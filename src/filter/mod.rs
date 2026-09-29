//! Filters: plain convolution, FIR (with windowed-sinc low-pass design) and biquad IIR (RBJ cookbook).
//!
//! Filters are stateful block processors: construct (allocates once), then `process(&mut [T])`
//! in place as often as needed without allocating, e.g. from an audio callback.
//!
//! [`MultiBiquad`] runs one biquad per channel, several channels at a time.

use crate::channels::{AudioBuffer, MultiProcessor};
use crate::units::*;

/// Full linear convolution of `a` and `b`; the result has `a.len() + b.len() - 1` samples
/// (empty if either input is empty). Allocates; for streaming use `Fir`.
pub fn convolve<T: Float>(a: &[T], b: &[T]) -> Vec<T> {
    if a.is_empty() || b.is_empty() {
        return Vec::new();
    }
    let mut out = vec![T::_ZERO; a.len() + b.len() - 1];
    for (i, &x) in a.iter().enumerate() {
        for (o, &h) in out[i..].iter_mut().zip(b) {
            *o = *o + x * h;
        }
    }
    out
}

/// |H(e^(i*w))| of the transfer function b(z) / a(z) at `frequency`,
/// with `b` and `a` given as coefficients of z^0, z^-1, z^-2, ...
fn magnitude_response<T: Float>(b: &[T], a: &[T], frequency: T, sample_rate: T) -> T {
    let w = T::_TAU * frequency / sample_rate;
    let poly = |coeffs: &[T]| {
        coeffs.iter().enumerate().fold(Complex::zero(), |acc, (k, &c)| acc + Complex::cis(-w * T::_lit(k as f64)) * c)
    };
    (poly(b) / poly(a)).norm()
}

// FIR =============================================================================================

/// Finite impulse response filter: `y[n] = sum_k h[k] * x[n - k]`.
///
/// Each output is one SIMD dot product (`simd::dot_kernel`) of the taps, stored reversed, with the
/// last N inputs, oldest first. Input is processed in chunks through a linear buffer: the previous
/// N-1 inputs, then the chunk. Every window is a plain slice of it, and the chunk is copied in before
/// any output is computed. Writing each sample just before reading it back as part of a wide vector
/// load would stall on every sample (the CPU can't forward a narrow store into a wider load).
/// Processing never allocates.
#[derive(Debug, Clone)]
pub struct Fir<T: Float> {
    taps: Vec<T>,
    /// taps in reverse order, aligned with the oldest-first input window
    reversed: Vec<T>,
    /// previous N-1 inputs (oldest first), followed by room for one chunk of new input
    buf: Vec<T>,
}

/// Samples per chunk in `Fir::process`; the cost of carrying history between chunks is spread over this many.
const FIR_CHUNK: usize = 128;

impl<T: Float> Fir<T> {
    /// Panics if `taps` is empty.
    pub fn new(taps: Vec<T>) -> Self {
        assert!(!taps.is_empty(), "a FIR filter needs at least one tap");
        let n = taps.len();
        let reversed = taps.iter().rev().copied().collect();
        Self { taps, reversed, buf: vec![T::_ZERO; n - 1 + FIR_CHUNK] }
    }
    /// Linear-phase low-pass (windowed sinc, Blackman window). See `design_lowpass`.
    pub fn lowpass(cutoff: T, sample_rate: T, num_taps: usize) -> Self {
        Self::new(design_lowpass(cutoff, sample_rate, num_taps))
    }
    pub fn taps(&self) -> &[T] {
        &self.taps
    }
    pub fn reset(&mut self) {
        self.buf.iter_mut().for_each(|s| *s = T::_ZERO);
    }
    /// Filters one sample. Costs an extra O(N) copy per call; for blocks use `process`.
    pub fn process_sample(&mut self, x: T) -> T {
        let mut one = [x];
        self.process(&mut one);
        one[0]
    }
    /// Filters `block` in place.
    pub fn process(&mut self, block: &mut [T]) {
        #[cfg(target_arch = "x86_64")]
        {
            if crate::simd::avx2_available() {
                // SAFETY: AVX2 support was just checked.
                return unsafe { self.process_avx2(block) };
            }
        }
        self.process_block(block)
    }
    #[cfg(target_arch = "x86_64")]
    #[target_feature(enable = "avx2")]
    unsafe fn process_avx2(&mut self, block: &mut [T]) {
        self.process_block(block)
    }
    /// The block loop; `inline(always)` so the baseline and AVX2 versions each get their own copy.
    #[inline(always)]
    fn process_block(&mut self, block: &mut [T]) {
        let n = self.taps.len();
        for chunk in block.chunks_mut(FIR_CHUNK) {
            let m = chunk.len();
            self.buf[n - 1..n - 1 + m].copy_from_slice(chunk);
            // output i uses inputs i ..= i + N-1 of the buffer: N-1 older samples then input i itself
            for (i, y) in chunk.iter_mut().enumerate() {
                *y = crate::simd::dot_kernel(&self.reversed, &self.buf[i..i + n]);
            }
            // the last N-1 inputs become the history for the next chunk
            self.buf.copy_within(m..m + n - 1, 0);
        }
    }
    /// Gain at `frequency` (1.0 = unchanged).
    pub fn magnitude_at(&self, frequency: T, sample_rate: T) -> T {
        magnitude_response(&self.taps, &[T::_ONE], frequency, sample_rate)
    }
    /// Delay in samples; constant across frequency because the designed taps are symmetric.
    pub fn group_delay(&self) -> T {
        T::_lit((self.taps.len() - 1) as f64 / 2.0)
    }
}

/// Windowed-sinc low-pass taps with unity DC gain. More taps = sharper transition band;
/// with a Blackman window the transition is roughly 5.5 * sample_rate / num_taps wide and
/// the stopband is ~74 dB down. Use an odd `num_taps` for a whole-sample group delay.
/// Panics unless 0 < cutoff < sample_rate / 2 and num_taps >= 2.
pub fn design_lowpass<T: Float>(cutoff: T, sample_rate: T, num_taps: usize) -> Vec<T> {
    let two = T::_lit(2.0);
    assert!(cutoff > T::_ZERO && cutoff < sample_rate / two, "cutoff must be between 0 and Nyquist");
    assert!(num_taps >= 2, "need at least 2 taps");
    let fc = cutoff / sample_rate; // normalized, cycles/sample
    let m = T::_lit((num_taps - 1) as f64);
    let mut taps: Vec<T> = (0..num_taps)
        .map(|i| {
            let n = T::_lit(i as f64);
            let x = n - m / two; // centered index
            let sinc = if x == T::_ZERO { two * fc } else { (T::_TAU * fc * x)._sin() / (T::_PI * x) };
            let window = T::_lit(0.42) - T::_lit(0.5) * (T::_TAU * n / m)._cos() + T::_lit(0.08) * (two * T::_TAU * n / m)._cos();
            sinc * window
        })
        .collect();
    let dc = taps.iter().fold(T::_ZERO, |acc, &h| acc + h);
    taps.iter_mut().for_each(|h| *h = *h / dc);
    taps
}

// BIQUAD ==========================================================================================

/// Q for a maximally flat (Butterworth) 2nd-order response.
pub const BUTTERWORTH_Q: f64 = std::f64::consts::FRAC_1_SQRT_2;

/// Normalized biquad coefficients (a0 = 1):
/// H(z) = (b0 + b1 z^-1 + b2 z^-2) / (1 + a1 z^-1 + a2 z^-2)
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct BiquadCoeffs<T: Float> {
    pub b0: T,
    pub b1: T,
    pub b2: T,
    pub a1: T,
    pub a2: T,
}

impl<T: Float> BiquadCoeffs<T> {
    /// Shared RBJ-cookbook setup: returns (cos w0, alpha).
    fn rbj(frequency: T, sample_rate: T, q: T) -> (T, T) {
        assert!(frequency > T::_ZERO && frequency < sample_rate / T::_lit(2.0), "frequency must be between 0 and Nyquist");
        assert!(q > T::_ZERO, "Q must be positive");
        let (sin_w, cos_w) = (T::_TAU * frequency / sample_rate)._sin_cos();
        (cos_w, sin_w / (T::_lit(2.0) * q))
    }
    fn normalized(b0: T, b1: T, b2: T, a0: T, a1: T, a2: T) -> Self {
        Self { b0: b0 / a0, b1: b1 / a0, b2: b2 / a0, a1: a1 / a0, a2: a2 / a0 }
    }
    /// -3 dB at `cutoff` when q = BUTTERWORTH_Q.
    pub fn lowpass(cutoff: T, sample_rate: T, q: T) -> Self {
        let (cos_w, alpha) = Self::rbj(cutoff, sample_rate, q);
        let b = (T::_ONE - cos_w) / T::_lit(2.0);
        Self::normalized(b, T::_ONE - cos_w, b, T::_ONE + alpha, T::_lit(-2.0) * cos_w, T::_ONE - alpha)
    }
    pub fn highpass(cutoff: T, sample_rate: T, q: T) -> Self {
        let (cos_w, alpha) = Self::rbj(cutoff, sample_rate, q);
        let b = (T::_ONE + cos_w) / T::_lit(2.0);
        Self::normalized(b, -(T::_ONE + cos_w), b, T::_ONE + alpha, T::_lit(-2.0) * cos_w, T::_ONE - alpha)
    }
    /// Unity gain at `center`; higher q = narrower band.
    pub fn bandpass(center: T, sample_rate: T, q: T) -> Self {
        let (cos_w, alpha) = Self::rbj(center, sample_rate, q);
        Self::normalized(alpha, T::_ZERO, -alpha, T::_ONE + alpha, T::_lit(-2.0) * cos_w, T::_ONE - alpha)
    }
    /// Zero gain at `center`; higher q = narrower notch.
    pub fn notch(center: T, sample_rate: T, q: T) -> Self {
        let (cos_w, alpha) = Self::rbj(center, sample_rate, q);
        let b1 = T::_lit(-2.0) * cos_w;
        Self::normalized(T::_ONE, b1, T::_ONE, T::_ONE + alpha, b1, T::_ONE - alpha)
    }
    /// Unity gain at every frequency; only the phase changes (by 180 degrees at `center`).
    /// Building block for phasers and phase-alignment.
    pub fn allpass(center: T, sample_rate: T, q: T) -> Self {
        let (cos_w, alpha) = Self::rbj(center, sample_rate, q);
        let a1 = T::_lit(-2.0) * cos_w;
        Self::normalized(T::_ONE - alpha, a1, T::_ONE + alpha, T::_ONE + alpha, a1, T::_ONE - alpha)
    }
    /// Boosts or cuts by `gain_db` around `center` (a bell), unity far away; higher q = narrower bell.
    pub fn peaking(center: T, sample_rate: T, q: T, gain_db: T) -> Self {
        let (cos_w, alpha) = Self::rbj(center, sample_rate, q);
        let a = Self::shelf_amplitude(gain_db);
        let a1 = T::_lit(-2.0) * cos_w;
        Self::normalized(T::_ONE + alpha * a, a1, T::_ONE - alpha * a, T::_ONE + alpha / a, a1, T::_ONE - alpha / a)
    }
    /// Boosts or cuts everything below `corner` by `gain_db`; q = BUTTERWORTH_Q gives the steepest
    /// slope without overshoot (cookbook shelf slope S = 1).
    pub fn low_shelf(corner: T, sample_rate: T, q: T, gain_db: T) -> Self {
        let (cos_w, alpha) = Self::rbj(corner, sample_rate, q);
        let a = Self::shelf_amplitude(gain_db);
        let (ap1, am1, two) = (a + T::_ONE, a - T::_ONE, T::_lit(2.0));
        let k = two * a._sqrt() * alpha;
        Self::normalized(
            a * (ap1 - am1 * cos_w + k),
            two * a * (am1 - ap1 * cos_w),
            a * (ap1 - am1 * cos_w - k),
            ap1 + am1 * cos_w + k,
            -two * (am1 + ap1 * cos_w),
            ap1 + am1 * cos_w - k,
        )
    }
    /// Boosts or cuts everything above `corner` by `gain_db`; see `low_shelf` for q.
    pub fn high_shelf(corner: T, sample_rate: T, q: T, gain_db: T) -> Self {
        let (cos_w, alpha) = Self::rbj(corner, sample_rate, q);
        let a = Self::shelf_amplitude(gain_db);
        let (ap1, am1, two) = (a + T::_ONE, a - T::_ONE, T::_lit(2.0));
        let k = two * a._sqrt() * alpha;
        Self::normalized(
            a * (ap1 + am1 * cos_w + k),
            -two * a * (am1 + ap1 * cos_w),
            a * (ap1 + am1 * cos_w - k),
            ap1 - am1 * cos_w + k,
            two * (am1 - ap1 * cos_w),
            ap1 - am1 * cos_w - k,
        )
    }
    /// The cookbook's A = 10^(dB/40): the square root of the linear gain, split between numerator and denominator.
    fn shelf_amplitude(gain_db: T) -> T {
        crate::gain::db_to_gain(gain_db)._sqrt()
    }
    /// Gain at `frequency` (1.0 = unchanged).
    pub fn magnitude_at(&self, frequency: T, sample_rate: T) -> T {
        magnitude_response(&[self.b0, self.b1, self.b2], &[T::_ONE, self.a1, self.a2], frequency, sample_rate)
    }
}

/// Which cookbook response a biquad was designed as.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum BiquadKind {
    Lowpass,
    Highpass,
    Bandpass,
    Notch,
    Allpass,
    Peaking,
    LowShelf,
    HighShelf,
}

impl BiquadKind {
    /// Whether `gain_db` affects this response (peaking and shelves).
    pub fn uses_gain(self) -> bool {
        matches!(self, BiquadKind::Peaking | BiquadKind::LowShelf | BiquadKind::HighShelf)
    }
}

/// The settings a biquad was designed from, so it can be inspected and re-designed (e.g. by a host
/// changing its frequency) rather than only holding opaque coefficients.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct BiquadDesign<T: Float> {
    pub kind: BiquadKind,
    /// cutoff, center or corner frequency in Hz
    pub frequency: T,
    pub q: T,
    /// used by peaking and shelves only
    pub gain_db: T,
    pub sample_rate: T,
}

impl<T: Float> BiquadDesign<T> {
    /// Panics if the frequency isn't between 0 and Nyquist or q isn't positive (see `BiquadCoeffs`).
    pub fn coeffs(&self) -> BiquadCoeffs<T> {
        let (f, fs, q, g) = (self.frequency, self.sample_rate, self.q, self.gain_db);
        match self.kind {
            BiquadKind::Lowpass => BiquadCoeffs::lowpass(f, fs, q),
            BiquadKind::Highpass => BiquadCoeffs::highpass(f, fs, q),
            BiquadKind::Bandpass => BiquadCoeffs::bandpass(f, fs, q),
            BiquadKind::Notch => BiquadCoeffs::notch(f, fs, q),
            BiquadKind::Allpass => BiquadCoeffs::allpass(f, fs, q),
            BiquadKind::Peaking => BiquadCoeffs::peaking(f, fs, q, g),
            BiquadKind::LowShelf => BiquadCoeffs::low_shelf(f, fs, q, g),
            BiquadKind::HighShelf => BiquadCoeffs::high_shelf(f, fs, q, g),
        }
    }
}

/// Second-order IIR filter in transposed direct form II (two state variables, good float behavior).
#[derive(Debug, Clone, Copy)]
pub struct Biquad<T: Float> {
    coeffs: BiquadCoeffs<T>,
    /// how the coefficients were designed; `None` when built from raw coefficients
    design: Option<BiquadDesign<T>>,
    s1: T,
    s2: T,
}

impl<T: Float> Biquad<T> {
    /// From raw coefficients (no design settings to inspect or change).
    pub fn new(coeffs: BiquadCoeffs<T>) -> Self {
        Self { coeffs, design: None, s1: T::_ZERO, s2: T::_ZERO }
    }
    pub fn from_design(design: BiquadDesign<T>) -> Self {
        Self { coeffs: design.coeffs(), design: Some(design), s1: T::_ZERO, s2: T::_ZERO }
    }
    fn designed(kind: BiquadKind, frequency: T, sample_rate: T, q: T, gain_db: T) -> Self {
        Self::from_design(BiquadDesign { kind, frequency, q, gain_db, sample_rate })
    }
    pub fn lowpass(cutoff: T, sample_rate: T, q: T) -> Self {
        Self::designed(BiquadKind::Lowpass, cutoff, sample_rate, q, T::_ZERO)
    }
    pub fn highpass(cutoff: T, sample_rate: T, q: T) -> Self {
        Self::designed(BiquadKind::Highpass, cutoff, sample_rate, q, T::_ZERO)
    }
    pub fn bandpass(center: T, sample_rate: T, q: T) -> Self {
        Self::designed(BiquadKind::Bandpass, center, sample_rate, q, T::_ZERO)
    }
    pub fn notch(center: T, sample_rate: T, q: T) -> Self {
        Self::designed(BiquadKind::Notch, center, sample_rate, q, T::_ZERO)
    }
    pub fn allpass(center: T, sample_rate: T, q: T) -> Self {
        Self::designed(BiquadKind::Allpass, center, sample_rate, q, T::_ZERO)
    }
    pub fn peaking(center: T, sample_rate: T, q: T, gain_db: T) -> Self {
        Self::designed(BiquadKind::Peaking, center, sample_rate, q, gain_db)
    }
    pub fn low_shelf(corner: T, sample_rate: T, q: T, gain_db: T) -> Self {
        Self::designed(BiquadKind::LowShelf, corner, sample_rate, q, gain_db)
    }
    pub fn high_shelf(corner: T, sample_rate: T, q: T, gain_db: T) -> Self {
        Self::designed(BiquadKind::HighShelf, corner, sample_rate, q, gain_db)
    }
    pub fn coeffs(&self) -> &BiquadCoeffs<T> {
        &self.coeffs
    }
    pub fn design(&self) -> Option<&BiquadDesign<T>> {
        self.design.as_ref()
    }
    /// Re-designs the filter, keeping its state (no click).
    pub fn set_design(&mut self, design: BiquadDesign<T>) {
        self.coeffs = design.coeffs();
        self.design = Some(design);
    }
    /// Swaps in raw coefficients, keeping state, so changes mid-stream don't click.
    /// The filter no longer has design settings afterwards.
    pub fn set_coeffs(&mut self, coeffs: BiquadCoeffs<T>) {
        self.coeffs = coeffs;
        self.design = None;
    }
    pub fn reset(&mut self) {
        self.s1 = T::_ZERO;
        self.s2 = T::_ZERO;
    }
    #[inline]
    pub fn process_sample(&mut self, x: T) -> T {
        let c = &self.coeffs;
        let y = c.b0 * x + self.s1;
        self.s1 = c.b1 * x - c.a1 * y + self.s2;
        self.s2 = c.b2 * x - c.a2 * y;
        y
    }
    /// Filters `block` in place.
    pub fn process(&mut self, block: &mut [T]) {
        for s in block {
            *s = self.process_sample(*s);
        }
    }
    pub fn magnitude_at(&self, frequency: T, sample_rate: T) -> T {
        self.coeffs.magnitude_at(frequency, sample_rate)
    }
}

// MULTICHANNEL BIQUAD =============================================================================

/// One biquad per channel, processed several channels at a time.
///
/// A biquad is recursive (each output needs the previous one), so a single channel can't be
/// vectorized along time and runs at the speed of its dependency chain. Channels are independent, so
/// running 4 in lockstep lets their chains overlap and fills SIMD lanes. Each channel does exactly
/// the arithmetic a lone `Biquad` would, so results are bit-identical to per-channel processing.
/// Allocates only in `new`.
#[derive(Debug, Clone)]
pub struct MultiBiquad<T: Float> {
    filters: Vec<Biquad<T>>,
}

impl<T: Float> MultiBiquad<T> {
    /// `make(channel)` builds each channel's filter (they may differ).
    pub fn new(channels: usize, make: impl FnMut(usize) -> Biquad<T>) -> Self {
        Self { filters: (0..channels).map(make).collect() }
    }
    pub fn channels(&self) -> &[Biquad<T>] {
        &self.filters
    }
    pub fn channel_mut(&mut self, ch: usize) -> &mut Biquad<T> {
        &mut self.filters[ch]
    }
    /// Re-designs every channel's filter, keeping their state (no click).
    pub fn set_design_all(&mut self, design: BiquadDesign<T>) {
        self.filters.iter_mut().for_each(|f| f.set_design(design));
    }
    pub fn reset(&mut self) {
        self.filters.iter_mut().for_each(Biquad::reset);
    }

    /// Filters every channel of `buffer` in place. Panics if the channel count differs.
    pub fn process_buffer(&mut self, buffer: &mut AudioBuffer<T>) {
        assert_eq!(buffer.channels(), self.filters.len(), "channel count mismatch");
        #[cfg(target_arch = "x86_64")]
        {
            if crate::simd::avx2_available() {
                // SAFETY: AVX2 support was just checked.
                return unsafe { self.process_avx2(buffer) };
            }
        }
        self.process_groups(buffer)
    }

    #[cfg(target_arch = "x86_64")]
    #[target_feature(enable = "avx2")]
    unsafe fn process_avx2(&mut self, buffer: &mut AudioBuffer<T>) {
        self.process_groups(buffer)
    }

    /// Channels in groups of 4, then 2, then 1.
    #[inline(always)]
    fn process_groups(&mut self, buffer: &mut AudioBuffer<T>) {
        let frames = buffer.frames();
        let mut channels = buffer.channels_mut();
        let mut filters = self.filters.as_mut_slice();
        while !filters.is_empty() {
            let lanes = match filters.len() {
                n if n >= 4 => 4,
                n if n >= 2 => 2,
                _ => 1,
            };
            let (group, rest) = filters.split_at_mut(lanes);
            match lanes {
                4 => run_lanes::<T, 4>(group, std::array::from_fn(|_| channels.next().unwrap()), frames),
                2 => run_lanes::<T, 2>(group, std::array::from_fn(|_| channels.next().unwrap()), frames),
                _ => run_lanes::<T, 1>(group, std::array::from_fn(|_| channels.next().unwrap()), frames),
            }
            filters = rest;
        }
    }
}

/// `L` biquads over `L` channels in lockstep, with coefficients and state in per-lane arrays.
#[inline(always)]
// indexing by frame is the point: every lane handles frame f before any moves on to f + 1
#[allow(clippy::needless_range_loop)]
fn run_lanes<T: Float, const L: usize>(filters: &mut [Biquad<T>], channels: [&mut [T]; L], frames: usize) {
    let lane = |f: fn(&Biquad<T>) -> T| -> [T; L] { std::array::from_fn(|l| f(&filters[l])) };
    let (b0, b1, b2) = (lane(|q| q.coeffs.b0), lane(|q| q.coeffs.b1), lane(|q| q.coeffs.b2));
    let (a1, a2) = (lane(|q| q.coeffs.a1), lane(|q| q.coeffs.a2));
    let (mut s1, mut s2) = (lane(|q| q.s1), lane(|q| q.s2));
    let channels = channels.map(|c| &mut c[..frames]);
    for f in 0..frames {
        for l in 0..L {
            let x = channels[l][f];
            let y = b0[l] * x + s1[l];
            s1[l] = b1[l] * x - a1[l] * y + s2[l];
            s2[l] = b2[l] * x - a2[l] * y;
            channels[l][f] = y;
        }
    }
    for (l, q) in filters.iter_mut().enumerate() {
        q.s1 = s1[l];
        q.s2 = s2[l];
    }
}

impl<T: Float> MultiProcessor<T> for MultiBiquad<T> {
    fn process(&mut self, buffer: &mut AudioBuffer<T>) {
        self.process_buffer(buffer)
    }
    fn reset(&mut self) {
        MultiBiquad::reset(self)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::osc::{Impulse, Noise, Sine};

    const FS: f64 = 48_000.0;

    fn assert_close(a: f64, b: f64, tol: f64, what: &str) {
        assert!((a - b).abs() <= tol, "{what}: {a} vs {b} (tol {tol})");
    }

    /// Steady-state gain of `process` for a unit sine at `freq`, from RMS (a unit sine has RMS 1/sqrt 2).
    /// RMS rather than peak: at high frequencies the samples straddle the true peak.
    /// The 24000-sample window must hold a whole number of cycles of `freq` for this to be exact.
    fn measured_gain(mut process: impl FnMut(&mut [f64]), freq: f64) -> f64 {
        let mut buf = vec![0.0; 48_000];
        Sine::new(freq, FS).fill(&mut buf);
        process(&mut buf);
        let tail = &buf[24_000..]; // skip the transient
        (2.0 * tail.iter().map(|s| s * s).sum::<f64>() / tail.len() as f64).sqrt()
    }

    #[test]
    fn convolve_known_answer() {
        assert_eq!(convolve(&[1.0, 2.0, 3.0], &[0.0, 1.0, 0.5]), [0.0, 1.0, 2.5, 4.0, 1.5]);
        assert!(convolve::<f64>(&[], &[1.0]).is_empty());
    }

    #[test]
    fn fir_matches_convolution_and_impulse_response_is_taps() {
        let taps = vec![0.5, -0.25, 0.125, 1.0];
        let input: Vec<f64> = Noise::new(7).take(64).collect();
        let mut out = input.clone();
        Fir::new(taps.clone()).process(&mut out);
        let reference = convolve(&input, &taps);
        for (a, b) in out.iter().zip(&reference) {
            assert_close(*a, *b, 1e-12, "fir vs convolve");
        }

        let mut ir = vec![0.0; 6];
        Impulse::new().fill(&mut ir);
        Fir::new(taps.clone()).process(&mut ir);
        assert_eq!(&ir[..4], &taps[..]);
        assert_eq!(&ir[4..], &[0.0, 0.0]);
    }

    #[test]
    fn multi_biquad_is_bit_identical_to_per_channel_processing() {
        for channels in 1..=9 {
            // different filters per channel
            let make = |ch: usize| Biquad::peaking(300.0 * (ch + 1) as f64, FS, 0.7 + ch as f64 * 0.1, ch as f64 - 3.0);
            let mut buf = AudioBuffer::new(channels, 700);
            for ch in 0..channels {
                Noise::new(ch as u64).fill(buf.channel_mut(ch));
            }
            let mut reference = buf.clone();
            let mut singles: Vec<Biquad<f64>> = (0..channels).map(make).collect();
            for (ch, f) in singles.iter_mut().enumerate() {
                f.process(reference.channel_mut(ch));
            }
            let mut multi = MultiBiquad::new(channels, make);
            // two calls, so state carries across blocks too
            buf.set_frames(300);
            multi.process_buffer(&mut buf);
            let first: Vec<Vec<f64>> = (0..channels).map(|ch| buf.channel(ch).to_vec()).collect();
            buf.set_frames(700);
            let mut tail = AudioBuffer::new(channels, 400);
            for ch in 0..channels {
                tail.channel_mut(ch).copy_from_slice(&buf.channel(ch)[300..]);
            }
            multi.process_buffer(&mut tail);
            for (ch, first) in first.iter().enumerate() {
                assert_eq!(&first[..], &reference.channel(ch)[..300], "{channels} channels, ch {ch}, first block");
                assert_eq!(tail.channel(ch), &reference.channel(ch)[300..], "{channels} channels, ch {ch}, second block");
            }
        }
    }

    #[test]
    fn fir_is_seamless_across_chunks_and_block_sizes() {
        // blocks smaller than, larger than and straddling the internal chunk size, plus single samples
        let taps: Vec<f64> = Noise::new(11).take(67).collect();
        let input: Vec<f64> = Noise::new(12).take(1_000).collect();
        let reference = convolve(&input, &taps);
        let mut fir = Fir::new(taps.clone());
        let mut out = input.clone();
        let mut start = 0;
        for size in [37, 300, 1, 128, 129, 5].iter().cycle() {
            if start == out.len() {
                break;
            }
            let end = (start + size).min(out.len());
            if *size == 1 {
                out[start] = fir.process_sample(out[start]);
            } else {
                fir.process(&mut out[start..end]);
            }
            start = end;
        }
        for (n, (a, b)) in out.iter().zip(&reference).enumerate() {
            assert_close(*a, *b, 1e-12, &format!("sample {n}"));
        }
    }

    #[test]
    fn fir_baseline_path_matches_dispatched_path() {
        // this machine / CI may always take the AVX2 path; exercise the baseline loop directly too
        let taps: Vec<f32> = Noise::new(13).take(31).collect();
        let input: Vec<f32> = Noise::new(14).take(500).collect();
        let (mut a, mut b) = (input.clone(), input);
        Fir::new(taps.clone()).process(&mut a);
        Fir::new(taps).process_block(&mut b);
        for (x, y) in a.iter().zip(&b) {
            assert!((x - y).abs() < 1e-5);
        }
    }

    #[test]
    fn single_tap_fir_is_a_gain() {
        let mut fir = Fir::new(vec![0.5]);
        let mut buf = [2.0, -4.0, 6.0];
        fir.process(&mut buf);
        assert_eq!(buf, [1.0, -2.0, 3.0]);
    }

    #[test]
    fn windowed_sinc_lowpass() {
        let fir = Fir::lowpass(1000.0, FS, 255);
        let taps = fir.taps();
        assert_close(taps.iter().sum(), 1.0, 1e-12, "DC gain");
        for (a, b) in taps.iter().zip(taps.iter().rev()) {
            assert_close(*a, *b, 1e-15, "symmetry (linear phase)");
        }
        assert_close(fir.magnitude_at(100.0, FS), 1.0, 1e-3, "passband");
        // transition ~ 5.5 * 48k / 255 ~ 1 kHz wide, so 3 kHz is well into the stopband
        assert!(fir.magnitude_at(3000.0, FS) < 1e-3, "stopband");
        assert_eq!(fir.group_delay(), 127.0);
        // time domain agrees with the analytic response
        for f in [200.0, 5000.0] {
            let mut fir = fir.clone();
            assert_close(measured_gain(|b| fir.process(b), f), fir.magnitude_at(f, FS), 1e-3, "fir measured gain");
        }
    }

    #[test]
    fn biquad_responses_match_cookbook_definitions() {
        let q = BUTTERWORTH_Q;
        let lp = BiquadCoeffs::lowpass(1000.0, FS, q);
        assert_close(lp.magnitude_at(1e-3, FS), 1.0, 1e-9, "lowpass DC");
        assert_close(lp.magnitude_at(1000.0, FS), BUTTERWORTH_Q, 1e-9, "lowpass -3 dB point");
        assert!(lp.magnitude_at(23_999.0, FS) < 1e-6, "lowpass Nyquist");

        let hp = BiquadCoeffs::highpass(1000.0, FS, q);
        assert_close(hp.magnitude_at(1000.0, FS), BUTTERWORTH_Q, 1e-9, "highpass -3 dB point");
        assert!(hp.magnitude_at(1e-3, FS) < 1e-6, "highpass DC");

        let bp = BiquadCoeffs::bandpass(2000.0, FS, 5.0);
        assert_close(bp.magnitude_at(2000.0, FS), 1.0, 1e-9, "bandpass peak");
        assert!(bp.magnitude_at(200.0, FS) < 0.05, "bandpass skirt");

        let notch = BiquadCoeffs::notch(2000.0, FS, 5.0);
        assert!(notch.magnitude_at(2000.0, FS) < 1e-9, "notch center");
        assert_close(notch.magnitude_at(200.0, FS), 1.0, 1e-3, "notch passband");
    }

    #[test]
    fn eq_shapes_hit_their_gains() {
        let db = |g: f64| 20.0 * g.log10();
        let q = BUTTERWORTH_Q;
        for gain_db in [-12.0, -3.0, 6.0, 12.0] {
            let bell = BiquadCoeffs::peaking(2000.0, FS, 1.0, gain_db);
            assert_close(db(bell.magnitude_at(2000.0, FS)), gain_db, 1e-9, "peaking center");
            assert_close(db(bell.magnitude_at(20.0, FS)), 0.0, 0.01, "peaking far below");

            let low = BiquadCoeffs::low_shelf(500.0, FS, q, gain_db);
            assert_close(db(low.magnitude_at(1e-3, FS)), gain_db, 1e-6, "low shelf DC");
            assert_close(db(low.magnitude_at(20_000.0, FS)), 0.0, 0.01, "low shelf top");
            // a shelf is half-way (in dB) at its corner frequency
            assert_close(db(low.magnitude_at(500.0, FS)), gain_db / 2.0, 1e-9, "low shelf corner");

            let high = BiquadCoeffs::high_shelf(5000.0, FS, q, gain_db);
            assert_close(db(high.magnitude_at(23_999.0, FS)), gain_db, 0.01, "high shelf top");
            assert_close(db(high.magnitude_at(1e-3, FS)), 0.0, 1e-6, "high shelf DC");
            assert_close(db(high.magnitude_at(5000.0, FS)), gain_db / 2.0, 1e-9, "high shelf corner");
        }
        let flat = BiquadCoeffs::peaking(2000.0, FS, 1.0, 0.0);
        assert_close(flat.magnitude_at(777.0, FS), 1.0, 1e-12, "0 dB peaking is transparent");
    }

    #[test]
    fn allpass_is_flat_and_shifts_phase_by_180_at_center() {
        let ap = BiquadCoeffs::allpass(3000.0, FS, 0.7);
        for f in [10.0, 300.0, 3000.0, 12_000.0, 23_000.0] {
            assert_close(ap.magnitude_at(f, FS), 1.0, 1e-9, "allpass magnitude");
        }
        // at the center frequency the output is the inverted input (steady state)
        let mut bq = Biquad::new(ap);
        let mut buf = vec![0.0; 48_000];
        Sine::new(3000.0, FS).fill(&mut buf);
        let input = buf.clone();
        bq.process(&mut buf);
        for n in 24_000..24_100 {
            assert_close(buf[n], -input[n], 1e-6, "allpass inversion at center");
        }
    }

    #[test]
    fn biquad_time_domain_matches_frequency_response() {
        for coeffs in [
            BiquadCoeffs::lowpass(1000.0, FS, BUTTERWORTH_Q),
            BiquadCoeffs::highpass(1000.0, FS, BUTTERWORTH_Q),
            BiquadCoeffs::bandpass(2000.0, FS, 2.0),
            BiquadCoeffs::notch(2000.0, FS, 2.0),
        ] {
            for f in [100.0, 1000.0, 2000.0, 8000.0] {
                let mut bq = Biquad::new(coeffs);
                assert_close(measured_gain(|b| bq.process(b), f), coeffs.magnitude_at(f, FS), 2e-3, &format!("{coeffs:?} at {f} Hz"));
            }
        }
    }

    #[test]
    fn biquad_f32_is_stable_and_reset_clears_state() {
        let mut bq = Biquad::<f32>::lowpass(50.0, 48_000.0, 0.707);
        let mut buf = [1.0f32; 48_000];
        bq.process(&mut buf);
        assert!((buf[47_999] - 1.0).abs() < 1e-3, "DC step settles to 1: {}", buf[47_999]);
        bq.reset();
        assert_eq!(bq.process_sample(0.0), 0.0);
    }
}
