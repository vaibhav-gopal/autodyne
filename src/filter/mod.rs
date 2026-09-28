//! Filters: plain convolution, FIR (with windowed-sinc low-pass design) and biquad IIR (RBJ cookbook).
//!
//! Filters are stateful block processors: construct (allocates once), then `process(&mut [T])`
//! in place as often as needed without allocating, e.g. from an audio callback.

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

/// Finite impulse response filter: y[n] = sum_k h[k] * x[n - k].
/// History is a ring buffer sized to the tap count, so processing never allocates.
#[derive(Debug, Clone)]
pub struct Fir<T: Float> {
    taps: Vec<T>,
    history: Vec<T>,
    /// index of the most recently written sample in `history`
    pos: usize,
}

impl<T: Float> Fir<T> {
    /// Panics if `taps` is empty.
    pub fn new(taps: Vec<T>) -> Self {
        assert!(!taps.is_empty(), "a FIR filter needs at least one tap");
        let n = taps.len();
        Self { taps, history: vec![T::_ZERO; n], pos: n - 1 }
    }
    /// Linear-phase low-pass (windowed sinc, Blackman window). See `design_lowpass`.
    pub fn lowpass(cutoff: T, sample_rate: T, num_taps: usize) -> Self {
        Self::new(design_lowpass(cutoff, sample_rate, num_taps))
    }
    pub fn taps(&self) -> &[T] {
        &self.taps
    }
    pub fn reset(&mut self) {
        self.history.iter_mut().for_each(|s| *s = T::_ZERO);
    }
    #[inline]
    pub fn process_sample(&mut self, x: T) -> T {
        let n = self.taps.len();
        self.pos = if self.pos + 1 == n { 0 } else { self.pos + 1 };
        self.history[self.pos] = x;
        // walk backwards through history (newest -> oldest) while walking forward through the taps
        let (newer, older) = self.history.split_at(self.pos + 1);
        let recent = newer.iter().rev().chain(older.iter().rev());
        self.taps.iter().zip(recent).fold(T::_ZERO, |acc, (&h, &x)| acc + h * x)
    }
    /// Filters `block` in place.
    pub fn process(&mut self, block: &mut [T]) {
        for s in block {
            *s = self.process_sample(*s);
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
    /// Gain at `frequency` (1.0 = unchanged).
    pub fn magnitude_at(&self, frequency: T, sample_rate: T) -> T {
        magnitude_response(&[self.b0, self.b1, self.b2], &[T::_ONE, self.a1, self.a2], frequency, sample_rate)
    }
}

/// Second-order IIR filter in transposed direct form II (two state variables, good float behavior).
#[derive(Debug, Clone, Copy)]
pub struct Biquad<T: Float> {
    coeffs: BiquadCoeffs<T>,
    s1: T,
    s2: T,
}

impl<T: Float> Biquad<T> {
    pub fn new(coeffs: BiquadCoeffs<T>) -> Self {
        Self { coeffs, s1: T::_ZERO, s2: T::_ZERO }
    }
    pub fn lowpass(cutoff: T, sample_rate: T, q: T) -> Self {
        Self::new(BiquadCoeffs::lowpass(cutoff, sample_rate, q))
    }
    pub fn highpass(cutoff: T, sample_rate: T, q: T) -> Self {
        Self::new(BiquadCoeffs::highpass(cutoff, sample_rate, q))
    }
    pub fn bandpass(center: T, sample_rate: T, q: T) -> Self {
        Self::new(BiquadCoeffs::bandpass(center, sample_rate, q))
    }
    pub fn notch(center: T, sample_rate: T, q: T) -> Self {
        Self::new(BiquadCoeffs::notch(center, sample_rate, q))
    }
    pub fn coeffs(&self) -> &BiquadCoeffs<T> {
        &self.coeffs
    }
    /// Swaps coefficients but keeps state, so parameter changes mid-stream don't click.
    pub fn set_coeffs(&mut self, coeffs: BiquadCoeffs<T>) {
        self.coeffs = coeffs;
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
