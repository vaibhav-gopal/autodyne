//! Sample-rate conversion by a rational factor (e.g. 48 kHz -> 44.1 kHz is 147/160).
//!
//! Conceptually: insert L-1 zeros between samples (upsample by L), low-pass to remove the images and
//! anything above the new Nyquist, then keep every M-th sample. The polyphase form computes only the
//! filter taps that land on real input samples, so each output costs about `taps_per_phase`
//! multiply-adds regardless of L.
//!
//! [`Oversampled`] uses a pair of resamplers to run any processor at a multiple of the sample rate.
//! [`Resample`] converts whole signals (`x.resampled(48_000, 44_100)`), aligned to the input;
//! [`upfirdn`] and `resample_poly` (feature `faer`) do it along an axis of n-d data as `scipy.signal` does.
//!
//! tend: Signal processing / resample

use crate::filter::design_lowpass;
use crate::signal::{SigResizeOps, Signal, SignalResizable};
use crate::units::*;

mod params;
mod oversample;
mod poly;
pub use oversample::*;
pub use poly::*;

/// Input samples per chunk in `Resampler::process`.
const RESAMPLE_CHUNK: usize = 128;

/// Streaming rational resampler. Allocates in `new`; `process` does not.
///
/// Each output is one SIMD dot product of a polyphase branch (stored reversed) with the recent input,
/// oldest first. As in `Fir`, input goes through a linear buffer a chunk at a time, so every window is a
/// plain slice written before it is read.
#[derive(Debug, Clone)]
pub struct Resampler<T: Float> {
    up: usize,
    down: usize,
    /// taps per polyphase branch
    branch_len: usize,
    /// branch p holds prototype taps h[p + kL] (zero padded), reversed: poly[p * branch_len + (branch_len - 1 - k)]
    poly: Vec<T>,
    /// previous branch_len - 1 inputs (oldest first), followed by room for one chunk of new input
    buf: Vec<T>,
    /// position of the next output in the upsampled timeline, relative to the newest input sample
    phase: usize,
    /// prototype filter delay in upsampled samples
    prototype_delay: f64,
}

impl<T: Float> Resampler<T> {
    /// Default quality: 64 taps per branch (transition band ~ 10% of the lower Nyquist, ~74 dB stopband).
    pub fn new(input_rate: u32, output_rate: u32) -> Self {
        Self::with_quality(input_rate, output_rate, 64)
    }

    /// Higher `taps_per_phase` = sharper anti-aliasing filter at more CPU per sample.
    /// Panics if either rate is zero or `taps_per_phase` is zero.
    pub fn with_quality(input_rate: u32, output_rate: u32, taps_per_phase: usize) -> Self {
        assert!(input_rate > 0 && output_rate > 0, "sample rates must be positive");
        assert!(taps_per_phase > 0, "need at least one tap per phase");
        let g = gcd(input_rate as usize, output_rate as usize) as u64;
        let (up, down) = ((output_rate as u64 / g) as usize, (input_rate as u64 / g) as usize);

        // Prototype runs at input_rate * up. Its cutoff sits below the lower of the two Nyquists,
        // and its length scales with max(up, down) so the transition band stays a fixed fraction of it.
        let fs_up = input_rate as f64 * up as f64;
        let cutoff = 0.45 * input_rate.min(output_rate) as f64;
        // Length 2*M*k + 1 (at least the requested quality) puts the linear-phase filter's delay,
        // (len - 1) / 2 = M*k upsampled samples, on exactly k output samples, so output can be aligned
        // with the input by dropping whole samples (see `delay`).
        let k = (taps_per_phase * up.max(down)).div_ceil(2 * down);
        let len = 2 * down * k + 1;
        let mut proto: Vec<T> = design_lowpass(T::_lit(cutoff), len, T::_lit(fs_up));
        // zero stuffing divides the signal level by L; the filter makes it back up
        proto.iter_mut().for_each(|h| *h = *h * T::_lit(up as f64));

        let branch_len = proto.len().div_ceil(up);
        let mut poly = vec![T::_ZERO; up * branch_len];
        for (i, &h) in proto.iter().enumerate() {
            poly[(i % up) * branch_len + (branch_len - 1 - i / up)] = h;
        }
        Self {
            up,
            down,
            branch_len,
            poly,
            buf: vec![T::_ZERO; branch_len - 1 + RESAMPLE_CHUNK],
            phase: 0,
            prototype_delay: (proto.len() - 1) as f64 / 2.0,
        }
    }

    /// (L, M): output_rate / input_rate reduced to lowest terms.
    pub fn ratio(&self) -> (usize, usize) {
        (self.up, self.down)
    }

    /// Filter delay, in output samples. Always a whole number: output `n + delay()` lines up with
    /// input time `n / output_rate`.
    pub fn delay(&self) -> f64 {
        self.prototype_delay / self.down as f64
    }

    /// Upper bound on how many samples `process` can write for `input_len` input samples.
    pub fn max_output_len(&self, input_len: usize) -> usize {
        (input_len * self.up).div_ceil(self.down) + 1
    }

    /// Clears the input history and restarts the phase.
    pub fn reset(&mut self) {
        self.buf.iter_mut().for_each(|s| *s = T::_ZERO);
        self.phase = 0;
    }

    /// Resamples `input` into the front of `out` and returns how many samples were written.
    /// Output count varies block to block (it averages input.len() * L / M); size `out` with
    /// `max_output_len`. Panics if `out` is too small.
    pub fn process(&mut self, input: &[T], out: &mut [T]) -> usize {
        assert!(out.len() >= self.max_output_len(input.len()), "output buffer too small; use max_output_len");
        #[cfg(target_arch = "x86_64")]
        {
            if crate::simd::avx2_available() {
                // SAFETY: AVX2 support was just checked.
                return unsafe { self.process_avx2(input, out) };
            }
        }
        self.process_block(input, out)
    }

    #[cfg(target_arch = "x86_64")]
    #[target_feature(enable = "avx2")]
    unsafe fn process_avx2(&mut self, input: &[T], out: &mut [T]) -> usize {
        self.process_block(input, out)
    }

    /// The block loop; `inline(always)` so the baseline and AVX2 versions each get their own copy.
    #[inline(always)]
    fn process_block(&mut self, input: &[T], out: &mut [T]) -> usize {
        let k = self.branch_len;
        let mut written = 0;
        for chunk in input.chunks(RESAMPLE_CHUNK) {
            let m = chunk.len();
            self.buf[k - 1..k - 1 + m].copy_from_slice(chunk);
            for i in 0..m {
                let window = &self.buf[i..i + k]; // the k inputs ending at chunk[i], oldest first
                // Every output whose upsampled position falls in [i * L, i * L + L) needs no newer input.
                while self.phase < self.up {
                    let branch = &self.poly[self.phase * k..][..k];
                    out[written] = crate::simd::dot_kernel(branch, window);
                    written += 1;
                    self.phase += self.down;
                }
                self.phase -= self.up;
            }
            // the last k-1 inputs become the history for the next chunk
            self.buf.copy_within(m..m + k - 1, 0);
        }
        written
    }
}

/// Sample-rate conversion of whole signals: the streaming [`Resampler`] run over a signal, aligned to
/// it. Implemented for every [`Signal`] (in the prelude).
pub trait Resample: Signal {
    /// The whole signal converted from `from_rate` to `to_rate`, aligned to the input (the filter
    /// delay removed) and `ceil(len * to / from)` samples long.
    fn resampled(&self, from_rate: u32, to_rate: u32) -> Vec<Self::Sample> {
        let input = self.samples();
        let mut rs = Resampler::new(from_rate, to_rate);
        let (up, down) = rs.ratio();
        let wanted = (input.len() * up).div_ceil(down);
        let delay = rs.delay() as usize;
        let mut out = Vec::with_capacity(wanted + delay + up);
        let chunk = 4096;
        let mut scratch = vec![Self::Sample::_ZERO; rs.max_output_len(chunk)];
        let zeros = vec![Self::Sample::_ZERO; chunk];
        let mut fed = 0;
        while out.len() < wanted + delay {
            // the input, then silence to flush the filter's delay
            let block = if fed < input.len() { &input[fed..(fed + chunk).min(input.len())] } else { &zeros[..] };
            fed += block.len();
            let n = rs.process(block, &mut scratch);
            out.extend_from_slice(&scratch[..n]);
        }
        out.drain(..delay);
        out.truncate(wanted);
        out
    }
    /// Converts the sample rate in place (see [`resampled`](Self::resampled)).
    fn resample(&mut self, from_rate: u32, to_rate: u32)
    where
        Self: SignalResizable,
    {
        let out = self.resampled(from_rate, to_rate);
        self.replace_with(&out);
    }
}

impl<S: Signal + ?Sized> Resample for S {}
#[cfg(test)]
mod tests {
    use super::*;
    use crate::osc::Sine;

    /// Resamples a unit sine and returns the worst error against the ideal sine at the new rate,
    /// after skipping the filter's start-up.
    fn tone_error(freq: f64, input_rate: u32, output_rate: u32) -> f64 {
        let mut rs = Resampler::<f64>::new(input_rate, output_rate);
        let input: Vec<f64> = Sine::new(freq, input_rate as f64).take(input_rate as usize / 2).collect();
        let mut out = vec![0.0; rs.max_output_len(input.len())];
        let n = rs.process(&input, &mut out);
        let delay = rs.delay();
        let skip = 4 * delay.ceil() as usize + 16;
        (skip..n - 16)
            .map(|i| {
                let t = (i as f64 - delay) / output_rate as f64;
                (out[i] - (std::f64::consts::TAU * freq * t).sin()).abs()
            })
            .fold(0.0, f64::max)
    }

    #[test]
    fn reduces_ratios() {
        assert_eq!(Resampler::<f64>::new(48_000, 44_100).ratio(), (147, 160));
        assert_eq!(Resampler::<f64>::new(44_100, 48_000).ratio(), (160, 147));
        assert_eq!(Resampler::<f64>::new(48_000, 96_000).ratio(), (2, 1));
        assert_eq!(Resampler::<f64>::new(48_000, 48_000).ratio(), (1, 1));
    }

    #[test]
    fn delay_is_a_whole_number_of_output_samples() {
        for (from, to) in [(48_000, 44_100), (44_100, 48_000), (48_000, 96_000), (48_000, 16_000), (22_050, 48_000)] {
            let d = Resampler::<f64>::new(from, to).delay();
            assert_eq!(d, d.round(), "{from} -> {to}: delay {d}");
        }
    }

    #[test]
    fn tones_survive_conversion() {
        for (input_rate, output_rate) in [(48_000, 44_100), (44_100, 48_000), (48_000, 96_000), (48_000, 16_000), (48_000, 48_000)] {
            let err = tone_error(1_000.0, input_rate, output_rate);
            assert!(err < 2e-3, "{input_rate} -> {output_rate}: max error {err}");
        }
    }

    #[test]
    fn output_length_tracks_the_ratio_across_blocks() {
        let mut rs = Resampler::<f32>::new(48_000, 44_100);
        let block = vec![0.0f32; 480];
        let mut out = vec![0.0f32; rs.max_output_len(block.len())];
        let total: usize = (0..100).map(|_| rs.process(&block, &mut out)).sum();
        assert_eq!(total, 48_000 * 147 / 160); // exactly 44100 for one second of input
    }

    #[test]
    fn block_size_does_not_change_the_result() {
        // one big call vs uneven blocks (crossing the internal chunk size) must give identical output
        let input: Vec<f64> = Sine::new(1_000.0, 48_000.0).take(5_000).collect();
        let mut whole = Resampler::new(48_000, 44_100);
        let mut expected = vec![0.0; whole.max_output_len(input.len())];
        let n = whole.process(&input, &mut expected);

        let mut split = Resampler::new(48_000, 44_100);
        let mut got = Vec::new();
        let mut scratch = vec![0.0; split.max_output_len(700)];
        for block in input.chunks(1).take(3).chain(input[3..].chunks(700)) {
            let m = split.process(block, &mut scratch);
            got.extend_from_slice(&scratch[..m]);
        }
        assert_eq!(got.len(), n);
        for (a, b) in got.iter().zip(&expected) {
            assert!((a - b).abs() < 1e-12);
        }
    }

    #[test]
    fn baseline_path_matches_dispatched_path() {
        let input: Vec<f32> = Sine::new(1_000.0, 48_000.0).take(2_000).collect();
        let (mut a, mut b) = (Resampler::new(48_000, 44_100), Resampler::new(48_000, 44_100));
        let (mut out_a, mut out_b) = (vec![0.0; a.max_output_len(2_000)], vec![0.0; b.max_output_len(2_000)]);
        let n = a.process(&input, &mut out_a);
        assert_eq!(b.process_block(&input, &mut out_b), n);
        for (x, y) in out_a[..n].iter().zip(&out_b[..n]) {
            assert!((x - y).abs() < 1e-5);
        }
    }

    #[test]
    fn downsampling_removes_content_above_the_new_nyquist() {
        // 10 kHz can't exist at 16 kHz (Nyquist 8 kHz): it must be filtered out, not aliased to 6 kHz.
        let mut rs = Resampler::<f64>::new(48_000, 16_000);
        let input: Vec<f64> = Sine::new(10_000.0, 48_000.0).take(48_000).collect();
        let mut out = vec![0.0; rs.max_output_len(input.len())];
        let n = rs.process(&input, &mut out);
        let peak = out[1_000..n].iter().fold(0.0f64, |m, s| m.max(s.abs()));
        assert!(peak < 1e-3, "aliased energy leaked through: peak {peak}");
    }

    #[test]
    fn resampled_is_aligned_and_sized() {
        use crate::osc::Sine;
        for (from, to) in [(48_000u32, 44_100u32), (44_100, 48_000), (48_000, 16_000), (48_000, 96_000)] {
            let tone: Vec<f64> = Sine::new(1_000.0, from as f64).take(from as usize / 10).collect();
            let out = tone.resampled(from, to);
            assert_eq!(out.len(), (tone.len() * to as usize).div_ceil(from as usize));
            // away from the edges (filter start-up and the flushed tail) it is the same sine at the new rate
            let margin = out.len() / 10;
            for (n, &y) in out.iter().enumerate().take(out.len() - margin).skip(margin) {
                let ideal = (std::f64::consts::TAU * 1_000.0 * n as f64 / to as f64).sin();
                assert!((y - ideal).abs() <= 2e-3, "{from} -> {to} sample {n}: {y} vs {ideal}");
            }
        }
        let mut v: Vec<f64> = vec![0.5; 480];
        v.resample(48_000, 96_000);
        assert_eq!(v.len(), 960);
    }
}