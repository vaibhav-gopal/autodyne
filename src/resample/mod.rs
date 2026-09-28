//! Sample-rate conversion by a rational factor (e.g. 48 kHz -> 44.1 kHz is 147/160).
//!
//! Conceptually: insert L-1 zeros between samples (upsample by L), low-pass to remove the images and
//! anything above the new Nyquist, then keep every M-th sample. The polyphase form computes only the
//! filter taps that land on real input samples, so each output costs about `taps_per_phase`
//! multiply-adds regardless of L.

use crate::filter::design_lowpass;
use crate::units::*;

fn gcd(a: u64, b: u64) -> u64 {
    if b == 0 { a } else { gcd(b, a % b) }
}

/// Streaming rational resampler. Allocates in `new`; `process` does not.
#[derive(Debug, Clone)]
pub struct Resampler<T: Float> {
    up: usize,
    down: usize,
    /// taps per polyphase branch
    branch_len: usize,
    /// branch p holds prototype taps h[p], h[p + L], h[p + 2L], ... (zero padded): poly[p * branch_len + k]
    poly: Vec<T>,
    /// ring of the last `branch_len` input samples
    history: Vec<T>,
    newest: usize,
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
        let g = gcd(input_rate as u64, output_rate as u64);
        let (up, down) = ((output_rate as u64 / g) as usize, (input_rate as u64 / g) as usize);

        // Prototype runs at input_rate * up. Its cutoff sits below the lower of the two Nyquists,
        // and its length scales with max(up, down) so the transition band stays a fixed fraction of it.
        let fs_up = input_rate as f64 * up as f64;
        let cutoff = 0.45 * input_rate.min(output_rate) as f64;
        let len = taps_per_phase * up.max(down);
        let mut proto: Vec<T> = design_lowpass(T::_lit(cutoff), T::_lit(fs_up), len.max(2));
        // zero stuffing divides the signal level by L; the filter makes it back up
        proto.iter_mut().for_each(|h| *h = *h * T::_lit(up as f64));

        let branch_len = proto.len().div_ceil(up);
        let mut poly = vec![T::_ZERO; up * branch_len];
        for (i, &h) in proto.iter().enumerate() {
            poly[(i % up) * branch_len + i / up] = h;
        }
        Self {
            up,
            down,
            branch_len,
            poly,
            history: vec![T::_ZERO; branch_len],
            newest: branch_len - 1,
            phase: 0,
            prototype_delay: (proto.len() - 1) as f64 / 2.0,
        }
    }

    /// (L, M): output_rate / input_rate reduced to lowest terms.
    pub fn ratio(&self) -> (usize, usize) {
        (self.up, self.down)
    }

    /// Filter delay, in output samples.
    pub fn delay(&self) -> f64 {
        self.prototype_delay / self.down as f64
    }

    /// Upper bound on how many samples `process` can write for `input_len` input samples.
    pub fn max_output_len(&self, input_len: usize) -> usize {
        (input_len * self.up).div_ceil(self.down) + 1
    }

    pub fn reset(&mut self) {
        self.history.iter_mut().for_each(|s| *s = T::_ZERO);
        self.phase = 0;
    }

    /// Resamples `input` into the front of `out` and returns how many samples were written.
    /// Output count varies block to block (it averages input.len() * L / M); size `out` with
    /// `max_output_len`. Panics if `out` is too small.
    pub fn process(&mut self, input: &[T], out: &mut [T]) -> usize {
        assert!(out.len() >= self.max_output_len(input.len()), "output buffer too small; use max_output_len");
        let mut written = 0;
        for &x in input {
            self.newest = if self.newest + 1 == self.branch_len { 0 } else { self.newest + 1 };
            self.history[self.newest] = x;
            // Every output whose upsampled position falls in [newest * L, newest * L + L) needs no newer input.
            while self.phase < self.up {
                let branch = &self.poly[self.phase * self.branch_len..][..self.branch_len];
                let (newer, older) = self.history.split_at(self.newest + 1);
                let recent = newer.iter().rev().chain(older.iter().rev());
                out[written] = branch.iter().zip(recent).fold(T::_ZERO, |acc, (&h, &s)| acc + h * s);
                written += 1;
                self.phase += self.down;
            }
            self.phase -= self.up;
        }
        written
    }
}

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
    fn downsampling_removes_content_above_the_new_nyquist() {
        // 10 kHz can't exist at 16 kHz (Nyquist 8 kHz): it must be filtered out, not aliased to 6 kHz.
        let mut rs = Resampler::<f64>::new(48_000, 16_000);
        let input: Vec<f64> = Sine::new(10_000.0, 48_000.0).take(48_000).collect();
        let mut out = vec![0.0; rs.max_output_len(input.len())];
        let n = rs.process(&input, &mut out);
        let peak = out[1_000..n].iter().fold(0.0f64, |m, s| m.max(s.abs()));
        assert!(peak < 1e-3, "aliased energy leaked through: peak {peak}");
    }
}
