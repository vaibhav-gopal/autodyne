//! [`Oversampled`]: runs any processor at a multiple of the sample rate, so the harmonics a nonlinear one
//! creates above Nyquist are filtered out instead of aliasing back.

use crate::alloc_prelude::*;
use super::Resampler;
use crate::processor::Processor;
use crate::units::*;

/// Runs a processor at `factor` times the sample rate: upsample, process, filter and downsample.
///
/// Nonlinear processors (saturation, clipping, folding) create harmonics above Nyquist that would
/// otherwise alias back into the audible band as inharmonic noise; at the higher rate they have room,
/// and the downsampling filter removes them. The wrapped processor must be built for the higher rate
/// (`sample_rate * factor`). Adds `latency()` samples of delay. Allocates only in `new`.
#[derive(Debug, Clone)]
pub struct Oversampled<P, T: Float> {
    inner: P,
    factor: usize,
    up: Resampler<T>,
    down: Resampler<T>,
    max_block: usize,
    high: Vec<T>,
    low: Vec<T>,
}

impl<P, T: Float> Oversampled<P, T> {
    /// `factor` is typically 2, 4 or 8; blocks longer than `max_block` are processed in pieces.
    /// Panics if `factor` or `max_block` is 0.
    pub fn new(inner: P, factor: usize, max_block: usize) -> Self {
        assert!(factor > 0 && max_block > 0, "factor and max_block must be positive");
        // only the ratio matters to the resamplers, so the rates can be given as 1 and factor
        let f = u32::try_from(factor).expect("oversampling factor fits in u32");
        let up = Resampler::new(1, f);
        let down = Resampler::new(f, 1);
        let high = vec![T::_ZERO; up.max_output_len(max_block)];
        let low = vec![T::_ZERO; down.max_output_len(high.len())];
        Self { inner, factor, up, down, max_block, high, low }
    }
    /// The oversampling factor.
    pub fn factor(&self) -> usize {
        self.factor
    }
    /// Delay added by the up- and down-sampling filters, in samples at the base rate.
    pub fn latency(&self) -> f64 {
        self.up.delay() / self.factor as f64 + self.down.delay()
    }
    /// The wrapped processor.
    pub fn inner(&self) -> &P {
        &self.inner
    }
    /// The wrapped processor, e.g. to change its settings while running.
    pub fn inner_mut(&mut self) -> &mut P {
        &mut self.inner
    }
}

impl<T: Float, P: Processor<T>> Oversampled<P, T> {
    /// Processes `block` in place (also available through `Processor`).
    pub fn process(&mut self, block: &mut [T]) {
        for chunk in block.chunks_mut(self.max_block) {
            let n_high = self.up.process(chunk, &mut self.high);
            self.inner.process(&mut self.high[..n_high]);
            let n_low = self.down.process(&self.high[..n_high], &mut self.low);
            // upsampling by an integer factor gives exactly factor samples per input, and
            // downsampling gives exactly one per factor, so the count always matches the chunk
            debug_assert_eq!(n_low, chunk.len());
            chunk.copy_from_slice(&self.low[..chunk.len()]);
        }
    }
    /// Clears the wrapped processor's and the filters' state.
    pub fn reset(&mut self) {
        self.inner.reset();
        self.up.reset();
        self.down.reset();
    }
}

impl<T: Float, P: Processor<T>> Processor<T> for Oversampled<P, T> {
    fn process(&mut self, block: &mut [T]) {
        Oversampled::process(self, block)
    }
    fn reset(&mut self) {
        Oversampled::reset(self)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::distortion::{Shape, Waveshaper};
    use crate::gain::Gain;
    use crate::osc::Sine;
    use crate::fft::Fft;

    const FS: f64 = 48_000.0;
    const N: usize = 8_192;
    /// a frequency on an exact FFT bin, so its harmonics and their aliases land on exact bins too
    const BIN: usize = 1_195; // 7001.95 Hz

    /// Energy outside the true harmonics, relative to the harmonics, in dB.
    fn alias_db(signal: &[f64]) -> f64 {
        let mut fft = Fft::new(N);
        let mut spectrum = vec![crate::units::Complex::zero(); N];
        fft.forward_real(&signal[..N], &mut spectrum);
        let (mut harmonic, mut alias) = (0.0, 0.0);
        for (k, z) in spectrum.iter().enumerate().take(N / 2).skip(1) {
            // tanh is odd-symmetric: only odd harmonics below Nyquist are genuine
            let is_harmonic = k % BIN == 0 && (k / BIN) % 2 == 1;
            if is_harmonic { harmonic += z.norm_sqr() } else { alias += z.norm_sqr() }
        }
        10.0 * (alias / harmonic).log10()
    }

    fn driven_tone(len: usize) -> Vec<f64> {
        let f = BIN as f64 * FS / N as f64;
        Sine::new(f, FS).with_amplitude(0.9).take(len).collect()
    }

    #[test]
    fn oversampling_removes_most_aliasing() {
        let saturator = |fs: f64| {
            let mut ws = Waveshaper::new(Shape::Tanh, fs);
            ws.set_drive_db(18.0);
            ws.reset();
            ws
        };
        let settle = 4_096;

        let mut plain = driven_tone(settle + N);
        saturator(FS).process(&mut plain);

        let mut over = driven_tone(settle + N);
        Oversampled::new(saturator(4.0 * FS), 4, 512).process(&mut over);

        let (a_plain, a_over) = (alias_db(&plain[settle..]), alias_db(&over[settle..]));
        println!("aliasing relative to the harmonics: {a_plain:.1} dB plain, {a_over:.1} dB with 4x oversampling");
        assert!(a_over < a_plain - 20.0, "aliasing {a_plain:.1} dB without vs {a_over:.1} dB with 4x oversampling");
    }

    #[test]
    fn transparent_inner_gives_the_input_delayed_by_the_latency() {
        let f = 440.0;
        let mut x: Vec<f64> = Sine::new(f, FS).take(9_600).collect();
        let mut ov = Oversampled::new(Gain::new(1.0, 0.0, 4.0 * FS), 4, 256);
        let latency = ov.latency();
        ov.process(&mut x);
        for (n, &y) in x.iter().enumerate().skip(2_000) {
            let ideal = (std::f64::consts::TAU * f * (n as f64 - latency) / FS).sin();
            assert!((y - ideal).abs() < 2e-3, "sample {n}: {y} vs {ideal}");
        }
    }
}
