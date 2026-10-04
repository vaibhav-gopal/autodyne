//! Bit-depth and sample-rate reduction.

use crate::gain::SmoothedValue;
use crate::units::*;

/// Lo-fi degradation: quantizes to fewer bits and holds samples at a lower rate (with the aliasing
/// that brings, on purpose), blended with the dry signal.
///
/// Bits may be fractional (8.5 bits sits between 8 and 9). Dither (triangular, one step wide)
/// trades the quantization's harsh, signal-dependent distortion for steady noise. The rate need not
/// divide the sample rate: a phase accumulator decides when to take the next sample.
#[derive(Debug, Clone, Copy)]
pub struct Bitcrusher<T: Float> {
    sample_rate: T,
    bits: T,
    step: T,
    rate_hz: T,
    increment: T,
    dither: bool,
    mix: T,
    mix_smoothed: SmoothedValue<T>,
    phase: T,
    held: T,
    noise: u64,
}

impl<T: Float> Bitcrusher<T> {
    /// 8 bits at a quarter of the sample rate, no dither, fully wet.
    pub fn new(sample_rate: T) -> Self {
        let mut b = Self {
            sample_rate,
            bits: T::_ZERO,
            step: T::_ZERO,
            rate_hz: T::_ZERO,
            increment: T::_ZERO,
            dither: false,
            mix: T::_ONE,
            mix_smoothed: SmoothedValue::new(T::_ONE).with_ramp_seconds(T::_lit(0.02), sample_rate),
            phase: T::_ONE,
            held: T::_ZERO,
            noise: 0x9E37_79B9_7F4A_7C15,
        };
        b.set_bits(T::_lit(8.0));
        b.set_rate(sample_rate / T::_lit(4.0));
        b
    }
    /// The sample rate in Hz.
    pub fn sample_rate(&self) -> T {
        self.sample_rate
    }
    /// Bit depth, 1 to 24 (fractional allowed): the signal is rounded to steps of 2^(1 - bits).
    pub fn set_bits(&mut self, bits: T) {
        self.bits = bits._clamp(T::_ONE, T::_lit(24.0));
        self.step = (T::_ONE - self.bits)._exp2();
    }
    /// Bit depth.
    pub fn bits(&self) -> T {
        self.bits
    }
    /// Rate at which samples are taken and held, in Hz (up to the sample rate: no reduction).
    pub fn set_rate(&mut self, hz: T) {
        self.rate_hz = hz._clamp(T::_ONE, self.sample_rate);
        self.increment = self.rate_hz / self.sample_rate;
    }
    /// Sample-and-hold rate in Hz.
    pub fn rate(&self) -> T {
        self.rate_hz
    }
    /// Adds triangular dither before quantizing (noise instead of distortion at low bit depths).
    pub fn set_dither(&mut self, on: bool) {
        self.dither = on;
    }
    /// Whether dither is on.
    pub fn dither(&self) -> bool {
        self.dither
    }
    /// Dry / wet, 0..1; changes ramp over 20 ms.
    pub fn set_mix(&mut self, mix: T) {
        self.mix = mix._clamp(T::_ZERO, T::_ONE);
        self.mix_smoothed.set_target(self.mix);
    }
    /// Dry / wet, 0..1.
    pub fn mix(&self) -> T {
        self.mix
    }
    /// Clears the held sample and finishes the mix ramp.
    pub fn reset(&mut self) {
        self.phase = T::_ONE;
        self.held = T::_ZERO;
        self.mix_smoothed.set_immediate(self.mix);
    }

    /// Uniform in [-0.5, 0.5) (xorshift).
    #[inline]
    fn uniform(&mut self) -> T {
        self.noise ^= self.noise << 13;
        self.noise ^= self.noise >> 7;
        self.noise ^= self.noise << 17;
        T::_lit((self.noise >> 11) as f64 / (1u64 << 53) as f64 - 0.5)
    }

    #[inline]
    fn quantize(&mut self, x: T) -> T {
        let dither = if self.dither { self.uniform() + self.uniform() } else { T::_ZERO };
        ((x / self.step + dither)._round() * self.step)._clamp(-T::_ONE, T::_ONE)
    }

    /// Processes one sample.
    #[inline]
    pub fn process_sample(&mut self, x: T) -> T {
        if self.phase >= T::_ONE {
            self.phase = self.phase - self.phase._floor();
            self.held = self.quantize(x);
        }
        self.phase = self.phase + self.increment;
        let mix = self.mix_smoothed.next_value();
        x + (self.held - x) * mix
    }
    /// Processes `block` in place.
    pub fn process(&mut self, block: &mut [T]) {
        for s in block {
            *s = self.process_sample(*s);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    const FS: f64 = 48_000.0;

    fn ramp(n: usize) -> Vec<f64> {
        (0..n).map(|i| -0.9 + 1.8 * i as f64 / n as f64).collect()
    }

    #[test]
    fn quantizes_to_the_bit_depth() {
        let mut b = Bitcrusher::new(FS);
        b.set_rate(FS);
        b.set_bits(4.0);
        let mut y = ramp(1_000);
        b.process(&mut y);
        let step = 1.0 / 8.0;
        assert!(y.iter().all(|v| ((v / step) - (v / step).round()).abs() < 1e-12), "multiples of 1/8");
        let mut distinct: Vec<i64> = y.iter().map(|v| (v / step).round() as i64).collect();
        distinct.dedup();
        assert_eq!(distinct.len(), 15, "-7/8 .. 7/8");
    }

    #[test]
    fn holds_samples_at_the_reduced_rate() {
        let mut b = Bitcrusher::new(FS);
        b.set_bits(24.0);
        b.set_rate(FS / 4.0);
        let x = ramp(400);
        let mut y = x.clone();
        b.process(&mut y);
        for chunk in y.chunks(4) {
            assert!(chunk.iter().all(|&v| v == chunk[0]), "held for 4 samples: {chunk:?}");
        }
        assert!((y[0] - x[0]).abs() < 1e-6 && (y[4] - x[4]).abs() < 1e-6, "each hold starts on a fresh sample");
    }

    #[test]
    fn dither_decorrelates_the_error_and_mix_blends() {
        // a slow ramp spanning 3 steps at 3 bits: without dither the error is a sawtooth that
        // follows the signal; with dither its average over each step is near zero
        let x: Vec<f64> = (0..30_000).map(|i| 0.1 + 0.6 * i as f64 / 30_000.0).collect();
        let mean_error = |dither: bool| {
            let mut b = Bitcrusher::new(FS);
            b.set_rate(FS);
            b.set_bits(3.0);
            b.set_dither(dither);
            let mut y = x.clone();
            b.process(&mut y);
            // error averaged over windows of 1000 samples: the worst window
            y.chunks(1_000).zip(x.chunks(1_000)).map(|(a, b)| (a.iter().zip(b).map(|(p, q)| p - q).sum::<f64>() / 1_000.0).abs()).fold(0.0, f64::max)
        };
        assert!(mean_error(false) > 0.05);
        assert!(mean_error(true) < 0.02, "{}", mean_error(true));
        let mut b = Bitcrusher::new(FS);
        b.set_mix(0.0);
        b.reset();
        let x = ramp(100);
        let mut y = x.clone();
        b.process(&mut y);
        assert_eq!(x, y, "dry");
    }
}