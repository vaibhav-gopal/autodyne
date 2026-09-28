//! Signal generators: oscillators, noise and impulses.
//!
//! Every generator can be used two ways:
//! - as an infinite `Iterator` (lazy, composes with the std adapters: `sine.take(n).zip(noise)...`)
//! - as a block processor: `fill` overwrites a buffer, `add_to` mixes into one.
//!   Neither allocates, so both are safe to call from a real-time audio callback.

use crate::units::*;

/// Implements the block-processing methods for a generator that has `fn next_sample(&mut self) -> T`.
macro_rules! impl_generator_blocks {
    ($Gen:ident) => {
        impl<T: Float> $Gen<T> {
            /// Overwrites `out` with the next `out.len()` samples.
            pub fn fill(&mut self, out: &mut [T]) {
                for s in out {
                    *s = self.next_sample();
                }
            }
            /// Adds the next `out.len()` samples onto `out` (mixing).
            pub fn add_to(&mut self, out: &mut [T]) {
                for s in out {
                    *s = *s + self.next_sample();
                }
            }
        }

        impl<T: Float> Iterator for $Gen<T> {
            type Item = T;
            fn next(&mut self) -> Option<T> {
                Some(self.next_sample())
            }
            fn size_hint(&self) -> (usize, Option<usize>) {
                (usize::MAX, None)
            }
        }
    };
}

// SINE ============================================================================================

/// Sine oscillator using a phase accumulator.
/// Phase is kept in cycles and wrapped to [0, 1), so precision doesn't degrade over long runs.
#[derive(Debug, Clone, Copy)]
pub struct Sine<T: Float> {
    /// current phase in cycles, [0, 1)
    phase: T,
    /// phase increment per sample: frequency / sample_rate
    increment: T,
    amplitude: T,
}

impl<T: Float> Sine<T> {
    pub fn new(frequency: T, sample_rate: T) -> Self {
        Self { phase: T::_ZERO, increment: frequency / sample_rate, amplitude: T::_ONE }
    }
    pub fn with_amplitude(mut self, amplitude: T) -> Self {
        self.amplitude = amplitude;
        self
    }
    /// Starting phase in cycles (0.25 = start at the peak, i.e. a cosine).
    pub fn with_phase(mut self, phase: T) -> Self {
        self.phase = phase._fract();
        self
    }
    /// Changes frequency without resetting phase, so there is no click.
    pub fn set_frequency(&mut self, frequency: T, sample_rate: T) {
        self.increment = frequency / sample_rate;
    }
    pub fn set_amplitude(&mut self, amplitude: T) {
        self.amplitude = amplitude;
    }
    pub fn reset(&mut self) {
        self.phase = T::_ZERO;
    }
    #[inline]
    pub fn next_sample(&mut self) -> T {
        let out = self.amplitude * (T::_TAU * self.phase)._sin();
        self.phase = self.phase + self.increment;
        if self.phase >= T::_ONE {
            self.phase = self.phase - T::_ONE;
        }
        out
    }
}

impl_generator_blocks!(Sine);

// NOISE ===========================================================================================

/// Uniform white noise in [-amplitude, amplitude).
/// Uses xorshift64* seeded through splitmix64: fast, allocation-free and deterministic per seed,
/// which keeps tests reproducible. Not cryptographic.
#[derive(Debug, Clone, Copy)]
pub struct Noise<T: Float> {
    state: u64,
    amplitude: T,
}

impl<T: Float> Noise<T> {
    pub fn new(seed: u64) -> Self {
        // splitmix64 spreads similar seeds apart and never yields the all-zero state xorshift can't leave.
        let mut z = seed.wrapping_add(0x9E37_79B9_7F4A_7C15);
        z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
        z ^= z >> 31;
        Self { state: if z == 0 { 0x9E37_79B9_7F4A_7C15 } else { z }, amplitude: T::_ONE }
    }
    pub fn with_amplitude(mut self, amplitude: T) -> Self {
        self.amplitude = amplitude;
        self
    }
    pub fn set_amplitude(&mut self, amplitude: T) {
        self.amplitude = amplitude;
    }
    #[inline]
    fn next_u64(&mut self) -> u64 {
        let mut x = self.state;
        x ^= x >> 12;
        x ^= x << 25;
        x ^= x >> 27;
        self.state = x;
        x.wrapping_mul(0x2545_F491_4F6C_DD1D)
    }
    #[inline]
    pub fn next_sample(&mut self) -> T {
        // top 53 bits -> uniform f64 in [0, 1) -> [-1, 1)
        let unit = (self.next_u64() >> 11) as f64 * (1.0 / (1u64 << 53) as f64);
        self.amplitude * T::_lit(unit * 2.0 - 1.0)
    }
}

impl_generator_blocks!(Noise);

// IMPULSE =========================================================================================

/// A single sample of `amplitude` followed by silence (the Kronecker delta).
/// Feeding it through a filter gives that filter's impulse response.
#[derive(Debug, Clone, Copy)]
pub struct Impulse<T: Float> {
    fired: bool,
    amplitude: T,
}

impl<T: Float> Impulse<T> {
    pub fn new() -> Self {
        Self { fired: false, amplitude: T::_ONE }
    }
    pub fn with_amplitude(mut self, amplitude: T) -> Self {
        self.amplitude = amplitude;
        self
    }
    /// Re-arms the impulse so the next sample fires again.
    pub fn reset(&mut self) {
        self.fired = false;
    }
    #[inline]
    pub fn next_sample(&mut self) -> T {
        if self.fired {
            T::_ZERO
        } else {
            self.fired = true;
            self.amplitude
        }
    }
}

impl<T: Float> Default for Impulse<T> {
    fn default() -> Self {
        Self::new()
    }
}

impl_generator_blocks!(Impulse);

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn sine_matches_closed_form() {
        let (f, fs) = (1000.0, 48_000.0);
        let mut osc = Sine::new(f, fs);
        for n in 0..4800 {
            let expected = (f64::_TAU * f * n as f64 / fs).sin();
            assert!((osc.next_sample() - expected).abs() < 1e-9, "sample {n}");
        }
    }

    #[test]
    fn sine_stays_accurate_over_long_runs() {
        // 10 s at 48 kHz: the wrapped phase accumulator must not drift audibly.
        let (f, fs) = (440.0, 48_000.0);
        let mut osc = Sine::new(f, fs);
        let n = 480_000;
        let last = osc.by_ref().nth(n).unwrap();
        let expected = (f64::_TAU * (f * n as f64 / fs).fract()).sin();
        assert!((last - expected).abs() < 1e-6);
    }

    #[test]
    fn sine_amplitude_phase_and_f32() {
        let mut osc = Sine::<f32>::new(1.0, 4.0).with_amplitude(0.5).with_phase(0.25);
        let got: Vec<f32> = osc.by_ref().take(4).collect();
        let expected = [0.5, 0.0, -0.5, 0.0];
        for (g, e) in got.iter().zip(expected) {
            assert!((g - e).abs() < 1e-6, "{got:?}");
        }
    }

    #[test]
    fn noise_is_bounded_centered_and_deterministic() {
        let samples: Vec<f64> = Noise::new(42).take(100_000).collect();
        assert!(samples.iter().all(|s| (-1.0..1.0).contains(s)));
        let mean = samples.iter().sum::<f64>() / samples.len() as f64;
        assert!(mean.abs() < 0.01, "mean {mean}");

        let again: Vec<f64> = Noise::new(42).take(1000).collect();
        assert_eq!(&samples[..1000], &again[..]);
        let other: Vec<f64> = Noise::new(43).take(1000).collect();
        assert_ne!(again, other);
    }

    #[test]
    fn impulse_fires_once() {
        let mut imp = Impulse::<f64>::new();
        let got: Vec<f64> = imp.by_ref().take(4).collect();
        assert_eq!(got, [1.0, 0.0, 0.0, 0.0]);
        imp.reset();
        assert_eq!(imp.next_sample(), 1.0);
    }

    #[test]
    fn fill_and_add_to() {
        let mut buf = [0.0f64; 4];
        Impulse::new().with_amplitude(2.0).fill(&mut buf);
        assert_eq!(buf, [2.0, 0.0, 0.0, 0.0]);
        Impulse::new().add_to(&mut buf);
        assert_eq!(buf, [3.0, 0.0, 0.0, 0.0]);
    }
}
