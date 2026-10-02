//! Levels and click-free parameter changes.
//!
//! Jumping a gain (or any parameter) from one value to another between two samples produces an audible
//! click; `SmoothedValue` ramps to each new target over a fixed time instead.

use crate::units::*;

/// Decibels to linear amplitude: 0 dB -> 1.0, +6.02 dB -> 2.0, -20 dB -> 0.1.
pub fn db_to_gain<T: Real>(db: T) -> T {
    T::lit(10.0).powf(db / T::lit(20.0))
}

/// Linear amplitude to decibels; 0.0 gives negative infinity.
pub fn gain_to_db<T: Real>(gain: T) -> T {
    T::lit(20.0) * gain.log10()
}

/// A value that moves linearly to each new target over a fixed number of samples
/// (the same idea as JUCE's `LinearSmoothedValue`). Call `next_value` once per sample.
#[derive(Debug, Clone, Copy)]
pub struct SmoothedValue<T: Float> {
    current: T,
    target: T,
    step: T,
    remaining: usize,
    ramp_samples: usize,
}

impl<T: Float> SmoothedValue<T> {
    /// Starts settled at `initial` with no ramp (changes apply immediately until a ramp length is set).
    pub fn new(initial: T) -> Self {
        Self { current: initial, target: initial, step: T::_ZERO, remaining: 0, ramp_samples: 0 }
    }
    /// Ramp length in samples for future `set_target` calls.
    pub fn with_ramp_samples(mut self, samples: usize) -> Self {
        self.ramp_samples = samples;
        self
    }
    /// Ramp length in seconds for future `set_target` calls (rounded to whole samples).
    pub fn with_ramp_seconds(self, seconds: T, sample_rate: T) -> Self {
        let samples = (seconds * sample_rate)._round().to_usize().unwrap_or(0);
        self.with_ramp_samples(samples)
    }
    /// Starts ramping from the current value towards `target`.
    pub fn set_target(&mut self, target: T) {
        if target == self.target {
            return;
        }
        self.target = target;
        if self.ramp_samples == 0 {
            self.set_immediate(target);
            return;
        }
        self.remaining = self.ramp_samples;
        self.step = (target - self.current) / T::_lit(self.ramp_samples as f64);
    }
    /// Jumps straight to `value` (e.g. on reset, when there is no audio to click).
    pub fn set_immediate(&mut self, value: T) {
        self.current = value;
        self.target = value;
        self.remaining = 0;
    }
    /// Advances one sample and returns the new value.
    #[inline]
    pub fn next_value(&mut self) -> T {
        if self.remaining > 0 {
            self.remaining -= 1;
            // land exactly on the target instead of accumulating rounding error
            self.current = if self.remaining == 0 { self.target } else { self.current + self.step };
        }
        self.current
    }
    pub fn current(&self) -> T {
        self.current
    }
    pub fn target(&self) -> T {
        self.target
    }
    pub fn is_smoothing(&self) -> bool {
        self.remaining > 0
    }
}

/// Multiplies a signal by a gain that ramps smoothly when changed.
#[derive(Debug, Clone, Copy)]
pub struct Gain<T: Float> {
    gain: SmoothedValue<T>,
}

impl<T: Float> Gain<T> {
    /// `ramp_seconds` is how long gain changes take; 0.02 (20 ms) is a common click-free choice.
    pub fn new(initial_gain: T, ramp_seconds: T, sample_rate: T) -> Self {
        Self { gain: SmoothedValue::new(initial_gain).with_ramp_seconds(ramp_seconds, sample_rate) }
    }
    pub fn set_gain(&mut self, gain: T) {
        self.gain.set_target(gain);
    }
    pub fn set_gain_db(&mut self, db: T) {
        self.gain.set_target(db_to_gain(db));
    }
    pub fn gain(&self) -> T {
        self.gain.target()
    }
    /// Finishes any ramp in progress immediately (the only state a gain has).
    pub fn reset(&mut self) {
        let target = self.gain.target();
        self.gain.set_immediate(target);
    }
    /// Applies the gain to `block` in place.
    pub fn process(&mut self, block: &mut [T]) {
        if self.gain.is_smoothing() {
            for s in block {
                *s = *s * self.gain.next_value();
            }
        } else {
            let g = self.gain.current();
            for s in block {
                *s = *s * g;
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn decibel_conversions() {
        assert_eq!(db_to_gain(0.0), 1.0);
        assert!((db_to_gain(-20.0f64) - 0.1).abs() < 1e-12);
        assert!((db_to_gain(20.0 * 2f64.log10()) - 2.0).abs() < 1e-12);
        for g in [0.001f64, 0.5, 1.0, 3.0] {
            assert!((db_to_gain(gain_to_db(g)) - g).abs() < 1e-12);
        }
        assert_eq!(gain_to_db(0.0f64), f64::NEG_INFINITY);
    }

    #[test]
    fn smoother_ramps_linearly_and_lands_exactly() {
        let mut v = SmoothedValue::new(0.0).with_ramp_samples(4);
        v.set_target(1.0);
        let ramp: Vec<f64> = (0..6).map(|_| v.next_value()).collect();
        assert_eq!(ramp, [0.25, 0.5, 0.75, 1.0, 1.0, 1.0]);
        assert!(!v.is_smoothing());
    }

    #[test]
    fn retargeting_mid_ramp_starts_from_the_current_value() {
        let mut v = SmoothedValue::new(0.0).with_ramp_samples(4);
        v.set_target(1.0);
        v.next_value();
        v.next_value(); // now at 0.5
        v.set_target(0.0);
        let ramp: Vec<f64> = (0..4).map(|_| v.next_value()).collect();
        assert_eq!(ramp, [0.375, 0.25, 0.125, 0.0]);
    }

    #[test]
    fn no_ramp_means_immediate() {
        let mut v = SmoothedValue::new(1.0f32);
        v.set_target(3.0);
        assert_eq!(v.next_value(), 3.0);
        let mut v = SmoothedValue::new(1.0).with_ramp_seconds(0.001, 48_000.0);
        v.set_target(2.0);
        assert_eq!((0..48).map(|_| v.next_value()).last(), Some(2.0)); // 1 ms at 48 kHz = 48 samples
        assert!(!v.is_smoothing());
    }

    #[test]
    fn gain_processor_ramps_then_holds() {
        // 4-sample ramp: 4 samples / 48 kHz
        let mut g = Gain::new(1.0, 4.0 / 48_000.0, 48_000.0);
        g.set_gain(0.5);
        let mut buf = [1.0f64; 6];
        g.process(&mut buf);
        assert_eq!(buf, [0.875, 0.75, 0.625, 0.5, 0.5, 0.5]);
        g.set_gain_db(0.0);
        assert_eq!(g.gain(), 1.0);
    }
}
