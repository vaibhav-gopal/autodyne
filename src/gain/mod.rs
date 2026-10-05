//! Levels: [`Gain`], constant-power panning ([`Panner`]), mid/side stereo width ([`StereoWidth`]),
//! and click-free parameter changes.
//!
//! Jumping a gain (or any parameter) from one value to another between two samples produces an audible
//! click; [`SmoothedValue`] ramps to each new target over a fixed time instead. Decibel conversions
//! are `units::{db_to_gain, gain_to_db}`.
//!
//! tend: Audio / gain

mod params;

use crate::channels::{AudioBuffer, MultiProcessor};
use crate::units::*;

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
    /// The value now (mid-ramp while smoothing).
    pub fn current(&self) -> T {
        self.current
    }
    /// The value being ramped to.
    pub fn target(&self) -> T {
        self.target
    }
    /// Whether a ramp is in progress.
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
    /// Linear gain to ramp to.
    pub fn set_gain(&mut self, gain: T) {
        self.gain.set_target(gain);
    }
    /// Gain in dB to ramp to.
    pub fn set_gain_db(&mut self, db: T) {
        self.gain.set_target(db_to_gain(db));
    }
    /// Target linear gain.
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

// STEREO ==========================================================================================

/// Mid/side stereo width: 0 = mono, 1 = unchanged, above 1 = wider. Needs a 2-channel buffer.
#[derive(Debug, Clone, Copy)]
pub struct StereoWidth<T: Float> {
    width: SmoothedValue<T>,
}

impl<T: Float> StereoWidth<T> {
    /// A width stage; changes ramp over 20 ms.
    pub fn new(width: T, sample_rate: T) -> Self {
        Self { width: SmoothedValue::new(width).with_ramp_seconds(T::_lit(0.02), sample_rate) }
    }
    /// Width, at least 0 (0 = mono, 1 = unchanged).
    pub fn set_width(&mut self, width: T) {
        self.width.set_target(width._max(T::_ZERO));
    }
    /// Target width.
    pub fn width(&self) -> T {
        self.width.target()
    }
}

impl<T: Float> MultiProcessor<T> for StereoWidth<T> {
    fn process(&mut self, buffer: &mut AudioBuffer<T>) {
        let half = T::_lit(0.5);
        let (left, right) = buffer.stereo_mut();
        for (l, r) in left.iter_mut().zip(right.iter_mut()) {
            let (mid, side) = ((*l + *r) * half, (*l - *r) * half * self.width.next_value());
            *l = mid + side;
            *r = mid - side;
        }
    }
    fn reset(&mut self) {
        let w = self.width.target();
        self.width.set_immediate(w);
    }
}

/// Places a mono signal in a stereo field with the constant-power pan law: -1 = hard left, 0 = center
/// (each side at -3 dB), 1 = hard right. The total power stays the same wherever it is panned.
#[derive(Debug, Clone, Copy)]
pub struct Panner<T: Float> {
    position: SmoothedValue<T>,
}

impl<T: Float> Panner<T> {
    /// A panner at `position` (-1..1); changes ramp over 20 ms.
    pub fn new(position: T, sample_rate: T) -> Self {
        Self { position: SmoothedValue::new(position._clamp(-T::_ONE, T::_ONE)).with_ramp_seconds(T::_lit(0.02), sample_rate) }
    }
    /// Pan position, -1 (left) .. 1 (right).
    pub fn set_position(&mut self, position: T) {
        self.position.set_target(position._clamp(-T::_ONE, T::_ONE));
    }
    /// Target pan position.
    pub fn position(&self) -> T {
        self.position.target()
    }
    /// (left gain, right gain) for a pan position.
    pub fn gains(position: T) -> (T, T) {
        let angle = (position + T::_ONE) * T::_PI / T::_lit(4.0);
        let (s, c) = angle._sin_cos();
        (c, s)
    }
    /// Writes `mono` panned into a stereo buffer (overwriting it; frames set to `mono.len()`).
    pub fn process(&mut self, mono: &[T], out: &mut AudioBuffer<T>) {
        out.set_frames(mono.len());
        let (left, right) = out.stereo_mut();
        for ((l, r), &x) in left.iter_mut().zip(right.iter_mut()).zip(mono) {
            let (gl, gr) = Self::gains(self.position.next_value());
            *l = x * gl;
            *r = x * gr;
        }
    }
}

crate::processor::forward_processor!(Gain);

#[cfg(test)]
mod tests {
    use super::*;

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

    const FS: f64 = 48_000.0;

    fn stereo_noise(frames: usize) -> AudioBuffer<f64> {
        let mut buf = AudioBuffer::new(2, frames);
        crate::osc::Noise::new(1).fill(buf.channel_mut(0));
        crate::osc::Noise::new(2).fill(buf.channel_mut(1));
        buf
    }

    #[test]
    fn width_zero_is_mono_and_one_is_transparent() {
        let original = stereo_noise(256);
        let mut same = original.clone();
        StereoWidth::new(1.0, FS).process(&mut same);
        assert_eq!(same.channel(0), original.channel(0));
        assert_eq!(same.channel(1), original.channel(1));

        let mut mono = original.clone();
        StereoWidth::new(0.0, FS).process(&mut mono);
        for f in 0..256 {
            let mid = (original.channel(0)[f] + original.channel(1)[f]) / 2.0;
            assert!((mono.channel(0)[f] - mid).abs() < 1e-15 && (mono.channel(1)[f] - mid).abs() < 1e-15);
        }
    }

    #[test]
    fn constant_power_pan_law() {
        for p in [-1.0, -0.5, 0.0, 0.3, 1.0] {
            let (l, r) = Panner::gains(p);
            assert!((l * l + r * r - 1.0f64).abs() < 1e-12, "power at {p}");
        }
        let (l, r) = Panner::gains(0.0f64);
        assert!((l - std::f64::consts::FRAC_1_SQRT_2).abs() < 1e-12 && (l - r).abs() < 1e-12, "center is -3 dB each side");
        let (l, r) = Panner::gains(-1.0f64);
        assert!((l - 1.0).abs() < 1e-12 && r.abs() < 1e-12, "hard left");

        let mut out = AudioBuffer::new(2, 4);
        Panner::new(1.0, FS).process(&[1.0; 4], &mut out);
        assert!(out.channel(0).iter().all(|s| s.abs() < 1e-12) && out.channel(1).iter().all(|&s| (s - 1.0).abs() < 1e-12));
    }
}
