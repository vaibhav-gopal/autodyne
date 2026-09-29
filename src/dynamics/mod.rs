//! Dynamics: envelope follower, compressor and limiter.
//!
//! The compressor is feed-forward: it measures the input level, decides how many dB to turn it down
//! (the static curve: threshold, ratio, soft knee), smooths that gain reduction with separate attack
//! and release times, and applies it. Smoothing happens on the gain in dB, not on the signal, which
//! keeps attack and release independent of level.

use crate::gain::{db_to_gain, gain_to_db};
use crate::units::*;

/// One-pole coefficient for a time constant: after `seconds`, a step response has covered
/// 1 - 1/e (~63%) of the way. Zero or negative times mean "instant" (coefficient 0).
fn time_coeff<T: Float>(seconds: T, sample_rate: T) -> T {
    if seconds <= T::_ZERO {
        T::_ZERO
    } else {
        (-T::_ONE / (seconds * sample_rate))._exp()
    }
}

// ENVELOPE FOLLOWER ===============================================================================

/// Tracks the peak level of a signal: rises with the attack time, falls with the release time.
#[derive(Debug, Clone, Copy)]
pub struct EnvelopeFollower<T: Float> {
    sample_rate: T,
    attack: T,
    release: T,
    envelope: T,
}

impl<T: Float> EnvelopeFollower<T> {
    pub fn new(attack_seconds: T, release_seconds: T, sample_rate: T) -> Self {
        Self {
            sample_rate,
            attack: time_coeff(attack_seconds, sample_rate),
            release: time_coeff(release_seconds, sample_rate),
            envelope: T::_ZERO,
        }
    }
    pub fn set_attack(&mut self, seconds: T) {
        self.attack = time_coeff(seconds, self.sample_rate);
    }
    pub fn set_release(&mut self, seconds: T) {
        self.release = time_coeff(seconds, self.sample_rate);
    }
    pub fn envelope(&self) -> T {
        self.envelope
    }
    pub fn reset(&mut self) {
        self.envelope = T::_ZERO;
    }
    /// Feeds one sample and returns the updated envelope.
    #[inline]
    pub fn process_sample(&mut self, x: T) -> T {
        let level = x._abs();
        let coeff = if level > self.envelope { self.attack } else { self.release };
        self.envelope = coeff * self.envelope + (T::_ONE - coeff) * level;
        self.envelope
    }
    /// Replaces each sample of `block` with the envelope at that point.
    pub fn process(&mut self, block: &mut [T]) {
        for s in block {
            *s = self.process_sample(*s);
        }
    }
}

// COMPRESSOR ======================================================================================

/// Feed-forward compressor with a soft knee. Also a limiter: see `Compressor::limiter`.
#[derive(Debug, Clone, Copy)]
pub struct Compressor<T: Float> {
    sample_rate: T,
    threshold_db: T,
    /// 1 / ratio: output dB per input dB above the threshold (0 for a limiter)
    slope: T,
    knee_db: T,
    makeup_db: T,
    attack: T,
    release: T,
    /// smoothed gain reduction in dB (>= 0)
    reduction_db: T,
}

impl<T: Float> Compressor<T> {
    /// A general-purpose starting point: threshold -18 dB, ratio 4:1, 6 dB knee, 10 ms attack,
    /// 100 ms release, no makeup gain.
    pub fn new(sample_rate: T) -> Self {
        Self {
            sample_rate,
            threshold_db: T::_lit(-18.0),
            slope: T::_lit(0.25),
            knee_db: T::_lit(6.0),
            makeup_db: T::_ZERO,
            attack: time_coeff(T::_lit(0.010), sample_rate),
            release: time_coeff(T::_lit(0.100), sample_rate),
            reduction_db: T::_ZERO,
        }
    }
    /// Brickwall limiter: infinite ratio, hard knee and instant attack, so no output sample exceeds
    /// `threshold_db`. (There is no lookahead, so gain changes land on the loud sample itself.)
    pub fn limiter(threshold_db: T, release_seconds: T, sample_rate: T) -> Self {
        let mut c = Self::new(sample_rate);
        c.set_threshold_db(threshold_db);
        c.slope = T::_ZERO;
        c.set_knee_db(T::_ZERO);
        c.set_attack(T::_ZERO);
        c.set_release(release_seconds);
        c
    }
    pub fn set_threshold_db(&mut self, db: T) {
        self.threshold_db = db;
    }
    /// Input dB above the threshold per output dB; values below 1 are treated as 1 (no compression).
    /// `T::_INFINITY` makes a limiter.
    pub fn set_ratio(&mut self, ratio: T) {
        self.slope = T::_ONE / ratio._max(T::_ONE);
    }
    /// Width of the soft knee in dB, centered on the threshold; 0 = hard knee.
    pub fn set_knee_db(&mut self, db: T) {
        self.knee_db = db._max(T::_ZERO);
    }
    pub fn set_makeup_db(&mut self, db: T) {
        self.makeup_db = db;
    }
    pub fn set_attack(&mut self, seconds: T) {
        self.attack = time_coeff(seconds, self.sample_rate);
    }
    pub fn set_release(&mut self, seconds: T) {
        self.release = time_coeff(seconds, self.sample_rate);
    }
    /// Current gain reduction in dB (positive = turning down), e.g. for a meter.
    pub fn gain_reduction_db(&self) -> T {
        self.reduction_db
    }
    pub fn reset(&mut self) {
        self.reduction_db = T::_ZERO;
    }
    /// The static curve: steady-state output level for a given input level, before makeup gain.
    pub fn output_level_db(&self, input_db: T) -> T {
        let (t, w, two) = (self.threshold_db, self.knee_db, T::_lit(2.0));
        let over = input_db - t;
        if two * over < -w {
            input_db // below the knee: untouched
        } else if w > T::_ZERO && two * over._abs() <= w {
            // inside the knee: a quadratic that blends smoothly between the two straight lines
            let x = over + w / two;
            input_db + (self.slope - T::_ONE) * x * x / (two * w)
        } else {
            t + over * self.slope // above the knee: the ratio applies
        }
    }
    /// Advances one sample given the detector level (a linear peak, >= 0) and returns the linear gain
    /// to apply, makeup included. `process_sample` feeds it |x|; linked multichannel compression feeds
    /// it the loudest channel so every channel gets the same gain.
    #[inline]
    pub fn gain_for_level(&mut self, level: T) -> T {
        // floor at -200 dB so silence gives a finite level (log of 0 is -infinity)
        let level_db = gain_to_db(level._max(T::_lit(1e-10)));
        let target = level_db - self.output_level_db(level_db);
        let coeff = if target > self.reduction_db { self.attack } else { self.release };
        self.reduction_db = coeff * self.reduction_db + (T::_ONE - coeff) * target;
        db_to_gain(self.makeup_db - self.reduction_db)
    }
    #[inline]
    pub fn process_sample(&mut self, x: T) -> T {
        x * self.gain_for_level(x._abs())
    }
    /// Compresses `block` in place.
    pub fn process(&mut self, block: &mut [T]) {
        for s in block {
            *s = self.process_sample(*s);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::osc::Sine;

    const FS: f64 = 48_000.0;

    fn assert_close(a: f64, b: f64, tol: f64, what: &str) {
        assert!((a - b).abs() <= tol, "{what}: {a} vs {b} (tol {tol})");
    }

    #[test]
    fn follower_attack_and_release_hit_63_percent_at_their_time_constants() {
        let mut f = EnvelopeFollower::new(0.010, 0.100, FS); // 480 and 4800 samples
        let mut up = vec![1.0; 480];
        f.process(&mut up);
        assert_close(up[479], 1.0 - (-1.0f64).exp(), 1e-9, "attack");

        let mut f = EnvelopeFollower::new(0.0, 0.100, FS);
        f.process_sample(1.0); // instant attack: envelope is now exactly 1
        assert_eq!(f.envelope(), 1.0);
        let mut down = vec![0.0; 4800];
        f.process(&mut down);
        assert_close(down[4799], (-1.0f64).exp(), 1e-9, "release");
    }

    #[test]
    fn static_curve_hard_knee() {
        let mut c = Compressor::new(FS);
        c.set_threshold_db(-20.0);
        c.set_ratio(4.0);
        c.set_knee_db(0.0);
        assert_eq!(c.output_level_db(-30.0), -30.0);
        assert_eq!(c.output_level_db(-20.0), -20.0);
        assert_eq!(c.output_level_db(-8.0), -17.0); // 12 dB over / 4 = 3 dB over
    }

    #[test]
    fn soft_knee_is_continuous_and_dips_below_threshold() {
        let mut c = Compressor::new(FS);
        c.set_threshold_db(-20.0);
        c.set_ratio(4.0);
        c.set_knee_db(10.0);
        // the knee meets both straight lines at its edges
        assert_close(c.output_level_db(-25.0), -25.0, 1e-12, "lower knee edge");
        assert_close(c.output_level_db(-15.0), -18.75, 1e-12, "upper knee edge");
        // at the threshold the knee is already compressing: T - (1 - 1/R) * W / 8
        assert_close(c.output_level_db(-20.0), -20.9375, 1e-12, "threshold");
        // and the curve never decreases
        let mut prev = f64::MIN;
        for i in 0..400 {
            let y = c.output_level_db(-40.0 + i as f64 * 0.1);
            assert!(y >= prev);
            prev = y;
        }
    }

    #[test]
    fn steady_state_matches_the_static_curve_plus_makeup() {
        let mut c = Compressor::new(FS);
        c.set_threshold_db(-20.0);
        c.set_ratio(4.0);
        c.set_knee_db(0.0);
        c.set_makeup_db(3.0);
        let mut block = vec![0.5; 48_000]; // constant level: -6.02 dB
        c.process(&mut block);
        let expected_db = c.output_level_db(gain_to_db(0.5)) + 3.0;
        assert_close(gain_to_db(block[47_999]), expected_db, 1e-6, "settled level");
        assert_close(c.gain_reduction_db(), gain_to_db(0.5) - c.output_level_db(gain_to_db(0.5)), 1e-6, "meter");
    }

    #[test]
    fn quiet_signals_pass_through() {
        let mut c = Compressor::new(FS); // threshold -18 dB, knee down to -21 dB
        let input: Vec<f64> = Sine::new(440.0, FS).with_amplitude(0.01).take(4_800).collect(); // -40 dB
        let mut out = input.clone();
        c.process(&mut out);
        for (a, b) in input.iter().zip(&out) {
            assert_close(*a, *b, 1e-12, "below threshold");
        }
    }

    #[test]
    fn limiter_never_exceeds_its_ceiling() {
        let ceiling = db_to_gain(-6.0);
        let mut lim = Compressor::limiter(-6.0, 0.05, FS);
        let mut block: Vec<f64> = Sine::new(100.0, FS).take(48_000).collect(); // 0 dBFS peaks
        lim.process(&mut block);
        let peak = block.iter().fold(0.0f64, |m, s| m.max(s.abs()));
        assert!(peak <= ceiling + 1e-12, "peak {peak} over ceiling {ceiling}");
        assert!(peak > 0.99 * ceiling, "limiter should reach the ceiling, peak {peak}");
    }

    #[test]
    fn reset_clears_gain_reduction() {
        let mut c = Compressor::new(FS);
        c.process(&mut [1.0; 1_000]);
        assert!(c.gain_reduction_db() > 0.0);
        c.reset();
        assert_eq!(c.gain_reduction_db(), 0.0);
    }
}
