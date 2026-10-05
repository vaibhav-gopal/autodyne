//! Dynamics: envelope follower, compressor, lookahead limiter, gate / expander, transient shaper.
//!
//! The compressor is feed-forward: it measures the input level, decides how many dB to turn it down
//! (the static curve: threshold, ratio, soft knee), smooths that gain reduction with separate attack
//! and release times, and applies it. Smoothing happens on the gain in dB, not on the signal, which
//! keeps attack and release independent of level.
//!
//! - [`Compressor`] (also a no-lookahead limiter), [`Gate`] (gate and downward expander) and
//!   [`TransientShaper`] are mono and [`GainComputer`]s: wrap them in
//!   [`Linked`] to drive every channel with one gain.
//! - [`LookaheadLimiter`] (with true-peak detection) and [`MultibandCompressor`] (Linkwitz-Riley
//!   bands) are multichannel (linked) by nature.
//!
//! tend: Audio / dynamics

mod params;
mod gate;
mod limiter;
mod multiband;
mod transient;

pub use gate::*;
pub use limiter::*;
pub use multiband::*;
pub use transient::*;

use crate::channels::{AudioBuffer, Linked, MultiProcessor};
use crate::gain::SmoothedValue;
use crate::units::*;

/// A level-driven gain stage: given a detector level (the loudest channel's, for linked
/// multichannel use), the gain to apply. [`Linked`] runs any of them
/// across the channels of a buffer.
pub trait GainComputer<T: Float> {
    /// Advances one sample at detector `level` (a linear peak, >= 0) and returns the linear gain.
    fn gain_for_level(&mut self, level: T) -> T;
    /// Clears the detector and gain state.
    fn reset(&mut self);
}

/// Linked dynamics (compressor, gate, transient shaper, ...): the loudest channel drives one gain
/// applied to all channels.
impl<T: Float, G: GainComputer<T>> MultiProcessor<T> for Linked<G> {
    fn process(&mut self, buffer: &mut AudioBuffer<T>) {
        buffer.apply_frame_gain(|peak| self.0.gain_for_level(peak));
    }
    fn reset(&mut self) {
        self.0.reset()
    }
}

/// One-pole coefficient for a time constant: after `seconds`, a step response has covered
/// 1 - 1/e (~63%) of the way. Zero or negative times mean "instant" (coefficient 0).
pub fn time_coeff<T: Real>(seconds: T, sample_rate: T) -> T {
    // branch-free: e^(-1/0) is already 0, and non-positive times select 0
    let zero = T::lit(0.0);
    T::select(seconds.greater(zero), (-T::lit(1.0) / (seconds * sample_rate)).exp(), zero)
}

/// One envelope-follower step over [`Real`]: rises toward the level with `attack_coeff`, falls
/// with `release_coeff` (`select`, not a branch, so it traces).
#[inline(always)]
pub fn envelope_step<T: Real>(attack_coeff: T, release_coeff: T, envelope: T, x: T) -> T {
    let level = x.abs();
    let coeff = T::select(level.greater(envelope), attack_coeff, release_coeff);
    coeff * envelope + (T::lit(1.0) - coeff) * level
}

/// The compressor's static curve and gain smoothing over [`Real`] (so it also traces and
/// differentiates in `flux`).
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct CompressorCurve<T> {
    /// Threshold in dB.
    pub threshold_db: T,
    /// 1 / ratio
    pub slope: T,
    /// Soft knee width in dB (0: hard).
    pub knee_db: T,
    /// One-pole coefficient of the attack (see [`time_coeff`]).
    pub attack_coeff: T,
    /// One-pole coefficient of the release.
    pub release_coeff: T,
}

impl<T: Real> CompressorCurve<T> {
    /// Steady-state output level for an input level, before makeup gain: untouched below the knee,
    /// a quadratic inside it, the ratio above it.
    pub fn output_level_db(&self, input_db: T) -> T {
        let (t, w, two, zero) = (self.threshold_db, self.knee_db, T::lit(2.0), T::lit(0.0));
        let over = input_db - t;
        let above = t + over * self.slope;
        // a zero-width knee never selects the quadratic; keep its division finite anyway
        let x = over + w / two;
        let knee = input_db + (self.slope - T::lit(1.0)) * x * x / (two * w.maximum(T::lit(1e-12)));
        let upper = T::select((two * over).abs().greater(w), above, knee);
        let upper = T::select(w.greater(zero), upper, above);
        T::select((two * over).less(-w), input_db, upper)
    }
    /// One step at detector `level` (a linear peak, >= 0) from gain reduction `reduction_db`:
    /// returns the next reduction and the linear gain (makeup not included).
    #[inline]
    pub fn tick(&self, reduction_db: T, level: T) -> (T, T) {
        // floor at -200 dB so silence gives a finite level (log of 0 is -infinity)
        let level_db = gain_to_db(level.maximum(T::lit(1e-10)));
        let target = level_db - self.output_level_db(level_db);
        let coeff = T::select(target.greater(reduction_db), self.attack_coeff, self.release_coeff);
        let reduction = coeff * reduction_db + (T::lit(1.0) - coeff) * target;
        (reduction, db_to_gain(-reduction))
    }
}

// ENVELOPE FOLLOWER ===============================================================================

/// Tracks the peak level of a signal: rises with the attack time, falls with the release time.
#[derive(Debug, Clone, Copy)]
pub struct EnvelopeFollower<T: Float> {
    sample_rate: T,
    attack_seconds: T,
    release_seconds: T,
    attack_coeff: T,
    release_coeff: T,
    envelope: T,
}

impl<T: Float> EnvelopeFollower<T> {
    /// A follower with attack and release times in seconds.
    pub fn new(attack_seconds: T, release_seconds: T, sample_rate: T) -> Self {
        Self {
            sample_rate,
            attack_seconds,
            release_seconds,
            attack_coeff: time_coeff(attack_seconds, sample_rate),
            release_coeff: time_coeff(release_seconds, sample_rate),
            envelope: T::_ZERO,
        }
    }
    /// Rise time constant in seconds.
    pub fn set_attack(&mut self, seconds: T) {
        self.attack_seconds = seconds;
        self.attack_coeff = time_coeff(seconds, self.sample_rate);
    }
    /// Fall time constant in seconds.
    pub fn set_release(&mut self, seconds: T) {
        self.release_seconds = seconds;
        self.release_coeff = time_coeff(seconds, self.sample_rate);
    }
    /// Attack time in seconds.
    pub fn attack(&self) -> T {
        self.attack_seconds
    }
    /// Release time in seconds.
    pub fn release(&self) -> T {
        self.release_seconds
    }
    /// The current envelope (a linear level).
    pub fn envelope(&self) -> T {
        self.envelope
    }
    /// Sets the envelope to zero.
    pub fn reset(&mut self) {
        self.envelope = T::_ZERO;
    }
    /// Feeds one sample and returns the updated envelope.
    #[inline]
    pub fn process_sample(&mut self, x: T) -> T {
        self.envelope = envelope_step(self.attack_coeff, self.release_coeff, self.envelope, x)._flush_denormal();
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
    /// makeup as a linear gain, ramped per sample so changes never click
    makeup: SmoothedValue<T>,
    attack_seconds: T,
    release_seconds: T,
    attack_coeff: T,
    release_coeff: T,
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
            makeup: SmoothedValue::new(T::_ONE).with_ramp_seconds(T::_lit(0.02), sample_rate),
            attack_seconds: T::_lit(0.010),
            release_seconds: T::_lit(0.100),
            attack_coeff: time_coeff(T::_lit(0.010), sample_rate),
            release_coeff: time_coeff(T::_lit(0.100), sample_rate),
            reduction_db: T::_ZERO,
        }
    }
    /// Brickwall limiter: infinite ratio, hard knee and instant attack, so no output sample exceeds
    /// `threshold_db`. (There is no lookahead, so gain changes land on the loud sample itself.)
    pub fn limiter(threshold_db: T, release_seconds: T, sample_rate: T) -> Self {
        let mut c = Self::new(sample_rate);
        c.set_threshold_db(threshold_db);
        c.set_slope(T::_ZERO);
        c.set_knee_db(T::_ZERO);
        c.set_attack(T::_ZERO);
        c.set_release(release_seconds);
        c
    }
    /// Threshold in dB above which the level is reduced.
    pub fn set_threshold_db(&mut self, db: T) {
        self.threshold_db = db;
    }
    /// Input dB above the threshold per output dB; values below 1 are treated as 1 (no compression).
    /// `T::_INFINITY` makes a limiter.
    pub fn set_ratio(&mut self, ratio: T) {
        self.slope = T::_ONE / ratio._max(T::_ONE);
    }
    /// Output dB per input dB above the threshold (1 / ratio), clamped to [0, 1]: 1 = no compression,
    /// 0.25 = 4:1, 0 = limiting. Unlike the ratio this is always finite, so it is what hosts see.
    pub fn set_slope(&mut self, slope: T) {
        self.slope = slope._clamp(T::_ZERO, T::_ONE);
    }
    /// Output dB per input dB above the threshold (1 / ratio).
    pub fn slope(&self) -> T {
        self.slope
    }
    /// Width of the soft knee in dB, centered on the threshold; 0 = hard knee.
    pub fn set_knee_db(&mut self, db: T) {
        self.knee_db = db._max(T::_ZERO);
    }
    /// Output gain after compression; changes ramp over 20 ms.
    pub fn set_makeup_db(&mut self, db: T) {
        self.makeup_db = db;
        self.makeup.set_target(db_to_gain(db));
    }
    /// Attack time constant in seconds (0: instant).
    pub fn set_attack(&mut self, seconds: T) {
        self.attack_seconds = seconds;
        self.attack_coeff = time_coeff(seconds, self.sample_rate);
    }
    /// Release time constant in seconds.
    pub fn set_release(&mut self, seconds: T) {
        self.release_seconds = seconds;
        self.release_coeff = time_coeff(seconds, self.sample_rate);
    }
    /// Threshold in dB.
    pub fn threshold_db(&self) -> T {
        self.threshold_db
    }
    /// The ratio; infinite for a limiter.
    pub fn ratio(&self) -> T {
        if self.slope == T::_ZERO { T::_INFINITY } else { T::_ONE / self.slope }
    }
    /// Knee width in dB.
    pub fn knee_db(&self) -> T {
        self.knee_db
    }
    /// Makeup gain in dB.
    pub fn makeup_db(&self) -> T {
        self.makeup_db
    }
    /// Attack time in seconds.
    pub fn attack(&self) -> T {
        self.attack_seconds
    }
    /// Release time in seconds.
    pub fn release(&self) -> T {
        self.release_seconds
    }
    /// Current gain reduction in dB (positive = turning down), e.g. for a meter.
    pub fn gain_reduction_db(&self) -> T {
        self.reduction_db
    }
    /// Releases all gain reduction and finishes the makeup ramp.
    pub fn reset(&mut self) {
        self.reduction_db = T::_ZERO;
        self.makeup.set_immediate(self.makeup.target());
    }
    /// The static curve: steady-state output level for a given input level, before makeup gain.
    pub fn output_level_db(&self, input_db: T) -> T {
        self.curve().output_level_db(input_db)
    }
    /// The static curve and smoothing (for driving the same compressor from generic or traced code).
    pub fn curve(&self) -> CompressorCurve<T> {
        CompressorCurve { threshold_db: self.threshold_db, slope: self.slope, knee_db: self.knee_db, attack_coeff: self.attack_coeff, release_coeff: self.release_coeff }
    }
    /// Advances one sample given the detector level (a linear peak, >= 0) and returns the linear gain
    /// to apply, makeup included. `process_sample` feeds it |x|; linked multichannel compression feeds
    /// it the loudest channel so every channel gets the same gain.
    #[inline]
    pub fn gain_for_level(&mut self, level: T) -> T {
        let (reduction, gain) = self.curve().tick(self.reduction_db, level);
        self.reduction_db = reduction;
        self.makeup.next_value() * gain
    }
    /// Processes one sample.
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

impl<T: Float> GainComputer<T> for Compressor<T> {
    fn gain_for_level(&mut self, level: T) -> T {
        Compressor::gain_for_level(self, level)
    }
    fn reset(&mut self) {
        Compressor::reset(self)
    }
}

crate::processor::forward_processor!(Compressor, EnvelopeFollower);

#[cfg(test)]
mod tests {
    use super::*;
    use crate::testing::assert_close;
    use crate::osc::Sine;

    const FS: f64 = 48_000.0;

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
