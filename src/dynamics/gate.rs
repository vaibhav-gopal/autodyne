//! Noise gate and downward expander.

use super::{time_coeff, GainComputer};
use crate::gain::{db_to_gain, gain_to_db};
use crate::units::*;

/// Detector release: fast, so the gate closes promptly; the hold time rides over the dips between
/// a low note's waveform peaks.
const DETECTOR_RELEASE: f64 = 0.005;

/// Noise gate / downward expander: turns quiet passages further down.
///
/// Below the threshold the signal is reduced by `ratio` (2 = every dB below the threshold becomes
/// 2 dB, a gentle expander; infinite = a gate), but never by more than `range`. Hysteresis keeps a
/// signal hovering around the threshold from chattering: once open, the gate only closes below
/// `threshold - hysteresis`, and only after the `hold` time. Attack sets how fast it opens,
/// release how fast it closes. For stereo, wrap it in [`Linked`](crate::channels::Linked).
#[derive(Debug, Clone, Copy)]
pub struct Gate<T: Float> {
    sample_rate: T,
    threshold_db: T,
    ratio: T,
    range_db: T,
    hysteresis_db: T,
    attack_seconds: T,
    hold_seconds: T,
    release_seconds: T,
    attack_coeff: T,
    release_coeff: T,
    detector_coeff: T,
    hold_samples: usize,
    level: T,
    open: bool,
    hold_left: usize,
    /// smoothed gain in dB (<= 0)
    gain_db: T,
}

impl<T: Float> Gate<T> {
    /// A gate: threshold -40 dB, infinite ratio, -80 dB range, 3 dB hysteresis, 0.5 ms attack,
    /// 20 ms hold, 100 ms release.
    pub fn new(sample_rate: T) -> Self {
        let lit = T::_lit;
        let mut g = Self {
            sample_rate,
            threshold_db: lit(-40.0),
            ratio: T::_INFINITY,
            range_db: lit(-80.0),
            hysteresis_db: lit(3.0),
            attack_seconds: T::_ZERO,
            hold_seconds: T::_ZERO,
            release_seconds: T::_ZERO,
            attack_coeff: T::_ZERO,
            release_coeff: T::_ZERO,
            detector_coeff: time_coeff(lit(DETECTOR_RELEASE), sample_rate),
            hold_samples: 0,
            level: T::_ZERO,
            open: false,
            hold_left: 0,
            gain_db: T::_ZERO,
        };
        g.set_attack(lit(0.0005));
        g.set_hold(lit(0.02));
        g.set_release(lit(0.1));
        g.gain_db = g.range_db;
        g
    }
    /// A downward expander: `ratio` dB out per dB below the threshold (no hold, no hysteresis).
    pub fn expander(threshold_db: T, ratio: T, sample_rate: T) -> Self {
        let mut g = Self::new(sample_rate);
        g.set_threshold_db(threshold_db);
        g.set_ratio(ratio);
        g.set_hysteresis_db(T::_ZERO);
        g.set_hold(T::_ZERO);
        g.set_release(T::_lit(0.05));
        g
    }
    pub fn set_threshold_db(&mut self, db: T) {
        self.threshold_db = db;
    }
    pub fn threshold_db(&self) -> T {
        self.threshold_db
    }
    /// Expansion ratio, at least 1 (no effect); `T::_INFINITY` gates.
    pub fn set_ratio(&mut self, ratio: T) {
        self.ratio = ratio._max(T::_ONE);
    }
    pub fn ratio(&self) -> T {
        self.ratio
    }
    /// The most the gain goes down, in dB (<= 0).
    pub fn set_range_db(&mut self, db: T) {
        self.range_db = db._min(T::_ZERO);
    }
    pub fn range_db(&self) -> T {
        self.range_db
    }
    pub fn set_hysteresis_db(&mut self, db: T) {
        self.hysteresis_db = db._max(T::_ZERO);
    }
    pub fn hysteresis_db(&self) -> T {
        self.hysteresis_db
    }
    pub fn set_attack(&mut self, seconds: T) {
        self.attack_seconds = seconds._max(T::_ZERO);
        self.attack_coeff = time_coeff(self.attack_seconds, self.sample_rate);
    }
    pub fn attack(&self) -> T {
        self.attack_seconds
    }
    /// How long the gate stays open after the level falls below the closing threshold.
    pub fn set_hold(&mut self, seconds: T) {
        self.hold_seconds = seconds._max(T::_ZERO);
        self.hold_samples = (self.hold_seconds * self.sample_rate).to_f64().unwrap_or(0.0).round() as usize;
    }
    pub fn hold(&self) -> T {
        self.hold_seconds
    }
    pub fn set_release(&mut self, seconds: T) {
        self.release_seconds = seconds._max(T::_ZERO);
        self.release_coeff = time_coeff(self.release_seconds, self.sample_rate);
    }
    pub fn release(&self) -> T {
        self.release_seconds
    }
    /// Whether the gate is open (passing the signal).
    pub fn is_open(&self) -> bool {
        self.open
    }
    /// Current gain reduction in dB (positive = turning down).
    pub fn gain_reduction_db(&self) -> T {
        -self.gain_db
    }
    pub fn reset(&mut self) {
        self.level = T::_ZERO;
        self.open = false;
        self.hold_left = 0;
        self.gain_db = self.range_db;
    }
    /// Advances one sample given the detector level (a linear peak) and returns the gain.
    #[inline]
    pub fn gain_for_level(&mut self, level: T) -> T {
        // peak detector: instant rise, short release
        self.level = if level > self.level { level } else { level + (self.level - level) * self.detector_coeff };
        let level_db = gain_to_db(self.level._max(T::_lit(1e-10)));
        if level_db >= self.threshold_db {
            self.open = true;
            self.hold_left = self.hold_samples;
        } else if level_db < self.threshold_db - self.hysteresis_db {
            if self.hold_left > 0 {
                self.hold_left -= 1;
            } else {
                self.open = false;
            }
        }
        let target = if self.open {
            T::_ZERO
        } else {
            let below = level_db - self.threshold_db; // negative
            let reduction = if self.ratio._is_finite() { below * (self.ratio - T::_ONE) } else { self.range_db };
            reduction._max(self.range_db)._min(T::_ZERO)
        };
        let coeff = if target > self.gain_db { self.attack_coeff } else { self.release_coeff };
        self.gain_db = (target + (self.gain_db - target) * coeff)._flush_denormal();
        db_to_gain(self.gain_db)
    }
    #[inline]
    pub fn process_sample(&mut self, x: T) -> T {
        x * self.gain_for_level(x._abs())
    }
    pub fn process(&mut self, block: &mut [T]) {
        for s in block {
            *s = self.process_sample(*s);
        }
    }
}

impl<T: Float> GainComputer<T> for Gate<T> {
    fn gain_for_level(&mut self, level: T) -> T {
        Gate::gain_for_level(self, level)
    }
    fn reset(&mut self) {
        Gate::reset(self)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::osc::{Noise, Sine};

    const FS: f64 = 48_000.0;

    fn rms_db(x: &[f64]) -> f64 {
        10.0 * (x.iter().map(|s| s * s).sum::<f64>() / x.len() as f64).log10()
    }

    #[test]
    fn gates_the_gaps_between_notes() {
        // 100 ms tone bursts at -10 dB with -60 dB noise between them
        let mut x: Vec<f64> = Noise::<f64>::new(3).take(48_000).map(|n| 1e-3 * n).collect();
        let mut tone = Sine::new(440.0, FS);
        for burst in [4_800, 19_200, 33_600] {
            for s in &mut x[burst..burst + 4_800] {
                *s += 0.3 * tone.next_sample();
            }
        }
        let mut gate = Gate::new(FS);
        gate.set_release(0.02);
        let mut y = x.clone();
        gate.process(&mut y);
        // the tone passes at full level once open
        let tone_in = rms_db(&x[20_000..24_000]);
        assert!((rms_db(&y[20_000..24_000]) - tone_in).abs() < 0.05);
        // the gaps (after the hold and most of the release) are down by nearly the range
        let gap = rms_db(&y[15_000..19_000]) - rms_db(&x[15_000..19_000]);
        assert!(gap < -70.0, "gap reduced by {gap} dB");
        assert!(!gate.is_open());
    }

    #[test]
    fn hysteresis_and_hold_prevent_chatter() {
        // a tone whose level wobbles 2 dB around the threshold: with 3 dB hysteresis the gate
        // opens once and stays open
        let mut gate = Gate::new(FS);
        gate.set_threshold_db(-20.0);
        let mut opened = 0;
        let mut was_open = false;
        for i in 0..96_000 {
            let t = i as f64 / FS;
            let level_db = -20.0 + 2.0 * (std::f64::consts::TAU * 3.0 * t).sin();
            let x = 10f64.powf(level_db / 20.0) * (std::f64::consts::TAU * 220.0 * t).sin();
            gate.process_sample(x);
            if gate.is_open() && !was_open {
                opened += 1;
            }
            was_open = gate.is_open();
        }
        assert_eq!(opened, 1);
    }

    #[test]
    fn expander_scales_the_level_below_the_threshold() {
        // a steady tone 10 dB under the threshold comes out 10 dB lower still at 2:1
        let mut exp = Gate::expander(-20.0, 2.0, FS);
        let x: Vec<f64> = Sine::new(1_000.0, FS).take(48_000).map(|s| s * 10f64.powf(-30.0 / 20.0)).collect();
        let mut y = x.clone();
        exp.process(&mut y);
        let change = rms_db(&y[24_000..]) - rms_db(&x[24_000..]);
        assert!((change + 10.0).abs() < 0.3, "{change} dB");
        // above the threshold: untouched
        let mut exp = Gate::expander(-20.0, 2.0, FS);
        let loud: Vec<f64> = Sine::new(1_000.0, FS).take(9_600).map(|s| 0.5 * s).collect();
        let mut y = loud.clone();
        exp.process(&mut y);
        assert!((rms_db(&y[4_800..]) - rms_db(&loud[4_800..])).abs() < 1e-6);
    }
}
