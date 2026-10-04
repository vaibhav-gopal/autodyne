//! Transient shaper.

use super::{time_coeff, GainComputer};
use crate::gain::SmoothedValue;
use crate::units::*;

/// Boosts or cuts the attacks and the sustain of sounds, independently of their level.
///
/// Two pairs of envelope followers compare a fast and a slow view of the signal: while a sound
/// starts, a fast-attack follower runs ahead of a slow-attack one (their ratio marks the attack);
/// while it decays, a slow-release follower lags behind a fast-release one (their ratio marks the
/// sustain). Each ratio, in dB, times its amount gives the gain: +1 doubles the attack (or
/// sustain) in dB, -1 removes it. There is no threshold, so loud and quiet hits are shaped alike.
/// For stereo, wrap it in [`Linked`](crate::channels::Linked).
#[derive(Debug, Clone, Copy)]
pub struct TransientShaper<T: Float> {
    sample_rate: T,
    attack_amount: T,
    sustain_amount: T,
    output_db: T,
    output: SmoothedValue<T>,
    /// (attack, release) coefficients of the four followers
    coeffs: [(T, T); 4],
    /// fast attack, slow attack, fast release, slow release
    env: [T; 4],
}

/// (attack, release) in seconds of the four followers.
const FOLLOWERS: [(f64, f64); 4] = [(0.0005, 0.1), (0.02, 0.1), (0.0005, 0.03), (0.0005, 0.25)];
/// Most gain change either way, dB.
const LIMIT_DB: f64 = 24.0;

impl<T: Float> TransientShaper<T> {
    /// Neutral: both amounts 0, so the signal passes unchanged.
    pub fn new(sample_rate: T) -> Self {
        let coeffs = FOLLOWERS.map(|(a, r)| (time_coeff(T::_lit(a), sample_rate), time_coeff(T::_lit(r), sample_rate)));
        Self {
            sample_rate,
            attack_amount: T::_ZERO,
            sustain_amount: T::_ZERO,
            output_db: T::_ZERO,
            output: SmoothedValue::new(T::_ONE).with_ramp_seconds(T::_lit(0.02), sample_rate),
            coeffs,
            env: [T::_ZERO; 4],
        }
    }
    pub fn sample_rate(&self) -> T {
        self.sample_rate
    }
    /// -1 (softer attacks) .. 1 (harder).
    pub fn set_attack(&mut self, amount: T) {
        self.attack_amount = amount._clamp(-T::_ONE, T::_ONE);
    }
    pub fn attack(&self) -> T {
        self.attack_amount
    }
    /// -1 (drier, shorter tails) .. 1 (longer).
    pub fn set_sustain(&mut self, amount: T) {
        self.sustain_amount = amount._clamp(-T::_ONE, T::_ONE);
    }
    pub fn sustain(&self) -> T {
        self.sustain_amount
    }
    /// Output gain; changes ramp over 20 ms.
    pub fn set_output_db(&mut self, db: T) {
        self.output_db = db;
        self.output.set_target(db_to_gain(db));
    }
    pub fn output_db(&self) -> T {
        self.output_db
    }
    pub fn reset(&mut self) {
        self.env = [T::_ZERO; 4];
        self.output.set_immediate(self.output.target());
    }
    /// Advances one sample given the detector level (a linear peak) and returns the gain.
    #[inline]
    pub fn gain_for_level(&mut self, level: T) -> T {
        for (env, &(attack, release)) in self.env.iter_mut().zip(&self.coeffs) {
            let coeff = if level > *env { attack } else { release };
            *env = (level + (*env - level) * coeff)._flush_denormal();
        }
        let floor = T::_lit(1e-9);
        let ratio_db = |a: T, b: T| T::_lit(20.0) * ((a + floor) / (b + floor))._log10();
        let attack_db = ratio_db(self.env[0], self.env[1]);
        let sustain_db = ratio_db(self.env[3], self.env[2]);
        let limit = T::_lit(LIMIT_DB);
        let db = (self.attack_amount * attack_db + self.sustain_amount * sustain_db)._clamp(-limit, limit);
        self.output.next_value() * db_to_gain(db)
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

impl<T: Float> GainComputer<T> for TransientShaper<T> {
    fn gain_for_level(&mut self, level: T) -> T {
        TransientShaper::gain_for_level(self, level)
    }
    fn reset(&mut self) {
        TransientShaper::reset(self)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::osc::Noise;

    const FS: f64 = 48_000.0;

    /// Noise hits decaying over 80 ms, at two levels, far enough apart for the detectors to settle.
    fn hits() -> Vec<f64> {
        let mut noise = Noise::<f64>::new(7);
        let mut x = vec![0.0; 120_000];
        for (start, level) in [(2_400, 0.8), (98_400, 0.05)] {
            for (i, s) in x[start..start + 19_200].iter_mut().enumerate() {
                *s = level * (-(i as f64) / (0.08 * FS)).exp() * noise.next_sample();
            }
        }
        x
    }

    fn energy_db(x: &[f64]) -> f64 {
        10.0 * x.iter().map(|s| s * s).sum::<f64>().log10()
    }

    /// (attack portion, tail portion) of each hit, in dB.
    fn shape(x: &[f64]) -> [(f64, f64); 2] {
        [2_400, 98_400].map(|s| (energy_db(&x[s..s + 240]), energy_db(&x[s + 9_600..s + 14_400])))
    }

    #[test]
    fn neutral_is_transparent() {
        let x = hits();
        let mut y = x.clone();
        TransientShaper::new(FS).process(&mut y);
        assert_eq!(x, y);
    }

    #[test]
    fn shapes_attack_and_sustain_at_any_level() {
        let x = hits();
        let before = shape(&x);
        for (attack, sustain) in [(1.0, 0.0), (-1.0, 0.0), (0.0, 1.0), (0.0, -1.0)] {
            let mut ts = TransientShaper::new(FS);
            ts.set_attack(attack);
            ts.set_sustain(sustain);
            let mut y = x.clone();
            ts.process(&mut y);
            let after = shape(&y);
            for (hit, ((a0, t0), (a1, t1))) in before.iter().zip(&after).enumerate() {
                // the attack-to-tail balance moves the requested way, by the same amount for both hits
                let change = (a1 - t1) - (a0 - t0);
                let wanted = if attack != 0.0 { attack } else { -sustain };
                assert!(change * wanted > 3.0, "attack {attack} sustain {sustain} hit {hit}: balance moved {change} dB");
            }
            let (c0, c1) = ((after[0].0 - after[0].1) - (before[0].0 - before[0].1), (after[1].0 - after[1].1) - (before[1].0 - before[1].1));
            assert!((c0 - c1).abs() < 1.0, "level independent: {c0} vs {c1}");
        }
    }
}
