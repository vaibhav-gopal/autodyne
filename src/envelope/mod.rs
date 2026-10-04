//! Envelope generators.
//!
//! [`Adsr`] shapes a note's level over time: Attack (rise to full level), Decay (fall to the sustain
//! level), Sustain (hold while the note is down), Release (fall to silence after it is let go).
//! Segments are linear with exact timing. Retriggering starts from the current level and release
//! always takes its full time from wherever the level is, so notes never click.
//!
//! Use it as a `Processor` (multiplies audio by the envelope, a VCA) or as a `Source` (produces the
//! envelope itself, e.g. to modulate a filter's frequency).

mod params;

use crate::signal::Source;
use crate::units::*;

/// Where an [`Adsr`] is in its cycle.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Stage {
    /// at 0, waiting for a note
    Idle,
    /// rising to 1
    Attack,
    /// falling to the sustain level
    Decay,
    /// holding the sustain level until note-off
    Sustain,
    /// falling to 0 after note-off
    Release,
}

/// Linear attack / decay / sustain / release envelope, from 0 to 1.
#[derive(Debug, Clone, Copy)]
pub struct Adsr<T: Float> {
    sample_rate: T,
    attack: T,
    decay: T,
    sustain: T,
    release: T,
    stage: Stage,
    level: T,
    /// per-sample fall during release, fixed at note-off so release takes exactly `release` seconds
    release_step: T,
}

impl<T: Float> Adsr<T> {
    /// Times in seconds, `sustain` as a level in [0, 1].
    pub fn new(attack: T, decay: T, sustain: T, release: T, sample_rate: T) -> Self {
        Self {
            sample_rate,
            attack: attack._max(T::_ZERO),
            decay: decay._max(T::_ZERO),
            sustain: sustain._clamp(T::_ZERO, T::_ONE),
            release: release._max(T::_ZERO),
            stage: Stage::Idle,
            level: T::_ZERO,
            release_step: T::_ZERO,
        }
    }
    /// Attack time in seconds.
    pub fn set_attack(&mut self, seconds: T) {
        self.attack = seconds._max(T::_ZERO);
    }
    /// Decay time in seconds.
    pub fn set_decay(&mut self, seconds: T) {
        self.decay = seconds._max(T::_ZERO);
    }
    /// Sustain level, 0..1.
    pub fn set_sustain(&mut self, level: T) {
        self.sustain = level._clamp(T::_ZERO, T::_ONE);
    }
    /// Release time in seconds.
    pub fn set_release(&mut self, seconds: T) {
        self.release = seconds._max(T::_ZERO);
    }
    /// Attack time in seconds.
    pub fn attack(&self) -> T {
        self.attack
    }
    /// Decay time in seconds.
    pub fn decay(&self) -> T {
        self.decay
    }
    /// Sustain level.
    pub fn sustain(&self) -> T {
        self.sustain
    }
    /// Release time in seconds.
    pub fn release(&self) -> T {
        self.release
    }
    /// The current stage.
    pub fn stage(&self) -> Stage {
        self.stage
    }
    /// The current level, 0..1.
    pub fn level(&self) -> T {
        self.level
    }
    /// Whether the envelope is producing anything (not idle).
    pub fn is_active(&self) -> bool {
        self.stage != Stage::Idle
    }
    /// Starts (or restarts) the attack from the current level.
    pub fn note_on(&mut self) {
        self.stage = Stage::Attack;
    }
    /// Starts the release from the current level; it reaches 0 after `release` seconds.
    pub fn note_off(&mut self) {
        if self.stage == Stage::Idle {
            return;
        }
        let samples = self.release * self.sample_rate;
        self.release_step = if samples > T::_ONE { self.level / samples } else { self.level };
        self.stage = Stage::Release;
    }
    /// Silences immediately.
    pub fn reset(&mut self) {
        self.stage = Stage::Idle;
        self.level = T::_ZERO;
    }

    /// Per-sample step to cover `distance` in `seconds` (the whole distance for zero time).
    fn step(&self, distance: T, seconds: T) -> T {
        let samples = seconds * self.sample_rate;
        if samples > T::_ONE { distance / samples } else { distance }
    }

    /// Advances one sample and returns the new level.
    #[inline]
    pub fn next_value(&mut self) -> T {
        // snap to each segment's target within rounding error, so accumulated steps can't overshoot
        // a segment end by a sample
        let eps = T::_lit(1e-9);
        match self.stage {
            Stage::Idle => {}
            Stage::Attack => {
                self.level = self.level + self.step(T::_ONE, self.attack);
                if self.level >= T::_ONE - eps {
                    self.level = T::_ONE;
                    self.stage = Stage::Decay;
                }
            }
            Stage::Decay => {
                self.level = self.level - self.step(T::_ONE - self.sustain, self.decay);
                if self.level <= self.sustain + eps {
                    self.level = self.sustain;
                    self.stage = Stage::Sustain;
                }
            }
            // follows sustain changes while held
            Stage::Sustain => self.level = self.sustain,
            Stage::Release => {
                self.level = self.level - self.release_step;
                if self.level <= eps {
                    self.level = T::_ZERO;
                    self.stage = Stage::Idle;
                }
            }
        }
        self.level
    }

    /// Multiplies `block` by the envelope (a VCA).
    pub fn process(&mut self, block: &mut [T]) {
        for s in block {
            *s = *s * self.next_value();
        }
    }
}

/// The envelope as a signal, e.g. to modulate a filter.
impl<T: Float> Source for Adsr<T> {
    type Sample = T;
    fn next_sample(&mut self) -> T {
        self.next_value()
    }
}

crate::processor::forward_processor!(Adsr);

#[cfg(test)]
mod tests {
    use super::*;

    fn run(env: &mut Adsr<f64>, n: usize) -> Vec<f64> {
        (0..n).map(|_| env.next_value()).collect()
    }

    #[test]
    fn segments_have_exact_timing() {
        // 1000 Hz so seconds are easy to count: 10-sample attack, 20-sample decay to 0.5, 40-sample release
        let mut env = Adsr::new(0.010, 0.020, 0.5, 0.040, 1_000.0);
        assert_eq!(env.next_value(), 0.0, "idle until note on");
        env.note_on();
        let a = run(&mut env, 10);
        assert!((a[4] - 0.5).abs() < 1e-12 && a[9] == 1.0, "{a:?}");
        let d = run(&mut env, 20);
        assert!((d[9] - 0.75).abs() < 1e-12 && d[19] == 0.5);
        assert_eq!(env.stage(), Stage::Sustain);
        assert_eq!(run(&mut env, 100).last(), Some(&0.5));
        env.note_off();
        let r = run(&mut env, 40);
        assert!((r[19] - 0.25).abs() < 1e-12 && r[39] == 0.0);
        assert!(!env.is_active());
    }

    #[test]
    fn early_release_and_retrigger_never_jump() {
        let mut env = Adsr::new(0.010, 0.0, 1.0, 0.010, 1_000.0);
        env.note_on();
        run(&mut env, 4); // part way up: 0.4
        env.note_off();
        let r = run(&mut env, 10);
        assert!((r[0] - 0.36).abs() < 1e-12, "release starts from the current level: {r:?}");
        assert_eq!(r[9], 0.0, "and still takes exactly the release time");

        env.note_on();
        run(&mut env, 5);
        let before = env.level();
        env.note_on(); // retrigger mid-attack
        assert!((env.next_value() - before - 0.1).abs() < 1e-12, "continues from where it was");
    }

    #[test]
    fn zero_times_are_instant_and_vca_multiplies() {
        let mut env = Adsr::new(0.0, 0.0, 0.25, 0.0, 48_000.0);
        env.note_on();
        assert_eq!(env.next_value(), 1.0);
        assert_eq!(env.next_value(), 0.25);
        let mut block = [2.0; 3];
        env.process(&mut block);
        assert_eq!(block, [0.5; 3]);
        env.note_off();
        assert_eq!(env.next_value(), 0.0);
    }
}
