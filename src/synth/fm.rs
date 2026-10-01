use super::{midi_to_hz, Voice};
use crate::envelope::Adsr;
use crate::units::*;

/// Operators per [`FmVoice`].
pub const OPERATORS: usize = 4;
/// Phase-modulation depth of a modulator at full level, in cycles (about 12.6 radians: enough
/// for metallic and noisy timbres).
const MOD_DEPTH: f64 = 2.0;
/// Self-feedback depth at full feedback, in cycles.
const FEEDBACK_DEPTH: f64 = 0.5;

/// How the four operators connect: which modulate which, and which are heard (carriers).
/// Operators are numbered 1-4 (indices 0-3); a modulator always has a higher number than the
/// operator it modulates, so they can be computed from 4 down to 1.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Algorithm {
    /// `modulators[i]`: bit `j` set = operator `j + 1` modulates operator `i + 1`
    pub modulators: [u8; OPERATORS],
    /// bit `i` set = operator `i + 1` is heard
    pub carriers: u8,
}

impl Algorithm {
    /// Eight classic 4-operator layouts, from one long stack to four sines added up.
    pub const ALL: [Algorithm; 8] = [
        Algorithm { modulators: [0b0010, 0b0100, 0b1000, 0], carriers: 0b0001 },
        Algorithm { modulators: [0b0010, 0b1100, 0, 0], carriers: 0b0001 },
        Algorithm { modulators: [0b0110, 0, 0b1000, 0], carriers: 0b0001 },
        Algorithm { modulators: [0b0110, 0b1000, 0b1000, 0], carriers: 0b0001 },
        Algorithm { modulators: [0b0010, 0, 0b1000, 0], carriers: 0b0101 },
        Algorithm { modulators: [0b1000, 0b1000, 0b1000, 0], carriers: 0b0111 },
        Algorithm { modulators: [0, 0, 0b1000, 0], carriers: 0b0111 },
        Algorithm { modulators: [0, 0, 0, 0], carriers: 0b1111 },
    ];
    /// `>` reads "modulates", `+` "together", `,` "separately heard".
    pub const NAMES: [&'static str; 8] = ["4>3>2>1", "3+4>2>1", "4>3, +2 >1", "4>2+3>1", "4>3, 2>1", "4>1,2,3", "4>3, 2, 1", "1, 2, 3, 4"];

    fn carrier_count(&self) -> u32 {
        self.carriers.count_ones()
    }
}

/// One operator: a sine whose phase other operators (or itself, for operator 4) modulate.
#[derive(Debug, Clone)]
struct Operator<T: Float> {
    /// frequency as a multiple of the note's
    ratio: T,
    detune_cents: T,
    level: T,
    env: Adsr<T>,
    phase: T,
    increment: T,
}

/// A 4-operator phase-modulation ("FM") voice in the Yamaha DX / TX tradition.
///
/// Each operator is a sine at `ratio` times the note's frequency (plus `detune` cents), shaped by
/// its own ADSR and level. The [`Algorithm`] decides which operators modulate which and which are
/// heard; operator 4 can modulate itself (`feedback`). A modulator at level `L` shifts its target's
/// phase by up to `2 L` cycles. Carriers are scaled by velocity; timbre (0.5 neutral) and pressure
/// scale every modulation depth, brightening the sound.
#[derive(Debug, Clone)]
pub struct FmVoice<T: Float> {
    sample_rate: T,
    ops: [Operator<T>; OPERATORS],
    algorithm: usize,
    feedback: T,
    /// operator 4's last two outputs (feedback averages them, which keeps it stable)
    history: [T; 2],
    note: u8,
    velocity: T,
    bend: T,
    timbre: T,
    pressure: T,
}

impl<T: Float> FmVoice<T> {
    /// An electric-piano-like starting point: algorithm 5 (two stacks), a 14:1 "tine" on operator 4.
    pub fn new(sample_rate: T) -> Self {
        let lit = T::_lit;
        let op = |ratio: f64, level: f64, a: f64, d: f64, r: f64| Operator {
            ratio: lit(ratio),
            detune_cents: T::_ZERO,
            level: lit(level),
            env: Adsr::new(lit(a), lit(d), T::_ZERO, lit(r), sample_rate),
            phase: T::_ZERO,
            increment: T::_ZERO,
        };
        let mut voice = Self {
            sample_rate,
            ops: [op(1.0, 1.0, 0.002, 1.6, 0.5), op(1.0, 0.3, 0.001, 0.9, 0.3), op(1.0, 0.6, 0.002, 2.2, 0.6), op(14.0, 0.07, 0.001, 0.15, 0.1)],
            algorithm: 4,
            feedback: T::_ZERO,
            history: [T::_ZERO; 2],
            note: 69,
            velocity: T::_ZERO,
            bend: T::_ZERO,
            timbre: lit(0.5),
            pressure: T::_ZERO,
        };
        voice.update_pitch();
        voice
    }
    /// One of [`Algorithm::ALL`] (clamped).
    pub fn set_algorithm(&mut self, index: usize) {
        self.algorithm = index.min(Algorithm::ALL.len() - 1);
    }
    pub fn algorithm(&self) -> usize {
        self.algorithm
    }
    /// Operator 4's self-modulation, 0..1 (towards saw-like, then noisy).
    pub fn set_feedback(&mut self, amount: T) {
        self.feedback = amount._clamp(T::_ZERO, T::_ONE);
    }
    pub fn feedback(&self) -> T {
        self.feedback
    }
    /// Operator `op` (0-based) frequency as a multiple of the note's (kept positive).
    pub fn set_ratio(&mut self, op: usize, ratio: T) {
        self.ops[op].ratio = ratio._max(T::_lit(0.001));
        self.update_pitch();
    }
    pub fn ratio(&self, op: usize) -> T {
        self.ops[op].ratio
    }
    pub fn set_detune(&mut self, op: usize, cents: T) {
        self.ops[op].detune_cents = cents;
        self.update_pitch();
    }
    pub fn detune(&self, op: usize) -> T {
        self.ops[op].detune_cents
    }
    /// Output level 0..1: loudness for a carrier, modulation depth for a modulator.
    pub fn set_level(&mut self, op: usize, level: T) {
        self.ops[op].level = level._clamp(T::_ZERO, T::_ONE);
    }
    pub fn level(&self, op: usize) -> T {
        self.ops[op].level
    }
    pub fn env(&self, op: usize) -> &Adsr<T> {
        &self.ops[op].env
    }
    pub fn env_mut(&mut self, op: usize) -> &mut Adsr<T> {
        &mut self.ops[op].env
    }

    fn update_pitch(&mut self) {
        let base = midi_to_hz(self.note as f64 + self.bend.to_f64().unwrap_or(0.0));
        let fs = self.sample_rate.to_f64().unwrap_or(48_000.0);
        for op in &mut self.ops {
            let cents = op.detune_cents.to_f64().unwrap_or(0.0);
            let hz = base * op.ratio.to_f64().unwrap_or(1.0) * 2f64.powf(cents / 1_200.0);
            op.increment = T::_lit((hz / fs).max(0.0));
        }
    }

    #[inline]
    fn next_sample(&mut self) -> T {
        let alg = Algorithm::ALL[self.algorithm];
        let depth = T::_lit(MOD_DEPTH) * (T::_lit(2.0) * self.timbre + self.pressure);
        let mut outs = [T::_ZERO; OPERATORS];
        for i in (0..OPERATORS).rev() {
            let mut pm = T::_ZERO;
            for (j, &out) in outs.iter().enumerate().skip(i + 1) {
                if alg.modulators[i] & (1 << j) != 0 {
                    pm = pm + out;
                }
            }
            pm = pm * depth;
            if i == OPERATORS - 1 {
                pm = pm + self.feedback * T::_lit(FEEDBACK_DEPTH) * (self.history[0] + self.history[1]) * T::_lit(0.5);
            }
            let op = &mut self.ops[i];
            let env = op.env.next_value();
            let y = (T::_TAU * (op.phase + pm))._sin() * op.level * env;
            outs[i] = y;
            let next = op.phase + op.increment;
            op.phase = next - next._floor();
        }
        self.history = [outs[OPERATORS - 1], self.history[0]];
        let mut sum = T::_ZERO;
        for (i, &out) in outs.iter().enumerate() {
            if alg.carriers & (1 << i) != 0 {
                sum = sum + out;
            }
        }
        // carriers add up uncorrelated: keep the loudness independent of their count
        sum * self.velocity / T::_lit(f64::from(alg.carrier_count().max(1)))._sqrt()
    }
}

impl<T: Float> Voice for FmVoice<T> {
    type Sample = T;

    fn note_on(&mut self, note: u8, velocity: T) {
        self.note = note;
        self.velocity = velocity._clamp(T::_ZERO, T::_ONE);
        self.update_pitch();
        for op in &mut self.ops {
            op.env.note_on();
        }
    }
    fn legato(&mut self, note: u8, _velocity: T) {
        self.note = note;
        self.update_pitch();
    }
    fn note_off(&mut self) {
        for op in &mut self.ops {
            op.env.note_off();
        }
    }
    /// Sounding while any carrier's envelope is.
    fn is_active(&self) -> bool {
        let carriers = Algorithm::ALL[self.algorithm].carriers;
        self.ops.iter().enumerate().any(|(i, op)| carriers & (1 << i) != 0 && op.env.is_active())
    }
    fn render(&mut self, out: &mut [T]) {
        if !self.is_active() {
            out.iter_mut().for_each(|s| *s = T::_ZERO);
            return;
        }
        for s in out {
            *s = self.next_sample();
        }
        self.history = [self.history[0]._flush_denormal(), self.history[1]._flush_denormal()];
    }
    fn set_pitch_bend(&mut self, semitones: T) {
        self.bend = semitones;
        self.update_pitch();
    }
    fn set_pressure(&mut self, pressure: T) {
        self.pressure = pressure._clamp(T::_ZERO, T::_ONE);
    }
    fn set_timbre(&mut self, timbre: T) {
        self.timbre = timbre._clamp(T::_ZERO, T::_ONE);
    }
    fn reset(&mut self) {
        for op in &mut self.ops {
            op.env.reset();
            op.phase = T::_ZERO;
        }
        self.history = [T::_ZERO; 2];
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    const FS: f64 = 48_000.0;

    /// Bessel function of the first kind, J_n(x), from its power series.
    fn bessel_j(n: u32, x: f64) -> f64 {
        let mut term = (x / 2.0).powi(n as i32) / (1..=n).map(f64::from).product::<f64>();
        let mut sum = term;
        for m in 1..40 {
            term *= -(x / 2.0).powi(2) / (m as f64 * (m + n) as f64);
            sum += term;
        }
        sum
    }

    /// Exact amplitude of the component at `freq` in `x` (every component completes whole cycles
    /// in the window, so they are orthogonal).
    fn amplitude(x: &[f64], freq: f64) -> f64 {
        let (mut re, mut im) = (0.0, 0.0);
        for (i, &v) in x.iter().enumerate() {
            let phase = std::f64::consts::TAU * freq * i as f64 / FS;
            re += v * phase.cos();
            im -= v * phase.sin();
        }
        2.0 * (re * re + im * im).sqrt() / x.len() as f64
    }

    /// Operators with instant, sustained envelopes (pure steady-state tones).
    fn steady(levels: [f64; 4], ratios: [f64; 4], algorithm: usize) -> FmVoice<f64> {
        let mut v = FmVoice::new(FS);
        v.set_algorithm(algorithm);
        for op in 0..4 {
            v.set_ratio(op, ratios[op]);
            v.set_level(op, levels[op]);
            *v.env_mut(op) = Adsr::new(0.0, 0.0, 1.0, 0.0, FS);
        }
        v
    }

    fn render(v: &mut FmVoice<f64>, n: usize) -> Vec<f64> {
        let mut out = vec![0.0; n];
        v.render(&mut out);
        out
    }

    #[test]
    fn sidebands_follow_the_bessel_functions() {
        // carrier 440 Hz modulated by 110 Hz at index beta = 1 radian: the classic FM spectrum,
        // components at 440 +- n 110 Hz with amplitudes |J_n(1)|
        let beta = 1.0;
        let level = beta / (std::f64::consts::TAU * MOD_DEPTH);
        let mut v = steady([1.0, level, 0.0, 0.0], [1.0, 0.25, 1.0, 1.0], 0); // 2 > 1
        v.note_on(69, 1.0);
        render(&mut v, 480);
        let x = render(&mut v, 48_000); // one second: every component completes whole cycles
        for n in 0..=3u32 {
            let expected = bessel_j(n, beta).abs();
            for freq in [440.0 + 110.0 * n as f64, 440.0 - 110.0 * n as f64] {
                let measured = amplitude(&x, freq);
                assert!((measured - expected).abs() < 2e-3, "{freq} Hz: {measured} vs J_{n}(1) = {expected}");
            }
        }
    }

    #[test]
    fn the_additive_algorithm_is_four_sines() {
        let mut v = steady([0.8, 0.4, 0.2, 0.1], [1.0, 2.0, 3.0, 4.0], 7);
        v.note_on(57, 1.0); // 220 Hz
        render(&mut v, 480);
        let x = render(&mut v, 48_000);
        for (h, level) in [0.8, 0.4, 0.2, 0.1].into_iter().enumerate() {
            let measured = amplitude(&x, 220.0 * (h + 1) as f64);
            assert!((measured - level / 2.0).abs() < 1e-3, "harmonic {}: {measured}", h + 1); // 4 carriers: / sqrt(4)
        }
    }

    #[test]
    fn feedback_turns_a_sine_toward_a_saw() {
        let tone = |feedback: f64| {
            let mut v = steady([0.0, 0.0, 0.0, 1.0], [1.0, 1.0, 1.0, 1.0], 7);
            v.set_feedback(feedback);
            v.note_on(57, 1.0);
            render(&mut v, 480);
            let x = render(&mut v, 48_000);
            amplitude(&x, 440.0) / amplitude(&x, 220.0)
        };
        assert!(tone(0.0) < 1e-6, "no feedback: a pure sine");
        assert!(tone(1.0) > 0.1, "full feedback: strong harmonics");
    }

    #[test]
    fn every_algorithm_plays_and_releases() {
        for algorithm in 0..Algorithm::ALL.len() {
            let mut v = FmVoice::new(FS);
            v.set_algorithm(algorithm);
            v.note_on(60, 0.9);
            let x = render(&mut v, 4_800);
            assert!(x.iter().all(|s| s.is_finite() && s.abs() <= 1.5), "algorithm {algorithm}");
            assert!(x.iter().any(|&s| s.abs() > 0.01), "algorithm {algorithm} is heard");
            v.note_off();
            render(&mut v, 48_000);
            assert!(!v.is_active(), "algorithm {algorithm} ends after its release");
        }
    }

    #[test]
    fn timbre_brightens_by_deepening_modulation() {
        // spectral centroid over the first 12 harmonics, in harmonics: deeper modulation spreads the
        // energy into more sidebands (Carson's rule), so it rises with timbre
        let centroid = |timbre: f64| {
            let mut v = steady([1.0, 0.1, 0.0, 0.0], [1.0, 1.0, 1.0, 1.0], 0);
            v.set_timbre(timbre);
            v.note_on(57, 1.0);
            render(&mut v, 480);
            let x = render(&mut v, 48_000);
            let power: Vec<f64> = (1..=12).map(|h| amplitude(&x, 220.0 * h as f64).powi(2)).collect();
            power.iter().enumerate().map(|(i, p)| (i + 1) as f64 * p).sum::<f64>() / power.iter().sum::<f64>()
        };
        let (none, neutral, full) = (centroid(0.0), centroid(0.5), centroid(1.0));
        assert!((none - 1.0).abs() < 1e-6, "timbre 0: no modulation, a pure sine");
        assert!(neutral > 1.2 && full > neutral + 0.3, "centroids {none:.2} < {neutral:.2} < {full:.2}");
    }
}