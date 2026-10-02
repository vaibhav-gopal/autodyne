//! Moog-style 4-pole ladder low-pass with zero-delay feedback.

use crate::distortion::Shape;
use crate::units::*;

/// Level where the loop input saturates (at drive 1): `HEADROOM * tanh(u / HEADROOM)` stays within
/// a few percent of linear up to about half of it, so full-scale input is only gently compressed.
const HEADROOM: f64 = 2.0;

/// The settings of a [`Ladder`] step, over [`Real`] (so it also traces and differentiates in
/// `flux`).
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct LadderCoeffs<T> {
    /// tan(pi fc / fs)
    pub g: T,
    /// 0..1
    pub resonance: T,
    pub drive: T,
    pub compensate: bool,
}

impl<T: Real> LadderCoeffs<T> {
    /// Cutoff kept within 1 Hz .. 0.49 x sample rate, resonance within 0..1, drive at least 0.01.
    pub fn new(cutoff: T, resonance: T, drive: T, compensate: bool, sample_rate: T) -> Self {
        let cutoff = cutoff.clip(T::lit(1.0), T::lit(0.49) * sample_rate);
        Self {
            g: (T::lit(std::f64::consts::PI) * cutoff / sample_rate).tan(),
            resonance: resonance.clip(T::lit(0.0), T::lit(1.0)),
            drive: drive.maximum(T::lit(0.01)),
            compensate,
        }
    }
    /// One step from the four stage states: `(next states, output)`.
    #[inline(always)]
    pub fn tick(&self, stages: [T; 4], x: T) -> ([T; 4], T) {
        let one = T::lit(1.0);
        let k = T::lit(4.0) * self.resonance;
        let gg = self.g / (one + self.g); // one stage's gain on its input
        let carry = one / (one + self.g); // and on its state
        // the 4th stage's output is gg^4 * u + sigma, where sigma collects the stored states;
        // feedback u = x - k * y4 then solves in closed form (the zero-delay feedback loop)
        let s = &stages;
        let sigma = carry * (gg * gg * gg * s[0] + gg * gg * s[1] + gg * s[2] + s[3]);
        let gain_in = if self.compensate { one + k } else { one };
        let gg4 = gg * gg * gg * gg;
        let u = (x * gain_in - k * sigma) / (one + k * gg4);
        let h = T::lit(HEADROOM);
        let mut y = h * Shape::Tanh.apply(self.drive * u / h) / self.drive;
        let mut next = stages;
        for state in &mut next {
            // trapezoidal one-pole low-pass
            let v = (y - *state) * gg;
            let lp = v + *state;
            *state = lp + v;
            y = lp;
        }
        (next, y)
    }
}

/// Four trapezoidal one-pole low-passes in a resonant feedback loop: the classic 24 dB/octave
/// synthesizer filter.
///
/// The loop is solved without the unit delay naive digital ladders put in the feedback path, so
/// cutoff and resonance stay accurate up to high frequencies and the filter self-oscillates exactly
/// at the cutoff when resonance reaches 1. With drive at 1 and moderate levels the response is the
/// bilinear transform of the analog ladder ([`magnitude_at`](Ladder::magnitude_at)); a `tanh`
/// saturator at the loop input adds the characteristic overdrive as levels or drive rise.
#[derive(Debug, Clone, Copy)]
pub struct Ladder<T: Float> {
    cutoff: T,
    /// 0..1 (feedback k = 4 x resonance)
    resonance: T,
    drive: T,
    compensate: bool,
    sample_rate: T,
    /// tan(pi fc / fs)
    g: T,
    stages: [T; 4],
}

impl<T: Float> Ladder<T> {
    /// Drive 1 (clean at moderate levels) with passband compensation on.
    pub fn new(cutoff: T, resonance: T, sample_rate: T) -> Self {
        let mut f = Self {
            cutoff,
            resonance: T::_ZERO,
            drive: T::_ONE,
            compensate: true,
            sample_rate,
            g: T::_ZERO,
            stages: [T::_ZERO; 4],
        };
        f.set_cutoff(cutoff);
        f.set_resonance(resonance);
        f
    }
    /// Cutoff in Hz, kept within 1 Hz .. 0.49 x sample rate.
    #[inline]
    pub fn set_cutoff(&mut self, hz: T) {
        self.cutoff = hz._max(T::_ONE)._min(T::_lit(0.49) * self.sample_rate);
        self.g = (T::_PI * self.cutoff / self.sample_rate)._tan();
    }
    pub fn cutoff(&self) -> T {
        self.cutoff
    }
    /// 0 (none) .. 1 (self-oscillation at the cutoff).
    pub fn set_resonance(&mut self, resonance: T) {
        self.resonance = resonance._clamp(T::_ZERO, T::_ONE);
    }
    pub fn resonance(&self) -> T {
        self.resonance
    }
    /// How hard the loop input saturates (linear, at least 0.01). Drive raises the level into the
    /// saturator and lowers its output by the same amount, so small signals are unchanged at any
    /// drive and automating it causes no level jump: loud signals just saturate earlier.
    /// At 1, full-scale input is compressed by under 1 dB.
    pub fn set_drive(&mut self, drive: T) {
        self.drive = drive._max(T::_lit(0.01));
    }
    pub fn drive(&self) -> T {
        self.drive
    }
    /// Passband compensation: resonance feedback lowers the gain below the cutoff by `1 + 4 x
    /// resonance`; with compensation on (the default) the input is raised to match, so the bass
    /// stays put as resonance rises.
    pub fn set_compensation(&mut self, on: bool) {
        self.compensate = on;
    }
    pub fn sample_rate(&self) -> T {
        self.sample_rate
    }
    pub fn reset(&mut self) {
        self.stages = [T::_ZERO; 4];
    }

    /// The step's settings (for driving the same filter from generic or traced code).
    pub fn coeffs(&self) -> LadderCoeffs<T> {
        LadderCoeffs { g: self.g, resonance: self.resonance, drive: self.drive, compensate: self.compensate }
    }

    #[inline]
    pub fn process_sample(&mut self, x: T) -> T {
        let (stages, y) = self.coeffs().tick(self.stages, x);
        self.stages = stages;
        y
    }

    /// Filters `block` in place.
    pub fn process(&mut self, block: &mut [T]) {
        for s in block {
            *s = self.process_sample(*s);
        }
        // once per block: a decaying state reaches zero instead of slow subnormal numbers
        self.stages.iter_mut().for_each(|s| *s = s._flush_denormal());
    }

    /// |H| at `frequency` for small signals with drive 1 (the analog ladder at the prewarped
    /// frequency): `gain_in / |(1 + jw)^4 + k|`.
    pub fn magnitude_at(&self, frequency: T) -> T {
        let w = (T::_PI * frequency / self.sample_rate)._tan() / self.g;
        let k = T::_lit(4.0) * self.resonance;
        // (1 + jw)^2 = (1 - w^2) + 2jw, squared again
        let (a, b) = (T::_ONE - w * w, T::_lit(2.0) * w);
        let (re, im) = (a * a - b * b, T::_lit(2.0) * a * b);
        let gain_in = if self.compensate { T::_ONE + k } else { T::_ONE };
        gain_in / (re + k)._hypot(im)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    const FS: f64 = 48_000.0;

    fn sine(freq: f64, amplitude: f64, n: usize) -> Vec<f64> {
        (0..n).map(|i| amplitude * (std::f64::consts::TAU * freq * i as f64 / FS).sin()).collect()
    }

    fn rms(v: &[f64]) -> f64 {
        (v.iter().map(|s| s * s).sum::<f64>() / v.len() as f64).sqrt()
    }

    #[test]
    fn small_signal_response_matches_the_analog_ladder() {
        for resonance in [0.0, 0.5, 0.9] {
            for compensate in [true, false] {
                let mut base = Ladder::new(1_000.0, resonance, FS);
                base.set_compensation(compensate);
                for freq in [50.0, 500.0, 1_000.0, 2_000.0, 6_000.0] {
                    let mut f = base;
                    let x = sine(freq, 1e-3, 48_000);
                    let y: Vec<f64> = x.iter().map(|&s| f.process_sample(s)).collect();
                    let (measured, analytic) = (rms(&y[24_000..]) / rms(&x[24_000..]), base.magnitude_at(freq));
                    assert!((measured / analytic - 1.0).abs() < 0.01, "r={resonance} comp={compensate} {freq} Hz: {measured} vs {analytic}");
                }
            }
        }
    }

    #[test]
    fn slopes_and_passband() {
        let f = Ladder::new(1_000.0, 0.0, FS);
        // four poles: about -24 dB per octave well above the cutoff
        let octave = 20.0 * (f.magnitude_at(8_000.0) / f.magnitude_at(4_000.0)).log10();
        assert!((octave + 24.0).abs() < 2.0, "{octave} dB/octave");
        assert!((f.magnitude_at(1.0) - 1.0).abs() < 1e-4, "unity in the passband");
        let mut resonant = Ladder::new(1_000.0, 0.8, FS);
        assert!((resonant.magnitude_at(1.0) - 1.0).abs() < 1e-4, "compensated: still unity");
        resonant.set_compensation(false);
        assert!((resonant.magnitude_at(1.0) - 1.0 / 4.2).abs() < 1e-4, "uncompensated: 1 / (1 + k)");
        assert!(resonant.magnitude_at(1_000.0) > 2.0 * resonant.magnitude_at(1.0), "a resonant peak at the cutoff");
    }

    #[test]
    fn full_resonance_self_oscillates_at_the_cutoff() {
        let mut f = Ladder::new(1_000.0, 1.0, FS);
        f.process_sample(0.5); // a kick, then silence
        let ring: Vec<f64> = (0..48_000).map(|_| f.process_sample(0.0)).collect();
        let tail = &ring[38_400..];
        let crossings = tail.windows(2).filter(|w| (w[0] < 0.0) != (w[1] < 0.0)).count();
        let freq = crossings as f64 / 2.0 / (tail.len() as f64 / FS);
        assert!((freq / 1_000.0 - 1.0).abs() < 0.02, "rings at {freq} Hz");
        let (early, late) = (rms(&ring[4_800..9_600]), rms(tail));
        assert!(late > 0.5 * early && late < 2.0, "sustained and bounded: {early} -> {late}");
    }

    #[test]
    fn drive_saturates_and_modulation_stays_bounded() {
        let mut clean = Ladder::new(5_000.0, 0.0, FS);
        let mut driven = clean;
        driven.set_drive(10.0);
        let x = sine(200.0, 1.0, 9_600);
        let peak = |f: &mut Ladder<f64>| x.iter().map(|&s| f.process_sample(s).abs()).fold(0.0, f64::max);
        assert!(peak(&mut driven) <= 2.0 / 10.0, "saturation bounds the loop at headroom / drive");
        assert!(peak(&mut clean) > 0.9, "drive 1: full scale only gently compressed");
        // small signals pass unchanged at any drive (no level jump when automating it)
        let mut small = Ladder::new(5_000.0, 0.0, FS);
        small.set_drive(10.0);
        let x = sine(200.0, 1e-3, 9_600);
        let y: Vec<f64> = x.iter().map(|&s| small.process_sample(s)).collect();
        assert!((rms(&y[4_800..]) / rms(&x[4_800..]) / small.magnitude_at(200.0) - 1.0).abs() < 0.01);
        // cutoff swept at audio rate with high resonance
        let mut f = Ladder::new(1_000.0, 0.95, FS);
        let mut worst = 0.0f64;
        for i in 0..48_000 {
            let t = i as f64 / FS;
            f.set_cutoff(30.0 * 600.0f64.powf(0.5 + 0.5 * (std::f64::consts::TAU * 200.0 * t).sin()));
            worst = worst.max(f.process_sample((std::f64::consts::TAU * 110.0 * t).sin()).abs());
        }
        assert!(worst.is_finite() && worst < 10.0, "bounded: {worst}");
    }
}
