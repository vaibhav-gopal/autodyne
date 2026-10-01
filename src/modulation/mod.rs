//! Modulation effects: chorus, flanger (both `ModulatedDelay`) and phaser.
//!
//! - Chorus / flanger: a short delay whose length is swept by a slow sine LFO. A chorus uses a longer
//!   delay (~20 ms) and no feedback, so it sounds like several detuned copies; a flanger uses a very
//!   short delay (~1-4 ms) with feedback, which gives a sweeping comb filter.
//! - Phaser: a chain of first-order all-pass filters whose corner frequency the LFO sweeps; mixing
//!   the phase-shifted signal with the dry one creates moving notches.
//!
//! For stereo width, run one instance per channel with LFO phases half a cycle apart (`with_lfo_phase`).

use crate::delay::DelayLine;
use crate::osc::Sine;
use crate::units::*;

// MODULATED DELAY (CHORUS / FLANGER) ==============================================================

/// A delay line whose delay time moves as `base + depth * sin(LFO)`.
#[derive(Debug, Clone)]
pub struct ModulatedDelay<T: Float> {
    line: DelayLine<T>,
    lfo: Sine<T>,
    lfo_phase: T,
    rate: T,
    sample_rate: T,
    base_samples: T,
    depth_samples: T,
    feedback: T,
    mix: T,
    current_delay: T,
}

impl<T: Float> ModulatedDelay<T> {
    /// Delay sweeps between `base - depth` and `base + depth` seconds at `rate_hz`.
    /// Panics if `depth > base` (the delay can't go negative).
    pub fn new(base_seconds: T, depth_seconds: T, rate_hz: T, sample_rate: T) -> Self {
        assert!(depth_seconds <= base_seconds, "depth must not exceed the base delay");
        let max = ((base_seconds + depth_seconds) * sample_rate)._ceil().to_usize().expect("delay must be finite") + 2;
        Self {
            line: DelayLine::new(max),
            lfo: Sine::new(rate_hz, sample_rate),
            lfo_phase: T::_ZERO,
            rate: rate_hz,
            sample_rate,
            base_samples: base_seconds * sample_rate,
            depth_samples: depth_seconds * sample_rate,
            feedback: T::_ZERO,
            mix: T::_lit(0.5),
            current_delay: base_seconds * sample_rate,
        }
    }
    /// Classic chorus: 20 ms +/- 3 ms at 0.8 Hz, no feedback, 50% mix.
    pub fn chorus(sample_rate: T) -> Self {
        Self::new(T::_lit(0.020), T::_lit(0.003), T::_lit(0.8), sample_rate)
    }
    /// Classic flanger: 2 ms +/- 1.5 ms at 0.25 Hz, 50% feedback, 50% mix.
    pub fn flanger(sample_rate: T) -> Self {
        let mut f = Self::new(T::_lit(0.002), T::_lit(0.0015), T::_lit(0.25), sample_rate);
        f.set_feedback(T::_lit(0.5));
        f
    }
    /// Starting LFO phase in cycles; use 0 and 0.5 on the left/right channels for a wide stereo image.
    pub fn with_lfo_phase(mut self, cycles: T) -> Self {
        self.lfo_phase = cycles;
        self.lfo = self.lfo.with_phase(cycles);
        self
    }
    pub fn set_rate(&mut self, hz: T) {
        self.rate = hz;
        self.lfo.set_frequency(hz, self.sample_rate);
    }
    pub fn rate(&self) -> T {
        self.rate
    }
    /// Sweep depth in seconds.
    pub fn depth(&self) -> T {
        self.depth_samples / self.sample_rate
    }
    /// Largest depth `set_depth` accepts (limited by the base delay and the allocated line).
    pub fn max_depth(&self) -> T {
        let room = T::_lit(self.line.max_delay() as f64 - 1.0) - self.base_samples;
        room._min(self.base_samples)._max(T::_ZERO) / self.sample_rate
    }
    pub fn feedback(&self) -> T {
        self.feedback
    }
    pub fn mix(&self) -> T {
        self.mix
    }
    /// Sweep depth in seconds, limited so the delay stays within the line allocated in `new`.
    pub fn set_depth(&mut self, seconds: T) {
        let room = T::_lit(self.line.max_delay() as f64 - 1.0) - self.base_samples;
        self.depth_samples = (seconds * self.sample_rate)._clamp(T::_ZERO, room._min(self.base_samples));
    }
    /// Negative feedback is allowed (it moves a flanger's peaks to where the notches were).
    /// Clamped to +/-0.95 to stay stable.
    pub fn set_feedback(&mut self, feedback: T) {
        let limit = T::_lit(0.95);
        self.feedback = feedback._clamp(-limit, limit);
    }
    /// 0 = dry only, 1 = delayed only.
    pub fn set_mix(&mut self, mix: T) {
        self.mix = mix._clamp(T::_ZERO, T::_ONE);
    }
    /// The delay used for the most recent sample, in samples.
    pub fn current_delay_samples(&self) -> T {
        self.current_delay
    }
    pub fn reset(&mut self) {
        self.line.reset();
        self.lfo = self.lfo.with_phase(self.lfo_phase);
    }
    #[inline]
    pub fn process_sample(&mut self, x: T) -> T {
        self.current_delay = self.base_samples + self.depth_samples * self.lfo.next_sample();
        // read before pushing x, so the newest stored sample is one step old (as in `Echo`)
        let delayed = self.line.read_frac(self.current_delay - T::_ONE);
        self.line.push((x + self.feedback * delayed)._flush_denormal());
        x * (T::_ONE - self.mix) + delayed * self.mix
    }
    pub fn process(&mut self, block: &mut [T]) {
        for s in block {
            *s = self.process_sample(*s);
        }
    }
}

// PHASER ==========================================================================================

/// First-order all-pass: unity gain, phase going from 0 to -180 degrees, -90 at its corner frequency.
#[derive(Debug, Clone, Copy)]
struct Allpass1<T: Float> {
    x1: T,
    y1: T,
}

impl<T: Float> Allpass1<T> {
    #[inline]
    fn process(&mut self, x: T, a: T) -> T {
        let y = (a * x + self.x1 - a * self.y1)._flush_denormal();
        self.x1 = x;
        self.y1 = y;
        y
    }
}

/// Coefficient putting a first-order all-pass's -90 degree point at `frequency`.
#[inline]
fn allpass_coeff<T: Float>(frequency: T, sample_rate: T) -> T {
    let t = (T::_PI * frequency / sample_rate)._tan();
    (t - T::_ONE) / (t + T::_ONE)
}

/// Phaser: `stages` first-order all-passes swept exponentially between `min` and `max` Hz.
#[derive(Debug, Clone)]
pub struct Phaser<T: Float> {
    stages: Vec<Allpass1<T>>,
    lfo: Sine<T>,
    lfo_phase: T,
    rate: T,
    sample_rate: T,
    min_hz: T,
    max_hz: T,
    feedback: T,
    mix: T,
    last_out: T,
}

impl<T: Float> Phaser<T> {
    /// `stages` all-pass stages (each pair adds a notch; 4-8 is typical). Sweeps 200 Hz - 2 kHz
    /// at 0.5 Hz, no feedback, 50% mix. Panics if `stages` is 0.
    pub fn new(stages: usize, sample_rate: T) -> Self {
        assert!(stages > 0, "a phaser needs at least one stage");
        Self {
            stages: vec![Allpass1 { x1: T::_ZERO, y1: T::_ZERO }; stages],
            lfo: Sine::new(T::_lit(0.5), sample_rate),
            lfo_phase: T::_ZERO,
            rate: T::_lit(0.5),
            sample_rate,
            min_hz: T::_lit(200.0),
            max_hz: T::_lit(2000.0),
            feedback: T::_ZERO,
            mix: T::_lit(0.5),
            last_out: T::_ZERO,
        }
    }
    pub fn with_lfo_phase(mut self, cycles: T) -> Self {
        self.lfo_phase = cycles;
        self.lfo = self.lfo.with_phase(cycles);
        self
    }
    pub fn set_rate(&mut self, hz: T) {
        self.rate = hz;
        self.lfo.set_frequency(hz, self.sample_rate);
    }
    pub fn rate(&self) -> T {
        self.rate
    }
    /// (min, max) sweep range in Hz.
    pub fn range(&self) -> (T, T) {
        (self.min_hz, self.max_hz)
    }
    pub fn feedback(&self) -> T {
        self.feedback
    }
    pub fn mix(&self) -> T {
        self.mix
    }
    pub fn sample_rate(&self) -> T {
        self.sample_rate
    }
    /// Sweep range; equal values hold the all-passes still. Panics unless 0 < min <= max < Nyquist.
    pub fn set_range(&mut self, min_hz: T, max_hz: T) {
        assert!(min_hz > T::_ZERO && min_hz <= max_hz && max_hz < self.sample_rate / T::_lit(2.0), "invalid phaser range");
        self.min_hz = min_hz;
        self.max_hz = max_hz;
    }
    /// Deepens the notches; clamped to +/-0.95.
    pub fn set_feedback(&mut self, feedback: T) {
        let limit = T::_lit(0.95);
        self.feedback = feedback._clamp(-limit, limit);
    }
    pub fn set_mix(&mut self, mix: T) {
        self.mix = mix._clamp(T::_ZERO, T::_ONE);
    }
    pub fn reset(&mut self) {
        self.stages.iter_mut().for_each(|s| *s = Allpass1 { x1: T::_ZERO, y1: T::_ZERO });
        self.lfo = self.lfo.with_phase(self.lfo_phase);
        self.last_out = T::_ZERO;
    }
    #[inline]
    pub fn process_sample(&mut self, x: T) -> T {
        // exponential sweep: equal LFO steps move by equal musical intervals
        let sweep = (self.lfo.next_sample() + T::_ONE) / T::_lit(2.0);
        let freq = self.min_hz * (self.max_hz / self.min_hz)._pow(sweep);
        let a = allpass_coeff(freq, self.sample_rate);
        let mut s = (x + self.feedback * self.last_out)._flush_denormal();
        for stage in &mut self.stages {
            s = stage.process(s, a);
        }
        self.last_out = s;
        x * (T::_ONE - self.mix) + s * self.mix
    }
    pub fn process(&mut self, block: &mut [T]) {
        for s in block {
            *s = self.process_sample(*s);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::osc::{Impulse, Sine};

    fn assert_close(a: f64, b: f64, tol: f64, what: &str) {
        assert!((a - b).abs() <= tol, "{what}: {a} vs {b} (tol {tol})");
    }

    /// Steady-state gain for a unit sine at `freq` (RMS over a whole number of cycles).
    fn gain_at(mut process: impl FnMut(&mut [f64]), freq: f64, fs: f64) -> f64 {
        let mut buf: Vec<f64> = Sine::new(freq, fs).take(fs as usize).collect();
        process(&mut buf);
        let tail = &buf[buf.len() / 2..];
        (2.0 * tail.iter().map(|s| s * s).sum::<f64>() / tail.len() as f64).sqrt()
    }

    #[test]
    fn without_depth_it_is_a_feedback_delay() {
        let fs = 1000.0;
        let mut d = ModulatedDelay::new(0.010, 0.0, 1.0, fs); // 10 samples, no sweep
        d.set_feedback(0.5);
        let mut buf = vec![0.0; 40];
        Impulse::new().fill(&mut buf);
        d.process(&mut buf);
        let expected = |n: usize| match n {
            0 | 10 => 0.5,
            20 => 0.25,
            30 => 0.125,
            _ => 0.0,
        };
        for (n, &y) in buf.iter().enumerate() {
            assert_close(y, expected(n), 1e-12, &format!("sample {n}"));
        }
    }

    #[test]
    fn delay_sweeps_between_base_minus_and_plus_depth() {
        let fs = 48_000.0;
        let mut f = ModulatedDelay::flanger(fs); // 2 ms +/- 1.5 ms at 0.25 Hz
        let (mut lo, mut hi) = (f64::MAX, f64::MIN);
        for _ in 0..(4.0 * fs) as usize {
            f.process_sample(0.0);
            lo = lo.min(f.current_delay_samples());
            hi = hi.max(f.current_delay_samples());
        }
        assert_close(lo, 0.0005 * fs, 1e-6, "shortest delay");
        assert_close(hi, 0.0035 * fs, 1e-6, "longest delay");
    }

    #[test]
    fn opposite_lfo_phases_sweep_in_opposite_directions() {
        let fs = 48_000.0;
        let mut left = ModulatedDelay::chorus(fs);
        let mut right = ModulatedDelay::chorus(fs).with_lfo_phase(0.5);
        for _ in 0..10_000 {
            left.process_sample(0.0);
            right.process_sample(0.0);
            let sum = left.current_delay_samples() + right.current_delay_samples();
            assert_close(sum, 2.0 * 0.020 * fs, 1e-9, "delays mirror around the base");
        }
    }

    #[test]
    fn reset_restarts_the_lfo_and_clears_the_line() {
        let fs = 48_000.0;
        let input: Vec<f64> = Sine::new(300.0, fs).take(5_000).collect();
        let mut fx = ModulatedDelay::chorus(fs).with_lfo_phase(0.3);
        let mut first = input.clone();
        fx.process(&mut first);
        fx.reset();
        let mut second = input;
        fx.process(&mut second);
        assert_eq!(first, second);
    }

    #[test]
    fn allpass_stage_has_unity_gain_and_minus_90_degrees_at_its_corner() {
        let fs = 48_000.0;
        let a = allpass_coeff(1_000.0, fs);
        let mut stage = Allpass1 { x1: 0.0, y1: 0.0 };
        assert_close(gain_at(|b| b.iter_mut().for_each(|s| *s = stage.process(*s, a)), 3_000.0, fs), 1.0, 1e-9, "unity gain");
        // -90 degrees at the corner: sin in, -cos out
        let mut stage = Allpass1 { x1: 0.0, y1: 0.0 };
        let mut osc = Sine::new(1_000.0, fs);
        let mut cos = Sine::new(1_000.0, fs).with_phase(0.25);
        for n in 0..48_000 {
            let y = stage.process(osc.next_sample(), a);
            let c = cos.next_sample();
            if n > 24_000 {
                assert_close(y, -c, 1e-9, "quadrature at corner");
            }
        }
    }

    #[test]
    fn frozen_phaser_notches_where_the_chain_reaches_180_degrees() {
        // 4 stages at f0: each stage is -2 atan(tan(pi f/fs) / tan(pi f0/fs)); the chain hits -180
        // degrees (a notch with 50% mix) where that atan is pi/8, and -360 (full level) at f0 itself.
        let (fs, f0) = (48_000.0, 1_000.0);
        let mut p = Phaser::new(4, fs);
        p.set_range(f0, f0);
        let notch = (std::f64::consts::FRAC_PI_8.tan() * (std::f64::consts::PI * f0 / fs).tan()).atan() * fs / std::f64::consts::PI;
        let mut probe = p.clone();
        assert!(gain_at(|b| probe.process(b), notch, fs) < 0.01, "notch at {notch:.1} Hz");
        let mut probe = p.clone();
        assert_close(gain_at(|b| probe.process(b), f0, fs), 1.0, 1e-6, "full level at f0");
    }
}
