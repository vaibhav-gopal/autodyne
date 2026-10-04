//! Delay lines and delay-based effects.
//!
//! `DelayLine` is the building block (echo, chorus, flanger, comb filters, reverbs); `Echo` is a
//! feedback delay built on it.

mod params;

use crate::gain::SmoothedValue;
use crate::units::*;

/// Ring buffer of past samples with integer and fractional (linearly interpolated) reads.
/// Allocates once in `new`; `push` and the reads never allocate.
#[derive(Debug, Clone)]
pub struct DelayLine<T: Float> {
    buffer: Vec<T>,
    /// index of the most recently pushed sample
    newest: usize,
}

impl<T: Float> DelayLine<T> {
    /// Can read up to `max_delay` samples back (fractional reads up to `max_delay` too).
    pub fn new(max_delay: usize) -> Self {
        // +1 so read(max_delay) exists, +1 more so a fractional read just under max_delay can interpolate
        Self { buffer: vec![T::_ZERO; max_delay + 2], newest: 0 }
    }
    pub fn max_delay(&self) -> usize {
        self.buffer.len() - 2
    }
    pub fn reset(&mut self) {
        self.buffer.iter_mut().for_each(|s| *s = T::_ZERO);
    }
    #[inline]
    pub fn push(&mut self, x: T) {
        self.newest = if self.newest + 1 == self.buffer.len() { 0 } else { self.newest + 1 };
        self.buffer[self.newest] = x;
    }
    /// The sample pushed `delay` pushes ago (0 = the most recent). Panics if `delay > max_delay()`.
    #[inline]
    pub fn read(&self, delay: usize) -> T {
        assert!(delay <= self.max_delay(), "delay {delay} exceeds max {}", self.max_delay());
        let len = self.buffer.len();
        self.buffer[(self.newest + len - delay) % len]
    }
    /// Like `read`, but between samples, by linear interpolation. `delay` is clamped to [0, max_delay].
    #[inline]
    pub fn read_frac(&self, delay: T) -> T {
        let d = delay._max(T::_ZERO)._min(T::_lit(self.max_delay() as f64));
        let whole = d._floor();
        let frac = d - whole;
        let i = whole.to_usize().unwrap_or(0);
        let (a, b) = (self.read(i), self.buffer[(self.newest + self.buffer.len() - i - 1) % self.buffer.len()]);
        a + (b - a) * frac
    }
    /// Like `read_frac`, with 4-point cubic (Hermite) interpolation: nearly flat frequency response, so
    /// repeated reads (as in feedback delay networks) don't dull the highs the way linear interpolation
    /// does. `delay` is clamped to [1, max_delay - 1], since the curve needs a sample on each side.
    #[inline]
    pub fn read_cubic(&self, delay: T) -> T {
        let d = delay._max(T::_ONE)._min(T::_lit(self.max_delay() as f64 - 1.0));
        let whole = d._floor();
        let t = d - whole;
        let i = whole.to_usize().unwrap_or(1);
        let (y0, y1, y2, y3) = (self.read(i - 1), self.read(i), self.read(i + 1), self.buffer[(self.newest + self.buffer.len() - i - 2) % self.buffer.len()]);
        let half = T::_lit(0.5);
        let c1 = half * (y2 - y0);
        let c2 = y0 - T::_lit(2.5) * y1 + T::_lit(2.0) * y2 - half * y3;
        let c3 = half * (y3 - y0) + T::_lit(1.5) * (y1 - y2);
        ((c3 * t + c2) * t + c1) * t + y1
    }
}

/// Feedback echo: each repeat is `feedback` times the previous one, mixed with the dry signal.
/// Delay time, feedback and mix are smoothed, so they can be automated while running.
#[derive(Debug, Clone)]
pub struct Echo<T: Float> {
    line: DelayLine<T>,
    sample_rate: T,
    delay_samples: SmoothedValue<T>,
    feedback: SmoothedValue<T>,
    mix: SmoothedValue<T>,
}

impl<T: Float> Echo<T> {
    /// Allocates room for delays up to `max_delay_seconds`. Starts at half that delay,
    /// feedback 0.5 and mix 0.5.
    pub fn new(max_delay_seconds: T, sample_rate: T) -> Self {
        let max = (max_delay_seconds * sample_rate)._ceil().to_usize().expect("max delay must be finite and >= 0");
        let ramp = T::_lit(0.05); // 50 ms parameter smoothing
        let smoothed = |v: T| SmoothedValue::new(v).with_ramp_seconds(ramp, sample_rate);
        Self {
            line: DelayLine::new(max.max(1)),
            sample_rate,
            delay_samples: smoothed(T::_lit(max.max(2) as f64 / 2.0)),
            feedback: smoothed(T::_lit(0.5)),
            mix: smoothed(T::_lit(0.5)),
        }
    }
    /// Clamped to [1 sample, the maximum given to `new`].
    pub fn set_delay_seconds(&mut self, seconds: T) {
        let samples = (seconds * self.sample_rate)._clamp(T::_ONE, T::_lit(self.line.max_delay() as f64));
        self.delay_samples.set_target(samples);
    }
    /// Level of each repeat relative to the one before; clamped below 1 so the echo always dies out.
    pub fn set_feedback(&mut self, feedback: T) {
        self.feedback.set_target(feedback._clamp(T::_ZERO, T::_lit(0.99)));
    }
    /// 0 = dry only, 1 = echoes only.
    pub fn set_mix(&mut self, mix: T) {
        self.mix.set_target(mix._clamp(T::_ZERO, T::_ONE));
    }
    /// Sets everything at once without ramping (e.g. before playback starts).
    pub fn set_immediate(&mut self, delay_seconds: T, feedback: T, mix: T) {
        self.set_delay_seconds(delay_seconds);
        self.set_feedback(feedback);
        self.set_mix(mix);
        for v in [&mut self.delay_samples, &mut self.feedback, &mut self.mix] {
            let target = v.target();
            v.set_immediate(target);
        }
    }
    /// Target delay time in seconds.
    pub fn delay_seconds(&self) -> T {
        self.delay_samples.target() / self.sample_rate
    }
    /// Longest delay this echo can reach (set by `new`).
    pub fn max_delay_seconds(&self) -> T {
        T::_lit(self.line.max_delay() as f64) / self.sample_rate
    }
    pub fn feedback(&self) -> T {
        self.feedback.target()
    }
    pub fn mix(&self) -> T {
        self.mix.target()
    }
    pub fn sample_rate(&self) -> T {
        self.sample_rate
    }
    pub fn reset(&mut self) {
        self.line.reset();
    }
    #[inline]
    pub fn process_sample(&mut self, x: T) -> T {
        let (delay, feedback, mix) = (self.delay_samples.next_value(), self.feedback.next_value(), self.mix.next_value());
        // Read before pushing x: the newest stored sample is from the previous step, so one less.
        let delayed = self.line.read_frac(delay - T::_ONE);
        // flushed: repeats die out to exact zero instead of slow subnormal numbers
        self.line.push((x + feedback * delayed)._flush_denormal());
        x * (T::_ONE - mix) + delayed * mix
    }
    /// Processes `block` in place.
    pub fn process(&mut self, block: &mut [T]) {
        for s in block {
            *s = self.process_sample(*s);
        }
    }
}

crate::processor::forward_processor!(Echo);

#[cfg(test)]
mod tests {
    use super::*;
    use crate::osc::Impulse;

    #[test]
    fn integer_reads_are_exact_delays() {
        let mut line = DelayLine::new(8);
        for x in 0..20 {
            line.push(x as f64);
        }
        for d in 0..=8 {
            assert_eq!(line.read(d), (19 - d) as f64);
        }
    }

    #[test]
    fn fractional_reads_interpolate() {
        // linear interpolation is exact on a ramp
        let mut line = DelayLine::new(16);
        for x in 0..10 {
            line.push(x as f64);
        }
        assert_eq!(line.read_frac(2.25), 6.75);
        assert_eq!(line.read_frac(0.0), 9.0);
        assert_eq!(line.read_frac(-3.0), 9.0); // clamped
    }

    #[test]
    fn cubic_reads_are_exact_on_smooth_signals() {
        // Hermite interpolation reproduces quadratics exactly
        let mut line = DelayLine::new(32);
        let f = |n: f64| 0.01 * n * n - 0.3 * n + 2.0;
        for n in 0..20 {
            line.push(f(n as f64));
        }
        for d in [1.0, 2.5, 7.25, 12.9] {
            assert!((line.read_cubic(d) - f(19.0 - d)).abs() < 1e-12, "delay {d}");
        }
    }

    #[test]
    #[should_panic(expected = "exceeds max")]
    fn reading_past_capacity_panics() {
        DelayLine::<f64>::new(4).read(5);
    }

    #[test]
    fn echo_impulse_response() {
        let fs = 1000.0f64;
        let mut echo = Echo::new(1.0, fs);
        echo.set_immediate(0.1, 0.5, 0.5); // 100 samples
        let mut buf = vec![0.0; 400];
        Impulse::new().fill(&mut buf);
        echo.process(&mut buf);
        let expected = |n: usize| match n {
            0 => 0.5,     // dry
            100 => 0.5,   // first repeat, wet
            200 => 0.25,  // x feedback
            300 => 0.125,
            _ => 0.0,
        };
        for (n, &y) in buf.iter().enumerate() {
            assert!((y - expected(n)).abs() < 1e-12, "sample {n}: {y}");
        }
    }

    #[test]
    fn echo_parameter_changes_are_smoothed() {
        let fs = 48_000.0;
        let mut echo = Echo::new(1.0, fs);
        echo.set_immediate(0.1, 0.0, 0.0);
        echo.set_mix(1.0);
        let mut buf = [1.0f64; 4];
        echo.process(&mut buf);
        // mix ramps over 50 ms (2400 samples), so the dry level only drops a little per sample
        assert!(buf.iter().all(|&y| y > 0.99), "{buf:?}");
    }
}
