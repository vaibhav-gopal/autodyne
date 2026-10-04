//! Lookahead brickwall limiter.

use crate::analysis::TruePeak;
use crate::channels::{AudioBuffer, MultiProcessor};

use crate::units::*;

/// Brickwall limiter with lookahead: the gain starts falling *before* a peak arrives, in a smooth
/// ramp that reaches exactly the needed reduction on the peak, so loud material is held under the
/// ceiling without the distortion of an instant gain change.
///
/// Per frame, the gain each sample needs (ceiling / its peak, across every channel: linked) is
/// smoothed by the release (instant down, exponential back up), held for the lookahead time
/// (a sliding minimum) and averaged over the lookahead (a box filter): the averaged ramp is never
/// above the held value it ends on, so no sample is under-limited. With true-peak detection, peaks
/// between samples (found by 4x oversampling) count too. A final clip catches rounding, so output
/// samples never exceed the ceiling.
///
/// Audio is delayed by [`latency`](Self::latency) samples (report it to the host). Multichannel by
/// nature; process an [`AudioBuffer`] or use [`process_mono`](Self::process_mono).
#[derive(Debug, Clone)]
pub struct LookaheadLimiter<T: Float> {
    sample_rate: T,
    ceiling_db: T,
    ceiling: T,
    release_seconds: T,
    release_coeff: T,
    true_peak: bool,
    lookahead: usize,
    detectors: Vec<TruePeak<T>>,
    /// true-peak delay: interpolated points trail the input by this many samples
    detect_delay: usize,
    /// per channel, the last lookahead + detect_delay + 1 inputs (a ring)
    delays: Vec<Vec<T>>,
    write: usize,
    /// release-smoothed required gain
    held: T,
    /// sliding minimum over lookahead + 1 frames: a monotonic deque of (gain, frame) in a ring
    min_gain: Vec<T>,
    min_time: Vec<u64>,
    min_head: usize,
    min_len: usize,
    /// box filter over lookahead frames, summed in f64 and re-summed each lap to avoid drift
    box_values: Vec<f64>,
    box_sum: f64,
    box_pos: usize,
    /// frames processed (timestamps for the sliding minimum)
    clock: u64,
    gain: T,
    /// one frame being processed, one sample per channel
    frame: Vec<T>,
}

impl<T: Float> LookaheadLimiter<T> {
    /// Ceiling -0.3 dB, true-peak detection on, 5 ms lookahead, 100 ms release.
    /// Panics if `channels` is 0.
    pub fn new(channels: usize, sample_rate: T) -> Self {
        Self::with_lookahead(channels, sample_rate, T::_lit(0.005))
    }
    /// With a lookahead time (at least one sample): longer gives gentler attacks and more latency.
    pub fn with_lookahead(channels: usize, sample_rate: T, seconds: T) -> Self {
        assert!(channels > 0, "a limiter needs at least one channel");
        let lookahead = (seconds * sample_rate)._round().to_f64().unwrap_or(1.0).max(1.0) as usize;
        let detector = TruePeak::new(sample_rate);
        let detect_delay = detector.latency();
        let mut limiter = Self {
            sample_rate,
            ceiling_db: T::_ZERO,
            ceiling: T::_ONE,
            release_seconds: T::_ZERO,
            release_coeff: T::_ZERO,
            true_peak: true,
            lookahead,
            detectors: vec![detector; channels],
            detect_delay,
            delays: vec![vec![T::_ZERO; lookahead + detect_delay + 1]; channels],
            write: 0,
            held: T::_ONE,
            min_gain: vec![T::_ONE; lookahead + 1],
            min_time: vec![0; lookahead + 1],
            min_head: 0,
            min_len: 0,
            box_values: vec![1.0; lookahead],
            box_sum: lookahead as f64,
            box_pos: 0,
            clock: 0,
            gain: T::_ONE,
            frame: vec![T::_ZERO; channels],
        };
        limiter.set_ceiling_db(T::_lit(-0.3));
        limiter.set_release(T::_lit(0.1));
        limiter
    }
    pub fn channels(&self) -> usize {
        self.delays.len()
    }
    /// Delay of the output, in samples.
    pub fn latency(&self) -> usize {
        self.lookahead + self.detect_delay
    }
    /// Highest output level in dBFS (dBTP with true-peak detection).
    pub fn set_ceiling_db(&mut self, db: T) {
        self.ceiling_db = db._min(T::_ZERO);
        self.ceiling = db_to_gain(self.ceiling_db);
    }
    pub fn ceiling_db(&self) -> T {
        self.ceiling_db
    }
    /// Time constant of the gain's recovery after a peak.
    pub fn set_release(&mut self, seconds: T) {
        self.release_seconds = seconds._max(T::_ZERO);
        self.release_coeff = if self.release_seconds > T::_ZERO { (-T::_ONE / (self.release_seconds * self.sample_rate))._exp() } else { T::_ZERO };
    }
    pub fn release(&self) -> T {
        self.release_seconds
    }
    /// Also limit peaks between samples (inter-sample overs that a DAC or codec would clip).
    pub fn set_true_peak(&mut self, on: bool) {
        self.true_peak = on;
    }
    pub fn true_peak(&self) -> bool {
        self.true_peak
    }
    /// Current gain reduction in dB (positive = turning down).
    pub fn gain_reduction_db(&self) -> T {
        -gain_to_db(self.gain)
    }
    pub fn reset(&mut self) {
        self.detectors.iter_mut().for_each(TruePeak::reset);
        self.delays.iter_mut().flatten().for_each(|s| *s = T::_ZERO);
        self.write = 0;
        self.held = T::_ONE;
        self.min_len = 0;
        self.box_values.iter_mut().for_each(|v| *v = 1.0);
        self.box_sum = self.lookahead as f64;
        self.box_pos = 0;
        self.gain = T::_ONE;
    }

    /// The gain for the frame now leaving the delay, given the newest frame's peak requirement.
    #[inline]
    fn next_gain(&mut self, required: T) -> T {
        // release: drop at once, recover exponentially
        self.held = if required < self.held { required } else { required + (self.held - required) * self.release_coeff };
        // sliding minimum over the last lookahead + 1 frames
        let cap = self.min_gain.len();
        while self.min_len > 0 && self.min_gain[(self.min_head + self.min_len - 1) % cap] >= self.held {
            self.min_len -= 1;
        }
        let tail = (self.min_head + self.min_len) % cap;
        self.min_gain[tail] = self.held;
        self.min_time[tail] = self.clock;
        self.min_len += 1;
        while self.min_time[self.min_head] + (self.lookahead as u64) < self.clock {
            self.min_head = (self.min_head + 1) % cap;
            self.min_len -= 1;
        }
        let minimum = self.min_gain[self.min_head].to_f64().unwrap_or(1.0);
        self.clock += 1;
        // box filter: a linear ramp ending on the minimum
        self.box_sum += minimum - self.box_values[self.box_pos];
        self.box_values[self.box_pos] = minimum;
        self.box_pos += 1;
        if self.box_pos == self.lookahead {
            self.box_pos = 0;
            self.box_sum = self.box_values.iter().sum();
        }
        self.gain = T::_lit(self.box_sum / self.lookahead as f64);
        self.gain
    }

    /// Limits the frame in `self.frame` in place.
    #[inline]
    fn process_frame(&mut self) {
        let len = self.delays[0].len();
        let (back_out, back_detect) = (self.lookahead + self.detect_delay, self.detect_delay);
        let mut peak = T::_ZERO;
        for (c, ring) in self.delays.iter_mut().enumerate() {
            let x = self.frame[c];
            ring[self.write] = x;
            let at = |back: usize| ring[(self.write + len - back) % len];
            // the samples at the detection point (and the next), plus the points between them
            peak = peak._max(at(back_detect)._abs())._max(at(back_detect.saturating_sub(1))._abs());
            if self.true_peak {
                peak = peak._max(self.detectors[c].interpolated_peak(x));
            }
        }
        let required = if peak > self.ceiling { self.ceiling / peak } else { T::_ONE };
        let gain = self.next_gain(required);
        let ceiling = self.ceiling;
        for (f, ring) in self.frame.iter_mut().zip(&self.delays) {
            *f = (ring[(self.write + len - back_out) % len] * gain)._clamp(-ceiling, ceiling);
        }
        self.write = (self.write + 1) % len;
    }

    /// Limits every channel of `buffer` (linked). Panics if the channel count differs.
    pub fn process_buffer(&mut self, buffer: &mut AudioBuffer<T>) {
        assert_eq!(buffer.channels(), self.channels(), "the buffer's channel count must match the limiter's");
        for i in 0..buffer.frames() {
            for c in 0..self.channels() {
                self.frame[c] = buffer.channel(c)[i];
            }
            self.process_frame();
            for c in 0..self.channels() {
                buffer.channel_mut(c)[i] = self.frame[c];
            }
        }
    }
    /// Limits a single channel. Panics unless the limiter has one channel.
    pub fn process_mono(&mut self, block: &mut [T]) {
        assert_eq!(self.channels(), 1, "process_mono needs a one-channel limiter");
        for s in block {
            self.frame[0] = *s;
            self.process_frame();
            *s = self.frame[0];
        }
    }
}
impl<T: Float> MultiProcessor<T> for LookaheadLimiter<T> {
    fn process(&mut self, buffer: &mut AudioBuffer<T>) {
        self.process_buffer(buffer);
    }
    fn reset(&mut self) {
        LookaheadLimiter::reset(self)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::analysis::TruePeak;
    use crate::osc::{Noise, Sine};

    const FS: f64 = 48_000.0;

    fn run_mono(limiter: &mut LookaheadLimiter<f64>, input: &[f64]) -> Vec<f64> {
        let mut out = input.to_vec();
        for block in out.chunks_mut(333) {
            limiter.process_mono(block);
        }
        out
    }

    #[test]
    fn holds_the_ceiling_and_delays_by_its_latency() {
        let mut limiter = LookaheadLimiter::new(1, FS);
        limiter.set_ceiling_db(-6.0);
        // loud noise with spikes
        let mut input: Vec<f64> = Noise::<f64>::new(1).take(48_000).map(|n| 0.8 * n).collect();
        for i in (1_000..48_000).step_by(7_919) {
            input[i] = 3.0;
        }
        let out = run_mono(&mut limiter, &input);
        let ceiling = db_to_gain(-6.0);
        let peak = out.iter().fold(0.0f64, |m, x| m.max(x.abs()));
        assert!(peak <= ceiling, "peak {peak} over {ceiling}");
        // quiet material comes through untouched, just late
        let mut limiter = LookaheadLimiter::new(1, FS);
        let quiet: Vec<f64> = Sine::new(440.0, FS).take(4_800).map(|s| 0.25 * s).collect();
        let out = run_mono(&mut limiter, &quiet);
        let latency = limiter.latency();
        assert_eq!(latency, 240 + 16);
        for i in latency..4_800 {
            assert!((out[i] - quiet[i - latency]).abs() < 1e-12, "sample {i}");
        }
    }

    #[test]
    fn the_gain_ramps_down_ahead_of_a_peak_and_recovers_with_the_release() {
        let mut limiter = LookaheadLimiter::new(1, FS);
        limiter.set_ceiling_db(-6.0);
        limiter.set_true_peak(false);
        let mut input = vec![0.1; 14_400];
        input[4_800] = 1.0;
        let out = run_mono(&mut limiter, &input);
        let latency = limiter.latency();
        let peak_out = 4_800 + latency;
        assert!((out[peak_out] - db_to_gain(-6.0)).abs() < 1e-9, "the peak lands exactly on the ceiling: {}", out[peak_out]);
        // a linear ramp over the lookahead before it (one sample longer: the detector also covers
        // the neighbouring sample), untouched before that
        let gain = |i: usize| out[i] / 0.1;
        assert!((gain(peak_out - 242) - 1.0).abs() < 1e-12);
        assert!(gain(peak_out - 241) < 1.0);
        let target = db_to_gain(-6.0);
        let mid = gain(peak_out - 120);
        assert!((mid - (1.0 + target) / 2.0).abs() < 0.01, "halfway down the ramp: {mid}");
        // recovers: 100 ms later about 63% of the way back
        let later = gain(peak_out + 4_800);
        let expected = 1.0 - (1.0 - target) * (-1.0f64).exp();
        assert!((later - expected).abs() < 0.02, "{later} vs {expected}");
    }

    #[test]
    fn true_peak_mode_catches_inter_sample_overs() {
        // a quarter-rate sine 45 degrees off its peaks: samples at 0.707, the wave at 1.0
        let input: Vec<f64> = (0..9_600).map(|n| (std::f64::consts::FRAC_PI_2 * n as f64 + std::f64::consts::FRAC_PI_4).sin()).collect();
        let true_peak_of = |x: &[f64]| {
            let mut tp = TruePeak::new(FS);
            tp.push(&x[2_000..]);
            tp.peak_db()
        };
        for (true_peak, expected_db) in [(false, 3.0), (true, 0.0)] {
            let mut limiter = LookaheadLimiter::new(1, FS);
            limiter.set_ceiling_db(-3.0);
            limiter.set_true_peak(true_peak);
            let out = run_mono(&mut limiter, &input);
            let tp = true_peak_of(&out) + 3.0;
            assert!((tp - expected_db).abs() < 0.3, "true peak {true_peak}: {tp} dB over the ceiling");
        }
    }

    #[test]
    fn channels_are_linked() {
        let mut limiter = LookaheadLimiter::new(2, FS);
        limiter.set_ceiling_db(-6.0);
        limiter.set_true_peak(false); // a DC step overshoots between samples; keep this about linking
        let mut buffer = AudioBuffer::new(2, 4_800);
        buffer.channel_mut(0).iter_mut().for_each(|s| *s = 1.0);
        buffer.channel_mut(1).iter_mut().for_each(|s| *s = 0.1);
        limiter.process_buffer(&mut buffer);
        let (l, r) = (buffer.channel(0)[4_000], buffer.channel(1)[4_000]);
        assert!((l / r - 10.0).abs() < 1e-9, "the balance is kept: {l} / {r}");
        assert!((limiter.gain_reduction_db() - 6.0).abs() < 0.01);
    }
}
