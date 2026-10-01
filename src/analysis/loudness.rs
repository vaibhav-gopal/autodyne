//! Loudness (ITU-R BS.1770 / EBU R128) and true peak.

use crate::channels::{AudioBuffer, MultiProcessor};
use crate::filter::{design_lowpass, Biquad, BiquadCoeffs};
use crate::processor::Processor;
use crate::units::*;

/// The two K-weighting stages of ITU-R BS.1770 for any sample rate: a high shelf (+4 dB above
/// about 1.7 kHz, the head's acoustic effect) and a high-pass at about 38 Hz. At 48 kHz they
/// equal the coefficients printed in the standard.
pub fn k_weighting<T: Float>(sample_rate: T) -> [BiquadCoeffs<T>; 2] {
    let fs = sample_rate.to_f64().unwrap_or(48_000.0);
    let pi = std::f64::consts::PI;
    let c = |x: f64| T::_lit(x);
    // the analog prototypes behind the standard's 48 kHz table, re-discretized for this rate
    let (f0, gain_db, q) = (1681.974450955533, 3.999843853973347, 0.7071752369554196);
    let k = (pi * f0 / fs).tan();
    let vh = 10f64.powf(gain_db / 20.0);
    let vb = vh.powf(0.4996667741545416);
    let a0 = 1.0 + k / q + k * k;
    let shelf = BiquadCoeffs {
        b0: c((vh + vb * k / q + k * k) / a0),
        b1: c(2.0 * (k * k - vh) / a0),
        b2: c((vh - vb * k / q + k * k) / a0),
        a1: c(2.0 * (k * k - 1.0) / a0),
        a2: c((1.0 - k / q + k * k) / a0),
    };
    let (f0, q) = (38.13547087602444, 0.5003270373238773);
    let k = (pi * f0 / fs).tan();
    let a0 = 1.0 + k / q + k * k;
    let highpass = BiquadCoeffs { b0: c(1.0), b1: c(-2.0), b2: c(1.0), a1: c(2.0 * (k * k - 1.0) / a0), a2: c((1.0 - k / q + k * k) / a0) };
    [shelf, highpass]
}

/// Taps per polyphase branch of the true-peak interpolator.
const TRUE_PEAK_TAPS: usize = 32;

/// True-peak detector (ITU-R BS.1770 annex 2): the signal is oversampled to at least 176.4 kHz
/// (4x below 88.2 kHz, 2x below 176.4 kHz) so peaks between samples are found, the overs a
/// sample-peak meter misses and a DAC or lossy codec then clips.
///
/// The interpolator is flat to about 0.41 x the sample rate (19.9 kHz at 48 kHz); content above
/// that reads slightly low. The reading also includes the samples themselves, so it is never
/// below the sample peak.
#[derive(Debug, Clone)]
pub struct TruePeak<T: Float> {
    factor: usize,
    /// one branch per output phase, each reversed (oldest input first)
    taps: Vec<T>,
    /// the last TRUE_PEAK_TAPS inputs, stored twice so they are always one contiguous slice
    history: Vec<T>,
    write: usize,
    peak: T,
}

impl<T: Float> TruePeak<T> {
    pub fn new(sample_rate: T) -> Self {
        let fs = sample_rate.to_f64().unwrap_or(48_000.0);
        let factor = if fs < 88_200.0 { 4 } else if fs < 176_400.0 { 2 } else { 1 };
        let mut taps = Vec::new();
        if factor > 1 {
            // odd length (one zero tap appended): the group delay is then a whole number of input
            // samples plus a whole phase, so the branches interpolate at exactly 1/factor steps
            // between the samples instead of straddling them
            let mut h: Vec<f64> = design_lowpass(fs / 2.0, fs * factor as f64, factor * TRUE_PEAK_TAPS - 1);
            h.push(0.0);
            for phase in 0..factor {
                // output phase p uses h[p + j * factor] on input x[n - j]
                taps.extend((0..TRUE_PEAK_TAPS).rev().map(|j| T::_lit(h[phase + j * factor] * factor as f64)));
            }
        }
        Self { factor, taps, history: vec![T::_ZERO; 2 * TRUE_PEAK_TAPS], write: 0, peak: T::_ZERO }
    }
    /// The oversampling factor (1, 2 or 4).
    pub fn factor(&self) -> usize {
        self.factor
    }
    /// Input samples by which the interpolated points trail the input: they lie between the
    /// samples `latency()` and `latency() - 1` back (0 without oversampling).
    pub fn latency(&self) -> usize {
        if self.factor == 1 { 0 } else { TRUE_PEAK_TAPS / 2 }
    }
    /// The largest absolute value among this sample and the interpolated points just before it.
    #[inline]
    pub fn process_sample(&mut self, x: T) -> T {
        x._abs()._max(self.interpolated_peak(x))
    }
    /// Feeds `x` and returns the largest interpolated point between the samples
    /// [`latency`](Self::latency) and `latency() - 1` back (not including the samples themselves;
    /// 0 without oversampling).
    #[inline]
    pub fn interpolated_peak(&mut self, x: T) -> T {
        if self.factor == 1 {
            return T::_ZERO;
        }
        self.history[self.write] = x;
        self.history[self.write + TRUE_PEAK_TAPS] = x;
        self.write = (self.write + 1) % TRUE_PEAK_TAPS;
        let recent = &self.history[self.write..self.write + TRUE_PEAK_TAPS];
        self.taps.as_chunks::<TRUE_PEAK_TAPS>().0.iter().fold(T::_ZERO, |m, branch| m._max(crate::simd::dot(recent, branch)._abs()))
    }
    /// Measures `block`, raising the held [`peak`](Self::peak).
    pub fn push(&mut self, block: &[T]) {
        let mut peak = self.peak;
        for &x in block {
            peak = peak._max(self.process_sample(x));
        }
        self.peak = peak;
    }
    /// Highest true peak since the last reset (linear).
    pub fn peak(&self) -> T {
        self.peak
    }
    /// Highest true peak in dBTP (-inf for silence).
    pub fn peak_db(&self) -> T {
        T::_lit(20.0) * self.peak._log10()
    }
    pub fn reset_peak(&mut self) {
        self.peak = T::_ZERO;
    }
    pub fn reset(&mut self) {
        self.history.iter_mut().for_each(|s| *s = T::_ZERO);
        self.write = 0;
        self.peak = T::_ZERO;
    }
}

impl<T: Float> Processor<T> for TruePeak<T> {
    /// Measures `block` and leaves it unchanged.
    fn process(&mut self, block: &mut [T]) {
        self.push(block);
    }
    fn reset(&mut self) {
        TruePeak::reset(self)
    }
}

/// Gating blocks below this are ignored entirely (silence and noise floors), LUFS.
const ABSOLUTE_GATE: f64 = -70.0;
const HISTOGRAM_STEP: f64 = 0.01;
const HISTOGRAM_BINS: usize = 10_000; // -70 .. +30 LUFS

fn loudness_of(mean_square: f64) -> f64 {
    -0.691 + 10.0 * mean_square.log10()
}

/// Gated measurements without storing every block: block counts and summed energies per
/// 0.01 LU of loudness, so gating at any threshold is exact to 0.01 LU and memory stays fixed.
#[derive(Debug, Clone)]
struct Histogram {
    counts: Vec<u64>,
    energies: Vec<f64>,
}

impl Histogram {
    fn new() -> Self {
        Self { counts: vec![0; HISTOGRAM_BINS], energies: vec![0.0; HISTOGRAM_BINS] }
    }
    fn add(&mut self, mean_square: f64) {
        let l = loudness_of(mean_square);
        if l > ABSOLUTE_GATE {
            let bin = (((l - ABSOLUTE_GATE) / HISTOGRAM_STEP) as usize).min(HISTOGRAM_BINS - 1);
            self.counts[bin] += 1;
            self.energies[bin] += mean_square;
        }
    }
    fn clear(&mut self) {
        self.counts.iter_mut().for_each(|c| *c = 0);
        self.energies.iter_mut().for_each(|e| *e = 0.0);
    }
    /// First bin whose center lies above `loudness`.
    fn first_bin_above(loudness: f64) -> usize {
        ((loudness - ABSOLUTE_GATE) / HISTOGRAM_STEP + 0.5).floor().clamp(0.0, HISTOGRAM_BINS as f64) as usize
    }
    fn bin_center(bin: usize) -> f64 {
        ABSOLUTE_GATE + (bin as f64 + 0.5) * HISTOGRAM_STEP
    }
    /// Loudness of the mean energy from `start` on.
    fn mean_loudness(&self, start: usize) -> Option<f64> {
        let count: u64 = self.counts[start..].iter().sum();
        (count > 0).then(|| loudness_of(self.energies[start..].iter().sum::<f64>() / count as f64))
    }
    /// The relative gate: blocks more than `relative_lu` below the absolute-gated mean are dropped.
    fn relative_start(&self, relative_lu: f64) -> Option<usize> {
        self.mean_loudness(0).map(|l| Self::first_bin_above(l + relative_lu))
    }
}

/// Loudness meter after ITU-R BS.1770-4 and EBU R128 (Tech 3341 / 3342): momentary (400 ms),
/// short-term (3 s) and gated integrated loudness in LUFS, loudness range in LU, and true peak.
///
/// Every channel is K-weighted; channel powers are summed with per-channel weights (1 by default;
/// for 5.1 set the surrounds to 1.41 and the LFE to 0). Readings update every 100 ms. Integrated
/// loudness and loudness range use histograms instead of a growing block list, so the meter never
/// allocates after `new` and can run for hours on the audio thread. Silence reads `-inf`.
#[derive(Debug, Clone)]
pub struct LoudnessMeter<T: Float> {
    filters: Vec<[Biquad<T>; 2]>,
    weights: Vec<f64>,
    true_peaks: Vec<TruePeak<T>>,
    /// samples per 100 ms step
    step_len: usize,
    step_fill: usize,
    /// weighted sum of squares so far in this step
    step_sum: f64,
    /// mean-square energies of the last 30 steps (3 s), a ring
    steps: [f64; 30],
    step_count: u64,
    momentary: f64,
    short_term: f64,
    max_momentary: f64,
    max_short_term: f64,
    blocks: Histogram,
    short_terms: Histogram,
}

impl<T: Float> LoudnessMeter<T> {
    /// Panics if `channels` is 0.
    pub fn new(channels: usize, sample_rate: T) -> Self {
        assert!(channels > 0, "a loudness meter needs at least one channel");
        let [shelf, highpass] = k_weighting(sample_rate);
        let fs = sample_rate.to_f64().unwrap_or(48_000.0);
        Self {
            filters: vec![[Biquad::new(shelf), Biquad::new(highpass)]; channels],
            weights: vec![1.0; channels],
            true_peaks: vec![TruePeak::new(sample_rate); channels],
            step_len: ((fs / 10.0).round() as usize).max(1),
            step_fill: 0,
            step_sum: 0.0,
            steps: [0.0; 30],
            step_count: 0,
            momentary: f64::NEG_INFINITY,
            short_term: f64::NEG_INFINITY,
            max_momentary: f64::NEG_INFINITY,
            max_short_term: f64::NEG_INFINITY,
            blocks: Histogram::new(),
            short_terms: Histogram::new(),
        }
    }
    pub fn channels(&self) -> usize {
        self.filters.len()
    }
    /// Power weight of `channel` (1 for front channels, 1.41 for surrounds, 0 to exclude the LFE).
    pub fn set_channel_weight(&mut self, channel: usize, weight: T) {
        self.weights[channel] = weight.to_f64().unwrap_or(1.0).max(0.0);
    }

    /// Measures one block per channel (`channels[c]`, all the same length). Panics if the
    /// channel count differs from the meter's.
    pub fn process(&mut self, channels: &[&[T]]) {
        assert_eq!(channels.len(), self.channels(), "one slice per channel");
        let frames = channels.iter().map(|c| c.len()).min().unwrap_or(0);
        self.run(frames, |c| channels[c]);
    }
    /// Measures every channel of `buffer`.
    pub fn process_buffer(&mut self, buffer: &AudioBuffer<T>) {
        assert_eq!(buffer.channels(), self.channels(), "the buffer's channel count must match the meter's");
        self.run(buffer.frames(), |c| buffer.channel(c));
    }

    fn run<'a>(&mut self, frames: usize, channel: impl Fn(usize) -> &'a [T])
    where
        T: 'a,
    {
        let mut start = 0;
        while start < frames {
            let n = (frames - start).min(self.step_len - self.step_fill);
            for (c, ([shelf, highpass], peak)) in self.filters.iter_mut().zip(self.true_peaks.iter_mut()).enumerate() {
                let input = &channel(c)[start..start + n];
                let mut sum = T::_ZERO;
                for &x in input {
                    let y = highpass.process_sample(shelf.process_sample(x));
                    sum = sum + y * y;
                }
                self.step_sum += self.weights[c] * sum.to_f64().unwrap_or(0.0);
                peak.push(input);
            }
            self.step_fill += n;
            start += n;
            if self.step_fill == self.step_len {
                self.finish_step();
            }
        }
    }

    fn finish_step(&mut self) {
        let ring = self.steps.len();
        self.steps[(self.step_count % ring as u64) as usize] = self.step_sum / self.step_len as f64;
        self.step_count += 1;
        self.step_sum = 0.0;
        self.step_fill = 0;
        for f in &mut self.filters {
            f.iter_mut().for_each(|b| b.flush_denormals());
        }
        let latest = |n: usize| (0..n).map(|i| self.steps[((self.step_count - 1 - i as u64) % ring as u64) as usize]).sum::<f64>() / n as f64;
        if self.step_count >= 4 {
            // a 400 ms gating block every 100 ms (75% overlap)
            let block = latest(4);
            self.momentary = loudness_of(block);
            self.max_momentary = self.max_momentary.max(self.momentary);
            self.blocks.add(block);
        }
        if self.step_count >= ring as u64 {
            let short = latest(ring);
            self.short_term = loudness_of(short);
            self.max_short_term = self.max_short_term.max(self.short_term);
            self.short_terms.add(short);
        }
    }

    /// Loudness of the last 400 ms, LUFS.
    pub fn momentary(&self) -> T {
        T::_lit(self.momentary)
    }
    /// Loudness of the last 3 s, LUFS.
    pub fn short_term(&self) -> T {
        T::_lit(self.short_term)
    }
    pub fn max_momentary(&self) -> T {
        T::_lit(self.max_momentary)
    }
    pub fn max_short_term(&self) -> T {
        T::_lit(self.max_short_term)
    }
    /// Gated loudness of everything measured since the last reset (BS.1770: blocks below -70 LUFS
    /// and those more than 10 LU below the remaining average are left out), LUFS.
    pub fn integrated(&self) -> T {
        let start = self.blocks.relative_start(-10.0);
        T::_lit(start.and_then(|s| self.blocks.mean_loudness(s)).unwrap_or(f64::NEG_INFINITY))
    }
    /// Loudness range (EBU Tech 3342): the spread between the 10th and 95th percentiles of the
    /// short-term loudness, after gating 20 LU below its average, in LU. 0 until 3 s are measured.
    pub fn loudness_range(&self) -> T {
        let Some(start) = self.short_terms.relative_start(-20.0) else {
            return T::_ZERO;
        };
        let counts = &self.short_terms.counts[start..];
        let total: u64 = counts.iter().sum();
        if total == 0 {
            return T::_ZERO;
        }
        let percentile = |p: f64| {
            let rank = ((total - 1) as f64 * p + 0.5) as u64;
            let mut seen = 0;
            for (i, &c) in counts.iter().enumerate() {
                seen += c;
                if seen > rank {
                    return Histogram::bin_center(start + i);
                }
            }
            Histogram::bin_center(HISTOGRAM_BINS - 1)
        };
        T::_lit(percentile(0.95) - percentile(0.10))
    }
    /// Highest true peak of any channel since the last reset, dBTP.
    pub fn true_peak_db(&self) -> T {
        let peak = self.true_peaks.iter().map(TruePeak::peak).fold(T::_ZERO, |m, p| m._max(p));
        T::_lit(20.0) * peak._log10()
    }
    pub fn channel_true_peak_db(&self, channel: usize) -> T {
        self.true_peaks[channel].peak_db()
    }

    /// Starts a new measurement (filters, readings, integrated loudness, range and peaks).
    pub fn reset(&mut self) {
        self.filters.iter_mut().flatten().for_each(Biquad::reset);
        self.true_peaks.iter_mut().for_each(TruePeak::reset);
        self.step_fill = 0;
        self.step_sum = 0.0;
        self.steps = [0.0; 30];
        self.step_count = 0;
        self.momentary = f64::NEG_INFINITY;
        self.short_term = f64::NEG_INFINITY;
        self.max_momentary = f64::NEG_INFINITY;
        self.max_short_term = f64::NEG_INFINITY;
        self.blocks.clear();
        self.short_terms.clear();
    }
}


impl<T: Float> MultiProcessor<T> for LoudnessMeter<T> {
    /// Measures `buffer` and leaves it unchanged.
    fn process(&mut self, buffer: &mut AudioBuffer<T>) {
        self.process_buffer(buffer);
    }
    fn reset(&mut self) {
        LoudnessMeter::reset(self)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::osc::Sine;

    /// Plays 1 kHz sine segments of (dBFS, seconds) on every channel of `meter`, a block at a time.
    fn play(meter: &mut LoudnessMeter<f64>, fs: f64, segments: &[(f64, f64)]) {
        let mut sine = Sine::new(1_000.0, fs);
        for &(db, seconds) in segments {
            let mut remaining = (seconds * fs).round() as usize;
            let gain = 10f64.powf(db / 20.0);
            while remaining > 0 {
                let block: Vec<f64> = (&mut sine).take(remaining.min(500)).map(|s| s * gain).collect();
                remaining -= block.len();
                let channels = vec![block.as_slice(); meter.channels()];
                meter.process(&channels);
            }
        }
    }

    #[test]
    fn k_weighting_matches_the_standard_at_48k() {
        let [shelf, highpass] = k_weighting(48_000.0);
        let expected: [(f64, f64); 10] = [
            (shelf.b0, 1.53512485958697),
            (shelf.b1, -2.69169618940638),
            (shelf.b2, 1.19839281085285),
            (shelf.a1, -1.69065929318241),
            (shelf.a2, 0.73248077421585),
            (highpass.b0, 1.0),
            (highpass.b1, -2.0),
            (highpass.b2, 1.0),
            (highpass.a1, -1.99004745483398),
            (highpass.a2, 0.99007225036621),
        ];
        for (i, (got, want)) in expected.into_iter().enumerate() {
            assert!((got - want).abs() < 1e-8, "coefficient {i}: {got} vs {want}");
        }
    }

    #[test]
    fn a_stereo_sine_at_minus_23_dbfs_reads_minus_23_lufs() {
        // EBU Tech 3341 test cases 1 and 2, at several sample rates
        for fs in [44_100.0, 48_000.0, 96_000.0] {
            for level in [-23.0, -33.0] {
                let mut meter = LoudnessMeter::new(2, fs);
                play(&mut meter, fs, &[(level, 6.0)]); // the standard plays 20 s; 6 covers the 3 s window twice
                for (name, reading) in [("momentary", meter.momentary()), ("short-term", meter.short_term()), ("integrated", meter.integrated())] {
                    assert!((reading - level).abs() < 0.1, "{fs} Hz {level} dBFS: {name} {reading}");
                }
            }
        }
        // one channel carries half the power: 3 dB less
        let mut mono = LoudnessMeter::new(1, 48_000.0);
        play(&mut mono, 48_000.0, &[(-23.0, 5.0)]);
        assert!((mono.integrated() + 26.01).abs() < 0.1, "{}", mono.integrated());
    }

    #[test]
    fn integrated_loudness_gates_quiet_passages() {
        // Tech 3341 cases 3 and 4: the quiet parts fall below the relative (or absolute) gate
        let fs = 48_000.0;
        let mut meter = LoudnessMeter::new(2, fs);
        play(&mut meter, fs, &[(-36.0, 10.0), (-23.0, 60.0), (-36.0, 10.0)]);
        assert!((meter.integrated() + 23.0).abs() < 0.1, "{}", meter.integrated());
        let mut meter = LoudnessMeter::new(2, fs);
        play(&mut meter, fs, &[(-72.0, 10.0), (-36.0, 10.0), (-23.0, 60.0), (-36.0, 10.0), (-72.0, 10.0)]);
        assert!((meter.integrated() + 23.0).abs() < 0.1, "{}", meter.integrated());
        assert!(meter.max_momentary() < -22.9 && meter.max_short_term() < -22.9);
        meter.reset();
        assert_eq!(meter.integrated(), f64::NEG_INFINITY, "nothing measured");
        play(&mut meter, fs, &[(-80.0, 2.0)]);
        assert_eq!(meter.integrated(), f64::NEG_INFINITY, "below the absolute gate");
    }

    #[test]
    fn loudness_range_spans_the_levels() {
        // Tech 3342 cases 1 and 2: 20 s at one level, then 20 s at another
        let fs = 48_000.0;
        for (a, b, range) in [(-20.0, -30.0, 10.0), (-20.0, -15.0, 5.0), (-40.0, -20.0, 20.0)] {
            let mut meter = LoudnessMeter::new(2, fs);
            play(&mut meter, fs, &[(a, 20.0), (b, 20.0)]);
            let lra = meter.loudness_range();
            assert!((lra - range).abs() < 1.0, "{a} then {b}: LRA {lra}");
        }
    }

    #[test]
    fn true_peak_finds_peaks_between_samples() {
        // a quarter-rate sine sampled 45 degrees off its peaks: samples reach 0.707, the wave 1.0
        for fs in [44_100.0, 48_000.0, 96_000.0, 192_000.0] {
            let mut tp = TruePeak::new(fs);
            let x: Vec<f64> = (0..4_800).map(|n| (std::f64::consts::FRAC_PI_2 * n as f64 + std::f64::consts::FRAC_PI_4).sin()).collect();
            tp.push(&x);
            if tp.factor() > 1 {
                assert!(tp.peak_db().abs() < 0.2, "{fs} Hz: {} dBTP", tp.peak_db());
            } else {
                assert!((tp.peak_db() + 3.01).abs() < 0.01, "no oversampling at 192 kHz");
            }
        }
        // in band, the reading is exact
        let mut tp = TruePeak::new(48_000.0);
        tp.push(&Sine::new(997.0, 48_000.0).take(48_000).map(|s| 0.5 * s).collect::<Vec<f64>>());
        assert!((tp.peak_db() + 6.0206).abs() < 0.02, "{}", tp.peak_db());
        // the meter reports the loudest channel
        let mut meter = LoudnessMeter::new(2, 48_000.0);
        let (loud, quiet) = (vec![0.9; 100], vec![0.1; 100]);
        meter.process(&[&quiet, &loud]);
        assert!(meter.true_peak_db() > meter.channel_true_peak_db(0));
    }
}
