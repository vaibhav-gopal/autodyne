//! Pitch with a voicing probability: probabilistic YIN (pYIN; Mauch and Dixon, 2014) and a hidden
//! Markov model over pitch and voicing, decoded as the audio streams.
//!
//! tend: Signal processing / analysis / voicing

use crate::alloc_prelude::*;
use crate::fft::RealFft;
use crate::processor::Processor;
use crate::special::betainc;
use crate::units::*;

/// One frame of a [`PitchTracker`]'s decoded track.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Voicing<T> {
    /// The frame's index: it covers samples `frame * hop .. frame * hop + frame_len` of the stream.
    pub frame: u64,
    /// The fundamental in Hz where the decoded track is voiced; `None` where it isn't.
    pub frequency: Option<T>,
    /// How likely the frame is voiced, 0..1, before smoothing: the probability its pitch candidates
    /// carry (pYIN's voiced probability).
    pub probability: T,
}

// pYIN's parameters, as the reference implementations have them (Mauch's pYIN Vamp plugin, librosa)
/// Thresholds the dips are weighed over.
const THRESHOLDS: usize = 100;
/// The thresholds' prior: Beta(2, 18), mean 0.1.
const BETA: (f64, f64) = (2.0, 18.0);
/// Earlier dips (shorter periods) are likelier: a Boltzmann prior over a dip's rank, this rate.
const BOLTZMANN: f64 = 2.0;
/// What a frame with no dip below a threshold gives its deepest dip, for that threshold.
const NO_TROUGH: f64 = 0.01;
/// Pitch states per semitone (10 cents apart).
const BINS_PER_SEMITONE: usize = 10;
/// The fastest pitch change the model allows, octaves per second.
const MAX_TRANSITION_RATE: f64 = 35.92;
/// The probability of switching between voiced and unvoiced from one frame to the next.
const SWITCH: f64 = 0.01;
/// Transitions less likely than this aren't searched (librosa's `transition_min_prob`).
const TRANSITION_MIN: f64 = 1e-4;
/// Added before every logarithm, as librosa does (`np.finfo(float).tiny`).
const TINY: f64 = f64::MIN_POSITIVE;

/// Monophonic pitch with voicing, for singing and speech: pYIN's candidates and probabilities per
/// frame, smoothed by a hidden Markov model whose states are pitches (10 cents apart) in a voiced
/// and an unvoiced layer.
///
/// YIN ([`PitchDetector`](super::PitchDetector)) takes the first dip of its difference function
/// below one threshold, and a breathy or low voice whose dips sit above it reads as unvoiced. pYIN
/// weighs every dip over a distribution of thresholds instead, so each candidate period gets a
/// probability, and their sum is how likely the frame is voiced; the model then picks the most
/// likely track through them, which bridges weak frames and rejects isolated octave jumps.
///
/// Decoding is fixed-lag Viterbi: a frame's decision comes `lag` frames after it is analyzed (the
/// track so far, traced back from its likeliest end), so it streams; [`finish`](Self::finish)
/// settles the last frames when the input ends, and [`track`](Self::track) decodes a whole
/// recording at once, exactly as offline pYIN does. The analysis follows librosa's `pyin`
/// (`center=False`) step for step; its outputs are this type's test fixtures.
///
/// Feed audio with [`push`](Self::push); it allocates only when built.
#[derive(Debug, Clone)]
pub struct PitchTracker<T: Float> {
    sample_rate: T,
    min_hz: T,
    max_hz: T,
    frame_len: usize,
    hop: usize,
    lag: usize,
    min_period: usize,
    max_period: usize,
    // YIN
    fft: RealFft<T>,
    padded: Vec<T>,
    spectrum: Vec<Complex<T>>,
    acf: Vec<T>,
    /// the difference function d(k), k = 0..=max_period
    difference: Vec<f64>,
    /// its cumulative-mean normalization, periods min_period..=max_period
    yin: Vec<f64>,
    troughs: Vec<usize>,
    trough_probs: Vec<f64>,
    // priors
    thresholds: Vec<f64>,
    beta_probs: Vec<f64>,
    /// exp(-BOLTZMANN k), by rank k
    rank_weight: Vec<f64>,
    /// the Boltzmann normalization for n dips, by n
    rank_norm: Vec<f64>,
    // the model
    bins: usize,
    width: usize,
    /// log transition from bin k to bin k + d - width / 2, within a layer and across, [k * width + d]
    log_within: Vec<f64>,
    log_across: Vec<f64>,
    log_obs: Vec<f64>,
    value: Vec<f64>,
    next: Vec<f64>,
    /// best predecessors, one row of 2 * bins per frame, a ring of lag + 1 frames
    ptrs: Vec<u16>,
    /// per frame of the ring: each bin's candidate frequency (NaN where none) and the voiced probability
    candidates: Vec<f64>,
    probabilities: Vec<f64>,
    path: Vec<u16>,
    analyzed: u64,
    decided: u64,
    // the stream
    history: Vec<T>,
    write: usize,
    seen: u64,
    frame: Vec<T>,
    latest: Option<Voicing<T>>,
}

impl<T: Float> PitchTracker<T> {
    /// Tracks fundamentals from `min_hz` to `max_hz`. Frames span two periods of `min_hz` (rounded
    /// up to a power of two), a quarter frame apart; decisions come about 50 ms of frames later.
    /// Panics unless 0 < min_hz < max_hz < sample_rate / 2.
    pub fn new(min_hz: T, max_hz: T, sample_rate: T) -> Self {
        let fs = sample_rate.to_f64().unwrap_or(48_000.0);
        let lo = min_hz.to_f64().unwrap_or(0.0);
        assert!(lo > 0.0, "need 0 < min_hz < max_hz < sample_rate / 2");
        let frame_len = (2 * (fs / lo).ceil() as usize).next_power_of_two();
        let hop = frame_len / 4;
        Self::build(min_hz, max_hz, sample_rate, frame_len, hop, (0.05 * fs / hop as f64).ceil() as usize)
    }

    /// The same tracker with frames of `frame_len` samples, `hop` apart (allocates). Frames shorter
    /// than a period of `min_hz` limit the longest period to `frame_len - 1`. Panics if `hop` is 0
    /// or the frame can't hold a period of `max_hz` and one more.
    pub fn with_frame(self, frame_len: usize, hop: usize) -> Self {
        Self::build(self.min_hz, self.max_hz, self.sample_rate, frame_len, hop, self.lag)
    }

    /// The same tracker deciding each frame `lag` frames after analyzing it (allocates). Longer is
    /// closer to offline decoding, shorter answers sooner; 0 takes each frame's likeliest state.
    pub fn with_lag(self, lag: usize) -> Self {
        Self::build(self.min_hz, self.max_hz, self.sample_rate, self.frame_len, self.hop, lag)
    }

    fn build(min_hz: T, max_hz: T, sample_rate: T, frame_len: usize, hop: usize, lag: usize) -> Self {
        let fs = sample_rate.to_f64().unwrap_or(48_000.0);
        let (lo, hi) = (min_hz.to_f64().unwrap_or(0.0), max_hz.to_f64().unwrap_or(0.0));
        assert!(lo > 0.0 && lo < hi && hi < fs / 2.0, "need 0 < min_hz < max_hz < sample_rate / 2");
        assert!(hop > 0, "hop must be at least 1");
        let min_period = (fs / hi).floor() as usize;
        let max_period = ((fs / lo).ceil() as usize).min(frame_len.saturating_sub(1));
        assert!(min_period >= 1 && max_period > min_period, "frame_len {frame_len} can't hold a period of max_hz and one more");
        let periods = max_period - min_period + 1;

        // the thresholds' prior: probability mass between consecutive thresholds 0, 0.01, ..., 1
        // (numpy's linspace: i * 0.01, the last exactly 1)
        let grid: Vec<f64> = (0..=THRESHOLDS).map(|i| if i == THRESHOLDS { 1.0 } else { i as f64 * (1.0 / THRESHOLDS as f64) }).collect();
        let cdf: Vec<f64> = grid.iter().map(|&x| betainc(BETA.0, BETA.1, x)).collect();
        let beta_probs = cdf.windows(2).map(|w| w[1] - w[0]).collect();
        let rank_weight = (0..periods).map(|k| (-BOLTZMANN * k as f64).exp()).collect();
        let rank_norm = (0..=periods).map(|n| (1.0 - (-BOLTZMANN).exp()) / (1.0 - (-BOLTZMANN * n as f64).exp())).collect();

        // pitch states: 10 per semitone from min_hz up
        let bins = (12.0 * BINS_PER_SEMITONE as f64 * (hi / lo).log2()).floor() as usize + 1;
        assert!(2 * bins <= u16::MAX as usize, "too many pitch states");
        let semitones = round_half_even(MAX_TRANSITION_RATE * 12.0 * hop as f64 / fs) as usize;
        let width = semitones * BINS_PER_SEMITONE + 1;
        let half = width / 2;
        // a triangle around the current bin (scipy's triang, symmetric), cut at the edges and
        // renormalized, then weighed by staying in or switching voicing (librosa's transition_local
        // and transition_loop)
        let triangle = |d: usize| 2.0 * ((d.min(width - 1 - d) + 1) as f64) / (width as f64 + 1.0);
        let stay = 1.0 - SWITCH;
        let switch = 1.0 - stay;
        let floor = (TRANSITION_MIN + TINY).ln();
        let (mut log_within, mut log_across) = (vec![f64::NEG_INFINITY; bins * width], vec![f64::NEG_INFINITY; bins * width]);
        for k in 0..bins {
            let inside = |d: usize| (k + d >= half) && (k + d - half < bins);
            let sum: f64 = (0..width).filter(|&d| inside(d)).map(triangle).sum();
            for d in (0..width).filter(|&d| inside(d)) {
                let p = triangle(d) / sum;
                for (table, w) in [(&mut log_within, stay), (&mut log_across, switch)] {
                    let l = (w * p + TINY).ln();
                    // pruned below the minimum, as librosa's Viterbi does
                    table[k * width + d] = if l >= floor { l } else { f64::NEG_INFINITY };
                }
            }
        }

        let n = (frame_len + frame_len - 1).next_power_of_two();
        let states = 2 * bins;
        Self {
            sample_rate,
            min_hz,
            max_hz,
            frame_len,
            hop,
            lag,
            min_period,
            max_period,
            fft: RealFft::new(n),
            padded: vec![T::_ZERO; n],
            spectrum: vec![Complex::zero(); n / 2 + 1],
            acf: vec![T::_ZERO; n],
            difference: vec![0.0; max_period + 1],
            yin: vec![0.0; periods],
            troughs: Vec::with_capacity(periods),
            trough_probs: Vec::with_capacity(periods),
            thresholds: grid[1..].to_vec(),
            beta_probs,
            rank_weight,
            rank_norm,
            bins,
            width,
            log_within,
            log_across,
            log_obs: vec![0.0; states],
            value: vec![0.0; states],
            next: vec![0.0; states],
            ptrs: vec![0; (lag + 1) * states],
            candidates: vec![f64::NAN; (lag + 1) * bins],
            probabilities: vec![0.0; lag + 1],
            path: vec![0; lag + 1],
            analyzed: 0,
            decided: 0,
            history: vec![T::_ZERO; frame_len],
            write: 0,
            seen: 0,
            frame: vec![T::_ZERO; frame_len],
            latest: None,
        }
    }

    /// Samples per frame.
    pub fn frame_len(&self) -> usize {
        self.frame_len
    }
    /// Samples between frames.
    pub fn hop(&self) -> usize {
        self.hop
    }
    /// Frames between analyzing a frame and deciding it.
    pub fn lag(&self) -> usize {
        self.lag
    }
    /// Samples from a frame's first sample to its decision: `frame_len + lag * hop`.
    pub fn latency(&self) -> usize {
        self.frame_len + self.lag * self.hop
    }
    /// The latest decided frame.
    pub fn voicing(&self) -> Option<Voicing<T>> {
        self.latest
    }

    /// Clears the stream and the model: the next sample starts frame 0.
    pub fn reset(&mut self) {
        self.history.iter_mut().for_each(|s| *s = T::_ZERO);
        self.write = 0;
        self.seen = 0;
        self.analyzed = 0;
        self.decided = 0;
        self.latest = None;
    }

    /// Feeds samples: analyzes a frame every hop and hands each decided frame to `emit`, in order.
    pub fn push(&mut self, block: &[T], mut emit: impl FnMut(Voicing<T>)) {
        let len = self.frame_len;
        for &x in block {
            self.history[self.write] = x;
            self.write = (self.write + 1) % len;
            self.seen += 1;
            if self.seen >= len as u64 && (self.seen - len as u64).is_multiple_of(self.hop as u64) {
                let mut frame = core::mem::take(&mut self.frame);
                for (i, f) in frame.iter_mut().enumerate() {
                    *f = self.history[(self.write + i) % len];
                }
                self.step(&frame);
                self.frame = frame;
                if self.analyzed > self.lag as u64 {
                    self.decide(1, &mut emit);
                }
            }
        }
    }

    /// Ends the stream: decides the frames still waiting (traced back from the likeliest end, as
    /// offline decoding does), then starts over as after [`reset`](Self::reset).
    pub fn finish(&mut self, mut emit: impl FnMut(Voicing<T>)) {
        let waiting = (self.analyzed - self.decided) as usize;
        if waiting > 0 {
            self.decide(waiting, &mut emit);
        }
        self.reset();
    }

    /// The decoded track of a whole recording, exactly as offline pYIN decodes it (one Viterbi pass
    /// over every frame). Allocates: two bytes per frame and pitch state for the trace-back.
    pub fn track(&self, signal: &[T]) -> Vec<Voicing<T>> {
        let frames = if signal.len() >= self.frame_len { (signal.len() - self.frame_len) / self.hop + 1 } else { 0 };
        let mut whole = Self::build(self.min_hz, self.max_hz, self.sample_rate, self.frame_len, self.hop, frames);
        let mut out = Vec::with_capacity(frames);
        whole.push(signal, |v| out.push(v));
        whole.finish(|v| out.push(v));
        out
    }

    /// Decides the oldest `count` undecided frames, tracing back from the likeliest state now.
    fn decide(&mut self, count: usize, emit: &mut impl FnMut(Voicing<T>)) {
        let ring = self.lag + 1;
        let states = 2 * self.bins;
        // the newest frame's likeliest state (the first, on a tie)
        let mut state = 0;
        for (j, &v) in self.value.iter().enumerate() {
            if v > self.value[state] {
                state = j;
            }
        }
        // back from the newest frame to the oldest undecided one
        let newest = self.analyzed - 1;
        let oldest = self.analyzed - (self.analyzed - self.decided);
        let mut frame = newest;
        loop {
            self.path[(frame % ring as u64) as usize] = state as u16;
            if frame == oldest {
                break;
            }
            state = self.ptrs[(frame % ring as u64) as usize * states + state] as usize;
            frame -= 1;
        }
        for _ in 0..count {
            let f = self.decided;
            let slot = (f % ring as u64) as usize;
            let state = self.path[slot] as usize;
            let frequency = (state < self.bins).then(|| {
                let found = self.candidates[slot * self.bins + state];
                let hz = if found.is_nan() {
                    self.min_hz.to_f64().unwrap_or(0.0) * (state as f64 / (12 * BINS_PER_SEMITONE) as f64).exp2()
                } else {
                    found
                };
                T::_lit(hz)
            });
            let v = Voicing { frame: f, frequency, probability: T::_lit(self.probabilities[slot]) };
            self.latest = Some(v);
            emit(v);
            self.decided += 1;
        }
    }

    /// Analyzes one frame and advances the model.
    fn step(&mut self, frame: &[T]) {
        let ring = self.lag + 1;
        let slot = (self.analyzed % ring as u64) as usize;
        let voiced = self.observe(frame, slot);
        self.probabilities[slot] = voiced;
        let (bins, width, half) = (self.bins, self.width, self.width / 2);
        if self.analyzed == 0 {
            let init = (1.0 / (2 * bins) as f64 + TINY).ln();
            for (v, &o) in self.value.iter_mut().zip(&self.log_obs) {
                *v = o + init;
            }
        } else {
            // V'(j) = obs(j) + max_k V(k) + log A(k, j): sources in index order (voiced then
            // unvoiced, bins ascending), the first best kept
            let ptrs = &mut self.ptrs[slot * 2 * bins..(slot + 1) * 2 * bins];
            for to_layer in 0..2 {
                for j in 0..bins {
                    let (lo, hi) = (j.saturating_sub(half), (j + half).min(bins - 1));
                    let (mut best, mut arg) = (f64::NEG_INFINITY, 0);
                    for from_layer in 0..2 {
                        let table = if from_layer == to_layer { &self.log_within } else { &self.log_across };
                        for k in lo..=hi {
                            let cost = self.value[from_layer * bins + k] + table[k * width + j + half - k];
                            if cost > best {
                                best = cost;
                                arg = from_layer * bins + k;
                            }
                        }
                    }
                    let to = to_layer * bins + j;
                    self.next[to] = self.log_obs[to] + best;
                    ptrs[to] = arg as u16;
                }
            }
            core::mem::swap(&mut self.value, &mut self.next);
            // keep the numbers small on endless streams (the decisions don't change)
            let top = self.value.iter().fold(f64::NEG_INFINITY, |m, &v| m.max(v));
            if top < -1e12 {
                self.value.iter_mut().for_each(|v| *v -= top);
            }
        }
        self.analyzed += 1;
    }

    /// pYIN's observation for one frame: the log probability of each state, each bin's candidate
    /// frequency into the ring at `slot`; returns the voiced probability.
    fn observe(&mut self, frame: &[T], slot: usize) -> f64 {
        let (min_period, max_period, bins) = (self.min_period, self.max_period, self.bins);
        // the autocorrelation of the whole frame, linear (zero padded)
        self.padded.iter_mut().for_each(|s| *s = T::_ZERO);
        self.padded[..frame.len()].copy_from_slice(frame);
        self.fft.forward(&self.padded, &mut self.spectrum);
        for s in self.spectrum.iter_mut() {
            *s = Complex::new(s.norm_sqr(), T::_ZERO);
        }
        self.fft.inverse(&self.spectrum, &mut self.acf);
        // d(k) = 2 (r(0) - r(k)) - sum_{m < k} y(m)^2 (librosa's whole-frame difference)
        let at = |x: T| x.to_f64().unwrap_or(0.0);
        let r0 = at(self.acf[0]);
        let mut energy = 0.0;
        self.difference[0] = 0.0;
        for k in 1..=max_period {
            energy += at(frame[k - 1]) * at(frame[k - 1]);
            self.difference[k] = 2.0 * (r0 - at(self.acf[k])) - energy;
        }
        // normalized by the running mean of d(1..=k)
        let mut running = 0.0;
        for k in 1..=max_period {
            running += self.difference[k];
            if k >= min_period {
                self.yin[k - min_period] = self.difference[k] / (running / k as f64 + TINY);
            }
        }

        // troughs: local minima, the ends compared with their one neighbour
        let y = &self.yin;
        let m = y.len();
        self.troughs.clear();
        for i in 0..m {
            let trough = if i == 0 {
                y[0] < y[1]
            } else if i == m - 1 {
                y[m - 1] < y[m - 2]
            } else {
                y[i] < y[i - 1] && y[i] <= y[i + 1]
            };
            if trough {
                self.troughs.push(i);
            }
        }
        let candidates = &mut self.candidates[slot * bins..(slot + 1) * bins];
        candidates.iter_mut().for_each(|c| *c = f64::NAN);
        self.log_obs[..bins].iter_mut().for_each(|o| *o = 0.0);
        let mut voiced = 0.0;
        if !self.troughs.is_empty() {
            // each threshold's prior shared among the troughs below it, earlier ones favoured
            self.trough_probs.clear();
            self.trough_probs.resize(self.troughs.len(), 0.0);
            for (j, &threshold) in self.thresholds.iter().enumerate() {
                let below = self.troughs.iter().filter(|&&i| y[i] < threshold).count();
                if below == 0 {
                    continue;
                }
                let norm = self.rank_norm[below];
                let mut rank = 0;
                for (p, &i) in self.trough_probs.iter_mut().zip(&self.troughs) {
                    if y[i] < threshold {
                        *p += norm * self.rank_weight[rank] * self.beta_probs[j];
                        rank += 1;
                    }
                }
            }
            // thresholds no trough is below go to the deepest
            let deepest = (0..self.troughs.len()).fold(0, |b, t| if y[self.troughs[t]] < y[self.troughs[b]] { t } else { b });
            let above = self.thresholds.iter().filter(|&&s| y[self.troughs[deepest]] >= s).count();
            self.trough_probs[deepest] += NO_TROUGH * self.beta_probs[..above].iter().sum::<f64>();

            // each candidate to its pitch bin (a later candidate in the same bin replaces an earlier
            // one, as in librosa)
            let fs = self.sample_rate.to_f64().unwrap_or(48_000.0);
            let fmin = self.min_hz.to_f64().unwrap_or(0.0);
            for (&i, &p) in self.troughs.iter().zip(&self.trough_probs) {
                if p == 0.0 {
                    continue;
                }
                // parabolic interpolation of the normalized difference (none at the ends, or past a neighbour)
                let shift = if i == 0 || i == m - 1 {
                    0.0
                } else {
                    let a = y[i + 1] + y[i - 1] - 2.0 * y[i];
                    let b = (y[i + 1] - y[i - 1]) / 2.0;
                    if b.abs() >= a.abs() { 0.0 } else { -b / a }
                };
                let f0 = fs / ((min_period + i) as f64 + shift);
                let bin = round_half_even((12 * BINS_PER_SEMITONE) as f64 * (f0 / fmin).log2()).max(0.0) as usize;
                if bin < bins {
                    self.log_obs[bin] = p;
                    candidates[bin] = f0;
                }
            }
            voiced = self.log_obs[..bins].iter().sum::<f64>().clamp(0.0, 1.0);
        }
        let unvoiced = (1.0 - voiced) / bins as f64;
        for (j, o) in self.log_obs.iter_mut().enumerate() {
            *o = (if j < bins { *o } else { unvoiced } + TINY).ln();
        }
        voiced
    }
}

/// Rounds to the nearest integer, ties to even (numpy's and Python's `round`).
fn round_half_even(x: f64) -> f64 {
    let r = x.round();
    if (x - x.trunc()).abs() == 0.5 && r % 2.0 != 0.0 { r - x.signum() } else { r }
}

impl<T: Float> Processor<T> for PitchTracker<T> {
    /// Tracks `block` and leaves it unchanged; the latest decision is [`voicing`](PitchTracker::voicing).
    fn process(&mut self, block: &mut [T]) {
        self.push(block, |_| {});
    }
    fn reset(&mut self) {
        PitchTracker::reset(self)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::osc::{Noise, Oscillator, Waveform};

    const FS: f64 = 16_000.0;

    fn cents(a: f64, b: f64) -> f64 {
        1_200.0 * (a / b).log2()
    }

    /// A saw at `f` Hz with `noise` of white noise added (relative to the saw's level).
    fn voice(f: f64, seconds: f64, noise: f64, seed: u64) -> Vec<f64> {
        let n = (seconds * FS) as usize;
        Oscillator::new(Waveform::Saw, f, FS).zip(Noise::<f64>::new(seed)).take(n).map(|(s, w)| 0.5 * s + 0.5 * noise * w).collect()
    }

    #[test]
    fn tracks_tones_and_marks_noise_and_silence_unvoiced() {
        let mut signal = vec![0.0; (0.3 * FS) as usize];
        signal.extend(voice(110.0, 0.6, 0.0, 1));
        signal.extend(Noise::<f64>::new(2).take((0.4 * FS) as usize).map(|x| 0.3 * x));
        signal.extend(voice(330.0, 0.6, 0.0, 3));
        let t = PitchTracker::new(60.0, 800.0, FS);
        let hop = t.hop() as f64 / FS;
        let track = t.track(&signal);
        // judged away from the edges, where frames straddle two segments
        let at = |s: f64| &track[(s / hop) as usize];
        for (s, want) in [(0.15, None), (0.6, Some(110.0)), (1.1, None), (1.6, Some(330.0))] {
            let v = at(s);
            match (want, v.frequency) {
                (None, None) => assert!(v.probability < 0.2, "{s} s: unvoiced with probability {}", v.probability),
                (Some(f), Some(got)) => {
                    assert!(cents(got, f).abs() < 5.0, "{s} s: {got} Hz, want {f}");
                    // a saw's whole-frame difference doesn't reach zero at its period, so not all
                    // thresholds pass its dip (librosa gives clean tones 0.5 to 0.95)
                    assert!(v.probability > 0.5, "{s} s: probability {}", v.probability)
                }
                _ => panic!("{s} s: {v:?}, want {want:?}"),
            }
        }
    }

    #[test]
    fn a_breathy_low_voice_stays_voiced_where_yin_drops_it() {
        // 85 Hz under noise half its level: YIN's fixed threshold rejects most frames
        let signal = voice(85.0, 1.0, 0.9, 7);
        let mut yin = crate::analysis::PitchDetector::new(60.0, 800.0, FS);
        let frames: Vec<&[f64]> = signal.chunks(yin.frame_len()).filter(|c| c.len() == yin.frame_len()).collect();
        let yin_voiced = frames.iter().filter(|f| yin.detect(f).is_some()).count() as f64 / frames.len() as f64;
        let track = PitchTracker::new(60.0, 800.0, FS).track(&signal);
        let settled = &track[8..track.len() - 8];
        let voiced: Vec<f64> = settled.iter().filter_map(|v| v.frequency).collect();
        let share = voiced.len() as f64 / settled.len() as f64;
        assert!(share > 0.95 && share > yin_voiced + 0.3, "pYIN voiced {share}, YIN {yin_voiced}");
        let worst = voiced.iter().map(|&f| cents(f, 85.0).abs()).fold(0.0, f64::max);
        assert!(worst < 30.0, "worst {worst} cents off 85 Hz");
    }

    #[test]
    fn streaming_with_a_long_lag_matches_offline_decoding() {
        let mut signal = voice(196.0, 0.5, 0.3, 11);
        signal.extend(voice(247.0, 0.5, 0.3, 12));
        let offline = PitchTracker::new(60.0, 800.0, FS).track(&signal);
        // a lag past the recording's length decides nothing until finish: the same pass
        let mut streamed = Vec::new();
        let mut t = PitchTracker::new(60.0, 800.0, FS).with_lag(offline.len());
        for block in signal.chunks(100) {
            t.push(block, |v| streamed.push(v));
        }
        assert!(streamed.is_empty());
        t.finish(|v| streamed.push(v));
        assert_eq!(streamed, offline);
        // the default lag decides as it goes, frame by frame in order, and agrees almost everywhere
        let mut live = Vec::new();
        let mut t = PitchTracker::new(60.0, 800.0, FS);
        for block in signal.chunks(100) {
            t.push(block, |v| live.push(v));
        }
        let before_finish = live.len();
        assert_eq!(before_finish, offline.len() - t.lag());
        t.finish(|v| live.push(v));
        assert!(live.iter().enumerate().all(|(i, v)| v.frame == i as u64));
        let same = live.iter().zip(&offline).filter(|(a, b)| a.frequency.is_some() == b.frequency.is_some()).count();
        assert!(same as f64 >= 0.97 * offline.len() as f64, "voicing agrees on {same} of {}", offline.len());
    }

    #[test]
    fn round_half_even_matches_numpy() {
        assert_eq!([0.5, 1.5, 2.5, -0.5, -1.5, 2.4, 2.6].map(round_half_even), [0.0, 2.0, 2.0, -0.0, -2.0, 2.0, 3.0]);
    }
}
