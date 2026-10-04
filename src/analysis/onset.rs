//! Onset (note attack) detection.

use super::Window;
use crate::processor::Processor;
use crate::fft::RealFft;
use crate::units::*;

/// A detected onset.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Onset<T> {
    /// estimated start, as a sample index into everything the detector has been fed
    pub position: u64,
    /// the detection function's value at the onset (how sudden and broadband the change was)
    pub strength: T,
}

/// Magnitudes are compressed as `log10(1 + COMPRESSION * |X|)` (|X| = 1 for a full-scale sine), so
/// quiet and loud attacks weigh alike.
const COMPRESSION: f64 = 1_000.0;
/// Frames between the compared spectra.
const LAG: usize = 2;

/// Streaming onset detector: SuperFlux-style spectral flux with adaptive peak picking (after Böck
/// and Widmer, 2013).
///
/// Each frame's log-compressed magnitude spectrum is compared with the one two frames earlier,
/// after spreading that one by a frequency-proportional maximum filter (about ±3%), so vibrato and
/// glides do not register as new energy; the rises are summed into a detection function. An onset
/// is a peak of that function that is the largest within 30 ms, stands `sensitivity` above the
/// average of the last 100 ms, and comes at least `min_interval` after the previous one.
///
/// Onsets are reported one frame (about 5 ms) after the frame that peaked, with their position
/// placed at the estimated start. Allocates only in `new`.
#[derive(Debug, Clone)]
pub struct OnsetDetector<T: Float> {
    sample_rate: T,
    fft: RealFft<T>,
    window: Vec<T>,
    scale: T,
    history: Vec<T>,
    write: usize,
    hop: usize,
    since: usize,
    frame: Vec<T>,
    spectrum: Vec<Complex<T>>,
    /// log magnitudes of this frame
    current: Vec<T>,
    /// max-filtered log magnitudes of the last LAG frames, a ring of `bins` rows
    past: Vec<T>,
    /// detection function of the last few frames, a ring
    odf: Vec<T>,
    frames: u64,
    samples: u64,
    pre_max: usize,
    pre_avg: usize,
    sensitivity: T,
    min_gap: u64,
    last_onset: Option<u64>,
}

/// Frames the peak picker waits after a candidate.
const POST: usize = 1;

impl<T: Float> OnsetDetector<T> {
    /// A frame of about 43 ms (2048 samples at 48 kHz) every eighth of a frame, sensitivity 0.006,
    /// at least 30 ms between onsets.
    pub fn new(sample_rate: T) -> Self {
        let fs = sample_rate.to_f64().unwrap_or(48_000.0);
        let fft_len = ((fs * 2_048.0 / 48_000.0).log2().round().exp2() as usize).max(64);
        let hop = fft_len / 8;
        let bins = fft_len / 2 + 1;
        let frames_in = |seconds: f64| ((seconds * fs / hop as f64).round() as usize).max(1);
        let (pre_max, pre_avg) = (frames_in(0.03), frames_in(0.1));
        let window: Vec<T> = Window::Hann.coefficients(fft_len);
        let sum = window.iter().fold(T::_ZERO, |acc, &w| acc + w);
        Self {
            sample_rate,
            fft: RealFft::new(fft_len),
            scale: T::_lit(2.0) / sum,
            window,
            history: vec![T::_ZERO; fft_len],
            write: 0,
            hop,
            since: 0,
            frame: vec![T::_ZERO; fft_len],
            spectrum: vec![Complex::zero(); bins],
            current: vec![T::_ZERO; bins],
            past: vec![T::_ZERO; LAG * bins],
            odf: vec![T::_ZERO; pre_avg + POST + 1],
            frames: 0,
            samples: 0,
            pre_max,
            pre_avg,
            sensitivity: T::_lit(0.006),
            min_gap: frames_in(0.03) as u64,
            last_onset: None,
        }
    }
    /// The sample rate in Hz.
    pub fn sample_rate(&self) -> T {
        self.sample_rate
    }
    /// Samples between frames.
    pub fn hop(&self) -> usize {
        self.hop
    }
    /// How far a peak must rise above the recent average (in detection-function units; lower
    /// finds softer onsets and more false ones).
    pub fn set_sensitivity(&mut self, threshold: T) {
        self.sensitivity = threshold._max(T::_ZERO);
    }
    /// Shortest time between two onsets, seconds.
    pub fn set_min_interval(&mut self, seconds: T) {
        let frames = (seconds * self.sample_rate / T::_lit(self.hop as f64))._round();
        self.min_gap = frames.to_f64().unwrap_or(1.0).max(1.0) as u64;
    }
    /// The detection function's latest value (for display).
    pub fn detection(&self) -> T {
        if self.frames == 0 { T::_ZERO } else { self.odf_at(self.frames - 1) }
    }
    /// Clears the history and the detection function.
    pub fn reset(&mut self) {
        self.history.iter_mut().chain(self.past.iter_mut()).chain(self.odf.iter_mut()).for_each(|s| *s = T::_ZERO);
        self.write = 0;
        self.since = 0;
        self.frames = 0;
        self.samples = 0;
        self.last_onset = None;
    }

    fn odf_at(&self, frame: u64) -> T {
        self.odf[(frame % self.odf.len() as u64) as usize]
    }

    /// Feeds samples, calling `on_onset` for each onset found.
    pub fn process(&mut self, block: &[T], mut on_onset: impl FnMut(Onset<T>)) {
        for &x in block {
            self.history[self.write] = x;
            self.write = (self.write + 1) % self.history.len();
            self.samples += 1;
            self.since += 1;
            if self.since >= self.hop {
                self.since = 0;
                self.analyze();
                if let Some(onset) = self.pick() {
                    on_onset(onset);
                }
            }
        }
    }

    fn analyze(&mut self) {
        let n = self.frame.len();
        for (i, (f, &w)) in self.frame.iter_mut().zip(&self.window).enumerate() {
            *f = self.history[(self.write + i) % n] * w;
        }
        self.fft.forward(&self.frame, &mut self.spectrum);
        let (scale, compression) = (self.scale, T::_lit(COMPRESSION));
        for (l, z) in self.current.iter_mut().zip(&self.spectrum) {
            *l = (T::_ONE + compression * scale * z.norm())._log10();
        }
        let bins = self.current.len();
        // compare with the max-filtered spectrum LAG frames back, then store this one filtered
        let slot = (self.frames % LAG as u64) as usize * bins;
        let reference = &mut self.past[slot..slot + bins];
        let mut flux = T::_ZERO;
        for (now, then) in self.current.iter().zip(reference.iter()) {
            flux = flux + (*now - *then)._max(T::_ZERO);
        }
        for (k, r) in reference.iter_mut().enumerate() {
            let spread = 1 + k * 3 / 100;
            let range = k.saturating_sub(spread)..(k + spread + 1).min(bins);
            *r = self.current[range].iter().fold(T::_ZERO, |m, &l| m._max(l));
        }
        let len = self.odf.len() as u64;
        self.odf[(self.frames % len) as usize] = flux / T::_lit(bins as f64);
        self.frames += 1;
    }

    /// Decides on the frame POST frames back now that the ones after it are known.
    fn pick(&mut self) -> Option<Onset<T>> {
        let latest = self.frames - 1;
        let candidate = latest.checked_sub(POST as u64)?;
        let value = self.odf_at(candidate);
        if value <= T::_ZERO || self.last_onset.is_some_and(|l| candidate - l < self.min_gap) {
            return None;
        }
        let since = |back: usize| candidate.saturating_sub(back as u64)..=latest;
        if since(self.pre_max).any(|f| self.odf_at(f) > value) {
            return None; // not the local maximum
        }
        let window = since(self.pre_avg);
        let count = T::_lit((window.end() - window.start() + 1) as f64);
        let mean = window.fold(T::_ZERO, |acc, f| acc + self.odf_at(f)) / count;
        if value < mean + self.sensitivity {
            return None;
        }
        self.last_onset = Some(candidate);
        // frame `candidate` ended (candidate + 1) * hop samples in; the attack sits ahead of the
        // end by a fixed fraction of the frame
        let end = (candidate + 1) * self.hop as u64;
        let position = end.saturating_sub(self.frame.len() as u64 * 11 / 32);
        Some(Onset { position, strength: value })
    }
}

impl<T: Float> Processor<T> for OnsetDetector<T> {
    /// Detects onsets in `block` (discarding them) and leaves it unchanged; use the inherent
    /// [`process`](OnsetDetector::process) to receive them.
    fn process(&mut self, block: &mut [T]) {
        OnsetDetector::process(self, block, |_| {});
    }
    fn reset(&mut self) {
        OnsetDetector::reset(self)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::osc::Noise;

    const FS: f64 = 48_000.0;

    fn detect(x: &[f64]) -> Vec<Onset<f64>> {
        let mut d = OnsetDetector::new(FS);
        let mut found = Vec::new();
        for block in x.chunks(256) {
            d.process(block, |o| found.push(o));
        }
        found
    }

    /// A plucked note: harmonics decaying exponentially from `start`.
    fn pluck(x: &mut [f64], start: usize, freq: f64, amplitude: f64) {
        for (i, s) in x[start..].iter_mut().enumerate() {
            let t = i as f64 / FS;
            let env = amplitude * (-t / 0.15).exp();
            *s += env * (1..=8).map(|h| (std::f64::consts::TAU * freq * h as f64 * t).sin() / h as f64).sum::<f64>();
        }
    }

    #[test]
    fn finds_every_pluck_and_nothing_else() {
        let starts = [4_800, 20_000, 31_000, 52_000, 60_500, 79_000, 97_000, 110_000];
        let notes = [(196.0, 0.5), (262.0, 0.1), (330.0, 0.3), (220.0, 0.05), (440.0, 0.4), (147.0, 0.2), (523.0, 0.02), (392.0, 0.3)];
        let mut x: Vec<f64> = Noise::<f64>::new(9).take(130_000).map(|n| 3e-4 * n).collect();
        for (&start, &(f, a)) in starts.iter().zip(&notes) {
            pluck(&mut x, start, f, a);
        }
        let found = detect(&x);
        let positions: Vec<u64> = found.iter().map(|o| o.position).collect();
        assert_eq!(found.len(), starts.len(), "found {positions:?}");
        for (o, &start) in found.iter().zip(&starts) {
            let error_ms = (o.position as f64 - start as f64) / FS * 1_000.0;
            assert!(error_ms.abs() < 10.0, "onset at {start}: found {} ({error_ms:.1} ms)", o.position);
        }
    }

    #[test]
    fn vibrato_and_steady_noise_are_not_onsets() {
        // a tone with vibrato (6 Hz, ±40 cents) and tremolo: one onset where it starts
        let mut x: Vec<f64> = vec![0.0; 4_800];
        let mut phase = 0.0;
        for i in 0..144_000 {
            let t = i as f64 / FS;
            let f = 330.0 * 2f64.powf(0.4 / 12.0 * (std::f64::consts::TAU * 6.0 * t).sin());
            phase += f / FS;
            let amp = 0.3 * (1.0 + 0.3 * (std::f64::consts::TAU * 4.0 * t).sin());
            x.push(amp * (1..=6).map(|h| (std::f64::consts::TAU * phase * h as f64).sin() / h as f64).sum::<f64>());
        }
        let found = detect(&x);
        assert_eq!(found.len(), 1, "{:?}", found.iter().map(|o| o.position).collect::<Vec<_>>());
        // stationary noise: one onset where it starts, nothing after
        let mut x = vec![0.0; 4_800];
        x.extend(Noise::<f64>::new(4).take(240_000).map(|n| 0.2 * n));
        let found = detect(&x);
        assert_eq!(found.len(), 1, "{:?}", found.iter().map(|o| o.position).collect::<Vec<_>>());
    }

    #[test]
    fn drum_hits_on_a_grid() {
        // noise bursts every 125 ms (16th notes at 120 BPM), alternating loud and soft
        let mut x = vec![0.0; 96_000];
        let mut noise = Noise::<f64>::new(2);
        let hits: Vec<usize> = (0..14).map(|i| 3_000 + i * 6_000).collect();
        for (i, &h) in hits.iter().enumerate() {
            let a = if i % 2 == 0 { 0.8 } else { 0.15 };
            for (j, s) in x[h..h + 4_000].iter_mut().enumerate() {
                *s += a * (-(j as f64) / 600.0).exp() * noise.next_sample();
            }
        }
        let found = detect(&x);
        assert_eq!(found.len(), hits.len(), "{:?}", found.iter().map(|o| o.position).collect::<Vec<_>>());
        for (o, &h) in found.iter().zip(&hits) {
            assert!((o.position as f64 - h as f64).abs() / FS < 0.01, "hit at {h}: found {}", o.position);
        }
    }
}
