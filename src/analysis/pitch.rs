//! Monophonic pitch detection (YIN).

use crate::processor::Processor;
use crate::fft::RealFft;
use crate::units::*;

/// One pitch estimate.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Pitch<T> {
    /// fundamental frequency in Hz
    pub frequency: T,
    /// how periodic the signal is, 0..1: 1 - YIN's normalized difference at the period
    /// (above 0.9 for clean tones, falling with noise and inharmonicity)
    pub clarity: T,
}

/// YIN pitch detector (de Cheveigné and Kawahara, 2002) for monophonic sources: tuners, pitch
/// tracking, pitch correction.
///
/// For each lag up to the longest period it measures how different the signal is from itself
/// shifted by that lag, normalizes by the running mean, and takes the first dip below a threshold
/// (refined between lags with a parabola), which avoids the octave-too-low errors of picking the
/// deepest dip. The difference function comes from FFT cross-correlation, so an estimate costs
/// two real FFTs instead of a quadratic sum.
///
/// Feed audio with [`process`](Self::process) (an estimate every hop, read with
/// [`pitch`](Self::pitch)) or analyze one window with [`detect`](Self::detect). Allocates only in `new`.
#[derive(Debug, Clone)]
pub struct PitchDetector<T: Float> {
    sample_rate: T,
    min_lag: usize,
    max_lag: usize,
    /// integration window, samples
    window: usize,
    threshold: T,
    fft: RealFft<T>,
    a: Vec<T>,
    b: Vec<T>,
    spectrum_a: Vec<Complex<T>>,
    spectrum_b: Vec<Complex<T>>,
    correlation: Vec<T>,
    /// cumulative-mean-normalized difference, by lag
    difference: Vec<T>,
    /// the raw difference, by lag (for interpolation)
    raw: Vec<f64>,
    /// the last window + max_lag samples, a ring
    history: Vec<T>,
    write: usize,
    frame: Vec<T>,
    hop: usize,
    since: usize,
    latest: Option<Pitch<T>>,
}

impl<T: Float> PitchDetector<T> {
    /// Detects fundamentals from `min_hz` to `max_hz`. The window spans one period of `min_hz`;
    /// estimates come every quarter window (`set_hop` to change). Threshold 0.15.
    /// Panics unless 0 < min_hz < max_hz < sample_rate / 4.
    pub fn new(min_hz: T, max_hz: T, sample_rate: T) -> Self {
        let fs = sample_rate.to_f64().unwrap_or(48_000.0);
        let (lo, hi) = (min_hz.to_f64().unwrap_or(0.0), max_hz.to_f64().unwrap_or(0.0));
        assert!(lo > 0.0 && lo < hi && hi < fs / 4.0, "need 0 < min_hz < max_hz < sample_rate / 4");
        let max_lag = (fs / lo).ceil() as usize + 1;
        let min_lag = ((fs / hi).floor() as usize).max(2);
        let window = max_lag;
        let n = (window + max_lag + 1).next_power_of_two();
        let bins = n / 2 + 1;
        Self {
            sample_rate,
            min_lag,
            max_lag,
            window,
            threshold: T::_lit(0.15),
            fft: RealFft::new(n),
            a: vec![T::_ZERO; n],
            b: vec![T::_ZERO; n],
            spectrum_a: vec![Complex::zero(); bins],
            spectrum_b: vec![Complex::zero(); bins],
            correlation: vec![T::_ZERO; n],
            difference: vec![T::_ZERO; max_lag + 2],
            raw: vec![0.0; max_lag + 2],
            history: vec![T::_ZERO; window + max_lag + 1],
            write: 0,
            frame: vec![T::_ZERO; window + max_lag + 1],
            hop: (window / 4).max(1),
            since: 0,
            latest: None,
        }
    }
    /// Samples one estimate needs: [`detect`](Self::detect) reads this many.
    pub fn frame_len(&self) -> usize {
        self.frame.len()
    }
    pub fn set_hop(&mut self, hop: usize) {
        self.hop = hop.max(1);
    }
    pub fn hop(&self) -> usize {
        self.hop
    }
    /// The dip threshold, 0.01..1: lower is stricter (fewer estimates on noisy or breathy sounds),
    /// higher accepts more but risks octave errors. YIN's authors suggest 0.1 to 0.15.
    pub fn set_threshold(&mut self, threshold: T) {
        self.threshold = threshold._clamp(T::_lit(0.01), T::_ONE);
    }
    /// The latest estimate from [`process`](Self::process); `None` while unvoiced or silent.
    pub fn pitch(&self) -> Option<Pitch<T>> {
        self.latest
    }
    pub fn reset(&mut self) {
        self.history.iter_mut().for_each(|s| *s = T::_ZERO);
        self.write = 0;
        self.since = 0;
        self.latest = None;
    }

    /// Feeds samples, estimating every hop. Returns true if a new estimate was made.
    pub fn process(&mut self, block: &[T]) -> bool {
        let len = self.history.len();
        let mut updated = false;
        for &x in block {
            self.history[self.write] = x;
            self.write = (self.write + 1) % len;
            self.since += 1;
            if self.since >= self.hop {
                self.since = 0;
                let mut frame = std::mem::take(&mut self.frame);
                for (i, f) in frame.iter_mut().enumerate() {
                    *f = self.history[(self.write + i) % len];
                }
                self.latest = self.detect(&frame);
                self.frame = frame;
                updated = true;
            }
        }
        updated
    }

    /// Estimates the pitch of `frame` (its first [`frame_len`](Self::frame_len) samples; panics if shorter).
    pub fn detect(&mut self, frame: &[T]) -> Option<Pitch<T>> {
        let (w, max_lag) = (self.window, self.max_lag);
        assert!(frame.len() > w + max_lag, "detect needs frame_len() samples");
        // r(tau) = sum_{j < w} x[j] x[j + tau] by FFT cross-correlation (zero padding stops wrap-around)
        self.a.iter_mut().for_each(|s| *s = T::_ZERO);
        self.b.iter_mut().for_each(|s| *s = T::_ZERO);
        self.a[..w].copy_from_slice(&frame[..w]);
        self.b[..w + max_lag + 1].copy_from_slice(&frame[..w + max_lag + 1]);
        self.fft.forward(&self.a, &mut self.spectrum_a);
        self.fft.forward(&self.b, &mut self.spectrum_b);
        for (sa, &sb) in self.spectrum_a.iter_mut().zip(&self.spectrum_b) {
            *sa = sa.conj() * sb;
        }
        self.fft.inverse(&self.spectrum_a, &mut self.correlation);

        // d(tau) = sum (x[j] - x[j + tau])^2 = e(0) + e(tau) - 2 r(tau), with e(tau) the energy of
        // the window shifted by tau (kept in f64: the sliding update would drift in f32)
        let energy = |s: &[T]| s.iter().fold(0.0, |acc, &x| acc + x.to_f64().unwrap_or(0.0).powi(2));
        let e0 = energy(&frame[..w]);
        if e0 <= 1e-20 * w as f64 {
            return None; // silence
        }
        let mut shifted = e0;
        let mut running = 0.0;
        self.difference[0] = T::_ONE;
        for tau in 1..=max_lag + 1 {
            let (gone, new) = (frame[tau - 1].to_f64().unwrap_or(0.0), frame[tau - 1 + w].to_f64().unwrap_or(0.0));
            shifted += new * new - gone * gone;
            let d = (e0 + shifted - 2.0 * self.correlation[tau].to_f64().unwrap_or(0.0)).max(0.0);
            running += d;
            self.raw[tau] = d;
            // cumulative mean normalization: d'(tau) = d(tau) tau / sum_{1..=tau} d
            self.difference[tau] = T::_lit(if running > 0.0 { d * tau as f64 / running } else { 1.0 });
        }

        let dn = &self.difference;
        let mut best = None;
        let mut tau = self.min_lag;
        while tau <= max_lag {
            if dn[tau] < self.threshold {
                // follow the dip down to its bottom
                while tau < max_lag && dn[tau + 1] < dn[tau] {
                    tau += 1;
                }
                best = Some(tau);
                break;
            }
            tau += 1;
        }
        let tau = best?;
        // parabola through the dip's three points of the raw difference: the normalization would
        // bias the vertex toward longer lags
        let (l, c, r) = (self.raw[tau - 1], self.raw[tau], self.raw[tau + 1]);
        let curvature = l + r - 2.0 * c;
        let shift = if curvature > 0.0 { (0.5 * (l - r) / curvature).clamp(-0.5, 0.5) } else { 0.0 };
        let period = T::_lit(tau as f64 + shift);
        Some(Pitch { frequency: self.sample_rate / period, clarity: (T::_ONE - dn[tau])._clamp(T::_ZERO, T::_ONE) })
    }
}

impl<T: Float> Processor<T> for PitchDetector<T> {
    /// Tracks the pitch of `block` and leaves it unchanged.
    fn process(&mut self, block: &mut [T]) {
        PitchDetector::process(self, block);
    }
    fn reset(&mut self) {
        PitchDetector::reset(self)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::osc::{Noise, Oscillator, Sine, Waveform};

    const FS: f64 = 48_000.0;

    fn cents(a: f64, b: f64) -> f64 {
        1_200.0 * (a / b).log2()
    }

    #[test]
    fn sines_across_the_range() {
        let mut d = PitchDetector::new(40.0, 2_000.0, FS);
        for f in [41.2, 55.0, 98.0, 220.0, 440.0, 1_046.5, 1_975.5] {
            let x: Vec<f64> = Sine::new(f, FS).take(d.frame_len()).collect();
            let p = d.detect(&x).expect("voiced");
            assert!(cents(p.frequency, f).abs() < 1.0, "{f} Hz -> {} Hz", p.frequency);
            assert!(p.clarity > 0.95, "{f} Hz clarity {}", p.clarity);
        }
    }

    #[test]
    fn rich_tones_and_a_missing_fundamental() {
        let mut d = PitchDetector::new(50.0, 1_500.0, FS);
        for (wave, f) in [(Waveform::Saw, 98.0), (Waveform::Pulse { pulse_width: 0.3 }, 196.0), (Waveform::Triangle, 330.0), (Waveform::Saw, 1_200.0)] {
            let x: Vec<f64> = Oscillator::new(wave, f, FS).take(d.frame_len()).collect();
            let p = d.detect(&x).expect("voiced");
            assert!(cents(p.frequency, f).abs() < 2.0, "{wave:?} {f} Hz -> {} Hz", p.frequency);
        }
        // harmonics 2..=5 of 150 Hz and no fundamental: the period is still 1 / 150 s
        let x: Vec<f64> = (0..d.frame_len())
            .map(|n| (2..=5).map(|h| (std::f64::consts::TAU * 150.0 * h as f64 * n as f64 / FS).sin() / h as f64).sum())
            .collect();
        let p = d.detect(&x).expect("voiced");
        assert!(cents(p.frequency, 150.0).abs() < 2.0, "missing fundamental -> {} Hz", p.frequency);
    }

    #[test]
    fn noise_and_silence_are_unvoiced() {
        let mut d = PitchDetector::new(50.0, 1_500.0, FS);
        let noise: Vec<f64> = Noise::<f64>::new(5).take(d.frame_len()).collect();
        assert_eq!(d.detect(&noise), None);
        assert_eq!(d.detect(&vec![0.0; d.frame_len()]), None);
        // a tone in moderate noise is still found, with lower clarity
        let clean: Vec<f64> = Sine::new(220.0, FS).take(d.frame_len()).collect();
        let clean = d.detect(&clean).unwrap().clarity;
        let noisy: Vec<f64> = Sine::new(220.0, FS).zip(Noise::<f64>::new(6)).take(d.frame_len()).map(|(s, n)| s + 0.15 * n).collect();
        let p = d.detect(&noisy).expect("voiced");
        assert!(cents(p.frequency, 220.0).abs() < 5.0 && p.clarity < clean - 0.01, "{p:?} vs clean clarity {clean}");
    }

    #[test]
    fn streaming_tracks_a_melody() {
        let mut d = PitchDetector::<f32>::new(60.0, 1_000.0, 48_000.0);
        let mut block = vec![0.0f32; 128];
        for f in [110.0f32, 165.0, 247.5, 440.0] {
            let mut osc = Oscillator::new(Waveform::Saw, f, 48_000.0);
            for _ in 0..(0.15 * 48_000.0 / 128.0) as usize {
                osc.fill(&mut block);
                d.process(&block);
            }
            let p = d.pitch().expect("voiced");
            assert!(cents(p.frequency as f64, f as f64).abs() < 3.0, "{f} Hz -> {} Hz", p.frequency);
        }
        d.reset();
        assert_eq!(d.pitch(), None);
    }
}
