//! A running spectrum for analyzers and GUIs.

use crate::processor::Processor;
use crate::fft::RealFft;
use crate::units::*;

/// Analysis window: trades frequency resolution (main-lobe width) against leakage (side lobes).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum Window {
    /// no window: the narrowest peak, but -13 dB side lobes
    Rectangular,
    /// -31 dB side lobes falling quickly: the usual choice
    Hann,
    /// 4-term Blackman-Harris: -92 dB side lobes, a wider peak; for a wide dynamic range
    BlackmanHarris,
}

impl Window {
    /// Every window, in parameter order.
    pub const ALL: [Window; 3] = [Window::Rectangular, Window::Hann, Window::BlackmanHarris];
    /// Display names, in the order of [`ALL`](Self::ALL).
    pub const NAMES: [&'static str; 3] = ["Rectangular", "Hann", "Blackman-Harris"];

    /// `len` coefficients in the periodic form used for spectral analysis (the window's period
    /// is `len`, so it tiles exactly at 75% overlap for Hann).
    pub fn coefficients<T: Float>(self, len: usize) -> Vec<T> {
        let terms: &[f64] = match self {
            Window::Rectangular => &[1.0],
            Window::Hann => &[0.5, 0.5],
            Window::BlackmanHarris => &[0.35875, 0.48829, 0.14128, 0.01168],
        };
        (0..len)
            .map(|n| {
                let x = std::f64::consts::TAU * n as f64 / len as f64;
                // a0 - a1 cos x + a2 cos 2x - a3 cos 3x
                let w = terms.iter().enumerate().fold(0.0, |acc, (k, a)| acc + if k % 2 == 0 { *a } else { -*a } * (k as f64 * x).cos());
                T::_lit(w)
            })
            .collect()
    }
}

/// Sliding-window magnitude spectrum with analyzer ballistics: feed it audio, read levels in dB.
///
/// Every `hop` samples the latest `fft_len` samples are windowed and transformed. Levels are
/// calibrated so a full-scale sine centered on a bin reads 0 dB in that bin. Displayed levels
/// rise at once and fall exponentially (the release time), and a peak trace falls at a constant
/// rate in dB per second. [`bands_db`](Self::bands_db) maps the bins onto log-spaced bands for drawing.
///
/// It allocates only in `new`, so it can run on the audio thread; as a [`Processor`] it passes
/// audio through unchanged (a tap in a chain).
#[derive(Debug, Clone)]
pub struct SpectrumAnalyzer<T: Float> {
    sample_rate: T,
    fft: RealFft<T>,
    window: Vec<T>,
    window_kind: Window,
    /// amplitude calibration: 2 / sum(window)
    scale: T,
    /// the last fft_len input samples, a ring
    history: Vec<T>,
    write: usize,
    hop: usize,
    since_frame: usize,
    frame: Vec<T>,
    spectrum: Vec<Complex<T>>,
    levels: Vec<T>,
    peaks: Vec<T>,
    release_seconds: T,
    /// per-frame multiplier for the distance above the new level
    release_coef: T,
    peak_fall_db_per_s: T,
    floor_db: T,
    frames: u64,
}

impl<T: Float> SpectrumAnalyzer<T> {
    /// Hann window, a hop of a quarter frame, 300 ms release, peaks falling 20 dB/s, floor -140 dB.
    /// Panics unless `fft_len` >= 16 (any length; powers of two are fastest).
    pub fn new(fft_len: usize, sample_rate: T) -> Self {
        assert!(fft_len >= 16, "fft_len must be at least 16, got {fft_len}");
        let bins = fft_len / 2 + 1;
        let floor_db = T::_lit(-140.0);
        let mut a = Self {
            sample_rate,
            fft: RealFft::new(fft_len),
            window: Vec::new(),
            window_kind: Window::Hann,
            scale: T::_ONE,
            history: vec![T::_ZERO; fft_len],
            write: 0,
            hop: fft_len / 4,
            since_frame: 0,
            frame: vec![T::_ZERO; fft_len],
            spectrum: vec![Complex::zero(); bins],
            levels: vec![floor_db; bins],
            peaks: vec![floor_db; bins],
            release_seconds: T::_lit(0.3),
            release_coef: T::_ZERO,
            peak_fall_db_per_s: T::_lit(20.0),
            floor_db,
            frames: 0,
        };
        a.set_window(Window::Hann);
        a.set_release(a.release_seconds);
        a
    }
    /// Samples per frame (the FFT length).
    pub fn fft_len(&self) -> usize {
        self.frame.len()
    }
    /// Number of bins: `fft_len / 2 + 1` (0 Hz to Nyquist).
    pub fn bins(&self) -> usize {
        self.levels.len()
    }
    /// The sample rate in Hz.
    pub fn sample_rate(&self) -> T {
        self.sample_rate
    }
    /// Center frequency of `bin` in Hz.
    pub fn bin_frequency(&self, bin: usize) -> T {
        T::_lit(bin as f64) * self.sample_rate / T::_lit(self.fft_len() as f64)
    }
    /// Changes the window (the level scale follows its gain).
    pub fn set_window(&mut self, window: Window) {
        self.window_kind = window;
        self.window = window.coefficients(self.fft_len());
        let sum = self.window.iter().fold(T::_ZERO, |acc, &w| acc + w);
        self.scale = T::_lit(2.0) / sum;
    }
    /// The window.
    pub fn window(&self) -> Window {
        self.window_kind
    }
    /// Samples between frames, 1 ..= fft_len (smaller = smoother animation, more work).
    pub fn set_hop(&mut self, hop: usize) {
        self.hop = hop.clamp(1, self.fft_len());
        self.set_release(self.release_seconds);
    }
    /// Samples between frames.
    pub fn hop(&self) -> usize {
        self.hop
    }
    /// Time for a level to fall about 8.7 dB toward a lower reading (exponential, in dB); 0 = no smoothing.
    pub fn set_release(&mut self, seconds: T) {
        self.release_seconds = seconds._max(T::_ZERO);
        let frame_seconds = T::_lit(self.hop as f64) / self.sample_rate;
        self.release_coef = if self.release_seconds > T::_ZERO { (-frame_seconds / self.release_seconds)._exp() } else { T::_ZERO };
    }
    /// How fast the peak trace falls, in dB per second.
    pub fn set_peak_fall(&mut self, db_per_second: T) {
        self.peak_fall_db_per_s = db_per_second._max(T::_ZERO);
    }
    /// Lowest level reported, in dB (silence reads this).
    pub fn set_floor_db(&mut self, db: T) {
        self.floor_db = db;
        self.levels.iter_mut().chain(self.peaks.iter_mut()).for_each(|l| *l = l._max(db));
    }
    /// Frames computed so far: a GUI can redraw when this changes.
    pub fn frames(&self) -> u64 {
        self.frames
    }
    /// Smoothed level of every bin in dB.
    pub fn levels_db(&self) -> &[T] {
        &self.levels
    }
    /// Peak trace of every bin in dB.
    pub fn peaks_db(&self) -> &[T] {
        &self.peaks
    }
    /// Clears the history, levels and peaks.
    pub fn reset(&mut self) {
        self.history.iter_mut().for_each(|s| *s = T::_ZERO);
        self.write = 0;
        self.since_frame = 0;
        let floor = self.floor_db;
        self.levels.iter_mut().chain(self.peaks.iter_mut()).for_each(|l| *l = floor);
    }

    /// Feeds samples; returns how many new frames were computed.
    pub fn push(&mut self, block: &[T]) -> usize {
        let mut new_frames = 0;
        for &x in block {
            self.history[self.write] = x;
            self.write = (self.write + 1) % self.history.len();
            self.since_frame += 1;
            if self.since_frame >= self.hop {
                self.since_frame = 0;
                self.analyze();
                new_frames += 1;
            }
        }
        new_frames
    }

    fn analyze(&mut self) {
        let n = self.fft_len();
        // oldest sample first: the ring from the write position on
        for (i, (f, &w)) in self.frame.iter_mut().zip(&self.window).enumerate() {
            *f = self.history[(self.write + i) % n] * w;
        }
        self.fft.forward(&self.frame, &mut self.spectrum);
        let power_scale = self.scale * self.scale;
        let fall = self.peak_fall_db_per_s * T::_lit(self.hop as f64) / self.sample_rate;
        let (floor, coef) = (self.floor_db, self.release_coef);
        for ((z, level), peak) in self.spectrum.iter().zip(self.levels.iter_mut()).zip(self.peaks.iter_mut()) {
            let power = z.norm_sqr() * power_scale;
            let db = if power > T::_ZERO { (T::_lit(10.0) * power._log10())._max(floor) } else { floor };
            *level = if db >= *level { db } else { db + (*level - db) * coef };
            *peak = (*peak - fall)._max(db)._max(floor);
        }
        self.frames += 1;
    }

    /// Smoothed levels on `out.len()` log-spaced bands from `low_hz` to `high_hz`, for drawing.
    /// A band covering several bins shows the loudest; a band narrower than a bin (at the low end)
    /// interpolates between the two nearest bins, so no band is empty.
    pub fn bands_db(&self, low_hz: T, high_hz: T, out: &mut [T]) {
        band_levels(&self.levels, self.sample_rate, low_hz, high_hz, out);
    }
    /// The peak trace on the same bands as [`bands_db`](Self::bands_db).
    pub fn peak_bands_db(&self, low_hz: T, high_hz: T, out: &mut [T]) {
        band_levels(&self.peaks, self.sample_rate, low_hz, high_hz, out);
    }
}

fn band_levels<T: Float>(levels: &[T], sample_rate: T, low_hz: T, high_hz: T, out: &mut [T]) {
    let bins = levels.len();
    if out.is_empty() || bins < 2 {
        return;
    }
    let nyquist = sample_rate / T::_lit(2.0);
    let low = low_hz._max(T::_lit(1e-3))._min(nyquist);
    let high = high_hz._max(low)._min(nyquist);
    let hz_per_bin = nyquist / T::_lit((bins - 1) as f64);
    let ratio = (high / low)._ln() / T::_lit(out.len() as f64);
    let last = (bins - 1) as f64;
    for (b, o) in out.iter_mut().enumerate() {
        let edge = |i: usize| low * (ratio * T::_lit(i as f64))._exp() / hz_per_bin;
        let (start, end) = (edge(b), edge(b + 1));
        let (first, past) = (start._ceil().to_f64().unwrap_or(0.0), end._floor().to_f64().unwrap_or(0.0));
        *o = if past >= first {
            // whole bins inside the band: the loudest
            let (i0, i1) = (first.clamp(0.0, last) as usize, past.clamp(0.0, last) as usize);
            levels[i0..=i1].iter().fold(T::_NEG_INFINITY, |m, &l| m._max(l))
        } else {
            // narrower than a bin: interpolate at the band's (geometric) center
            let center = (start * end)._sqrt().to_f64().unwrap_or(0.0).clamp(0.0, last);
            let i = (center.floor() as usize).min(bins - 2);
            let frac = T::_lit(center - i as f64);
            levels[i] + (levels[i + 1] - levels[i]) * frac
        };
    }
}

impl<T: Float> Processor<T> for SpectrumAnalyzer<T> {
    /// Analyzes `block` and leaves it unchanged.
    fn process(&mut self, block: &mut [T]) {
        self.push(block);
    }
    fn reset(&mut self) {
        SpectrumAnalyzer::reset(self)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::osc::{Noise, Sine};

    const FS: f64 = 48_000.0;

    #[test]
    fn a_full_scale_sine_reads_zero_db_in_its_bin() {
        for window in Window::ALL {
            let mut a = SpectrumAnalyzer::new(1024, FS);
            a.set_window(window);
            a.set_release(0.0);
            let bin = 100;
            let tone: Vec<f64> = Sine::new(a.bin_frequency(bin), FS).take(4096).collect();
            assert_eq!(a.push(&tone), 16);
            let levels = a.levels_db();
            assert!(levels[bin].abs() < 0.01, "{window:?}: {} dB", levels[bin]);
            if window == Window::BlackmanHarris {
                // far from the tone only the side lobes remain
                let leak = levels.iter().enumerate().filter(|(k, _)| k.abs_diff(bin) > 4).map(|(_, &l)| l).fold(f64::MIN, f64::max);
                assert!(leak < -90.0, "side lobes {leak} dB");
            }
        }
    }

    #[test]
    fn ballistics_release_and_peaks() {
        let mut a = SpectrumAnalyzer::new(1024, FS);
        a.set_release(0.1);
        a.set_peak_fall(20.0);
        let bin = 64;
        let tone: Vec<f64> = Sine::new(a.bin_frequency(bin), FS).take(48_000).collect();
        a.push(&tone);
        assert!(a.levels_db()[bin].abs() < 0.01 && a.peaks_db()[bin].abs() < 0.01);
        // silence: after 0.5 s (well past the window) the peak has fallen 10 dB, the level far more
        a.push(&vec![0.0; 24_000]);
        let peak = a.peaks_db()[bin];
        assert!((peak + 10.0).abs() < 0.4, "peak {peak} dB"); // it starts falling once the tone leaves the window
        let level = a.levels_db()[bin];
        assert!(level < -100.0, "level {level} dB");
        a.reset();
        assert!(a.levels_db().iter().all(|&l| l == -140.0));
    }

    #[test]
    fn bands_cover_the_spectrum_without_gaps() {
        let mut a = SpectrumAnalyzer::new(2048, FS);
        let noise: Vec<f64> = Noise::new(3).take(48_000).collect();
        a.push(&noise);
        let mut bands = [0.0; 96];
        a.bands_db(20.0, 20_000.0, &mut bands);
        // white noise: every band, even the sub-bin ones at the bottom, has a level
        assert!(bands.iter().all(|&b| b > -60.0 && b < 0.0), "{bands:?}");
        // a tone lights its band
        let mut a = SpectrumAnalyzer::new(2048, FS);
        a.push(&Sine::new(1_000.0, FS).take(8192).collect::<Vec<f64>>());
        a.bands_db(20.0, 20_000.0, &mut bands);
        let loudest = bands.iter().enumerate().fold((0, f64::MIN), |m, (i, &b)| if b > m.1 { (i, b) } else { m }).0;
        let center = 20.0 * 1000f64.powf((loudest as f64 + 0.5) / 96.0);
        assert!((center / 1_000.0).log2().abs() < 0.1, "loudest band at {center} Hz");
    }
}
