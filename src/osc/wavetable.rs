//! Wavetable synthesis: single-cycle frames you morph between, band-limited for every pitch.

use std::sync::Arc;

use super::band_limited_increment;
use crate::signal::Source;
use crate::spectral::RealFft;
use crate::units::*;

/// The most harmonics a table keeps (its full-bandwidth level, used for the lowest notes).
pub const MAX_HARMONICS: usize = 1023;
/// Band limits of successive levels are half an octave apart.
const LEVEL_RATIO: f64 = std::f64::consts::SQRT_2;

/// One band-limited version of every frame.
#[derive(Debug, Clone)]
struct Level<T> {
    /// the highest harmonic kept
    harmonics: usize,
    /// samples per frame: a power of two, 64 per period of the highest harmonic (up to 2048), so the
    /// images linear interpolation creates stay below about -90 dB
    len: usize,
    /// frames x (len + 1): each frame ends with a copy of its first sample for interpolation
    data: Vec<T>,
}

/// A set of single-cycle waveforms ("frames") an oscillator can morph between, stored
/// band-limited: every frame is kept at a series of levels whose band limits fall half an octave
/// apart (all harmonics, then 1/sqrt(2) as many, and so on down to a sine), built from its spectrum.
/// [`WavetableOsc`] plays the levels whose harmonics all stay below Nyquist at the current pitch,
/// crossfading between two, so even high notes don't alias and sweeps stay smooth.
///
/// Frames are normalized to a peak of 1 and their DC is removed. Building a table runs FFTs and
/// allocates; share one between oscillators and voices through an `Arc`.
#[derive(Debug, Clone)]
pub struct Wavetable<T: Float> {
    frames: usize,
    levels: Vec<Level<T>>,
}

impl<T: Float> Wavetable<T> {
    /// From single-cycle waveforms of any length (at least 2 samples each): one frame each.
    /// Panics if there are no frames or a frame is too short.
    pub fn from_frames<F: AsRef<[T]>>(frames: &[F]) -> Self {
        let spectra: Vec<Vec<Complex<f64>>> = frames.iter().map(|f| harmonics_of(f.as_ref())).collect();
        Self::from_spectra(spectra)
    }

    /// From harmonic amplitudes: `frames[f][h]` is the amplitude of harmonic `h + 1` (in sine
    /// phase) in frame `f`. Panics if there are no frames.
    pub fn from_harmonics<F: AsRef<[T]>>(frames: &[F]) -> Self {
        let spectra = frames
            .iter()
            .map(|f| {
                let amplitudes = f.as_ref();
                let mut c = vec![Complex::zero(); MAX_HARMONICS + 1];
                for (h, &a) in amplitudes.iter().take(MAX_HARMONICS).enumerate() {
                    // a sin(x) = a cos(x - pi/2): coefficient -i a
                    c[h + 1] = Complex::new(0.0, -a.to_f64().unwrap_or(0.0));
                }
                c
            })
            .collect();
        Self::from_spectra(spectra)
    }

    /// The classic morph: sine, triangle, saw, square (4 frames, in that order).
    pub fn classic() -> Self {
        let harmonics = |amplitude: &dyn Fn(usize) -> f64| -> Vec<T> { (1..=MAX_HARMONICS).map(|h| T::_lit(amplitude(h))).collect() };
        let sine = harmonics(&|h| if h == 1 { 1.0 } else { 0.0 });
        let triangle = harmonics(&|h| if h % 2 == 1 { (if (h / 2) % 2 == 0 { 1.0 } else { -1.0 }) / (h * h) as f64 } else { 0.0 });
        let saw = harmonics(&|h| 1.0 / h as f64);
        let square = harmonics(&|h| if h % 2 == 1 { 1.0 / h as f64 } else { 0.0 });
        Self::from_harmonics(&[sine, triangle, saw, square])
    }

    /// [`classic`](Self::classic), built once per sample type and shared: every call after the first
    /// just clones an `Arc` (cheap enough to give every voice its own handle).
    pub fn shared_classic() -> Arc<Self> {
        use std::any::{Any, TypeId};
        use std::sync::OnceLock;
        static F32: OnceLock<Arc<Wavetable<f32>>> = OnceLock::new();
        static F64: OnceLock<Arc<Wavetable<f64>>> = OnceLock::new();
        let shared: Option<&dyn Any> = if TypeId::of::<T>() == TypeId::of::<f32>() {
            Some(F32.get_or_init(|| Arc::new(Wavetable::classic())))
        } else if TypeId::of::<T>() == TypeId::of::<f64>() {
            Some(F64.get_or_init(|| Arc::new(Wavetable::classic())))
        } else {
            None
        };
        shared.and_then(|s| s.downcast_ref::<Arc<Self>>()).cloned().unwrap_or_else(|| Arc::new(Self::classic()))
    }
    /// Frames from per-frame harmonic coefficients `c[k]` (k = 1..), where the waveform is
    /// `sum |c_k| cos(2 pi k t + arg c_k)`.
    fn from_spectra(mut spectra: Vec<Vec<Complex<f64>>>) -> Self {
        assert!(!spectra.is_empty(), "a wavetable needs at least one frame");
        // the band limits, half an octave apart, down to a single harmonic
        let mut limits = Vec::new();
        let mut h = MAX_HARMONICS as f64;
        loop {
            let k = (h.floor() as usize).max(1);
            if limits.last() != Some(&k) {
                limits.push(k);
            }
            if k == 1 {
                break;
            }
            h /= LEVEL_RATIO;
        }
        let len_for = |k: usize| (64 * k).next_power_of_two().clamp(64, 2048);
        // normalize every frame to a peak of 1 at full bandwidth
        let mut ffts: Vec<(usize, RealFft<f64>)> = Vec::new();
        let mut render = |c: &[Complex<f64>], keep: usize, len: usize| -> Vec<f64> {
            let fft = match ffts.iter_mut().position(|(l, _)| *l == len) {
                Some(i) => &mut ffts[i].1,
                None => {
                    ffts.push((len, RealFft::new(len)));
                    &mut ffts.last_mut().expect("just pushed").1
                }
            };
            let mut spectrum = vec![Complex::zero(); len / 2 + 1];
            let scale = len as f64 / 2.0;
            for k in 1..=keep.min(len / 2 - 1).min(c.len().saturating_sub(1)) {
                spectrum[k] = c[k] * scale;
            }
            let mut out = vec![0.0; len];
            fft.inverse(&spectrum, &mut out);
            out
        };
        for c in &mut spectra {
            let full = render(c, limits[0], len_for(limits[0]));
            let peak = full.iter().fold(0.0f64, |m, &x| m.max(x.abs()));
            if peak > 0.0 {
                c.iter_mut().for_each(|z| *z *= 1.0 / peak);
            }
        }
        let levels = limits
            .iter()
            .map(|&harmonics| {
                let len = len_for(harmonics);
                let mut data = Vec::with_capacity(spectra.len() * (len + 1));
                for c in &spectra {
                    let wave = render(c, harmonics, len);
                    data.extend(wave.iter().map(|&x| T::_lit(x)));
                    data.push(T::_lit(wave[0]));
                }
                Level { harmonics, len, data }
            })
            .collect();
        Self { frames: spectra.len(), levels }
    }

    pub fn frames(&self) -> usize {
        self.frames
    }

    /// The level to play at `increment` cycles per sample, and the weight of that level against the
    /// next darker one (1 = only this level). Both levels keep every harmonic below Nyquist.
    fn select(&self, increment: T) -> (usize, T) {
        let last = self.levels.len() - 1;
        if increment <= T::_ZERO {
            return (0, T::_ONE);
        }
        // harmonics that fit below Nyquist at this pitch
        let fit = T::_lit(0.5) / increment;
        let a = self.levels.iter().position(|l| T::_lit(l.harmonics as f64) <= fit).unwrap_or(last);
        if a == last {
            return (last, T::_ONE);
        }
        // fade from the darker neighbor (a + 1) to `a` as the pitch falls, reaching `a` alone where
        // the brighter level (a - 1; one ratio above the top for level 0) becomes safe: continuous,
        // and never aliasing
        let this = self.levels[a].harmonics as f64;
        let brighter = if a == 0 { this * LEVEL_RATIO } else { self.levels[a - 1].harmonics as f64 };
        (a, ((fit - T::_lit(this)) / T::_lit(brighter - this))._clamp(T::_ZERO, T::_ONE))
    }

    /// One level's value at `phase` (cycles, [0, 1)) and `position` (frames, 0..frames-1).
    #[inline]
    fn read(&self, level: usize, position: T, phase: T) -> T {
        let lv = &self.levels[level];
        let stride = lv.len + 1;
        let f0 = position._floor().to_usize().unwrap_or(0).min(self.frames - 1);
        let f1 = (f0 + 1).min(self.frames - 1);
        let ff = position - T::_lit(f0 as f64);
        let x = phase * T::_lit(lv.len as f64);
        let i = x._floor().to_usize().unwrap_or(0).min(lv.len - 1);
        let frac = x - T::_lit(i as f64);
        let at = |frame: usize| {
            let row = &lv.data[frame * stride + i..frame * stride + i + 2];
            row[0] + (row[1] - row[0]) * frac
        };
        let a = at(f0);
        if f1 == f0 { a } else { a + (at(f1) - a) * ff }
    }
}

/// An oscillator playing a [`Wavetable`]: morph position 0..1 across its frames, band-limited at
/// every pitch. Frequencies are limited to 0 ..= Nyquist.
#[derive(Debug, Clone)]
pub struct WavetableOsc<T: Float> {
    table: Arc<Wavetable<T>>,
    phase: T,
    increment: T,
    /// morph position scaled to frames (0..frames-1)
    position: T,
    level: usize,
    weight: T,
}

impl<T: Float> WavetableOsc<T> {
    pub fn new(table: Arc<Wavetable<T>>, frequency: T, sample_rate: T) -> Self {
        let mut osc = Self { table, phase: T::_ZERO, increment: T::_ZERO, position: T::_ZERO, level: 0, weight: T::_ONE };
        osc.set_frequency(frequency, sample_rate);
        osc
    }
    /// Starting phase in cycles, [0, 1).
    pub fn with_phase(mut self, cycles: T) -> Self {
        self.phase = cycles - cycles._floor();
        self
    }
    /// Changes frequency without resetting phase (no click).
    pub fn set_frequency(&mut self, frequency: T, sample_rate: T) {
        self.increment = band_limited_increment(frequency, sample_rate);
        (self.level, self.weight) = self.table.select(self.increment);
    }
    /// Morph position, 0 (first frame) .. 1 (last frame), blending linearly between frames.
    pub fn set_position(&mut self, position: T) {
        let span = T::_lit((self.table.frames() - 1) as f64);
        self.position = position._clamp(T::_ZERO, T::_ONE) * span;
    }
    pub fn position(&self) -> T {
        let span = (self.table.frames() - 1).max(1) as f64;
        self.position / T::_lit(span)
    }
    /// Swaps the table (keeps phase, frequency and position).
    pub fn set_table(&mut self, table: Arc<Wavetable<T>>) {
        let position = self.position();
        self.table = table;
        self.set_position(position);
        (self.level, self.weight) = self.table.select(self.increment);
    }
    pub fn table(&self) -> &Arc<Wavetable<T>> {
        &self.table
    }
    /// Moves to cycles (in [0, 1)) without changing anything else.
    pub fn set_phase(&mut self, cycles: T) {
        self.phase = cycles - cycles._floor();
    }
    pub fn reset(&mut self) {
        self.phase = T::_ZERO;
    }
    #[inline]
    pub fn next_sample(&mut self) -> T {
        let table = &self.table;
        let mut value = table.read(self.level, self.position, self.phase);
        if self.weight < T::_ONE {
            let darker = table.read(self.level + 1, self.position, self.phase);
            value = darker + (value - darker) * self.weight;
        }
        self.phase = self.phase + self.increment;
        if self.phase >= T::_ONE {
            self.phase = self.phase - T::_ONE;
        }
        value
    }
    /// Overwrites `out` with the next `out.len()` samples.
    pub fn fill(&mut self, out: &mut [T]) {
        for s in out {
            *s = self.next_sample();
        }
    }
}

impl<T: Float> Source for WavetableOsc<T> {
    type Sample = T;
    fn next_sample(&mut self) -> T {
        WavetableOsc::next_sample(self)
    }
    fn fill(&mut self, out: &mut [T]) {
        WavetableOsc::fill(self, out)
    }
}

/// Harmonic coefficients `c[k]` (k = 1.. up to the frame's resolution and [`MAX_HARMONICS`]) of one
/// single-cycle frame: `x(t) = sum |c_k| cos(2 pi k t + arg c_k)`.
fn harmonics_of<T: Float>(frame: &[T]) -> Vec<Complex<f64>> {
    let n = frame.len();
    assert!(n >= 2, "a frame needs at least 2 samples");
    let x: Vec<f64> = frame.iter().map(|s| s.to_f64().unwrap_or(0.0)).collect();
    // harmonics strictly below the frame's own Nyquist
    let keep = ((n - 1) / 2).min(MAX_HARMONICS);
    let mut c = vec![Complex::zero(); keep + 1];
    if n.is_power_of_two() {
        let mut fft = RealFft::new(n);
        let mut spectrum = vec![Complex::zero(); n / 2 + 1];
        fft.forward(&x, &mut spectrum);
        for k in 1..=keep {
            c[k] = spectrum[k] * (2.0 / n as f64);
        }
    } else {
        for (k, ck) in c.iter_mut().enumerate().skip(1) {
            let mut sum = Complex::zero();
            for (i, &v) in x.iter().enumerate() {
                sum += Complex::cis(-std::f64::consts::TAU * (k * i) as f64 / n as f64) * v;
            }
            *ck = sum * (2.0 / n as f64);
        }
    }
    c
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::signal::Signal;
    use crate::spectral::Fft;

    const FS: f64 = 48_000.0;

    /// Magnitudes (bins 0..n/2) of `x` through a 4-term Blackman-Harris window (sidelobes < -92 dB).
    fn windowed_spectrum(x: &[f64]) -> Vec<f64> {
        let n = x.len();
        let w = |i: usize| {
            let t = std::f64::consts::TAU * i as f64 / n as f64;
            0.35875 - 0.48829 * t.cos() + 0.14128 * (2.0 * t).cos() - 0.01168 * (3.0 * t).cos()
        };
        let input: Vec<f64> = x.iter().enumerate().map(|(i, &v)| v * w(i)).collect();
        let mut z = vec![Complex::zero(); n];
        Fft::new(n).forward_real(&input, &mut z);
        z[..n / 2].iter().map(|c| c.norm_sqr().sqrt()).collect()
    }

    fn play(table: Wavetable<f64>, freq: f64, position: f64, n: usize) -> Vec<f64> {
        let mut osc = WavetableOsc::new(Arc::new(table), freq, FS);
        osc.set_position(position);
        (0..n).map(|_| osc.next_sample()).collect()
    }

    /// Level of the harmonic at `freq` relative to the fundamental at `f0`, in dB.
    fn harmonic_db(mags: &[f64], f0: f64, freq: f64) -> f64 {
        let bin = |f: f64| (f * mags.len() as f64 * 2.0 / FS).round() as usize;
        let peak = |f: f64| (bin(f) - 3..=bin(f) + 3).map(|i| mags[i]).fold(0.0, f64::max);
        20.0 * (peak(freq) / peak(f0)).log10()
    }

    #[test]
    fn a_sine_frame_of_any_length_plays_a_pure_sine() {
        // 600 samples: not a power of two, so the frame's spectrum comes from a direct DFT
        let frame: Vec<f64> = (0..600).map(|i| 0.5 * (std::f64::consts::TAU * i as f64 / 600.0).sin()).collect();
        let x = play(Wavetable::from_frames(&[frame]), 440.0, 0.0, 48_000);
        assert!((x.rms().unwrap() - std::f64::consts::FRAC_1_SQRT_2).abs() < 1e-3, "normalized to a peak of 1");
        let mags = windowed_spectrum(&x[..32_768]);
        assert!(harmonic_db(&mags, 440.0, 880.0) < -90.0, "no harmonics");
    }

    #[test]
    fn high_notes_do_not_alias() {
        // a saw at 3.1 kHz: harmonics up to 7 fit below Nyquist; everything else must be (window-)silence
        let f0 = 3_100.0;
        let x = play(Wavetable::classic(), f0, 2.0 / 3.0, 65_536);
        let mags = windowed_spectrum(&x[..65_536]);
        let top = mags.iter().cloned().fold(0.0, f64::max);
        let bin_hz = FS / 65_536.0;
        let near_harmonic = |i: usize| {
            let f = i as f64 * bin_hz;
            let h = (f / f0).round();
            h >= 1.0 && (f - h * f0).abs() < 8.0 * bin_hz
        };
        let worst = mags.iter().enumerate().filter(|&(i, _)| i > 8 && !near_harmonic(i)).map(|(_, &m)| m).fold(0.0, f64::max);
        assert!(20.0 * (worst / top).log10() < -80.0, "aliasing at {:.1} dB", 20.0 * (worst / top).log10());
        // the test can see aliasing: a naive saw at the same pitch fails it badly
        let naive: Vec<f64> = (0..65_536).map(|i| 2.0 * ((i as f64 * f0 / FS).fract()) - 1.0).collect();
        let mags = windowed_spectrum(&naive);
        let worst = mags.iter().enumerate().filter(|&(i, _)| i > 8 && !near_harmonic(i)).map(|(_, &m)| m).fold(0.0, f64::max);
        assert!(20.0 * (worst / top).log10() > -40.0);
    }

    #[test]
    fn the_classic_table_morphs_sine_triangle_saw_square() {
        // a pitch exactly on an FFT bin, so every harmonic is measured at its true level
        let f0 = 137.0 * FS / 32_768.0;
        let at = |position: f64| windowed_spectrum(&play(Wavetable::classic(), f0, position, 32_768));
        let db = |mags: &[f64], h: f64| harmonic_db(mags, f0, h * f0);
        let expect = |measured: f64, ratio: f64, what: &str| assert!((measured - 20.0 * ratio.log10()).abs() < 0.05, "{what}: {measured:.3} dB");
        let (sine, triangle, saw, square) = (at(0.0), at(1.0 / 3.0), at(2.0 / 3.0), at(1.0));
        assert!(db(&sine, 2.0) < -90.0 && db(&sine, 3.0) < -90.0, "frame 0: a sine");
        expect(db(&triangle, 3.0), 1.0 / 9.0, "frame 1: a triangle (odd harmonics at 1/h^2)");
        expect(db(&saw, 2.0), 0.5, "frame 2: a saw (harmonics at 1/h)");
        expect(db(&saw, 3.0), 1.0 / 3.0, "frame 2: a saw");
        assert!(db(&square, 2.0) < -90.0, "frame 3: a square (no even harmonics)");
        expect(db(&square, 3.0), 1.0 / 3.0, "frame 3: a square (odd harmonics at 1/h)");
        // halfway between the saw and the square (no even harmonics), the 2nd harmonic is halved
        let level = |mags: &[f64], h: f64| {
            let bin = (h * f0 * mags.len() as f64 * 2.0 / FS).round() as usize;
            mags[bin]
        };
        let between = at(5.0 / 6.0);
        assert!((level(&between, 2.0) / level(&saw, 2.0) - 0.5).abs() < 1e-3, "the morph blends linearly");
    }
    #[test]
    fn level_selection_never_aliases_and_moves_continuously() {
        let table = Wavetable::<f64>::classic();
        let k = |level: usize| table.levels[level].harmonics as f64;
        // the effective band limit (harmonics weighted by the crossfade) for a pitch
        let effective = |increment: f64| {
            let (a, w) = table.select(increment);
            let fit = 0.5 / increment;
            assert!(k(a) <= fit, "level {a} aliases at increment {increment}");
            if w < 1.0 {
                assert!(k(a + 1) <= fit, "level {} aliases at increment {increment}", a + 1);
                (1.0 - w) * k(a + 1) + w * k(a)
            } else {
                k(a)
            }
        };
        let steps = 200_000;
        let mut last = effective(1e-4);
        for i in 1..=steps {
            let increment = 1e-4 * (0.5f64 / 1e-4).powf(i as f64 / steps as f64); // ~1.6 Hz .. Nyquist at 48 kHz
            let now = effective(increment);
            assert!((now - last).abs() <= 1e-3 * last + 1e-9, "a jump from {last} to {now} at increment {increment}");
            last = now;
        }
    }
}