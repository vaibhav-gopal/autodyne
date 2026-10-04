//! A parametric equalizer with any number of bands (up to [`MAX_EQ_BANDS`]).

use super::{Biquad, BiquadCoeffs, BUTTERWORTH_Q};
use crate::units::*;

/// Most bands a [`ParametricEq`] has.
pub const MAX_EQ_BANDS: usize = 8;
/// Parameters per band: on, type, frequency, gain, Q, slope.
pub const EQ_BAND_PARAMS: usize = 6;
/// Biquad sections per band (cuts up to 48 dB/octave).
const MAX_STAGES: usize = 4;

pub(crate) const EQ_PARAM_IDS: [[&str; EQ_BAND_PARAMS]; MAX_EQ_BANDS] = [
    ["band1_on", "band1_type", "band1_freq_hz", "band1_gain_db", "band1_q", "band1_slope"],
    ["band2_on", "band2_type", "band2_freq_hz", "band2_gain_db", "band2_q", "band2_slope"],
    ["band3_on", "band3_type", "band3_freq_hz", "band3_gain_db", "band3_q", "band3_slope"],
    ["band4_on", "band4_type", "band4_freq_hz", "band4_gain_db", "band4_q", "band4_slope"],
    ["band5_on", "band5_type", "band5_freq_hz", "band5_gain_db", "band5_q", "band5_slope"],
    ["band6_on", "band6_type", "band6_freq_hz", "band6_gain_db", "band6_q", "band6_slope"],
    ["band7_on", "band7_type", "band7_freq_hz", "band7_gain_db", "band7_q", "band7_slope"],
    ["band8_on", "band8_type", "band8_freq_hz", "band8_gain_db", "band8_q", "band8_slope"],
];
pub(crate) const EQ_PARAM_NAMES: [[&str; EQ_BAND_PARAMS]; MAX_EQ_BANDS] = [
    ["Band 1 on", "Band 1 type", "Band 1 frequency", "Band 1 gain", "Band 1 Q", "Band 1 slope"],
    ["Band 2 on", "Band 2 type", "Band 2 frequency", "Band 2 gain", "Band 2 Q", "Band 2 slope"],
    ["Band 3 on", "Band 3 type", "Band 3 frequency", "Band 3 gain", "Band 3 Q", "Band 3 slope"],
    ["Band 4 on", "Band 4 type", "Band 4 frequency", "Band 4 gain", "Band 4 Q", "Band 4 slope"],
    ["Band 5 on", "Band 5 type", "Band 5 frequency", "Band 5 gain", "Band 5 Q", "Band 5 slope"],
    ["Band 6 on", "Band 6 type", "Band 6 frequency", "Band 6 gain", "Band 6 Q", "Band 6 slope"],
    ["Band 7 on", "Band 7 type", "Band 7 frequency", "Band 7 gain", "Band 7 Q", "Band 7 slope"],
    ["Band 8 on", "Band 8 type", "Band 8 frequency", "Band 8 gain", "Band 8 Q", "Band 8 slope"],
];

/// What one band of a [`ParametricEq`] does.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum EqBandKind {
    /// a bell: boost or cut around the frequency, width set by Q
    Peak,
    /// boost or cut below the frequency
    LowShelf,
    /// boost or cut above the frequency
    HighShelf,
    /// high-pass (removes lows); Butterworth at slopes above 12 dB/octave
    LowCut,
    /// low-pass (removes highs)
    HighCut,
    /// removes a narrow band around the frequency
    Notch,
    /// passes a band around the frequency
    BandPass,
}

impl EqBandKind {
    /// Every kind, in parameter order.
    pub const ALL: [EqBandKind; 7] = [EqBandKind::Peak, EqBandKind::LowShelf, EqBandKind::HighShelf, EqBandKind::LowCut, EqBandKind::HighCut, EqBandKind::Notch, EqBandKind::BandPass];
    /// Display names, in the order of [`ALL`](Self::ALL).
    pub const NAMES: [&'static str; 7] = ["Peak", "Low shelf", "High shelf", "Low cut", "High cut", "Notch", "Band pass"];
    /// Whether the gain setting applies.
    pub fn uses_gain(self) -> bool {
        matches!(self, EqBandKind::Peak | EqBandKind::LowShelf | EqBandKind::HighShelf)
    }
    /// Whether the slope setting applies.
    pub fn uses_slope(self) -> bool {
        matches!(self, EqBandKind::LowCut | EqBandKind::HighCut)
    }
}

/// Slopes a cut band offers.
pub const EQ_SLOPE_NAMES: [&str; MAX_STAGES] = ["12 dB/oct", "24 dB/oct", "36 dB/oct", "48 dB/oct"];

/// One band's settings.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct EqBand {
    /// Whether the band is applied.
    pub enabled: bool,
    /// What the band does.
    pub kind: EqBandKind,
    /// Center, corner or cutoff in Hz.
    pub frequency: f64,
    /// Boost or cut in dB (peak and shelves).
    pub gain_db: f64,
    /// Width (peak, notch, band pass), shelf slope, or the resonance of a 12 dB/octave cut.
    pub q: f64,
    /// cut bands: 1..=4 sections, 12..48 dB/octave
    pub slope: usize,
}

impl EqBand {
    /// A band of `kind` at `frequency`, off, at 0 dB and Q 0.707, 12 dB/octave.
    pub fn new(kind: EqBandKind, frequency: f64) -> Self {
        Self { enabled: false, kind, frequency, gain_db: 0.0, q: BUTTERWORTH_Q, slope: 1 }
    }
}

/// Band `i` of a new EQ with `bands` bands (see [`ParametricEq::new`]).
pub(crate) fn default_band(bands: usize, i: usize) -> EqBand {
    if bands == 6 {
        const SIX: [(EqBandKind, f64); 6] = [
            (EqBandKind::LowCut, 30.0),
            (EqBandKind::LowShelf, 100.0),
            (EqBandKind::Peak, 300.0),
            (EqBandKind::Peak, 1_000.0),
            (EqBandKind::Peak, 4_000.0),
            (EqBandKind::HighShelf, 10_000.0),
        ];
        EqBand::new(SIX[i].0, SIX[i].1)
    } else {
        let position = if bands == 1 { 0.5 } else { i as f64 / (bands - 1) as f64 };
        EqBand::new(EqBandKind::Peak, 100.0 * 100f64.powf(position))
    }
}

/// A parametric EQ: a fixed number of bands, each a bell, shelf, cut (12-48 dB/octave), notch or
/// band-pass at any frequency, gain and Q, switched on or off individually. Only enabled bands
/// cost anything. [`magnitude_db_at`](Self::magnitude_db_at) gives the combined curve for drawing.
///
/// Changing a band redesigns its sections and keeps their state; wrap the EQ in
/// [`Smoothed`](crate::params::Smoothed) for click-free automation. For stereo, use a linked
/// `PerChannel` of EQs.
#[derive(Debug, Clone)]
pub struct ParametricEq<T: Float> {
    sample_rate: T,
    bands: Vec<EqBand>,
    stages: Vec<[Biquad<T>; MAX_STAGES]>,
    /// sections in use per band (0 when disabled)
    active: Vec<usize>,
}

impl<T: Float> ParametricEq<T> {
    /// `bands` bands (1 to MAX_EQ_BANDS), all off: with 6, a low cut at 30 Hz, a low shelf at
    /// 100 Hz, bells at 300 Hz, 1 kHz and 4 kHz, and a high shelf at 10 kHz; otherwise bells
    /// spread from 100 Hz to 10 kHz. Panics on a band count out of range.
    pub fn new(bands: usize, sample_rate: T) -> Self {
        assert!((1..=MAX_EQ_BANDS).contains(&bands), "an EQ has 1 to {MAX_EQ_BANDS} bands");
        let layout: Vec<EqBand> = (0..bands).map(|i| default_band(bands, i)).collect();
        let identity = Biquad::new(BiquadCoeffs { b0: T::_ONE, b1: T::_ZERO, b2: T::_ZERO, a1: T::_ZERO, a2: T::_ZERO });
        let mut eq = Self { sample_rate, bands: layout, stages: vec![[identity; MAX_STAGES]; bands], active: vec![0; bands] };
        for i in 0..bands {
            eq.redesign(i);
        }
        eq
    }
    /// The sample rate in Hz.
    pub fn sample_rate(&self) -> T {
        self.sample_rate
    }
    /// The number of bands.
    pub fn band_count(&self) -> usize {
        self.bands.len()
    }
    /// Band `i`'s settings.
    pub fn band(&self, i: usize) -> &EqBand {
        &self.bands[i]
    }
    /// Replaces band `i` (frequency kept within 10 Hz .. 0.45 x the sample rate, Q 0.05..40,
    /// slope 1..=4).
    pub fn set_band(&mut self, i: usize, band: EqBand) {
        self.bands[i] = band;
        self.redesign(i);
    }
    /// Switches band `i` on or off.
    pub fn set_enabled(&mut self, i: usize, on: bool) {
        self.bands[i].enabled = on;
        self.redesign(i);
    }
    /// Changes band `i`'s kind.
    pub fn set_kind(&mut self, i: usize, kind: EqBandKind) {
        self.bands[i].kind = kind;
        self.redesign(i);
    }
    /// Moves band `i` (kept within 10 Hz .. 0.45 x the sample rate).
    pub fn set_frequency(&mut self, i: usize, hz: f64) {
        self.bands[i].frequency = hz;
        self.redesign(i);
    }
    /// Band `i`'s boost or cut in dB.
    pub fn set_gain_db(&mut self, i: usize, db: f64) {
        self.bands[i].gain_db = db;
        self.redesign(i);
    }
    /// Band `i`'s Q (kept within 0.05..40).
    pub fn set_q(&mut self, i: usize, q: f64) {
        self.bands[i].q = q;
        self.redesign(i);
    }
    /// Band `i`'s slope as a number of 12 dB/octave sections (1..=4).
    pub fn set_slope(&mut self, i: usize, sections: usize) {
        self.bands[i].slope = sections;
        self.redesign(i);
    }

    fn redesign(&mut self, i: usize) {
        let fs = self.sample_rate.to_f64().unwrap_or(48_000.0);
        let band = &mut self.bands[i];
        band.frequency = band.frequency.clamp(10.0, 0.45 * fs);
        band.q = band.q.clamp(0.05, 40.0);
        band.slope = band.slope.clamp(1, MAX_STAGES);
        let b = *band;
        if !b.enabled {
            self.active[i] = 0;
            return;
        }
        let (f, q, gain, sr) = (T::_lit(b.frequency), T::_lit(b.q), T::_lit(b.gain_db), self.sample_rate);
        let butterworth = |stage: usize, sections: usize| {
            // Q of section `stage` of a Butterworth filter of order 2 x sections
            let angle = (2 * stage + 1) as f64 * std::f64::consts::PI / (4 * sections) as f64;
            T::_lit(1.0 / (2.0 * angle.cos()))
        };
        let designs: [Option<BiquadCoeffs<T>>; MAX_STAGES] = std::array::from_fn(|s| match b.kind {
            EqBandKind::Peak if s == 0 => Some(BiquadCoeffs::peaking(f, q, gain, sr)),
            EqBandKind::LowShelf if s == 0 => Some(BiquadCoeffs::low_shelf(f, q, gain, sr)),
            EqBandKind::HighShelf if s == 0 => Some(BiquadCoeffs::high_shelf(f, q, gain, sr)),
            EqBandKind::Notch if s == 0 => Some(BiquadCoeffs::notch(f, q, sr)),
            EqBandKind::BandPass if s == 0 => Some(BiquadCoeffs::bandpass(f, q, sr)),
            // one section: the band's own Q (resonance); more: a Butterworth cascade
            EqBandKind::LowCut if s < b.slope => Some(BiquadCoeffs::highpass(f, if b.slope == 1 { q } else { butterworth(s, b.slope) }, sr)),
            EqBandKind::HighCut if s < b.slope => Some(BiquadCoeffs::lowpass(f, if b.slope == 1 { q } else { butterworth(s, b.slope) }, sr)),
            _ => None,
        });
        let mut used = 0;
        for (stage, design) in self.stages[i].iter_mut().zip(designs) {
            if let Some(coeffs) = design {
                // a section coming into use starts clean; one already running keeps its state
                if used >= self.active[i] {
                    stage.reset();
                }
                stage.set_coeffs(coeffs);
                used += 1;
            }
        }
        self.active[i] = used;
    }

    /// Clears every section's state.
    pub fn reset(&mut self) {
        self.stages.iter_mut().flatten().for_each(Biquad::reset);
    }
    /// Filters one sample through every enabled band.
    #[inline]
    pub fn process_sample(&mut self, x: T) -> T {
        let mut y = x;
        for (stages, &n) in self.stages.iter_mut().zip(&self.active) {
            for s in &mut stages[..n] {
                y = s.process_sample(y);
            }
        }
        y
    }
    /// Equalizes `block` in place.
    pub fn process(&mut self, block: &mut [T]) {
        for s in block.iter_mut() {
            *s = self.process_sample(*s);
        }
        self.stages.iter_mut().flatten().for_each(Biquad::flush_denormals);
    }
    /// The whole EQ's response at `frequency` in dB (enabled bands only).
    pub fn magnitude_db_at(&self, frequency: T) -> T {
        let mut db = T::_ZERO;
        for (stages, &n) in self.stages.iter().zip(&self.active) {
            for s in &stages[..n] {
                db = db + T::_lit(20.0) * s.magnitude_at(frequency, self.sample_rate)._max(T::_lit(1e-12))._log10();
            }
        }
        db
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    const FS: f64 = 48_000.0;

    fn measured_db(eq: &mut ParametricEq<f64>, freq: f64) -> f64 {
        eq.reset();
        let x: Vec<f64> = (0..48_000).map(|i| (std::f64::consts::TAU * freq * i as f64 / FS).sin()).collect();
        let mut y = x.clone();
        eq.process(&mut y);
        let rms = |v: &[f64]| (v.iter().map(|s| s * s).sum::<f64>() / v.len() as f64).sqrt();
        20.0 * (rms(&y[24_000..]) / rms(&x[24_000..])).log10()
    }

    #[test]
    fn disabled_bands_are_transparent() {
        let mut eq = ParametricEq::new(6, FS);
        let x: Vec<f64> = (0..1_000).map(|i| (i as f64 * 0.37).sin()).collect();
        let mut y = x.clone();
        eq.process(&mut y);
        assert_eq!(x, y);
        assert_eq!(eq.magnitude_db_at(1_000.0), 0.0);
    }

    #[test]
    fn bands_shape_the_response_as_drawn() {
        let mut eq = ParametricEq::new(6, FS);
        eq.set_band(0, EqBand { enabled: true, kind: EqBandKind::LowCut, frequency: 80.0, gain_db: 0.0, q: 0.7, slope: 4 });
        eq.set_band(2, EqBand { enabled: true, kind: EqBandKind::Peak, frequency: 500.0, gain_db: 6.0, q: 1.4, slope: 1 });
        eq.set_band(5, EqBand { enabled: true, kind: EqBandKind::HighShelf, frequency: 8_000.0, gain_db: -4.0, q: 0.7, slope: 1 });
        for freq in [20.0, 60.0, 80.0, 500.0, 2_000.0, 12_000.0] {
            let (measured, drawn) = (measured_db(&mut eq, freq), eq.magnitude_db_at(freq));
            assert!((measured - drawn).abs() < 0.05 * drawn.abs().max(1.0), "{freq} Hz: measured {measured} dB, drawn {drawn} dB");
        }
        assert!((eq.magnitude_db_at(500.0) - 6.0).abs() < 0.1, "the bell peaks at its gain");
        // the cut alone: Butterworth, -3 dB at its frequency, 48 dB/octave below
        let mut cut = ParametricEq::new(1, FS);
        cut.set_band(0, EqBand { enabled: true, kind: EqBandKind::LowCut, frequency: 80.0, gain_db: 0.0, q: 0.7, slope: 4 });
        assert!((cut.magnitude_db_at(80.0) + 3.01).abs() < 0.01, "{}", cut.magnitude_db_at(80.0));
        let slope = cut.magnitude_db_at(10.0) - cut.magnitude_db_at(20.0);
        assert!((slope + 48.0).abs() < 1.0, "{slope} dB/octave");
    }

    #[test]
    fn switching_kinds_and_slopes_changes_the_section_count() {
        let mut eq = ParametricEq::<f64>::new(2, FS);
        eq.set_enabled(0, true);
        eq.set_kind(0, EqBandKind::HighCut);
        eq.set_slope(0, 3);
        assert_eq!(eq.active[0], 3);
        eq.set_kind(0, EqBandKind::Notch);
        assert_eq!(eq.active[0], 1);
        eq.set_frequency(0, 1e9);
        assert_eq!(eq.band(0).frequency, 0.45 * FS);
        eq.set_enabled(0, false);
        assert_eq!(eq.active[0], 0);
    }
}