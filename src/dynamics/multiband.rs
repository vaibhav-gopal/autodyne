//! Multiband compressor.

use crate::alloc_prelude::*;
use super::Compressor;
use crate::channels::{AudioBuffer, MultiProcessor};
use crate::filter::{Crossover, MAX_BANDS};
use crate::units::*;

/// Parameters per band: threshold, slope, attack, release, makeup, bypass.
pub const MULTIBAND_BAND_PARAMS: usize = 6;
pub(crate) const MULTIBAND_PARAM_IDS: [[&str; MULTIBAND_BAND_PARAMS]; MAX_BANDS] = [
    ["band1_threshold_db", "band1_slope", "band1_attack_s", "band1_release_s", "band1_makeup_db", "band1_bypass"],
    ["band2_threshold_db", "band2_slope", "band2_attack_s", "band2_release_s", "band2_makeup_db", "band2_bypass"],
    ["band3_threshold_db", "band3_slope", "band3_attack_s", "band3_release_s", "band3_makeup_db", "band3_bypass"],
    ["band4_threshold_db", "band4_slope", "band4_attack_s", "band4_release_s", "band4_makeup_db", "band4_bypass"],
];
pub(crate) const MULTIBAND_PARAM_NAMES: [[&str; MULTIBAND_BAND_PARAMS]; MAX_BANDS] = [
    ["Band 1 threshold", "Band 1 slope", "Band 1 attack", "Band 1 release", "Band 1 makeup", "Band 1 bypass"],
    ["Band 2 threshold", "Band 2 slope", "Band 2 attack", "Band 2 release", "Band 2 makeup", "Band 2 bypass"],
    ["Band 3 threshold", "Band 3 slope", "Band 3 attack", "Band 3 release", "Band 3 makeup", "Band 3 bypass"],
    ["Band 4 threshold", "Band 4 slope", "Band 4 attack", "Band 4 release", "Band 4 makeup", "Band 4 bypass"],
];
pub(crate) const CROSSOVER_PARAM_IDS: [&str; MAX_BANDS - 1] = ["crossover1_hz", "crossover2_hz", "crossover3_hz"];
pub(crate) const CROSSOVER_PARAM_NAMES: [&str; MAX_BANDS - 1] = ["Crossover 1", "Crossover 2", "Crossover 3"];

/// A compressor per frequency band: Linkwitz-Riley crossovers split the signal (2 to 4 bands),
/// each band is compressed on its own (linked across channels), and the bands are summed back.
///
/// With every band bypassed, or below its threshold, the output is the input through the
/// crossover's all-pass: the same level at every frequency. Multichannel by nature; process an
/// [`AudioBuffer`] or use [`process_mono`](Self::process_mono).
#[derive(Debug, Clone)]
pub struct MultibandCompressor<T: Float> {
    crossovers: Vec<Crossover<T>>,
    compressors: [Compressor<T>; MAX_BANDS],
    bypass: [bool; MAX_BANDS],
    /// one frame's bands, per channel
    split: Vec<[T; MAX_BANDS]>,
}

impl<T: Float> MultibandCompressor<T> {
    /// `bands` bands (2 to 4) for `channels` channels. Crossovers: 1 kHz (2 bands), 200 Hz and
    /// 2 kHz (3), 120 Hz, 1 kHz and 6 kHz (4); each band starts as [`Compressor::new`]. Panics on
    /// other band counts or no channels.
    pub fn new(channels: usize, bands: usize, sample_rate: T) -> Self {
        assert!(channels > 0, "need at least one channel");
        let frequencies: &[f64] = match bands {
            2 => &[1_000.0],
            3 => &[200.0, 2_000.0],
            4 => &[120.0, 1_000.0, 6_000.0],
            _ => panic!("a multiband compressor has 2 to {MAX_BANDS} bands"),
        };
        let frequencies: Vec<T> = frequencies.iter().map(|&f| T::_lit(f)).collect();
        Self {
            crossovers: vec![Crossover::new(&frequencies, sample_rate); channels],
            compressors: [Compressor::new(sample_rate); MAX_BANDS],
            bypass: [false; MAX_BANDS],
            split: vec![[T::_ZERO; MAX_BANDS]; channels],
        }
    }
    /// Number of channels.
    pub fn channels(&self) -> usize {
        self.crossovers.len()
    }
    /// Number of bands.
    pub fn bands(&self) -> usize {
        self.crossovers[0].bands()
    }
    /// Crossover `split`'s frequency in Hz.
    pub fn crossover_frequency(&self, split: usize) -> T {
        self.crossovers[0].frequency(split)
    }
    /// Moves crossover `split` (kept a third of an octave from its neighbours).
    pub fn set_crossover_frequency(&mut self, split: usize, hz: T) {
        self.crossovers.iter_mut().for_each(|c| c.set_frequency(split, hz));
    }
    /// Band `band`'s compressor, to read or change its settings.
    pub fn band(&self, band: usize) -> &Compressor<T> {
        &self.compressors[band]
    }
    /// Band `band`'s compressor, mutably.
    pub fn band_mut(&mut self, band: usize) -> &mut Compressor<T> {
        &mut self.compressors[band]
    }
    /// Passes band `band` uncompressed.
    pub fn set_bypass(&mut self, band: usize, bypass: bool) {
        self.bypass[band] = bypass;
    }
    /// Whether band `band` is bypassed.
    pub fn bypass(&self, band: usize) -> bool {
        self.bypass[band]
    }
    /// Clears the crossovers' and compressors' state.
    pub fn reset(&mut self) {
        self.crossovers.iter_mut().for_each(Crossover::reset);
        self.compressors.iter_mut().for_each(Compressor::reset);
    }

    /// Compresses the frame held in `self.split[c][0]` (input) for every channel; returns via the
    /// same slot.
    #[inline]
    fn process_frame(&mut self) {
        let bands = self.bands();
        for (crossover, frame) in self.crossovers.iter_mut().zip(self.split.iter_mut()) {
            let x = frame[0];
            crossover.split(x, frame);
        }
        let mut gains = [T::_ONE; MAX_BANDS];
        for (b, gain) in gains[..bands].iter_mut().enumerate() {
            let level = self.split.iter().fold(T::_ZERO, |m, f| m._max(f[b]._abs()));
            let g = self.compressors[b].gain_for_level(level);
            if !self.bypass[b] {
                *gain = g;
            }
        }
        for frame in &mut self.split {
            let y = frame[..bands].iter().zip(&gains).fold(T::_ZERO, |acc, (&s, &g)| acc + s * g);
            frame[0] = y;
        }
    }

    /// Compresses every channel of `buffer`. Panics if the channel count differs.
    pub fn process_buffer(&mut self, buffer: &mut AudioBuffer<T>) {
        assert_eq!(buffer.channels(), self.channels(), "the buffer's channel count must match");
        for i in 0..buffer.frames() {
            for c in 0..self.channels() {
                self.split[c][0] = buffer.channel(c)[i];
            }
            self.process_frame();
            for c in 0..self.channels() {
                buffer.channel_mut(c)[i] = self.split[c][0];
            }
        }
        self.crossovers.iter_mut().for_each(Crossover::flush_denormals);
    }
    /// Compresses one channel. Panics unless built for one channel.
    pub fn process_mono(&mut self, block: &mut [T]) {
        assert_eq!(self.channels(), 1, "process_mono needs a one-channel multiband compressor");
        for s in block {
            self.split[0][0] = *s;
            self.process_frame();
            *s = self.split[0][0];
        }
        self.crossovers[0].flush_denormals();
    }
}

impl<T: Float> MultiProcessor<T> for MultibandCompressor<T> {
    fn process(&mut self, buffer: &mut AudioBuffer<T>) {
        self.process_buffer(buffer);
    }
    fn reset(&mut self) {
        MultibandCompressor::reset(self)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    const FS: f64 = 48_000.0;

    fn tone(freq: f64, amplitude: f64) -> Vec<f64> {
        (0..48_000).map(|i| amplitude * (std::f64::consts::TAU * freq * i as f64 / FS).sin()).collect()
    }
    fn rms_db(x: &[f64]) -> f64 {
        10.0 * (x.iter().map(|s| s * s).sum::<f64>() / x.len() as f64).log10()
    }

    #[test]
    fn transparent_below_threshold_or_bypassed() {
        for bands in 2..=4 {
            for freq in [60.0, 500.0, 3_000.0, 12_000.0] {
                let mut mb = MultibandCompressor::new(1, bands, FS);
                for b in 0..bands {
                    mb.set_bypass(b, true);
                }
                let x = tone(freq, 0.9);
                let mut y = x.clone();
                mb.process_mono(&mut y);
                let change = rms_db(&y[24_000..]) - rms_db(&x[24_000..]);
                assert!(change.abs() < 1e-3, "{bands} bands, {freq} Hz: {change} dB");
            }
        }
    }

    #[test]
    fn compresses_only_the_band_a_tone_is_in() {
        let mut mb = MultibandCompressor::new(1, 3, FS);
        for b in [0, 2] {
            mb.set_bypass(b, true);
        }
        mb.band_mut(1).set_threshold_db(-30.0);
        mb.band_mut(1).set_ratio(10.0);
        mb.band_mut(1).set_knee_db(0.0);
        let mut low = tone(60.0, 0.5);
        let mut mid = tone(700.0, 0.5);
        let (low_in, mid_in) = (rms_db(&low[24_000..]), rms_db(&mid[24_000..]));
        mb.process_mono(&mut low);
        mb.reset();
        mb.process_mono(&mut mid);
        assert!((rms_db(&low[24_000..]) - low_in).abs() < 0.1, "the low band is untouched");
        let reduction = mid_in - rms_db(&mid[24_000..]);
        assert!(reduction > 15.0, "the mid tone is compressed by {reduction} dB");
    }

    #[test]
    fn stereo_bands_are_linked() {
        let mut mb = MultibandCompressor::new(2, 2, FS);
        let mut buffer = AudioBuffer::new(2, 24_000);
        buffer.channel_mut(0).copy_from_slice(&tone(300.0, 0.9)[..24_000]);
        buffer.channel_mut(1).copy_from_slice(&tone(300.0, 0.09)[..24_000]);
        mb.process_buffer(&mut buffer);
        let ratio = rms_db(&buffer.channel(0)[12_000..]) - rms_db(&buffer.channel(1)[12_000..]);
        assert!((ratio - 20.0).abs() < 0.01, "the 20 dB balance is kept: {ratio}");
    }
}