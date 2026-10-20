//! Phase vocoder: pitch shifting and time stretching.

use crate::alloc_prelude::*;
use crate::fft::RealFft;
use crate::processor::Processor;
use crate::units::*;

fn princarg<T: Float>(x: T) -> T {
    x - T::_TAU * (x / T::_TAU)._round()
}

/// The frame-by-frame engine shared by [`PitchShifter`] and [`time_stretch`]: analysis, true
/// frequencies from phase differences, spectral peaks with their regions, and resynthesis with
/// identity phase locking (Laroche and Dolson, 1999). Each peak's phase advances at its (shifted)
/// true frequency; the bins around it keep their phase offsets from it, so partials stay coherent
/// instead of smearing ("phasiness").
#[derive(Debug, Clone)]
struct Vocoder<T: Float> {
    fft: RealFft<T>,
    window: Vec<T>,
    frame: Vec<T>,
    spectrum: Vec<Complex<T>>,
    magnitude: Vec<T>,
    phase: Vec<T>,
    previous_phase: Vec<T>,
    /// synthesis phase of every bin in the last output frame
    synthesis_phase: Vec<T>,
    peaks: Vec<usize>,
    output: Vec<Complex<T>>,
    first: bool,
}

impl<T: Float> Vocoder<T> {
    fn new(fft_len: usize) -> Self {
        let bins = fft_len / 2 + 1;
        Self {
            fft: RealFft::new(fft_len),
            window: (0..fft_len).map(|i| T::_lit(0.5 - 0.5 * (core::f64::consts::TAU * i as f64 / fft_len as f64).cos())).collect(),
            frame: vec![T::_ZERO; fft_len],
            spectrum: vec![Complex::zero(); bins],
            magnitude: vec![T::_ZERO; bins],
            phase: vec![T::_ZERO; bins],
            previous_phase: vec![T::_ZERO; bins],
            synthesis_phase: vec![T::_ZERO; bins],
            peaks: Vec::with_capacity(bins),
            output: vec![Complex::zero(); bins],
            first: true,
        }
    }
    fn len(&self) -> usize {
        self.frame.len()
    }
    fn reset(&mut self) {
        self.first = true;
    }
    /// Sum of the squared window over overlapping frames `hop` apart (the overlap-add gain of
    /// analysis and synthesis windows): constant for a Hann window at a hop of a quarter frame.
    fn overlap_gain(&self, hop: usize) -> T {
        let total: f64 = self.window.iter().map(|w| w.to_f64().unwrap_or(0.0).powi(2)).sum();
        T::_lit(total / hop as f64)
    }

    /// Transforms `self.frame` (raw samples, oldest first): analyzed `analysis_hop` samples after
    /// the previous frame, resynthesized `synthesis_hop` after the previous output frame, with every
    /// frequency scaled by `ratio`. Leaves the windowed output frame in `self.frame`.
    fn process(&mut self, analysis_hop: usize, synthesis_hop: usize, ratio: T) {
        let n = self.len();
        let bins = self.spectrum.len();
        for (f, &w) in self.frame.iter_mut().zip(&self.window) {
            *f = *f * w;
        }
        self.fft.forward(&self.frame, &mut self.spectrum);
        let mut loudest = T::_ZERO;
        for ((m, p), z) in self.magnitude.iter_mut().zip(self.phase.iter_mut()).zip(&self.spectrum) {
            *m = z.norm();
            *p = z.arg();
            loudest = loudest._max(*m);
        }
        // peaks: local maxima over two bins each side, above a floor
        self.peaks.clear();
        let floor = loudest * T::_lit(1e-5);
        for k in 2..bins.saturating_sub(2) {
            let m = self.magnitude[k];
            if m > floor && m > self.magnitude[k - 1] && m >= self.magnitude[k + 1] && m > self.magnitude[k - 2] && m >= self.magnitude[k + 2] {
                self.peaks.push(k);
            }
        }
        self.output.iter_mut().for_each(|z| *z = Complex::zero());
        let bin_omega = T::_TAU / T::_lit(n as f64);
        let (ha, hs) = (T::_lit(analysis_hop as f64), T::_lit(synthesis_hop as f64));
        let mut region_start = 0;
        for (i, &p) in self.peaks.iter().enumerate() {
            // the region runs to the lowest bin between this peak and the next
            let region_end = match self.peaks.get(i + 1) {
                Some(&next) => (p..next).min_by(|&a, &b| self.magnitude[a].partial_cmp(&self.magnitude[b]).unwrap_or(core::cmp::Ordering::Equal)).unwrap_or(p) + 1,
                None => bins,
            };
            // true frequency of the peak from its phase advance, in radians per sample
            let expected = bin_omega * T::_lit(p as f64);
            let omega = if self.first { expected } else { expected + princarg(self.phase[p] - self.previous_phase[p] - expected * ha) / ha };
            let target = (T::_lit(p as f64) * ratio)._round().to_f64().unwrap_or(0.0) as isize;
            let shift = target - p as isize;
            // the peak's synthesis phase: advanced at its new frequency from where that bin was
            let peak_phase = if self.first {
                self.phase[p]
            } else {
                let from = (target.clamp(0, bins as isize - 1)) as usize;
                princarg(self.synthesis_phase[from] + omega * ratio * hs)
            };
            for k in region_start..region_end {
                let j = k as isize + shift;
                if j <= 0 || j >= bins as isize - 1 {
                    continue;
                }
                let angle = peak_phase + self.phase[k] - self.phase[p];
                self.output[j as usize] = self.output[j as usize] + Complex::from_polar(self.magnitude[k], angle);
            }
            region_start = region_end;
        }
        for (s, z) in self.synthesis_phase.iter_mut().zip(&self.output) {
            *s = z.arg();
        }
        self.previous_phase.copy_from_slice(&self.phase);
        self.first = false;
        self.fft.inverse(&self.output, &mut self.frame);
        for (f, &w) in self.frame.iter_mut().zip(&self.window) {
            *f = *f * w;
        }
    }
}


/// Real-time pitch shifter: changes pitch without changing duration.
///
/// A phase vocoder with identity phase locking moves each spectral peak (and the bins around it)
/// to its new frequency and keeps the partials' phases coherent, so tonal material shifts cleanly;
/// sharp transients soften somewhat, and formants move with the pitch. Output is delayed by
/// [`latency`](Self::latency) (the frame length). At a ratio of exactly 1 it passes the input
/// through (delayed by the same latency), so automation through 0 semitones is seamless.
#[derive(Debug, Clone)]
pub struct PitchShifter<T: Float> {
    vocoder: Vocoder<T>,
    hop: usize,
    semitones: T,
    ratio: T,
    /// the last fft_len inputs (a ring)
    input: Vec<T>,
    write: usize,
    since: usize,
    /// overlap-add accumulator, fft_len ahead (a ring)
    output: Vec<T>,
    read: usize,
    norm: T,
}

impl<T: Float> PitchShifter<T> {
    /// A 2048-sample frame (43 ms at 48 kHz) with 4x overlap.
    pub fn new() -> Self {
        Self::with_fft_len(2_048)
    }
    /// Longer frames resolve low notes better and smear transients more. Panics unless `fft_len`
    /// is a power of two >= 64.
    pub fn with_fft_len(fft_len: usize) -> Self {
        assert!(fft_len >= 64 && fft_len.is_power_of_two(), "fft_len must be a power of two >= 64");
        let vocoder = Vocoder::new(fft_len);
        let hop = fft_len / 4;
        let norm = T::_ONE / vocoder.overlap_gain(hop);
        Self { vocoder, hop, semitones: T::_ZERO, ratio: T::_ONE, input: vec![T::_ZERO; fft_len], write: 0, since: 0, output: vec![T::_ZERO; fft_len], read: 0, norm }
    }
    /// Shift in semitones (-36..36).
    pub fn set_semitones(&mut self, semitones: T) {
        self.semitones = semitones._clamp(T::_lit(-36.0), T::_lit(36.0));
        self.ratio = (self.semitones / T::_lit(12.0))._exp2();
    }
    /// Shift in semitones.
    pub fn semitones(&self) -> T {
        self.semitones
    }
    /// The frequency ratio (2 = an octave up).
    pub fn ratio(&self) -> T {
        self.ratio
    }
    /// Delay of the output in samples.
    pub fn latency(&self) -> usize {
        self.input.len()
    }
    /// Clears the buffers and the vocoder's phases.
    pub fn reset(&mut self) {
        self.input.iter_mut().chain(self.output.iter_mut()).for_each(|s| *s = T::_ZERO);
        self.write = 0;
        self.since = 0;
        self.read = 0;
        self.vocoder.reset();
    }
    /// Processes one sample.
    #[inline]
    pub fn process_sample(&mut self, x: T) -> T {
        let n = self.input.len();
        let delayed = self.input[self.write];
        self.input[self.write] = x;
        self.write = (self.write + 1) % n;
        let y = self.output[self.read];
        self.output[self.read] = T::_ZERO;
        self.read = (self.read + 1) % n;
        self.since += 1;
        if self.since == self.hop {
            self.since = 0;
            for (i, f) in self.vocoder.frame.iter_mut().enumerate() {
                *f = self.input[(self.write + i) % n];
            }
            self.vocoder.process(self.hop, self.hop, self.ratio);
            // the new frame plays out over the next fft_len samples
            for (i, &s) in self.vocoder.frame.iter().enumerate() {
                let at = (self.read + i) % n;
                self.output[at] = self.output[at] + s * self.norm;
            }
        }
        if self.ratio == T::_ONE { delayed } else { y }
    }
    /// Processes `block` in place.
    pub fn process(&mut self, block: &mut [T]) {
        for s in block {
            *s = self.process_sample(*s);
        }
    }
}

impl<T: Float> Default for PitchShifter<T> {
    fn default() -> Self {
        Self::new()
    }
}

impl<T: Float> Processor<T> for PitchShifter<T> {
    fn process(&mut self, block: &mut [T]) {
        PitchShifter::process(self, block)
    }
    fn reset(&mut self) {
        PitchShifter::reset(self)
    }
}

/// Real-time time stretching: plays a sample buffer at a speed (0.5 = half speed, twice as long)
/// without changing its pitch, the speed changeable while playing, with an optional loop.
///
/// The same phase-locked vocoder as [`time_stretch`], driven by the output: [`render`](Self::render)
/// fills a block, reading the source `speed` samples further for each sample it writes (a frame
/// every quarter frame of output). Reads wrap inside the loop region, so frames that straddle its
/// end hold the loop's start, and playback stays continuous across it. At a constant speed the
/// output is exactly [`time_stretch`]'s. Allocates only when constructed.
///
/// tend: Signal processing / spectral / streaming time stretch
#[derive(Debug, Clone)]
pub struct TimeStretcher<T: Float> {
    vocoder: Vocoder<T>,
    hop: usize,
    speed: f64,
    /// source position of the next frame's first sample
    cursor: f64,
    /// where the last frame started (for the analysis hop), none right after a seek
    previous: Option<isize>,
    latest: f64,
    /// output samples since the last frame
    elapsed: usize,
    looping: Option<(usize, usize)>,
    /// overlap-add accumulator, a frame ahead (a ring)
    output: Vec<T>,
    read: usize,
    norm: T,
}

impl<T: Float> TimeStretcher<T> {
    /// A 2048-sample frame (43 ms at 48 kHz) with 4x overlap, at speed 1, from the source's start.
    pub fn new() -> Self {
        Self::with_fft_len(2_048)
    }
    /// Longer frames resolve low notes better and smear transients more. Panics unless `fft_len`
    /// is a power of two >= 64.
    pub fn with_fft_len(fft_len: usize) -> Self {
        assert!(fft_len >= 64 && fft_len.is_power_of_two(), "fft_len must be a power of two >= 64");
        let vocoder = Vocoder::new(fft_len);
        let hop = fft_len / 4;
        let norm = T::_ONE / vocoder.overlap_gain(hop);
        let mut s = Self {
            vocoder,
            hop,
            speed: 1.0,
            cursor: 0.0,
            previous: None,
            latest: 0.0,
            elapsed: 0,
            looping: None,
            output: vec![T::_ZERO; fft_len],
            read: 0,
            norm,
        };
        s.seek(0.0);
        s
    }

    /// Playback speed, 0.1..=10 (clamped): 1 is the original tempo, 0.5 half speed. Takes effect
    /// from the next frame (a quarter frame of output at most).
    pub fn set_speed(&mut self, speed: T) {
        self.speed = speed.to_f64().unwrap_or(1.0).clamp(0.1, 10.0);
    }
    /// Playback speed.
    pub fn speed(&self) -> T {
        T::_lit(self.speed)
    }
    /// Loops source samples `start..end` (playback that reaches `end` continues from `start`), or
    /// stops looping (`None`). Playback before `start` runs into the loop. Panics on an empty region.
    pub fn set_loop(&mut self, region: Option<core::ops::Range<usize>>) {
        self.looping = region.map(|r| {
            assert!(r.start < r.end, "empty loop region");
            (r.start, r.end)
        });
    }
    /// Plays from source position `position` (samples; before 0 or past the end is silence): it is
    /// heard [`latency`](Self::latency) samples of output later. What is already playing fades
    /// out under the new frames.
    pub fn seek(&mut self, position: f64) {
        let half = (self.vocoder.len() / 2) as f64;
        self.cursor = position - half;
        self.latest = self.cursor;
        self.previous = None;
        self.elapsed = 0;
        self.vocoder.reset();
    }
    /// The source position heard now, in samples (inside the loop when looping).
    pub fn position(&self) -> f64 {
        let half = (self.vocoder.len() / 2) as f64;
        self.wrap_position(self.latest + half + self.speed * (self.elapsed as f64 - half))
    }
    /// Output samples from a [`seek`](Self::seek) to hearing its position: half a frame.
    pub fn latency(&self) -> usize {
        self.vocoder.len() / 2
    }
    /// Silence, from the source's start, at the current speed and loop.
    pub fn reset(&mut self) {
        self.output.iter_mut().for_each(|s| *s = T::_ZERO);
        self.read = 0;
        self.seek(0.0);
    }

    /// Fills `out` with the next samples of `source` played at the current speed.
    pub fn render(&mut self, source: &[T], out: &mut [T]) {
        let n = self.output.len();
        for o in out {
            if self.previous.is_none() || self.elapsed == self.hop {
                self.frame(source);
            }
            *o = self.output[self.read];
            self.output[self.read] = T::_ZERO;
            self.read = (self.read + 1) % n;
            self.elapsed += 1;
        }
    }

    /// Analyzes the frame at the cursor and adds its resynthesis to the output, a frame ahead.
    fn frame(&mut self, source: &[T]) {
        let n = self.vocoder.len();
        let start = self.cursor.round() as isize;
        let analysis_hop = self.previous.map_or(1, |p| (start - p).max(1) as usize);
        for (i, f) in self.vocoder.frame.iter_mut().enumerate() {
            *f = Self::sample(source, self.looping, start + i as isize);
        }
        self.vocoder.process(analysis_hop, self.hop, T::_ONE);
        for (i, &s) in self.vocoder.frame.iter().enumerate() {
            let at = (self.read + i) % n;
            self.output[at] = self.output[at] + s * self.norm;
        }
        self.previous = Some(start);
        self.latest = start as f64;
        self.elapsed = 0;
        self.cursor += self.speed * self.hop as f64;
        // keep the numbers small while looping: a whole number of loops back
        if let (Some((a, b)), Some(p)) = (self.looping, self.previous) {
            let len = (b - a) as isize;
            if p >= b as isize + n as isize {
                let back = len * ((p - a as isize) / len);
                self.previous = Some(p - back);
                self.latest -= back as f64;
                self.cursor -= back as f64;
            }
        }
    }

    /// Source sample `i`, wrapped into the loop past its end; silence outside the source.
    fn sample(source: &[T], looping: Option<(usize, usize)>, i: isize) -> T {
        let i = match looping {
            Some((a, b)) if i >= b as isize => a as isize + (i - a as isize) % (b - a) as isize,
            _ => i,
        };
        if i >= 0 && (i as usize) < source.len() { source[i as usize] } else { T::_ZERO }
    }

    fn wrap_position(&self, p: f64) -> f64 {
        match self.looping {
            Some((a, b)) if p >= b as f64 => a as f64 + (p - a as f64) % (b - a) as f64,
            _ => p,
        }
    }
}

impl<T: Float> Default for TimeStretcher<T> {
    fn default() -> Self {
        Self::new()
    }
}

/// Changes the duration of `input` by `factor` (2 = twice as long) without changing its pitch,
/// with the same phase-locked vocoder as [`PitchShifter`] (frames of `fft_len`, a power of two;
/// 2048 suits most music at 44.1-48 kHz). The result has `round(input.len() * factor)` samples.
/// Offline: allocates the output. Panics unless `factor` is in 0.1..=10.
pub fn time_stretch<T: Float>(input: &[T], factor: f64, fft_len: usize) -> Vec<T> {
    assert!((0.1..=10.0).contains(&factor), "stretch factor must be within 0.1..=10");
    let mut vocoder = Vocoder::<T>::new(fft_len);
    let n = fft_len;
    let hop = n / 4;
    let norm = T::_ONE / vocoder.overlap_gain(hop);
    // pad so the first and last samples sit inside full frames
    let pad = n;
    let padded_len = input.len() + 2 * pad;
    let at = |i: isize| if i >= pad as isize && ((i as usize) - pad) < input.len() { input[i as usize - pad] } else { T::_ZERO };
    let out_len = ((input.len() as f64) * factor).round() as usize;
    let offset = (pad as f64 * factor).round() as usize;
    let mut output = vec![T::_ZERO; offset + out_len + 2 * n];
    let mut previous = 0usize;
    let mut m = 0usize;
    loop {
        let synthesis = m * hop;
        let analysis = (synthesis as f64 / factor).round() as usize;
        if analysis >= padded_len || synthesis >= output.len() - n {
            break;
        }
        for (i, f) in vocoder.frame.iter_mut().enumerate() {
            *f = at((analysis + i) as isize);
        }
        vocoder.process((analysis - previous).max(1), hop, T::_ONE);
        for (i, &s) in vocoder.frame.iter().enumerate() {
            output[synthesis + i] = output[synthesis + i] + s * norm;
        }
        previous = analysis;
        m += 1;
    }
    output.drain(..offset);
    output.truncate(out_len);
    output
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::analysis::PitchDetector;
    use crate::osc::{Oscillator, Sine, Waveform};

    const FS: f64 = 48_000.0;

    fn cents(a: f64, b: f64) -> f64 {
        1_200.0 * (a / b).log2()
    }
    fn rms(x: &[f64]) -> f64 {
        (x.iter().map(|s| s * s).sum::<f64>() / x.len() as f64).sqrt()
    }
    fn pitch_of(x: &[f64]) -> f64 {
        let mut d = PitchDetector::new(40.0, 2_000.0, FS);
        let start = x.len() - d.frame_len();
        d.detect(&x[start..]).expect("voiced").frequency
    }

    #[test]
    fn shifts_tones_by_the_requested_interval() {
        for (wave, freq, semitones) in [(Waveform::Sine, 440.0, 7.0), (Waveform::Saw, 220.0, -5.0), (Waveform::Saw, 110.0, 12.0), (Waveform::Triangle, 330.0, 3.5)] {
            let x: Vec<f64> = Oscillator::new(wave, freq, FS).take(48_000).map(|s| 0.5 * s).collect();
            let mut shifter = PitchShifter::new();
            shifter.set_semitones(semitones);
            let mut y = x.clone();
            shifter.process(&mut y);
            let expected = freq * 2f64.powf(semitones / 12.0);
            let found = pitch_of(&y);
            assert!(cents(found, expected).abs() < 5.0, "{wave:?} {freq} Hz {semitones:+} st: {found} Hz vs {expected}");
            let level = 20.0 * (rms(&y[24_000..]) / rms(&x[24_000..])).log10();
            assert!(level.abs() < 1.5, "{wave:?} {semitones:+} st: level {level} dB");
        }
    }

    #[test]
    fn zero_semitones_is_a_pure_delay() {
        let x: Vec<f64> = Sine::new(440.0, FS).take(10_000).collect();
        let mut shifter = PitchShifter::new();
        shifter.set_semitones(0.0);
        let mut y = x.clone();
        shifter.process(&mut y);
        let latency = shifter.latency();
        assert!((latency..10_000).all(|i| y[i] == x[i - latency]));
    }

    #[test]
    fn time_stretch_keeps_pitch_and_level() {
        let x: Vec<f64> = Oscillator::new(Waveform::Saw, 196.0, FS).take(48_000).map(|s| 0.5 * s).collect();
        for factor in [0.5, 0.8, 1.5, 2.0] {
            let y = time_stretch(&x, factor, 2_048);
            assert_eq!(y.len(), (48_000.0 * factor) as usize);
            let found = pitch_of(&y[..y.len() * 3 / 4]);
            assert!(cents(found, 196.0).abs() < 5.0, "x{factor}: {found} Hz");
            let middle = &y[y.len() / 4..y.len() * 3 / 4];
            let level = 20.0 * (rms(middle) / rms(&x[12_000..36_000])).log10();
            assert!(level.abs() < 1.0, "x{factor}: level {level} dB");
        }
    }

    #[test]
    fn streaming_at_a_constant_speed_is_time_stretch() {
        let x: Vec<f64> = Oscillator::new(Waveform::Saw, 220.0, FS).zip(Sine::new(3.0, FS)).take(24_000).map(|(s, m)| 0.5 * s * (0.6 + 0.4 * m)).collect();
        let n = 2_048;
        for speed in [1.0, 0.5, 2.0, 0.25] {
            let factor = 1.0 / speed;
            let offline = time_stretch(&x, factor, n);
            // time_stretch pads n zeros ahead and drops their output: start that early
            let mut s = TimeStretcher::with_fft_len(n);
            s.set_speed(speed);
            s.seek(-((n / 2) as f64));
            let offset = (n as f64 * factor).round() as usize;
            let mut out = vec![0.0; offset + offline.len()];
            for block in out.chunks_mut(300) {
                s.render(&x, block);
            }
            // the same frames, overlap-added in the same order: the same bits
            assert!(out[offset..] == offline[..], "speed {speed}: streaming differs from time_stretch");
        }
    }

    #[test]
    fn loops_keep_pitch_and_level_and_stay_inside() {
        let x: Vec<f64> = Oscillator::new(Waveform::Saw, 196.0, FS).take(48_000).map(|s| 0.5 * s).collect();
        let mut s = TimeStretcher::new();
        s.set_speed(0.75);
        s.set_loop(Some(12_000..36_000));
        s.seek(12_000.0);
        let mut out = vec![0.0; 3 * 48_000];
        for block in out.chunks_mut(256) {
            s.render(&x, block);
            let p = s.position();
            assert!((11_000.0..36_000.0).contains(&p), "position {p} left the loop");
        }
        // 3 s at 0.75 covers the 0.5 s loop four and a half times
        let found = pitch_of(&out);
        assert!(cents(found, 196.0).abs() < 5.0, "looped: {found} Hz");
        let level = 20.0 * (rms(&out[24_000..]) / rms(&x[12_000..36_000])).log10();
        assert!(level.abs() < 1.0, "looped: level {level} dB");
        assert!(out.iter().all(|v| v.is_finite()));
    }

    #[test]
    fn speed_changes_while_playing() {
        let x: Vec<f64> = Oscillator::new(Waveform::Saw, 247.0, FS).take(5 * 48_000).map(|s| 0.5 * s).collect();
        let mut s = TimeStretcher::new();
        s.seek(1_000.0);
        let mut out = vec![0.0; 48_000];
        s.render(&x, &mut out[..24_000]);
        s.set_speed(0.5);
        s.render(&x, &mut out[24_000..]);
        // 0.5 s at 1x, then 0.5 s at 0.5x: 36,000 samples of source past the seek, less the latency
        let want = 1_000.0 + 24_000.0 + 12_000.0 - 0.5 * s.latency() as f64;
        assert!((s.position() - want).abs() < s.latency() as f64, "position {} vs {want}", s.position());
        let found = pitch_of(&out);
        assert!(cents(found, 247.0).abs() < 5.0, "after the change: {found} Hz");
        assert_eq!(s.speed(), 0.5);
    }

    #[test]
    fn time_stretch_moves_events_in_time() {
        // a tone burst from 0.25 s to 0.5 s lands at 0.5 s to 1 s when doubled
        let mut x = vec![0.0; 48_000];
        let mut tone = Sine::new(500.0, FS);
        for s in &mut x[12_000..24_000] {
            *s = tone.next_sample();
        }
        let y = time_stretch(&x, 2.0, 2_048);
        let level = |a: usize, b: usize| rms(&y[a..b]);
        assert!(level(26_000, 46_000) > 0.6, "the burst is there");
        assert!(level(0, 20_000) < 0.05 && level(52_000, 96_000) < 0.05, "and nowhere else");
    }
}
