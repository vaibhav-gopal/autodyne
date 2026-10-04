use std::ops::RangeInclusive;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::Arc;

use super::voice::{MAX_STRETCH, SINC_CUTOFF, SINC_ZEROS};
use crate::filter::design_lowpass;
use crate::units::*;
use crate::wav::{self, WavError, WavLoopKind};

/// Zero samples around every level's data, so interpolation near the ends needs no bounds checks:
/// the widest sinc kernel's reach plus a little.
pub(crate) const PAD: usize = (SINC_ZEROS as f64 * MAX_STRETCH / SINC_CUTOFF) as usize + 4;
/// Mipmap levels stop at this many frames.
const MIN_LEVEL_FRAMES: usize = 64;

/// How a [`Sample`] loops.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum LoopMode {
    /// plays once to the end
    Off,
    /// repeats start..end until the note's release has faded
    Forward,
    /// bounces between start and end
    PingPong,
    /// repeats while the key is held, then plays on to the end (the release portion of the recording)
    Sustain,
}

impl LoopMode {
    pub const ALL: [LoopMode; 4] = [LoopMode::Off, LoopMode::Forward, LoopMode::PingPong, LoopMode::Sustain];
    pub const NAMES: [&'static str; 4] = ["Off", "Forward", "Ping-pong", "Sustain"];
}

/// Loop points in frames of the original sample.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Loop {
    pub start: usize,
    /// one past the last looped frame
    pub end: usize,
    pub mode: LoopMode,
    /// frames over which the end of the loop fades into the frames before its start (forward and
    /// sustain loops), hiding the seam of loops that are not cut perfectly
    pub crossfade: usize,
}

/// One resolution of a sample: planar channels padded with [`PAD`] zeros at both ends.
#[derive(Debug, Clone)]
pub(crate) struct Level<T> {
    pub channels: Vec<Vec<T>>,
    pub frames: usize,
}

/// Recorded audio ready to play: one or two channels, its sample rate, the key it sounds at
/// unpitched, and an optional loop.
///
/// For playback far above the root (more than an octave up), [`with_mipmaps`](Self::with_mipmaps)
/// adds half-rate copies so high notes stay alias-free without wide interpolation kernels; they
/// cost up to the sample's size again in memory.
#[derive(Debug, Clone)]
pub struct Sample<T: Float> {
    pub(crate) levels: Vec<Level<T>>,
    sample_rate: f64,
    root_key: f64,
    looping: Option<Loop>,
}

impl<T: Float> Sample<T> {
    /// From planar channels (at least one; more than two are dropped; all the same length).
    /// Root key 60 (middle C), no loop. Panics on no channels or uneven lengths.
    pub fn new(channels: Vec<Vec<T>>, sample_rate: T) -> Self {
        assert!(!channels.is_empty(), "a sample needs at least one channel");
        let frames = channels[0].len();
        assert!(channels.iter().all(|c| c.len() == frames), "every channel needs the same length");
        let padded = channels
            .into_iter()
            .take(2)
            .map(|c| {
                let mut v = Vec::with_capacity(frames + 2 * PAD);
                v.resize(PAD, T::_ZERO);
                v.extend(c);
                v.resize(frames + 2 * PAD, T::_ZERO);
                v
            })
            .collect();
        Self { levels: vec![Level { channels: padded, frames }], sample_rate: sample_rate.to_f64().unwrap_or(48_000.0), root_key: 60.0, looping: None }
    }
    pub fn from_mono(data: Vec<T>, sample_rate: T) -> Self {
        Self::new(vec![data], sample_rate)
    }
    /// From a WAV file in memory, taking its root key and first loop (forward loops become
    /// [`LoopMode::Forward`], ping-pong ones [`LoopMode::PingPong`]) from its `smpl` chunk.
    pub fn from_wav(bytes: &[u8]) -> Result<Self, WavError> {
        let file: wav::Wav<T> = wav::read(bytes)?;
        let mut sample = Self::new(file.channels, T::_lit(file.sample_rate as f64));
        if let Some(root) = file.root_key {
            sample.root_key = root;
        }
        if let Some(l) = file.loops.first() {
            let mode = if l.kind == WavLoopKind::PingPong { LoopMode::PingPong } else { LoopMode::Forward };
            sample = sample.with_loop(l.start, l.end, mode, 0);
        }
        Ok(sample)
    }
    /// The MIDI key (fractional for fine tuning) at which the sample plays at its recorded pitch.
    pub fn with_root_key(mut self, key: f64) -> Self {
        self.root_key = key;
        self
    }
    /// Loops `start..end` (clamped to the sample; ignored if empty); `crossfade` is limited to the
    /// loop length and to the frames before `start`.
    pub fn with_loop(mut self, start: usize, end: usize, mode: LoopMode, crossfade: usize) -> Self {
        let end = end.min(self.frames());
        self.looping = (start < end && mode != LoopMode::Off).then(|| Loop { start, end, mode, crossfade: crossfade.min(end - start).min(start) });
        self
    }
    /// Removes the loop.
    pub fn without_loop(mut self) -> Self {
        self.looping = None;
        self
    }
    /// Adds half-rate copies for `octaves` octaves (or until they are very short), so notes far
    /// above the root are band-limited cheaply.
    pub fn with_mipmaps(mut self, octaves: usize) -> Self {
        let taps: Vec<f64> = design_lowpass(0.225, 127, 1.0);
        let center = taps.len() / 2;
        while self.levels.len() <= octaves {
            let prev = self.levels.last().unwrap();
            let frames = prev.frames.div_ceil(2);
            if frames < MIN_LEVEL_FRAMES {
                break;
            }
            let channels = prev
                .channels
                .iter()
                .map(|data| {
                    let x = |i: isize| if i >= 0 && (i as usize) < prev.frames { data[PAD + i as usize].to_f64().unwrap_or(0.0) } else { 0.0 };
                    let mut out = vec![T::_ZERO; frames + 2 * PAD];
                    for (m, o) in out[PAD..PAD + frames].iter_mut().enumerate() {
                        // centered low-pass at frame 2m, then keep every other frame
                        let at = (2 * m) as isize;
                        let y: f64 = taps.iter().enumerate().map(|(j, h)| h * x(at + center as isize - j as isize)).sum();
                        *o = T::_lit(y);
                    }
                    out
                })
                .collect();
            self.levels.push(Level { channels, frames });
        }
        self
    }
    pub fn frames(&self) -> usize {
        self.levels[0].frames
    }
    pub fn channels(&self) -> usize {
        self.levels[0].channels.len()
    }
    pub fn sample_rate(&self) -> f64 {
        self.sample_rate
    }
    pub fn root_key(&self) -> f64 {
        self.root_key
    }
    pub fn loop_points(&self) -> Option<Loop> {
        self.looping
    }
    /// Frame `i` of `channel` in the original resolution (0 outside the sample).
    pub fn frame(&self, channel: usize, i: usize) -> T {
        let level = &self.levels[0];
        if i < level.frames { level.channels[channel][PAD + i] } else { T::_ZERO }
    }
}

/// Where a sample plays: a key range, a velocity range, and how it is tuned, leveled and panned.
#[derive(Debug, Clone)]
pub struct Zone<T: Float> {
    pub sample: Arc<Sample<T>>,
    pub keys: RangeInclusive<u8>,
    /// MIDI velocities 1..=127
    pub velocities: RangeInclusive<u8>,
    /// overrides the sample's root key
    pub root_key: Option<f64>,
    pub tune_cents: f64,
    pub gain_db: f64,
    /// -1 (left) .. 1 (right)
    pub pan: f64,
}

impl<T: Float> Zone<T> {
    /// Every key and velocity, untuned, unity gain, centered.
    pub fn new(sample: Arc<Sample<T>>) -> Self {
        Self { sample, keys: 0..=127, velocities: 0..=127, root_key: None, tune_cents: 0.0, gain_db: 0.0, pan: 0.0 }
    }
    pub fn keys(mut self, keys: RangeInclusive<u8>) -> Self {
        self.keys = keys;
        self
    }
    pub fn velocities(mut self, velocities: RangeInclusive<u8>) -> Self {
        self.velocities = velocities;
        self
    }
    pub fn root_key(mut self, key: f64) -> Self {
        self.root_key = Some(key);
        self
    }
    pub fn tune_cents(mut self, cents: f64) -> Self {
        self.tune_cents = cents;
        self
    }
    pub fn gain_db(mut self, db: f64) -> Self {
        self.gain_db = db;
        self
    }
    pub fn pan(mut self, pan: f64) -> Self {
        self.pan = pan.clamp(-1.0, 1.0);
        self
    }
    /// The effective root key: the zone's, else the sample's.
    pub fn root(&self) -> f64 {
        self.root_key.unwrap_or(self.sample.root_key())
    }
    pub fn contains(&self, key: u8, velocity: u8) -> bool {
        self.keys.contains(&key) && self.velocities.contains(&velocity)
    }
}

/// A set of zones: a multisampled instrument. Shared by voices through `Arc`; to swap instruments
/// while playing, give each voice the new map and keep the old one alive (dropped off the audio
/// thread) until they have switched.
#[derive(Debug)]
pub struct SampleMap<T: Float> {
    zones: Vec<Zone<T>>,
    /// round-robin counter shared by every voice
    next: AtomicUsize,
}

impl<T: Float> SampleMap<T> {
    pub fn new(zones: Vec<Zone<T>>) -> Self {
        Self { zones, next: AtomicUsize::new(0) }
    }
    /// One sample across the whole keyboard.
    pub fn single(sample: Sample<T>) -> Self {
        Self::new(vec![Zone::new(Arc::new(sample))])
    }
    pub fn zones(&self) -> &[Zone<T>] {
        &self.zones
    }
    /// The zone for `key` at MIDI `velocity`: the matching one, taking turns among several.
    pub fn find(&self, key: u8, velocity: u8) -> Option<&Zone<T>> {
        let matching = self.zones.iter().filter(|z| z.contains(key, velocity)).count();
        let turn = match matching {
            0 => return None,
            1 => 0,
            n => self.next.fetch_add(1, Ordering::Relaxed) % n,
        };
        self.zones.iter().filter(|z| z.contains(key, velocity)).nth(turn)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn construction_loops_and_metadata() {
        let s = Sample::new(vec![vec![0.5f64; 1_000], vec![-0.5; 1_000], vec![0.0; 1_000]], 44_100.0);
        assert_eq!((s.channels(), s.frames(), s.root_key()), (2, 1_000, 60.0), "at most two channels");
        assert_eq!((s.frame(1, 999), s.frame(0, 1_000)), (-0.5, 0.0));
        let s = s.with_loop(100, 5_000, LoopMode::Forward, 500);
        assert_eq!(s.loop_points(), Some(Loop { start: 100, end: 1_000, mode: LoopMode::Forward, crossfade: 100 }), "clamped");
        assert_eq!(s.clone().with_loop(10, 10, LoopMode::Forward, 0).loop_points(), None);
        assert_eq!(s.with_loop(0, 10, LoopMode::Off, 0).loop_points(), None);
    }

    #[test]
    fn wav_metadata_becomes_the_root_and_loop() {
        let file = wav::Wav {
            sample_rate: 32_000,
            channels: vec![vec![0.25f32; 400]],
            root_key: Some(64.5),
            loops: vec![wav::WavLoop { start: 50, end: 350, kind: WavLoopKind::PingPong }],
        };
        let s = Sample::<f32>::from_wav(&wav::write(&file, wav::WavFormat::Float32)).unwrap();
        assert_eq!((s.sample_rate(), s.root_key(), s.frames()), (32_000.0, 64.5, 400));
        assert_eq!(s.loop_points(), Some(Loop { start: 50, end: 350, mode: LoopMode::PingPong, crossfade: 0 }));
    }

    #[test]
    fn mipmaps_halve_the_rate_and_keep_the_band() {
        // a low tone survives decimation at full level; one above the new Nyquist is removed
        let fs = 48_000.0;
        let tone = |f: f64| (0..8_192).map(|i| (std::f64::consts::TAU * f * i as f64 / fs).sin()).collect::<Vec<f64>>();
        let s = Sample::from_mono(tone(1_000.0), fs).with_mipmaps(3);
        assert_eq!(s.levels.len(), 4);
        assert_eq!(s.levels[1].frames, 4_096);
        let rms = |l: &Level<f64>| {
            let edge = l.frames / 8; // away from the start and end transients
            (l.channels[0][PAD + edge..PAD + l.frames - edge].iter().map(|x| x * x).sum::<f64>() / (l.frames - 2 * edge) as f64).sqrt()
        };
        for level in &s.levels[1..] {
            assert!((rms(level) - 0.5f64.sqrt()).abs() < 0.01, "level keeps a 1 kHz tone: {}", rms(level));
        }
        // aligned: level 1 frame m is the original's frame 2m
        let (l0, l1) = (&s.levels[0].channels[0], &s.levels[1].channels[0]);
        assert!((l1[PAD + 1_500] - l0[PAD + 3_000]).abs() < 0.01);
        let high = Sample::from_mono(tone(15_000.0), fs).with_mipmaps(1);
        assert!(rms(&high.levels[1]) < 1e-3, "15 kHz is above level 1's 12 kHz Nyquist: {}", rms(&high.levels[1]));
    }

    #[test]
    fn zones_select_by_key_and_velocity_and_round_robin() {
        let s = |v: f64| Arc::new(Sample::from_mono(vec![v; 10], 48_000.0));
        let map = SampleMap::new(vec![
            Zone::new(s(1.0)).keys(0..=59).velocities(0..=63),
            Zone::new(s(2.0)).keys(0..=59).velocities(64..=127),
            Zone::new(s(3.0)).keys(60..=127),
            Zone::new(s(4.0)).keys(60..=127),
        ]);
        let first = |key, vel| map.find(key, vel).map(|z| z.sample.frame(0, 0));
        assert_eq!((first(40, 10), first(40, 100)), (Some(1.0), Some(2.0)));
        let turns: Vec<_> = (0..4).map(|_| first(72, 90).unwrap()).collect();
        assert_eq!(turns, vec![3.0, 4.0, 3.0, 4.0]);
        assert!(SampleMap::new(vec![Zone::new(s(1.0)).keys(10..=20)]).find(30, 64).is_none());
    }
}
