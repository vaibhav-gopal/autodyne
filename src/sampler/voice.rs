use std::sync::{Arc, OnceLock};

use super::sample::{Level, PAD};
use super::{LoopMode, SampleMap, Zone};
use crate::envelope::Adsr;
use crate::synth::Voice;
use crate::special::bessel_i0;
use crate::units::*;

/// How a [`SamplerVoice`] reads between recorded frames when playing at another pitch.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum Interpolation {
    /// cheapest; dull highs and audible aliasing when pitched up
    Linear,
    /// 4-point Catmull-Rom: brighter, still some aliasing when pitched up
    Cubic,
    /// Kaiser-windowed sinc (64 taps, up to 128 when pitched up): band-limited, with flat response to 0.82 x Nyquist;
    /// the most expensive
    Sinc,
}

impl Interpolation {
    /// Every method, in parameter order.
    pub const ALL: [Interpolation; 3] = [Interpolation::Linear, Interpolation::Cubic, Interpolation::Sinc];
    /// Display names, in the order of [`ALL`](Self::ALL).
    pub const NAMES: [&'static str; 3] = ["Linear", "Cubic", "Sinc"];
}

/// Zero crossings of the sinc kernel on each side: 64 taps at unity speed, for a transition band
/// of about 0.18 x Nyquist at 90 dB stop-band rejection.
pub(crate) const SINC_ZEROS: usize = 32;
/// Kernel table points per zero crossing (linearly interpolated).
const SINC_RES: usize = 512;
/// The kernel widens up to this factor when pitching up; beyond an octave the sample's mipmaps
/// take over (without them, notes more than an octave up alias).
pub(crate) const MAX_STRETCH: f64 = 2.0;
/// The kernel's -6 dB point relative to Nyquist: its transition band then ends at Nyquist, so the
/// pass band reaches about 0.82 x Nyquist (19.7 kHz at 48 kHz) with nothing folding back.
pub(crate) const SINC_CUTOFF: f64 = 0.91;
const MAX_TAPS: usize = 2 * (SINC_ZEROS * MAX_STRETCH as usize + 2);

/// Kaiser-windowed sinc, sampled from 0 to SINC_ZEROS zero crossings.
fn sinc_table() -> &'static [f64] {
    static TABLE: OnceLock<Vec<f64>> = OnceLock::new();
    TABLE.get_or_init(|| {
        let beta = 9.0;
        let len = SINC_ZEROS * SINC_RES;
        (0..=len + 1)
            .map(|i| {
                let t = i as f64 / SINC_RES as f64;
                if i > len {
                    return 0.0;
                }
                let sinc = if i == 0 { 1.0 } else { (std::f64::consts::PI * t).sin() / (std::f64::consts::PI * t) };
                let r = t / SINC_ZEROS as f64;
                sinc * bessel_i0(beta * (1.0 - r * r).max(0.0).sqrt()) / bessel_i0(beta)
            })
            .collect()
    })
}

/// A [`Voice`] playing a [`SampleMap`]: on each note it picks the zone for the key and velocity,
/// resamples it to the note's pitch, follows the sample's loop, and shapes it with an amplitude
/// envelope, velocity, and the zone's gain and pan.
///
/// Pitched far above the root, it reads the sample's mipmaps (if built) and widens the sinc kernel,
/// so it stays band-limited. Positions are kept in `f64`, exact for samples of any length.
#[derive(Debug, Clone)]
pub struct SamplerVoice<T: Float> {
    sample_rate: f64,
    map: Arc<SampleMap<T>>,
    zone: Option<Zone<T>>,
    interpolation: Interpolation,
    amp_env: Adsr<T>,
    velocity_sensitivity: T,
    tune: T,
    start: T,
    note: u8,
    velocity: T,
    bend: T,
    gains: (T, T),
    /// playback position in frames of the original sample, and direction (+1 / -1)
    pos: f64,
    dir: f64,
    /// has passed through the loop at least once
    wrapped: bool,
    released: bool,
    finished: bool,
    /// frames of the original per output sample
    increment: f64,
    level: usize,
    /// sinc cutoff relative to the level's Nyquist
    cutoff: f64,
}

impl<T: Float> SamplerVoice<T> {
    /// Sinc interpolation; a 1 ms attack, full sustain and a 250 ms release.
    pub fn new(map: Arc<SampleMap<T>>, sample_rate: T) -> Self {
        sinc_table();
        let lit = T::_lit;
        Self {
            sample_rate: sample_rate.to_f64().unwrap_or(48_000.0),
            map,
            zone: None,
            interpolation: Interpolation::Sinc,
            amp_env: Adsr::new(lit(0.001), T::_ZERO, T::_ONE, lit(0.25), sample_rate),
            velocity_sensitivity: T::_ONE,
            tune: T::_ZERO,
            start: T::_ZERO,
            note: 60,
            velocity: T::_ZERO,
            bend: T::_ZERO,
            gains: (T::_ZERO, T::_ZERO),
            pos: 0.0,
            dir: 1.0,
            wrapped: false,
            released: false,
            finished: true,
            increment: 1.0,
            level: 0,
            cutoff: SINC_CUTOFF,
        }
    }
    /// Plays another instrument from the next note on. Dropping the last `Arc` of the old map here
    /// would free it on this thread: keep a reference elsewhere if this runs on the audio thread.
    pub fn set_map(&mut self, map: Arc<SampleMap<T>>) {
        self.map = map;
    }
    /// The instrument played.
    pub fn map(&self) -> &Arc<SampleMap<T>> {
        &self.map
    }
    /// How samples are read between frames.
    pub fn set_interpolation(&mut self, interpolation: Interpolation) {
        self.interpolation = interpolation;
    }
    /// How samples are read between frames.
    pub fn interpolation(&self) -> Interpolation {
        self.interpolation
    }
    /// The amplitude envelope.
    pub fn amp_env(&self) -> &Adsr<T> {
        &self.amp_env
    }
    /// The amplitude envelope, to change its settings.
    pub fn amp_env_mut(&mut self) -> &mut Adsr<T> {
        &mut self.amp_env
    }
    /// 0: every velocity plays at full level .. 1: level proportional to velocity.
    pub fn set_velocity_sensitivity(&mut self, amount: T) {
        self.velocity_sensitivity = amount._clamp(T::_ZERO, T::_ONE);
    }
    /// Velocity sensitivity, 0..1.
    pub fn velocity_sensitivity(&self) -> T {
        self.velocity_sensitivity
    }
    /// Transposition in semitones (fractions for fine tuning).
    pub fn set_tune(&mut self, semitones: T) {
        self.tune = semitones;
        self.update_pitch();
    }
    /// Transposition in semitones.
    pub fn tune(&self) -> T {
        self.tune
    }
    /// Where notes start, as a fraction of the sample (skips attacks or silence).
    pub fn set_start(&mut self, fraction: T) {
        self.start = fraction._clamp(T::_ZERO, T::_ONE);
    }
    /// Start position as a fraction of the sample.
    pub fn start(&self) -> T {
        self.start
    }
    /// The zone playing (the last one, after the note ends).
    pub fn zone(&self) -> Option<&Zone<T>> {
        self.zone.as_ref()
    }
    /// Playback position in frames of the playing sample.
    pub fn position(&self) -> f64 {
        self.pos
    }

    fn update_pitch(&mut self) {
        let Some(zone) = &self.zone else { return };
        let sample = &zone.sample;
        let semitones = self.note as f64 + self.bend.to_f64().unwrap_or(0.0) + self.tune.to_f64().unwrap_or(0.0) + zone.tune_cents / 100.0 - zone.root();
        let ratio = (sample.sample_rate() / self.sample_rate * 2f64.powf(semitones / 12.0)).clamp(1e-6, 1e6);
        self.increment = ratio;
        self.level = if ratio >= 2.0 { (ratio.log2().floor() as usize).min(sample.levels.len() - 1) } else { 0 };
        let speed = ratio / (1u64 << self.level) as f64;
        self.cutoff = SINC_CUTOFF * (1.0 / speed).clamp(1.0 / MAX_STRETCH, 1.0);
    }

    fn update_gains(&mut self) {
        let Some(zone) = &self.zone else { return };
        let s = self.velocity_sensitivity;
        let level = T::_lit(10f64.powf(zone.gain_db / 20.0)) * (T::_ONE - s + s * self.velocity);
        // balance: the far side is turned down, the near side stays
        let pan = zone.pan;
        self.gains = (level * T::_lit((1.0 - pan).min(1.0)), level * T::_lit((1.0 + pan).min(1.0)));
    }

    /// Whether the loop is in effect right now.
    fn looping(&self) -> Option<super::Loop> {
        let l = self.zone.as_ref()?.sample.loop_points()?;
        match l.mode {
            LoopMode::Off => None,
            LoopMode::Sustain if self.released => None,
            _ => Some(l),
        }
    }

    /// One stereo frame at position `pos` of the playing sample.
    fn read(&self, pos: f64, looping: Option<super::Loop>) -> (T, T) {
        let zone = self.zone.as_ref().unwrap();
        let level: &Level<T> = &zone.sample.levels[self.level];
        let scale = (1u64 << self.level) as f64;
        let p = pos / scale;
        // loop points at this level, for reading across the seam of a seamless (uncrossfaded) loop
        let seam = looping.filter(|l| l.crossfade == 0).map(|l| ((l.start as f64 / scale).round() as isize, (l.end as f64 / scale).round() as isize, l.mode));
        let frames = level.frames as isize;
        let wrapped = self.wrapped;
        let index = |k: isize| -> usize {
            let k = match seam {
                Some((s, e, mode)) if e > s && (k >= e || (wrapped && k < s)) => {
                    let len = e - s;
                    if mode == LoopMode::PingPong {
                        let m = (k - s).rem_euclid(2 * len);
                        s + if m > len { 2 * len - m } else { m }
                    } else {
                        s + (k - s).rem_euclid(len)
                    }
                }
                _ => k,
            };
            (k.clamp(-(PAD as isize), frames + PAD as isize - 1) + PAD as isize) as usize
        };
        let left = &level.channels[0];
        let right = level.channels.get(1).unwrap_or(left);
        let base = p.floor();
        let i = base as isize;
        let frac = p - base;
        if frac == 0.0 && self.increment == 1.0 {
            let at = index(i);
            return (left[at], right[at]);
        }
        let mut sum = (0.0, 0.0);
        let mut add = |k: isize, w: f64| {
            let at = index(k);
            sum.0 += w * left[at].to_f64().unwrap_or(0.0);
            sum.1 += w * right[at].to_f64().unwrap_or(0.0);
        };
        match self.interpolation {
            Interpolation::Linear => {
                add(i, 1.0 - frac);
                add(i + 1, frac);
            }
            Interpolation::Cubic => {
                // Catmull-Rom weights for the frames at i-1, i, i+1, i+2
                let (f2, f3) = (frac * frac, frac * frac * frac);
                add(i - 1, -0.5 * f3 + f2 - 0.5 * frac);
                add(i, 1.5 * f3 - 2.5 * f2 + 1.0);
                add(i + 1, -1.5 * f3 + 2.0 * f2 + 0.5 * frac);
                add(i + 2, 0.5 * f3 - 0.5 * f2);
            }
            Interpolation::Sinc => {
                let table = sinc_table();
                let c = self.cutoff;
                let reach = SINC_ZEROS as f64 / c;
                let first = (p - reach).ceil() as isize;
                let last = ((p + reach).floor() as isize).min(first + MAX_TAPS as isize - 1);
                let mut weights = [0.0f64; MAX_TAPS];
                let taps = (last - first + 1) as usize;
                for (w, k) in weights[..taps].iter_mut().zip(first..=last) {
                    let x = (p - k as f64).abs() * c * SINC_RES as f64;
                    let j = (x as usize).min(table.len() - 2);
                    *w = c * (table[j] + (table[j + 1] - table[j]) * (x - j as f64));
                }
                let crosses_seam = seam.is_some_and(|(s, e, _)| last >= e || (wrapped && first < s));
                if crosses_seam {
                    for (&w, k) in weights[..taps].iter().zip(first..=last) {
                        add(k, w);
                    }
                } else {
                    // the usual case: one contiguous run of frames (the padding covers the ends)
                    let at = (first + PAD as isize) as usize;
                    let dot = |data: &[T]| data[at..at + taps].iter().zip(&weights[..taps]).fold(0.0, |acc, (x, w)| acc + w * x.to_f64().unwrap_or(0.0));
                    sum = (dot(left), dot(right));
                }
            }
        }
        (T::_lit(sum.0), T::_lit(sum.1))
    }

    #[inline]
    fn next_frame(&mut self) -> (T, T) {
        if self.finished {
            return (T::_ZERO, T::_ZERO);
        }
        let looping = self.looping();
        let (mut l, mut r) = self.read(self.pos, looping);
        if let Some(lp) = looping.filter(|lp| lp.crossfade > 0 && lp.mode != LoopMode::PingPong) {
            let fade_start = (lp.end - lp.crossfade) as f64;
            if self.pos >= fade_start {
                // equal-power crossfade from the loop's end into the frames leading up to its start
                let g = (self.pos - fade_start) / lp.crossfade as f64 * std::f64::consts::FRAC_PI_2;
                let (into, out) = (T::_lit(g.sin()), T::_lit(g.cos()));
                let (bl, br) = self.read(self.pos - (lp.end - lp.start) as f64, looping);
                l = l * out + bl * into;
                r = r * out + br * into;
            }
        }
        let env = self.amp_env.next_value();
        self.advance(looping);
        (l * self.gains.0 * env, r * self.gains.1 * env)
    }

    fn advance(&mut self, looping: Option<super::Loop>) {
        self.pos += self.dir * self.increment;
        match looping {
            Some(lp) => {
                let (start, end) = (lp.start as f64, lp.end as f64);
                let len = end - start;
                if lp.mode == LoopMode::PingPong {
                    // bounce off the ends (reflecting the overshoot)
                    if self.dir > 0.0 && self.pos >= end {
                        self.pos = 2.0 * end - self.pos;
                        self.dir = -1.0;
                        self.wrapped = true;
                    } else if self.dir < 0.0 && self.pos < start {
                        self.pos = 2.0 * start - self.pos;
                        self.dir = 1.0;
                    }
                    if self.wrapped {
                        self.pos = self.pos.clamp(start, end); // steps longer than the loop
                    }
                } else if self.pos >= end {
                    self.pos = start + (self.pos - start).rem_euclid(len);
                    self.wrapped = true;
                }
            }
            None => {
                let frames = self.zone.as_ref().map_or(0, |z| z.sample.frames()) as f64;
                if self.pos >= frames || self.pos < 0.0 {
                    self.finished = true;
                }
            }
        }
    }
}

impl<T: Float> Voice for SamplerVoice<T> {
    type Sample = T;

    fn note_on(&mut self, note: u8, velocity: T) {
        self.note = note;
        self.velocity = velocity._clamp(T::_ZERO, T::_ONE);
        let vel7 = (self.velocity.to_f64().unwrap_or(0.0) * 127.0).round().clamp(1.0, 127.0) as u8;
        self.zone = self.map.find(note, vel7).cloned();
        self.finished = self.zone.is_none();
        let frames = self.zone.as_ref().map_or(0, |z| z.sample.frames());
        self.pos = (self.start.to_f64().unwrap_or(0.0) * frames as f64).floor().min(frames.saturating_sub(1) as f64);
        self.dir = 1.0;
        self.wrapped = false;
        self.released = false;
        self.update_pitch();
        self.update_gains();
        self.amp_env.note_on();
    }
    fn legato(&mut self, note: u8, _velocity: T) {
        self.note = note;
        self.update_pitch();
    }
    fn note_off(&mut self) {
        self.released = true;
        self.amp_env.note_off();
    }
    fn is_active(&self) -> bool {
        !self.finished && self.amp_env.is_active()
    }
    /// The mid (average) of the stereo output.
    fn render(&mut self, out: &mut [T]) {
        let half = T::_lit(0.5);
        for s in out {
            let (l, r) = self.next_frame();
            *s = (l + r) * half;
        }
    }
    fn render_stereo(&mut self, left: &mut [T], right: &mut [T]) {
        for (l, r) in left.iter_mut().zip(right.iter_mut()) {
            (*l, *r) = self.next_frame();
        }
    }
    fn set_pitch_bend(&mut self, semitones: T) {
        self.bend = semitones;
        self.update_pitch();
    }
    fn reset(&mut self) {
        self.finished = true;
        self.amp_env.reset();
    }
}

#[cfg(test)]
mod tests {
    use super::super::{Sample, SampleMap, Zone};
    use super::*;
    use crate::fft::RealFft;

    const FS: f64 = 48_000.0;

    fn sine(freq: f64, frames: usize, fs: f64) -> Vec<f64> {
        (0..frames).map(|i| (std::f64::consts::TAU * freq * i as f64 / fs).sin()).collect()
    }

    fn voice(sample: Sample<f64>) -> SamplerVoice<f64> {
        let mut v = SamplerVoice::new(Arc::new(SampleMap::single(sample)), FS);
        v.amp_env_mut().set_attack(0.0);
        v
    }

    fn play(v: &mut SamplerVoice<f64>, note: u8, frames: usize) -> Vec<f64> {
        v.note_on(note, 1.0);
        let mut out = vec![0.0; frames];
        v.render(&mut out);
        out
    }

    const N: usize = 16_384;

    /// Frequency of FFT bin `k` (an output tone there needs no analysis window: no leakage).
    fn bin(k: usize) -> f64 {
        k as f64 * FS / N as f64
    }

    /// Level in dB of the strongest component near `freq` (a bin-centered tone), and of the
    /// strongest of everything else.
    fn tone_and_rest(x: &[f64], freq: f64) -> (f64, f64) {
        let n = N;
        let frame = &x[x.len() - n..];
        let mut spectrum = vec![Complex::zero(); n / 2 + 1];
        RealFft::new(n).forward(frame, &mut spectrum);
        let db = |z: &Complex<f64>| 20.0 * (z.norm() * 2.0 / n as f64).max(1e-12).log10();
        let bin = |f: f64| (f * n as f64 / FS).round() as usize;
        let (lo, hi) = (bin(freq) - 4, bin(freq) + 4);
        let tone = spectrum[lo..=hi].iter().map(db).fold(f64::MIN, f64::max);
        let rest = spectrum.iter().enumerate().filter(|(k, _)| (*k < lo || *k > hi) && *k > 3).map(|(_, z)| db(z)).fold(f64::MIN, f64::max);
        (tone, rest)
    }

    #[test]
    fn plays_at_the_root_unchanged_and_transposes() {
        let data = sine(440.0, 48_000, FS);
        let mut v = voice(Sample::from_mono(data.clone(), FS).with_root_key(69.0));
        let out = play(&mut v, 69, 1_000);
        assert_eq!(&out[..], &data[..1_000], "bit-exact at the root");
        // an octave up reads every other frame; a fifth down is 2/3 the speed
        for (note, freq) in [(81, 880.0), (62, 293.66), (76, 659.26)] {
            for interpolation in Interpolation::ALL {
                v.set_interpolation(interpolation);
                let out = play(&mut v, note, 24_000);
                let crossings = out.windows(2).filter(|w| w[0] < 0.0 && w[1] >= 0.0).count() as f64;
                let measured = crossings / 0.5;
                assert!((measured / freq - 1.0).abs() < 0.01, "{interpolation:?} note {note}: {measured} Hz");
            }
        }
    }

    #[test]
    fn sinc_interpolation_is_clean_where_the_others_are_not() {
        // a 5 kHz tone pitched a minor third up at a 44.1 kHz recording rate: imaging and aliasing
        // products land away from the tone
        let fs_sample = 44_100.0;
        let target = bin(2_030); // 5947 Hz
        let data = sine(target / 2f64.powf(3.0 / 12.0), 88_200, fs_sample);
        let mut levels = Vec::new();
        for interpolation in Interpolation::ALL {
            let mut v = voice(Sample::from_mono(data.clone(), fs_sample).with_root_key(60.0));
            v.set_interpolation(interpolation);
            let out = play(&mut v, 63, 40_000);
            let (tone, rest) = tone_and_rest(&out, target);
            assert!(tone.abs() < 0.5, "{interpolation:?}: tone at {tone} dB");
            levels.push(rest - tone);
        }
        assert!(levels[2] < -85.0, "sinc artifacts {} dB", levels[2]);
        assert!(levels[0] > -60.0 && levels[1] > levels[2] + 20.0, "linear {} dB, cubic {} dB", levels[0], levels[1]);
    }

    #[test]
    fn high_notes_stay_band_limited() {
        // tones pushed past Nyquist would alias without band-limiting: the widened sinc kernel
        // (within an octave) and the mipmaps (beyond) remove them instead
        for (freq, semitones, mipmaps) in [(15_000.0, 11, 0), (9_000.0, 19, 3)] {
            let data = sine(freq, 192_000, FS); // long enough not to run out at 3x speed
            let mut v = voice(Sample::from_mono(data, FS).with_root_key(60.0).with_mipmaps(mipmaps));
            let out = play(&mut v, 60 + semitones, 40_000);
            let peak = out[20_000..].iter().fold(0.0f64, |m, x| m.max(x.abs()));
            assert!(peak < 1e-4, "{freq} Hz up {semitones}: {peak}");
        }
        // and a 2 kHz tone 2.5 octaves up (to 11.3 kHz) plays cleanly from the mipmaps
        let target = bin(3_861);
        let data = sine(target / 2f64.powf(2.5), 200_000, FS);
        let mut v = voice(Sample::from_mono(data, FS).with_root_key(60.0).with_mipmaps(3));
        let out = play(&mut v, 90, 30_000);
        let (tone, rest) = tone_and_rest(&out, target);
        assert!(tone.abs() < 0.5 && rest - tone < -70.0, "tone {tone} dB, rest {rest} dB");
    }

    #[test]
    fn loops_forward_ping_pong_and_sustain() {
        // a ramp 0..1000 so positions are visible in the output
        let ramp: Vec<f64> = (0..1_000).map(|i| i as f64 / 1_000.0).collect();
        let mut v = voice(Sample::from_mono(ramp.clone(), FS).with_root_key(60.0).with_loop(200, 300, LoopMode::Forward, 0));
        v.set_interpolation(Interpolation::Linear);
        let out = play(&mut v, 60, 600);
        assert_eq!(out[350], ramp[250], "forward: 350 frames in, back at 250");
        assert!(v.is_active());
        let mut v = voice(Sample::from_mono(ramp.clone(), FS).with_root_key(60.0).with_loop(200, 300, LoopMode::PingPong, 0));
        v.set_interpolation(Interpolation::Linear);
        let out = play(&mut v, 60, 600);
        assert!((out[320] - 0.28).abs() < 1e-9, "ping-pong: bounced at 300, coming back: {}", out[320]);
        assert!((out[430] - 0.23).abs() < 1e-9, "and forward again after 200: {}", out[430]);
        // sustain: loops while held, then plays out to the end and stops
        let mut v = voice(Sample::from_mono(ramp.clone(), FS).with_root_key(60.0).with_loop(200, 300, LoopMode::Sustain, 0));
        v.amp_env_mut().set_release(10.0);
        let held = play(&mut v, 60, 2_000);
        assert!(held[1_999] < 0.3 && v.is_active());
        v.note_off();
        let mut tail = vec![0.0; 2_000];
        v.render(&mut tail);
        assert!(!v.is_active(), "the end of the sample ends the voice");
        assert!(tail.iter().any(|&x| x > 0.9), "played past the loop to the end");
        // without a loop the voice ends with the sample
        let mut v = voice(Sample::from_mono(ramp, FS));
        play(&mut v, 60, 1_001);
        assert!(!v.is_active());
    }

    #[test]
    fn seamless_and_crossfaded_loops_have_no_click() {
        // a 100 Hz sine with a loop of exactly 4 cycles is seamless: sinc reads across the seam
        let data = sine(100.0, 9_600, FS);
        let mut v = voice(Sample::from_mono(data, FS).with_root_key(60.0).with_loop(1_920, 3_840, LoopMode::Forward, 0));
        let out = play(&mut v, 61, 20_000);
        let jump = out.windows(2).skip(100).map(|w| (w[1] - w[0]).abs()).fold(0.0, f64::max);
        let expected = std::f64::consts::TAU * 100.0 * 2f64.powf(1.0 / 12.0) / FS;
        assert!(jump < expected * 1.01, "largest step {jump} vs a sine's {expected}");
        // a loop cut mid-cycle clicks, unless crossfaded
        let data = sine(100.0, 9_600, FS);
        let step = |crossfade| {
            let mut v = voice(Sample::from_mono(data.clone(), FS).with_root_key(60.0).with_loop(2_000, 4_123, LoopMode::Forward, crossfade));
            let out = play(&mut v, 60, 20_000);
            out.windows(2).skip(100).map(|w| (w[1] - w[0]).abs()).fold(0.0, f64::max)
        };
        assert!(step(0) > 0.2, "an uncrossfaded bad loop jumps");
        assert!(step(1_000) < 0.03, "crossfaded: {}", step(1_000));
    }

    #[test]
    fn zones_velocity_pan_and_stereo() {
        let left_only = Arc::new(Sample::new(vec![vec![0.5; 4_800], vec![0.0; 4_800]], FS));
        let map = Arc::new(SampleMap::new(vec![
            Zone::new(left_only.clone()).keys(0..=59).gain_db(-6.0206),
            Zone::new(Arc::new(Sample::from_mono(vec![0.5; 4_800], FS))).keys(60..=127).pan(0.5),
        ]));
        let mut v = SamplerVoice::new(map, FS);
        v.amp_env_mut().set_attack(0.0);
        let (mut l, mut r) = (vec![0.0; 100], vec![0.0; 100]);
        v.note_on(40, 1.0);
        v.render_stereo(&mut l, &mut r);
        assert!((l[50] - 0.25).abs() < 1e-3 && r[50] == 0.0, "stereo sample, -6 dB: {} {}", l[50], r[50]);
        v.note_on(70, 0.5);
        v.render_stereo(&mut l, &mut r);
        assert!((l[50] - 0.125).abs() < 1e-3 && (r[50] - 0.25).abs() < 1e-3, "half velocity, panned right: {} {}", l[50], r[50]);
        v.set_velocity_sensitivity(0.0);
        v.note_on(70, 0.1);
        v.render_stereo(&mut l, &mut r);
        assert!((r[50] - 0.5).abs() < 1e-3, "no velocity sensitivity");
    }
}
