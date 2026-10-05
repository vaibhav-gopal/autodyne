//! Playable synthesis: MIDI messages, a voice interface, polyphony, and a ready-made voice.
//!
//! - [`MidiMessage`]: parse / produce MIDI channel messages; [`midi_to_hz`] for note pitches.
//! - [`Voice`]: what one playable voice does (note on / off / legato, render a block, pitch bend,
//!   pressure and timbre).
//! - [`Poly`]: a pool of voices driven by MIDI: allocation with voice stealing, sustain pedal,
//!   pitch bend, all-notes-off, sample-accurate event timing within a block, mono and legato modes
//!   with last-note priority, and MPE (per-note pitch bend, pressure and timbre on member channels)
//!   plus per-note expression by key (CLAP note expressions, polyphonic aftertouch).
//! - [`SynthVoice`]: a subtractive voice (up to 7 detuned unison oscillators -> state-variable or
//!   ladder filter swept by its own envelope, pressure and timbre -> amplitude ADSR with velocity),
//!   with glide.
//! - [`FmVoice`]: a 4-operator phase-modulation voice (8 algorithms, ratios, detune, per-operator
//!   envelopes, feedback).
//!
//! Everything is allocated when built; handling events and rendering never allocate, so both can run
//! inside an audio callback.
//!
//! tend: Audio / synth

use crate::alloc_prelude::*;
mod params;
mod fm;
mod midi;
mod voice;

pub use fm::*;
pub use midi::*;
pub use voice::*;

use crate::signal::Source;
use crate::units::*;

/// One playable voice.
pub trait Voice {
    /// The sample type.
    type Sample: Float;

    /// Starts (or retriggers) `note` at `velocity` in [0, 1].
    fn note_on(&mut self, note: u8, velocity: Self::Sample);
    /// Releases the current note (the voice keeps sounding through its release).
    fn note_off(&mut self);
    /// Whether the voice is producing sound (including its release tail).
    fn is_active(&self) -> bool;
    /// Overwrites `out` with the voice's next samples (silence when inactive).
    fn render(&mut self, out: &mut [Self::Sample]);
    /// Overwrites `left` and `right` (the same length) with the voice's next samples in stereo.
    /// By default the mono render goes to both sides.
    fn render_stereo(&mut self, left: &mut [Self::Sample], right: &mut [Self::Sample]) {
        self.render(left);
        right.copy_from_slice(left);
    }
    /// Detunes the voice by this many semitones (pitch bend). No-op by default.
    fn set_pitch_bend(&mut self, _semitones: Self::Sample) {}
    /// Moves to `note` without retriggering, gliding if the voice glides (legato playing).
    /// Retriggers by default.
    fn legato(&mut self, note: u8, velocity: Self::Sample) {
        self.note_on(note, velocity);
    }
    /// Pressure (aftertouch) in [0, 1]. No-op by default.
    fn set_pressure(&mut self, _pressure: Self::Sample) {}
    /// Timbre (MPE "slide", brightness) in [0, 1], 0.5 neutral. No-op by default.
    fn set_timbre(&mut self, _timbre: Self::Sample) {}
    /// Silences immediately.
    fn reset(&mut self);
}

/// A MIDI message to apply `offset` samples into the next rendered block.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct TimedEvent {
    /// samples into the block
    pub offset: usize,
    /// the message
    pub message: MidiMessage,
}

/// How [`Poly`] assigns notes to voices.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum VoiceMode {
    /// every note gets its own voice
    Poly,
    /// one voice; every new note retriggers it (last-note priority: releasing a key returns to the
    /// one still held before it)
    Mono,
    /// one voice; notes played while another is held glide over without retriggering
    Legato,
}

impl VoiceMode {
    /// Every mode, in parameter order.
    pub const ALL: [VoiceMode; 3] = [VoiceMode::Poly, VoiceMode::Mono, VoiceMode::Legato];
    /// Display names, in the order of [`ALL`](Self::ALL).
    pub const NAMES: [&'static str; 3] = ["Poly", "Mono", "Legato"];
}

/// The MPE lower zone's master channel (MIDI channel 1); member channels are 2-16.
pub const MPE_MASTER_CHANNEL: u8 = 0;
/// Most keys the mono and legato modes remember.
const HELD_KEYS: usize = 16;

#[derive(Debug, Clone, Copy)]
struct Slot<T> {
    /// the note this voice was started for, until it is released (None while releasing or idle)
    note: Option<u8>,
    channel: u8,
    /// note-off arrived while the sustain pedal was down: release when the pedal comes up
    sustained: bool,
    /// when the note started (for stealing the oldest)
    started: u64,
    /// per-note tuning in semitones (CLAP note expressions), on top of the bends
    note_bend: T,
}

impl<T: Float> Default for Slot<T> {
    fn default() -> Self {
        Self { note: None, channel: 0, sustained: false, started: 0, note_bend: T::_ZERO }
    }
}

/// A pool of voices played by MIDI.
///
/// Voice allocation: a repeated note retriggers the voice already playing it; otherwise a free voice
/// is used; otherwise the oldest *releasing* voice is stolen, and only then the oldest held one.
/// Responds to note on/off, the sustain pedal (CC 64), all notes off (CC 123), all sound off (CC 120),
/// pitch bend, channel and polyphonic aftertouch, and timbre (CC 74), on every channel (omni).
///
/// With MPE on (lower zone), each note plays on its own member channel (2-16) and that channel's
/// pitch bend (over [`member_bend_range`](Self::set_member_bend_range), 48 semitones by default),
/// pressure and timbre shape that note alone; the master channel (1) still bends and controls every
/// voice.
#[derive(Debug, Clone)]
pub struct Poly<V: Voice> {
    voices: Vec<V>,
    slots: Vec<Slot<V::Sample>>,
    clock: u64,
    sustain: bool,
    mode: VoiceMode,
    /// mono / legato: keys held, oldest first
    held: [(u8, V::Sample); HELD_KEYS],
    held_len: usize,
    mpe: bool,
    bend_range: V::Sample,
    member_bend_range: V::Sample,
    /// semitones from the master (or omni) pitch bend
    master_bend: V::Sample,
    /// per-channel state for MPE members (index = channel)
    channel_bend: [V::Sample; 16],
    channel_pressure: [V::Sample; 16],
    channel_timbre: [V::Sample; 16],
    /// non-MPE: the pressure and timbre every voice gets
    pressure: V::Sample,
    timbre: V::Sample,
    max_block: usize,
    scratch: Vec<V::Sample>,
    scratch_right: Vec<V::Sample>,
}

impl<V: Voice> Poly<V> {
    /// `count` voices built by `make(index)`; blocks longer than `max_block` are rendered in pieces.
    /// Panics if `count` or `max_block` is 0.
    pub fn new(count: usize, max_block: usize, make: impl FnMut(usize) -> V) -> Self {
        assert!(count > 0 && max_block > 0, "need at least one voice and a positive block size");
        let zero = V::Sample::_ZERO;
        let neutral = V::Sample::_lit(0.5);
        Self {
            voices: (0..count).map(make).collect(),
            slots: vec![Slot::default(); count],
            clock: 0,
            sustain: false,
            mode: VoiceMode::Poly,
            held: [(0, zero); HELD_KEYS],
            held_len: 0,
            mpe: false,
            bend_range: V::Sample::_lit(2.0),
            member_bend_range: V::Sample::_lit(48.0),
            master_bend: zero,
            channel_bend: [zero; 16],
            channel_pressure: [zero; 16],
            channel_timbre: [neutral; 16],
            pressure: zero,
            timbre: neutral,
            max_block,
            scratch: vec![zero; max_block],
            scratch_right: vec![zero; max_block],
        }
    }
    /// Pitch bend range in semitones for a full bend (default 2).
    pub fn set_bend_range(&mut self, semitones: V::Sample) {
        self.bend_range = semitones;
    }
    /// Pitch bend range in semitones.
    pub fn bend_range(&self) -> V::Sample {
        self.bend_range
    }
    /// Pitch bend range of MPE member channels (default 48 semitones, the MPE standard).
    pub fn set_member_bend_range(&mut self, semitones: V::Sample) {
        self.member_bend_range = semitones;
    }
    /// Turns MPE (lower zone) on or off. Releases everything first.
    pub fn set_mpe(&mut self, on: bool) {
        if on != self.mpe {
            self.all_notes_off();
            self.mpe = on;
            self.channel_bend = [V::Sample::_ZERO; 16];
        }
    }
    /// Whether MPE is on.
    pub fn mpe(&self) -> bool {
        self.mpe
    }
    /// Switches between poly, mono and legato. Releases everything first.
    pub fn set_mode(&mut self, mode: VoiceMode) {
        if mode != self.mode {
            self.all_notes_off();
            self.mode = mode;
        }
    }
    /// The voice mode.
    pub fn mode(&self) -> VoiceMode {
        self.mode
    }
    /// All voices.
    pub fn voices(&self) -> &[V] {
        &self.voices
    }
    /// All voices, e.g. to change a sound parameter on every one.
    pub fn voices_mut(&mut self) -> &mut [V] {
        &mut self.voices
    }
    /// Number of voices currently sounding.
    pub fn active_voices(&self) -> usize {
        self.voices.iter().filter(|v| v.is_active()).count()
    }

    /// Whether `channel` is an MPE member channel (per-note controls) right now.
    fn is_member(&self, channel: u8) -> bool {
        self.mpe && channel != MPE_MASTER_CHANNEL
    }
    /// Total bend of a voice: master + its member channel + its own tuning.
    fn bend_of(&self, slot: &Slot<V::Sample>) -> V::Sample {
        let member = if self.is_member(slot.channel) { self.channel_bend[slot.channel as usize & 15] } else { V::Sample::_ZERO };
        self.master_bend + member + slot.note_bend
    }
    /// Gives voice `i` its current bend, pressure and timbre.
    fn refresh(&mut self, i: usize) {
        let slot = self.slots[i];
        let ch = slot.channel as usize & 15;
        let (pressure, timbre) = if self.is_member(slot.channel) { (self.channel_pressure[ch], self.channel_timbre[ch]) } else { (self.pressure, self.timbre) };
        let bend = self.bend_of(&slot);
        let voice = &mut self.voices[i];
        voice.set_pitch_bend(bend);
        voice.set_pressure(pressure);
        voice.set_timbre(timbre);
    }

    /// Note on, on channel 0 (see [`note_on_channel`](Self::note_on_channel)).
    pub fn note_on(&mut self, note: u8, velocity: V::Sample) {
        self.note_on_channel(0, note, velocity);
    }

    /// Note on, on `channel` (which matters with MPE: the note follows that channel's controls).
    pub fn note_on_channel(&mut self, channel: u8, note: u8, velocity: V::Sample) {
        self.clock += 1;
        if self.mode != VoiceMode::Poly {
            self.mono_note_on(channel, note, velocity);
            return;
        }
        let mpe = self.mpe;
        let same_note = |s: &Slot<V::Sample>| s.note == Some(note) && (!mpe || s.channel == channel);
        let oldest = |slots: &[Slot<V::Sample>], pick: &dyn Fn(usize) -> bool| {
            (0..slots.len()).filter(|&i| pick(i)).min_by_key(|&i| slots[i].started)
        };
        let index = self
            .slots
            .iter()
            .position(same_note)
            .or_else(|| (0..self.voices.len()).find(|&i| !self.voices[i].is_active()))
            .or_else(|| oldest(&self.slots, &|i| self.slots[i].note.is_none()))
            .or_else(|| oldest(&self.slots, &|_| true))
            .expect("the pool has at least one voice");
        self.slots[index] = Slot { note: Some(note), channel, sustained: false, started: self.clock, note_bend: V::Sample::_ZERO };
        self.refresh(index); // the note starts with its channel's current bend, pressure, timbre
        self.voices[index].note_on(note, velocity);
    }

    /// Note off, on channel 0 (see [`note_off_channel`](Self::note_off_channel)).
    pub fn note_off(&mut self, note: u8) {
        self.note_off_channel(0, note);
    }

    /// Note off, on `channel` (with MPE only the note on that channel is released).
    pub fn note_off_channel(&mut self, channel: u8, note: u8) {
        if self.mode != VoiceMode::Poly {
            self.mono_note_off(note);
            return;
        }
        let mpe = self.mpe;
        for (voice, slot) in self.voices.iter_mut().zip(&mut self.slots) {
            if slot.note == Some(note) && (!mpe || slot.channel == channel) && !slot.sustained {
                if self.sustain {
                    slot.sustained = true;
                } else {
                    voice.note_off();
                    slot.note = None;
                }
            }
        }
    }

    fn mono_note_on(&mut self, channel: u8, note: u8, velocity: V::Sample) {
        self.forget_key(note);
        let legato = self.mode == VoiceMode::Legato && self.held_len > 0 && self.voices[0].is_active();
        if self.held_len == HELD_KEYS {
            self.held.copy_within(1.., 0); // full: forget the oldest key
            self.held_len -= 1;
        }
        self.held[self.held_len] = (note, velocity);
        self.held_len += 1;
        self.slots[0] = Slot { note: Some(note), channel, sustained: false, started: self.clock, note_bend: V::Sample::_ZERO };
        self.refresh(0);
        if legato {
            self.voices[0].legato(note, velocity);
        } else {
            self.voices[0].note_on(note, velocity);
        }
    }

    fn mono_note_off(&mut self, note: u8) {
        let sounding = self.held_len > 0 && self.held[self.held_len - 1].0 == note;
        self.forget_key(note);
        if !sounding {
            return;
        }
        if self.held_len > 0 {
            // back to the key still held before it
            let (previous, velocity) = self.held[self.held_len - 1];
            self.slots[0].note = Some(previous);
            if self.mode == VoiceMode::Legato {
                self.voices[0].legato(previous, velocity);
            } else {
                self.voices[0].note_on(previous, velocity);
            }
        } else if self.sustain {
            self.slots[0].sustained = true;
        } else {
            self.voices[0].note_off();
            self.slots[0].note = None;
        }
    }

    fn forget_key(&mut self, note: u8) {
        if let Some(i) = self.held[..self.held_len].iter().position(|&(n, _)| n == note) {
            self.held.copy_within(i + 1..self.held_len, i);
            self.held_len -= 1;
        }
    }

    /// Sustain pedal: while down, note-offs are held until it comes back up.
    pub fn set_sustain(&mut self, down: bool) {
        self.sustain = down;
        if !down {
            for (voice, slot) in self.voices.iter_mut().zip(&mut self.slots) {
                if slot.sustained {
                    voice.note_off();
                    slot.note = None;
                    slot.sustained = false;
                }
            }
        }
    }

    /// `amount` in [-1, 1]; scaled by the bend range. Bends every voice.
    pub fn set_pitch_bend(&mut self, amount: V::Sample) {
        self.master_bend = amount._clamp(-V::Sample::_ONE, V::Sample::_ONE) * self.bend_range;
        for i in 0..self.voices.len() {
            self.refresh(i);
        }
    }

    /// Pitch bend received on `channel`: a member channel bends only its notes (MPE), any other
    /// channel bends every voice.
    pub fn set_channel_bend(&mut self, channel: u8, amount: V::Sample) {
        if !self.is_member(channel) {
            self.set_pitch_bend(amount);
            return;
        }
        let amount = amount._clamp(-V::Sample::_ONE, V::Sample::_ONE);
        self.channel_bend[channel as usize & 15] = amount * self.member_bend_range;
        self.refresh_channel(channel);
    }

    /// Channel pressure in [0, 1]: a member channel's notes only (MPE), otherwise every voice.
    pub fn set_channel_pressure(&mut self, channel: u8, pressure: V::Sample) {
        let p = pressure._clamp(V::Sample::_ZERO, V::Sample::_ONE);
        if self.is_member(channel) {
            self.channel_pressure[channel as usize & 15] = p;
            self.refresh_channel(channel);
        } else {
            self.pressure = p;
            for i in 0..self.voices.len() {
                self.refresh(i);
            }
        }
    }

    /// Timbre in [0, 1] (CC 74, 0.5 neutral): a member channel's notes only (MPE), otherwise every voice.
    pub fn set_channel_timbre(&mut self, channel: u8, timbre: V::Sample) {
        let t = timbre._clamp(V::Sample::_ZERO, V::Sample::_ONE);
        if self.is_member(channel) {
            self.channel_timbre[channel as usize & 15] = t;
            self.refresh_channel(channel);
        } else {
            self.timbre = t;
            for i in 0..self.voices.len() {
                self.refresh(i);
            }
        }
    }

    fn refresh_channel(&mut self, channel: u8) {
        for i in 0..self.voices.len() {
            if self.slots[i].channel == channel && self.slots[i].note.is_some() {
                self.refresh(i);
            }
        }
    }

    /// Pressure for the voices playing `note` (polyphonic aftertouch, CLAP note expressions).
    pub fn set_note_pressure(&mut self, note: u8, pressure: V::Sample) {
        let p = pressure._clamp(V::Sample::_ZERO, V::Sample::_ONE);
        for (voice, slot) in self.voices.iter_mut().zip(&self.slots) {
            if slot.note == Some(note) {
                voice.set_pressure(p);
            }
        }
    }

    /// Timbre for the voices playing `note` (CLAP brightness expression).
    pub fn set_note_timbre(&mut self, note: u8, timbre: V::Sample) {
        let t = timbre._clamp(V::Sample::_ZERO, V::Sample::_ONE);
        for (voice, slot) in self.voices.iter_mut().zip(&self.slots) {
            if slot.note == Some(note) {
                voice.set_timbre(t);
            }
        }
    }

    /// Per-note tuning in semitones for the voices playing `note` (CLAP tuning expression), on top
    /// of the pitch bends.
    pub fn set_note_tuning(&mut self, note: u8, semitones: V::Sample) {
        for i in 0..self.voices.len() {
            if self.slots[i].note == Some(note) {
                self.slots[i].note_bend = semitones;
                self.refresh(i);
            }
        }
    }

    /// Releases every note (they fade out through their release).
    pub fn all_notes_off(&mut self) {
        self.sustain = false;
        self.held_len = 0;
        for (voice, slot) in self.voices.iter_mut().zip(&mut self.slots) {
            if voice.is_active() {
                voice.note_off();
            }
            *slot = Slot { started: slot.started, ..Slot::default() };
        }
    }

    /// Silences everything immediately.
    pub fn reset(&mut self) {
        self.sustain = false;
        self.held_len = 0;
        self.voices.iter_mut().for_each(Voice::reset);
        self.slots.iter_mut().for_each(|s| *s = Slot::default());
    }

    /// Applies one MIDI message.
    pub fn handle(&mut self, message: MidiMessage) {
        let seven_bit = |v: u8| V::Sample::_lit(v as f64 / 127.0);
        match message {
            MidiMessage::NoteOn { channel, note, velocity } => self.note_on_channel(channel, note, seven_bit(velocity)),
            MidiMessage::NoteOff { channel, note, .. } => self.note_off_channel(channel, note),
            MidiMessage::ControlChange { controller: cc::SUSTAIN, value, .. } => self.set_sustain(value >= 64),
            MidiMessage::ControlChange { controller: cc::ALL_NOTES_OFF, .. } => self.all_notes_off(),
            MidiMessage::ControlChange { controller: cc::ALL_SOUND_OFF, .. } => self.reset(),
            MidiMessage::ControlChange { channel, controller: cc::TIMBRE, value } => self.set_channel_timbre(channel, seven_bit(value)),
            MidiMessage::ControlChange { .. } => {}
            MidiMessage::PitchBend { channel, value } => self.set_channel_bend(channel, V::Sample::_lit(value as f64 / 8192.0)),
            MidiMessage::ChannelPressure { channel, value } => self.set_channel_pressure(channel, seven_bit(value)),
            MidiMessage::PolyPressure { note, value, .. } => self.set_note_pressure(note, seven_bit(value)),
        }
    }

    /// Overwrites `out` with the sum of all active voices.
    pub fn render(&mut self, out: &mut [V::Sample]) {
        for chunk in out.chunks_mut(self.max_block) {
            chunk.iter_mut().for_each(|s| *s = V::Sample::_ZERO);
            let scratch = &mut self.scratch[..chunk.len()];
            for voice in self.voices.iter_mut().filter(|v| v.is_active()) {
                voice.render(scratch);
                for (o, &s) in chunk.iter_mut().zip(scratch.iter()) {
                    *o = *o + s;
                }
            }
        }
    }

    /// Renders `out`, applying each event at its sample offset (events must be sorted by offset;
    /// offsets past the end apply at the end of the block).
    pub fn render_events(&mut self, out: &mut [V::Sample], events: &[TimedEvent]) {
        let mut pos = 0;
        for event in events {
            let at = event.offset.clamp(pos, out.len());
            self.render(&mut out[pos..at]);
            self.handle(event.message);
            pos = at;
        }
        self.render(&mut out[pos..]);
    }

    /// Overwrites `left` and `right` with the stereo sum of all active voices
    /// ([`Voice::render_stereo`]). Panics if their lengths differ.
    pub fn render_stereo(&mut self, left: &mut [V::Sample], right: &mut [V::Sample]) {
        assert_eq!(left.len(), right.len(), "left and right must be the same length");
        for (l_chunk, r_chunk) in left.chunks_mut(self.max_block).zip(right.chunks_mut(self.max_block)) {
            l_chunk.iter_mut().chain(r_chunk.iter_mut()).for_each(|s| *s = V::Sample::_ZERO);
            let (sl, sr) = (&mut self.scratch[..l_chunk.len()], &mut self.scratch_right[..l_chunk.len()]);
            for voice in self.voices.iter_mut().filter(|v| v.is_active()) {
                voice.render_stereo(sl, sr);
                for ((l, r), (&a, &b)) in l_chunk.iter_mut().zip(r_chunk.iter_mut()).zip(sl.iter().zip(sr.iter())) {
                    *l = *l + a;
                    *r = *r + b;
                }
            }
        }
    }

    /// [`render_stereo`](Self::render_stereo) with events at their sample offsets, as in
    /// [`render_events`](Self::render_events).
    pub fn render_stereo_events(&mut self, left: &mut [V::Sample], right: &mut [V::Sample], events: &[TimedEvent]) {
        let mut pos = 0;
        for event in events {
            let at = event.offset.clamp(pos, left.len());
            self.render_stereo(&mut left[pos..at], &mut right[pos..at]);
            self.handle(event.message);
            pos = at;
        }
        self.render_stereo(&mut left[pos..], &mut right[pos..]);
    }
}

/// A polyphonic synth is a signal source (it generates rather than processes).
impl<V: Voice> Source for Poly<V> {
    type Sample = V::Sample;
    fn next_sample(&mut self) -> V::Sample {
        let mut one = [V::Sample::_ZERO];
        self.render(&mut one);
        one[0]
    }
    fn fill(&mut self, out: &mut [V::Sample]) {
        self.render(out);
    }
}
#[cfg(test)]
mod tests {
    use super::*;
    use crate::params::Parameterized;

    /// A voice that outputs a constant equal to its velocity while held, and is instantly silent
    /// when released: easy to reason about.
    #[derive(Debug, Clone, Default)]
    struct Dc {
        note: Option<u8>,
        level: f64,
        bend: f64,
        pressure: f64,
        timbre: f64,
        /// how the last note arrived: true = legato (no retrigger)
        legato: bool,
    }

    impl Voice for Dc {
        type Sample = f64;
        fn note_on(&mut self, note: u8, velocity: f64) {
            self.note = Some(note);
            self.level = velocity;
            self.legato = false;
        }
        fn legato(&mut self, note: u8, _velocity: f64) {
            self.note = Some(note);
            self.legato = true;
        }
        fn set_pressure(&mut self, pressure: f64) {
            self.pressure = pressure;
        }
        fn set_timbre(&mut self, timbre: f64) {
            self.timbre = timbre;
        }
        fn note_off(&mut self) {
            self.note = None;
        }
        fn is_active(&self) -> bool {
            self.note.is_some()
        }
        fn render(&mut self, out: &mut [f64]) {
            let v = if self.note.is_some() { self.level } else { 0.0 };
            out.iter_mut().for_each(|s| *s = v);
        }
        fn set_pitch_bend(&mut self, semitones: f64) {
            self.bend = semitones;
        }
        fn reset(&mut self) {
            self.note = None;
        }
    }

    fn notes(p: &Poly<Dc>) -> Vec<Option<u8>> {
        p.voices().iter().map(|v| v.note).collect()
    }

    #[test]
    fn allocation_and_stealing() {
        let mut p = Poly::new(2, 64, |_| Dc::default());
        p.note_on(60, 1.0);
        p.note_on(64, 1.0);
        assert_eq!(notes(&p), [Some(60), Some(64)]);
        p.note_on(67, 1.0); // no free voice: steals the oldest (60)
        assert_eq!(notes(&p), [Some(67), Some(64)]);
        p.note_on(64, 0.5); // same note retriggers its own voice
        assert_eq!(notes(&p), [Some(67), Some(64)]);
        assert_eq!(p.voices()[1].level, 0.5);
        p.note_off(67);
        assert_eq!(p.active_voices(), 1);
        p.note_on(72, 1.0); // uses the freed voice
        assert_eq!(notes(&p), [Some(72), Some(64)]);
    }

    #[test]
    fn sustain_pedal_holds_releases() {
        let mut p = Poly::new(4, 64, |_| Dc::default());
        p.handle(MidiMessage::NoteOn { channel: 0, note: 60, velocity: 127 });
        p.handle(MidiMessage::ControlChange { channel: 0, controller: cc::SUSTAIN, value: 127 });
        p.handle(MidiMessage::NoteOff { channel: 0, note: 60, velocity: 0 });
        assert_eq!(p.active_voices(), 1, "held by the pedal");
        p.handle(MidiMessage::ControlChange { channel: 0, controller: cc::SUSTAIN, value: 0 });
        assert_eq!(p.active_voices(), 0, "released when the pedal comes up");
    }

    #[test]
    fn mixing_bend_and_all_notes_off() {
        let mut p = Poly::new(4, 3, |_| Dc::default());
        p.handle(MidiMessage::NoteOn { channel: 0, note: 60, velocity: 127 });
        p.handle(MidiMessage::NoteOn { channel: 5, note: 67, velocity: 127 }); // omni
        let mut out = [0.0; 8]; // longer than max_block: rendered in pieces
        p.render(&mut out);
        assert!(out.iter().all(|&s| (s - 2.0).abs() < 1e-12), "two voices summed: {out:?}");
        p.handle(MidiMessage::PitchBend { channel: 0, value: 8191 });
        assert!((p.voices()[0].bend - 2.0).abs() < 1e-3, "full bend = 2 semitones");
        p.handle(MidiMessage::ControlChange { channel: 0, controller: cc::ALL_NOTES_OFF, value: 0 });
        assert_eq!(p.active_voices(), 0);
    }

    #[test]
    fn stereo_rendering_sums_voices_per_side() {
        // mono voices land on both sides; the stereo sum equals the mono one
        let mut poly = Poly::new(4, 3, |_| Dc::default());
        poly.note_on(60, 0.25);
        poly.note_on(64, 0.5);
        let (mut l, mut r, mut m) = (vec![0.0; 7], vec![0.0; 7], vec![0.0; 7]);
        poly.render_stereo(&mut l, &mut r);
        poly.render(&mut m);
        assert_eq!((l.clone(), r.clone()), (m.clone(), m));
        let events = [TimedEvent { offset: 2, message: MidiMessage::NoteOff { channel: 0, note: 64, velocity: 0 } }];
        poly.render_stereo_events(&mut l, &mut r, &events);
        assert_eq!((l[1], l[2], r[6]), (0.75, 0.25, 0.25));
    }
    #[test]
    fn events_land_on_their_sample() {
        let mut p = Poly::new(2, 16, |_| Dc::default());
        let mut out = [0.0; 10];
        let on = |offset, note| TimedEvent { offset, message: MidiMessage::NoteOn { channel: 0, note, velocity: 127 } };
        let off = |offset| TimedEvent { offset, message: MidiMessage::NoteOff { channel: 0, note: 60, velocity: 0 } };
        p.render_events(&mut out, &[on(3, 60), on(5, 62), off(8)]);
        assert_eq!(out, [0.0, 0.0, 0.0, 1.0, 1.0, 2.0, 2.0, 2.0, 1.0, 1.0]);
    }

    fn on(channel: u8, note: u8) -> MidiMessage {
        MidiMessage::NoteOn { channel, note, velocity: 127 }
    }
    fn off(channel: u8, note: u8) -> MidiMessage {
        MidiMessage::NoteOff { channel, note, velocity: 0 }
    }
    /// The voice playing `note` (on `channel`).
    fn voice_of(p: &Poly<Dc>, channel: u8, note: u8) -> &Dc {
        let i = (0..p.voices().len()).find(|&i| p.slots[i].note == Some(note) && p.slots[i].channel == channel).unwrap();
        &p.voices()[i]
    }

    #[test]
    fn mpe_member_channels_shape_their_own_notes() {
        let mut p = Poly::new(4, 64, |_| Dc::default());
        p.set_mpe(true);
        p.handle(on(1, 60));
        p.handle(on(2, 64));
        p.handle(on(3, 60)); // the same key twice, on two member channels
        p.handle(MidiMessage::PitchBend { channel: 1, value: 8191 });
        p.handle(MidiMessage::ChannelPressure { channel: 2, value: 127 });
        p.handle(MidiMessage::ControlChange { channel: 3, controller: cc::TIMBRE, value: 0 });
        let near = |a: f64, b: f64| (a - b).abs() < 1e-2;
        assert!(near(voice_of(&p, 1, 60).bend, 48.0), "a member bends its note over the member range");
        assert_eq!(voice_of(&p, 2, 64).bend, 0.0);
        assert_eq!((voice_of(&p, 2, 64).pressure, voice_of(&p, 1, 60).pressure), (1.0, 0.0));
        assert_eq!((voice_of(&p, 3, 60).timbre, voice_of(&p, 1, 60).timbre), (0.0, 0.5));
        // the master channel bends everything, on top of the member bends
        p.handle(MidiMessage::PitchBend { channel: MPE_MASTER_CHANNEL, value: 4096 });
        assert!(near(voice_of(&p, 1, 60).bend, 49.0) && near(voice_of(&p, 2, 64).bend, 1.0));
        // a note-off releases only the note on its own channel
        p.handle(off(3, 60));
        assert_eq!(p.active_voices(), 2);
        assert!(voice_of(&p, 1, 60).is_active());
        // a new note on a member channel starts with that channel's current controls
        p.handle(on(2, 67));
        assert_eq!(voice_of(&p, 2, 67).pressure, 1.0);
    }

    #[test]
    fn without_mpe_controls_reach_every_voice_and_keys_by_note() {
        let mut p = Poly::new(4, 64, |_| Dc::default());
        p.handle(on(1, 60));
        p.handle(on(2, 64));
        p.handle(MidiMessage::ChannelPressure { channel: 5, value: 127 });
        assert!(p.voices().iter().filter(|v| v.is_active()).all(|v| v.pressure == 1.0), "omni");
        p.handle(MidiMessage::PolyPressure { channel: 0, note: 64, value: 0 });
        assert_eq!((voice_of(&p, 1, 60).pressure, voice_of(&p, 2, 64).pressure), (1.0, 0.0), "per key");
        p.set_note_tuning(60, 0.25); // a CLAP per-note tuning expression
        p.set_pitch_bend(0.5);
        assert!((voice_of(&p, 1, 60).bend - 1.25).abs() < 1e-12 && (voice_of(&p, 2, 64).bend - 1.0).abs() < 1e-12);
    }

    #[test]
    fn mono_and_legato_follow_the_last_held_key() {
        for mode in [VoiceMode::Mono, VoiceMode::Legato] {
            let mut p = Poly::new(4, 64, |_| Dc::default());
            p.set_mode(mode);
            p.note_on(60, 1.0);
            assert!(!p.voices()[0].legato, "the first key always triggers");
            p.note_on(64, 1.0);
            p.note_on(67, 1.0);
            assert_eq!((p.active_voices(), p.voices()[0].note), (1, Some(67)), "{mode:?}: one voice, newest key");
            assert_eq!(p.voices()[0].legato, mode == VoiceMode::Legato, "{mode:?}: legato only glides over");
            p.note_off(64); // not sounding: just forgotten
            assert_eq!(p.voices()[0].note, Some(67));
            p.note_off(67); // back to the key still held
            assert_eq!(p.voices()[0].note, Some(60));
            p.set_sustain(true);
            p.note_off(60);
            assert_eq!(p.active_voices(), 1, "{mode:?}: the pedal holds the last note");
            p.set_sustain(false);
            assert_eq!(p.active_voices(), 0);
        }
    }

    #[test]
    fn mode_and_mpe_are_parameters() {
        let mut p = Poly::new(2, 64, |_| SynthVoice::<f32>::new(48_000.0));
        let voice_params = p.voices()[0].param_count();
        assert_eq!(p.param_count(), voice_params + 3);
        p.set_param_by_id("voice_mode", 2.0).unwrap();
        p.set_param_by_id("bend_range", 12.0).unwrap();
        p.set_param_by_id("mpe", 1.0).unwrap();
        assert_eq!((p.mode(), p.bend_range(), p.mpe()), (VoiceMode::Legato, 12.0, true));
        assert_eq!(p.param_group(voice_params), Some("Polyphony"));
        assert_eq!(p.param_info(voice_params).unwrap().format(2.0), "Legato");
    }
}