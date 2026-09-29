//! Playable synthesis: MIDI messages, a voice interface, polyphony, and a ready-made voice.
//!
//! - [`MidiMessage`]: parse / produce MIDI channel messages; [`midi_to_hz`] for note pitches.
//! - [`Voice`]: what one playable voice does (note on / off, render a block, pitch bend).
//! - [`Poly`]: a pool of voices driven by MIDI: allocation with voice stealing, sustain pedal,
//!   pitch bend, all-notes-off, and sample-accurate event timing within a block.
//! - [`SynthVoice`]: a subtractive voice (band-limited oscillator -> enveloped resonant low-pass ->
//!   amplitude ADSR with velocity).
//!
//! Everything is allocated when built; handling events and rendering never allocate, so both can run
//! inside an audio callback.

mod midi;
mod voice;

pub use midi::*;
pub use voice::*;

use crate::signal::Source;
use crate::units::*;

/// One playable voice.
pub trait Voice {
    type Sample: Float;

    /// Starts (or retriggers) `note` at `velocity` in [0, 1].
    fn note_on(&mut self, note: u8, velocity: Self::Sample);
    /// Releases the current note (the voice keeps sounding through its release).
    fn note_off(&mut self);
    /// Whether the voice is producing sound (including its release tail).
    fn is_active(&self) -> bool;
    /// Overwrites `out` with the voice's next samples (silence when inactive).
    fn render(&mut self, out: &mut [Self::Sample]);
    /// Detunes the voice by this many semitones (pitch bend). No-op by default.
    fn set_pitch_bend(&mut self, _semitones: Self::Sample) {}
    /// Silences immediately.
    fn reset(&mut self);
}

/// A MIDI message to apply `offset` samples into the next rendered block.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct TimedEvent {
    pub offset: usize,
    pub message: MidiMessage,
}

#[derive(Debug, Clone, Copy, Default)]
struct Slot {
    /// the note this voice was started for, until it is released (None while releasing or idle)
    note: Option<u8>,
    /// note-off arrived while the sustain pedal was down: release when the pedal comes up
    sustained: bool,
    /// when the note started (for stealing the oldest)
    started: u64,
}

/// A pool of voices played by MIDI.
///
/// Voice allocation: a repeated note retriggers the voice already playing it; otherwise a free voice
/// is used; otherwise the oldest *releasing* voice is stolen, and only then the oldest held one.
/// Responds to note on/off, the sustain pedal (CC 64), all notes off (CC 123), all sound off (CC 120)
/// and pitch bend, on every channel (omni).
#[derive(Debug, Clone)]
pub struct Poly<V: Voice> {
    voices: Vec<V>,
    slots: Vec<Slot>,
    clock: u64,
    sustain: bool,
    bend_range: V::Sample,
    max_block: usize,
    scratch: Vec<V::Sample>,
}

impl<V: Voice> Poly<V> {
    /// `count` voices built by `make(index)`; blocks longer than `max_block` are rendered in pieces.
    /// Panics if `count` or `max_block` is 0.
    pub fn new(count: usize, max_block: usize, make: impl FnMut(usize) -> V) -> Self {
        assert!(count > 0 && max_block > 0, "need at least one voice and a positive block size");
        Self {
            voices: (0..count).map(make).collect(),
            slots: vec![Slot::default(); count],
            clock: 0,
            sustain: false,
            bend_range: V::Sample::_lit(2.0),
            max_block,
            scratch: vec![V::Sample::_ZERO; max_block],
        }
    }
    /// Pitch bend range in semitones for a full bend (default 2).
    pub fn set_bend_range(&mut self, semitones: V::Sample) {
        self.bend_range = semitones;
    }
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

    pub fn note_on(&mut self, note: u8, velocity: V::Sample) {
        self.clock += 1;
        let oldest = |slots: &[Slot], pick: &dyn Fn(usize) -> bool| {
            (0..slots.len()).filter(|&i| pick(i)).min_by_key(|&i| slots[i].started)
        };
        let index = self
            .slots
            .iter()
            .position(|s| s.note == Some(note))
            .or_else(|| (0..self.voices.len()).find(|&i| !self.voices[i].is_active()))
            .or_else(|| oldest(&self.slots, &|i| self.slots[i].note.is_none()))
            .or_else(|| oldest(&self.slots, &|_| true))
            .expect("the pool has at least one voice");
        self.voices[index].note_on(note, velocity);
        self.slots[index] = Slot { note: Some(note), sustained: false, started: self.clock };
    }

    pub fn note_off(&mut self, note: u8) {
        for (voice, slot) in self.voices.iter_mut().zip(&mut self.slots) {
            if slot.note == Some(note) && !slot.sustained {
                if self.sustain {
                    slot.sustained = true;
                } else {
                    voice.note_off();
                    slot.note = None;
                }
            }
        }
    }

    /// Sustain pedal: while down, note-offs are held until it comes back up.
    pub fn set_sustain(&mut self, down: bool) {
        self.sustain = down;
        if !down {
            for (voice, slot) in self.voices.iter_mut().zip(&mut self.slots) {
                if slot.sustained {
                    voice.note_off();
                    *slot = Slot { note: None, sustained: false, started: slot.started };
                }
            }
        }
    }

    /// `amount` in [-1, 1]; scaled by the bend range.
    pub fn set_pitch_bend(&mut self, amount: V::Sample) {
        let semitones = amount._clamp(-V::Sample::_ONE, V::Sample::_ONE) * self.bend_range;
        self.voices.iter_mut().for_each(|v| v.set_pitch_bend(semitones));
    }

    /// Releases every note (they fade out through their release).
    pub fn all_notes_off(&mut self) {
        self.sustain = false;
        for (voice, slot) in self.voices.iter_mut().zip(&mut self.slots) {
            if voice.is_active() {
                voice.note_off();
            }
            *slot = Slot { note: None, sustained: false, started: slot.started };
        }
    }

    /// Silences everything immediately.
    pub fn reset(&mut self) {
        self.sustain = false;
        self.voices.iter_mut().for_each(Voice::reset);
        self.slots.iter_mut().for_each(|s| *s = Slot::default());
    }

    /// Applies one MIDI message.
    pub fn handle(&mut self, message: MidiMessage) {
        let seven_bit = |v: u8| V::Sample::_lit(v as f64 / 127.0);
        match message {
            MidiMessage::NoteOn { note, velocity, .. } => self.note_on(note, seven_bit(velocity)),
            MidiMessage::NoteOff { note, .. } => self.note_off(note),
            MidiMessage::ControlChange { controller: cc::SUSTAIN, value, .. } => self.set_sustain(value >= 64),
            MidiMessage::ControlChange { controller: cc::ALL_NOTES_OFF, .. } => self.all_notes_off(),
            MidiMessage::ControlChange { controller: cc::ALL_SOUND_OFF, .. } => self.reset(),
            MidiMessage::ControlChange { .. } => {}
            MidiMessage::PitchBend { value, .. } => self.set_pitch_bend(V::Sample::_lit(value as f64 / 8192.0)),
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

    /// A voice that outputs a constant equal to its velocity while held, and is instantly silent
    /// when released: easy to reason about.
    #[derive(Debug, Clone, Default)]
    struct Dc {
        note: Option<u8>,
        level: f64,
        bend: f64,
    }

    impl Voice for Dc {
        type Sample = f64;
        fn note_on(&mut self, note: u8, velocity: f64) {
            self.note = Some(note);
            self.level = velocity;
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
    fn events_land_on_their_sample() {
        let mut p = Poly::new(2, 16, |_| Dc::default());
        let mut out = [0.0; 10];
        let on = |offset, note| TimedEvent { offset, message: MidiMessage::NoteOn { channel: 0, note, velocity: 127 } };
        let off = |offset| TimedEvent { offset, message: MidiMessage::NoteOff { channel: 0, note: 60, velocity: 0 } };
        p.render_events(&mut out, &[on(3, 60), on(5, 62), off(8)]);
        assert_eq!(out, [0.0, 0.0, 0.0, 1.0, 1.0, 2.0, 2.0, 2.0, 1.0, 1.0]);
    }
}
