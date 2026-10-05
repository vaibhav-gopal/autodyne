//! Minimal MIDI 1.0 channel-message parsing: the messages a synth needs.

#[cfg(not(any(feature = "std", test)))]
use crate::alloc_prelude::*;

/// A MIDI channel message. Channels are 0-15; data values are 0-127.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum MidiMessage {
    /// A key pressed (velocity 1-127: parsing turns a velocity of 0 into `NoteOff`).
    NoteOn {
        /// MIDI channel, 0-15.
        channel: u8,
        /// Key number (60 = middle C).
        note: u8,
        /// How hard the key was struck.
        velocity: u8,
    },
    /// A key released.
    NoteOff {
        /// MIDI channel, 0-15.
        channel: u8,
        /// Key number.
        note: u8,
        /// Release velocity (often 0 or 64).
        velocity: u8,
    },
    /// A controller moved (see [`cc`]).
    ControlChange {
        /// MIDI channel, 0-15.
        channel: u8,
        /// Controller number.
        controller: u8,
        /// Its new value.
        value: u8,
    },
    /// -8192 (full down) ..= 8191 (full up), 0 = centered
    PitchBend {
        /// MIDI channel, 0-15.
        channel: u8,
        /// The bend.
        value: i16,
    },
    /// aftertouch for the whole channel (in MPE: one note's pressure)
    ChannelPressure {
        /// MIDI channel, 0-15.
        channel: u8,
        /// The pressure.
        value: u8,
    },
    /// aftertouch for one note
    PolyPressure {
        /// MIDI channel, 0-15.
        channel: u8,
        /// Key number.
        note: u8,
        /// The pressure.
        value: u8,
    },
}

/// Controller numbers with meaning to a synth.
pub mod cc {
    /// modulation wheel
    pub const MOD_WHEEL: u8 = 1;
    /// sustain (damper) pedal: values >= 64 are down
    pub const SUSTAIN: u8 = 64;
    /// brightness / timbre: MPE's third dimension ("slide")
    pub const TIMBRE: u8 = 74;
    /// all sound off: silence immediately
    pub const ALL_SOUND_OFF: u8 = 120;
    /// all notes off: release every note
    pub const ALL_NOTES_OFF: u8 = 123;
}

impl MidiMessage {
    /// Parses one complete channel message. Returns `None` for other messages (system, clock,
    /// program change, ...), truncated input or out-of-range data bytes.
    /// A note-on with velocity 0 is a note-off, as the MIDI spec defines.
    pub fn parse(bytes: &[u8]) -> Option<MidiMessage> {
        let (&status, data) = bytes.split_first()?;
        let channel = status & 0x0F;
        let byte = |i: usize| data.get(i).copied().filter(|b| *b < 0x80);
        match status & 0xF0 {
            0x80 => Some(MidiMessage::NoteOff { channel, note: byte(0)?, velocity: byte(1)? }),
            0x90 => {
                let (note, velocity) = (byte(0)?, byte(1)?);
                Some(if velocity == 0 {
                    MidiMessage::NoteOff { channel, note, velocity: 0 }
                } else {
                    MidiMessage::NoteOn { channel, note, velocity }
                })
            }
            0xA0 => Some(MidiMessage::PolyPressure { channel, note: byte(0)?, value: byte(1)? }),
            0xB0 => Some(MidiMessage::ControlChange { channel, controller: byte(0)?, value: byte(1)? }),
            0xD0 => Some(MidiMessage::ChannelPressure { channel, value: byte(0)? }),
            0xE0 => {
                let raw = ((byte(1)? as i16) << 7) | byte(0)? as i16;
                Some(MidiMessage::PitchBend { channel, value: raw - 8192 })
            }
            _ => None,
        }
    }

    /// The raw bytes of this message (e.g. to send or log it): the first [`wire_len`](Self::wire_len) are
    /// meaningful (channel pressure has 2).
    pub fn to_bytes(self) -> [u8; 3] {
        match self {
            MidiMessage::NoteOn { channel, note, velocity } => [0x90 | channel, note, velocity],
            MidiMessage::NoteOff { channel, note, velocity } => [0x80 | channel, note, velocity],
            MidiMessage::ControlChange { channel, controller, value } => [0xB0 | channel, controller, value],
            MidiMessage::PitchBend { channel, value } => {
                let raw = (value.clamp(-8192, 8191) + 8192) as u16;
                [0xE0 | channel, (raw & 0x7F) as u8, (raw >> 7) as u8]
            }
            MidiMessage::ChannelPressure { channel, value } => [0xD0 | channel, value, 0],
            MidiMessage::PolyPressure { channel, note, value } => [0xA0 | channel, note, value],
        }
    }
    /// Number of bytes the message takes on the wire.
    pub fn wire_len(self) -> usize {
        if matches!(self, MidiMessage::ChannelPressure { .. }) { 2 } else { 3 }
    }
    /// The channel the message is on.
    pub fn channel(self) -> u8 {
        match self {
            MidiMessage::NoteOn { channel, .. }
            | MidiMessage::NoteOff { channel, .. }
            | MidiMessage::ControlChange { channel, .. }
            | MidiMessage::PitchBend { channel, .. }
            | MidiMessage::ChannelPressure { channel, .. }
            | MidiMessage::PolyPressure { channel, .. } => channel,
        }
    }
}

/// Frequency in Hz of a (possibly fractional) MIDI note number, A4 = note 69 = 440 Hz.
pub fn midi_to_hz(note: f64) -> f64 {
    440.0 * 2f64.powf((note - 69.0) / 12.0)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parses_channel_messages() {
        assert_eq!(MidiMessage::parse(&[0x93, 60, 100]), Some(MidiMessage::NoteOn { channel: 3, note: 60, velocity: 100 }));
        assert_eq!(MidiMessage::parse(&[0x90, 60, 0]), Some(MidiMessage::NoteOff { channel: 0, note: 60, velocity: 0 }));
        assert_eq!(MidiMessage::parse(&[0x81, 61, 40]), Some(MidiMessage::NoteOff { channel: 1, note: 61, velocity: 40 }));
        assert_eq!(MidiMessage::parse(&[0xB0, cc::SUSTAIN, 127]), Some(MidiMessage::ControlChange { channel: 0, controller: 64, value: 127 }));
        assert_eq!(MidiMessage::parse(&[0xE0, 0x00, 0x40]), Some(MidiMessage::PitchBend { channel: 0, value: 0 }));
        assert_eq!(MidiMessage::parse(&[0xE0, 0x7F, 0x7F]), Some(MidiMessage::PitchBend { channel: 0, value: 8191 }));
        assert_eq!(MidiMessage::parse(&[0xE0, 0x00, 0x00]), Some(MidiMessage::PitchBend { channel: 0, value: -8192 }));
    }

    #[test]
    fn rejects_other_and_malformed_messages() {
        assert_eq!(MidiMessage::parse(&[]), None);
        assert_eq!(MidiMessage::parse(&[0xF8]), None, "clock");
        assert_eq!(MidiMessage::parse(&[0xC0, 5]), None, "program change");
        assert_eq!(MidiMessage::parse(&[0xD0]), None, "truncated channel pressure");
        assert_eq!(MidiMessage::parse(&[0x90, 60]), None, "truncated");
        assert_eq!(MidiMessage::parse(&[0x90, 200, 10]), None, "data byte out of range");
    }

    #[test]
    fn bytes_roundtrip() {
        for m in [
            MidiMessage::NoteOn { channel: 9, note: 36, velocity: 90 },
            MidiMessage::NoteOff { channel: 0, note: 36, velocity: 64 },
            MidiMessage::ControlChange { channel: 2, controller: 1, value: 33 },
            MidiMessage::PitchBend { channel: 15, value: -1234 },
            MidiMessage::ChannelPressure { channel: 4, value: 99 },
            MidiMessage::PolyPressure { channel: 1, note: 60, value: 12 },
        ] {
            assert_eq!(MidiMessage::parse(&m.to_bytes()[..m.wire_len()]), Some(m));
        }
    }

    #[test]
    fn note_frequencies() {
        assert_eq!(midi_to_hz(69.0), 440.0);
        assert!((midi_to_hz(60.0) - 261.625_565).abs() < 1e-5);
        assert!((midi_to_hz(81.0) - 880.0).abs() < 1e-9);
    }
}
