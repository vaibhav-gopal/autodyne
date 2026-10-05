//! Musical time: tempo, beat position and note lengths, for tempo-synced LFOs, delays and
//! arpeggiators.

#[cfg(not(any(feature = "std", test)))]
use crate::alloc_prelude::*;

/// Where the host's timeline is. Positions are in beats (quarter notes), which is how DAWs report
/// them. A host updates it once per block (or calls [`advance`](Self::advance) itself when it runs
/// the clock).
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Transport {
    /// beats (quarter notes) per minute
    pub tempo: f64,
    /// bar length in beats (quarter notes): 4 in 4/4, 3 in 3/4, and also 3 in 6/8 (six eighth
    /// notes); [`bar_length`](Self::bar_length) converts a time signature
    pub beats_per_bar: f64,
    /// position at the start of the current block, in beats since the start of the timeline
    pub position: f64,
    /// whether the host is playing (the position moves only while it is)
    pub playing: bool,
}

impl Default for Transport {
    /// 120 BPM, 4/4, stopped at the start.
    fn default() -> Self {
        Self { tempo: 120.0, beats_per_bar: 4.0, position: 0.0, playing: false }
    }
}

impl Transport {
    /// Stopped at the start, in 4/4 at `tempo` BPM.
    pub fn new(tempo: f64) -> Self {
        Self { tempo, ..Self::default() }
    }
    /// Bar length in beats (quarter notes) of a time signature: numerator x 4 / denominator.
    /// Falls back to 4/4 for a signature with a zero part.
    pub fn bar_length(numerator: u32, denominator: u32) -> f64 {
        if numerator == 0 || denominator == 0 { 4.0 } else { numerator as f64 * 4.0 / denominator as f64 }
    }
    /// Length of one beat in seconds.
    pub fn seconds_per_beat(&self) -> f64 {
        60.0 / self.tempo
    }
    /// Beats that pass in one sample at `sample_rate`.
    pub fn beats_per_sample(&self, sample_rate: f64) -> f64 {
        self.tempo / (60.0 * sample_rate)
    }
    /// The position `samples` from now (moves only while playing).
    pub fn position_after(&self, samples: usize, sample_rate: f64) -> f64 {
        if self.playing { self.position + samples as f64 * self.beats_per_sample(sample_rate) } else { self.position }
    }
    /// Moves the position forward by `samples` (only while playing).
    pub fn advance(&mut self, samples: usize, sample_rate: f64) {
        self.position = self.position_after(samples, sample_rate);
    }
    /// Bar number (from 0) and beat within the bar at the current position.
    pub fn bar_and_beat(&self) -> (f64, f64) {
        let bar = (self.position / self.beats_per_bar).floor();
        (bar, self.position - bar * self.beats_per_bar)
    }
}

/// A note length for tempo sync, in beats (quarter notes): `Division::EIGHTH` is half a beat.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Division {
    /// length in beats (quarter notes)
    pub beats: f64,
}

impl Division {
    /// A whole note: four beats.
    pub const WHOLE: Division = Division { beats: 4.0 };
    /// A half note: two beats.
    pub const HALF: Division = Division { beats: 2.0 };
    /// A quarter note: one beat.
    pub const QUARTER: Division = Division { beats: 1.0 };
    /// An eighth note.
    pub const EIGHTH: Division = Division { beats: 0.5 };
    /// A sixteenth note.
    pub const SIXTEENTH: Division = Division { beats: 0.25 };
    /// A thirty-second note.
    pub const THIRTY_SECOND: Division = Division { beats: 0.125 };

    /// A note this many bars long (in 4/4).
    pub const fn bars(n: f64) -> Division {
        Division { beats: 4.0 * n }
    }
    /// One and a half times as long.
    pub const fn dotted(self) -> Division {
        Division { beats: self.beats * 1.5 }
    }
    /// Three in the time of two.
    pub const fn triplet(self) -> Division {
        Division { beats: self.beats * 2.0 / 3.0 }
    }
    /// Its length in seconds at `tempo` BPM.
    pub fn seconds(self, tempo: f64) -> f64 {
        self.beats * 60.0 / tempo
    }
    /// Repetitions per second at `tempo` (an LFO rate).
    pub fn hz(self, tempo: f64) -> f64 {
        1.0 / self.seconds(tempo)
    }

    /// The common divisions, shortest first, for a choice parameter (names in [`NAMES`](Self::NAMES)).
    pub const ALL: [Division; 15] = [
        Division::THIRTY_SECOND,
        Division::SIXTEENTH.triplet(),
        Division::SIXTEENTH,
        Division::EIGHTH.triplet(),
        Division::SIXTEENTH.dotted(),
        Division::EIGHTH,
        Division::QUARTER.triplet(),
        Division::EIGHTH.dotted(),
        Division::QUARTER,
        Division::HALF.triplet(),
        Division::QUARTER.dotted(),
        Division::HALF,
        Division::WHOLE,
        Division::bars(2.0),
        Division::bars(4.0),
    ];
    /// Display names, in the order of [`ALL`](Self::ALL) ("T": triplet, ".": dotted).
    pub const NAMES: [&'static str; 15] =
        ["1/32", "1/16T", "1/16", "1/8T", "1/16.", "1/8", "1/4T", "1/8.", "1/4", "1/2T", "1/4.", "1/2", "1/1", "2/1", "4/1"];
    /// Index of `QUARTER` in [`ALL`](Self::ALL).
    pub const QUARTER_INDEX: usize = 8;
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn beats_seconds_and_divisions() {
        let t = Transport::new(120.0);
        assert_eq!(t.seconds_per_beat(), 0.5);
        assert_eq!(Division::QUARTER.seconds(120.0), 0.5);
        assert_eq!(Division::EIGHTH.dotted().beats, 0.75);
        assert!((Division::QUARTER.triplet().beats - 2.0 / 3.0).abs() < 1e-12);
        assert_eq!(Division::WHOLE.hz(120.0), 0.5);
        assert_eq!(Division::ALL[Division::QUARTER_INDEX], Division::QUARTER);
        assert!(Division::ALL.windows(2).all(|w| w[0].beats < w[1].beats), "shortest first");
        assert_eq!(Division::ALL.len(), Division::NAMES.len());
    }

    #[test]
    fn advancing_moves_only_while_playing() {
        let mut t = Transport::new(120.0);
        t.advance(48_000, 48_000.0);
        assert_eq!(t.position, 0.0, "stopped");
        t.playing = true;
        t.advance(48_000, 48_000.0); // one second at 120 BPM = 2 beats
        assert!((t.position - 2.0).abs() < 1e-12);
        t.position = 9.5;
        assert_eq!(t.bar_and_beat(), (2.0, 1.5));
        t.beats_per_bar = Transport::bar_length(6, 8); // six eighths: three quarter-note beats
        assert_eq!(t.beats_per_bar, 3.0);
        assert_eq!(t.bar_and_beat(), (3.0, 0.5));
        assert_eq!((Transport::bar_length(7, 8), Transport::bar_length(3, 4), Transport::bar_length(0, 4)), (3.5, 3.0, 4.0));
    }
}
