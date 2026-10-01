//! Low-frequency oscillator: a control signal for modulation, free-running in Hz or locked to the
//! host's beat grid.

use super::transport::{Division, Transport};
use crate::osc::Noise;
use crate::params::{ParamError, ParamInfo, ParamUnit, Parameterized};

/// The shape of one LFO cycle. All shapes are bipolar (-1 to 1) and start where a sine does.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum LfoShape {
    Sine,
    Triangle,
    /// rises from -1 to 1
    SawUp,
    /// falls from 1 to -1
    SawDown,
    Square,
    /// a new random value every cycle, held
    SampleHold,
    /// random values every cycle, eased between with a cosine
    SmoothRandom,
}

impl LfoShape {
    pub const ALL: [LfoShape; 7] = [
        LfoShape::Sine,
        LfoShape::Triangle,
        LfoShape::SawUp,
        LfoShape::SawDown,
        LfoShape::Square,
        LfoShape::SampleHold,
        LfoShape::SmoothRandom,
    ];
    pub const NAMES: [&'static str; 7] = ["Sine", "Triangle", "Saw up", "Saw down", "Square", "Sample & hold", "Smooth random"];
}

/// An LFO, advanced in blocks: [`advance`](Lfo::advance) moves it forward and returns its value,
/// typically once per control step (every few dozen samples) to drive a modulation matrix.
///
/// With tempo sync on and the transport playing, the phase is computed from the beat position, so
/// the LFO stays locked to the host's grid even across loops and jumps. Otherwise it runs freely
/// (at the synced rate if sync is on).
#[derive(Debug, Clone)]
pub struct Lfo {
    shape: LfoShape,
    rate_hz: f64,
    sync: bool,
    division: usize,
    phase_offset: f64,
    fade_seconds: f64,
    sample_rate: f64,
    /// free-running phase in cycles, [0, 1), without the offset
    phase: f64,
    /// count of completed cycles (random shapes draw a new value on each)
    cycle: i64,
    samples_since_trigger: usize,
    noise: Noise<f64>,
    previous: f64,
    next: f64,
    value: f64,
}

impl Lfo {
    /// A 1 Hz sine, free-running, no fade.
    pub fn new(sample_rate: f64) -> Self {
        let mut noise = Noise::new(0x004C_464F);
        let (previous, next) = (noise.next_sample(), noise.next_sample());
        Self {
            shape: LfoShape::Sine,
            rate_hz: 1.0,
            sync: false,
            division: Division::QUARTER_INDEX,
            phase_offset: 0.0,
            fade_seconds: 0.0,
            sample_rate,
            phase: 0.0,
            cycle: 0,
            samples_since_trigger: usize::MAX,
            noise,
            previous,
            next,
            value: 0.0,
        }
    }
    pub fn with_shape(mut self, shape: LfoShape) -> Self {
        self.shape = shape;
        self
    }
    /// Seed for the random shapes (deterministic: the same seed gives the same sequence).
    pub fn with_seed(mut self, seed: u64) -> Self {
        self.noise = Noise::new(seed);
        (self.previous, self.next) = (self.noise.next_sample(), self.noise.next_sample());
        self
    }
    pub fn set_shape(&mut self, shape: LfoShape) {
        self.shape = shape;
    }
    pub fn shape(&self) -> LfoShape {
        self.shape
    }
    /// Free-running rate in Hz (used while sync is off).
    pub fn set_rate(&mut self, hz: f64) {
        self.rate_hz = hz.max(0.0);
    }
    pub fn rate(&self) -> f64 {
        self.rate_hz
    }
    pub fn set_sync(&mut self, on: bool) {
        self.sync = on;
    }
    pub fn sync(&self) -> bool {
        self.sync
    }
    /// The synced cycle length, as an index into [`Division::ALL`] (clamped).
    pub fn set_division(&mut self, index: usize) {
        self.division = index.min(Division::ALL.len() - 1);
    }
    pub fn division(&self) -> Division {
        Division::ALL[self.division]
    }
    pub fn division_index(&self) -> usize {
        self.division
    }
    /// Phase offset in cycles, [0, 1).
    pub fn set_phase_offset(&mut self, cycles: f64) {
        self.phase_offset = cycles.rem_euclid(1.0);
    }
    pub fn phase_offset(&self) -> f64 {
        self.phase_offset
    }
    /// Fade-in time after each trigger (0 = full depth at once): delayed vibrato.
    pub fn set_fade(&mut self, seconds: f64) {
        self.fade_seconds = seconds.max(0.0);
    }
    pub fn fade(&self) -> f64 {
        self.fade_seconds
    }
    /// The most recent value, in [-1, 1].
    pub fn value(&self) -> f64 {
        self.value
    }
    /// Restarts the cycle (free-running mode) and the fade-in: call on note-on for a per-note LFO.
    pub fn trigger(&mut self) {
        self.phase = 0.0;
        self.samples_since_trigger = 0;
    }
    /// Back to the start: phase, fade (finished) and random sequence position kept.
    pub fn reset(&mut self) {
        self.phase = 0.0;
        self.cycle = 0;
        self.samples_since_trigger = usize::MAX;
        self.value = 0.0;
    }

    /// Cycles per second right now.
    pub fn frequency(&self, transport: Option<&Transport>) -> f64 {
        if self.sync { self.division().hz(transport.map_or(120.0, |t| t.tempo)) } else { self.rate_hz }
    }

    /// Moves `samples` forward and returns the value at the end of that span.
    /// `transport` locks a synced LFO to the beat grid while it plays; pass `None` to run freely.
    pub fn advance(&mut self, samples: usize, transport: Option<&Transport>) -> f64 {
        let (phase, cycle) = match transport {
            Some(t) if self.sync && t.playing => {
                let cycles = t.position_after(samples, self.sample_rate) / self.division().beats;
                // keep the free-running phase in step, so stopping the transport doesn't jump
                self.phase = cycles.rem_euclid(1.0);
                ((cycles + self.phase_offset).rem_euclid(1.0), cycles.floor() as i64)
            }
            _ => {
                let total = self.phase + self.frequency(transport) * samples as f64 / self.sample_rate;
                let wraps = total.floor();
                self.phase = total - wraps;
                ((self.phase + self.phase_offset).rem_euclid(1.0), self.cycle + wraps as i64)
            }
        };
        if cycle != self.cycle {
            // one fresh random value per cycle (at most two matter: the previous and the next)
            for _ in 0..(cycle - self.cycle).unsigned_abs().min(2) {
                self.previous = self.next;
                self.next = self.noise.next_sample();
            }
            self.cycle = cycle;
        }
        self.samples_since_trigger = self.samples_since_trigger.saturating_add(samples);
        let fade = if self.fade_seconds > 0.0 {
            (self.samples_since_trigger as f64 / (self.fade_seconds * self.sample_rate)).min(1.0)
        } else {
            1.0
        };
        self.value = fade * self.shape_at(phase);
        self.value
    }

    fn shape_at(&self, p: f64) -> f64 {
        match self.shape {
            LfoShape::Sine => (std::f64::consts::TAU * p).sin(),
            LfoShape::Triangle => {
                if p < 0.25 {
                    4.0 * p
                } else if p < 0.75 {
                    2.0 - 4.0 * p
                } else {
                    4.0 * p - 4.0
                }
            }
            LfoShape::SawUp => 2.0 * p - 1.0,
            LfoShape::SawDown => 1.0 - 2.0 * p,
            LfoShape::Square => {
                if p < 0.5 {
                    1.0
                } else {
                    -1.0
                }
            }
            LfoShape::SampleHold => self.next,
            LfoShape::SmoothRandom => self.previous + (self.next - self.previous) * (0.5 - 0.5 * (std::f64::consts::PI * p).cos()),
        }
    }
}

impl Parameterized for Lfo {
    fn param_count(&self) -> usize {
        6
    }
    fn param_info(&self, index: usize) -> Option<ParamInfo> {
        Some(match index {
            0 => ParamInfo::choice("shape", "Shape", &LfoShape::NAMES, 0),
            1 => ParamInfo::new("rate_hz", "Rate", ParamUnit::Hertz, 0.01, 50.0, 1.0).log(),
            2 => ParamInfo::toggle("sync", "Tempo sync", false),
            3 => ParamInfo::choice("division", "Division", &Division::NAMES, Division::QUARTER_INDEX),
            4 => ParamInfo::new("phase", "Phase", ParamUnit::Fraction, 0.0, 1.0, 0.0).instant(),
            5 => ParamInfo::new("fade_s", "Fade in", ParamUnit::Seconds, 0.0, 10.0, 0.0),
            _ => return None,
        })
    }
    fn param_group(&self, index: usize) -> Option<&'static str> {
        (index < self.param_count()).then_some("LFO")
    }
    fn get_param(&self, index: usize) -> Option<f64> {
        Some(match index {
            0 => LfoShape::ALL.iter().position(|&s| s == self.shape).unwrap_or(0) as f64,
            1 => self.rate_hz,
            2 => f64::from(u8::from(self.sync)),
            3 => self.division as f64,
            4 => self.phase_offset,
            5 => self.fade_seconds,
            _ => return None,
        })
    }
    fn set_param(&mut self, index: usize, value: f64) -> Result<f64, ParamError> {
        let v = self.param_info(index).ok_or(ParamError::UnknownIndex(index))?.validate(value)?;
        match index {
            0 => self.shape = LfoShape::ALL[v as usize],
            1 => self.set_rate(v),
            2 => self.sync = v >= 0.5,
            3 => self.set_division(v as usize),
            4 => self.set_phase_offset(v),
            _ => self.set_fade(v),
        }
        Ok(v)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    const FS: f64 = 48_000.0;

    fn at_phase(shape: LfoShape, phase: f64) -> f64 {
        let mut lfo = Lfo::new(FS).with_shape(shape);
        lfo.set_rate(1.0);
        lfo.advance((phase * FS) as usize, None)
    }

    #[test]
    fn shapes_hit_their_landmarks() {
        let close = |a: f64, b: f64| (a - b).abs() < 1e-3;
        for (shape, quarter, three_quarters) in [
            (LfoShape::Sine, 1.0, -1.0),
            (LfoShape::Triangle, 1.0, -1.0),
            (LfoShape::SawUp, -0.5, 0.5),
            (LfoShape::SawDown, 0.5, -0.5),
            (LfoShape::Square, 1.0, -1.0),
        ] {
            assert!(close(at_phase(shape, 0.25), quarter), "{shape:?} at 1/4");
            assert!(close(at_phase(shape, 0.75), three_quarters), "{shape:?} at 3/4");
        }
    }

    #[test]
    fn free_running_rate_is_exact() {
        let mut lfo = Lfo::new(FS);
        lfo.set_rate(5.0);
        let mut crossings = 0;
        let mut last = lfo.value();
        // 1.05 s: the 10th crossing (at exactly 1 s) falls inside the window
        for _ in 0..(1.05 * FS) as usize / 32 {
            let v = lfo.advance(32, None);
            if (last < 0.0) != (v < 0.0) {
                crossings += 1;
            }
            last = v;
        }
        assert_eq!(crossings, 10, "5 cycles per second cross zero 10 times");
    }

    #[test]
    fn synced_lfo_locks_to_the_beat_grid() {
        let mut lfo = Lfo::new(FS).with_shape(LfoShape::SawUp);
        lfo.set_sync(true);
        lfo.set_division(Division::QUARTER_INDEX);
        let mut transport = Transport { playing: true, position: 7.25, ..Transport::new(90.0) };
        // wherever the host jumps, the value follows the position: a quarter note per cycle
        assert!((lfo.advance(0, Some(&transport)) - (2.0 * 0.25 - 1.0)).abs() < 1e-12);
        transport.position = 12.5;
        assert!((lfo.advance(0, Some(&transport)) - 0.0).abs() < 1e-12);
        // and advancing within a block moves it by the tempo
        let half_beat = (0.5 * 60.0 / 90.0 * FS) as usize;
        assert!((lfo.advance(half_beat, Some(&transport)) + 1.0).abs() < 1e-3, "back at the cycle start");
        // stopped: it keeps running freely at the synced rate (90 BPM quarter = 1.5 Hz)
        transport.playing = false;
        assert!((lfo.frequency(Some(&transport)) - 1.5).abs() < 1e-12);
    }

    #[test]
    fn random_shapes_change_once_per_cycle_and_are_deterministic() {
        let run = |seed| {
            let mut lfo = Lfo::new(FS).with_shape(LfoShape::SampleHold).with_seed(seed);
            lfo.set_rate(10.0); // 4,800 samples per cycle
            (0..40).map(|_| lfo.advance(480, None)).collect::<Vec<_>>()
        };
        let values = run(7);
        assert_eq!(values, run(7), "same seed, same sequence");
        assert_ne!(values, run(8));
        let changes = values.windows(2).filter(|w| w[0] != w[1]).count();
        assert_eq!(changes, 3, "4 cycles in 40 x 480 samples: a new value at each of 3 wraps");
        assert!(values.iter().all(|v| (-1.0..1.0).contains(v)));
    }

    #[test]
    fn fade_in_ramps_depth_after_a_trigger() {
        let mut lfo = Lfo::new(FS).with_shape(LfoShape::Square);
        lfo.set_fade(1.0);
        lfo.trigger();
        assert!((lfo.advance((0.25 * FS) as usize, None) - 0.25).abs() < 1e-9, "a quarter of the way in");
        lfo.advance(FS as usize, None);
        assert_eq!(lfo.value().abs(), 1.0, "full depth after the fade");
    }

    #[test]
    fn parameters_round_trip() {
        let mut lfo = Lfo::new(FS);
        lfo.set_param_by_id("shape", 4.0).unwrap();
        lfo.set_param_by_id("sync", 1.0).unwrap();
        lfo.set_param_by_id("division", 5.0).unwrap();
        assert_eq!((lfo.shape(), lfo.sync(), lfo.division()), (LfoShape::Square, true, Division::EIGHTH));
        let info = lfo.param_info(3).unwrap();
        assert_eq!(info.format(lfo.get_param(3).unwrap()), "1/8");
    }
}
