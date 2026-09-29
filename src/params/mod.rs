//! Parameters and metadata: what a host (DAW, game engine, UI, preset system) needs to discover,
//! display, automate and save a processor's settings.
//!
//! - [`ParamInfo`]: id, display name, unit, range, default, scale, and conversion to the normalized
//!   0..1 range plugin hosts (VST3, CLAP, ...) use.
//! - [`Parameterized`]: index-based access, as plugin APIs do, plus lookup by id, normalized values,
//!   snapshot / restore, and `param_group` to tell which stage of a chain a parameter belongs to.
//!
//! Values cross the API as `f64` whatever the processor's sample type. Setting a value validates it
//! (must be finite), clamps it into range and returns what was applied. Chains (tuples, `Vec`) expose
//! their stages' parameters one after another; `PerChannel` exposes one set that controls every channel.

use thiserror::Error;

use crate::channels::{Linked, Panner, PerChannel, StereoWidth};
use crate::delay::Echo;
use crate::dynamics::{Compressor, EnvelopeFollower};
use crate::distortion::Waveshaper;
use crate::envelope::Adsr;
use crate::resample::Oversampled;
use crate::reverb::{Convolver, Reverb};
use crate::synth::{Poly, SynthVoice, Voice};
use crate::filter::{Biquad, Fir, MultiBiquad};
use crate::gain::{gain_to_db, Gain};
use crate::modulation::{ModulatedDelay, Phaser};
use crate::units::*;

/// What a parameter's value measures (for display and host units).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum ParamUnit {
    None,
    Hertz,
    Decibels,
    Seconds,
    /// a ratio such as 4:1
    Ratio,
    /// output change per input change above a threshold (1 / ratio, 0..1); displayed as a ratio,
    /// with 0 shown as ∞:1
    Slope,
    /// 0..1, displayed as a percentage
    Fraction,
}

/// How a parameter maps onto a 0..1 control (knob, slider, host automation lane).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum ParamScale {
    Linear,
    /// equal steps are equal ratios (frequencies, times); requires min > 0
    Log,
}

/// Description of one parameter.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct ParamInfo {
    /// stable identifier (for presets and scripting), e.g. "threshold_db"
    pub id: &'static str,
    /// human-readable name, e.g. "Threshold"
    pub name: &'static str,
    pub unit: ParamUnit,
    pub min: f64,
    pub max: f64,
    pub default: f64,
    pub scale: ParamScale,
}

impl ParamInfo {
    /// A linear parameter.
    pub const fn new(id: &'static str, name: &'static str, unit: ParamUnit, min: f64, max: f64, default: f64) -> Self {
        Self { id, name, unit, min, max, default, scale: ParamScale::Linear }
    }
    /// The same parameter on a log scale.
    pub const fn log(mut self) -> Self {
        self.scale = ParamScale::Log;
        self
    }
    pub fn clamp(&self, value: f64) -> f64 {
        value.clamp(self.min, self.max)
    }
    /// Checks the value is finite and clamps it into range.
    pub fn validate(&self, value: f64) -> Result<f64, ParamError> {
        if value.is_finite() { Ok(self.clamp(value)) } else { Err(ParamError::NotFinite(self.id)) }
    }
    /// The value as a 0..1 control position.
    pub fn to_normalized(&self, value: f64) -> f64 {
        let v = self.clamp(value);
        if self.max == self.min {
            return 0.0;
        }
        match self.scale {
            ParamScale::Linear => (v - self.min) / (self.max - self.min),
            ParamScale::Log => (v / self.min).ln() / (self.max / self.min).ln(),
        }
    }
    /// The value at a 0..1 control position.
    pub fn from_normalized(&self, normalized: f64) -> f64 {
        let n = normalized.clamp(0.0, 1.0);
        match self.scale {
            ParamScale::Linear => self.min + n * (self.max - self.min),
            ParamScale::Log => self.min * (self.max / self.min).powf(n),
        }
    }
    /// The value formatted for display, e.g. "1.20 kHz", "-18.0 dB", "10.0 ms", "4.0:1", "50%".
    pub fn format(&self, value: f64) -> String {
        match self.unit {
            ParamUnit::Hertz if value.abs() >= 1000.0 => format!("{:.2} kHz", value / 1000.0),
            ParamUnit::Hertz => format!("{value:.1} Hz"),
            ParamUnit::Decibels => format!("{value:.1} dB"),
            ParamUnit::Seconds if value.abs() < 1.0 => format!("{:.1} ms", value * 1000.0),
            ParamUnit::Seconds => format!("{value:.2} s"),
            ParamUnit::Ratio => format!("{value:.1}:1"),
            ParamUnit::Slope if value <= 0.0 => "∞:1".to_string(),
            ParamUnit::Slope => format!("{:.1}:1", 1.0 / value),
            ParamUnit::Fraction => format!("{:.0}%", value * 100.0),
            ParamUnit::None => format!("{value:.3}"),
        }
    }
}

#[derive(Error, Debug, Clone, PartialEq, Eq)]
pub enum ParamError {
    #[error("no parameter at index {0}")]
    UnknownIndex(usize),
    #[error("no parameter with id {0:?}")]
    UnknownId(String),
    #[error("parameter {0:?} must be finite")]
    NotFinite(&'static str),
    #[error("snapshot has {got} values but there are {expected} parameters")]
    SnapshotLength { expected: usize, got: usize },
}

/// A processor (or chain) whose settings can be discovered and changed at runtime.
pub trait Parameterized {
    fn param_count(&self) -> usize;
    fn param_info(&self, index: usize) -> Option<ParamInfo>;
    /// Name of the processor that owns the parameter, e.g. "Compressor" (useful in chains).
    fn param_group(&self, index: usize) -> Option<&'static str>;
    fn get_param(&self, index: usize) -> Option<f64>;
    /// Validates and clamps `value`, applies it, and returns the value applied.
    fn set_param(&mut self, index: usize, value: f64) -> Result<f64, ParamError>;

    /// Index of the first parameter with this id.
    fn param_index(&self, id: &str) -> Option<usize> {
        (0..self.param_count()).find(|&i| self.param_info(i).is_some_and(|p| p.id == id))
    }
    fn get_param_by_id(&self, id: &str) -> Option<f64> {
        self.get_param(self.param_index(id)?)
    }
    fn set_param_by_id(&mut self, id: &str, value: f64) -> Result<f64, ParamError> {
        let index = self.param_index(id).ok_or_else(|| ParamError::UnknownId(id.to_string()))?;
        self.set_param(index, value)
    }
    /// Current value as a 0..1 control position.
    fn get_normalized(&self, index: usize) -> Option<f64> {
        Some(self.param_info(index)?.to_normalized(self.get_param(index)?))
    }
    /// Sets from a 0..1 control position; returns the value applied.
    fn set_normalized(&mut self, index: usize, normalized: f64) -> Result<f64, ParamError> {
        let info = self.param_info(index).ok_or(ParamError::UnknownIndex(index))?;
        self.set_param(index, info.from_normalized(normalized))
    }
    /// All parameter descriptions, in index order.
    fn params(&self) -> Vec<ParamInfo> {
        (0..self.param_count()).filter_map(|i| self.param_info(i)).collect()
    }
    /// All current values, in index order (a preset).
    fn snapshot(&self) -> Vec<f64> {
        (0..self.param_count()).filter_map(|i| self.get_param(i)).collect()
    }
    /// Applies a snapshot taken from a processor with the same parameters.
    fn restore(&mut self, values: &[f64]) -> Result<(), ParamError> {
        if values.len() != self.param_count() {
            return Err(ParamError::SnapshotLength { expected: self.param_count(), got: values.len() });
        }
        for (i, &v) in values.iter().enumerate() {
            self.set_param(i, v)?;
        }
        Ok(())
    }
    /// Resets every parameter to its default.
    fn reset_params(&mut self) -> Result<(), ParamError> {
        for i in 0..self.param_count() {
            let default = self.param_info(i).ok_or(ParamError::UnknownIndex(i))?.default;
            self.set_param(i, default)?;
        }
        Ok(())
    }
}

// CHAINS ==========================================================================================

/// Finds which stage owns a chain-wide index: (stage, index within the stage).
fn locate(stages: &[&dyn Parameterized], mut index: usize) -> Option<(usize, usize)> {
    for (s, stage) in stages.iter().enumerate() {
        let n = stage.param_count();
        if index < n {
            return Some((s, index));
        }
        index -= n;
    }
    None
}

macro_rules! chain_params {
    ($($P:ident . $i:tt),+) => {
        impl<$($P: Parameterized),+> Parameterized for ($($P,)+) {
            fn param_count(&self) -> usize {
                0 $(+ self.$i.param_count())+
            }
            fn param_info(&self, index: usize) -> Option<ParamInfo> {
                let stages: &[&dyn Parameterized] = &[$(&self.$i),+];
                let (s, i) = locate(stages, index)?;
                stages[s].param_info(i)
            }
            fn param_group(&self, index: usize) -> Option<&'static str> {
                let stages: &[&dyn Parameterized] = &[$(&self.$i),+];
                let (s, i) = locate(stages, index)?;
                stages[s].param_group(i)
            }
            fn get_param(&self, index: usize) -> Option<f64> {
                let stages: &[&dyn Parameterized] = &[$(&self.$i),+];
                let (s, i) = locate(stages, index)?;
                stages[s].get_param(i)
            }
            fn set_param(&mut self, index: usize, value: f64) -> Result<f64, ParamError> {
                let (s, i) = locate(&[$(&self.$i),+], index).ok_or(ParamError::UnknownIndex(index))?;
                let stages: &mut [&mut dyn Parameterized] = &mut [$(&mut self.$i),+];
                stages[s].set_param(i, value)
            }
        }
    };
}

chain_params!(A.0);
chain_params!(A.0, B.1);
chain_params!(A.0, B.1, C.2);
chain_params!(A.0, B.1, C.2, D.3);
chain_params!(A.0, B.1, C.2, D.3, E.4);
chain_params!(A.0, B.1, C.2, D.3, E.4, F.5);
chain_params!(A.0, B.1, C.2, D.3, E.4, F.5, G.6);
chain_params!(A.0, B.1, C.2, D.3, E.4, F.5, G.6, H.7);

/// Like `locate`, over a slice of stages, without building a list (so automation never allocates).
fn locate_in<P: Parameterized>(stages: &[P], mut index: usize) -> Option<(usize, usize)> {
    for (s, stage) in stages.iter().enumerate() {
        let n = stage.param_count();
        if index < n {
            return Some((s, index));
        }
        index -= n;
    }
    None
}

impl<P: Parameterized> Parameterized for Vec<P> {
    fn param_count(&self) -> usize {
        self.iter().map(Parameterized::param_count).sum()
    }
    fn param_info(&self, index: usize) -> Option<ParamInfo> {
        let (s, i) = locate_in(self, index)?;
        self[s].param_info(i)
    }
    fn param_group(&self, index: usize) -> Option<&'static str> {
        let (s, i) = locate_in(self, index)?;
        self[s].param_group(i)
    }
    fn get_param(&self, index: usize) -> Option<f64> {
        let (s, i) = locate_in(self, index)?;
        self[s].get_param(i)
    }
    fn set_param(&mut self, index: usize, value: f64) -> Result<f64, ParamError> {
        let (s, i) = locate_in(self, index).ok_or(ParamError::UnknownIndex(index))?;
        self[s].set_param(i, value)
    }
}

impl<P: Parameterized + ?Sized> Parameterized for Box<P> {
    fn param_count(&self) -> usize {
        (**self).param_count()
    }
    fn param_info(&self, index: usize) -> Option<ParamInfo> {
        (**self).param_info(index)
    }
    fn param_group(&self, index: usize) -> Option<&'static str> {
        (**self).param_group(index)
    }
    fn get_param(&self, index: usize) -> Option<f64> {
        (**self).get_param(index)
    }
    fn set_param(&mut self, index: usize, value: f64) -> Result<f64, ParamError> {
        (**self).set_param(index, value)
    }
}

/// A linked processor has exactly the parameters of the processor it wraps.
impl<P: Parameterized> Parameterized for Linked<P> {
    fn param_count(&self) -> usize {
        self.0.param_count()
    }
    fn param_info(&self, index: usize) -> Option<ParamInfo> {
        self.0.param_info(index)
    }
    fn param_group(&self, index: usize) -> Option<&'static str> {
        self.0.param_group(index)
    }
    fn get_param(&self, index: usize) -> Option<f64> {
        self.0.get_param(index)
    }
    fn set_param(&mut self, index: usize, value: f64) -> Result<f64, ParamError> {
        self.0.set_param(index, value)
    }
}

/// One set of parameters controlling every channel (linked stereo / multichannel control).
impl<P: Parameterized> Parameterized for PerChannel<P> {
    fn param_count(&self) -> usize {
        self.channels().first().map_or(0, Parameterized::param_count)
    }
    fn param_info(&self, index: usize) -> Option<ParamInfo> {
        self.channels().first()?.param_info(index)
    }
    fn param_group(&self, index: usize) -> Option<&'static str> {
        self.channels().first()?.param_group(index)
    }
    fn get_param(&self, index: usize) -> Option<f64> {
        self.channels().first()?.get_param(index)
    }
    fn set_param(&mut self, index: usize, value: f64) -> Result<f64, ParamError> {
        let mut applied = Err(ParamError::UnknownIndex(index));
        for p in self.iter_mut() {
            applied = Ok(p.set_param(index, value)?);
        }
        applied
    }
}

// PROCESSORS ======================================================================================

fn f<T: Float>(x: T) -> f64 {
    x.to_f64().unwrap_or(f64::NAN)
}

fn t<T: Float>(v: f64) -> T {
    T::_lit(v)
}

/// Implements `Parameterized` for a processor from a list of infos (may depend on the instance) and
/// read / write mappings by index. Validation, clamping and the bookkeeping are shared.
macro_rules! parameterized {
    ($Ty:ident, $group:literal,
     infos: |$s:ident| $infos:expr,
     read: |$r:ident, $ri:ident| $read:expr,
     write: |$w:ident, $wi:ident, $wv:ident| $write:expr $(,)?) => {
        impl<T: Float> Parameterized for $Ty<T> {
            fn param_count(&self) -> usize {
                let $s = self;
                $infos.len()
            }
            fn param_info(&self, index: usize) -> Option<ParamInfo> {
                let $s = self;
                $infos.get(index).copied()
            }
            fn param_group(&self, index: usize) -> Option<&'static str> {
                (index < self.param_count()).then_some($group)
            }
            fn get_param(&self, index: usize) -> Option<f64> {
                if index >= self.param_count() {
                    return None;
                }
                let ($r, $ri) = (self, index);
                Some($read)
            }
            fn set_param(&mut self, index: usize, value: f64) -> Result<f64, ParamError> {
                let applied = self.param_info(index).ok_or(ParamError::UnknownIndex(index))?.validate(value)?;
                let ($w, $wi, $wv) = (self, index, applied);
                $write;
                Ok(applied)
            }
        }
    };
}

use ParamUnit::{Decibels, Fraction, Hertz, Seconds, Slope};

parameterized!(Gain, "Gain",
    infos: |_s| [ParamInfo::new("gain_db", "Gain", Decibels, -60.0, 24.0, 0.0)],
    read: |p, _i| f(gain_to_db(p.gain())).max(-60.0),
    write: |p, _i, v| p.set_gain_db(t(v)),
);

parameterized!(Echo, "Echo",
    infos: |s| [
        ParamInfo::new("delay_s", "Delay", Seconds, 1.0 / f(s.sample_rate()), f(s.max_delay_seconds()), (0.3f64).min(f(s.max_delay_seconds()))).log(),
        ParamInfo::new("feedback", "Feedback", Fraction, 0.0, 0.99, 0.5),
        ParamInfo::new("mix", "Mix", Fraction, 0.0, 1.0, 0.5),
    ],
    read: |p, i| match i { 0 => f(p.delay_seconds()), 1 => f(p.feedback()), _ => f(p.mix()) },
    write: |p, i, v| match i { 0 => p.set_delay_seconds(t(v)), 1 => p.set_feedback(t(v)), _ => p.set_mix(t(v)) },
);

parameterized!(Compressor, "Compressor",
    infos: |_s| [
        ParamInfo::new("threshold_db", "Threshold", Decibels, -60.0, 0.0, -18.0),
        // the slope (1 / ratio) rather than the ratio: finite for every setting, 0 is a limiter
        ParamInfo::new("slope", "Slope", Slope, 0.0, 1.0, 0.25),
        ParamInfo::new("knee_db", "Knee", Decibels, 0.0, 24.0, 6.0),
        ParamInfo::new("attack_s", "Attack", Seconds, 0.0, 0.5, 0.010),
        ParamInfo::new("release_s", "Release", Seconds, 0.001, 5.0, 0.100).log(),
        ParamInfo::new("makeup_db", "Makeup", Decibels, -24.0, 24.0, 0.0),
    ],
    read: |p, i| match i {
        0 => f(p.threshold_db()),
        1 => f(p.slope()),
        2 => f(p.knee_db()),
        3 => f(p.attack()),
        4 => f(p.release()),
        _ => f(p.makeup_db()),
    },
    write: |p, i, v| match i {
        0 => p.set_threshold_db(t(v)),
        1 => p.set_slope(t(v)),
        2 => p.set_knee_db(t(v)),
        3 => p.set_attack(t(v)),
        4 => p.set_release(t(v)),
        _ => p.set_makeup_db(t(v)),
    },
);

parameterized!(EnvelopeFollower, "Envelope follower",
    infos: |_s| [
        ParamInfo::new("attack_s", "Attack", Seconds, 0.0, 1.0, 0.010),
        ParamInfo::new("release_s", "Release", Seconds, 0.0, 5.0, 0.100),
    ],
    read: |p, i| if i == 0 { f(p.attack()) } else { f(p.release()) },
    write: |p, i, v| if i == 0 { p.set_attack(t(v)) } else { p.set_release(t(v)) },
);

parameterized!(ModulatedDelay, "Modulated delay",
    infos: |s| [
        ParamInfo::new("rate_hz", "Rate", Hertz, 0.01, 20.0, 0.5).log(),
        ParamInfo::new("depth_s", "Depth", Seconds, 0.0, f(s.max_depth()), f(s.max_depth()) / 2.0),
        ParamInfo::new("feedback", "Feedback", Fraction, -0.95, 0.95, 0.0),
        ParamInfo::new("mix", "Mix", Fraction, 0.0, 1.0, 0.5),
    ],
    read: |p, i| match i { 0 => f(p.rate()), 1 => f(p.depth()), 2 => f(p.feedback()), _ => f(p.mix()) },
    write: |p, i, v| match i { 0 => p.set_rate(t(v)), 1 => p.set_depth(t(v)), 2 => p.set_feedback(t(v)), _ => p.set_mix(t(v)) },
);

parameterized!(Phaser, "Phaser",
    infos: |s| {
        let top = (0.49 * f(s.sample_rate())).min(20_000.0);
        [
            ParamInfo::new("rate_hz", "Rate", Hertz, 0.01, 20.0, 0.5).log(),
            ParamInfo::new("min_hz", "Sweep low", Hertz, 20.0, top, 200.0).log(),
            ParamInfo::new("max_hz", "Sweep high", Hertz, 20.0, top, 2_000.0f64.min(top)).log(),
            ParamInfo::new("feedback", "Feedback", Fraction, -0.95, 0.95, 0.0),
            ParamInfo::new("mix", "Mix", Fraction, 0.0, 1.0, 0.5),
        ]
    },
    read: |p, i| {
        let (lo, hi) = p.range();
        match i { 0 => f(p.rate()), 1 => f(lo), 2 => f(hi), 3 => f(p.feedback()), _ => f(p.mix()) }
    },
    write: |p, i, v| {
        let (lo, hi) = p.range();
        match i {
            0 => p.set_rate(t(v)),
            // keep low <= high by moving the other end when they cross
            1 => p.set_range(t(v), hi._max(t(v))),
            2 => p.set_range(lo._min(t(v)), t(v)),
            3 => p.set_feedback(t(v)),
            _ => p.set_mix(t(v)),
        }
    },
);

parameterized!(Adsr, "ADSR",
    infos: |_s| [
        ParamInfo::new("attack_s", "Attack", Seconds, 0.0, 10.0, 0.005),
        ParamInfo::new("decay_s", "Decay", Seconds, 0.0, 10.0, 0.1),
        ParamInfo::new("sustain", "Sustain", Fraction, 0.0, 1.0, 0.7),
        ParamInfo::new("release_s", "Release", Seconds, 0.0, 20.0, 0.3),
    ],
    read: |p, i| match i { 0 => f(p.attack()), 1 => f(p.decay()), 2 => f(p.sustain()), _ => f(p.release()) },
    write: |p, i, v| match i { 0 => p.set_attack(t(v)), 1 => p.set_decay(t(v)), 2 => p.set_sustain(t(v)), _ => p.set_release(t(v)) },
);

parameterized!(Waveshaper, "Waveshaper",
    infos: |_s| [
        ParamInfo::new("drive_db", "Drive", Decibels, 0.0, 48.0, 0.0),
        ParamInfo::new("output_db", "Output", Decibels, -24.0, 24.0, 0.0),
        ParamInfo::new("mix", "Mix", Fraction, 0.0, 1.0, 1.0),
    ],
    read: |p, i| match i { 0 => f(p.drive_db()), 1 => f(p.output_db()), _ => f(p.mix()) },
    write: |p, i, v| match i { 0 => p.set_drive_db(t(v)), 1 => p.set_output_db(t(v)), _ => p.set_mix(t(v)) },
);

/// An oversampled processor has exactly the parameters of the processor inside.
impl<P: Parameterized, T: Float> Parameterized for Oversampled<P, T> {
    fn param_count(&self) -> usize {
        self.inner().param_count()
    }
    fn param_info(&self, index: usize) -> Option<ParamInfo> {
        self.inner().param_info(index)
    }
    fn param_group(&self, index: usize) -> Option<&'static str> {
        self.inner().param_group(index)
    }
    fn get_param(&self, index: usize) -> Option<f64> {
        self.inner().get_param(index)
    }
    fn set_param(&mut self, index: usize, value: f64) -> Result<f64, ParamError> {
        self.inner_mut().set_param(index, value)
    }
}

parameterized!(SynthVoice, "Synth voice",
    infos: |_s| [
        ParamInfo::new("cutoff_hz", "Cutoff", Hertz, 20.0, 20_000.0, 800.0).log(),
        ParamInfo::new("resonance", "Resonance", ParamUnit::None, 0.5, 12.0, 1.2).log(),
        ParamInfo::new("env_amount", "Filter env (octaves)", ParamUnit::None, 0.0, 6.0, 3.0),
        ParamInfo::new("amp_attack_s", "Amp attack", Seconds, 0.0, 10.0, 0.005),
        ParamInfo::new("amp_decay_s", "Amp decay", Seconds, 0.0, 10.0, 0.3),
        ParamInfo::new("amp_sustain", "Amp sustain", Fraction, 0.0, 1.0, 0.6),
        ParamInfo::new("amp_release_s", "Amp release", Seconds, 0.0, 20.0, 0.3),
        ParamInfo::new("filter_attack_s", "Filter attack", Seconds, 0.0, 10.0, 0.002),
        ParamInfo::new("filter_decay_s", "Filter decay", Seconds, 0.0, 10.0, 0.25),
        ParamInfo::new("filter_sustain", "Filter sustain", Fraction, 0.0, 1.0, 0.2),
        ParamInfo::new("filter_release_s", "Filter release", Seconds, 0.0, 20.0, 0.3),
    ],
    read: |p, i| match i {
        0 => f(p.cutoff()),
        1 => f(p.resonance()),
        2 => f(p.env_amount()),
        3 => f(p.amp_env().attack()),
        4 => f(p.amp_env().decay()),
        5 => f(p.amp_env().sustain()),
        6 => f(p.amp_env().release()),
        7 => f(p.filter_env().attack()),
        8 => f(p.filter_env().decay()),
        9 => f(p.filter_env().sustain()),
        _ => f(p.filter_env().release()),
    },
    write: |p, i, v| match i {
        0 => p.set_cutoff(t(v)),
        1 => p.set_resonance(t(v)),
        2 => p.set_env_amount(t(v)),
        3 => p.amp_env_mut().set_attack(t(v)),
        4 => p.amp_env_mut().set_decay(t(v)),
        5 => p.amp_env_mut().set_sustain(t(v)),
        6 => p.amp_env_mut().set_release(t(v)),
        7 => p.filter_env_mut().set_attack(t(v)),
        8 => p.filter_env_mut().set_decay(t(v)),
        9 => p.filter_env_mut().set_sustain(t(v)),
        _ => p.filter_env_mut().set_release(t(v)),
    },
);

/// One set of parameters controlling every voice.
impl<V: Voice + Parameterized> Parameterized for Poly<V> {
    fn param_count(&self) -> usize {
        self.voices().first().map_or(0, Parameterized::param_count)
    }
    fn param_info(&self, index: usize) -> Option<ParamInfo> {
        self.voices().first()?.param_info(index)
    }
    fn param_group(&self, index: usize) -> Option<&'static str> {
        self.voices().first()?.param_group(index)
    }
    fn get_param(&self, index: usize) -> Option<f64> {
        self.voices().first()?.get_param(index)
    }
    fn set_param(&mut self, index: usize, value: f64) -> Result<f64, ParamError> {
        let mut applied = Err(ParamError::UnknownIndex(index));
        for v in self.voices_mut() {
            applied = Ok(v.set_param(index, value)?);
        }
        applied
    }
}

parameterized!(Reverb, "Reverb",
    infos: |_s| [
        ParamInfo::new("size", "Size", ParamUnit::None, 0.25, 2.0, 1.0),
        ParamInfo::new("decay_s", "Decay", Seconds, 0.05, 30.0, 1.8).log(),
        ParamInfo::new("damping_hz", "Damping", Hertz, 500.0, 24_000.0, 6_000.0).log(),
        ParamInfo::new("predelay_s", "Pre-delay", Seconds, 0.0, 0.25, 0.02),
        ParamInfo::new("mix", "Mix", Fraction, 0.0, 1.0, 0.3),
        ParamInfo::new("width", "Width", Fraction, 0.0, 1.0, 1.0),
        ParamInfo::new("modulation", "Modulation", Fraction, 0.0, 1.0, 0.5),
    ],
    read: |p, i| match i {
        0 => f(p.size()),
        1 => f(p.decay()),
        2 => f(p.damping()),
        3 => f(p.predelay()),
        4 => f(p.mix()),
        5 => f(p.width()),
        _ => f(p.modulation()),
    },
    write: |p, i, v| match i {
        0 => p.set_size(t(v)),
        1 => p.set_decay(t(v)),
        2 => p.set_damping(t(v)),
        3 => p.set_predelay(t(v)),
        4 => p.set_mix(t(v)),
        5 => p.set_width(t(v)),
        _ => p.set_modulation(t(v)),
    },
);

parameterized!(Convolver, "Convolver",
    infos: |_s| [ParamInfo::new("mix", "Mix", Fraction, 0.0, 1.0, 1.0)],
    read: |p, _i| f(p.mix()),
    write: |p, _i, v| p.set_mix(t(v)),
);

parameterized!(StereoWidth, "Stereo width",
    infos: |_s| [ParamInfo::new("width", "Width", Fraction, 0.0, 4.0, 1.0)],
    read: |p, _i| f(p.width()),
    write: |p, _i, v| p.set_width(t(v)),
);

parameterized!(Panner, "Panner",
    infos: |_s| [ParamInfo::new("position", "Pan", ParamUnit::None, -1.0, 1.0, 0.0)],
    read: |p, _i| f(p.position()),
    write: |p, _i, v| p.set_position(t(v)),
);

/// A designed biquad exposes frequency, Q and gain; one built from raw coefficients has no parameters.
impl<T: Float> Parameterized for Biquad<T> {
    fn param_count(&self) -> usize {
        if self.design().is_some() { 3 } else { 0 }
    }
    fn param_info(&self, index: usize) -> Option<ParamInfo> {
        let d = self.design()?;
        let top = 0.49 * f(d.sample_rate);
        [
            ParamInfo::new("frequency_hz", "Frequency", Hertz, 10.0, top, 1_000.0f64.min(top)).log(),
            ParamInfo::new("q", "Q", ParamUnit::None, 0.1, 20.0, std::f64::consts::FRAC_1_SQRT_2).log(),
            ParamInfo::new("gain_db", "Gain", Decibels, -24.0, 24.0, 0.0),
        ]
        .get(index)
        .copied()
    }
    fn param_group(&self, index: usize) -> Option<&'static str> {
        (index < self.param_count()).then_some("Biquad")
    }
    fn get_param(&self, index: usize) -> Option<f64> {
        let d = self.design()?;
        [f(d.frequency), f(d.q), f(d.gain_db)].get(index).copied()
    }
    fn set_param(&mut self, index: usize, value: f64) -> Result<f64, ParamError> {
        let applied = self.param_info(index).ok_or(ParamError::UnknownIndex(index))?.validate(value)?;
        let mut d = *self.design().expect("param_info succeeded, so there is a design");
        match index {
            0 => d.frequency = t(applied),
            1 => d.q = t(applied),
            _ => d.gain_db = t(applied),
        }
        self.set_design(d);
        Ok(applied)
    }
}

/// One set of parameters (frequency, Q, gain) applied to every channel's filter.
impl<T: Float> Parameterized for MultiBiquad<T> {
    fn param_count(&self) -> usize {
        self.channels().first().map_or(0, Parameterized::param_count)
    }
    fn param_info(&self, index: usize) -> Option<ParamInfo> {
        self.channels().first()?.param_info(index)
    }
    fn param_group(&self, index: usize) -> Option<&'static str> {
        (index < self.param_count()).then_some("Multichannel biquad")
    }
    fn get_param(&self, index: usize) -> Option<f64> {
        self.channels().first()?.get_param(index)
    }
    fn set_param(&mut self, index: usize, value: f64) -> Result<f64, ParamError> {
        let mut applied = Err(ParamError::UnknownIndex(index));
        for ch in 0..self.channels().len() {
            applied = Ok(self.channel_mut(ch).set_param(index, value)?);
        }
        applied
    }
}

/// FIR taps are data rather than settings, so a FIR has no parameters.
impl<T: Float> Parameterized for Fir<T> {
    fn param_count(&self) -> usize {
        0
    }
    fn param_info(&self, _index: usize) -> Option<ParamInfo> {
        None
    }
    fn param_group(&self, _index: usize) -> Option<&'static str> {
        None
    }
    fn get_param(&self, _index: usize) -> Option<f64> {
        None
    }
    fn set_param(&mut self, index: usize, _value: f64) -> Result<f64, ParamError> {
        Err(ParamError::UnknownIndex(index))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::filter::BUTTERWORTH_Q;

    const FS: f64 = 48_000.0;

    #[test]
    fn info_scaling_and_formatting() {
        let freq = ParamInfo::new("f", "Freq", Hertz, 20.0, 20_000.0, 1_000.0).log();
        assert!((freq.to_normalized(632.455_532) - 0.5).abs() < 1e-6); // geometric middle
        assert!((freq.from_normalized(0.5) - 632.455_532).abs() < 1e-3);
        assert_eq!(freq.to_normalized(1e9), 1.0);
        let mix = ParamInfo::new("m", "Mix", Fraction, 0.0, 1.0, 0.5);
        assert_eq!(mix.from_normalized(0.25), 0.25);
        assert_eq!(freq.format(1_200.0), "1.20 kHz");
        assert_eq!(mix.format(0.5), "50%");
        assert_eq!(ParamInfo::new("a", "A", Seconds, 0.0, 1.0, 0.0).format(0.01), "10.0 ms");
        assert_eq!(mix.validate(f64::NAN), Err(ParamError::NotFinite("m")));
        assert_eq!(mix.validate(3.0), Ok(1.0));
    }

    #[test]
    fn compressor_parameters_round_trip() {
        let mut c = Compressor::<f32>::new(48_000.0);
        assert_eq!(c.param_count(), 6);
        assert_eq!(c.get_param_by_id("threshold_db"), Some(-18.0));
        assert_eq!(c.set_param_by_id("slope", 0.125), Ok(0.125));
        assert!((c.ratio() - 8.0).abs() < 1e-6);
        assert_eq!(c.param_info(1).unwrap().format(0.125), "8.0:1");
        assert_eq!(c.set_param_by_id("threshold_db", -200.0), Ok(-60.0)); // clamped
        assert_eq!(c.set_param_by_id("nope", 1.0), Err(ParamError::UnknownId("nope".into())));
        let limiter = Compressor::<f64>::limiter(-1.0, 0.05, FS);
        assert_eq!(limiter.get_param_by_id("slope"), Some(0.0));
        assert_eq!(limiter.param_info(1).unwrap().format(0.0), "∞:1");
    }

    #[test]
    fn snapshot_restore_and_defaults() {
        let mut echo = Echo::new(1.0, FS);
        echo.set_param_by_id("feedback", 0.8).unwrap();
        let preset = echo.snapshot();
        echo.reset_params().unwrap();
        assert_eq!(echo.get_param_by_id("feedback"), Some(0.5));
        echo.restore(&preset).unwrap();
        assert_eq!(echo.get_param_by_id("feedback"), Some(0.8));
        assert!(echo.restore(&[0.1]).is_err());
        // ranges follow the instance: this echo was built for at most 1 s of delay
        assert_eq!(echo.param_info(0).unwrap().max, 1.0);
    }

    #[test]
    fn limiter_survives_a_preset_round_trip() {
        // snapshot -> restore into a default compressor must give back the same limiter
        let original = Compressor::<f64>::limiter(-6.0, 0.05, FS);
        let mut restored = Compressor::new(FS);
        restored.restore(&original.snapshot()).unwrap();
        assert!(restored.ratio().is_infinite());

        let loud: Vec<f64> = crate::osc::Sine::new(100.0, FS).take(9_600).collect();
        let (mut a, mut b) = (loud.clone(), loud);
        original.clone().process(&mut a);
        restored.process(&mut b);
        assert_eq!(a, b);
    }

    #[test]
    fn biquad_parameters_redesign_the_filter() {
        let mut bq = Biquad::lowpass(1_000.0, FS, BUTTERWORTH_Q);
        bq.set_param_by_id("frequency_hz", 2_000.0).unwrap();
        assert!((bq.magnitude_at(2_000.0, FS) - BUTTERWORTH_Q).abs() < 1e-9, "-3 dB moved to 2 kHz");
        assert_eq!(Biquad::new(*bq.coeffs()).param_count(), 0);
    }

    #[test]
    fn chains_concatenate_and_group() {
        let mut chain = (Gain::new(1.0, 0.0, FS), Compressor::new(FS), Biquad::peaking(1_000.0, FS, 1.0, 3.0));
        assert_eq!(chain.param_count(), 1 + 6 + 3);
        assert_eq!(chain.param_group(0), Some("Gain"));
        assert_eq!(chain.param_group(1), Some("Compressor"));
        assert_eq!(chain.param_group(9), Some("Biquad"));
        assert_eq!(chain.param_info(9).unwrap().id, "gain_db");
        // ids repeat across stages; the index disambiguates, param_index finds the first
        assert_eq!(chain.param_index("gain_db"), Some(0));
        chain.set_param(9, 6.0).unwrap();
        assert_eq!(chain.2.design().unwrap().gain_db, 6.0);
        assert_eq!(chain.param_group(10), None);

        let boxed: Vec<Box<dyn Parameterized>> = vec![Box::new(Gain::new(1.0, 0.0, FS)), Box::new(StereoWidth::new(1.0, FS))];
        assert_eq!(boxed.param_count(), 2);
        assert_eq!(boxed.param_info(1).unwrap().id, "width");
    }

    #[test]
    fn per_channel_parameters_are_linked() {
        let mut stereo = PerChannel::new(2, |_| Compressor::new(FS));
        assert_eq!(stereo.param_count(), 6);
        stereo.set_param_by_id("threshold_db", -30.0).unwrap();
        assert!(stereo.iter_mut().all(|c| c.threshold_db() == -30.0));
    }

    #[test]
    fn phaser_range_stays_ordered() {
        let mut p = Phaser::new(4, FS);
        p.set_param_by_id("min_hz", 5_000.0).unwrap(); // above the current max (2 kHz)
        let (lo, hi) = p.range();
        assert!(lo <= hi && lo == 5_000.0);
    }
}
