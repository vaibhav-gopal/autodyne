//! Parameters and metadata: what a host (DAW, game engine, UI, preset system) needs to discover,
//! display, automate and save a processor's settings.
//!
//! - [`ParamInfo`]: id, display name, unit, range, default, scale, and conversion to the normalized
//!   0..1 range plugin hosts (VST3, CLAP, ...) use.
//! - [`Parameterized`]: index-based access, as plugin APIs do, plus lookup by id, normalized values,
//!   snapshot / restore, and `param_group` to tell which stage of a chain a parameter belongs to.
//! - [`Smoothed`]: wraps any processor so changes to continuous parameters ramp instead of jumping;
//!   [`ParamEvent`] with [`process_events`] / [`process_buffer_events`] applies changes at exact
//!   sample offsets. Each `ParamInfo` declares its [`Smoothing`]: ramped, smoothed internally, or instant.
//!
//! Values cross the API as `f64` whatever the processor's sample type. Setting a value validates it
//! (must be finite), clamps it into range and returns what was applied. Chains (tuples, `Vec`) expose
//! their stages' parameters one after another; `PerChannel` exposes one set that controls every channel.

use thiserror::Error;

mod smoothed;
pub use smoothed::*;

use crate::channels::{Linked, PerChannel};
use crate::units::*;

/// What a parameter's value measures (for display and host units).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum ParamUnit {
    /// a plain number
    None,
    /// a frequency
    Hertz,
    /// a level in dB
    Decibels,
    /// a time
    Seconds,
    /// a ratio such as 4:1
    Ratio,
    /// output change per input change above a threshold (1 / ratio, 0..1); displayed as a ratio,
    /// with 0 shown as ∞:1
    Slope,
    /// 0..1, displayed as a percentage
    Fraction,
    /// hundredths of a semitone
    Cents,
    /// pitch intervals (transposition, bend ranges)
    Semitones,
}

/// How a parameter maps onto a 0..1 control (knob, slider, host automation lane).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum ParamScale {
    /// equal steps are equal differences
    Linear,
    /// equal steps are equal ratios (frequencies, times); requires min > 0
    Log,
}

/// What kind of value a parameter holds.
///
/// Values always cross the API as `f64`, the way plugin hosts model parameters: the discrete kinds hold
/// whole numbers (an index for a choice, 0 or 1 for a toggle), and setting one rounds to the nearest.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum ParamKind {
    /// any value in range
    Continuous,
    /// whole numbers in range, e.g. a voice count or an octave
    Integer,
    /// off (0) or on (1)
    Toggle,
    /// one of several named options: the value is the index into this list
    Choice(&'static [&'static str]),
}

/// How a change to a parameter should reach the audio.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum Smoothing {
    /// ramp to new values (the default for continuous parameters): applied by [`Smoothed`], so
    /// automation never jumps audibly
    Ramp,
    /// the processor already smooths this parameter sample by sample: pass changes straight through
    Internal,
    /// apply immediately (discrete parameters, and ones where a glide would sound wrong)
    Instant,
}

/// Description of one parameter.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct ParamInfo {
    /// stable identifier (for presets and scripting), e.g. "threshold_db"
    pub id: &'static str,
    /// human-readable name, e.g. "Threshold"
    pub name: &'static str,
    /// what the value measures
    pub unit: ParamUnit,
    /// smallest value
    pub min: f64,
    /// largest value
    pub max: f64,
    /// value of a fresh processor
    pub default: f64,
    /// how the value maps onto a 0..1 control
    pub scale: ParamScale,
    /// what kind of value it is
    pub kind: ParamKind,
    /// how changes reach the audio
    pub smoothing: Smoothing,
}

impl ParamInfo {
    /// A linear, continuous parameter.
    pub const fn new(id: &'static str, name: &'static str, unit: ParamUnit, min: f64, max: f64, default: f64) -> Self {
        Self { id, name, unit, min, max, default, scale: ParamScale::Linear, kind: ParamKind::Continuous, smoothing: Smoothing::Ramp }
    }
    /// An on/off switch (0 or 1).
    pub const fn toggle(id: &'static str, name: &'static str, default: bool) -> Self {
        let default = if default { 1.0 } else { 0.0 };
        Self { kind: ParamKind::Toggle, smoothing: Smoothing::Instant, ..Self::new(id, name, ParamUnit::None, 0.0, 1.0, default) }
    }
    /// A choice between named options; the value is an index into `options`.
    /// Panics if `options` is empty or `default` is out of range (at compile time in a `const`).
    pub const fn choice(id: &'static str, name: &'static str, options: &'static [&'static str], default: usize) -> Self {
        assert!(!options.is_empty() && default < options.len(), "a choice needs options and a default among them");
        let max = (options.len() - 1) as f64;
        Self { kind: ParamKind::Choice(options), smoothing: Smoothing::Instant, ..Self::new(id, name, ParamUnit::None, 0.0, max, default as f64) }
    }
    /// The same parameter on a log scale.
    pub const fn log(mut self) -> Self {
        self.scale = ParamScale::Log;
        self
    }
    /// The same parameter restricted to whole numbers (applied instantly: no ramp between steps).
    pub const fn integer(mut self) -> Self {
        self.kind = ParamKind::Integer;
        self.smoothing = Smoothing::Instant;
        self
    }
    /// Marks the parameter as smoothed by its processor already (no extra ramp).
    pub const fn smoothed_internally(mut self) -> Self {
        self.smoothing = Smoothing::Internal;
        self
    }
    /// Marks the parameter as applied immediately (no ramp).
    pub const fn instant(mut self) -> Self {
        self.smoothing = Smoothing::Instant;
        self
    }
    /// Whether the parameter only takes whole-number values (integer, toggle or choice).
    pub const fn is_discrete(&self) -> bool {
        !matches!(self.kind, ParamKind::Continuous)
    }
    /// Clamps into range, rounding discrete parameters to the nearest whole number.
    pub fn clamp(&self, value: f64) -> f64 {
        let value = value.clamp(self.min, self.max);
        if self.is_discrete() { value.round().clamp(self.min, self.max) } else { value }
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
    /// The value at a 0..1 control position (rounded for discrete parameters).
    pub fn from_normalized(&self, normalized: f64) -> f64 {
        let n = normalized.clamp(0.0, 1.0);
        let value = match self.scale {
            ParamScale::Linear => self.min + n * (self.max - self.min),
            ParamScale::Log => self.min * (self.max / self.min).powf(n),
        };
        if self.is_discrete() { self.clamp(value) } else { value }
    }
    /// The value formatted for display, e.g. "1.20 kHz", "-18.0 dB", "10.0 ms", "4.0:1", "50%", or the
    /// option's name for a choice and "On" / "Off" for a toggle.
    pub fn format(&self, value: f64) -> String {
        match self.kind {
            ParamKind::Toggle => return if value >= 0.5 { "On" } else { "Off" }.to_string(),
            ParamKind::Choice(options) => return options[self.clamp(value) as usize].to_string(),
            ParamKind::Integer if self.unit == ParamUnit::None => return format!("{:.0}", value.round()),
            _ => {}
        }
        match self.unit {
            // unit switches compare against the rounded display value, so that parsing the text and
            // formatting again gives the same text (999.96 Hz shows as "1.00 kHz", not "1000.0 Hz")
            ParamUnit::Hertz if value.abs() >= 999.95 => format!("{:.2} kHz", value / 1000.0),
            ParamUnit::Hertz => format!("{value:.1} Hz"),
            ParamUnit::Decibels => format!("{value:.1} dB"),
            ParamUnit::Seconds if value.abs() < 0.999_95 => format!("{:.1} ms", value * 1000.0),
            ParamUnit::Seconds => format!("{value:.2} s"),
            ParamUnit::Ratio => format!("{value:.1}:1"),
            ParamUnit::Slope if value <= 0.0 => "∞:1".to_string(),
            ParamUnit::Slope => format!("{:.1}:1", 1.0 / value),
            ParamUnit::Fraction => format!("{:.0}%", value * 100.0),
            ParamUnit::Cents => format!("{value:.1} ct"),
            ParamUnit::Semitones => format!("{value:.2} st"),
            ParamUnit::None => format!("{value:.3}"),
        }
    }
    /// Reads a value typed by a user or produced by [`format`](Self::format): the inverse of
    /// `format`, e.g. "1.2 kHz" → 1200, "10 ms" → 0.01, "50%" → 0.5, "8:1" on a slope → 0.125.
    /// Units may be omitted; a bare number is in the display unit (seconds, Hz, percent, ratio).
    /// Choices accept an option's name (any case) or its index, toggles "on" / "off" (or yes / no,
    /// true / false, 1 / 0). Not clamped: pass the result to [`validate`](Self::validate) or
    /// `set_param`.
    pub fn parse(&self, text: &str) -> Option<f64> {
        let text = text.trim();
        match self.kind {
            ParamKind::Toggle => {
                return match text.to_ascii_lowercase().as_str() {
                    "on" | "yes" | "true" | "1" => Some(1.0),
                    "off" | "no" | "false" | "0" => Some(0.0),
                    _ => None,
                };
            }
            ParamKind::Choice(options) => {
                if let Some(index) = options.iter().position(|o| o.eq_ignore_ascii_case(text)) {
                    return Some(index as f64);
                }
            }
            _ => {}
        }
        if self.unit == ParamUnit::Slope && (text.starts_with('∞') || text.to_ascii_lowercase().starts_with("inf")) {
            return Some(0.0);
        }
        let end = text
            .char_indices()
            .find(|&(i, c)| !(c.is_ascii_digit() || c == '.' || ((c == '-' || c == '+') && i == 0)))
            .map_or(text.len(), |(i, _)| i);
        let number: f64 = text[..end].parse().ok()?;
        let suffix = text[end..].trim().to_ascii_lowercase();
        let value = match self.unit {
            ParamUnit::Hertz if suffix.starts_with('k') => number * 1000.0,
            ParamUnit::Seconds if suffix.starts_with("ms") => number / 1000.0,
            ParamUnit::Fraction => number / 100.0,
            ParamUnit::Slope if number > 0.0 => 1.0 / number,
            ParamUnit::Slope => return None,
            _ => number,
        };
        value.is_finite().then_some(value)
    }
}

/// Errors reading or setting parameters.
#[derive(Error, Debug, Clone, PartialEq, Eq)]
pub enum ParamError {
    /// No parameter at this index.
    #[error("no parameter at index {0}")]
    UnknownIndex(usize),
    /// No parameter with this id.
    #[error("no parameter with id {0:?}")]
    UnknownId(String),
    /// The value was NaN or infinite (the parameter's id).
    #[error("parameter {0:?} must be finite")]
    NotFinite(&'static str),
    /// A snapshot with the wrong number of values.
    #[error("snapshot has {got} values but there are {expected} parameters")]
    SnapshotLength {
        /// Number of parameters.
        expected: usize,
        /// Values in the snapshot.
        got: usize,
    },
}

/// A processor (or chain) whose settings can be discovered and changed at runtime.
pub trait Parameterized {
    /// Number of parameters.
    fn param_count(&self) -> usize;
    /// Description of parameter `index` (`None` past the last one).
    fn param_info(&self, index: usize) -> Option<ParamInfo>;
    /// Name of the processor that owns the parameter, e.g. "Compressor" (useful in chains).
    fn param_group(&self, index: usize) -> Option<&'static str>;
    /// Current value of parameter `index`, in its own units.
    fn get_param(&self, index: usize) -> Option<f64>;
    /// Validates and clamps `value`, applies it, and returns the value applied.
    fn set_param(&mut self, index: usize, value: f64) -> Result<f64, ParamError>;

    /// Index of the first parameter with this id.
    fn param_index(&self, id: &str) -> Option<usize> {
        (0..self.param_count()).find(|&i| self.param_info(i).is_some_and(|p| p.id == id))
    }
    /// Current value of the parameter with this id.
    fn get_param_by_id(&self, id: &str) -> Option<f64> {
        self.get_param(self.param_index(id)?)
    }
    /// Sets the parameter with this id (see [`set_param`](Self::set_param)).
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
//
// Each processor module declares its own parameters (`<module>/params.rs`), with the macro and the
// conversions below; the channel wrappers' (`Linked`, `PerChannel`) are above, since `channels`
// sits below `params`.

/// A processor's value as the `f64` parameters cross the API in.
pub(crate) fn to_f64<T: Float>(x: T) -> f64 {
    x.to_f64().unwrap_or(f64::NAN)
}

/// A parameter value in the processor's sample type.
pub(crate) fn from_f64<T: Float>(v: f64) -> T {
    T::_lit(v)
}

/// Implements `Parameterized` for a processor from a list of infos (may depend on the instance) and
/// read / write mappings by index. Validation, clamping and the bookkeeping are shared.
macro_rules! parameterized {
    ($Ty:ident, $group:literal,
     infos: |$s:ident| $infos:expr,
     read: |$r:ident, $ri:ident| $read:expr,
     write: |$w:ident, $wi:ident, $wv:ident| $write:expr $(,)?) => {
        impl<T: $crate::units::Float> $crate::params::Parameterized for $Ty<T> {
            fn param_count(&self) -> usize {
                let $s = self;
                $infos.len()
            }
            fn param_info(&self, index: usize) -> Option<$crate::params::ParamInfo> {
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
            fn set_param(&mut self, index: usize, value: f64) -> Result<f64, $crate::params::ParamError> {
                let applied = self.param_info(index).ok_or($crate::params::ParamError::UnknownIndex(index))?.validate(value)?;
                let ($w, $wi, $wv) = (self, index, applied);
                $write;
                Ok(applied)
            }
        }
    };
}
pub(crate) use parameterized;

#[cfg(test)]
mod tests {
    use super::*;
    use super::ParamUnit::{Decibels, Fraction, Hertz, Seconds, Slope};
    use crate::delay::Echo;
    use crate::distortion::Bitcrusher;
    use crate::dynamics::{Compressor, Gate, LookaheadLimiter, MultibandCompressor, TransientShaper};
    use crate::filter::{Biquad, ParametricEq, BUTTERWORTH_Q};
    use crate::gain::{Gain, StereoWidth};
    use crate::modulation::Phaser;
    use crate::osc::Waveform;
    use crate::sampler::{Interpolation, SamplerVoice};
    use crate::spectral::PitchShifter;
    use crate::synth::{FmVoice, Poly, SynthVoice};

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
    fn discrete_kinds_round_format_and_parse() {
        const SHAPES: [&str; 3] = ["Sine", "Saw", "Square"];
        let shape = ParamInfo::choice("shape", "Shape", &SHAPES, 1);
        assert_eq!((shape.min, shape.max, shape.default), (0.0, 2.0, 1.0));
        assert_eq!(shape.validate(1.6), Ok(2.0), "rounds to the nearest option");
        assert_eq!(shape.validate(7.0), Ok(2.0));
        assert_eq!(shape.from_normalized(0.4), 1.0);
        assert_eq!(shape.format(2.0), "Square");
        assert_eq!(shape.parse("saw"), Some(1.0), "by name, any case");
        assert_eq!(shape.parse("2"), Some(2.0), "or by index");
        assert_eq!(shape.parse("Triangle"), None);

        let bypass = ParamInfo::toggle("bypass", "Bypass", false);
        assert_eq!((bypass.validate(0.7), bypass.format(1.0), bypass.format(0.0)), (Ok(1.0), "On".into(), "Off".into()));
        assert_eq!((bypass.parse("ON"), bypass.parse("no"), bypass.parse("maybe")), (Some(1.0), Some(0.0), None));

        let voices = ParamInfo::new("voices", "Voices", ParamUnit::None, 1.0, 16.0, 8.0).integer();
        assert!(voices.is_discrete() && !ParamInfo::new("x", "X", ParamUnit::None, 0.0, 1.0, 0.0).is_discrete());
        assert_eq!((voices.validate(3.4), voices.format(12.0)), (Ok(3.0), "12".into()));
        for info in [shape, bypass, voices] {
            for i in 0..=100 {
                let v = info.from_normalized(i as f64 / 100.0);
                assert_eq!(v, v.round(), "{} stays whole", info.id);
                assert_eq!(info.parse(&info.format(v)), Some(v), "{} text round trip", info.id);
            }
        }
    }

    #[test]
    fn synth_voice_waveform_is_a_choice_that_keeps_pulse_width() {
        let mut voice = SynthVoice::<f32>::new(48_000.0);
        let info = voice.param_info(voice.param_index("waveform").unwrap()).unwrap();
        assert_eq!(info.kind, ParamKind::Choice(&["Sine", "Saw", "Pulse", "Triangle", "Wavetable"]));
        assert_eq!(info.format(voice.get_param_by_id("waveform").unwrap()), "Saw");
        voice.set_param_by_id("pulse_width", 0.25).unwrap(); // remembered while the saw plays
        assert_eq!(voice.set_param_by_id("waveform", 1.8), Ok(2.0));
        assert_eq!(voice.waveform(), Waveform::Pulse { pulse_width: 0.25 });
        voice.set_param_by_id("waveform", 0.0).unwrap();
        voice.set_param_by_id("pulse_width", 0.75).unwrap();
        assert_eq!(voice.waveform(), Waveform::Sine);
        voice.set_param_by_id("waveform", 2.0).unwrap();
        assert_eq!(voice.waveform(), Waveform::Pulse { pulse_width: 0.75 });
        // the last option plays the wavetable; going back restores the classic waveform
        voice.set_param_by_id("waveform", 4.0).unwrap();
        assert!(voice.wavetable_source());
        assert_eq!(voice.get_param_by_id("waveform"), Some(4.0));
        voice.set_param_by_id("waveform", 1.0).unwrap();
        assert!(!voice.wavetable_source() && voice.waveform() == Waveform::Saw);
    }

    #[test]
    fn fm_voice_parameters_cover_every_operator() {
        let mut fm = FmVoice::<f32>::new(48_000.0);
        assert_eq!(fm.param_count(), 2 + 4 * 7);
        let mut ids: Vec<&str> = (0..fm.param_count()).map(|i| fm.param_info(i).unwrap().id).collect();
        ids.sort_unstable();
        ids.dedup();
        assert_eq!(ids.len(), fm.param_count(), "unique ids");
        fm.set_param_by_id("op3_ratio", 2.5).unwrap();
        fm.set_param_by_id("op4_level", 0.5).unwrap();
        fm.set_param_by_id("op2_release_s", 1.25).unwrap();
        fm.set_param_by_id("algorithm", 7.0).unwrap();
        assert_eq!((fm.ratio(2), fm.level(3), fm.env(1).release(), fm.algorithm()), (2.5, 0.5, 1.25, 7));
        let info = fm.param_info(fm.param_index("algorithm").unwrap()).unwrap();
        assert_eq!(info.format(7.0), "1, 2, 3, 4");
        // what the defaults say is what a new voice has
        for i in 0..fm.param_count() {
            let fresh = FmVoice::<f64>::new(48_000.0);
            let (info, value) = (fresh.param_info(i).unwrap(), fresh.get_param(i).unwrap());
            assert!((value - info.default).abs() < 1e-6, "{}: {} vs default {}", info.id, value, info.default);
        }
    }
    #[test]
    fn sampler_voice_parameters() {
        use crate::sampler::{Sample, SampleMap};
        let map = std::sync::Arc::new(SampleMap::single(Sample::from_mono(vec![0.5f64; 100], 48_000.0)));
        let mut v = SamplerVoice::new(map.clone(), 48_000.0);
        for i in 0..v.param_count() {
            let info = v.param_info(i).unwrap();
            assert!((v.get_param(i).unwrap() - info.default).abs() < 1e-9, "{} default", info.id);
        }
        v.set_param_by_id("interpolation", 0.0).unwrap();
        v.set_param_by_id("tune", -7.5).unwrap();
        v.set_param_by_id("amp_release_s", 2.0).unwrap();
        assert_eq!((v.interpolation(), v.tune(), v.amp_env().release()), (Interpolation::Linear, -7.5, 2.0));
        let tune = v.param_info(v.param_index("tune").unwrap()).unwrap();
        assert_eq!(tune.format(-7.5), "-7.50 st");
        assert_eq!(tune.parse("12 st"), Some(12.0));
        let poly = Poly::new(4, 64, |_| SamplerVoice::new(map.clone(), 48_000.0));
        // the voice's parameters, then voice mode, bend range and MPE
        assert_eq!(poly.param_count(), v.param_count() + 3);
    }
    #[test]
    fn parse_inverts_format() {
        let freq = ParamInfo::new("f", "Freq", Hertz, 20.0, 20_000.0, 1_000.0);
        let time = ParamInfo::new("t", "Time", Seconds, 0.0, 10.0, 0.1);
        let mix = ParamInfo::new("m", "Mix", Fraction, 0.0, 1.0, 0.5);
        let slope = ParamInfo::new("s", "Slope", Slope, 0.0, 1.0, 0.25);
        let gain = ParamInfo::new("g", "Gain", Decibels, -60.0, 12.0, 0.0);
        assert_eq!(freq.parse("1.2 kHz"), Some(1200.0));
        assert_eq!(freq.parse("440"), Some(440.0));
        assert_eq!(time.parse("250 ms"), Some(0.25));
        assert_eq!(time.parse("2 s"), Some(2.0));
        assert_eq!(mix.parse("30%"), Some(0.3));
        assert_eq!(slope.parse("4:1"), Some(0.25));
        assert_eq!(slope.parse("∞:1"), Some(0.0));
        assert_eq!(slope.parse("0:1"), None);
        assert_eq!(gain.parse("-18dB"), Some(-18.0));
        assert_eq!(gain.parse("loud"), None);
        // text -> value -> text is stable, including around the unit switches
        for info in [freq, time, mix, slope, gain] {
            for i in 0..=2000 {
                let v = info.from_normalized(i as f64 / 2000.0);
                let v = if info.unit == Hertz { 999.9 + i as f64 * 1e-4 } else { v };
                let text = info.format(v);
                let back = info.parse(&text).unwrap_or_else(|| panic!("{} can't parse {text:?}", info.id));
                assert_eq!(info.format(back), text, "{} {v}", info.id);
            }
        }
        assert_eq!(time.format(0.99996), "1.00 s");
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

    /// Every parameter of a freshly built processor reads its declared default, and survives a
    /// snapshot / restore.
    fn assert_defaults(name: &str, p: &mut impl Parameterized) {
        for i in 0..p.param_count() {
            let info = p.param_info(i).unwrap();
            let value = p.get_param(i).unwrap();
            assert!((value - info.default).abs() < 1e-9 * info.default.abs().max(1.0), "{name} {}: {value} vs default {}", info.id, info.default);
        }
        let snapshot = p.snapshot();
        p.restore(&snapshot).unwrap();
        assert_eq!(p.snapshot(), snapshot, "{name} round trip");
    }

    #[test]
    fn effects_start_at_their_defaults() {
        assert_defaults("limiter", &mut LookaheadLimiter::<f64>::new(2, FS));
        assert_defaults("gate", &mut Gate::<f64>::new(FS));
        assert_defaults("transient shaper", &mut TransientShaper::<f64>::new(FS));
        assert_defaults("bitcrusher", &mut Bitcrusher::<f64>::new(FS));
        assert_defaults("pitch shifter", &mut PitchShifter::<f64>::new());
        for bands in [1, 4, 6, 8] {
            assert_defaults("parametric EQ", &mut ParametricEq::<f64>::new(bands, FS));
        }
        for bands in 2..=4 {
            assert_defaults("multiband", &mut MultibandCompressor::<f64>::new(2, bands, FS));
        }
        // the gate's top ratio is an infinite one
        let mut gate = Gate::<f64>::new(FS);
        gate.set_param_by_id("ratio", 4.0).unwrap();
        assert_eq!(gate.ratio(), 4.0);
        gate.set_param_by_id("ratio", 100.0).unwrap();
        assert!(gate.ratio().is_infinite());
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
        let mut bq = Biquad::lowpass(1_000.0, BUTTERWORTH_Q, FS);
        bq.set_param_by_id("frequency_hz", 2_000.0).unwrap();
        assert!((bq.magnitude_at(2_000.0, FS) - BUTTERWORTH_Q).abs() < 1e-9, "-3 dB moved to 2 kHz");
        assert_eq!(Biquad::new(*bq.coeffs()).param_count(), 0);
    }

    #[test]
    fn chains_concatenate_and_group() {
        let mut chain = (Gain::new(1.0, 0.0, FS), Compressor::new(FS), Biquad::peaking(1_000.0, 1.0, 3.0, FS));
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
