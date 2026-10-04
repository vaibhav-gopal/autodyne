//! The host-facing parameters ([`Parameterized`](crate::params::Parameterized)) of the dynamics processors.

use super::*;
use crate::params::{from_f64, parameterized, to_f64, ParamError, ParamInfo, ParamUnit, Parameterized};
use crate::params::ParamUnit::{Decibels, Hertz, Seconds, Slope};
use crate::units::*;

parameterized!(Compressor, "Compressor",
    infos: |_s| [
        ParamInfo::new("threshold_db", "Threshold", Decibels, -60.0, 0.0, -18.0),
        // the slope (1 / ratio) rather than the ratio: finite for every setting, 0 is a limiter
        ParamInfo::new("slope", "Slope", Slope, 0.0, 1.0, 0.25),
        ParamInfo::new("knee_db", "Knee", Decibels, 0.0, 24.0, 6.0),
        ParamInfo::new("attack_s", "Attack", Seconds, 0.0, 0.5, 0.010),
        ParamInfo::new("release_s", "Release", Seconds, 0.001, 5.0, 0.100).log(),
        ParamInfo::new("makeup_db", "Makeup", Decibels, -24.0, 24.0, 0.0).smoothed_internally(),
    ],
    read: |p, i| match i {
        0 => to_f64(p.threshold_db()),
        1 => to_f64(p.slope()),
        2 => to_f64(p.knee_db()),
        3 => to_f64(p.attack()),
        4 => to_f64(p.release()),
        _ => to_f64(p.makeup_db()),
    },
    write: |p, i, v| match i {
        0 => p.set_threshold_db(from_f64(v)),
        1 => p.set_slope(from_f64(v)),
        2 => p.set_knee_db(from_f64(v)),
        3 => p.set_attack(from_f64(v)),
        4 => p.set_release(from_f64(v)),
        _ => p.set_makeup_db(from_f64(v)),
    },
);

parameterized!(EnvelopeFollower, "Envelope follower",
    infos: |_s| [
        ParamInfo::new("attack_s", "Attack", Seconds, 0.0, 1.0, 0.010),
        ParamInfo::new("release_s", "Release", Seconds, 0.0, 5.0, 0.100),
    ],
    read: |p, i| if i == 0 { to_f64(p.attack()) } else { to_f64(p.release()) },
    write: |p, i, v| if i == 0 { p.set_attack(from_f64(v)) } else { p.set_release(from_f64(v)) },
);

parameterized!(LookaheadLimiter, "Limiter",
    infos: |_s| [
        ParamInfo::new("ceiling_db", "Ceiling", Decibels, -24.0, 0.0, -0.3),
        ParamInfo::new("release_s", "Release", Seconds, 0.001, 2.0, 0.1).log(),
        ParamInfo::toggle("true_peak", "True peak", true),
    ],
    read: |p, i| match i { 0 => to_f64(p.ceiling_db()), 1 => to_f64(p.release()), _ => p.true_peak() as u8 as f64 },
    write: |p, i, v| match i { 0 => p.set_ceiling_db(from_f64(v)), 1 => p.set_release(from_f64(v)), _ => p.set_true_peak(v >= 0.5) },
);

/// Expander ratio at which the gate parameter means "gate" (an infinite ratio).
const GATE_RATIO: f64 = 100.0;

parameterized!(Gate, "Gate",
    infos: |_s| [
        ParamInfo::new("threshold_db", "Threshold", Decibels, -90.0, 0.0, -40.0),
        ParamInfo::new("ratio", "Ratio (100 = gate)", ParamUnit::Ratio, 1.0, GATE_RATIO, GATE_RATIO).log(),
        ParamInfo::new("range_db", "Range", Decibels, -100.0, 0.0, -80.0),
        ParamInfo::new("hysteresis_db", "Hysteresis", Decibels, 0.0, 12.0, 3.0),
        ParamInfo::new("attack_s", "Attack", Seconds, 0.0, 0.1, 0.0005),
        ParamInfo::new("hold_s", "Hold", Seconds, 0.0, 2.0, 0.02),
        ParamInfo::new("release_s", "Release", Seconds, 0.001, 5.0, 0.1).log(),
    ],
    read: |p, i| match i {
        0 => to_f64(p.threshold_db()),
        1 => to_f64(p.ratio()).min(GATE_RATIO),
        2 => to_f64(p.range_db()),
        3 => to_f64(p.hysteresis_db()),
        4 => to_f64(p.attack()),
        5 => to_f64(p.hold()),
        _ => to_f64(p.release()),
    },
    write: |p, i, v| match i {
        0 => p.set_threshold_db(from_f64(v)),
        1 => p.set_ratio(if v >= GATE_RATIO { T::_INFINITY } else { from_f64(v) }),
        2 => p.set_range_db(from_f64(v)),
        3 => p.set_hysteresis_db(from_f64(v)),
        4 => p.set_attack(from_f64(v)),
        5 => p.set_hold(from_f64(v)),
        _ => p.set_release(from_f64(v)),
    },
);

parameterized!(TransientShaper, "Transient shaper",
    infos: |_s| [
        ParamInfo::new("attack", "Attack", ParamUnit::None, -1.0, 1.0, 0.0),
        ParamInfo::new("sustain", "Sustain", ParamUnit::None, -1.0, 1.0, 0.0),
        ParamInfo::new("output_db", "Output", Decibels, -24.0, 24.0, 0.0).smoothed_internally(),
    ],
    read: |p, i| match i { 0 => to_f64(p.attack()), 1 => to_f64(p.sustain()), _ => to_f64(p.output_db()) },
    write: |p, i, v| match i { 0 => p.set_attack(from_f64(v)), 1 => p.set_sustain(from_f64(v)), _ => p.set_output_db(from_f64(v)) },
);

const MULTIBAND_GROUPS: [&str; 5] = ["Crossovers", "Band 1", "Band 2", "Band 3", "Band 4"];

/// The crossover frequencies, then six parameters per band (threshold, slope, attack, release,
/// makeup, bypass), grouped by band.
impl<T: Float> Parameterized for MultibandCompressor<T> {
    fn param_count(&self) -> usize {
        self.bands() - 1 + self.bands() * MULTIBAND_BAND_PARAMS
    }
    fn param_info(&self, index: usize) -> Option<ParamInfo> {
        let splits = self.bands() - 1;
        if index < splits {
            let defaults: &[f64] = match self.bands() {
                2 => &[1_000.0],
                3 => &[200.0, 2_000.0],
                _ => &[120.0, 1_000.0, 6_000.0],
            };
            return Some(ParamInfo::new(CROSSOVER_PARAM_IDS[index], CROSSOVER_PARAM_NAMES[index], Hertz, 20.0, 20_000.0, defaults[index]).log());
        }
        let (b, field) = ((index - splits) / MULTIBAND_BAND_PARAMS, (index - splits) % MULTIBAND_BAND_PARAMS);
        if b >= self.bands() {
            return None;
        }
        let (id, name) = (MULTIBAND_PARAM_IDS[b][field], MULTIBAND_PARAM_NAMES[b][field]);
        Some(match field {
            0 => ParamInfo::new(id, name, Decibels, -60.0, 0.0, -18.0),
            1 => ParamInfo::new(id, name, Slope, 0.0, 1.0, 0.25),
            2 => ParamInfo::new(id, name, Seconds, 0.0, 0.5, 0.010),
            3 => ParamInfo::new(id, name, Seconds, 0.001, 5.0, 0.100).log(),
            4 => ParamInfo::new(id, name, Decibels, -24.0, 24.0, 0.0).smoothed_internally(),
            _ => ParamInfo::toggle(id, name, false),
        })
    }
    fn param_group(&self, index: usize) -> Option<&'static str> {
        let splits = self.bands() - 1;
        (index < self.param_count()).then(|| if index < splits { MULTIBAND_GROUPS[0] } else { MULTIBAND_GROUPS[1 + (index - splits) / MULTIBAND_BAND_PARAMS] })
    }
    fn get_param(&self, index: usize) -> Option<f64> {
        let splits = self.bands() - 1;
        if index < splits {
            return Some(to_f64(self.crossover_frequency(index)));
        }
        let (b, field) = ((index - splits) / MULTIBAND_BAND_PARAMS, (index - splits) % MULTIBAND_BAND_PARAMS);
        if b >= self.bands() {
            return None;
        }
        let c = self.band(b);
        Some(match field {
            0 => to_f64(c.threshold_db()),
            1 => to_f64(c.slope()),
            2 => to_f64(c.attack()),
            3 => to_f64(c.release()),
            4 => to_f64(c.makeup_db()),
            _ => self.bypass(b) as u8 as f64,
        })
    }
    fn set_param(&mut self, index: usize, value: f64) -> Result<f64, ParamError> {
        let v = self.param_info(index).ok_or(ParamError::UnknownIndex(index))?.validate(value)?;
        let splits = self.bands() - 1;
        if index < splits {
            self.set_crossover_frequency(index, from_f64(v));
            return Ok(to_f64(self.crossover_frequency(index)));
        }
        let (b, field) = ((index - splits) / MULTIBAND_BAND_PARAMS, (index - splits) % MULTIBAND_BAND_PARAMS);
        match field {
            0 => self.band_mut(b).set_threshold_db(from_f64(v)),
            1 => self.band_mut(b).set_slope(from_f64(v)),
            2 => self.band_mut(b).set_attack(from_f64(v)),
            3 => self.band_mut(b).set_release(from_f64(v)),
            4 => self.band_mut(b).set_makeup_db(from_f64(v)),
            _ => self.set_bypass(b, v >= 0.5),
        }
        Ok(v)
    }
}
