//! The host-facing parameters ([`Parameterized`](crate::params::Parameterized)) of the sampler voice.

use super::*;
use crate::params::{from_f64, parameterized, to_f64, ParamInfo};
use crate::params::ParamUnit::{Fraction, Seconds, Semitones};

parameterized!(SamplerVoice, "Sampler voice",
    infos: |_s| [
        ParamInfo::choice("interpolation", "Interpolation", &Interpolation::NAMES, 2),
        ParamInfo::new("tune", "Tune", Semitones, -24.0, 24.0, 0.0),
        ParamInfo::new("start", "Sample start", Fraction, 0.0, 1.0, 0.0),
        ParamInfo::new("velocity_sens", "Velocity sensitivity", Fraction, 0.0, 1.0, 1.0),
        ParamInfo::new("amp_attack_s", "Amp attack", Seconds, 0.0, 10.0, 0.001),
        ParamInfo::new("amp_decay_s", "Amp decay", Seconds, 0.0, 10.0, 0.0),
        ParamInfo::new("amp_sustain", "Amp sustain", Fraction, 0.0, 1.0, 1.0),
        ParamInfo::new("amp_release_s", "Amp release", Seconds, 0.0, 20.0, 0.25),
    ],
    read: |p, i| match i {
        0 => Interpolation::ALL.iter().position(|&m| m == p.interpolation()).unwrap_or(0) as f64,
        1 => to_f64(p.tune()),
        2 => to_f64(p.start()),
        3 => to_f64(p.velocity_sensitivity()),
        4 => to_f64(p.amp_env().attack()),
        5 => to_f64(p.amp_env().decay()),
        6 => to_f64(p.amp_env().sustain()),
        _ => to_f64(p.amp_env().release()),
    },
    write: |p, i, v| match i {
        0 => p.set_interpolation(Interpolation::ALL[v as usize]),
        1 => p.set_tune(from_f64(v)),
        2 => p.set_start(from_f64(v)),
        3 => p.set_velocity_sensitivity(from_f64(v)),
        4 => p.amp_env_mut().set_attack(from_f64(v)),
        5 => p.amp_env_mut().set_decay(from_f64(v)),
        6 => p.amp_env_mut().set_sustain(from_f64(v)),
        _ => p.amp_env_mut().set_release(from_f64(v)),
    },
);
