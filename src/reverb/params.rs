//! The host-facing parameters ([`Parameterized`](crate::params::Parameterized)) of the reverbs.

use super::*;
use crate::params::{from_f64, parameterized, to_f64, ParamInfo, ParamUnit};
use crate::params::ParamUnit::{Fraction, Hertz, Seconds};

parameterized!(Reverb, "Reverb",
    infos: |_s| [
        ParamInfo::new("size", "Size", ParamUnit::None, 0.25, 2.0, 1.0).instant(),
        ParamInfo::new("decay_s", "Decay", Seconds, 0.05, 30.0, 1.8).log(),
        ParamInfo::new("damping_hz", "Damping", Hertz, 500.0, 24_000.0, 6_000.0).log(),
        ParamInfo::new("predelay_s", "Pre-delay", Seconds, 0.0, 0.25, 0.02).instant(),
        ParamInfo::new("mix", "Mix", Fraction, 0.0, 1.0, 0.3).smoothed_internally(),
        ParamInfo::new("width", "Width", Fraction, 0.0, 1.0, 1.0).smoothed_internally(),
        ParamInfo::new("modulation", "Modulation", Fraction, 0.0, 1.0, 0.5),
    ],
    read: |p, i| match i {
        0 => to_f64(p.size()),
        1 => to_f64(p.decay()),
        2 => to_f64(p.damping()),
        3 => to_f64(p.predelay()),
        4 => to_f64(p.mix()),
        5 => to_f64(p.width()),
        _ => to_f64(p.modulation()),
    },
    write: |p, i, v| match i {
        0 => p.set_size(from_f64(v)),
        1 => p.set_decay(from_f64(v)),
        2 => p.set_damping(from_f64(v)),
        3 => p.set_predelay(from_f64(v)),
        4 => p.set_mix(from_f64(v)),
        5 => p.set_width(from_f64(v)),
        _ => p.set_modulation(from_f64(v)),
    },
);

parameterized!(Convolver, "Convolver",
    infos: |_s| [ParamInfo::new("mix", "Mix", Fraction, 0.0, 1.0, 1.0).smoothed_internally()],
    read: |p, _i| to_f64(p.mix()),
    write: |p, _i, v| p.set_mix(from_f64(v)),
);
