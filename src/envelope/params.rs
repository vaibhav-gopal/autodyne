//! The host-facing parameters ([`Parameterized`](crate::params::Parameterized)) of the ADSR envelope.

use super::*;
use crate::params::{from_f64, parameterized, to_f64, ParamInfo};
use crate::params::ParamUnit::{Fraction, Seconds};

parameterized!(Adsr, "ADSR",
    infos: |_s| [
        ParamInfo::new("attack_s", "Attack", Seconds, 0.0, 10.0, 0.005),
        ParamInfo::new("decay_s", "Decay", Seconds, 0.0, 10.0, 0.1),
        ParamInfo::new("sustain", "Sustain", Fraction, 0.0, 1.0, 0.7),
        ParamInfo::new("release_s", "Release", Seconds, 0.0, 20.0, 0.3),
    ],
    read: |p, i| match i { 0 => to_f64(p.attack()), 1 => to_f64(p.decay()), 2 => to_f64(p.sustain()), _ => to_f64(p.release()) },
    write: |p, i, v| match i { 0 => p.set_attack(from_f64(v)), 1 => p.set_decay(from_f64(v)), 2 => p.set_sustain(from_f64(v)), _ => p.set_release(from_f64(v)) },
);
