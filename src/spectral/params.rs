//! The host-facing parameters ([`Parameterized`](crate::params::Parameterized)) of the pitch shifter.

use super::*;
use crate::params::{from_f64, parameterized, to_f64, ParamInfo};
use crate::params::ParamUnit::Semitones;

parameterized!(PitchShifter, "Pitch shifter",
    infos: |_s| [ParamInfo::new("semitones", "Shift", Semitones, -24.0, 24.0, 0.0)],
    read: |p, _i| to_f64(p.semitones()),
    write: |p, _i, v| p.set_semitones(from_f64(v)),
);
