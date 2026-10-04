//! The host-facing parameters ([`Parameterized`](crate::params::Parameterized)) of the gain stage.

use super::*;
use crate::params::{from_f64, parameterized, to_f64, ParamInfo, ParamUnit};
use crate::params::ParamUnit::{Decibels, Fraction};

parameterized!(Gain, "Gain",
    infos: |_s| [ParamInfo::new("gain_db", "Gain", Decibels, -60.0, 24.0, 0.0).smoothed_internally()],
    read: |p, _i| to_f64(gain_to_db(p.gain())).max(-60.0),
    write: |p, _i, v| p.set_gain_db(from_f64(v)),
);

parameterized!(StereoWidth, "Stereo width",
    infos: |_s| [ParamInfo::new("width", "Width", Fraction, 0.0, 4.0, 1.0).smoothed_internally()],
    read: |p, _i| to_f64(p.width()),
    write: |p, _i, v| p.set_width(from_f64(v)),
);

parameterized!(Panner, "Panner",
    infos: |_s| [ParamInfo::new("position", "Pan", ParamUnit::None, -1.0, 1.0, 0.0).smoothed_internally()],
    read: |p, _i| to_f64(p.position()),
    write: |p, _i, v| p.set_position(from_f64(v)),
);
