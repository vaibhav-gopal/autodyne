//! The host-facing parameters ([`Parameterized`](crate::params::Parameterized)) of the distortion effects.

use super::*;
use crate::params::{from_f64, parameterized, to_f64, ParamInfo, ParamUnit};
use crate::params::ParamUnit::{Decibels, Fraction, Hertz};

parameterized!(Waveshaper, "Waveshaper",
    infos: |_s| [
        ParamInfo::new("drive_db", "Drive", Decibels, 0.0, 48.0, 0.0).smoothed_internally(),
        ParamInfo::new("output_db", "Output", Decibels, -24.0, 24.0, 0.0).smoothed_internally(),
        ParamInfo::new("mix", "Mix", Fraction, 0.0, 1.0, 1.0).smoothed_internally(),
    ],
    read: |p, i| match i { 0 => to_f64(p.drive_db()), 1 => to_f64(p.output_db()), _ => to_f64(p.mix()) },
    write: |p, i, v| match i { 0 => p.set_drive_db(from_f64(v)), 1 => p.set_output_db(from_f64(v)), _ => p.set_mix(from_f64(v)) },
);

parameterized!(Bitcrusher, "Bitcrusher",
    infos: |s| [
        ParamInfo::new("bits", "Bits", ParamUnit::None, 1.0, 24.0, 8.0),
        ParamInfo::new("rate_hz", "Rate", Hertz, 100.0, to_f64(s.sample_rate()), to_f64(s.sample_rate()) / 4.0).log(),
        ParamInfo::toggle("dither", "Dither", false),
        ParamInfo::new("mix", "Mix", Fraction, 0.0, 1.0, 1.0).smoothed_internally(),
    ],
    read: |p, i| match i { 0 => to_f64(p.bits()), 1 => to_f64(p.rate()), 2 => p.dither() as u8 as f64, _ => to_f64(p.mix()) },
    write: |p, i, v| match i { 0 => p.set_bits(from_f64(v)), 1 => p.set_rate(from_f64(v)), 2 => p.set_dither(v >= 0.5), _ => p.set_mix(from_f64(v)) },
);
