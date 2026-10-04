//! The host-facing parameters ([`Parameterized`](crate::params::Parameterized)) of the modulation effects.

use super::*;
use crate::params::{from_f64, parameterized, to_f64, ParamInfo};
use crate::params::ParamUnit::{Fraction, Hertz, Seconds};

parameterized!(ModulatedDelay, "Modulated delay",
    infos: |s| [
        ParamInfo::new("rate_hz", "Rate", Hertz, 0.01, 20.0, 0.5).log(),
        ParamInfo::new("depth_s", "Depth", Seconds, 0.0, to_f64(s.max_depth()), to_f64(s.max_depth()) / 2.0),
        ParamInfo::new("feedback", "Feedback", Fraction, -0.95, 0.95, 0.0),
        ParamInfo::new("mix", "Mix", Fraction, 0.0, 1.0, 0.5).smoothed_internally(),
    ],
    read: |p, i| match i { 0 => to_f64(p.rate()), 1 => to_f64(p.depth()), 2 => to_f64(p.feedback()), _ => to_f64(p.mix()) },
    write: |p, i, v| match i { 0 => p.set_rate(from_f64(v)), 1 => p.set_depth(from_f64(v)), 2 => p.set_feedback(from_f64(v)), _ => p.set_mix(from_f64(v)) },
);

parameterized!(Phaser, "Phaser",
    infos: |s| {
        let top = (0.49 * to_f64(s.sample_rate())).min(20_000.0);
        [
            ParamInfo::new("rate_hz", "Rate", Hertz, 0.01, 20.0, 0.5).log(),
            ParamInfo::new("min_hz", "Sweep low", Hertz, 20.0, top, 200.0).log(),
            ParamInfo::new("max_hz", "Sweep high", Hertz, 20.0, top, 2_000.0f64.min(top)).log(),
            ParamInfo::new("feedback", "Feedback", Fraction, -0.95, 0.95, 0.0),
            ParamInfo::new("mix", "Mix", Fraction, 0.0, 1.0, 0.5).smoothed_internally(),
        ]
    },
    read: |p, i| {
        let (lo, hi) = p.range();
        match i { 0 => to_f64(p.rate()), 1 => to_f64(lo), 2 => to_f64(hi), 3 => to_f64(p.feedback()), _ => to_f64(p.mix()) }
    },
    write: |p, i, v| {
        let (lo, hi) = p.range();
        match i {
            0 => p.set_rate(from_f64(v)),
            // keep low <= high by moving the other end when they cross
            1 => p.set_range(from_f64(v), hi._max(from_f64(v))),
            2 => p.set_range(lo._min(from_f64(v)), from_f64(v)),
            3 => p.set_feedback(from_f64(v)),
            _ => p.set_mix(from_f64(v)),
        }
    },
);
