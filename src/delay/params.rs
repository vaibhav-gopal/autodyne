//! The host-facing parameters ([`Parameterized`](crate::params::Parameterized)) of the echo.

use super::*;
use crate::params::{from_f64, parameterized, to_f64, ParamInfo};
use crate::params::ParamUnit::{Fraction, Seconds};

parameterized!(Echo, "Echo",
    infos: |s| [
        ParamInfo::new("delay_s", "Delay", Seconds, 1.0 / to_f64(s.sample_rate()), to_f64(s.max_delay_seconds()), (0.3f64).min(to_f64(s.max_delay_seconds()))).log().smoothed_internally(),
        ParamInfo::new("feedback", "Feedback", Fraction, 0.0, 0.99, 0.5).smoothed_internally(),
        ParamInfo::new("mix", "Mix", Fraction, 0.0, 1.0, 0.5).smoothed_internally(),
    ],
    read: |p, i| match i { 0 => to_f64(p.delay_seconds()), 1 => to_f64(p.feedback()), _ => to_f64(p.mix()) },
    write: |p, i, v| match i { 0 => p.set_delay_seconds(from_f64(v)), 1 => p.set_feedback(from_f64(v)), _ => p.set_mix(from_f64(v)) },
);
