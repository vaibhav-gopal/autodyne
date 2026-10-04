//! The host-facing parameters ([`Parameterized`](crate::params::Parameterized)) of the filters and the parametric EQ.

use super::*;
use crate::params::{from_f64, parameterized, to_f64, ParamError, ParamInfo, ParamUnit, Parameterized};
use crate::params::ParamUnit::{Decibels, Fraction, Hertz};

const EQ_GROUPS: [&str; 8] = ["EQ band 1", "EQ band 2", "EQ band 3", "EQ band 4", "EQ band 5", "EQ band 6", "EQ band 7", "EQ band 8"];

/// Six parameters per band: on, type, frequency, gain, Q, slope (cut bands), grouped by band.
impl<T: Float> Parameterized for ParametricEq<T> {
    fn param_count(&self) -> usize {
        self.band_count() * EQ_BAND_PARAMS
    }
    fn param_info(&self, index: usize) -> Option<ParamInfo> {
        let (b, field) = (index / EQ_BAND_PARAMS, index % EQ_BAND_PARAMS);
        if b >= self.band_count() {
            return None;
        }
        let (id, name) = (EQ_PARAM_IDS[b][field], EQ_PARAM_NAMES[b][field]);
        let d = default_band(self.band_count(), b);
        Some(match field {
            0 => ParamInfo::toggle(id, name, d.enabled),
            1 => ParamInfo::choice(id, name, &EqBandKind::NAMES, EqBandKind::ALL.iter().position(|&k| k == d.kind).unwrap_or(0)),
            2 => ParamInfo::new(id, name, Hertz, 20.0, 20_000.0, d.frequency).log(),
            3 => ParamInfo::new(id, name, Decibels, -24.0, 24.0, d.gain_db),
            4 => ParamInfo::new(id, name, ParamUnit::None, 0.1, 18.0, d.q).log(),
            _ => ParamInfo::choice(id, name, &EQ_SLOPE_NAMES, d.slope - 1),
        })
    }
    fn param_group(&self, index: usize) -> Option<&'static str> {
        (index < self.param_count()).then(|| EQ_GROUPS[index / EQ_BAND_PARAMS])
    }
    fn get_param(&self, index: usize) -> Option<f64> {
        let (b, field) = (index / EQ_BAND_PARAMS, index % EQ_BAND_PARAMS);
        let band = (b < self.band_count()).then(|| self.band(b))?;
        Some(match field {
            0 => band.enabled as u8 as f64,
            1 => EqBandKind::ALL.iter().position(|&k| k == band.kind).unwrap_or(0) as f64,
            2 => band.frequency,
            3 => band.gain_db,
            4 => band.q,
            _ => (band.slope - 1) as f64,
        })
    }
    fn set_param(&mut self, index: usize, value: f64) -> Result<f64, ParamError> {
        let v = self.param_info(index).ok_or(ParamError::UnknownIndex(index))?.validate(value)?;
        let b = index / EQ_BAND_PARAMS;
        match index % EQ_BAND_PARAMS {
            0 => self.set_enabled(b, v >= 0.5),
            1 => self.set_kind(b, EqBandKind::ALL[v as usize]),
            2 => self.set_frequency(b, v),
            3 => self.set_gain_db(b, v),
            4 => self.set_q(b, v),
            _ => self.set_slope(b, v as usize + 1),
        }
        Ok(v)
    }
}

parameterized!(Svf, "State-variable filter",
    infos: |s| {
        let top = (0.49 * to_f64(s.sample_rate())).min(20_000.0);
        [
            ParamInfo::choice("mode", "Mode", &SvfMode::NAMES, 0),
            ParamInfo::new("cutoff_hz", "Cutoff", Hertz, 20.0, top, 1_000.0f64.min(top)).log(),
            ParamInfo::new("q", "Q", ParamUnit::None, 0.1, 25.0, std::f64::consts::FRAC_1_SQRT_2).log(),
        ]
    },
    read: |p, i| match i {
        0 => SvfMode::ALL.iter().position(|&m| m == p.mode()).unwrap_or(0) as f64,
        1 => to_f64(p.cutoff()),
        _ => to_f64(p.q()),
    },
    write: |p, i, v| match i {
        0 => p.set_mode(SvfMode::ALL[v as usize]),
        1 => p.set_cutoff(from_f64(v)),
        _ => p.set_q(from_f64(v)),
    },
);

parameterized!(Ladder, "Ladder filter",
    infos: |s| {
        let top = (0.49 * to_f64(s.sample_rate())).min(20_000.0);
        [
            ParamInfo::new("cutoff_hz", "Cutoff", Hertz, 20.0, top, 1_000.0f64.min(top)).log(),
            ParamInfo::new("resonance", "Resonance", Fraction, 0.0, 1.0, 0.3),
            ParamInfo::new("drive_db", "Drive", Decibels, 0.0, 24.0, 0.0),
        ]
    },
    read: |p, i| match i { 0 => to_f64(p.cutoff()), 1 => to_f64(p.resonance()), _ => to_f64(gain_to_db(p.drive())) },
    write: |p, i, v| match i { 0 => p.set_cutoff(from_f64(v)), 1 => p.set_resonance(from_f64(v)), _ => p.set_drive(T::_lit(10f64.powf(v / 20.0))) },
);

/// A designed biquad exposes frequency, Q and gain; one built from raw coefficients has no parameters.
impl<T: Float> Parameterized for Biquad<T> {
    fn param_count(&self) -> usize {
        if self.design().is_some() { 3 } else { 0 }
    }
    fn param_info(&self, index: usize) -> Option<ParamInfo> {
        let d = self.design()?;
        let top = 0.49 * to_f64(d.sample_rate);
        [
            ParamInfo::new("frequency_hz", "Frequency", Hertz, 10.0, top, 1_000.0f64.min(top)).log(),
            ParamInfo::new("q", "Q", ParamUnit::None, 0.1, 20.0, std::f64::consts::FRAC_1_SQRT_2).log(),
            ParamInfo::new("gain_db", "Gain", Decibels, -24.0, 24.0, 0.0),
        ]
        .get(index)
        .copied()
    }
    fn param_group(&self, index: usize) -> Option<&'static str> {
        (index < self.param_count()).then_some("Biquad")
    }
    fn get_param(&self, index: usize) -> Option<f64> {
        let d = self.design()?;
        [to_f64(d.frequency), to_f64(d.q), to_f64(d.gain_db)].get(index).copied()
    }
    fn set_param(&mut self, index: usize, value: f64) -> Result<f64, ParamError> {
        let applied = self.param_info(index).ok_or(ParamError::UnknownIndex(index))?.validate(value)?;
        let mut d = *self.design().expect("param_info succeeded, so there is a design");
        match index {
            0 => d.frequency = from_f64(applied),
            1 => d.q = from_f64(applied),
            _ => d.gain_db = from_f64(applied),
        }
        self.set_design(d);
        Ok(applied)
    }
}

/// One set of parameters (frequency, Q, gain) applied to every channel's filter.
impl<T: Float> Parameterized for MultiBiquad<T> {
    fn param_count(&self) -> usize {
        self.channels().first().map_or(0, Parameterized::param_count)
    }
    fn param_info(&self, index: usize) -> Option<ParamInfo> {
        self.channels().first()?.param_info(index)
    }
    fn param_group(&self, index: usize) -> Option<&'static str> {
        (index < self.param_count()).then_some("Multichannel biquad")
    }
    fn get_param(&self, index: usize) -> Option<f64> {
        self.channels().first()?.get_param(index)
    }
    fn set_param(&mut self, index: usize, value: f64) -> Result<f64, ParamError> {
        let mut applied = Err(ParamError::UnknownIndex(index));
        for ch in 0..self.channels().len() {
            applied = Ok(self.channel_mut(ch).set_param(index, value)?);
        }
        applied
    }
}

/// FIR taps are data rather than settings, so a FIR has no parameters.
impl<T: Float> Parameterized for Fir<T> {
    fn param_count(&self) -> usize {
        0
    }
    fn param_info(&self, _index: usize) -> Option<ParamInfo> {
        None
    }
    fn param_group(&self, _index: usize) -> Option<&'static str> {
        None
    }
    fn get_param(&self, _index: usize) -> Option<f64> {
        None
    }
    fn set_param(&mut self, index: usize, _value: f64) -> Result<f64, ParamError> {
        Err(ParamError::UnknownIndex(index))
    }
}
