//! The host-facing parameters ([`Parameterized`](crate::params::Parameterized)) of the synth voices and the polyphonic voice pool.

use super::*;
use crate::osc::Waveform;
use crate::params::{from_f64, parameterized, to_f64, ParamError, ParamInfo, ParamUnit, Parameterized};
use crate::params::ParamUnit::{Cents, Decibels, Fraction, Hertz, Seconds};

parameterized!(SynthVoice, "Synth voice",
    infos: |_s| [
        ParamInfo::choice("waveform", "Waveform", &SynthVoice::<f64>::SOURCE_NAMES, 1),
        ParamInfo::new("pulse_width", "Pulse width", Fraction, 0.05, 0.95, 0.5),
        ParamInfo::new("wt_position", "Wavetable position", Fraction, 0.0, 1.0, 0.0),
        ParamInfo::new("unison", "Unison", ParamUnit::None, 1.0, MAX_UNISON as f64, 1.0).integer(),
        ParamInfo::new("detune_cents", "Detune", Cents, 0.0, 100.0, 15.0),
        ParamInfo::choice("filter", "Filter", &VoiceFilter::NAMES, 0),
        ParamInfo::new("cutoff_hz", "Cutoff", Hertz, 20.0, 20_000.0, 800.0).log(),
        ParamInfo::new("resonance", "Resonance", Fraction, 0.0, 1.0, 0.22),
        ParamInfo::new("drive_db", "Drive (ladder)", Decibels, 0.0, 24.0, 0.0),
        ParamInfo::new("env_amount", "Filter env (octaves)", ParamUnit::None, 0.0, 6.0, 3.0),
        ParamInfo::new("amp_attack_s", "Amp attack", Seconds, 0.0, 10.0, 0.005),
        ParamInfo::new("amp_decay_s", "Amp decay", Seconds, 0.0, 10.0, 0.3),
        ParamInfo::new("amp_sustain", "Amp sustain", Fraction, 0.0, 1.0, 0.6),
        ParamInfo::new("amp_release_s", "Amp release", Seconds, 0.0, 20.0, 0.3),
        ParamInfo::new("filter_attack_s", "Filter attack", Seconds, 0.0, 10.0, 0.002),
        ParamInfo::new("filter_decay_s", "Filter decay", Seconds, 0.0, 10.0, 0.25),
        ParamInfo::new("filter_sustain", "Filter sustain", Fraction, 0.0, 1.0, 0.2),
        ParamInfo::new("filter_release_s", "Filter release", Seconds, 0.0, 20.0, 0.3),
        ParamInfo::new("glide_s", "Glide", Seconds, 0.0, 5.0, 0.0),
        ParamInfo::new("pressure_octaves", "Pressure > cutoff (octaves)", ParamUnit::None, 0.0, 4.0, 1.0),
        ParamInfo::new("timbre_octaves", "Timbre > cutoff (octaves)", ParamUnit::None, 0.0, 4.0, 1.0),
    ],
    read: |p, i| match i {
        0 => if p.wavetable_source() { 4.0 } else { p.waveform().index() as f64 },
        1 => to_f64(p.pulse_width()),
        2 => to_f64(p.wavetable_position()),
        3 => p.unison() as f64,
        4 => to_f64(p.detune()),
        5 => VoiceFilter::ALL.iter().position(|&v| v == p.filter()).unwrap_or(0) as f64,
        6 => to_f64(p.cutoff()),
        7 => to_f64(p.resonance()),
        8 => to_f64(p.drive_db()),
        9 => to_f64(p.env_amount()),
        10 => to_f64(p.amp_env().attack()),
        11 => to_f64(p.amp_env().decay()),
        12 => to_f64(p.amp_env().sustain()),
        13 => to_f64(p.amp_env().release()),
        14 => to_f64(p.filter_env().attack()),
        15 => to_f64(p.filter_env().decay()),
        16 => to_f64(p.filter_env().sustain()),
        17 => to_f64(p.filter_env().release()),
        18 => to_f64(p.glide()),
        19 => to_f64(p.pressure_amount()),
        _ => to_f64(p.timbre_amount()),
    },
    write: |p, i, v| match i {
        0 => {
            // the classic waveforms, or (the last option) the wavetable
            p.set_wavetable_source(v as usize == SynthVoice::<f64>::SOURCE_NAMES.len() - 1);
            if !p.wavetable_source() {
                p.set_waveform(Waveform::from_index(v as usize, p.pulse_width()));
            }
        }
        1 => p.set_pulse_width(from_f64(v)),
        2 => p.set_wavetable_position(from_f64(v)),
        3 => p.set_unison(v as usize),
        4 => p.set_detune(from_f64(v)),
        5 => p.set_filter(VoiceFilter::ALL[v as usize]),
        6 => p.set_cutoff(from_f64(v)),
        7 => p.set_resonance(from_f64(v)),
        8 => p.set_drive_db(from_f64(v)),
        9 => p.set_env_amount(from_f64(v)),
        10 => p.amp_env_mut().set_attack(from_f64(v)),
        11 => p.amp_env_mut().set_decay(from_f64(v)),
        12 => p.amp_env_mut().set_sustain(from_f64(v)),
        13 => p.amp_env_mut().set_release(from_f64(v)),
        14 => p.filter_env_mut().set_attack(from_f64(v)),
        15 => p.filter_env_mut().set_decay(from_f64(v)),
        16 => p.filter_env_mut().set_sustain(from_f64(v)),
        17 => p.filter_env_mut().set_release(from_f64(v)),
        18 => p.set_glide(from_f64(v)),
        19 => p.set_pressure_amount(from_f64(v)),
        _ => p.set_timbre_amount(from_f64(v)),
    },
);

parameterized!(FmVoice, "FM voice",
    infos: |_s| [
        ParamInfo::choice("algorithm", "Algorithm", &Algorithm::NAMES, 4),
        ParamInfo::new("feedback", "Feedback (op 4)", Fraction, 0.0, 1.0, 0.0),
        ParamInfo::new("op1_ratio", "Op 1 ratio", ParamUnit::None, 0.25, 16.0, 1.0).log(),
        ParamInfo::new("op1_detune", "Op 1 detune", Cents, -50.0, 50.0, 0.0),
        ParamInfo::new("op1_level", "Op 1 level", Fraction, 0.0, 1.0, 1.0),
        ParamInfo::new("op1_attack_s", "Op 1 attack", Seconds, 0.0, 10.0, 0.002),
        ParamInfo::new("op1_decay_s", "Op 1 decay", Seconds, 0.0, 10.0, 1.6),
        ParamInfo::new("op1_sustain", "Op 1 sustain", Fraction, 0.0, 1.0, 0.0),
        ParamInfo::new("op1_release_s", "Op 1 release", Seconds, 0.0, 20.0, 0.5),
        ParamInfo::new("op2_ratio", "Op 2 ratio", ParamUnit::None, 0.25, 16.0, 1.0).log(),
        ParamInfo::new("op2_detune", "Op 2 detune", Cents, -50.0, 50.0, 0.0),
        ParamInfo::new("op2_level", "Op 2 level", Fraction, 0.0, 1.0, 0.3),
        ParamInfo::new("op2_attack_s", "Op 2 attack", Seconds, 0.0, 10.0, 0.001),
        ParamInfo::new("op2_decay_s", "Op 2 decay", Seconds, 0.0, 10.0, 0.9),
        ParamInfo::new("op2_sustain", "Op 2 sustain", Fraction, 0.0, 1.0, 0.0),
        ParamInfo::new("op2_release_s", "Op 2 release", Seconds, 0.0, 20.0, 0.3),
        ParamInfo::new("op3_ratio", "Op 3 ratio", ParamUnit::None, 0.25, 16.0, 1.0).log(),
        ParamInfo::new("op3_detune", "Op 3 detune", Cents, -50.0, 50.0, 0.0),
        ParamInfo::new("op3_level", "Op 3 level", Fraction, 0.0, 1.0, 0.6),
        ParamInfo::new("op3_attack_s", "Op 3 attack", Seconds, 0.0, 10.0, 0.002),
        ParamInfo::new("op3_decay_s", "Op 3 decay", Seconds, 0.0, 10.0, 2.2),
        ParamInfo::new("op3_sustain", "Op 3 sustain", Fraction, 0.0, 1.0, 0.0),
        ParamInfo::new("op3_release_s", "Op 3 release", Seconds, 0.0, 20.0, 0.6),
        ParamInfo::new("op4_ratio", "Op 4 ratio", ParamUnit::None, 0.25, 16.0, 14.0).log(),
        ParamInfo::new("op4_detune", "Op 4 detune", Cents, -50.0, 50.0, 0.0),
        ParamInfo::new("op4_level", "Op 4 level", Fraction, 0.0, 1.0, 0.07),
        ParamInfo::new("op4_attack_s", "Op 4 attack", Seconds, 0.0, 10.0, 0.001),
        ParamInfo::new("op4_decay_s", "Op 4 decay", Seconds, 0.0, 10.0, 0.15),
        ParamInfo::new("op4_sustain", "Op 4 sustain", Fraction, 0.0, 1.0, 0.0),
        ParamInfo::new("op4_release_s", "Op 4 release", Seconds, 0.0, 20.0, 0.1),
    ],
    read: |p, i| match i {
        0 => p.algorithm() as f64,
        1 => to_f64(p.feedback()),
        _ => {
            let (op, field) = ((i - 2) / 7, (i - 2) % 7);
            match field {
                0 => to_f64(p.ratio(op)),
                1 => to_f64(p.detune(op)),
                2 => to_f64(p.level(op)),
                3 => to_f64(p.env(op).attack()),
                4 => to_f64(p.env(op).decay()),
                5 => to_f64(p.env(op).sustain()),
                _ => to_f64(p.env(op).release()),
            }
        }
    },
    write: |p, i, v| match i {
        0 => p.set_algorithm(v as usize),
        1 => p.set_feedback(from_f64(v)),
        _ => {
            let (op, field) = ((i - 2) / 7, (i - 2) % 7);
            match field {
                0 => p.set_ratio(op, from_f64(v)),
                1 => p.set_detune(op, from_f64(v)),
                2 => p.set_level(op, from_f64(v)),
                3 => p.env_mut(op).set_attack(from_f64(v)),
                4 => p.env_mut(op).set_decay(from_f64(v)),
                5 => p.env_mut(op).set_sustain(from_f64(v)),
                _ => p.env_mut(op).set_release(from_f64(v)),
            }
        }
    },
);

/// The voices' parameters (one set controlling every voice), followed by the pool's own: voice mode,
/// pitch bend range and MPE.
impl<V: Voice + Parameterized> Parameterized for Poly<V> {
    fn param_count(&self) -> usize {
        self.voices().first().map_or(0, Parameterized::param_count) + POLY_PARAMS
    }
    fn param_info(&self, index: usize) -> Option<ParamInfo> {
        let voice = self.voices().first()?;
        match index.checked_sub(voice.param_count()) {
            None => voice.param_info(index),
            Some(0) => Some(ParamInfo::choice("voice_mode", "Voice mode", &VoiceMode::NAMES, 0)),
            Some(1) => Some(ParamInfo::new("bend_range", "Bend range", ParamUnit::None, 0.0, 48.0, 2.0).integer()),
            Some(2) => Some(ParamInfo::toggle("mpe", "MPE", false)),
            Some(_) => None,
        }
    }
    fn param_group(&self, index: usize) -> Option<&'static str> {
        let voice = self.voices().first()?;
        if index < voice.param_count() {
            voice.param_group(index)
        } else {
            (index < self.param_count()).then_some("Polyphony")
        }
    }
    fn get_param(&self, index: usize) -> Option<f64> {
        let voice = self.voices().first()?;
        match index.checked_sub(voice.param_count()) {
            None => voice.get_param(index),
            Some(0) => Some(VoiceMode::ALL.iter().position(|&m| m == self.mode()).unwrap_or(0) as f64),
            Some(1) => Some(to_f64(self.bend_range())),
            Some(2) => Some(f64::from(u8::from(self.mpe()))),
            Some(_) => None,
        }
    }
    fn set_param(&mut self, index: usize, value: f64) -> Result<f64, ParamError> {
        let voice_params = self.voices().first().map_or(0, Parameterized::param_count);
        if index < voice_params {
            let mut applied = Err(ParamError::UnknownIndex(index));
            for v in self.voices_mut() {
                applied = Ok(v.set_param(index, value)?);
            }
            return applied;
        }
        let applied = self.param_info(index).ok_or(ParamError::UnknownIndex(index))?.validate(value)?;
        match index - voice_params {
            0 => self.set_mode(VoiceMode::ALL[applied as usize]),
            1 => self.set_bend_range(from_f64(applied)),
            _ => self.set_mpe(applied >= 0.5),
        }
        Ok(applied)
    }
}

/// Parameters `Poly` adds after its voices' own.
const POLY_PARAMS: usize = 3;
