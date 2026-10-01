//! autodyne's polyphonic subtractive synth as an instrument plugin.
//!
//! ```text
//! cargo xtask bundle autodyne-synth --release   # target/bundled: .clap and .vst3
//! ```
//!
//! Signal path: MIDI -> 16-voice `Poly<SynthVoice>` (up to 7 detuned oscillators -> state-variable
//! or ladder filter swept by its envelope, pressure and timbre -> ADSR; poly, mono or legato with
//! glide) -> level -> FDN reverb. Notes land on their exact sample; the sustain pedal,
//! all-notes-off, pitch bend, aftertouch, MPE and CLAP per-note expressions (pressure, tuning,
//! brightness) are handled. An LFO (free or locked to the host's tempo) modulates cutoff, pulse width
//! and wavetable position through a modulation matrix. Every parameter comes from the chain's `Parameterized`
//! implementation through `ParamBridge`, grouped by stage, and host automation ramps through
//! `Smoothed`.

use std::sync::Arc;

use autodyne::control::{Lfo, Modulated, Route, Transport};
use autodyne::gain::Gain;
use autodyne::params::{Parameterized, Smoothed};
use autodyne::reverb::Reverb;
use autodyne::synth::{MidiMessage, Poly, SynthVoice};
use autodyne_plug::ParamBridge;
use nice_plug::midi::{Channel, Key};
use nice_plug::prelude::*;

const VOICES: usize = 16;
/// How long parameter changes take to glide to their new value.
const RAMP_SECONDS: f64 = 0.02;
/// Voices render in pieces of at most this many samples, whatever the host's buffer size.
const VOICE_BLOCK: usize = 256;

/// Modulation source index of the LFO.
const LFO: usize = 0;

/// The voices, with modulation routes into their parameters.
pub type Voices = Modulated<Poly<SynthVoice<f32>>>;

/// The whole instrument, as one parameterized chain: LFO, voices, output level, reverb.
pub type Patch = (Lfo, Voices, Gain<f32>, Reverb<f32>);

/// The patch at `sample_rate`, with its default sound (modulation depths at zero).
pub fn patch(sample_rate: f32) -> Patch {
    let mut poly = Poly::new(VOICES, VOICE_BLOCK, |_| SynthVoice::new(sample_rate));
    for (id, value) in [("cutoff_hz", 700.0), ("resonance", 0.35), ("env_amount", 2.5), ("amp_release_s", 0.4)] {
        poly.set_param_by_id(id, value).expect("known parameter");
    }
    let mut voices = Modulated::new(poly, 1, 3);
    let routes = [("cutoff_hz", "LFO > cutoff"), ("pulse_width", "LFO > pulse width"), ("wt_position", "LFO > wavetable position")];
    for (slot, (id, name)) in routes.into_iter().enumerate() {
        let destination = voices.param_index(id).expect("known parameter");
        voices.set_route(slot, Some(Route { source: LFO, destination, via: None })).expect("valid route");
        voices.set_depth_name(slot, name).expect("valid slot");
    }
    let mut reverb = Reverb::new(sample_rate);
    reverb.set_decay(2.2);
    reverb.set_mix(0.2);
    // several voices sum: start at -14 dB for headroom, ramping level changes over 20 ms
    (Lfo::new(sample_rate as f64), voices, Gain::new(0.2, 0.02, sample_rate), reverb)
}

/// Renders the voices into `out`: parameter ramps (`Smoothed::run`) and LFO modulation
/// (`Modulated::run`) advance as they go, and `transport` moves with the rendered samples.
pub fn render_voices(patch: &mut Smoothed<Patch>, out: &mut [f32], transport: &mut Transport, sample_rate: f64) {
    patch.run(out.len(), |p, range| {
        let (lfo, voices, _, _) = p;
        let part = &mut out[range];
        voices.run(
            part.len(),
            |sources, n| {
                sources[LFO] = lfo.advance(n, Some(transport));
                transport.advance(n, sample_rate);
            },
            |poly, r| poly.render(&mut part[r]),
        );
    });
}

pub struct AutodyneSynth {
    params: Arc<ParamBridge>,
    patch: Smoothed<Patch>,
    sample_rate: f64,
}

impl Default for AutodyneSynth {
    fn default() -> Self {
        let patch = Smoothed::new(patch(48_000.0), RAMP_SECONDS, 48_000.0);
        Self { params: Arc::new(ParamBridge::new(&patch)), patch, sample_rate: 48_000.0 }
    }
}

impl Plugin for AutodyneSynth {
    const NAME: &'static str = "Autodyne Synth";
    const VENDOR: &'static str = "autodyne";
    const URL: &'static str = "https://github.com/vaibhav-gopal/autodyne";
    const EMAIL: &'static str = "";
    const VERSION: &'static str = env!("CARGO_PKG_VERSION");

    const AUDIO_IO_LAYOUTS: &'static [AudioIOLayout] = &[AudioIOLayout {
        main_input_channels: None,
        main_output_channels: NonZeroU32::new(2),
        ..AudioIOLayout::const_default()
    }];

    // CCs and pitch bend, besides notes
    const MIDI_INPUT: MidiConfig = MidiConfig::MidiCCs;
    const SAMPLE_ACCURATE_AUTOMATION: bool = true;

    type Editor = ();
    type SysExMessage = ();
    type BackgroundTask = ();

    fn params(&self) -> Arc<dyn Params> {
        self.params.clone()
    }

    fn activate(&mut self, _layout: &AudioIOLayout, config: &BufferConfig, _context: &mut impl ActivateContext<Self>) -> bool {
        // allocation is fine here: this runs outside the audio callback
        self.sample_rate = config.sample_rate as f64;
        self.patch = Smoothed::new(patch(config.sample_rate), RAMP_SECONDS, self.sample_rate);
        // the new patch takes every current host setting at once: no glide from the defaults when a
        // preset loads or the sample rate changes (this runs off the audio thread)
        self.params.invalidate();
        self.params.apply(&mut self.patch);
        self.patch.settle();
        true
    }

    fn reset(&mut self) {
        self.patch.settle();
        let (lfo, voices, _, reverb) = self.patch.inner_mut();
        lfo.reset();
        voices.inner_mut().reset();
        reverb.reset();
    }

    fn process(&mut self, buffer: &mut Buffer, _aux: &mut AuxiliaryBuffers, context: &mut impl ProcessContext<Self>) -> ProcessStatus {
        self.params.apply(&mut self.patch);
        let host = context.transport();
        let mut transport = Transport {
            tempo: host.tempo.unwrap_or(120.0),
            beats_per_bar: match (host.time_sig_numerator, host.time_sig_denominator) {
                (Some(n), Some(d)) => Transport::bar_length(n.max(0) as u32, d.max(0) as u32),
                _ => 4.0,
            },
            position: host.pos_beats().unwrap_or(0.0),
            playing: host.playing,
        };
        let [left, right] = buffer.as_slice() else { return ProcessStatus::Normal };

        // render up to each event, apply it, carry on: every event lands on its sample
        let mut pos = 0;
        while let Some(event) = context.next_event() {
            let at = (event.timing() as usize).clamp(pos, left.len());
            render_voices(&mut self.patch, &mut left[pos..at], &mut transport, self.sample_rate);
            pos = at;
            handle(self.patch.inner_mut().1.inner_mut(), event);
        }
        render_voices(&mut self.patch, &mut left[pos..], &mut transport, self.sample_rate);

        let (_, _, level, reverb) = self.patch.inner_mut();
        level.process(left);
        right.copy_from_slice(left);
        reverb.process_stereo(left, right);
        ProcessStatus::Normal
    }
}

/// Applies one host note event to the voices. MIDI stays channel-aware, so MPE controllers shape
/// each note through its member channel (with MPE on); CLAP note expressions arrive per key.
fn handle(voices: &mut Poly<SynthVoice<f32>>, event: NoteEvent<()>) {
    let number = |channel: Channel| if let Channel::Number(n) = channel { n } else { 0 };
    match event {
        NoteEvent::NoteOn { channel, key: Key::Number(note), velocity, .. } => voices.note_on_channel(number(channel), note, velocity),
        NoteEvent::NoteOff { channel, key: Key::Number(note), .. } | NoteEvent::Choke { channel, key: Key::Number(note), .. } => {
            voices.note_off_channel(number(channel), note)
        }
        // a wildcard key means every note: release them all, or silence everything for a choke
        NoteEvent::NoteOff { key: Key::Wildcard, .. } => voices.all_notes_off(),
        NoteEvent::Choke { key: Key::Wildcard, .. } => voices.reset(),
        NoteEvent::PolyPressure { key: Key::Number(note), pressure, .. } => voices.set_note_pressure(note, pressure),
        NoteEvent::PolyTuning { key: Key::Number(note), tuning, .. } => voices.set_note_tuning(note, tuning),
        NoteEvent::PolyBrightness { key: Key::Number(note), brightness, .. } => voices.set_note_timbre(note, brightness),
        // sustain, all notes off, all sound off and timbre (CC 74) are handled as MIDI controllers
        NoteEvent::MidiCC { channel, cc, value, .. } => voices.handle(MidiMessage::ControlChange {
            channel,
            controller: cc,
            value: (value * 127.0).round() as u8,
        }),
        // 0..1 with 0.5 centered
        NoteEvent::MidiPitchBend { channel, value, .. } => voices.set_channel_bend(channel, value * 2.0 - 1.0),
        NoteEvent::MidiChannelPressure { channel, pressure, .. } => voices.set_channel_pressure(channel, pressure),
        _ => {}
    }
}
impl ClapPlugin for AutodyneSynth {
    const CLAP_ID: &'static str = "com.github.vaibhav-gopal.autodyne.synth";
    const CLAP_DESCRIPTION: Option<&'static str> = Some("Polyphonic subtractive synth with an FDN reverb");
    const CLAP_MANUAL_URL: Option<&'static str> = Some(Self::URL);
    const CLAP_SUPPORT_URL: Option<&'static str> = None;
    const CLAP_FEATURES: &'static [ClapFeature] = &[ClapFeature::Instrument, ClapFeature::Synthesizer, ClapFeature::Stereo];
}

nice_export_clap!(AutodyneSynth);

#[cfg(feature = "vst3")]
impl Vst3Plugin for AutodyneSynth {
    const VST3_CLASS_ID: [u8; 16] = *b"AutodyneSynth001";
    const VST3_SUBCATEGORIES: &'static [Vst3SubCategory] = &[Vst3SubCategory::Instrument, Vst3SubCategory::Synth, Vst3SubCategory::Stereo];
}

#[cfg(feature = "vst3")]
nice_export_vst3!(AutodyneSynth);

#[cfg(test)]
mod tests {
    use super::*;
    use autodyne::synth::Voice;
    use nice_plug::midi::VoiceID;

    fn note_on(note: u8) -> NoteEvent<()> {
        NoteEvent::NoteOn { timing: 0, voice_id: VoiceID::Wildcard, channel: Channel::Number(0), key: Key::Number(note), velocity: 0.8 }
    }

    fn note_off(key: Key) -> NoteEvent<()> {
        NoteEvent::NoteOff { timing: 0, voice_id: VoiceID::Wildcard, channel: Channel::Number(0), key, velocity: 0.0 }
    }

    #[test]
    fn exposes_the_whole_chain_with_unique_ids() {
        let synth = AutodyneSynth::default();
        let map = synth.params.param_map();
        assert_eq!(map.len(), synth.patch.param_count());
        let mut ids: Vec<&str> = map.iter().map(|(id, _, _)| id.as_str()).collect();
        ids.sort_unstable();
        ids.dedup();
        assert_eq!(ids.len(), map.len(), "ids must be unique for host state");
        assert!(map.iter().any(|(id, _, group)| id == "cutoff_hz" && group == "Synth voice"));
        assert!(map.iter().any(|(id, _, group)| id == "decay_s" && group == "Reverb"));
    }

    #[test]
    fn notes_pedal_and_bend() {
        let mut voices = patch(48_000.0).1.into_inner();
        handle(&mut voices, note_on(60));
        handle(&mut voices, note_on(64));
        assert_eq!(voices.active_voices(), 2);
        let pedal = |value| NoteEvent::MidiCC { timing: 0, channel: 0, cc: 64, value };
        handle(&mut voices, pedal(1.0));
        handle(&mut voices, note_off(Key::Number(60)));
        let mut out = vec![0.0; 48_000]; // longer than the 0.4 s release
        voices.render(&mut out);
        assert_eq!(voices.active_voices(), 2, "held by the sustain pedal");
        handle(&mut voices, pedal(0.0));
        voices.render(&mut out);
        assert_eq!(voices.active_voices(), 1, "released with the pedal");
        handle(&mut voices, NoteEvent::MidiPitchBend { timing: 0, channel: 0, value: 0.5 }); // centered: no-op
        assert!(out.iter().all(|s| s.is_finite()));
    }

    #[test]
    fn wildcard_keys_mean_every_note() {
        let mut voices = patch(48_000.0).1.into_inner();
        let mut out = vec![0.0; 48_000]; // longer than the 0.4 s release
        [60, 64, 67].into_iter().for_each(|n| handle(&mut voices, note_on(n)));
        handle(&mut voices, note_off(Key::Wildcard));
        voices.render(&mut out);
        assert_eq!(voices.active_voices(), 0, "a wildcard note-off releases every note");

        [60, 64].into_iter().for_each(|n| handle(&mut voices, note_on(n)));
        let choke = NoteEvent::Choke { timing: 0, voice_id: VoiceID::Wildcard, channel: Channel::Wildcard, key: Key::Wildcard };
        handle(&mut voices, choke);
        assert_eq!(voices.active_voices(), 0, "a wildcard choke silences immediately");
    }

    #[test]
    fn a_synced_lfo_sweeps_the_cutoff_while_rendering() {
        let fs = 48_000.0;
        let mut patch = Smoothed::new(patch(fs as f32), RAMP_SECONDS, fs);
        for (id, value) in [("sync", 1.0), ("mod_1_depth", 0.2)] {
            patch.set_param_by_id(id, value).unwrap();
        }
        patch.inner_mut().1.inner_mut().note_on(60, 0.8);
        let mut transport = Transport { playing: true, ..Transport::new(120.0) };
        let cutoff = patch.inner().1.param_index("cutoff_hz").unwrap();
        let mut seen = Vec::new();
        let mut out = vec![0.0f32; 512];
        for _ in 0..47 {
            // ~0.5 s: one quarter-note LFO cycle at 120 BPM
            render_voices(&mut patch, &mut out, &mut transport, fs);
            seen.push(patch.inner().1.modulated_value(cutoff).unwrap());
        }
        let (lo, hi) = seen.iter().fold((f64::MAX, f64::MIN), |(lo, hi), &v| (lo.min(v), hi.max(v)));
        assert!(lo < 600.0 && hi > 800.0, "the cutoff swings around its 700 Hz base: {lo:.0}..{hi:.0}");
        assert!((transport.position - 47.0 * 512.0 / fs * 2.0).abs() < 1e-9, "the transport moved with the audio");
        assert!(out.iter().all(|s| s.is_finite()));
    }

    #[test]
    fn mpe_and_note_expressions_reach_single_notes() {
        let mut voices = patch(48_000.0).1.into_inner();
        voices.set_mpe(true);
        let on = |channel, note| NoteEvent::NoteOn { timing: 0, voice_id: VoiceID::Wildcard, channel: Channel::Number(channel), key: Key::Number(note), velocity: 0.8 };
        handle(&mut voices, on(1, 60));
        handle(&mut voices, on(2, 64));
        // a member channel's pitch bend moves only its note (48 semitones full scale)
        handle(&mut voices, NoteEvent::MidiPitchBend { timing: 0, channel: 1, value: 0.5 + 0.5 / 48.0 });
        // a CLAP tuning expression detunes one key
        let tuning = NoteEvent::PolyTuning { timing: 0, voice_id: VoiceID::Wildcard, channel: Channel::Number(2), key: Key::Number(64), tuning: -0.5 };
        handle(&mut voices, tuning);
        let pitches: Vec<f32> = voices.voices().iter().filter(|v| v.is_active()).map(|v| v.current_pitch()).collect();
        assert!(pitches.iter().any(|p| (p - 61.0).abs() < 1e-3), "{pitches:?}");
        assert!(pitches.iter().any(|p| (p - 63.5).abs() < 1e-3), "{pitches:?}");
    }

    #[test]
    fn random_parameter_values_while_playing_never_panic() {
        // what a host's automation (and clap-validator's parameter fuzzing) does
        let fs = 48_000.0;
        let mut patch = Smoothed::new(patch(fs as f32), RAMP_SECONDS, fs);
        let mut transport = Transport { playing: true, ..Transport::new(120.0) };
        let mut out = vec![0.0f32; 256];
        let mut seed = 0x2545_f491_4f6c_dd1du64;
        let mut random = || {
            seed ^= seed << 13;
            seed ^= seed >> 7;
            seed ^= seed << 17;
            (seed >> 11) as f64 / (1u64 << 53) as f64
        };
        for round in 0..300 {
            for index in 0..patch.param_count() {
                let normalized = random();
                patch.set_normalized(index, normalized).unwrap_or_else(|e| panic!("param {index}: {e}"));
            }
            let voices = patch.inner_mut().1.inner_mut();
            if round % 3 == 0 {
                // the whole key range, on any channel (MPE members included)
                voices.note_on_channel((random() * 16.0) as u8, (random() * 128.0) as u8, 0.8);
            }
            // bends, aftertouch, timbre and per-note tuning to their extremes
            voices.set_channel_bend((random() * 16.0) as u8, random() as f32 * 2.0 - 1.0);
            voices.set_channel_pressure((random() * 16.0) as u8, random() as f32);
            voices.set_channel_timbre((random() * 16.0) as u8, random() as f32);
            voices.set_note_tuning((random() * 128.0) as u8, (random() as f32 * 2.0 - 1.0) * 120.0);
            render_voices(&mut patch, &mut out, &mut transport, fs);
            assert!(out.iter().all(|s| s.is_finite()), "round {round}");
        }
    }
}