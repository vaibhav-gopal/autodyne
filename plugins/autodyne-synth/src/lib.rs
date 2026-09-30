//! autodyne's polyphonic subtractive synth as an instrument plugin.
//!
//! ```text
//! cargo xtask bundle autodyne-synth --release   # target/bundled: .clap and .vst3
//! ```
//!
//! Signal path: MIDI -> 16-voice `Poly<SynthVoice>` (band-limited saw -> enveloped resonant
//! low-pass -> ADSR) -> level -> FDN reverb. Notes land on their exact sample; the sustain pedal,
//! all-notes-off and pitch bend (±2 semitones) are handled. Every parameter comes from the chain's
//! `Parameterized` implementation through `ParamBridge`, grouped by stage.

use std::sync::Arc;

use autodyne::gain::Gain;
use autodyne::params::Parameterized;
use autodyne::reverb::Reverb;
use autodyne::synth::{MidiMessage, Poly, SynthVoice};
use autodyne_nih::ParamBridge;
use nih_plug::prelude::*;

const VOICES: usize = 16;
/// Voices render in pieces of at most this many samples, whatever the host's buffer size.
const VOICE_BLOCK: usize = 256;

/// The whole instrument, as one parameterized chain: voices, output level, reverb.
pub type Patch = (Poly<SynthVoice<f32>>, Gain<f32>, Reverb<f32>);

/// The patch at `sample_rate`, with its default sound.
pub fn patch(sample_rate: f32) -> Patch {
    let mut voices = Poly::new(VOICES, VOICE_BLOCK, |_| SynthVoice::new(sample_rate));
    for (id, value) in [("cutoff_hz", 700.0), ("resonance", 2.0), ("env_amount", 2.5), ("amp_release_s", 0.4)] {
        voices.set_param_by_id(id, value).expect("known parameter");
    }
    let mut reverb = Reverb::new(sample_rate);
    reverb.set_decay(2.2);
    reverb.set_mix(0.2);
    // several voices sum: start at -14 dB for headroom, ramping level changes over 20 ms
    (voices, Gain::new(0.2, 0.02, sample_rate), reverb)
}

pub struct AutodyneSynth {
    params: Arc<ParamBridge>,
    patch: Patch,
}

impl Default for AutodyneSynth {
    fn default() -> Self {
        let patch = patch(48_000.0);
        Self { params: Arc::new(ParamBridge::new(&patch)), patch }
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

    type SysExMessage = ();
    type BackgroundTask = ();

    fn params(&self) -> Arc<dyn Params> {
        self.params.clone()
    }

    fn initialize(&mut self, _layout: &AudioIOLayout, config: &BufferConfig, _context: &mut impl InitContext<Self>) -> bool {
        // allocation is fine here: this runs outside the audio callback
        self.patch = patch(config.sample_rate);
        self.params.invalidate(); // the new patch must receive every current setting
        true
    }

    fn reset(&mut self) {
        self.patch.0.reset();
        self.patch.2.reset();
    }

    fn process(&mut self, buffer: &mut Buffer, _aux: &mut AuxiliaryBuffers, context: &mut impl ProcessContext<Self>) -> ProcessStatus {
        self.params.apply(&mut self.patch);
        let (voices, level, reverb) = &mut self.patch;
        let [left, right] = buffer.as_slice() else { return ProcessStatus::Normal };

        // render up to each event, apply it, carry on: every event lands on its sample
        let mut pos = 0;
        while let Some(event) = context.next_event() {
            let at = (event.timing() as usize).clamp(pos, left.len());
            voices.render(&mut left[pos..at]);
            pos = at;
            handle(voices, event);
        }
        voices.render(&mut left[pos..]);

        level.process(left);
        right.copy_from_slice(left);
        reverb.process_stereo(left, right);
        ProcessStatus::Normal
    }
}

/// Applies one host note event to the voices.
fn handle(voices: &mut Poly<SynthVoice<f32>>, event: NoteEvent<()>) {
    match event {
        NoteEvent::NoteOn { note, velocity, .. } => voices.note_on(note, velocity),
        NoteEvent::NoteOff { note, .. } | NoteEvent::Choke { note, .. } => voices.note_off(note),
        // sustain, all notes off and all sound off are handled as MIDI controllers
        NoteEvent::MidiCC { channel, cc, value, .. } => voices.handle(MidiMessage::ControlChange {
            channel,
            controller: cc,
            value: (value * 127.0).round() as u8,
        }),
        // 0..1 with 0.5 centered
        NoteEvent::MidiPitchBend { value, .. } => voices.set_pitch_bend(value * 2.0 - 1.0),
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

nih_export_clap!(AutodyneSynth);

#[cfg(feature = "vst3")]
impl Vst3Plugin for AutodyneSynth {
    const VST3_CLASS_ID: [u8; 16] = *b"AutodyneSynth001";
    const VST3_SUBCATEGORIES: &'static [Vst3SubCategory] = &[Vst3SubCategory::Instrument, Vst3SubCategory::Synth, Vst3SubCategory::Stereo];
}

#[cfg(feature = "vst3")]
nih_export_vst3!(AutodyneSynth);

#[cfg(test)]
mod tests {
    use super::*;

    fn note_on(note: u8) -> NoteEvent<()> {
        NoteEvent::NoteOn { timing: 0, voice_id: None, channel: 0, note, velocity: 0.8 }
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
        let mut voices = patch(48_000.0).0;
        handle(&mut voices, note_on(60));
        handle(&mut voices, note_on(64));
        assert_eq!(voices.active_voices(), 2);
        let pedal = |value| NoteEvent::MidiCC { timing: 0, channel: 0, cc: 64, value };
        handle(&mut voices, pedal(1.0));
        handle(&mut voices, NoteEvent::NoteOff { timing: 0, voice_id: None, channel: 0, note: 60, velocity: 0.0 });
        let mut out = vec![0.0; 48_000]; // longer than the 0.4 s release
        voices.render(&mut out);
        assert_eq!(voices.active_voices(), 2, "held by the sustain pedal");
        handle(&mut voices, pedal(0.0));
        voices.render(&mut out);
        assert_eq!(voices.active_voices(), 1, "released with the pedal");
        handle(&mut voices, NoteEvent::MidiPitchBend { timing: 0, channel: 0, value: 0.5 }); // centered: no-op
        assert!(out.iter().all(|s| s.is_finite()));
    }
}
