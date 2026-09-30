//! autodyne's 8-line feedback-delay-network reverb as a plugin.
//!
//! ```text
//! cargo xtask bundle autodyne-reverb --release   # target/bundled: .clap and .vst3
//! ```
//!
//! Parameters come straight from `Reverb`'s `Parameterized` implementation through `ParamBridge`.

use std::sync::Arc;

use autodyne::reverb::Reverb;
use autodyne_nih::ParamBridge;
use nih_plug::prelude::*;

pub struct AutodyneReverb {
    params: Arc<ParamBridge>,
    reverb: Reverb<f32>,
    /// the second channel when the host runs the plugin in mono
    mono_scratch: Vec<f32>,
    sample_rate: f32,
}

impl Default for AutodyneReverb {
    fn default() -> Self {
        let reverb = Reverb::new(48_000.0);
        Self { params: Arc::new(ParamBridge::new(&reverb)), reverb, mono_scratch: Vec::new(), sample_rate: 48_000.0 }
    }
}

impl Plugin for AutodyneReverb {
    const NAME: &'static str = "Autodyne Reverb";
    const VENDOR: &'static str = "autodyne";
    const URL: &'static str = "https://github.com/vaibhav-gopal/autodyne";
    const EMAIL: &'static str = "";
    const VERSION: &'static str = env!("CARGO_PKG_VERSION");

    const AUDIO_IO_LAYOUTS: &'static [AudioIOLayout] = &[
        AudioIOLayout {
            main_input_channels: NonZeroU32::new(2),
            main_output_channels: NonZeroU32::new(2),
            ..AudioIOLayout::const_default()
        },
        AudioIOLayout {
            main_input_channels: NonZeroU32::new(1),
            main_output_channels: NonZeroU32::new(1),
            ..AudioIOLayout::const_default()
        },
    ];

    const SAMPLE_ACCURATE_AUTOMATION: bool = true;

    type SysExMessage = ();
    type BackgroundTask = ();

    fn params(&self) -> Arc<dyn Params> {
        self.params.clone()
    }

    fn initialize(&mut self, _layout: &AudioIOLayout, config: &BufferConfig, _context: &mut impl InitContext<Self>) -> bool {
        // allocation is fine here: this runs outside the audio callback
        self.sample_rate = config.sample_rate;
        self.reverb = Reverb::new(config.sample_rate);
        self.mono_scratch = vec![0.0; config.max_buffer_size as usize];
        self.params.invalidate(); // the new reverb must receive every current setting
        true
    }

    fn reset(&mut self) {
        self.reverb.reset();
    }

    fn process(&mut self, buffer: &mut Buffer, _aux: &mut AuxiliaryBuffers, _context: &mut impl ProcessContext<Self>) -> ProcessStatus {
        self.params.apply(&mut self.reverb);
        match buffer.as_slice() {
            [left, right] => self.reverb.process_stereo(left, right),
            [mono] => {
                let n = mono.len();
                let other = &mut self.mono_scratch[..n];
                other.copy_from_slice(mono);
                self.reverb.process_stereo(mono, other);
                for (m, o) in mono.iter_mut().zip(other.iter()) {
                    *m = 0.5 * (*m + *o);
                }
            }
            _ => {}
        }
        // keep processing after the input stops, until the tail has decayed
        ProcessStatus::Tail((self.reverb.decay() * self.sample_rate) as u32)
    }
}

impl ClapPlugin for AutodyneReverb {
    const CLAP_ID: &'static str = "com.github.vaibhav-gopal.autodyne.reverb";
    const CLAP_DESCRIPTION: Option<&'static str> = Some("Feedback delay network reverb with modulated delay lines");
    const CLAP_MANUAL_URL: Option<&'static str> = Some(Self::URL);
    const CLAP_SUPPORT_URL: Option<&'static str> = None;
    const CLAP_FEATURES: &'static [ClapFeature] = &[ClapFeature::AudioEffect, ClapFeature::Reverb, ClapFeature::Stereo, ClapFeature::Mono];
}

nih_export_clap!(AutodyneReverb);

#[cfg(feature = "vst3")]
impl Vst3Plugin for AutodyneReverb {
    const VST3_CLASS_ID: [u8; 16] = *b"AutodyneReverb01";
    const VST3_SUBCATEGORIES: &'static [Vst3SubCategory] = &[Vst3SubCategory::Fx, Vst3SubCategory::Reverb];
}

#[cfg(feature = "vst3")]
nih_export_vst3!(AutodyneReverb);
