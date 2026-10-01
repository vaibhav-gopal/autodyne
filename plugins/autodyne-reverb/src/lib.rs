//! autodyne's 8-line feedback-delay-network reverb as a plugin.
//!
//! ```text
//! cargo xtask bundle autodyne-reverb --release   # target/bundled: .clap and .vst3
//! ```
//!
//! Parameters come straight from `Reverb`'s `Parameterized` implementation through `ParamBridge`,
//! and ramp through `Smoothed` so automation never clicks.

use std::sync::Arc;

use autodyne::params::Smoothed;
use autodyne::reverb::Reverb;
use autodyne_plug::ParamBridge;
use nice_plug::prelude::*;

/// How long parameter changes take to glide to their new value.
const RAMP_SECONDS: f64 = 0.02;

pub struct AutodyneReverb {
    params: Arc<ParamBridge>,
    reverb: Smoothed<Reverb<f32>>,
    /// the second channel when the host runs the plugin in mono
    mono_scratch: Vec<f32>,
    sample_rate: f32,
}

impl Default for AutodyneReverb {
    fn default() -> Self {
        let reverb = Smoothed::new(Reverb::new(48_000.0), RAMP_SECONDS, 48_000.0);
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

    type Editor = ();
    type SysExMessage = ();
    type BackgroundTask = ();

    fn params(&self) -> Arc<dyn Params> {
        self.params.clone()
    }

    fn activate(&mut self, _layout: &AudioIOLayout, config: &BufferConfig, _context: &mut impl ActivateContext<Self>) -> bool {
        // allocation is fine here: this runs outside the audio callback
        self.sample_rate = config.sample_rate;
        self.reverb = Smoothed::new(Reverb::new(config.sample_rate), RAMP_SECONDS, config.sample_rate as f64);
        self.mono_scratch = vec![0.0; config.max_buffer_size as usize];
        // the new reverb takes every current host setting at once: no glide from the defaults when a
        // preset loads or the sample rate changes (this runs off the audio thread)
        self.params.invalidate();
        self.params.apply(&mut self.reverb);
        self.reverb.settle();
        true
    }

    fn reset(&mut self) {
        self.reverb.settle();
    }

    fn process(&mut self, buffer: &mut Buffer, _aux: &mut AuxiliaryBuffers, _context: &mut impl ProcessContext<Self>) -> ProcessStatus {
        self.params.apply(&mut self.reverb);
        match buffer.as_slice() {
            [left, right] => self.reverb.run(left.len(), |r, range| r.process_stereo(&mut left[range.clone()], &mut right[range])),
            [mono] => {
                let n = mono.len();
                let other = &mut self.mono_scratch[..n];
                other.copy_from_slice(mono);
                self.reverb.run(n, |r, range| r.process_stereo(&mut mono[range.clone()], &mut other[range]));
                for (m, o) in mono.iter_mut().zip(other.iter()) {
                    *m = 0.5 * (*m + *o);
                }
            }
            _ => {}
        }
        // keep processing after the input stops, until the tail has decayed
        ProcessStatus::Tail((self.reverb.inner().decay() * self.sample_rate) as u32)
    }
}

impl ClapPlugin for AutodyneReverb {
    const CLAP_ID: &'static str = "com.github.vaibhav-gopal.autodyne.reverb";
    const CLAP_DESCRIPTION: Option<&'static str> = Some("Feedback delay network reverb with modulated delay lines");
    const CLAP_MANUAL_URL: Option<&'static str> = Some(Self::URL);
    const CLAP_SUPPORT_URL: Option<&'static str> = None;
    const CLAP_FEATURES: &'static [ClapFeature] = &[ClapFeature::AudioEffect, ClapFeature::Reverb, ClapFeature::Stereo, ClapFeature::Mono];
}

nice_export_clap!(AutodyneReverb);

#[cfg(feature = "vst3")]
impl Vst3Plugin for AutodyneReverb {
    const VST3_CLASS_ID: [u8; 16] = *b"AutodyneReverb01";
    const VST3_SUBCATEGORIES: &'static [Vst3SubCategory] = &[Vst3SubCategory::Fx, Vst3SubCategory::Reverb];
}

#[cfg(feature = "vst3")]
nice_export_vst3!(AutodyneReverb);
