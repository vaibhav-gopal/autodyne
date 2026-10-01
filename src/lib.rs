//! Autodyne: DSP and numerical building blocks for real-time audio and signal processing.
//!
//! ```
//! use autodyne::prelude::*;
//! use autodyne::filter::{Biquad, BUTTERWORTH_Q};
//! use autodyne::osc::Sine;
//!
//! let mut block = vec![0.0f32; 512];
//! let mut chain = Sine::new(440.0, 48_000.0).through(Biquad::lowpass(1_000.0, 48_000.0, BUTTERWORTH_Q as f32));
//! chain.fill(&mut block);          // generate + filter, a block at a time
//! assert!(block.rms().unwrap() > 0.5);
//! ```
//!
//! - Data: [`units`] (number traits, `Complex`, `DType` reflection), [`signal`] (the signal traits,
//!   capability tiers, sources, streams, n-d arrays), [`channels`] (multichannel buffers).
//! - Processing: [`processor`] (the shared trait and chains), [`osc`], [`filter`], [`spectral`],
//!   [`iq`], [`gain`], [`delay`], [`dynamics`], [`envelope`], [`distortion`], [`reverb`], [`modulation`], [`resample`], [`simd`].
//! - Measurement: [`analysis`] (spectrum, loudness and true peak, pitch, onsets).
//! - Instruments: [`synth`] (MIDI, voices, polyphony), [`control`] (LFOs, modulation matrix, transport).
//! - Integration: [`params`] (parameters and metadata for hosts), [`dynamic`] (runtime-typed data
//!   and processors), [`prelude`] (one import for all the traits).
//!
//! Processors allocate only when constructed; processing runs in place without allocating, so it is
//! safe inside an audio callback.

pub mod units;
pub mod signal;
pub mod simd;
pub mod processor;
pub mod channels;
pub mod osc;
pub mod filter;
pub mod spectral;
pub mod iq;
pub mod gain;
pub mod delay;
pub mod dynamics;
pub mod envelope;
pub mod distortion;
pub mod synth;
pub mod reverb;
pub mod modulation;
pub mod analysis;
pub mod control;
pub mod resample;
pub mod params;
pub mod dynamic;
pub mod prelude;
