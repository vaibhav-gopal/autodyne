//! Sample playback: recorded sounds mapped across keys and velocities, pitched by resampling.
//!
//! - [`Sample`]: audio (mono or stereo) with a root key and an optional loop, shared between voices
//!   through `Arc`. Loads WAV files (with their `smpl` loop and root key) via [`Sample::from_wav`].
//! - [`Zone`] and [`SampleMap`]: which sample plays for which key and velocity, with per-zone
//!   tuning, gain and pan; overlapping zones alternate (round robin).
//! - [`SamplerVoice`]: a [`Voice`](crate::synth::Voice) for [`Poly`](crate::synth::Poly) that plays
//!   the map: windowed-sinc, cubic or linear interpolation, forward / ping-pong / sustain loops with
//!   crossfades, an amplitude envelope and velocity, stereo output.
//!
//! Samples live in memory; everything is prepared when loading, so playing never allocates.
//!
//! tend: Audio / sampler

mod params;
mod sample;
mod voice;

pub use sample::*;
pub use voice::*;
