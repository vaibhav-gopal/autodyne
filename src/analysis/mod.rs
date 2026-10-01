//! Measurement and analysis for meters, tuners, visualizers and GUIs.
//!
//! - [`SpectrumAnalyzer`]: a running magnitude spectrum with analyzer ballistics and log-spaced
//!   bands for drawing.
//! - [`LoudnessMeter`]: ITU-R BS.1770 / EBU R128 momentary, short-term and integrated loudness,
//!   loudness range, and true peak ([`TruePeak`] on its own for limiters and peak meters).
//! - [`PitchDetector`]: YIN fundamental-frequency estimation for monophonic sources.
//! - [`OnsetDetector`]: note attacks by SuperFlux-style spectral flux.
//!
//! All of them allocate only when constructed, so they can run on the audio thread, and all are
//! [`Processor`](crate::processor::Processor)s (or a `MultiProcessor`) that pass audio through
//! unchanged, so they can tap a chain.

mod loudness;
mod onset;
mod pitch;
mod spectrum;

pub use loudness::*;
pub use onset::*;
pub use pitch::*;
pub use spectrum::*;