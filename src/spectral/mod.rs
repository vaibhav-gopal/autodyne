//! Spectral analysis and processing built on the FFT (`fft`): windows ([`get_window`]), estimates as
//! `scipy.signal` makes them (periodogram, Welch, CSD, coherence, spectrogram, STFT / ISTFT,
//! Lomb-Scargle), the analytic signal ([`hilbert`]), and the phase vocoder: [`PitchShifter`] and
//! [`TimeStretcher`] (real time; the stretcher plays a sample buffer at a changeable speed, with a
//! loop) and [`time_stretch`] (offline).
//!
//! tend: Signal processing / spectral

mod analytic;
mod estimate;
mod params;
mod vocoder;
mod windows;
pub use analytic::*;
pub use estimate::*;
pub use vocoder::*;
pub use windows::*;
