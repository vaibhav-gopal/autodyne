//! Spectral analysis and processing built on the FFT (`fft`): windows ([`get_window`]), estimates as
//! `scipy.signal` makes them (periodogram, Welch, CSD, coherence, spectrogram, STFT / ISTFT,
//! Lomb-Scargle), and the phase vocoder: [`PitchShifter`] (real time) and [`time_stretch`].

mod params;
mod estimate;
mod vocoder;
mod windows;
pub use estimate::*;
pub use vocoder::*;
pub use windows::*;