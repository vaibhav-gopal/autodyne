//! Autodyne: DSP and numerical building blocks for real-time audio and signal processing.
//!
//! ```
//! use autodyne::prelude::*;
//! use autodyne::filter::{Biquad, BUTTERWORTH_Q};
//! use autodyne::osc::Sine;
//!
//! let mut block = vec![0.0f32; 512];
//! let mut chain = Sine::new(440.0, 48_000.0).through(Biquad::lowpass(1_000.0, BUTTERWORTH_Q as f32, 48_000.0));
//! chain.fill(&mut block);          // generate + filter, a block at a time
//! assert!(block.rms().unwrap() > 0.5);
//! ```
//!
//! - Data: [`units`] (number traits, `Complex`, `DType` reflection), [`signal`] (the signal traits,
//!   capability tiers, sources, streams, n-d arrays), [`channels`] (multichannel buffers).
//! - Processing: [`processor`] (the shared trait and chains), [`osc`], [`filter`], [`fft`],
//!   [`spectral`], [`iq`], [`gain`] (levels, pan, width), [`delay`], [`dynamics`], [`envelope`],
//!   [`distortion`], [`reverb`], [`modulation`], [`resample`], [`simd`].
//! - Maths: [`special`] (gamma, error and normal functions, Bessel and elliptic functions),
//!   [`random`] (generators, distributions, random processes), [`stats`] (order statistics,
//!   moments, histograms, covariance, autocorrelation and AR fits), [`ode`] (initial value
//!   problems), [`linalg`] (matrix products, solves, least squares, eigen / Schur / SVD, matrix
//!   equations, polynomials) and [`systems`] (LTI systems: transfer functions, zeros-poles-gain,
//!   state space, responses, discretization, controllability, LQR, margins); `linalg` and
//!   `systems` need the feature `faer`, on by default.
//! - Measurement: [`analysis`] (spectrum, loudness and true peak, pitch, onsets).
//! - Instruments: [`synth`] (MIDI, voices, polyphony), [`sampler`] (multisampled playback), [`control`] (LFOs, modulation matrix, transport).
//! - Differentiable programs: `flux` (feature `flux`) traces code written over [`units::Real`],
//!   differentiates it and emits StableHLO for XLA / IREE, or runs it in process (feature jit:
//!   scalar scans compiled to machine code).
//! - GPU: gpu (feature gpu): arrays in GPU memory with CubeCL kernels through wgpu.
//! - Integration: [`dlpack`] (zero-copy tensor exchange with NumPy, PyTorch, JAX...), [`wav`] (WAV files with loop metadata), [`params`] (parameters and metadata for hosts), [`dynamic`] (runtime-typed data
//!   and processors), [`prelude`] (one import for all the traits).
//!
//! Processors allocate only when constructed; processing runs in place without allocating, so it is
//! safe inside an audio callback.
//!
//! Errors: operations that can fail on their inputs (shapes, designs, files, external tools) return
//! `Result` with their module's error type (`signal::NdError`, `filter::FilterError`, ...), which wraps
//! the lower layers' errors rather than flattening them. Processors clamp settings into range instead
//! of failing; the few constructors that panic say so ("Panics if ...").
//!
//! Layering (each module uses only those before it): `units`; `simd`, `fft`, `special`, `processor`;
//! `signal`; `random`, `channels`, `linalg`; `stats`; `params`; then the processors, instruments and analysis,
//! each declaring its own `Processor` and `Parameterized` implementations (`<module>/params.rs`).

#![warn(missing_docs)]
// Without the `std` feature: core + alloc (see Cargo.toml). Unit tests link std either way (the
// harness and the tests need it); the library code under test keeps its no-std paths.
#![cfg_attr(not(any(feature = "std", test)), no_std)]

extern crate alloc;

/// What the standard prelude gives every module, taken from `alloc` so the same code builds with and
/// without `std` (modules that use these glob-import it). Without `std` it also brings the float
/// methods `core` lacks (`x.sqrt()` ...), from `libm`.
pub(crate) mod alloc_prelude {
    #[allow(unused_imports)]
    pub(crate) use alloc::{
        borrow::ToOwned,
        boxed::Box,
        format,
        string::{String, ToString},
        vec,
        vec::Vec,
    };
    #[cfg(not(feature = "std"))]
    #[allow(unused_imports)]
    pub(crate) use crate::units::float_math::FloatMath;
}

/// With the `mimalloc` feature, mimalloc allocates for the whole program (see the feature's note in
/// `Cargo.toml`): new arrays reuse freed pages instead of faulting fresh ones in.
#[cfg(feature = "mimalloc")]
#[global_allocator]
static ALLOCATOR: mimalloc::MiMalloc = mimalloc::MiMalloc;

pub mod units;
pub mod signal;
pub mod simd;
pub mod processor;
pub mod channels;
pub mod fft;
pub mod special;
pub mod osc;
pub mod random;
pub mod stats;
pub mod ode;
pub mod filter;
pub mod spectral;
pub mod iq;
pub mod gain;
pub mod delay;
pub mod dynamics;
pub mod envelope;
pub mod distortion;
pub mod synth;
pub mod sampler;
pub mod reverb;
pub mod modulation;
pub mod analysis;
pub mod control;
pub mod resample;
pub mod params;
pub mod dynamic;
#[cfg(feature = "faer")]
mod gemm;
#[cfg(feature = "faer")]
pub mod linalg;
#[cfg(feature = "faer")]
pub mod systems;
#[cfg(feature = "gpu")]
pub mod gpu;
pub mod dlpack;
pub mod interop;
#[cfg(feature = "flux")]
pub mod flux;
pub mod wav;
pub mod prelude;
#[cfg(test)]
mod testing;
