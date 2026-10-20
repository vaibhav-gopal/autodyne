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
//! Features: `std` and `dsp` (on by default) and the optional backends below. Without default
//! features autodyne is `no_std` and lean: `units`, `simd` and `geometry` only; `dsp` adds the rest.
//!
//! Layering (each module uses only those before it): `units`; `simd`, `geometry`, `fft`, `special`, `processor`;
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

// Always built: the number traits, the SIMD kernels and the small geometry types. Everything else is
// the `dsp` feature (on by default), so `default-features = false` compiles only these three.
pub mod units;
pub mod simd;
pub mod geometry;

// (one attribute per module: declared through a macro, their #[macro_export] macros couldn't be
// named by path)
#[cfg(feature = "dsp")]
pub mod signal;
#[cfg(feature = "dsp")]
pub mod processor;
#[cfg(feature = "dsp")]
pub mod channels;
#[cfg(feature = "dsp")]
pub mod fft;
#[cfg(feature = "dsp")]
pub mod special;
#[cfg(feature = "dsp")]
pub mod osc;
#[cfg(feature = "dsp")]
pub mod random;
#[cfg(feature = "dsp")]
pub mod stats;
#[cfg(feature = "dsp")]
pub mod ode;
#[cfg(feature = "dsp")]
pub mod filter;
#[cfg(feature = "dsp")]
pub mod spectral;
#[cfg(feature = "dsp")]
pub mod iq;
#[cfg(feature = "dsp")]
pub mod gain;
#[cfg(feature = "dsp")]
pub mod delay;
#[cfg(feature = "dsp")]
pub mod dynamics;
#[cfg(feature = "dsp")]
pub mod envelope;
#[cfg(feature = "dsp")]
pub mod distortion;
#[cfg(feature = "dsp")]
pub mod synth;
#[cfg(feature = "dsp")]
pub mod sampler;
#[cfg(feature = "dsp")]
pub mod reverb;
#[cfg(feature = "dsp")]
pub mod modulation;
#[cfg(feature = "dsp")]
pub mod analysis;
#[cfg(feature = "dsp")]
pub mod control;
#[cfg(feature = "dsp")]
pub mod resample;
#[cfg(feature = "dsp")]
pub mod params;
#[cfg(feature = "dsp")]
pub mod dynamic;
#[cfg(feature = "dsp")]
pub mod dlpack;
#[cfg(feature = "dsp")]
pub mod interop;
#[cfg(feature = "dsp")]
pub mod wav;
#[cfg(feature = "faer")]
mod gemm;
#[cfg(feature = "faer")]
pub mod linalg;
#[cfg(feature = "faer")]
pub mod systems;
#[cfg(feature = "gpu")]
pub mod gpu;
#[cfg(feature = "flux")]
pub mod flux;
pub mod prelude;
#[cfg(all(test, feature = "dsp"))]
mod testing;
