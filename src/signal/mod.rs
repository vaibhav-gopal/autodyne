//! Signals: the crate's core view of sample data.
//!
//! A signal is any contiguous run of samples: `[T]`, `Vec<T>`, arrays, a channel of an `AudioBuffer`,
//! or memory handed over by a host. Rather than a wrapper type, signal capabilities are extension
//! traits implemented for slices, so every buffer already *is* a signal, with no copies or conversions:
//!
//! - [`Signal`]: read-only analysis: levels (rms, peak, crest factor), norms, statistics, ordering
//!   (argmax, ...), relations between two signals (inner product, angle, distance) and
//!   length-changing operations that return new data (convolve, correlate, resampled).
//! - [`SignalMut`]: in-place transforms: gain, normalization, DC removal, fades, running sums and
//!   differences, projection, and pointwise math between signals with a [`Broadcast`] policy for
//!   mismatched lengths.
//! - [`ComplexSignal`]: the same ideas for complex data (IQ baseband, spectra).
//! - [`Source`]: procedural signals (oscillators, noise, closures), composed lazily with `scaled`,
//!   `mix` and `through(processor)`.
//! - Streams ([`SignalRead`], [`SignalWrite`], [`SignalSeek`], [`SignalStream`]): signals moved a block
//!   at a time as they are generated or arrive at runtime, with [`SampleReader`] / [`SampleWriter`]
//!   converting to and from bytes (files, sockets, FFI) at the boundary.
//! - Capability tiers ([`SignalOwned`], [`SignalResizable`]) and the operations they unlock
//!   ([`SigOwnedOps`], [`SigResizeOps`]).
//!
//! Stateful block processing (filters, effects) lives in `processor`; a `Source` feeds it with `through`.
//!
//! ```
//! use autodyne::signal::{Signal, SignalMut, Source};
//! use autodyne::osc::Sine;
//!
//! let mut tone = vec![0.0f64; 4800];
//! Sine::new(1_000.0, 48_000.0).fill(&mut tone);
//! assert!((tone.rms().unwrap() - 0.5f64.sqrt()).abs() < 1e-9);
//!
//! tone.normalize_peak(0.5);
//! assert!((tone.peak() - 0.5).abs() < 1e-12);
//! ```

mod analysis;
mod complex;
mod container;
mod source;
mod stream;
mod transform;

pub use analysis::*;
pub use complex::*;
pub use container::*;
pub use source::*;
pub use stream::*;
pub use transform::*;

use thiserror::Error;

/// Errors from operations that combine or measure signals.
#[derive(Error, Debug, Clone, Copy, PartialEq, Eq)]
pub enum SignalError {
    #[error("signal lengths differ: {0} vs {1}")]
    LengthMismatch(usize, usize),
    #[error("the operation needs a non-empty signal")]
    Empty,
    #[error("the operation is undefined for a signal with zero norm")]
    ZeroNorm,
}
