//! flux: trace DSP and array code written over [`Real`](crate::units::Real) or
//! [`ArrayMath`](crate::signal::ArrayMath), differentiate it, and compile it with XLA or IREE (cargo
//! feature `flux`). The same generic code runs eagerly on `f32` samples or `NdArray`s.
//!
//! Generic code runs on [`Tracer`] instead of `f32`; every operation is recorded into a flat
//! [`Graph`] of primitives on f32 arrays: element-wise arithmetic and functions, comparisons and
//! `select`, broadcasting, reshapes and transposes, slices, padding and concatenation, sums,
//! `dot_general` and real FFTs. From there:
//!
//! - [`Graph::eval`] interprets it (f32, the reference semantics; values are
//!   [`NdArray`](crate::signal::NdArray)s);
//! - [`vjp`] differentiates it in reverse mode, recording the backward pass into the same trace;
//! - [`Scan`] runs a traced step over a whole signal, and its gradient as a reverse scan, with
//!   respect to the parameters, the initial state and the input signal;
//! - [`Loss`] scores a scan's whole output: mean squared error, the multi-resolution STFT loss
//!   usual for audio ([`Loss::stft`]), or any traced function; [`optim`] updates the parameters;
//! - [`Graph::program`] and the `Scan` programs emit textual StableHLO, which a [`Backend`]
//!   compiles and runs: [`Iree`] (its command-line tools), [`Pjrt`] (a PJRT plugin such as XLA's,
//!   loaded in-process through the PJRT C API) or [`Xla`] (XLA through JAX, where no plugin exists).
//!
//! flux is a front end only: no IR of its own beyond the trace, no code generation, nothing linked;
//! the compilers are external tools found at run time. The real-time path is unchanged: the same
//! generic code monomorphized for `f32` / `f64`, never touching a trace.
//!
//! ```
//! use autodyne::filter::OnePole;
//! use autodyne::flux::{scalar, vector, Scan};
//! use autodyne::units::Elementwise;
//!
//! // fit a one-pole's cutoff: the gradient of the error with respect to the cutoff
//! let scan = Scan::trace(&[&[]], &[&[]], &[], |p, s, x| {
//!     let (s, y) = OnePole::lowpass(p[0], Elementwise::lit(48_000.0)).tick(s[0], x);
//!     (vec![s], y)
//! });
//! let xs = vector(&(0..256).map(|i| if i % 32 < 16 { 1.0 } else { -1.0 }).collect::<Vec<f32>>());
//! let (target, _) = scan.run(&[scalar(2_000.0)], &xs, &[scalar(0.0)]);
//! let g = scan.loss_grad(&[scalar(500.0)], &xs, &target, &[scalar(0.0)]);
//! assert!(g.params[0].as_slice()[0] < 0.0, "raising the cutoff lowers the error");
//! let program = scan.loss_grad_program(256); // StableHLO for IREE / XLA
//! assert!(program.text.contains("stablehlo.while"));
//! ```

mod ad;
mod graph;
mod hlo;
mod interp;
mod iree;
mod loss;
pub mod optim;
mod pjrt;
mod runtime;
mod scan;
mod xla;

pub use ad::vjp;
pub use graph::{scalar, trace, vector, Cmp, Graph, Mask, Node, Op, Part, Tracer};
pub use hlo::Program;
pub use iree::{Iree, IreeTarget};
pub use pjrt::Pjrt;
pub use runtime::{Backend, Executable};
pub use loss::{frames, multi_resolution_stft, stft_magnitude, Loss, StftResolution};
pub use scan::{LossGrad, Scan, ScanVjp};
pub use xla::Xla;

/// Errors from compiling or running programs.
#[derive(Debug, thiserror::Error)]
pub enum FluxError {
    #[error("i/o: {0}")]
    Io(#[from] std::io::Error),
    #[error("{tool} failed: {message}")]
    Tool { tool: &'static str, message: String },
    #[error("bad .npy data: {0}")]
    Npy(String),
    #[error("shape mismatch: {0}")]
    Shape(String),
}

#[cfg(test)]
mod processors;
#[cfg(test)]
mod tests;
