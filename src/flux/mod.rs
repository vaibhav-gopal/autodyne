//! flux: trace DSP and array code written over [`Real`](crate::units::Real) or
//! [`ArrayMath`](crate::signal::ArrayMath), differentiate it, and compile it with XLA or IREE (cargo
//! feature `flux`). The same generic code runs eagerly on `f32` samples or `NdArray`s.
//!
//! Generic code runs on [`Tracer`] instead of `f32`; every operation is recorded into a flat
//! [`Graph`] of primitives on real arrays, run in `f32` or `f64` ([`FluxFloat`], chosen when the graph
//! is evaluated or emitted): element-wise arithmetic and functions, comparisons and
//! `select`, broadcasting, reshapes and transposes, slices, padding, concatenation and reversal,
//! sums, products, maxima and minima, `dot_general`, real and complex FFTs, and gathers (`take`, for
//! wavetables and modulated delays) with their scatter. From there:
//!
//! - [`Graph::eval`] runs it in process, in `f32` or `f64` (values are
//!   [`NdArray`](crate::signal::NdArray)s): an interpreter that fuses element-wise chains into
//!   single passes, the reference the compiled backends are checked against;
//! - [`vjp`] differentiates it in reverse mode, recording the backward pass into the same trace
//!   ([`jvp`], forward mode, transposes it);
//! - [`Scan`] runs a traced step over a whole signal, and its gradient as a reverse scan, with
//!   respect to the parameters, the initial state and the input signal;
//! - [`Loss`] scores a scan's whole output: mean squared error, the multi-resolution STFT loss
//!   usual for audio ([`Loss::stft`]), or any traced function; [`optim`] updates the parameters;
//! - [`Graph::program`] and the `Scan` programs emit textual StableHLO, which a [`Backend`]
//!   compiles and runs: [`Iree`] (its command-line tools) or [`Pjrt`] (a PJRT plugin such as XLA's,
//!   loaded in-process through the PJRT C API).
//!
//! In process, a scan whose state is scalars runs as a register program, or (feature `jit`) as
//! machine code compiled with Cranelift on first use. For whole programs flux is a front end: no IR
//! of its own beyond the trace, the compilers (IREE's tools, XLA's or another PJRT plugin) found at
//! run time; no Python anywhere. The real-time path is unchanged: the same generic code monomorphized for `f32` / `f64`,
//! never touching a trace.
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
mod fuse;
mod graph;
mod hlo;
mod interp;
mod iree;
#[cfg(feature = "jit")]
mod jit;
mod loss;
pub mod optim;
mod pjrt;
mod runtime;
mod scalar;
mod scan;

pub use ad::{jvp, vjp};
pub use graph::{scalar, trace, vector, Cmp, FluxFloat, Graph, Kind, Mask, Node, Op, Reduction, Tracer};
pub use hlo::{Emit, Program};
pub use iree::{Iree, IreeTarget};
pub use pjrt::{Pjrt, PjrtOption};
pub use runtime::{Backend, DeviceArray, Executable, ExecutableExt, HostArray, HostRef};
pub use loss::{multi_resolution_stft, stft_magnitude, Loss, StftResolution};
pub use scan::{LossGrad, Scan, ScanVjp};

/// Errors from compiling or running programs.
/// Not `Clone` or `PartialEq`: it can hold an `io::Error`.
#[derive(Debug, thiserror::Error)]
pub enum FluxError {
    /// Reading or writing files, or running a tool, failed.
    #[error("i/o: {0}")]
    Io(#[from] std::io::Error),
    /// An external tool (`iree-compile`, a Python with JAX, ...) failed.
    #[error("{tool} failed: {message}")]
    Tool {
        /// Which tool.
        tool: &'static str,
        /// What it reported.
        message: String,
    },
    /// A `.npy` file that couldn't be parsed.
    #[error("bad .npy data: {0}")]
    Npy(String),
    /// Arrays of the wrong shape for a program.
    #[error("shape mismatch: {0}")]
    Shape(String),
}

#[cfg(test)]
mod processors;
#[cfg(test)]
mod tests;
