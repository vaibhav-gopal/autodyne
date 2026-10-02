//! flux: trace DSP code written over [`Real`](crate::units::Real), differentiate it, and compile
//! it with XLA / IREE (cargo feature `flux`).
//!
//! Generic code runs on [`Tracer`] instead of `f32`; every operation is recorded into a flat
//! [`Graph`] of primitives. From there:
//!
//! - [`Graph::eval`] interprets it (f32, the reference semantics);
//! - [`vjp`] differentiates it in reverse mode, recording the backward pass into the same trace;
//! - [`Scan`] runs a traced per-sample step over a whole signal, and its gradient as a reverse
//!   scan;
//! - the `*_hlo` methods emit textual StableHLO, which [`Iree`] compiles and runs (XLA through
//!   PJRT reads the same text).
//!
//! flux is a front end only: no IR of its own beyond the trace, no code generation, nothing linked;
//! the compilers are external tools used at run time. The real-time path is unchanged: the same
//! generic code monomorphized for `f32` / `f64`, never touching a trace.
//!
//! ```
//! use autodyne::filter::OnePole;
//! use autodyne::flux::Scan;
//! use autodyne::units::Real;
//!
//! // fit a one-pole's cutoff: the gradient of the error with respect to the cutoff
//! let scan = Scan::trace(1, 1, |p, s, x| {
//!     let (s, y) = OnePole::lowpass(p[0], Real::lit(48_000.0)).tick(s[0], x);
//!     (vec![s], y)
//! });
//! let xs: Vec<f32> = (0..256).map(|i| if i % 32 < 16 { 1.0 } else { -1.0 }).collect();
//! let (target, _) = scan.run(&[2_000.0], &xs, &[0.0]);
//! let g = scan.loss_grad(&[500.0], &xs, &target, &[0.0]);
//! assert!(g.params[0] < 0.0, "raising the cutoff lowers the error");
//! let hlo = scan.loss_grad_hlo(xs.len()); // StableHLO for IREE / XLA
//! assert!(hlo.contains("stablehlo.while"));
//! ```

mod ad;
mod graph;
mod hlo;
mod interp;
mod iree;
mod scan;

pub use ad::vjp;
pub use graph::{trace, Cmp, Graph, Mask, Op, Tracer};
pub use iree::{Iree, Module, Tensor};
pub use scan::{LossGrad, Scan};

/// Errors from running compiled programs.
#[derive(Debug, thiserror::Error)]
pub enum FluxError {
    #[error("i/o: {0}")]
    Io(#[from] std::io::Error),
    #[error("{tool} failed: {message}")]
    Tool { tool: &'static str, message: String },
    #[error("bad .npy data: {0}")]
    Npy(String),
}

#[cfg(test)]
mod tests;
