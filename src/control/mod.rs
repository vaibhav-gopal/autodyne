//! Control signals: what moves parameters over time.
//!
//! - [`Transport`] and [`Division`]: the host's tempo and beat position, and note lengths for
//!   tempo sync.
//! - [`Lfo`]: low-frequency oscillator (sine, triangle, saws, square, sample & hold, smooth
//!   random), free-running in Hz or locked to the transport's beat grid, with fade-in and retrigger.
//! - [`Modulated`]: a modulation matrix around any [`Parameterized`](crate::params::Parameterized)
//!   processor: [`Route`]s connect sources to any parameter with automatable depths.
//!
//! Control signals update at control rate (every [`RAMP_STEP`](crate::params::RAMP_STEP)
//! samples), which is plenty for musical modulation and keeps coefficient recalculation cheap.
//!
//! tend: Audio / control

mod lfo;
mod matrix;
mod transport;

pub use lfo::*;
pub use matrix::*;
pub use transport::*;
