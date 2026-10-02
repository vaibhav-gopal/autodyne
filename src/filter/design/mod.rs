//! Filter design, as in `scipy.signal`: IIR filters from analog prototypes ([`butter`],
//! [`cheby1`], [`cheby2`], [`ellip`], [`bessel`], [`iirfilter`]) returned as zeros-poles-gain
//! (convert with `to_sos()`), and FIR filters ([`firwin`], [`firwin2`], [`firls`], [`remez`]).
//!
//! ```
//! use autodyne::filter::design::{ellip, firwin, Band, Design};
//! use autodyne::spectral::WindowSpec;
//!
//! // 6th-order elliptic band-pass, 0.5 dB ripple, 60 dB stopband, 300-3400 Hz at 16 kHz
//! let sos = ellip(6, 0.5, 60.0, Band::Bandpass(300.0, 3_400.0), Design::Digital { fs: 16_000.0 })
//!     .unwrap()
//!     .to_sos()
//!     .unwrap();
//! assert_eq!(sos.len(), 6);
//! // a 101-tap Kaiser-window low-pass at 1 kHz
//! let taps = firwin(101, &[1_000.0], WindowSpec::Kaiser { beta: 8.6 }, true, true, 16_000.0).unwrap();
//! assert!((taps.iter().sum::<f64>() - 1.0).abs() < 1e-12);
//! ```

mod fir;
mod iir;
mod special;

pub use fir::*;
pub use iir::*;
pub use special::{arc_jac_sn, bessel_i0, ellipj, ellipk, ellipkm1};

use thiserror::Error;

#[derive(Debug, Clone, PartialEq, Eq, Error)]
pub enum DesignError {
    #[error("invalid design: {0}")]
    Invalid(String),
}

#[cfg(test)]
mod tests;
