//! Linear time-invariant systems: transfer functions, zeros-poles-gain and state space, continuous
//! (`s`) or discrete (`z`, with a sample period), converted into one another and into second-order
//! sections; frequency responses, stability, time responses and discretization.
//!
//! Conventions follow `scipy.signal`: polynomials are coefficient vectors, highest power first;
//! a digital filter's `(b, a)` is the same vector read in powers of `z⁻¹`. Coefficients are `f64`
//! (designs need the precision); the filtering functions apply them to `f32` or `f64` data.
//!
//! ```
//! use autodyne::filter::design::{butter, Band, Design};
//!
//! // 4th-order Butterworth low-pass at 1 kHz, 48 kHz sampling, as second-order sections
//! let zpk = butter(4, Band::Lowpass(1_000.0), Design::Digital { fs: 48_000.0 }).unwrap();
//! assert!(zpk.is_stable());
//! let sos = zpk.to_sos().unwrap();
//! assert_eq!(sos.len(), 2);
//! let h = autodyne::systems::sosfreqz(&sos, &[0.0, 1_000.0], 48_000.0);
//! assert!((h[0].norm() - 1.0).abs() < 1e-9 && (h[1].norm() - 0.5f64.sqrt()).abs() < 1e-9);
//! ```

mod convert;
mod discretize;
mod response;

pub use convert::*;
pub use discretize::*;
pub use response::*;

use thiserror::Error;

use crate::linalg::LinalgError;
use crate::signal::NdArray;
use crate::units::Complex;

/// A complex number in double precision.
pub type C64 = Complex<f64>;

/// Second-order sections: each row is `[b0, b1, b2, a0, a1, a2]`, applied in order.
pub type Sos = Vec<[f64; 6]>;

/// Continuous time (`s`), or discrete time (`z`) with a sample period `dt` in seconds.
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum Domain {
    Continuous,
    Discrete { dt: f64 },
}

impl Domain {
    /// Discrete at sample rate `fs`.
    pub fn sampled(fs: f64) -> Domain {
        Domain::Discrete { dt: 1.0 / fs }
    }
    pub fn is_discrete(self) -> bool {
        matches!(self, Domain::Discrete { .. })
    }
}

#[derive(Debug, Clone, PartialEq, Error)]
pub enum SystemError {
    #[error(transparent)]
    Linalg(#[from] LinalgError),
    #[error("invalid system: {0}")]
    Invalid(String),
    #[error("the systems are in different domains")]
    Domain,
}

/// A transfer function `num(x) / den(x)` (polynomials, highest power first).
#[derive(Debug, Clone, PartialEq)]
pub struct TransferFunction {
    pub num: Vec<f64>,
    pub den: Vec<f64>,
    pub domain: Domain,
}

/// Zeros, poles and gain: `k Π(x - z) / Π(x - p)`.
#[derive(Debug, Clone, PartialEq)]
pub struct Zpk {
    pub zeros: Vec<C64>,
    pub poles: Vec<C64>,
    pub gain: f64,
    pub domain: Domain,
}

/// State space: `x' = A x + B u`, `y = C x + D u` (`x'` the derivative, or the next state when
/// discrete). `A` n x n, `B` n x inputs, `C` outputs x n, `D` outputs x inputs.
#[derive(Debug, Clone, PartialEq)]
pub struct StateSpace {
    pub a: NdArray<f64>,
    pub b: NdArray<f64>,
    pub c: NdArray<f64>,
    pub d: NdArray<f64>,
    pub domain: Domain,
}

/// Whether poles lie strictly inside the stability region: the left half-plane (continuous) or the
/// unit circle (discrete).
fn stable(poles: &[C64], domain: Domain) -> bool {
    match domain {
        Domain::Continuous => poles.iter().all(|p| p.re < 0.0),
        Domain::Discrete { .. } => poles.iter().all(|p| p.norm() < 1.0),
    }
}

/// How far the least stable pole is inside the stability boundary: `-max Re(p)` (continuous) or
/// `1 - max |p|` (discrete). Negative when unstable; infinite with no poles.
fn margin(poles: &[C64], domain: Domain) -> f64 {
    match domain {
        Domain::Continuous => -poles.iter().map(|p| p.re).fold(f64::NEG_INFINITY, f64::max),
        Domain::Discrete { .. } => 1.0 - poles.iter().map(|p| p.norm()).fold(f64::NEG_INFINITY, f64::max),
    }
}

impl Zpk {
    pub fn new(zeros: Vec<C64>, poles: Vec<C64>, gain: f64, domain: Domain) -> Self {
        Zpk { zeros, poles, gain, domain }
    }
    /// Asymptotically stable: every pole strictly inside the stability region.
    pub fn is_stable(&self) -> bool {
        stable(&self.poles, self.domain)
    }
    /// The distance of the least stable pole from the stability boundary (negative: unstable).
    pub fn stability_margin(&self) -> f64 {
        margin(&self.poles, self.domain)
    }
    /// Minimum phase: stable, with every zero inside the stability region too.
    pub fn is_minimum_phase(&self) -> bool {
        self.is_stable() && stable(&self.zeros, self.domain)
    }
    pub fn to_tf(&self) -> TransferFunction {
        let (num, den) = zpk2tf(&self.zeros, &self.poles, self.gain);
        TransferFunction { num, den, domain: self.domain }
    }
    /// Second-order sections, pairing poles with their nearest zeros (`scipy.signal.zpk2sos`,
    /// pairing `nearest`).
    pub fn to_sos(&self) -> Result<Sos, SystemError> {
        zpk2sos(&self.zeros, &self.poles, self.gain, Pairing::Nearest, self.domain)
    }
    pub fn to_ss(&self) -> Result<StateSpace, SystemError> {
        self.to_tf().to_ss()
    }
}

impl TransferFunction {
    pub fn new(num: Vec<f64>, den: Vec<f64>, domain: Domain) -> Self {
        TransferFunction { num, den, domain }
    }
    pub fn poles(&self) -> Result<Vec<C64>, SystemError> {
        Ok(crate::linalg::roots(&self.den)?)
    }
    pub fn zeros(&self) -> Result<Vec<C64>, SystemError> {
        Ok(crate::linalg::roots(&self.num)?)
    }
    pub fn is_stable(&self) -> Result<bool, SystemError> {
        Ok(stable(&self.poles()?, self.domain))
    }
    pub fn stability_margin(&self) -> Result<f64, SystemError> {
        Ok(margin(&self.poles()?, self.domain))
    }
    pub fn to_zpk(&self) -> Result<Zpk, SystemError> {
        let (zeros, poles, gain) = tf2zpk(&self.num, &self.den)?;
        Ok(Zpk { zeros, poles, gain, domain: self.domain })
    }
    pub fn to_ss(&self) -> Result<StateSpace, SystemError> {
        let (a, b, c, d) = tf2ss(&self.num, &self.den)?;
        Ok(StateSpace { a, b, c, d, domain: self.domain })
    }
    pub fn to_sos(&self) -> Result<Sos, SystemError> {
        self.to_zpk()?.to_sos()
    }
    /// The series connection `self` then `other` (the product of the transfer functions).
    pub fn series(&self, other: &TransferFunction) -> Result<TransferFunction, SystemError> {
        if self.domain != other.domain {
            return Err(SystemError::Domain);
        }
        Ok(TransferFunction {
            num: crate::linalg::polymul(&self.num, &other.num),
            den: crate::linalg::polymul(&self.den, &other.den),
            domain: self.domain,
        })
    }
    /// The parallel connection (the sum of the transfer functions).
    pub fn parallel(&self, other: &TransferFunction) -> Result<TransferFunction, SystemError> {
        if self.domain != other.domain {
            return Err(SystemError::Domain);
        }
        let num = crate::linalg::polyadd(&crate::linalg::polymul(&self.num, &other.den), &crate::linalg::polymul(&other.num, &self.den));
        Ok(TransferFunction { num, den: crate::linalg::polymul(&self.den, &other.den), domain: self.domain })
    }
    /// Negative feedback through `feedback`: `G / (1 + G H)`.
    pub fn feedback(&self, feedback: &TransferFunction) -> Result<TransferFunction, SystemError> {
        if self.domain != feedback.domain {
            return Err(SystemError::Domain);
        }
        let num = crate::linalg::polymul(&self.num, &feedback.den);
        let den = crate::linalg::polyadd(&crate::linalg::polymul(&self.den, &feedback.den), &crate::linalg::polymul(&self.num, &feedback.num));
        Ok(TransferFunction { num, den, domain: self.domain })
    }
}

impl StateSpace {
    pub fn new(a: NdArray<f64>, b: NdArray<f64>, c: NdArray<f64>, d: NdArray<f64>, domain: Domain) -> Result<Self, SystemError> {
        let n = a.shape().first().copied().unwrap_or(0);
        let ok = a.shape() == [n, n] && b.ndim() == 2 && b.shape()[0] == n && c.ndim() == 2 && c.shape()[1] == n && d.shape() == [c.shape()[0], b.shape()[1]];
        if !ok {
            return Err(SystemError::Invalid(format!(
                "state-space shapes A {:?}, B {:?}, C {:?}, D {:?} do not agree",
                a.shape(),
                b.shape(),
                c.shape(),
                d.shape()
            )));
        }
        Ok(StateSpace { a, b, c, d, domain })
    }
    /// The number of states.
    pub fn order(&self) -> usize {
        self.a.shape()[0]
    }
    /// The poles: the eigenvalues of `A`.
    pub fn poles(&self) -> Result<Vec<C64>, SystemError> {
        if self.order() == 0 {
            return Ok(Vec::new());
        }
        Ok(crate::linalg::eigvals(self.a.view())?)
    }
    pub fn is_stable(&self) -> Result<bool, SystemError> {
        Ok(stable(&self.poles()?, self.domain))
    }
    pub fn stability_margin(&self) -> Result<f64, SystemError> {
        Ok(margin(&self.poles()?, self.domain))
    }
    /// The transfer function from `input` to every output: one numerator per output, a shared
    /// denominator (`scipy.signal.ss2tf`).
    pub fn to_tf(&self, input: usize) -> Result<(Vec<Vec<f64>>, Vec<f64>), SystemError> {
        ss2tf(self, input)
    }
    /// The single-input, single-output transfer function.
    pub fn to_tf_siso(&self) -> Result<TransferFunction, SystemError> {
        let (mut num, den) = ss2tf(self, 0)?;
        if num.len() != 1 || self.b.shape()[1] != 1 {
            return Err(SystemError::Invalid("not a single-input, single-output system".into()));
        }
        Ok(TransferFunction { num: num.remove(0), den, domain: self.domain })
    }
    pub fn to_zpk(&self) -> Result<Zpk, SystemError> {
        self.to_tf_siso()?.to_zpk()
    }
}

/// A complex number with no imaginary part.
pub(crate) fn real(x: f64) -> C64 {
    C64::new(x, 0.0)
}

#[cfg(test)]
mod tests;
