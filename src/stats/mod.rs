//! Statistics, as NumPy, `scipy.stats` and `statsmodels` have them: order statistics along an axis
//! ([`median_axis`], [`quantile_axis`] with NumPy's methods), moments ([`skew_axis`],
//! [`kurtosis_axis`], [`moment_axis`], [`zscore_axis`]), [`histogram`]s (fixed counts, explicit
//! edges or NumPy's bin-width rules), covariance and correlation matrices ([`cov`], [`corrcoef`]),
//! and time-series estimates ([`acovf`], [`acf`], [`pacf`], [`levinson_durbin`],
//! [`yule_walker`], [`burg`]).
//!
//! Inputs are views with any strides; computation is in f64.
//!
//! tend: Numerics / stats

use thiserror::Error;

mod correlation;
mod histogram;
mod moments;
mod quantile;
mod timeseries;

pub use correlation::*;
pub use histogram::*;
pub use moments::*;
pub use quantile::*;
pub use timeseries::*;

/// Errors from statistics.
#[derive(Debug, Clone, PartialEq, Eq, Error)]
pub enum StatsError {
    /// An argument is out of range (a quantile above 1, too few samples for the order asked).
    #[error("invalid argument: {0}")]
    Invalid(String),
    /// An n-d layout error (an axis out of range).
    #[error(transparent)]
    Nd(#[from] crate::signal::NdError),
    /// A linear algebra failure (feature `faer`).
    #[cfg(feature = "faer")]
    #[error(transparent)]
    Linalg(#[from] crate::linalg::LinalgError),
}

impl StatsError {
    pub(crate) fn invalid(message: impl Into<String>) -> Self {
        StatsError::Invalid(message.into())
    }
}

/// An array of `shape` (with `axis` of length `len`, or removed when `len` is `None`) whose lanes
/// along `axis` are `lanes`.
fn assemble<T: crate::units::Float + Default>(shape: &[usize], axis: usize, len: Option<usize>, lanes: &[Vec<f64>]) -> Result<crate::signal::NdArray<T>, StatsError> {
    let mut shape = shape.to_vec();
    match len {
        Some(l) => shape[axis] = l,
        None => {
            shape.remove(axis);
            let data = lanes.iter().map(|l| T::_lit(l[0])).collect();
            return Ok(crate::signal::NdArray::from_vec(data, &shape)?);
        }
    }
    let mut out = crate::signal::NdArray::<T>::zeros(&shape)?;
    for (mut dst, src) in out.lanes_mut(axis)?.zip(lanes) {
        for (d, &s) in dst.iter_mut().zip(src) {
            *d = T::_lit(s);
        }
    }
    Ok(out)
}
