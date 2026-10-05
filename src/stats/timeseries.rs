//! Time-series estimates, as `statsmodels.tsa.stattools` makes them: autocovariance and
//! autocorrelation ([`acovf`], [`acf`]; by FFT, so `O(n log n)` for every lag at once), partial
//! autocorrelation ([`pacf`]), and autoregressive fits ([`levinson_durbin`], [`yule_walker`],
//! [`burg`]).

use crate::alloc_prelude::*;
use super::StatsError;
use crate::fft::{next_fast_len, RealFft};
use crate::units::*;

fn to_f64<T: Float>(x: &[T], demean: bool) -> Vec<f64> {
    let mut v: Vec<f64> = x.iter().map(|s| s.to_f64().unwrap_or(f64::NAN)).collect();
    if demean && !v.is_empty() {
        let mean = v.iter().sum::<f64>() / v.len() as f64;
        v.iter_mut().for_each(|s| *s -= mean);
    }
    v
}

/// The autocovariances of `x` at lags `0..=nlags` (`statsmodels.tsa.stattools.acovf`): `sum_t
/// x[t] x[t + k]` over `n` (or over `n - k` when `adjusted`), after removing the mean when
/// `demean`. Computed by FFT: one transform of the zero-padded signal gives every lag. Errors if
/// `nlags >= n`.
pub fn acovf<T: Float>(x: &[T], nlags: usize, adjusted: bool, demean: bool) -> Result<Vec<f64>, StatsError> {
    let n = x.len();
    if nlags >= n {
        return Err(StatsError::invalid(format!("nlags ({nlags}) must be below the length ({n})")));
    }
    let x = to_f64(x, demean);
    let size = next_fast_len(2 * n - 1);
    let mut fft = RealFft::<f64>::new(size);
    let mut padded = vec![0.0; size];
    padded[..n].copy_from_slice(&x);
    let mut spectrum = vec![Complex::zero(); fft.spectrum_len()];
    fft.forward(&padded, &mut spectrum);
    // |X|² transforms back to the (circular, but padded so linear) autocorrelation
    spectrum.iter_mut().for_each(|z| *z = Complex::new(z.norm_sqr(), 0.0));
    fft.inverse(&spectrum, &mut padded);
    Ok((0..=nlags).map(|k| padded[k] / if adjusted { (n - k) as f64 } else { n as f64 }).collect())
}

/// The autocorrelations of `x` at lags `0..=nlags` (`statsmodels.tsa.stattools.acf`): [`acovf`]
/// (demeaned) divided by the variance, so lag 0 is 1.
pub fn acf<T: Float>(x: &[T], nlags: usize, adjusted: bool) -> Result<Vec<f64>, StatsError> {
    let c = acovf(x, nlags, adjusted, true)?;
    let c0 = c[0];
    Ok(c.into_iter().map(|v| v / c0).collect())
}

/// An autoregressive model fitted by [`levinson_durbin`], [`yule_walker`] or [`burg`].
#[derive(Clone, Debug, PartialEq)]
pub struct ArFit {
    /// The coefficients `phi_1..phi_p` of `x[t] = sum_i phi_i x[t - i] + e[t]`.
    pub ar: Vec<f64>,
    /// The variance of the innovations `e`.
    pub sigma2: f64,
    /// The partial autocorrelations at lags `0..=p` (lag 0 is 1): the last coefficient of each
    /// order's fit.
    pub pacf: Vec<f64>,
}

/// The order-`order` autoregression whose autocovariances are `acov[0..=order]`, by the
/// Levinson-Durbin recursion (`statsmodels.tsa.stattools.levinson_durbin` with `isacov=True`):
/// `O(order²)`, and its partial autocorrelations come free. Errors if there are too few
/// autocovariances or the variance is not positive.
pub fn levinson_durbin(acov: &[f64], order: usize) -> Result<ArFit, StatsError> {
    if acov.len() <= order {
        return Err(StatsError::invalid(format!("order {order} needs {} autocovariances, got {}", order + 1, acov.len())));
    }
    if acov[0].is_nan() || acov[0] <= 0.0 {
        return Err(StatsError::invalid("the lag-0 autocovariance (the variance) must be positive"));
    }
    let mut phi = vec![0.0; order];
    let mut pacf = vec![1.0];
    let mut sigma = acov[0];
    for k in 1..=order {
        let reflection = (acov[k] - (1..k).map(|j| phi[j - 1] * acov[k - j]).sum::<f64>()) / sigma;
        let previous = phi[..k - 1].to_vec();
        for j in 1..k {
            phi[j - 1] = previous[j - 1] - reflection * previous[k - j - 1];
        }
        phi[k - 1] = reflection;
        sigma *= 1.0 - reflection * reflection;
        pacf.push(reflection);
    }
    Ok(ArFit { ar: phi, sigma2: sigma, pacf })
}

/// How [`yule_walker`] and [`pacf`] estimate the autocovariances.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub enum YuleWalker {
    /// Dividing lag `k`'s sum by `n - k` (statsmodels' `"adjusted"`, its default).
    #[default]
    Adjusted,
    /// Dividing by `n` (`"mle"`): biased, but always a stationary fit.
    Mle,
}

/// An order-`order` autoregression fitted to `x` by the Yule-Walker equations
/// (`statsmodels.regression.linear_model.yule_walker`, demeaned): the autocovariances of `x`
/// solved by [`levinson_durbin`]. The returned `sigma2` is the innovation variance (statsmodels
/// returns its square root).
pub fn yule_walker<T: Float>(x: &[T], order: usize, method: YuleWalker) -> Result<ArFit, StatsError> {
    let acov = acovf(x, order, method == YuleWalker::Adjusted, true)?;
    let mut fit = levinson_durbin(&acov, order)?;
    // statsmodels' sigma² = r0 - sum phi_k r_k (the same in exact arithmetic; this keeps its rounding)
    fit.sigma2 = acov[0] - fit.ar.iter().zip(&acov[1..]).map(|(p, r)| p * r).sum::<f64>();
    Ok(fit)
}

/// The partial autocorrelations of `x` at lags `0..=nlags` (`statsmodels.tsa.stattools.pacf` with
/// `method="yw"` or `"ywm"`): the last coefficient of the Yule-Walker fit of each order. Errors
/// unless `nlags < n / 2`.
pub fn pacf<T: Float>(x: &[T], nlags: usize, method: YuleWalker) -> Result<Vec<f64>, StatsError> {
    if nlags >= x.len() / 2 {
        return Err(StatsError::invalid(format!("pacf needs nlags ({nlags}) below half the length ({})", x.len())));
    }
    let acov = acovf(x, nlags, method == YuleWalker::Adjusted, true)?;
    Ok(levinson_durbin(&acov, nlags)?.pacf)
}

/// An order-`order` autoregression fitted to `x` by Burg's method
/// (`statsmodels.regression.linear_model.burg`, demeaned): reflection coefficients from forward and
/// backward prediction errors together, so the fit is always stationary and sharper than
/// Yule-Walker on short records. Errors unless `1 <= order < n`.
pub fn burg<T: Float>(x: &[T], order: usize) -> Result<ArFit, StatsError> {
    let n = x.len();
    if order == 0 || order >= n {
        return Err(StatsError::invalid(format!("burg needs 1 <= order < n, got order {order} for {n} samples")));
    }
    let x = to_f64(x, true);
    let p = order;
    // forward errors f[t] and backward errors b[t] of the order-m predictor, starting from the signal
    let (mut f, mut b) = (x.clone(), x);
    let mut pacf = Vec::with_capacity(p + 1);
    pacf.push(1.0);
    let mut sigma2 = 0.0;
    for m in 1..=p {
        // the reflection minimizing the summed forward and backward error powers, over t = m..n
        let (mut num, mut den) = (0.0, 0.0);
        for t in m..n {
            num += f[t] * b[t - 1];
            den += f[t] * f[t] + b[t - 1] * b[t - 1];
        }
        let k = 2.0 * num / den;
        // descending, so b[t - 1] is still the previous order's when b[t] is written
        for t in (m..n).rev() {
            let (ft, bt) = (f[t], b[t - 1]);
            f[t] = ft - k * bt;
            b[t] = bt - k * ft;
        }
        pacf.push(k);
        // statsmodels' estimate: (1 - k²) times the error power, per remaining sample
        sigma2 = (1.0 - k * k) * den / (2.0 * (n - m) as f64);
    }
    // the coefficients from the reflections (statsmodels' levinson_durbin_pacf)
    let mut ar = pacf[1..].to_vec();
    for i in 1..p {
        let prev = ar[..i].to_vec();
        for j in 0..i {
            ar[j] = prev[j] - ar[i] * prev[i - 1 - j];
        }
    }
    Ok(ArFit { ar, sigma2, pacf })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::random::{process, Rng};

    #[test]
    fn autocorrelation_by_fft_matches_the_sums() {
        let x = crate::testing::noise_f64(300, 2);
        let mean = x.iter().sum::<f64>() / 300.0;
        let fast = acovf(&x, 20, false, true).unwrap();
        for (k, &c) in fast.iter().enumerate() {
            let slow: f64 = (0..300 - k).map(|t| (x[t] - mean) * (x[t + k] - mean)).sum::<f64>() / 300.0;
            assert!((c - slow).abs() < 1e-14, "lag {k}");
        }
        let r = acf(&x, 5, true).unwrap();
        assert_eq!(r[0], 1.0);
        assert!(acovf(&x, 300, false, true).is_err());
    }

    #[test]
    fn ar_fits_recover_a_known_process() {
        // x[t] = 0.6 x[t-1] - 0.3 x[t-2] + e, sigma 1
        let mut rng = Rng::new(9);
        let x = process::arma(&mut rng, &[1.0, -0.6, 0.3], &[1.0], 50_000, 1.0, 500).unwrap();
        for fit in [yule_walker(&x, 2, YuleWalker::Adjusted).unwrap(), yule_walker(&x, 2, YuleWalker::Mle).unwrap(), burg(&x, 2).unwrap()] {
            assert!((fit.ar[0] - 0.6).abs() < 0.015 && (fit.ar[1] + 0.3).abs() < 0.015, "{:?}", fit.ar);
            assert!((fit.sigma2 - 1.0).abs() < 0.02, "{}", fit.sigma2);
        }
        // the partial autocorrelation cuts off after lag 2
        let p = pacf(&x, 6, YuleWalker::Adjusted).unwrap();
        assert!((p[2] + 0.3).abs() < 0.015 && p[3..].iter().all(|v| v.abs() < 0.02), "{p:?}");
    }
}
