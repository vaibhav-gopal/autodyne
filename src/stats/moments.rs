//! Moments along an axis, as `scipy.stats` computes them: [`moment_axis`] (central moments),
//! [`skew_axis`], [`kurtosis_axis`] and [`zscore_axis`]. Two passes (the mean, then deviations
//! from it), so large offsets don't cost precision.

use super::{assemble, StatsError};
use crate::signal::{lanes_f64, NdArray, NdView};
use crate::units::*;

/// The mean of `lane` and its central moments of orders 2, 3 and 4 (dividing by `n`).
fn central(lane: &[f64]) -> (f64, f64, f64, f64) {
    let n = lane.len() as f64;
    let mean = lane.iter().sum::<f64>() / n;
    let (mut m2, mut m3, mut m4) = (0.0, 0.0, 0.0);
    for &v in lane {
        let d = v - mean;
        let d2 = d * d;
        m2 += d2;
        m3 += d2 * d;
        m4 += d2 * d2;
    }
    (mean, m2 / n, m3 / n, m4 / n)
}

fn per_lane<T: Float + Default>(x: NdView<'_, T>, axis: usize, f: impl Fn(&[f64]) -> f64) -> Result<NdArray<T>, StatsError> {
    if x.shape().get(axis) == Some(&0) {
        return Err(StatsError::invalid("moments of an empty axis"));
    }
    let lanes = lanes_f64(&x, axis)?;
    let out: Vec<Vec<f64>> = lanes.iter().map(|l| vec![f(l)]).collect();
    assemble(x.shape(), axis, None, &out)
}

/// The `order`-th central moment of the lanes along `axis` (removed) (`scipy.stats.moment`):
/// the mean of `(x - mean)^order`; order 0 is 1 and order 1 is 0.
pub fn moment_axis<T: Float + Default>(x: NdView<'_, T>, order: u32, axis: usize) -> Result<NdArray<T>, StatsError> {
    per_lane(x, axis, |l| {
        let mean = l.iter().sum::<f64>() / l.len() as f64;
        match order {
            0 => 1.0,
            1 => 0.0,
            _ => l.iter().map(|&v| (v - mean).powi(order as i32)).sum::<f64>() / l.len() as f64,
        }
    })
}

/// The skewness of the lanes along `axis` (removed) (`scipy.stats.skew`): `m3 / m2^1.5`, or with
/// `bias` false the adjusted Fisher-Pearson estimate (times `sqrt(n (n - 1)) / (n - 2)`, for
/// `n > 2`). Zero for a constant lane, as SciPy gives it when the variance vanishes at its
/// precision.
pub fn skew_axis<T: Float + Default>(x: NdView<'_, T>, axis: usize, bias: bool) -> Result<NdArray<T>, StatsError> {
    per_lane(x, axis, |l| {
        let (mean, m2, m3, _) = central(l);
        let n = l.len() as f64;
        // SciPy treats a variance below its rounding as zero
        if m2 <= (f64::EPSILON * mean).powi(2) {
            return if m2 == 0.0 { f64::NAN } else { 0.0 };
        }
        let g = m3 / m2.powf(1.5);
        if bias || n <= 2.0 { g } else { g * (n * (n - 1.0)).sqrt() / (n - 2.0) }
    })
}

/// The kurtosis of the lanes along `axis` (removed) (`scipy.stats.kurtosis`): `m4 / m2²`, minus 3
/// when `fisher` (so a normal distribution scores 0); with `bias` false the unbiased estimate (for
/// `n > 3`).
pub fn kurtosis_axis<T: Float + Default>(x: NdView<'_, T>, axis: usize, fisher: bool, bias: bool) -> Result<NdArray<T>, StatsError> {
    per_lane(x, axis, |l| {
        let (mean, m2, _, m4) = central(l);
        let n = l.len() as f64;
        if m2 <= (f64::EPSILON * mean).powi(2) {
            return f64::NAN;
        }
        let mut k = m4 / (m2 * m2);
        if !bias && n > 3.0 {
            k = ((n * n - 1.0) * k - 3.0 * (n - 1.0) * (n - 1.0)) / ((n - 2.0) * (n - 3.0)) + 3.0;
        }
        if fisher { k - 3.0 } else { k }
    })
}

/// Standard scores along `axis` (`scipy.stats.zscore`): `(x - mean) / std`, the standard deviation
/// dividing by `n - ddof`. Same shape as `x`.
pub fn zscore_axis<T: Float + Default>(x: NdView<'_, T>, axis: usize, ddof: usize) -> Result<NdArray<T>, StatsError> {
    let lanes = lanes_f64(&x, axis)?;
    let n = x.shape()[axis];
    if n <= ddof {
        return Err(StatsError::invalid(format!("zscore needs more than ddof ({ddof}) samples, got {n}")));
    }
    let out: Vec<Vec<f64>> = lanes
        .iter()
        .map(|l| {
            let (mean, m2, _, _) = central(l);
            let std = (m2 * n as f64 / (n - ddof) as f64).sqrt();
            l.iter().map(|&v| (v - mean) / std).collect()
        })
        .collect();
    assemble(x.shape(), axis, Some(n), &out)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn moments_match_scipy() {
        // scipy.stats on [2, 8, 0, 4, 1, 9, 9, 0]
        let x = NdArray::from_vec(vec![2.0, 8.0, 0.0, 4.0, 1.0, 9.0, 9.0, 0.0], &[8]).unwrap();
        let one = |a: NdArray<f64>| a.as_slice()[0];
        assert!((one(skew_axis(x.view(), 0, true).unwrap()) - 0.2650554122698573).abs() < 1e-12);
        assert!((one(skew_axis(x.view(), 0, false).unwrap()) - 0.3305821804079746).abs() < 1e-12);
        assert!((one(kurtosis_axis(x.view(), 0, true, true).unwrap()) - -1.6660010752838508).abs() < 1e-12);
        assert!((one(kurtosis_axis(x.view(), 0, false, false).unwrap()) - 0.9013977419039132).abs() < 1e-12);
        assert!((one(moment_axis(x.view(), 3, 0).unwrap()) - 13.67578125).abs() < 1e-10);
        let z = zscore_axis(x.view(), 0, 1).unwrap();
        assert!((z.as_slice()[0] - -0.533938378146085).abs() < 1e-12);
    }
}
