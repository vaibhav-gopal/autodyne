//! Covariance and correlation matrices (`numpy.cov`, `numpy.corrcoef`).

use super::StatsError;
use crate::signal::{NdArray, NdView};
use crate::units::*;

/// The variables of `x` as centred rows: `x` is 1-D (one variable) or 2-D with variables on rows
/// (`rowvar`) or columns.
fn centred_rows<T: Float>(x: NdView<'_, T>, rowvar: bool) -> Result<Vec<Vec<f64>>, StatsError> {
    let f = |v: &T| v.to_f64().unwrap_or(f64::NAN);
    let rows: Vec<Vec<f64>> = match x.ndim() {
        1 => vec![x.iter().map(f).collect()],
        2 => {
            let v = if rowvar { x } else { x.transpose() };
            v.lanes(1)?.map(|l| l.iter().map(f).collect()).collect()
        }
        d => return Err(StatsError::invalid(format!("cov needs a 1-D or 2-D array, got {d} dimensions"))),
    };
    Ok(rows
        .into_iter()
        .map(|mut r| {
            let mean = r.iter().sum::<f64>() / r.len() as f64;
            r.iter_mut().for_each(|v| *v -= mean);
            r
        })
        .collect())
}

/// The covariance matrix of the variables of `x` (`numpy.cov`): rows are variables and columns
/// observations when `rowvar` (else the other way round); a 1-D input is one variable. Entry
/// `(i, j)` is `sum (x_i - mean_i)(x_j - mean_j) / (n - ddof)` (`ddof` 1: the unbiased estimate).
/// Errors unless there are more than `ddof` observations.
pub fn cov<T: Float + Default>(x: NdView<'_, T>, rowvar: bool, ddof: usize) -> Result<NdArray<T>, StatsError> {
    let rows = centred_rows(x, rowvar)?;
    let n = rows[0].len();
    if n <= ddof {
        return Err(StatsError::invalid(format!("cov needs more than ddof ({ddof}) observations, got {n}")));
    }
    let k = rows.len();
    let scale = 1.0 / (n - ddof) as f64;
    // X Xᵀ as one matrix product (the blocked GEMM) when linalg is there, else a dot per pair
    #[cfg(feature = "faer")]
    let c: Vec<f64> = {
        let x = NdArray::from_vec(rows.concat(), &[k, n])?;
        crate::linalg::matmul(x.view(), x.view().transpose())?.into_vec()
    };
    #[cfg(not(feature = "faer"))]
    let c: Vec<f64> = {
        let mut c = vec![0.0; k * k];
        for i in 0..k {
            for j in i..k {
                let v = crate::simd::dot(&rows[i], &rows[j]);
                c[i * k + j] = v;
                c[j * k + i] = v;
            }
        }
        c
    };
    Ok(NdArray::from_vec(c.into_iter().map(|v| T::_lit(v * scale)).collect(), &[k, k])?)
}

/// The Pearson correlation matrix of the variables of `x` (`numpy.corrcoef`): the covariance
/// normalized by the standard deviations, clipped to [-1, 1]. A constant variable gives NaN.
pub fn corrcoef<T: Float + Default>(x: NdView<'_, T>, rowvar: bool) -> Result<NdArray<T>, StatsError> {
    let c = cov(x, rowvar, 1)?;
    let k = c.shape()[0];
    let c: Vec<f64> = c.as_slice().iter().map(|v| v.to_f64().unwrap_or(f64::NAN)).collect();
    let d: Vec<f64> = (0..k).map(|i| c[i * k + i].sqrt()).collect();
    let r = (0..k * k).map(|ij| T::_lit((c[ij] / d[ij / k] / d[ij % k]).clamp(-1.0, 1.0)));
    Ok(NdArray::from_vec(r.collect(), &[k, k])?)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn covariance_and_correlation_follow_numpy() {
        // numpy.cov([[0, 1, 2], [2, 1, 0.5]]) and corrcoef
        let x = NdArray::from_vec(vec![0.0, 1.0, 2.0, 2.0, 1.0, 0.5], &[2, 3]).unwrap();
        let c = cov(x.view(), true, 1).unwrap();
        let want = [1.0, -0.75, -0.75, 0.5833333333333334];
        assert!(c.as_slice().iter().zip(want).all(|(a, b)| (a - b).abs() < 1e-12), "{:?}", c.as_slice());
        // the same data with variables in columns
        let t = x.view().transpose().to_owned();
        assert_eq!(cov(t.view(), false, 1).unwrap().as_slice(), c.as_slice());
        let r = corrcoef(x.view(), true).unwrap();
        assert!((r.as_slice()[1] - -0.9819805060619657).abs() < 1e-12 && r.as_slice()[0] == 1.0);
        assert!(cov(NdArray::from_vec(vec![1.0], &[1]).unwrap().view(), true, 1).is_err());
    }
}
