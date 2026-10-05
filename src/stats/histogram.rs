//! Histograms as NumPy makes them (`numpy.histogram`, `numpy.histogram_bin_edges`): a number of
//! equal bins, explicit edges, or a bin-width rule; the last bin includes its right edge.

use crate::alloc_prelude::*;
use super::{QuantileMethod, StatsError};
use crate::signal::NdArray;
use crate::units::*;

/// How [`histogram`] chooses its bins (`bins` in `numpy.histogram`).
#[derive(Clone, Debug, PartialEq)]
pub enum Bins {
    /// This many equal bins over the range.
    Count(usize),
    /// These edges (increasing): `edges.len() - 1` bins.
    Edges(Vec<f64>),
    /// Equal bins whose width a rule estimates from the data.
    Rule(BinRule),
}

/// NumPy's bin-width estimators (`bins="auto"`, `"fd"`, ...).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum BinRule {
    /// The smaller of the Sturges width and the Freedman-Diaconis width, the latter at least half
    /// the `Sqrt` width so heavy tails don't make thousands of bins (NumPy 2's rule).
    Auto,
    /// Freedman-Diaconis: `2 IQR n^(-1/3)`, robust to outliers.
    Fd,
    /// Sturges: `range / (log2 n + 1)`, for small, roughly normal samples.
    Sturges,
    /// Scott: `(24 √π / n)^(1/3) σ`.
    Scott,
    /// Rice: `range / (2 n^(1/3))`.
    Rice,
    /// `range / √n`.
    Sqrt,
}

fn min_max(x: &[f64]) -> (f64, f64) {
    x.iter().fold((f64::INFINITY, f64::NEG_INFINITY), |(lo, hi), &v| (lo.min(v), hi.max(v)))
}

/// The bin width `rule` estimates for `x` (non-empty, finite).
fn rule_width(rule: BinRule, x: &[f64]) -> Result<f64, StatsError> {
    let n = x.len() as f64;
    let (lo, hi) = min_max(x);
    let ptp = hi - lo;
    let sturges = ptp / (n.log2() + 1.0);
    let fd = || -> Result<f64, StatsError> {
        let a = NdArray::from_vec(x.to_vec(), &[x.len()])?;
        let q = super::quantile_axis(a.view(), &[0.75, 0.25], 0, QuantileMethod::Linear, true)?;
        Ok(2.0 * (q.as_slice()[0] - q.as_slice()[1]) * n.powf(-1.0 / 3.0))
    };
    Ok(match rule {
        BinRule::Sturges => sturges,
        BinRule::Sqrt => ptp / n.sqrt(),
        BinRule::Rice => ptp / (2.0 * n.powf(1.0 / 3.0)),
        BinRule::Scott => {
            let mean = x.iter().sum::<f64>() / n;
            let std = (x.iter().map(|v| (v - mean) * (v - mean)).sum::<f64>() / n).sqrt();
            (24.0 * core::f64::consts::PI.sqrt() / n).powf(1.0 / 3.0) * std
        }
        BinRule::Fd => fd()?,
        // NumPy 2: the FD width, at least half the √n width (capping the bin count), or Sturges'
        BinRule::Auto => fd()?.max(ptp / n.sqrt() / 2.0).min(sturges),
    })
}

/// The bin edges [`histogram`] would use (`numpy.histogram_bin_edges`): over `range` (default:
/// the data's minimum and maximum, widened by 0.5 each way when they are equal). Errors on
/// non-finite data or range, decreasing edges, or zero bins.
pub fn histogram_bin_edges<T: Float>(x: &[T], bins: &Bins, range: Option<(f64, f64)>) -> Result<Vec<f64>, StatsError> {
    let data: Vec<f64> = x.iter().map(|v| v.to_f64().unwrap_or(f64::NAN)).collect();
    if let Bins::Edges(e) = bins {
        if e.len() < 2 || e.windows(2).any(|w| w[0] > w[1]) {
            return Err(StatsError::invalid("bin edges must increase monotonically, at least two of them"));
        }
        return Ok(e.clone());
    }
    let (mut first, mut last) = match range {
        Some((lo, hi)) => {
            if lo > hi || !lo.is_finite() || !hi.is_finite() {
                return Err(StatsError::invalid(format!("the range must be finite with min <= max, got ({lo}, {hi})")));
            }
            (lo, hi)
        }
        None if data.is_empty() => (0.0, 1.0),
        None => {
            let (lo, hi) = min_max(&data);
            if !lo.is_finite() || !hi.is_finite() {
                return Err(StatsError::invalid("the data range is not finite (NaN or infinite values)"));
            }
            (lo, hi)
        }
    };
    if first == last {
        first -= 0.5;
        last += 0.5;
    }
    let count = match bins {
        Bins::Count(0) => return Err(StatsError::invalid("at least one bin is needed")),
        Bins::Count(c) => *c,
        Bins::Rule(rule) => {
            let inside: Vec<f64> = data.iter().copied().filter(|&v| v >= first && v <= last).collect();
            if inside.is_empty() {
                1
            } else {
                let w = rule_width(*rule, &inside)?;
                if w > 0.0 { ((last - first) / w).ceil() as usize } else { 1 }
            }
        }
        Bins::Edges(_) => unreachable!("handled above"),
    };
    // numpy.linspace(first, last, count + 1)
    let step = (last - first) / count as f64;
    Ok((0..=count).map(|i| if i == count { last } else { first + i as f64 * step }).collect())
}

/// A histogram of `x` (`numpy.histogram`): `(values, edges)`, where bin `i` counts the samples in
/// `[edges[i], edges[i + 1])` (the last bin also its right edge); samples outside the range are
/// left out. With `weights`, each sample adds its weight instead of 1; with `density`, the values
/// are divided by the total and the bin widths so they integrate to 1.
pub fn histogram<T: Float>(x: &[T], bins: &Bins, range: Option<(f64, f64)>, weights: Option<&[f64]>, density: bool) -> Result<(Vec<f64>, Vec<f64>), StatsError> {
    if let Some(w) = weights {
        if w.len() != x.len() {
            return Err(StatsError::invalid(format!("{} weights for {} samples", w.len(), x.len())));
        }
    }
    let edges = histogram_bin_edges(x, bins, range)?;
    let count = edges.len() - 1;
    let (first, last) = (edges[0], edges[count]);
    let mut hist = vec![0.0; count];
    let uniform = matches!(bins, Bins::Count(_) | Bins::Rule(_));
    let norm = count as f64 / (last - first);
    for (i, v) in x.iter().enumerate() {
        let v = v.to_f64().unwrap_or(f64::NAN);
        if !(v >= first && v <= last) {
            continue;
        }
        let bin = if uniform {
            // NumPy's computed index, corrected against the edges it rounds near
            let mut b = (((v - first) * norm) as usize).min(count - 1);
            if v < edges[b] {
                b -= 1;
            } else if b + 1 < count && v >= edges[b + 1] {
                b += 1;
            }
            b
        } else {
            // the last edge at or below v, the right edge closing the last bin
            edges.partition_point(|&e| e <= v).saturating_sub(1).min(count - 1)
        };
        hist[bin] += weights.map_or(1.0, |w| w[i]);
    }
    if density {
        let total: f64 = hist.iter().sum();
        for (h, w) in hist.iter_mut().zip(edges.windows(2)) {
            *h /= total * (w[1] - w[0]);
        }
    }
    Ok((hist, edges))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn histograms_follow_numpy() {
        let x = [0.5, 1.0, 1.0, 2.5, 3.0, 3.0, 3.0, 4.0];
        // numpy.histogram(x, 4): [3, 0, 4, 1] over [0.5, 1.375, 2.25, 3.125, 4]
        let (h, e) = histogram(&x, &Bins::Count(4), None, None, false).unwrap();
        assert_eq!(h, [3.0, 0.0, 4.0, 1.0]);
        assert_eq!(e, [0.5, 1.375, 2.25, 3.125, 4.0]);
        // explicit edges: the last bin includes 4; 0.5 is left out
        let (h, _) = histogram(&x, &Bins::Edges(vec![1.0, 2.0, 4.0]), None, None, false).unwrap();
        assert_eq!(h, [2.0, 5.0]);
        let (h, _) = histogram(&x, &Bins::Count(2), Some((0.0, 4.0)), Some(&[1.0, 2.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0]), true).unwrap();
        assert_eq!(h, [0.375, 0.125]);
        // a constant sample: one unit-wide bin around it
        let (h, e) = histogram(&[2.0, 2.0], &Bins::Count(1), None, None, false).unwrap();
        assert_eq!((h, e), (vec![2.0], vec![1.5, 2.5]));
        assert!(histogram(&x, &Bins::Edges(vec![2.0, 1.0]), None, None, false).is_err());
    }
}
