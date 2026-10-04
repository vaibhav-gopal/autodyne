//! Order statistics along an axis: [`median_axis`] and [`quantile_axis`] (`numpy.median`,
//! `numpy.quantile` / `numpy.percentile`), found by selection (`O(n)`) rather than sorting.

use super::{assemble, StatsError};
use crate::signal::{lanes_f64, NdArray, NdView};
use crate::units::*;

/// How a quantile between two samples is chosen (`method` in `numpy.quantile`).
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub enum QuantileMethod {
    /// Linear interpolation between the two (NumPy's default).
    #[default]
    Linear,
    /// The lower one.
    Lower,
    /// The higher one.
    Higher,
    /// The nearer one (ties to the even index, as NumPy rounds).
    Nearest,
    /// Their average.
    Midpoint,
}

/// NumPy's `_lerp`: `a + t (b - a)`, written from `b`'s side above one half so the result is
/// monotonic in `t` and exact at both ends.
fn lerp(a: f64, b: f64, t: f64) -> f64 {
    let d = b - a;
    if t >= 0.5 { b - d * (1.0 - t) } else { a + d * t }
}

/// The `q`-quantile of `lane` (reordered in place), NaN if it holds one.
fn quantile_of(lane: &mut [f64], q: f64, method: QuantileMethod) -> f64 {
    let n = lane.len();
    if n == 0 || lane.iter().any(|v| v.is_nan()) {
        return f64::NAN;
    }
    let h = (n - 1) as f64 * q;
    let lo = h.floor() as usize;
    let hi = (lo + 1).min(n - 1);
    let mut select = |k: usize| -> f64 {
        let (_, v, _) = lane.select_nth_unstable_by(k, f64::total_cmp);
        *v
    };
    match method {
        QuantileMethod::Lower => select(lo),
        QuantileMethod::Higher => select(h.ceil() as usize),
        QuantileMethod::Nearest => select(h.round_ties_even() as usize),
        QuantileMethod::Linear | QuantileMethod::Midpoint => {
            let a = select(lo);
            // after selecting lo, everything above it is in lane[lo + 1..]: its minimum is the next one
            let b = if hi == lo { a } else { lane[lo + 1..].iter().copied().fold(f64::INFINITY, f64::min) };
            // NumPy's midpoint is the sample itself when h falls on one
            let t = if method == QuantileMethod::Midpoint { if h == lo as f64 { 0.0 } else { 0.5 } } else { h - lo as f64 };
            lerp(a, b, t)
        }
    }
}

fn check_q(q: f64) -> Result<f64, StatsError> {
    if (0.0..=1.0).contains(&q) { Ok(q) } else { Err(StatsError::invalid(format!("quantiles must be in [0, 1], got {q}"))) }
}

/// The `q`-quantiles (each in [0, 1]) of the lanes along `axis` (`numpy.quantile`): the axis is
/// replaced by one of length `q.len()` (or removed when `q` has one value and `keep` is false). A
/// lane holding NaN gives NaN, as in NumPy; an empty one errors.
pub fn quantile_axis<T: Float + Default>(x: NdView<'_, T>, q: &[f64], axis: usize, method: QuantileMethod, keep: bool) -> Result<NdArray<T>, StatsError> {
    let q: Vec<f64> = q.iter().map(|&v| check_q(v)).collect::<Result<_, _>>()?;
    if q.is_empty() {
        return Err(StatsError::invalid("no quantiles asked for"));
    }
    let mut lanes = lanes_f64(&x, axis)?;
    if x.shape()[axis] == 0 {
        return Err(StatsError::invalid("quantiles of an empty axis"));
    }
    let out: Vec<Vec<f64>> = lanes.iter_mut().map(|lane| q.iter().map(|&p| quantile_of(lane, p, method)).collect()).collect();
    let len = if q.len() == 1 && !keep { None } else { Some(q.len()) };
    assemble(x.shape(), axis, len, &out)
}

/// The medians of the lanes along `axis` (removed) (`numpy.median`): the middle value, or the
/// mean of the two middle values; NaN where a lane holds NaN.
pub fn median_axis<T: Float + Default>(x: NdView<'_, T>, axis: usize) -> Result<NdArray<T>, StatsError> {
    let mut lanes = lanes_f64(&x, axis)?;
    let n = x.shape()[axis];
    if n == 0 {
        return Err(StatsError::invalid("the median of an empty axis"));
    }
    let out: Vec<Vec<f64>> = lanes
        .iter_mut()
        .map(|lane| {
            if lane.iter().any(|v| v.is_nan()) {
                return vec![f64::NAN];
            }
            let (_, &mut m, _) = lane.select_nth_unstable_by(n / 2, f64::total_cmp);
            if n % 2 == 1 {
                return vec![m];
            }
            // the lower middle value is the largest of the lower half
            let below = lane[..n / 2].iter().copied().fold(f64::NEG_INFINITY, f64::max);
            vec![(below + m) / 2.0]
        })
        .collect();
    assemble(x.shape(), axis, None, &out)
}

/// [`quantile_axis`] with `q` in percent (`numpy.percentile`).
pub fn percentile_axis<T: Float + Default>(x: NdView<'_, T>, q: &[f64], axis: usize, method: QuantileMethod, keep: bool) -> Result<NdArray<T>, StatsError> {
    let fractions: Vec<f64> = q.iter().map(|p| p / 100.0).collect();
    quantile_axis(x, &fractions, axis, method, keep)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn arr(v: &[f64], shape: &[usize]) -> NdArray<f64> {
        NdArray::from_vec(v.to_vec(), shape).unwrap()
    }

    #[test]
    fn quantiles_follow_numpy() {
        let x = arr(&[3.0, 1.0, 4.0, 1.5, 9.0, 2.0, 6.0], &[7]);
        // numpy.quantile(x, [0, .1, .25, .5, .9, 1]) and the other methods at 0.4
        let q = quantile_axis(x.view(), &[0.0, 0.1, 0.25, 0.5, 0.9, 1.0], 0, QuantileMethod::Linear, true).unwrap();
        let want = [1.0, 1.3, 1.75, 3.0, 7.199999999999999, 9.0];
        assert!(q.as_slice().iter().zip(want).all(|(a, b)| (a - b).abs() < 1e-12), "{:?}", q.as_slice());
        let at = |m| quantile_axis(x.view(), &[0.4], 0, m, false).unwrap().as_slice()[0];
        assert_eq!(at(QuantileMethod::Lower), 2.0);
        assert_eq!(at(QuantileMethod::Higher), 3.0);
        assert_eq!(at(QuantileMethod::Nearest), 2.0); // h = 2.4 -> 2
        assert_eq!(at(QuantileMethod::Midpoint), 2.5);
        assert!((at(QuantileMethod::Linear) - 2.4).abs() < 1e-12);
        assert!(quantile_axis(x.view(), &[1.5], 0, QuantileMethod::Linear, false).is_err());
    }

    #[test]
    fn medians_along_axes() {
        let x = arr(&[3.0, 1.0, 2.0, 6.0, 5.0, 4.0, 0.0, 8.0], &[2, 4]);
        assert_eq!(median_axis(x.view(), 1).unwrap().as_slice(), &[2.5, 4.5]);
        assert_eq!(median_axis(x.view(), 0).unwrap().as_slice(), &[4.0, 2.5, 1.0, 7.0]);
        assert_eq!(median_axis(arr(&[5.0, 1.0, 3.0], &[3]).view(), 0).unwrap().as_slice(), &[3.0]);
        assert!(median_axis(arr(&[1.0, f64::NAN], &[2]).view(), 0).unwrap().as_slice()[0].is_nan());
        let p = percentile_axis(x.view(), &[50.0], 1, QuantileMethod::Linear, false).unwrap();
        assert_eq!(p.as_slice(), &[2.5, 4.5]);
    }
}
