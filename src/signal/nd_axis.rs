//! Operations along one axis of n-d data: reductions (min, max, product, argmin / argmax, variance,
//! standard deviation), cumulative sums and products, sorting, differences; and joining arrays
//! ([`concatenate`], [`stack`]). NumPy's semantics: NaN propagates through min / max and sorts last.

use std::cmp::Ordering;

use super::ndarray::{check_axis, NdArray, NdError, NdView};
use super::Storage;
use crate::units::*;

/// Runs `f` on every lane along `axis` (copied out when strided), writing `out_len` results per
/// lane; the result has `axis` of length `out_len` (removed when `keep` is false and `out_len` is 1).
fn lanes_to<T: Copy, R: Copy + Default>(v: &NdView<'_, T>, axis: usize, out_len: usize, keep: bool, mut f: impl FnMut(&[T], &mut [R])) -> Result<NdArray<R>, NdError> {
    check_axis(axis, v.ndim())?;
    let mut shape = v.shape().to_vec();
    shape[axis] = out_len;
    let mut out = NdArray::<R>::zeros(&shape)?;
    let mut buf: Vec<T> = Vec::with_capacity(v.shape()[axis]);
    let mut res = vec![R::default(); out_len];
    for (lane, mut dst) in v.lanes(axis)?.zip(out.lanes_mut(axis)?) {
        buf.clear();
        buf.extend(lane.iter().copied());
        f(&buf, &mut res);
        for (d, &r) in dst.iter_mut().zip(&res) {
            *d = r;
        }
    }
    if !keep && out_len == 1 {
        shape.remove(axis);
        return NdArray::from_vec(out.into_vec(), &shape);
    }
    Ok(out)
}

/// NumPy's total order for sorting: NaN last.
fn nan_last<T: Float>(a: &T, b: &T) -> Ordering {
    match (a._is_nan(), b._is_nan()) {
        (true, true) => Ordering::Equal,
        (true, false) => Ordering::Greater,
        (false, true) => Ordering::Less,
        _ => a.partial_cmp(b).unwrap_or(Ordering::Equal),
    }
}

impl<'a, T: Float + Default> NdView<'a, T> {
    /// The minimum along `axis` (removed); NaN if the lane holds one. Errors on an empty axis.
    pub fn min_axis(&self, axis: usize) -> Result<NdArray<T>, NdError> {
        self.extreme(axis, |a, b| a < b)
    }

    /// The maximum along `axis` (removed); NaN if the lane holds one.
    pub fn max_axis(&self, axis: usize) -> Result<NdArray<T>, NdError> {
        self.extreme(axis, |a, b| a > b)
    }

    fn extreme(&self, axis: usize, better: fn(T, T) -> bool) -> Result<NdArray<T>, NdError> {
        check_axis(axis, self.ndim())?;
        if self.shape()[axis] == 0 {
            return Err(NdError::OutOfBounds);
        }
        lanes_to(self, axis, 1, false, |lane, out| {
            out[0] = lane.iter().skip(1).fold(lane[0], |m, &x| if m._is_nan() || x._is_nan() { T::_NAN } else if better(x, m) { x } else { m });
        })
    }

    /// The product along `axis` (removed).
    pub fn prod_axis(&self, axis: usize) -> Result<NdArray<T>, NdError> {
        lanes_to(self, axis, 1, false, |lane, out| out[0] = lane.iter().fold(T::_ONE, |p, &x| p * x))
    }

    /// The index of the minimum along `axis` (removed); the first NaN wins, as in NumPy.
    pub fn argmin_axis(&self, axis: usize) -> Result<NdArray<usize>, NdError> {
        self.arg_extreme(axis, |a, b| a < b)
    }

    /// The index of the maximum along `axis` (removed); the first NaN wins.
    pub fn argmax_axis(&self, axis: usize) -> Result<NdArray<usize>, NdError> {
        self.arg_extreme(axis, |a, b| a > b)
    }

    fn arg_extreme(&self, axis: usize, better: fn(T, T) -> bool) -> Result<NdArray<usize>, NdError> {
        check_axis(axis, self.ndim())?;
        if self.shape()[axis] == 0 {
            return Err(NdError::OutOfBounds);
        }
        lanes_to(self, axis, 1, false, |lane: &[T], out: &mut [usize]| {
            let mut best = 0;
            for (i, &x) in lane.iter().enumerate() {
                if x._is_nan() {
                    best = i;
                    break;
                }
                if better(x, lane[best]) {
                    best = i;
                }
            }
            out[0] = best;
        })
    }

    /// The variance along `axis` (removed), dividing by `n - ddof` (0: population, 1: sample).
    pub fn var_axis(&self, axis: usize, ddof: usize) -> Result<NdArray<T>, NdError> {
        lanes_to(self, axis, 1, false, |lane, out| {
            let n = lane.len();
            let mean = lane.iter().fold(T::_ZERO, |s, &x| s + x) / T::_lit(n as f64);
            let ss = lane.iter().fold(T::_ZERO, |s, &x| s + (x - mean) * (x - mean));
            out[0] = if n > ddof { ss / T::_lit((n - ddof) as f64) } else { T::_NAN };
        })
    }

    /// The standard deviation along `axis` (removed), with `ddof` as in [`var_axis`](Self::var_axis).
    pub fn std_axis(&self, axis: usize, ddof: usize) -> Result<NdArray<T>, NdError> {
        Ok(self.var_axis(axis, ddof)?.map(|v| v._sqrt()))
    }

    /// Running sums along `axis` (same shape).
    pub fn cumsum_axis(&self, axis: usize) -> Result<NdArray<T>, NdError> {
        check_axis(axis, self.ndim())?;
        lanes_to(self, axis, self.shape()[axis], true, |lane, out| {
            let mut acc = T::_ZERO;
            for (o, &x) in out.iter_mut().zip(lane) {
                acc = acc + x;
                *o = acc;
            }
        })
    }

    /// Running products along `axis` (same shape).
    pub fn cumprod_axis(&self, axis: usize) -> Result<NdArray<T>, NdError> {
        check_axis(axis, self.ndim())?;
        lanes_to(self, axis, self.shape()[axis], true, |lane, out| {
            let mut acc = T::_ONE;
            for (o, &x) in out.iter_mut().zip(lane) {
                acc = acc * x;
                *o = acc;
            }
        })
    }

    /// Each lane along `axis` sorted ascending (NaN last).
    pub fn sort_axis(&self, axis: usize) -> Result<NdArray<T>, NdError> {
        check_axis(axis, self.ndim())?;
        lanes_to(self, axis, self.shape()[axis], true, |lane, out| {
            out.copy_from_slice(lane);
            out.sort_by(nan_last);
        })
    }

    /// The indices that sort each lane along `axis` (stable: equal values keep their order).
    pub fn argsort_axis(&self, axis: usize) -> Result<NdArray<usize>, NdError> {
        check_axis(axis, self.ndim())?;
        lanes_to(self, axis, self.shape()[axis], true, |lane: &[T], out: &mut [usize]| {
            for (i, o) in out.iter_mut().enumerate() {
                *o = i;
            }
            out.sort_by(|&a, &b| nan_last(&lane[a], &lane[b]));
        })
    }

    /// The `n`-th discrete difference along `axis` (that axis shrinks by `n`).
    pub fn diff_axis(&self, axis: usize, n: usize) -> Result<NdArray<T>, NdError> {
        check_axis(axis, self.ndim())?;
        let len = self.shape()[axis].saturating_sub(n);
        lanes_to(self, axis, len, true, |lane, out| {
            let mut d = lane.to_vec();
            for _ in 0..n {
                for i in 0..d.len().saturating_sub(1) {
                    d[i] = d[i + 1] - d[i];
                }
                d.pop();
            }
            out.copy_from_slice(&d[..len]);
        })
    }
}

/// Joins arrays along an existing `axis` (`numpy.concatenate`): every other axis must match.
pub fn concatenate<T: Copy + Default>(arrays: &[NdView<'_, T>], axis: usize) -> Result<NdArray<T>, NdError> {
    let first = arrays.first().ok_or(NdError::OutOfBounds)?;
    check_axis(axis, first.ndim())?;
    let mut shape = first.shape().to_vec();
    shape[axis] = 0;
    for a in arrays {
        let same = a.ndim() == first.ndim() && a.shape().iter().zip(first.shape()).enumerate().all(|(i, (x, y))| i == axis || x == y);
        if !same {
            return Err(NdError::ShapeMismatch { expected: first.len(), got: a.len() });
        }
        shape[axis] += a.shape()[axis];
    }
    // contiguous parts: for each index of the axes before `axis`, each part's block in turn
    let slices: Option<Vec<&[T]>> = arrays.iter().map(|a| a.as_slice()).collect();
    if let Some(slices) = slices {
        let outer: usize = shape[..axis].iter().product();
        let blocks: Vec<usize> = arrays.iter().map(|a| a.shape()[axis..].iter().product()).collect();
        let mut data = Vec::with_capacity(shape.iter().product());
        for o in 0..outer {
            for (s, &b) in slices.iter().zip(&blocks) {
                data.extend_from_slice(&s[o * b..(o + 1) * b]);
            }
        }
        return NdArray::from_vec(data, &shape);
    }
    let mut out = NdArray::<T>::zeros(&shape)?;
    let mut offset = 0;
    for a in arrays {
        let n = a.shape()[axis];
        let mut target = out.view_mut().slice_axis(axis, offset..offset + n)?;
        target.assign(a)?;
        offset += n;
    }
    Ok(out)
}

/// Joins arrays of one shape along a new `axis` (`numpy.stack`).
pub fn stack<T: Copy + Default>(arrays: &[NdView<'_, T>], axis: usize) -> Result<NdArray<T>, NdError> {
    let first = arrays.first().ok_or(NdError::OutOfBounds)?;
    check_axis(axis, first.ndim() + 1)?;
    let expanded: Vec<NdView<'_, T>> = arrays
        .iter()
        .map(|a| if a.shape() == first.shape() { a.insert_axis(axis) } else { Err(NdError::ShapeMismatch { expected: first.len(), got: a.len() }) })
        .collect::<Result<_, _>>()?;
    concatenate(&expanded, axis)
}

/// The same operations on owned arrays.
macro_rules! forward {
    ($($name:ident($($arg:ident: $ty:ty),*) -> $out:ty;)+) => {
        impl<T: Float + Default, S: Storage<Elem = T>> NdArray<T, S> {
            $(
                #[doc = concat!("See [`NdView::", stringify!($name), "`].")]
                pub fn $name(&self, $($arg: $ty),*) -> Result<$out, NdError> {
                    self.view().$name($($arg),*)
                }
            )+
        }
    };
}

forward! {
    min_axis(axis: usize) -> NdArray<T>;
    max_axis(axis: usize) -> NdArray<T>;
    prod_axis(axis: usize) -> NdArray<T>;
    argmin_axis(axis: usize) -> NdArray<usize>;
    argmax_axis(axis: usize) -> NdArray<usize>;
    var_axis(axis: usize, ddof: usize) -> NdArray<T>;
    std_axis(axis: usize, ddof: usize) -> NdArray<T>;
    cumsum_axis(axis: usize) -> NdArray<T>;
    cumprod_axis(axis: usize) -> NdArray<T>;
    sort_axis(axis: usize) -> NdArray<T>;
    argsort_axis(axis: usize) -> NdArray<usize>;
    diff_axis(axis: usize, n: usize) -> NdArray<T>;
}

#[cfg(test)]
mod tests {
    use super::*;

    fn arr(data: &[f64], shape: &[usize]) -> NdArray<f64> {
        NdArray::from_vec(data.to_vec(), shape).unwrap()
    }

    #[test]
    fn reductions_along_axes() {
        let a = arr(&[3.0, 1.0, 2.0, 6.0, 5.0, 4.0], &[2, 3]);
        assert_eq!(a.min_axis(1).unwrap().as_slice(), &[1.0, 4.0]);
        assert_eq!(a.max_axis(0).unwrap().as_slice(), &[6.0, 5.0, 4.0]);
        assert_eq!(a.prod_axis(1).unwrap().as_slice(), &[6.0, 120.0]);
        assert_eq!(a.argmin_axis(1).unwrap().as_slice(), &[1, 2]);
        assert_eq!(a.argmax_axis(0).unwrap().as_slice(), &[1, 1, 1]);
        assert_eq!(a.var_axis(1, 0).unwrap().as_slice(), &[2.0 / 3.0, 2.0 / 3.0]);
        assert_eq!(a.std_axis(1, 1).unwrap().as_slice(), &[1.0, 1.0]);
        // on a transposed view: reductions follow the logical axes
        assert_eq!(a.view().transpose().min_axis(0).unwrap().as_slice(), &[1.0, 4.0]);
        let nan = arr(&[1.0, f64::NAN, 0.0], &[3]);
        assert!(nan.min_axis(0).unwrap().as_slice()[0].is_nan());
        assert_eq!(nan.argmax_axis(0).unwrap().as_slice(), &[1]);
    }

    #[test]
    fn cumulative_sorting_and_differences() {
        let a = arr(&[3.0, 1.0, 2.0, 6.0, 5.0, 4.0], &[2, 3]);
        assert_eq!(a.cumsum_axis(1).unwrap().as_slice(), &[3.0, 4.0, 6.0, 6.0, 11.0, 15.0]);
        assert_eq!(a.cumprod_axis(0).unwrap().as_slice(), &[3.0, 1.0, 2.0, 18.0, 5.0, 8.0]);
        assert_eq!(a.sort_axis(1).unwrap().as_slice(), &[1.0, 2.0, 3.0, 4.0, 5.0, 6.0]);
        assert_eq!(a.argsort_axis(1).unwrap().as_slice(), &[1, 2, 0, 2, 1, 0]);
        let with_nan = arr(&[2.0, f64::NAN, 1.0], &[3]).sort_axis(0).unwrap();
        assert_eq!(&with_nan.as_slice()[..2], &[1.0, 2.0]);
        assert!(with_nan.as_slice()[2].is_nan());
        assert_eq!(a.diff_axis(1, 1).unwrap().as_slice(), &[-2.0, 1.0, -1.0, -1.0]);
        assert_eq!(a.diff_axis(1, 2).unwrap().as_slice(), &[3.0, 0.0]);
    }

    #[test]
    fn concatenate_and_stack() {
        let a = arr(&[1.0, 2.0, 3.0, 4.0], &[2, 2]);
        let b = arr(&[5.0, 6.0], &[1, 2]);
        let c = concatenate(&[a.view(), b.view()], 0).unwrap();
        assert_eq!(c.shape(), &[3, 2]);
        assert_eq!(c.as_slice(), &[1.0, 2.0, 3.0, 4.0, 5.0, 6.0]);
        let d = concatenate(&[a.view(), a.view().transpose()], 1).unwrap();
        assert_eq!(d.as_slice(), &[1.0, 2.0, 1.0, 3.0, 3.0, 4.0, 2.0, 4.0]);
        let s = stack(&[a.view(), a.view()], 2).unwrap();
        assert_eq!(s.shape(), &[2, 2, 2]);
        assert_eq!(s.as_slice(), &[1.0, 1.0, 2.0, 2.0, 3.0, 3.0, 4.0, 4.0]);
        assert!(concatenate(&[a.view(), b.view()], 1).is_err());
    }
}
