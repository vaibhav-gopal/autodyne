//! [`ArrayMath`]: array operations written once and run either eagerly on [`NdArray`]s or traced
//! (`flux::Tracer`), like NumPy and JAX's `jax.numpy` sharing one API.

use std::ops::{Add, Mul};

use super::nd_ops::{broadcast_shapes, Zip};
use super::ndarray::NdArray;
use crate::spectral::{Fft, RealFft};
use crate::units::*;

/// Array maths on top of [`Elementwise`]: shapes, broadcasting, reductions and tensor products.
/// [`RealArrayMath`] adds ordering and real FFTs; [`ComplexArrayMath`] complex FFTs.
///
/// Implemented eagerly by `NdArray<f32 / f64 / Complex>` (every call computes a new array), by the
/// runtime-typed `DynArray`, and by `flux::Tracer` (every call records a node), so one generic
/// function serves all: run it on arrays, or trace it to differentiate it and compile it with XLA
/// or IREE.
///
/// Arithmetic operators and the [`Elementwise`] functions broadcast NumPy-style. Values are taken
/// by value; `clone()` one to use it twice (free for tracers).
///
/// ```
/// use autodyne::signal::{NdArray, RealArrayMath};
///
/// /// Filters each row of `x` by a per-bin gain, in the frequency domain.
/// fn spectral_gain<A: RealArrayMath>(x: A, gain: A) -> A {
///     let n = *x.shape().last().unwrap();
///     let (re, im) = x.rfft();
///     A::irfft(re * gain.clone(), im * gain, n)
/// }
///
/// let x = NdArray::from_vec(vec![1.0f64, 0.0, 0.0, 0.0], &[1, 4]).unwrap(); // an impulse
/// let gain = NdArray::from_vec(vec![1.0, 0.5, 0.0], &[3]).unwrap();
/// let y = spectral_gain(x, gain);
/// assert_eq!(y.shape(), &[1, 4]);
/// assert!((y.as_slice()[0] - 0.5).abs() < 1e-12); // (1 + 2 * 0.5 + 0) / 4
/// ```
pub trait ArrayMath: Elementwise {
    /// A constant array of `shape` from row-major `values` (windows, weights), rounded to this type.
    fn array(values: &[f64], shape: &[usize]) -> Self;
    /// The shape (`[]` for a scalar).
    fn shape(&self) -> Vec<usize>;
    /// Stretches to `shape`, NumPy style (trailing axes line up; axes of length 1 stretch).
    fn broadcast_to(self, shape: &[usize]) -> Self;
    /// Axis `i` becomes axis `dims[i]` of `shape` (`dims` increasing); axes of length 1 stretch and
    /// the other axes of `shape` repeat the value.
    fn broadcast_in_dim(self, shape: &[usize], dims: &[usize]) -> Self;
    /// The same elements (row-major) in another shape.
    fn reshape(self, shape: &[usize]) -> Self;
    /// Axis `i` of the result is axis `perm[i]` of `self`.
    fn transpose(self, perm: &[usize]) -> Self;
    /// Sum over `axes`, which are removed.
    fn sum_axes(self, axes: &[usize]) -> Self;
    /// Contracts axes `ca` of `self` with axes `cb` of `rhs` (pairwise). The result has the
    /// remaining axes of `self`, then those of `rhs`.
    fn dot_general(self, rhs: Self, ca: &[usize], cb: &[usize]) -> Self;
    /// Indices `start[i]..limit[i]` by `stride[i]` along each axis (`a[s:l:k, ...]`).
    fn slice(self, start: &[usize], limit: &[usize], stride: &[usize]) -> Self;
    /// Zero padding: `low[i]` zeros before axis `i`, `high[i]` after, and `interior[i]` between
    /// neighbouring elements.
    fn pad(self, low: &[usize], high: &[usize], interior: &[usize]) -> Self;
    /// Joins arrays along `axis` (every other axis must match), like `numpy.concatenate`.
    fn concatenate(parts: &[Self], axis: usize) -> Self;
    /// Product over `axes`, which are removed.
    fn prod_axes(self, axes: &[usize]) -> Self;
    /// The elements in reverse order along each of `axes` (`numpy.flip`).
    fn reverse(self, axes: &[usize]) -> Self;

    /// `start..end` along `axis`, the other axes whole.
    fn slice_axis(self, axis: usize, start: usize, end: usize) -> Self {
        let shape = self.shape();
        let (mut lo, mut hi) = (vec![0; shape.len()], shape.clone());
        (lo[axis], hi[axis]) = (start, end);
        self.slice(&lo, &hi, &vec![1; shape.len()])
    }

    /// Sum of every element (a scalar).
    fn sum_all(self) -> Self {
        let n = self.shape().len();
        self.sum_axes(&(0..n).collect::<Vec<_>>())
    }
    /// Mean of every element (a scalar).
    fn mean_all(self) -> Self {
        let count: usize = self.shape().iter().product();
        self.sum_all() / Self::lit(count as f64)
    }
    /// Contracts the last axis of `self` with the first of `rhs`: matrix · vector, vector · vector
    /// (a scalar), matrix · matrix.
    fn dot(self, rhs: Self) -> Self {
        let n = self.shape().len();
        assert!(n >= 1 && !rhs.shape().is_empty(), "dot: both operands need an axis");
        self.dot_general(rhs, &[n - 1], &[0])
    }
}

/// Real arrays: [`ArrayMath`] with ordering ([`RealValued`]), real FFTs and their complex
/// counterpart type.
///
/// Spectra are values of [`Complex`](Self::Complex) (`NdArray<Complex<T>>` for `NdArray<T>`; the
/// same type for `DynArray` and `flux::Tracer`), so a spectral computation stays complex from one
/// FFT to the next: `A::irfft_complex(x.rfft_complex() * gain.to_complex(), n)`. The `(real,
/// imaginary)` pair methods are built on these.
pub trait RealArrayMath: ArrayMath + RealValued {
    /// Complex arrays of the same precision.
    type Complex: ComplexArrayMath;
    /// The real FFT along the last axis: `n` samples become `n / 2 + 1` complex bins.
    fn rfft_complex(self) -> Self::Complex;
    /// The inverse of [`rfft_complex`](Self::rfft_complex): `n` samples along the last axis from
    /// `n / 2 + 1` bins, scaled by `1 / n`. The imaginary parts of bins 0 and `n / 2` are ignored.
    fn irfft_complex(spectrum: Self::Complex, n: usize) -> Self;
    /// `self + 0i`.
    fn to_complex(self) -> Self::Complex;
    /// `re + i·im`.
    fn complex(re: Self, im: Self) -> Self::Complex;
    fn real_part(z: Self::Complex) -> Self;
    fn imag_part(z: Self::Complex) -> Self;

    /// The real FFT along the last axis: `(real parts, imaginary parts)`, `n / 2 + 1` bins each.
    fn rfft(self) -> (Self, Self) {
        let z = self.rfft_complex();
        (Self::real_part(z.clone()), Self::imag_part(z))
    }
    /// The inverse of [`rfft`](Self::rfft): `n` samples along the last axis from `n / 2 + 1` bins,
    /// scaled by `1 / n`. The imaginary parts of bins 0 and `n / 2` are ignored.
    fn irfft(re: Self, im: Self, n: usize) -> Self {
        Self::irfft_complex(Self::complex(re, im), n)
    }
    /// The complex DFT along the last axis of `re + i·im` (same shapes), as `(real parts,
    /// imaginary parts)`.
    fn fft_parts(re: Self, im: Self) -> (Self, Self) {
        let z = Self::complex(re, im).fft();
        (Self::real_part(z.clone()), Self::imag_part(z))
    }
    /// The inverse of [`fft_parts`](Self::fft_parts) (scaled by `1 / n`).
    fn ifft_parts(re: Self, im: Self) -> (Self, Self) {
        let z = Self::complex(re, im).ifft();
        (Self::real_part(z.clone()), Self::imag_part(z))
    }
    /// Maximum over `axes`, which are removed (NaN propagates; an empty axis gives -∞).
    fn max_axes(self, axes: &[usize]) -> Self;
    /// Minimum over `axes`, which are removed (NaN propagates; an empty axis gives +∞).
    fn min_axes(self, axes: &[usize]) -> Self;
    /// Rows of `self` along its first axis at `indices` (`numpy.take(self, indices, axis=0)`): the
    /// result has the indices' shape followed by `self`'s other axes. Indices are real numbers,
    /// rounded down and clamped to the table (so the lookup never fails, as in XLA); interpolate
    /// between `take(floor(i))` and `take(floor(i) + 1)` for fractional positions (wavetables,
    /// modulated delays). The gradient flows to the table, not the indices.
    fn take(self, indices: Self) -> Self;
}

/// Complex arrays: [`ArrayMath`] with complex FFTs along the last axis.
pub trait ComplexArrayMath: ArrayMath {
    /// The DFT along the last axis.
    fn fft(self) -> Self;
    /// The inverse DFT along the last axis (scaled by `1 / n`).
    fn ifft(self) -> Self;
    /// The complex conjugate.
    fn conj(self) -> Self;
}

/// Broadcasts both operands to their common shape (panics, naming the shapes, if they can't).
fn aligned<'a, T: Copy>(a: &'a NdArray<T>, b: &'a NdArray<T>) -> (super::NdView<'a, T>, super::NdView<'a, T>) {
    let (shape, n) = broadcast_shapes(a.shape(), b.shape()).unwrap_or_else(|_| panic!("shapes {:?} and {:?} do not broadcast", a.shape(), b.shape()));
    (a.view().broadcast_to(&shape[..n]).expect("broadcasts"), b.view().broadcast_to(&shape[..n]).expect("broadcasts"))
}

fn zip_with<T: Copy, R: Copy>(a: &NdArray<T>, b: &NdArray<T>, f: impl Fn(T, T) -> R) -> NdArray<R> {
    let (a, b) = aligned(a, b);
    Zip::from(a).and(b).expect("same shape").map_collect(|&x, &y| f(x, y))
}

/// `broadcast_in_dim` for any element type (masks included).
pub(crate) fn broadcast_in_dim<T: Copy>(a: &NdArray<T>, shape: &[usize], dims: &[usize]) -> NdArray<T> {
    assert_eq!(a.ndim(), dims.len(), "broadcast_in_dim: one result axis per operand axis");
    assert!(dims.windows(2).all(|w| w[0] < w[1]), "broadcast_in_dim: dims must increase");
    // put the operand's axes at `dims` (length 1 elsewhere): the row-major order is unchanged
    let mut placed = vec![1; shape.len()];
    for (&d, &n) in dims.iter().zip(a.shape()) {
        assert!(d < shape.len() && (n == 1 || n == shape[d]), "cannot broadcast {:?} to {shape:?} along {dims:?}", a.shape());
        placed[d] = n;
    }
    a.view().reshape(&placed).expect("same element count").broadcast_to(shape).expect("checked").to_owned()
}

// shape operations for any element type (used by every implementation)

pub(crate) fn broadcast_to_any<T: Copy>(a: NdArray<T>, shape: &[usize]) -> NdArray<T> {
    if a.shape() == shape {
        return a;
    }
    a.view().broadcast_to(shape).unwrap_or_else(|_| panic!("cannot broadcast {:?} to {shape:?}", a.shape())).to_owned()
}

pub(crate) fn reshape_any<T>(a: NdArray<T>, shape: &[usize]) -> NdArray<T> {
    let from = a.shape().to_vec();
    a.reshape(shape).unwrap_or_else(|_| panic!("reshape: {from:?} to {shape:?} changes the element count"))
}

pub(crate) fn transpose_any<T: Copy>(a: NdArray<T>, perm: &[usize]) -> NdArray<T> {
    a.view().permute(perm).unwrap_or_else(|_| panic!("transpose: {perm:?} is not a permutation of {:?}", a.shape())).to_owned()
}

/// The shape of a slice (checking its bounds).
pub(crate) fn slice_shape(shape: &[usize], start: &[usize], limit: &[usize], stride: &[usize]) -> Vec<usize> {
    let n = shape.len();
    assert!(start.len() == n && limit.len() == n && stride.len() == n, "slice: one start, limit and stride per axis of {shape:?}");
    (0..n)
        .map(|i| {
            assert!(start[i] <= limit[i] && limit[i] <= shape[i] && stride[i] >= 1, "slice: {start:?}..{limit:?} by {stride:?} is out of range for {shape:?}");
            (limit[i] - start[i]).div_ceil(stride[i])
        })
        .collect()
}

/// The shape after padding (checking the arguments).
pub(crate) fn pad_shape(shape: &[usize], low: &[usize], high: &[usize], interior: &[usize]) -> Vec<usize> {
    let n = shape.len();
    assert!(low.len() == n && high.len() == n && interior.len() == n, "pad: one low, high and interior count per axis of {shape:?}");
    (0..n).map(|i| low[i] + high[i] + if shape[i] == 0 { 0 } else { shape[i] + (shape[i] - 1) * interior[i] }).collect()
}

/// The shape of a concatenation (checking that the other axes match).
pub(crate) fn concat_shape(shapes: &[Vec<usize>], axis: usize) -> Vec<usize> {
    let first = shapes.first().expect("concatenate: needs at least one array");
    assert!(axis < first.len(), "concatenate: axis {axis} is out of range for {first:?}");
    let mut shape = first.clone();
    shape[axis] = 0;
    for s in shapes {
        assert!(s.len() == first.len() && (0..s.len()).all(|i| i == axis || s[i] == first[i]), "concatenate: {s:?} does not match {first:?} off axis {axis}");
        shape[axis] += s[axis];
    }
    shape
}

pub(crate) fn slice_any<T: Copy>(a: &NdArray<T>, start: &[usize], limit: &[usize], stride: &[usize]) -> NdArray<T> {
    let shape = slice_shape(a.shape(), start, limit, stride);
    let mut v = a.view();
    for i in 0..shape.len() {
        v = v.slice_axis(i, start[i]..limit[i]).expect("checked");
        if stride[i] > 1 && shape[i] > 0 {
            v = v.step_axis(i, stride[i] as isize).expect("checked");
        }
    }
    v.to_owned()
}

pub(crate) fn pad_any<T: Copy + Default>(a: &NdArray<T>, low: &[usize], high: &[usize], interior: &[usize]) -> NdArray<T> {
    let (from, n) = (a.shape(), a.ndim());
    let shape = pad_shape(from, low, high, interior);
    let mut strides = vec![1; n];
    for i in (0..n.saturating_sub(1)).rev() {
        strides[i] = strides[i + 1] * shape[i + 1];
    }
    let mut out = vec![T::default(); shape.iter().product()];
    if a.is_empty() {
        return NdArray::from_vec(out, &shape).expect("valid shape");
    }
    if n == 0 {
        out[0] = a.as_slice()[0];
        return NdArray::from_vec(out, &shape).expect("valid shape");
    }
    // a row (last axis) at a time: one copy when nothing goes between its elements
    let (row, step) = (from[n - 1], interior[n - 1] + 1);
    let mut index = [0usize; super::MAX_DIMS];
    for chunk in a.as_slice().chunks(row) {
        let base = (0..n - 1).map(|i| (low[i] + index[i] * (interior[i] + 1)) * strides[i]).sum::<usize>() + low[n - 1];
        if step == 1 {
            out[base..base + row].copy_from_slice(chunk);
        } else {
            chunk.iter().enumerate().for_each(|(j, &x)| out[base + j * step] = x);
        }
        for i in (0..n - 1).rev() {
            index[i] += 1;
            if index[i] < from[i] {
                break;
            }
            index[i] = 0;
        }
    }
    NdArray::from_vec(out, &shape).expect("valid shape")
}

pub(crate) fn concatenate_any<T: Copy + Default>(parts: &[NdArray<T>], axis: usize) -> NdArray<T> {
    concat_shape(&parts.iter().map(|p| p.shape().to_vec()).collect::<Vec<_>>(), axis);
    super::concatenate(&parts.iter().map(|p| p.view()).collect::<Vec<_>>(), axis).expect("checked")
}

/// Sums over `axes` by plain accumulation, for element types without a pairwise kernel.
pub(crate) fn sum_axes_any<T: Copy + Default + Add<Output = T>>(a: NdArray<T>, axes: &[usize]) -> NdArray<T> {
    sum_axes_with(a, axes, |s, x| s + x)
}

/// `select` for any element type: `if_true` where `mask` holds, else `if_false` (broadcast).
pub(crate) fn select_any<T: Copy>(mask: &NdArray<bool>, if_true: &NdArray<T>, if_false: &NdArray<T>) -> NdArray<T> {
    let (s, n) = broadcast_shapes(if_true.shape(), if_false.shape()).expect("operands broadcast");
    let (shape, n) = broadcast_shapes(&s[..n], mask.shape()).expect("the mask broadcasts");
    let shape = &shape[..n];
    let m = mask.view().broadcast_to(shape).expect("broadcasts");
    let (a, b) = (if_true.view().broadcast_to(shape).expect("broadcasts"), if_false.view().broadcast_to(shape).expect("broadcasts"));
    Zip::from(m).and(a).expect("same shape").and(b).expect("same shape").map_collect(|&m, &x, &y| if m { x } else { y })
}

/// Sums over `axes` with an explicit addition (wrapping for integers).
pub(crate) fn sum_axes_with<T: Copy + Default>(a: NdArray<T>, axes: &[usize], add: impl Fn(T, T) -> T) -> NdArray<T> {
    reduce_axes_with(a, axes, T::default(), add)
}

/// Folds `axes` (removed) with `f`, starting each fold from `init` (what an empty axis gives).
pub(crate) fn reduce_axes_with<T: Copy>(a: NdArray<T>, axes: &[usize], init: T, f: impl Fn(T, T) -> T) -> NdArray<T> {
    let shape = a.shape().to_vec();
    let mut axes = axes.to_vec();
    axes.sort_unstable();
    axes.dedup();
    assert!(axes.iter().all(|&x| x < shape.len()), "reduction: axis out of range for {shape:?}");
    let reduced: Vec<usize> = shape.iter().enumerate().filter(|(i, _)| !axes.contains(i)).map(|(_, &n)| n).collect();
    // move the summed axes last, then add up each run
    let kept: Vec<usize> = (0..shape.len()).filter(|i| !axes.contains(i)).collect();
    let perm: Vec<usize> = kept.iter().copied().chain(axes.iter().copied().filter(|x| *x < shape.len())).collect();
    let moved = a.view().permute(&perm).expect("a permutation").to_vec();
    let run: usize = axes.iter().map(|&x| shape[x]).product();
    let out: Vec<T> = if run == 0 {
        vec![init; reduced.iter().product()]
    } else {
        moved.chunks(run).map(|c| c.iter().fold(init, |s, &x| f(s, x))).collect()
    };
    NdArray::from_vec(out, &reduced).expect("valid shape")
}

/// The larger, NaN if either is (StableHLO's `maximum`).
fn max_nan<T: Float>(a: T, b: T) -> T {
    if a._is_nan() || b._is_nan() {
        T::_NAN
    } else if b > a {
        b
    } else {
        a
    }
}

/// The smaller, NaN if either is.
fn min_nan<T: Float>(a: T, b: T) -> T {
    if a._is_nan() || b._is_nan() {
        T::_NAN
    } else if b < a {
        b
    } else {
        a
    }
}

pub(crate) fn reverse_any<T: Copy>(a: NdArray<T>, axes: &[usize]) -> NdArray<T> {
    let mut v = a.view();
    for &axis in axes {
        assert!(axis < a.ndim(), "reverse: axis {axis} is out of range for {:?}", a.shape());
        v = v.flip(axis).expect("checked");
    }
    v.to_owned()
}

/// An index into `n` rows from a real value: rounded down, then clamped to `0..n` (as StableHLO's
/// gather clamps), NaN to 0.
pub(crate) fn clamp_index(i: f64, n: usize) -> usize {
    if i.is_nan() {
        0
    } else {
        i.floor().clamp(0.0, (n - 1) as f64) as usize
    }
}

/// Rows of `table` (along its first axis) at `indices` (any shape): the result is
/// `[indices..., table's other axes...]`.
pub(crate) fn take_any<T: Copy>(table: &NdArray<T>, indices: impl Iterator<Item = f64>, index_shape: &[usize]) -> NdArray<T> {
    let (&n, rest) = table.shape().split_first().expect("take: the table needs an axis");
    assert!(n > 0, "take: the table is empty");
    let row: usize = rest.iter().product();
    let data = table.as_slice();
    let mut out = Vec::with_capacity(index_shape.iter().product::<usize>() * row);
    for i in indices {
        let k = clamp_index(i, n);
        out.extend_from_slice(&data[k * row..(k + 1) * row]);
    }
    NdArray::from_vec(out, &[index_shape, rest].concat()).expect("valid shape")
}

/// `dot_general` as one matrix product of the permuted operands: `[free, contracted] ·
/// [contracted, free]`. Returns the operands as matrices and the output shape.
#[allow(clippy::type_complexity)]
pub(crate) fn dot_operands<T: Copy>(a: &NdArray<T>, b: &NdArray<T>, ca: &[usize], cb: &[usize]) -> (Vec<T>, Vec<T>, usize, usize, usize, Vec<usize>) {
    let (sa, sb) = (a.shape(), b.shape());
    assert_eq!(ca.len(), cb.len(), "dot_general: pair each contracted axis");
    for (&i, &j) in ca.iter().zip(cb) {
        assert!(i < sa.len() && j < sb.len() && sa[i] == sb[j], "dot_general: cannot contract {sa:?} axis {i} with {sb:?} axis {j}");
    }
    let fa: Vec<usize> = (0..sa.len()).filter(|x| !ca.contains(x)).collect();
    let fb: Vec<usize> = (0..sb.len()).filter(|x| !cb.contains(x)).collect();
    let am = a.view().permute(&[fa.as_slice(), ca].concat()).expect("a permutation").to_vec();
    let bm = b.view().permute(&[cb, fb.as_slice()].concat()).expect("a permutation").to_vec();
    let m: usize = fa.iter().map(|&x| sa[x]).product();
    let k: usize = ca.iter().map(|&x| sa[x]).product();
    let n: usize = fb.iter().map(|&x| sb[x]).product();
    let shape: Vec<usize> = fa.iter().map(|&x| sa[x]).chain(fb.iter().map(|&x| sb[x])).collect();
    (am, bm, m, k, n, shape)
}

/// `dot_general` reading both operands in place (any strides) when each is a matrix once its free
/// axes and its contracted axes are merged (always so for 2-D operands): `None` otherwise, or for
/// element types the matrix product does not cover.
#[cfg(feature = "faer")]
fn dot_in_place<T: Copy + 'static>(a: &NdArray<T>, b: &NdArray<T>, ca: &[usize], cb: &[usize]) -> Option<NdArray<T>> {
    let (sa, sb) = (a.shape(), b.shape());
    if ca.len() != cb.len() || ca.iter().zip(cb).any(|(&i, &j)| i >= sa.len() || j >= sb.len() || sa[i] != sb[j]) {
        return None;
    }
    let fa: Vec<usize> = (0..sa.len()).filter(|x| !ca.contains(x)).collect();
    let fb: Vec<usize> = (0..sb.len()).filter(|x| !cb.contains(x)).collect();
    let av = a.view().permute(&[fa.as_slice(), ca].concat()).ok()?;
    let bv = b.view().permute(&[cb, fb.as_slice()].concat()).ok()?;
    let ma = merged(av.shape(), av.strides(), fa.len())?;
    let mb = merged(bv.shape(), bv.strides(), cb.len())?;
    let out = crate::linalg::gemm_strided(av.as_ptr(), ma, bv.as_ptr(), mb)?;
    let shape: Vec<usize> = fa.iter().map(|&x| sa[x]).chain(fb.iter().map(|&x| sb[x])).collect();
    NdArray::from_vec(out, &shape).ok()
}

/// A view's axes as a matrix: the first `split` merged into rows, the rest into columns, if each
/// group steps through memory with one stride (axes of length 1 aside). `(rows, columns, row
/// stride, column stride)`, strides in elements.
#[cfg(feature = "faer")]
fn merged(shape: &[usize], strides: &[isize], split: usize) -> Option<(usize, usize, isize, isize)> {
    fn group(shape: &[usize], strides: &[isize]) -> Option<(usize, isize)> {
        let mut stride: Option<isize> = None;
        let mut inner = 1usize;
        for (&len, &st) in shape.iter().zip(strides).rev() {
            if len == 1 {
                continue;
            }
            match stride {
                None => stride = Some(st),
                Some(s) if st == s * inner as isize => {}
                Some(_) => return None,
            }
            inner *= len;
        }
        Some((shape.iter().product(), stride.unwrap_or(1)))
    }
    let (rows, rs) = group(&shape[..split], &strides[..split])?;
    let (cols, cs) = group(&shape[split..], &strides[split..])?;
    Some((rows, cols, rs, cs))
}

/// The triple-loop matrix product.
pub(crate) fn matmul_naive<T: Copy + Default + Add<Output = T> + Mul<Output = T>>(a: &[T], b: &[T], m: usize, k: usize, n: usize) -> Vec<T> {
    let mut out = Vec::with_capacity(m * n);
    for i in 0..m {
        for j in 0..n {
            out.push((0..k).fold(T::default(), |s, p| s + a[i * k + p] * b[p * n + j]));
        }
    }
    out
}

impl<T: Float + Default> Elementwise for NdArray<T> {
    fn lit(v: f64) -> Self {
        NdArray::from_vec(vec![T::_lit(v)], &[]).expect("a scalar")
    }
    fn exp(self) -> Self {
        self.map(|&x| x.exp())
    }
    fn ln(self) -> Self {
        self.map(|&x| x.ln())
    }
    fn sin(self) -> Self {
        self.map(|&x| x.sin())
    }
    fn cos(self) -> Self {
        self.map(|&x| x.cos())
    }
    fn tanh(self) -> Self {
        self.map(|&x| x.tanh())
    }
    fn sqrt(self) -> Self {
        self.map(|&x| x.sqrt())
    }
    fn powf(self, e: Self) -> Self {
        zip_with(&self, &e, |x, y| x.powf(y))
    }
}

impl<T: Float + Default> RealValued for NdArray<T> {
    type Mask = NdArray<bool>;
    fn abs(self) -> Self {
        self.map(|&x| x.abs())
    }
    fn minimum(self, other: Self) -> Self {
        zip_with(&self, &other, |x, y| x.minimum(y))
    }
    fn maximum(self, other: Self) -> Self {
        zip_with(&self, &other, |x, y| x.maximum(y))
    }
    fn less(self, other: Self) -> NdArray<bool> {
        zip_with(&self, &other, |x, y| x < y)
    }
    fn greater(self, other: Self) -> NdArray<bool> {
        zip_with(&self, &other, |x, y| x > y)
    }
    fn select(mask: NdArray<bool>, if_true: Self, if_false: Self) -> Self {
        select_any(&mask, &if_true, &if_false)
    }
    fn floor(self) -> Self {
        self.map(|&x| x.floor())
    }
}

impl<T: Float + Default> ArrayMath for NdArray<T> {
    fn array(values: &[f64], shape: &[usize]) -> Self {
        NdArray::from_vec(values.iter().map(|&v| T::_lit(v)).collect(), shape).unwrap_or_else(|_| panic!("array: {} values do not fill {shape:?}", values.len()))
    }
    fn shape(&self) -> Vec<usize> {
        NdArray::shape(self).to_vec()
    }
    fn broadcast_to(self, shape: &[usize]) -> Self {
        broadcast_to_any(self, shape)
    }
    fn broadcast_in_dim(self, shape: &[usize], dims: &[usize]) -> Self {
        broadcast_in_dim(&self, shape, dims)
    }
    fn reshape(self, shape: &[usize]) -> Self {
        reshape_any(self, shape)
    }
    fn transpose(self, perm: &[usize]) -> Self {
        transpose_any(self, perm)
    }

    fn sum_axes(self, axes: &[usize]) -> Self {
        // pairwise summation through sum_into: keep the summed axes at length 1, then drop them
        let shape = NdArray::shape(&self).to_vec();
        assert!(axes.iter().all(|&a| a < shape.len()), "sum_axes: axis out of range for {shape:?}");
        let kept: Vec<usize> = shape.iter().enumerate().map(|(i, &n)| if axes.contains(&i) { 1 } else { n }).collect();
        let mut out = NdArray::<T>::zeros(&kept).expect("valid shape");
        self.view().sum_into(&mut out.view_mut()).expect("broadcasts to the input");
        let reduced: Vec<usize> = shape.iter().enumerate().filter(|(i, _)| !axes.contains(i)).map(|(_, &n)| n).collect();
        NdArray::reshape(out, &reduced).expect("same element count")
    }

    fn dot_general(self, rhs: Self, ca: &[usize], cb: &[usize]) -> Self {
        #[cfg(feature = "faer")]
        if let Some(out) = dot_in_place(&self, &rhs, ca, cb) {
            return out;
        }
        let (a, b, m, k, n, shape) = dot_operands(&self, &rhs, ca, cb);
        #[cfg(feature = "faer")]
        if let Some(out) = crate::linalg::gemm_any(&a, &b, m, k, n) {
            return NdArray::from_vec(out, &shape).expect("valid shape");
        }
        NdArray::from_vec(matmul_naive(&a, &b, m, k, n), &shape).expect("valid shape")
    }
    fn slice(self, start: &[usize], limit: &[usize], stride: &[usize]) -> Self {
        slice_any(&self, start, limit, stride)
    }
    fn pad(self, low: &[usize], high: &[usize], interior: &[usize]) -> Self {
        pad_any(&self, low, high, interior)
    }
    fn concatenate(parts: &[Self], axis: usize) -> Self {
        concatenate_any(parts, axis)
    }
    fn prod_axes(self, axes: &[usize]) -> Self {
        reduce_axes_with(self, axes, T::_ONE, |p, x| p * x)
    }
    fn reverse(self, axes: &[usize]) -> Self {
        reverse_any(self, axes)
    }
}

impl<T: Float + Default> RealArrayMath for NdArray<T> {
    type Complex = NdArray<Complex<T>>;
    fn to_complex(self) -> NdArray<Complex<T>> {
        self.map(|&x| Complex::new(x, T::_ZERO))
    }
    fn complex(re: Self, im: Self) -> NdArray<Complex<T>> {
        join_complex(&re, &im)
    }
    fn real_part(z: NdArray<Complex<T>>) -> Self {
        z.map(|c| c.re)
    }
    fn imag_part(z: NdArray<Complex<T>>) -> Self {
        z.map(|c| c.im)
    }
    fn max_axes(self, axes: &[usize]) -> Self {
        reduce_axes_with(self, axes, T::_NEG_INFINITY, max_nan)
    }
    fn min_axes(self, axes: &[usize]) -> Self {
        reduce_axes_with(self, axes, T::_INFINITY, min_nan)
    }
    fn take(self, indices: Self) -> Self {
        take_any(&self, indices.as_slice().iter().map(|i| i.to_f64().unwrap_or(f64::NAN)), indices.shape())
    }
    fn rfft_complex(self) -> NdArray<Complex<T>> {
        let mut shape = NdArray::shape(&self).to_vec();
        let n = *shape.last().expect("rfft: needs an axis");
        assert!(n >= 1, "rfft: empty axis");
        let m = n / 2 + 1;
        let rows = self.len() / n;
        let mut out = vec![Complex::new(T::_ZERO, T::_ZERO); rows * m];
        let mut fft = RealFft::<T>::new(n);
        let input = if self.view().is_contiguous() { self } else { self.view().to_owned() };
        for (row, bins) in input.as_slice().chunks(n).zip(out.chunks_mut(m)) {
            fft.forward(row, bins);
        }
        *shape.last_mut().unwrap() = m;
        NdArray::from_vec(out, &shape).expect("valid shape")
    }

    fn irfft_complex(spectrum: NdArray<Complex<T>>, n: usize) -> Self {
        let mut shape = NdArray::shape(&spectrum).to_vec();
        let m = n / 2 + 1;
        assert!(n >= 1 && shape.last() == Some(&m), "irfft: {n} samples need {m} bins, got {shape:?}");
        let rows = spectrum.len() / m;
        let mut out = vec![T::_ZERO; rows * n];
        let mut fft = RealFft::<T>::new(n);
        let spectrum = if spectrum.view().is_contiguous() { spectrum } else { spectrum.view().to_owned() };
        for (bins, row) in spectrum.as_slice().chunks(m).zip(out.chunks_mut(n)) {
            fft.inverse(bins, row);
        }
        *shape.last_mut().unwrap() = n;
        NdArray::from_vec(out, &shape).expect("valid shape")
    }
}

impl<T: Float + Default> Elementwise for NdArray<Complex<T>> {
    fn lit(v: f64) -> Self {
        NdArray::from_vec(vec![Complex::new(T::_lit(v), T::_ZERO)], &[]).expect("a scalar")
    }
    fn exp(self) -> Self {
        self.map(|&z| z.exp())
    }
    fn ln(self) -> Self {
        self.map(|&z| z.ln())
    }
    fn sin(self) -> Self {
        self.map(|&z| z.sin())
    }
    fn cos(self) -> Self {
        self.map(|&z| z.cos())
    }
    fn tanh(self) -> Self {
        self.map(|&z| z.tanh())
    }
    fn sqrt(self) -> Self {
        self.map(|&z| z.sqrt())
    }
    fn powf(self, e: Self) -> Self {
        zip_with(&self, &e, Elementwise::powf)
    }
}

impl<T: Float + Default> ArrayMath for NdArray<Complex<T>> {
    fn array(values: &[f64], shape: &[usize]) -> Self {
        NdArray::from_vec(values.iter().map(|&v| Complex::new(T::_lit(v), T::_ZERO)).collect(), shape)
            .unwrap_or_else(|_| panic!("array: {} values do not fill {shape:?}", values.len()))
    }
    fn shape(&self) -> Vec<usize> {
        NdArray::shape(self).to_vec()
    }
    fn broadcast_to(self, shape: &[usize]) -> Self {
        broadcast_to_any(self, shape)
    }
    fn broadcast_in_dim(self, shape: &[usize], dims: &[usize]) -> Self {
        broadcast_in_dim(&self, shape, dims)
    }
    fn reshape(self, shape: &[usize]) -> Self {
        reshape_any(self, shape)
    }
    fn transpose(self, perm: &[usize]) -> Self {
        transpose_any(self, perm)
    }
    fn sum_axes(self, axes: &[usize]) -> Self {
        let last = self.ndim().saturating_sub(1);
        if self.ndim() > 0 && !axes.contains(&last) && self.view().is_contiguous() {
            // the real and imaginary parts as a real array with the last axis doubled: the real sums
            // (pairwise, vectorized) along the same axes
            let mut shape = self.shape().to_vec();
            shape[last] *= 2;
            let mut data = std::mem::ManuallyDrop::new(self.into_vec());
            // SAFETY: Complex<T> is repr(C) { re, im }: a Vec of n of them is a Vec of 2n T's
            let reals = unsafe { Vec::from_raw_parts(data.as_mut_ptr().cast::<T>(), data.len() * 2, data.capacity() * 2) };
            let sums = NdArray::from_vec(reals, &shape).expect("valid shape").sum_axes(axes);
            let mut out_shape = sums.shape().to_vec();
            let n = out_shape.len();
            out_shape[n - 1] /= 2;
            let mut sums = std::mem::ManuallyDrop::new(sums.into_vec());
            // SAFETY: an even number of T's laid out as (re, im) pairs, as above
            let pairs = unsafe { Vec::from_raw_parts(sums.as_mut_ptr().cast::<Complex<T>>(), sums.len() / 2, sums.capacity() / 2) };
            return NdArray::from_vec(pairs, &out_shape).expect("valid shape");
        }
        sum_axes_any(self, axes)
    }
    fn dot_general(self, rhs: Self, ca: &[usize], cb: &[usize]) -> Self {
        let (a, b, m, k, n, shape) = dot_operands(&self, &rhs, ca, cb);
        NdArray::from_vec(matmul_naive(&a, &b, m, k, n), &shape).expect("valid shape")
    }
    fn slice(self, start: &[usize], limit: &[usize], stride: &[usize]) -> Self {
        slice_any(&self, start, limit, stride)
    }
    fn pad(self, low: &[usize], high: &[usize], interior: &[usize]) -> Self {
        pad_any(&self, low, high, interior)
    }
    fn concatenate(parts: &[Self], axis: usize) -> Self {
        concatenate_any(parts, axis)
    }
    fn prod_axes(self, axes: &[usize]) -> Self {
        reduce_axes_with(self, axes, Complex::one(), |p, x| p * x)
    }
    fn reverse(self, axes: &[usize]) -> Self {
        reverse_any(self, axes)
    }
}

impl<T: Float + Default> ComplexArrayMath for NdArray<Complex<T>> {
    fn fft(self) -> Self {
        complex_fft(self, false)
    }
    fn ifft(self) -> Self {
        complex_fft(self, true)
    }
    fn conj(self) -> Self {
        self.map(|z| z.conj())
    }
}

/// The DFT (or its inverse) of every row along the last axis, in f64.
fn join_complex<T: Float + Default>(re: &NdArray<T>, im: &NdArray<T>) -> NdArray<Complex<T>> {
    assert_eq!(re.shape(), im.shape(), "complex parts differ in shape");
    NdArray::from_vec(re.as_slice().iter().zip(im.as_slice()).map(|(&r, &i)| Complex::new(r, i)).collect(), re.shape()).expect("same shape")
}

fn complex_fft<T: Float + Default>(a: NdArray<Complex<T>>, inverse: bool) -> NdArray<Complex<T>> {
    let n = *a.shape().last().expect("fft: needs an axis");
    assert!(n >= 1, "fft: empty axis");
    let mut fft = Fft::<T>::new(n);
    let mut out = if a.view().is_contiguous() { a } else { a.view().to_owned() };
    for row in out.as_mut_slice().chunks_mut(n) {
        if inverse {
            fft.inverse(row);
        } else {
            fft.forward(row);
        }
    }
    out
}
#[cfg(test)]
mod tests {
    use super::*;

    fn arr(data: &[f64], shape: &[usize]) -> NdArray<f64> {
        NdArray::from_vec(data.to_vec(), shape).unwrap()
    }

    #[test]
    fn element_wise_broadcasts() {
        let a = arr(&[1.0, 2.0, 3.0, 4.0, 5.0, 6.0], &[2, 3]);
        let b = arr(&[10.0, 20.0, 30.0], &[3]);
        assert_eq!((a.clone() + b.clone()).as_slice(), &[11.0, 22.0, 33.0, 14.0, 25.0, 36.0]);
        assert_eq!((a.clone() * NdArray::lit(2.0)).as_slice(), &[2.0, 4.0, 6.0, 8.0, 10.0, 12.0]);
        assert_eq!(a.clone().maximum(arr(&[3.0], &[1])).as_slice(), &[3.0, 3.0, 3.0, 4.0, 5.0, 6.0]);
        let picked = NdArray::select(a.clone().greater(arr(&[2.5], &[])), a, -b);
        assert_eq!(picked.as_slice(), &[-10.0, -20.0, 3.0, 4.0, 5.0, 6.0]);
    }

    #[test]
    fn shapes_reductions_and_products() {
        let a = arr(&[1.0, 2.0, 3.0, 4.0, 5.0, 6.0], &[2, 3]);
        assert_eq!(a.clone().transpose(&[1, 0]).as_slice(), &[1.0, 4.0, 2.0, 5.0, 3.0, 6.0]);
        assert_eq!(a.clone().sum_axes(&[0]).as_slice(), &[5.0, 7.0, 9.0]);
        assert_eq!(a.clone().sum_axes(&[1]).as_slice(), &[6.0, 15.0]);
        assert_eq!(a.clone().mean_all().as_slice(), &[3.5]);
        assert_eq!(arr(&[1.0, 2.0], &[2]).broadcast_in_dim(&[2, 3], &[0]).as_slice(), &[1.0, 1.0, 1.0, 2.0, 2.0, 2.0]);
        let x = arr(&[1.0, 0.0, -1.0], &[3]);
        assert_eq!(a.clone().dot(x).as_slice(), &[-2.0, -2.0]);
        let gram = a.clone().dot_general(a, &[0], &[0]);
        assert_eq!(ArrayMath::shape(&gram), vec![3, 3]);
        assert_eq!(gram.as_slice(), &[17.0, 22.0, 27.0, 22.0, 29.0, 36.0, 27.0, 36.0, 45.0]);
    }

    #[test]
    fn fft_round_trip() {
        for n in [8usize, 7] {
            let x = arr(&(0..2 * n).map(|i| (i as f64 * 0.7).sin()).collect::<Vec<_>>(), &[2, n]);
            let (re, im) = x.clone().rfft();
            let back = NdArray::irfft(re, im, n);
            assert!(back.as_slice().iter().zip(x.as_slice()).all(|(a, b)| (a - b).abs() < 1e-12), "n = {n}");
        }
    }

    #[test]
    fn complex_arrays() {
        let z = NdArray::from_vec(vec![Complex::new(1.0f64, 0.0), Complex::new(0.0, 1.0), Complex::new(-1.0, 0.0), Complex::new(0.0, -1.0)], &[1, 4]).unwrap();
        // the DFT of e^(iπn/2) is 4 at bin 1
        let f = z.clone().fft();
        assert!((f.as_slice()[1] - Complex::new(4.0, 0.0)).norm() < 1e-12);
        assert!(f.as_slice().iter().enumerate().all(|(k, v)| k == 1 || v.norm() < 1e-12));
        let back = f.ifft();
        assert!(back.as_slice().iter().zip(z.as_slice()).all(|(a, b)| (*a - *b).norm() < 1e-12));
        let s = z.clone().sum_all();
        assert!(s.as_slice()[0].norm() < 1e-12);
        let w = z.clone() * z.clone().conj(); // |z|² = 1
        assert!(w.as_slice().iter().all(|v| (*v - Complex::new(1.0, 0.0)).norm() < 1e-12));
        let e = NdArray::from_vec(vec![Complex::new(0.0, std::f64::consts::PI)], &[]).unwrap().exp();
        assert!((e.as_slice()[0] - Complex::new(-1.0, 0.0)).norm() < 1e-12);
        let m = NdArray::from_vec(vec![Complex::new(0.0, 1.0); 4], &[2, 2]).unwrap();
        let p = m.clone().dot(m);
        assert!(p.as_slice().iter().all(|v| (*v - Complex::new(-2.0, 0.0)).norm() < 1e-12));
    }
}