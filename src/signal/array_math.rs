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

/// Real arrays: [`ArrayMath`] with ordering ([`RealValued`]) and real FFTs.
pub trait RealArrayMath: ArrayMath + RealValued {
    /// The real FFT along the last axis: `(real parts, imaginary parts)`, `n / 2 + 1` bins each.
    fn rfft(self) -> (Self, Self);
    /// The inverse of [`rfft`](Self::rfft): `n` samples along the last axis from `n / 2 + 1` bins,
    /// scaled by `1 / n`. The imaginary parts of bins 0 and `n / 2` are ignored.
    fn irfft(re: Self, im: Self, n: usize) -> Self;
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
    let shape = a.shape().to_vec();
    assert!(axes.iter().all(|&x| x < shape.len()), "sum_axes: axis out of range for {shape:?}");
    let reduced: Vec<usize> = shape.iter().enumerate().filter(|(i, _)| !axes.contains(i)).map(|(_, &n)| n).collect();
    // move the summed axes last, then add up each run
    let kept: Vec<usize> = (0..shape.len()).filter(|i| !axes.contains(i)).collect();
    let perm: Vec<usize> = kept.iter().copied().chain(axes.iter().copied().filter(|x| *x < shape.len())).collect();
    let moved = a.view().permute(&perm).expect("a permutation").to_vec();
    let run: usize = axes.iter().map(|&x| shape[x]).product();
    let out: Vec<T> = if run == 0 {
        vec![T::default(); reduced.iter().product()]
    } else {
        moved.chunks(run).map(|c| c.iter().fold(T::default(), |s, &x| add(s, x))).collect()
    };
    NdArray::from_vec(out, &reduced).expect("valid shape")
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
        let (a, b, m, k, n, shape) = dot_operands(&self, &rhs, ca, cb);
        #[cfg(feature = "faer")]
        if let Some(out) = crate::linalg::gemm_any(&a, &b, m, k, n) {
            return NdArray::from_vec(out, &shape).expect("valid shape");
        }
        NdArray::from_vec(matmul_naive(&a, &b, m, k, n), &shape).expect("valid shape")
    }
}

impl<T: Float + Default> RealArrayMath for NdArray<T> {
    fn rfft(self) -> (Self, Self) {
        let mut shape = NdArray::shape(&self).to_vec();
        let n = *shape.last().expect("rfft: needs an axis");
        assert!(n >= 1, "rfft: empty axis");
        let m = n / 2 + 1;
        let rows = self.len() / n;
        let (mut re, mut im) = (Vec::with_capacity(rows * m), Vec::with_capacity(rows * m));
        let mut fft = RealFft::<f64>::new(n);
        for row in self.as_slice().chunks(n) {
            for z in rfft_row(&mut fft, row) {
                re.push(T::_lit(z.re));
                im.push(T::_lit(z.im));
            }
        }
        *shape.last_mut().unwrap() = m;
        (NdArray::from_vec(re, &shape).expect("valid shape"), NdArray::from_vec(im, &shape).expect("valid shape"))
    }

    fn irfft(re: Self, im: Self, n: usize) -> Self {
        let mut shape = NdArray::shape(&re).to_vec();
        assert_eq!(shape.as_slice(), NdArray::shape(&im), "irfft: real and imaginary parts differ in shape");
        let m = n / 2 + 1;
        assert!(n >= 1 && shape.last() == Some(&m), "irfft: {n} samples need {m} bins, got {shape:?}");
        let mut out = Vec::with_capacity(re.len() / m * n);
        let mut fft = RealFft::<f64>::new(n);
        for (r, i) in re.as_slice().chunks(m).zip(im.as_slice().chunks(m)) {
            out.extend(irfft_row(&mut fft, r, i).into_iter().map(T::_lit));
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
        sum_axes_any(self, axes)
    }
    fn dot_general(self, rhs: Self, ca: &[usize], cb: &[usize]) -> Self {
        let (a, b, m, k, n, shape) = dot_operands(&self, &rhs, ca, cb);
        NdArray::from_vec(matmul_naive(&a, &b, m, k, n), &shape).expect("valid shape")
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
fn complex_fft<T: Float + Default>(a: NdArray<Complex<T>>, inverse: bool) -> NdArray<Complex<T>> {
    let n = *a.shape().last().expect("fft: needs an axis");
    assert!(n >= 1, "fft: empty axis");
    let mut fft = Fft::<f64>::new(n);
    let mut row = vec![Complex::zero(); n];
    let mut out = Vec::with_capacity(a.len());
    for chunk in a.as_slice().chunks(n) {
        for (r, z) in row.iter_mut().zip(chunk) {
            *r = Complex::new(z.re.to_f64().unwrap_or(f64::NAN), z.im.to_f64().unwrap_or(f64::NAN));
        }
        if inverse {
            fft.inverse(&mut row);
        } else {
            fft.forward(&mut row);
        }
        out.extend(row.iter().map(|z| Complex::new(T::_lit(z.re), T::_lit(z.im))));
    }
    NdArray::from_vec(out, a.shape()).expect("same shape")
}

/// Bins 0..=n/2 of a real signal's DFT, computed in f64.
fn rfft_row<T: Float>(fft: &mut RealFft<f64>, x: &[T]) -> Vec<Complex<f64>> {
    let x: Vec<f64> = x.iter().map(|v| v.to_f64().unwrap()).collect();
    let mut out = vec![Complex::new(0.0, 0.0); fft.spectrum_len()];
    fft.forward(&x, &mut out);
    out
}

/// `n` samples from bins 0..=n/2, scaled by 1/n, in f64; imaginary parts of bins 0 and n/2 ignored.
fn irfft_row<T: Float>(fft: &mut RealFft<f64>, re: &[T], im: &[T]) -> Vec<f64> {
    let spectrum: Vec<Complex<f64>> = re.iter().zip(im).map(|(r, i)| Complex::new(r.to_f64().unwrap(), i.to_f64().unwrap())).collect();
    let mut out = vec![0.0f64; fft.len()];
    fft.inverse(&spectrum, &mut out);
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