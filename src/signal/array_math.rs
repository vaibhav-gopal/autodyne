//! [`ArrayMath`]: array operations written once and run either eagerly on [`NdArray`]s or traced
//! (`flux::Tracer`), like NumPy and JAX's `jax.numpy` sharing one API.

use super::nd_ops::{broadcast_shapes, Zip};
use super::ndarray::NdArray;
use crate::spectral::RealFft;
use crate::units::*;

/// Array maths on top of [`Elementwise`]: shapes, broadcasting, reductions, tensor products and real
/// FFTs.
///
/// Implemented eagerly by `NdArray<f32>` / `NdArray<f64>` (every call computes a new array) and by
/// `flux::Tracer` (every call records a node), so one generic function serves both: run it on
/// arrays, or trace it to differentiate it and compile it with XLA or IREE.
///
/// Arithmetic operators and the [`Elementwise`] functions broadcast NumPy-style. Values are taken
/// by value; `clone()` one to use it twice (free for tracers).
///
/// ```
/// use autodyne::signal::{ArrayMath, NdArray};
///
/// /// Filters each row of `x` by a per-bin gain, in the frequency domain.
/// fn spectral_gain<A: ArrayMath>(x: A, gain: A) -> A {
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
    /// The real FFT along the last axis: `(real parts, imaginary parts)`, `n / 2 + 1` bins each.
    fn rfft(self) -> (Self, Self);
    /// The inverse of [`rfft`](Self::rfft): `n` samples along the last axis from `n / 2 + 1` bins,
    /// scaled by `1 / n`. The imaginary parts of bins 0 and `n / 2` are ignored.
    fn irfft(re: Self, im: Self, n: usize) -> Self;

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

impl<T: Float + Default> Elementwise for NdArray<T> {
    type Mask = NdArray<bool>;
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
    fn abs(self) -> Self {
        self.map(|&x| x.abs())
    }
    fn powf(self, e: Self) -> Self {
        zip_with(&self, &e, |x, y| x.powf(y))
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
        let (s, n) = broadcast_shapes(if_true.shape(), if_false.shape()).expect("operands broadcast");
        let (shape, n) = broadcast_shapes(&s[..n], mask.shape()).expect("the mask broadcasts");
        let shape = &shape[..n];
        let m = mask.view().broadcast_to(shape).expect("broadcasts");
        let (a, b) = (if_true.view().broadcast_to(shape).expect("broadcasts"), if_false.view().broadcast_to(shape).expect("broadcasts"));
        Zip::from(m).and(a).expect("same shape").and(b).expect("same shape").map_collect(|&m, &x, &y| if m { x } else { y })
    }
}

impl<T: Float + Default> ArrayMath for NdArray<T> {
    fn shape(&self) -> Vec<usize> {
        NdArray::shape(self).to_vec()
    }

    fn broadcast_to(self, shape: &[usize]) -> Self {
        if NdArray::shape(&self) == shape {
            return self;
        }
        self.view().broadcast_to(shape).unwrap_or_else(|_| panic!("cannot broadcast {:?} to {shape:?}", NdArray::shape(&self))).to_owned()
    }

    fn broadcast_in_dim(self, shape: &[usize], dims: &[usize]) -> Self {
        broadcast_in_dim(&self, shape, dims)
    }

    fn reshape(self, shape: &[usize]) -> Self {
        let from = NdArray::shape(&self).to_vec();
        NdArray::reshape(self, shape).unwrap_or_else(|_| panic!("reshape: {from:?} to {shape:?} changes the element count"))
    }

    fn transpose(self, perm: &[usize]) -> Self {
        self.view().permute(perm).unwrap_or_else(|_| panic!("transpose: {perm:?} is not a permutation of {:?}", NdArray::shape(&self))).to_owned()
    }

    fn sum_axes(self, axes: &[usize]) -> Self {
        let shape = NdArray::shape(&self).to_vec();
        assert!(axes.iter().all(|&a| a < shape.len()), "sum_axes: axis out of range for {shape:?}");
        // keep the summed axes at length 1, then drop them
        let kept: Vec<usize> = shape.iter().enumerate().map(|(i, &n)| if axes.contains(&i) { 1 } else { n }).collect();
        let mut out = NdArray::<T>::zeros(&kept).expect("valid shape");
        self.view().sum_into(&mut out.view_mut()).expect("broadcasts to the input");
        let reduced: Vec<usize> = shape.iter().enumerate().filter(|(i, _)| !axes.contains(i)).map(|(_, &n)| n).collect();
        NdArray::reshape(out, &reduced).expect("same element count")
    }

    fn dot_general(self, rhs: Self, ca: &[usize], cb: &[usize]) -> Self {
        let (sa, sb) = (NdArray::shape(&self).to_vec(), NdArray::shape(&rhs).to_vec());
        assert_eq!(ca.len(), cb.len(), "dot_general: pair each contracted axis");
        for (&i, &j) in ca.iter().zip(cb) {
            assert!(i < sa.len() && j < sb.len() && sa[i] == sb[j], "dot_general: cannot contract {sa:?} axis {i} with {sb:?} axis {j}");
        }
        // [free, contracted] · [contracted, free] as one matrix product, accumulated in f64
        let fa: Vec<usize> = (0..sa.len()).filter(|x| !ca.contains(x)).collect();
        let fb: Vec<usize> = (0..sb.len()).filter(|x| !cb.contains(x)).collect();
        let a = self.view().permute(&[fa.as_slice(), ca].concat()).expect("a permutation").to_vec();
        let b = rhs.view().permute(&[cb, fb.as_slice()].concat()).expect("a permutation").to_vec();
        let m: usize = fa.iter().map(|&x| sa[x]).product();
        let k: usize = ca.iter().map(|&x| sa[x]).product();
        let n: usize = fb.iter().map(|&x| sb[x]).product();
        let mut out = Vec::with_capacity(m * n);
        for i in 0..m {
            for j in 0..n {
                let sum: f64 = (0..k).map(|p| a[i * k + p].to_f64().unwrap() * b[p * n + j].to_f64().unwrap()).sum();
                out.push(T::_lit(sum));
            }
        }
        let shape: Vec<usize> = fa.iter().map(|&x| sa[x]).chain(fb.iter().map(|&x| sb[x])).collect();
        NdArray::from_vec(out, &shape).expect("valid shape")
    }

    fn rfft(self) -> (Self, Self) {
        let mut shape = NdArray::shape(&self).to_vec();
        let n = *shape.last().expect("rfft: needs an axis");
        assert!(n >= 1, "rfft: empty axis");
        let m = n / 2 + 1;
        let rows = self.len() / n;
        let (mut re, mut im) = (Vec::with_capacity(rows * m), Vec::with_capacity(rows * m));
        for row in self.as_slice().chunks(n) {
            for z in rfft_row(row) {
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
        for (r, i) in re.as_slice().chunks(m).zip(im.as_slice().chunks(m)) {
            out.extend(irfft_row(r, i, n).into_iter().map(T::_lit));
        }
        *shape.last_mut().unwrap() = n;
        NdArray::from_vec(out, &shape).expect("valid shape")
    }
}

/// Bins 0..=n/2 of a real signal's DFT, in f64 (the FFT for power-of-two lengths, the DFT by
/// definition otherwise).
fn rfft_row<T: Float>(x: &[T]) -> Vec<Complex<f64>> {
    let n = x.len();
    let x: Vec<f64> = x.iter().map(|v| v.to_f64().unwrap()).collect();
    if n >= 2 && n.is_power_of_two() {
        let mut out = vec![Complex::new(0.0, 0.0); n / 2 + 1];
        RealFft::<f64>::new(n).forward(&x, &mut out);
        return out;
    }
    (0..n / 2 + 1)
        .map(|k| {
            x.iter().enumerate().fold(Complex::new(0.0, 0.0), |acc, (t, &v)| {
                let phase = -std::f64::consts::TAU * ((k * t) % n) as f64 / n as f64;
                acc + Complex::new(v * phase.cos(), v * phase.sin())
            })
        })
        .collect()
}

/// `n` samples from bins 0..=n/2, scaled by 1/n, in f64; imaginary parts of bins 0 and n/2 ignored.
fn irfft_row<T: Float>(re: &[T], im: &[T], n: usize) -> Vec<f64> {
    let spectrum: Vec<Complex<f64>> = re.iter().zip(im).map(|(r, i)| Complex::new(r.to_f64().unwrap(), i.to_f64().unwrap())).collect();
    if n >= 2 && n.is_power_of_two() {
        let mut out = vec![0.0f64; n];
        RealFft::<f64>::new(n).inverse(&spectrum, &mut out);
        return out;
    }
    (0..n)
        .map(|t| {
            let mut sum = spectrum[0].re;
            for (k, z) in spectrum.iter().enumerate().skip(1) {
                let phase = std::f64::consts::TAU * ((k * t) % n) as f64 / n as f64;
                let term = z.re * phase.cos() - z.im * phase.sin();
                sum += if n.is_multiple_of(2) && k == n / 2 { term } else { 2.0 * term };
            }
            sum / n as f64
        })
        .collect()
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
}
