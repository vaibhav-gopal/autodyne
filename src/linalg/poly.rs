//! Polynomials as coefficient slices, highest power first (NumPy's `poly1d` order):
//! `[1, -3, 2]` is `x² - 3x + 2`.

use super::{eigvals, LinalgError, LinalgFloat};
use crate::signal::NdArray;
use crate::units::*;

/// The roots of `p` (highest power first), like `numpy.roots`: the eigenvalues of its companion
/// matrix. Leading zeros are dropped; trailing zeros are roots at 0.
pub fn roots<T: LinalgFloat>(p: &[T]) -> Result<Vec<Complex<T>>, LinalgError> {
    let start = p.iter().position(|&c| c != T::_ZERO).unwrap_or(p.len());
    let p = &p[start..];
    let end = p.iter().rposition(|&c| c != T::_ZERO).map_or(0, |i| i + 1);
    let zeros = p.len() - end;
    let p = &p[..end];
    let mut out = Vec::with_capacity(p.len().saturating_sub(1) + zeros);
    let n = p.len().saturating_sub(1);
    if n > 0 {
        // first row -p[1..] / p[0], ones on the subdiagonal
        let mut c = NdArray::<T>::zeros(&[n, n]).expect("n x n");
        let data = c.as_mut_slice();
        for j in 0..n {
            data[j] = -p[j + 1] / p[0];
        }
        for i in 1..n {
            data[i * n + i - 1] = T::_ONE;
        }
        out.extend(eigvals(c.view())?);
    }
    out.extend(std::iter::repeat_n(Complex::zero(), zeros));
    Ok(out)
}

/// The monic polynomial with these roots (highest power first), complex coefficients.
pub fn poly_complex<T: Float>(roots: &[Complex<T>]) -> Vec<Complex<T>> {
    let mut p = vec![Complex::new(T::_ONE, T::_ZERO)];
    for &r in roots {
        // multiply by (x - r)
        p.push(Complex::zero());
        for i in (1..p.len()).rev() {
            p[i] = p[i] - r * p[i - 1];
        }
    }
    p
}

/// The monic polynomial with these roots, real coefficients: the roots must be real or come in
/// conjugate pairs (the imaginary parts of the expansion, rounding error, are dropped).
pub fn poly<T: Float>(roots: &[Complex<T>]) -> Vec<T> {
    poly_complex(roots).into_iter().map(|c| c.re).collect()
}

/// `p(x)` by Horner's rule.
pub fn polyval<T: Float>(p: &[T], x: T) -> T {
    p.iter().fold(T::_ZERO, |acc, &c| acc * x + c)
}

/// `p(z)` at a complex point.
pub fn polyval_complex<T: Float>(p: &[T], z: Complex<T>) -> Complex<T> {
    p.iter().fold(Complex::zero(), |acc, &c| acc * z + Complex::new(c, T::_ZERO))
}

/// The product of two polynomials (the convolution of their coefficients).
pub fn polymul<T: Float>(a: &[T], b: &[T]) -> Vec<T> {
    if a.is_empty() || b.is_empty() {
        return Vec::new();
    }
    let mut out = vec![T::_ZERO; a.len() + b.len() - 1];
    for (i, &x) in a.iter().enumerate() {
        for (j, &y) in b.iter().enumerate() {
            out[i + j] = out[i + j] + x * y;
        }
    }
    out
}

/// The sum of two polynomials (aligned at their constant terms).
pub fn polyadd<T: Float>(a: &[T], b: &[T]) -> Vec<T> {
    let n = a.len().max(b.len());
    let at = |p: &[T], i: usize| if i + p.len() >= n { p[i + p.len() - n] } else { T::_ZERO };
    (0..n).map(|i| at(a, i) + at(b, i)).collect()
}
