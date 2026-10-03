//! Linear algebra on autodyne's arrays: matrix products, solves, least squares, decompositions (LU,
//! QR, Cholesky, eigen, SVD), inverses, determinants, and polynomials (roots, expansion).
//!
//! Backed by [faer](https://docs.rs/faer) (feature `faer`, on by default): pure Rust,
//! LAPACK-class algorithms, multithreaded matrix products. Inputs are [`NdView`]s with any strides
//! (transposed, reversed, every other row): faer reads them in place, nothing is copied on the way
//! in. Results are new row-major [`NdArray`]s.
//!
//! ```
//! use autodyne::linalg;
//! use autodyne::signal::NdArray;
//!
//! let a = NdArray::from_vec(vec![3.0f64, 1.0, 1.0, 2.0], &[2, 2]).unwrap();
//! let b = NdArray::from_vec(vec![9.0, 8.0], &[2]).unwrap();
//! let x = linalg::solve(a.view(), b.view()).unwrap();
//! assert!((x.as_slice()[0] - 2.0).abs() < 1e-12 && (x.as_slice()[1] - 3.0).abs() < 1e-12);
//! ```

use faer::linalg::solvers::{DenseSolveCore, Solve};
use faer::{Accum, MatMut, MatRef, Side};
use thiserror::Error;

use crate::signal::{NdArray, NdView};
use crate::units::*;

mod expm;
mod poly;
mod values;
pub use expm::expm;
pub use poly::*;

/// Element types linear algebra works on: `f32` and `f64`.
pub trait LinalgFloat: Float + Default + faer::traits::RealField {}
impl LinalgFloat for f32 {}
impl LinalgFloat for f64 {}

#[derive(Debug, Clone, PartialEq, Eq, Error)]
pub enum LinalgError {
    #[error("expected a {expected} array, got shape {got:?}")]
    Dims { expected: &'static str, got: Vec<usize> },
    #[error("expected a square matrix, got shape {0:?}")]
    NotSquare(Vec<usize>),
    #[error("shapes {0:?} and {1:?} are not aligned")]
    Mismatch(Vec<usize>, Vec<usize>),
    #[error("the matrix is singular")]
    Singular,
    #[error("the matrix is not positive definite")]
    NotPositiveDefinite,
    #[error("the iteration did not converge")]
    NoConvergence,
}

/// Uses `threads` threads for large matrix operations (0: all cores, 1: single-threaded).
pub fn set_threads(threads: usize) {
    faer::set_global_parallelism(if threads == 1 { faer::Par::Seq } else { faer::Par::rayon(threads) });
}

/// A 2-D view as a faer matrix, in place.
fn matrix<'a, T>(v: NdView<'a, T>) -> Result<MatRef<'a, T>, LinalgError> {
    match (v.shape(), v.strides()) {
        // SAFETY: the view's layout is validated: every (i, j) inside the shape is inside its memory,
        // which stays borrowed for 'a
        (&[rows, cols], &[rs, cs]) => Ok(unsafe { MatRef::from_raw_parts(v.as_ptr(), rows, cols, rs, cs) }),
        shape => Err(LinalgError::Dims { expected: "2-D", got: shape.0.to_vec() }),
    }
}

/// A 1-D or 2-D view as a matrix (a 1-D view becomes one column).
fn columns<'a, T>(v: NdView<'a, T>) -> Result<MatRef<'a, T>, LinalgError> {
    match (v.shape(), v.strides()) {
        // SAFETY: as in `matrix`
        (&[n], &[s]) => Ok(unsafe { MatRef::from_raw_parts(v.as_ptr(), n, 1, s, 0) }),
        _ => matrix(v),
    }
}

fn square<'a, T>(v: NdView<'a, T>) -> Result<MatRef<'a, T>, LinalgError> {
    let m = matrix(v)?;
    if m.nrows() != m.ncols() {
        return Err(LinalgError::NotSquare(v.shape().to_vec()));
    }
    Ok(m)
}

/// A new row-major array of `shape` holding `m` (whose element count matches).
fn to_array<T: Copy>(m: MatRef<'_, T>, shape: &[usize]) -> NdArray<T> {
    let mut data = Vec::with_capacity(m.nrows() * m.ncols());
    for i in 0..m.nrows() {
        for j in 0..m.ncols() {
            data.push(m[(i, j)]);
        }
    }
    NdArray::from_vec(data, shape).expect("element count matches")
}

/// Matrix product, NumPy `matmul` style for 1-D and 2-D operands: matrix · matrix, matrix · vector,
/// vector · matrix, vector · vector (a scalar, shape `[]`).
pub fn matmul<T: LinalgFloat>(a: NdView<'_, T>, b: NdView<'_, T>) -> Result<NdArray<T>, LinalgError> {
    let (sa, sb) = (a.shape().to_vec(), b.shape().to_vec());
    let (lhs, rhs) = match (sa.len(), sb.len()) {
        // a 1-D left operand is one row
        // SAFETY: as in `matrix`
        (1, 1 | 2) => (unsafe { MatRef::from_raw_parts(a.as_ptr(), 1, sa[0], 0, a.strides()[0]) }, columns(b)?),
        (2, 1 | 2) => (matrix(a)?, columns(b)?),
        _ => return Err(LinalgError::Dims { expected: "1-D or 2-D", got: if sa.len() > 2 || sa.is_empty() { sa } else { sb } }),
    };
    if lhs.ncols() != rhs.nrows() {
        return Err(LinalgError::Mismatch(sa, sb));
    }
    let shape: Vec<usize> = match (sa.len(), sb.len()) {
        (1, 1) => vec![],
        (1, 2) => vec![sb[1]],
        (2, 1) => vec![sa[0]],
        _ => vec![sa[0], sb[1]],
    };
    let (m, n) = (lhs.nrows(), rhs.ncols());
    let mut out = NdArray::<T>::zeros(&shape).expect("valid shape");
    // SAFETY: `out` is a fresh contiguous m x n (row-major) buffer, borrowed mutably for the call
    let dst = unsafe { MatMut::from_raw_parts_mut(out.as_mut_slice().as_mut_ptr(), m, n, n as isize, 1) };
    faer::linalg::matmul::matmul(dst, Accum::Replace, lhs, rhs, T::_ONE, faer::get_global_parallelism());
    Ok(out)
}

/// Row-major matrix product of contiguous buffers, for any element type: `Some(a[m x k] · b[k x n])`
/// for `f32` / `f64`, `None` for the others (used by `ArrayMath::dot_general`).
#[allow(clippy::ptr_arg)] // Vecs, not slices: they are downcast through Any, which needs sized types
pub(crate) fn gemm_any<T: Copy + 'static>(a: &Vec<T>, b: &Vec<T>, m: usize, k: usize, n: usize) -> Option<Vec<T>> {
    use std::any::Any;
    fn go<F: LinalgFloat>(a: &[F], b: &[F], m: usize, k: usize, n: usize) -> Vec<F> {
        let mut out = vec![F::_ZERO; m * n];
        gemm(a, b, &mut out, m, k, n);
        out
    }
    let (a, b): (&dyn Any, &dyn Any) = (a, b);
    let out: Box<dyn Any> = if let (Some(a), Some(b)) = (a.downcast_ref::<Vec<f64>>(), b.downcast_ref::<Vec<f64>>()) {
        Box::new(go(a, b, m, k, n))
    } else if let (Some(a), Some(b)) = (a.downcast_ref::<Vec<f32>>(), b.downcast_ref::<Vec<f32>>()) {
        Box::new(go(a, b, m, k, n))
    } else {
        return None;
    };
    out.downcast::<Vec<T>>().ok().map(|b| *b)
}

/// Row-major matrix product of contiguous buffers: `out[m x n] = a[m x k] · b[k x n]`.
fn gemm<T: LinalgFloat>(a: &[T], b: &[T], out: &mut [T], m: usize, k: usize, n: usize) {
    // SAFETY: the buffers hold m*k, k*n and m*n elements (the caller sizes them)
    let (lhs, rhs, dst) = unsafe {
        (
            MatRef::from_raw_parts(a.as_ptr(), m, k, k as isize, 1),
            MatRef::from_raw_parts(b.as_ptr(), k, n, n as isize, 1),
            MatMut::from_raw_parts_mut(out.as_mut_ptr(), m, n, n as isize, 1),
        )
    };
    faer::linalg::matmul::matmul(dst, Accum::Replace, lhs, rhs, T::_ONE, faer::get_global_parallelism());
}

/// The solution `x` of `a · x = b` (`a` square; `b` a vector or a matrix of right-hand sides).
pub fn solve<T: LinalgFloat>(a: NdView<'_, T>, b: NdView<'_, T>) -> Result<NdArray<T>, LinalgError> {
    let m = square(a)?;
    let rhs = columns(b)?;
    if rhs.nrows() != m.nrows() {
        return Err(LinalgError::Mismatch(a.shape().to_vec(), b.shape().to_vec()));
    }
    let lu = m.partial_piv_lu();
    check_nonsingular(lu.U())?;
    Ok(to_array(lu.solve(rhs).as_ref(), b.shape()))
}

/// Errors if the triangular factor `u` has a zero (or non-finite) pivot.
fn check_nonsingular<T: LinalgFloat>(u: MatRef<'_, T>) -> Result<(), LinalgError> {
    let n = u.nrows().min(u.ncols());
    if (0..n).any(|i| u[(i, i)] == T::_ZERO || !u[(i, i)]._is_finite()) {
        return Err(LinalgError::Singular);
    }
    Ok(())
}

/// The inverse of a square matrix.
pub fn inv<T: LinalgFloat>(a: NdView<'_, T>) -> Result<NdArray<T>, LinalgError> {
    let m = square(a)?;
    let lu = m.partial_piv_lu();
    check_nonsingular(lu.U())?;
    Ok(to_array(lu.inverse().as_ref(), a.shape()))
}

/// The determinant of a square matrix.
pub fn det<T: LinalgFloat>(a: NdView<'_, T>) -> Result<T, LinalgError> {
    Ok(square(a)?.determinant())
}

/// A least-squares solution and what it found out about `a`.
#[derive(Debug, Clone, PartialEq)]
pub struct Lstsq<T> {
    /// The `x` minimizing `|a · x - b|` (of minimum norm when `a` is rank deficient).
    pub solution: NdArray<T>,
    /// The effective rank of `a`.
    pub rank: usize,
    /// Singular values of `a`, largest first.
    pub singular_values: Vec<T>,
}

/// Least squares, like `numpy.linalg.lstsq`: through the SVD, so rank-deficient and
/// underdetermined systems get the minimum-norm solution. Singular values below
/// `eps · max(m, n) · s_max` count as zero.
pub fn lstsq<T: LinalgFloat>(a: NdView<'_, T>, b: NdView<'_, T>) -> Result<Lstsq<T>, LinalgError> {
    let m = matrix(a)?;
    let rhs = columns(b)?;
    if rhs.nrows() != m.nrows() {
        return Err(LinalgError::Mismatch(a.shape().to_vec(), b.shape().to_vec()));
    }
    let svd = m.thin_svd().map_err(|_| LinalgError::NoConvergence)?;
    let s: Vec<T> = svd.S().column_vector().iter().copied().collect();
    let largest = s.first().copied().unwrap_or(T::_ZERO);
    let tol = T::_EPSILON * T::_lit(m.nrows().max(m.ncols()) as f64) * largest;
    let rank = s.iter().filter(|&&v| v > tol).count();
    // x = V · diag(1/s) · Uᵀ · b over the significant singular values
    let ut_b = svd.U().transpose() * rhs;
    let mut scaled = ut_b.clone();
    for i in 0..scaled.nrows() {
        let inv = if i < rank { T::_ONE / s[i] } else { T::_ZERO };
        for j in 0..scaled.ncols() {
            scaled[(i, j)] = ut_b[(i, j)] * inv;
        }
    }
    let x = svd.V() * &scaled;
    let shape: Vec<usize> = if b.ndim() == 1 { vec![m.ncols()] } else { vec![m.ncols(), rhs.ncols()] };
    Ok(Lstsq { solution: to_array(x.as_ref(), &shape), rank, singular_values: s })
}

/// The reduced QR decomposition `a = q · r`: `q` is m x k with orthonormal columns, `r` is k x n
/// upper triangular, k = min(m, n).
pub fn qr<T: LinalgFloat>(a: NdView<'_, T>) -> Result<(NdArray<T>, NdArray<T>), LinalgError> {
    let m = matrix(a)?;
    let k = m.nrows().min(m.ncols());
    let qr = m.qr();
    let q = qr.compute_thin_Q();
    let r = qr.R();
    let r = r.subrows(0, k);
    Ok((to_array(q.as_ref(), &[m.nrows(), k]), to_array(r, &[k, m.ncols()])))
}

/// The Cholesky factor `l` (lower triangular, `a = l · lᵀ`) of a symmetric positive definite matrix.
pub fn cholesky<T: LinalgFloat>(a: NdView<'_, T>) -> Result<NdArray<T>, LinalgError> {
    let m = square(a)?;
    let llt = m.llt(Side::Lower).map_err(|_| LinalgError::NotPositiveDefinite)?;
    Ok(to_array(llt.L(), a.shape()))
}

/// Eigenvalues and eigenvectors of a general square matrix.
#[derive(Debug, Clone, PartialEq)]
pub struct Eig<T: Float> {
    pub values: Vec<Complex<T>>,
    /// Column `i` is the (unit-norm) eigenvector of `values[i]`.
    pub vectors: NdArray<Complex<T>>,
}

/// The eigendecomposition of a general (non-symmetric) square matrix: complex eigenvalues come in
/// conjugate pairs.
pub fn eig<T: LinalgFloat>(a: NdView<'_, T>) -> Result<Eig<T>, LinalgError> {
    let m = square(a)?;
    let e = m.eigen().map_err(|_| LinalgError::NoConvergence)?;
    let values = e.S().column_vector().iter().map(|z| Complex::new(z.re, z.im)).collect();
    let u = e.U();
    let n = m.nrows();
    let mut data = Vec::with_capacity(n * n);
    for i in 0..n {
        for j in 0..n {
            let z = u[(i, j)];
            data.push(Complex::new(z.re, z.im));
        }
    }
    Ok(Eig { values, vectors: NdArray::from_vec(data, &[n, n]).expect("n x n") })
}

/// The eigenvalues of a general square matrix.
pub fn eigvals<T: LinalgFloat>(a: NdView<'_, T>) -> Result<Vec<Complex<T>>, LinalgError> {
    let m = square(a)?;
    Ok(m.eigenvalues().map_err(|_| LinalgError::NoConvergence)?.iter().map(|z| Complex::new(z.re, z.im)).collect())
}

/// Eigenvalues (ascending) and orthonormal eigenvectors (columns) of a symmetric matrix; only its
/// lower triangle is read.
pub fn eigh<T: LinalgFloat>(a: NdView<'_, T>) -> Result<(Vec<T>, NdArray<T>), LinalgError> {
    let m = square(a)?;
    let e = m.self_adjoint_eigen(Side::Lower).map_err(|_| LinalgError::NoConvergence)?;
    Ok((e.S().column_vector().iter().copied().collect(), to_array(e.U(), a.shape())))
}

/// The eigenvalues (ascending) of a symmetric matrix, without eigenvectors; only its lower
/// triangle is read.
///
/// faer reduces the matrix to tridiagonal form; the eigenvalues of that come from the
/// Pal-Walker-Kahan QL/QR iteration (LAPACK's `dsterf`), much faster than the divide and conquer
/// faer's own values-only path uses.
pub fn eigvalsh<T: LinalgFloat>(a: NdView<'_, T>) -> Result<Vec<T>, LinalgError> {
    use faer::linalg::evd::tridiag::{tridiag_in_place, tridiag_in_place_scratch};
    let m = square(a)?;
    let n = m.nrows();
    if n == 0 {
        return Ok(Vec::new());
    }
    let par = faer::get_global_parallelism();
    // scaled into range, as LAPACK's dsyev does: rmin = sqrt(safmin / eps), rmax = 1 / rmin
    let rmin = (min_positive::<T>() / f64_of(T::_EPSILON)).sqrt();
    let scale = scaling(m, true, rmin, 1.0 / rmin);
    let k = of_f64::<T>(scale);
    let mut trid = faer::Mat::<T>::from_fn(n, n, |i, j| if i >= j { m[(i, j)] * k } else { T::_ZERO });
    if n > 1 {
        let bs = faer::linalg::qr::no_pivoting::factor::recommended_block_size::<T>(n, n);
        let mut householder = faer::Mat::<T>::zeros(bs, n - 1);
        let mut buf = faer::dyn_stack::MemBuffer::new(tridiag_in_place_scratch::<T>(n, par, Default::default()));
        tridiag_in_place(trid.as_mut(), householder.as_mut(), par, faer::dyn_stack::MemStack::new(&mut buf), Default::default());
    }
    let mut diag: Vec<f64> = (0..n).map(|i| f64_of(trid[(i, i)])).collect();
    let off: Vec<f64> = (0..n - 1).map(|i| f64_of(trid[(i + 1, i)])).collect();
    values::tridiagonal_eigenvalues(&mut diag, &off).map_err(|_| LinalgError::NoConvergence)?;
    Ok(diag.into_iter().map(|v| of_f64(v / scale)).collect())
}

/// A power of two bringing the largest magnitude of `m` (its lower triangle if `lower`) into
/// `[lo, hi]`, so the reductions neither underflow nor overflow; exact both ways. 1 when it is
/// in range already, or zero, or not finite.
fn scaling<T: LinalgFloat>(m: MatRef<'_, T>, lower: bool, lo: f64, hi: f64) -> f64 {
    let mut top = 0.0f64;
    for j in 0..m.ncols() {
        for i in if lower { j } else { 0 }..m.nrows() {
            top = top.max(f64_of(m[(i, j)]).abs());
        }
    }
    if top > 0.0 && top < lo {
        (lo / top).log2().ceil().exp2()
    } else if top > hi && top.is_finite() {
        (-(top / hi).log2().ceil()).exp2()
    } else {
        1.0
    }
}

/// The smallest positive normal value of `T`.
fn min_positive<T: LinalgFloat>() -> f64 {
    2f64.powi(T::_MIN_EXP - 1)
}

fn f64_of<T: LinalgFloat>(x: T) -> f64 {
    x.to_f64().expect("f32 and f64 convert to f64")
}

fn of_f64<T: LinalgFloat>(x: f64) -> T {
    T::from_f64(x).expect("f64 converts to f32 and f64")
}

/// A singular value decomposition `a = u · diag(s) · vt`.
#[derive(Debug, Clone, PartialEq)]
pub struct Svd<T> {
    pub u: NdArray<T>,
    /// Non-negative, largest first.
    pub s: Vec<T>,
    pub vt: NdArray<T>,
}

/// The singular value decomposition: reduced (`u` m x k, `vt` k x n, k = min(m, n)) unless `full`
/// (`u` m x m, `vt` n x n).
pub fn svd<T: LinalgFloat>(a: NdView<'_, T>, full: bool) -> Result<Svd<T>, LinalgError> {
    let m = matrix(a)?;
    let svd = if full { m.svd() } else { m.thin_svd() }.map_err(|_| LinalgError::NoConvergence)?;
    let (u, v) = (svd.U(), svd.V());
    let s = svd.S().column_vector().iter().copied().collect();
    Ok(Svd { u: to_array(u, &[u.nrows(), u.ncols()]), s, vt: to_array(v.transpose(), &[v.ncols(), v.nrows()]) })
}

/// The singular values, largest first.
///
/// faer reduces the matrix to bidiagonal form; the singular values of that come from dqds
/// (LAPACK's `dlasq1`), each to high relative accuracy and much faster than the divide and
/// conquer faer's own values-only path uses.
pub fn svdvals<T: LinalgFloat>(a: NdView<'_, T>) -> Result<Vec<T>, LinalgError> {
    use faer::linalg::svd::bidiag::{bidiag_in_place, bidiag_in_place_scratch};
    let m = matrix(a)?;
    // the transpose has the same singular values; reduce the tall one
    let m = if m.nrows() < m.ncols() { m.transpose() } else { m };
    let (rows, n) = (m.nrows(), m.ncols());
    if n == 0 {
        return Ok(Vec::new());
    }
    let par = faer::get_global_parallelism();
    // scaled into range, as LAPACK's dgesvd does: smlnum = sqrt(safmin) / eps, bignum = 1 / smlnum
    let smlnum = min_positive::<T>().sqrt() / f64_of(T::_EPSILON);
    let scale = scaling(m, false, smlnum, 1.0 / smlnum);
    let k = of_f64::<T>(scale);
    let mut bid = faer::Mat::<T>::from_fn(rows, n, |i, j| m[(i, j)] * k);
    let bs = faer::linalg::qr::no_pivoting::factor::recommended_block_size::<T>(rows, n);
    let (mut hl, mut hr) = (faer::Mat::<T>::zeros(bs, n), faer::Mat::<T>::zeros(bs, n - 1));
    let mut buf = faer::dyn_stack::MemBuffer::new(bidiag_in_place_scratch::<T>(rows, n, par, Default::default()));
    bidiag_in_place(bid.as_mut(), hl.as_mut(), hr.as_mut(), par, faer::dyn_stack::MemStack::new(&mut buf), Default::default());
    let mut diag: Vec<f64> = (0..n).map(|i| f64_of(bid[(i, i)])).collect();
    let off: Vec<f64> = (0..n - 1).map(|i| f64_of(bid[(i, i + 1)])).collect();
    values::bidiagonal_singular_values(&mut diag, &off).map_err(|_| LinalgError::NoConvergence)?;
    Ok(diag.into_iter().map(|v| of_f64(v / scale)).collect())
}

/// The Moore-Penrose pseudo-inverse (n x m for an m x n matrix).
pub fn pinv<T: LinalgFloat>(a: NdView<'_, T>) -> Result<NdArray<T>, LinalgError> {
    let m = matrix(a)?;
    let svd = m.thin_svd().map_err(|_| LinalgError::NoConvergence)?;
    let p = svd.pseudoinverse();
    Ok(to_array(p.as_ref(), &[m.ncols(), m.nrows()]))
}

/// The effective rank (singular values above `eps · max(m, n) · s_max`).
pub fn matrix_rank<T: LinalgFloat>(a: NdView<'_, T>) -> Result<usize, LinalgError> {
    let m = matrix(a)?;
    let s = svdvals(a)?;
    let tol = T::_EPSILON * T::_lit(m.nrows().max(m.ncols()) as f64) * s.first().copied().unwrap_or(T::_ZERO);
    Ok(s.iter().filter(|&&v| v > tol).count())
}

/// The 2-norm condition number (largest over smallest singular value).
pub fn cond<T: LinalgFloat>(a: NdView<'_, T>) -> Result<T, LinalgError> {
    let s = svdvals(a)?;
    match (s.first(), s.last()) {
        (Some(&hi), Some(&lo)) => Ok(hi / lo),
        _ => Err(LinalgError::Dims { expected: "non-empty", got: a.shape().to_vec() }),
    }
}

#[cfg(test)]
mod tests;
