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

use crate::gemm::{small, written};
use crate::signal::{NdArray, NdView, NdViewMut};
use crate::units::*;

mod bdc;
mod dc;
pub(crate) mod dense;
mod equations;
mod expm;
mod poly;
mod reorder;
mod schur;
mod values;
pub use equations::*;
pub use expm::expm;
pub use poly::*;
pub use schur::schur;

/// Element types linear algebra works on: `f32` and `f64`.
pub trait LinalgFloat: Float + Default + faer::traits::RealField {}
impl LinalgFloat for f32 {}
impl LinalgFloat for f64 {}

/// Errors from linear algebra.
#[derive(Debug, Clone, PartialEq, Eq, Error)]
pub enum LinalgError {
    #[error("expected a {expected} array, got shape {got:?}")]
    /// An operand with the wrong number of axes.
    Dims {
        /// What was needed, e.g. "2-D".
        expected: &'static str,
        /// The shape given.
        got: Vec<usize>,
    },
    /// A square matrix was needed (the shape given).
    #[error("expected a square matrix, got shape {0:?}")]
    NotSquare(Vec<usize>),
    /// Operand shapes that don't fit together.
    #[error("shapes {0:?} and {1:?} are not aligned")]
    Mismatch(Vec<usize>, Vec<usize>),
    /// The matrix has no inverse.
    #[error("the matrix is singular")]
    Singular,
    /// Cholesky needs a symmetric positive definite matrix.
    #[error("the matrix is not positive definite")]
    NotPositiveDefinite,
    /// An iterative algorithm didn't converge.
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
    let (_, _, shape) = product(a, b)?;
    let mut strides = vec![1isize; shape.len()];
    for i in (0..shape.len().saturating_sub(1)).rev() {
        strides[i] = strides[i + 1] * shape[i + 1] as isize;
    }
    let data = written(shape.iter().product(), |out| {
        // SAFETY: a row-major buffer of the product's shape, borrowed mutably for the call
        let view = unsafe { NdViewMut::from_raw_parts(out, &shape, &strides) }.expect("a valid layout");
        matmul_into(a, b, view)
    })?;
    Ok(NdArray::from_vec(data, &shape).expect("valid shape"))
}

/// [`matmul`] into `out` (the product's shape, any strides), which is overwritten: no new array.
pub fn matmul_into<T: LinalgFloat>(a: NdView<'_, T>, b: NdView<'_, T>, mut out: NdViewMut<'_, T>) -> Result<(), LinalgError> {
    let (lhs, rhs, shape) = product(a, b)?;
    if out.shape() != shape.as_slice() {
        return Err(LinalgError::Mismatch(shape, out.shape().to_vec()));
    }
    let (m, n) = (lhs.nrows(), rhs.ncols());
    // the output as an m x n matrix: [m, n], [n] (m = 1), [m] (n = 1) or [] (both 1)
    let (rs, cs) = match (out.strides(), shape.len()) {
        (&[rs, cs], 2) => (rs, cs),
        (&[s], 1) if m == 1 => (0, s),
        (&[s], 1) => (s, 0),
        _ => (0, 0),
    };
    if lhs.ncols() == 0 {
        out.map_inplace(|x| *x = T::_ZERO);
        return Ok(());
    }
    // small row-major products: a register-blocked kernel (no packing)
    let k = lhs.ncols();
    let row_major = |rs: isize, cs: isize, cols: usize| cs == 1 && rs == cols as isize;
    if shape.len() == 2 && row_major(lhs.row_stride(), lhs.col_stride(), k) && row_major(rhs.row_stride(), rhs.col_stride(), n) && row_major(rs, cs, n) && small::matmul(lhs.as_ptr(), rhs.as_ptr(), out.as_mut_ptr(), m, k, n) {
        return Ok(());
    }
    // SAFETY: `out` is a validated view of the product's shape, borrowed mutably for the call
    let dst = unsafe { MatMut::from_raw_parts_mut(out.as_mut_ptr(), m, n, rs, cs) };
    faer::linalg::matmul::matmul(dst, Accum::Replace, lhs, rhs, T::_ONE, product_parallelism(m, n, k));
    Ok(())
}

/// The global parallelism for products big enough to share out, else sequential: a small output
/// (a 50 x 50 Gram matrix of long rows, say) or little work runs several times slower on a thread
/// pool than on one core, the hand-offs costing more than the arithmetic.
fn product_parallelism(m: usize, n: usize, k: usize) -> faer::Par {
    if m * n < 64 * 64 || m * n * k < 1 << 24 { faer::Par::Seq } else { faer::get_global_parallelism() }
}

/// The operands of a product as matrices, and the product's shape.
#[allow(clippy::type_complexity)]
fn product<'a, T: LinalgFloat>(a: NdView<'a, T>, b: NdView<'a, T>) -> Result<(MatRef<'a, T>, MatRef<'a, T>, Vec<usize>), LinalgError> {
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
    Ok((lhs, rhs, matmul_shape(&sa, &sb)?))
}

/// The shape of [`matmul`]'s result for operands of these shapes.
pub fn matmul_shape(a: &[usize], b: &[usize]) -> Result<Vec<usize>, LinalgError> {
    let inner = |s: &[usize], first: bool| if s.len() == 1 { s[0] } else if first { s[1] } else { s[0] };
    match (a.len(), b.len()) {
        (1 | 2, 1 | 2) if inner(a, true) != inner(b, false) => Err(LinalgError::Mismatch(a.to_vec(), b.to_vec())),
        (1, 1) => Ok(vec![]),
        (1, 2) => Ok(vec![b[1]]),
        (2, 1) => Ok(vec![a[0]]),
        (2, 2) => Ok(vec![a[0], b[1]]),
        _ => Err(LinalgError::Dims { expected: "1-D or 2-D", got: if a.len() > 2 || a.is_empty() { a.to_vec() } else { b.to_vec() } }),
    }
}

/// The solution `x` of `a · x = b` (`a` square; `b` a vector or a matrix of right-hand sides).
pub fn solve<T: LinalgFloat>(a: NdView<'_, T>, b: NdView<'_, T>) -> Result<NdArray<T>, LinalgError> {
    let m = square(a)?;
    let rhs = columns(b)?;
    if rhs.nrows() != m.nrows() {
        return Err(LinalgError::Mismatch(a.shape().to_vec(), b.shape().to_vec()));
    }
    let (m, transposed) = column_major(m);
    let lu = m.partial_piv_lu();
    check_nonsingular(lu.U())?;
    // aᵀ's factors solve a · x = b as well
    let x = if transposed { lu.solve_transpose(rhs) } else { lu.solve(rhs) };
    Ok(to_array(x.as_ref(), b.shape()))
}

/// `m`, or its transpose when that is column-major and `m` is not, and whether it was transposed.
/// faer factors a column-major copy of its input: a row-major matrix (autodyne's own layout) read
/// in place is its transpose in column-major order, copied in memory order, while copying it as
/// is reads across the rows.
fn column_major<T>(m: MatRef<'_, T>) -> (MatRef<'_, T>, bool) {
    if m.col_stride() == 1 && m.row_stride() != 1 { (m.transpose(), true) } else { (m, false) }
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
    // det(aᵀ) = det(a)
    Ok(column_major(square(a)?).0.determinant())
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
    /// The eigenvalues, in no particular order.
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
///
/// faer reduces the matrix to tridiagonal form; autodyne's divide and conquer (LAPACK's `dstedc`)
/// solves that, and faer's blocked Householder products carry the vectors back.
pub fn eigh<T: LinalgFloat>(a: NdView<'_, T>) -> Result<(Vec<T>, NdArray<T>), LinalgError> {
    use faer::linalg::evd::tridiag::{tridiag_in_place, tridiag_in_place_scratch};
    use faer::linalg::householder::{apply_block_householder_sequence_on_the_left_in_place_scratch, apply_block_householder_sequence_on_the_left_in_place_with_conj};
    let m = square(a)?;
    let n = m.nrows();
    if n == 0 {
        return Ok((Vec::new(), NdArray::zeros(&[0, 0]).expect("valid shape")));
    }
    let par = faer::get_global_parallelism();
    let (mut trid, scale) = lower_scaled(m);
    let bs = faer::linalg::qr::no_pivoting::factor::recommended_block_size::<T>(n, n);
    let mut householder = faer::Mat::<T>::zeros(bs, n.saturating_sub(1));
    let mut buf = faer::dyn_stack::MemBuffer::new(faer::dyn_stack::StackReq::any_of(&[
        tridiag_in_place_scratch::<T>(n, par, Default::default()),
        apply_block_householder_sequence_on_the_left_in_place_scratch::<T>(n.saturating_sub(1), bs, n),
    ]));
    if n > 1 {
        tridiag_in_place(trid.as_mut(), householder.as_mut(), par, faer::dyn_stack::MemStack::new(&mut buf), Default::default());
    }
    let mut values: Vec<f64> = (0..n).map(|i| f64_of(trid[(i, i)])).collect();
    let off: Vec<f64> = (0..n - 1).map(|i| f64_of(trid[(i + 1, i)])).collect();
    let mut q = vec![0.0; n * n];
    dc::tridiagonal_eigen(&mut values, &off, &mut q).map_err(|_| LinalgError::NoConvergence)?;
    let mut u = faer::Mat::<T>::from_fn(n, n, |i, j| of_f64(q[j * n + i]));
    if n > 1 {
        apply_block_householder_sequence_on_the_left_in_place_with_conj(
            trid.as_ref().submatrix(1, 0, n - 1, n - 1),
            householder.as_ref(),
            faer::Conj::No,
            u.as_mut().subrows_mut(1, n - 1),
            par,
            faer::dyn_stack::MemStack::new(&mut buf),
        );
    }
    Ok((values.into_iter().map(|v| of_f64(v / scale)).collect(), to_array(u.as_ref(), a.shape())))
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
    let (mut trid, scale) = lower_scaled(m);
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

/// The lower triangle of `m` in a new column-major matrix (zeros above), scaled into range as
/// LAPACK's dsyev does (rmin = sqrt(safmin / eps), rmax = 1 / rmin), and the power of two it was
/// scaled by: one pass over the input in its own memory order.
fn lower_scaled<T: LinalgFloat>(m: MatRef<'_, T>) -> (faer::Mat<T>, f64) {
    let n = m.nrows();
    let mut out = faer::Mat::<T>::zeros(n, n);
    let mut top = T::_ZERO;
    if m.col_stride() == 1 {
        // row-major: read rows, contiguous
        for i in 0..n {
            for j in 0..=i {
                let v = m[(i, j)];
                top = top.maximum(v.abs());
                out[(i, j)] = v;
            }
        }
    } else {
        for j in 0..n {
            for i in j..n {
                let v = m[(i, j)];
                top = top.maximum(v.abs());
                out[(i, j)] = v;
            }
        }
    }
    let rmin = (min_positive::<T>() / f64_of(T::_EPSILON)).sqrt();
    let top = f64_of(top);
    let scale = if top > 0.0 && top < rmin {
        (rmin / top).log2().ceil().exp2()
    } else if top > 1.0 / rmin && top.is_finite() {
        (-(top * rmin).log2().ceil()).exp2()
    } else {
        1.0
    };
    if scale != 1.0 {
        let k = of_f64::<T>(scale);
        for j in 0..n {
            for i in j..n {
                out[(i, j)] *= k;
            }
        }
    }
    (out, scale)
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
    /// Left singular vectors (columns).
    pub u: NdArray<T>,
    /// Non-negative, largest first.
    pub s: Vec<T>,
    /// Right singular vectors, transposed (rows).
    pub vt: NdArray<T>,
}

/// The singular value decomposition: reduced (`u` m x k, `vt` k x n, k = min(m, n)) unless `full`
/// (`u` m x m, `vt` n x n).
///
/// As LAPACK's `gesdd`: faer reduces the matrix to bidiagonal form, autodyne's divide and conquer
/// solves that, and faer's blocked Householder products carry the vectors back.
pub fn svd<T: LinalgFloat>(a: NdView<'_, T>, full: bool) -> Result<Svd<T>, LinalgError> {
    let m = matrix(a)?;
    if m.nrows() < m.ncols() {
        // the transpose's: a^T = U S V^T, so a = V S U^T
        let Svd { u, s, vt } = svd_tall(m.transpose(), full)?;
        let (ur, uc) = (u.shape()[0], u.shape()[1]);
        let (vr, vc) = (vt.shape()[0], vt.shape()[1]);
        let u_t = NdArray::from_vec((0..uc).flat_map(|j| (0..ur).map(move |i| (i, j))).map(|(i, j)| u.as_slice()[i * uc + j]).collect(), &[uc, ur]).expect("valid shape");
        let vt_t = NdArray::from_vec((0..vc).flat_map(|j| (0..vr).map(move |i| (i, j))).map(|(i, j)| vt.as_slice()[i * vc + j]).collect(), &[vc, vr]).expect("valid shape");
        return Ok(Svd { u: vt_t, s, vt: u_t });
    }
    svd_tall(m, full)
}

/// [`svd`] of a matrix with at least as many rows as columns.
fn svd_tall<T: LinalgFloat>(m: MatRef<'_, T>, full: bool) -> Result<Svd<T>, LinalgError> {
    use faer::linalg::householder::{apply_block_householder_sequence_on_the_left_in_place_scratch, apply_block_householder_sequence_on_the_left_in_place_with_conj};
    use faer::linalg::svd::bidiag::{bidiag_in_place, bidiag_in_place_scratch};
    let (rows, n) = (m.nrows(), m.ncols());
    let ucols = if full { rows } else { n };
    if n == 0 {
        let u = faer::Mat::<T>::from_fn(rows, ucols, |i, j| if i == j { T::_ONE } else { T::_ZERO });
        return Ok(Svd { u: to_array(u.as_ref(), &[rows, ucols]), s: Vec::new(), vt: NdArray::zeros(&[0, 0]).expect("valid shape") });
    }
    let par = faer::get_global_parallelism();
    // scaled into range, as LAPACK's gesdd does: smlnum = sqrt(safmin) / eps, bignum = 1 / smlnum
    let smlnum = min_positive::<T>().sqrt() / f64_of(T::_EPSILON);
    let scale = scaling(m, false, smlnum, 1.0 / smlnum);
    let k = of_f64::<T>(scale);
    let mut bid = faer::Mat::<T>::from_fn(rows, n, |i, j| m[(i, j)] * k);
    let bs = faer::linalg::qr::no_pivoting::factor::recommended_block_size::<T>(rows, n);
    let (mut hl, mut hr) = (faer::Mat::<T>::zeros(bs, n), faer::Mat::<T>::zeros(bs, n - 1));
    let mut buf = faer::dyn_stack::MemBuffer::new(faer::dyn_stack::StackReq::any_of(&[
        bidiag_in_place_scratch::<T>(rows, n, par, Default::default()),
        apply_block_householder_sequence_on_the_left_in_place_scratch::<T>(n, bs, ucols),
        apply_block_householder_sequence_on_the_left_in_place_scratch::<T>(n.saturating_sub(1), bs, n),
    ]));
    bidiag_in_place(bid.as_mut(), hl.as_mut(), hr.as_mut(), par, faer::dyn_stack::MemStack::new(&mut buf), Default::default());
    let mut s: Vec<f64> = (0..n).map(|i| f64_of(bid[(i, i)])).collect();
    let off: Vec<f64> = (0..n - 1).map(|i| f64_of(bid[(i, i + 1)])).collect();
    let (mut ub, mut vb) = (vec![0.0; n * n], vec![0.0; n * n]);
    bdc::bidiagonal_svd(&mut s, &off, &mut ub, &mut vb).map_err(|_| LinalgError::NoConvergence)?;
    // U: [U_B 0; 0 I] through the left reflectors
    let mut u = faer::Mat::<T>::from_fn(rows, ucols, |i, j| if i < n && j < n { of_f64(ub[j * n + i]) } else if i == j { T::_ONE } else { T::_ZERO });
    apply_block_householder_sequence_on_the_left_in_place_with_conj(bid.as_ref(), hl.as_ref(), faer::Conj::No, u.as_mut(), par, faer::dyn_stack::MemStack::new(&mut buf));
    // V: V_B through the right reflectors (stored above the superdiagonal, mirrored below as faer does)
    let mut v = faer::Mat::<T>::from_fn(n, n, |i, j| of_f64(vb[j * n + i]));
    if n > 1 {
        for j in 1..n {
            for i in 0..j {
                bid[(j, i)] = bid[(i, j)];
            }
        }
        apply_block_householder_sequence_on_the_left_in_place_with_conj(
            bid.as_ref().submatrix(1, 0, n - 1, n - 1),
            hr.as_ref(),
            faer::Conj::Yes,
            v.as_mut().subrows_mut(1, n - 1),
            par,
            faer::dyn_stack::MemStack::new(&mut buf),
        );
    }
    Ok(Svd { u: to_array(u.as_ref(), &[rows, ucols]), s: s.into_iter().map(|x| of_f64(x / scale)).collect(), vt: to_array(v.as_ref().transpose(), &[n, n]) })
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
