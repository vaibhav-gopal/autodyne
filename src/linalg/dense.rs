//! A small dense f64 matrix for the algorithms written element by element here (the Schur QR
//! iteration, Bartels-Stewart, the Riccati iterations); products, inverses and solves go to faer.

use faer::linalg::solvers::DenseSolveCore;
use faer::Mat as FaerMat;

use super::LinalgError;
use crate::signal::{NdArray, NdView};
use crate::units::*;

/// Row-major `rows x cols` f64.
#[derive(Clone, Debug, PartialEq)]
pub(crate) struct Mat {
    pub(crate) rows: usize,
    pub(crate) cols: usize,
    pub(crate) data: Vec<f64>,
}

impl Mat {
    pub(crate) fn zeros(rows: usize, cols: usize) -> Self {
        Self { rows, cols, data: vec![0.0; rows * cols] }
    }
    pub(crate) fn identity(n: usize) -> Self {
        let mut m = Self::zeros(n, n);
        for i in 0..n {
            m.data[i * n + i] = 1.0;
        }
        m
    }
    #[inline]
    pub(crate) fn at(&self, i: usize, j: usize) -> f64 {
        self.data[i * self.cols + j]
    }
    #[inline]
    pub(crate) fn set(&mut self, i: usize, j: usize, v: f64) {
        self.data[i * self.cols + j] = v;
    }
    /// A 2-D view (a 1-D one as a column), any strides, as f64.
    pub(crate) fn from_view<T: Float>(v: NdView<'_, T>) -> Result<Self, LinalgError> {
        let (rows, cols) = match v.shape() {
            &[r, c] => (r, c),
            &[r] => (r, 1),
            s => return Err(LinalgError::Dims { expected: "1-D or 2-D", got: s.to_vec() }),
        };
        Ok(Self { rows, cols, data: v.iter().map(|x| x.to_f64().unwrap_or(f64::NAN)).collect() })
    }
    /// A square 2-D view.
    pub(crate) fn square<T: Float>(v: NdView<'_, T>) -> Result<Self, LinalgError> {
        let m = Self::from_view(v)?;
        if v.ndim() != 2 || m.rows != m.cols {
            return Err(LinalgError::NotSquare(v.shape().to_vec()));
        }
        Ok(m)
    }
    pub(crate) fn to_array<T: Float + Default>(&self) -> NdArray<T> {
        NdArray::from_vec(self.data.iter().map(|&v| T::_lit(v)).collect(), &[self.rows, self.cols]).expect("rows x cols")
    }
    fn faer(&self) -> FaerMat<f64> {
        FaerMat::from_fn(self.rows, self.cols, |i, j| self.at(i, j))
    }
    fn from_faer(m: faer::MatRef<'_, f64>) -> Self {
        let mut out = Self::zeros(m.nrows(), m.ncols());
        for i in 0..m.nrows() {
            for j in 0..m.ncols() {
                out.set(i, j, m[(i, j)]);
            }
        }
        out
    }
    pub(crate) fn t(&self) -> Self {
        let mut out = Self::zeros(self.cols, self.rows);
        for i in 0..self.rows {
            for j in 0..self.cols {
                out.set(j, i, self.at(i, j));
            }
        }
        out
    }
    pub(crate) fn mul(&self, other: &Mat) -> Self {
        assert_eq!(self.cols, other.rows, "inner dimensions");
        Self::from_faer((self.faer() * other.faer()).as_ref())
    }
    pub(crate) fn add(&self, other: &Mat) -> Self {
        Self { rows: self.rows, cols: self.cols, data: self.data.iter().zip(&other.data).map(|(a, b)| a + b).collect() }
    }
    pub(crate) fn sub(&self, other: &Mat) -> Self {
        Self { rows: self.rows, cols: self.cols, data: self.data.iter().zip(&other.data).map(|(a, b)| a - b).collect() }
    }
    pub(crate) fn scale(&self, s: f64) -> Self {
        Self { rows: self.rows, cols: self.cols, data: self.data.iter().map(|v| v * s).collect() }
    }
    /// `(m + mᵀ) / 2`.
    pub(crate) fn symmetrized(&self) -> Self {
        self.add(&self.t()).scale(0.5)
    }
    pub(crate) fn norm(&self) -> f64 {
        self.data.iter().map(|v| v * v).sum::<f64>().sqrt()
    }
    /// The inverse, by LU with partial pivoting; errors when singular.
    pub(crate) fn inv(&self) -> Result<Self, LinalgError> {
        let lu = self.faer().partial_piv_lu();
        let u = lu.U();
        let biggest = (0..u.nrows()).map(|i| u[(i, i)].abs()).fold(0.0, f64::max);
        if (0..u.nrows()).any(|i| u[(i, i)].abs() <= biggest * f64::EPSILON * self.rows as f64) {
            return Err(LinalgError::Singular);
        }
        Ok(Self::from_faer(lu.inverse().as_ref()))
    }
    /// `|det|^(1 / n)` (for scaling the sign iteration), from the LU factors' diagonal.
    pub(crate) fn det_root(&self) -> f64 {
        let lu = self.faer().partial_piv_lu();
        let u = lu.U();
        let log: f64 = (0..u.nrows()).map(|i| u[(i, i)].abs().ln()).sum();
        (log / self.rows as f64).exp()
    }
    /// The least-squares solution of `self x = b`.
    pub(crate) fn lstsq(&self, b: &Mat) -> Result<Self, LinalgError> {
        let fit = super::lstsq(self.to_array::<f64>().view(), b.to_array::<f64>().view())?;
        Ok(Self { rows: self.cols, cols: b.cols, data: fit.solution.into_vec() })
    }
}
