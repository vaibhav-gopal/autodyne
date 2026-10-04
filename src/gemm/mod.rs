//! Matrix-product kernels on plain buffers (crate-internal; feature `faer`): what both the arrays'
//! `ArrayMath::dot_general` (`signal`) and `linalg::matmul` run, kept below both so neither depends
//! on the other for it. faer's multithreaded gemm in general, and a register-blocked AVX2 kernel
//! ([`small::matmul`]) for small row-major products, where packing and dispatch would dominate.

use std::any::{Any, TypeId};
use std::convert::Infallible;

use faer::{Accum, MatMut, MatRef};

use crate::units::*;

pub(crate) mod small;

/// Element types faer multiplies: `f32` and `f64`.
trait Elem: Float + faer::traits::RealField {}
impl Elem for f32 {}
impl Elem for f64 {}

/// `len` values written by `write` into fresh, uninitialized memory (no zero fill first). `write`
/// must write every element.
pub(crate) fn written<T, E>(len: usize, write: impl FnOnce(*mut T) -> Result<(), E>) -> Result<Vec<T>, E> {
    let mut data: Vec<std::mem::MaybeUninit<T>> = Vec::with_capacity(len);
    // SAFETY: MaybeUninit needs no initialization
    unsafe { data.set_len(len) };
    write(data.as_mut_ptr().cast::<T>())?;
    let mut data = std::mem::ManuallyDrop::new(data);
    // SAFETY: every element was written; MaybeUninit<T> has T's layout
    Ok(unsafe { Vec::from_raw_parts(data.as_mut_ptr().cast::<T>(), data.len(), data.capacity()) })
}

/// Row-major matrix product of contiguous buffers, for any element type: `Some(a[m x k] · b[k x n])`
/// for `f32` / `f64`, `None` for the others (used by `ArrayMath::dot_general`).
#[allow(clippy::ptr_arg)] // Vecs, not slices: they are downcast through Any, which needs sized types
pub(crate) fn gemm_any<T: Copy + 'static>(a: &Vec<T>, b: &Vec<T>, m: usize, k: usize, n: usize) -> Option<Vec<T>> {
    fn go<F: Elem>(a: &[F], b: &[F], m: usize, k: usize, n: usize) -> Vec<F> {
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

/// The row-major product of two strided matrices, each `(rows, columns, row stride, column
/// stride)` from its first element, for `f32` / `f64` (`None` for other element types).
pub(crate) fn gemm_strided<T: Copy + 'static>(a: *const T, (m, k, ars, acs): (usize, usize, isize, isize), b: *const T, (k2, n, brs, bcs): (usize, usize, isize, isize)) -> Option<Vec<T>> {
    assert_eq!(k, k2, "gemm_strided: inner dimensions differ");
    fn go<F: Elem>(a: *const F, b: *const F, m: usize, k: usize, n: usize, s: [isize; 4]) -> Vec<F> {
        if k == 0 {
            return vec![F::_ZERO; m * n];
        }
        let filled = written(m * n, |out| {
            // SAFETY: the caller's views cover these matrices (their elements are inside the views'
            // memory, borrowed for the call); `out` is a fresh m x n row-major buffer
            let (lhs, rhs, dst) = unsafe { (MatRef::from_raw_parts(a, m, k, s[0], s[1]), MatRef::from_raw_parts(b, k, n, s[2], s[3]), MatMut::from_raw_parts_mut(out, m, n, n as isize, 1)) };
            // with something contracted, faer overwrites every element
            faer::linalg::matmul::matmul(dst, Accum::Replace, lhs, rhs, F::_ONE, faer::get_global_parallelism());
            Ok::<(), Infallible>(())
        });
        match filled {
            Ok(v) => v,
            Err(never) => match never {},
        }
    }
    let s = [ars, acs, brs, bcs];
    let out: Box<dyn Any> = if TypeId::of::<T>() == TypeId::of::<f32>() {
        Box::new(go::<f32>(a.cast(), b.cast(), m, k, n, s))
    } else if TypeId::of::<T>() == TypeId::of::<f64>() {
        Box::new(go::<f64>(a.cast(), b.cast(), m, k, n, s))
    } else {
        return None;
    };
    out.downcast::<Vec<T>>().ok().map(|b| *b)
}

/// Row-major matrix product of contiguous buffers: `out[m x n] = a[m x k] · b[k x n]`.
fn gemm<T: Elem>(a: &[T], b: &[T], out: &mut [T], m: usize, k: usize, n: usize) {
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