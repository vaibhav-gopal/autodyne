//! The CubeCL kernels behind [`GpuArray`](super::GpuArray): element-wise arithmetic (with
//! scalars, same-layout arrays, rows and columns), unary functions, the layout gather that
//! materializes a permuted view, reductions (all, along rows, down columns) and the per-lane FIR.
//! Generic over `f32` / `f64`; the operation is a compile-time constant, so each is its own shader.

use cubecl::prelude::*;

/// Binary operations, chosen at kernel compile time.
pub(super) const ADD: u32 = 0;
pub(super) const SUB: u32 = 1;
pub(super) const MUL: u32 = 2;
pub(super) const DIV: u32 = 3;

#[cube]
fn apply<F: Float, N: Size>(a: Vector<F, N>, b: Vector<F, N>, #[comptime] op: u32) -> Vector<F, N> {
    let mut out = a + b;
    if op == SUB {
        out = a - b;
    } else if op == MUL {
        out = a * b;
    } else if op == DIV {
        out = a / b;
    }
    out
}

#[cube(launch_unchecked)]
pub(super) fn axpb_kernel<F: Float + CubeElement, N: Size>(x: &Array<Vector<F, N>>, out: &mut Array<Vector<F, N>>, a: F, b: F) {
    if ABSOLUTE_POS < x.len() {
        out[ABSOLUTE_POS] = x[ABSOLUTE_POS] * Vector::new(a) + Vector::new(b);
    }
}

/// `x op s`, or `s op x` when `scalar_first`.
#[cube(launch_unchecked)]
pub(super) fn scalar_kernel<F: Float + CubeElement, N: Size>(x: &Array<Vector<F, N>>, out: &mut Array<Vector<F, N>>, s: F, #[comptime] op: u32, #[comptime] scalar_first: bool) {
    if ABSOLUTE_POS < x.len() {
        let v = x[ABSOLUTE_POS];
        let sv = Vector::new(s);
        if scalar_first {
            out[ABSOLUTE_POS] = apply::<F, N>(sv, v, op);
        } else {
            out[ABSOLUTE_POS] = apply::<F, N>(v, sv, op);
        }
    }
}

#[cube(launch_unchecked)]
pub(super) fn binary_kernel<F: Float, N: Size>(x: &Array<Vector<F, N>>, y: &Array<Vector<F, N>>, out: &mut Array<Vector<F, N>>, #[comptime] op: u32) {
    if ABSOLUTE_POS < x.len() {
        out[ABSOLUTE_POS] = apply::<F, N>(x[ABSOLUTE_POS], y[ABSOLUTE_POS], op);
    }
}

/// `x` (rows x cols in memory, cols in vectors) with a vector along its rows (cols, in vectors).
#[cube(launch_unchecked)]
pub(super) fn row_kernel<F: Float, N: Size>(x: &Array<Vector<F, N>>, row: &Array<Vector<F, N>>, out: &mut Array<Vector<F, N>>, cols: usize, #[comptime] op: u32) {
    if ABSOLUTE_POS < x.len() {
        out[ABSOLUTE_POS] = apply::<F, N>(x[ABSOLUTE_POS], row[ABSOLUTE_POS % cols], op);
    }
}

/// `x` (rows x cols in memory, cols in vectors) with a vector down its columns (rows).
#[cube(launch_unchecked)]
pub(super) fn column_kernel<F: Float, N: Size>(x: &Array<Vector<F, N>>, column: &Array<F>, out: &mut Array<Vector<F, N>>, cols: usize, #[comptime] op: u32) {
    if ABSOLUTE_POS < x.len() {
        out[ABSOLUTE_POS] = apply::<F, N>(x[ABSOLUTE_POS], Vector::new(column[ABSOLUTE_POS / cols]), op);
    }
}

/// Unary functions, chosen at kernel compile time.
pub(super) const EXP: u32 = 0;
pub(super) const LN: u32 = 1;
pub(super) const TANH: u32 = 2;
pub(super) const SIN: u32 = 3;
pub(super) const COS: u32 = 4;
pub(super) const SQRT: u32 = 5;
pub(super) const ABS: u32 = 6;
pub(super) const NEG: u32 = 7;

#[cube(launch_unchecked)]
pub(super) fn unary_kernel<F: Float, N: Size>(x: &Array<Vector<F, N>>, out: &mut Array<Vector<F, N>>, #[comptime] op: u32) {
    if ABSOLUTE_POS < x.len() {
        let v = x[ABSOLUTE_POS];
        let mut r = v;
        if op == EXP {
            r = v.exp();
        } else if op == LN {
            r = v.ln();
        } else if op == TANH {
            r = v.tanh();
        } else if op == SIN {
            r = v.sin();
        } else if op == COS {
            r = v.cos();
        } else if op == SQRT {
            r = v.sqrt();
        } else if op == ABS {
            r = v.abs();
        } else if op == NEG {
            r = Vector::new(F::from_int(0)) - v;
        }
        out[ABSOLUTE_POS] = r;
    }
}


/// Row-major `out` of `shape`, element `i` read from `x` at the `strides`-weighted index: any
/// permutation of a layout.
#[cube(launch_unchecked)]
pub(super) fn gather_kernel<F: Float>(x: &Array<F>, out: &mut Array<F>, shape: &Array<u32>, strides: &Array<u32>) {
    if ABSOLUTE_POS < out.len() {
        let mut rest = ABSOLUTE_POS as u32;
        let mut offset = 0u32;
        let mut d = shape.len();
        while d > 0 {
            d -= 1;
            let s = shape[d];
            offset += (rest % s) * strides[d];
            rest /= s;
        }
        out[ABSOLUTE_POS] = x[offset as usize];
    }
}

/// The sum of a cube's values in shared memory (a power of two of them), into slot 0.
#[cube]
fn reduce_shared<F: Float>(shared: &mut SharedMemory<F>, #[comptime] threads: u32) {
    let unit = UNIT_POS as usize;
    #[unroll]
    for k in 0..comptime![threads.trailing_zeros()] {
        let half = comptime![(threads >> (k + 1)) as usize];
        if unit < half {
            shared[unit] = shared[unit] + shared[unit + half];
        }
        sync_cube();
    }
}

/// Partial sums: each cube sums a grid-strided share of `x` into `partial[CUBE_POS]`.
#[cube(launch_unchecked)]
pub(super) fn sum_partial_kernel<F: Float>(x: &Array<F>, partial: &mut Array<F>, #[comptime] threads: u32) {
    let mut s = F::from_int(0);
    let mut i = ABSOLUTE_POS;
    let stride = CUBE_COUNT * CUBE_DIM as usize;
    while i < x.len() {
        s += x[i];
        i += stride;
    }
    let mut shared = SharedMemory::<F>::new(threads as usize);
    shared[UNIT_POS as usize] = s;
    sync_cube();
    reduce_shared::<F>(&mut shared, threads);
    if UNIT_POS == 0 {
        partial[CUBE_POS] = shared[0];
    }
}

/// One cube per row of `x` (rows x cols): the row's sum.
#[cube(launch_unchecked)]
pub(super) fn row_sum_kernel<F: Float>(x: &Array<F>, out: &mut Array<F>, cols: usize, #[comptime] threads: u32) {
    let row = CUBE_POS;
    let mut s = F::from_int(0);
    let mut j = UNIT_POS as usize;
    while j < cols {
        s += x[row * cols + j];
        j += threads as usize;
    }
    let mut shared = SharedMemory::<F>::new(threads as usize);
    shared[UNIT_POS as usize] = s;
    sync_cube();
    reduce_shared::<F>(&mut shared, threads);
    if UNIT_POS == 0 {
        out[row] = shared[0];
    }
}

/// Column sums of `x` (rows x cols, cols in vectors) over a chunk of rows per cube row:
/// `partial[chunk * cols + c]`.
#[cube(launch_unchecked)]
pub(super) fn column_partial_kernel<F: Float, N: Size>(x: &Array<Vector<F, N>>, partial: &mut Array<Vector<F, N>>, rows: usize, cols: usize, chunk: usize) {
    let c = ABSOLUTE_POS_X as usize;
    let part = CUBE_POS_Y as usize;
    if c < cols {
        let mut acc = Vector::<F, N>::new(F::from_int(0));
        let start = part * chunk;
        let mut end = start + chunk;
        if end > rows {
            end = rows;
        }
        for r in start..end {
            acc += x[r * cols + c];
        }
        partial[part * cols + c] = acc;
    }
}

/// The sums down `parts` rows of `partial` (parts x cols, cols in vectors).
#[cube(launch_unchecked)]
pub(super) fn column_total_kernel<F: Float, N: Size>(partial: &Array<Vector<F, N>>, out: &mut Array<Vector<F, N>>, parts: usize, cols: usize) {
    if ABSOLUTE_POS < cols {
        let mut acc = Vector::<F, N>::new(F::from_int(0));
        for p in 0..parts {
            acc += partial[p * cols + ABSOLUTE_POS];
        }
        out[ABSOLUTE_POS] = acc;
    }
}

/// A causal FIR along each lane of `x` (lanes x len): `out[l, t] = sum_j taps[j] x[l, t - j]`.
#[cube(launch_unchecked)]
pub(super) fn fir_kernel<F: Float>(x: &Array<F>, taps: &Array<F>, out: &mut Array<F>, len: usize) {
    let t = ABSOLUTE_POS_X as usize;
    let lane = CUBE_POS_Y as usize;
    if t < len {
        let base = lane * len;
        let mut acc = F::from_int(0);
        let k = taps.len();
        let mut j = 0usize;
        while j < k && j <= t {
            acc += taps[j] * x[base + t - j];
            j += 1;
        }
        out[base + t] = acc;
    }
}
