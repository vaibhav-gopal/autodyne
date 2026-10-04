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

/// Matrix product tiling: a cube of `MM_THREADS` x `MM_THREADS` units computes a `MM_TILE` square
/// of the output, each unit 4 x 4 of it, with 16 of the inner dimension staged through shared memory
/// at a time (the kernel spells these sizes out).
pub(super) const MM_THREADS: u32 = 16;
pub(super) const MM_TILE: usize = 64;

/// `c = a b` for row-major `a` (m x k) and `b` (k x n), register-tiled (see [`MM_TILE`]).
#[cube(launch_unchecked)]
pub(super) fn matmul_kernel<F: Float>(a: &Array<F>, b: &Array<F>, c: &mut Array<F>, m: usize, k: usize, n: usize) {
    let tx = UNIT_POS_X as usize;
    let ty = UNIT_POS_Y as usize;
    let unit = ty * 16 + tx;
    let row0 = CUBE_POS_Y as usize * 64;
    let col0 = CUBE_POS_X as usize * 64;
    // sa[r * 16 + kk] and sb[kk * 64 + col]
    let mut sa = SharedMemory::<F>::new(1024usize);
    let mut sb = SharedMemory::<F>::new(1024usize);
    let mut acc = Array::<F>::new(16usize);
    #[unroll]
    for i in 0..16usize {
        acc[i] = F::from_int(0);
    }
    let mut av = Array::<F>::new(4usize);
    let mut bv = Array::<F>::new(4usize);
    let mut k0 = 0usize;
    while k0 < k {
        // 1024 values of each tile, 4 per unit
        #[unroll]
        for l in 0..4usize {
            let idx = unit + l * 256;
            let (r, kk) = (idx / 16, idx % 16);
            let mut va = F::from_int(0);
            if row0 + r < m && k0 + kk < k {
                va = a[(row0 + r) * k + k0 + kk];
            }
            sa[idx] = va;
            let (kb, col) = (idx / 64, idx % 64);
            let mut vb = F::from_int(0);
            if k0 + kb < k && col0 + col < n {
                vb = b[(k0 + kb) * n + col0 + col];
            }
            sb[idx] = vb;
        }
        sync_cube();
        #[unroll]
        for kk in 0..16usize {
            #[unroll]
            for i in 0..4usize {
                av[i] = sa[(ty * 4 + i) * 16 + kk];
                bv[i] = sb[kk * 64 + tx * 4 + i];
            }
            #[unroll]
            for i in 0..4usize {
                #[unroll]
                for j in 0..4usize {
                    acc[i * 4 + j] += av[i] * bv[j];
                }
            }
        }
        sync_cube();
        k0 += 16;
    }
    #[unroll]
    for i in 0..4usize {
        #[unroll]
        for j in 0..4usize {
            let (r, col) = (row0 + ty * 4 + i, col0 + tx * 4 + j);
            if r < m && col < n {
                c[r * n + col] = acc[i * 4 + j];
            }
        }
    }
}

/// Real lanes (lanes x n) into interleaved complex ones (lanes x n x 2), imaginary parts zero.
#[cube(launch_unchecked)]
pub(super) fn real_to_complex_kernel<F: Float>(x: &Array<F>, z: &mut Array<F>) {
    if ABSOLUTE_POS < x.len() {
        z[2 * ABSOLUTE_POS] = x[ABSOLUTE_POS];
        z[2 * ABSOLUTE_POS + 1] = F::from_int(0);
    }
}

/// One radix-2 Stockham pass over every lane (lanes x n complex, interleaved): butterfly `i` of
/// `n / 2` combines `x[i]` and `x[i + n/2]` with the twiddle `tw[k (n / 2ns)]`, `k = i mod ns`,
/// and writes `y[2i - k]` and `y[2i - k + ns]`; `ns` doubles each pass, and the output comes out
/// in natural order (no bit reversal). `sign` -1 conjugates the twiddles (the inverse).
#[cube(launch_unchecked)]
pub(super) fn stockham_kernel<F: Float + CubeElement>(x: &Array<F>, y: &mut Array<F>, tw: &Array<F>, n: usize, ns: usize, sign: F) {
    let i = ABSOLUTE_POS_X as usize;
    let lane = CUBE_POS_Y as usize;
    let half = n / 2;
    if i < half {
        let base = lane * n * 2;
        let k = i % ns;
        let t = k * (half / ns);
        let (wr, wi) = (tw[2 * t], sign * tw[2 * t + 1]);
        let (ar, ai) = (x[base + 2 * i], x[base + 2 * i + 1]);
        let (br0, bi0) = (x[base + 2 * (i + half)], x[base + 2 * (i + half) + 1]);
        let br = br0 * wr - bi0 * wi;
        let bi = br0 * wi + bi0 * wr;
        let j = 2 * i - k;
        y[base + 2 * j] = ar + br;
        y[base + 2 * j + 1] = ai + bi;
        y[base + 2 * (j + ns)] = ar - br;
        y[base + 2 * (j + ns) + 1] = ai - bi;
    }
}

/// Whole transforms of rows that fit in shared memory, one cube per row: the row is loaded in
/// bit-reversed order (real input as complex values with zero imaginary parts when `real_input`),
/// transformed by `bits` in-place radix-2 stages separated by barriers, and its first `bins`
/// values written, times `scale`. `cap` (comptime) is the shared buffer's size in values,
/// `2 n` at most. `sign` -1 conjugates the twiddles (the inverse).
#[cube(launch_unchecked)]
#[allow(clippy::too_many_arguments)]
pub(super) fn fft_shared_kernel<F: Float + CubeElement>(
    x: &Array<F>,
    out: &mut Array<F>,
    tw: &Array<F>,
    n: usize,
    bits: u32,
    bins: usize,
    sign: F,
    scale: F,
    #[comptime] real_input: bool,
    #[comptime] cap: usize,
) {
    let lane = CUBE_POS;
    let threads = CUBE_DIM as usize;
    let mut s = SharedMemory::<F>::new(cap);
    // load, bit-reversed
    let mut i = UNIT_POS as usize;
    while i < n {
        let mut r = 0u32;
        let mut v = i as u32;
        for _ in 0..bits {
            r = (r << 1) | (v & 1);
            v >>= 1;
        }
        let j = r as usize;
        if real_input {
            s[2 * j] = x[lane * n + i];
            s[2 * j + 1] = F::from_int(0);
        } else {
            s[2 * j] = x[(lane * n + i) * 2];
            s[2 * j + 1] = x[(lane * n + i) * 2 + 1];
        }
        i += threads;
    }
    sync_cube();
    let mut half = 1usize;
    while half < n {
        let step = n / (2 * half);
        let mut b = UNIT_POS as usize;
        while b < n / 2 {
            let pos = b % half;
            let j = (b / half) * 2 * half + pos;
            let t = pos * step;
            let (wr, wi) = (tw[2 * t], sign * tw[2 * t + 1]);
            let (cr, ci) = (s[2 * (j + half)], s[2 * (j + half) + 1]);
            let tr = cr * wr - ci * wi;
            let ti = cr * wi + ci * wr;
            let (ar, ai) = (s[2 * j], s[2 * j + 1]);
            s[2 * (j + half)] = ar - tr;
            s[2 * (j + half) + 1] = ai - ti;
            s[2 * j] = ar + tr;
            s[2 * j + 1] = ai + ti;
            b += threads;
        }
        sync_cube();
        half *= 2;
    }
    let mut k = UNIT_POS as usize;
    while k < bins {
        out[(lane * bins + k) * 2] = s[2 * k] * scale;
        out[(lane * bins + k) * 2 + 1] = s[2 * k + 1] * scale;
        k += threads;
    }
}

/// `out[l, k] = z[l, k] w[k]` for `k < n` and zero up to `m` (complex, interleaved; `z` lanes of
/// `zn`, `out` lanes of `m`): the chirp products of Bluestein's algorithm.
#[cube(launch_unchecked)]
pub(super) fn chirp_kernel<F: Float>(z: &Array<F>, w: &Array<F>, out: &mut Array<F>, zn: usize, n: usize, m: usize) {
    let k = ABSOLUTE_POS_X as usize;
    let lane = CUBE_POS_Y as usize;
    if k < m {
        let o = (lane * m + k) * 2;
        if k < n {
            let s = (lane * zn + k) * 2;
            let (a, b, c, d) = (z[s], z[s + 1], w[2 * k], w[2 * k + 1]);
            out[o] = a * c - b * d;
            out[o + 1] = a * d + b * c;
        } else {
            out[o] = F::from_int(0);
            out[o + 1] = F::from_int(0);
        }
    }
}

/// `z[l, k] *= h[k]` (complex, interleaved; `h` shared by every lane of length `m`).
#[cube(launch_unchecked)]
pub(super) fn complex_mul_kernel<F: Float>(z: &mut Array<F>, h: &Array<F>, m: usize) {
    if ABSOLUTE_POS < z.len() / 2 {
        let k = ABSOLUTE_POS % m;
        let (a, b, c, d) = (z[2 * ABSOLUTE_POS], z[2 * ABSOLUTE_POS + 1], h[2 * k], h[2 * k + 1]);
        z[2 * ABSOLUTE_POS] = a * c - b * d;
        z[2 * ABSOLUTE_POS + 1] = a * d + b * c;
    }
}

/// The first `bins` complex values of each lane of `z` (lanes of `n`), times `scale`.
#[cube(launch_unchecked)]
pub(super) fn bins_kernel<F: Float + CubeElement>(z: &Array<F>, out: &mut Array<F>, n: usize, bins: usize, scale: F) {
    let k = ABSOLUTE_POS_X as usize;
    let lane = CUBE_POS_Y as usize;
    if k < bins {
        let (s, o) = ((lane * n + k) * 2, (lane * bins + k) * 2);
        out[o] = z[s] * scale;
        out[o + 1] = z[s + 1] * scale;
    }
}

/// The full Hermitian spectrum (lanes of `n`) from bins `0..=n/2` (lanes of `bins`):
/// `z[n - k] = conj(z[k])`; the imaginary parts of DC (and Nyquist, for even `n`) are dropped.
#[cube(launch_unchecked)]
pub(super) fn hermitian_kernel<F: Float>(half: &Array<F>, z: &mut Array<F>, n: usize, bins: usize) {
    let k = ABSOLUTE_POS_X as usize;
    let lane = CUBE_POS_Y as usize;
    if k < n {
        let o = (lane * n + k) * 2;
        if k < bins {
            let s = (lane * bins + k) * 2;
            z[o] = half[s];
            let mut im = half[s + 1];
            if k == 0 || 2 * k == n {
                im = F::from_int(0);
            }
            z[o + 1] = im;
        } else {
            let s = (lane * bins + (n - k)) * 2;
            z[o] = half[s];
            z[o + 1] = F::from_int(0) - half[s + 1];
        }
    }
}

/// The real parts of complex lanes, times `scale`.
#[cube(launch_unchecked)]
pub(super) fn real_part_kernel<F: Float + CubeElement>(z: &Array<F>, out: &mut Array<F>, scale: F) {
    if ABSOLUTE_POS < out.len() {
        out[ABSOLUTE_POS] = z[2 * ABSOLUTE_POS] * scale;
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
