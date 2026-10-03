//! Arrays on the GPU (feature `gpu`): [`GpuArray`] keeps `f32` data in device memory between
//! operations, so a chain of element-wise work, reductions and filtering never crosses the bus
//! until [`GpuArray::to_host`]. Kernels are CubeCL's, run through wgpu (Vulkan, Metal, DirectX 12).
//!
//! ```no_run
//! use autodyne::gpu::GpuArray;
//! use autodyne::signal::NdArray;
//!
//! let x = NdArray::from_vec((0..12).map(|i| i as f32).collect(), &[3, 4]).unwrap();
//! let g = GpuArray::from_host(&x.view());
//! let y = g.axpb(2.0, 0.5).tanh();
//! let rows = y.sum_axis(1);
//! assert_eq!(rows.to_host().shape(), [3]);
//! ```

use std::sync::OnceLock;

use cubecl::bytes::Bytes;
use cubecl::prelude::*;
use cubecl::server::Handle;
use cubecl::wgpu::{WgpuDevice, WgpuRuntime};

use crate::signal::{NdArray, NdView};

type Client = ComputeClient<WgpuRuntime>;

/// The default GPU's compute client, made on first use.
fn client() -> &'static Client {
    static CLIENT: OnceLock<Client> = OnceLock::new();
    CLIENT.get_or_init(|| WgpuRuntime::client(&WgpuDevice::default()))
}

/// Waits until the GPU has finished all submitted work.
pub fn sync() {
    cubecl::future::block_on(client().sync()).expect("the GPU finishes its work");
}

const THREADS: u32 = 256;

// KERNELS =========================================================================================

/// Binary operations, chosen at kernel compile time.
const ADD: u32 = 0;
const SUB: u32 = 1;
const MUL: u32 = 2;
const DIV: u32 = 3;

#[cube]
fn apply<N: Size>(a: Vector<f32, N>, b: Vector<f32, N>, #[comptime] op: u32) -> Vector<f32, N> {
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
fn axpb_kernel<N: Size>(x: &Array<Vector<f32, N>>, out: &mut Array<Vector<f32, N>>, a: f32, b: f32) {
    if ABSOLUTE_POS < x.len() {
        out[ABSOLUTE_POS] = x[ABSOLUTE_POS] * Vector::new(a) + Vector::new(b);
    }
}

#[cube(launch_unchecked)]
fn binary_kernel<N: Size>(x: &Array<Vector<f32, N>>, y: &Array<Vector<f32, N>>, out: &mut Array<Vector<f32, N>>, #[comptime] op: u32) {
    if ABSOLUTE_POS < x.len() {
        out[ABSOLUTE_POS] = apply::<N>(x[ABSOLUTE_POS], y[ABSOLUTE_POS], op);
    }
}

/// `x` (rows x cols, in vectors) with a row vector (cols, in vectors) broadcast down the rows.
#[cube(launch_unchecked)]
fn row_kernel<N: Size>(x: &Array<Vector<f32, N>>, row: &Array<Vector<f32, N>>, out: &mut Array<Vector<f32, N>>, cols: usize, #[comptime] op: u32) {
    if ABSOLUTE_POS < x.len() {
        out[ABSOLUTE_POS] = apply::<N>(x[ABSOLUTE_POS], row[ABSOLUTE_POS % cols], op);
    }
}

/// `x` (rows x cols, cols in vectors) with a column vector (rows) broadcast along the rows.
#[cube(launch_unchecked)]
fn column_kernel<N: Size>(x: &Array<Vector<f32, N>>, column: &Array<f32>, out: &mut Array<Vector<f32, N>>, cols: usize, #[comptime] op: u32) {
    if ABSOLUTE_POS < x.len() {
        out[ABSOLUTE_POS] = apply::<N>(x[ABSOLUTE_POS], Vector::new(column[ABSOLUTE_POS / cols]), op);
    }
}

/// Unary functions, chosen at kernel compile time.
const EXP: u32 = 0;
const LN: u32 = 1;
const TANH: u32 = 2;
const SIN: u32 = 3;
const COS: u32 = 4;
const SQRT: u32 = 5;
const ABS: u32 = 6;
const NEG: u32 = 7;

#[cube(launch_unchecked)]
fn unary_kernel<N: Size>(x: &Array<Vector<f32, N>>, out: &mut Array<Vector<f32, N>>, #[comptime] op: u32) {
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
            r = Vector::new(0.0) - v;
        }
        out[ABSOLUTE_POS] = r;
    }
}

/// The sum of a cube's values in shared memory (a power of two of them), into slot 0.
#[cube]
fn reduce_shared(shared: &mut SharedMemory<f32>, #[comptime] threads: u32) {
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
fn sum_partial_kernel(x: &Array<f32>, partial: &mut Array<f32>, #[comptime] threads: u32) {
    let mut s = 0.0f32;
    let mut i = ABSOLUTE_POS;
    let stride = CUBE_COUNT * CUBE_DIM as usize;
    while i < x.len() {
        s += x[i];
        i += stride;
    }
    let mut shared = SharedMemory::<f32>::new(threads as usize);
    shared[UNIT_POS as usize] = s;
    sync_cube();
    reduce_shared(&mut shared, threads);
    if UNIT_POS == 0 {
        partial[CUBE_POS] = shared[0];
    }
}

/// One cube per row of `x` (rows x cols, cols in vectors): the row's sum.
#[cube(launch_unchecked)]
fn row_sum_kernel(x: &Array<f32>, out: &mut Array<f32>, cols: usize, #[comptime] threads: u32) {
    let row = CUBE_POS;
    let mut s = 0.0f32;
    let mut j = UNIT_POS as usize;
    while j < cols {
        s += x[row * cols + j];
        j += threads as usize;
    }
    let mut shared = SharedMemory::<f32>::new(threads as usize);
    shared[UNIT_POS as usize] = s;
    sync_cube();
    reduce_shared(&mut shared, threads);
    if UNIT_POS == 0 {
        out[row] = shared[0];
    }
}

/// Column sums of `x` (rows x cols, cols in vectors) over a chunk of rows per cube row:
/// `partial[chunk * cols + c]`.
#[cube(launch_unchecked)]
fn column_partial_kernel<N: Size>(x: &Array<Vector<f32, N>>, partial: &mut Array<Vector<f32, N>>, rows: usize, cols: usize, chunk: usize) {
    let c = ABSOLUTE_POS_X as usize;
    let part = CUBE_POS_Y as usize;
    if c < cols {
        let mut acc = Vector::<f32, N>::new(0.0);
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
fn column_total_kernel<N: Size>(partial: &Array<Vector<f32, N>>, out: &mut Array<Vector<f32, N>>, parts: usize, cols: usize) {
    if ABSOLUTE_POS < cols {
        let mut acc = Vector::<f32, N>::new(0.0);
        for p in 0..parts {
            acc += partial[p * cols + ABSOLUTE_POS];
        }
        out[ABSOLUTE_POS] = acc;
    }
}

/// A causal FIR along each lane of `x` (lanes x len): `out[l, t] = sum_j taps[j] x[l, t - j]`.
#[cube(launch_unchecked)]
fn fir_kernel(x: &Array<f32>, taps: &Array<f32>, out: &mut Array<f32>, len: usize) {
    let t = ABSOLUTE_POS_X as usize;
    let lane = CUBE_POS_Y as usize;
    if t < len {
        let base = lane * len;
        let mut acc = 0.0f32;
        let k = taps.len();
        let mut j = 0usize;
        while j < k && j <= t {
            acc += taps[j] * x[base + t - j];
            j += 1;
        }
        out[base + t] = acc;
    }
}

// ARRAYS ==========================================================================================

/// `f32` values in GPU memory, row-major, with a shape.
#[derive(Clone)]
pub struct GpuArray {
    handle: Handle,
    shape: Vec<usize>,
}

impl std::fmt::Debug for GpuArray {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("GpuArray").field("shape", &self.shape).finish()
    }
}

/// Vectors of 4 when `n` allows, else single values.
fn width(n: usize) -> usize {
    if n.is_multiple_of(4) { 4 } else { 1 }
}

fn cubes(units: usize) -> CubeCount {
    CubeCount::Static((units as u32).div_ceil(THREADS).max(1), 1, 1)
}

impl GpuArray {
    /// Copies `x` (any strides) to the GPU.
    pub fn from_host(x: &NdView<'_, f32>) -> GpuArray {
        let data = match x.as_slice() {
            Some(s) => s.to_vec(),
            None => x.to_vec(),
        };
        GpuArray { handle: client().create(Bytes::from_elems(data)), shape: x.shape().to_vec() }
    }

    /// Moves `x` to the GPU (its buffer is handed over, not copied, when it is contiguous).
    pub fn from_array(x: NdArray<f32>) -> GpuArray {
        if !x.view().is_contiguous() {
            return GpuArray::from_host(&x.view());
        }
        let shape = x.shape().to_vec();
        GpuArray { handle: client().create(Bytes::from_elems(x.into_vec())), shape }
    }

    /// Copies the values back.
    pub fn to_host(&self) -> NdArray<f32> {
        let bytes = client().read_one(self.handle.clone()).expect("the GPU returns the array");
        let mut data = match bytes.try_into_vec::<f32>() {
            Ok(v) => v,
            Err(bytes) => f32::from_bytes(&bytes).to_vec(),
        };
        data.truncate(self.len());
        NdArray::from_vec(data, &self.shape).expect("the array's shape")
    }

    pub fn shape(&self) -> &[usize] {
        &self.shape
    }
    pub fn len(&self) -> usize {
        self.shape.iter().product()
    }
    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    fn empty(shape: &[usize]) -> GpuArray {
        let len: usize = shape.iter().product();
        GpuArray { handle: client().empty(len.max(1) * 4), shape: shape.to_vec() }
    }


    /// `a * x + b`, one pass.
    pub fn axpb(&self, a: f32, b: f32) -> GpuArray {
        let out = GpuArray::empty(&self.shape);
        let (n, w) = (self.len(), width(self.len()));
        if n > 0 {
            // SAFETY: both arrays hold n values; the launch covers n / w vectors
            unsafe { axpb_kernel::launch_unchecked::<WgpuRuntime>(client(), cubes(n / w), CubeDim::new_1d(THREADS), w, ArrayArg::from_raw_parts(self.handle.clone(), n), ArrayArg::from_raw_parts(out.handle.clone(), n), a, b) };
        }
        out
    }

    fn binary(&self, other: &GpuArray, op: u32) -> GpuArray {
        let out = GpuArray::empty(&self.shape);
        let n = self.len();
        if n == 0 {
            return out;
        }
        let cols = *self.shape.last().unwrap_or(&1);
        if other.shape == self.shape {
            let w = width(n);
            // SAFETY: all three hold n values
            unsafe {
                binary_kernel::launch_unchecked::<WgpuRuntime>(
                    client(),
                    cubes(n / w),
                    CubeDim::new_1d(THREADS),
                    w,
                    ArrayArg::from_raw_parts(self.handle.clone(), n),
                    ArrayArg::from_raw_parts(other.handle.clone(), n),
                    ArrayArg::from_raw_parts(out.handle.clone(), n),
                    op,
                )
            };
        } else if other.shape.as_slice() == [cols] || other.shape.as_slice() == [1, cols] {
            let w = width(cols);
            // SAFETY: x and out hold n values, the row cols
            unsafe {
                row_kernel::launch_unchecked::<WgpuRuntime>(
                    client(),
                    cubes(n / w),
                    CubeDim::new_1d(THREADS),
                    w,
                    ArrayArg::from_raw_parts(self.handle.clone(), n),
                    ArrayArg::from_raw_parts(other.handle.clone(), cols),
                    ArrayArg::from_raw_parts(out.handle.clone(), n),
                    cols / w,
                    op,
                )
            };
        } else if self.shape.len() == 2 && (other.shape.as_slice() == [self.shape[0], 1] || other.shape.as_slice() == [self.shape[0]]) {
            let w = width(cols);
            // SAFETY: x and out hold n values, the column one per row
            unsafe {
                column_kernel::launch_unchecked::<WgpuRuntime>(
                    client(),
                    cubes(n / w),
                    CubeDim::new_1d(THREADS),
                    w,
                    ArrayArg::from_raw_parts(self.handle.clone(), n),
                    ArrayArg::from_raw_parts(other.handle.clone(), self.shape[0]),
                    ArrayArg::from_raw_parts(out.handle.clone(), n),
                    cols / w,
                    op,
                )
            };
        } else {
            panic!("GpuArray: cannot combine shapes {:?} and {:?} (the same shape, a row or a column)", self.shape, other.shape);
        }
        out
    }

    /// Element by element; `other` has this shape, or is a row (`[cols]`, `[1, cols]`) or a
    /// column (`[rows, 1]`) broadcast across a matrix.
    pub fn add(&self, other: &GpuArray) -> GpuArray {
        self.binary(other, ADD)
    }
    pub fn sub(&self, other: &GpuArray) -> GpuArray {
        self.binary(other, SUB)
    }
    pub fn mul(&self, other: &GpuArray) -> GpuArray {
        self.binary(other, MUL)
    }
    pub fn div(&self, other: &GpuArray) -> GpuArray {
        self.binary(other, DIV)
    }

    fn unary(&self, op: u32) -> GpuArray {
        let out = GpuArray::empty(&self.shape);
        let (n, w) = (self.len(), width(self.len()));
        if n > 0 {
            // SAFETY: both hold n values
            unsafe { unary_kernel::launch_unchecked::<WgpuRuntime>(client(), cubes(n / w), CubeDim::new_1d(THREADS), w, ArrayArg::from_raw_parts(self.handle.clone(), n), ArrayArg::from_raw_parts(out.handle.clone(), n), op) };
        }
        out
    }
    pub fn exp(&self) -> GpuArray {
        self.unary(EXP)
    }
    pub fn ln(&self) -> GpuArray {
        self.unary(LN)
    }
    pub fn tanh(&self) -> GpuArray {
        self.unary(TANH)
    }
    pub fn sin(&self) -> GpuArray {
        self.unary(SIN)
    }
    pub fn cos(&self) -> GpuArray {
        self.unary(COS)
    }
    pub fn sqrt(&self) -> GpuArray {
        self.unary(SQRT)
    }
    pub fn abs(&self) -> GpuArray {
        self.unary(ABS)
    }
    pub fn neg(&self) -> GpuArray {
        self.unary(NEG)
    }

    /// The sum of every element, as a one-element array (still on the GPU).
    pub fn sum(&self) -> GpuArray {
        let n = self.len();
        let parts = (n as u32).div_ceil(THREADS).clamp(1, 512);
        let partial = GpuArray::empty(&[parts as usize]);
        // SAFETY: x holds n values, partial one per cube
        unsafe {
            sum_partial_kernel::launch_unchecked::<WgpuRuntime>(
                client(),
                CubeCount::Static(parts, 1, 1),
                CubeDim::new_1d(THREADS),
                ArrayArg::from_raw_parts(self.handle.clone(), n),
                ArrayArg::from_raw_parts(partial.handle.clone(), parts as usize),
                THREADS,
            )
        };
        if parts == 1 {
            return GpuArray { handle: partial.handle, shape: vec![] };
        }
        let total = GpuArray::empty(&[]);
        // SAFETY: partial holds `parts` values, total one
        unsafe {
            sum_partial_kernel::launch_unchecked::<WgpuRuntime>(
                client(),
                CubeCount::Static(1, 1, 1),
                CubeDim::new_1d(THREADS),
                ArrayArg::from_raw_parts(partial.handle.clone(), parts as usize),
                ArrayArg::from_raw_parts(total.handle.clone(), 1),
                THREADS,
            )
        };
        total
    }

    /// Sums along `axis` of a matrix (0: down the columns, 1: along the rows), still on the GPU.
    pub fn sum_axis(&self, axis: usize) -> GpuArray {
        assert!(self.shape.len() == 2 && axis < 2, "GpuArray::sum_axis: a matrix and axis 0 or 1");
        let (rows, cols) = (self.shape[0], self.shape[1]);
        let w = width(cols);
        if axis == 1 {
            let out = GpuArray::empty(&[rows]);
            if rows > 0 {
                // SAFETY: x holds rows * cols values, out one per row (one cube each)
                unsafe {
                    row_sum_kernel::launch_unchecked::<WgpuRuntime>(
                        client(),
                        CubeCount::Static(rows as u32, 1, 1),
                        CubeDim::new_1d(THREADS),
                        ArrayArg::from_raw_parts(self.handle.clone(), rows * cols),
                        ArrayArg::from_raw_parts(out.handle.clone(), rows),
                        cols,
                        THREADS,
                    )
                };
            }
            return out;
        }
        let out = GpuArray::empty(&[cols]);
        if rows == 0 || cols == 0 {
            return out;
        }
        // enough row chunks to keep the GPU busy, each a few hundred rows
        let parts = rows.div_ceil(64).clamp(1, 64);
        let chunk = rows.div_ceil(parts);
        let partial = GpuArray::empty(&[parts, cols]);
        // SAFETY: x holds rows * cols values, partial parts * cols
        unsafe {
            column_partial_kernel::launch_unchecked::<WgpuRuntime>(
                client(),
                CubeCount::Static(((cols / w) as u32).div_ceil(THREADS), parts as u32, 1),
                CubeDim::new_1d(THREADS),
                w,
                ArrayArg::from_raw_parts(self.handle.clone(), rows * cols),
                ArrayArg::from_raw_parts(partial.handle.clone(), parts * cols),
                rows,
                cols / w,
                chunk,
            );
            column_total_kernel::launch_unchecked::<WgpuRuntime>(
                client(),
                cubes(cols / w),
                CubeDim::new_1d(THREADS),
                w,
                ArrayArg::from_raw_parts(partial.handle.clone(), parts * cols),
                ArrayArg::from_raw_parts(out.handle.clone(), cols),
                parts,
                cols / w,
            );
        }
        out
    }

    /// A causal FIR with `taps` along the last axis (each lane starts from silence).
    pub fn fir(&self, taps: &[f32]) -> GpuArray {
        let len = *self.shape.last().unwrap_or(&1);
        let lanes = self.len() / len.max(1);
        let out = GpuArray::empty(&self.shape);
        if self.is_empty() || taps.is_empty() {
            return out;
        }
        let taps_gpu = client().create(Bytes::from_elems(taps.to_vec()));
        // SAFETY: x and out hold lanes * len values, the taps taps.len()
        unsafe {
            fir_kernel::launch_unchecked::<WgpuRuntime>(
                client(),
                CubeCount::Static((len as u32).div_ceil(THREADS), lanes as u32, 1),
                CubeDim::new_1d(THREADS),
                ArrayArg::from_raw_parts(self.handle.clone(), lanes * len),
                ArrayArg::from_raw_parts(taps_gpu, taps.len()),
                ArrayArg::from_raw_parts(out.handle.clone(), lanes * len),
                len,
            )
        };
        out
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Whether a GPU adapter is there (the tests pass without one).
    fn gpu() -> bool {
        static AVAILABLE: OnceLock<bool> = OnceLock::new();
        *AVAILABLE.get_or_init(|| std::panic::catch_unwind(|| client().clone()).is_ok())
    }

    fn data(shape: &[usize], seed: u32) -> NdArray<f32> {
        let n = shape.iter().product();
        let mut s = seed;
        NdArray::from_vec(
            (0..n)
                .map(|_| {
                    s = s.wrapping_mul(1_664_525).wrapping_add(1_013_904_223);
                    (s >> 8) as f32 / (1u32 << 24) as f32 * 2.0 - 1.0
                })
                .collect(),
            shape,
        )
        .unwrap()
    }

    fn close(got: &NdArray<f32>, want: &[f32], tol: f32, what: &str) {
        assert_eq!(got.len(), want.len(), "{what}: length");
        for (i, (a, b)) in got.as_slice().iter().zip(want).enumerate() {
            assert!((a - b).abs() <= tol * (1.0 + b.abs()), "{what}: element {i}: {a} vs {b}");
        }
    }

    #[test]
    fn element_wise_matches_the_cpu() {
        if !gpu() {
            return;
        }
        for shape in [vec![7], vec![3, 4], vec![5, 7], vec![64, 33]] {
            let (x, y) = (data(&shape, 1), data(&shape, 2));
            let (gx, gy) = (GpuArray::from_host(&x.view()), GpuArray::from_host(&y.view()));
            let xs = x.as_slice();
            close(&gx.axpb(2.0, 0.5).to_host(), &xs.iter().map(|v| 2.0 * v + 0.5).collect::<Vec<_>>(), 1e-6, "axpb");
            let ys = y.as_slice();
            for (name, got, f) in [
                ("add", gx.add(&gy), (|a: f32, b: f32| a + b) as fn(f32, f32) -> f32),
                ("sub", gx.sub(&gy), |a, b| a - b),
                ("mul", gx.mul(&gy), |a, b| a * b),
                ("div", gx.div(&gy), |a, b| a / b),
            ] {
                close(&got.to_host(), &xs.iter().zip(ys).map(|(&a, &b)| f(a, b)).collect::<Vec<_>>(), 1e-5, name);
            }
            for (name, got, f) in [
                ("exp", gx.exp(), (|a: f32| a.exp()) as fn(f32) -> f32),
                ("tanh", gx.tanh(), |a| a.tanh()),
                ("sin", gx.sin(), |a| a.sin()),
                ("cos", gx.cos(), |a| a.cos()),
                ("abs", gx.abs(), |a| a.abs()),
                ("neg", gx.neg(), |a| -a),
                ("sqrt", gx.abs().sqrt(), |a| a.abs().sqrt()),
                ("ln", gx.abs().ln(), |a| a.abs().ln()),
            ] {
                close(&got.to_host(), &xs.iter().map(|&a| f(a)).collect::<Vec<_>>(), 1e-4, name);
            }
            if shape.len() == 2 {
                let (rows, cols) = (shape[0], shape[1]);
                let row = data(&[cols], 3);
                let col = data(&[rows, 1], 4);
                let got = gx.mul(&GpuArray::from_host(&row.view())).to_host();
                close(&got, &(0..rows * cols).map(|k| xs[k] * row.as_slice()[k % cols]).collect::<Vec<_>>(), 1e-6, "row broadcast");
                let got = gx.sub(&GpuArray::from_host(&col.view())).to_host();
                close(&got, &(0..rows * cols).map(|k| xs[k] - col.as_slice()[k / cols]).collect::<Vec<_>>(), 1e-6, "column broadcast");
            }
        }
    }

    #[test]
    fn reductions_match_the_cpu() {
        if !gpu() {
            return;
        }
        for n in [1, 5, 1000, 1023, 300_000, 4_000_001] {
            let x = data(&[n], 5);
            let want: f64 = x.as_slice().iter().map(|&v| v as f64).sum();
            let got = GpuArray::from_host(&x.view()).sum().to_host().as_slice()[0] as f64;
            assert!((got - want).abs() <= 1e-4 * (n as f64).sqrt().max(1.0), "sum of {n}: {got} vs {want}");
        }
        for (rows, cols) in [(1, 1), (3, 5), (100, 64), (2000, 2000), (513, 7)] {
            let x = data(&[rows, cols], 6);
            let g = GpuArray::from_host(&x.view());
            let xs = x.as_slice();
            let row_sums: Vec<f32> = (0..rows).map(|i| xs[i * cols..(i + 1) * cols].iter().sum()).collect();
            let col_sums: Vec<f32> = (0..cols).map(|j| (0..rows).map(|i| xs[i * cols + j]).sum()).collect();
            close(&g.sum_axis(1).to_host(), &row_sums, 1e-4, "row sums");
            close(&g.sum_axis(0).to_host(), &col_sums, 1e-4, "column sums");
        }
    }

    #[test]
    fn fir_matches_the_cpu() {
        if !gpu() {
            return;
        }
        let (lanes, len) = (3, 1000);
        let x = data(&[lanes, len], 7);
        let taps = crate::filter::design_lowpass(2_000.0f32, 48_000.0, 63);
        let got = GpuArray::from_host(&x.view()).fir(&taps).to_host();
        let mut want = Vec::new();
        for l in 0..lanes {
            let mut f = crate::filter::Fir::new(taps.clone());
            let mut lane = x.as_slice()[l * len..(l + 1) * len].to_vec();
            f.process(&mut lane);
            want.extend(lane);
        }
        close(&got, &want, 1e-5, "fir");
    }
}