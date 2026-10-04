//! Arrays on the GPU (feature `gpu`): [`GpuArray`] keeps `f32` or `f64` data in device memory
//! between operations, so a chain of element-wise work, reductions and filtering never crosses the
//! bus until [`GpuArray::to_host`]. Kernels are CubeCL's, run through wgpu (Vulkan, Metal,
//! DirectX 12).
//!
//! Like `NdView` on the CPU, a `GpuArray` is a view: [`transpose`](GpuArray::transpose) and
//! [`permute`](GpuArray::permute) only reorder its axes, element-wise work runs in memory order and
//! keeps the layout (as NumPy keeps an input's memory order), and data is rearranged only where a
//! kernel needs it ([`contiguous`](GpuArray::contiguous)).
//!
//! ```no_run
//! use autodyne::gpu::GpuArray;
//! use autodyne::signal::NdArray;
//!
//! let x = NdArray::from_vec((0..12).map(|i| i as f32).collect(), &[3, 4]).unwrap();
//! let g = GpuArray::from_host(&x.view()).unwrap();
//! let y = g.transpose().axpb(2.0, 0.5).tanh();
//! let rows = y.sum_axis(1);
//! assert_eq!(rows.to_host().shape(), [4]);
//! ```

use std::marker::PhantomData;
use std::sync::OnceLock;

use cubecl::bytes::Bytes;
use cubecl::prelude::*;
use cubecl::server::Handle;
use cubecl::wgpu::{WgpuDevice, WgpuRuntime};
use thiserror::Error;

use crate::signal::{NdArray, NdView};

type Client = ComputeClient<WgpuRuntime>;

/// Errors making GPU arrays.
#[derive(Debug, Clone, PartialEq, Eq, Error)]
pub enum GpuError {
    /// No GPU adapter was found.
    #[error("no GPU adapter is available")]
    NoDevice,
    /// The GPU has no arithmetic in this element type (its name).
    #[error("this GPU does not compute in {0}")]
    Unsupported(&'static str),
}

/// The default GPU's compute client, made on first use (`None` without an adapter).
fn device() -> Option<&'static Client> {
    static CLIENT: OnceLock<Option<Client>> = OnceLock::new();
    CLIENT
        .get_or_init(|| {
            // wgpu panics when it finds no adapter: that means no GPU here
            let hook = std::panic::take_hook();
            std::panic::set_hook(Box::new(|_| {}));
            let client = std::panic::catch_unwind(|| WgpuRuntime::client(&WgpuDevice::default())).ok();
            std::panic::set_hook(hook);
            client
        })
        .as_ref()
}

fn client() -> &'static Client {
    device().expect("a GpuArray exists only once a GPU was found")
}

/// Whether a GPU adapter is available.
pub fn available() -> bool {
    device().is_some()
}

/// Whether the GPU computes in `T` (`f32` everywhere; `f64` where the device has it, e.g. Vulkan
/// on most desktop GPUs).
pub fn supports<T: GpuFloat>() -> bool {
    device().is_some_and(|c| c.properties().supports_type(T::as_type_native_unchecked()))
}

/// Waits until the GPU has finished all submitted work (nothing to wait for without a GPU).
pub fn sync() {
    if let Some(c) = device() {
        cubecl::future::block_on(c.sync()).expect("the GPU finishes its work");
    }
}

mod sealed {
    pub trait Sealed {}
    impl Sealed for f32 {}
    impl Sealed for f64 {}
}

/// Element types a [`GpuArray`] holds: `f32` and `f64`.
pub trait GpuFloat: Float + CubeElement + Copy + Default + Send + Sync + sealed::Sealed + 'static {
    /// The element type's name ("f32", "f64").
    const NAME: &'static str;
}
impl GpuFloat for f32 {
    const NAME: &'static str = "f32";
}
impl GpuFloat for f64 {
    const NAME: &'static str = "f64";
}

const THREADS: u32 = 256;

// KERNELS =========================================================================================

/// Binary operations, chosen at kernel compile time.
const ADD: u32 = 0;
const SUB: u32 = 1;
const MUL: u32 = 2;
const DIV: u32 = 3;

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
fn axpb_kernel<F: Float + CubeElement, N: Size>(x: &Array<Vector<F, N>>, out: &mut Array<Vector<F, N>>, a: F, b: F) {
    if ABSOLUTE_POS < x.len() {
        out[ABSOLUTE_POS] = x[ABSOLUTE_POS] * Vector::new(a) + Vector::new(b);
    }
}

/// `x op s`, or `s op x` when `scalar_first`.
#[cube(launch_unchecked)]
fn scalar_kernel<F: Float + CubeElement, N: Size>(x: &Array<Vector<F, N>>, out: &mut Array<Vector<F, N>>, s: F, #[comptime] op: u32, #[comptime] scalar_first: bool) {
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
fn binary_kernel<F: Float, N: Size>(x: &Array<Vector<F, N>>, y: &Array<Vector<F, N>>, out: &mut Array<Vector<F, N>>, #[comptime] op: u32) {
    if ABSOLUTE_POS < x.len() {
        out[ABSOLUTE_POS] = apply::<F, N>(x[ABSOLUTE_POS], y[ABSOLUTE_POS], op);
    }
}

/// `x` (rows x cols in memory, cols in vectors) with a vector along its rows (cols, in vectors).
#[cube(launch_unchecked)]
fn row_kernel<F: Float, N: Size>(x: &Array<Vector<F, N>>, row: &Array<Vector<F, N>>, out: &mut Array<Vector<F, N>>, cols: usize, #[comptime] op: u32) {
    if ABSOLUTE_POS < x.len() {
        out[ABSOLUTE_POS] = apply::<F, N>(x[ABSOLUTE_POS], row[ABSOLUTE_POS % cols], op);
    }
}

/// `x` (rows x cols in memory, cols in vectors) with a vector down its columns (rows).
#[cube(launch_unchecked)]
fn column_kernel<F: Float, N: Size>(x: &Array<Vector<F, N>>, column: &Array<F>, out: &mut Array<Vector<F, N>>, cols: usize, #[comptime] op: u32) {
    if ABSOLUTE_POS < x.len() {
        out[ABSOLUTE_POS] = apply::<F, N>(x[ABSOLUTE_POS], Vector::new(column[ABSOLUTE_POS / cols]), op);
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
fn unary_kernel<F: Float, N: Size>(x: &Array<Vector<F, N>>, out: &mut Array<Vector<F, N>>, #[comptime] op: u32) {
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

/// Double-precision transcendentals. Shader languages have `exp`, `log`, `sin`, `cos` and `tanh`
/// for 16- and 32-bit floats only (SPIR-V's GLSL.std.450, HLSL, WGSL), so for `f64` they are built
/// here from arithmetic, `round`, `fma` and comparisons, which every f64-capable GPU has: Cody-Waite
/// range reduction (fused, so the compiler cannot fold the split constants back together), Taylor
/// or atanh series to below an ulp, then an exact power of two. Arguments of sin and cos are
/// reduced with a three-part pi/2, accurate for |x| up to about 1e6. Within 4 ulps of the CPU.
// full-precision constants and NaN idioms such as `x != x` are the point here
#[allow(clippy::excessive_precision, clippy::approx_constant, clippy::eq_op, clippy::manual_clamp, clippy::if_same_then_else)]
mod wide {
    use super::*;

    /// `2^k`, exact (repeated squaring of 2 or 1/2), for |k| up to about 1100.
    #[cube]
    fn pow2_f64(k: i32) -> f64 {
        let mut base = 2.0f64;
        let mut n = u32::cast_from(k);
        if k < 0 {
            base = 0.5f64;
            n = u32::cast_from(-k);
        }
        let mut r = 1.0f64;
        while n > 0 {
            if (n & 1) == 1 {
                r *= base;
            }
            base *= base;
            n >>= 1;
        }
        r
    }

    /// `e^r - 1` for |r| <= 0.35 (Taylor to r^14).
    #[cube]
    fn expm1_small_f64(r: f64) -> f64 {
        let mut p = 1.1470745597729725e-11f64; // 1/14!
        p = p * r + 1.6059043836821613e-10f64;
        p = p * r + 2.08767569878681e-9f64;
        p = p * r + 2.505210838544172e-8f64;
        p = p * r + 2.755731922398589e-7f64;
        p = p * r + 2.7557319223985893e-6f64;
        p = p * r + 2.48015873015873e-5f64;
        p = p * r + 1.984126984126984e-4f64;
        p = p * r + 1.3888888888888889e-3f64;
        p = p * r + 8.333333333333333e-3f64;
        p = p * r + 4.1666666666666664e-2f64;
        p = p * r + 0.16666666666666666f64;
        p = p * r + 0.5f64;
        p * r * r + r
    }

    #[cube]
    fn exp_f64(x: f64) -> f64 {
        // beyond +-800 e^x is 0 or infinite anyway
        let mut y = x;
        if y > 800.0f64 {
            y = 800.0f64;
        }
        if y < -800.0f64 {
            y = -800.0f64;
        }
        let q = (y * 1.4426950408889634f64).round();
        // fused steps: plain ones may be re-associated into one product with a rounded ln 2
        let r = fma(-q, 1.90821492927058770002e-10f64, fma(-q, 6.93147180369123816490e-1f64, y));
        let mut k = i32::cast_from(q);
        // (NaN casts to anything)
        if k > 1100 {
            k = 1100;
        }
        if k < -1100 {
            k = -1100;
        }
        // split the power so neither factor overflows before the product rounds
        let half = k / 2;
        (1.0f64 + expm1_small_f64(r)) * pow2_f64(half) * pow2_f64(k - half)
    }

    #[cube]
    fn ln_f64(x: f64) -> f64 {
        let mut m = x;
        let mut k = 0i32;
        // bring m within f32's range (2^-100 .. 2^100) by exact steps
        while m > 1.2676506002282294e30f64 && k < 1100 {
            m *= 7.888609052210118e-31f64;
            k += 100;
        }
        while m < 7.888609052210118e-31f64 && m > 0.0f64 && k > -1200 {
            m *= 1.2676506002282294e30f64;
            k -= 100;
        }
        // the exponent from an f32 logarithm (off by one at most: the series takes either side)
        let e = i32::cast_from((f32::cast_from(m).ln() * 1.442695f32).round());
        m *= pow2_f64(-e);
        k += e;
        // ln m = 2 atanh(s), s = (m - 1) / (m + 1), |s| < 0.18
        let s = (m - 1.0f64) / (m + 1.0f64);
        let z = s * s;
        let mut p = 0.043478260869565216f64;
        p = p * z + 0.047619047619047616f64;
        p = p * z + 0.05263157894736842f64;
        p = p * z + 0.058823529411764705f64;
        p = p * z + 0.06666666666666667f64;
        p = p * z + 0.07692307692307693f64;
        p = p * z + 0.09090909090909091f64;
        p = p * z + 0.1111111111111111f64;
        p = p * z + 0.14285714285714285f64;
        p = p * z + 0.2f64;
        p = p * z + 0.3333333333333333f64;
        let kf = f64::cast_from(k);
        let small = fma(kf, 1.90821492927058770002e-10f64, 2.0f64 * s * z * p);
        let mut out = fma(kf, 6.93147180369123816490e-1f64, 2.0f64 * s + small);
        if x == 0.0f64 {
            out = -1.0f64 / x;
        } else if x < 0.0f64 {
            out = (x - x) / (x - x);
        } else if x > 1.7976931348623157e308f64 {
            out = x;
        } else if x != x {
            out = x;
        }
        out
    }

    #[cube]
    fn tanh_f64(x: f64) -> f64 {
        let a = x.abs();
        // tanh a = e / (e + 2) with e = e^(2a) - 1, exact near 0 through expm1
        let y = 2.0f64 * a;
        let mut e = exp_f64(y) - 1.0f64;
        if y < 0.35f64 {
            e = expm1_small_f64(y);
        }
        let mut t = e / (e + 2.0f64);
        if a > 22.0f64 {
            t = 1.0f64;
        }
        if x < 0.0f64 {
            t = -t;
        }
        t
    }

    /// sin x (`cosine` false) or cos x.
    #[cube]
    fn sin_cos_f64(x: f64, #[comptime] cosine: bool) -> f64 {
        let q = (x * 0.6366197723675814f64).round();
        let r = fma(-q, 2.02226624879595063154e-21f64, fma(-q, 6.07710050633881403649e-11f64, fma(-q, 1.57079632673412561417e0f64, x)));
        let z = r * r;
        // sin r to r^17, cos r to r^16, on |r| <= pi/4
        let mut s = 2.8114572543455206e-15f64;
        s = s * z - 7.647163731819816e-13f64;
        s = s * z + 1.6059043836821613e-10f64;
        s = s * z - 2.505210838544172e-8f64;
        s = s * z + 2.7557319223985893e-6f64;
        s = s * z - 1.984126984126984e-4f64;
        s = s * z + 8.333333333333333e-3f64;
        s = s * z - 0.16666666666666666f64;
        let sin_r = r + r * z * s;
        let mut c = 4.779477332387385e-14f64;
        c = c * z - 1.1470745597729725e-11f64;
        c = c * z + 2.08767569878681e-9f64;
        c = c * z - 2.755731922398589e-7f64;
        c = c * z + 2.48015873015873e-5f64;
        c = c * z - 1.3888888888888889e-3f64;
        c = c * z + 4.1666666666666664e-2f64;
        c = c * z - 0.5f64;
        let cos_r = 1.0f64 + z * c;
        let mut quadrant = i32::cast_from(q);
        if cosine {
            quadrant += 1;
        }
        quadrant &= 3;
        let mut out = sin_r;
        if quadrant == 1 {
            out = cos_r;
        } else if quadrant == 2 {
            out = -sin_r;
        } else if quadrant == 3 {
            out = -cos_r;
        }
        out
    }

    #[cube(launch_unchecked)]
    pub(super) fn transcendental_f64_kernel(x: &Array<f64>, out: &mut Array<f64>, #[comptime] op: u32) {
        if ABSOLUTE_POS < x.len() {
            let v = x[ABSOLUTE_POS];
            let mut r = v;
            if op == EXP {
                r = exp_f64(v);
            } else if op == LN {
                r = ln_f64(v);
            } else if op == TANH {
                r = tanh_f64(v);
            } else if op == SIN {
                r = sin_cos_f64(v, false);
            } else if op == COS {
                r = sin_cos_f64(v, true);
            }
            out[ABSOLUTE_POS] = r;
        }
    }
}

/// Row-major `out` of `shape`, element `i` read from `x` at the `strides`-weighted index: any
/// permutation of a layout.
#[cube(launch_unchecked)]
fn gather_kernel<F: Float>(x: &Array<F>, out: &mut Array<F>, shape: &Array<u32>, strides: &Array<u32>) {
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
fn sum_partial_kernel<F: Float>(x: &Array<F>, partial: &mut Array<F>, #[comptime] threads: u32) {
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
fn row_sum_kernel<F: Float>(x: &Array<F>, out: &mut Array<F>, cols: usize, #[comptime] threads: u32) {
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
fn column_partial_kernel<F: Float, N: Size>(x: &Array<Vector<F, N>>, partial: &mut Array<Vector<F, N>>, rows: usize, cols: usize, chunk: usize) {
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
fn column_total_kernel<F: Float, N: Size>(partial: &Array<Vector<F, N>>, out: &mut Array<Vector<F, N>>, parts: usize, cols: usize) {
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
fn fir_kernel<F: Float>(x: &Array<F>, taps: &Array<F>, out: &mut Array<F>, len: usize) {
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

// ARRAYS ==========================================================================================

/// `f32` / `f64` values in GPU memory: a row-major buffer of shape `memory`, seen through an axis
/// order (axis `j` of the array is axis `perm[j]` of the buffer).
pub struct GpuArray<T: GpuFloat = f32> {
    handle: Handle,
    memory: Vec<usize>,
    perm: Vec<usize>,
    _elem: PhantomData<T>,
}

impl<T: GpuFloat> Clone for GpuArray<T> {
    /// Another view of the same memory (no copy).
    fn clone(&self) -> Self {
        GpuArray { handle: self.handle.clone(), memory: self.memory.clone(), perm: self.perm.clone(), _elem: PhantomData }
    }
}

impl<T: GpuFloat> std::fmt::Debug for GpuArray<T> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("GpuArray").field("dtype", &T::NAME).field("shape", &self.shape()).field("perm", &self.perm).finish()
    }
}

enum Pairing {
    Same,
    AlongRows,
    DownColumns,
}

/// Vectors of 4 when `n` allows, else single values.
fn width(n: usize) -> usize {
    if n.is_multiple_of(4) { 4 } else { 1 }
}

fn cubes(units: usize) -> CubeCount {
    CubeCount::Static((units as u32).div_ceil(THREADS).max(1), 1, 1)
}

/// Row-major strides of `shape`.
fn row_major(shape: &[usize]) -> Vec<usize> {
    let mut s = vec![1; shape.len()];
    for i in (0..shape.len().saturating_sub(1)).rev() {
        s[i] = s[i + 1] * shape[i + 1];
    }
    s
}

fn identity(n: usize) -> Vec<usize> {
    (0..n).collect()
}

impl<T: GpuFloat> GpuArray<T> {
    fn check() -> Result<(), GpuError> {
        if !available() {
            return Err(GpuError::NoDevice);
        }
        if !supports::<T>() {
            return Err(GpuError::Unsupported(T::NAME));
        }
        Ok(())
    }

    fn new(handle: Handle, memory: Vec<usize>, perm: Vec<usize>) -> GpuArray<T> {
        GpuArray { handle, memory, perm, _elem: PhantomData }
    }

    /// Copies `x` to the GPU. A permuted contiguous view (a transpose, say) goes up as it lies in
    /// memory and stays the same view; anything else is gathered first.
    pub fn from_host(x: &NdView<'_, T>) -> Result<GpuArray<T>, GpuError> {
        Self::check()?;
        let n = x.ndim();
        let order = x.memory_order();
        let order = &order[..n];
        let in_order = x.permute(order).expect("a permutation");
        let (data, memory, perm) = match in_order.as_slice() {
            // buffer axis i holds array axis order[i]
            Some(s) => {
                let mut perm = vec![0; n];
                for (i, &a) in order.iter().enumerate() {
                    perm[a] = i;
                }
                (s.to_vec(), in_order.shape().to_vec(), perm)
            }
            None => (x.to_vec(), x.shape().to_vec(), identity(n)),
        };
        Ok(GpuArray::new(client().create(Bytes::from_elems(data)), memory, perm))
    }

    /// Moves `x` to the GPU (its buffer handed over, not copied, when it is contiguous).
    pub fn from_array(x: NdArray<T>) -> Result<GpuArray<T>, GpuError> {
        if !x.view().is_contiguous() {
            return GpuArray::from_host(&x.view());
        }
        Self::check()?;
        let memory = x.shape().to_vec();
        let perm = identity(memory.len());
        Ok(GpuArray::new(client().create(Bytes::from_elems(x.into_vec())), memory, perm))
    }

    /// Copies the values back, contiguous in this array's axis order.
    pub fn to_host(&self) -> NdArray<T> {
        let bytes = client().read_one(self.handle.clone()).expect("the GPU returns the array");
        let mut data = match bytes.try_into_vec::<T>() {
            Ok(v) => v,
            Err(bytes) => T::from_bytes(&bytes).to_vec(),
        };
        data.truncate(self.len());
        let buffer = NdArray::from_vec(data, &self.memory).expect("the buffer's shape");
        if self.is_standard() {
            return buffer;
        }
        buffer.view().permute(&self.perm).expect("a permutation").to_owned()
    }

    /// Length of each axis (of this view).
    pub fn shape(&self) -> Vec<usize> {
        self.perm.iter().map(|&p| self.memory[p]).collect()
    }
    /// Number of axes.
    pub fn ndim(&self) -> usize {
        self.memory.len()
    }
    /// Number of elements.
    pub fn len(&self) -> usize {
        self.memory.iter().product()
    }
    /// Whether there are no elements.
    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }
    /// Whether the array is its buffer in row-major order (not a permuted view).
    pub fn is_standard(&self) -> bool {
        self.perm.iter().enumerate().all(|(i, &p)| i == p)
    }

    /// The axes in reverse order (a matrix transposed): a view, nothing moves.
    pub fn transpose(&self) -> GpuArray<T> {
        let axes: Vec<usize> = (0..self.ndim()).rev().collect();
        self.permute(&axes)
    }

    /// Axis `j` of the result is axis `axes[j]` of this array (`numpy.transpose`): a view.
    pub fn permute(&self, axes: &[usize]) -> GpuArray<T> {
        let mut seen = vec![false; self.ndim()];
        let valid = axes.len() == self.ndim() && axes.iter().all(|&a| a < self.ndim() && !std::mem::replace(&mut seen[a], true));
        assert!(valid, "GpuArray::permute: {axes:?} is not a permutation of {} axes", self.ndim());
        GpuArray::new(self.handle.clone(), self.memory.clone(), axes.iter().map(|&a| self.perm[a]).collect())
    }

    fn empty(memory: &[usize], perm: Vec<usize>) -> GpuArray<T> {
        let len: usize = memory.iter().product();
        GpuArray::new(client().empty(len.max(1) * size_of::<T>()), memory.to_vec(), perm)
    }

    /// A new buffer of row-major `shape`, element `i` read from this buffer at the
    /// `strides`-weighted index of `i`.
    fn gathered(&self, shape: &[usize], strides: &[usize]) -> Handle {
        let len: usize = shape.iter().product();
        let out = client().empty(len.max(1) * size_of::<T>());
        if len > 0 {
            let shape_u: Vec<u32> = shape.iter().map(|&s| s as u32).collect();
            let strides_u: Vec<u32> = strides.iter().map(|&s| s as u32).collect();
            let (sh, st) = (client().create(Bytes::from_elems(shape_u)), client().create(Bytes::from_elems(strides_u)));
            // SAFETY: every index read is inside the buffer (strides of a permutation of its layout)
            unsafe {
                gather_kernel::launch_unchecked::<T, WgpuRuntime>(
                    client(),
                    cubes(len),
                    CubeDim::new_1d(THREADS),
                    ArrayArg::from_raw_parts(self.handle.clone(), self.len()),
                    ArrayArg::from_raw_parts(out.clone(), len),
                    ArrayArg::from_raw_parts(sh, shape.len()),
                    ArrayArg::from_raw_parts(st, shape.len()),
                )
            };
        }
        out
    }

    /// The same values with the buffer in this array's own row-major order (a copy only for a
    /// permuted view).
    pub fn contiguous(&self) -> GpuArray<T> {
        if self.is_standard() {
            return self.clone();
        }
        let shape = self.shape();
        let mstrides = row_major(&self.memory);
        let strides: Vec<usize> = self.perm.iter().map(|&p| mstrides[p]).collect();
        let n = shape.len();
        GpuArray::new(self.gathered(&shape, &strides), shape, identity(n))
    }

    /// `other`'s values (same shape) in a buffer laid out like this one's.
    fn laid_out_like(&self, other: &GpuArray<T>) -> Handle {
        if other.memory == self.memory && other.perm == self.perm {
            return other.handle.clone();
        }
        // buffer axis i here is array axis inv[i]; read that axis from other's buffer
        let mut inv = vec![0; self.ndim()];
        for (j, &p) in self.perm.iter().enumerate() {
            inv[p] = j;
        }
        let ostrides = row_major(&other.memory);
        let strides: Vec<usize> = inv.iter().map(|&j| ostrides[other.perm[j]]).collect();
        other.gathered(&self.memory, &strides)
    }

    /// `a * x + b`, one pass.
    pub fn axpb(&self, a: T, b: T) -> GpuArray<T> {
        let out = GpuArray::empty(&self.memory, self.perm.clone());
        let (n, w) = (self.len(), width(self.len()));
        if n > 0 {
            // SAFETY: both arrays hold n values; the launch covers n / w vectors
            unsafe {
                axpb_kernel::launch_unchecked::<T, WgpuRuntime>(client(), cubes(n / w), CubeDim::new_1d(THREADS), w, ArrayArg::from_raw_parts(self.handle.clone(), n), ArrayArg::from_raw_parts(out.handle.clone(), n), a, b)
            };
        }
        out
    }

    /// How `other` lines up with this array element by element (`None`: it doesn't).
    fn pairing(&self, other: &GpuArray<T>) -> Option<Pairing> {
        let (shape, oshape) = (self.shape(), other.shape());
        if oshape == shape {
            return Some(Pairing::Same);
        }
        if shape.len() != 2 {
            return None;
        }
        let (rows, cols) = (shape[0], shape[1]);
        if oshape.as_slice() == [cols] || oshape.as_slice() == [1, cols] {
            Some(Pairing::AlongRows)
        } else if oshape.as_slice() == [rows, 1] {
            Some(Pairing::DownColumns)
        } else {
            None
        }
    }

    /// Whether [`add`](Self::add) and the other element-wise operations take `other`: the same
    /// shape, or a row or a column of this matrix.
    pub fn can_combine(&self, other: &GpuArray<T>) -> bool {
        self.pairing(other).is_some()
    }

    fn binary(&self, other: &GpuArray<T>, op: u32) -> GpuArray<T> {
        let out = GpuArray::empty(&self.memory, self.perm.clone());
        let n = self.len();
        if n == 0 {
            return out;
        }
        let (shape, oshape) = (self.shape(), other.shape());
        if oshape == shape {
            let w = width(n);
            let y = self.laid_out_like(other);
            // SAFETY: all three hold n values in one layout
            unsafe {
                binary_kernel::launch_unchecked::<T, WgpuRuntime>(
                    client(),
                    cubes(n / w),
                    CubeDim::new_1d(THREADS),
                    w,
                    ArrayArg::from_raw_parts(self.handle.clone(), n),
                    ArrayArg::from_raw_parts(y, n),
                    ArrayArg::from_raw_parts(out.handle.clone(), n),
                    op,
                )
            };
            return out;
        }
        let along_rows = match self.pairing(other) {
            Some(Pairing::AlongRows) => true,
            Some(_) => false,
            None => panic!("GpuArray: cannot combine shapes {shape:?} and {oshape:?} (the same shape, or a row or a column of a matrix)"),
        };
        let vector = other.contiguous();
        // in a transposed buffer, the array's rows are the buffer's columns
        let mcols = self.memory[1];
        let w = width(mcols);
        if along_rows == self.is_standard() {
            // SAFETY: x and out hold n values, the vector one per buffer column
            unsafe {
                row_kernel::launch_unchecked::<T, WgpuRuntime>(
                    client(),
                    cubes(n / w),
                    CubeDim::new_1d(THREADS),
                    w,
                    ArrayArg::from_raw_parts(self.handle.clone(), n),
                    ArrayArg::from_raw_parts(vector.handle.clone(), mcols),
                    ArrayArg::from_raw_parts(out.handle.clone(), n),
                    mcols / w,
                    op,
                )
            };
        } else {
            // SAFETY: x and out hold n values, the vector one per buffer row
            unsafe {
                column_kernel::launch_unchecked::<T, WgpuRuntime>(
                    client(),
                    cubes(n / w),
                    CubeDim::new_1d(THREADS),
                    w,
                    ArrayArg::from_raw_parts(self.handle.clone(), n),
                    ArrayArg::from_raw_parts(vector.handle.clone(), self.memory[0]),
                    ArrayArg::from_raw_parts(out.handle.clone(), n),
                    mcols / w,
                    op,
                )
            };
        }
        out
    }

    /// Element by element. `other` has this shape (in any layout), or is a row (`[cols]`,
    /// `[1, cols]`) or a column (`[rows, 1]`) broadcast across a matrix. The result keeps this
    /// array's layout.
    pub fn add(&self, other: &GpuArray<T>) -> GpuArray<T> {
        self.binary(other, ADD)
    }
    /// `self - other` (see [`add`](Self::add) for the shapes allowed).
    pub fn sub(&self, other: &GpuArray<T>) -> GpuArray<T> {
        self.binary(other, SUB)
    }
    /// `self * other` (see [`add`](Self::add)).
    pub fn mul(&self, other: &GpuArray<T>) -> GpuArray<T> {
        self.binary(other, MUL)
    }
    /// `self / other` (see [`add`](Self::add)).
    pub fn div(&self, other: &GpuArray<T>) -> GpuArray<T> {
        self.binary(other, DIV)
    }

    fn scalar(&self, s: T, op: u32, scalar_first: bool) -> GpuArray<T> {
        let out = GpuArray::empty(&self.memory, self.perm.clone());
        let (n, w) = (self.len(), width(self.len()));
        if n > 0 {
            // SAFETY: both hold n values
            unsafe {
                scalar_kernel::launch_unchecked::<T, WgpuRuntime>(
                    client(),
                    cubes(n / w),
                    CubeDim::new_1d(THREADS),
                    w,
                    ArrayArg::from_raw_parts(self.handle.clone(), n),
                    ArrayArg::from_raw_parts(out.handle.clone(), n),
                    s,
                    op,
                    scalar_first,
                )
            };
        }
        out
    }
    /// `x + s`
    pub fn add_scalar(&self, s: T) -> GpuArray<T> {
        self.scalar(s, ADD, false)
    }
    /// `x - s`
    pub fn sub_scalar(&self, s: T) -> GpuArray<T> {
        self.scalar(s, SUB, false)
    }
    /// `x * s`
    pub fn mul_scalar(&self, s: T) -> GpuArray<T> {
        self.scalar(s, MUL, false)
    }
    /// `x / s`
    pub fn div_scalar(&self, s: T) -> GpuArray<T> {
        self.scalar(s, DIV, false)
    }
    /// `s - x`
    pub fn scalar_sub(&self, s: T) -> GpuArray<T> {
        self.scalar(s, SUB, true)
    }
    /// `s / x`
    pub fn scalar_div(&self, s: T) -> GpuArray<T> {
        self.scalar(s, DIV, true)
    }

    fn unary(&self, op: u32) -> GpuArray<T> {
        let out = GpuArray::empty(&self.memory, self.perm.clone());
        let (n, w) = (self.len(), width(self.len()));
        let own = matches!(op, EXP | LN | TANH | SIN | COS) && std::any::TypeId::of::<T>() == std::any::TypeId::of::<f64>();
        if n > 0 && own {
            // SAFETY: both hold n values (f64, checked above)
            unsafe {
                wide::transcendental_f64_kernel::launch_unchecked::<WgpuRuntime>(client(), cubes(n), CubeDim::new_1d(THREADS), ArrayArg::from_raw_parts(self.handle.clone(), n), ArrayArg::from_raw_parts(out.handle.clone(), n), op)
            };
        } else if n > 0 {
            // SAFETY: both hold n values
            unsafe {
                unary_kernel::launch_unchecked::<T, WgpuRuntime>(client(), cubes(n / w), CubeDim::new_1d(THREADS), w, ArrayArg::from_raw_parts(self.handle.clone(), n), ArrayArg::from_raw_parts(out.handle.clone(), n), op)
            };
        }
        out
    }
    /// e to the power of each element.
    pub fn exp(&self) -> GpuArray<T> {
        self.unary(EXP)
    }
    /// Natural logarithm of each element.
    pub fn ln(&self) -> GpuArray<T> {
        self.unary(LN)
    }
    /// Hyperbolic tangent of each element.
    pub fn tanh(&self) -> GpuArray<T> {
        self.unary(TANH)
    }
    /// Sine of each element.
    pub fn sin(&self) -> GpuArray<T> {
        self.unary(SIN)
    }
    /// Cosine of each element.
    pub fn cos(&self) -> GpuArray<T> {
        self.unary(COS)
    }
    /// Square root of each element.
    pub fn sqrt(&self) -> GpuArray<T> {
        self.unary(SQRT)
    }
    /// Absolute value of each element.
    pub fn abs(&self) -> GpuArray<T> {
        self.unary(ABS)
    }
    /// Each element negated.
    pub fn neg(&self) -> GpuArray<T> {
        self.unary(NEG)
    }

    /// The sum of every element, as a one-element array (still on the GPU).
    pub fn sum(&self) -> GpuArray<T> {
        let n = self.len();
        let parts = (n as u32).div_ceil(THREADS).clamp(1, 512);
        let partial = GpuArray::<T>::empty(&[parts as usize], vec![0]);
        // SAFETY: x holds n values, partial one per cube
        unsafe {
            sum_partial_kernel::launch_unchecked::<T, WgpuRuntime>(
                client(),
                CubeCount::Static(parts, 1, 1),
                CubeDim::new_1d(THREADS),
                ArrayArg::from_raw_parts(self.handle.clone(), n),
                ArrayArg::from_raw_parts(partial.handle.clone(), parts as usize),
                THREADS,
            )
        };
        if parts == 1 {
            return GpuArray::new(partial.handle, vec![], vec![]);
        }
        let total = GpuArray::<T>::empty(&[], vec![]);
        // SAFETY: partial holds `parts` values, total one
        unsafe {
            sum_partial_kernel::launch_unchecked::<T, WgpuRuntime>(
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
    pub fn sum_axis(&self, axis: usize) -> GpuArray<T> {
        assert!(self.ndim() == 2 && axis < 2, "GpuArray::sum_axis: a matrix and axis 0 or 1");
        // summing along the buffer's rows or down its columns, whichever the axis is in memory
        let (rows, cols) = (self.memory[0], self.memory[1]);
        if self.perm[axis] == 1 {
            let out = GpuArray::empty(&[rows], vec![0]);
            if rows > 0 {
                // SAFETY: x holds rows * cols values, out one per row (one cube each)
                unsafe {
                    row_sum_kernel::launch_unchecked::<T, WgpuRuntime>(
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
        let out = GpuArray::empty(&[cols], vec![0]);
        if rows == 0 || cols == 0 {
            return out;
        }
        let w = width(cols);
        // enough row chunks to keep the GPU busy
        let parts = rows.div_ceil(64).clamp(1, 64);
        let chunk = rows.div_ceil(parts);
        let partial = GpuArray::<T>::empty(&[parts, cols], vec![0, 1]);
        // SAFETY: x holds rows * cols values, partial parts * cols
        unsafe {
            column_partial_kernel::launch_unchecked::<T, WgpuRuntime>(
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
            column_total_kernel::launch_unchecked::<T, WgpuRuntime>(
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
    pub fn fir(&self, taps: &[T]) -> GpuArray<T> {
        let x = self.contiguous();
        let len = *x.memory.last().unwrap_or(&1);
        let lanes = x.len() / len.max(1);
        let out = GpuArray::empty(&x.memory, x.perm.clone());
        if x.is_empty() || taps.is_empty() {
            return out;
        }
        let taps_gpu = client().create(Bytes::from_elems(taps.to_vec()));
        // SAFETY: x and out hold lanes * len values, the taps taps.len()
        unsafe {
            fir_kernel::launch_unchecked::<T, WgpuRuntime>(
                client(),
                CubeCount::Static((len as u32).div_ceil(THREADS), lanes as u32, 1),
                CubeDim::new_1d(THREADS),
                ArrayArg::from_raw_parts(x.handle.clone(), lanes * len),
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

    fn up(x: &NdArray<f32>) -> GpuArray<f32> {
        GpuArray::from_host(&x.view()).unwrap()
    }

    fn close(got: &NdArray<f32>, want: &[f32], tol: f32, what: &str) {
        assert_eq!(got.len(), want.len(), "{what}: length");
        for (i, (a, b)) in got.as_slice().iter().zip(want).enumerate() {
            assert!((a - b).abs() <= tol * (1.0 + b.abs()), "{what}: element {i}: {a} vs {b}");
        }
    }

    #[test]
    fn no_gpu_is_an_error() {
        if !available() {
            let x = data(&[4], 0);
            assert_eq!(GpuArray::from_host(&x.view()).unwrap_err(), GpuError::NoDevice);
        }
    }

    #[test]
    fn element_wise_matches_the_cpu() {
        if !available() {
            return;
        }
        for shape in [vec![7], vec![3, 4], vec![5, 7], vec![64, 33]] {
            let (x, y) = (data(&shape, 1), data(&shape, 2));
            let (gx, gy) = (up(&x), up(&y));
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
            for (name, got, f) in [
                ("x + s", gx.add_scalar(1.5), (|a: f32| a + 1.5) as fn(f32) -> f32),
                ("x - s", gx.sub_scalar(1.5), |a| a - 1.5),
                ("x * s", gx.mul_scalar(1.5), |a| a * 1.5),
                ("x / s", gx.div_scalar(3.0), |a| a / 3.0),
                ("s - x", gx.scalar_sub(1.5), |a| 1.5 - a),
                ("s / x", gx.scalar_div(3.0), |a| 3.0 / a),
            ] {
                // (Vulkan's f32 division is good to 2.5 ulps, not correctly rounded)
                close(&got.to_host(), &xs.iter().map(|&a| f(a)).collect::<Vec<_>>(), 1e-6, name);
            }
            assert!(gx.can_combine(&gy) && !gx.can_combine(&up(&data(&[3], 9))) == (shape.last() != Some(&3)));
            if shape.len() == 2 {
                let (rows, cols) = (shape[0], shape[1]);
                let row = data(&[cols], 3);
                let col = data(&[rows, 1], 4);
                close(&gx.mul(&up(&row)).to_host(), &(0..rows * cols).map(|k| xs[k] * row.as_slice()[k % cols]).collect::<Vec<_>>(), 1e-6, "row broadcast");
                close(&gx.sub(&up(&col)).to_host(), &(0..rows * cols).map(|k| xs[k] - col.as_slice()[k / cols]).collect::<Vec<_>>(), 1e-6, "column broadcast");
            }
        }
    }

    #[test]
    fn views_match_the_cpu() {
        if !available() {
            return;
        }
        for (rows, cols) in [(37, 20), (16, 12), (5, 5)] {
            let x = data(&[rows, cols], 11);
            let t = up(&x).transpose();
            assert_eq!(t.shape(), [cols, rows]);
            assert!(!t.is_standard());
            let xt = x.view().transpose().to_owned();
            assert_eq!(t.to_host(), xt);
            let xs = xt.as_slice();
            // element-wise work keeps the view; a transposed host view goes up as it lies
            close(&t.axpb(2.0, 1.0).to_host(), &xs.iter().map(|v| 2.0 * v + 1.0).collect::<Vec<_>>(), 1e-6, "transposed axpb");
            let lifted = GpuArray::from_host(&x.view().transpose()).unwrap();
            assert!(!lifted.is_standard());
            assert_eq!(lifted.to_host(), xt);
            // mixed layouts
            let y = data(&[cols, rows], 12);
            let sum: Vec<f32> = xs.iter().zip(y.as_slice()).map(|(a, b)| a + b).collect();
            close(&t.add(&up(&y)).to_host(), &sum, 1e-6, "transposed + standard");
            close(&up(&y).add(&t).to_host(), &sum, 1e-6, "standard + transposed");
            // broadcasts and sums on the transposed view (cols x rows)
            let row = data(&[rows], 13);
            let col = data(&[cols, 1], 14);
            close(&t.mul(&up(&row)).to_host(), &(0..rows * cols).map(|k| xs[k] * row.as_slice()[k % rows]).collect::<Vec<_>>(), 1e-6, "row on transposed");
            close(&t.sub(&up(&col)).to_host(), &(0..rows * cols).map(|k| xs[k] - col.as_slice()[k / rows]).collect::<Vec<_>>(), 1e-6, "column on transposed");
            close(&t.mul(&up(&row).permute(&[0])).to_host(), &(0..rows * cols).map(|k| xs[k] * row.as_slice()[k % rows]).collect::<Vec<_>>(), 1e-6, "row on transposed again");
            let row_sums: Vec<f32> = (0..cols).map(|i| xs[i * rows..(i + 1) * rows].iter().sum()).collect();
            let col_sums: Vec<f32> = (0..rows).map(|j| (0..cols).map(|i| xs[i * rows + j]).sum()).collect();
            close(&t.sum_axis(1).to_host(), &row_sums, 1e-4, "row sums of transposed");
            close(&t.sum_axis(0).to_host(), &col_sums, 1e-4, "column sums of transposed");
            let c = t.contiguous();
            assert!(c.is_standard());
            assert_eq!(c.to_host(), xt);
            assert_eq!(t.transpose().to_host(), x);
        }
        // permuting a 3-D array
        let z = data(&[2, 3, 4], 15);
        let want = z.view().permute(&[2, 0, 1]).unwrap().to_owned();
        let gz = up(&z).permute(&[2, 0, 1]);
        assert_eq!(gz.shape(), [4, 2, 3]);
        assert_eq!(gz.to_host(), want);
        assert_eq!(gz.contiguous().to_host(), want);
        close(&gz.exp().to_host(), &want.as_slice().iter().map(|v| v.exp()).collect::<Vec<_>>(), 1e-6, "exp of a permuted view");
    }

    #[test]
    fn reductions_match_the_cpu() {
        if !available() {
            return;
        }
        for n in [1, 5, 1000, 1023, 300_000, 4_000_001] {
            let x = data(&[n], 5);
            let want: f64 = x.as_slice().iter().map(|&v| v as f64).sum();
            let got = up(&x).sum().to_host().as_slice()[0] as f64;
            assert!((got - want).abs() <= 1e-4 * (n as f64).sqrt().max(1.0), "sum of {n}: {got} vs {want}");
        }
        for (rows, cols) in [(1, 1), (3, 5), (100, 64), (2000, 2000), (513, 7)] {
            let x = data(&[rows, cols], 6);
            let g = up(&x);
            let xs = x.as_slice();
            let row_sums: Vec<f32> = (0..rows).map(|i| xs[i * cols..(i + 1) * cols].iter().sum()).collect();
            let col_sums: Vec<f32> = (0..cols).map(|j| (0..rows).map(|i| xs[i * cols + j]).sum()).collect();
            close(&g.sum_axis(1).to_host(), &row_sums, 1e-4, "row sums");
            close(&g.sum_axis(0).to_host(), &col_sums, 1e-4, "column sums");
        }
    }

    #[test]
    fn fir_matches_the_cpu() {
        if !available() {
            return;
        }
        let (lanes, len) = (3, 1000);
        let x = data(&[lanes, len], 7);
        let taps = crate::filter::design_lowpass(2_000.0f32, 63, 48_000.0);
        let mut want = Vec::new();
        for l in 0..lanes {
            let mut f = crate::filter::Fir::new(taps.clone());
            let mut lane = x.as_slice()[l * len..(l + 1) * len].to_vec();
            f.process(&mut lane);
            want.extend(lane);
        }
        close(&up(&x).fir(&taps).to_host(), &want, 1e-5, "fir");
        // lanes along the last axis of a transposed view
        let xt = x.view().transpose().to_owned();
        close(&up(&xt).transpose().fir(&taps).to_host(), &want, 1e-5, "fir of a transposed view");
    }

    #[test]
    fn double_precision_where_supported() {
        if !available() {
            return;
        }
        let x = NdArray::from_vec((0..1000).map(|i| (i as f64 * 0.1).sin()).collect(), &[10, 100]).unwrap();
        match GpuArray::from_host(&x.view()) {
            Ok(g) => {
                assert!(supports::<f64>());
                let want: f64 = x.as_slice().iter().map(|v| 2.0 * v + 0.5f64).sum();
                let got = g.axpb(2.0, 0.5).sum().to_host().as_slice()[0];
                assert!((got - want).abs() < 1e-9 * want.abs().max(1.0), "{got} vs {want}");
                // the transcendentals built from arithmetic, to within a few ulps
                let wide = NdArray::from_vec((0..4000).map(|i| (i as f64 - 2000.0) * 0.37 + 0.001).collect(), &[4000]).unwrap();
                let gw = GpuArray::from_host(&wide.view()).unwrap();
                let positive = NdArray::from_vec([1e-310, 1e-300, 1e-20, 0.5, 0.75, 1.0, 1.5, 2.0, 3.0, 10.0, 1e20, 1e300].to_vec(), &[12]).unwrap();
                let gp = GpuArray::from_host(&positive.view()).unwrap();
                for (name, got, input, f) in [
                    ("exp", gw.axpb(0.002, 0.0).exp(), wide.as_slice().iter().map(|v| v * 0.002).collect::<Vec<_>>(), f64::exp as fn(f64) -> f64),
                    ("exp wide", gw.exp(), wide.as_slice().to_vec(), f64::exp),
                    ("tanh", gw.axpb(0.01, 0.0).tanh(), wide.as_slice().iter().map(|v| v * 0.01).collect(), f64::tanh),
                    ("sin", gw.sin(), wide.as_slice().to_vec(), f64::sin),
                    ("cos", gw.cos(), wide.as_slice().to_vec(), f64::cos),
                    ("ln", gw.abs().ln(), wide.as_slice().iter().map(|v| v.abs()).collect(), f64::ln),
                    ("ln of extremes", gp.ln(), positive.as_slice().to_vec(), f64::ln),
                ] {
                    for (i, (a, &v)) in got.to_host().as_slice().iter().zip(&input).enumerate() {
                        let b = f(v);
                        let ok = if b.is_finite() { (a - b).abs() <= 4.0 * f64::EPSILON * b.abs().max(f64::MIN_POSITIVE) + 1e-300 } else { *a == b };
                        assert!(ok, "{name}: element {i} ({v}): {a} vs {b}");
                    }
                }
                let special = NdArray::from_vec(vec![0.0, -1.0, f64::INFINITY, 1000.0, -1000.0], &[5]).unwrap();
                let gs = GpuArray::from_host(&special.view()).unwrap();
                let ln = gs.ln().to_host();
                assert!(ln.as_slice()[0] == f64::NEG_INFINITY && ln.as_slice()[1].is_nan() && ln.as_slice()[2] == f64::INFINITY);
                let exp = gs.exp().to_host();
                assert!(exp.as_slice()[2] == f64::INFINITY && exp.as_slice()[3] == f64::INFINITY && exp.as_slice()[4] == 0.0);
                let t = g.transpose();
                let col_sums: Vec<f64> = (0..100).map(|j| (0..10).map(|i| x.as_slice()[i * 100 + j]).sum()).collect();
                let got = t.sum_axis(1).to_host();
                for (a, b) in got.as_slice().iter().zip(&col_sums) {
                    assert!((a - b).abs() < 1e-12, "{a} vs {b}");
                }
            }
            Err(e) => assert_eq!(e, GpuError::Unsupported("f64")),
        }
    }
}
