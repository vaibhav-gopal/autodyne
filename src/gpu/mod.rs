//! Arrays on the GPU (feature `gpu`): [`GpuArray`] keeps `f32` or `f64` data in device memory
//! between operations, so a chain of element-wise work, reductions, filtering, matrix products and
//! FFTs never crosses the bus until [`GpuArray::to_host`]. Kernels are CubeCL's, run through wgpu (Vulkan, Metal,
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
    /// `v` rounded to this type (host-side constants: twiddles, scales).
    fn of(v: f64) -> Self;
}
impl GpuFloat for f32 {
    const NAME: &'static str = "f32";
    fn of(v: f64) -> Self {
        v as f32
    }
}
impl GpuFloat for f64 {
    const NAME: &'static str = "f64";
    fn of(v: f64) -> Self {
        v
    }
}

const THREADS: u32 = 256;

mod kernels;
mod wide;
#[cfg(test)]
mod tests;

use kernels::*;

// FFT =============================================================================================

/// `e^(-2 pi i t / n)` for `t` in `0..n / 2`, interleaved, computed in f64 and kept on the GPU
/// for the life of the process (one table per length and type).
fn twiddles<T: GpuFloat>(n: usize) -> Handle {
    static CACHE: OnceLock<std::sync::Mutex<std::collections::HashMap<(&'static str, usize), Handle>>> = OnceLock::new();
    let mut cache = CACHE.get_or_init(Default::default).lock().unwrap_or_else(std::sync::PoisonError::into_inner);
    cache
        .entry((T::NAME, n))
        .or_insert_with(|| {
            let tw: Vec<T> = (0..n / 2)
                .flat_map(|t| {
                    let a = -std::f64::consts::TAU * t as f64 / n as f64;
                    [T::of(a.cos()), T::of(a.sin())]
                })
                .collect();
            client().create(Bytes::from_elems(if tw.is_empty() { vec![T::of(0.0); 2] } else { tw }))
        })
        .clone()
}

/// The longest row transformed in shared memory: 32 KiB of interleaved complex values.
fn shared_fft_max<T: GpuFloat>() -> usize {
    32 * 1024 / (2 * size_of::<T>())
}

/// Transforms of `lanes` rows of power-of-two length `n <= shared_fft_max` in one launch (one cube
/// per row, see `fft_shared_kernel`): real input when `real_input`, the first `bins` values out,
/// scaled by `scale`.
fn fft_shared<T: GpuFloat>(x: &Handle, lanes: usize, n: usize, real_input: bool, bins: usize, inverse: bool, scale: f64) -> Handle {
    let out = client().empty((2 * lanes * bins).max(1) * size_of::<T>());
    let tw = twiddles::<T>(n);
    let input_len = if real_input { lanes * n } else { 2 * lanes * n };
    let threads = (n / 2).clamp(1, 256) as u32;
    // SAFETY: x holds `input_len` values, out lanes * bins complex ones, the twiddles n / 2; each
    // cube's shared buffer holds its row (2 n <= cap values)
    unsafe {
        fft_shared_kernel::launch_unchecked::<T, WgpuRuntime>(
            client(),
            CubeCount::Static(lanes as u32, 1, 1),
            CubeDim::new_1d(threads),
            ArrayArg::from_raw_parts(x.clone(), input_len),
            ArrayArg::from_raw_parts(out.clone(), 2 * lanes * bins),
            ArrayArg::from_raw_parts(tw, n.max(2)),
            n,
            n.trailing_zeros(),
            bins,
            T::of(if inverse { -1.0 } else { 1.0 }),
            T::of(scale),
            real_input,
            2 * shared_fft_max::<T>(),
        )
    };
    out
}

/// Unscaled radix-2 transforms of `lanes` interleaved complex lanes of `n` (a power of two) in
/// `x`: log2(n) Stockham passes ping-ponging between two buffers. `x` itself is left untouched.
fn radix2<T: GpuFloat>(x: &Handle, lanes: usize, n: usize, inverse: bool) -> Handle {
    let len = 2 * lanes * n;
    let tw = twiddles::<T>(n);
    let sign = T::of(if inverse { -1.0 } else { 1.0 });
    // pass p writes buffer p mod 2 and reads the previous one (the input for the first): the
    // input is never written, and a one-point transform (no passes) is the input itself
    let buffers = [client().empty(len * size_of::<T>()), client().empty(len * size_of::<T>())];
    let mut src = x.clone();
    let (mut ns, mut pass) = (1, 0);
    while ns < n {
        let dst = buffers[pass % 2].clone();
        // SAFETY: src and dst hold `len` values; the twiddle table n / 2 complex ones
        unsafe {
            stockham_kernel::launch_unchecked::<T, WgpuRuntime>(
                client(),
                CubeCount::Static(((n / 2) as u32).div_ceil(THREADS), lanes as u32, 1),
                CubeDim::new_1d(THREADS),
                ArrayArg::from_raw_parts(src.clone(), len),
                ArrayArg::from_raw_parts(dst.clone(), len),
                ArrayArg::from_raw_parts(tw.clone(), n.max(2)),
                n,
                ns,
                sign,
            )
        };
        src = dst;
        ns *= 2;
        pass += 1;
    }
    src
}

/// Unscaled transforms of `lanes` interleaved complex lanes of any length `n` by Bluestein's
/// algorithm: `X[k] = c[k] sum_j (x[j] c[j]) conj(c[k - j])` with `c[k] = e^(-i pi k² / n)`, the
/// convolution done by radix-2 transforms of length `m >= 2n - 1` (the filter's spectrum from the
/// CPU FFT, in f64).
fn bluestein<T: GpuFloat>(x: &Handle, lanes: usize, n: usize, inverse: bool) -> Handle {
    let m = (2 * n - 1).next_power_of_two();
    let sign = if inverse { 1.0 } else { -1.0 };
    // the chirp, k² reduced modulo 2n so the angle stays exact
    let chirp: Vec<(f64, f64)> = (0..n)
        .map(|k| {
            let a = sign * std::f64::consts::PI * ((k as u128 * k as u128) % (2 * n as u128)) as f64 / n as f64;
            (a.cos(), a.sin())
        })
        .collect();
    let mut filter = vec![crate::units::Complex::<f64>::zero(); m];
    filter[0] = crate::units::Complex::new(chirp[0].0, -chirp[0].1);
    for k in 1..n {
        let c = crate::units::Complex::new(chirp[k].0, -chirp[k].1);
        filter[k] = c;
        filter[m - k] = c;
    }
    crate::fft::Fft::<f64>::new(m).forward(&mut filter);
    let to_gpu = |v: Vec<(f64, f64)>| client().create(Bytes::from_elems(v.into_iter().flat_map(|(a, b)| [T::of(a), T::of(b)]).collect::<Vec<T>>()));
    let chirp_gpu = to_gpu(chirp);
    let filter_gpu = to_gpu(filter.iter().map(|z| (z.re, z.im)).collect());
    let padded = client().empty(2 * lanes * m * size_of::<T>());
    // SAFETY: x holds lanes * n complex values, padded lanes * m, the chirp n
    unsafe {
        chirp_kernel::launch_unchecked::<T, WgpuRuntime>(
            client(),
            CubeCount::Static((m as u32).div_ceil(THREADS), lanes as u32, 1),
            CubeDim::new_1d(THREADS),
            ArrayArg::from_raw_parts(x.clone(), 2 * lanes * n),
            ArrayArg::from_raw_parts(chirp_gpu.clone(), 2 * n),
            ArrayArg::from_raw_parts(padded.clone(), 2 * lanes * m),
            n,
            n,
            m,
        )
    };
    // the convolution: forward, times the filter's spectrum, inverse (scaled by 1 / m)
    let spectrum = radix2::<T>(&padded, lanes, m, false);
    // SAFETY: spectrum holds lanes * m complex values, the filter m
    unsafe {
        complex_mul_kernel::launch_unchecked::<T, WgpuRuntime>(client(), cubes(lanes * m), CubeDim::new_1d(THREADS), ArrayArg::from_raw_parts(spectrum.clone(), 2 * lanes * m), ArrayArg::from_raw_parts(filter_gpu, 2 * m), m)
    };
    let conv = radix2::<T>(&spectrum, lanes, m, true);
    let out = client().empty(2 * lanes * n * size_of::<T>());
    // the chirp again on the first n values, with the inverse transform's 1 / m folded in
    let chirp_scaled: Vec<T> = {
        let s = 1.0 / m as f64;
        (0..n)
            .flat_map(|k| {
                let a = sign * std::f64::consts::PI * ((k as u128 * k as u128) % (2 * n as u128)) as f64 / n as f64;
                [T::of(a.cos() * s), T::of(a.sin() * s)]
            })
            .collect()
    };
    let chirp_scaled = client().create(Bytes::from_elems(chirp_scaled));
    // SAFETY: conv holds lanes * m complex values, out lanes * n
    unsafe {
        chirp_kernel::launch_unchecked::<T, WgpuRuntime>(
            client(),
            CubeCount::Static((n as u32).div_ceil(THREADS), lanes as u32, 1),
            CubeDim::new_1d(THREADS),
            ArrayArg::from_raw_parts(conv, 2 * lanes * m),
            ArrayArg::from_raw_parts(chirp_scaled, 2 * n),
            ArrayArg::from_raw_parts(out.clone(), 2 * lanes * n),
            m,
            n,
            n,
        )
    };
    out
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

    /// The matrix product `self @ other` of two matrices (m x k and k x n), still on the GPU: a
    /// register-tiled kernel, each cube computing a 64 x 64 block of the result through shared
    /// memory. Transposed views are made contiguous first. Panics unless the inner dimensions agree.
    pub fn matmul(&self, other: &GpuArray<T>) -> GpuArray<T> {
        let (a, b) = (self.contiguous(), other.contiguous());
        assert!(a.ndim() == 2 && b.ndim() == 2 && a.memory[1] == b.memory[0], "GpuArray::matmul: {:?} @ {:?}", self.shape(), other.shape());
        let (m, k, n) = (a.memory[0], a.memory[1], b.memory[1]);
        let out = GpuArray::empty(&[m, n], vec![0, 1]);
        if m == 0 || n == 0 {
            return out;
        }
        if k == 0 {
            return out.mul_scalar(T::of(0.0));
        }
        let tiles = |d: usize| d.div_ceil(MM_TILE) as u32;
        // SAFETY: a holds m * k values, b k * n, out m * n; every access is bounds-checked against them
        unsafe {
            matmul_kernel::launch_unchecked::<T, WgpuRuntime>(
                client(),
                CubeCount::Static(tiles(n), tiles(m), 1),
                CubeDim::new_2d(MM_THREADS, MM_THREADS),
                ArrayArg::from_raw_parts(a.handle.clone(), m * k),
                ArrayArg::from_raw_parts(b.handle.clone(), k * n),
                ArrayArg::from_raw_parts(out.handle.clone(), m * n),
                m,
                k,
                n,
            )
        };
        out
    }

    /// The discrete Fourier transform along the next-to-last axis of interleaved complex data
    /// (`[..., n, 2]`: real and imaginary parts last), any `n >= 1`: radix-2 Stockham passes for
    /// powers of two, Bluestein's algorithm (a chirp convolution by power-of-two transforms)
    /// otherwise. Twiddles are computed on the host in f64. Same sign convention as `fft::Fft`.
    pub fn fft(&self) -> GpuArray<T> {
        self.complex_transform(false)
    }

    /// The inverse of [`fft`](Self::fft) (scaled by `1 / n`).
    pub fn ifft(&self) -> GpuArray<T> {
        self.complex_transform(true)
    }

    /// The spectrum of real lanes along the last axis (`[..., n]` to `[..., n / 2 + 1, 2]`,
    /// interleaved complex bins), as `numpy.fft.rfft`.
    pub fn rfft(&self) -> GpuArray<T> {
        let x = self.contiguous();
        let mut shape = x.memory.clone();
        let n = shape.pop().expect("GpuArray::rfft: an axis to transform");
        let lanes = x.len() / n.max(1);
        let bins = n / 2 + 1;
        if lanes > 0 && n.is_power_of_two() && n <= shared_fft_max::<T>() {
            // one launch: packed, transformed and cut to the bins in shared memory
            let out = fft_shared::<T>(&x.handle, lanes, n, true, bins, false, 1.0);
            shape.extend([bins, 2]);
            return GpuArray::new(out, shape.clone(), identity(shape.len()));
        }
        let z = GpuArray::<T>::empty(&[lanes, n, 2], vec![0, 1, 2]);
        if !x.is_empty() {
            // SAFETY: x holds lanes * n values, z twice as many
            unsafe {
                real_to_complex_kernel::launch_unchecked::<T, WgpuRuntime>(client(), cubes(x.len()), CubeDim::new_1d(THREADS), ArrayArg::from_raw_parts(x.handle.clone(), x.len()), ArrayArg::from_raw_parts(z.handle.clone(), 2 * x.len()))
            };
        }
        let spectrum = z.complex_transform(false);
        let out = GpuArray::<T>::empty(&[lanes, bins, 2], vec![0, 1, 2]);
        if lanes > 0 {
            // SAFETY: spectrum holds lanes * n complex values, out lanes * bins
            unsafe {
                bins_kernel::launch_unchecked::<T, WgpuRuntime>(
                    client(),
                    CubeCount::Static((bins as u32).div_ceil(THREADS), lanes as u32, 1),
                    CubeDim::new_1d(THREADS),
                    ArrayArg::from_raw_parts(spectrum.handle.clone(), 2 * lanes * n),
                    ArrayArg::from_raw_parts(out.handle.clone(), 2 * lanes * bins),
                    n,
                    bins,
                    T::of(1.0),
                )
            };
        }
        shape.extend([bins, 2]);
        GpuArray::new(out.handle, shape.clone(), identity(shape.len()))
    }

    /// Real lanes of length `n` from their bins (`[..., n / 2 + 1, 2]` to `[..., n]`), as
    /// `numpy.fft.irfft`: the Hermitian spectrum completed, inverted, the real parts kept.
    pub fn irfft(&self, n: usize) -> GpuArray<T> {
        let x = self.contiguous();
        let mut shape = x.memory.clone();
        assert!(shape.len() >= 2 && shape[shape.len() - 1] == 2 && shape[shape.len() - 2] == n / 2 + 1 && n >= 1, "GpuArray::irfft: bins [..., n / 2 + 1, 2] for n = {n}, got {shape:?}");
        shape.truncate(shape.len() - 2);
        let bins = n / 2 + 1;
        let lanes = x.len() / (2 * bins);
        let full = GpuArray::<T>::empty(&[lanes, n, 2], vec![0, 1, 2]);
        if lanes > 0 {
            // SAFETY: x holds lanes * bins complex values, full lanes * n
            unsafe {
                hermitian_kernel::launch_unchecked::<T, WgpuRuntime>(
                    client(),
                    CubeCount::Static((n as u32).div_ceil(THREADS), lanes as u32, 1),
                    CubeDim::new_1d(THREADS),
                    ArrayArg::from_raw_parts(x.handle.clone(), 2 * lanes * bins),
                    ArrayArg::from_raw_parts(full.handle.clone(), 2 * lanes * n),
                    n,
                    bins,
                )
            };
        }
        let z = full.complex_transform(true);
        let out = GpuArray::<T>::empty(&[lanes * n], vec![0]);
        if lanes > 0 {
            // SAFETY: z holds lanes * n complex values, out lanes * n
            unsafe {
                real_part_kernel::launch_unchecked::<T, WgpuRuntime>(client(), cubes(lanes * n), CubeDim::new_1d(THREADS), ArrayArg::from_raw_parts(z.handle.clone(), 2 * lanes * n), ArrayArg::from_raw_parts(out.handle.clone(), lanes * n), T::of(1.0))
            };
        }
        shape.push(n);
        GpuArray::new(out.handle, shape.clone(), identity(shape.len()))
    }

    /// `[..., n, 2]` complex lanes transformed along `n` (inverse scaled by `1 / n`).
    fn complex_transform(&self, inverse: bool) -> GpuArray<T> {
        let x = self.contiguous();
        assert!(x.ndim() >= 2 && x.memory[x.ndim() - 1] == 2, "GpuArray::fft: interleaved complex data [..., n, 2], got {:?}", self.shape());
        let n = x.memory[x.ndim() - 2];
        let lanes = x.len() / (2 * n.max(1));
        if lanes == 0 || n == 0 {
            return x;
        }
        if n.is_power_of_two() && n <= shared_fft_max::<T>() {
            let out = fft_shared::<T>(&x.handle, lanes, n, false, n, inverse, if inverse { 1.0 / n as f64 } else { 1.0 });
            return GpuArray::new(out, x.memory.clone(), identity(x.ndim()));
        }
        let z = if n.is_power_of_two() { radix2::<T>(&x.handle, lanes, n, inverse) } else { bluestein::<T>(&x.handle, lanes, n, inverse) };
        let z = if inverse {
            let scaled = GpuArray::<T>::empty(&[lanes, n, 2], vec![0, 1, 2]);
            // SAFETY: z and scaled hold lanes * n complex values
            unsafe {
                bins_kernel::launch_unchecked::<T, WgpuRuntime>(
                    client(),
                    CubeCount::Static((n as u32).div_ceil(THREADS), lanes as u32, 1),
                    CubeDim::new_1d(THREADS),
                    ArrayArg::from_raw_parts(z.clone(), 2 * lanes * n),
                    ArrayArg::from_raw_parts(scaled.handle.clone(), 2 * lanes * n),
                    n,
                    n,
                    T::of(1.0 / n as f64),
                )
            };
            scaled.handle
        } else {
            z
        };
        GpuArray::new(z, x.memory.clone(), identity(x.ndim()))
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
