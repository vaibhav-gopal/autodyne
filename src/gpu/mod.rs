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

mod kernels;
mod wide;
#[cfg(test)]
mod tests;

use kernels::*;

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
