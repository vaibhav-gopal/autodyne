//! DLPack: zero-copy tensor exchange with NumPy, PyTorch, JAX, CuPy, TensorFlow and anything else
//! that speaks the [DLPack](https://dmlc.github.io/dlpack/latest/) C ABI.
//!
//! - Export: [`NdArray::into_dlpack`] hands an array's storage over to another library (versioned,
//!   v1) along with a deleter; [`NdArray::into_dlpack_legacy`] produces the pre-1.0 struct. Shared
//!   (`Arc`) arrays are exported read-only, without copying.
//! - Import: [`DlpackTensor::from_raw`] / [`DlpackTensor::from_raw_legacy`] take ownership of a
//!   tensor from another library (any strides, including negative ones) and release it exactly
//!   once, when dropped. It is a runtime-typed view ([`DlpackTensor::view`]), a typed one
//!   ([`DlpackTensor::typed`]), or, when contiguous, a full [`NdArray`] over the foreign memory
//!   ([`DlpackTensor::into_array`]).
//!
//! Only memory the CPU can read is accepted (CPU, pinned / managed host memory); bool, float16 /
//! bfloat16 and vector (lanes > 1) element types are rejected.

use std::ffi::c_void;
use std::marker::PhantomData;
use std::mem::{align_of, size_of};

use thiserror::Error;

use crate::dynamic::{DynArray, DynElement, DynError, DynView, DynViewMut};
use crate::signal::{NdArray, NdView, NdViewMut, Storage, StorageMut, MAX_DIMS};
use crate::units::*;

// THE C ABI =======================================================================================

/// `DLDevice`: where the memory lives.
#[repr(C)]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct DLDevice {
    /// The kind of device (`DL_CPU`, `DL_CUDA`, ...).
    pub device_type: i32,
    /// Which device of that kind (0 for the CPU).
    pub device_id: i32,
}

/// Host memory.
pub const DL_CPU: i32 = 1;
/// CUDA device memory.
pub const DL_CUDA: i32 = 2;
/// Pinned host memory allocated by CUDA (readable by the CPU).
pub const DL_CUDA_HOST: i32 = 3;
/// Pinned host memory allocated by ROCm (readable by the CPU).
pub const DL_ROCM_HOST: i32 = 11;
/// CUDA unified memory (readable by the CPU).
pub const DL_CUDA_MANAGED: i32 = 13;

/// `DLDataType`: element kind, bits, and vector lanes.
#[repr(C)]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct DLDataType {
    /// The kind of element (`DL_INT`, `DL_FLOAT`, ...).
    pub code: u8,
    /// Bits per element (per lane).
    pub bits: u8,
    /// Vector lanes per element (1 for scalars, the only kind accepted here).
    pub lanes: u16,
}

/// Signed integers.
pub const DL_INT: u8 = 0;
/// Unsigned integers.
pub const DL_UINT: u8 = 1;
/// IEEE floats.
pub const DL_FLOAT: u8 = 2;
/// bfloat16 (not supported here).
pub const DL_BFLOAT: u8 = 4;
/// Complex numbers (two floats of `bits / 2` each).
pub const DL_COMPLEX: u8 = 5;
/// Booleans (not supported here).
pub const DL_BOOL: u8 = 6;

/// `DLTensor`: pointer, device, shape and strides (in elements; null strides mean row-major).
#[repr(C)]
#[derive(Debug, Clone, Copy)]
pub struct DLTensor {
    /// The memory, at `byte_offset` before the first element.
    pub data: *mut c_void,
    /// Where the memory lives.
    pub device: DLDevice,
    /// Number of axes.
    pub ndim: i32,
    /// The element type.
    pub dtype: DLDataType,
    /// `ndim` axis lengths.
    pub shape: *mut i64,
    /// `ndim` element strides, or null for row-major.
    pub strides: *mut i64,
    /// Bytes from `data` to the first element.
    pub byte_offset: u64,
}

/// `DLManagedTensor` (pre-1.0, "legacy").
#[repr(C)]
#[derive(Debug)]
pub struct DLManagedTensor {
    /// The tensor.
    pub dl_tensor: DLTensor,
    /// The producer's context, for `deleter`.
    pub manager_ctx: *mut c_void,
    /// Called once by the consumer when it no longer needs the memory.
    pub deleter: Option<unsafe extern "C" fn(*mut DLManagedTensor)>,
}

/// `DLPackVersion`: the ABI version of a versioned tensor.
#[repr(C)]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct DLPackVersion {
    /// Incompatible changes.
    pub major: u32,
    /// Compatible additions.
    pub minor: u32,
}

/// The DLPack version this module produces.
pub const DLPACK_VERSION: DLPackVersion = DLPackVersion { major: 1, minor: 1 };
/// `flags` bit: the consumer must not write to the memory.
pub const DLPACK_FLAG_READ_ONLY: u64 = 1;
/// `flags` bit: the producer made a copy for this export.
pub const DLPACK_FLAG_IS_COPIED: u64 = 2;

/// `DLManagedTensorVersioned` (DLPack 1.x).
#[repr(C)]
#[derive(Debug)]
pub struct DLManagedTensorVersioned {
    /// The ABI version the producer used.
    pub version: DLPackVersion,
    /// The producer's context, for `deleter`.
    pub manager_ctx: *mut c_void,
    /// Called once by the consumer when it no longer needs the memory.
    pub deleter: Option<unsafe extern "C" fn(*mut DLManagedTensorVersioned)>,
    /// `DLPACK_FLAG_*` bits.
    pub flags: u64,
    /// The tensor.
    pub dl_tensor: DLTensor,
}

/// Errors importing or exporting DLPack tensors.
#[derive(Debug, Clone, PartialEq, Eq, Error)]
pub enum DlpackError {
    /// A null tensor pointer.
    #[error("null tensor pointer")]
    Null,
    /// A DLPack major version other than 1.
    #[error("unsupported DLPack major version {0} (this is version 1)")]
    Version(u32),
    /// Memory the CPU can't read (the device type).
    #[error("memory on device type {0} is not readable by the CPU")]
    Device(i32),
    /// An element type autodyne has no `DType` for.
    #[error("unsupported element type (code {code}, {bits} bits, {lanes} lanes)")]
    DataType {
        /// The element kind (`DL_*`).
        code: u8,
        /// Bits per lane.
        bits: u8,
        /// Vector lanes.
        lanes: u16,
    },
    /// More axes than `MAX_DIMS` (or a negative count).
    #[error("{0} dimensions is not supported (at most {MAX_DIMS})")]
    Dims(i32),
    /// Negative or overflowing shape, strides or offset.
    #[error("negative or overflowing shape or strides")]
    Layout,
    /// Writing was asked of a read-only tensor.
    #[error("the tensor is read-only")]
    ReadOnly,
    /// An owned array needs contiguous row-major memory.
    #[error("the tensor is not contiguous row-major, so it can't become an NdArray (use view())")]
    NotContiguous,
    /// A runtime-typed array error.
    #[error(transparent)]
    Dyn(#[from] DynError),
}

/// The DLPack element type of a `DType` (`None` for the packed `I24`).
pub fn to_dl_dtype(dtype: DType) -> Option<DLDataType> {
    let (code, bits) = match dtype {
        DType::F32 => (DL_FLOAT, 32),
        DType::F64 => (DL_FLOAT, 64),
        DType::I8 => (DL_INT, 8),
        DType::I16 => (DL_INT, 16),
        DType::I32 => (DL_INT, 32),
        DType::I64 => (DL_INT, 64),
        DType::U8 => (DL_UINT, 8),
        DType::U16 => (DL_UINT, 16),
        DType::U32 => (DL_UINT, 32),
        DType::U64 => (DL_UINT, 64),
        DType::ComplexF32 => (DL_COMPLEX, 64),
        DType::ComplexF64 => (DL_COMPLEX, 128),
        DType::I24 => return None,
    };
    Some(DLDataType { code, bits, lanes: 1 })
}

/// The `DType` of a DLPack element type.
pub fn from_dl_dtype(t: DLDataType) -> Result<DType, DlpackError> {
    let unsupported = DlpackError::DataType { code: t.code, bits: t.bits, lanes: t.lanes };
    if t.lanes != 1 {
        return Err(unsupported);
    }
    Ok(match (t.code, t.bits) {
        (DL_FLOAT, 32) => DType::F32,
        (DL_FLOAT, 64) => DType::F64,
        (DL_INT, 8) => DType::I8,
        (DL_INT, 16) => DType::I16,
        (DL_INT, 32) => DType::I32,
        (DL_INT, 64) => DType::I64,
        (DL_UINT, 8) => DType::U8,
        (DL_UINT, 16) => DType::U16,
        (DL_UINT, 32) => DType::U32,
        (DL_UINT, 64) => DType::U64,
        (DL_COMPLEX, 64) => DType::ComplexF32,
        (DL_COMPLEX, 128) => DType::ComplexF64,
        _ => return Err(unsupported),
    })
}

// EXPORT ==========================================================================================

/// One heap block per export: the managed struct first (so a pointer to it is a pointer to the
/// block), then the shape and strides it points into, then the storage that owns the elements.
#[repr(C)]
struct Exported<M, S> {
    managed: M,
    shape: [i64; MAX_DIMS],
    strides: [i64; MAX_DIMS],
    storage: S,
}

unsafe extern "C" fn delete_versioned<S>(m: *mut DLManagedTensorVersioned) {
    if !m.is_null() {
        // SAFETY: `m` came from `Box::into_raw` of this block in `into_dlpack`, and DLPack calls the
        // deleter once
        drop(unsafe { Box::from_raw(m.cast::<Exported<DLManagedTensorVersioned, S>>()) });
    }
}

unsafe extern "C" fn delete_legacy<S>(m: *mut DLManagedTensor) {
    if !m.is_null() {
        // SAFETY: as above, from `into_dlpack_legacy`
        drop(unsafe { Box::from_raw(m.cast::<Exported<DLManagedTensor, S>>()) });
    }
}

/// The tensor header for `array` seen through `axes` (exported axis k is the array's axis
/// `axes[k]`; `None` keeps the order).
fn tensor_for<T: DynElement, S: Storage<Elem = T>>(array: &NdArray<T, S>, axes: Option<&[usize]>) -> (DLTensor, [i64; MAX_DIMS], [i64; MAX_DIMS]) {
    let mut own_shape = [0i64; MAX_DIMS];
    let mut own_strides = [0i64; MAX_DIMS];
    let n = array.ndim();
    let mut acc = 1i64;
    for i in (0..n).rev() {
        own_shape[i] = array.shape()[i] as i64;
        own_strides[i] = acc;
        acc *= own_shape[i];
    }
    let (mut shape, mut strides) = (own_shape, own_strides);
    if let Some(axes) = axes {
        for (k, &a) in axes.iter().enumerate() {
            shape[k] = own_shape[a];
            strides[k] = own_strides[a];
        }
    }
    let tensor = DLTensor {
        // set once the storage has reached its final place (see into_dlpack)
        data: std::ptr::null_mut(),
        device: DLDevice { device_type: DL_CPU, device_id: 0 },
        ndim: n as i32,
        dtype: to_dl_dtype(T::DTYPE).expect("array element types all have a DLPack type"),
        shape: std::ptr::null_mut(),
        strides: std::ptr::null_mut(),
        byte_offset: 0,
    };
    (tensor, shape, strides)
}

impl<T: DynElement, S: Storage<Elem = T> + 'static> NdArray<T, S> {
    /// Hands the array to another library as a DLPack 1.x tensor, without copying. The receiver owns
    /// it and must call its deleter (which frees the storage) exactly once. Storage that can't be
    /// written in place (shared `Arc` arrays) is flagged read-only. Axis labels are not carried.
    pub fn into_dlpack(self) -> *mut DLManagedTensorVersioned {
        let (tensor, shape, strides) = tensor_for(&self, None);
        let flags = if S::WRITABLE { 0 } else { DLPACK_FLAG_READ_ONLY };
        let managed = DLManagedTensorVersioned { version: DLPACK_VERSION, manager_ctx: std::ptr::null_mut(), deleter: Some(delete_versioned::<S>), flags, dl_tensor: tensor };
        let block = Box::into_raw(Box::new(Exported { managed, shape, strides, storage: self.into_storage() }));
        // SAFETY: `block` is a live allocation; point the tensor at its own shape and strides, and at
        // the elements through the storage in its final place (so the pointer stays valid and, for
        // writable storage, carries write permission)
        unsafe {
            (*block).managed.dl_tensor.data = (*block).storage.as_raw_ptr().cast();
            (*block).managed.dl_tensor.shape = (*block).shape.as_mut_ptr();
            (*block).managed.dl_tensor.strides = (*block).strides.as_mut_ptr();
            (*block).managed.manager_ctx = block.cast();
        }
        block.cast()
    }
}

impl<T: DynElement, S: Storage<Elem = T> + 'static> NdArray<T, S> {
    /// [`into_dlpack`](Self::into_dlpack) of the array seen through an axis permutation (exported
    /// axis k is axis `axes[k]`), still without copying: e.g. a result computed in an input's memory
    /// order goes back with that input's strides, as NumPy's `order='K'` results do. Panics unless
    /// `axes` is a permutation of the array's axes.
    pub fn into_dlpack_permuted(self, axes: &[usize]) -> *mut DLManagedTensorVersioned {
        let n = self.ndim();
        let mut seen = [false; MAX_DIMS];
        assert!(axes.len() == n && axes.iter().all(|&a| a < n && !std::mem::replace(&mut seen[a], true)), "axes must be a permutation");
        let (tensor, shape, strides) = tensor_for(&self, Some(axes));
        let flags = if S::WRITABLE { 0 } else { DLPACK_FLAG_READ_ONLY };
        let managed = DLManagedTensorVersioned { version: DLPACK_VERSION, manager_ctx: std::ptr::null_mut(), deleter: Some(delete_versioned::<S>), flags, dl_tensor: tensor };
        let block = Box::into_raw(Box::new(Exported { managed, shape, strides, storage: self.into_storage() }));
        // SAFETY: as in `into_dlpack`
        unsafe {
            (*block).managed.dl_tensor.data = (*block).storage.as_raw_ptr().cast();
            (*block).managed.dl_tensor.shape = (*block).shape.as_mut_ptr();
            (*block).managed.dl_tensor.strides = (*block).strides.as_mut_ptr();
            (*block).managed.manager_ctx = block.cast();
        }
        block.cast()
    }
}

impl<T: DynElement, S: StorageMut<Elem = T> + 'static> NdArray<T, S> {
    /// The pre-1.0 `DLManagedTensor`, for consumers that don't speak DLPack 1.x. It has no read-only
    /// flag, so only writable storage can be exported this way.
    pub fn into_dlpack_legacy(self) -> *mut DLManagedTensor {
        let (tensor, shape, strides) = tensor_for(&self, None);
        let managed = DLManagedTensor { dl_tensor: tensor, manager_ctx: std::ptr::null_mut(), deleter: Some(delete_legacy::<S>) };
        let block = Box::into_raw(Box::new(Exported { managed, shape, strides, storage: self.into_storage() }));
        // SAFETY: as in `into_dlpack`
        unsafe {
            (*block).managed.dl_tensor.data = (*block).storage.as_raw_ptr().cast();
            (*block).managed.dl_tensor.shape = (*block).shape.as_mut_ptr();
            (*block).managed.dl_tensor.strides = (*block).strides.as_mut_ptr();
            (*block).managed.manager_ctx = block.cast();
        }
        block.cast()
    }
}

impl DynArray {
    /// [`NdArray::into_dlpack`] for whatever element type the array holds.
    pub fn into_dlpack(self) -> *mut DLManagedTensorVersioned {
        crate::dyn_match!(self, a => a.into_dlpack())
    }
    /// [`NdArray::into_dlpack_permuted`] for whatever element type the array holds.
    pub fn into_dlpack_permuted(self, axes: &[usize]) -> *mut DLManagedTensorVersioned {
        crate::dyn_match!(self, a => a.into_dlpack_permuted(axes))
    }
    /// [`NdArray::into_dlpack_legacy`] for whatever element type the array holds.
    pub fn into_dlpack_legacy(self) -> *mut DLManagedTensor {
        crate::dyn_match!(self, a => a.into_dlpack_legacy())
    }
}

// IMPORT ==========================================================================================

#[derive(Debug)]
enum Managed {
    Versioned(*mut DLManagedTensorVersioned),
    Legacy(*mut DLManagedTensor),
}

/// A tensor received from another library through DLPack. Owns it: dropping calls the producer's
/// deleter (once). Not `Send`: some producers' deleters must run on the thread that made them.
#[derive(Debug)]
pub struct DlpackTensor {
    managed: Managed,
    dtype: DType,
    ndim: usize,
    shape: [usize; MAX_DIMS],
    strides: [isize; MAX_DIMS],
    /// data + byte_offset: the element at index [0, 0, ...]
    origin: *mut u8,
    read_only: bool,
}

impl Drop for DlpackTensor {
    fn drop(&mut self) {
        // SAFETY: we own the managed tensor (taken in `from_raw*`) and drop runs once
        unsafe {
            match self.managed {
                Managed::Versioned(m) => {
                    if let Some(deleter) = (*m).deleter {
                        deleter(m);
                    }
                }
                Managed::Legacy(m) => {
                    if let Some(deleter) = (*m).deleter {
                        deleter(m);
                    }
                }
            }
        }
    }
}

impl DlpackTensor {
    /// Takes ownership of a DLPack 1.x tensor. On error the tensor is released (its deleter runs).
    ///
    /// # Safety
    /// `ptr` must be null or a valid `DLManagedTensorVersioned` that nobody else will use or delete,
    /// describing memory that stays valid until its deleter is called.
    pub unsafe fn from_raw(ptr: *mut DLManagedTensorVersioned) -> Result<Self, DlpackError> {
        if ptr.is_null() {
            return Err(DlpackError::Null);
        }
        // SAFETY: valid per the contract
        let (version, flags, tensor) = unsafe { ((*ptr).version, (*ptr).flags, (*ptr).dl_tensor) };
        let mut taken = Self::owning(Managed::Versioned(ptr), flags & DLPACK_FLAG_READ_ONLY != 0);
        if version.major != 1 {
            return Err(DlpackError::Version(version.major));
        }
        // SAFETY: as above
        unsafe { taken.describe(&tensor)? };
        Ok(taken)
    }

    /// Takes ownership of a pre-1.0 `DLManagedTensor` (no read-only flag: assumed writable).
    ///
    /// # Safety
    /// As [`from_raw`](Self::from_raw).
    pub unsafe fn from_raw_legacy(ptr: *mut DLManagedTensor) -> Result<Self, DlpackError> {
        if ptr.is_null() {
            return Err(DlpackError::Null);
        }
        // SAFETY: valid per the contract
        let tensor = unsafe { (*ptr).dl_tensor };
        let mut taken = Self::owning(Managed::Legacy(ptr), false);
        // SAFETY: as above
        unsafe { taken.describe(&tensor)? };
        Ok(taken)
    }

    /// Ownership first, so an error from here on still runs the deleter (through `Drop`).
    fn owning(managed: Managed, read_only: bool) -> Self {
        Self { managed, dtype: DType::U8, ndim: 0, shape: [0; MAX_DIMS], strides: [0; MAX_DIMS], origin: std::ptr::null_mut(), read_only }
    }

    /// # Safety
    /// `t` describes valid memory (the `from_raw*` contract).
    unsafe fn describe(&mut self, t: &DLTensor) -> Result<(), DlpackError> {
        if ![DL_CPU, DL_CUDA_HOST, DL_ROCM_HOST, DL_CUDA_MANAGED].contains(&t.device.device_type) {
            return Err(DlpackError::Device(t.device.device_type));
        }
        self.dtype = from_dl_dtype(t.dtype)?;
        if t.ndim < 0 || t.ndim as usize > MAX_DIMS {
            return Err(DlpackError::Dims(t.ndim));
        }
        self.ndim = t.ndim as usize;
        let n = self.ndim;
        for i in 0..n {
            // SAFETY: shape has ndim entries per the DLPack contract
            let d = unsafe { *t.shape.add(i) };
            self.shape[i] = usize::try_from(d).map_err(|_| DlpackError::Layout)?;
        }
        if t.strides.is_null() {
            let mut acc = 1isize;
            for i in (0..n).rev() {
                self.strides[i] = acc;
                acc = acc.checked_mul(self.shape[i] as isize).ok_or(DlpackError::Layout)?;
            }
        } else {
            for i in 0..n {
                // SAFETY: strides has ndim entries when not null
                let s = unsafe { *t.strides.add(i) };
                self.strides[i] = isize::try_from(s).map_err(|_| DlpackError::Layout)?;
            }
        }
        let offset = usize::try_from(t.byte_offset).map_err(|_| DlpackError::Layout)?;
        self.origin = t.data.cast::<u8>().wrapping_add(offset);
        Ok(())
    }

    /// The element type.
    pub fn dtype(&self) -> DType {
        self.dtype
    }
    /// Length of each axis.
    pub fn shape(&self) -> &[usize] {
        &self.shape[..self.ndim]
    }
    /// Element strides per axis.
    pub fn strides(&self) -> &[isize] {
        &self.strides[..self.ndim]
    }
    /// Number of elements.
    pub fn len(&self) -> usize {
        self.shape().iter().product()
    }
    /// Whether there are no elements.
    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }
    /// Whether the producer forbade writing.
    pub fn is_read_only(&self) -> bool {
        self.read_only
    }
    /// Whether the elements are one contiguous row-major block.
    pub fn is_contiguous(&self) -> bool {
        let mut expected = 1isize;
        for i in (0..self.ndim).rev() {
            if self.shape[i] != 1 {
                if self.strides[i] != expected {
                    return false;
                }
                expected *= self.shape[i] as isize;
            }
        }
        true
    }

    /// The bytes spanned by the layout, and the element offset of index [0, 0, ...] within them.
    fn span(&self) -> Result<(*mut u8, usize, usize), DlpackError> {
        if self.is_empty() {
            return Ok((std::ptr::NonNull::<u8>::dangling().as_ptr(), 0, 0));
        }
        let (mut lo, mut hi) = (0isize, 0isize);
        for (&d, &s) in self.shape().iter().zip(self.strides()) {
            let reach = (d as isize - 1).checked_mul(s).ok_or(DlpackError::Layout)?;
            if reach < 0 {
                lo = lo.checked_add(reach).ok_or(DlpackError::Layout)?;
            } else {
                hi = hi.checked_add(reach).ok_or(DlpackError::Layout)?;
            }
        }
        let size = self.dtype.size_bytes();
        let elements = (hi - lo + 1) as usize;
        let start = self.origin.wrapping_offset(lo * size as isize);
        Ok((start, elements * size, (-lo) as usize))
    }

    /// A runtime-typed view of the elements, without copying.
    pub fn view(&self) -> Result<DynView<'_>, DlpackError> {
        let (start, bytes, offset) = self.span()?;
        // SAFETY: the producer guarantees the described elements are readable memory for as long
        // as the tensor lives; the slice covers exactly the bytes they span and borrows `self`
        let bytes = unsafe { std::slice::from_raw_parts(start, bytes) };
        Ok(DynView::with_strides(bytes, self.dtype, self.shape(), Some(self.strides()), offset)?)
    }
    /// A mutable runtime-typed view; errors on read-only tensors.
    pub fn view_mut(&mut self) -> Result<DynViewMut<'_>, DlpackError> {
        if self.read_only {
            return Err(DlpackError::ReadOnly);
        }
        let (start, bytes, offset) = self.span()?;
        // SAFETY: as in `view`, writable (not flagged read-only) and exclusively borrowed
        let bytes = unsafe { std::slice::from_raw_parts_mut(start, bytes) };
        Ok(DynViewMut::with_strides(bytes, self.dtype, self.shape(), Some(self.strides()), offset)?)
    }
    /// The typed view, if the element type is `T` and the memory is aligned for it.
    pub fn typed<T: DynElement>(&self) -> Result<NdView<'_, T>, DlpackError> {
        let view = self.view()?;
        // the typed view borrows `self` (through the bytes), not the temporary DynView
        let typed: NdView<'_, T> = view.typed::<T>()?;
        Ok(typed)
    }
    /// The typed mutable view (see [`typed`](Self::typed)); errors on read-only tensors.
    pub fn typed_mut<T: DynElement>(&mut self) -> Result<NdViewMut<'_, T>, DlpackError> {
        if self.read_only {
            return Err(DlpackError::ReadOnly);
        }
        if T::DTYPE != self.dtype {
            return Err(DynError::DTypeMismatch { expected: self.dtype, found: T::DTYPE }.into());
        }
        let (start, bytes, offset) = self.span()?;
        if !(start as usize).is_multiple_of(align_of::<T>()) {
            return Err(DynError::Unaligned(self.dtype).into());
        }
        // SAFETY: aligned (checked), sized and valid elements of type T (dtype checked) for the
        // tensor's lifetime, writable and exclusively borrowed
        let elements = unsafe { std::slice::from_raw_parts_mut(start.cast::<T>(), bytes / size_of::<T>()) };
        Ok(NdViewMut::from_parts(elements, self.shape(), self.strides(), offset).map_err(DynError::from)?)
    }

    /// The tensor as an `NdArray` over its own (foreign) memory, without copying: it must be
    /// contiguous row-major, writable, aligned and of element type `T`. On error the tensor is
    /// released; check [`is_contiguous`](Self::is_contiguous) and friends first to keep it.
    pub fn into_array<T: DynElement>(self) -> Result<NdArray<T, ForeignBuffer<T>>, DlpackError> {
        if T::DTYPE != self.dtype {
            return Err(DynError::DTypeMismatch { expected: self.dtype, found: T::DTYPE }.into());
        }
        if self.read_only {
            return Err(DlpackError::ReadOnly);
        }
        if !self.is_contiguous() {
            return Err(DlpackError::NotContiguous);
        }
        let len = self.len();
        let ptr = if len == 0 { std::ptr::NonNull::<T>::dangling().as_ptr() } else { self.origin.cast::<T>() };
        if !(ptr as usize).is_multiple_of(align_of::<T>()) {
            return Err(DynError::Unaligned(self.dtype).into());
        }
        let shape = self.shape;
        let ndim = self.ndim;
        let buffer = ForeignBuffer { ptr, len, _tensor: self, _elem: PhantomData };
        Ok(NdArray::from_storage(buffer, &shape[..ndim]).map_err(DynError::from)?)
    }
}

/// Storage owned by another library (received through DLPack): the elements stay where the
/// producer put them, and are released through its deleter when the array is dropped.
#[derive(Debug)]
pub struct ForeignBuffer<T> {
    ptr: *mut T,
    len: usize,
    _tensor: DlpackTensor,
    _elem: PhantomData<T>,
}

// SAFETY: the producer's memory stays put until the deleter runs (when this buffer drops); the
// pointer came from the producer with write access (read-only tensors never become buffers)
unsafe impl<T> Storage for ForeignBuffer<T> {
    type Elem = T;
    fn as_slice(&self) -> &[T] {
        // SAFETY: checked contiguous, aligned and typed in `into_array`; alive while `_tensor` is
        unsafe { std::slice::from_raw_parts(self.ptr, self.len) }
    }
    fn as_raw_ptr(&mut self) -> *mut T {
        self.ptr
    }
}

impl<T> StorageMut for ForeignBuffer<T> {
    fn as_mut_slice(&mut self) -> &mut [T] {
        // SAFETY: as above, and not read-only (checked in `into_array`)
        unsafe { std::slice::from_raw_parts_mut(self.ptr, self.len) }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::atomic::{AtomicUsize, Ordering};
    use std::sync::Arc;

    #[test]
    #[cfg(target_pointer_width = "64")]
    fn struct_layouts_match_the_c_header() {
        assert_eq!((size_of::<DLDevice>(), size_of::<DLDataType>()), (8, 4));
        assert_eq!(size_of::<DLTensor>(), 48);
        assert_eq!(size_of::<DLManagedTensor>(), 64);
        assert_eq!(size_of::<DLManagedTensorVersioned>(), 80);
        assert_eq!(std::mem::offset_of!(DLManagedTensorVersioned, dl_tensor), 32);
    }

    /// Storage that counts its drops, to check the deleter runs exactly once.
    struct Counted(Vec<f32>, Arc<AtomicUsize>);
    impl Drop for Counted {
        fn drop(&mut self) {
            self.1.fetch_add(1, Ordering::SeqCst);
        }
    }
    // SAFETY: Vec-backed, pointer from a mutable borrow
    unsafe impl Storage for Counted {
        type Elem = f32;
        fn as_slice(&self) -> &[f32] {
            &self.0
        }
        fn as_raw_ptr(&mut self) -> *mut f32 {
            self.0.as_mut_ptr()
        }
    }
    impl StorageMut for Counted {
        fn as_mut_slice(&mut self) -> &mut [f32] {
            &mut self.0
        }
    }

    #[test]
    fn round_trips_without_copying_and_frees_once() {
        let drops = Arc::new(AtomicUsize::new(0));
        let data: Vec<f32> = (0..6).map(|i| i as f32).collect();
        let address = data.as_ptr();
        let array = NdArray::from_storage(Counted(data, drops.clone()), &[2, 3]).unwrap();
        let raw = array.into_dlpack();
        let tensor = unsafe { DlpackTensor::from_raw(raw) }.unwrap();
        assert_eq!((tensor.dtype(), tensor.shape(), tensor.strides()), (DType::F32, &[2, 3][..], &[3, 1][..]));
        assert!(!tensor.is_read_only() && tensor.is_contiguous());
        assert_eq!(tensor.typed::<f32>().unwrap().transpose().to_vec(), [0.0, 3.0, 1.0, 4.0, 2.0, 5.0]);
        assert!(tensor.typed::<f64>().is_err());
        let mut back = tensor.into_array::<f32>().unwrap();
        assert_eq!(back.as_slice().as_ptr(), address, "the same memory, not a copy");
        back.as_mut_slice()[0] = 10.0;
        assert_eq!(back.view().sum(), 25.0);
        assert_eq!(drops.load(Ordering::SeqCst), 0);
        drop(back);
        assert_eq!(drops.load(Ordering::SeqCst), 1, "the deleter freed the storage once");
    }

    #[test]
    fn legacy_and_dynamic_exports() {
        let a = DynArray::from_array(NdArray::from_vec(vec![1i16, -2, 3, -4], &[4]).unwrap());
        let tensor = unsafe { DlpackTensor::from_raw(a.into_dlpack()) }.unwrap();
        assert_eq!(tensor.dtype(), DType::I16);
        assert_eq!(tensor.view().unwrap().to_array().unwrap(), DynArray::from_array(NdArray::from_vec(vec![1i16, -2, 3, -4], &[4]).unwrap()));
        let legacy = NdArray::from_vec(vec![Complex::new(1.0f64, 2.0); 3], &[3]).unwrap().into_dlpack_legacy();
        let mut tensor = unsafe { DlpackTensor::from_raw_legacy(legacy) }.unwrap();
        assert_eq!(tensor.dtype(), DType::ComplexF64);
        tensor.typed_mut::<Complex<f64>>().unwrap().fill(Complex::new(0.0, 1.0));
        assert_eq!(tensor.typed::<Complex<f64>>().unwrap().to_vec(), [Complex::new(0.0, 1.0); 3]);
    }

    #[test]
    fn permuted_exports_carry_the_strides() {
        // a [2, 3] result exported as its transpose: shape [3, 2], strides [1, 3], same memory
        let a = NdArray::from_vec(vec![0.0f64, 1.0, 2.0, 3.0, 4.0, 5.0], &[2, 3]).unwrap();
        let address = a.as_slice().as_ptr();
        let tensor = unsafe { DlpackTensor::from_raw(a.into_dlpack_permuted(&[1, 0])) }.unwrap();
        assert_eq!((tensor.shape(), tensor.strides()), (&[3, 2][..], &[1, 3][..]));
        let view = tensor.typed::<f64>().unwrap();
        assert_eq!(view.to_vec(), [0.0, 3.0, 1.0, 4.0, 2.0, 5.0]);
        assert_eq!(view.as_ptr(), address);
    }

    #[test]
    fn shared_arrays_export_read_only() {
        let shared = NdArray::from_vec(vec![1.0f64, 2.0], &[2]).unwrap().into_shared();
        let keep = shared.clone();
        let mut tensor = unsafe { DlpackTensor::from_raw(shared.into_dlpack()) }.unwrap();
        assert!(tensor.is_read_only());
        assert_eq!(tensor.view_mut().err(), Some(DlpackError::ReadOnly));
        assert_eq!(tensor.typed::<f64>().unwrap().as_slice().unwrap().as_ptr(), keep.as_slice().as_ptr());
        assert_eq!(tensor.into_array::<f64>().err(), Some(DlpackError::ReadOnly));
    }

    static FOREIGN_DELETES: AtomicUsize = AtomicUsize::new(0);

    /// A tensor as another library would produce it: our own allocation, our own deleter.
    struct Foreign {
        managed: DLManagedTensorVersioned,
        shape: [i64; 2],
        strides: [i64; 2],
        data: Vec<f64>,
    }

    unsafe extern "C" fn delete_foreign(m: *mut DLManagedTensorVersioned) {
        FOREIGN_DELETES.fetch_add(1, Ordering::SeqCst);
        drop(unsafe { Box::from_raw((*m).manager_ctx.cast::<Foreign>()) });
    }

    fn foreign(shape: [i64; 2], strides: [i64; 2], byte_offset: u64, device: i32, dtype: DLDataType) -> *mut DLManagedTensorVersioned {
        let tensor = DLTensor { data: std::ptr::null_mut(), device: DLDevice { device_type: device, device_id: 0 }, ndim: 2, dtype, shape: std::ptr::null_mut(), strides: std::ptr::null_mut(), byte_offset };
        let managed = DLManagedTensorVersioned { version: DLPACK_VERSION, manager_ctx: std::ptr::null_mut(), deleter: Some(delete_foreign), flags: 0, dl_tensor: tensor };
        let block = Box::into_raw(Box::new(Foreign { managed, shape, strides, data: (0..6).map(f64::from).collect() }));
        unsafe {
            (*block).managed.dl_tensor.data = (*block).data.as_mut_ptr().cast();
            (*block).managed.dl_tensor.shape = (*block).shape.as_mut_ptr();
            (*block).managed.dl_tensor.strides = (*block).strides.as_mut_ptr();
            (*block).managed.manager_ctx = block.cast();
            &raw mut (*block).managed
        }
    }

    #[test]
    fn imports_strided_and_reversed_foreign_tensors() {
        let f64_type = DLDataType { code: DL_FLOAT, bits: 64, lanes: 1 };
        let before = FOREIGN_DELETES.load(Ordering::SeqCst);
        // column-major [3, 2] over 0..6 (what a transposed NumPy array exports)
        let tensor = unsafe { DlpackTensor::from_raw(foreign([3, 2], [1, 3], 0, DL_CPU, f64_type)) }.unwrap();
        assert!(!tensor.is_contiguous());
        assert_eq!(tensor.typed::<f64>().unwrap().to_vec(), [0.0, 3.0, 1.0, 4.0, 2.0, 5.0]);
        assert_eq!(tensor.into_array::<f64>().err(), Some(DlpackError::NotContiguous));
        // rows reversed: the origin is the last row, reached through byte_offset
        let tensor = unsafe { DlpackTensor::from_raw(foreign([2, 3], [-3, 1], 24, DL_CPU, f64_type)) }.unwrap();
        assert_eq!(tensor.typed::<f64>().unwrap().to_vec(), [3.0, 4.0, 5.0, 0.0, 1.0, 2.0]);
        drop(tensor);
        // rejected tensors are still released
        let gpu = unsafe { DlpackTensor::from_raw(foreign([2, 3], [3, 1], 0, DL_CUDA, f64_type)) };
        assert_eq!(gpu.err(), Some(DlpackError::Device(DL_CUDA)));
        let vector = unsafe { DlpackTensor::from_raw(foreign([2, 3], [3, 1], 0, DL_CPU, DLDataType { code: DL_FLOAT, bits: 32, lanes: 4 })) };
        assert!(matches!(vector.err(), Some(DlpackError::DataType { lanes: 4, .. })));
        assert_eq!(FOREIGN_DELETES.load(Ordering::SeqCst) - before, 4, "every tensor deleted exactly once");
        assert_eq!(unsafe { DlpackTensor::from_raw(std::ptr::null_mut()) }.err(), Some(DlpackError::Null));
    }

    #[test]
    fn dtype_mapping_round_trips() {
        for d in DType::ALL {
            match to_dl_dtype(d) {
                Some(t) => assert_eq!(from_dl_dtype(t), Ok(d)),
                None => assert_eq!(d, DType::I24),
            }
        }
        assert!(from_dl_dtype(DLDataType { code: DL_BOOL, bits: 8, lanes: 1 }).is_err());
        assert!(from_dl_dtype(DLDataType { code: DL_BFLOAT, bits: 16, lanes: 1 }).is_err());
    }
}
