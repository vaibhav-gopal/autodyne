//! Runtime-typed data, for APIs where the element type is only known at runtime (game engines, DAWs,
//! ML frameworks, file formats).
//!
//! - [`DynArray`]: an owned n-d array of any [`DType`], with typed access, casting, raw byte
//!   (de)serialization and zero-copy export ([`DynArray::as_bytes`]).
//! - [`DynView`] / [`DynViewMut`]: raw bytes from elsewhere, described by dtype + shape (+ strides),
//!   viewed as typed [`NdView`]s without copying when aligned.
//! - [`dyn_match!`](crate::dyn_match): runs generic code on whatever element type a `DynArray` holds.
//! - Runtime-typed math on views and arrays ([`DynView::binary`], [`DynView::unary`], reductions, casts) with
//!   explicit, loss-free [`Promotion`] rules.
//! - [`DynProcessor`]: processors with a runtime sample type and runtime-accessible parameters, built
//!   with [`build_dyn`] from a [`ProcessorFactory`].

use std::mem::size_of;

use thiserror::Error;

use crate::signal::{Endian, NdArray, NdError, NdView, NdViewMut, MAX_DIMS};
use crate::units::*;

mod processor;
pub use processor::*;

#[derive(Error, Debug, Clone, PartialEq, Eq)]
pub enum DynError {
    #[error("expected element type {expected}, found {found}")]
    DTypeMismatch { expected: DType, found: DType },
    #[error("element type {0} is not supported here")]
    Unsupported(DType),
    #[error("cannot cast complex values to the real type {0}")]
    ComplexToReal(DType),
    #[error("values of type {from} don't convert exactly to {to}; cast explicitly to accept the loss")]
    Inexact { from: DType, to: DType },
    #[error("{bytes} bytes is not a whole number of {size}-byte elements for the shape (needs {needed})")]
    ByteLength { bytes: usize, size: usize, needed: usize },
    #[error("the memory is not aligned for {0} elements; use to_array() to copy instead")]
    Unaligned(DType),
    #[error(transparent)]
    Nd(#[from] NdError),
}

mod sealed {
    pub trait Sealed {}
}

/// Element types a `DynArray` can hold: the primitive numbers and complex floats (every `DType` except
/// the packed `I24`). Sealed: implemented only for plain-data types, which is what makes the
/// zero-copy byte views sound.
pub trait DynElement: Reflection + Default + sealed::Sealed {
    #[doc(hidden)]
    fn wrap(array: NdArray<Self>) -> DynArray;
    #[doc(hidden)]
    fn peek(array: &DynArray) -> Option<&NdArray<Self>>;
    #[doc(hidden)]
    fn peek_mut(array: &mut DynArray) -> Option<&mut NdArray<Self>>;
    #[doc(hidden)]
    #[allow(clippy::result_large_err)] // hands the array back on mismatch, like Any::downcast
    fn take(array: DynArray) -> Result<NdArray<Self>, DynArray>;
    /// The value as (real, imaginary) f64; imaginary is 0 for real types.
    #[doc(hidden)]
    fn to_c64(self) -> (f64, f64);
    /// Numeric cast from (real, imaginary); real types drop the imaginary part, integers saturate.
    #[doc(hidden)]
    fn from_c64(value: (f64, f64)) -> Self;
    #[doc(hidden)]
    fn write_raw(self, endian: Endian, out: &mut [u8]);
    #[doc(hidden)]
    fn read_raw(bytes: &[u8], endian: Endian) -> Self;
}

macro_rules! real_element {
    ($($T:ty => $V:ident),+) => {$(
        impl sealed::Sealed for $T {}
        impl DynElement for $T {
            fn wrap(array: NdArray<Self>) -> DynArray {
                DynArray::$V(array)
            }
            fn peek(array: &DynArray) -> Option<&NdArray<Self>> {
                if let DynArray::$V(a) = array { Some(a) } else { None }
            }
            fn peek_mut(array: &mut DynArray) -> Option<&mut NdArray<Self>> {
                if let DynArray::$V(a) = array { Some(a) } else { None }
            }
            #[allow(clippy::result_large_err)]
            fn take(array: DynArray) -> Result<NdArray<Self>, DynArray> {
                if let DynArray::$V(a) = array { Ok(a) } else { Err(array) }
            }
            fn to_c64(self) -> (f64, f64) {
                (self as f64, 0.0)
            }
            fn from_c64(value: (f64, f64)) -> Self {
                value.0 as $T // `as` saturates and maps NaN to 0 for integers
            }
            fn write_raw(self, endian: Endian, out: &mut [u8]) {
                out.copy_from_slice(&match endian { Endian::Little => self.to_le_bytes(), Endian::Big => self.to_be_bytes() });
            }
            fn read_raw(bytes: &[u8], endian: Endian) -> Self {
                let b = bytes.try_into().expect("element-sized chunk");
                match endian { Endian::Little => <$T>::from_le_bytes(b), Endian::Big => <$T>::from_be_bytes(b) }
            }
        }
    )+};
}

real_element!(f32 => F32, f64 => F64, i8 => I8, i16 => I16, i32 => I32, i64 => I64, u8 => U8, u16 => U16, u32 => U32, u64 => U64);

macro_rules! complex_element {
    ($($F:ty => $V:ident),+) => {$(
        impl sealed::Sealed for Complex<$F> {}
        impl DynElement for Complex<$F> {
            fn wrap(array: NdArray<Self>) -> DynArray {
                DynArray::$V(array)
            }
            fn peek(array: &DynArray) -> Option<&NdArray<Self>> {
                if let DynArray::$V(a) = array { Some(a) } else { None }
            }
            fn peek_mut(array: &mut DynArray) -> Option<&mut NdArray<Self>> {
                if let DynArray::$V(a) = array { Some(a) } else { None }
            }
            #[allow(clippy::result_large_err)]
            fn take(array: DynArray) -> Result<NdArray<Self>, DynArray> {
                if let DynArray::$V(a) = array { Ok(a) } else { Err(array) }
            }
            fn to_c64(self) -> (f64, f64) {
                (self.re as f64, self.im as f64)
            }
            fn from_c64(value: (f64, f64)) -> Self {
                Complex::new(value.0 as $F, value.1 as $F)
            }
            fn write_raw(self, endian: Endian, out: &mut [u8]) {
                let half = size_of::<$F>();
                self.re.write_raw(endian, &mut out[..half]);
                self.im.write_raw(endian, &mut out[half..]);
            }
            fn read_raw(bytes: &[u8], endian: Endian) -> Self {
                let half = size_of::<$F>();
                Complex::new(<$F>::read_raw(&bytes[..half], endian), <$F>::read_raw(&bytes[half..], endian))
            }
        }
    )+};
}

complex_element!(f32 => ComplexF32, f64 => ComplexF64);

/// Reinterprets bytes as elements, if the length and alignment fit exactly.
fn cast_slice<T: DynElement>(bytes: &[u8]) -> Option<&[T]> {
    // SAFETY: `DynElement` is sealed and implemented only for primitive numbers and `repr(C)`
    // `Complex` of floats: plain data with no padding and no invalid bit patterns, so any correctly
    // aligned, element-sized bytes are valid `T`s. `align_to` puts only correctly aligned elements in
    // the middle slice; requiring an empty prefix and suffix means it covers exactly `bytes`.
    let (prefix, elements, suffix) = unsafe { bytes.align_to::<T>() };
    (prefix.is_empty() && suffix.is_empty()).then_some(elements)
}

/// Mutable variant of `cast_slice`.
fn cast_slice_mut<T: DynElement>(bytes: &mut [u8]) -> Option<&mut [T]> {
    // SAFETY: as in `cast_slice`; additionally every bit pattern written through the `&mut [T]` is a
    // valid byte sequence, so the bytes stay valid for their owner.
    let (prefix, elements, suffix) = unsafe { bytes.align_to_mut::<T>() };
    (prefix.is_empty() && suffix.is_empty()).then_some(elements)
}

// OWNED ===========================================================================================

/// An owned n-d array whose element type is chosen at runtime.
#[derive(Debug, Clone, PartialEq)]
pub enum DynArray {
    F32(NdArray<f32>),
    F64(NdArray<f64>),
    I8(NdArray<i8>),
    I16(NdArray<i16>),
    I32(NdArray<i32>),
    I64(NdArray<i64>),
    U8(NdArray<u8>),
    U16(NdArray<u16>),
    U32(NdArray<u32>),
    U64(NdArray<u64>),
    ComplexF32(NdArray<Complex<f32>>),
    ComplexF64(NdArray<Complex<f64>>),
}

/// Runs `$body` with `$a` bound to the typed `NdArray` inside a `DynArray` (or `&`/`&mut` of one),
/// whatever its element type. The body is compiled once per type, so it can call generic code:
///
/// ```
/// use autodyne::dynamic::DynArray;
/// use autodyne::dyn_match;
/// use autodyne::units::DType;
///
/// let a = DynArray::zeros(DType::I16, &[2, 3]).unwrap();
/// let count = dyn_match!(&a, arr => arr.len());
/// assert_eq!(count, 6);
/// ```
#[macro_export]
macro_rules! dyn_match {
    ($value:expr, $a:ident => $body:expr) => {
        match $value {
            $crate::dynamic::DynArray::F32($a) => $body,
            $crate::dynamic::DynArray::F64($a) => $body,
            $crate::dynamic::DynArray::I8($a) => $body,
            $crate::dynamic::DynArray::I16($a) => $body,
            $crate::dynamic::DynArray::I32($a) => $body,
            $crate::dynamic::DynArray::I64($a) => $body,
            $crate::dynamic::DynArray::U8($a) => $body,
            $crate::dynamic::DynArray::U16($a) => $body,
            $crate::dynamic::DynArray::U32($a) => $body,
            $crate::dynamic::DynArray::U64($a) => $body,
            $crate::dynamic::DynArray::ComplexF32($a) => $body,
            $crate::dynamic::DynArray::ComplexF64($a) => $body,
        }
    };
}

/// Runs `$body` with `$T` as the Rust type for a runtime `DType`, or evaluates `$otherwise` for
/// dtypes with no array type (`I24`).
macro_rules! with_dtype {
    ($dtype:expr, $T:ident => $body:expr, $otherwise:expr) => {
        match $dtype {
            DType::F32 => { type $T = f32; $body }
            DType::F64 => { type $T = f64; $body }
            DType::I8 => { type $T = i8; $body }
            DType::I16 => { type $T = i16; $body }
            DType::I32 => { type $T = i32; $body }
            DType::I64 => { type $T = i64; $body }
            DType::U8 => { type $T = u8; $body }
            DType::U16 => { type $T = u16; $body }
            DType::U32 => { type $T = u32; $body }
            DType::U64 => { type $T = u64; $body }
            DType::ComplexF32 => { type $T = Complex<f32>; $body }
            DType::ComplexF64 => { type $T = Complex<f64>; $body }
            DType::I24 => $otherwise,
        }
    };
}

impl DynArray {
    /// A zero-filled array of a runtime dtype.
    pub fn zeros(dtype: DType, shape: &[usize]) -> Result<Self, DynError> {
        with_dtype!(dtype, T => Ok(T::wrap(NdArray::<T>::zeros(shape)?)), Err(DynError::Unsupported(dtype)))
    }
    pub fn from_array<T: DynElement>(array: NdArray<T>) -> Self {
        T::wrap(array)
    }
    pub fn dtype(&self) -> DType {
        dyn_match!(self, a => dtype_of(a))
    }
    pub fn shape(&self) -> &[usize] {
        dyn_match!(self, a => a.shape())
    }
    pub fn len(&self) -> usize {
        dyn_match!(self, a => a.len())
    }
    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }
    /// The typed array, if the element type is `T`.
    pub fn as_array<T: DynElement>(&self) -> Result<&NdArray<T>, DynError> {
        let found = self.dtype();
        T::peek(self).ok_or(DynError::DTypeMismatch { expected: T::DTYPE, found })
    }
    pub fn as_array_mut<T: DynElement>(&mut self) -> Result<&mut NdArray<T>, DynError> {
        let found = self.dtype();
        T::peek_mut(self).ok_or(DynError::DTypeMismatch { expected: T::DTYPE, found })
    }
    /// Unwraps into the typed array, or gives the `DynArray` back if the type differs.
    #[allow(clippy::result_large_err)] // hands the array back on mismatch, like Any::downcast
    pub fn into_array<T: DynElement>(self) -> Result<NdArray<T>, DynArray> {
        T::take(self)
    }
    /// A copy converted to another element type, saturating values that don't fit (floats to
    /// integers truncate; real to complex sets a zero imaginary part). Exact for every value that
    /// fits, 64-bit integers included. Casting complex to real is an error. See
    /// [`cast_with`](Self::cast_with) for checked or wrapping casts.
    pub fn cast(&self, dtype: DType) -> Result<DynArray, DynError> {
        self.cast_with(dtype, CastMode::Saturating)
    }
    /// The elements' memory, without copying (native byte order), e.g. to hand to another runtime.
    pub fn as_bytes(&self) -> &[u8] {
        dyn_match!(self, a => {
            let s = a.as_slice();
            // SAFETY: the elements are plain data (see `DynElement`), so their memory is `len * size`
            // initialized bytes; the byte slice borrows `self`, so it can't outlive or alias a write.
            unsafe { std::slice::from_raw_parts(s.as_ptr().cast::<u8>(), std::mem::size_of_val(s)) }
        })
    }
    /// Serializes the elements (row-major) in the given byte order.
    pub fn to_bytes(&self, endian: Endian) -> Vec<u8> {
        let size = self.dtype().size_bytes();
        let mut out = vec![0u8; self.len() * size];
        dyn_match!(self, a => {
            for (chunk, &x) in out.chunks_exact_mut(size).zip(a.as_slice()) {
                x.write_raw(endian, chunk);
            }
        });
        out
    }
    /// Deserializes row-major elements from bytes in the given byte order.
    pub fn from_bytes(bytes: &[u8], dtype: DType, shape: &[usize], endian: Endian) -> Result<Self, DynError> {
        let size = dtype.size_bytes();
        let needed = shape.iter().product::<usize>() * size;
        if bytes.len() != needed {
            return Err(DynError::ByteLength { bytes: bytes.len(), size, needed });
        }
        with_dtype!(dtype, T => {
            let data: Vec<T> = bytes.chunks_exact(size).map(|c| T::read_raw(c, endian)).collect();
            Ok(T::wrap(NdArray::from_vec(data, shape)?))
        }, Err(DynError::Unsupported(dtype)))
    }
}

mod math;
mod ops;
pub use ops::*;

fn dtype_of<T: DynElement>(_: &NdArray<T>) -> DType {
    T::DTYPE
}

// BORROWED ========================================================================================

/// Layout of an external buffer, in elements.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
struct RawLayout {
    dtype: DType,
    shape: [usize; MAX_DIMS],
    strides: [isize; MAX_DIMS],
    ndim: usize,
    offset: usize,
}

impl RawLayout {
    fn new(dtype: DType, shape: &[usize], strides: Option<&[isize]>, offset: usize) -> Result<Self, DynError> {
        if dtype == DType::I24 {
            return Err(DynError::Unsupported(dtype));
        }
        if shape.len() > MAX_DIMS {
            return Err(NdError::TooManyDims(shape.len()).into());
        }
        let mut s = [0; MAX_DIMS];
        s[..shape.len()].copy_from_slice(shape);
        let mut st = [0; MAX_DIMS];
        match strides {
            Some(strides) if strides.len() != shape.len() => {
                return Err(NdError::ShapeMismatch { expected: shape.len(), got: strides.len() }.into())
            }
            Some(strides) => st[..strides.len()].copy_from_slice(strides),
            None => {
                let mut acc = 1isize;
                for i in (0..shape.len()).rev() {
                    st[i] = acc;
                    acc *= shape[i] as isize;
                }
            }
        }
        Ok(Self { dtype, shape: s, strides: st, ndim: shape.len(), offset })
    }
    fn check_bytes(&self, bytes: usize) -> Result<(), DynError> {
        let size = self.dtype.size_bytes();
        if !bytes.is_multiple_of(size) {
            return Err(DynError::ByteLength { bytes, size, needed: bytes.div_ceil(size) * size });
        }
        Ok(())
    }
    fn check_type<T: DynElement>(&self) -> Result<(), DynError> {
        if T::DTYPE != self.dtype {
            return Err(DynError::DTypeMismatch { expected: self.dtype, found: T::DTYPE });
        }
        Ok(())
    }
}

/// Borrowed bytes from another runtime, described by dtype, shape and (optionally) element strides.
/// Native byte order. Typed access is zero-copy when the memory is aligned for the element type.
#[derive(Debug, Clone, Copy)]
pub struct DynView<'a> {
    bytes: &'a [u8],
    layout: RawLayout,
}

impl<'a> DynView<'a> {
    /// A contiguous row-major view.
    pub fn new(bytes: &'a [u8], dtype: DType, shape: &[usize]) -> Result<Self, DynError> {
        Self::with_strides(bytes, dtype, shape, None, 0)
    }
    /// A view with explicit element strides (negative ones walk backwards) and the element offset of
    /// index [0, 0, ...] (e.g. a column-major, sliced or reversed tensor). Bounds are checked when a
    /// typed view is taken.
    pub fn with_strides(bytes: &'a [u8], dtype: DType, shape: &[usize], strides: Option<&[isize]>, offset: usize) -> Result<Self, DynError> {
        let layout = RawLayout::new(dtype, shape, strides, offset)?;
        layout.check_bytes(bytes.len())?;
        Ok(Self { bytes, layout })
    }
    pub fn dtype(&self) -> DType {
        self.layout.dtype
    }
    pub fn shape(&self) -> &[usize] {
        &self.layout.shape[..self.layout.ndim]
    }
    /// The typed view over the same memory. Errors on a dtype mismatch, misaligned memory
    /// (`to_array` copies instead) or a layout that doesn't fit the bytes.
    pub fn typed<T: DynElement>(&self) -> Result<NdView<'a, T>, DynError> {
        self.layout.check_type::<T>()?;
        let elements = cast_slice::<T>(self.bytes).ok_or(DynError::Unaligned(self.layout.dtype))?;
        let l = &self.layout;
        Ok(NdView::from_parts(elements, &l.shape[..l.ndim], &l.strides[..l.ndim], l.offset)?)
    }
    /// A contiguous owned copy; works for any alignment.
    pub fn to_array(&self) -> Result<DynArray, DynError> {
        let l = &self.layout;
        let size = l.dtype.size_bytes();
        with_dtype!(l.dtype, T => {
            let elements: Vec<T> = self.bytes.chunks_exact(size).map(|c| T::read_raw(c, native_endian())).collect();
            let view = NdView::from_parts(&elements, &l.shape[..l.ndim], &l.strides[..l.ndim], l.offset)?;
            Ok(T::wrap(view.to_owned()))
        }, Err(DynError::Unsupported(l.dtype)))
    }
}

/// Mutable borrowed bytes from another runtime (see [`DynView`]).
#[derive(Debug)]
pub struct DynViewMut<'a> {
    bytes: &'a mut [u8],
    layout: RawLayout,
}

impl<'a> DynViewMut<'a> {
    pub fn new(bytes: &'a mut [u8], dtype: DType, shape: &[usize]) -> Result<Self, DynError> {
        Self::with_strides(bytes, dtype, shape, None, 0)
    }
    pub fn with_strides(bytes: &'a mut [u8], dtype: DType, shape: &[usize], strides: Option<&[isize]>, offset: usize) -> Result<Self, DynError> {
        let layout = RawLayout::new(dtype, shape, strides, offset)?;
        layout.check_bytes(bytes.len())?;
        Ok(Self { bytes, layout })
    }
    pub fn dtype(&self) -> DType {
        self.layout.dtype
    }
    pub fn shape(&self) -> &[usize] {
        &self.layout.shape[..self.layout.ndim]
    }
    /// The typed mutable view over the same memory (see `DynView::typed`).
    pub fn typed_mut<T: DynElement>(&mut self) -> Result<NdViewMut<'_, T>, DynError> {
        self.layout.check_type::<T>()?;
        let dtype = self.layout.dtype;
        let l = self.layout;
        let elements = cast_slice_mut::<T>(self.bytes).ok_or(DynError::Unaligned(dtype))?;
        Ok(NdViewMut::from_parts(elements, &l.shape[..l.ndim], &l.strides[..l.ndim], l.offset)?)
    }
}

fn native_endian() -> Endian {
    if cfg!(target_endian = "little") { Endian::Little } else { Endian::Big }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn typed_access_and_dispatch() {
        let a = DynArray::from_array(NdArray::from_vec(vec![1.5f32, -2.0], &[2]).unwrap());
        assert_eq!(a.dtype(), DType::F32);
        assert_eq!(a.as_array::<f32>().unwrap().as_slice(), [1.5, -2.0]);
        assert_eq!(a.as_array::<f64>().unwrap_err(), DynError::DTypeMismatch { expected: DType::F64, found: DType::F32 });
        assert!(a.clone().into_array::<i16>().is_err());
        for d in DType::ALL {
            match DynArray::zeros(d, &[3, 2]) {
                Ok(z) => assert_eq!((z.dtype(), z.shape(), z.len()), (d, &[3usize, 2][..], 6)),
                Err(e) => assert_eq!((d, e), (DType::I24, DynError::Unsupported(DType::I24))),
            }
        }
    }

    #[test]
    fn casting() {
        let a = DynArray::from_array(NdArray::from_vec(vec![1.7f64, -40_000.0, 3.0], &[3]).unwrap());
        let i = a.cast(DType::I16).unwrap();
        assert_eq!(i.as_array::<i16>().unwrap().as_slice(), [1, -32_768, 3]); // truncates, saturates
        let c = a.cast(DType::ComplexF32).unwrap();
        assert_eq!(c.as_array::<Complex<f32>>().unwrap().as_slice()[0], Complex::new(1.7, 0.0));
        assert_eq!(c.cast(DType::F32), Err(DynError::ComplexToReal(DType::F32)));
    }

    #[test]
    fn byte_serialization_roundtrips_every_dtype() {
        for d in DType::ALL.into_iter().filter(|&d| d != DType::I24) {
            let a = DynArray::from_array(NdArray::from_vec(vec![1.0f64, 2.0, 3.0, 4.0], &[2, 2]).unwrap()).cast(d).unwrap();
            for endian in [Endian::Little, Endian::Big] {
                let bytes = a.to_bytes(endian);
                assert_eq!(bytes.len(), 4 * d.size_bytes());
                assert_eq!(DynArray::from_bytes(&bytes, d, &[2, 2], endian).unwrap(), a, "{d} {endian:?}");
            }
        }
        assert!(matches!(DynArray::from_bytes(&[0; 3], DType::F32, &[1], Endian::Little), Err(DynError::ByteLength { .. })));
    }

    #[test]
    fn zero_copy_views_of_external_memory() {
        let a = DynArray::from_array(NdArray::from_vec(vec![1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0], &[2, 3]).unwrap());
        let bytes = a.as_bytes();
        let view = DynView::new(bytes, DType::F32, &[2, 3]).unwrap();
        let typed = view.typed::<f32>().unwrap();
        assert_eq!(typed.get(&[1, 2]), Some(&6.0));
        // same memory, not a copy
        assert_eq!(typed.as_slice().unwrap().as_ptr().cast::<u8>(), bytes.as_ptr());
        assert!(matches!(view.typed::<f64>(), Err(DynError::DTypeMismatch { .. })));
        // column-major interpretation of the same bytes
        let t = DynView::with_strides(bytes, DType::F32, &[3, 2], Some(&[1, 3]), 0).unwrap();
        assert_eq!(t.typed::<f32>().unwrap().get(&[2, 0]), Some(&3.0));
        // a layout reaching past the bytes is rejected
        assert!(DynView::new(bytes, DType::F32, &[3, 3]).unwrap().typed::<f32>().is_err());
    }

    #[test]
    fn misaligned_memory_is_an_error_but_copyable() {
        // u32 storage is 4-byte aligned; its bytes 4..12 hold the f32s 7.0 and 8.0
        let words = [0, 7.0f32.to_bits(), 8.0f32.to_bits(), 0];
        let storage = DynArray::from_array(NdArray::from_vec(words.to_vec(), &[4]).unwrap());
        let bytes = storage.as_bytes();
        let aligned = DynView::new(&bytes[4..12], DType::F32, &[2]).unwrap();
        assert_eq!(aligned.typed::<f32>().unwrap().to_vec(), [7.0, 8.0]);
        // starting one byte in can never be 4-byte aligned
        let misaligned = DynView::new(&bytes[1..9], DType::F32, &[2]).unwrap();
        assert_eq!(misaligned.typed::<f32>().unwrap_err(), DynError::Unaligned(DType::F32));
        // the copying path works for any alignment
        let mut shifted = [0u8; 9];
        shifted[1..].copy_from_slice(&bytes[4..12]);
        let copied = DynView::new(&shifted[1..], DType::F32, &[2]).unwrap().to_array().unwrap();
        assert_eq!(copied.as_array::<f32>().unwrap().as_slice(), [7.0, 8.0]);
    }

    #[test]
    fn mutable_views_write_through() {
        let mut a = DynArray::from_array(NdArray::from_vec(vec![0i16; 4], &[2, 2]).unwrap());
        let mut bytes = a.to_bytes(native_endian());
        {
            let mut v = DynViewMut::new(&mut bytes, DType::I16, &[2, 2]).unwrap();
            if let Ok(mut typed) = v.typed_mut::<i16>() {
                *typed.get_mut(&[1, 1]).unwrap() = 42;
            } else {
                return; // unaligned Vec<u8> allocation: nothing to test on this platform
            }
        }
        a = DynArray::from_bytes(&bytes, DType::I16, &[2, 2], native_endian()).unwrap();
        assert_eq!(a.as_array::<i16>().unwrap()[&[1, 1][..]], 42);
    }
}
