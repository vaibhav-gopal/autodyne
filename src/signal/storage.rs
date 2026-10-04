//! Where an [`NdArray`](super::NdArray)'s elements live.
//!
//! An array is a layout over *storage*: anything that holds its elements as one contiguous run.
//! `Vec<T>` (the default) and `Box<[T]>` are owned and writable; `Arc<[T]>` is shared (clones are
//! cheap, writing copies first: [`NdArray::make_mut`](super::NdArray::make_mut)); memory owned by
//! another library comes in through [`ForeignBuffer`](crate::dlpack::ForeignBuffer) (DLPack). Every
//! kind gives the same views, so every operation works on all of them without copying.

use std::sync::Arc;

/// Contiguous element memory an `NdArray` can be built on.
///
/// # Safety
/// Views and DLPack exports keep raw pointers into the storage, so implementations promise:
/// - `as_slice` returns the same memory every time while the storage is not mutated or dropped
///   (moving the storage value must not move the elements: heap memory, as in `Vec` or `Box`);
/// - `as_raw_ptr` points at the first element of that memory, and when `WRITABLE` is true it carries
///   permission to write all of it (derive it from a mutable borrow, not from `as_slice`).
pub unsafe trait Storage {
    /// The element type.
    type Elem;
    /// The elements.
    fn as_slice(&self) -> &[Self::Elem];
    /// A raw pointer to the first element, for handing the memory to foreign code (DLPack). Writing
    /// through it is allowed only when [`WRITABLE`](Self::WRITABLE).
    fn as_raw_ptr(&mut self) -> *mut Self::Elem;
    /// Whether the elements may be written in place (false for shared storage, which copies on
    /// write instead).
    const WRITABLE: bool = true;
}

/// Storage whose elements can be written in place.
pub trait StorageMut: Storage {
    /// The elements, mutably.
    fn as_mut_slice(&mut self) -> &mut [Self::Elem];
}

// SAFETY: heap memory that stays put while the Vec isn't mutated; the raw pointer comes from a
// mutable borrow
unsafe impl<T> Storage for Vec<T> {
    type Elem = T;
    fn as_slice(&self) -> &[T] {
        self
    }
    fn as_raw_ptr(&mut self) -> *mut T {
        self.as_mut_ptr()
    }
}
impl<T> StorageMut for Vec<T> {
    fn as_mut_slice(&mut self) -> &mut [T] {
        self
    }
}

// SAFETY: as for Vec
unsafe impl<T> Storage for Box<[T]> {
    type Elem = T;
    fn as_slice(&self) -> &[T] {
        self
    }
    fn as_raw_ptr(&mut self) -> *mut T {
        self.as_mut_ptr()
    }
}
impl<T> StorageMut for Box<[T]> {
    fn as_mut_slice(&mut self) -> &mut [T] {
        self
    }
}

// SAFETY: heap memory shared by the Arcs, never moved; not writable, so the pointer may be read-only
unsafe impl<T> Storage for Arc<[T]> {
    type Elem = T;
    fn as_slice(&self) -> &[T] {
        self
    }
    fn as_raw_ptr(&mut self) -> *mut T {
        self.as_ptr().cast_mut()
    }
    const WRITABLE: bool = false;
}
