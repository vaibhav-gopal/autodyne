//! Where an [`NdArray`](super::NdArray)'s elements live.
//!
//! An array is a layout over *storage*: anything that holds its elements as one contiguous run.
//! `Vec<T>` (the default) and `Box<[T]>` are owned and writable; `Arc<[T]>` is shared (clones are
//! cheap, writing copies first: [`NdArray::make_mut`](super::NdArray::make_mut)); memory owned by
//! another library comes in through [`ForeignBuffer`](crate::dlpack::ForeignBuffer) (DLPack). Every
//! kind gives the same views, so every operation works on all of them without copying.

use std::sync::Arc;

/// Contiguous element memory an `NdArray` can be built on.
pub trait Storage {
    type Elem;
    fn as_slice(&self) -> &[Self::Elem];
    /// Whether the elements may be written in place (false for shared storage, which copies on
    /// write instead).
    const WRITABLE: bool = true;
}

/// Storage whose elements can be written in place.
pub trait StorageMut: Storage {
    fn as_mut_slice(&mut self) -> &mut [Self::Elem];
}

impl<T> Storage for Vec<T> {
    type Elem = T;
    fn as_slice(&self) -> &[T] {
        self
    }
}
impl<T> StorageMut for Vec<T> {
    fn as_mut_slice(&mut self) -> &mut [T] {
        self
    }
}

impl<T> Storage for Box<[T]> {
    type Elem = T;
    fn as_slice(&self) -> &[T] {
        self
    }
}
impl<T> StorageMut for Box<[T]> {
    fn as_mut_slice(&mut self) -> &mut [T] {
        self
    }
}

impl<T> Storage for Arc<[T]> {
    type Elem = T;
    fn as_slice(&self) -> &[T] {
        self
    }
    const WRITABLE: bool = false;
}
