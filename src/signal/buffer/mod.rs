use std::fmt::Debug;
use std::ops::Index;

#[repr(transparent)]
pub struct Buffer<T> {
    data: Vec<T>
}

#[repr(transparent)]
pub struct BufferView<'a, T> {
    slice: &'a [T],
}

#[repr(transparent)]
pub struct BufferViewMut<'a, T> {
    slice: &'a mut [T],
}

/// Trait for general buffer applications : stored in program memory
pub trait IntoBuffer: PartialEq + Debug + IntoIterator + Index<usize> + AsRef<[Self::Item]> {
    type Item;
}

pub trait FromBuffer: IntoBuffer {}

/// Trait that provides static buffer operations
pub trait BufferOps<T: Buffer> {
    // associated adapter methods
    fn conv(input: &T, impulse: &T, output: &mut T, state: &mut T);
}