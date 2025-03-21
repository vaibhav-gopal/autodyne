/// using extension traits to represent buffers via vectors, slices and arrays
use super::*;
use delegate::delegate;

macro_rules! impl_buffer_traits {
    ($ty:ty) => {
        impl<T: PrimitiveUnit> Buffer for $ty {
            type Item = T;
        }
        impl<T: PrimitiveUnit> Buffer for &$ty {
            type Item = T;
        }
        impl<T: PrimitiveUnit> Buffer for &mut $ty {
            type Item = T;
        }
        impl<T: PrimitiveUnit> BufferMut for $ty {}
        impl<T: PrimitiveUnit> BufferMut for &mut $ty {}
    };
}
macro_rules! impl_buffer_traits_sized {
    ($ty:ty) => {
        impl<T: PrimitiveUnit, const N: usize> Buffer for $ty {
            type Item = T;
        }
        impl<T: PrimitiveUnit, const N: usize> Buffer for &$ty {
            type Item = T;
        }
        impl<T: PrimitiveUnit, const N: usize> Buffer for &mut $ty {
            type Item = T;
        }
        impl<T: PrimitiveUnit, const N: usize> BufferMut for $ty {}
        impl<T: PrimitiveUnit, const N: usize> BufferMut for &mut $ty {}
    };
}

impl_buffer_traits!(Vec<T>);
impl_buffer_traits!([T]);
impl_buffer_traits_sized!([T; N]);
