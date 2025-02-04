/// using extension traits to represent buffers via vectors, slices and arrays
use super::*;
use delegate::delegate;

impl<T: PrimitiveUnit> Buffer for Vec<T> {}
impl<T: PrimitiveUnit> Buffer for &Vec<T> {}
impl<T: PrimitiveUnit> Buffer for &mut Vec<T> {}
impl<T: PrimitiveUnit> Buffer for [T] {}
impl<T: PrimitiveUnit> Buffer for &[T] {}
impl<T: PrimitiveUnit> Buffer for &mut [T] {}
impl<T: PrimitiveUnit, const N: usize> Buffer for [T; N] {}
impl<T: PrimitiveUnit, const N: usize> Buffer for &[T; N] {}
impl<T: PrimitiveUnit, const N: usize> Buffer for &mut [T; N] {}
