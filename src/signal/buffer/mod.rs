// /// 1-D Buffer Signal
// use std::vec::IntoIter;
// use std::ops::{AddAssign, DivAssign, MulAssign, SubAssign};
// use super::*;
// 
// #[macro_use]
// mod binop;
// 
// #[repr(transparent)]
// #[derive(Debug, Clone, Hash, Ord, PartialOrd, Eq, PartialEq, Default)]
// pub struct Buffer<T: Unit> {
//     data: Vec<T>
// }
// 
// // super trait implementations ==========================================
// 
// impl<T: Unit> IntoIterator for Buffer<T> {
//     type Item = T;
//     type IntoIter = IntoIter<T>;
//     fn into_iter(self) -> Self::IntoIter {
//         self.data.into_iter()
//     }
// }
// 
// impl<T: Unit> AsRef<[T]> for Buffer<T> {
//     fn as_ref(&self) -> &[T] { self.data.as_slice() }
// }
// 
// impl<T: Unit> AsMut<[T]> for Buffer<T> {
//     fn as_mut(&mut self) -> &mut [T] { self.data.as_mut_slice() }
// }
// 
// impl<T: Unit> Deref for Buffer<T> {
//     type Target = [T];
//     fn deref(&self) -> &Self::Target { self.data.as_slice() }
// }
// 
// impl<T: Unit> DerefMut for Buffer<T> {
//     fn deref_mut(&mut self) -> &mut Self::Target { self.data.as_mut_slice() }
// }
// 
// // std::ops and SignalOps Implementations ===========================================
// 
// impl<T: Unit> SignalOps for Buffer<T> {
//     impl_buffer_sigop_and_assign!(add, sig_add, sig_add_assign, BufferMismatch, |(x, y)| x + y);
//     impl_buffer_sigop_and_assign!(sub, sig_sub, sig_sub_assign, BufferMismatch, |(x, y)| x - y);
//     impl_buffer_sigop_and_assign!(mul, sig_mul, sig_mul_assign, BufferMismatch, |(x, y)| x * y);
//     impl_buffer_sigop_and_assign!(div, sig_div, sig_div_assign, BufferMismatch, |(x, y)| x / y);
// }
// impl_buffer_binop_and_assign_scalar!(Add, add, AddAssign, add_assign);
// impl_buffer_binop_and_assign_scalar!(Sub, sub, SubAssign, sub_assign);
// impl_buffer_binop_and_assign_scalar!(Mul, mul, MulAssign, mul_assign);
// impl_buffer_binop_and_assign_scalar!(Div, div, DivAssign, div_assign);
// impl_buffer_unop!(Neg, neg);
// 
// // std::io Implementations ============================================
// 
// // Signal Trait Implementations =======================================
// 
// impl<T: Unit> Signal for Buffer<T> {
//     type Sample = T;
// }
// 
// impl<T: Unit> SignalMut for Buffer<T> {
// }
// 
// impl<T: Unit> SignalOwned for Buffer<T> {
//     type Container = Vec<T>;
//     fn as_container(&self) -> &Self::Container {
//         &self.data
//     }
//     fn as_container_mut(&mut self) -> &mut Self::Container {
//         &mut self.data
//     }
//     fn into_container(self) -> Self::Container {
//         self.data
//     }
// }
// 
// impl<T: Unit> SignalResizable for Buffer<T> {
//     fn resize(&mut self, new_len: usize, fill_value: Self::Sample) {
//         todo!()
//     }
//     fn clear(&mut self) {
//         todo!()
//     }
//     fn append(&mut self, value: Self::Sample) {
//         todo!()
//     }
// }
// 
// // impl<T: Unit> SignalStream for Buffer<T> {}