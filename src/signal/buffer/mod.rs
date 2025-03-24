/// 1-D Buffer Signal
///

use std::vec::IntoIter;
use super::*;

#[repr(transparent)]
#[derive(Debug, Clone, Hash, Ord, PartialOrd, Eq, PartialEq, Default)]
pub struct Buffer<T: Unit> {
    data: Vec<T>
}

// std::iter implementations ==========================================

impl<T: Unit> IntoIterator for Buffer<T> {
    type Item = T;
    type IntoIter = IntoIter<T>;
    fn into_iter(self) -> Self::IntoIter {
        self.data.into_iter()
    }
}

// std::ops Implementations ===========================================

macro_rules! impl_buffer_binop_and_assign {
    ($trait:ident, $method:ident, $assign_trait:ident, $assign_method:ident, $op:expr) => {
        use std::ops::*;

        // Buffer + Buffer
        impl<T> $trait for Buffer<T>
        where
            T: $trait<Output = T> + Copy,
        {
            type Output = Buffer<T>;
            fn $method(self, rhs: Self) -> Self::Output {
                assert_eq!(self.len(), rhs.len(), "Buffers must match");
                let data = self
                    .data
                    .into_iter()
                    .zip(rhs.data.into_iter())
                    .map($op)
                    .collect();
                Buffer { data }
            }
        }

        // &Buffer + &Buffer
        impl<'a, 'b, T> $trait<&'b Buffer<T>> for &'a Buffer<T>
        where
            T: $trait<Output = T> + Copy,
        {
            type Output = Buffer<T>;
            fn $method(self, rhs: &'b Buffer<T>) -> Self::Output {
                assert_eq!(self.len(), rhs.len(), "Buffers must match");
                let data = self
                    .data
                    .iter()
                    .copied()
                    .zip(rhs.data.iter().copied())
                    .map($op)
                    .collect();
                Buffer { data }
            }
        }

        // Buffer + &Buffer
        impl<'b, T> $trait<&'b Buffer<T>> for Buffer<T>
        where
            T: $trait<Output = T> + Copy,
        {
            type Output = Buffer<T>;
            fn $method(self, rhs: &'b Buffer<T>) -> Self::Output {
                assert_eq!(self.len(), rhs.len(), "Buffers must match");
                let data = self
                    .data
                    .into_iter()
                    .zip(rhs.data.iter().copied())
                    .map($op)
                    .collect();
                Buffer { data }
            }
        }

        // Buffer + &mut Buffer
        impl<'b, T> $trait<&'b mut Buffer<T>> for Buffer<T>
        where
            T: $trait<Output = T> + Copy,
        {
            type Output = Buffer<T>;
            fn $method(self, rhs: &'b mut Buffer<T>) -> Self::Output {
                assert_eq!(self.len(), rhs.len(), "Buffers must match");
                let data = self
                    .data
                    .into_iter()
                    .zip(rhs.data.iter().copied())
                    .map($op)
                    .collect();
                Buffer { data }
            }
        }

        // &Buffer + Buffer
        impl<'a, T> $trait<Buffer<T>> for &'a Buffer<T>
        where
            T: $trait<Output = T> + Copy,
        {
            type Output = Buffer<T>;
            fn $method(self, rhs: Buffer<T>) -> Self::Output {
                assert_eq!(self.len(), rhs.len(), "Buffers must match");
                let data = self
                    .data
                    .iter()
                    .copied()
                    .zip(rhs.data.into_iter())
                    .map($op)
                    .collect();
                Buffer { data }
            }
        }

        // Buffer += Buffer
        impl<T> $assign_trait<Buffer<T>> for Buffer<T>
        where
            T: $trait<Output = T> + Copy,
        {
            fn $assign_method(&mut self, rhs: Buffer<T>) {
                assert_eq!(self.len(), rhs.len());
                for (a, b) in self.data.iter_mut().zip(rhs.data.into_iter()) {
                    *a = $op(*a, b);
                }
            }
        }

        // Buffer += &Buffer
        impl<T> $assign_trait<&Buffer<T>> for Buffer<T>
        where
            T: $trait<Output = T> + Copy,
        {
            fn $assign_method(&mut self, rhs: &Buffer<T>) {
                assert_eq!(self.len(), rhs.len());
                for (a, b) in self.data.iter_mut().zip(rhs.data.iter()) {
                    *a = $op(*a, *b);
                }
            }
        }

        // Buffer += &mut Buffer
        impl<T> $assign_trait<&mut Buffer<T>> for Buffer<T>
        where
            T: $trait<Output = T> + Copy,
        {
            fn $assign_method(&mut self, rhs: &mut Buffer<T>) {
                assert_eq!(self.len(), rhs.len());
                for (a, b) in self.data.iter_mut().zip(rhs.data.iter()) {
                    *a = $op(*a, *b);
                }
            }
        }

        // &mut Buffer += Buffer
        impl<T> $assign_trait<Buffer<T>> for &mut Buffer<T>
        where
            T: $trait<Output = T> + Copy,
        {
            fn $assign_method(&mut self, rhs: Buffer<T>) {
                assert_eq!(self.len(), rhs.len());
                for (a, b) in self.data.iter_mut().zip(rhs.data.into_iter()) {
                    *a = $op(*a, b);
                }
            }
        }

        // &mut Buffer += &Buffer
        impl<T> $assign_trait<&Buffer<T>> for &mut Buffer<T>
        where
            T: $trait<Output = T> + Copy,
        {
            fn $assign_method(&mut self, rhs: &Buffer<T>) {
                assert_eq!(self.len(), rhs.len());
                for (a, b) in self.data.iter_mut().zip(rhs.data.iter()) {
                    *a = $op(*a, *b);
                }
            }
        }

        // &mut Buffer += &mut Buffer
        impl<T> $assign_trait<&mut Buffer<T>> for &mut Buffer<T>
        where
            T: $trait<Output = T> + Copy,
        {
            fn $assign_method(&mut self, rhs: &mut Buffer<T>) {
                assert_eq!(self.len(), rhs.len());
                for (a, b) in self.data.iter_mut().zip(rhs.data.iter()) {
                    *a = $op(*a, *b);
                }
            }
        }
    };
}


// std::io Implementations ============================================

// Signal Trait Implementations =======================================

impl<T: Unit> Signal for Buffer<T> {
    type Sample = T;
    fn view(&self) -> &[Self::Sample] {
        self.data.as_slice()
    }
}

impl<T: Unit> SignalMut for Buffer<T> {
    fn view_mut(&mut self) -> &mut [Self::Sample] {
        self.data.as_mut_slice()
    }
}

impl<T: Unit> SignalOwned for Buffer<T> {
    type Container = Vec<T>;
    fn as_container(&self) -> &Self::Container {
        &self.data
    }
    fn as_container_mut(&mut self) -> &mut Self::Container {
        &mut self.data
    }
    fn into_container(self) -> Self::Container {
        self.data
    }
}

impl<T: Unit> SignalResizable for Buffer<T> {
    fn resize(&mut self, new_len: usize, fill_value: Self::Sample) {
        todo!()
    }
    fn clear(&mut self) {
        todo!()
    }
    fn append(&mut self, value: Self::Sample) {
        todo!()
    }
}

// impl<T: Unit> SignalStream for Buffer<T> {}
// impl<T: Unit> SignalOps for Buffer<T> {}