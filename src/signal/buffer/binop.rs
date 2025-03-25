/// (trait, method, assign_trait, assign_method, op)
macro_rules! impl_buffer_binop_and_assign {
    ($trait:ident, $method:ident, $assign_trait:ident, $assign_method:ident, $op:expr) => {
        // Buffer + &[T]
        impl<T, Rhs> $trait<Rhs> for Buffer<T>
        where
            T: Unit,
            Rhs: AsRef<[T]>,
        {
            type Output = Buffer<T>;
            fn $method(self, rhs: Rhs) -> Self::Output {
                let rhs = rhs.as_ref();
                assert_eq!(self.len(), rhs.len(), "Buffers must match");
                let data = self
                    .data
                    .into_iter()
                    .zip(rhs.iter().copied())
                    .map($op)
                    .collect();
                Buffer { data }
            }
        }

        // &Buffer + &[T]
        impl<'a, T, Rhs> $trait<Rhs> for &'a Buffer<T>
        where
            T: Unit,
            Rhs: AsRef<[T]>,
        {
            type Output = Buffer<T>;
            fn $method(&self, rhs: Rhs) -> Self::Output {
                let rhs = rhs.as_ref();
                assert_eq!(self.len(), rhs.len(), "Buffers must match");
                let data = self
                    .data
                    .iter()
                    .copied()
                    .zip(rhs.iter().copied())
                    .map($op)
                    .collect();
                Buffer { data }
            }
        }
        
        // &mut Buffer + &[T]
        impl<'a, T, Rhs> $trait<Rhs> for &mut'a Buffer<T>
        where
            T: Unit,
            Rhs: AsRef<[T]>,
        {
            type Output = Buffer<T>;
            fn $method(&self, rhs: Rhs) -> Self::Output {
                let rhs = rhs.as_ref();
                assert_eq!(self.len(), rhs.len(), "Buffers must match");
                let data = self
                    .data
                    .iter()
                    .copied()
                    .zip(rhs.iter().copied())
                    .map($op)
                    .collect();
                Buffer { data }
            }
        }
        
        // Buffer += &[T]
        impl<T, Rhs> $assign_trait<Rhs> for Buffer<T>
        where
            T: Unit,
            Rhs: AsRef<[T]>
        {
            fn $assign_method(&mut self, rhs: Rhs) {
                let rhs = rhs.as_ref();
                assert_eq!(self.len(), rhs.len(), "Buffers must match");
                for (a, b) in self.data.iter_mut().zip(rhs.iter().copied()) {
                    *a = (*a).$method(b);
                }
            }
        }
        
        // &mut Buffer += &[T]
        impl<T, Rhs> $assign_trait<Rhs> for &mut Buffer<T>
        where
            T: Unit,
            Rhs: AsRef<[T]>,
        {
            fn $assign_method(&mut self, rhs: Rhs) {
                let rhs = rhs.as_ref();
                assert_eq!(self.len(), rhs.len(), "Buffers must match");
                for (a, b) in self.data.iter_mut().zip(rhs.iter().copied()) {
                    *a = (*a).$method(b);
                }
            }
        }
        
    };
}

/// (trait, method, assign_trait, assign_method, op)
macro_rules! impl_buffer_binop_and_assign_scalar {
    () => {
        // Buffer + &[T]
        impl<T, Rhs> $trait<Rhs> for Buffer<T>
        where
            T: Unit,
            Rhs: AsRef<[T]>,
        {
            type Output = Buffer<T>;
            fn $method(self, rhs: Rhs) -> Self::Output {
                let rhs = rhs.as_ref();
                assert_eq!(self.len(), rhs.len(), "Buffers must match");
                let data = self
                    .data
                    .into_iter()
                    .zip(rhs.iter().copied())
                    .map($op)
                    .collect();
                Buffer { data }
            }
        }  
    };
}

/// (trait, method, op)
macro_rules! impl_buffer_unop {
    () => {
        // Buffer + &[T]
        impl<T, Rhs> $trait<Rhs> for Buffer<T>
        where
            T: Unit,
            Rhs: AsRef<[T]>,
        {
            type Output = Buffer<T>;
            fn $method(self, rhs: Rhs) -> Self::Output {
                let rhs = rhs.as_ref();
                assert_eq!(self.len(), rhs.len(), "Buffers must match");
                let data = self
                    .data
                    .into_iter()
                    .zip(rhs.iter().copied())
                    .map($op)
                    .collect();
                Buffer { data }
            }
        }  
    };
}

