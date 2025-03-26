
/// (std::ops method, method, assign_method, error_type, op)
macro_rules! impl_buffer_sigop_and_assign {
    ($atomic_method:ident, $method:ident, $assign_method:ident, $error_variant:ident, $op:expr) => {
        // &Buffer + &[T]
        fn $method<U: AsRef<[Self::Sample]>>(&self, rhs: U) -> Result<Self, SignalOpsError> {
            let rhs = rhs.as_ref();
            if (self.len() != rhs.len()) {
                return Err(SignalOpsError::$error_variant(self.len(), rhs.len()));
            }
            let data = self
                .data
                .iter()
                .copied()
                .zip(rhs.iter().copied())
                .map($op)
                .collect();
            Ok(Self { data })
        }
        
        // &mut Buffer += &[T]
        fn $assign_method<U: AsRef<[Self::Sample]>>(&mut self, rhs: U) -> Result<(), SignalOpsError> {
            let rhs = rhs.as_ref();
            if (self.len() != rhs.len()) {
                return Err(SignalOpsError::$error_variant(self.len(), rhs.len()));
            }
            for (a, b) in self.data.iter_mut().zip(rhs.iter().copied()) {
                *a = (*a).$atomic_method(b);
            }
            Ok(())
        }
    };
}

/// (trait, method, assign_trait, assign_method, op)
macro_rules! impl_buffer_binop_and_assign_scalar {
    ($trait:ident, $method:ident, $assign_trait:ident, $assign_method:ident) => {
        // Buffer + T
        impl<T> $trait<T> for Buffer<T>
        where
            T: Unit
        {
            type Output = Buffer<T>;
            fn $method(self, rhs: T) -> Self::Output {
                let data = self
                    .data
                    .into_iter()
                    .map(|x| x.$method(rhs))
                    .collect();
                Buffer { data }
            }
        }
        
        // Buffer + &T
        impl<T> $trait<&T> for Buffer<T>
        where
            T: Unit
        {
            type Output = Buffer<T>;
            fn $method(self, rhs: &T) -> Self::Output {
                let data = self
                    .data
                    .into_iter()
                    .map(|x| x.$method(*rhs))
                    .collect();
                Buffer { data }
            }
        }
        
        // &Buffer + T
        impl<'a, T> $trait<T> for &'a Buffer<T>
        where
            T: Unit,
        {
            type Output = Buffer<T>;
            fn $method(self, rhs: T) -> Self::Output {
                let data = self
                    .data
                    .iter()
                    .copied()
                    .map(|x| x.$method(rhs))
                    .collect();
                Buffer { data }
            }
        }
        
        // &Buffer + &T
        impl<'a, T> $trait<&T> for &'a Buffer<T>
        where
            T: Unit,
        {
            type Output = Buffer<T>;
            fn $method(self, rhs: &T) -> Self::Output {
                let data = self
                    .data
                    .iter()
                    .copied()
                    .map(|x| x.$method(*rhs))
                    .collect();
                Buffer { data }
            }
        }
        
        // Buffer += T
        impl<T> $assign_trait<T> for Buffer<T>
        where
            T: Unit
        {
            fn $assign_method(&mut self, rhs: T) {
                for a in self.data.iter_mut() {
                    *a = a.$method(rhs);
                }
            }
        }
        
        // Buffer += &T
        impl<T> $assign_trait<&T> for Buffer<T>
        where
            T: Unit
        {
            fn $assign_method(&mut self, rhs: &T) {
                for a in self.data.iter_mut() {
                    *a = a.$method(*rhs);
                }
            }
        }
    };
}

/// (trait, method, op)
macro_rules! impl_buffer_unop {
    ($trait:ident, $method:ident) => {
        // Buffer
        impl<T> $trait for Buffer<T>
        where
            T: Unit,
        {
            type Output = Buffer<T>;
            fn $method(self) -> Self::Output {
                let data = self
                    .data
                    .into_iter()
                    .map(|x| x.$method())
                    .collect();
                Buffer { data }
            }
        }  
        
        // &Buffer
        impl<'a, T> $trait for &'a Buffer<T>
        where
            T: Unit,
        {
            type Output = Buffer<T>;
            fn $method(self) -> Self::Output {
                let data = self
                    .data
                    .iter()
                    .copied()
                    .map(|x| x.$method())
                    .collect();
                Buffer { data }
            }
        }  
    };
}