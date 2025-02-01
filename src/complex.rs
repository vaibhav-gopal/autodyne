use crate::numbers::{};

/// Complex number in Cartesian form
/// Usually you'll want to use Complex32 and Complex64 instead
pub struct Complex<T: Num<T>>{
    /// Real portion
    pub re: T,
    /// Complex portion
    pub im: T,
}

pub type Complex32 = Complex<f32>;
pub type Complex64 = Complex<f64>;
pub type Complex32i = Complex<i32>;

impl<T: Num<T>> Complex<T> {
    /// Create a new Complex number
    pub fn new(re: T, im: T) -> Complex<T> {
        Complex {re, im}
    }

    /// Create an imaginary unit
    pub fn i() -> Complex<T> {
        Complex::new(T::zero(), T::one())
    }


}
