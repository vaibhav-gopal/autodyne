mod numbers;
pub use numbers::*;

/// Complex number in Cartesian form
/// Usually you'll want to use Complex32 and Complex64 instead
pub struct Complex<T: Number>{
    /// Real portion
    pub re: T,
    /// Complex portion
    pub im: T,
}

pub type Complex16 = Complex<Real16>;
pub type Complex32 = Complex<Real32>;
pub type Complex64 = Complex<Real64>;
pub type Complex128 = Complex<Real128>;
pub type ComplexF32 = Complex<RealF32>;
pub type ComplexF64 = Complex<RealF64>;

#[macro_export]
macro_rules! complex {
    () => {
        Complex::default()
    };
    ($re:expr, $im: expr) => {
        Complex::new($re, $im)
    };
}
#[macro_export]
macro_rules! re {
    () => {
        Complex::re()
    };
    ($re:expr) => {
        Complex::re().scale($re)
    };
}
#[macro_export]
macro_rules! im {
    () => {
        Complex::im()
    };
    ($im:expr) => {
        Complex::im().scale($im)
    };
}

impl<T: Number> Default for Complex<T> {
    /// Create an empty Complex number
    fn default() -> Self {
        Complex::new(T::zero(), T::zero())
    }
}

impl<T: Number> Complex<T> {
    /// Create a new Complex number
    pub fn new(re: T, im: T) -> Complex<T> {
        Complex {re, im}
    }
    
    /// Create a real unit
    pub fn re() -> Complex<T> {
        complex!(T::one(), T::zero())
    }

    /// Create an imaginary unit
    pub fn im() -> Complex<T> {
        complex!(T::zero(), T::one())
    }

    /// Square of the norm ; re^2 + im^2
    pub fn norm_sqr(&self) -> T {
        self.re * self.re + self.im * self.im;
    }
    
    /// Multiply by a constant/scalar
    pub fn scale(&self, k: T) -> Complex<T> {
        complex!(self.re * k, self.im * k)
    }
    
    /// Divide by a constant/scalar
    pub fn unscale(&self, k: T) -> Complex<T> {
        complex!(self.re / k, self.im * k)
    }
    
    /// Return complex conjugate
    pub fn conj(&self) -> Complex<T> {
        complex!(self.re, -self.im)
    }
    
    /// Return complex inverse
    pub fn inv(&self) -> Complex<T> {
        let norm_sqr = self.norm_sqr();
        complex!(self.re / norm_sqr, -self.im / norm_sqr)
    }
    
    /// Get magnitude / abs of complex number
    pub fn norm(&self) -> T {
        self.im.atan2
    }
}
