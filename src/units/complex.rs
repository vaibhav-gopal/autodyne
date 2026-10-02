use std::ops::{Add, AddAssign, Div, DivAssign, Mul, MulAssign, Neg, Sub, SubAssign};
use super::*;

/// Complex number over any Float ; the basis for FFTs and IQ (de)modulation.
/// Deliberately not a `Unit`: it has no bit-level `PhysicalRepr` and no total order.
/// `repr(C)`: laid out as `re` then `im`, the same as C99 / NumPy complex numbers, so buffers of it
/// can be shared with other runtimes without copying.
#[derive(Debug, Clone, Copy, PartialEq, Default)]
#[repr(C)]
pub struct Complex<T: Float> {
    pub re: T,
    pub im: T,
}

impl<T: Float> Complex<T> {
    pub const fn new(re: T, im: T) -> Self {
        Self { re, im }
    }
    pub fn zero() -> Self {
        Self::new(T::_ZERO, T::_ZERO)
    }
    pub fn one() -> Self {
        Self::new(T::_ONE, T::_ZERO)
    }
    /// The imaginary unit i
    pub fn i() -> Self {
        Self::new(T::_ZERO, T::_ONE)
    }
    /// r * e^(i*theta)
    pub fn from_polar(r: T, theta: T) -> Self {
        let (s, c) = theta._sin_cos();
        Self::new(r * c, r * s)
    }
    /// e^(i*theta) ; a unit phasor, the building block of oscillators and the DFT
    pub fn cis(theta: T) -> Self {
        Self::from_polar(T::_ONE, theta)
    }
    pub fn conj(self) -> Self {
        Self::new(self.re, -self.im)
    }
    /// |z|^2 ; cheaper than norm() when only comparing magnitudes
    pub fn norm_sqr(self) -> T {
        self.re * self.re + self.im * self.im
    }
    /// |z|
    pub fn norm(self) -> T {
        self.re._hypot(self.im)
    }
    /// Phase angle in (-pi, pi]
    pub fn arg(self) -> T {
        self.im._atan2(self.re)
    }
    /// (|z|, arg z)
    pub fn to_polar(self) -> (T, T) {
        (self.norm(), self.arg())
    }
    /// e^z
    pub fn exp(self) -> Self {
        Self::from_polar(self.re._exp(), self.im)
    }
    pub fn recip(self) -> Self {
        let d = self.norm_sqr();
        Self::new(self.re / d, -self.im / d)
    }
    /// Principal square root (non-negative real part; on the branch cut, the sign of `im` picks the
    /// side, as in NumPy).
    pub fn sqrt(self) -> Self {
        let r = self.norm();
        let half = T::_lit(0.5);
        let re = ((r + self.re) * half)._sqrt();
        let im = ((r - self.re) * half)._sqrt();
        let negative = self.im.to_f64().is_some_and(f64::is_sign_negative);
        Self::new(re, if negative { -im } else { im })
    }
    /// Principal natural logarithm: `ln|z| + i arg z`.
    pub fn ln(self) -> Self {
        Self::new(self.norm()._ln(), self.arg())
    }
    /// `self` raised to a real power (principal branch); `0^e` is 0 for `e > 0`.
    pub fn powf(self, e: T) -> Self {
        if self.re == T::_ZERO && self.im == T::_ZERO {
            return if e == T::_ZERO { Self::one() } else { Self::zero() };
        }
        (self.ln() * e).exp()
    }
    /// `self` raised to an integer power, by repeated squaring.
    pub fn powi(self, n: i32) -> Self {
        let mut base = if n < 0 { self.recip() } else { self };
        let mut e = n.unsigned_abs();
        let mut acc = Self::one();
        while e > 0 {
            if e & 1 == 1 {
                acc *= base;
            }
            base = base * base;
            e >>= 1;
        }
        acc
    }
    /// Principal arc sine: `-i ln(i z + sqrt(1 - z²))`.
    pub fn asin(self) -> Self {
        let i = Self::i();
        let root = (Self::one() - self * self).sqrt();
        -(i * (i * self + root).ln())
    }
}

impl<T: Float> From<T> for Complex<T> {
    fn from(re: T) -> Self {
        Self::new(re, T::_ZERO)
    }
}

// Complex (op) Complex ============================================================================

impl<T: Float> Add for Complex<T> {
    type Output = Self;
    fn add(self, rhs: Self) -> Self {
        Self::new(self.re + rhs.re, self.im + rhs.im)
    }
}

impl<T: Float> Sub for Complex<T> {
    type Output = Self;
    fn sub(self, rhs: Self) -> Self {
        Self::new(self.re - rhs.re, self.im - rhs.im)
    }
}

impl<T: Float> Mul for Complex<T> {
    type Output = Self;
    fn mul(self, rhs: Self) -> Self {
        Self::new(self.re * rhs.re - self.im * rhs.im, self.re * rhs.im + self.im * rhs.re)
    }
}

impl<T: Float> Div for Complex<T> {
    type Output = Self;
    fn div(self, rhs: Self) -> Self {
        let d = rhs.norm_sqr();
        Self::new(
            (self.re * rhs.re + self.im * rhs.im) / d,
            (self.im * rhs.re - self.re * rhs.im) / d,
        )
    }
}

impl<T: Float> Neg for Complex<T> {
    type Output = Self;
    fn neg(self) -> Self {
        Self::new(-self.re, -self.im)
    }
}

// Complex (op) scalar =============================================================================

impl<T: Float> Mul<T> for Complex<T> {
    type Output = Self;
    fn mul(self, rhs: T) -> Self {
        Self::new(self.re * rhs, self.im * rhs)
    }
}

impl<T: Float> Div<T> for Complex<T> {
    type Output = Self;
    fn div(self, rhs: T) -> Self {
        Self::new(self.re / rhs, self.im / rhs)
    }
}

// Assign variants =================================================================================

macro_rules! impl_complex_assign {
    ($assign_trait:ident, $assign_method:ident, $op:tt, $Rhs:ty) => {
        impl<T: Float> $assign_trait<$Rhs> for Complex<T> {
            fn $assign_method(&mut self, rhs: $Rhs) {
                *self = *self $op rhs;
            }
        }
    };
}

impl_complex_assign!(AddAssign, add_assign, +, Complex<T>);
impl_complex_assign!(SubAssign, sub_assign, -, Complex<T>);
impl_complex_assign!(MulAssign, mul_assign, *, Complex<T>);
impl_complex_assign!(DivAssign, div_assign, /, Complex<T>);
impl_complex_assign!(MulAssign, mul_assign, *, T);
impl_complex_assign!(DivAssign, div_assign, /, T);

#[cfg(test)]
mod tests {
    use super::*;

    type C = Complex<f64>;

    fn close(a: C, b: C) -> bool {
        (a - b).norm() < 1e-12
    }

    #[test]
    fn arithmetic() {
        let a = C::new(1.0, 2.0);
        let b = C::new(3.0, -4.0);
        assert_eq!(a + b, C::new(4.0, -2.0));
        assert_eq!(a - b, C::new(-2.0, 6.0));
        // (1+2i)(3-4i) = 3 - 4i + 6i - 8i^2 = 11 + 2i
        assert_eq!(a * b, C::new(11.0, 2.0));
        assert!(close((a * b) / b, a));
        assert!(close(a * a.recip(), C::one()));
    }

    #[test]
    fn i_squared_is_minus_one() {
        assert_eq!(C::i() * C::i(), C::new(-1.0, 0.0));
    }

    #[test]
    fn polar_roundtrip() {
        let z = C::new(-3.0, 4.0);
        let (r, theta) = z.to_polar();
        assert!((r - 5.0).abs() < 1e-12);
        assert!(close(C::from_polar(r, theta), z));
    }

    #[test]
    fn eulers_identity() {
        // e^(i*pi) + 1 = 0
        assert!(close(C::new(0.0, f64::_PI).exp() + C::one(), C::zero()));
        assert!(close(C::cis(f64::_PI / 2.0), C::i()));
    }

    #[test]
    fn conj_and_norm() {
        let z = C::new(3.0, 4.0);
        assert_eq!(z * z.conj(), C::new(25.0, 0.0));
        assert_eq!(z.norm_sqr(), 25.0);
    }

    #[test]
    fn assign_ops() {
        let mut z = C::new(1.0, 1.0);
        z *= C::i();
        assert_eq!(z, C::new(-1.0, 1.0));
        z *= 2.0;
        assert_eq!(z, C::new(-2.0, 2.0));
    }
}
