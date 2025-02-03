use std::ops::{Add, Sub, Mul, Div, Neg};
use std::fmt::{Debug};

mod markers;
pub mod complex;
pub mod real;
pub mod real_fixed;

// Currently using the newtype and generic/impl trait pattern
// only use static dispatch and zero-cost abstractions (no dynamic dispatch)
// avoid runtime overheads
// TODO: SIMD!!!
// TODO: Specialization macro? (for T specific associated constants on traits)
// TODO: Delegation macro? (for underlying type methods on newtypes)

// GENERAL =========================================================================================
/// Trait for general number operations
pub trait Unit: Add + Sub + Mul + Div + Neg + PartialEq + Copy + Debug{}

/// Marker trait for identifying single field types ; used by units composed of other units
pub trait PrimitiveUnit: Unit {}

/// Trait for data containing integer number operations
/// Defines minimum operations that can apply to any integer and return a valid, usable result
pub trait IntUnit: Unit + PartialOrd {
    fn zero() -> Self;
    fn one() -> Self;
    fn pow(self, n: Self) -> Self;
    fn signum(self) -> Self;
    fn abs(self) -> Self;
    fn max(self, other: Self) -> Self;
    fn min(self, other: Self) -> Self;
    fn clamp(self, min: Self, max: Self) -> Self;
}

/// Trait for data containing real number operations
/// Defines minimum operations that can apply to any real number and return a valid, usable result
pub trait RealUnit: IntUnit {
    const NAN: Self;
    const INFINITY: Self;
    const NEG_INFINITY: Self;
    const EPSILON: Self;
    const MANTISSA_DIGITS: u32;
    const PI: Self;
    const E: Self;
    const TAU: Self;
    fn recip(self) -> Self;
    fn floor(self) -> Self;
    fn ceil(self) -> Self;
    fn round(self) -> Self;
    fn trunc(self) -> Self;
    fn fract(self) -> Self;
    fn powf(self, n: Self) -> Self;
    fn sqrt(self) -> Self;
    fn cbrt(self) -> Self;
    fn exp(self) -> Self;
    fn ln(self) -> Self;
    fn log(self, base: Self) -> Self;
    fn log2(self) -> Self;
    fn log10(self) -> Self;
    fn hypot(self, other: Self) -> Self;
    fn sin(self) -> Self;
    fn cos(self) -> Self;
    fn tan(self) -> Self;
    fn asin(self) -> Self;
    fn acos(self) -> Self;
    fn atan(self) -> Self;
    fn atan2(self, other: Self) -> Self;
    fn sinh(self) -> Self;
    fn cosh(self) -> Self;
    fn tanh(self) -> Self;
    fn asinh(self) -> Self;
    fn acosh(self) -> Self;
    fn atanh(self) -> Self;
    fn to_deg(self) -> Self;
    fn to_rad(self) -> Self;
    fn is_nan(&self) -> bool;
    fn is_inf(&self) -> bool;
}

/// Trait for data containing complex number operations
pub trait ComplexUnit<T>: Unit {
    /// Return a real unit
    fn re() -> Self;
    /// Return an imaginary unit
    fn im() -> Self;
    /// Square of the norm ; re^2 + im^2
    fn norm_sqr(&self) -> T;
    /// Multiply by a constant/scalar
    fn scale(&self, k: T) -> Self;
    /// Divide by a constant/scalar
    fn unscale(&self, k: T) -> Self;
    /// Return complex conjugate
    fn conj(&self) -> Self;
    /// Return complex inverse
    fn inv(&self) -> Self;
    /// Get magnitude / abs of complex number
    fn norm(&self) -> T;
    /// Calculate the principal argument
    fn arg(&self) -> T;
    /// Get polar form representation
    fn to_polar(&self) -> (T, T);
    /// Get complex number from polar form
    fn from_polar(r: T, theta: T) -> Self;
    /// Compute e^(self)
    fn exp(&self) -> Self;
}
