use std::ops::{Add, Sub, Mul, Div, Neg};
use std::fmt::{Debug};

mod markers;
pub mod real;
pub mod realf;
pub mod complex;

// Currently using the newtype and generic/impl trait pattern
// only use static dispatch and zero-cost abstractions (no dynamic dispatch)
// avoid runtime overheads
// TODO: SIMD!!!

// GENERAL =========================================================================================
/// Trait for general data operations
pub trait Unit: Add + Sub + Mul + Div + Neg + PartialEq + PartialOrd + Copy + Debug{}

/// Trait for data containing integer number operations
/// Defines minimum operations that can apply to any integer and return a valid, usable result
pub trait IntUnit: Unit {
    fn zero() -> Self;
    fn one() -> Self;
    fn pow(self, n: &impl crate::numbers::IntUnit) -> Self;
    fn signum(self) -> Self;
    fn abs(self) -> Self;
    fn max(self, other: &impl crate::numbers::IntUnit) -> Self;
    fn min(self, other: &impl crate::numbers::IntUnit) -> Self;
    fn clamp(self, min: &impl crate::numbers::IntUnit, max: &impl crate::numbers::IntUnit) -> Self;
}

/// Trait for data containing real number operations
/// Defines minimum operations that can apply to any real number and return a valid, usable result
pub trait RealUnit: IntUnit {
    const NAN: Self;
    const INFINITY: Self;
    const NEG_INFINITY: Self;
    const EPSILON: Self;
    const MANTISSA_DIGITS: u32;
    fn inv(self) -> Self;
    fn floor(self) -> Self;
    fn ceil(self) -> Self;
    fn round(self) -> Self;
    fn trunc(self) -> Self;
    fn fract(self) -> Self;
    fn powf(self, n: &impl crate::numbers::RealUnit) -> Self;
    fn sqrt(self) -> Self;
    fn cbrt(self) -> Self;
    fn exp(self) -> Self;
    fn ln(self) -> Self;
    fn log(self, base: &impl crate::numbers::RealUnit) -> Self;
    fn log2(self) -> Self;
    fn log10(self) -> Self;
    fn hypot(self, other: &impl crate::numbers::RealUnit) -> Self;
    fn sin(self) -> Self;
    fn cos(self) -> Self;
    fn tan(self) -> Self;
    fn asin(self) -> Self;
    fn acos(self) -> Self;
    fn atan(self) -> Self;
    fn atan2(self) -> Self;
    fn sinh(self) -> Self;
    fn cosh(self) -> Self;
    fn tanh(self) -> Self;
    fn asinh(self) -> Self;
    fn acosh(self) -> Self;
    fn atanh(self) -> Self;
    fn to_deg(self) -> Self;
    fn to_rad(self) -> Self;
    fn pi() -> Self;
    fn e() -> Self;
    fn tau() -> Self;
    fn is_nan(&self) -> bool;
    fn is_inf(&self) -> bool;
}

/// Marker trait for identifying single field types
pub trait PrimitiveUnit: Unit {}

/// Trait for data containing complex number operations
pub trait ComplexUnit: Unit {}