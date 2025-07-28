/// Module for extending the primitive types and introducing a system for working with primitive data (numbers)
/// todo:
/// - add documentation to all traits
/// - add documentation for each new unit type
/// - abstract SIMD operations (later ; prob needs to be its own module)
/// - implement the fixed-point unit type (later)
/// - add tests (next)
/// - make more macros to derive more arithmetic operations (next)

mod extension;

mod traits;
pub use traits::*;

/// Trait for data containing real number operations
/// Defines minimum operations that can apply to any real number and return a valid, usable result
pub trait RealUnit {
    const NAN: Self;
    const INFINITY: Self;
    const NEG_INFINITY: Self;
    const EPSILON: Self;
    const MANTISSA_DIGITS: u32;
    const PI: Self;
    const E: Self;
    const TAU: Self;
    fn pow(self, n: Self) -> Self;
    fn signum(self) -> Self;
    fn abs(self) -> Self;
    fn max(self, other: Self) -> Self;
    fn min(self, other: Self) -> Self;
    fn clamp(self, min: Self, max: Self) -> Self;
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