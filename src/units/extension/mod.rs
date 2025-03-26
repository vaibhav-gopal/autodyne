/// Using extension traits compared to newtypes ; extending the primitive types

use super::*;
use delegate::delegate;

impl Unit for f32 {}
impl Unit for f64 {}

impl PrimitiveUnit for f32 {}
impl PrimitiveUnit for f64 {}

impl RealUnit for f32 {
    const NAN: Self = Self::NAN;
    const INFINITY: Self = Self::INFINITY;
    const NEG_INFINITY: Self =  Self::NEG_INFINITY;
    const EPSILON: Self = Self::EPSILON;
    const MANTISSA_DIGITS: u32 = Self::MANTISSA_DIGITS;
    const PI: Self = std::f32::consts::PI;
    const E: Self = std::f32::consts::E;
    const TAU: Self = std::f32::consts::TAU;
    fn zero() -> Self { 0f32 }
    fn one() -> Self { 1f32 }

    delegate! {
        #[through(f32)]
        #[allow(unconditional_recursion)]
        to self {
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
        }
    }
    fn is_nan(&self) -> bool {
        todo!()
    }
    fn is_inf(&self) -> bool {
        todo!()
    }
}

impl RealUnit for f64 {
    const NAN: Self = Self::NAN;
    const INFINITY: Self = Self::INFINITY;
    const NEG_INFINITY: Self =  Self::NEG_INFINITY;
    const EPSILON: Self = Self::EPSILON;
    const MANTISSA_DIGITS: u32 = Self::MANTISSA_DIGITS;
    const PI: Self = std::f64::consts::PI;
    const E: Self = std::f64::consts::E;
    const TAU: Self = std::f64::consts::TAU;
    fn zero() -> Self { 0f64 }
    fn one() -> Self { 1f64 }
    
    delegate! {
        #[through(f64)]
        #[allow(unconditional_recursion)]
        to self {
            fn signum(self) -> Self;
            fn abs(self) -> Self;
            fn pow(self, n: Self) -> Self;
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
        }
    }
    fn is_nan(&self) -> bool {
        todo!()
    }
    fn is_inf(&self) -> bool {
        todo!()
    }
}
