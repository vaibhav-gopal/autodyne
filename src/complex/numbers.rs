use std::cmp::Ordering;
use std::ops::{Add, Sub, Mul, Div, Neg};
use std::fmt::{Debug, Formatter};

// create Float marker trait that identifies floating point arithmetic
// create Fixed marker trait that identifies fixed point arithmetic
// SHOULD NOT INTRODUCE RUNTIME CONSTRAINTS (should never check if number is floatnumber or fixednumber!)
// USE OPERATOR OVERLOADS AND ZERO-COST ABSTRACTIONS WHERE YOU CAN
// HOWEVER, YOU CAN!! --> use floatnumber and fixednumber as trait bounds (compile time not run time)

// GENERAL =========================================================================================
/// General number methods and identifier
pub trait Number: Add + Sub + Mul + Div + Neg + PartialEq + PartialOrd + Copy + Debug + From<Self>{
    fn zero() -> Self;
    fn one() -> Self;
    fn floor(self) -> Self;
    fn ceil(self) -> Self;
    fn round(self) -> Self;
    fn trunc(self) -> Self;
    fn fract(self) -> Self;
    fn pow(self, n: &Self) -> Self;
    fn powf(self, n: &Self) -> Self;
    fn sqrt(self) -> Self;
    fn cbrt(self) -> Self;
    fn exp(self) -> Self;
    fn ln(self) -> Self;
    fn log(self, base: &Self) -> Self;
    fn log2(self) -> Self;
    fn log10(self) -> Self;
    fn hypot(self, other: &Self) -> Self;
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
    fn abs(self) -> Self;
    fn clamp(self, min: &Self, max: &Self) -> Self;
    fn signum(self) -> Self;
    fn max(self, other: &Self) -> Self;
    fn min(self, other: &Self) -> Self;
    fn to_deg(self) -> Self;
    fn to_rad(self) -> Self;
    fn inv(self) -> Self;
    fn pi() -> Self;
    fn e() -> Self;
    fn tau() -> Self;
}

/// Float-point number identifier
pub trait FloatNumber: Number {}
/// Fixed-point number identifier
pub trait FixedNumber: Number {}

// PRIVATE =========================================================================================

/// The Real number implementation for this library (can create your own)
pub struct Real<T: FixedType>(T);
pub struct RealF<T: FloatType>(T);

pub type Real16 = Real<i16>;
pub type Real32 = Real<i32>;
pub type Real64 = Real<i64>;
pub type Real128 = Real<i128>;
pub type RealF32 = RealF<f32>;
pub type RealF64 = RealF<f64>;

/// scale factor (how many bits wide is the fractional digits) for fixed point arithmetic (compile time const for less overhead), for more flexibility (in increasing strength and performance costs):
/// if you want end-user flexibility: look into reading env variables
/// if you want runtime (global scale factor across all numbers): look into having a global state of parameters or global store of callbacks to parameters needed
/// if you want runtime (local scale factor per number): look into encapsulating every fixed point number with its own scale factor
const SCALE_FACTOR_16: usize = 8;
const SCALE_FACTOR_32: usize = 16;
const SCALE_FACTOR_64: usize = 32;
const SCALE_FACTOR_128: usize = 64;

// Private marker traits for QOL
trait Sz16 {}
trait Sz32 {}
trait Sz64 {}
trait Sz128 {}
trait FloatType: NumberType {}
trait FixedType: NumberType {}
trait NumberType: Add + Sub + Mul + Div + Neg + Copy + Debug + PartialEq + PartialOrd {}

impl NumberType for f32 {}
impl NumberType for f64 {}
impl NumberType for i16 {}
impl NumberType for i32 {}
impl NumberType for i64 {}
impl NumberType for i128 {}
impl Sz32 for f32 {}
impl Sz64 for f64 {}
impl Sz16 for i16 {}
impl Sz32 for i32 {}
impl Sz64 for i64 {}
impl Sz128 for i128 {}
impl FloatType for f32 {}
impl FloatType for f64 {}
impl FixedType for i16 {}
impl FixedType for i32 {}
impl FixedType for i64 {}
impl FixedType for i128 {}

impl<T: FloatType> FloatNumber for RealF<T> {}
impl<T: FixedType> FixedNumber for Real<T> {}

/// Generic Number implementation for float types
/// TODO: instead of using explicit float primitives, define trait bounds to require From and Into using f32 for example instead
impl<T: FloatType> Number for RealF<T> {
    fn zero() -> Self {
        RealF(0.0)
    }

    fn one() -> Self {
        RealF(1.0)
    }

    fn floor(self) -> Self {
        todo!()
    }

    fn ceil(self) -> Self {
        todo!()
    }

    fn round(self) -> Self {
        todo!()
    }

    fn trunc(self) -> Self {
        todo!()
    }

    fn fract(self) -> Self {
        todo!()
    }

    fn pow(self, n: &Self) -> Self {
        todo!()
    }

    fn powf(self, n: &Self) -> Self {
        todo!()
    }

    fn sqrt(self) -> Self {
        todo!()
    }

    fn cbrt(self) -> Self {
        todo!()
    }

    fn exp(self) -> Self {
        todo!()
    }

    fn ln(self) -> Self {
        todo!()
    }

    fn log(self, base: &Self) -> Self {
        todo!()
    }

    fn log2(self) -> Self {
        todo!()
    }

    fn log10(self) -> Self {
        todo!()
    }

    fn hypot(self, other: &Self) -> Self {
        todo!()
    }

    fn sin(self) -> Self {
        todo!()
    }

    fn cos(self) -> Self {
        todo!()
    }

    fn tan(self) -> Self {
        todo!()
    }

    fn asin(self) -> Self {
        todo!()
    }

    fn acos(self) -> Self {
        todo!()
    }

    fn atan(self) -> Self {
        todo!()
    }

    fn atan2(self) -> Self {
        todo!()
    }

    fn sinh(self) -> Self {
        todo!()
    }

    fn cosh(self) -> Self {
        todo!()
    }

    fn tanh(self) -> Self {
        todo!()
    }

    fn asinh(self) -> Self {
        todo!()
    }

    fn acosh(self) -> Self {
        todo!()
    }

    fn atanh(self) -> Self {
        todo!()
    }

    fn abs(self) -> Self {
        todo!()
    }

    fn clamp(self, min: &Self, max: &Self) -> Self {
        todo!()
    }

    fn signum(self) -> Self {
        todo!()
    }

    fn max(self, other: &Self) -> Self {
        todo!()
    }

    fn min(self, other: &Self) -> Self {
        todo!()
    }

    fn to_deg(self) -> Self {
        todo!()
    }

    fn to_rad(self) -> Self {
        todo!()
    }

    fn inv(self) -> Self {
        todo!()
    }

    fn pi() -> Self {
        todo!()
    }

    fn e() -> Self {
        todo!()
    }

    fn tau() -> Self {
        todo!()
    }
}

/// Generic Number implementation for fixed types
/// TODO: instead of using explicit integer primitives, define trait bounds to require From and Into using i32 for example instead
impl<T: FixedType> Number for Real<T> {
    fn zero() -> Self {
        Real(0)
    }

    fn one() -> Self {
        Real(1)
    }

    fn floor(self) -> Self {
        todo!()
    }

    fn ceil(self) -> Self {
        todo!()
    }

    fn round(self) -> Self {
        todo!()
    }

    fn trunc(self) -> Self {
        todo!()
    }

    fn fract(self) -> Self {
        todo!()
    }

    fn pow(self, n: &Self) -> Self {
        todo!()
    }

    fn powf(self, n: &Self) -> Self {
        todo!()
    }

    fn sqrt(self) -> Self {
        todo!()
    }

    fn cbrt(self) -> Self {
        todo!()
    }

    fn exp(self) -> Self {
        todo!()
    }

    fn ln(self) -> Self {
        todo!()
    }

    fn log(self, base: &Self) -> Self {
        todo!()
    }

    fn log2(self) -> Self {
        todo!()
    }

    fn log10(self) -> Self {
        todo!()
    }

    fn hypot(self, other: &Self) -> Self {
        todo!()
    }

    fn sin(self) -> Self {
        todo!()
    }

    fn cos(self) -> Self {
        todo!()
    }

    fn tan(self) -> Self {
        todo!()
    }

    fn asin(self) -> Self {
        todo!()
    }

    fn acos(self) -> Self {
        todo!()
    }

    fn atan(self) -> Self {
        todo!()
    }

    fn atan2(self) -> Self {
        todo!()
    }

    fn sinh(self) -> Self {
        todo!()
    }

    fn cosh(self) -> Self {
        todo!()
    }

    fn tanh(self) -> Self {
        todo!()
    }

    fn asinh(self) -> Self {
        todo!()
    }

    fn acosh(self) -> Self {
        todo!()
    }

    fn atanh(self) -> Self {
        todo!()
    }

    fn abs(self) -> Self {
        todo!()
    }

    fn clamp(self, min: &Self, max: &Self) -> Self {
        todo!()
    }

    fn signum(self) -> Self {
        todo!()
    }

    fn max(self, other: &Self) -> Self {
        todo!()
    }

    fn min(self, other: &Self) -> Self {
        todo!()
    }

    fn to_deg(self) -> Self {
        todo!()
    }

    fn to_rad(self) -> Self {
        todo!()
    }

    fn inv(self) -> Self {
        todo!()
    }

    fn pi() -> Self {
        todo!()
    }

    fn e() -> Self {
        todo!()
    }

    fn tau() -> Self {
        todo!()
    }
}

// OTHER TRAIT IMPLS

impl<T: FloatType> Clone for RealF<T> {
    fn clone(&self) -> Self {
        self.0.clone()
    }
}
impl<T: FixedType> Clone for Real<T> {
    fn clone(&self) -> Self {
        self.0.clone()
    }
}

impl<T: FloatType> Copy for RealF<T> {}
impl<T: FixedType> Copy for Real<T> {}

impl<T: FloatType> Debug for RealF<T> {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        todo!()
    }
}
impl<T: FixedType> Debug for Real<T> {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        todo!()
    }
}

// FLOAT OPERATOR OVERLOADS
// can rely on auto-deref for FloatType instead of impl all the overloads again, but for the sake of type safety...
// only same type operations will be supported! (float - float or fixed - fixed)

impl<T: FloatType> Add for RealF<T> {
    type Output = Self;
    fn add(self, rhs: Self) -> Self::Output {
        self.0 + rhs.0
    }
}
impl<T: FloatType> Sub for RealF<T> {
    type Output = Self;
    fn sub(self, rhs: Self) -> Self::Output {
        self.0 - rhs.0
    }
}
impl<T: FloatType> Mul for RealF<T> {
    type Output = Self;
    fn mul(self, rhs: Self) -> Self::Output {
        self.0 * rhs.0
    }
}
impl<T: FloatType> Div for RealF<T> {
    type Output = Self;
    fn div(self, rhs: Self) -> Self::Output {
        self.0 / rhs.0
    }
}
impl<T: FloatType> Neg for RealF<T> {
    type Output = Self;
    fn neg(self) -> Self::Output {
        -self.0
    }
}
impl<T: FloatType> PartialEq for RealF<T> {
    fn eq(&self, other: &Self) -> bool {
        todo!()
    }
}
impl<T: FloatType> PartialOrd for RealF<T> {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        todo!()
    }
}
impl<T: FloatType> From<Self> for RealF<T> {
    fn from(value: T) -> Self {
        todo!()
    }
}

// FIXED OPERATOR OVERLOADS (WITH OTHERS)
// use macros please!
// add checks to make sure no overflow occurs!

macro_rules! fixed_add {
    ( $([$from:ident, $to:ident, $scale_from:expr, $scale_to:expr]),* ) => {
        $(
        impl<T: FixedType + $from, A: FixedType + $to> Add<Real<A>> for Real<T> {
            type Output = Self;
            fn add(self, rhs: Real<A>) -> Self::Output { self.0 + rhs.0 }
        }
        )*
    };
}
macro_rules! fixed_sub {
    ( $([$from:ident, $to:ident, $scale_from:expr, $scale_to:expr]),* ) => {
        $(
        impl<T: FixedType + $from, A: FixedType + $to> Sub<Real<A>> for Real<T> {
            type Output = Self;
            fn sub(self, rhs: Real<A>) -> Self::Output { self.0 - rhs.0 }
        }
        )*
    };
}
macro_rules! fixed_mul {
    ( $([$from:ident, $to:ident, $scale_from:expr, $scale_to:expr]),* ) => {
        $(
        impl<T: FixedType + $from, A: FixedType + $to> Mul<Real<A>> for Real<T> {
            type Output = Self;
            fn mul(self, rhs: Real<A>) -> Self::Output { self.0 * rhs.0 }
        }
        )*
    };
}
macro_rules! fixed_div {
    ( $([$from:ident, $to:ident, $scale_from:expr, $scale_to:expr]),* ) => {
        $(
        impl<T: FixedType + $from, A: FixedType + $to> Div<Real<A>> for Real<T> {
            type Output = Self;
            fn div(self, rhs: Real<A>) -> Self::Output { self.0 / rhs.0 }
        }
        )*
    };
}
macro_rules! apply_fixed_ops {
    ($($in_macro:ident),*) => {
        $(
        $in_macro!(
            [Sz128, Sz64, 128, 64],
            [Sz128, Sz32, 128, 32],
            [Sz128, Sz16, 128, 16],
            [Sz64, Sz128, 64, 128],
            [Sz64, Sz32, 64, 32],
            [Sz64, Sz16, 64, 16],
            [Sz32, Sz128, 32, 128],
            [Sz32, Sz64, 32, 64],
            [Sz32, Sz16, 32, 16],
            [Sz16, Sz128, 16, 128],
            [Sz16, Sz64, 16, 64],
            [Sz16, Sz32, 16, 32]
        );
        )*
    };
}
apply_fixed_ops!(fixed_add, fixed_sub, fixed_mul, fixed_div);

// FIXED OPERATOR OVERLOADS (WITH ITSELF)
// minimum requirement for number trait

impl<T: FixedType> Add for Real<T> {
    type Output = Self;
    fn add(self, rhs: Self) -> Self::Output {
        self.0 + rhs.0
    }
}
impl<T: FixedType> Sub for Real<T> {
    type Output = Self;
    fn sub(self, rhs: Self) -> Self::Output {
        self.0 - rhs.0
    }
}
impl<T: FixedType> Mul for Real<T> {
    type Output = Self;
    fn mul(self, rhs: Self) -> Self::Output {
        self.0 * rhs.0
    }
}
impl<T: FixedType> Div for Real<T> {
    type Output = Self;
    fn div(self, rhs: Self) -> Self::Output {
        self.0 / rhs.0
    }
}
impl<T: FixedType> Neg for Real<T> {
    type Output = Self;
    fn neg(self) -> Self::Output {
        -self.0
    }
}
impl<T: FixedType> PartialEq for Real<T> {
    fn eq(&self, other: &Self) -> bool {
        todo!()
    }
}
impl<T: FixedType> PartialOrd for Real<T> {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        todo!()
    }
}
impl<T: FixedType> From<Self> for Real<T> {
    fn from(value: T) -> Self {
        todo!()
    }
}
