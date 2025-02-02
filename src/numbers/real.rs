use std::cmp::Ordering;
use std::fmt::{Debug, Formatter};
use std::ops::{Add, Div, Mul, Neg, Sub};
use super::*;
use super::markers::*;

/// Real (fixed-point) number type
#[repr(transparent)]
pub struct Real<T: FixedType>(T);
pub type Real16 = Real<i16>;
pub type Real32 = Real<i32>;
pub type Real64 = Real<i64>;
pub type Real128 = Real<i128>;

// GENERAL IMPLEMENTATIONS =========================================================================

impl<T: FixedType> Real<T> {
    pub fn new(val: T) -> Self {
        Self(val)
    }
}
impl<T: FixedType> Clone for Real<T> {
    fn clone(&self) -> Self {
        self.0.clone()
    }
}
impl<T: FixedType> Copy for Real<T> {}
impl<T: FixedType> Debug for Real<T> {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        todo!()
    }
}

// UNIT IMPLEMENTATIONS ============================================================================

impl<T: FixedType> Unit for Real<T> {}
impl<T: FixedType> PrimitiveUnit for Real<T> {}

impl<T: FixedType> IntUnit for Real<T> {
    fn zero() -> Self {
        Real::new(0)
    }
    fn one() -> Self {
        Real::new(1)
    }
    fn pow(self, n: &impl IntUnit) -> Self {
        todo!()
    }
    fn signum(self) -> Self {
        todo!()
    }
    fn abs(self) -> Self {
        todo!()
    }
    fn max(self, other: &impl IntUnit) -> Self {
        todo!()
    }
    fn min(self, other: &impl IntUnit) -> Self {
        todo!()
    }
    fn clamp(self, min: &impl IntUnit, max: &impl IntUnit) -> Self {
        todo!()
    }
}

macro_rules! specialization {
    () => {};
}

impl<T: FixedType + Sz16> RealUnit for Real<T> {
    const MANTISSA_DIGITS: u32 = 8;
    const EPSILON: Self = Real(0);
    const INFINITY: Self = Real(0);
    const NAN: Self = Real(0);
    const NEG_INFINITY: Self = Real(0);
}
impl<T: FixedType + Sz32> RealUnit for Real<T> {
    const MANTISSA_DIGITS: u32 = 16;
    const EPSILON: Self = Real(0);
    const INFINITY: Self = Real(0);
    const NAN: Self = Real(0);
    const NEG_INFINITY: Self = Real(0);
}
impl<T: FixedType + Sz64> RealUnit for Real<T> {
    const MANTISSA_DIGITS: u32 = 32;
    const EPSILON: Self = Real(0);
    const INFINITY: Self = Real(0);
    const NAN: Self = Real(0);
    const NEG_INFINITY: Self = Real(0);
}
impl<T: FixedType + Sz128> RealUnit for Real<T> {
    const MANTISSA_DIGITS: u32 = 64;
    const EPSILON: Self = Real(0);
    const INFINITY: Self = Real(0);
    const NAN: Self = Real(0);
    const NEG_INFINITY: Self = Real(0);
}

// FIXED OPERATOR OVERLOADS (WITH OTHERS)
// use macros please!
// add checks to make sure no overflow occurs!

macro_rules! fixed_add {
    ( $([$from:ident, $to:ident, $scale_from:ident, $scale_to:expr]),* ) => {
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
            [Sz128, Sz64, SCALE_FACTOR_128, SCALE_FACTOR_64],
            [Sz128, Sz32, SCALE_FACTOR_128, SCALE_FACTOR_32],
            [Sz128, Sz16, SCALE_FACTOR_128, SCALE_FACTOR_16],
            [Sz64, Sz128, SCALE_FACTOR_64, SCALE_FACTOR_128],
            [Sz64, Sz32, SCALE_FACTOR_64, SCALE_FACTOR_32],
            [Sz64, Sz16, SCALE_FACTOR_64, SCALE_FACTOR_16],
            [Sz32, Sz128, SCALE_FACTOR_32, SCALE_FACTOR_128],
            [Sz32, Sz64, SCALE_FACTOR_32, SCALE_FACTOR_64],
            [Sz32, Sz16, SCALE_FACTOR_32, SCALE_FACTOR_16],
            [Sz16, Sz128, SCALE_FACTOR_16, SCALE_FACTOR_128],
            [Sz16, Sz64, SCALE_FACTOR_16, SCALE_FACTOR_64],
            [Sz16, Sz32, SCALE_FACTOR_16, SCALE_FACTOR_32]
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
        self.0.eq(other.0)
    }
}
impl<T: FixedType> PartialOrd for Real<T> {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        self.0.partial_cmp(other.0)
    }
}
