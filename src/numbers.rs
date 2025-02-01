use std::ops::{Add, Sub, Mul, Div, Neg};

// create Float marker trait that identifies floating point arithmetic
// create Fixed marker trait that identifies fixed point arithmetic
// SHOULD NOT INTRODUCE RUNTIME CONSTRAINTS (should never check if number is floatnumber or fixednumber!)
// USE OPERATOR OVERLOADS AND ZERO-COST ABSTRACTIONS WHERE YOU CAN
// HOWEVER, YOU CAN!! --> use floatnumber and fixednumber as trait bounds (compile time not run time)

// GENERAL =========================================================================================
/// General number methods and identifier
pub trait Number: Add<Self::T> + Sub<Self::T> + Mul<Self::T> + Div<Self::T> + Neg {
    type T;

    fn zero() -> Self::T;
    fn one() -> Self::T;
}

/// Float-point number specific methods and identifier
pub trait FloatNumber: Number {
    type T;
}

/// Fixed-point number specific methods and identifier
pub trait FixedNumber: Number {
    type T;
}

// PRIVATE =========================================================================================

/// The Real number implementation for this library (can create your own)
#[derive(Debug, Copy, Clone)]
pub struct Real<T: NumberType>(T);

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
trait NumberType: Add + Sub + Mul + Div + Neg {}

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

/// Generic Number implementation for float types
/// TODO: instead of using explicit float primitives, define trait bounds to require From and Into using f32 for example instead
impl<T: FloatType> Number for Real<T> {
    type T = T;

    fn zero() -> Self::T {
        0.0
    }

    fn one() -> Self::T {
        1.0
    }
}

/// Generic Number implementation for fixed types
/// TODO: instead of using explicit integer primitives, define trait bounds to require From and Into using i32 for example instead
impl<T: FixedType> Number for Real<T> {
    type T = T;

    fn zero() -> Self::T {
        0
    }

    fn one() -> Self::T {
        1
    }
}

// FLOAT/FIXED FUNCTIONS

impl<T: FloatType> FloatNumber for Real<T> {
    type T = T;
}

impl<T: FixedType> FixedNumber for Real<T> {
    type T = T;
}

// FLOAT OPERATOR OVERLOADS
// can rely on auto-deref for FloatType instead of impl all the overloads again, but for the sake of type safety...
// only same type operations will be supported

impl<T: FloatType, A: FloatType> Add<A> for Real<T> {
    type Output = Self;
    fn add(self, rhs: Real<A>) -> Self::Output {
        self.0 + rhs.0
    }
}
impl<T: FloatType, A: FloatType> Sub<A> for Real<T> {
    type Output = Self;
    fn sub(self, rhs: Real<A>) -> Self::Output {
        self.0 - rhs.0
    }
}
impl<T: FloatType, A: FloatType> Mul<A> for Real<T> {
    type Output = Self;
    fn mul(self, rhs: Real<A>) -> Self::Output {
        self.0 * rhs.0
    }
}
impl<T: FloatType, A: FloatType> Div<A> for Real<T> {
    type Output = Self;
    fn div(self, rhs: Real<A>) -> Self::Output {
        self.0 / rhs.0
    }
}
impl<T: FloatType> Neg for Real<T> {
    type Output = Self;
    fn neg(self) -> Self::Output {
        -self.0
    }
}

// FIXED OPERATOR OVERLOADS
// use macros please!
// add checks to make sure no overflow occurs!

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
impl<T: FixedType> Neg for Real<T> {
    type Output = Self;
    fn neg(self) -> Self::Output {
        -self.0
    }
}

macro_rules! fixed_add {
    ($([$to:ident, $sz:ident, $scale_from:expr, $scale_to:expr]),*) => {
        $(
        impl<T: FixedType + $sz, A: FixedType + $to> Add<A> for Real<T> {
            type Output = Self;
            fn add(self, rhs: Real<A>) -> Self::Output {
                let max_scale = $self_from.max($scale_to);
                let lhs_val = self.0.checked_shl(max_scale - $scale_from)?;
                let rhs_val = self.0.checked_shl(max_scale - $scale_to)?;
                lhs_val.checked_add(rhs_val)?
            }
        }
        )*
    };
}

macro_rules! fixed_sub {
    ($([$to:ident, $sz:ident, $scale_from:expr, $scale_to:expr]),*) => {
        $(
        impl<T: FixedType + $sz, A: FixedType + $to> Sub<A> for Real<T> {
            type Output = Self;
            fn sub(self, rhs: Real<A>) -> Self::Output {
                let max_scale = $self_from.max($scale_to);
                let lhs_val = self.0.checked_shl(max_scale - $scale_from)?;
                let rhs_val = self.0.checked_shl(max_scale - $scale_to)?;
                lhs_val.checked_sub(rhs_val)?
            }
        }
        )*
    };
}

macro_rules! fixed_mul {
    ($([$to:ident, $sz:ident, $scale_from:expr, $scale_to:expr]),*) => {
        $(
        impl<T: FixedType + $sz, A: FixedType + $to> Mul<A> for Real<T> {
            type Output = Self;
            fn mul(self, rhs: Real<A>) -> Self::Output {
                (self.0 as i64 * rhs.0 as i64) >> ($scale_from + $scale_to)
            }
        }
        )*
    };
}

macro_rules! fixed_div {
    ($([$to:ident, $sz:ident, $scale_from:expr, $scale_to:expr]),*) => {
        $(
        impl<T: FixedType + $sz, A: FixedType + $to> Div<A> for Real<T> {
            type Output = Self;
            fn div(self, rhs: Real<A>) -> Self::Output {
                (self.0 as i64 / rhs.0 as i64) << ($scale_from.max($scale_to))
            }
        )*
    };
}
