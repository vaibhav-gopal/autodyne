use std::ops::{BitAnd, BitOr, BitXor, Not, Shl, Shr};
use super::*;

// Marker traits
/// Bit operations: not, and, or, xor and shifts.
pub trait Bitwise: Not<Output = Self> + BitAnd<Output = Self> + BitOr<Output = Self> + BitXor<Output = Self> + Shl<Output = Self> + Shr<Output = Self> + Sized {}

/// Squares and square roots.
pub trait ExpBasic: Unit {
    /// The result type (`Self` for floats).
    type Output: Unit;
    /// self^2
    fn _sq(self) -> <Self as ExpBasic>::Output;
    /// self^1/2
    fn _sqrt(self) -> <Self as ExpBasic>::Output;
}

// Opt-In Traits
/// Powers with an exponent of type `RHS`.
pub trait ExpPowDynamic<RHS>: ExpBasic {
    /// self^rhs
    fn _pow(self, rhs: RHS) -> <Self as ExpBasic>::Output;
}

/// Roots with a degree of type `RHS`.
pub trait ExpRootDynamic<RHS>: ExpBasic {
    /// self^(1/rhs)
    fn _root(self, n: RHS) -> <Self as ExpBasic>::Output;
}

/// Trigonometric functions (radians)
pub trait Trig: Unit {
    /// Sine.
    fn _sin(self) -> Self;
    /// Cosine.
    fn _cos(self) -> Self;
    /// Tangent.
    fn _tan(self) -> Self;
    /// (sin(self), cos(self)) ; cheaper than calling both separately
    fn _sin_cos(self) -> (Self, Self);
    /// Arcsine, in [-π/2, π/2].
    fn _asin(self) -> Self;
    /// Arccosine, in [0, π].
    fn _acos(self) -> Self;
    /// Arctangent, in (-π/2, π/2).
    fn _atan(self) -> Self;
    /// four-quadrant arctangent of self / other
    fn _atan2(self, other: Self) -> Self;
    /// sqrt(self^2 + other^2) without intermediate overflow
    fn _hypot(self, other: Self) -> Self;
}

/// Exponentials and logarithms.
pub trait ExpFloat<RHS = Self>: ExpRootDynamic<RHS> + ExpPowDynamic<RHS> {
    /// e^self
    fn _exp(self) -> <Self as ExpBasic>::Output;
    /// 2^self
    fn _exp2(self) -> <Self as ExpBasic>::Output;
    /// e^self - 1
    fn _exp_m1(self) -> <Self as ExpBasic>::Output;
    /// log(self)
    fn _log(self, base: RHS) -> <Self as ExpBasic>::Output;
    /// log_2(self)
    fn _log2(self) -> <Self as ExpBasic>::Output;
    /// log_10(self)
    fn _log10(self) -> <Self as ExpBasic>::Output;
    /// ln(self)
    fn _ln(self) -> <Self as ExpBasic>::Output;
    /// ln(self + 1)
    fn _ln_1p(self) -> <Self as ExpBasic>::Output;
}
