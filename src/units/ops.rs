use std::ops::{BitAnd, BitOr, BitXor, Neg, Not, Shl, Shr};
use super::*;

// Marker traits
pub trait Bitwise: Not<Output = Self> + BitAnd<Output = Self> + BitOr<Output = Self> + BitXor<Output = Self> + Shl<Output = Self> + Shr<Output = Self> + Sized {}

pub trait ExpBasic: Unit {
    type Output: Unit;
    /// self^2
    fn _sq(self) -> <Self as ExpBasic>::Output;
    /// self^1/2
    fn _sqrt(self) -> <Self as ExpBasic>::Output;
}

// Opt-In Traits
pub trait ExpPowDynamic<RHS>: ExpBasic {
    /// self^rhs
    fn _pow(self, rhs: RHS) -> <Self as ExpBasic>::Output;
}

pub trait ExpRootDynamic<RHS>: ExpBasic {
    /// self^(1/rhs)
    fn _root(self, n: RHS) -> <Self as ExpBasic>::Output;
}

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
