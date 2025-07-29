use std::ops::{BitAnd, BitOr, BitXor, Neg, Not, Shl, Shr};
use super::*;

// Marker traits
pub trait Bitwise: Not<Output = Self> + BitAnd<Output = Self> + BitOr<Output = Self> + BitXor<Output = Self> + Shl<Output = Self> + Shr<Output = Self> {}

// Opt-In Traits
pub trait Exp<RHS = Self> {
    type Output: Unit;
    fn pow(self, rhs: RHS) -> Self::Output;
    fn root(self, n: RHS) -> Self::Output;
    fn sq(self) -> Self::Output;
    fn cb(self) -> Self::Output;
    fn sqrt(self) -> Self::Output;
    fn cbrt(self) -> Self::Output;
}
