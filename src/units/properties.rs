use std::ops::Neg;
use super::*;

/// Describes a unit that can be ordered w.r.t itself
pub trait Ordered: PartialOrd {
    fn _min(self, other: Self) -> Self;
    fn _max(self, other: Self) -> Self;
    fn _clamp(self, min: Self, max: Self) -> Self;
}

/// Describes a unit that has total order and reflexivity
pub trait OrderedReflexive: Ordered + Ord + Eq {}

/// Defines bounds on values
pub trait Bounded {
    const _MIN: Self;
    const _MAX: Self;
}

/// Describes the property of a unit that is capable of negative numbers
pub trait Signed: Unit + Neg<Output = Self> {
    const _SIGN_MASK: Self::BitsRepr;
    const _NEG_ONE: Self;
    fn _abs(self) -> Self;
    fn _signum(self) -> Self;
    fn _is_positive(self) -> bool;
    fn _is_negative(self) -> bool;
}

/// Describes a signal that is both bounded and signed
pub trait BoundedSigned: Bounded + Signed {
    const _MIN_POSITIVE: Self;
}