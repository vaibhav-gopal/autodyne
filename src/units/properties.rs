use std::ops::Neg;
use super::*;

/// Describes a unit that can be ordered w.r.t itself
pub trait Ordered: PartialOrd {
    fn min(self, other: self) -> Self;
    fn max(self, other: self) -> Self;
    fn clamp(self, min: self, max: self) -> Self;
}

/// Describes a unit that has total order and reflexivity
pub trait OrderedReflexive: Ordered + Ord + Eq {}

/// Defines bounds on values
pub trait Bounded {
    const MIN: Self;
    const MAX: Self;
}

/// Describes the property of a unit that is capable of negative numbers
pub trait Signed: Unit + Neg<Output = Self> {
    const SIGN_MASK: Self::BitsRepr;
    const NEG_ONE: Self = Self::ONE.neg();
    fn abs(self) -> Self;
    fn signum(self) -> Self;
    fn is_positive(self) -> bool;
    fn is_negative(self) -> bool;
}

/// Describes a signal that is both bounded and signed
pub trait BoundedSigned: Bounded + Signed {
    const MIN_POSITIVE: Self;
}