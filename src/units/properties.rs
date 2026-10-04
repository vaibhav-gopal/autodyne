use std::ops::Neg;
use super::*;

/// Describes a unit that can be ordered w.r.t itself
pub trait Ordered: PartialOrd {
    /// The smaller of the two (for floats, the other one when one is NaN).
    fn _min(self, other: Self) -> Self;
    /// The larger of the two (for floats, the other one when one is NaN).
    fn _max(self, other: Self) -> Self;
    /// `self` limited to `[min, max]`.
    fn _clamp(self, min: Self, max: Self) -> Self;
}

/// Describes a unit that has total order and reflexivity
pub trait OrderedReflexive: Ordered + Ord + Eq {}

/// Defines bounds on values
pub trait Bounded {
    /// The smallest value (for floats, the most negative finite one).
    const _MIN: Self;
    /// The largest (finite) value.
    const _MAX: Self;
}

/// Describes the property of a unit that is capable of negative numbers
pub trait Signed: Unit + Neg<Output = Self> {
    /// Mask of the sign bit in the bit representation.
    const _SIGN_MASK: Self::BitsRepr;
    /// -1.
    const _NEG_ONE: Self;
    /// The absolute value.
    fn _abs(self) -> Self;
    /// -1, 0 or 1 for integers; ±1 for floats (by the sign bit), NaN for NaN.
    fn _signum(self) -> Self;
    /// Whether `self` is positive (floats: by the sign bit, so `+0.0` counts).
    fn _is_positive(self) -> bool;
    /// Whether `self` is negative (floats: by the sign bit, so `-0.0` counts).
    fn _is_negative(self) -> bool;
}

/// Describes a unit that is both bounded and signed
pub trait BoundedSigned: Bounded + Signed {
    /// The smallest positive normal value for floats (0 for integers).
    const _MIN_POSITIVE: Self;
}