use std::ops::{Add, Div, Mul, Rem, Sub};
use std::fmt::Debug;
use super::*;

/// Elementary operations
pub trait UnitOps: Add<Output = Self> + Sub<Output = Self> + Mul<Output = Self> + Div<Output = Self> + Rem<Output = Self> + PartialEq + Copy + Sized {}

/// Units are:
/// 1. In memory (Copy, implies Sized)
/// 2. Support elementary arithmetic (UnitOps)
/// 3. Part of a set of values (PartialEq)
/// 4. Have a multiplicative and additive identity (Zero and One)
pub trait Unit: PhysicalRepr + Zero + One + UnitOps + Inv + Symbolic {}

/// Defines the additive identity
pub trait Zero: UnitOps {
    const _ZERO: Self;
    fn _is_zero(&self) -> bool {
        self.eq(&Self::_ZERO)
    }
    fn _set_zero(&mut self) {
        *self = Self::_ZERO;
    }
}

/// Defines the multiplicative identity
pub trait One: UnitOps {
    const _ONE: Self;
    fn _is_one(&self) -> bool {
        self.eq(&Self::_ONE)
    }
    fn _set_one(&mut self) {
        *self = Self::_ONE;
    }
    fn _recip(self) -> Self {
        Self::_ONE / self
    }
}

pub trait PhysicalRepr: Copy + Sized + Debug {
    /// bit-width of the datatype
    const _BITS: u32;
    const _BYTES: usize;
    /// A type that can represent the bits / base-2 internal representation of the unit
    type BitsRepr: Unit + Bitwise + Bounded + Eq;
    type BytesRepr;
    /// Casting from raw bits and bytes
    fn _from_bits(v: Self::BitsRepr) -> Self;
    fn _to_bits(self) -> Self::BitsRepr;
    fn _from_be_bytes(bytes: Self::BytesRepr) -> Self;
    fn _from_le_bytes(bytes: Self::BytesRepr) -> Self;
    fn _from_ne_bytes(bytes: Self::BytesRepr) -> Self;
    fn _to_be_bytes(self) -> Self::BytesRepr;
    fn _to_le_bytes(self) -> Self::BytesRepr;
    fn _to_ne_bytes(self) -> Self::BytesRepr;
}

/// Describes the property of unit to be simplified/alternatively viewed w.r.t another unit
pub trait Symbolic: UnitOps {
    type Base: Unit;
    fn _dismantle(self) -> Option<Self::Base> {
        None
    }
}

/// Describes the property of a unit having an inverse representation (guarantees self.inv().inv() == self)
pub trait Inv: UnitOps {
    fn _inv(self) -> Self;
}
