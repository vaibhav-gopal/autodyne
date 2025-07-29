use std::ops::{Add, Div, Mul, Rem, Sub};
use std::fmt::Debug;
use super::*;

/// Elementary operations
pub trait UnitOps: Add<Output = Self> + Sub<Output = Self> + Mul<Output = Self> + Div<Output = Self> + Rem<Output = Self> {}

/// Units are:
/// 1. In memory (Copy, implies Sized)
/// 2. Support elementary arithmetic (UnitOps)
/// 3. Part of a set of values (PartialEq)
/// 4. Have a multiplicative and additive identity (Zero and One)
pub trait Unit: PhysicalRepr + PartialEq + Zero + One + UnitOps + Inv + Symbolic {}

/// Defines the additive identity
pub trait Zero: Add<Output=Self> + Sub<Output=Self> {
    const ZERO: Self;
    fn is_zero(&self) -> bool {
        self.eq(Self::ZERO)
    }
    fn set_zero(&mut self) {
        self = Self::ZERO;
    }
}

/// Defines the multiplicative identity
pub trait One: Mul<Output = Self> + Div<Output = Self> + Sized {
    const ONE: Self;
    fn is_one(&self) -> bool {
        self.eq(Self::ONE)
    }
    fn set_one(&mut self) {
        self = Self::ONE;
    }
    fn recip(self) -> Self {
        Self::ONE / self
    }
}

pub trait PhysicalRepr: Copy + Sized + Debug {
    /// bit-width of the datatype
    const BITS: u32;
    const BYTES: usize;
    /// A type that can represent the bits / base-2 internal representation of the unit
    type BitsRepr: Unit + Bitwise + Bounded + Eq;
    /// Casting from raw bits and bytes
    fn from_bits(v: Self::BitsRepr) -> Self;
    fn to_bits(self) -> Self::BitsRepr;
    fn from_be_bytes(bytes: [u8; Self::BYTES]) -> Self;
    fn from_le_bytes(bytes: [u8; Self::BYTES]) -> Self;
    fn from_ne_bytes(bytes: [u8; Self::BYTES]) -> Self;
    fn to_be_bytes(self) -> [u8; Self::BYTES];
    fn to_le_bytes(self) -> [u8; Self::BYTES];
    fn to_ne_bytes(self) -> [u8; Self::BYTES];
}

/// Describes the property of unit to be simplified/alternatively viewed w.r.t another unit
pub trait Symbolic {
    type Base: Unit;
    fn dismantle(self) -> Option<Self::Base> {
        None
    }
}

/// Describes the property of a unit having an inverse representation (guarantees self.inv().inv() == self)
pub trait Inv {
    fn inv(self) -> Self;
}
