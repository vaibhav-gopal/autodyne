//! [`Unit`]: the arithmetic every number type shares ([`UnitOps`], [`Zero`], [`One`], [`Inv`]) and its
//! bit-level representation ([`PhysicalRepr`]).

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
pub trait Unit: PhysicalRepr + Zero + One + UnitOps + Inv {}

/// Defines the additive identity
pub trait Zero: UnitOps {
    /// The additive identity.
    const _ZERO: Self;
    /// Whether `self` is the additive identity.
    fn _is_zero(&self) -> bool {
        self.eq(&Self::_ZERO)
    }
    /// Sets `self` to the additive identity.
    fn _set_zero(&mut self) {
        *self = Self::_ZERO;
    }
}

/// Defines the multiplicative identity
pub trait One: UnitOps {
    /// The multiplicative identity.
    const _ONE: Self;
    /// Whether `self` is the multiplicative identity.
    fn _is_one(&self) -> bool {
        self.eq(&Self::_ONE)
    }
    /// Sets `self` to the multiplicative identity.
    fn _set_one(&mut self) {
        *self = Self::_ONE;
    }
    /// `1 / self`.
    fn _recip(self) -> Self {
        Self::_ONE / self
    }
}

/// A type's machine representation: its bits and bytes.
pub trait PhysicalRepr: Copy + Sized + Debug + 'static {
    /// bit-width of the datatype
    const _BITS: u32;
    /// Size in bytes.
    const _BYTES: usize;
    /// A type that can represent the bits / base-2 internal representation of the unit
    type BitsRepr: Unit + Bitwise + Bounded + Eq;
    /// The byte array type (`[u8; BYTES]`).
    type BytesRepr;
    /// Casting from raw bits and bytes
    fn _from_bits(v: Self::BitsRepr) -> Self;
    /// The raw bits.
    fn _to_bits(self) -> Self::BitsRepr;
    /// From big-endian bytes.
    fn _from_be_bytes(bytes: Self::BytesRepr) -> Self;
    /// From little-endian bytes.
    fn _from_le_bytes(bytes: Self::BytesRepr) -> Self;
    /// From bytes in the platform's order.
    fn _from_ne_bytes(bytes: Self::BytesRepr) -> Self;
    /// As big-endian bytes.
    fn _to_be_bytes(self) -> Self::BytesRepr;
    /// As little-endian bytes.
    fn _to_le_bytes(self) -> Self::BytesRepr;
    /// As bytes in the platform's order.
    fn _to_ne_bytes(self) -> Self::BytesRepr;
}

/// A unit with a reciprocal.
pub trait Inv: UnitOps {
    /// `1 / self` (truncated for integers).
    fn _inv(self) -> Self;
}
