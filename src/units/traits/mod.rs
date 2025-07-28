use std::fmt::Debug;
use std::ops::{Add, Div, Mul, Neg, Rem, Sub};

/// Elementary operations
pub trait UnitOps: Add<Output = Self> + Sub<Output = Self> + Mul<Output = Self> + Div<Output = Self> + Rem<Output = Self> {}

/// Units are:
/// 1. In memory (Copy, implies Sized)
/// 2. Support elementary arithmetic (UnitOps)
/// 3. Part of a set of values (PartialEq)
/// 4. Have a multiplicative and additive identity (Zero and One)
pub trait Unit: PhysicalRepr + PartialEq + Zero + One + UnitOps + Inv + Symbolic {} 

/// Defines the additive identity
pub trait Zero: Add<Output = Self> + Sub<Output = Self> {
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
    const BYTES: u32;
    /// A type that can represent the bits / base-2 internal representation of the unit
    type IntRepr: Unit + Bitwise + Bounded + Eq;
    /// Casting from raw bits and bytes
    fn from_bits(v: Self::IntRepr) -> Self;
    fn to_bits(self) -> Self::IntRepr;
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

mod properties {
    use std::ops::Neg;
    use super::Unit;
    
    /// Describes a unit that can be ordered w.r.t itself
    pub trait Ordered: Unit + PartialOrd {
        fn min(self, other: self) -> Self;
        fn max(self, other: self) -> Self;
        fn clamp(self, min: self, max: self) -> Self;
    }
    
    /// Describes a unit that has total order and reflexivity
    pub trait OrderedReflexive: Ordered + Ord + Eq {}
    
    /// Defines bounds on values
    pub trait Bounded: Unit {
        const MIN: Self;
        const MAX: Self;
    }
    
    /// Describes the property of a unit that is capable of negative numbers
    pub trait Signed: Unit + Neg<Output = Self> {
        const SIGN_MASK: Self::IntRepr;
        const NEG_ONE: Self = Self::ONE.neg();
        fn abs(self) -> Self;
        fn signum(self) -> Self;
        fn is_sign_positive(self) -> bool;
        fn is_sign_negative(self) -> bool;
    }
    
    /// Describes a signal that is both bounded and signed
    pub trait BoundedSigned: Bounded + Signed {
        const MIN_POSITIVE: Self;
    }
}
pub use properties::*;

mod ops {
    use std::ops::{BitAnd, BitOr, BitXor, Neg, Not, Shl, Shr};
    use super::Unit;
    
    // Marker traits
    pub trait Bitwise: Unit + Not<Output = Self> + BitAnd<Output = Self> + BitOr<Output = Self> + BitXor<Output = Self> + Shl<Output = Self> + Shr<Output = Self> {}

    // Opt-In Traits
    pub trait Exp<RHS = Self>: Unit {
        type Output: Unit;
        fn pow(self, rhs: RHS) -> Self::Output;
        fn root(self, n: RHS) -> Self::Output;
        fn sq(self) -> Self::Output;
        fn cb(self) -> Self::Output;
        fn sqrt(self) -> Self::Output;
        fn cbrt(self) -> Self::Output;
    }
}
pub use ops::*;

mod cast {
    pub trait FromPrimitive: Copy {
        fn from_i64(n: i64) -> Option<Self>;
        fn from_u64(n: u64) -> Option<Self>;
        fn from_isize(n: isize) -> Option<Self>;
        fn from_i8(n: i8) -> Option<Self>;
        fn from_i16(n: i16) -> Option<Self>;
        fn from_i32(n: i32) -> Option<Self>;
        fn from_i128(n: i128) -> Option<Self>;
        fn from_usize(n: usize) -> Option<Self>;
        fn from_u8(n: u8) -> Option<Self>;
        fn from_u16(n: u16) -> Option<Self>;
        fn from_u32(n: u32) -> Option<Self>;
        fn from_u128(n: u128) -> Option<Self>;
        fn from_f32(n: f32) -> Option<Self>;
        fn from_f64(n: f64) -> Option<Self>;
        fn from_str(n: &str) -> Option<Self>;
    }
    
    pub trait ToPrimitive: Copy {
        fn to_i64(self) -> Option<i64>;
        fn to_u64(self) -> Option<u64>;
        fn to_isize(self) -> Option<isize>;
        fn to_i8(self) -> Option<i8>;
        fn to_i16(self) -> Option<i16>;
        fn to_i32(self) -> Option<i32>;
        fn to_i128(self) -> Option<i128>;
        fn to_usize(self) -> Option<usize>;
        fn to_u8(self) -> Option<u8>;
        fn to_u16(self) -> Option<u16>;
        fn to_u32(self) -> Option<u32>;
        fn to_u128(self) -> Option<u128>;
        fn to_f32(self) -> Option<f32>;
        fn to_f64(self) -> Option<f64>;
        fn to_str(self) -> Option<String>;
    }
    
    pub trait AsPrimitive<T: 'static + Copy>: 'static + Copy {
        fn as_(self) -> T;
    }
    
    pub trait CastPrimitive: FromPrimitive + ToPrimitive {}
}
pub use cast::*;

pub trait Integer: Unit + OrderedReflexive + Bounded + Bitwise + Exp + CastPrimitive {}
pub trait Float: Unit + Ordered + BoundedSigned + Exp + CastPrimitive {
    /// Special states
    const NAN: Self;
    const INFINITY: Self;
    const NEG_INFINITY: Self;
    /// Bounds, and bit-widths for machine representation
    const EPSILON: Self;
    const SIG_BITS: u32;
    const EXP_BITS: u32;
    const SIG_MASK: Self::IntRepr;
    const EXP_MASK: Self::IntRepr;
    const DIGITS: u32;
    const MANTISSA_DIGITS: u32;
    const MIN_EXP: i32;
    const MAX_EXP: i32;
    const MIN_10_EXP: i32;
    const MAX_10_EXP: i32;
    /// Mathematical constants
    const PI: Self;
    const E: Self;
    const TAU: Self;
    fn floor(self) -> Self;
    fn ceil(self) -> Self;
    fn round(self) -> Self;
    fn trunc(self) -> Self;
    fn fract(self) -> Self;
}