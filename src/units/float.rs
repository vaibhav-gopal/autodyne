use super::*;

pub trait Float: Unit + Ordered + BoundedSigned + Exp + CastPrimitive {
    /// Special states
    const NAN: Self;
    const INFINITY: Self;
    const NEG_INFINITY: Self;
    /// Bounds, and bit-widths for machine representation
    const EPSILON: Self;
    const SIG_BITS: u32;
    const EXP_BITS: u32;
    const SIG_MASK: Self::BitsRepr;
    const EXP_MASK: Self::BitsRepr;
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

macro_rules! impl_float {
    ($SrcT:ident) => {
        impl Float for $SrcT {
            const NAN: Self = $SrcT::NAN;
            const INFINITY: Self = $SrcT::INFINITY;
            const NEG_INFINITY: Self = $SrcT::NEG_INFINITY;
            const EPSILON: Self = $SrcT::EPSILON;
            const SIG_BITS: u32 = $SrcT::SIG_BITS;
            const EXP_BITS: u32 = $SrcT::EXP_BITS;
            const SIG_MASK: Self::BitsRepr = $SrcT::SIG_MASK;
            const EXP_MASK: Self::BitsRepr = $SrcT::EXP_MASK;
            const DIGITS: u32 = $SrcT::DIGITS;
            const MANTISSA_DIGITS: u32 = $SrcT::MANTISSA_DIGITS;
            const MIN_EXP: i32 = $SrcT::MIN_EXP;
            const MAX_EXP: i32 = $SrcT::MAX_EXP;
            const MIN_10_EXP: i32 = $SrcT::MIN_10_EXP;
            const MAX_10_EXP: i32 = $SrcT::MAX_10_EXP;
            const PI: Self = $SrcT::PI;
            const E: Self = $SrcT::E;
            const TAU: Self = $SrcT::TAU;
            fn floor(self) -> Self {
                $SrcT::floor(self)
            }
            fn ceil(self) -> Self {
                $SrcT::ceil(self)
            }
            fn round(self) -> Self {
                $SrcT::round(self)
            }
            fn trunc(self) -> Self {
                $SrcT::trunc(self)
            }
            fn fract(self) -> Self {
                $SrcT::fract(self)
            }
        }
        impl Exp for $SrcT {
            type Output = $SrcT;
            fn pow(self, rhs: RHS) -> Self::Output {
                $SrcT::powf(self, rhs)
            }
            fn root(self, n: RHS) -> Self::Output {
                $SrcT::powf(self, n.recip())
            }
            fn sq(self) -> Self::Output {
                self.pow(2)
            }
            fn cb(self) -> Self::Output {
                self.pow(3)
            }
            fn sqrt(self) -> Self::Output {
                self.root(2)
            }
            fn cbrt(self) -> Self::Output {
                self.root(3)
            }
        }
    }
}

impl_float!(f32);
impl_float!(f64);

macro_rules! impl_basic_unit_bounds {
    ($SrcT:ident, $SrcReprT:ident) => {
        impl Unit for $SrcT {}
        impl UnitOps for $SrcT {}
        impl Zero for $SrcT {
            const ZERO: Self = $SrcT::ZERO;
        }
        impl One for $SrcT {
            const ONE: Self = $SrcT::ONE;
            fn recip(self) -> Self {
                $SrcT::recip(self)
            }
        }
        impl Inv for $SrcT {
            fn inv(self) -> Self {
                self.recip()
            }
        }
        impl Symbolic for $SrcT {
            type Base = $SrcT;
        }
        impl PhysicalRepr for $SrcT {
            const BITS: u32 = $SrcT::BITS;
            const BYTES: usize = size_of::<$SrcT>();
            type BitsRepr: Unit + Bitwise + Bounded + Eq = $SrcReprT;
            fn from_bits(v: Self::BitsRepr) -> Self {
                $SrcT::from_bits(v)
            }
            fn to_bits(self) -> Self::BitsRepr {
                $SrcT::to_bits(self)
            }
            fn from_be_bytes(bytes: [u8; Self::BYTES]) -> Self {
                $SrcT::from_be_bytes(bytes)
            }
            fn from_le_bytes(bytes: [u8; Self::BYTES]) -> Self {
                $SrcT::from_le_bytes(bytes)
            }
            fn from_ne_bytes(bytes: [u8; Self::BYTES]) -> Self {
                $SrcT::from_ne_bytes(bytes)
            }
            fn to_be_bytes(self) -> [u8; Self::BYTES] {
                $SrcT::to_be_bytes(self)
            }
            fn to_le_bytes(self) -> [u8; Self::BYTES] {
                $SrcT::to_le_bytes(self)
            }
            fn to_ne_bytes(self) -> [u8; Self::BYTES] {
                $SrcT::to_ne_bytes(self)
            }
        }
    }
}

impl_basic_unit_bounds!(f32, u32);
impl_basic_unit_bounds!(f64, u64);

macro_rules! impl_properties {
    ($SrcT:ident) => {
        impl Ordered for $SrcT {
            fn min(self, other: self) -> Self {
                $SrcT::min(self, other)
            }
            fn max(self, other: self) -> Self {
                $SrcT::max(self, other)
            }
            fn clamp(self, min: self, max: self) -> Self {
                $SrcT::clamp(self, min, max)
            }
        }
        impl OrderedReflexive for $SrcT {}
        impl Bounded for $SrcT {
            const MIN: Self = $SrcT::MIN;
            const MAX: Self = $SrcT::MAX;
        }
    }
}

impl_properties!(f32);
impl_properties!(f64);

macro_rules! impl_properties_signed {
    ($SrcT:ident) => {
        impl Signed for $SrcT {
            const NEG_ONE: Self = $SrcT::NEG_ONE;
            const SIGN_MASK: Self::BitsRepr = $SrcT::SIGN_MASK;
            fn abs(self) -> Self {
                $SrcT::abs(self)
            }
            fn signum(self) -> Self {
                $SrcT::signum(self)
            }
            fn is_positive(self) -> bool {
                $SrcT::is_sign_positive(self)
            }
            fn is_negative(self) -> bool {
                $SrcT::is_sign_negative(self)
            }
        }
        impl BoundedSigned for $SrcT {
            const MIN_POSITIVE: Self = $SrcT::MIN_POSITIVE;
        }
    }
}

impl_properties_signed!(f32);
impl_properties_signed!(f64);