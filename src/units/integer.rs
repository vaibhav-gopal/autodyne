use super::*;

pub trait Integer: Unit + OrderedReflexive + Bounded + Bitwise + Exp + CastPrimitive {}

macro_rules! impl_integer {
    ($SrcT:ident) => {
        impl Integer for $SrcT {}
        impl Bitwise for $SrcT {}
        impl Exp for $SrcT {
            type Output = $SrcT;
            fn pow(self, rhs: RHS) -> Self::Output {
                $SrcT::pow(self, rhs)
            }
            fn root(self, n: RHS) -> Self::Output {
                $SrcT::root(self, n)
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

impl_integer!(u8);
impl_integer!(u16);
impl_integer!(u32);
impl_integer!(u64);
impl_integer!(u128);
impl_integer!(usize);
impl_integer!(i8);
impl_integer!(i16);
impl_integer!(i32);
impl_integer!(i64);
impl_integer!(i128);
impl_integer!(isize);

macro_rules! impl_basic_unit_bounds {
    ($SrcT:ident, $SrcReprT:ident) => {
        impl Unit for $SrcT {}
        impl UnitOps for $SrcT {}
        impl Zero for $SrcT {
            const ZERO: Self = $SrcT::ZERO;
        }
        impl One for $SrcT {
            const ONE: Self = $SrcT::ONE;
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
                v as Self
            }
            fn to_bits(self) -> Self::BitsRepr {
                self as Self::BitsRepr
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

impl_basic_unit_bounds!(u8, u8);
impl_basic_unit_bounds!(u16, u16);
impl_basic_unit_bounds!(u32, u32);
impl_basic_unit_bounds!(u64, u64);
impl_basic_unit_bounds!(u128, u128);
impl_basic_unit_bounds!(usize, usize);
impl_basic_unit_bounds!(i8, u8);
impl_basic_unit_bounds!(i16, u16);
impl_basic_unit_bounds!(i32, u32);
impl_basic_unit_bounds!(i64, u64);
impl_basic_unit_bounds!(i128, u128);
impl_basic_unit_bounds!(isize, usize);

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

impl_properties!(u8);
impl_properties!(u16);
impl_properties!(u32);
impl_properties!(u64);
impl_properties!(u128);
impl_properties!(usize);
impl_properties!(i8);
impl_properties!(i16);
impl_properties!(i32);
impl_properties!(i64);
impl_properties!(i128);
impl_properties!(isize);

macro_rules! impl_properties_signed {
    ($SrcT:ident) => {
        impl Signed for $SrcT {
            const NEG_ONE: Self = -$SrcT::ONE;
            const SIGN_MASK: Self::BitsRepr = Self::BitsRepr::ONE << ($SrcT::BITS - 1);
            fn abs(self) -> Self {
                $SrcT::abs(self)
            }
            fn signum(self) -> Self {
                $SrcT::signum(self)
            }
            fn is_positive(self) -> bool {
                $SrcT::is_positive(self)
            }
            fn is_negative(self) -> bool {
                $SrcT::is_negative(self)
            }
        }
        impl BoundedSigned for $SrcT {
            const MIN_POSITIVE: Self = $SrcT::ZERO;
        }
    }
}

impl_properties_signed!(i8);
impl_properties_signed!(i16);
impl_properties_signed!(i32);
impl_properties_signed!(i64);
impl_properties_signed!(i128);
impl_properties_signed!(isize);