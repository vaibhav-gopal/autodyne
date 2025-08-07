use super::*;

pub trait Integer: Unit + OrderedReflexive + Bounded + Bitwise + ExpPowDynamic<u32> + CastPrimitive {}

macro_rules! impl_integer {
    ($SrcT:ident) => {
        impl Integer for $SrcT {}
        impl Bitwise for $SrcT {}
        impl ExpBasic for $SrcT {
            type Output = $SrcT;
            fn _sq(self) -> <Self as ExpBasic>::Output {
                $SrcT::pow(self, 2u32)
            }
            fn _sqrt(self) -> <Self as ExpBasic>::Output {
                $SrcT::isqrt(self)
            }
        }
        impl ExpPowDynamic<u32> for $SrcT {
            fn _pow(self, rhs: u32) -> <Self as ExpBasic>::Output {
                $SrcT::pow(self, rhs)
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
            const _ZERO: Self = 0 as $SrcT;
        }
        impl One for $SrcT {
            const _ONE: Self = 1 as $SrcT;
        }
        impl Inv for $SrcT {
            fn _inv(self) -> Self {
                self._recip()
            }
        }
        impl Symbolic for $SrcT {
            type Base = $SrcT;
        }
        impl PhysicalRepr for $SrcT {
            const _BITS: u32 = size_of::<$SrcT>() as u32 * 8;
            const _BYTES: usize = size_of::<$SrcT>();
            type BitsRepr = $SrcReprT;
            type BytesRepr = [u8; size_of::<$SrcT>()];
            fn _from_bits(v: Self::BitsRepr) -> Self {
                v as Self
            }
            fn _to_bits(self) -> Self::BitsRepr {
                self as Self::BitsRepr
            }
            fn _from_be_bytes(bytes: Self::BytesRepr) -> Self {
                $SrcT::from_be_bytes(bytes)
            }
            fn _from_le_bytes(bytes: Self::BytesRepr) -> Self {
                $SrcT::from_le_bytes(bytes)
            }
            fn _from_ne_bytes(bytes: Self::BytesRepr) -> Self {
                $SrcT::from_ne_bytes(bytes)
            }
            fn _to_be_bytes(self) -> Self::BytesRepr {
                $SrcT::to_be_bytes(self)
            }
            fn _to_le_bytes(self) -> Self::BytesRepr {
                $SrcT::to_le_bytes(self)
            }
            fn _to_ne_bytes(self) -> Self::BytesRepr {
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
            fn _min(self, other: Self) -> Self {
                $SrcT::min(self, other)
            }
            fn _max(self, other: Self) -> Self {
                $SrcT::max(self, other)
            }
            fn _clamp(self, min: Self, max: Self) -> Self {
                $SrcT::clamp(self, min, max)
            }
        }
        impl OrderedReflexive for $SrcT {}
        impl Bounded for $SrcT {
            const _MIN: Self = $SrcT::MIN;
            const _MAX: Self = $SrcT::MAX;
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
            const _NEG_ONE: Self = -$SrcT::_ONE;
            const _SIGN_MASK: Self::BitsRepr = Self::BitsRepr::_ONE << ($SrcT::_BITS - 1);
            fn _abs(self) -> Self {
                $SrcT::abs(self)
            }
            fn _signum(self) -> Self {
                $SrcT::signum(self)
            }
            fn _is_positive(self) -> bool {
                $SrcT::is_positive(self)
            }
            fn _is_negative(self) -> bool {
                $SrcT::is_negative(self)
            }
        }
        impl BoundedSigned for $SrcT {
            const _MIN_POSITIVE: Self = $SrcT::_ZERO;
        }
    }
}

impl_properties_signed!(i8);
impl_properties_signed!(i16);
impl_properties_signed!(i32);
impl_properties_signed!(i64);
impl_properties_signed!(i128);
impl_properties_signed!(isize);