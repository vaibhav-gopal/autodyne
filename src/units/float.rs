use super::*;

pub trait Float: Unit + Ordered + BoundedSigned + ExpFloat + CastPrimitive {
    /// Special states
    const _NAN: Self;
    const _INFINITY: Self;
    const _NEG_INFINITY: Self;
    /// Bounds, and bit-widths for machine representation
    const _EPSILON: Self;
    const _SIG_BITS: u32;
    const _EXP_BITS: u32;
    const _SIG_MASK: Self::BitsRepr;
    const _EXP_MASK: Self::BitsRepr;
    const _DIGITS: u32;
    const _MANTISSA_DIGITS: u32;
    const _MIN_EXP: i32;
    const _MAX_EXP: i32;
    const _MIN_10_EXP: i32;
    const _MAX_10_EXP: i32;
    /// Mathematical constants
    const _PI: Self;
    const _E: Self;
    const _TAU: Self;
    fn _floor(self) -> Self;
    fn _ceil(self) -> Self;
    fn _round(self) -> Self;
    fn _trunc(self) -> Self;
    fn _fract(self) -> Self;
}

macro_rules! impl_float {
    ($SrcT:ident) => {
        impl Float for $SrcT {
            const _NAN: Self = $SrcT::NAN;
            const _INFINITY: Self = $SrcT::INFINITY;
            const _NEG_INFINITY: Self = $SrcT::NEG_INFINITY;
            const _EPSILON: Self = $SrcT::EPSILON;
            const _SIG_BITS: u32 = $SrcT::SIG_BITS;
            const _EXP_BITS: u32 = $SrcT::EXP_BITS;
            const _SIG_MASK: Self::BitsRepr = $SrcT::SIG_MASK;
            const _EXP_MASK: Self::BitsRepr = $SrcT::EXP_MASK;
            const _DIGITS: u32 = $SrcT::DIGITS;
            const _MANTISSA_DIGITS: u32 = $SrcT::MANTISSA_DIGITS;
            const _MIN_EXP: i32 = $SrcT::MIN_EXP;
            const _MAX_EXP: i32 = $SrcT::MAX_EXP;
            const _MIN_10_EXP: i32 = $SrcT::MIN_10_EXP;
            const _MAX_10_EXP: i32 = $SrcT::MAX_10_EXP;
            const _PI: Self = $SrcT::PI;
            const _E: Self = $SrcT::E();
            const _TAU: Self = $SrcT::TAU();
            fn _floor(self) -> Self {
                $SrcT::floor(self)
            }
            fn _ceil(self) -> Self {
                $SrcT::ceil(self)
            }
            fn _round(self) -> Self {
                $SrcT::round(self)
            }
            fn _trunc(self) -> Self {
                $SrcT::trunc(self)
            }
            fn _fract(self) -> Self {
                $SrcT::fract(self)
            }
        }
        impl ExpBasic for $SrcT {
            type Output = $SrcT;
            fn _sq(self) -> <Self as ExpBasic>::Output {
                $SrcT::powi(self, 2i32)
            }
            fn _sqrt(self) -> <Self as ExpBasic>::Output {
                $SrcT::sqrt(self)
            }
        }
        impl ExpPowDynamic<Self> for $SrcT {
            fn _pow(self, rhs: Self) -> <Self as ExpBasic>::Output {
                $SrcT::powf(self, rhs)
            }
        }
        impl ExpRootDynamic<Self> for $SrcT {
            fn _root(self, n: Self) -> <Self as ExpBasic>::Output {
                $SrcT::powf(self, n._recip())
            }
        }
        impl ExpFloat for $SrcT {
            fn _exp(self) -> <Self as ExpBasic>::Output {
                $SrcT::exp(self)
            }
            fn _exp2(self) -> <Self as ExpBasic>::Output {
                $SrcT::exp2(self)
            }
            fn _exp_m1(self) -> <Self as ExpBasic>::Output {
                $SrcT::exp_m1(self)
            }
            fn _log(self, base: Self) -> <Self as ExpBasic>::Output {
                $SrcT::log(self, base)
            }
            fn _log2(self) -> <Self as ExpBasic>::Output {
                $SrcT::log2(self)
            }
            fn _log10(self) -> <Self as ExpBasic>::Output {
                $SrcT::log10(self)
            }
            fn _ln(self) -> <Self as ExpBasic>::Output {
                $SrcT::ln(self)
            }
            fn _ln_1p(self) -> <Self as ExpBasic>::Output {
                $SrcT::ln_1p(self)
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
            const _ZERO: Self = $SrcT::ZERO;
        }
        impl One for $SrcT {
            const _ONE: Self = $SrcT::ONE;
            fn _recip(self) -> Self {
                $SrcT::recip(self)
            }
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
            const _BITS: u32 = $SrcT::BITS;
            const _BYTES: usize = size_of::<$SrcT>();
            type BitsRepr = $SrcReprT;
            type BytesRepr = [u8; size_of::<$SrcT>()];
            fn _from_bits(v: Self::BitsRepr) -> Self {
                $SrcT::from_bits(v)
            }
            fn _to_bits(self) -> Self::BitsRepr {
                $SrcT::to_bits(self)
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

impl_basic_unit_bounds!(f32, u32);
impl_basic_unit_bounds!(f64, u64);

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
        impl Bounded for $SrcT {
            const _MIN: Self = $SrcT::MIN;
            const _MAX: Self = $SrcT::MAX;
        }
    }
}

impl_properties!(f32);
impl_properties!(f64);

macro_rules! impl_properties_signed {
    ($SrcT:ident) => {
        impl Signed for $SrcT {
            const _NEG_ONE: Self = $SrcT::NEG_ONE;
            const _SIGN_MASK: Self::BitsRepr = $SrcT::SIGN_MASK;
            fn _abs(self) -> Self {
                $SrcT::abs(self)
            }
            fn _signum(self) -> Self {
                $SrcT::signum(self)
            }
            fn _is_positive(self) -> bool {
                $SrcT::is_sign_positive(self)
            }
            fn _is_negative(self) -> bool {
                $SrcT::is_sign_negative(self)
            }
        }
        impl BoundedSigned for $SrcT {
            const _MIN_POSITIVE: Self = $SrcT::MIN_POSITIVE;
        }
    }
}

impl_properties_signed!(f32);
impl_properties_signed!(f64);