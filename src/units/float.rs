//! [`Float`]: the concrete floating-point types (`f32`, `f64`), real arithmetic plus everything that
//! inspects a value.

use super::float_math::FloatMath;
use super::*;

/// A concrete floating-point number (`f32`, `f64`): [`Real`] arithmetic plus everything that
/// inspects a value (bits, rounding, NaN checks, ordering).
pub trait Float: Real + Unit + Ordered + BoundedSigned + ExpBasic<Output = Self> + ExpFloat + Trig + CastPrimitive {
    /// Not a number.
    const _NAN: Self;
    /// Positive infinity.
    const _INFINITY: Self;
    /// Negative infinity.
    const _NEG_INFINITY: Self;
    /// The gap between 1 and the next larger value.
    const _EPSILON: Self;
    /// Explicitly stored significand bits (23 for `f32`, 52 for `f64`).
    const _SIG_BITS: u32;
    /// Exponent bits.
    const _EXP_BITS: u32;
    /// Mask of the significand bits in the bit representation.
    const _SIG_MASK: Self::BitsRepr;
    /// Mask of the exponent bits in the bit representation.
    const _EXP_MASK: Self::BitsRepr;
    /// Decimal digits that survive a round trip through this type.
    const _DIGITS: u32;
    /// Significand digits in base 2, the implicit leading bit included.
    const _MANTISSA_DIGITS: u32;
    /// One more than the smallest normal power of two.
    const _MIN_EXP: i32;
    /// One more than the largest power of two.
    const _MAX_EXP: i32;
    /// The smallest power of ten that is a normal value.
    const _MIN_10_EXP: i32;
    /// The largest power of ten that is a finite value.
    const _MAX_10_EXP: i32;
    /// π.
    const _PI: Self;
    /// Euler's number e.
    const _E: Self;
    /// τ = 2π, a full turn in radians.
    const _TAU: Self;
    /// Magnitude below which `_flush_denormal` returns zero: 1e-30, about -600 dB.
    const _FLUSH_THRESHOLD: Self;
    /// The largest integer at most `self`.
    fn _floor(self) -> Self;
    /// The smallest integer at least `self`.
    fn _ceil(self) -> Self;
    /// The nearest integer, halves away from zero.
    fn _round(self) -> Self;
    /// The integer part (rounded toward zero).
    fn _trunc(self) -> Self;
    /// `self - trunc(self)`.
    fn _fract(self) -> Self;
    /// Whether `self` is NaN.
    fn _is_nan(self) -> bool;
    /// Whether `self` is neither infinite nor NaN.
    fn _is_finite(self) -> bool;
    /// Converts an f64 constant (sample rates, frequencies, coefficients) into this float type.
    /// Panics only if the value is out of range for Self, which f64 -> f32 constants never are in practice.
    fn _lit(v: f64) -> Self {
        Self::from_f64(v).expect("f64 constant out of range for target float")
    }
    /// `self`, or exactly zero when its magnitude is below 1e-30 (about -600 dB, far below anything
    /// audible).
    ///
    /// Feedback loops (reverbs, recursive filters, feedback delays, envelope followers) apply this to
    /// their state so a decaying tail reaches zero instead of sinking into subnormal numbers, which many
    /// CPUs process 10-100x slower. This works without changing the CPU's floating-point mode
    /// (flush-to-zero), which is unsafe in Rust and up to the host.
    #[inline(always)]
    fn _flush_denormal(self) -> Self {
        if self._abs() < Self::_FLUSH_THRESHOLD { Self::_ZERO } else { self }
    }
}

macro_rules! impl_float {
    ($SrcT:ident, $SigBits:expr) => {
        impl Float for $SrcT {
            const _NAN: Self = $SrcT::NAN;
            const _INFINITY: Self = $SrcT::INFINITY;
            const _NEG_INFINITY: Self = $SrcT::NEG_INFINITY;
            const _EPSILON: Self = $SrcT::EPSILON;
            const _SIG_BITS: u32 = $SigBits;
            const _EXP_BITS: u32 = Self::_BITS - Self::_SIG_BITS - 1;
            const _SIG_MASK: Self::BitsRepr = (1 << Self::_SIG_BITS) - 1;
            const _EXP_MASK: Self::BitsRepr = ! (Self::_SIG_MASK | Self::_SIGN_MASK);
            const _DIGITS: u32 = $SrcT::DIGITS;
            const _MANTISSA_DIGITS: u32 = $SrcT::MANTISSA_DIGITS;
            const _MIN_EXP: i32 = $SrcT::MIN_EXP;
            const _MAX_EXP: i32 = $SrcT::MAX_EXP;
            const _MIN_10_EXP: i32 = $SrcT::MIN_10_EXP;
            const _MAX_10_EXP: i32 = $SrcT::MAX_10_EXP;
            const _PI: Self = core::$SrcT::consts::PI;
            const _E: Self = core::$SrcT::consts::E;
            const _TAU: Self = core::$SrcT::consts::TAU;
            const _FLUSH_THRESHOLD: Self = 1e-30;
            fn _floor(self) -> Self {
                FloatMath::floor(&self)
            }
            fn _ceil(self) -> Self {
                FloatMath::ceil(&self)
            }
            fn _round(self) -> Self {
                FloatMath::round(&self)
            }
            fn _trunc(self) -> Self {
                FloatMath::trunc(&self)
            }
            fn _fract(self) -> Self {
                FloatMath::fract(&self)
            }
            fn _is_nan(self) -> bool {
                $SrcT::is_nan(self)
            }
            fn _is_finite(self) -> bool {
                $SrcT::is_finite(self)
            }
        }
        impl ExpBasic for $SrcT {
            type Output = $SrcT;
            fn _sq(self) -> <Self as ExpBasic>::Output {
                FloatMath::powi(&self, 2i32)
            }
            fn _sqrt(self) -> <Self as ExpBasic>::Output {
                FloatMath::sqrt(&self)
            }
        }
        impl ExpPowDynamic<Self> for $SrcT {
            fn _pow(self, rhs: Self) -> <Self as ExpBasic>::Output {
                FloatMath::powf(&self, rhs)
            }
        }
        impl ExpRootDynamic<Self> for $SrcT {
            fn _root(self, n: Self) -> <Self as ExpBasic>::Output {
                FloatMath::powf(&self, n._recip())
            }
        }
        impl Trig for $SrcT {
            fn _sin(self) -> Self {
                FloatMath::sin(&self)
            }
            fn _cos(self) -> Self {
                FloatMath::cos(&self)
            }
            fn _tan(self) -> Self {
                FloatMath::tan(&self)
            }
            fn _sin_cos(self) -> (Self, Self) {
                FloatMath::sin_cos(&self)
            }
            fn _asin(self) -> Self {
                FloatMath::asin(&self)
            }
            fn _acos(self) -> Self {
                FloatMath::acos(&self)
            }
            fn _atan(self) -> Self {
                FloatMath::atan(&self)
            }
            fn _atan2(self, other: Self) -> Self {
                FloatMath::atan2(&self, other)
            }
            fn _hypot(self, other: Self) -> Self {
                FloatMath::hypot(&self, other)
            }
        }
        impl ExpFloat for $SrcT {
            fn _exp(self) -> <Self as ExpBasic>::Output {
                FloatMath::exp(&self)
            }
            fn _exp2(self) -> <Self as ExpBasic>::Output {
                FloatMath::exp2(&self)
            }
            fn _exp_m1(self) -> <Self as ExpBasic>::Output {
                FloatMath::exp_m1(&self)
            }
            fn _log(self, base: Self) -> <Self as ExpBasic>::Output {
                FloatMath::log(&self, base)
            }
            fn _log2(self) -> <Self as ExpBasic>::Output {
                FloatMath::log2(&self)
            }
            fn _log10(self) -> <Self as ExpBasic>::Output {
                FloatMath::log10(&self)
            }
            fn _ln(self) -> <Self as ExpBasic>::Output {
                FloatMath::ln(&self)
            }
            fn _ln_1p(self) -> <Self as ExpBasic>::Output {
                FloatMath::ln_1p(&self)
            }
        }
    }
}

impl_float!(f32, 23);
impl_float!(f64, 52);

macro_rules! impl_basic_unit_bounds {
    ($SrcT:ident, $SrcReprT:ident) => {
        impl Unit for $SrcT {}
        impl UnitOps for $SrcT {}
        impl Zero for $SrcT {
            const _ZERO: Self = 0 as $SrcT;
        }
        impl One for $SrcT {
            const _ONE: Self = 1 as $SrcT;
            fn _recip(self) -> Self {
                $SrcT::recip(self)
            }
        }
        impl Inv for $SrcT {
            fn _inv(self) -> Self {
                self._recip()
            }
        }
        impl PhysicalRepr for $SrcT {
            const _BITS: u32 = size_of::<$SrcT>() as u32 * 8;
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
            const _NEG_ONE: Self = -$SrcT::_ONE;
            const _SIGN_MASK: Self::BitsRepr = 1 << ($SrcT::_BITS - 1);
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