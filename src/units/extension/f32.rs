use super::*;
use delegate::delegate;

impl Float for f32 {
    const NAN: Self = f32::NAN;
    const INFINITY: Self = f32::INFINITY;
    const NEG_INFINITY: Self = f32::NEG_INFINITY;
    const EPSILON: Self = f32::EPSILON;
    const SIG_BITS: u32 = f32::SIG_BITS;
    const EXP_BITS: u32 = f32::EXP_BITS;
    const SIG_MASK: Self::IntRepr = f32::SIG_MASK;
    const EXP_MASK: Self::IntRepr = f32::EXP_MASK;
    const DIGITS: u32 = f32::DIGITS;
    const MANTISSA_DIGITS: u32 = f32::MANTISSA_DIGITS;
    const MIN_EXP: i32 = f32::MIN_EXP;
    const MAX_EXP: i32 = f32::MAX_EXP;
    const MIN_10_EXP: i32 = f32::MIN_10_EXP;
    const MAX_10_EXP: i32 = f32::MAX_10_EXP;
    const PI: Self = f32::PI;
    const E: Self = f32::E;
    const TAU: Self = f32::TAU;
    
    delegate! {
        #[through(f32)]
        #[allow(unconditional_recursion)]
        to self {
            fn floor(self) -> Self;
            fn ceil(self) -> Self;
            fn round(self) -> Self;
            fn trunc(self) -> Self;
            fn fract(self) -> Self;
        }
    }
}

impl FromPrimitive for f32 {
    fn from_i64(n: i64) -> Option<Self> {
        todo!()
    }

    fn from_u64(n: u64) -> Option<Self> {
        todo!()
    }

    fn from_isize(n: isize) -> Option<Self> {
        todo!()
    }

    fn from_i8(n: i8) -> Option<Self> {
        todo!()
    }

    fn from_i16(n: i16) -> Option<Self> {
        todo!()
    }

    fn from_i32(n: i32) -> Option<Self> {
        todo!()
    }

    fn from_i128(n: i128) -> Option<Self> {
        todo!()
    }

    fn from_usize(n: usize) -> Option<Self> {
        todo!()
    }

    fn from_u8(n: u8) -> Option<Self> {
        todo!()
    }

    fn from_u16(n: u16) -> Option<Self> {
        todo!()
    }

    fn from_u32(n: u32) -> Option<Self> {
        todo!()
    }

    fn from_u128(n: u128) -> Option<Self> {
        todo!()
    }

    fn from_f32(n: f32) -> Option<Self> {
        todo!()
    }

    fn from_f64(n: f64) -> Option<Self> {
        todo!()
    }

    fn from_str(n: &str) -> Option<Self> {
        todo!()
    }
}

impl ToPrimitive for f32 {
    fn to_i64(self) -> Option<i64> {
        todo!()
    }

    fn to_u64(self) -> Option<u64> {
        todo!()
    }

    fn to_isize(self) -> Option<isize> {
        todo!()
    }

    fn to_i8(self) -> Option<i8> {
        todo!()
    }

    fn to_i16(self) -> Option<i16> {
        todo!()
    }

    fn to_i32(self) -> Option<i32> {
        todo!()
    }

    fn to_i128(self) -> Option<i128> {
        todo!()
    }

    fn to_usize(self) -> Option<usize> {
        todo!()
    }

    fn to_u8(self) -> Option<u8> {
        todo!()
    }

    fn to_u16(self) -> Option<u16> {
        todo!()
    }

    fn to_u32(self) -> Option<u32> {
        todo!()
    }

    fn to_u128(self) -> Option<u128> {
        todo!()
    }

    fn to_f32(self) -> Option<f32> {
        todo!()
    }

    fn to_f64(self) -> Option<f64> {
        todo!()
    }

    fn to_str(self) -> Option<String> {
        todo!()
    }
}

impl<T> AsPrimitive<T> for f32 {
    fn as_(self) -> T {
        todo!()
    }
}

impl CastPrimitive for f32 {}

impl Exp for f32 {
    type Output = Self;
    
    delegate! {
        #[through(f32)]
        #[allow(unconditional_recursion)]
        to self {
            #[call(powf)]
            fn pow(self, rhs: self) -> Self::Output;
            fn sqrt(self) -> Self::Output;
            fn cbrt(self) -> Self::Output;
        }
    }
    
    fn root(self, n: self) -> Self::Output {
        self.pow(Self::recip(n))
    }
    
    fn sq(self) -> Self::Output {
        self.pow(2f32)
    }

    fn cb(self) -> Self::Output {
        self.pow(3f32)
    }
}

impl Ordered for f32 {
    delegate! {
        #[through(f32)]
        #[allow(unconditional_recursion)]
        to self {
            fn min(self, other: self) -> Self;
            fn max(self, other: self) -> Self;
            fn clamp(self, min: self, max: self) -> Self;
        }
    }
}

impl BoundedSigned for f32 {
    const MIN_POSITIVE: Self = f32::MIN_POSITIVE;
}

impl Bounded for f32 {
    const MIN: Self = f32::MIN;
    const MAX: Self = f32::MAX;
}

impl Signed for f32 {
    const SIGN_MASK: Self::IntRepr = f32::SIGN_MASK;
    const NEG_ONE: Self = f32::NEG_ONE;
    delegate! {
        #[through(f32)]
        #[allow(unconditional_recursion)]
        to self {
            fn signum(self) -> Self;
            fn abs(self) -> Self;
            fn is_sign_negative(self) -> Self;
            fn is_sign_positive(self) -> Self;
        }
    }
}

impl PhysicalRepr for f32 {
    const BITS: u32 = f32::BITS;
    const BYTES: u32 = f32::BITS / 8;
    type IntRepr = u32;
    delegate! {
        #[through(f32)]
        #[allow(unconditional_recursion)]
        to self {
            fn to_bits(self) -> Self::IntRepr;
            fn to_be_bytes(self) -> [u8; Self::BYTES as usize];
            fn to_le_bytes(self) -> [u8; Self::BYTES as usize];
            fn to_ne_bytes(self) -> [u8; Self::BYTES as usize];
        }
        to f32 {
            fn from_bits(v: Self::IntRepr) -> Self;
            fn from_be_bytes(bytes: [u8; Self::BYTES as usize]) -> Self;
            fn from_le_bytes(bytes: [u8; Self::BYTES as usize]) -> Self;
            fn from_ne_bytes(bytes: [u8; Self::BYTES as usize]) -> Self;
        }
    }
}

impl Symbolic for f32 {
    type Base = f32;
}

impl Inv for f32 {
    fn inv(self) -> Self {
        self.recip()
    }
}

impl Zero for f32 {
    const ZERO: Self = f32::ZERO;
}

impl One for f32 {
    const ONE: Self = f32::ONE;
    delegate! {
        #[through(f32)]
        #[allow(unconditional_recursion)]
        to self {
            fn recip(self) -> Self;
        }
    }
}

impl UnitOps for f32 {}

impl Unit for f32 {}