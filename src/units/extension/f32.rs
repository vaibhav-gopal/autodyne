use super::*;
use delegate::delegate;

impl Float for f32 {
    const NAN: Self = f32::NAN;
    const INFINITY: Self = f32::INFINITY;
    const NEG_INFINITY: Self = f32::NEG_INFINITY;
    const EPSILON: Self = f32::EPSILON;
    const MANTISSA_DIGITS: u32 = f32::MANTISSA_DIGITS;
    const PI: Self = f32::PI;
    const E: Self = f32::E;
    const TAU: Self = f32::TAU;

    fn floor(self) -> Self {
        todo!()
    }

    fn ceil(self) -> Self {
        todo!()
    }

    fn round(self) -> Self {
        todo!()
    }

    fn trunc(self) -> Self {
        todo!()
    }

    fn fract(self) -> Self {
        todo!()
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

impl Inv for f32 {
    type Output = Self;
    fn inv(self) -> Self::Output {
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