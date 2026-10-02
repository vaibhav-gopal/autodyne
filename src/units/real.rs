use std::ops::{Add, Div, Mul, Neg, Sub};

/// Real-valued arithmetic that can be traced: the part of [`Float`](super::Float) a processor needs
/// for its maths, without anything that inspects a concrete value.
///
/// Code written over `Real` runs unchanged on `f32` / `f64` (monomorphized, the real-time path) and
/// on `flux::Tracer`, which records the operations into a graph that can be differentiated and
/// compiled. So it may not branch on values: a comparison returns a [`Mask`](Real::Mask) (a plain
/// `bool` for floats, a traced boolean for tracers) and [`select`](Real::select) picks between two
/// results, which becomes `stablehlo.select` when traced.
///
/// Every [`Float`](super::Float) is `Real`.
pub trait Real: Copy + Add<Output = Self> + Sub<Output = Self> + Mul<Output = Self> + Div<Output = Self> + Neg<Output = Self> {
    /// The result of a comparison: `bool` for floats.
    type Mask: Copy;
    /// A constant (sample rates, frequencies, coefficients), rounded to this type.
    fn lit(v: f64) -> Self;
    fn exp(self) -> Self;
    /// Natural logarithm.
    fn ln(self) -> Self;
    fn sin(self) -> Self;
    fn cos(self) -> Self;
    fn tanh(self) -> Self;
    fn sqrt(self) -> Self;
    fn abs(self) -> Self;
    /// `self` raised to the power `e`.
    fn powf(self, e: Self) -> Self;
    fn min(self, other: Self) -> Self;
    fn max(self, other: Self) -> Self;
    /// `self < other`.
    fn less(self, other: Self) -> Self::Mask;
    /// `self > other`.
    fn greater(self, other: Self) -> Self::Mask;
    /// `if mask { if_true } else { if_false }`, without branching on a traced value.
    fn select(mask: Self::Mask, if_true: Self, if_false: Self) -> Self;
}

macro_rules! impl_real {
    ($($T:ident),+) => {$(
        impl Real for $T {
            type Mask = bool;
            #[inline(always)]
            fn lit(v: f64) -> Self {
                v as $T
            }
            #[inline(always)]
            fn exp(self) -> Self {
                $T::exp(self)
            }
            #[inline(always)]
            fn ln(self) -> Self {
                $T::ln(self)
            }
            #[inline(always)]
            fn sin(self) -> Self {
                $T::sin(self)
            }
            #[inline(always)]
            fn cos(self) -> Self {
                $T::cos(self)
            }
            #[inline(always)]
            fn tanh(self) -> Self {
                $T::tanh(self)
            }
            #[inline(always)]
            fn sqrt(self) -> Self {
                $T::sqrt(self)
            }
            #[inline(always)]
            fn abs(self) -> Self {
                $T::abs(self)
            }
            #[inline(always)]
            fn powf(self, e: Self) -> Self {
                $T::powf(self, e)
            }
            #[inline(always)]
            fn min(self, other: Self) -> Self {
                $T::min(self, other)
            }
            #[inline(always)]
            fn max(self, other: Self) -> Self {
                $T::max(self, other)
            }
            #[inline(always)]
            fn less(self, other: Self) -> bool {
                self < other
            }
            #[inline(always)]
            fn greater(self, other: Self) -> bool {
                self > other
            }
            #[inline(always)]
            fn select(mask: bool, if_true: Self, if_false: Self) -> Self {
                if mask { if_true } else { if_false }
            }
        }
    )+};
}

impl_real!(f32, f64);
