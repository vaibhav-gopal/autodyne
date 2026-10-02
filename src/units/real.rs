use std::ops::{Add, Div, Mul, Neg, Sub};

/// Element-wise real arithmetic that can be traced: the maths a processor needs, without anything
/// that inspects a concrete value.
///
/// Implemented by single numbers (`f32`, `f64`), by `NdArray<f32 / f64>` (element-wise, NumPy-style
/// broadcasting) and by `flux::Tracer`, which records the operations into a graph that can be
/// differentiated and compiled. Code written over this trait may not branch on values: a
/// comparison returns a [`Mask`](Elementwise::Mask) (`bool` for numbers, an array of `bool` for
/// arrays, a traced mask for tracers) and [`select`](Elementwise::select) picks between two
/// results, which becomes `stablehlo.select` when traced.
///
/// Per-sample code uses [`Real`] (the `Copy` types); array code uses
/// [`ArrayMath`](crate::signal::ArrayMath), which adds shapes, reductions, products and FFTs.
/// Names follow NumPy: `minimum` / `maximum` are element-wise (`min` / `max` are reductions).
pub trait Elementwise: Clone + Add<Output = Self> + Sub<Output = Self> + Mul<Output = Self> + Div<Output = Self> + Neg<Output = Self> {
    /// The result of a comparison.
    type Mask: Clone;
    /// A constant (sample rates, frequencies, coefficients), rounded to this type. For arrays, a
    /// scalar that broadcasts against any shape.
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
    /// The smaller of each pair of elements.
    fn minimum(self, other: Self) -> Self;
    /// The larger of each pair of elements.
    fn maximum(self, other: Self) -> Self;
    /// `self < other`.
    fn less(self, other: Self) -> Self::Mask;
    /// `self > other`.
    fn greater(self, other: Self) -> Self::Mask;
    /// `if mask { if_true } else { if_false }`, without branching on a traced value.
    fn select(mask: Self::Mask, if_true: Self, if_false: Self) -> Self;
}

/// [`Elementwise`] arithmetic on values that are `Copy`: single numbers (`f32`, `f64`) and
/// `flux::Tracer`. The trait per-sample processors are written over, so the same code runs on the
/// audio thread and traces into a differentiable program.
///
/// Every [`Float`](super::Float) is `Real`.
pub trait Real: Elementwise + Copy {}

impl<T: Elementwise + Copy> Real for T {}

macro_rules! impl_elementwise {
    ($($T:ident),+) => {$(
        impl Elementwise for $T {
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
            fn minimum(self, other: Self) -> Self {
                $T::min(self, other)
            }
            #[inline(always)]
            fn maximum(self, other: Self) -> Self {
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

impl_elementwise!(f32, f64);
