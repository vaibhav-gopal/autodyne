use std::ops::{Add, Div, Mul, Neg, Sub};

use super::{Complex, Float};

/// Element-wise arithmetic and elementary functions that can be traced: the maths a processor
/// needs, without anything that inspects a concrete value.
///
/// Implemented by single numbers (`f32`, `f64`, `Complex`), by `NdArray`s of them (element-wise,
/// NumPy-style broadcasting), by runtime-typed `DynArray`s, and by `flux::Tracer`, which records
/// the operations into a graph that can be differentiated and compiled.
///
/// Ordering (comparisons, `select`, `abs`, `minimum` / `maximum`) is [`RealValued`]; per-sample
/// code uses [`Real`] (the `Copy` real types); array code uses
/// [`ArrayMath`](crate::signal::ArrayMath), which adds shapes, reductions, products and FFTs.
pub trait Elementwise: Clone + Add<Output = Self> + Sub<Output = Self> + Mul<Output = Self> + Div<Output = Self> + Neg<Output = Self> {
    /// A constant (sample rates, frequencies, coefficients), rounded to this type. For arrays, a
    /// scalar that broadcasts against any shape.
    fn lit(v: f64) -> Self;
    /// e raised to `self`.
    fn exp(self) -> Self;
    /// Natural logarithm.
    fn ln(self) -> Self;
    /// Sine (radians).
    fn sin(self) -> Self;
    /// Cosine (radians).
    fn cos(self) -> Self;
    /// Hyperbolic tangent.
    fn tanh(self) -> Self;
    /// Square root.
    fn sqrt(self) -> Self;
    /// `self` raised to the power `e`.
    fn powf(self, e: Self) -> Self;
    /// `sin / cos` (numbers override it with their native tangent).
    fn tan(self) -> Self {
        self.clone().sin() / self.cos()
    }
    /// Base-10 logarithm.
    fn log10(self) -> Self {
        self.ln() / Self::lit(std::f64::consts::LN_10)
    }
}

/// [`Elementwise`] values that are real numbers: they can be compared, and so selected between, and
/// have an absolute value of their own type. Code over this trait may not branch on values: a
/// comparison returns a [`Mask`](RealValued::Mask) (a `bool` for numbers, an array of `bool` for
/// arrays, a traced mask for tracers) and [`select`](RealValued::select) picks between two results,
/// which becomes `stablehlo.select` when traced. Names follow NumPy: `minimum` / `maximum` are
/// element-wise (`min` / `max` are reductions).
pub trait RealValued: Elementwise {
    /// The result of a comparison.
    type Mask: Clone;
    /// The absolute value.
    fn abs(self) -> Self;
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
    /// The largest integer not above `self` (its derivative is zero where defined).
    fn floor(self) -> Self;
    /// `self` limited to `[lo, hi]` (NumPy's `clip`).
    fn clip(self, lo: Self, hi: Self) -> Self {
        self.maximum(lo).minimum(hi)
    }
}

/// [`RealValued`] arithmetic on values that are `Copy`: single numbers (`f32`, `f64`) and
/// `flux::Tracer`. The trait per-sample processors are written over, so the same code runs on the
/// audio thread and traces into a differentiable program.
///
/// Every [`Float`](super::Float) is `Real`.
pub trait Real: RealValued + Copy {}

impl<T: RealValued + Copy> Real for T {}

macro_rules! impl_elementwise {
    ($($T:ident),+) => {$(
        impl Elementwise for $T {
            #[inline(always)]
            fn lit(v: f64) -> Self {
                v as $T
            }
            #[inline(always)]
            fn exp(self) -> Self {
                <$T as super::fastmath::Transcendental>::t_exp(self)
            }
            #[inline(always)]
            fn ln(self) -> Self {
                <$T as super::fastmath::Transcendental>::t_ln(self)
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
                <$T as super::fastmath::Transcendental>::t_tanh(self)
            }
            #[inline(always)]
            fn sqrt(self) -> Self {
                $T::sqrt(self)
            }
            #[inline(always)]
            fn powf(self, e: Self) -> Self {
                $T::powf(self, e)
            }
            #[inline(always)]
            fn tan(self) -> Self {
                $T::tan(self)
            }
            #[inline(always)]
            fn log10(self) -> Self {
                $T::log10(self)
            }
        }

        impl RealValued for $T {
            type Mask = bool;
            #[inline(always)]
            fn abs(self) -> Self {
                $T::abs(self)
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
            #[inline(always)]
            fn floor(self) -> Self {
                $T::floor(self)
            }
        }
    )+};
}

impl_elementwise!(f32, f64);

/// Complex numbers: arithmetic and the principal branches of the elementary functions.
impl<T: Float> Elementwise for Complex<T> {
    fn lit(v: f64) -> Self {
        Complex::new(T::_lit(v), T::_ZERO)
    }
    fn exp(self) -> Self {
        Complex::exp(self)
    }
    fn ln(self) -> Self {
        Complex::ln(self)
    }
    fn sin(self) -> Self {
        Complex::sin(self)
    }
    fn cos(self) -> Self {
        Complex::cos(self)
    }
    fn tanh(self) -> Self {
        Complex::tanh(self)
    }
    fn sqrt(self) -> Self {
        Complex::sqrt(self)
    }
    fn powf(self, e: Self) -> Self {
        if self.re == T::_ZERO && self.im == T::_ZERO {
            return if e.re == T::_ZERO && e.im == T::_ZERO { Complex::one() } else { Complex::zero() };
        }
        (e * Complex::ln(self)).exp()
    }
}
