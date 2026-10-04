//! Fixed-point numbers in Q format: an integer holding the value times `2^FRAC`.
//!
//! Arithmetic saturates (the DSP convention: an overflow clips instead of wrapping around), and
//! multiplication rounds to nearest; `wrapping_*` and `checked_*` variants exist. `Fixed` implements
//! [`Real`](super::Real), so processors written over it run in fixed point unchanged, which
//! simulates a fixed-point implementation (elementary functions are evaluated in `f64` and rounded
//! to the format).
//!
//! ```
//! use autodyne::units::{Q15, Elementwise};
//!
//! let a = Q15::from_f64(0.75);
//! let b = Q15::from_f64(0.5);
//! assert_eq!((a * b).to_f64(), 0.375);
//! assert_eq!((a + b).to_f64(), Q15::MAX.to_f64()); // saturates just below 1
//! assert_eq!(Q15::lit(0.1).to_bits(), 3277);
//! ```

use std::fmt;
use std::ops::{Add, Div, Mul, Neg, Rem, Sub};

use super::{Elementwise, RealValued};

/// The integer types fixed-point values are stored in.
pub trait FixedStorage: Copy + Ord + Eq + std::hash::Hash + fmt::Debug + Default + Send + Sync + 'static {
    /// Width in bits.
    const BITS: u32;
    /// The smallest value.
    const MIN: Self;
    /// The largest value.
    const MAX: Self;
    /// The value widened to `i128`.
    fn fixed_raw(self) -> i128;
    /// Clamps to the type's range.
    fn fixed_saturate(v: i128) -> Self;
    /// Keeps the low bits (two's complement wrap-around).
    fn fixed_wrap(v: i128) -> Self;
}

macro_rules! storage {
    ($($T:ty),+) => {$(
        impl FixedStorage for $T {
            const BITS: u32 = <$T>::BITS;
            const MIN: Self = <$T>::MIN;
            const MAX: Self = <$T>::MAX;
            #[inline]
            fn fixed_raw(self) -> i128 {
                self as i128
            }
            #[inline]
            fn fixed_saturate(v: i128) -> Self {
                v.clamp(<$T>::MIN as i128, <$T>::MAX as i128) as $T
            }
            #[inline]
            fn fixed_wrap(v: i128) -> Self {
                v as $T
            }
        }
    )+};
}

storage!(i8, i16, i32, i64);

/// A signed fixed-point number: `bits / 2^FRAC`, stored in `I`.
#[derive(Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, Default)]
#[repr(transparent)]
pub struct Fixed<I: FixedStorage, const FRAC: u32>(I);

/// Q0.7: [-1, 1) in steps of 2⁻⁷.
pub type Q7 = Fixed<i8, 7>;
/// Q0.15: [-1, 1) in steps of 2⁻¹⁵ (16-bit audio).
pub type Q15 = Fixed<i16, 15>;
/// Q0.31: [-1, 1) in steps of 2⁻³¹ (32-bit audio).
pub type Q31 = Fixed<i32, 31>;
/// Q16.16: ±32768 in steps of 2⁻¹⁶.
pub type Q16_16 = Fixed<i32, 16>;
/// Q32.32: ±2³¹ in steps of 2⁻³².
pub type Q32_32 = Fixed<i64, 32>;

impl<I: FixedStorage, const FRAC: u32> Fixed<I, FRAC> {
    /// The smallest value.
    pub const MIN: Self = Fixed(I::MIN);
    /// The largest value.
    pub const MAX: Self = Fixed(I::MAX);
    /// One step: `2^-FRAC`.
    pub fn epsilon() -> Self {
        Fixed(I::fixed_saturate(1))
    }
    /// The value with raw representation `bits`.
    pub const fn from_bits(bits: I) -> Self {
        Fixed(bits)
    }
    /// The raw representation.
    pub const fn to_bits(self) -> I {
        self.0
    }
    fn scale() -> f64 {
        2f64.powi(FRAC as i32)
    }
    /// The nearest representable value (ties away from zero), saturating; NaN gives 0.
    pub fn from_f64(v: f64) -> Self {
        if v.is_nan() {
            return Fixed(I::fixed_saturate(0));
        }
        let scaled = (v * Self::scale()).round();
        Fixed(I::fixed_saturate(if scaled >= i128::MAX as f64 { i128::MAX } else if scaled <= i128::MIN as f64 { i128::MIN } else { scaled as i128 }))
    }
    /// The value as `f64` (exact while the format has at most 53 significant bits).
    pub fn to_f64(self) -> f64 {
        self.0.fixed_raw() as f64 / Self::scale()
    }
    /// The integer `n` (saturating).
    pub fn from_int(n: i64) -> Self {
        Fixed(I::fixed_saturate((n as i128) << FRAC))
    }

    fn raw(self) -> i128 {
        self.0.fixed_raw()
    }
    /// The product's raw value, rounded to nearest (ties away from zero).
    fn mul_raw(self, rhs: Self) -> i128 {
        let p = self.raw() * rhs.raw();
        if FRAC == 0 {
            return p;
        }
        let half = 1i128 << (FRAC - 1);
        if p >= 0 { (p + half) >> FRAC } else { -((-p + half) >> FRAC) }
    }
    /// The quotient's raw value, rounded toward zero; `None` for division by zero.
    fn div_raw(self, rhs: Self) -> Option<i128> {
        (rhs.raw() != 0).then(|| (self.raw() << FRAC) / rhs.raw())
    }

    /// `self + rhs`, clamped to the representable range.
    pub fn saturating_add(self, rhs: Self) -> Self {
        Fixed(I::fixed_saturate(self.raw() + rhs.raw()))
    }
    /// `self - rhs`, clamped to the representable range.
    pub fn saturating_sub(self, rhs: Self) -> Self {
        Fixed(I::fixed_saturate(self.raw() - rhs.raw()))
    }
    /// `self * rhs` (rounded to nearest), clamped to the representable range.
    pub fn saturating_mul(self, rhs: Self) -> Self {
        Fixed(I::fixed_saturate(self.mul_raw(rhs)))
    }
    /// Saturates; dividing by zero gives the extreme of the dividend's sign.
    pub fn saturating_div(self, rhs: Self) -> Self {
        match self.div_raw(rhs) {
            Some(q) => Fixed(I::fixed_saturate(q)),
            None if self.raw() < 0 => Self::MIN,
            None => Self::MAX,
        }
    }
    /// `self + rhs`, wrapping around on overflow (two's complement).
    pub fn wrapping_add(self, rhs: Self) -> Self {
        Fixed(I::fixed_wrap(self.raw() + rhs.raw()))
    }
    /// `self - rhs`, wrapping around on overflow.
    pub fn wrapping_sub(self, rhs: Self) -> Self {
        Fixed(I::fixed_wrap(self.raw() - rhs.raw()))
    }
    /// `self * rhs` (rounded to nearest), wrapping around on overflow.
    pub fn wrapping_mul(self, rhs: Self) -> Self {
        Fixed(I::fixed_wrap(self.mul_raw(rhs)))
    }
    fn checked(v: i128) -> Option<Self> {
        let s = I::fixed_saturate(v);
        (s.fixed_raw() == v).then_some(Fixed(s))
    }
    /// `self + rhs`, or `None` on overflow.
    pub fn checked_add(self, rhs: Self) -> Option<Self> {
        Self::checked(self.raw() + rhs.raw())
    }
    /// `self - rhs`, or `None` on overflow.
    pub fn checked_sub(self, rhs: Self) -> Option<Self> {
        Self::checked(self.raw() - rhs.raw())
    }
    /// `self * rhs` (rounded to nearest), or `None` on overflow.
    pub fn checked_mul(self, rhs: Self) -> Option<Self> {
        Self::checked(self.mul_raw(rhs))
    }
    /// `self / rhs` (rounded toward zero), or `None` on overflow or division by zero.
    pub fn checked_div(self, rhs: Self) -> Option<Self> {
        self.div_raw(rhs).and_then(Self::checked)
    }
}

impl<I: FixedStorage, const FRAC: u32> Add for Fixed<I, FRAC> {
    type Output = Self;
    fn add(self, rhs: Self) -> Self {
        self.saturating_add(rhs)
    }
}
impl<I: FixedStorage, const FRAC: u32> Sub for Fixed<I, FRAC> {
    type Output = Self;
    fn sub(self, rhs: Self) -> Self {
        self.saturating_sub(rhs)
    }
}
impl<I: FixedStorage, const FRAC: u32> Mul for Fixed<I, FRAC> {
    type Output = Self;
    fn mul(self, rhs: Self) -> Self {
        self.saturating_mul(rhs)
    }
}
impl<I: FixedStorage, const FRAC: u32> Div for Fixed<I, FRAC> {
    type Output = Self;
    fn div(self, rhs: Self) -> Self {
        self.saturating_div(rhs)
    }
}
impl<I: FixedStorage, const FRAC: u32> Rem for Fixed<I, FRAC> {
    type Output = Self;
    /// The remainder (the sign of the dividend); 0 for division by zero.
    fn rem(self, rhs: Self) -> Self {
        if rhs.raw() == 0 { Fixed(I::fixed_saturate(0)) } else { Fixed(I::fixed_saturate(self.raw() % rhs.raw())) }
    }
}
impl<I: FixedStorage, const FRAC: u32> Neg for Fixed<I, FRAC> {
    type Output = Self;
    /// Saturating: `-MIN` is `MAX`.
    fn neg(self) -> Self {
        Fixed(I::fixed_saturate(-self.raw()))
    }
}

/// Exact decimal: every fixed-point value has a finite decimal expansion.
impl<I: FixedStorage, const FRAC: u32> fmt::Display for Fixed<I, FRAC> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let raw = self.raw();
        let (neg, mag) = (raw < 0, raw.unsigned_abs());
        let int = mag >> FRAC;
        let mut frac = mag & ((1u128 << FRAC) - 1);
        let mut digits = String::new();
        let limit = f.precision();
        while frac != 0 && limit.is_none_or(|p| digits.len() < p) {
            frac *= 10;
            digits.push(char::from(b'0' + (frac >> FRAC) as u8));
            frac &= (1u128 << FRAC) - 1;
        }
        if let Some(p) = limit {
            while digits.len() < p {
                digits.push('0');
            }
        }
        let sign = if neg { "-" } else { "" };
        if digits.is_empty() { write!(f, "{sign}{int}") } else { write!(f, "{sign}{int}.{digits}") }
    }
}

impl<I: FixedStorage, const FRAC: u32> fmt::Debug for Fixed<I, FRAC> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{self} (Q{}.{FRAC})", I::BITS - 1 - FRAC.min(I::BITS - 1))
    }
}

/// Elementary functions in `f64`, rounded to the format.
impl<I: FixedStorage, const FRAC: u32> Elementwise for Fixed<I, FRAC> {
    fn lit(v: f64) -> Self {
        Self::from_f64(v)
    }
    fn exp(self) -> Self {
        Self::from_f64(self.to_f64().exp())
    }
    fn ln(self) -> Self {
        Self::from_f64(self.to_f64().ln())
    }
    fn sin(self) -> Self {
        Self::from_f64(self.to_f64().sin())
    }
    fn cos(self) -> Self {
        Self::from_f64(self.to_f64().cos())
    }
    fn tanh(self) -> Self {
        Self::from_f64(self.to_f64().tanh())
    }
    fn sqrt(self) -> Self {
        Self::from_f64(self.to_f64().sqrt())
    }
    fn powf(self, e: Self) -> Self {
        Self::from_f64(self.to_f64().powf(e.to_f64()))
    }
}

impl<I: FixedStorage, const FRAC: u32> RealValued for Fixed<I, FRAC> {
    type Mask = bool;
    fn abs(self) -> Self {
        if self.raw() < 0 { -self } else { self }
    }
    fn minimum(self, other: Self) -> Self {
        self.min(other)
    }
    fn maximum(self, other: Self) -> Self {
        self.max(other)
    }
    fn less(self, other: Self) -> bool {
        self < other
    }
    fn greater(self, other: Self) -> bool {
        self > other
    }
    fn select(mask: bool, if_true: Self, if_false: Self) -> Self {
        if mask { if_true } else { if_false }
    }
    fn floor(self) -> Self {
        // clear the fraction bits (arithmetic shift rounds toward negative infinity)
        Fixed(I::fixed_saturate((self.raw() >> FRAC) << FRAC))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::filter::OnePole;

    #[test]
    fn arithmetic_rounds_and_saturates() {
        let a = Q15::from_f64(0.5);
        assert_eq!(a.to_bits(), 16384);
        assert_eq!((a * a).to_f64(), 0.25);
        assert_eq!(Q15::from_f64(2.0), Q15::MAX);
        assert_eq!(Q15::from_f64(-2.0), Q15::MIN);
        assert_eq!(-Q15::MIN, Q15::MAX);
        assert_eq!((Q15::MAX + Q15::epsilon()), Q15::MAX);
        assert_eq!(Q15::MAX.wrapping_add(Q15::epsilon()), Q15::MIN);
        assert_eq!(Q15::MAX.checked_add(Q15::epsilon()), None);
        // rounding to nearest in the product: 3 ulp * 0.5 = 1.5 ulp -> 2 ulp
        assert_eq!((Q15::from_bits(3) * a).to_bits(), 2);
        let q = Q16_16::from_int(7) / Q16_16::from_int(2);
        assert_eq!(q.to_f64(), 3.5);
        assert_eq!(Q16_16::from_int(1) / Q16_16::from_int(0), Q16_16::MAX);
        assert_eq!(format!("{}", Q16_16::from_f64(-3.25)), "-3.25");
        assert_eq!(format!("{:.3}", Q15::from_bits(1)), "0.000");
        assert_eq!(format!("{}", Q15::from_bits(1)), "0.000030517578125");
    }

    #[test]
    fn generic_filters_run_in_fixed_point() {
        // the same OnePole code in Q16.16 and in f64: within a few steps of the format
        let fixed = OnePole::<Q16_16>::lowpass(Q16_16::lit(1_000.0), Q16_16::lit(16_000.0));
        let float = OnePole::<f64>::lowpass(1_000.0, 16_000.0);
        let (mut sf, mut sd) = (Q16_16::lit(0.0), 0.0f64);
        for n in 0..64 {
            let x = if n % 16 < 8 { 1.0 } else { -1.0 };
            let (a, ya) = fixed.tick(sf, Q16_16::lit(x));
            let (b, yb) = float.tick(sd, x);
            sf = a;
            sd = b;
            assert!((ya.to_f64() - yb).abs() < 1e-3, "sample {n}: {} vs {yb}", ya.to_f64());
        }
    }
}
