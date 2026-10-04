//! Arbitrary-precision binary floating point.

use std::cell::{Cell, RefCell};
use std::cmp::Ordering;
use std::collections::HashMap;
use std::fmt;
use std::ops::{Add, Div, Mul, Neg, Sub};
use std::str::FromStr;

use super::BigInt;
use crate::units::{Elementwise, RealValued};

/// What a [`BigFloat`] holds besides finite values.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Kind {
    Finite,
    Inf,
    Nan,
}

/// A binary floating-point number of a chosen precision: `±mant · 2^exp` with `mant` exactly
/// `prec` bits (or zero), plus ±∞ and NaN.
///
/// Every operation rounds to nearest, ties to even, at the larger precision of its operands;
/// elementary functions (`exp`, `ln`, `sin`, `cos`, `tanh`, `sqrt`, `powf`) evaluate with guard
/// bits and round once. `BigFloat` implements [`Elementwise`] and [`RealValued`], so generic code
/// runs in arbitrary precision; [`Elementwise::lit`] uses the thread's default precision
/// ([`BigFloat::set_default_precision`], 128 bits unless set).
///
/// ```
/// use autodyne::units::{BigFloat, Elementwise};
///
/// let pi = BigFloat::pi(200);
/// assert!(pi.to_string_digits(50).starts_with("3.1415926535897932384626433832795028841971693993751"));
/// let e = BigFloat::from_f64(1.0, 200).exp();
/// assert!(e.to_string_digits(30).starts_with("2.71828182845904523536028747135"));
/// ```
#[derive(Clone)]
pub struct BigFloat {
    kind: Kind,
    neg: bool,
    mant: BigInt,
    exp: i64,
    prec: u32,
}

thread_local! {
    static DEFAULT_PRECISION: Cell<u32> = const { Cell::new(128) };
    static CONSTANTS: RefCell<HashMap<(u8, u32), BigFloat>> = RefCell::new(HashMap::new());
}

const GUARD: u32 = 32;

impl BigFloat {
    /// The precision `lit` and `Default` use on this thread (128 bits unless set).
    pub fn default_precision() -> u32 {
        DEFAULT_PRECISION.with(Cell::get)
    }
    /// Sets the precision `lit` and `Default` use on this thread. Panics below 2 bits.
    pub fn set_default_precision(bits: u32) {
        assert!(bits >= 2, "precision must be at least 2 bits");
        DEFAULT_PRECISION.with(|p| p.set(bits));
    }

    /// Positive zero at `prec` bits.
    pub fn zero(prec: u32) -> Self {
        BigFloat { kind: Kind::Finite, neg: false, mant: BigInt::zero(), exp: 0, prec }
    }
    /// Not a number, at `prec` bits.
    pub fn nan(prec: u32) -> Self {
        BigFloat { kind: Kind::Nan, neg: false, mant: BigInt::zero(), exp: 0, prec }
    }
    /// Infinity of the given sign, at `prec` bits.
    pub fn infinity(negative: bool, prec: u32) -> Self {
        BigFloat { kind: Kind::Inf, neg: negative, mant: BigInt::zero(), exp: 0, prec }
    }

    /// `±mant · 2^exp` rounded to `prec` bits (ties to even); `sticky` says whether nonzero bits
    /// below `mant` were dropped already.
    fn round(neg: bool, mant: BigInt, exp: i64, prec: u32, sticky: bool) -> Self {
        let mant = mant.abs();
        let bits = mant.bits();
        if bits == 0 {
            return BigFloat { neg, ..BigFloat::zero(prec) };
        }
        let p = prec as u64;
        if bits <= p {
            let shift = (p - bits) as usize;
            // a sticky bit below an exactly representable value only matters for exact ties,
            // which cannot happen here (all dropped bits were below the last kept one)
            return BigFloat { kind: Kind::Finite, neg, mant: mant << shift, exp: exp - shift as i64, prec };
        }
        let shift = bits - p;
        let half = mant.bit(shift - 1);
        let below = sticky || (0..shift - 1).any(|i| mant.bit(i));
        let mut m = mant >> shift as usize;
        let mut e = exp + shift as i64;
        if half && (below || m.bit(0)) {
            m = m + BigInt::one();
            if m.bits() > p {
                m = m >> 1;
                e += 1;
            }
        }
        BigFloat { kind: Kind::Finite, neg, mant: m, exp: e, prec }
    }

    /// `v` exactly when it fits `prec` bits, otherwise rounded to nearest (ties to even).
    pub fn from_f64(v: f64, prec: u32) -> Self {
        if v.is_nan() {
            return BigFloat::nan(prec);
        }
        if v.is_infinite() {
            return BigFloat::infinity(v < 0.0, prec);
        }
        if v == 0.0 {
            return BigFloat { neg: v.is_sign_negative(), ..BigFloat::zero(prec) };
        }
        let bits = v.to_bits();
        let raw_exp = ((bits >> 52) & 0x7ff) as i64;
        let frac = bits & ((1 << 52) - 1);
        let (m, e) = if raw_exp == 0 { (frac, -1074) } else { (frac | (1 << 52), raw_exp - 1075) };
        BigFloat::round(v < 0.0, BigInt::from(m), e, prec, false)
    }

    /// The integer `v`, rounded to `prec` bits.
    pub fn from_bigint(v: &BigInt, prec: u32) -> Self {
        BigFloat::round(v.is_negative(), v.clone(), 0, prec, false)
    }

    /// The same value at another precision (rounded).
    pub fn with_precision(&self, prec: u32) -> Self {
        match self.kind {
            Kind::Finite => BigFloat::round(self.neg, self.mant.clone(), self.exp, prec, false),
            _ => BigFloat { prec, ..self.clone() },
        }
    }

    /// The number of significand bits.
    pub fn precision(&self) -> u32 {
        self.prec
    }
    /// Whether the value is not a number.
    pub fn is_nan(&self) -> bool {
        self.kind == Kind::Nan
    }
    /// Whether the value is ±∞.
    pub fn is_infinite(&self) -> bool {
        self.kind == Kind::Inf
    }
    /// Whether the value is neither ±∞ nor NaN.
    pub fn is_finite(&self) -> bool {
        self.kind == Kind::Finite
    }
    /// Whether the value is ±0.
    pub fn is_zero(&self) -> bool {
        self.kind == Kind::Finite && self.mant.is_zero()
    }
    /// Whether the sign is negative (including -0 and -∞; never for NaN).
    pub fn is_negative(&self) -> bool {
        self.neg && !self.is_nan()
    }

    /// The exponent of the leading bit (`floor(log2 |x|)`) of a finite nonzero value.
    fn magnitude(&self) -> i64 {
        self.exp + self.mant.bits() as i64 - 1
    }

    /// The nearest `f64`.
    pub fn to_f64(&self) -> f64 {
        match self.kind {
            Kind::Nan => f64::NAN,
            Kind::Inf => if self.neg { f64::NEG_INFINITY } else { f64::INFINITY },
            Kind::Finite if self.is_zero() => if self.neg { -0.0 } else { 0.0 },
            Kind::Finite => {
                let r = self.with_precision(53);
                let m = r.mant.to_f64();
                // scale in steps so intermediate results stay in range
                let mut v = m;
                let mut e = r.exp;
                while e > 0 {
                    let s = e.min(1000);
                    v *= 2f64.powi(s as i32);
                    e -= s;
                }
                while e < 0 {
                    let s = (-e).min(1000);
                    v /= 2f64.powi(s as i32);
                    e += s;
                }
                if self.neg { -v } else { v }
            }
        }
    }

    /// `self · 2^k`.
    pub fn mul_pow2(&self, k: i64) -> Self {
        let mut r = self.clone();
        if r.is_finite() && !r.is_zero() {
            r.exp += k;
        }
        r
    }

    fn add_impl(&self, other: &Self, prec: u32, negate_other: bool) -> Self {
        let other_neg = other.neg ^ negate_other;
        match (self.kind, other.kind) {
            (Kind::Nan, _) | (_, Kind::Nan) => return BigFloat::nan(prec),
            (Kind::Inf, Kind::Inf) => return if self.neg == other_neg { BigFloat::infinity(self.neg, prec) } else { BigFloat::nan(prec) },
            (Kind::Inf, _) => return BigFloat::infinity(self.neg, prec),
            (_, Kind::Inf) => return BigFloat::infinity(other_neg, prec),
            _ => {}
        }
        if other.is_zero() {
            return if self.is_zero() { BigFloat { neg: self.neg && other_neg, ..BigFloat::zero(prec) } } else { self.with_precision(prec) };
        }
        if self.is_zero() {
            let mut r = other.with_precision(prec);
            r.neg = other_neg;
            return r;
        }
        // align; an operand far below the other only contributes a sticky bit
        let (big, small, big_neg, small_neg) = if self.magnitude() >= other.magnitude() { (self, other, self.neg, other_neg) } else { (other, self, other_neg, self.neg) };
        let gap = big.exp - (small.magnitude() + 1);
        let limit = prec as i64 + 3;
        let (sm, se, sticky) = if gap > limit {
            // stand-in: a value just below the rounding position, with the small operand's sign
            (BigInt::one(), big.exp - limit, true)
        } else {
            (small.mant.clone(), small.exp, false)
        };
        let e = big.exp.min(se);
        let a = &big.mant << (big.exp - e) as usize;
        let b = &sm << (se - e) as usize;
        let a = if big_neg { -a } else { a };
        let b = if small_neg { -b } else { b };
        let s = a + b;
        if s.is_zero() {
            return BigFloat::zero(prec);
        }
        BigFloat::round(s.is_negative(), s, e, prec, sticky)
    }

    fn mul_impl(&self, other: &Self, prec: u32) -> Self {
        let neg = self.neg != other.neg;
        match (self.kind, other.kind) {
            (Kind::Nan, _) | (_, Kind::Nan) => BigFloat::nan(prec),
            (Kind::Inf, _) | (_, Kind::Inf) => if self.is_zero() || other.is_zero() { BigFloat::nan(prec) } else { BigFloat::infinity(neg, prec) },
            _ => BigFloat::round(neg, &self.mant * &other.mant, self.exp + other.exp, prec, false),
        }
    }

    fn div_impl(&self, other: &Self, prec: u32) -> Self {
        let neg = self.neg != other.neg;
        match (self.kind, other.kind) {
            (Kind::Nan, _) | (_, Kind::Nan) | (Kind::Inf, Kind::Inf) => return BigFloat::nan(prec),
            (Kind::Inf, _) => return BigFloat::infinity(neg, prec),
            (_, Kind::Inf) => return BigFloat { neg, ..BigFloat::zero(prec) },
            _ => {}
        }
        if other.is_zero() {
            return if self.is_zero() { BigFloat::nan(prec) } else { BigFloat::infinity(neg, prec) };
        }
        if self.is_zero() {
            return BigFloat { neg, ..BigFloat::zero(prec) };
        }
        // enough quotient bits for prec plus rounding
        let shift = (prec as u64 + 2 + other.mant.bits()).saturating_sub(self.mant.bits()) as usize + 2;
        let (q, r) = (&self.mant << shift).div_rem(&other.mant);
        BigFloat::round(neg, q, self.exp - other.exp - shift as i64, prec, !r.is_zero())
    }

    /// The square root (NaN below zero).
    pub fn sqrt_prec(&self, prec: u32) -> Self {
        match self.kind {
            Kind::Nan => return BigFloat::nan(prec),
            Kind::Inf => return if self.neg { BigFloat::nan(prec) } else { BigFloat::infinity(false, prec) },
            _ => {}
        }
        if self.is_zero() {
            return BigFloat { neg: self.neg, ..BigFloat::zero(prec) };
        }
        if self.neg {
            return BigFloat::nan(prec);
        }
        let (mut m, mut e) = (self.mant.clone(), self.exp);
        if e % 2 != 0 {
            m = m << 1;
            e -= 1;
        }
        // at least 2 (prec + 2) bits under the root
        let want = 2 * (prec as u64 + 2);
        let mut k = want.saturating_sub(m.bits()) as usize;
        if k % 2 == 1 {
            k += 1;
        }
        let scaled = m << k;
        let root = scaled.isqrt();
        let exact = &root * &root == scaled;
        BigFloat::round(false, root, (e - k as i64) / 2, prec, !exact)
    }

    // CONSTANTS ===================================================================================

    fn cached(id: u8, prec: u32, make: impl FnOnce() -> BigFloat) -> BigFloat {
        if let Some(v) = CONSTANTS.with(|c| c.borrow().get(&(id, prec)).cloned()) {
            return v;
        }
        let v = make();
        CONSTANTS.with(|c| c.borrow_mut().insert((id, prec), v.clone()));
        v
    }

    /// `Σ ±1 / ((2k+1) n^(2k+1))` scaled by `2^w`: atanh(1/n) (`alternate` false) or atan(1/n).
    fn arc_recip(n: u64, w: usize, alternate: bool) -> BigInt {
        let n = BigInt::from(n);
        let n2 = &n * &n;
        let mut term = (BigInt::one() << w) / &n;
        let mut sum = term.clone();
        let mut k = 1u64;
        while !term.is_zero() {
            term = &term / &n2;
            let t = &term / &BigInt::from(2 * k + 1);
            sum = if alternate && k % 2 == 1 { sum - t } else { sum + t };
            k += 1;
        }
        sum
    }

    /// π to `prec` bits (Machin: π = 16 atan(1/5) - 4 atan(1/239)).
    pub fn pi(prec: u32) -> BigFloat {
        BigFloat::cached(0, prec, || {
            let w = (prec + GUARD) as usize;
            let v = (BigFloat::arc_recip(5, w, true) << 4) - (BigFloat::arc_recip(239, w, true) << 2);
            BigFloat::round(false, v, -(w as i64), prec, true)
        })
    }

    /// ln 2 to `prec` bits (2 atanh(1/3)).
    pub fn ln2(prec: u32) -> BigFloat {
        BigFloat::cached(1, prec, || {
            let w = (prec + GUARD) as usize;
            BigFloat::round(false, BigFloat::arc_recip(3, w, false) << 1, -(w as i64), prec, true)
        })
    }

    // FUNCTIONS ===================================================================================

    /// `e^self` at `prec` bits.
    fn exp_prec(&self, prec: u32) -> BigFloat {
        match self.kind {
            Kind::Nan => return BigFloat::nan(prec),
            Kind::Inf => return if self.neg { BigFloat::zero(prec) } else { BigFloat::infinity(false, prec) },
            _ => {}
        }
        if self.is_zero() {
            return BigFloat::from_f64(1.0, prec);
        }
        if self.magnitude() > 62 {
            return if self.neg { BigFloat::zero(prec) } else { BigFloat::infinity(false, prec) };
        }
        // x = k ln2 + r with |r| <= ln2 / 2
        let w = prec + GUARD + self.magnitude().max(0) as u32;
        let ln2 = BigFloat::ln2(w);
        let k = self.div_impl(&ln2, w).to_f64().round();
        let r = self.add_impl(&ln2.mul_impl(&BigFloat::from_f64(k, w), w), w, true);
        // halve r s times so the series converges fast, then square back
        let s = ((prec as f64).sqrt() / 2.0) as i64 + 1;
        let r = r.mul_pow2(-s);
        let wf = w + s as u32;
        let one = BigFloat::from_f64(1.0, wf);
        let (mut sum, mut term) = (one.clone(), one);
        let mut n = 1u64;
        loop {
            term = term.mul_impl(&r, wf).div_impl(&BigFloat::from_f64(n as f64, wf), wf);
            if term.is_zero() || term.magnitude() < sum.magnitude() - wf as i64 - 2 {
                break;
            }
            sum = sum.add_impl(&term, wf, false);
            n += 1;
        }
        for _ in 0..s {
            sum = sum.mul_impl(&sum, wf);
        }
        sum.mul_pow2(k as i64).with_precision(prec)
    }

    /// The natural logarithm at `prec` bits (NaN below zero, -∞ at zero).
    fn ln_prec(&self, prec: u32) -> BigFloat {
        match self.kind {
            Kind::Nan => return BigFloat::nan(prec),
            Kind::Inf => return if self.neg { BigFloat::nan(prec) } else { BigFloat::infinity(false, prec) },
            _ => {}
        }
        if self.is_zero() {
            return BigFloat::infinity(true, prec);
        }
        if self.neg {
            return BigFloat::nan(prec);
        }
        // x = m 2^e with m in [1/√2, √2): ln x = 2 atanh((m - 1)/(m + 1)) + e ln2
        let mut e = self.magnitude();
        let w = prec + GUARD + 64;
        let wide = self.with_precision(w);
        let mut m = BigFloat { exp: -(wide.mant.bits() as i64 - 1), ..wide };
        if m.to_f64() > std::f64::consts::SQRT_2 {
            m = m.mul_pow2(-1);
            e += 1;
        }
        let one = BigFloat::from_f64(1.0, w);
        let z = m.add_impl(&one, w, true).div_impl(&m.add_impl(&one, w, false), w);
        let z2 = z.mul_impl(&z, w);
        let (mut sum, mut power) = (z.clone(), z.clone());
        let mut k = 1u64;
        while !power.is_zero() {
            power = power.mul_impl(&z2, w);
            let t = power.div_impl(&BigFloat::from_f64((2 * k + 1) as f64, w), w);
            if t.is_zero() || (!sum.is_zero() && t.magnitude() < sum.magnitude() - w as i64 - 2) {
                break;
            }
            sum = sum.add_impl(&t, w, false);
            k += 1;
        }
        let ln_m = sum.mul_pow2(1);
        let tail = BigFloat::ln2(w).mul_impl(&BigFloat::from_f64(e as f64, w), w);
        ln_m.add_impl(&tail, w, false).with_precision(prec)
    }

    /// `(sin r, cos r)` by Taylor series, for |r| <= π/4, at `w` bits.
    fn sin_cos_small(r: &BigFloat, w: u32) -> (BigFloat, BigFloat) {
        let r2 = r.mul_impl(r, w);
        let series = |start: BigFloat, first: u64| {
            let (mut sum, mut term) = (start.clone(), start);
            let mut n = first;
            loop {
                term = term.mul_impl(&r2, w).div_impl(&BigFloat::from_f64(((n + 1) * (n + 2)) as f64, w), w);
                if term.is_zero() || (!sum.is_zero() && term.magnitude() < sum.magnitude() - w as i64 - 2) {
                    break;
                }
                term = -term;
                sum = sum.add_impl(&term, w, false);
                n += 2;
            }
            sum
        };
        (series(r.with_precision(w), 1), series(BigFloat::from_f64(1.0, w), 0))
    }

    /// `(sin x, cos x)` at `prec` bits: reduced by multiples of π/2.
    fn sin_cos_prec(&self, prec: u32) -> (BigFloat, BigFloat) {
        if !self.is_finite() {
            return (BigFloat::nan(prec), BigFloat::nan(prec));
        }
        if self.is_zero() {
            return (self.with_precision(prec), BigFloat::from_f64(1.0, prec));
        }
        // the reduction cancels about magnitude(x) bits: carry them
        let w = prec + GUARD + self.magnitude().max(0) as u32;
        let half_pi = BigFloat::pi(w).mul_pow2(-1);
        let kf = self.div_impl(&half_pi, w);
        let k = BigFloat::round_to_integer(&kf);
        let r = self.add_impl(&half_pi.mul_impl(&BigFloat::from_bigint(&k, w), w), w, true);
        let (s, c) = BigFloat::sin_cos_small(&r, w);
        let quadrant = (&k % &BigInt::from(4)).to_i64().unwrap_or(0).rem_euclid(4);
        let (s, c) = match quadrant {
            0 => (s, c),
            1 => (c, -s),
            2 => (-s, -c),
            _ => (-c, s),
        };
        (s.with_precision(prec), c.with_precision(prec))
    }

    /// The nearest integer (ties away from zero).
    fn round_to_integer(x: &BigFloat) -> BigInt {
        if x.is_zero() || !x.is_finite() {
            return BigInt::zero();
        }
        let half = BigFloat::from_f64(0.5, x.prec + 2);
        let shifted = if x.neg { x.add_impl(&half, x.prec + 2, true) } else { x.add_impl(&half, x.prec + 2, false) };
        let mag = if shifted.exp >= 0 { &shifted.mant << shifted.exp as usize } else { &shifted.mant >> (-shifted.exp) as usize };
        if shifted.neg { -mag } else { mag }
    }

    fn tanh_prec(&self, prec: u32) -> BigFloat {
        match self.kind {
            Kind::Nan => return BigFloat::nan(prec),
            Kind::Inf => return BigFloat::from_f64(if self.neg { -1.0 } else { 1.0 }, prec),
            _ => {}
        }
        if self.is_zero() {
            return self.with_precision(prec);
        }
        // beyond this, tanh rounds to ±1
        if self.magnitude() > 32 || self.to_f64().abs() > prec as f64 * 0.35 + 2.0 {
            return BigFloat::from_f64(if self.neg { -1.0 } else { 1.0 }, prec);
        }
        // tanh x = (e^2x - 1) / (e^2x + 1), with bits for the cancellation near 0
        let w = prec + GUARD + (-self.magnitude()).max(0) as u32;
        let e2 = self.mul_pow2(1).exp_prec(w);
        let one = BigFloat::from_f64(1.0, w);
        e2.add_impl(&one, w, true).div_impl(&e2.add_impl(&one, w, false), w).with_precision(prec)
    }

    /// Whether the value is an integer, and its parity.
    fn integer_parity(&self) -> Option<bool> {
        if !self.is_finite() {
            return None;
        }
        if self.is_zero() {
            return Some(false);
        }
        if self.exp >= 0 {
            return Some(self.exp == 0 && self.mant.bit(0));
        }
        let drop = (-self.exp) as u64;
        if (0..drop).any(|i| self.mant.bit(i)) {
            return None;
        }
        Some(self.mant.bit(drop))
    }

    fn powf_prec(&self, e: &BigFloat, prec: u32) -> BigFloat {
        if self.is_nan() || e.is_nan() {
            return BigFloat::nan(prec);
        }
        if e.is_zero() {
            return BigFloat::from_f64(1.0, prec);
        }
        if self.is_zero() {
            return if e.neg { BigFloat::infinity(false, prec) } else { BigFloat::zero(prec) };
        }
        let odd = match (self.neg, e.integer_parity()) {
            (true, None) => return BigFloat::nan(prec),
            (true, Some(odd)) => odd,
            _ => false,
        };
        // e ln|x| with bits for the size of the exponent
        let w = prec + GUARD + e.magnitude().max(0) as u32 + 64;
        let r = self.abs().ln_prec(w).mul_impl(e, w).exp_prec(w).with_precision(prec);
        if odd { -r } else { r }
    }

    /// `|self|`.
    pub fn abs(&self) -> BigFloat {
        BigFloat { neg: false, ..self.clone() }
    }

    /// The decimal value to `digits` significant digits (round half to even), like `1.234e-5` for
    /// very large or small magnitudes.
    pub fn to_string_digits(&self, digits: usize) -> String {
        match self.kind {
            Kind::Nan => return "NaN".into(),
            Kind::Inf => return if self.neg { "-inf".into() } else { "inf".into() },
            _ => {}
        }
        let sign = if self.neg { "-" } else { "" };
        if self.is_zero() {
            return format!("{sign}0");
        }
        let digits = digits.max(1);
        // estimate the decimal exponent, then scale to `digits` integer digits
        let mut d = (self.magnitude() as f64 * std::f64::consts::LOG10_2).floor() as i64;
        let scaled = loop {
            let k = digits as i64 - 1 - d;
            let mut num = self.mant.clone();
            let mut den = BigInt::one();
            if k >= 0 { num = num * BigInt::from(10u64).pow(k as u32) } else { den = den * BigInt::from(10u64).pow((-k) as u32) }
            if self.exp >= 0 { num = num << self.exp as usize } else { den = den << (-self.exp) as usize }
            let (q, r) = num.div_rem(&den);
            // round half to even
            let twice = r << 1;
            let q = match twice.cmp(&den) {
                Ordering::Greater => q + BigInt::one(),
                Ordering::Equal if q.bit(0) => q + BigInt::one(),
                _ => q,
            };
            let text = q.to_string();
            if text.len() > digits {
                d += 1;
            } else if text.len() < digits {
                d -= 1;
            } else {
                break text;
            }
        };
        let (head, tail) = scaled.split_at(1);
        let tail = tail.trim_end_matches('0');
        if (-6..21).contains(&d) {
            // positional
            let all = format!("{head}{}", &scaled[1..]);
            let s = if d >= 0 {
                let point = (d + 1) as usize;
                if point >= all.len() {
                    format!("{all}{}", "0".repeat(point - all.len()))
                } else {
                    let (i, f) = all.split_at(point);
                    let f = f.trim_end_matches('0');
                    if f.is_empty() { i.to_string() } else { format!("{i}.{f}") }
                }
            } else {
                let f = format!("{}{}", "0".repeat((-d - 1) as usize), all.trim_end_matches('0'));
                format!("0.{f}")
            };
            format!("{sign}{s}")
        } else if tail.is_empty() {
            format!("{sign}{head}e{d}")
        } else {
            format!("{sign}{head}.{tail}e{d}")
        }
    }
}

impl fmt::Display for BigFloat {
    /// Enough significant digits to identify the value (or `{:.N}` for N significant digits).
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let digits = f.precision().unwrap_or((self.prec as f64 * std::f64::consts::LOG10_2).ceil() as usize + 1);
        f.write_str(&self.to_string_digits(digits))
    }
}

impl fmt::Debug for BigFloat {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{self} ({} bits)", self.prec)
    }
}

/// An invalid decimal number.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ParseBigFloatError;

impl fmt::Display for ParseBigFloatError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str("invalid decimal number")
    }
}

impl std::error::Error for ParseBigFloatError {}

impl BigFloat {
    /// Parses a decimal number (`-12.5e-3`, `inf`, `nan`) to `prec` bits, correctly rounded.
    pub fn parse(s: &str, prec: u32) -> Result<BigFloat, ParseBigFloatError> {
        let t = s.trim();
        let lower = t.to_ascii_lowercase();
        let (neg, body) = match lower.as_bytes().first() {
            Some(b'-') => (true, &lower[1..]),
            Some(b'+') => (false, &lower[1..]),
            _ => (false, &lower[..]),
        };
        match body {
            "inf" | "infinity" => return Ok(BigFloat::infinity(neg, prec)),
            "nan" => return Ok(BigFloat::nan(prec)),
            _ => {}
        }
        let (mantissa, exp10) = match body.split_once('e') {
            Some((m, e)) => (m, e.parse::<i64>().map_err(|_| ParseBigFloatError)?),
            None => (body, 0),
        };
        let (int, frac) = mantissa.split_once('.').unwrap_or((mantissa, ""));
        if int.is_empty() && frac.is_empty() {
            return Err(ParseBigFloatError);
        }
        let digits = format!("{int}{frac}");
        if !digits.bytes().all(|b| b.is_ascii_digit()) {
            return Err(ParseBigFloatError);
        }
        let n: BigInt = digits.parse().map_err(|_| ParseBigFloatError)?;
        let e = exp10 - frac.len() as i64;
        let v = if e >= 0 {
            BigFloat::round(neg, n * BigInt::from(10u64).pow(e as u32), 0, prec, false)
        } else {
            // exact quotient, correctly rounded
            let den = BigInt::from(10u64).pow((-e) as u32);
            let shift = (prec as u64 + 2 + den.bits()).saturating_sub(n.bits()) as usize + 2;
            let (q, r) = (n << shift).div_rem(&den);
            BigFloat::round(neg, q, -(shift as i64), prec, !r.is_zero())
        };
        Ok(v)
    }
}

impl FromStr for BigFloat {
    type Err = ParseBigFloatError;
    /// Parses at the default precision.
    fn from_str(s: &str) -> Result<Self, Self::Err> {
        BigFloat::parse(s, BigFloat::default_precision())
    }
}

impl Default for BigFloat {
    fn default() -> Self {
        BigFloat::zero(BigFloat::default_precision())
    }
}

impl PartialEq for BigFloat {
    fn eq(&self, other: &Self) -> bool {
        self.partial_cmp(other) == Some(Ordering::Equal)
    }
}

impl PartialOrd for BigFloat {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        if self.is_nan() || other.is_nan() {
            return None;
        }
        let sign = |x: &BigFloat| if x.is_zero() { 0 } else if x.neg { -1 } else { 1 };
        let (sa, sb) = (sign(self), sign(other));
        if sa != sb {
            return Some(sa.cmp(&sb));
        }
        if sa == 0 {
            return Some(Ordering::Equal);
        }
        let mag = match (self.kind, other.kind) {
            (Kind::Inf, Kind::Inf) => Ordering::Equal,
            (Kind::Inf, _) => Ordering::Greater,
            (_, Kind::Inf) => Ordering::Less,
            _ => self.magnitude().cmp(&other.magnitude()).then_with(|| {
                let e = self.exp.min(other.exp);
                (&self.mant << (self.exp - e) as usize).cmp(&(&other.mant << (other.exp - e) as usize))
            }),
        };
        Some(if sa > 0 { mag } else { mag.reverse() })
    }
}

macro_rules! op {
    ($Trait:ident $method:ident |$a:ident, $b:ident, $p:ident| $body:expr) => {
        impl $Trait<&BigFloat> for &BigFloat {
            type Output = BigFloat;
            fn $method(self, rhs: &BigFloat) -> BigFloat {
                let ($a, $b, $p) = (self, rhs, self.prec.max(rhs.prec));
                $body
            }
        }
        impl $Trait for BigFloat {
            type Output = BigFloat;
            fn $method(self, rhs: BigFloat) -> BigFloat {
                $Trait::$method(&self, &rhs)
            }
        }
    };
}

op!(Add add |a, b, p| a.add_impl(b, p, false));
op!(Sub sub |a, b, p| a.add_impl(b, p, true));
op!(Mul mul |a, b, p| a.mul_impl(b, p));
op!(Div div |a, b, p| a.div_impl(b, p));

impl Neg for BigFloat {
    type Output = BigFloat;
    fn neg(mut self) -> BigFloat {
        if !self.is_nan() {
            self.neg = !self.neg;
        }
        self
    }
}

impl Neg for &BigFloat {
    type Output = BigFloat;
    fn neg(self) -> BigFloat {
        -self.clone()
    }
}

/// Elementary functions correctly rounded in practice (guard bits, one final rounding).
impl Elementwise for BigFloat {
    fn lit(v: f64) -> Self {
        BigFloat::from_f64(v, BigFloat::default_precision())
    }
    fn exp(self) -> Self {
        self.exp_prec(self.prec)
    }
    fn ln(self) -> Self {
        self.ln_prec(self.prec)
    }
    fn sin(self) -> Self {
        self.sin_cos_prec(self.prec).0
    }
    fn cos(self) -> Self {
        self.sin_cos_prec(self.prec).1
    }
    fn tanh(self) -> Self {
        self.tanh_prec(self.prec)
    }
    fn sqrt(self) -> Self {
        self.sqrt_prec(self.prec)
    }
    fn powf(self, e: Self) -> Self {
        let p = self.prec.max(e.prec);
        self.powf_prec(&e, p)
    }
}

impl RealValued for BigFloat {
    type Mask = bool;
    fn abs(self) -> Self {
        BigFloat::abs(&self)
    }
    fn minimum(self, other: Self) -> Self {
        if self.is_nan() || other.is_nan() { BigFloat::nan(self.prec.max(other.prec)) } else if other < self { other } else { self }
    }
    fn maximum(self, other: Self) -> Self {
        if self.is_nan() || other.is_nan() { BigFloat::nan(self.prec.max(other.prec)) } else if other > self { other } else { self }
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
        if !self.is_finite() || self.is_zero() || self.exp >= 0 {
            return self;
        }
        // drop the fraction bits; negative values with a fraction go one further down
        let drop = (-self.exp) as usize;
        let int = &self.mant >> drop;
        let had_fraction = (0..drop as u64).any(|i| self.mant.bit(i));
        let int = if self.neg && had_fraction { int + BigInt::one() } else { int };
        BigFloat::round(self.neg, int, 0, self.prec, false)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn f(v: f64) -> BigFloat {
        BigFloat::from_f64(v, 53)
    }

    #[test]
    fn double_precision_agrees_with_f64() {
        // at 53 bits, correctly rounded basic operations equal IEEE doubles
        let xs = [0.1, -2.5, 3.0, 1e-300, 7.0e200, std::f64::consts::PI, 1.0 / 3.0];
        for &a in &xs {
            for &b in &xs {
                assert_eq!((f(a) + f(b)).to_f64(), a + b, "{a} + {b}");
                assert_eq!((f(a) - f(b)).to_f64(), a - b, "{a} - {b}");
                assert_eq!((f(a) * f(b)).to_f64(), a * b, "{a} * {b}");
                assert_eq!((f(a) / f(b)).to_f64(), a / b, "{a} / {b}");
            }
            if a >= 0.0 {
                assert_eq!(f(a).sqrt().to_f64(), a.sqrt(), "sqrt {a}");
            }
        }
        // transcendental functions within an ulp of libm
        for &x in &[0.5, -1.25, 3.0, 10.0, 1e-5] {
            let close = |got: f64, want: f64| (got - want).abs() <= 2.0 * f64::EPSILON * want.abs();
            assert!(close(f(x).exp().to_f64(), x.exp()), "exp {x}");
            assert!(close(f(x).sin().to_f64(), x.sin()), "sin {x}");
            assert!(close(f(x).cos().to_f64(), x.cos()), "cos {x}");
            assert!(close(f(x).tanh().to_f64(), x.tanh()), "tanh {x}");
            if x > 0.0 {
                assert!(close(f(x).ln().to_f64(), x.ln()), "ln {x}");
                assert!(close(f(x).powf(f(1.7)).to_f64(), x.powf(1.7)), "pow {x}");
            }
        }
        assert!(f(-1.0).ln().is_nan());
        assert!(f(-8.0).powf(f(1.0 / 3.0)).is_nan());
        assert_eq!(f(-2.0).powf(f(3.0)).to_f64(), -8.0);
    }

    #[test]
    fn high_precision_constants_and_identities() {
        let p = 300;
        let pi = BigFloat::pi(p);
        assert!(pi.to_string_digits(80).starts_with("3.14159265358979323846264338327950288419716939937510582097494459230781640628620"));
        let ln2 = BigFloat::ln2(p);
        assert!(ln2.to_string_digits(60).starts_with("0.69314718055994530941723212145817656807550013436025525412068"));
        // exp(ln 2) = 2, sin² + cos² = 1, sqrt(2)² = 2, at 300 bits
        let two = BigFloat::from_f64(2.0, p);
        let tol = BigFloat::from_f64(1.0, p).mul_pow2(-(p as i64) + 4);
        let near = |a: &BigFloat, b: &BigFloat| (a - b).abs() < tol;
        assert!(near(&ln2.clone().exp(), &two));
        let x = BigFloat::parse("1.2345678901234567890123456789", p).unwrap();
        let (s, c) = x.sin_cos_prec(p);
        assert!(near(&(&(&s * &s) + &(&c * &c)), &BigFloat::from_f64(1.0, p)));
        let r = two.clone().sqrt();
        assert!(near(&(&r * &r), &two));
        // sin(π) is tiny, cos(π) = -1
        let (s, c) = pi.sin_cos_prec(p);
        assert!(s.abs() < tol && near(&c, &BigFloat::from_f64(-1.0, p)));
        // large arguments reduce correctly: sin(1e10) = -0.4875060250875106...
        assert!(BigFloat::from_f64(1e10, 200).sin().to_string_digits(15).starts_with("-0.487506025087511"));
    }

    #[test]
    fn parsing_formatting_and_generic_code() {
        let x = BigFloat::parse("-12.375e2", 64).unwrap();
        assert_eq!(x.to_f64(), -1237.5);
        assert_eq!(x.to_string_digits(6), "-1237.5");
        assert_eq!(BigFloat::parse("1e-30", 64).unwrap().to_string_digits(3), "1e-30");
        assert_eq!(BigFloat::from_f64(0.000125, 53).to_string_digits(3), "0.000125");
        assert!("abc".parse::<BigFloat>().is_err());
        // generic Real-style code at 256 bits
        BigFloat::set_default_precision(256);
        fn softplus<T: Elementwise>(x: T) -> T {
            (x.exp() + T::lit(1.0)).ln()
        }
        let y = softplus(BigFloat::from_f64(0.0, 256));
        assert!(y.to_string_digits(70).starts_with("0.69314718055994530941723212145817656807550013436025"));
        BigFloat::set_default_precision(128);
        assert!(f(1.0) < f(2.0) && f(-1.0) < f(0.0) && f(f64::NAN).partial_cmp(&f(1.0)).is_none());
    }
}
