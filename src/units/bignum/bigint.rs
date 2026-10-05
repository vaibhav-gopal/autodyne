//! Arbitrary-precision signed integers.

use crate::alloc_prelude::*;
use core::cmp::Ordering;
use core::fmt;
use core::ops::{Add, Div, Mul, Neg, Rem, Shl, Shr, Sub};
use core::str::FromStr;

/// An arbitrary-precision integer: a sign and a magnitude in 64-bit limbs (least significant
/// first, no leading zero limbs; zero is an empty magnitude, never negative).
///
/// Multiplication switches to Karatsuba above 32 limbs; division is Knuth's algorithm D.
/// Division and remainder truncate toward zero, as Rust's integer operators do.
///
/// ```
/// use autodyne::units::BigInt;
///
/// let f: BigInt = (1..=30u64).map(BigInt::from).fold(BigInt::from(1u64), |a, b| a * b);
/// assert_eq!(f.to_string(), "265252859812191058636308480000000");
/// assert_eq!("-123456789012345678901234567890".parse::<BigInt>().unwrap() % BigInt::from(97u64), BigInt::from(-52i64));
/// ```
#[derive(Clone, PartialEq, Eq, Hash, Default)]
pub struct BigInt {
    neg: bool,
    mag: Vec<u64>,
}

// MAGNITUDES ======================================================================================

fn trim(v: &mut Vec<u64>) {
    while v.last() == Some(&0) {
        v.pop();
    }
}

fn cmp_mag(a: &[u64], b: &[u64]) -> Ordering {
    a.len().cmp(&b.len()).then_with(|| a.iter().rev().cmp(b.iter().rev()))
}

fn add_mag(a: &[u64], b: &[u64]) -> Vec<u64> {
    let (a, b) = if a.len() >= b.len() { (a, b) } else { (b, a) };
    let mut out = Vec::with_capacity(a.len() + 1);
    let mut carry = 0u64;
    for (i, &x) in a.iter().enumerate() {
        let (s1, c1) = x.overflowing_add(*b.get(i).unwrap_or(&0));
        let (s2, c2) = s1.overflowing_add(carry);
        out.push(s2);
        carry = (c1 as u64) + (c2 as u64);
    }
    if carry > 0 {
        out.push(carry);
    }
    out
}

/// `a - b` for `a >= b`.
fn sub_mag(a: &[u64], b: &[u64]) -> Vec<u64> {
    let mut out = Vec::with_capacity(a.len());
    let mut borrow = 0u64;
    for (i, &x) in a.iter().enumerate() {
        let (d1, b1) = x.overflowing_sub(*b.get(i).unwrap_or(&0));
        let (d2, b2) = d1.overflowing_sub(borrow);
        out.push(d2);
        borrow = (b1 as u64) + (b2 as u64);
    }
    debug_assert_eq!(borrow, 0, "sub_mag needs a >= b");
    trim(&mut out);
    out
}

fn mul_school(a: &[u64], b: &[u64]) -> Vec<u64> {
    if a.is_empty() || b.is_empty() {
        return Vec::new();
    }
    let mut out = vec![0u64; a.len() + b.len()];
    for (i, &x) in a.iter().enumerate() {
        let mut carry = 0u128;
        for (j, &y) in b.iter().enumerate() {
            let t = out[i + j] as u128 + x as u128 * y as u128 + carry;
            out[i + j] = t as u64;
            carry = t >> 64;
        }
        let mut k = i + b.len();
        while carry > 0 {
            let t = out[k] as u128 + carry;
            out[k] = t as u64;
            carry = t >> 64;
            k += 1;
        }
    }
    trim(&mut out);
    out
}

const KARATSUBA: usize = 32;

fn mul_mag(a: &[u64], b: &[u64]) -> Vec<u64> {
    if a.len() < KARATSUBA || b.len() < KARATSUBA {
        return mul_school(a, b);
    }
    // a = a1 B + a0, b = b1 B + b0: ab = z2 B² + (z1 - z2 - z0) B + z0, z1 = (a0 + a1)(b0 + b1)
    let half = a.len().max(b.len()) / 2;
    let split = |v: &[u64]| -> (Vec<u64>, Vec<u64>) {
        let (lo, hi) = v.split_at(half.min(v.len()));
        let mut lo = lo.to_vec();
        trim(&mut lo);
        (lo, hi.to_vec())
    };
    let (a0, a1) = split(a);
    let (b0, b1) = split(b);
    let z0 = mul_mag(&a0, &b0);
    let z2 = mul_mag(&a1, &b1);
    let z1 = mul_mag(&add_mag(&a0, &a1), &add_mag(&b0, &b1));
    let middle = sub_mag(&sub_mag(&z1, &z2), &z0);
    let mut out = z0;
    let shifted = |v: &[u64], limbs: usize| -> Vec<u64> { if v.is_empty() { Vec::new() } else { core::iter::repeat_n(0, limbs).chain(v.iter().copied()).collect() } };
    out = add_mag(&out, &shifted(&middle, half));
    out = add_mag(&out, &shifted(&z2, 2 * half));
    trim(&mut out);
    out
}

fn shl_mag(a: &[u64], bits: usize) -> Vec<u64> {
    if a.is_empty() {
        return Vec::new();
    }
    let (limbs, bits) = (bits / 64, (bits % 64) as u32);
    let mut out = vec![0u64; limbs];
    if bits == 0 {
        out.extend_from_slice(a);
    } else {
        let mut carry = 0u64;
        for &x in a {
            out.push((x << bits) | carry);
            carry = x >> (64 - bits);
        }
        if carry > 0 {
            out.push(carry);
        }
    }
    out
}

fn shr_mag(a: &[u64], bits: usize) -> Vec<u64> {
    let (limbs, bits) = (bits / 64, (bits % 64) as u32);
    if limbs >= a.len() {
        return Vec::new();
    }
    let a = &a[limbs..];
    let mut out: Vec<u64> = if bits == 0 {
        a.to_vec()
    } else {
        (0..a.len()).map(|i| (a[i] >> bits) | a.get(i + 1).map_or(0, |&h| h << (64 - bits))).collect()
    };
    trim(&mut out);
    out
}

/// Quotient and remainder of a magnitude by one limb.
fn divrem_small(a: &[u64], d: u64) -> (Vec<u64>, u64) {
    let mut q = vec![0u64; a.len()];
    let mut r = 0u128;
    for i in (0..a.len()).rev() {
        let cur = (r << 64) | a[i] as u128;
        q[i] = (cur / d as u128) as u64;
        r = cur % d as u128;
    }
    trim(&mut q);
    (q, r as u64)
}

/// Quotient and remainder of magnitudes (Knuth, TAOCP vol. 2, algorithm D).
fn divrem_mag(a: &[u64], b: &[u64]) -> (Vec<u64>, Vec<u64>) {
    assert!(!b.is_empty(), "division by zero");
    if cmp_mag(a, b) == Ordering::Less {
        return (Vec::new(), a.to_vec());
    }
    if b.len() == 1 {
        let (q, r) = divrem_small(a, b[0]);
        return (q, if r == 0 { Vec::new() } else { vec![r] });
    }
    // normalize so the divisor's top limb has its high bit set
    let shift = b[b.len() - 1].leading_zeros() as usize;
    let v = shl_mag(b, shift);
    let mut u = shl_mag(a, shift);
    if u.len() == a.len() {
        u.push(0);
    }
    let (n, m) = (v.len(), u.len() - v.len());
    let mut q = vec![0u64; m];
    let (vtop, vnext) = (v[n - 1] as u128, v[n - 2] as u128);
    for j in (0..m).rev() {
        let num = ((u[j + n] as u128) << 64) | u[j + n - 1] as u128;
        let mut qhat = num / vtop;
        let mut rhat = num % vtop;
        while qhat >= 1 << 64 || qhat * vnext > ((rhat << 64) | u[j + n - 2] as u128) {
            qhat -= 1;
            rhat += vtop;
            if rhat >= 1 << 64 {
                break;
            }
        }
        // u[j..=j+n] -= qhat * v
        let mut borrow = 0i128;
        let mut carry = 0u128;
        for i in 0..n {
            let p = qhat * v[i] as u128 + carry;
            carry = p >> 64;
            let t = u[i + j] as i128 - (p as u64) as i128 + borrow;
            u[i + j] = t as u64;
            borrow = t >> 64;
        }
        let t = u[j + n] as i128 - carry as i128 + borrow;
        u[j + n] = t as u64;
        if t < 0 {
            // qhat was one too large: add v back
            qhat -= 1;
            let mut c = 0u128;
            for i in 0..n {
                let s = u[i + j] as u128 + v[i] as u128 + c;
                u[i + j] = s as u64;
                c = s >> 64;
            }
            u[j + n] = u[j + n].wrapping_add(c as u64);
        }
        q[j] = qhat as u64;
    }
    trim(&mut q);
    u.truncate(n);
    trim(&mut u);
    (q, shr_mag(&u, shift))
}

// BIGINT ==========================================================================================

impl BigInt {
    fn from_parts(neg: bool, mut mag: Vec<u64>) -> Self {
        trim(&mut mag);
        BigInt { neg: neg && !mag.is_empty(), mag }
    }
    /// Zero.
    pub fn zero() -> Self {
        BigInt::default()
    }
    /// One.
    pub fn one() -> Self {
        BigInt::from(1u64)
    }
    /// Whether the value is zero.
    pub fn is_zero(&self) -> bool {
        self.mag.is_empty()
    }
    /// Whether the value is below zero.
    pub fn is_negative(&self) -> bool {
        self.neg
    }
    /// -1, 0 or 1.
    pub fn signum(&self) -> i32 {
        if self.is_zero() { 0 } else if self.neg { -1 } else { 1 }
    }
    /// The magnitude.
    pub fn abs(&self) -> BigInt {
        BigInt { neg: false, mag: self.mag.clone() }
    }
    /// The number of significant bits of the magnitude (0 for zero).
    pub fn bits(&self) -> u64 {
        self.mag.last().map_or(0, |&top| 64 * (self.mag.len() as u64 - 1) + (64 - top.leading_zeros() as u64))
    }
    /// Whether bit `i` of the magnitude is set.
    pub fn bit(&self, i: u64) -> bool {
        let (limb, bit) = ((i / 64) as usize, i % 64);
        self.mag.get(limb).is_some_and(|&l| (l >> bit) & 1 == 1)
    }
    /// Quotient and remainder, truncated toward zero (the remainder has the dividend's sign).
    /// Panics on division by zero.
    pub fn div_rem(&self, d: &BigInt) -> (BigInt, BigInt) {
        let (q, r) = divrem_mag(&self.mag, &d.mag);
        (BigInt::from_parts(self.neg != d.neg, q), BigInt::from_parts(self.neg, r))
    }
    /// `self^e`.
    pub fn pow(&self, mut e: u32) -> BigInt {
        let mut base = self.clone();
        let mut acc = BigInt::one();
        while e > 0 {
            if e & 1 == 1 {
                acc = &acc * &base;
            }
            e >>= 1;
            if e > 0 {
                base = &base * &base;
            }
        }
        acc
    }
    /// The greatest common divisor (non-negative).
    pub fn gcd(&self, other: &BigInt) -> BigInt {
        let (mut a, mut b) = (self.abs(), other.abs());
        while !b.is_zero() {
            let r = &a % &b;
            a = b;
            b = r;
        }
        a
    }
    /// The integer square root `floor(sqrt(self))`; panics for negative values.
    pub fn isqrt(&self) -> BigInt {
        assert!(!self.neg, "square root of a negative number");
        if self.is_zero() {
            return BigInt::zero();
        }
        // Newton from above: x = (x + n / x) / 2 while it decreases
        let mut x = BigInt::one() << self.bits().div_ceil(2) as usize;
        loop {
            let y = (&x + &(self / &x)) >> 1;
            if y >= x {
                return x;
            }
            x = y;
        }
    }
    /// The value as `f64`, rounded to nearest.
    pub fn to_f64(&self) -> f64 {
        let bits = self.bits();
        if bits == 0 {
            return 0.0;
        }
        let v = if bits <= 64 {
            self.mag[0] as f64
        } else {
            // the top 64 bits, with any lower bit set folded in as a sticky bit for correct rounding
            let shift = bits - 64;
            let top = shr_mag(&self.mag, shift as usize)[0];
            let sticky = self.mag.iter().take((shift / 64) as usize).any(|&l| l != 0) || (!shift.is_multiple_of(64) && self.mag[(shift / 64) as usize] << (64 - shift % 64) != 0);
            let top = if sticky { top | 1 } else { top };
            top as f64 * 2f64.powi(shift as i32)
        };
        if self.neg { -v } else { v }
    }
    /// The integer part of `v` (truncated toward zero); `None` for NaN or infinities.
    pub fn from_f64(v: f64) -> Option<BigInt> {
        if !v.is_finite() {
            return None;
        }
        let t = v.trunc().abs();
        if t < 1.0 {
            return Some(BigInt::zero());
        }
        let bits = t.to_bits();
        let exp = ((bits >> 52) & 0x7ff) as i64 - 1075;
        let mant = (bits & ((1 << 52) - 1)) | (1 << 52);
        let m = BigInt::from(mant);
        let mag = if exp >= 0 { m << exp as usize } else { m >> (-exp) as usize };
        Some(if v < 0.0 { -mag } else { mag })
    }
    /// The value as `i64`, if it fits.
    pub fn to_i64(&self) -> Option<i64> {
        match self.mag.len() {
            0 => Some(0),
            1 if !self.neg => i64::try_from(self.mag[0]).ok(),
            1 => {
                let m = self.mag[0];
                if m <= 1 << 63 { Some((m as i64).wrapping_neg()) } else { None }
            }
            _ => None,
        }
    }
    /// Parses digits in `radix` (2 to 36), with an optional sign.
    pub fn from_str_radix(s: &str, radix: u32) -> Result<BigInt, ParseBigIntError> {
        assert!((2..=36).contains(&radix), "radix must be in 2..=36");
        let (neg, digits) = match s.as_bytes().first() {
            Some(b'-') => (true, &s[1..]),
            Some(b'+') => (false, &s[1..]),
            _ => (false, s),
        };
        let digits: Vec<char> = digits.chars().filter(|&c| c != '_').collect();
        if digits.is_empty() {
            return Err(ParseBigIntError);
        }
        let mut mag: Vec<u64> = Vec::new();
        for c in digits {
            let d = c.to_digit(radix).ok_or(ParseBigIntError)? as u64;
            // mag = mag * radix + d
            let mut carry = d as u128;
            for limb in mag.iter_mut() {
                let t = *limb as u128 * radix as u128 + carry;
                *limb = t as u64;
                carry = t >> 64;
            }
            if carry > 0 {
                mag.push(carry as u64);
            }
        }
        Ok(BigInt::from_parts(neg, mag))
    }
    /// The digits in `radix` (2 to 36), lowercase, with a leading `-` when negative.
    pub fn to_str_radix(&self, radix: u32) -> String {
        assert!((2..=36).contains(&radix), "radix must be in 2..=36");
        if self.is_zero() {
            return "0".into();
        }
        // peel off the largest power of the radix that fits a limb at a time
        let (chunk, width) = {
            let (mut p, mut w) = (radix as u64, 1usize);
            while let Some(next) = p.checked_mul(radix as u64) {
                p = next;
                w += 1;
            }
            (p, w)
        };
        let mut parts = Vec::new();
        let mut mag = self.mag.clone();
        while !mag.is_empty() {
            let (q, r) = divrem_small(&mag, chunk);
            parts.push(r);
            mag = q;
        }
        let digit = |d: u64| core::char::from_digit(d as u32, radix).expect("in range");
        let mut s = String::new();
        if self.neg {
            s.push('-');
        }
        for (i, &p) in parts.iter().rev().enumerate() {
            let mut buf = Vec::with_capacity(width);
            let mut v = p;
            while v > 0 {
                buf.push(digit(v % radix as u64));
                v /= radix as u64;
            }
            if i > 0 {
                buf.resize(width, '0');
            }
            s.extend(buf.iter().rev());
        }
        s
    }
}

/// An invalid digit string.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ParseBigIntError;

impl fmt::Display for ParseBigIntError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str("invalid digits for a big integer")
    }
}

impl core::error::Error for ParseBigIntError {}

impl FromStr for BigInt {
    type Err = ParseBigIntError;
    fn from_str(s: &str) -> Result<Self, Self::Err> {
        BigInt::from_str_radix(s, 10)
    }
}

impl fmt::Display for BigInt {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.pad_integral(!self.neg, "", self.abs().to_str_radix(10).as_str())
    }
}

impl fmt::Debug for BigInt {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        fmt::Display::fmt(self, f)
    }
}

impl fmt::LowerHex for BigInt {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.pad_integral(!self.neg, "0x", self.abs().to_str_radix(16).as_str())
    }
}

macro_rules! from_unsigned {
    ($($T:ty),+) => {$(
        impl From<$T> for BigInt {
            fn from(v: $T) -> Self {
                let v = v as u128;
                BigInt::from_parts(false, vec![v as u64, (v >> 64) as u64])
            }
        }
    )+};
}
macro_rules! from_signed {
    ($($T:ty),+) => {$(
        impl From<$T> for BigInt {
            fn from(v: $T) -> Self {
                let m = (v as i128).unsigned_abs();
                BigInt::from_parts(v < 0, vec![m as u64, (m >> 64) as u64])
            }
        }
    )+};
}
from_unsigned!(u8, u16, u32, u64, u128, usize);
from_signed!(i8, i16, i32, i64, i128, isize);

impl Ord for BigInt {
    fn cmp(&self, other: &Self) -> Ordering {
        match (self.neg, other.neg) {
            (false, true) => Ordering::Greater,
            (true, false) => Ordering::Less,
            (false, false) => cmp_mag(&self.mag, &other.mag),
            (true, true) => cmp_mag(&other.mag, &self.mag),
        }
    }
}

impl PartialOrd for BigInt {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}

fn add_signed(a: &BigInt, b: &BigInt) -> BigInt {
    if a.neg == b.neg {
        return BigInt::from_parts(a.neg, add_mag(&a.mag, &b.mag));
    }
    match cmp_mag(&a.mag, &b.mag) {
        Ordering::Less => BigInt::from_parts(b.neg, sub_mag(&b.mag, &a.mag)),
        _ => BigInt::from_parts(a.neg, sub_mag(&a.mag, &b.mag)),
    }
}

/// Each operator for owned and borrowed operands.
macro_rules! binop {
    ($Trait:ident $method:ident |$a:ident, $b:ident| $body:expr) => {
        impl $Trait<&BigInt> for &BigInt {
            type Output = BigInt;
            fn $method(self, rhs: &BigInt) -> BigInt {
                let ($a, $b) = (self, rhs);
                $body
            }
        }
        impl $Trait for BigInt {
            type Output = BigInt;
            fn $method(self, rhs: BigInt) -> BigInt {
                $Trait::$method(&self, &rhs)
            }
        }
        impl $Trait<&BigInt> for BigInt {
            type Output = BigInt;
            fn $method(self, rhs: &BigInt) -> BigInt {
                $Trait::$method(&self, rhs)
            }
        }
        impl $Trait<BigInt> for &BigInt {
            type Output = BigInt;
            fn $method(self, rhs: BigInt) -> BigInt {
                $Trait::$method(self, &rhs)
            }
        }
    };
}

binop!(Add add |a, b| add_signed(a, b));
binop!(Sub sub |a, b| add_signed(a, &-b.clone()));
binop!(Mul mul |a, b| BigInt::from_parts(a.neg != b.neg, mul_mag(&a.mag, &b.mag)));
binop!(Div div |a, b| a.div_rem(b).0);
binop!(Rem rem |a, b| a.div_rem(b).1);

impl Neg for BigInt {
    type Output = BigInt;
    fn neg(self) -> BigInt {
        let neg = !self.neg;
        BigInt::from_parts(neg, self.mag)
    }
}

impl Neg for &BigInt {
    type Output = BigInt;
    fn neg(self) -> BigInt {
        -self.clone()
    }
}

/// Multiplies by `2^bits` (the magnitude; the sign is kept).
impl Shl<usize> for BigInt {
    type Output = BigInt;
    fn shl(self, bits: usize) -> BigInt {
        BigInt::from_parts(self.neg, shl_mag(&self.mag, bits))
    }
}

impl Shl<usize> for &BigInt {
    type Output = BigInt;
    fn shl(self, bits: usize) -> BigInt {
        BigInt::from_parts(self.neg, shl_mag(&self.mag, bits))
    }
}

/// Divides the magnitude by `2^bits`, truncating toward zero.
impl Shr<usize> for BigInt {
    type Output = BigInt;
    fn shr(self, bits: usize) -> BigInt {
        BigInt::from_parts(self.neg, shr_mag(&self.mag, bits))
    }
}

impl Shr<usize> for &BigInt {
    type Output = BigInt;
    fn shr(self, bits: usize) -> BigInt {
        BigInt::from_parts(self.neg, shr_mag(&self.mag, bits))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn b(s: &str) -> BigInt {
        s.parse().unwrap()
    }

    #[test]
    fn arithmetic_matches_i128() {
        let samples: [i128; 9] = [0, 1, -1, 12345, -987654321, i64::MAX as i128, i64::MIN as i128, 1 << 100, -(3 << 90) + 7];
        for &x in &samples {
            for &y in &samples {
                let (bx, by) = (BigInt::from(x), BigInt::from(y));
                if let Some(s) = x.checked_add(y) {
                    assert_eq!(&bx + &by, BigInt::from(s), "{x} + {y}");
                }
                if let Some(s) = x.checked_sub(y) {
                    assert_eq!(&bx - &by, BigInt::from(s), "{x} - {y}");
                }
                if let Some(p) = x.checked_mul(y) {
                    assert_eq!(&bx * &by, BigInt::from(p), "{x} * {y}");
                }
                if y != 0 {
                    assert_eq!(&bx / &by, BigInt::from(x / y), "{x} / {y}");
                    assert_eq!(&bx % &by, BigInt::from(x % y), "{x} % {y}");
                }
                assert_eq!(bx.cmp(&by), x.cmp(&y));
            }
        }
    }

    #[test]
    fn large_values_round_trip() {
        let a = b("123456789012345678901234567890123456789012345678901234567890");
        let c = b("-98765432109876543210987654321");
        let (q, r) = a.div_rem(&c);
        assert_eq!(&q * &c + &r, a);
        assert!(r.abs() < c.abs());
        assert_eq!(a.to_string(), "123456789012345678901234567890123456789012345678901234567890");
        assert_eq!(format!("{:x}", BigInt::from(255u32)), "ff");
        assert_eq!(BigInt::from_str_radix("-ff", 16).unwrap(), BigInt::from(-255));
        // Karatsuba against schoolbook on large operands
        let x = BigInt::from(3u64).pow(4000);
        let y = BigInt::from(7u64).pow(3000);
        assert_eq!(BigInt::from_parts(false, mul_mag(&x.mag, &y.mag)), BigInt::from_parts(false, mul_school(&x.mag, &y.mag)));
        assert_eq!((&x * &y) / &y, x);
        assert_eq!(BigInt::from(10u64).pow(40).isqrt(), BigInt::from(10u64).pow(20));
        assert_eq!(b("99999999999999999999").isqrt(), b("9999999999"));
        assert_eq!(BigInt::from(462u64).gcd(&BigInt::from(1071u64)), BigInt::from(21u64));
        assert_eq!(BigInt::from(1u64) << 200 >> 199, BigInt::from(2u64));
    }

    #[test]
    fn float_conversions() {
        assert_eq!(BigInt::from_f64(1e30).unwrap().to_f64(), 1e30);
        assert_eq!(BigInt::from_f64(-2.75).unwrap(), BigInt::from(-2));
        assert_eq!(BigInt::from(u128::MAX).to_f64(), u128::MAX as f64);
        assert_eq!((BigInt::from(1u64) << 1100).to_f64(), f64::INFINITY);
        assert_eq!(BigInt::from(i64::MIN).to_i64(), Some(i64::MIN));
        assert_eq!((BigInt::from(1u64) << 63).to_i64(), None);
    }
}
