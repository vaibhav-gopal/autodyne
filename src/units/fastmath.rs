//! `f32` exponential, logarithm and hyperbolic tangent as branch-free arithmetic (Cephes' range
//! reductions and polynomials): within 2 ulp of the correctly rounded results everywhere, and with
//! no calls or branches, so loops over them vectorize (std's versions call the platform's libm,
//! one element at a time).

// the coefficients as Cephes publishes them (rounded to f32 by the compiler)
#![allow(clippy::excessive_precision)]

/// The exponential, logarithm and hyperbolic tangent each float type computes with: these for
/// `f32`, std's for `f64`.
pub(crate) trait Transcendental: Sized {
    fn t_exp(self) -> Self;
    fn t_ln(self) -> Self;
    fn t_tanh(self) -> Self;
}

impl Transcendental for f32 {
    #[inline(always)]
    fn t_exp(self) -> f32 {
        exp(self)
    }
    #[inline(always)]
    fn t_ln(self) -> f32 {
        ln(self)
    }
    #[inline(always)]
    fn t_tanh(self) -> f32 {
        tanh(self)
    }
}

impl Transcendental for f64 {
    #[inline(always)]
    fn t_exp(self) -> f64 {
        f64::exp(self)
    }
    #[inline(always)]
    fn t_ln(self) -> f64 {
        f64::ln(self)
    }
    #[inline(always)]
    fn t_tanh(self) -> f64 {
        f64::tanh(self)
    }
}

/// `2^k` for an integer-valued `k` in `[-126, 127]`, from its bits.
#[inline(always)]
fn pow2(k: i32) -> f32 {
    f32::from_bits(((k + 127) as u32) << 23)
}

/// `x` rounded to the nearest integer (ties to even), for `|x| < 2^22`.
#[inline(always)]
fn round(x: f32) -> f32 {
    const SHIFT: f32 = 12_582_912.0; // 1.5 * 2^23
    (x + SHIFT) - SHIFT
}

/// `e^x`.
#[inline(always)]
pub fn exp(x: f32) -> f32 {
    const LOG2E: f32 = std::f32::consts::LOG2_E;
    // ln 2 in two parts: the first exact in few bits, so k * C1 is exact
    const C1: f32 = 0.693_359_375;
    const C2: f32 = -2.121_944_4e-4;
    const MAX: f32 = 88.722_84; // ln(f32::MAX)
    const MIN: f32 = -103.972_08; // ln of the smallest subnormal / 2
    let xc = x.clamp(MIN, MAX);
    let k = round(xc * LOG2E);
    let r = (xc - k * C1) - k * C2;
    let z = r * r;
    let p = (((((1.987_569_15e-4 * r + 1.398_199_950_7e-3) * r + 8.333_451_907_3e-3) * r + 4.166_579_589_4e-2) * r + 1.666_666_545_9e-1) * r + 5.000_000_120_1e-1) * z + r + 1.0;
    // 2^k in two factors, each a normal number, so subnormal results come out right
    let k = k as i32;
    let half = k >> 1;
    let y = p * pow2(half) * pow2(k - half);
    let y = if x > MAX { f32::INFINITY } else { y };
    let y = if x < MIN { 0.0 } else { y };
    if x.is_nan() { x } else { y }
}

/// The natural logarithm.
#[inline(always)]
pub fn ln(x: f32) -> f32 {
    const SQRT_HALF: f32 = std::f32::consts::FRAC_1_SQRT_2;
    // subnormals: scaled into the normal range first
    let subnormal = x < f32::MIN_POSITIVE;
    let xs = if subnormal { x * 8_388_608.0 } else { x };
    let bits = xs.to_bits();
    let e = ((bits >> 23) & 0xff) as i32 - 126 - if subnormal { 23 } else { 0 };
    // x = m 2^e, m in [0.5, 1); then in [sqrt(1/2), sqrt(2)) around 1
    let m = f32::from_bits((bits & 0x007f_ffff) | 0x3f00_0000);
    let low = m < SQRT_HALF;
    let e = (if low { e - 1 } else { e }) as f32;
    let f = if low { m + m - 1.0 } else { m - 1.0 };
    let z = f * f;
    let p = ((((((((7.037_683_629_2e-2 * f - 1.151_461_031_0e-1) * f + 1.167_699_874_0e-1) * f - 1.242_014_084_6e-1) * f + 1.424_932_278_7e-1) * f - 1.666_805_766_5e-1) * f + 2.000_071_476_5e-1) * f
        - 2.499_999_399_3e-1)
        * f
        + 3.333_333_117_4e-1)
        * f
        * z;
    let y = p + -2.121_944_4e-4 * e - 0.5 * z;
    let r = f + y + 0.693_359_375 * e;
    let r = if x == 0.0 { f32::NEG_INFINITY } else { r };
    let r = if x < 0.0 { f32::NAN } else { r };
    let r = if x == f32::INFINITY { x } else { r };
    if x.is_nan() { x } else { r }
}

/// The hyperbolic tangent.
#[inline(always)]
pub fn tanh(x: f32) -> f32 {
    let a = x.abs();
    let z = x * x;
    // near zero, an odd polynomial (no cancellation)
    let near = ((((-5.704_988_727_45e-3 * z + 2.063_908_879_54e-2) * z - 5.373_971_555_31e-2) * z + 1.333_144_220_36e-1) * z - 3.333_328_194_22e-1) * z * x + x;
    // elsewhere 1 - 2 / (e^2|x| + 1), with the sign of x; past 9, 1 (tanh 9 rounds to it)
    let e = exp(2.0 * a.min(9.0));
    let far = (if a > 9.0 { 1.0 } else { 1.0 - 2.0 / (e + 1.0) }).copysign(x);
    let y = if a < 0.625 { near.copysign(x) } else { far };
    if x.is_nan() { x } else { y }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Distance in units in the last place between `got` and the correctly rounded `want`.
    fn ulps(got: f32, want: f64) -> f64 {
        let w = want as f32;
        if got == w || (got.is_nan() && w.is_nan()) {
            return 0.0;
        }
        if got.is_infinite() || w.is_infinite() {
            return f64::INFINITY;
        }
        let ulp = (f32::from_bits(w.abs().to_bits() + 1) - w.abs()) as f64;
        ((got as f64) - want).abs() / ulp.max(f32::from_bits(1) as f64)
    }

    /// Values spread over `[lo, hi]`, densely and log-spaced through small magnitudes.
    fn samples(lo: f32, hi: f32) -> Vec<f32> {
        let mut v: Vec<f32> = (0..200_000).map(|i| lo + (hi - lo) * i as f32 / 199_999.0).collect();
        for e in -40..=38 {
            for k in 1..20 {
                let x = k as f32 * 10f32.powi(e);
                if x >= lo && x <= hi {
                    v.push(x);
                }
                if -x >= lo && -x <= hi {
                    v.push(-x);
                }
            }
        }
        v
    }

    #[test]
    #[ignore]
    fn timing() {
        let x: Vec<f32> = (0..4369).map(|i| 0.01 + i as f32 * 0.37).collect();
        let mut y = vec![0.0f32; x.len()];
        let mut best = f64::MAX;
        for _ in 0..200 {
            let t = std::time::Instant::now();
            for (o, &v) in y.iter_mut().zip(&x) {
                *o = ln(v);
            }
            std::hint::black_box(&y);
            best = best.min(t.elapsed().as_secs_f64() * 1e9 / x.len() as f64);
        }
        let mut best_std = f64::MAX;
        for _ in 0..200 {
            let t = std::time::Instant::now();
            for (o, &v) in y.iter_mut().zip(&x) {
                *o = v.ln();
            }
            std::hint::black_box(&y);
            best_std = best_std.min(t.elapsed().as_secs_f64() * 1e9 / x.len() as f64);
        }
        #[target_feature(enable = "avx2,fma")]
        unsafe fn avx(y: &mut [f32], x: &[f32]) {
            for (o, &v) in y.iter_mut().zip(x) {
                *o = ln(v);
            }
        }
        let mut best_avx = f64::MAX;
        for _ in 0..200 {
            let t = std::time::Instant::now();
            unsafe { avx(&mut y, &x) };
            std::hint::black_box(&y);
            best_avx = best_avx.min(t.elapsed().as_secs_f64() * 1e9 / x.len() as f64);
        }
        eprintln!("TIMING ln {best:.3} ns/elem, avx2 {best_avx:.3}, std {best_std:.3}");
    }

    #[test]
    fn exp_within_two_ulp() {
        let worst = samples(-103.0, 88.7).into_iter().map(|x| ulps(exp(x), (x as f64).exp())).fold(0.0, f64::max);
        assert!(worst <= 2.0, "exp: {worst} ulp");
        assert_eq!(exp(0.0), 1.0);
        assert_eq!(exp(100.0), f32::INFINITY);
        assert_eq!(exp(f32::INFINITY), f32::INFINITY);
        assert_eq!(exp(-200.0), 0.0);
        assert_eq!(exp(f32::NEG_INFINITY), 0.0);
        assert!(exp(f32::NAN).is_nan());
    }

    #[test]
    fn ln_within_two_ulp() {
        let mut xs = samples(0.0, 1e6);
        xs.extend((0..100_000).map(|i| f32::from_bits(1 + i * 83))); // subnormals
        xs.extend([f32::MIN_POSITIVE, f32::MAX, 1.0, 2.0, 0.5, std::f32::consts::E]);
        let worst = xs.into_iter().filter(|&x| x > 0.0).map(|x| ulps(ln(x), (x as f64).ln())).fold(0.0, f64::max);
        assert!(worst <= 2.0, "ln: {worst} ulp");
        assert_eq!(ln(1.0), 0.0);
        assert_eq!(ln(0.0), f32::NEG_INFINITY);
        assert_eq!(ln(f32::INFINITY), f32::INFINITY);
        assert!(ln(-1.0).is_nan() && ln(f32::NAN).is_nan());
    }

    #[test]
    fn tanh_within_two_ulp() {
        let worst = samples(-20.0, 20.0).into_iter().map(|x| ulps(tanh(x), (x as f64).tanh())).fold(0.0, f64::max);
        assert!(worst <= 2.0, "tanh: {worst} ulp");
        assert_eq!(tanh(0.0), 0.0);
        assert_eq!(tanh(f32::INFINITY), 1.0);
        assert_eq!(tanh(f32::NEG_INFINITY), -1.0);
        assert!(tanh(f32::NAN).is_nan());
        assert!(tanh(-0.0).is_sign_negative());
    }
}
