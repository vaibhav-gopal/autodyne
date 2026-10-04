//! Double-precision transcendentals. Shader languages have `exp`, `log`, `sin`, `cos` and `tanh`
//! for 16- and 32-bit floats only (SPIR-V's GLSL.std.450, HLSL, WGSL), so for `f64` they are built
//! here from arithmetic, `round`, `fma` and comparisons, which every f64-capable GPU has: Cody-Waite
//! range reduction (fused, so the compiler cannot fold the split constants back together), Taylor
//! or atanh series to below an ulp, then an exact power of two. Arguments of sin and cos are
//! reduced with a three-part pi/2, accurate for |x| up to about 1e6. Within 4 ulps of the CPU.

// full-precision constants and NaN idioms such as `x != x` are the point here
#![allow(clippy::excessive_precision, clippy::approx_constant, clippy::eq_op, clippy::manual_clamp, clippy::if_same_then_else)]

use cubecl::prelude::*;

use super::kernels::{COS, EXP, LN, SIN, TANH};

/// `2^k`, exact (repeated squaring of 2 or 1/2), for |k| up to about 1100.
#[cube]
fn pow2_f64(k: i32) -> f64 {
    let mut base = 2.0f64;
    let mut n = u32::cast_from(k);
    if k < 0 {
        base = 0.5f64;
        n = u32::cast_from(-k);
    }
    let mut r = 1.0f64;
    while n > 0 {
        if (n & 1) == 1 {
            r *= base;
        }
        base *= base;
        n >>= 1;
    }
    r
}

/// `e^r - 1` for |r| <= 0.35 (Taylor to r^14).
#[cube]
fn expm1_small_f64(r: f64) -> f64 {
    let mut p = 1.1470745597729725e-11f64; // 1/14!
    p = p * r + 1.6059043836821613e-10f64;
    p = p * r + 2.08767569878681e-9f64;
    p = p * r + 2.505210838544172e-8f64;
    p = p * r + 2.755731922398589e-7f64;
    p = p * r + 2.7557319223985893e-6f64;
    p = p * r + 2.48015873015873e-5f64;
    p = p * r + 1.984126984126984e-4f64;
    p = p * r + 1.3888888888888889e-3f64;
    p = p * r + 8.333333333333333e-3f64;
    p = p * r + 4.1666666666666664e-2f64;
    p = p * r + 0.16666666666666666f64;
    p = p * r + 0.5f64;
    p * r * r + r
}

#[cube]
fn exp_f64(x: f64) -> f64 {
    // beyond +-800 e^x is 0 or infinite anyway
    let mut y = x;
    if y > 800.0f64 {
        y = 800.0f64;
    }
    if y < -800.0f64 {
        y = -800.0f64;
    }
    let q = (y * 1.4426950408889634f64).round();
    // fused steps: plain ones may be re-associated into one product with a rounded ln 2
    let r = fma(-q, 1.90821492927058770002e-10f64, fma(-q, 6.93147180369123816490e-1f64, y));
    let mut k = i32::cast_from(q);
    // (NaN casts to anything)
    if k > 1100 {
        k = 1100;
    }
    if k < -1100 {
        k = -1100;
    }
    // split the power so neither factor overflows before the product rounds
    let half = k / 2;
    (1.0f64 + expm1_small_f64(r)) * pow2_f64(half) * pow2_f64(k - half)
}

#[cube]
fn ln_f64(x: f64) -> f64 {
    let mut m = x;
    let mut k = 0i32;
    // bring m within f32's range (2^-100 .. 2^100) by exact steps
    while m > 1.2676506002282294e30f64 && k < 1100 {
        m *= 7.888609052210118e-31f64;
        k += 100;
    }
    while m < 7.888609052210118e-31f64 && m > 0.0f64 && k > -1200 {
        m *= 1.2676506002282294e30f64;
        k -= 100;
    }
    // the exponent from an f32 logarithm (off by one at most: the series takes either side)
    let e = i32::cast_from((f32::cast_from(m).ln() * 1.442695f32).round());
    m *= pow2_f64(-e);
    k += e;
    // ln m = 2 atanh(s), s = (m - 1) / (m + 1), |s| < 0.18
    let s = (m - 1.0f64) / (m + 1.0f64);
    let z = s * s;
    let mut p = 0.043478260869565216f64;
    p = p * z + 0.047619047619047616f64;
    p = p * z + 0.05263157894736842f64;
    p = p * z + 0.058823529411764705f64;
    p = p * z + 0.06666666666666667f64;
    p = p * z + 0.07692307692307693f64;
    p = p * z + 0.09090909090909091f64;
    p = p * z + 0.1111111111111111f64;
    p = p * z + 0.14285714285714285f64;
    p = p * z + 0.2f64;
    p = p * z + 0.3333333333333333f64;
    let kf = f64::cast_from(k);
    let small = fma(kf, 1.90821492927058770002e-10f64, 2.0f64 * s * z * p);
    let mut out = fma(kf, 6.93147180369123816490e-1f64, 2.0f64 * s + small);
    if x == 0.0f64 {
        out = -1.0f64 / x;
    } else if x < 0.0f64 {
        out = (x - x) / (x - x);
    } else if x > 1.7976931348623157e308f64 {
        out = x;
    } else if x != x {
        out = x;
    }
    out
}

#[cube]
fn tanh_f64(x: f64) -> f64 {
    let a = x.abs();
    // tanh a = e / (e + 2) with e = e^(2a) - 1, exact near 0 through expm1
    let y = 2.0f64 * a;
    let mut e = exp_f64(y) - 1.0f64;
    if y < 0.35f64 {
        e = expm1_small_f64(y);
    }
    let mut t = e / (e + 2.0f64);
    if a > 22.0f64 {
        t = 1.0f64;
    }
    if x < 0.0f64 {
        t = -t;
    }
    t
}

/// sin x (`cosine` false) or cos x.
#[cube]
fn sin_cos_f64(x: f64, #[comptime] cosine: bool) -> f64 {
    let q = (x * 0.6366197723675814f64).round();
    let r = fma(-q, 2.02226624879595063154e-21f64, fma(-q, 6.07710050633881403649e-11f64, fma(-q, 1.57079632673412561417e0f64, x)));
    let z = r * r;
    // sin r to r^17, cos r to r^16, on |r| <= pi/4
    let mut s = 2.8114572543455206e-15f64;
    s = s * z - 7.647163731819816e-13f64;
    s = s * z + 1.6059043836821613e-10f64;
    s = s * z - 2.505210838544172e-8f64;
    s = s * z + 2.7557319223985893e-6f64;
    s = s * z - 1.984126984126984e-4f64;
    s = s * z + 8.333333333333333e-3f64;
    s = s * z - 0.16666666666666666f64;
    let sin_r = r + r * z * s;
    let mut c = 4.779477332387385e-14f64;
    c = c * z - 1.1470745597729725e-11f64;
    c = c * z + 2.08767569878681e-9f64;
    c = c * z - 2.755731922398589e-7f64;
    c = c * z + 2.48015873015873e-5f64;
    c = c * z - 1.3888888888888889e-3f64;
    c = c * z + 4.1666666666666664e-2f64;
    c = c * z - 0.5f64;
    let cos_r = 1.0f64 + z * c;
    let mut quadrant = i32::cast_from(q);
    if cosine {
        quadrant += 1;
    }
    quadrant &= 3;
    let mut out = sin_r;
    if quadrant == 1 {
        out = cos_r;
    } else if quadrant == 2 {
        out = -sin_r;
    } else if quadrant == 3 {
        out = -cos_r;
    }
    out
}

#[cube(launch_unchecked)]
pub(super) fn transcendental_f64_kernel(x: &Array<f64>, out: &mut Array<f64>, #[comptime] op: u32) {
    if ABSOLUTE_POS < x.len() {
        let v = x[ABSOLUTE_POS];
        let mut r = v;
        if op == EXP {
            r = exp_f64(v);
        } else if op == LN {
            r = ln_f64(v);
        } else if op == TANH {
            r = tanh_f64(v);
        } else if op == SIN {
            r = sin_cos_f64(v, false);
        } else if op == COS {
            r = sin_cos_f64(v, true);
        }
        out[ABSOLUTE_POS] = r;
    }
}
