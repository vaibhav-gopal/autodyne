//! Special functions (`scipy.special` names): the modified Bessel function I₀ (Kaiser windows and
//! the sampler's interpolation kernel), complete elliptic integrals, and Jacobi elliptic functions
//! and their inverses (elliptic filter design).
//!
//! tend: Numerics / special

use crate::units::Complex;

type C64 = Complex<f64>;

/// The modified Bessel function of the first kind, order 0 (power series; exact to rounding for the
/// arguments Kaiser windows use).
pub fn bessel_i0(x: f64) -> f64 {
    let y = x * x / 4.0;
    let mut term = 1.0;
    let mut sum = 1.0;
    let mut k = 1.0;
    while term > sum * 1e-17 {
        term *= y / (k * k);
        sum += term;
        k += 1.0;
    }
    sum
}

/// The arithmetic-geometric mean of `a` and `b`.
fn agm(mut a: f64, mut b: f64) -> f64 {
    for _ in 0..64 {
        if (a - b).abs() <= f64::EPSILON * a.abs() {
            break;
        }
        (a, b) = ((a + b) / 2.0, (a * b).sqrt());
    }
    a
}

/// The complete elliptic integral of the first kind `K(m)` (parameter `m = k²`, `scipy.special.ellipk`).
pub fn ellipk(m: f64) -> f64 {
    if m >= 1.0 {
        return if m == 1.0 { f64::INFINITY } else { f64::NAN };
    }
    std::f64::consts::FRAC_PI_2 / agm(1.0, (1.0 - m).sqrt())
}

/// `K(1 - p)`, accurate for small `p` (`scipy.special.ellipkm1`).
pub fn ellipkm1(p: f64) -> f64 {
    if p <= 0.0 {
        return if p == 0.0 { f64::INFINITY } else { f64::NAN };
    }
    std::f64::consts::FRAC_PI_2 / agm(1.0, p.sqrt())
}

/// Jacobi elliptic functions `(sn, cn, dn, φ)` of `u` with parameter `m` (`scipy.special.ellipj`,
/// Cephes' AGM method with its series near `m = 0` and `m = 1`).
pub fn ellipj(u: f64, m: f64) -> (f64, f64, f64, f64) {
    if !(0.0..=1.0).contains(&m) {
        return (f64::NAN, f64::NAN, f64::NAN, f64::NAN);
    }
    if m < 1e-9 {
        let (t, b) = (u.sin(), u.cos());
        let ai = 0.25 * m * (u - t * b);
        return (t - ai * b, b + ai * t, 1.0 - 0.5 * m * t * t, u - ai);
    }
    if m >= 0.9999999999 {
        let mut ai = 0.25 * (1.0 - m);
        let b = u.cosh();
        let t = u.tanh();
        let phi = 1.0 / b;
        let twon = b * u.sinh();
        let sn = t + ai * (twon - u) / (b * b);
        let ph = 2.0 * u.exp().atan() - std::f64::consts::FRAC_PI_2 + ai * (twon - u) / b;
        ai *= t * phi;
        return (sn, phi - ai * (twon - u), phi + ai * (twon + u), ph);
    }
    // descending AGM sequence, then back down for the amplitude φ
    let mut a = [0.0f64; 9];
    let mut c = [0.0f64; 9];
    a[0] = 1.0;
    let mut b = (1.0 - m).sqrt();
    c[0] = m.sqrt();
    let mut twon = 1.0;
    let mut i = 0;
    while (c[i] / a[i]).abs() > f64::EPSILON {
        if i > 7 {
            break;
        }
        let ai = a[i];
        i += 1;
        c[i] = (ai - b) / 2.0;
        let t = (ai * b).sqrt();
        a[i] = (ai + b) / 2.0;
        b = t;
        twon *= 2.0;
    }
    let mut phi = twon * a[i] * u;
    let mut prev = phi;
    while i > 0 {
        let t = c[i] * phi.sin() / a[i];
        prev = phi;
        phi = (t.asin() + phi) / 2.0;
        i -= 1;
    }
    let (sn, cn) = (phi.sin(), phi.cos());
    (sn, cn, cn / (prev - phi).cos(), phi)
}

/// `sqrt((1 - x)(1 + x))`, accurate for small `x`.
fn complement(x: C64) -> C64 {
    ((C64::one() - x) * (C64::one() + x)).sqrt()
}

/// The inverse Jacobi `sn` for a complex argument, by descending Landen transformations
/// (`scipy.signal._filter_design._arc_jac_sn`).
pub fn arc_jac_sn(w: C64, m: f64) -> C64 {
    let k = m.sqrt();
    if k > 1.0 {
        return C64::new(f64::NAN, f64::NAN);
    }
    if k == 1.0 {
        // atanh(w) = ln((1 + w) / (1 - w)) / 2
        return ((C64::one() + w) / (C64::one() - w)).ln() * 0.5;
    }
    let mut ks = vec![k];
    while *ks.last().expect("non-empty") != 0.0 {
        let kn = *ks.last().expect("non-empty");
        let kp = complement(C64::new(kn, 0.0)).re;
        ks.push((1.0 - kp) / (1.0 + kp));
        if ks.len() > 11 {
            break;
        }
    }
    let big_k: f64 = ks[1..].iter().map(|k| 1.0 + k).product::<f64>() * std::f64::consts::FRAC_PI_2;
    let mut wn = w;
    for pair in ks.windows(2) {
        let (kn, knext) = (pair[0], pair[1]);
        wn = wn * 2.0 / ((C64::one() + complement(wn * kn)) * (1.0 + knext));
    }
    wn.asin() * (2.0 / std::f64::consts::PI) * big_k
}

/// The real inverse Jacobi `sc` with complementary parameter (`_arc_jac_sc1`):
/// `arc_jac_sn(i w, m)` is purely imaginary; its imaginary part.
pub fn arc_jac_sc1(w: f64, m: f64) -> f64 {
    arc_jac_sn(C64::new(0.0, w), m).im
}

// GAMMA FUNCTION ==================================================================================

/// Lanczos approximation, g = 7, 9 terms (relative error about 1e-15).
const LANCZOS_G: f64 = 7.0;
const LANCZOS: [f64; 9] = [
    0.999_999_999_999_809_9,
    676.520_368_121_885_1,
    -1_259.139_216_722_402_8,
    771.323_428_777_653_1,
    -176.615_029_162_140_6,
    12.507_343_278_686_905,
    -0.138_571_095_265_720_12,
    9.984_369_578_019_572e-6,
    1.505_632_735_149_311_6e-7,
];

/// `ln |Γ(x)|` (`scipy.special.gammaln`): Lanczos for `x >= 0.5`, the reflection formula below.
/// Infinite at the poles (0, -1, -2, ...).
pub fn gammaln(x: f64) -> f64 {
    if x.is_nan() {
        return f64::NAN;
    }
    if x <= 0.0 && x == x.floor() {
        return f64::INFINITY;
    }
    if x < 0.5 {
        // Γ(x) Γ(1 - x) = π / sin(π x)
        let s = (std::f64::consts::PI * x).sin().abs();
        return std::f64::consts::PI.ln() - s.ln() - gammaln(1.0 - x);
    }
    let x = x - 1.0;
    let t = x + LANCZOS_G + 0.5;
    let series = LANCZOS[1..].iter().enumerate().fold(LANCZOS[0], |s, (i, &c)| s + c / (x + (i + 1) as f64));
    0.5 * std::f64::consts::TAU.ln() + (x + 0.5) * t.ln() - t + series.ln()
}

/// `Γ(x)` (`scipy.special.gamma`): the sign from the reflection formula, the magnitude from
/// [`gammaln`]; exact factorials for small positive integers. NaN at the poles.
pub fn gamma(x: f64) -> f64 {
    if x <= 0.0 && x == x.floor() {
        return f64::NAN;
    }
    if x == x.floor() && x <= 23.0 {
        return (1..x as u64).map(|k| k as f64).product();
    }
    // Γ is negative on (-1, 0), (-3, -2), ...
    let sign = if x < 0.0 && (x.floor() as i64).rem_euclid(2) == 1 { -1.0 } else { 1.0 };
    sign * gammaln(x).exp()
}

// ERROR FUNCTION AND THE NORMAL DISTRIBUTION ======================================================

/// `exp(-s x²)` for `s` of 1 or 1/2, without the rounding of `x²` (relative error `s x² ε`, 3e-15
/// at x = 6) that `exp(-s * x * x)` has: `x = m + f` with `m` a multiple of 1/128, so `s m²` is
/// exact and only the small remainder `s (2 m f + f²)` is rounded (cephes' `expx2`).
fn exp_neg_sq(x: f64, s: f64) -> f64 {
    let x = x.abs();
    let m = (x * 128.0).round() / 128.0;
    let f = x - m;
    (-(s * m * m)).exp() * (-(s * (2.0 * m * f + f * f))).exp()
}

/// The continued fraction `F(x)` with `erfc(x) = exp(-x²) / (F(x) √π)`, for `x >= 0.5` (modified
/// Lentz; a few hundred terms near 0.5, a handful above 4).
fn erfc_fraction(x: f64) -> f64 {
    // F(x) = x + (1/2) / (x + 1 / (x + (3/2) / (x + 2 / (x + ...))))
    let tiny = 1e-300;
    let mut f = x;
    let (mut c, mut d) = (x, 0.0);
    for k in 1..5000 {
        let a = k as f64 / 2.0;
        d = x + a * d;
        d = if d.abs() < tiny { tiny } else { d };
        c = x + a / c;
        c = if c.abs() < tiny { tiny } else { c };
        d = 1.0 / d;
        let delta = c * d;
        f *= delta;
        if (delta - 1.0).abs() < 1e-16 {
            break;
        }
    }
    f
}

/// `erfc(x)` for `x >= 0.5`, accurate to rounding.
fn erfc_tail(x: f64) -> f64 {
    exp_neg_sq(x, 1.0) / (erfc_fraction(x) * std::f64::consts::PI.sqrt())
}

/// Where [`erfc`] switches from `1 - erf` (whose cancellation would cost digits above it) to the
/// continued fraction.
const ERFC_SPLIT: f64 = 0.5;

/// The error function (`scipy.special.erf`): its Taylor series below 2, `1 - erfc` above.
pub fn erf(x: f64) -> f64 {
    if x.is_nan() {
        return f64::NAN;
    }
    if x.abs() >= 2.0 {
        return x.signum() * (1.0 - erfc_tail(x.abs()));
    }
    // 2/√π Σ (-1)^n x^(2n+1) / (n! (2n+1))
    let (x2, mut term, mut sum) = (x * x, x, x);
    for n in 1..100 {
        term *= -x2 / n as f64;
        let add = term / (2 * n + 1) as f64;
        sum += add;
        if add.abs() < 1e-17 * sum.abs() {
            break;
        }
    }
    sum * 2.0 / std::f64::consts::PI.sqrt()
}

/// The complementary error function `1 - erf(x)` (`scipy.special.erfc`), accurate in the tails
/// (no cancellation for large `x`).
pub fn erfc(x: f64) -> f64 {
    if x.is_nan() {
        return f64::NAN;
    }
    if x >= ERFC_SPLIT {
        erfc_tail(x)
    } else if x <= -2.0 {
        2.0 - erfc_tail(-x)
    } else {
        1.0 - erf(x)
    }
}

/// The standard normal cumulative distribution function (`scipy.special.ndtr`). In the tails the
/// Gaussian factor `exp(-x²/2)` comes from `x` itself, not from `x / √2` (whose rounding the
/// exponential would amplify), so tiny probabilities keep their digits.
pub fn ndtr(x: f64) -> f64 {
    let z = x.abs() / std::f64::consts::SQRT_2;
    if z < ERFC_SPLIT {
        return 0.5 * erfc(-x / std::f64::consts::SQRT_2);
    }
    let tail = 0.5 * exp_neg_sq(x, 0.5) / (erfc_fraction(z) * std::f64::consts::PI.sqrt());
    if x < 0.0 { tail } else { 1.0 - tail }
}

/// The inverse of [`ndtr`]: the standard normal quantile of probability `p` (`scipy.special.ndtri`).
/// Acklam's rational approximation refined by Halley steps on [`erfc`], so accurate to rounding;
/// `-inf` / `inf` at 0 / 1, NaN outside.
pub fn ndtri(p: f64) -> f64 {
    if !(0.0..=1.0).contains(&p) {
        return f64::NAN;
    }
    if p == 0.0 {
        return f64::NEG_INFINITY;
    }
    if p == 1.0 {
        return f64::INFINITY;
    }
    const A: [f64; 6] = [-3.969_683_028_665_376e1, 2.209_460_984_245_205e2, -2.759_285_104_469_687e2, 1.383_577_518_672_69e2, -3.066_479_806_614_716e1, 2.506_628_277_459_239];
    const B: [f64; 5] = [-5.447_609_879_822_406e1, 1.615_858_368_580_409e2, -1.556_989_798_598_866e2, 6.680_131_188_771_972e1, -1.328_068_155_288_572e1];
    const C: [f64; 6] = [-7.784_894_002_430_293e-3, -3.223_964_580_411_365e-1, -2.400_758_277_161_838, -2.549_732_539_343_734, 4.374_664_141_464_968, 2.938_163_982_698_783];
    const D: [f64; 4] = [7.784_695_709_041_462e-3, 3.224_671_290_700_398e-1, 2.445_134_137_142_996, 3.754_408_661_907_416];
    let horner = |c: &[f64], x: f64| c.iter().fold(0.0, |acc, &k| acc * x + k);
    let tail = |q: f64| horner(&C, q) / (horner(&D, q) * q + 1.0);
    let low = 0.02425;
    let mut x = if p < low {
        tail((-2.0 * p.ln()).sqrt())
    } else if p > 1.0 - low {
        -tail((-2.0 * (1.0 - p).ln()).sqrt())
    } else {
        let q = p - 0.5;
        let r = q * q;
        horner(&A, r) * q / (horner(&B, r) * r + 1.0)
    };
    // Halley's method on ndtr(x) - p, the error taken from the nearer tail so it keeps its digits
    for _ in 0..2 {
        // ndtr(x) - p, the upper tail written as (1 - p) - ndtr(-x)
        let e = if x < 0.0 { ndtr(x) - p } else { (1.0 - p) - ndtr(-x) };
        let u = e * std::f64::consts::TAU.sqrt() * (x * x / 2.0).exp();
        x -= u / (1.0 + x * u / 2.0);
    }
    x
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn elliptic_integrals_match_known_values() {
        // K(0) = π/2; K(1/2) = 1.854074677301372; K(0.99) = 3.695637362989875
        assert!((ellipk(0.0) - std::f64::consts::FRAC_PI_2).abs() < 1e-15);
        assert!((ellipk(0.5) - 1.854_074_677_301_372).abs() < 1e-14);
        assert!((ellipk(0.99) - 3.695_637_362_989_875).abs() < 1e-13);
        assert!((ellipkm1(0.01) - ellipk(0.99)).abs() < 1e-13);
    }

    #[test]
    fn jacobi_functions_satisfy_their_identities() {
        for &(u, m) in &[(0.3, 0.2), (1.1, 0.7), (2.0, 0.95), (0.5, 1e-12), (0.8, 1.0 - 1e-12)] {
            let (sn, cn, dn, _) = ellipj(u, m);
            assert!((sn * sn + cn * cn - 1.0).abs() < 1e-12, "u {u} m {m}");
            assert!((dn * dn + m * sn * sn - 1.0).abs() < 1e-12, "u {u} m {m}");
        }
        // sn(K(m), m) = 1
        let m = 0.6;
        assert!((ellipj(ellipk(m), m).0 - 1.0).abs() < 1e-12);
        // the inverse: arc_jac_sn(sn(u)) = u
        let (sn, ..) = ellipj(0.7, m);
        assert!((arc_jac_sn(C64::new(sn, 0.0), m).re - 0.7).abs() < 1e-12);
    }

    /// Relative error, or absolute below 1.
    fn rel(got: f64, want: f64) -> f64 {
        (got - want).abs() / want.abs().max(1.0)
    }

    #[test]
    fn gamma_matches_scipy() {
        let cases = [
            (0.1, 2.252712651734206),
            (0.5, 0.5723649429247),
            (1.0, 0.0),
            (1.5, -0.12078223763524526),
            (2.5, 0.2846828704729192),
            (7.3, 7.147892523022249),
            (30.0, 71.257038967168),
            (171.5, 709.1431630309283),
            (1e5, 1051287.7089736569),
            (-0.5, 1.2655121234846454),
            (-2.7, -0.0714070853156458),
        ];
        for (x, want) in cases {
            assert!(rel(gammaln(x), want) < 1e-14, "gammaln({x}) = {} vs {want}", gammaln(x));
        }
        for (x, want) in [(0.1, 9.513507698668732), (0.5, 1.7724538509055159), (4.5, 11.63172839656745), (-0.5, -3.5449077018110318), (-2.7, -0.931082784838964), (20.0, 1.21645100408832e17)] {
            assert!((gamma(x) - want).abs() < 1e-13 * want.abs(), "gamma({x}) = {} vs {want}", gamma(x));
        }
        assert_eq!(gammaln(0.0), f64::INFINITY);
        assert!(gamma(-3.0).is_nan());
    }

    #[test]
    fn error_functions_match_scipy() {
        let xs = [-3.5, -1.0, -0.3, 0.0, 1e-5, 0.4, 1.0, 1.7, 2.5, 4.0, 6.0, 10.0];
        let erfs = [-0.9999992569016276, -0.8427007929497148, -0.3286267594591274, 0.0, 1.1283791670579e-05, 0.42839235504666845, 0.8427007929497148, 0.9837904585907745, 0.999593047982555, 0.9999999845827421, 1.0, 1.0];
        let erfcs = [1.9999992569016276, 1.8427007929497148, 1.3286267594591274, 1.0, 0.9999887162083294, 0.5716076449533316, 0.15729920705028516, 0.01620954140922544, 0.00040695201744495886, 1.541725790028002e-08, 2.1519736712498913e-17, 2.0884875837625446e-45];
        for ((&x, &e), &c) in xs.iter().zip(&erfs).zip(&erfcs) {
            assert!((erf(x) - e).abs() <= 1e-15 * e.abs() + 1e-20, "erf({x}) = {} vs {e}", erf(x));
            assert!((erfc(x) - c).abs() < 1e-14 * c, "erfc({x}) = {} vs {c}", erfc(x));
        }
        assert!((erfc(0.5) - 0.4795001221869535).abs() < 1e-15);
        for (x, want) in [(-10.0, 7.61985302416047e-24), (-2.0, 0.022750131948179195), (0.0, 0.5), (0.5, 0.6914624612740131), (3.0, 0.9986501019683699)] {
            assert!((ndtr(x) - want).abs() < 1e-14 * want, "ndtr({x}) = {} vs {want}", ndtr(x));
        }
    }

    #[test]
    fn normal_quantiles_match_scipy() {
        let cases = [
            (1e-300, -37.0470962993612),
            (1e-20, -9.262340089798409),
            (1e-5, -4.264890793922825),
            (0.02, -2.053748910631823),
            (0.1, -1.2815515655446004),
            (0.3, -0.5244005127080409),
            (0.5, 0.0),
            (0.7, 0.5244005127080407),
            (0.975, 1.959963984540054),
            (0.99999, 4.264890793923841),
        ];
        for (p, want) in cases {
            assert!((ndtri(p) - want).abs() < 1e-13 * want.abs().max(1.0), "ndtri({p}) = {} vs {want}", ndtri(p));
        }
        assert_eq!((ndtri(0.0), ndtri(1.0)), (f64::NEG_INFINITY, f64::INFINITY));
        assert!(ndtri(1.5).is_nan());
    }

    #[test]
    fn bessel_i0_known_values() {
        assert_eq!(bessel_i0(0.0), 1.0);
        assert!((bessel_i0(1.0) - 1.266_065_877_752_008_4).abs() < 1e-15);
        assert!((bessel_i0(10.0) - 2_815.716_628_466_254).abs() < 1e-9);
    }
}
