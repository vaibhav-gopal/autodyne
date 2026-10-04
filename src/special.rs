//! Special functions (`scipy.special` names): the modified Bessel function I₀ (Kaiser windows and
//! the sampler's interpolation kernel), complete elliptic integrals, and Jacobi elliptic functions
//! and their inverses (elliptic filter design).

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

    #[test]
    fn bessel_i0_known_values() {
        assert_eq!(bessel_i0(0.0), 1.0);
        assert!((bessel_i0(1.0) - 1.266_065_877_752_008_4).abs() < 1e-15);
        assert!((bessel_i0(10.0) - 2_815.716_628_466_254).abs() < 1e-9);
    }
}
