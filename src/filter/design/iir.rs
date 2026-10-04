//! IIR design: analog prototypes (Butterworth, Chebyshev I and II, elliptic, Bessel), frequency
//! transformations, and digital filters through the bilinear transform (`scipy.signal.iirfilter`).

use std::f64::consts::PI;

use super::special::{arc_jac_sc1, ellipj, ellipk, ellipkm1};
use super::DesignError;
use crate::systems::{bilinear_zpk, Domain, Zpk, C64};

/// Which band a filter passes, with its edge frequencies: Hz for a digital design, rad/s for an
/// analog one.
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum Band {
    /// Passes below the edge.
    Lowpass(f64),
    /// Passes above the edge.
    Highpass(f64),
    /// Passes between the two edges.
    Bandpass(f64, f64),
    /// Stops between the two edges.
    Bandstop(f64, f64),
}

/// A digital design at sample rate `fs` (edges in Hz), or an analog one (edges in rad/s).
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum Design {
    /// A digital filter (bilinear transform with prewarping).
    Digital {
        /// The sample rate in Hz.
        fs: f64,
    },
    /// An analog filter (`s`-domain).
    Analog,
}

/// How a Bessel filter is normalized (`scipy.signal.besselap`'s `norm`).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum BesselNorm {
    /// Phase-matched: half the maximum phase shift at the edge; same asymptotes as Butterworth.
    Phase,
    /// A group delay of 1 / edge at low frequencies.
    Delay,
    /// -3 dB at the edge.
    Mag,
}

/// The analog prototype family and its specifications.
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum IirKind {
    /// Maximally flat passband.
    Butterworth,
    /// Passband ripple `rp` dB.
    Chebyshev1 {
        /// Passband ripple in dB.
        rp: f64,
    },
    /// Stopband attenuation `rs` dB.
    Chebyshev2 {
        /// Stopband attenuation in dB.
        rs: f64,
    },
    /// Passband ripple `rp` dB and stopband attenuation `rs` dB.
    Elliptic {
        /// Passband ripple in dB.
        rp: f64,
        /// Stopband attenuation in dB.
        rs: f64,
    },
    /// Maximally flat group delay.
    Bessel {
        /// How the edge frequency is defined.
        norm: BesselNorm,
    },
}

fn real(x: f64) -> C64 {
    C64::new(x, 0.0)
}

fn prod(v: &[C64]) -> C64 {
    v.iter().fold(C64::one(), |acc, &x| acc * x)
}

/// Odd-spaced indices `-n+1, -n+3, ..., n-1`.
fn spread(n: usize) -> impl Iterator<Item = f64> {
    (0..n).map(move |i| -(n as f64) + 1.0 + 2.0 * i as f64)
}

/// Butterworth analog prototype: poles evenly on the unit circle's left half (`buttap`).
pub fn buttap(n: usize) -> (Vec<C64>, Vec<C64>, f64) {
    let poles = spread(n).map(|m| -C64::cis(PI * m / (2.0 * n as f64))).collect();
    (Vec::new(), poles, 1.0)
}

/// Chebyshev type I analog prototype with `rp` dB passband ripple (`cheb1ap`).
pub fn cheb1ap(n: usize, rp: f64) -> (Vec<C64>, Vec<C64>, f64) {
    if n == 0 {
        return (Vec::new(), Vec::new(), 10f64.powf(-rp / 20.0));
    }
    let eps = (10f64.powf(0.1 * rp) - 1.0).sqrt();
    let mu = (1.0 / eps).asinh() / n as f64;
    let poles: Vec<C64> = spread(n)
        .map(|m| {
            // -sinh(mu + iθ)
            let theta = PI * m / (2.0 * n as f64);
            -C64::new(mu.sinh() * theta.cos(), mu.cosh() * theta.sin())
        })
        .collect();
    let mut k = prod(&poles.iter().map(|&p| -p).collect::<Vec<_>>()).re;
    if n.is_multiple_of(2) {
        k /= (1.0 + eps * eps).sqrt();
    }
    (Vec::new(), poles, k)
}

/// Chebyshev type II analog prototype with `rs` dB stopband attenuation (`cheb2ap`).
pub fn cheb2ap(n: usize, rs: f64) -> (Vec<C64>, Vec<C64>, f64) {
    if n == 0 {
        return (Vec::new(), Vec::new(), 1.0);
    }
    let de = 1.0 / (10f64.powf(0.1 * rs) - 1.0).sqrt();
    let mu = (1.0 / de).asinh() / n as f64;
    let ms: Vec<f64> = if n % 2 == 1 {
        spread(n).filter(|&m| m != 0.0).collect()
    } else {
        spread(n).collect()
    };
    // zeros on the imaginary axis: -conj(i / sin(mπ / 2n))
    let zeros: Vec<C64> = ms.iter().map(|&m| -(C64::new(0.0, 1.0) / (m * PI / (2.0 * n as f64)).sin()).conj()).collect();
    let poles: Vec<C64> = spread(n)
        .map(|m| {
            let p = -C64::cis(PI * m / (2.0 * n as f64));
            C64::new(mu.sinh() * p.re, mu.cosh() * p.im).recip()
        })
        .collect();
    let neg = |v: &[C64]| v.iter().map(|&x| -x).collect::<Vec<_>>();
    let k = (prod(&neg(&poles)) / prod(&neg(&zeros))).re;
    (zeros, poles, k)
}

/// `10^x - 1`, accurate for small `x`.
fn pow10m1(x: f64) -> f64 {
    (10f64.ln() * x).exp_m1()
}

/// Solves the elliptic degree equation with nomes (`_ellipdeg`).
fn ellipdeg(n: usize, m1: f64) -> f64 {
    let q1 = (-PI * ellipkm1(m1) / ellipk(m1)).exp();
    let q = q1.powf(1.0 / n as f64);
    let num: f64 = (0..=7).map(|m| q.powi(m * (m + 1))).sum();
    let den = 1.0 + 2.0 * (1..=8).map(|m| q.powi(m * m)).sum::<f64>();
    16.0 * q * (num / den).powi(4)
}

/// Elliptic (Cauer) analog prototype: `rp` dB passband ripple, `rs` dB stopband attenuation, the
/// steepest transition for its order (`ellipap`).
pub fn ellipap(n: usize, rp: f64, rs: f64) -> Result<(Vec<C64>, Vec<C64>, f64), DesignError> {
    if n == 0 {
        return Ok((Vec::new(), Vec::new(), 10f64.powf(-rp / 20.0)));
    }
    if n == 1 {
        let p = -(1.0 / pow10m1(0.1 * rp)).sqrt();
        return Ok((Vec::new(), vec![real(p)], -p));
    }
    let eps_sq = pow10m1(0.1 * rp);
    let eps = eps_sq.sqrt();
    let ck1_sq = eps_sq / pow10m1(0.1 * rs);
    if ck1_sq == 0.0 {
        return Err(DesignError::Invalid("cannot design an elliptic filter with these rp and rs".into()));
    }
    let val0 = ellipk(ck1_sq);
    let m = ellipdeg(n, ck1_sq);
    let capk = ellipk(m);
    let js: Vec<f64> = ((1 - n % 2)..n).step_by(2).map(|j| j as f64).collect();
    let sncndn: Vec<(f64, f64, f64)> = js.iter().map(|&j| { let (s, c, d, _) = ellipj(j * capk / n as f64, m); (s, c, d) }).collect();
    const EPSILON: f64 = 2e-16;
    let mut zeros: Vec<C64> = sncndn.iter().filter(|(s, _, _)| s.abs() > EPSILON).map(|(s, _, _)| C64::new(0.0, 1.0 / (m.sqrt() * s))).collect();
    let conj: Vec<C64> = zeros.iter().map(|z| z.conj()).collect();
    zeros.extend(conj);
    let r = arc_jac_sc1(1.0 / eps, ck1_sq);
    let v0 = capk * r / (n as f64 * val0);
    let (sv, cv, dv, _) = ellipj(v0, 1.0 - m);
    let mut poles: Vec<C64> = sncndn.iter().map(|&(s, c, d)| -C64::new(c * d * sv * cv, s * dv) / (1.0 - (d * sv).powi(2))).collect();
    if n % 2 == 1 {
        let total: f64 = poles.iter().map(|p| p.norm_sqr()).sum::<f64>().sqrt();
        let newp: Vec<C64> = poles.iter().filter(|p| p.im.abs() > EPSILON * total).map(|p| p.conj()).collect();
        poles.extend(newp);
    } else {
        let conj: Vec<C64> = poles.iter().map(|p| p.conj()).collect();
        poles.extend(conj);
    }
    let neg = |v: &[C64]| v.iter().map(|&x| -x).collect::<Vec<_>>();
    let mut k = (prod(&neg(&poles)) / prod(&neg(&zeros))).re;
    if n.is_multiple_of(2) {
        k /= (1.0 + eps_sq).sqrt();
    }
    Ok((zeros, poles, k))
}

/// `n!` as a float.
fn factorial(n: usize) -> f64 {
    (1..=n).map(|k| k as f64).product()
}

/// Bessel analog prototype (`besselap`): the roots of the reverse Bessel polynomial, normalized.
pub fn besselap(n: usize, norm: BesselNorm) -> Result<(Vec<C64>, Vec<C64>, f64), DesignError> {
    if n == 0 {
        return Ok((Vec::new(), Vec::new(), 1.0));
    }
    // θ_n(s) = Σ a_k s^k, a_k = (2n - k)! / (2^(n-k) k! (n - k)!), highest power first
    let coeffs: Vec<f64> = (0..=n).rev().map(|k| factorial(2 * n - k) / (2f64.powi((n - k) as i32) * factorial(k) * factorial(n - k))).collect();
    let mut poles = crate::linalg::roots(&coeffs).map_err(|e| DesignError::Invalid(e.to_string()))?;
    // polish with Newton steps on θ_n (the companion matrix loses digits as n grows)
    let deriv: Vec<f64> = coeffs.iter().enumerate().take(n).map(|(i, &c)| c * (n - i) as f64).collect();
    for p in &mut poles {
        for _ in 0..4 {
            let f = crate::linalg::polyval_complex(&coeffs, *p);
            let df = crate::linalg::polyval_complex(&deriv, *p);
            if df.norm() == 0.0 {
                break;
            }
            *p -= f / df;
        }
    }
    let a_last = coeffs[n]; // θ_n(0) = (2n)! / (2^n n!)
    let (poles, k) = match norm {
        BesselNorm::Phase => {
            let scale = 10f64.powf(-a_last.log10() / n as f64);
            (poles.into_iter().map(|p| p * scale).collect(), 1.0)
        }
        BesselNorm::Delay => (poles, a_last),
        BesselNorm::Mag => {
            // the frequency where |H(jw)| = 1/sqrt(2), by bisection then Newton-free refinement
            let gain = |w: f64| (a_last / prod(&poles.iter().map(|&p| C64::new(0.0, w) - p).collect::<Vec<_>>()).norm()).abs();
            let target = 0.5f64.sqrt();
            let (mut lo, mut hi) = (0.0, 1.0);
            while gain(hi) > target {
                hi *= 2.0;
            }
            for _ in 0..200 {
                let mid = 0.5 * (lo + hi);
                if gain(mid) > target {
                    lo = mid;
                } else {
                    hi = mid;
                }
            }
            let w = 0.5 * (lo + hi);
            (poles.into_iter().map(|p| p / w).collect(), a_last / w.powi(n as i32))
        }
    };
    Ok((Vec::new(), poles, k))
}

/// Low-pass prototype to low-pass with cutoff `wo` (`lp2lp_zpk`).
pub fn lp2lp_zpk(z: &[C64], p: &[C64], k: f64, wo: f64) -> (Vec<C64>, Vec<C64>, f64) {
    let degree = p.len() as i32 - z.len() as i32;
    (z.iter().map(|&x| x * wo).collect(), p.iter().map(|&x| x * wo).collect(), k * wo.powi(degree))
}

/// Low-pass prototype to high-pass with cutoff `wo` (`lp2hp_zpk`).
pub fn lp2hp_zpk(z: &[C64], p: &[C64], k: f64, wo: f64) -> (Vec<C64>, Vec<C64>, f64) {
    let degree = p.len().saturating_sub(z.len());
    let mut zh: Vec<C64> = z.iter().map(|&x| real(wo) / x).collect();
    zh.extend(std::iter::repeat_n(C64::zero(), degree));
    let ph = p.iter().map(|&x| real(wo) / x).collect();
    let neg = |v: &[C64]| v.iter().map(|&x| -x).collect::<Vec<_>>();
    (zh, ph, k * (prod(&neg(z)) / prod(&neg(p))).re)
}

/// Low-pass prototype to band-pass at center `wo` with bandwidth `bw` (`lp2bp_zpk`).
pub fn lp2bp_zpk(z: &[C64], p: &[C64], k: f64, wo: f64, bw: f64) -> (Vec<C64>, Vec<C64>, f64) {
    let degree = p.len().saturating_sub(z.len());
    let split = |v: &[C64]| -> Vec<C64> {
        let lp: Vec<C64> = v.iter().map(|&x| x * (bw / 2.0)).collect();
        let roots: Vec<C64> = lp.iter().map(|&x| (x * x - real(wo * wo)).sqrt()).collect();
        lp.iter().zip(&roots).map(|(&a, &r)| a + r).chain(lp.iter().zip(&roots).map(|(&a, &r)| a - r)).collect()
    };
    let mut zb = split(z);
    zb.extend(std::iter::repeat_n(C64::zero(), degree));
    (zb, split(p), k * bw.powi(degree as i32))
}

/// Low-pass prototype to band-stop at center `wo` with bandwidth `bw` (`lp2bs_zpk`).
pub fn lp2bs_zpk(z: &[C64], p: &[C64], k: f64, wo: f64, bw: f64) -> (Vec<C64>, Vec<C64>, f64) {
    let degree = p.len().saturating_sub(z.len());
    let split = |v: &[C64]| -> Vec<C64> {
        let hp: Vec<C64> = v.iter().map(|&x| real(bw / 2.0) / x).collect();
        let roots: Vec<C64> = hp.iter().map(|&x| (x * x - real(wo * wo)).sqrt()).collect();
        hp.iter().zip(&roots).map(|(&a, &r)| a + r).chain(hp.iter().zip(&roots).map(|(&a, &r)| a - r)).collect()
    };
    let mut zb = split(z);
    zb.extend(std::iter::repeat_n(C64::new(0.0, wo), degree));
    zb.extend(std::iter::repeat_n(C64::new(0.0, -wo), degree));
    let neg = |v: &[C64]| v.iter().map(|&x| -x).collect::<Vec<_>>();
    (zb, split(p), k * (prod(&neg(z)) / prod(&neg(p))).re)
}

/// An IIR filter of `order` from an analog prototype (`scipy.signal.iirfilter`): the prototype is
/// moved to the band (edges prewarped for a digital design) and, for a digital design, mapped to
/// the z-plane by the bilinear transform. Convert the result with `to_sos()` (recommended) or
/// `to_tf()`.
pub fn iirfilter(order: usize, band: Band, kind: IirKind, design: Design) -> Result<Zpk, DesignError> {
    let (z, p, k) = match kind {
        IirKind::Butterworth => buttap(order),
        IirKind::Chebyshev1 { rp } => cheb1ap(order, rp),
        IirKind::Chebyshev2 { rs } => cheb2ap(order, rs),
        IirKind::Elliptic { rp, rs } => ellipap(order, rp, rs)?,
        IirKind::Bessel { norm } => besselap(order, norm)?,
    };
    // edges: rad/s for analog; for digital, prewarped from Hz as in SciPy (fs = 2 after
    // normalizing to the Nyquist frequency)
    let (edges, domain): (Vec<f64>, Domain) = match design {
        Design::Analog => (band_edges(band), Domain::Continuous),
        Design::Digital { fs } => {
            let nyquist = fs / 2.0;
            let normalized: Vec<f64> = band_edges(band).iter().map(|&w| w / nyquist).collect();
            for &w in &normalized {
                if !(w > 0.0 && w < 1.0) {
                    return Err(DesignError::Invalid(format!("digital filter edges must be between 0 and the Nyquist frequency {nyquist} Hz")));
                }
            }
            (normalized.iter().map(|&w| 4.0 * (PI * w / 2.0).tan()).collect(), Domain::sampled(fs))
        }
    };
    let (z, p, k) = match band {
        Band::Lowpass(_) => lp2lp_zpk(&z, &p, k, edges[0]),
        Band::Highpass(_) => lp2hp_zpk(&z, &p, k, edges[0]),
        Band::Bandpass(..) | Band::Bandstop(..) => {
            if edges[1] <= edges[0] {
                return Err(DesignError::Invalid("band edges must increase".into()));
            }
            let (bw, wo) = (edges[1] - edges[0], (edges[0] * edges[1]).sqrt());
            if matches!(band, Band::Bandpass(..)) {
                lp2bp_zpk(&z, &p, k, wo, bw)
            } else {
                lp2bs_zpk(&z, &p, k, wo, bw)
            }
        }
    };
    match domain {
        Domain::Continuous => Ok(Zpk::new(z, p, k, domain)),
        Domain::Discrete { .. } => {
            // the bilinear transform at the normalized rate fs = 2
            let (z, p, k) = bilinear_zpk(&z, &p, k, 2.0);
            Ok(Zpk::new(z, p, k, domain))
        }
    }
}

fn band_edges(band: Band) -> Vec<f64> {
    match band {
        Band::Lowpass(w) | Band::Highpass(w) => vec![w],
        Band::Bandpass(a, b) | Band::Bandstop(a, b) => vec![a, b],
    }
}

/// A Butterworth filter: maximally flat passband (`scipy.signal.butter`).
pub fn butter(order: usize, band: Band, design: Design) -> Result<Zpk, DesignError> {
    iirfilter(order, band, IirKind::Butterworth, design)
}

/// A Chebyshev type I filter: `rp` dB passband ripple, a steeper edge (`scipy.signal.cheby1`).
pub fn cheby1(order: usize, rp: f64, band: Band, design: Design) -> Result<Zpk, DesignError> {
    iirfilter(order, band, IirKind::Chebyshev1 { rp }, design)
}

/// A Chebyshev type II filter: flat passband, `rs` dB stopband ripple (`scipy.signal.cheby2`). The
/// band edges are where the stopband starts.
pub fn cheby2(order: usize, rs: f64, band: Band, design: Design) -> Result<Zpk, DesignError> {
    iirfilter(order, band, IirKind::Chebyshev2 { rs }, design)
}

/// An elliptic (Cauer) filter: `rp` dB passband ripple, `rs` dB stopband attenuation, the steepest
/// edge for its order (`scipy.signal.ellip`).
pub fn ellip(order: usize, rp: f64, rs: f64, band: Band, design: Design) -> Result<Zpk, DesignError> {
    iirfilter(order, band, IirKind::Elliptic { rp, rs }, design)
}

/// A Bessel filter: maximally flat group delay (`scipy.signal.bessel`).
pub fn bessel(order: usize, band: Band, norm: BesselNorm, design: Design) -> Result<Zpk, DesignError> {
    iirfilter(order, band, IirKind::Bessel { norm }, design)
}
