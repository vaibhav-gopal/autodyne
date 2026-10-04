//! Window functions (`scipy.signal.windows`): tapers for spectral analysis and FIR design.

use std::f64::consts::PI;
use std::str::FromStr;

use crate::fft::Fft;
use crate::special::bessel_i0;
use crate::units::Complex;

/// A window shape and its parameters (`scipy.signal.get_window`'s `window` argument).
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum WindowSpec {
    /// Rectangular (no taper).
    Boxcar,
    /// Triangular, without zero ends.
    Triang,
    /// Triangular, with zero ends.
    Bartlett,
    /// Hann (raised cosine).
    Hann,
    /// Hamming.
    Hamming,
    /// Blackman.
    Blackman,
    /// 4-term Blackman-Harris.
    BlackmanHarris,
    /// 4-term Nuttall.
    Nuttall,
    /// Flat top (accurate amplitudes, wide peaks).
    Flattop,
    /// Bohman.
    Bohman,
    /// Parzen (de la Vallée Poussin).
    Parzen,
    /// Cosine (sine) window.
    Cosine,
    /// Kaiser with shape `beta` (0: rectangular; 8.6: about Blackman).
    Kaiser {
        /// Shape: 0 is rectangular, larger values taper more.
        beta: f64,
    },
    /// Gaussian with standard deviation `std` samples.
    Gaussian {
        /// Standard deviation in samples.
        std: f64,
    },
    /// Tukey (tapered cosine): `alpha` is the tapered fraction (0: rectangular, 1: Hann).
    Tukey {
        /// Fraction of the window that tapers.
        alpha: f64,
    },
    /// Exponential (Poisson), centered, with decay `tau` samples.
    Exponential {
        /// Decay constant in samples.
        tau: f64,
    },
    /// Dolph-Chebyshev with side lobes `attenuation` dB down.
    Chebwin {
        /// Side-lobe level below the main lobe, in dB.
        attenuation: f64,
    },
}

/// The names `FromStr` accepts (parameterless windows).
impl FromStr for WindowSpec {
    type Err = super::SpectralError;
    fn from_str(name: &str) -> Result<Self, Self::Err> {
        Ok(match name.to_ascii_lowercase().as_str() {
            "boxcar" | "rectangular" | "rect" | "ones" => WindowSpec::Boxcar,
            "triang" | "triangle" | "tri" => WindowSpec::Triang,
            "bartlett" | "bart" | "brt" => WindowSpec::Bartlett,
            "hann" | "han" | "hanning" => WindowSpec::Hann,
            "hamming" | "hamm" | "ham" => WindowSpec::Hamming,
            "blackman" | "black" | "blk" => WindowSpec::Blackman,
            "blackmanharris" | "blackharr" | "bkh" => WindowSpec::BlackmanHarris,
            "nuttall" | "nutl" | "nut" => WindowSpec::Nuttall,
            "flattop" | "flat" | "flt" => WindowSpec::Flattop,
            "bohman" | "bman" | "bmn" => WindowSpec::Bohman,
            "parzen" | "parz" | "par" => WindowSpec::Parzen,
            "cosine" | "halfcosine" => WindowSpec::Cosine,
            "tukey" | "tuk" => WindowSpec::Tukey { alpha: 0.5 },
            "exponential" | "poisson" => WindowSpec::Exponential { tau: 1.0 },
            other => return Err(super::SpectralError::invalid(format!("unknown window '{other}' (parameterized windows: use the enum)"))),
        })
    }
}

/// `n` window coefficients (`scipy.signal.get_window`). `periodic` (SciPy's `fftbins`, its
/// default) gives the window of period `n` (for spectral analysis: frames tile exactly);
/// otherwise the symmetric window (for filter design).
pub fn get_window(spec: WindowSpec, n: usize, periodic: bool) -> Vec<f64> {
    if n <= 1 {
        return vec![1.0; n];
    }
    // a periodic window is the symmetric one of n + 1 points without its last point
    let m = if periodic { n + 1 } else { n };
    let mut w = symmetric(spec, m);
    w.truncate(n);
    w
}

/// The symmetric window of `m >= 2` points.
fn symmetric(spec: WindowSpec, m: usize) -> Vec<f64> {
    let mf = m as f64;
    let idx = |i: usize| i as f64;
    match spec {
        WindowSpec::Boxcar => vec![1.0; m],
        WindowSpec::Triang => {
            let half: Vec<f64> = (1..=m.div_ceil(2)).map(|n| if m.is_multiple_of(2) { (2.0 * n as f64 - 1.0) / mf } else { 2.0 * n as f64 / (mf + 1.0) }).collect();
            let mut w = half.clone();
            let tail = if m.is_multiple_of(2) { &half[..] } else { &half[..half.len() - 1] };
            w.extend(tail.iter().rev());
            w
        }
        WindowSpec::Bartlett => (0..m).map(|i| if idx(i) <= (mf - 1.0) / 2.0 { 2.0 * idx(i) / (mf - 1.0) } else { 2.0 - 2.0 * idx(i) / (mf - 1.0) }).collect(),
        WindowSpec::Hann => cosine_sum(m, &[0.5, 0.5]),
        WindowSpec::Hamming => cosine_sum(m, &[0.54, 0.46]),
        WindowSpec::Blackman => cosine_sum(m, &[0.42, 0.50, 0.08]),
        WindowSpec::BlackmanHarris => cosine_sum(m, &[0.35875, 0.48829, 0.14128, 0.01168]),
        WindowSpec::Nuttall => cosine_sum(m, &[0.3635819, 0.4891775, 0.1365995, 0.0106411]),
        WindowSpec::Flattop => cosine_sum(m, &[0.21557895, 0.41663158, 0.277263158, 0.083578947, 0.006947368]),
        WindowSpec::Bohman => {
            let mut w = vec![0.0; m];
            for (i, wi) in w.iter_mut().enumerate().take(m - 1).skip(1) {
                let fac = (-1.0 + 2.0 * idx(i) / (mf - 1.0)).abs();
                *wi = (1.0 - fac) * (PI * fac).cos() + (PI * fac).sin() / PI;
            }
            w
        }
        WindowSpec::Parzen => (0..m)
            .map(|i| {
                let n = idx(i) - (mf - 1.0) / 2.0;
                let r = n.abs() / (mf / 2.0);
                if n.abs() <= (mf - 1.0) / 4.0 { 1.0 - 6.0 * r * r + 6.0 * r * r * r } else { 2.0 * (1.0 - r).powi(3) }
            })
            .collect(),
        WindowSpec::Cosine => (0..m).map(|i| (PI / mf * (idx(i) + 0.5)).sin()).collect(),
        WindowSpec::Kaiser { beta } => {
            let alpha = (mf - 1.0) / 2.0;
            let i0b = bessel_i0(beta);
            (0..m).map(|i| bessel_i0(beta * (1.0 - ((idx(i) - alpha) / alpha).powi(2)).max(0.0).sqrt()) / i0b).collect()
        }
        WindowSpec::Gaussian { std } => (0..m).map(|i| (-(idx(i) - (mf - 1.0) / 2.0).powi(2) / (2.0 * std * std)).exp()).collect(),
        WindowSpec::Tukey { alpha } => {
            if alpha <= 0.0 {
                return vec![1.0; m];
            }
            if alpha >= 1.0 {
                return cosine_sum(m, &[0.5, 0.5]);
            }
            let width = (alpha * (mf - 1.0) / 2.0).floor() as usize;
            (0..m)
                .map(|i| {
                    let n = idx(i);
                    if i <= width {
                        0.5 * (1.0 + (PI * (-1.0 + 2.0 * n / alpha / (mf - 1.0))).cos())
                    } else if i < m - width - 1 {
                        1.0
                    } else {
                        0.5 * (1.0 + (PI * (-2.0 / alpha + 1.0 + 2.0 * n / alpha / (mf - 1.0))).cos())
                    }
                })
                .collect()
        }
        WindowSpec::Exponential { tau } => (0..m).map(|i| (-(idx(i) - (mf - 1.0) / 2.0).abs() / tau).exp()).collect(),
        WindowSpec::Chebwin { attenuation } => chebwin(m, attenuation),
    }
}

/// `Σ_k (-1)^k a_k cos(2πk n / (m - 1))`: the generalized cosine windows.
fn cosine_sum(m: usize, a: &[f64]) -> Vec<f64> {
    (0..m)
        .map(|i| {
            let fac = -PI + 2.0 * PI * i as f64 / (m - 1) as f64;
            a.iter().enumerate().map(|(k, &ak)| ak * (k as f64 * fac).cos()).sum()
        })
        .collect()
}

/// Dolph-Chebyshev: the Chebyshev polynomial sampled in frequency, transformed to time.
fn chebwin(m: usize, at: f64) -> Vec<f64> {
    let order = m as f64 - 1.0;
    let beta = ((10f64.powf(at.abs() / 20.0)).acosh() / order).cosh();
    let p: Vec<f64> = (0..m)
        .map(|k| {
            let x = beta * (PI * k as f64 / m as f64).cos();
            if x > 1.0 {
                (order * x.acosh()).cosh()
            } else if x < -1.0 {
                (2.0 * (m % 2) as f64 - 1.0) * (order * (-x).acosh()).cosh()
            } else {
                (order * x.acos()).cos()
            }
        })
        .collect();
    let mut buf: Vec<Complex<f64>> = if m % 2 == 1 {
        p.iter().map(|&v| Complex::new(v, 0.0)).collect()
    } else {
        p.iter().enumerate().map(|(k, &v)| Complex::cis(PI / m as f64 * k as f64) * v).collect()
    };
    Fft::new(m).forward(&mut buf);
    let re: Vec<f64> = buf.iter().map(|z| z.re).collect();
    let w: Vec<f64> = if m % 2 == 1 {
        let n = m.div_ceil(2);
        re[1..n].iter().rev().chain(&re[..n]).copied().collect()
    } else {
        let n = m / 2 + 1;
        re[1..n].iter().rev().chain(&re[1..n]).copied().collect()
    };
    let peak = w.iter().copied().fold(f64::NEG_INFINITY, f64::max);
    w.into_iter().map(|v| v / peak).collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    fn close(a: &[f64], b: &[f64]) -> bool {
        a.len() == b.len() && a.iter().zip(b).all(|(x, y)| (x - y).abs() < 1e-12)
    }

    #[test]
    fn known_windows() {
        assert!(close(&get_window(WindowSpec::Hann, 4, true), &[0.0, 0.5, 1.0, 0.5]));
        assert!(close(&get_window(WindowSpec::Hann, 5, false), &[0.0, 0.5, 1.0, 0.5, 0.0]));
        assert!(close(&get_window(WindowSpec::Hamming, 3, false), &[0.08, 1.0, 0.08]));
        assert!(close(&get_window(WindowSpec::Triang, 4, false), &[0.25, 0.75, 0.75, 0.25]));
        assert!(close(&get_window(WindowSpec::Triang, 5, false), &[1.0 / 3.0, 2.0 / 3.0, 1.0, 2.0 / 3.0, 1.0 / 3.0]));
        assert!(close(&get_window(WindowSpec::Bartlett, 5, false), &[0.0, 0.5, 1.0, 0.5, 0.0]));
        assert!(close(&get_window(WindowSpec::Kaiser { beta: 0.0 }, 4, false), &[1.0; 4]));
        assert!(close(&get_window(WindowSpec::Tukey { alpha: 1.0 }, 5, false), &get_window(WindowSpec::Hann, 5, false)));
        assert_eq!("hanning".parse::<WindowSpec>(), Ok(WindowSpec::Hann));
        let cheb = get_window(WindowSpec::Chebwin { attenuation: 100.0 }, 9, false);
        assert!((cheb[4] - 1.0).abs() < 1e-12 && (cheb[0] - cheb[8]).abs() < 1e-12);
    }
}
