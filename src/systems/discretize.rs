//! Continuous to discrete time: zero-order hold, the generalized bilinear family (Tustin, Euler,
//! backward difference), prewarped Tustin; `bilinear` and `bilinear_zpk` for filter design.

use super::{real, Domain, StateSpace, SystemError, TransferFunction, Zpk, C64};
use crate::linalg::{expm, matmul, solve};
use crate::signal::NdArray;

/// How a continuous system becomes a discrete one (`scipy.signal.cont2discrete`).
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum Method {
    /// Zero-order hold: exact for inputs held constant over each sample.
    Zoh,
    /// Tustin's bilinear transform (`gbt` with alpha 1/2): maps the imaginary axis onto the unit
    /// circle, warping frequencies.
    Bilinear,
    /// Tustin with the frequency `w0` (rad/s) preserved exactly.
    BilinearPrewarp(f64),
    /// Forward Euler (`gbt` with alpha 0).
    Euler,
    /// Backward difference (`gbt` with alpha 1).
    BackwardDiff,
    /// The generalized bilinear transform with weight `alpha` in [0, 1].
    Gbt(f64),
}

fn eye(n: usize) -> NdArray<f64> {
    NdArray::from_fn(&[n, n], |i| if i[0] == i[1] { 1.0 } else { 0.0 }).expect("n x n")
}

fn scaled(m: &NdArray<f64>, s: f64) -> NdArray<f64> {
    m.map(|&x| x * s)
}

fn add(a: &NdArray<f64>, b: &NdArray<f64>) -> NdArray<f64> {
    a + b
}

impl StateSpace {
    /// The discrete system for sample period `dt` (`scipy.signal.cont2discrete`).
    pub fn discretize(&self, dt: f64, method: Method) -> Result<StateSpace, SystemError> {
        if self.domain.is_discrete() {
            return Err(SystemError::Invalid("the system is already discrete".into()));
        }
        let (n, m) = (self.order(), self.b.shape()[1]);
        let domain = Domain::Discrete { dt };
        let alpha = match method {
            Method::Zoh => {
                // e^([[A, B], [0, 0]] dt) = [[Ad, Bd], [0, I]]
                let block = NdArray::from_fn(&[n + m, n + m], |i| {
                    let (r, c) = (i[0], i[1]);
                    if r >= n {
                        0.0
                    } else if c < n {
                        self.a.as_slice()[r * n + c] * dt
                    } else {
                        self.b.as_slice()[r * m + c - n] * dt
                    }
                })
                .expect("shape");
                let e = expm(block.view())?;
                let ad = NdArray::from_fn(&[n, n], |i| e.as_slice()[i[0] * (n + m) + i[1]]).expect("shape");
                let bd = NdArray::from_fn(&[n, m], |i| e.as_slice()[i[0] * (n + m) + n + i[1]]).expect("shape");
                return StateSpace::new(ad, bd, self.c.clone(), self.d.clone(), domain);
            }
            Method::Bilinear => 0.5,
            Method::BilinearPrewarp(w0) => {
                // Tustin with the period that maps w0 exactly: 2 tan(w0 dt / 2) / w0
                let effective = 2.0 * (w0 * dt / 2.0).tan() / w0;
                let mut sys = self.gbt(effective, 0.5)?;
                sys.domain = domain;
                return Ok(sys);
            }
            Method::Euler => 0.0,
            Method::BackwardDiff => 1.0,
            Method::Gbt(alpha) => alpha,
        };
        let mut sys = self.gbt(dt, alpha)?;
        sys.domain = domain;
        Ok(sys)
    }

    /// The generalized bilinear transform with period `dt` and weight `alpha`.
    fn gbt(&self, dt: f64, alpha: f64) -> Result<StateSpace, SystemError> {
        if !(0.0..=1.0).contains(&alpha) {
            return Err(SystemError::Invalid(format!("gbt alpha must be in [0, 1], got {alpha}")));
        }
        let n = self.order();
        let ima = add(&eye(n), &scaled(&self.a, -alpha * dt));
        let ad = solve(ima.view(), add(&eye(n), &scaled(&self.a, (1.0 - alpha) * dt)).view())?;
        let bd = solve(ima.view(), scaled(&self.b, dt).view())?;
        // cd = (ima⁻ᵀ cᵀ)ᵀ
        let cd = solve(ima.view().transpose(), self.c.view().transpose())?;
        let cd = cd.view().transpose().to_owned();
        let dd = add(&self.d, &scaled(&matmul(self.c.view(), bd.view())?, alpha));
        StateSpace::new(ad, bd, cd, dd, self.domain)
    }
}

impl TransferFunction {
    /// The discrete transfer function for sample period `dt` (through state space, as
    /// `scipy.signal.cont2discrete` does).
    pub fn discretize(&self, dt: f64, method: Method) -> Result<TransferFunction, SystemError> {
        if self.domain.is_discrete() {
            return Err(SystemError::Invalid("the system is already discrete".into()));
        }
        let ss = self.to_ss()?.discretize(dt, method)?;
        let (mut num, den) = ss.to_tf(0)?;
        Ok(TransferFunction { num: num.remove(0), den, domain: ss.domain })
    }
}

impl Zpk {
    /// The digital system for sample rate `fs` by the bilinear transform (`bilinear_zpk`).
    pub fn bilinear(&self, fs: f64) -> Result<Zpk, SystemError> {
        if self.domain.is_discrete() {
            return Err(SystemError::Invalid("the system is already discrete".into()));
        }
        let (zeros, poles, gain) = bilinear_zpk(&self.zeros, &self.poles, self.gain, fs);
        Ok(Zpk { zeros, poles, gain, domain: Domain::sampled(fs) })
    }
}

/// The bilinear transform of an analog zeros-poles-gain system to sample rate `fs`
/// (`scipy.signal.bilinear_zpk`); zeros at infinity land at z = -1.
pub fn bilinear_zpk(zeros: &[C64], poles: &[C64], gain: f64, fs: f64) -> (Vec<C64>, Vec<C64>, f64) {
    let degree = poles.len().saturating_sub(zeros.len());
    let fs2 = real(2.0 * fs);
    let mut z: Vec<C64> = zeros.iter().map(|&q| (fs2 + q) / (fs2 - q)).collect();
    let p: Vec<C64> = poles.iter().map(|&q| (fs2 + q) / (fs2 - q)).collect();
    z.extend(std::iter::repeat_n(real(-1.0), degree));
    let num = zeros.iter().fold(C64::one(), |acc, &q| acc * (fs2 - q));
    let den = poles.iter().fold(C64::one(), |acc, &q| acc * (fs2 - q));
    (z, p, gain * (num / den).re)
}

fn binomial(n: usize, k: usize) -> f64 {
    (0..k).fold(1.0, |acc, i| acc * (n - i) as f64 / (i + 1) as f64)
}

/// The bilinear transform of an analog transfer function `b(s) / a(s)` to sample rate `fs`
/// (`scipy.signal.bilinear`), normalized.
pub fn bilinear(b: &[f64], a: &[f64], fs: f64) -> Result<(Vec<f64>, Vec<f64>), SystemError> {
    let (n, d) = (b.len() - 1, a.len() - 1);
    let m = n.max(d);
    let transform = |c: &[f64], degree: usize| -> Vec<f64> {
        (0..=m)
            .map(|j| {
                let mut val = 0.0;
                for i in 0..=degree {
                    for k in 0..=i {
                        if j >= k && j - k <= m - i {
                            let sign = if k % 2 == 0 { 1.0 } else { -1.0 };
                            val += binomial(i, k) * binomial(m - i, j - k) * c[degree - i] * (2.0 * fs).powi(i as i32) * sign;
                        }
                    }
                }
                val
            })
            .collect()
    };
    super::normalize(&transform(b, n), &transform(a, d))
}
