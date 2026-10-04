//! Frequency responses (`freqz`, `sosfreqz`, `freqs`, group delay) and time responses (impulse,
//! step, simulation).

use super::{Domain, Method, StateSpace, SystemError, TransferFunction, C64};
use crate::linalg::{expm, matmul, polyval_complex};
use crate::signal::NdArray;

/// `n` evenly spaced frequencies in Hz from 0 up to (not including) the Nyquist frequency, or the
/// whole circle up to `fs` when `whole` (the default grid of `scipy.signal.freqz`).
pub fn freq_grid(n: usize, whole: bool, fs: f64) -> Vec<f64> {
    let top = if whole { fs } else { fs / 2.0 };
    (0..n).map(|i| top * i as f64 / n as f64).collect()
}

/// `e^(-iω)` for a frequency in Hz at sample rate `fs`.
fn unit(f: f64, fs: f64) -> C64 {
    C64::cis(-std::f64::consts::TAU * f / fs)
}

/// `Σ c[k] z^k` for `z = e^(-iω)`: a polynomial in `z⁻¹`, lowest power first.
fn eval_inverse(c: &[f64], z: C64) -> C64 {
    c.iter().rev().fold(C64::zero(), |acc, &x| acc * z + C64::new(x, 0.0))
}

/// The frequency response of a digital filter `b(z⁻¹) / a(z⁻¹)` at frequencies in Hz
/// (`scipy.signal.freqz`).
pub fn freqz(b: &[f64], a: &[f64], freqs: &[f64], fs: f64) -> Vec<C64> {
    freqs.iter().map(|&f| eval_inverse(b, unit(f, fs)) / eval_inverse(a, unit(f, fs))).collect()
}

/// The frequency response of second-order sections in series (`scipy.signal.sosfreqz`).
pub fn sosfreqz(sos: &[[f64; 6]], freqs: &[f64], fs: f64) -> Vec<C64> {
    freqs
        .iter()
        .map(|&f| {
            let z = unit(f, fs);
            sos.iter().fold(C64::one(), |h, s| h * eval_inverse(&s[..3], z) / eval_inverse(&s[3..], z))
        })
        .collect()
}

/// The frequency response of a digital zeros-poles-gain system (`scipy.signal.freqz_zpk`).
pub fn freqz_zpk(zeros: &[C64], poles: &[C64], gain: f64, freqs: &[f64], fs: f64) -> Vec<C64> {
    freqs
        .iter()
        .map(|&f| {
            let z = unit(f, fs).conj(); // e^(iω)
            let num = zeros.iter().fold(C64::new(gain, 0.0), |h, &q| h * (z - q));
            poles.iter().fold(num, |h, &p| h / (z - p))
        })
        .collect()
}

/// The frequency response of an analog system `b(s) / a(s)` at angular frequencies in rad/s
/// (`scipy.signal.freqs`).
pub fn freqs(b: &[f64], a: &[f64], w: &[f64]) -> Vec<C64> {
    w.iter().map(|&w| polyval_complex(b, C64::new(0.0, w)) / polyval_complex(a, C64::new(0.0, w))).collect()
}

/// The frequency response of an analog zeros-poles-gain system (`scipy.signal.freqs_zpk`).
pub fn freqs_zpk(zeros: &[C64], poles: &[C64], gain: f64, w: &[f64]) -> Vec<C64> {
    w.iter()
        .map(|&w| {
            let s = C64::new(0.0, w);
            let num = zeros.iter().fold(C64::new(gain, 0.0), |h, &q| h * (s - q));
            poles.iter().fold(num, |h, &p| h / (s - p))
        })
        .collect()
}

/// The group delay in samples of a digital filter at frequencies in Hz
/// (`scipy.signal.group_delay`); 0 where the response is (numerically) zero.
pub fn group_delay(b: &[f64], a: &[f64], freqs: &[f64], fs: f64) -> Vec<f64> {
    // c = b * reversed(a); delay = Re(Σ k c_k z^-k / Σ c_k z^-k) - (len(a) - 1)
    let reversed: Vec<f64> = a.iter().rev().copied().collect();
    let c = crate::linalg::polymul(b, &reversed);
    let cr: Vec<f64> = c.iter().enumerate().map(|(k, x)| k as f64 * x).collect();
    freqs
        .iter()
        .map(|&f| {
            let z = unit(f, fs);
            let den = eval_inverse(&c, z);
            if den.norm() < 10.0 * f64::EPSILON {
                0.0
            } else {
                (eval_inverse(&cr, z) / den).re - (a.len() as f64 - 1.0)
            }
        })
        .collect()
}

impl TransferFunction {
    /// The frequency response at frequencies in Hz (digital) or rad/s (analog).
    pub fn frequency_response(&self, freqs: &[f64]) -> Vec<C64> {
        match self.domain {
            Domain::Continuous => self::freqs(&self.num, &self.den, freqs),
            Domain::Discrete { dt } => {
                // positive powers of z: align both polynomials at their highest power
                let n = self.num.len().max(self.den.len());
                let pad = |p: &[f64]| [vec![0.0; n - p.len()], p.to_vec()].concat();
                freqz(&pad(&self.num), &pad(&self.den), freqs, 1.0 / dt)
            }
        }
    }

    /// The impulse response: `n` samples (discrete), or at `n` times `dt` apart from 0
    /// (continuous, exact for the states; the direct term's impulse is left out, as in SciPy).
    pub fn impulse(&self, n: usize, dt: Option<f64>) -> Result<Vec<f64>, SystemError> {
        self.to_ss()?.impulse(n, dt).map(|y| y.as_slice().to_vec())
    }

    /// The unit step response: `n` samples (discrete), or at `n` times `dt` apart from 0
    /// (continuous, exact: a constant input is what zero-order hold assumes).
    pub fn step(&self, n: usize, dt: Option<f64>) -> Result<Vec<f64>, SystemError> {
        self.to_ss()?.step(n, dt).map(|y| y.as_slice().to_vec())
    }
}

impl StateSpace {
    /// Runs a discrete system on inputs `u` (`[steps, inputs]`, or `[steps]` for one input) from
    /// state `x0` (zero if `None`): returns the outputs `[steps, outputs]` and the states
    /// `[steps, states]` (`scipy.signal.dlsim`). A continuous system must be discretized first.
    pub fn simulate(&self, u: &NdArray<f64>, x0: Option<&[f64]>) -> Result<(NdArray<f64>, NdArray<f64>), SystemError> {
        if !self.domain.is_discrete() {
            return Err(SystemError::invalid("simulate needs a discrete system; use lsim or discretize"));
        }
        let (n, inputs, outputs) = (self.order(), self.b.shape()[1], self.c.shape()[0]);
        let steps = u.shape().first().copied().unwrap_or(0);
        let width = if u.ndim() == 1 { 1 } else { u.shape()[1] };
        if width != inputs || u.ndim() > 2 {
            return Err(SystemError::invalid(format!("inputs should be [steps, {inputs}], got {:?}", u.shape())));
        }
        let mut x = match x0 {
            Some(x0) if x0.len() == n => x0.to_vec(),
            Some(x0) => return Err(SystemError::invalid(format!("x0 has {} values for {n} states", x0.len()))),
            None => vec![0.0; n],
        };
        let (a, b, c, d) = (self.a.as_slice(), self.b.as_slice(), self.c.as_slice(), self.d.as_slice());
        let mut ys = Vec::with_capacity(steps * outputs);
        let mut xs = Vec::with_capacity(steps * n);
        let mut next = vec![0.0; n];
        for k in 0..steps {
            let uk = &u.as_slice()[k * inputs..(k + 1) * inputs];
            xs.extend_from_slice(&x);
            for o in 0..outputs {
                let cx: f64 = (0..n).map(|j| c[o * n + j] * x[j]).sum();
                let du: f64 = (0..inputs).map(|j| d[o * inputs + j] * uk[j]).sum();
                ys.push(cx + du);
            }
            for (i, xi) in next.iter_mut().enumerate() {
                *xi = (0..n).map(|j| a[i * n + j] * x[j]).sum::<f64>() + (0..inputs).map(|j| b[i * inputs + j] * uk[j]).sum::<f64>();
            }
            std::mem::swap(&mut x, &mut next);
        }
        Ok((NdArray::from_vec(ys, &[steps, outputs]).expect("shape"), NdArray::from_vec(xs, &[steps, n]).expect("shape")))
    }

    /// Simulates a continuous system on inputs `u` sampled every `dt` seconds, holding each input
    /// constant over its interval (exact for piecewise-constant inputs; `scipy.signal.lsim` with
    /// zero-order hold): returns the outputs and states at the sample times.
    pub fn lsim(&self, u: &NdArray<f64>, dt: f64, x0: Option<&[f64]>) -> Result<(NdArray<f64>, NdArray<f64>), SystemError> {
        if self.domain.is_discrete() {
            return Err(SystemError::invalid("lsim needs a continuous system; use simulate"));
        }
        self.discretize(dt, Method::Zoh)?.simulate(u, x0)
    }

    /// The response to a unit impulse on the first input (`[n, outputs]`): discrete, or continuous
    /// sampled every `dt` (`C e^(A t) B`, the direct term's impulse left out).
    pub fn impulse(&self, n: usize, dt: Option<f64>) -> Result<NdArray<f64>, SystemError> {
        let (states, inputs) = (self.order(), self.b.shape()[1]);
        match self.domain {
            Domain::Discrete { .. } => {
                let u = NdArray::from_fn(&[n, inputs], |i| if i[0] == 0 && i[1] == 0 { 1.0 } else { 0.0 }).expect("shape");
                Ok(self.simulate(&u, None)?.0)
            }
            Domain::Continuous => {
                let dt = dt.ok_or_else(|| SystemError::invalid("a continuous impulse response needs a time step"))?;
                let phi = expm(self.a.map(|&x| x * dt).view())?;
                let mut x = NdArray::from_fn(&[states, 1], |i| self.b.as_slice()[i[0] * inputs]).expect("shape");
                let outputs = self.c.shape()[0];
                let mut ys = Vec::with_capacity(n * outputs);
                for _ in 0..n {
                    ys.extend_from_slice(matmul(self.c.view(), x.view())?.as_slice());
                    x = matmul(phi.view(), x.view())?;
                }
                Ok(NdArray::from_vec(ys, &[n, outputs]).expect("shape"))
            }
        }
    }

    /// The response to a unit step on the first input (`[n, outputs]`): discrete, or continuous
    /// sampled every `dt`.
    pub fn step(&self, n: usize, dt: Option<f64>) -> Result<NdArray<f64>, SystemError> {
        let inputs = self.b.shape()[1];
        let u = NdArray::from_fn(&[n, inputs], |i| if i[1] == 0 { 1.0 } else { 0.0 }).expect("shape");
        match self.domain {
            Domain::Discrete { .. } => Ok(self.simulate(&u, None)?.0),
            Domain::Continuous => {
                let dt = dt.ok_or_else(|| SystemError::invalid("a continuous step response needs a time step"))?;
                Ok(self.lsim(&u, dt, None)?.0)
            }
        }
    }
}
