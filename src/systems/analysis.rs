//! Analysis and design on state-space systems and transfer functions, as `python-control` has
//! them: controllability and observability ([`StateSpace::ctrb`], [`StateSpace::obsv`] and their
//! ranks), Gramians, DC gain, natural frequencies and damping of the poles ([`damp`]), linear
//! quadratic regulators ([`lqr`], [`dlqr`]), and gain and phase margins ([`stability_margins`]).

use super::{Domain, StateSpace, SystemError, TransferFunction, C64};
use crate::linalg::{self, matrix_rank, polyadd, polymul, roots, solve_continuous_are, solve_continuous_lyapunov, solve_discrete_are, solve_discrete_lyapunov};
use crate::signal::NdArray;

fn mm(a: &NdArray<f64>, b: &NdArray<f64>) -> Result<NdArray<f64>, SystemError> {
    Ok(linalg::matmul(a.view(), b.view())?)
}

fn transposed(a: &NdArray<f64>) -> NdArray<f64> {
    a.view().transpose().to_owned()
}

impl StateSpace {
    /// The controllability matrix `[B, AB, A²B, ..., Aⁿ⁻¹B]` (`control.ctrb`): n x (n inputs).
    pub fn ctrb(&self) -> Result<NdArray<f64>, SystemError> {
        let n = self.order();
        let m = self.b.shape()[1];
        let mut block = self.b.clone();
        let mut out = NdArray::<f64>::zeros(&[n, n * m])?;
        for k in 0..n {
            for i in 0..n {
                for j in 0..m {
                    out.as_mut_slice()[i * n * m + k * m + j] = block.as_slice()[i * m + j];
                }
            }
            block = mm(&self.a, &block)?;
        }
        Ok(out)
    }

    /// The observability matrix `[C; CA; CA²; ...; CAⁿ⁻¹]` (`control.obsv`): (n outputs) x n.
    pub fn obsv(&self) -> Result<NdArray<f64>, SystemError> {
        let n = self.order();
        let p = self.c.shape()[0];
        let mut block = self.c.clone();
        let mut data = Vec::with_capacity(n * p * n);
        for _ in 0..n {
            data.extend_from_slice(block.as_slice());
            block = mm(&block, &self.a)?;
        }
        Ok(NdArray::from_vec(data, &[n * p, n])?)
    }

    /// Whether every state can be steered by the inputs: [`ctrb`](Self::ctrb) has full rank.
    pub fn is_controllable(&self) -> Result<bool, SystemError> {
        Ok(matrix_rank(self.ctrb()?.view())? == self.order())
    }

    /// Whether every state can be told from the outputs: [`obsv`](Self::obsv) has full rank.
    pub fn is_observable(&self) -> Result<bool, SystemError> {
        Ok(matrix_rank(self.obsv()?.view())? == self.order())
    }

    /// The controllability Gramian `Wc` (`control.gram(sys, "c")`): `A Wc + Wc Aᵀ + B Bᵀ = 0`
    /// (continuous) or `A Wc Aᵀ - Wc + B Bᵀ = 0` (discrete). Needs a stable system.
    pub fn controllability_gramian(&self) -> Result<NdArray<f64>, SystemError> {
        self.require_stable("the controllability Gramian")?;
        let bbt = mm(&self.b, &transposed(&self.b))?;
        Ok(match self.domain {
            Domain::Continuous => solve_continuous_lyapunov(self.a.view(), bbt.map(|v| -v).view())?,
            Domain::Discrete { .. } => solve_discrete_lyapunov(self.a.view(), bbt.view())?,
        })
    }

    /// The observability Gramian `Wo` (`control.gram(sys, "o")`): `Aᵀ Wo + Wo A + Cᵀ C = 0`
    /// (continuous) or `Aᵀ Wo A - Wo + Cᵀ C = 0` (discrete). Needs a stable system.
    pub fn observability_gramian(&self) -> Result<NdArray<f64>, SystemError> {
        self.require_stable("the observability Gramian")?;
        let ctc = mm(&transposed(&self.c), &self.c)?;
        let at = transposed(&self.a);
        Ok(match self.domain {
            Domain::Continuous => solve_continuous_lyapunov(at.view(), ctc.map(|v| -v).view())?,
            Domain::Discrete { .. } => solve_discrete_lyapunov(at.view(), ctc.view())?,
        })
    }

    fn require_stable(&self, what: &str) -> Result<(), SystemError> {
        if !self.is_stable()? {
            return Err(SystemError::invalid(format!("{what} needs a stable system")));
        }
        Ok(())
    }

    /// The steady-state gain (`control.dcgain`), outputs x inputs: `D - C A⁻¹ B` (continuous) or
    /// `D + C (I - A)⁻¹ B` (discrete). Errors when a pole sits at `s = 0` (`z = 1`).
    pub fn dcgain(&self) -> Result<NdArray<f64>, SystemError> {
        let n = self.order();
        let m = match self.domain {
            Domain::Continuous => self.a.map(|v| -v),
            Domain::Discrete { .. } => NdArray::from_fn(&[n, n], |i| if i[0] == i[1] { 1.0 } else { 0.0 } - self.a.as_slice()[i[0] * n + i[1]])?,
        };
        let x = linalg::solve(m.view(), self.b.view()).map_err(|_| SystemError::invalid("a pole at s = 0 (z = 1): the DC gain is infinite"))?;
        let cx = mm(&self.c, &x)?;
        Ok(NdArray::from_vec(cx.as_slice().iter().zip(self.d.as_slice()).map(|(a, b)| a + b).collect(), self.d.shape())?)
    }
}

/// The poles of a system with their natural frequencies (rad/s) and damping ratios
/// (`control.damp`): `wn = |p|`, `zeta = -Re(p) / |p|` for continuous poles; discrete poles are
/// first mapped to `s = ln(z) / dt`.
pub fn damp(poles: &[C64], domain: Domain) -> Vec<(C64, f64, f64)> {
    poles
        .iter()
        .map(|&p| {
            let s = match domain {
                Domain::Continuous => p,
                Domain::Discrete { dt } => C64::new(p.norm().ln(), p.im.atan2(p.re)) / dt,
            };
            let wn = s.norm();
            (p, wn, if wn == 0.0 { -1.0 } else { -s.re / wn })
        })
        .collect()
}

/// A linear quadratic regulator: the state feedback `u = -K x` minimizing the integral of
/// `xᵀ Q x + uᵀ R u`, the Riccati solution `S`, and the closed-loop poles.
#[derive(Clone, Debug, PartialEq)]
pub struct Lqr {
    /// The gain `K` (inputs x n).
    pub k: NdArray<f64>,
    /// The solution of the Riccati equation.
    pub s: NdArray<f64>,
    /// The eigenvalues of `A - B K`.
    pub poles: Vec<C64>,
}

fn closed_loop(a: &NdArray<f64>, b: &NdArray<f64>, k: &NdArray<f64>) -> Result<Vec<C64>, SystemError> {
    let bk = mm(b, k)?;
    let acl = NdArray::from_vec(a.as_slice().iter().zip(bk.as_slice()).map(|(x, y)| x - y).collect(), a.shape())?;
    Ok(linalg::eigvals(acl.view())?)
}

/// The continuous-time LQR for `x' = A x + B u` (`control.lqr`): `K = R⁻¹ Bᵀ S` with `S` from
/// [`solve_continuous_are`].
pub fn lqr(a: &NdArray<f64>, b: &NdArray<f64>, q: &NdArray<f64>, r: &NdArray<f64>) -> Result<Lqr, SystemError> {
    let s = solve_continuous_are(a.view(), b.view(), q.view(), r.view())?;
    let k = linalg::solve(r.view(), mm(&transposed(b), &s)?.view())?;
    let poles = closed_loop(a, b, &k)?;
    Ok(Lqr { k, s, poles })
}

/// The discrete-time LQR for `x[k+1] = A x[k] + B u[k]` (`control.dlqr`):
/// `K = (R + Bᵀ S B)⁻¹ Bᵀ S A` with `S` from [`solve_discrete_are`].
pub fn dlqr(a: &NdArray<f64>, b: &NdArray<f64>, q: &NdArray<f64>, r: &NdArray<f64>) -> Result<Lqr, SystemError> {
    let s = solve_discrete_are(a.view(), b.view(), q.view(), r.view())?;
    let bt = transposed(b);
    let lhs = mm(&mm(&bt, &s)?, b)?;
    let lhs = NdArray::from_vec(lhs.as_slice().iter().zip(r.as_slice()).map(|(x, y)| x + y).collect(), r.shape())?;
    let k = linalg::solve(lhs.view(), mm(&mm(&bt, &s)?, a)?.view())?;
    let poles = closed_loop(a, b, &k)?;
    Ok(Lqr { k, s, poles })
}

/// Gain and phase margins of an open loop (`control.stability_margins`).
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Margins {
    /// How much the loop gain can grow before the closed loop goes unstable: `1 / |L|` where the
    /// phase crosses -180° (infinite if it never does).
    pub gain_margin: f64,
    /// How much phase lag can be added: `180° + arg L` where `|L| = 1`, in degrees (infinite if
    /// the gain never crosses 1).
    pub phase_margin: f64,
    /// The frequency (rad/s) of the phase crossover the gain margin is measured at (NaN if none).
    pub phase_crossover: f64,
    /// The frequency (rad/s) of the gain crossover the phase margin is measured at (NaN if none).
    pub gain_crossover: f64,
}

/// `p(j w)` split into real and imaginary polynomials in `w` (highest power first).
fn on_imaginary_axis(p: &[f64]) -> (Vec<f64>, Vec<f64>) {
    let deg = p.len().saturating_sub(1);
    let (mut re, mut im) = (vec![0.0; p.len()], vec![0.0; p.len()]);
    for (i, &c) in p.iter().enumerate() {
        let k = deg - i; // the power of s
        // (j w)^k = j^k w^k
        match k % 4 {
            0 => re[i] = c,
            1 => im[i] = c,
            2 => re[i] = -c,
            _ => im[i] = -c,
        }
    }
    (re, im)
}

fn sub(a: &[f64], b: &[f64]) -> Vec<f64> {
    polyadd(a, &b.iter().map(|v| -v).collect::<Vec<_>>())
}

/// The roots of `p` that are real and positive (frequencies), or on the unit circle (as angles in
/// (0, π]) when `unit_circle`.
fn crossings(p: &[f64], unit_circle: bool) -> Result<Vec<f64>, SystemError> {
    if p.iter().all(|&c| c == 0.0) {
        return Ok(Vec::new());
    }
    let tol = 1e-8;
    Ok(roots(p)?
        .into_iter()
        .filter_map(|r| {
            if unit_circle {
                ((r.norm() - 1.0).abs() < tol && r.im >= -tol).then(|| r.im.atan2(r.re).abs()).filter(|&w| w > 0.0)
            } else {
                (r.im.abs() < tol * (1.0 + r.re.abs()) && r.re > 0.0).then_some(r.re)
            }
        })
        .collect())
}

/// The gain and phase margins of the open-loop transfer function `l` (`control.stability_margins`
/// for SISO systems), from the exact crossover frequencies: the positive real roots of
/// `|num(jw)|² - |den(jw)|²` (gain crossovers) and of `Im[num(jw) conj(den(jw))]` with the real
/// part negative (phase crossovers); for a discrete system, the roots on the unit circle of the
/// same equations in `z`, frequencies in rad/s through `dt`. Where there are several crossings, the
/// smallest margin is reported.
pub fn stability_margins(l: &TransferFunction) -> Result<Margins, SystemError> {
    let eval = |w: f64| -> C64 {
        match l.domain {
            Domain::Continuous => crate::linalg::polyval_complex(&l.num, C64::new(0.0, w)) / crate::linalg::polyval_complex(&l.den, C64::new(0.0, w)),
            Domain::Discrete { dt } => {
                let z = C64::cis(w * dt);
                crate::linalg::polyval_complex(&l.num, z) / crate::linalg::polyval_complex(&l.den, z)
            }
        }
    };
    let (gain_eq, phase_eq, unit, scale) = match l.domain {
        Domain::Continuous => {
            let (nr, ni) = on_imaginary_axis(&l.num);
            let (dr, di) = on_imaginary_axis(&l.den);
            let gain = sub(&polyadd(&polymul(&nr, &nr), &polymul(&ni, &ni)), &polyadd(&polymul(&dr, &dr), &polymul(&di, &di)));
            let phase = sub(&polymul(&ni, &dr), &polymul(&nr, &di));
            (gain, phase, false, 1.0)
        }
        Domain::Discrete { dt } => {
            // on |z| = 1, conj(p(z)) = p(1/z); multiplied through by z^d with d the common degree
            let d = l.num.len().max(l.den.len());
            let pad = |p: &[f64]| [vec![0.0; d - p.len()], p.to_vec()].concat();
            let (n, den) = (pad(&l.num), pad(&l.den));
            let rev = |p: &[f64]| p.iter().rev().copied().collect::<Vec<_>>();
            let gain = sub(&polymul(&n, &rev(&n)), &polymul(&den, &rev(&den)));
            let phase = sub(&polymul(&n, &rev(&den)), &polymul(&rev(&n), &den));
            (gain, phase, true, 1.0 / dt)
        }
    };
    // polynomial roots carry the rounding of the companion matrix's eigenvalues (a few digits on the
    // unit circle): polish each crossing with secant steps on the response itself
    let polish = |w0: f64, f: &dyn Fn(f64) -> f64| -> f64 {
        let (mut a, mut b) = (w0, w0 * (1.0 + 1e-7));
        let (mut fa, mut fb) = (f(a), f(b));
        for _ in 0..8 {
            if fb == fa || fb == 0.0 {
                break;
            }
            let c = b - fb * (b - a) / (fb - fa);
            if !c.is_finite() || (c - w0).abs() > 1e-3 * w0 {
                break;
            }
            (a, fa, b, fb) = (b, fb, c, f(c));
        }
        if fb.abs() <= f(w0).abs() { b } else { w0 }
    };
    let gain_error = |w: f64| eval(w).norm().ln();
    let phase_error = |w: f64| {
        let h = eval(w);
        h.im / h.norm()
    };
    let mut margins = Margins { gain_margin: f64::INFINITY, phase_margin: f64::INFINITY, phase_crossover: f64::NAN, gain_crossover: f64::NAN };
    for w in crossings(&phase_eq, unit)? {
        let w = polish(w * scale, &phase_error);
        let h = eval(w);
        if h.re < 0.0 && h.norm() > 0.0 {
            let gm = 1.0 / h.norm();
            if gm < margins.gain_margin {
                (margins.gain_margin, margins.phase_crossover) = (gm, w);
            }
        }
    }
    for w in crossings(&gain_eq, unit)? {
        let w = polish(w * scale, &gain_error);
        let phase = eval(w).arg().to_degrees();
        // 180° + the phase, wrapped into (-180, 180]
        let mut pm = (180.0 + phase).rem_euclid(360.0);
        if pm > 180.0 {
            pm -= 360.0;
        }
        if pm.abs() < margins.phase_margin.abs() {
            (margins.phase_margin, margins.gain_crossover) = (pm, w);
        }
    }
    Ok(margins)
}
