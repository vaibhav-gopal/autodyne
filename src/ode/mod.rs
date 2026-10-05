//! Initial value problems `y' = f(t, y)`, as `scipy.integrate.solve_ivp` solves them: adaptive
//! explicit Runge-Kutta pairs ([`OdeMethod::Rk45`], Dormand-Prince 5(4), the default;
//! [`OdeMethod::Rk23`], Bogacki-Shampine 3(2)) and, for stiff problems, an L-stable Rosenbrock
//! method ([`OdeMethod::Rosenbrock23`], MATLAB's `ode23s`). Steps are chosen as SciPy chooses them
//! (its initial step, error norm and step factors), outputs can be interpolated at requested
//! times from each step's dense output, and events (zero crossings of functions of `(t, y)`) are
//! located on it, optionally stopping the integration.
//!
//! ```
//! use autodyne::ode::{solve_ivp, OdeOptions};
//!
//! // y' = -y from y(0) = 1: y(1) = e^-1
//! let sol = solve_ivp(|_t, y, dy| dy[0] = -y[0], (0.0, 1.0), &[1.0], OdeOptions { rtol: 1e-10, atol: 1e-12, ..Default::default() }).unwrap();
//! assert!((sol.y.last().unwrap()[0] - (-1.0f64).exp()).abs() < 1e-9);
//! ```
//!
//! tend: Numerics / ode

use crate::alloc_prelude::*;
use thiserror::Error;

/// Errors from solving initial value problems.
#[derive(Debug, Clone, PartialEq, Eq, Error)]
pub enum OdeError {
    /// An argument is out of range (a negative tolerance, an empty state).
    #[error("invalid problem: {0}")]
    Invalid(String),
}

impl OdeError {
    pub(crate) fn invalid(message: impl Into<String>) -> Self {
        OdeError::Invalid(message.into())
    }
}

/// The integration method.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub enum OdeMethod {
    /// Dormand-Prince 5(4): explicit, fifth order with a fourth-order error estimate and dense
    /// output (SciPy's default `RK45`). For non-stiff problems.
    #[default]
    Rk45,
    /// Bogacki-Shampine 3(2) (`RK23`): cheaper steps for loose tolerances.
    Rk23,
    /// The Rosenbrock method of MATLAB's `ode23s`: linearly implicit, L-stable, second order with
    /// a third-order error estimate. For stiff problems; the Jacobian is taken by finite
    /// differences.
    Rosenbrock23,
}

/// An event function: `g(t, y)`, whose sign changes mark the events.
pub type EventFn<'a> = Box<dyn FnMut(f64, &[f64]) -> f64 + 'a>;

/// A function of `(t, y)` whose zero crossings [`solve_ivp`] locates (`events` in SciPy).
pub struct Event<'a> {
    /// The event function.
    pub g: EventFn<'a>,
    /// Stop the integration at the first occurrence.
    pub terminal: bool,
    /// Only crossings going up (`> 0`), going down (`< 0`), or either (`0`).
    pub direction: f64,
}

/// Settings for [`solve_ivp`].
pub struct OdeOptions<'a> {
    /// The method.
    pub method: OdeMethod,
    /// Relative tolerance (SciPy's default 1e-3).
    pub rtol: f64,
    /// Absolute tolerance (SciPy's default 1e-6).
    pub atol: f64,
    /// The first step's size (`None`: chosen from the problem, as SciPy does).
    pub first_step: Option<f64>,
    /// The largest step allowed.
    pub max_step: f64,
    /// Times to report the solution at, inside the span in the direction of integration (`None`:
    /// every step's end).
    pub t_eval: Option<Vec<f64>>,
    /// Events to locate.
    pub events: Vec<Event<'a>>,
}

impl Default for OdeOptions<'_> {
    fn default() -> Self {
        Self { method: OdeMethod::Rk45, rtol: 1e-3, atol: 1e-6, first_step: None, max_step: f64::INFINITY, t_eval: None, events: Vec::new() }
    }
}

/// Why the integration stopped.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum OdeStatus {
    /// It reached the end of the span.
    Finished,
    /// A terminal event occurred.
    Event,
    /// The step size fell below what the arithmetic can resolve.
    StepTooSmall,
}

/// The result of [`solve_ivp`].
#[derive(Clone, Debug, PartialEq)]
pub struct OdeSolution {
    /// The output times.
    pub t: Vec<f64>,
    /// The state at each output time.
    pub y: Vec<Vec<f64>>,
    /// For each event, the times it occurred.
    pub t_events: Vec<Vec<f64>>,
    /// For each event, the states where it occurred.
    pub y_events: Vec<Vec<Vec<f64>>>,
    /// Evaluations of `f` (finite-difference Jacobians included).
    pub nfev: usize,
    /// Jacobian evaluations.
    pub njev: usize,
    /// LU decompositions.
    pub nlu: usize,
    /// How it ended.
    pub status: OdeStatus,
}

/// SciPy's step controller.
const SAFETY: f64 = 0.9;
const MIN_FACTOR: f64 = 0.2;
const MAX_FACTOR: f64 = 10.0;

/// The root mean square.
fn rms(v: impl Iterator<Item = f64>) -> f64 {
    let (mut s, mut n) = (0.0, 0);
    for x in v {
        s += x * x;
        n += 1;
    }
    (s / n.max(1) as f64).sqrt()
}

/// A Runge-Kutta tableau with an error estimator and a dense-output polynomial (SciPy's `rk.py`).
struct Tableau {
    c: &'static [f64],
    a: &'static [&'static [f64]],
    b: &'static [f64],
    /// error weights over the stages and the FSAL stage
    e: &'static [f64],
    /// dense output: stage j's contribution is `sum_k p[j][k] x^(k + 1)`
    p: &'static [&'static [f64]],
    error_order: usize,
}

const RK45: Tableau = Tableau {
    c: &[0.0, 1.0 / 5.0, 3.0 / 10.0, 4.0 / 5.0, 8.0 / 9.0, 1.0],
    a: &[
        &[],
        &[1.0 / 5.0],
        &[3.0 / 40.0, 9.0 / 40.0],
        &[44.0 / 45.0, -56.0 / 15.0, 32.0 / 9.0],
        &[19372.0 / 6561.0, -25360.0 / 2187.0, 64448.0 / 6561.0, -212.0 / 729.0],
        &[9017.0 / 3168.0, -355.0 / 33.0, 46732.0 / 5247.0, 49.0 / 176.0, -5103.0 / 18656.0],
    ],
    b: &[35.0 / 384.0, 0.0, 500.0 / 1113.0, 125.0 / 192.0, -2187.0 / 6784.0, 11.0 / 84.0],
    e: &[-71.0 / 57600.0, 0.0, 71.0 / 16695.0, -71.0 / 1920.0, 17253.0 / 339200.0, -22.0 / 525.0, 1.0 / 40.0],
    p: &[
        &[1.0, -8048581381.0 / 2820520608.0, 8663915743.0 / 2820520608.0, -12715105075.0 / 11282082432.0],
        &[0.0, 0.0, 0.0, 0.0],
        &[0.0, 131558114200.0 / 32700410799.0, -68118460800.0 / 10900136933.0, 87487479700.0 / 32700410799.0],
        &[0.0, -1754552775.0 / 470086768.0, 14199869525.0 / 1410260304.0, -10690763975.0 / 1880347072.0],
        &[0.0, 127303824393.0 / 49829197408.0, -318862633887.0 / 49829197408.0, 701980252875.0 / 199316789632.0],
        &[0.0, -282668133.0 / 205662961.0, 2019193451.0 / 616988883.0, -1453857185.0 / 822651844.0],
        &[0.0, 40617522.0 / 29380423.0, -110615467.0 / 29380423.0, 69997945.0 / 29380423.0],
    ],
    error_order: 4,
};

const RK23: Tableau = Tableau {
    c: &[0.0, 1.0 / 2.0, 3.0 / 4.0],
    a: &[&[], &[1.0 / 2.0], &[0.0, 3.0 / 4.0]],
    b: &[2.0 / 9.0, 1.0 / 3.0, 4.0 / 9.0],
    e: &[5.0 / 72.0, -1.0 / 12.0, -1.0 / 9.0, 1.0 / 8.0],
    p: &[&[1.0, -4.0 / 3.0, 5.0 / 9.0], &[0.0, 1.0, -2.0 / 3.0], &[0.0, 4.0 / 3.0, -8.0 / 9.0], &[0.0, -1.0, 1.0]],
    error_order: 2,
};

/// One step's dense output: `y(t_old + x h)` for `x` in [0, 1].
enum Dense {
    /// `y_old + h sum_j q[j] x^(j + 1)` (the Runge-Kutta polynomials, coefficients per state)
    Poly { t_old: f64, h: f64, y_old: Vec<f64>, q: Vec<Vec<f64>> },
    /// cubic Hermite from the end values and slopes
    Hermite { t_old: f64, h: f64, y0: Vec<f64>, y1: Vec<f64>, f0: Vec<f64>, f1: Vec<f64> },
}

impl Dense {
    fn eval(&self, t: f64) -> Vec<f64> {
        match self {
            Dense::Poly { t_old, h, y_old, q } => {
                let x = (t - t_old) / h;
                y_old
                    .iter()
                    .enumerate()
                    .map(|(i, &y)| {
                        let mut acc = 0.0;
                        let mut xp = x;
                        for qk in q {
                            acc += qk[i] * xp;
                            xp *= x;
                        }
                        y + h * acc
                    })
                    .collect()
            }
            Dense::Hermite { t_old, h, y0, y1, f0, f1 } => {
                let x = (t - t_old) / h;
                let (h00, h10, h01, h11) = (2.0 * x * x * x - 3.0 * x * x + 1.0, x * x * x - 2.0 * x * x + x, -2.0 * x * x * x + 3.0 * x * x, x * x * x - x * x);
                (0..y0.len()).map(|i| h00 * y0[i] + h10 * h * f0[i] + h01 * y1[i] + h11 * h * f1[i]).collect()
            }
        }
    }
}

/// Solves `m x = b` in place by Gaussian elimination with partial pivoting (`m` n x n row-major,
/// factored once per Rosenbrock step and reused for its three solves).
struct Lu {
    n: usize,
    lu: Vec<f64>,
    pivots: Vec<usize>,
}

impl Lu {
    fn new(mut m: Vec<f64>, n: usize) -> Option<Self> {
        let mut pivots = vec![0; n];
        for k in 0..n {
            let p = (k..n).max_by(|&a, &b| m[a * n + k].abs().total_cmp(&m[b * n + k].abs()))?;
            if m[p * n + k] == 0.0 {
                return None;
            }
            pivots[k] = p;
            if p != k {
                for j in 0..n {
                    m.swap(k * n + j, p * n + j);
                }
            }
            for i in k + 1..n {
                m[i * n + k] /= m[k * n + k];
                let f = m[i * n + k];
                for j in k + 1..n {
                    m[i * n + j] -= f * m[k * n + j];
                }
            }
        }
        Some(Self { n, lu: m, pivots })
    }

    fn solve(&self, b: &mut [f64]) {
        let n = self.n;
        for k in 0..n {
            b.swap(k, self.pivots[k]);
        }
        for i in 0..n {
            let s: f64 = (0..i).map(|j| self.lu[i * n + j] * b[j]).sum();
            b[i] -= s;
        }
        for i in (0..n).rev() {
            let s: f64 = (i + 1..n).map(|j| self.lu[i * n + j] * b[j]).sum();
            b[i] = (b[i] - s) / self.lu[i * n + i];
        }
    }
}

/// Solves `y' = f(t, y)`, `y(t0) = y0` over `t_span = (t0, t1)` (`scipy.integrate.solve_ivp`); `f`
/// writes the derivative into its third argument. Integrates backwards when `t1 < t0`.
///
/// Errors on non-positive tolerances, an empty state, or output times outside the span or out of
/// order.
pub fn solve_ivp<'a, F: FnMut(f64, &[f64], &mut [f64])>(mut f: F, t_span: (f64, f64), y0: &[f64], mut options: OdeOptions<'a>) -> Result<OdeSolution, OdeError> {
    let (t0, t1) = t_span;
    let n = y0.len();
    if n == 0 {
        return Err(OdeError::invalid("the state is empty"));
    }
    let (rtol, atol) = (options.rtol, options.atol);
    if !(rtol > 0.0 && atol >= 0.0) {
        return Err(OdeError::invalid(format!("tolerances must be positive (rtol {rtol}, atol {atol})")));
    }
    if !(t0.is_finite() && t1.is_finite()) || options.max_step.is_nan() || options.max_step <= 0.0 {
        return Err(OdeError::invalid("the span must be finite and max_step positive"));
    }
    let dir = if t1 >= t0 { 1.0 } else { -1.0 };
    if let Some(te) = &options.t_eval {
        let inside = te.iter().all(|&t| (t - t0) * dir >= 0.0 && (t1 - t) * dir >= 0.0);
        let ordered = te.windows(2).all(|w| (w[1] - w[0]) * dir >= 0.0);
        if !inside || !ordered {
            return Err(OdeError::invalid("t_eval must lie in the span, ordered in the direction of integration"));
        }
    }
    let mut nfev = 0;
    let mut eval = |t: f64, y: &[f64], out: &mut [f64], nfev: &mut usize| {
        *nfev += 1;
        f(t, y, out);
    };
    let tableau = match options.method {
        OdeMethod::Rk45 => Some(&RK45),
        OdeMethod::Rk23 => Some(&RK23),
        OdeMethod::Rosenbrock23 => None,
    };
    let error_order = tableau.map_or(2, |t| t.error_order);
    let error_exponent = -1.0 / (error_order as f64 + 1.0);

    let mut t = t0;
    let mut y = y0.to_vec();
    let mut fy = vec![0.0; n];
    eval(t, &y, &mut fy, &mut nfev);
    let interval = (t1 - t0).abs();
    let mut h_abs = match options.first_step {
        Some(h) if h > 0.0 => h.min(interval),
        Some(h) => return Err(OdeError::invalid(format!("first_step must be positive, got {h}"))),
        None => {
            // scipy.integrate._ivp.common.select_initial_step
            let scale: Vec<f64> = y.iter().map(|v| atol + v.abs() * rtol).collect();
            let d0 = rms(y.iter().zip(&scale).map(|(v, s)| v / s));
            let d1 = rms(fy.iter().zip(&scale).map(|(v, s)| v / s));
            let h0 = if d0 < 1e-5 || d1 < 1e-5 { 1e-6 } else { 0.01 * d0 / d1 };
            let h0 = h0.min(interval);
            let y1: Vec<f64> = y.iter().zip(&fy).map(|(v, d)| v + h0 * dir * d).collect();
            let mut f1 = vec![0.0; n];
            eval(t + h0 * dir, &y1, &mut f1, &mut nfev);
            let d2 = rms(f1.iter().zip(&fy).zip(&scale).map(|((a, b), s)| (a - b) / s)) / h0;
            // SciPy passes the error estimator's order here, not the method's
            let h1 = if d1 <= 1e-15 && d2 <= 1e-15 { (h0 * 1e-3).max(1e-6) } else { (0.01 / d1.max(d2)).powf(1.0 / (error_order as f64 + 1.0)) };
            (100.0 * h0).min(h1).min(interval)
        }
    };
    h_abs = h_abs.min(options.max_step);

    let mut sol = OdeSolution { t: Vec::new(), y: Vec::new(), t_events: vec![Vec::new(); options.events.len()], y_events: vec![Vec::new(); options.events.len()], nfev: 0, njev: 0, nlu: 0, status: OdeStatus::Finished };
    let mut next_eval = 0;
    match &options.t_eval {
        None => {
            sol.t.push(t);
            sol.y.push(y.clone());
        }
        Some(te) => {
            while next_eval < te.len() && te[next_eval] == t0 {
                sol.t.push(t0);
                sol.y.push(y.clone());
                next_eval += 1;
            }
        }
    }
    let mut g_prev: Vec<f64> = options.events.iter_mut().map(|e| (e.g)(t, &y)).collect();

    // working storage
    let stages = tableau.map_or(0, |tb| tb.b.len());
    let mut k: Vec<Vec<f64>> = vec![vec![0.0; n]; stages + 1];
    let mut y_new = vec![0.0; n];
    let mut f_new = vec![0.0; n];
    let mut tmp = vec![0.0; n];
    let mut rejected = false;
    let d = 1.0 / (2.0 + core::f64::consts::SQRT_2);
    let e32 = 6.0 + core::f64::consts::SQRT_2;
    // the Rosenbrock step's Jacobian and ∂f/∂t, refreshed every step
    let mut jac = vec![0.0; n * n];
    let mut dfdt = vec![0.0; n];

    while (t1 - t) * dir > 0.0 {
        let min_step = 10.0 * (next_up(t.abs()) - t.abs());
        h_abs = h_abs.min(options.max_step).max(min_step);
        let dense: Dense;
        let (mut t_new, mut h);
        if options.method == OdeMethod::Rosenbrock23 {
            // the Jacobian by forward differences, and ∂f/∂t
            for j in 0..n {
                let delta = (f64::EPSILON * y[j].abs().max(1.0)).sqrt();
                tmp.copy_from_slice(&y);
                tmp[j] += delta;
                eval(t, &tmp, &mut f_new, &mut nfev);
                for i in 0..n {
                    jac[i * n + j] = (f_new[i] - fy[i]) / delta;
                }
            }
            let dt_fd = (f64::EPSILON * t.abs().max(1.0)).sqrt() * dir;
            eval(t + dt_fd, &y, &mut f_new, &mut nfev);
            for i in 0..n {
                dfdt[i] = (f_new[i] - fy[i]) / dt_fd;
            }
            sol.njev += 1;
        }
        loop {
            if h_abs < min_step {
                sol.status = OdeStatus::StepTooSmall;
                return Ok(finish(sol, nfev));
            }
            h = h_abs * dir;
            t_new = t + h;
            if (t_new - t1) * dir > 0.0 {
                t_new = t1;
            }
            h = t_new - t;
            h_abs = h.abs();
            let error_norm;
            if let Some(tb) = tableau {
                // the stages; k[0] is f(t, y) (first same as last)
                k[0].copy_from_slice(&fy);
                for s in 1..stages {
                    for i in 0..n {
                        tmp[i] = y[i] + h * tb.a[s].iter().enumerate().map(|(j, &a)| a * k[j][i]).sum::<f64>();
                    }
                    eval(t + tb.c[s] * h, &tmp, &mut k[s], &mut nfev);
                }
                for i in 0..n {
                    y_new[i] = y[i] + h * tb.b.iter().enumerate().map(|(j, &b)| b * k[j][i]).sum::<f64>();
                }
                eval(t_new, &y_new, &mut f_new, &mut nfev);
                k[stages].copy_from_slice(&f_new);
                error_norm = rms((0..n).map(|i| {
                    let err = h * tb.e.iter().enumerate().map(|(j, &e)| e * k[j][i]).sum::<f64>();
                    err / (atol + y[i].abs().max(y_new[i].abs()) * rtol)
                }));
            } else {
                // W = I - h d J
                let mut w = vec![0.0; n * n];
                for i in 0..n {
                    for j in 0..n {
                        w[i * n + j] = if i == j { 1.0 } else { 0.0 } - h * d * jac[i * n + j];
                    }
                }
                sol.nlu += 1;
                let Some(lu) = Lu::new(w, n) else {
                    h_abs *= 0.5;
                    rejected = true;
                    continue;
                };
                let mut k1: Vec<f64> = (0..n).map(|i| fy[i] + h * d * dfdt[i]).collect();
                lu.solve(&mut k1);
                for i in 0..n {
                    tmp[i] = y[i] + 0.5 * h * k1[i];
                }
                let mut f1 = vec![0.0; n];
                eval(t + 0.5 * h, &tmp, &mut f1, &mut nfev);
                let mut k2: Vec<f64> = (0..n).map(|i| f1[i] - k1[i]).collect();
                lu.solve(&mut k2);
                for i in 0..n {
                    k2[i] += k1[i];
                    y_new[i] = y[i] + h * k2[i];
                }
                eval(t_new, &y_new, &mut f_new, &mut nfev);
                let mut k3: Vec<f64> = (0..n).map(|i| f_new[i] - e32 * (k2[i] - f1[i]) - 2.0 * (k1[i] - fy[i]) + h * d * dfdt[i]).collect();
                lu.solve(&mut k3);
                error_norm = rms((0..n).map(|i| {
                    let err = h / 6.0 * (k1[i] - 2.0 * k2[i] + k3[i]);
                    err / (atol + y[i].abs().max(y_new[i].abs()) * rtol)
                }));
            }
            if error_norm < 1.0 {
                let mut factor = if error_norm == 0.0 { MAX_FACTOR } else { MAX_FACTOR.min(SAFETY * error_norm.powf(error_exponent)) };
                if rejected {
                    factor = factor.min(1.0);
                }
                let step = h;
                // the step's dense output
                dense = match tableau {
                    Some(tb) => Dense::Poly {
                        t_old: t,
                        h: step,
                        y_old: y.clone(),
                        q: (0..tb.p[0].len()).map(|kk| (0..n).map(|i| tb.p.iter().enumerate().map(|(j, row)| row[kk] * k[j][i]).sum()).collect()).collect(),
                    },
                    None => Dense::Hermite { t_old: t, h: step, y0: y.clone(), y1: y_new.clone(), f0: fy.clone(), f1: f_new.clone() },
                };
                h_abs *= factor;
                rejected = false;
                break;
            }
            h_abs *= MIN_FACTOR.max(SAFETY * error_norm.powf(error_exponent));
            rejected = true;
        }
        let t_old = t;
        // events in (t_old, t_new]
        let mut stop_at: Option<(f64, Vec<f64>)> = None;
        for (e, ev) in options.events.iter_mut().enumerate() {
            let g_new = (ev.g)(t_new, &y_new);
            let up = g_prev[e] < 0.0 && g_new >= 0.0;
            let down = g_prev[e] > 0.0 && g_new <= 0.0;
            if (up && ev.direction >= 0.0) || (down && ev.direction <= 0.0) {
                // bisection on the dense output to the precision of t
                let (mut a, mut b, ga) = (t_old, t_new, g_prev[e]);
                for _ in 0..200 {
                    let m = 0.5 * (a + b);
                    if m == a || m == b {
                        break;
                    }
                    let gm = (ev.g)(m, &dense.eval(m));
                    if (gm < 0.0) == (ga < 0.0) && gm != 0.0 { a = m } else { b = m }
                }
                let ye = dense.eval(b);
                sol.t_events[e].push(b);
                sol.y_events[e].push(ye.clone());
                if ev.terminal && stop_at.as_ref().is_none_or(|(ts, _)| (b - ts) * dir < 0.0) {
                    stop_at = Some((b, ye));
                }
            }
            g_prev[e] = g_new;
        }
        let t_end = stop_at.as_ref().map_or(t_new, |s| s.0);
        // outputs
        match &options.t_eval {
            None => {
                sol.t.push(t_end);
                sol.y.push(stop_at.as_ref().map_or_else(|| y_new.clone(), |s| s.1.clone()));
            }
            Some(te) => {
                while next_eval < te.len() && (te[next_eval] - t_end) * dir <= 0.0 {
                    let tv = te[next_eval];
                    sol.t.push(tv);
                    sol.y.push(if tv == t_new { y_new.clone() } else { dense.eval(tv) });
                    next_eval += 1;
                }
            }
        }
        if stop_at.is_some() {
            sol.status = OdeStatus::Event;
            return Ok(finish(sol, nfev));
        }
        t = t_new;
        y.copy_from_slice(&y_new);
        fy.copy_from_slice(&f_new);
    }
    Ok(finish(sol, nfev))
}

fn finish(mut sol: OdeSolution, nfev: usize) -> OdeSolution {
    sol.nfev = nfev;
    sol
}

/// The next representable f64 above `x >= 0` (for SciPy's minimum step, 10 ulps of `t`).
fn next_up(x: f64) -> f64 {
    if x == 0.0 { f64::from_bits(1) } else { f64::from_bits(x.to_bits() + 1) }
}

#[cfg(test)]
mod tests;
