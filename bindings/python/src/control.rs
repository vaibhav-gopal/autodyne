//! Matrix equations (`autodyne.linalg`) and control analysis (`autodyne.control`): the Python
//! layer gives them `scipy.linalg`'s and `python-control`'s names.

use autodyne::linalg;
use autodyne::signal::{NdArray, NdView};
use autodyne::systems::{self, Domain, StateSpace, TransferFunction};
use autodyne::units::*;
use pyo3::exceptions::{PyTypeError, PyValueError};
use pyo3::prelude::*;

use crate::{input, numpy_array, numpy_complex, numpy_out, typed, value_error, Input};

type Obj = Py<PyAny>;

/// A float input as an owned f64 array.
fn matrix(obj: &Bound<'_, PyAny>) -> PyResult<NdArray<f64>> {
    let held = input(obj)?;
    crate::with_float_input!(held, T, v => Ok(v.map(|x| x.to_f64().unwrap_or(f64::NAN))))
}

fn tuple(py: Python<'_>, items: Vec<Obj>) -> PyResult<Obj> {
    Ok(pyo3::types::PyTuple::new(py, items)?.into_any().unbind())
}

fn float(py: Python<'_>, v: f64) -> PyResult<Obj> {
    Ok(v.into_pyobject(py)?.into_any().unbind())
}

#[pyfunction]
fn schur(py: Python<'_>, a: &Bound<'_, PyAny>) -> PyResult<Obj> {
    let (t, z) = linalg::schur(matrix(a)?.view()).map_err(value_error)?;
    tuple(py, vec![numpy_array(py, t)?, numpy_array(py, z)?])
}

#[pyfunction]
fn solve_sylvester(py: Python<'_>, a: &Bound<'_, PyAny>, b: &Bound<'_, PyAny>, q: &Bound<'_, PyAny>) -> PyResult<Obj> {
    numpy_array(py, linalg::solve_sylvester(matrix(a)?.view(), matrix(b)?.view(), matrix(q)?.view()).map_err(value_error)?)
}

#[pyfunction]
fn solve_continuous_lyapunov(py: Python<'_>, a: &Bound<'_, PyAny>, q: &Bound<'_, PyAny>) -> PyResult<Obj> {
    numpy_array(py, linalg::solve_continuous_lyapunov(matrix(a)?.view(), matrix(q)?.view()).map_err(value_error)?)
}

#[pyfunction]
fn solve_discrete_lyapunov(py: Python<'_>, a: &Bound<'_, PyAny>, q: &Bound<'_, PyAny>) -> PyResult<Obj> {
    numpy_array(py, linalg::solve_discrete_lyapunov(matrix(a)?.view(), matrix(q)?.view()).map_err(value_error)?)
}

#[pyfunction]
fn solve_continuous_are(py: Python<'_>, a: &Bound<'_, PyAny>, b: &Bound<'_, PyAny>, q: &Bound<'_, PyAny>, r: &Bound<'_, PyAny>) -> PyResult<Obj> {
    numpy_array(py, linalg::solve_continuous_are(matrix(a)?.view(), matrix(b)?.view(), matrix(q)?.view(), matrix(r)?.view()).map_err(value_error)?)
}

#[pyfunction]
fn solve_discrete_are(py: Python<'_>, a: &Bound<'_, PyAny>, b: &Bound<'_, PyAny>, q: &Bound<'_, PyAny>, r: &Bound<'_, PyAny>) -> PyResult<Obj> {
    numpy_array(py, linalg::solve_discrete_are(matrix(a)?.view(), matrix(b)?.view(), matrix(q)?.view(), matrix(r)?.view()).map_err(value_error)?)
}

fn domain(dt: Option<f64>) -> Domain {
    match dt {
        Some(dt) if dt > 0.0 => Domain::Discrete { dt },
        _ => Domain::Continuous,
    }
}

fn state_space(a: &Bound<'_, PyAny>, b: &Bound<'_, PyAny>, c: &Bound<'_, PyAny>, d: &Bound<'_, PyAny>, dt: Option<f64>) -> PyResult<StateSpace> {
    StateSpace::new(matrix(a)?, matrix(b)?, matrix(c)?, matrix(d)?, domain(dt)).map_err(value_error)
}

#[pyfunction]
fn ctrb(py: Python<'_>, a: &Bound<'_, PyAny>, b: &Bound<'_, PyAny>) -> PyResult<Obj> {
    let (a, b) = (matrix(a)?, matrix(b)?);
    let n = a.shape()[0];
    let p = b.shape().get(1).copied().unwrap_or(1);
    let sys = StateSpace::new(a, b, NdArray::zeros(&[1, n]).map_err(value_error)?, NdArray::zeros(&[1, p]).map_err(value_error)?, Domain::Continuous).map_err(value_error)?;
    numpy_array(py, sys.ctrb().map_err(value_error)?)
}

#[pyfunction]
fn obsv(py: Python<'_>, a: &Bound<'_, PyAny>, c: &Bound<'_, PyAny>) -> PyResult<Obj> {
    let (a, c) = (matrix(a)?, matrix(c)?);
    let n = a.shape()[0];
    let q = c.shape()[0];
    let sys = StateSpace::new(a, NdArray::zeros(&[n, 1]).map_err(value_error)?, c, NdArray::zeros(&[q, 1]).map_err(value_error)?, Domain::Continuous).map_err(value_error)?;
    numpy_array(py, sys.obsv().map_err(value_error)?)
}

/// `kind` "c" (controllability) or "o" (observability).
#[pyfunction]
#[pyo3(signature = (a, b, c, d, kind, dt=None))]
fn gram(py: Python<'_>, a: &Bound<'_, PyAny>, b: &Bound<'_, PyAny>, c: &Bound<'_, PyAny>, d: &Bound<'_, PyAny>, kind: &str, dt: Option<f64>) -> PyResult<Obj> {
    let sys = state_space(a, b, c, d, dt)?;
    let w = match kind {
        "c" => sys.controllability_gramian(),
        "o" => sys.observability_gramian(),
        other => return Err(PyValueError::new_err(format!("kind must be 'c' or 'o', got {other:?}"))),
    };
    numpy_array(py, w.map_err(value_error)?)
}

#[pyfunction]
#[pyo3(signature = (a, b, c, d, dt=None))]
fn dcgain(py: Python<'_>, a: &Bound<'_, PyAny>, b: &Bound<'_, PyAny>, c: &Bound<'_, PyAny>, d: &Bound<'_, PyAny>, dt: Option<f64>) -> PyResult<Obj> {
    numpy_array(py, state_space(a, b, c, d, dt)?.dcgain().map_err(value_error)?)
}

/// `(wn, zeta, poles)` of the state matrix's eigenvalues.
#[pyfunction]
#[pyo3(signature = (a, dt=None))]
fn damp(py: Python<'_>, a: &Bound<'_, PyAny>, dt: Option<f64>) -> PyResult<Obj> {
    let poles = linalg::eigvals(matrix(a)?.view()).map_err(value_error)?;
    let d = systems::damp(&poles, domain(dt));
    let n = d.len();
    let wn: Vec<f64> = d.iter().map(|x| x.1).collect();
    let zeta: Vec<f64> = d.iter().map(|x| x.2).collect();
    let poles = NdArray::from_vec(d.iter().map(|x| x.0).collect(), &[n]).map_err(value_error)?;
    tuple(py, vec![numpy_out(py, wn, &[n], None)?, numpy_out(py, zeta, &[n], None)?, numpy_complex(py, poles)?])
}

/// `(K, S, E)`: gain, Riccati solution, closed-loop poles.
#[pyfunction]
#[pyo3(signature = (a, b, q, r, discrete=false))]
fn lqr(py: Python<'_>, a: &Bound<'_, PyAny>, b: &Bound<'_, PyAny>, q: &Bound<'_, PyAny>, r: &Bound<'_, PyAny>, discrete: bool) -> PyResult<Obj> {
    let (a, b, q, r) = (matrix(a)?, matrix(b)?, matrix(q)?, matrix(r)?);
    let reg = if discrete { systems::dlqr(&a, &b, &q, &r) } else { systems::lqr(&a, &b, &q, &r) }.map_err(value_error)?;
    let n = reg.poles.len();
    let poles = NdArray::from_vec(reg.poles, &[n]).map_err(value_error)?;
    tuple(py, vec![numpy_array(py, reg.k)?, numpy_array(py, reg.s)?, numpy_complex(py, poles)?])
}

/// `(gm, pm, wpc, wgc)` of a SISO transfer function `num / den`.
#[pyfunction]
#[pyo3(signature = (num, den, dt=None))]
fn stability_margins(py: Python<'_>, num: Vec<f64>, den: Vec<f64>, dt: Option<f64>) -> PyResult<Obj> {
    let m = systems::stability_margins(&TransferFunction::new(num, den, domain(dt))).map_err(value_error)?;
    tuple(py, vec![float(py, m.gain_margin)?, float(py, m.phase_margin)?, float(py, m.phase_crossover)?, float(py, m.gain_crossover)?])
}

pub(crate) fn register(m: &Bound<'_, PyModule>) -> PyResult<()> {
    for f in [
        wrap_pyfunction!(schur, m)?,
        wrap_pyfunction!(solve_sylvester, m)?,
        wrap_pyfunction!(solve_continuous_lyapunov, m)?,
        wrap_pyfunction!(solve_discrete_lyapunov, m)?,
        wrap_pyfunction!(solve_continuous_are, m)?,
        wrap_pyfunction!(solve_discrete_are, m)?,
    ] {
        m.add_function(f)?;
    }
    let c = PyModule::new(m.py(), "control")?;
    for f in [
        wrap_pyfunction!(ctrb, &c)?,
        wrap_pyfunction!(obsv, &c)?,
        wrap_pyfunction!(gram, &c)?,
        wrap_pyfunction!(dcgain, &c)?,
        wrap_pyfunction!(damp, &c)?,
        wrap_pyfunction!(lqr, &c)?,
        wrap_pyfunction!(stability_margins, &c)?,
    ] {
        c.add_function(f)?;
    }
    m.add_submodule(&c)
}

