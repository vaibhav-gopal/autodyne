//! `autodyne.integrate.solve_ivp`: autodyne's integrators driving a Python right-hand side.

use std::cell::RefCell;

use autodyne::ode::{self, Event, OdeMethod, OdeOptions, OdeStatus};
use numpy::{PyArray1, PyArrayMethods};
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use pyo3::types::PyDict;

use crate::{numpy_out, value_error};

type Obj = Py<PyAny>;

/// `out = fun(t, y)` through Python; the first exception is kept and the derivative made NaN, so the
/// integration stops shortly (its steps fail) and the exception is raised afterwards.
fn call(py: Python<'_>, fun: &Bound<'_, PyAny>, t: f64, y: &[f64], out: &mut [f64], error: &RefCell<Option<PyErr>>) {
    if error.borrow().is_some() {
        out.fill(f64::NAN);
        return;
    }
    let result = (|| -> PyResult<()> {
        let ya = PyArray1::from_slice(py, y);
        let r = fun.call1((t, ya))?;
        // a float64 NumPy array is read directly; anything else is converted once
        let arr = match r.cast::<PyArray1<f64>>() {
            Ok(a) => a.clone(),
            Err(_) => numpy::PyArray1::from_vec(py, r.extract::<Vec<f64>>()?),
        };
        let ro = arr.readonly();
        let s = ro.as_slice().map_err(|_| PyValueError::new_err("fun must return a 1-D array"))?;
        if s.len() != out.len() {
            return Err(PyValueError::new_err(format!("fun returned {} values for {} states", s.len(), out.len())));
        }
        out.copy_from_slice(s);
        Ok(())
    })();
    if let Err(e) = result {
        *error.borrow_mut() = Some(e);
        out.fill(f64::NAN);
    }
}

/// Solves `y' = fun(t, y)` over `t_span`; returns a dict like SciPy's `OdeResult` (the Python layer
/// wraps it).
#[pyfunction]
#[pyo3(signature = (fun, t0, t1, y0, method, rtol, atol, first_step, max_step, t_eval, events, terminal, direction))]
#[allow(clippy::too_many_arguments)]
fn solve_ivp(
    py: Python<'_>,
    fun: &Bound<'_, PyAny>,
    t0: f64,
    t1: f64,
    y0: Vec<f64>,
    method: &str,
    rtol: f64,
    atol: f64,
    first_step: Option<f64>,
    max_step: f64,
    t_eval: Option<Vec<f64>>,
    events: Vec<Bound<'_, PyAny>>,
    terminal: Vec<bool>,
    direction: Vec<f64>,
) -> PyResult<Obj> {
    let method = match method {
        "RK45" => OdeMethod::Rk45,
        "RK23" => OdeMethod::Rk23,
        "Rosenbrock23" | "ode23s" => OdeMethod::Rosenbrock23,
        other => return Err(PyValueError::new_err(format!("unknown method {other:?} (RK45, RK23 or Rosenbrock23)"))),
    };
    let error = RefCell::new(None);
    let event_list: Vec<Event<'_>> = events
        .iter()
        .zip(terminal.iter().zip(&direction))
        .map(|(g, (&terminal, &direction))| {
            let error = &error;
            Event {
                g: Box::new(move |t: f64, y: &[f64]| -> f64 {
                    if error.borrow().is_some() {
                        return f64::NAN;
                    }
                    match g.call1((t, PyArray1::from_slice(py, y))).and_then(|v| v.extract::<f64>()) {
                        Ok(v) => v,
                        Err(e) => {
                            *error.borrow_mut() = Some(e);
                            f64::NAN
                        }
                    }
                }),
                terminal,
                direction,
            }
        })
        .collect();
    let options = OdeOptions { method, rtol, atol, first_step, max_step, t_eval, events: event_list };
    let sol = ode::solve_ivp(|t, y, dy| call(py, fun, t, y, dy, &error), (t0, t1), &y0, options).map_err(value_error)?;
    if let Some(e) = error.into_inner() {
        return Err(e);
    }
    let n = y0.len();
    let points = sol.t.len();
    // y as (states, points), as SciPy returns it
    let mut y = vec![0.0; n * points];
    for (j, state) in sol.y.iter().enumerate() {
        for i in 0..n {
            y[i * points + j] = state[i];
        }
    }
    let out = PyDict::new(py);
    out.set_item("t", numpy_out(py, sol.t, &[points], None)?)?;
    out.set_item("y", numpy_out(py, y, &[n, points], None)?)?;
    let t_events: Vec<Obj> = sol.t_events.into_iter().map(|t| { let k = t.len(); numpy_out(py, t, &[k], None) }).collect::<PyResult<_>>()?;
    let y_events: Vec<Obj> = sol.y_events.into_iter().map(|ys| { let k = ys.len(); numpy_out(py, ys.concat(), &[k, n], None) }).collect::<PyResult<_>>()?;
    out.set_item("t_events", t_events)?;
    out.set_item("y_events", y_events)?;
    out.set_item("nfev", sol.nfev)?;
    out.set_item("njev", sol.njev)?;
    out.set_item("nlu", sol.nlu)?;
    let (status, message) = match sol.status {
        OdeStatus::Finished => (0, "The solver successfully reached the end of the integration interval."),
        OdeStatus::Event => (1, "A termination event occurred."),
        OdeStatus::StepTooSmall => (-1, "Required step size is less than spacing between numbers."),
    };
    out.set_item("status", status)?;
    out.set_item("message", message)?;
    Ok(out.into_any().unbind())
}

pub(crate) fn register(m: &Bound<'_, PyModule>) -> PyResult<()> {
    let i = PyModule::new(m.py(), "integrate")?;
    i.add_function(wrap_pyfunction!(solve_ivp, &i)?)?;
    m.add_submodule(&i)
}

