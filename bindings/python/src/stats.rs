//! `autodyne.stats`: order statistics, moments, histograms, covariance and time-series estimates
//! (the Python layer gives them NumPy's, SciPy's and statsmodels' names and defaults).

use autodyne::signal::NdView;
use autodyne::stats::{self, BinRule, Bins, QuantileMethod, YuleWalker};
use autodyne::units::*;
use pyo3::exceptions::{PyTypeError, PyValueError};
use pyo3::prelude::*;

use crate::{axis_index, input, numpy_array, numpy_out, typed, value_error, Input};

type Obj = Py<PyAny>;

macro_rules! float_view {
    ($obj:expr, $T:ident, $v:ident => $body:expr) => {{
        let held = input($obj)?;
        crate::with_float_input!(held, $T, $v => $body)
    }};
}

fn vec_out(py: Python<'_>, v: Vec<f64>) -> PyResult<Obj> {
    let n = v.len();
    numpy_out(py, v, &[n], None)
}

fn tuple(py: Python<'_>, items: Vec<Obj>) -> PyResult<Obj> {
    Ok(pyo3::types::PyTuple::new(py, items)?.into_any().unbind())
}

/// Runs `$body` with `$s` a slice of the 1-D float input `$obj`: its own memory when contiguous,
/// else a copy.
macro_rules! with_samples {
    ($obj:expr, $s:ident => $body:expr) => {
        float_view!($obj, T, v => {
            if v.ndim() != 1 {
                return Err(PyValueError::new_err(format!("expected a 1-D array, got {} dimensions", v.ndim())));
            }
            let owned;
            let $s: &[T] = match v.as_slice() {
                Some(s) => s,
                None => {
                    owned = v.to_owned().into_vec();
                    &owned
                }
            };
            $body
        })
    };
}

fn method(name: &str) -> PyResult<QuantileMethod> {
    match name {
        "linear" => Ok(QuantileMethod::Linear),
        "lower" => Ok(QuantileMethod::Lower),
        "higher" => Ok(QuantileMethod::Higher),
        "nearest" => Ok(QuantileMethod::Nearest),
        "midpoint" => Ok(QuantileMethod::Midpoint),
        other => Err(PyValueError::new_err(format!("unknown method {other:?}"))),
    }
}

/// Quantiles along `axis`, which they replace (one per `q`).
#[pyfunction]
fn quantile(py: Python<'_>, x: &Bound<'_, PyAny>, q: Vec<f64>, axis: isize, method_by: &str) -> PyResult<Obj> {
    let m = method(method_by)?;
    float_view!(x, T, v => {
        let axis = axis_index(axis, v.ndim())?;
        numpy_array(py, stats::quantile_axis(v, &q, axis, m, true).map_err(value_error)?)
    })
}

#[pyfunction]
fn median(py: Python<'_>, x: &Bound<'_, PyAny>, axis: isize) -> PyResult<Obj> {
    float_view!(x, T, v => {
        let axis = axis_index(axis, v.ndim())?;
        numpy_array(py, stats::median_axis(v, axis).map_err(value_error)?)
    })
}

#[pyfunction]
fn skew(py: Python<'_>, x: &Bound<'_, PyAny>, axis: isize, bias: bool) -> PyResult<Obj> {
    float_view!(x, T, v => {
        let axis = axis_index(axis, v.ndim())?;
        numpy_array(py, stats::skew_axis(v, axis, bias).map_err(value_error)?)
    })
}

#[pyfunction]
fn kurtosis(py: Python<'_>, x: &Bound<'_, PyAny>, axis: isize, fisher: bool, bias: bool) -> PyResult<Obj> {
    float_view!(x, T, v => {
        let axis = axis_index(axis, v.ndim())?;
        numpy_array(py, stats::kurtosis_axis(v, axis, fisher, bias).map_err(value_error)?)
    })
}

#[pyfunction]
fn moment(py: Python<'_>, x: &Bound<'_, PyAny>, order: u32, axis: isize) -> PyResult<Obj> {
    float_view!(x, T, v => {
        let axis = axis_index(axis, v.ndim())?;
        numpy_array(py, stats::moment_axis(v, order, axis).map_err(value_error)?)
    })
}

#[pyfunction]
fn zscore(py: Python<'_>, x: &Bound<'_, PyAny>, axis: isize, ddof: usize) -> PyResult<Obj> {
    float_view!(x, T, v => {
        let axis = axis_index(axis, v.ndim())?;
        numpy_array(py, stats::zscore_axis(v, axis, ddof).map_err(value_error)?)
    })
}

fn bins_of(bins: &Bound<'_, PyAny>) -> PyResult<Bins> {
    if let Ok(name) = bins.extract::<String>() {
        let rule = match name.as_str() {
            "auto" => BinRule::Auto,
            "fd" => BinRule::Fd,
            "sturges" => BinRule::Sturges,
            "scott" => BinRule::Scott,
            "rice" => BinRule::Rice,
            "sqrt" => BinRule::Sqrt,
            other => return Err(PyValueError::new_err(format!("unknown bin rule {other:?}"))),
        };
        return Ok(Bins::Rule(rule));
    }
    if let Ok(n) = bins.extract::<usize>() {
        return Ok(Bins::Count(n));
    }
    Ok(Bins::Edges(bins.extract::<Vec<f64>>().map_err(|_| PyTypeError::new_err("bins must be an int, a rule name or a sequence of edges"))?))
}

/// `(values, edges)` of a 1-D sample.
#[pyfunction]
#[pyo3(signature = (x, bins, range=None, weights=None, density=false))]
fn histogram(py: Python<'_>, x: &Bound<'_, PyAny>, bins: &Bound<'_, PyAny>, range: Option<(f64, f64)>, weights: Option<Vec<f64>>, density: bool) -> PyResult<Obj> {
    let bins = bins_of(bins)?;
    let (h, e) = with_samples!(x, s => stats::histogram(s, &bins, range, weights.as_deref(), density).map_err(value_error))?;
    tuple(py, vec![vec_out(py, h)?, vec_out(py, e)?])
}

#[pyfunction]
#[pyo3(signature = (x, bins, range=None))]
fn histogram_bin_edges(py: Python<'_>, x: &Bound<'_, PyAny>, bins: &Bound<'_, PyAny>, range: Option<(f64, f64)>) -> PyResult<Obj> {
    let bins = bins_of(bins)?;
    vec_out(py, with_samples!(x, s => stats::histogram_bin_edges(s, &bins, range).map_err(value_error))?)
}

#[pyfunction]
fn cov(py: Python<'_>, x: &Bound<'_, PyAny>, rowvar: bool, ddof: usize) -> PyResult<Obj> {
    float_view!(x, T, v => numpy_array(py, stats::cov(v, rowvar, ddof).map_err(value_error)?))
}

#[pyfunction]
fn corrcoef(py: Python<'_>, x: &Bound<'_, PyAny>, rowvar: bool) -> PyResult<Obj> {
    float_view!(x, T, v => numpy_array(py, stats::corrcoef(v, rowvar).map_err(value_error)?))
}

fn yw(name: &str) -> PyResult<YuleWalker> {
    match name {
        "adjusted" | "yw" | "ywadjusted" | "unbiased" => Ok(YuleWalker::Adjusted),
        "mle" | "ywm" | "ywmle" => Ok(YuleWalker::Mle),
        other => Err(PyValueError::new_err(format!("unknown method {other:?}"))),
    }
}

#[pyfunction]
fn acovf(py: Python<'_>, x: &Bound<'_, PyAny>, nlag: usize, adjusted: bool, demean: bool) -> PyResult<Obj> {
    vec_out(py, with_samples!(x, s => stats::acovf(s, nlag, adjusted, demean).map_err(value_error))?)
}

#[pyfunction]
fn acf(py: Python<'_>, x: &Bound<'_, PyAny>, nlags: usize, adjusted: bool) -> PyResult<Obj> {
    vec_out(py, with_samples!(x, s => stats::acf(s, nlags, adjusted).map_err(value_error))?)
}

#[pyfunction]
fn pacf(py: Python<'_>, x: &Bound<'_, PyAny>, nlags: usize, method_by: &str) -> PyResult<Obj> {
    vec_out(py, with_samples!(x, s => stats::pacf(s, nlags, yw(method_by)?).map_err(value_error))?)
}

/// `(ar, sigma2)` of a Yule-Walker fit.
#[pyfunction]
fn yule_walker(py: Python<'_>, x: &Bound<'_, PyAny>, order: usize, method_by: &str) -> PyResult<Obj> {
    let method = yw(method_by)?;
    let fit = with_samples!(x, s => stats::yule_walker(s, order, method).map_err(value_error))?;
    tuple(py, vec![vec_out(py, fit.ar)?, fit.sigma2.into_pyobject(py)?.into_any().unbind()])
}

/// `(ar, sigma2)` of a Burg fit.
#[pyfunction]
fn burg(py: Python<'_>, x: &Bound<'_, PyAny>, order: usize) -> PyResult<Obj> {
    let fit = with_samples!(x, s => stats::burg(s, order).map_err(value_error))?;
    tuple(py, vec![vec_out(py, fit.ar)?, fit.sigma2.into_pyobject(py)?.into_any().unbind()])
}

/// `(sigma2, ar, pacf)` from autocovariances.
#[pyfunction]
fn levinson_durbin(py: Python<'_>, acov: Vec<f64>, order: usize) -> PyResult<Obj> {
    let fit = stats::levinson_durbin(&acov, order).map_err(value_error)?;
    tuple(py, vec![fit.sigma2.into_pyobject(py)?.into_any().unbind(), vec_out(py, fit.ar)?, vec_out(py, fit.pacf)?])
}

pub(crate) fn register(m: &Bound<'_, PyModule>) -> PyResult<()> {
    let s = PyModule::new(m.py(), "stats")?;
    for f in [
        wrap_pyfunction!(quantile, &s)?,
        wrap_pyfunction!(median, &s)?,
        wrap_pyfunction!(skew, &s)?,
        wrap_pyfunction!(kurtosis, &s)?,
        wrap_pyfunction!(moment, &s)?,
        wrap_pyfunction!(zscore, &s)?,
        wrap_pyfunction!(histogram, &s)?,
        wrap_pyfunction!(histogram_bin_edges, &s)?,
        wrap_pyfunction!(cov, &s)?,
        wrap_pyfunction!(corrcoef, &s)?,
        wrap_pyfunction!(acovf, &s)?,
        wrap_pyfunction!(acf, &s)?,
        wrap_pyfunction!(pacf, &s)?,
        wrap_pyfunction!(yule_walker, &s)?,
        wrap_pyfunction!(burg, &s)?,
        wrap_pyfunction!(levinson_durbin, &s)?,
    ] {
        s.add_function(f)?;
    }
    m.add_submodule(&s)
}

