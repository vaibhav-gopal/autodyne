//! The SciPy-like surface: linear algebra, filter design and filtering, spectral estimation and
//! FFTs. Arguments follow SciPy's names; the Python wrappers in `autodyne.linalg`,
//! `autodyne.signal` and `autodyne.fft` supply its defaults.

use autodyne::filter::design::{self, Band, BesselNorm, Design, IirKind, RemezType};
use autodyne::filter::{self, Pad};
use autodyne::linalg::{self, LinalgFloat};
use autodyne::signal::{NdArray, NdView};
use autodyne::fft::Fft;
use autodyne::spectral::{self, Average, Boundary, Detrend, IstftOptions, Scaling, Segments, SpectrogramMode, StftOptions, WindowSpec};
use autodyne::systems::{self, Domain, Pairing, Zpk, C64};
use autodyne::units::*;
use numpy::PyReadonlyArrayDyn;
use pyo3::exceptions::{PyTypeError, PyValueError};
use pyo3::prelude::*;
use pyo3::types::PyTuple;

use crate::{axis_index, import, input, numpy_array, numpy_complex, numpy_out, typed, value_error, Input};

type Obj = Py<PyAny>;

fn tuple(py: Python<'_>, items: Vec<Obj>) -> PyResult<Obj> {
    Ok(PyTuple::new(py, items)?.into_any().unbind())
}

fn vec_out(py: Python<'_>, v: Vec<f64>) -> PyResult<Obj> {
    let n = v.len();
    numpy_out(py, v, &[n], None)
}

fn complex_vec_out(py: Python<'_>, v: Vec<C64>) -> PyResult<Obj> {
    let n = v.len();
    numpy_complex(py, NdArray::from_vec(v, &[n]).map_err(value_error)?)
}

/// A 1-D float64 coefficient vector from any sequence or array.
fn coeffs(obj: &Bound<'_, PyAny>) -> PyResult<Vec<f64>> {
    if let Ok(a) = obj.extract::<PyReadonlyArrayDyn<'_, f64>>() {
        return Ok(a.as_array().iter().copied().collect());
    }
    if let Ok(v) = obj.extract::<f64>() {
        return Ok(vec![v]);
    }
    obj.extract::<Vec<f64>>()
}

/// Second-order sections from an `(n, 6)` array.
fn sos_in(obj: &Bound<'_, PyAny>) -> PyResult<Vec<[f64; 6]>> {
    let a: PyReadonlyArrayDyn<'_, f64> = obj.extract()?;
    let v = a.as_array();
    if v.ndim() != 2 || v.shape()[1] != 6 {
        return Err(PyValueError::new_err("sos must have shape (n_sections, 6)"));
    }
    Ok(v.outer_iter().map(|row| [row[0], row[1], row[2], row[3], row[4], row[5]]).collect())
}

fn sos_out(py: Python<'_>, sos: &[[f64; 6]]) -> PyResult<Obj> {
    numpy_out(py, sos.concat(), &[sos.len(), 6], None)
}

/// Complex values (an array with `__dlpack__`: the Python wrappers pass `numpy.asarray(z, complex)`).
fn complexes(obj: &Bound<'_, PyAny>) -> PyResult<Vec<C64>> {
    Ok(complex_input(obj)?.into_vec())
}

/// SciPy's window argument: a name, or a `(name, parameter)` tuple.
pub(crate) fn window_spec(obj: &Bound<'_, PyAny>) -> PyResult<WindowSpec> {
    if let Ok(name) = obj.extract::<String>() {
        return name.parse().map_err(PyValueError::new_err);
    }
    let t: (String, f64) = obj.extract().map_err(|_| PyTypeError::new_err("window must be a name or a (name, parameter) tuple"))?;
    Ok(match t.0.as_str() {
        "kaiser" | "ksr" => WindowSpec::Kaiser { beta: t.1 },
        "gaussian" | "gauss" | "gss" => WindowSpec::Gaussian { std: t.1 },
        "tukey" | "tuk" => WindowSpec::Tukey { alpha: t.1 },
        "exponential" | "poisson" => WindowSpec::Exponential { tau: t.1 },
        "chebwin" | "cheb" => WindowSpec::Chebwin { attenuation: t.1 },
        other => return Err(PyValueError::new_err(format!("unknown parameterized window {other:?}"))),
    })
}

// LINEAR ALGEBRA ==================================================================================

/// Runs `$body` with the input view as f32 or f64.
macro_rules! float_view {
    ($obj:expr, $T:ident, $v:ident => $body:expr) => {{
        let held = input($obj)?;
        crate::with_float_input!(held, $T, $v => $body)
    }};
}

fn linalg_error(e: linalg::LinalgError) -> PyErr {
    PyValueError::new_err(e.to_string())
}

/// Calls `f` with the second of two float inputs as a view of the first's dtype: read in place when
/// it already has that dtype, else converted once.
fn with_pair<T: LinalgFloat + DynElementBridge + 'static, R>(b: &Bound<'_, PyAny>, f: impl FnOnce(NdView<'_, T>) -> PyResult<R>) -> PyResult<R> {
    let held = input(b)?;
    crate::with_float_input!(held, U, v => {
        if std::any::TypeId::of::<U>() == std::any::TypeId::of::<T>() {
            // SAFETY: U and T are the same type, so the view is reinterpreted as itself
            let same: NdView<'_, T> = unsafe { std::mem::transmute_copy(&v) };
            f(same)
        } else {
            let converted = NdArray::from_vec(v.iter().map(|x| T::_lit(x.to_f64().unwrap_or(f64::NAN))).collect(), v.shape()).map_err(value_error)?;
            f(converted.view())
        }
    })
}

/// Marker for the element types the linear algebra bindings use.
pub(crate) trait DynElementBridge: Copy + Default + numpy::Element {}
impl DynElementBridge for f32 {}
impl DynElementBridge for f64 {}

/// Matrix product of 1-D or 2-D arrays, `numpy.matmul` style (lists and integer arrays become float64).
#[pyfunction]
fn matmul<'py>(py: Python<'py>, a: &Bound<'py, PyAny>, b: &Bound<'py, PyAny>) -> PyResult<Obj> {
    let float = |x: &Bound<'_, PyAny>| x.cast::<numpy::PyArrayDyn<f64>>().is_ok() || x.cast::<numpy::PyArrayDyn<f32>>().is_ok();
    if float(a) && float(b) {
        return matmul_floats(py, a, b);
    }
    // anything else (lists, integer arrays, other tensors) through NumPy, as float64 unless float
    let np = py.import("numpy")?;
    let convert = |x: &Bound<'py, PyAny>| -> PyResult<Bound<'py, PyAny>> {
        if float(x) {
            return Ok(x.clone());
        }
        let arr = np.call_method1("asarray", (x,))?;
        let kind: String = arr.getattr("dtype")?.getattr("kind")?.extract()?;
        if kind == "f" { Ok(arr) } else { np.call_method1("asarray", (x, "float64")) }
    };
    matmul_floats(py, &convert(a)?, &convert(b)?)
}

fn matmul_floats(py: Python<'_>, a: &Bound<'_, PyAny>, b: &Bound<'_, PyAny>) -> PyResult<Obj> {
    // the product written straight into a new NumPy array, as NumPy's own does
    float_view!(a, T, va => with_pair::<T, _>(b, |vb| {
        let shape = linalg::matmul_shape(va.shape(), vb.shape()).map_err(linalg_error)?;
        // SAFETY: matmul_into overwrites every element before the array is returned
        let out = unsafe { numpy::PyArray::<T, numpy::ndarray::IxDyn>::new(py, numpy::ndarray::IxDyn(&shape), false) };
        let mut strides = vec![1isize; shape.len()];
        for i in (0..shape.len().saturating_sub(1)).rev() {
            strides[i] = strides[i + 1] * shape[i + 1] as isize;
        }
        // SAFETY: a fresh C-ordered array of `shape`, borrowed by nothing else
        let view = unsafe { autodyne::signal::NdViewMut::from_raw_parts(numpy::PyArrayMethods::data(&out), &shape, &strides) }.map_err(value_error)?;
        linalg::matmul_into(va, vb, view).map_err(linalg_error)?;
        Ok(out.into_any().unbind())
    }))
}

#[pyfunction]
fn solve(py: Python<'_>, a: &Bound<'_, PyAny>, b: &Bound<'_, PyAny>) -> PyResult<Obj> {
    float_view!(a, T, va => with_pair::<T, _>(b, |vb| numpy_array(py, linalg::solve(va, vb).map_err(linalg_error)?)))
}

#[pyfunction]
fn inv(py: Python<'_>, a: &Bound<'_, PyAny>) -> PyResult<Obj> {
    float_view!(a, T, va => numpy_array(py, linalg::inv(va).map_err(linalg_error)?))
}

#[pyfunction]
fn det(a: &Bound<'_, PyAny>) -> PyResult<f64> {
    float_view!(a, T, va => Ok(linalg::det(va).map_err(linalg_error)?.to_f64().unwrap_or(f64::NAN)))
}

/// `(x, rank, singular values)`.
#[pyfunction]
fn lstsq(py: Python<'_>, a: &Bound<'_, PyAny>, b: &Bound<'_, PyAny>) -> PyResult<Obj> {
    float_view!(a, T, va => with_pair::<T, _>(b, |vb| {
        let r = linalg::lstsq(va, vb).map_err(linalg_error)?;
        let s: Vec<f64> = r.singular_values.iter().map(|v| v.to_f64().unwrap_or(f64::NAN)).collect();
        tuple(py, vec![numpy_array(py, r.solution)?, r.rank.into_pyobject(py)?.into_any().unbind(), vec_out(py, s)?])
    }))
}

/// `(eigenvalues, eigenvectors)`, complex.
#[pyfunction]
fn eig(py: Python<'_>, a: &Bound<'_, PyAny>) -> PyResult<Obj> {
    float_view!(a, T, va => {
        let e = linalg::eig(va).map_err(linalg_error)?;
        let w = NdArray::from_vec(e.values.clone(), &[e.values.len()]).map_err(value_error)?;
        tuple(py, vec![numpy_complex(py, w)?, numpy_complex(py, e.vectors)?])
    })
}

#[pyfunction]
fn eigvals(py: Python<'_>, a: &Bound<'_, PyAny>) -> PyResult<Obj> {
    float_view!(a, T, va => {
        let w = linalg::eigvals(va).map_err(linalg_error)?;
        let n = w.len();
        numpy_complex(py, NdArray::from_vec(w, &[n]).map_err(value_error)?)
    })
}

/// `(eigenvalues ascending, eigenvectors)` of a symmetric matrix.
#[pyfunction]
fn eigh(py: Python<'_>, a: &Bound<'_, PyAny>) -> PyResult<Obj> {
    float_view!(a, T, va => {
        let (w, v) = linalg::eigh(va).map_err(linalg_error)?;
        let n = w.len();
        tuple(py, vec![numpy_array(py, NdArray::from_vec(w, &[n]).map_err(value_error)?)?, numpy_array(py, v)?])
    })
}

/// `(u, s, vt)`.
#[pyfunction]
#[pyo3(signature = (a, full_matrices=true))]
fn svd(py: Python<'_>, a: &Bound<'_, PyAny>, full_matrices: bool) -> PyResult<Obj> {
    float_view!(a, T, va => {
        let r = linalg::svd(va, full_matrices).map_err(linalg_error)?;
        let n = r.s.len();
        tuple(py, vec![numpy_array(py, r.u)?, numpy_array(py, NdArray::from_vec(r.s, &[n]).map_err(value_error)?)?, numpy_array(py, r.vt)?])
    })
}

#[pyfunction]
fn svdvals(py: Python<'_>, a: &Bound<'_, PyAny>) -> PyResult<Obj> {
    float_view!(a, T, va => {
        let s = linalg::svdvals(va).map_err(linalg_error)?;
        let n = s.len();
        numpy_array(py, NdArray::from_vec(s, &[n]).map_err(value_error)?)
    })
}

#[pyfunction]
fn pinv(py: Python<'_>, a: &Bound<'_, PyAny>) -> PyResult<Obj> {
    float_view!(a, T, va => numpy_array(py, linalg::pinv(va).map_err(linalg_error)?))
}

/// Reduced `(q, r)`.
#[pyfunction]
fn qr(py: Python<'_>, a: &Bound<'_, PyAny>) -> PyResult<Obj> {
    float_view!(a, T, va => {
        let (q, r) = linalg::qr(va).map_err(linalg_error)?;
        tuple(py, vec![numpy_array(py, q)?, numpy_array(py, r)?])
    })
}

#[pyfunction]
fn cholesky(py: Python<'_>, a: &Bound<'_, PyAny>) -> PyResult<Obj> {
    float_view!(a, T, va => numpy_array(py, linalg::cholesky(va).map_err(linalg_error)?))
}

#[pyfunction]
fn expm(py: Python<'_>, a: &Bound<'_, PyAny>) -> PyResult<Obj> {
    float_view!(a, T, va => numpy_array(py, linalg::expm(va).map_err(linalg_error)?))
}

#[pyfunction]
fn roots(py: Python<'_>, p: &Bound<'_, PyAny>) -> PyResult<Obj> {
    complex_vec_out(py, linalg::roots(&coeffs(p)?).map_err(linalg_error)?)
}

// FILTER DESIGN ===================================================================================

fn design_error(e: impl std::fmt::Display) -> PyErr {
    PyValueError::new_err(e.to_string())
}

/// `zpk` in SciPy's requested `output` form.
fn design_out(py: Python<'_>, zpk: Zpk, output: &str) -> PyResult<Obj> {
    match output {
        "zpk" => tuple(py, vec![complex_vec_out(py, zpk.zeros.clone())?, complex_vec_out(py, zpk.poles.clone())?, zpk.gain.into_pyobject(py)?.into_any().unbind()]),
        "ba" => {
            let tf = zpk.to_tf();
            tuple(py, vec![vec_out(py, tf.num)?, vec_out(py, tf.den)?])
        }
        "sos" => {
            let pairing = if zpk.domain.is_discrete() { Pairing::Nearest } else { Pairing::Minimal };
            sos_out(py, &systems::zpk2sos(&zpk.zeros, &zpk.poles, zpk.gain, pairing, zpk.domain).map_err(design_error)?)
        }
        other => Err(PyValueError::new_err(format!("unknown output {other:?}"))),
    }
}

/// `scipy.signal.iirfilter`: `wn` one or two edges (Hz with `fs`, else normalized to Nyquist for
/// digital, rad/s for analog).
#[pyfunction]
#[pyo3(signature = (n, wn, rp=None, rs=None, btype="bandpass", analog=false, ftype="butter", output="ba", fs=None, norm="phase"))]
#[allow(clippy::too_many_arguments)]
fn iirfilter(py: Python<'_>, n: usize, wn: &Bound<'_, PyAny>, rp: Option<f64>, rs: Option<f64>, btype: &str, analog: bool, ftype: &str, output: &str, fs: Option<f64>, norm: &str) -> PyResult<Obj> {
    let edges = coeffs(wn)?;
    // digital edges normalized to Nyquist are Hz at fs = 2
    let fs = fs.unwrap_or(2.0);
    let band = match (btype, edges.as_slice()) {
        ("lowpass" | "low", [w]) => Band::Lowpass(*w),
        ("highpass" | "high", [w]) => Band::Highpass(*w),
        ("bandpass" | "band" | "pass", [a, b]) => Band::Bandpass(*a, *b),
        ("bandstop" | "stop", [a, b]) => Band::Bandstop(*a, *b),
        _ => return Err(PyValueError::new_err(format!("btype {btype:?} does not fit {} edge(s)", edges.len()))),
    };
    let need = |v: Option<f64>, name: &str| v.ok_or_else(|| PyValueError::new_err(format!("{ftype} needs {name}")));
    let kind = match ftype {
        "butter" | "butterworth" => IirKind::Butterworth,
        "cheby1" => IirKind::Chebyshev1 { rp: need(rp, "rp")? },
        "cheby2" => IirKind::Chebyshev2 { rs: need(rs, "rs")? },
        "ellip" | "elliptic" => IirKind::Elliptic { rp: need(rp, "rp")?, rs: need(rs, "rs")? },
        "bessel" => IirKind::Bessel {
            norm: match norm {
                "phase" => BesselNorm::Phase,
                "delay" => BesselNorm::Delay,
                "mag" => BesselNorm::Mag,
                other => return Err(PyValueError::new_err(format!("unknown Bessel norm {other:?}"))),
            },
        },
        other => return Err(PyValueError::new_err(format!("unknown ftype {other:?}"))),
    };
    let design = if analog { Design::Analog } else { Design::Digital { fs } };
    design_out(py, design::iirfilter(n, band, kind, design).map_err(design_error)?, output)
}

#[pyfunction]
#[pyo3(signature = (numtaps, cutoff, window, pass_zero, scale, fs))]
fn firwin(py: Python<'_>, numtaps: usize, cutoff: &Bound<'_, PyAny>, window: &Bound<'_, PyAny>, pass_zero: bool, scale: bool, fs: f64) -> PyResult<Obj> {
    vec_out(py, design::firwin(numtaps, &coeffs(cutoff)?, window_spec(window)?, pass_zero, scale, fs).map_err(design_error)?)
}

#[pyfunction]
#[pyo3(signature = (numtaps, freq, gain, nfreqs, window, antisymmetric, fs))]
#[allow(clippy::too_many_arguments)]
fn firwin2(py: Python<'_>, numtaps: usize, freq: &Bound<'_, PyAny>, gain: &Bound<'_, PyAny>, nfreqs: Option<usize>, window: Option<&Bound<'_, PyAny>>, antisymmetric: bool, fs: f64) -> PyResult<Obj> {
    let window = window.map(window_spec).transpose()?;
    vec_out(py, design::firwin2(numtaps, &coeffs(freq)?, &coeffs(gain)?, nfreqs, window, antisymmetric, fs).map_err(design_error)?)
}

fn pairs(v: Vec<f64>) -> PyResult<Vec<(f64, f64)>> {
    if !v.len().is_multiple_of(2) {
        return Err(PyValueError::new_err("band edges come in pairs"));
    }
    Ok(v.chunks(2).map(|c| (c[0], c[1])).collect())
}

#[pyfunction]
#[pyo3(signature = (numtaps, bands, desired, weight, fs))]
fn firls(py: Python<'_>, numtaps: usize, bands: &Bound<'_, PyAny>, desired: &Bound<'_, PyAny>, weight: Option<&Bound<'_, PyAny>>, fs: f64) -> PyResult<Obj> {
    let weight = weight.map(coeffs).transpose()?;
    vec_out(py, design::firls(numtaps, &pairs(coeffs(bands)?)?, &pairs(coeffs(desired)?)?, weight.as_deref(), fs).map_err(design_error)?)
}

#[pyfunction]
#[pyo3(signature = (numtaps, bands, desired, weight, kind, maxiter, grid_density, fs))]
#[allow(clippy::too_many_arguments)]
fn remez(py: Python<'_>, numtaps: usize, bands: &Bound<'_, PyAny>, desired: &Bound<'_, PyAny>, weight: Option<&Bound<'_, PyAny>>, kind: &str, maxiter: usize, grid_density: usize, fs: f64) -> PyResult<Obj> {
    let kind = match kind {
        "bandpass" => RemezType::Bandpass,
        "differentiator" => RemezType::Differentiator,
        "hilbert" => RemezType::Hilbert,
        other => return Err(PyValueError::new_err(format!("unknown type {other:?}"))),
    };
    let weight = weight.map(coeffs).transpose()?;
    vec_out(py, design::remez(numtaps, &pairs(coeffs(bands)?)?, &coeffs(desired)?, weight.as_deref(), kind, maxiter, grid_density, fs).map_err(design_error)?)
}

#[pyfunction]
fn kaiserord(ripple: f64, width: f64) -> PyResult<(usize, f64)> {
    design::kaiserord(ripple, width).map_err(design_error)
}

#[pyfunction]
#[pyo3(signature = (window, nx, fftbins=true))]
fn get_window(py: Python<'_>, window: &Bound<'_, PyAny>, nx: usize, fftbins: bool) -> PyResult<Obj> {
    vec_out(py, spectral::get_window(window_spec(window)?, nx, fftbins))
}

#[pyfunction]
#[pyo3(signature = (z, p, k, pairing="nearest", analog=false))]
fn zpk2sos(py: Python<'_>, z: &Bound<'_, PyAny>, p: &Bound<'_, PyAny>, k: f64, pairing: &str, analog: bool) -> PyResult<Obj> {
    let pairing = match pairing {
        "nearest" => Pairing::Nearest,
        "keep_odd" => Pairing::KeepOdd,
        "minimal" => Pairing::Minimal,
        other => return Err(PyValueError::new_err(format!("unknown pairing {other:?}"))),
    };
    let domain = if analog { Domain::Continuous } else { Domain::Discrete { dt: 1.0 } };
    sos_out(py, &systems::zpk2sos(&complexes(z)?, &complexes(p)?, k, pairing, domain).map_err(design_error)?)
}

#[pyfunction]
fn tf2zpk(py: Python<'_>, b: &Bound<'_, PyAny>, a: &Bound<'_, PyAny>) -> PyResult<Obj> {
    let (z, p, k) = systems::tf2zpk(&coeffs(b)?, &coeffs(a)?).map_err(design_error)?;
    tuple(py, vec![complex_vec_out(py, z)?, complex_vec_out(py, p)?, k.into_pyobject(py)?.into_any().unbind()])
}

// FREQUENCY RESPONSE ===============================================================================

#[pyfunction]
fn freqz(py: Python<'_>, b: &Bound<'_, PyAny>, a: &Bound<'_, PyAny>, freqs: &Bound<'_, PyAny>, fs: f64) -> PyResult<Obj> {
    complex_vec_out(py, systems::freqz(&coeffs(b)?, &coeffs(a)?, &coeffs(freqs)?, fs))
}

#[pyfunction]
fn sosfreqz(py: Python<'_>, sos: &Bound<'_, PyAny>, freqs: &Bound<'_, PyAny>, fs: f64) -> PyResult<Obj> {
    complex_vec_out(py, systems::sosfreqz(&sos_in(sos)?, &coeffs(freqs)?, fs))
}

#[pyfunction]
fn group_delay(py: Python<'_>, b: &Bound<'_, PyAny>, a: &Bound<'_, PyAny>, freqs: &Bound<'_, PyAny>, fs: f64) -> PyResult<Obj> {
    vec_out(py, systems::group_delay(&coeffs(b)?, &coeffs(a)?, &coeffs(freqs)?, fs))
}

// FILTERING =======================================================================================

/// A float array of the same dtype as the main input (for initial states).
fn state_as<T: Float + Default>(obj: &Bound<'_, PyAny>) -> PyResult<NdArray<T>> {
    let held = input(obj)?;
    crate::with_float_input!(held, U, v => Ok(v.to_owned().map(|&x| T::_lit(x.to_f64().unwrap_or(f64::NAN)))))
}

#[pyfunction]
#[pyo3(signature = (b, a, x, axis=-1, zi=None))]
fn lfilter(py: Python<'_>, b: &Bound<'_, PyAny>, a: &Bound<'_, PyAny>, x: &Bound<'_, PyAny>, axis: isize, zi: Option<&Bound<'_, PyAny>>) -> PyResult<Obj> {
    let (b, a) = (coeffs(b)?, coeffs(a)?);
    float_view!(x, T, v => {
        let axis = axis_index(axis, v.ndim())?;
        match zi {
            None => numpy_array(py, filter::lfilter(&b, &a, v, axis).map_err(design_error)?),
            Some(zi) => {
                let zi = state_as::<T>(zi)?;
                let (y, zf) = filter::lfilter_with_state(&b, &a, v, axis, zi.view()).map_err(design_error)?;
                tuple(py, vec![numpy_array(py, y)?, numpy_array(py, zf)?])
            }
        }
    })
}

#[pyfunction]
fn lfilter_zi(py: Python<'_>, b: &Bound<'_, PyAny>, a: &Bound<'_, PyAny>) -> PyResult<Obj> {
    vec_out(py, filter::lfilter_zi(&coeffs(b)?, &coeffs(a)?).map_err(design_error)?)
}

#[pyfunction]
#[pyo3(signature = (sos, x, axis=-1, zi=None))]
fn sosfilt(py: Python<'_>, sos: &Bound<'_, PyAny>, x: &Bound<'_, PyAny>, axis: isize, zi: Option<&Bound<'_, PyAny>>) -> PyResult<Obj> {
    let sos = sos_in(sos)?;
    float_view!(x, T, v => {
        let axis = axis_index(axis, v.ndim())?;
        match zi {
            None => numpy_array(py, filter::sosfilt(&sos, v, axis).map_err(design_error)?),
            Some(zi) => {
                let zi = state_as::<T>(zi)?;
                let (y, zf) = filter::sosfilt_with_state(&sos, v, axis, zi.view()).map_err(design_error)?;
                tuple(py, vec![numpy_array(py, y)?, numpy_array(py, zf)?])
            }
        }
    })
}

#[pyfunction]
fn sosfilt_zi(py: Python<'_>, sos: &Bound<'_, PyAny>) -> PyResult<Obj> {
    let zi = filter::sosfilt_zi(&sos_in(sos)?).map_err(design_error)?;
    let n = zi.len();
    numpy_out(py, zi.concat(), &[n, 2], None)
}

fn pad(padtype: Option<&str>, padlen: Option<usize>) -> PyResult<Pad> {
    Ok(match padtype {
        None => Pad::None,
        Some("odd") => Pad::Odd(padlen),
        Some("even") => Pad::Even(padlen),
        Some("constant") => Pad::Constant(padlen),
        Some(other) => return Err(PyValueError::new_err(format!("unknown padtype {other:?}"))),
    })
}

#[pyfunction]
#[pyo3(signature = (b, a, x, axis=-1, padtype=Some("odd"), padlen=None))]
#[allow(clippy::too_many_arguments)]
fn filtfilt(py: Python<'_>, b: &Bound<'_, PyAny>, a: &Bound<'_, PyAny>, x: &Bound<'_, PyAny>, axis: isize, padtype: Option<&str>, padlen: Option<usize>) -> PyResult<Obj> {
    let (b, a, pad) = (coeffs(b)?, coeffs(a)?, pad(padtype, padlen)?);
    float_view!(x, T, v => {
        let axis = axis_index(axis, v.ndim())?;
        numpy_array(py, filter::filtfilt(&b, &a, v, axis, pad).map_err(design_error)?)
    })
}

#[pyfunction]
#[pyo3(signature = (sos, x, axis=-1, padtype=Some("odd"), padlen=None))]
fn sosfiltfilt(py: Python<'_>, sos: &Bound<'_, PyAny>, x: &Bound<'_, PyAny>, axis: isize, padtype: Option<&str>, padlen: Option<usize>) -> PyResult<Obj> {
    let (sos, pad) = (sos_in(sos)?, pad(padtype, padlen)?);
    float_view!(x, T, v => {
        let axis = axis_index(axis, v.ndim())?;
        numpy_array(py, filter::sosfiltfilt(&sos, v, axis, pad).map_err(design_error)?)
    })
}

// SPECTRAL ESTIMATION =============================================================================

#[allow(clippy::too_many_arguments)]
fn segments(window: &Bound<'_, PyAny>, nperseg: Option<usize>, noverlap: Option<usize>, nfft: Option<usize>, detrend: &str, onesided: bool, scaling: &str) -> PyResult<Segments> {
    Ok(Segments {
        window: window_spec(window)?,
        nperseg,
        noverlap,
        nfft,
        detrend: match detrend {
            "constant" => Detrend::Constant,
            "linear" => Detrend::Linear,
            "none" | "" | "false" => Detrend::None,
            other => return Err(PyValueError::new_err(format!("unknown detrend {other:?}"))),
        },
        onesided,
        scaling: match scaling {
            "density" | "psd" => Scaling::Density,
            "spectrum" => Scaling::Spectrum,
            other => return Err(PyValueError::new_err(format!("unknown scaling {other:?}"))),
        },
    })
}

fn average(name: &str) -> PyResult<Average> {
    match name {
        "mean" => Ok(Average::Mean),
        "median" => Ok(Average::Median),
        other => Err(PyValueError::new_err(format!("unknown average {other:?}"))),
    }
}

fn spectral_error(e: spectral::SpectralError) -> PyErr {
    PyValueError::new_err(e.to_string())
}

#[pyfunction]
#[pyo3(signature = (x, fs, window, nperseg, noverlap, nfft, detrend, return_onesided, scaling, axis, average_by))]
#[allow(clippy::too_many_arguments)]
fn welch(py: Python<'_>, x: &Bound<'_, PyAny>, fs: f64, window: &Bound<'_, PyAny>, nperseg: Option<usize>, noverlap: Option<usize>, nfft: Option<usize>, detrend: &str, return_onesided: bool, scaling: &str, axis: isize, average_by: &str) -> PyResult<Obj> {
    let seg = segments(window, nperseg, noverlap, nfft, detrend, return_onesided, scaling)?;
    float_view!(x, T, v => {
        let axis = axis_index(axis, v.ndim())?;
        let (f, p) = spectral::welch(v, fs, axis, &seg, average(average_by)?).map_err(spectral_error)?;
        tuple(py, vec![vec_out(py, f)?, numpy_array(py, p)?])
    })
}

#[pyfunction]
#[pyo3(signature = (x, y, fs, window, nperseg, noverlap, nfft, detrend, return_onesided, scaling, axis, average_by))]
#[allow(clippy::too_many_arguments)]
fn csd(py: Python<'_>, x: &Bound<'_, PyAny>, y: &Bound<'_, PyAny>, fs: f64, window: &Bound<'_, PyAny>, nperseg: Option<usize>, noverlap: Option<usize>, nfft: Option<usize>, detrend: &str, return_onesided: bool, scaling: &str, axis: isize, average_by: &str) -> PyResult<Obj> {
    let seg = segments(window, nperseg, noverlap, nfft, detrend, return_onesided, scaling)?;
    float_view!(x, T, v => {
        let other = state_as::<T>(y)?;
        let axis = axis_index(axis, v.ndim())?;
        let (f, p) = spectral::csd(v, other.view(), fs, axis, &seg, average(average_by)?).map_err(spectral_error)?;
        tuple(py, vec![vec_out(py, f)?, numpy_complex(py, p)?])
    })
}

#[pyfunction]
#[pyo3(signature = (x, y, fs, window, nperseg, noverlap, nfft, detrend, axis))]
#[allow(clippy::too_many_arguments)]
fn coherence(py: Python<'_>, x: &Bound<'_, PyAny>, y: &Bound<'_, PyAny>, fs: f64, window: &Bound<'_, PyAny>, nperseg: Option<usize>, noverlap: Option<usize>, nfft: Option<usize>, detrend: &str, axis: isize) -> PyResult<Obj> {
    let seg = segments(window, nperseg, noverlap, nfft, detrend, true, "density")?;
    float_view!(x, T, v => {
        let other = state_as::<T>(y)?;
        let axis = axis_index(axis, v.ndim())?;
        let (f, c) = spectral::coherence(v, other.view(), fs, axis, &seg).map_err(spectral_error)?;
        tuple(py, vec![vec_out(py, f)?, numpy_array(py, c)?])
    })
}

#[pyfunction]
#[pyo3(signature = (x, fs, window, nperseg, noverlap, nfft, detrend, return_onesided, scaling, axis, mode))]
#[allow(clippy::too_many_arguments)]
fn spectrogram(py: Python<'_>, x: &Bound<'_, PyAny>, fs: f64, window: &Bound<'_, PyAny>, nperseg: Option<usize>, noverlap: Option<usize>, nfft: Option<usize>, detrend: &str, return_onesided: bool, scaling: &str, axis: isize, mode: &str) -> PyResult<Obj> {
    let seg = segments(window, nperseg, noverlap, nfft, detrend, return_onesided, scaling)?;
    let mode = match mode {
        "psd" => SpectrogramMode::Psd,
        "magnitude" => SpectrogramMode::Magnitude,
        "angle" => SpectrogramMode::Angle,
        "phase" => SpectrogramMode::Phase,
        other => return Err(PyValueError::new_err(format!("unknown mode {other:?}"))),
    };
    float_view!(x, T, v => {
        let axis = axis_index(axis, v.ndim())?;
        let (f, t, s) = spectral::spectrogram(v, fs, axis, &seg, mode).map_err(spectral_error)?;
        tuple(py, vec![vec_out(py, f)?, vec_out(py, t)?, numpy_array(py, s)?])
    })
}

fn boundary(name: Option<&str>) -> PyResult<Boundary> {
    Ok(match name {
        None => Boundary::None,
        Some("zeros") => Boundary::Zeros,
        Some("even") => Boundary::Even,
        Some("odd") => Boundary::Odd,
        Some("constant") => Boundary::Constant,
        Some(other) => return Err(PyValueError::new_err(format!("unknown boundary {other:?}"))),
    })
}

#[pyfunction]
#[pyo3(signature = (x, fs, window, nperseg, noverlap, nfft, detrend, return_onesided, boundary_by, padded, axis, scaling))]
#[allow(clippy::too_many_arguments)]
fn stft(py: Python<'_>, x: &Bound<'_, PyAny>, fs: f64, window: &Bound<'_, PyAny>, nperseg: usize, noverlap: Option<usize>, nfft: Option<usize>, detrend: &str, return_onesided: bool, boundary_by: Option<&str>, padded: bool, axis: isize, scaling: &str) -> PyResult<Obj> {
    let seg = segments(window, Some(nperseg), noverlap, nfft, detrend, return_onesided, scaling)?;
    let options = StftOptions { boundary: boundary(boundary_by)?, padded };
    float_view!(x, T, v => {
        let axis = axis_index(axis, v.ndim())?;
        let (f, t, z) = spectral::stft(v, fs, axis, &seg, options).map_err(spectral_error)?;
        tuple(py, vec![vec_out(py, f)?, vec_out(py, t)?, numpy_complex(py, z)?])
    })
}

/// A complex input (any DLPack producer), as f64 complex.
fn complex_input(obj: &Bound<'_, PyAny>) -> PyResult<NdArray<Complex<f64>>> {
    let tensor = import(obj)?;
    match tensor.dtype() {
        DType::ComplexF64 => Ok(typed::<Complex<f64>>(&tensor)?.to_owned()),
        DType::ComplexF32 => Ok(typed::<Complex<f32>>(&tensor)?.map(|z| Complex::new(z.re as f64, z.im as f64))),
        DType::F64 => Ok(typed::<f64>(&tensor)?.map(|&x| Complex::new(x, 0.0))),
        DType::F32 => Ok(typed::<f32>(&tensor)?.map(|&x| Complex::new(x as f64, 0.0))),
        d => Err(PyTypeError::new_err(format!("expected complex or float values, got {d}"))),
    }
}

#[pyfunction]
#[pyo3(signature = (zxx, fs, window, nperseg, noverlap, nfft, input_onesided, boundary_by, scaling))]
#[allow(clippy::too_many_arguments)]
fn istft(py: Python<'_>, zxx: &Bound<'_, PyAny>, fs: f64, window: &Bound<'_, PyAny>, nperseg: Option<usize>, noverlap: Option<usize>, nfft: Option<usize>, input_onesided: bool, boundary_by: bool, scaling: &str) -> PyResult<Obj> {
    let z = complex_input(zxx)?;
    let scaling = match scaling {
        "spectrum" => Scaling::Spectrum,
        "psd" | "density" => Scaling::Density,
        other => return Err(PyValueError::new_err(format!("unknown scaling {other:?}"))),
    };
    let options = IstftOptions { window: window_spec(window)?, nperseg, noverlap, nfft, onesided: input_onesided, boundary: boundary_by, scaling };
    let (t, x) = spectral::istft(z.view(), fs, options).map_err(spectral_error)?;
    tuple(py, vec![vec_out(py, t)?, numpy_array(py, x)?])
}

// FFT =============================================================================================

/// The DFT (or inverse) along the last axis, any length, complex128 out.
#[pyfunction]
#[pyo3(signature = (x, inverse=false))]
fn fft(py: Python<'_>, x: &Bound<'_, PyAny>, inverse: bool) -> PyResult<Obj> {
    let mut z = complex_input(x)?;
    let n = *z.shape().last().ok_or_else(|| PyValueError::new_err("fft needs at least one axis"))?;
    if n == 0 {
        return Err(PyValueError::new_err("fft needs a non-empty last axis"));
    }
    let mut plan = Fft::<f64>::new(n);
    for row in z.as_mut_slice().chunks_mut(n) {
        if inverse { plan.inverse(row) } else { plan.forward(row) }
    }
    numpy_complex(py, z)
}

/// The inverse real FFT along the last axis to `n` samples (default `2 (m - 1)`).
#[pyfunction]
#[pyo3(signature = (x, n=None))]
fn irfft(py: Python<'_>, x: &Bound<'_, PyAny>, n: Option<usize>) -> PyResult<Obj> {
    let z = complex_input(x)?;
    let m = *z.shape().last().ok_or_else(|| PyValueError::new_err("irfft needs at least one axis"))?;
    let n = n.unwrap_or(2 * m.saturating_sub(1));
    if n == 0 {
        return Err(PyValueError::new_err("irfft needs at least one output sample"));
    }
    let bins = n / 2 + 1;
    let mut plan = autodyne::fft::RealFft::<f64>::new(n);
    let rows = z.len() / m.max(1);
    let mut out = Vec::with_capacity(rows * n);
    let mut spec = vec![Complex::zero(); bins];
    let mut buf = vec![0.0; n];
    for row in z.as_slice().chunks(m) {
        // numpy takes the first n/2 + 1 bins, zero-filling missing ones
        spec.iter_mut().for_each(|s| *s = Complex::zero());
        for (s, v) in spec.iter_mut().zip(row) {
            *s = *v;
        }
        plan.inverse(&spec, &mut buf);
        out.extend_from_slice(&buf);
    }
    let mut shape = z.shape().to_vec();
    *shape.last_mut().expect("an axis") = n;
    numpy_out(py, out, &shape, None)
}

/// Threads for large matrix operations (0: all cores, 1: single-threaded).
#[pyfunction]
fn set_threads(threads: usize) {
    linalg::set_threads(threads);
}

pub(crate) fn register(m: &Bound<'_, PyModule>) -> PyResult<()> {
    for f in [
        wrap_pyfunction!(set_threads, m)?,
        wrap_pyfunction!(matmul, m)?,
        wrap_pyfunction!(solve, m)?,
        wrap_pyfunction!(inv, m)?,
        wrap_pyfunction!(det, m)?,
        wrap_pyfunction!(lstsq, m)?,
        wrap_pyfunction!(eig, m)?,
        wrap_pyfunction!(eigvals, m)?,
        wrap_pyfunction!(eigh, m)?,
        wrap_pyfunction!(svd, m)?,
        wrap_pyfunction!(svdvals, m)?,
        wrap_pyfunction!(pinv, m)?,
        wrap_pyfunction!(qr, m)?,
        wrap_pyfunction!(cholesky, m)?,
        wrap_pyfunction!(expm, m)?,
        wrap_pyfunction!(roots, m)?,
        wrap_pyfunction!(iirfilter, m)?,
        wrap_pyfunction!(firwin, m)?,
        wrap_pyfunction!(firwin2, m)?,
        wrap_pyfunction!(firls, m)?,
        wrap_pyfunction!(remez, m)?,
        wrap_pyfunction!(kaiserord, m)?,
        wrap_pyfunction!(get_window, m)?,
        wrap_pyfunction!(zpk2sos, m)?,
        wrap_pyfunction!(tf2zpk, m)?,
        wrap_pyfunction!(freqz, m)?,
        wrap_pyfunction!(sosfreqz, m)?,
        wrap_pyfunction!(group_delay, m)?,
        wrap_pyfunction!(lfilter, m)?,
        wrap_pyfunction!(lfilter_zi, m)?,
        wrap_pyfunction!(sosfilt, m)?,
        wrap_pyfunction!(sosfilt_zi, m)?,
        wrap_pyfunction!(filtfilt, m)?,
        wrap_pyfunction!(sosfiltfilt, m)?,
        wrap_pyfunction!(welch, m)?,
        wrap_pyfunction!(csd, m)?,
        wrap_pyfunction!(coherence, m)?,
        wrap_pyfunction!(spectrogram, m)?,
        wrap_pyfunction!(stft, m)?,
        wrap_pyfunction!(istft, m)?,
        wrap_pyfunction!(fft, m)?,
        wrap_pyfunction!(irfft, m)?,
    ] {
        m.add_function(f)?;
    }
    Ok(())
}
