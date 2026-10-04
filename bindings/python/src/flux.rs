//! `autodyne.flux` from Python: trace functions written with [`Tracer`] operations (NumPy-style
//! operators and methods), differentiate them, run scans and losses, and compile the result for
//! IREE or a PJRT plugin (XLA's, for instance). The trace, its derivatives and the programs are all built in Rust;
//! Python only describes the computation once.

use std::cell::RefCell;

use autodyne::distortion::Shape;
use autodyne::dynamics::{envelope_step, time_coeff, CompressorCurve};
use autodyne::filter::{BiquadCoeffs, BiquadKind, LadderCoeffs, OnePole, SvfCoeffs, SvfMode};
use autodyne::flux::{self as fx, Backend, Emit, Executable, ExecutableExt, FluxFloat, Graph, Iree, IreeTarget, Loss, Mask, Pjrt, PjrtOption, Program, Scan, StftResolution, Tracer};
use autodyne::signal::{ArrayMath, ComplexArrayMath, NdArray, NdView, RealArrayMath};
use autodyne::units::*;
use pyo3::exceptions::{PyIndexError, PyTypeError, PyValueError};
use pyo3::prelude::*;
use pyo3::types::{PyDict, PyList, PySlice, PyTuple};

use crate::{input, numpy_array, typed, runtime_error, value_error, Input};

type Obj = Py<PyAny>;

// TRACERS =========================================================================================

/// A traced array: operations on it are recorded (only valid inside the trace that made it).
#[pyclass(name = "Tracer", module = "autodyne.flux", unsendable, skip_from_py_object)]
#[derive(Clone, Copy)]
struct PyTracer(Tracer);

/// A traced comparison, for `flux.where`.
#[pyclass(name = "Mask", module = "autodyne.flux", unsendable, skip_from_py_object)]
#[derive(Clone, Copy)]
struct PyMask(Mask);

fn wrap(py: Python<'_>, t: Tracer) -> PyResult<Obj> {
    Ok(Py::new(py, PyTracer(t))?.into_any())
}

/// A tracer, a number (a constant) or an array (a constant array).
fn tr(x: &Bound<'_, PyAny>) -> PyResult<Tracer> {
    if let Ok(t) = x.cast::<PyTracer>() {
        return Ok(t.borrow().0);
    }
    if let Ok(v) = x.extract::<f64>() {
        return Ok(Tracer::lit(v));
    }
    if x.is_instance_of::<pyo3::types::PyString>() {
        return Err(PyTypeError::new_err("expected a tracer, a number or an array, not a string"));
    }
    let (a, _) = host(x)?;
    Ok(Tracer::constant(&a))
}

fn axes(axis: Option<&Bound<'_, PyAny>>, ndim: usize) -> PyResult<Vec<usize>> {
    let norm = |a: isize| crate::axis_index(a, ndim);
    match axis {
        None => Ok((0..ndim).collect()),
        Some(a) => match a.extract::<isize>() {
            Ok(a) => Ok(vec![norm(a)?]),
            Err(_) => a.extract::<Vec<isize>>()?.into_iter().map(norm).collect(),
        },
    }
}

#[pymethods]
impl PyTracer {
    #[getter]
    fn shape(&self) -> Vec<usize> {
        self.0.shape()
    }
    #[getter]
    fn ndim(&self) -> usize {
        self.0.shape().len()
    }
    fn __repr__(&self) -> String {
        format!("Tracer(shape={:?})", self.0.shape())
    }

    fn __add__(&self, o: &Bound<'_, PyAny>) -> PyResult<Self> {
        Ok(PyTracer(self.0 + tr(o)?))
    }
    fn __radd__(&self, o: &Bound<'_, PyAny>) -> PyResult<Self> {
        Ok(PyTracer(tr(o)? + self.0))
    }
    fn __sub__(&self, o: &Bound<'_, PyAny>) -> PyResult<Self> {
        Ok(PyTracer(self.0 - tr(o)?))
    }
    fn __rsub__(&self, o: &Bound<'_, PyAny>) -> PyResult<Self> {
        Ok(PyTracer(tr(o)? - self.0))
    }
    fn __mul__(&self, o: &Bound<'_, PyAny>) -> PyResult<Self> {
        Ok(PyTracer(self.0 * tr(o)?))
    }
    fn __rmul__(&self, o: &Bound<'_, PyAny>) -> PyResult<Self> {
        Ok(PyTracer(tr(o)? * self.0))
    }
    fn __truediv__(&self, o: &Bound<'_, PyAny>) -> PyResult<Self> {
        Ok(PyTracer(self.0 / tr(o)?))
    }
    fn __rtruediv__(&self, o: &Bound<'_, PyAny>) -> PyResult<Self> {
        Ok(PyTracer(tr(o)? / self.0))
    }
    fn __pow__(&self, o: &Bound<'_, PyAny>, _modulo: Option<&Bound<'_, PyAny>>) -> PyResult<Self> {
        Ok(PyTracer(self.0.powf(tr(o)?)))
    }
    fn __rpow__(&self, o: &Bound<'_, PyAny>, _modulo: Option<&Bound<'_, PyAny>>) -> PyResult<Self> {
        Ok(PyTracer(tr(o)?.powf(self.0)))
    }
    fn __matmul__(&self, o: &Bound<'_, PyAny>) -> PyResult<Self> {
        Ok(PyTracer(self.0.dot(tr(o)?)))
    }
    fn __neg__(&self) -> Self {
        PyTracer(-self.0)
    }
    fn __pos__(&self) -> Self {
        *self
    }
    fn __abs__(&self) -> Self {
        PyTracer(self.0.abs())
    }
    fn __lt__(&self, o: &Bound<'_, PyAny>) -> PyResult<PyMask> {
        Ok(PyMask(self.0.less(tr(o)?)))
    }
    fn __gt__(&self, o: &Bound<'_, PyAny>) -> PyResult<PyMask> {
        Ok(PyMask(self.0.greater(tr(o)?)))
    }

    fn exp(&self) -> Self {
        PyTracer(self.0.exp())
    }
    fn log(&self) -> Self {
        PyTracer(self.0.ln())
    }
    fn log10(&self) -> Self {
        PyTracer(self.0.log10())
    }
    fn sin(&self) -> Self {
        PyTracer(self.0.sin())
    }
    fn cos(&self) -> Self {
        PyTracer(self.0.cos())
    }
    fn tan(&self) -> Self {
        PyTracer(self.0.tan())
    }
    fn tanh(&self) -> Self {
        PyTracer(self.0.tanh())
    }
    fn sqrt(&self) -> Self {
        PyTracer(self.0.sqrt())
    }
    fn abs(&self) -> Self {
        PyTracer(self.0.abs())
    }
    fn floor(&self) -> Self {
        PyTracer(self.0.floor())
    }
    fn minimum(&self, o: &Bound<'_, PyAny>) -> PyResult<Self> {
        Ok(PyTracer(self.0.minimum(tr(o)?)))
    }
    fn maximum(&self, o: &Bound<'_, PyAny>) -> PyResult<Self> {
        Ok(PyTracer(self.0.maximum(tr(o)?)))
    }
    fn clip(&self, lo: &Bound<'_, PyAny>, hi: &Bound<'_, PyAny>) -> PyResult<Self> {
        Ok(PyTracer(self.0.clip(tr(lo)?, tr(hi)?)))
    }

    #[pyo3(signature = (*shape))]
    fn reshape(&self, shape: &Bound<'_, PyTuple>) -> PyResult<Self> {
        let shape: Vec<usize> = if shape.len() == 1 && shape.get_item(0)?.extract::<usize>().is_err() { shape.get_item(0)?.extract()? } else { shape.extract()? };
        Ok(PyTracer(self.0.reshape(&shape)))
    }
    #[pyo3(signature = (*perm))]
    fn transpose(&self, perm: Vec<usize>) -> Self {
        let n = self.0.shape().len();
        let perm = if perm.is_empty() { (0..n).rev().collect() } else { perm };
        PyTracer(self.0.transpose(&perm))
    }
    #[getter(T)]
    fn t(&self) -> Self {
        self.transpose(Vec::new())
    }
    #[pyo3(signature = (axis=None))]
    fn sum(&self, axis: Option<&Bound<'_, PyAny>>) -> PyResult<Self> {
        Ok(PyTracer(self.0.sum_axes(&axes(axis, self.ndim())?)))
    }
    #[pyo3(signature = (axis=None))]
    fn mean(&self, axis: Option<&Bound<'_, PyAny>>) -> PyResult<Self> {
        let axes = axes(axis, self.ndim())?;
        let shape = self.0.shape();
        let count: usize = axes.iter().map(|&a| shape[a]).product();
        Ok(PyTracer(self.0.sum_axes(&axes) / Tracer::lit(count as f64)))
    }
    #[pyo3(signature = (axis=None))]
    fn max(&self, axis: Option<&Bound<'_, PyAny>>) -> PyResult<Self> {
        Ok(PyTracer(self.0.max_axes(&axes(axis, self.ndim())?)))
    }
    #[pyo3(signature = (axis=None))]
    fn min(&self, axis: Option<&Bound<'_, PyAny>>) -> PyResult<Self> {
        Ok(PyTracer(self.0.min_axes(&axes(axis, self.ndim())?)))
    }
    #[pyo3(signature = (axis=None))]
    fn prod(&self, axis: Option<&Bound<'_, PyAny>>) -> PyResult<Self> {
        Ok(PyTracer(self.0.prod_axes(&axes(axis, self.ndim())?)))
    }
    fn dot(&self, o: &Bound<'_, PyAny>) -> PyResult<Self> {
        Ok(PyTracer(self.0.dot(tr(o)?)))
    }
    /// Whether the values are complex (spectra).
    #[getter]
    fn is_complex(&self) -> bool {
        self.0.is_complex()
    }
    /// The real part.
    #[getter]
    fn real(&self) -> Self {
        PyTracer(self.0.re())
    }
    /// The imaginary part.
    #[getter]
    fn imag(&self) -> Self {
        PyTracer(self.0.im())
    }
    fn conj(&self) -> Self {
        PyTracer(self.0.conj())
    }
    /// The real FFT along the last axis as complex values (`n / 2 + 1` bins).
    fn rfft_complex(&self) -> Self {
        PyTracer(self.0.rfft_complex())
    }
    /// The complex FFT along the last axis (a real tracer is promoted).
    #[pyo3(signature = (inverse=false))]
    fn fft(&self, inverse: bool) -> Self {
        PyTracer(if inverse { self.0.ifft() } else { self.0.fft() })
    }
    /// The real FFT along the last axis: `(real parts, imaginary parts)`.
    fn rfft(&self) -> (Self, Self) {
        let (re, im) = self.0.rfft();
        (PyTracer(re), PyTracer(im))
    }
    /// Rows along the first axis at real `indices` (rounded down, clamped).
    fn take(&self, indices: &Bound<'_, PyAny>) -> PyResult<Self> {
        Ok(PyTracer(self.0.take(tr(indices)?)))
    }
    #[pyo3(signature = (axis=None))]
    fn reverse(&self, axis: Option<&Bound<'_, PyAny>>) -> PyResult<Self> {
        Ok(PyTracer(self.0.reverse(&axes(axis, self.ndim())?)))
    }
    /// Zero padding: `low[i]` before axis `i`, `high[i]` after, `interior[i]` between elements.
    #[pyo3(signature = (low, high, interior=None))]
    fn pad(&self, low: Vec<usize>, high: Vec<usize>, interior: Option<Vec<usize>>) -> Self {
        let interior = interior.unwrap_or_else(|| vec![0; low.len()]);
        PyTracer(self.0.pad(&low, &high, &interior))
    }
    fn broadcast_to(&self, shape: Vec<usize>) -> Self {
        PyTracer(self.0.broadcast_to(&shape))
    }

    /// Static indexing: integers and slices with positive steps, one per leading axis.
    fn __getitem__(&self, key: &Bound<'_, PyAny>) -> PyResult<Self> {
        let shape = self.0.shape();
        let items: Vec<Bound<'_, PyAny>> = match key.cast::<PyTuple>() {
            Ok(t) => t.iter().collect(),
            Err(_) => vec![key.clone()],
        };
        if items.len() > shape.len() {
            return Err(PyIndexError::new_err(format!("too many indices for shape {shape:?}")));
        }
        let (mut start, mut limit, mut stride) = (vec![0; shape.len()], shape.clone(), vec![1; shape.len()]);
        let mut dropped = Vec::new();
        for (axis, item) in items.iter().enumerate() {
            let n = shape[axis] as isize;
            if let Ok(s) = item.cast::<PySlice>() {
                let ix = s.indices(n)?;
                if ix.step <= 0 {
                    return Err(PyValueError::new_err("only positive slice steps (use reverse())"));
                }
                start[axis] = ix.start as usize;
                limit[axis] = (ix.stop.max(ix.start)) as usize;
                stride[axis] = ix.step as usize;
            } else {
                let i: isize = item.extract().map_err(|_| PyTypeError::new_err("indices must be integers or slices"))?;
                let i = if i < 0 { i + n } else { i };
                if i < 0 || i >= n {
                    return Err(PyIndexError::new_err(format!("index out of range for axis {axis} of {shape:?}")));
                }
                (start[axis], limit[axis]) = (i as usize, i as usize + 1);
                dropped.push(axis);
            }
        }
        let sliced = self.0.slice(&start, &limit, &stride);
        let kept: Vec<usize> = sliced.shape().iter().enumerate().filter(|(a, _)| !dropped.contains(a)).map(|(_, &d)| d).collect();
        Ok(PyTracer(sliced.reshape(&kept)))
    }
}

// FUNCTIONS ON TRACERS ============================================================================

/// `if_true` where `mask` holds, else `if_false`.
#[pyfunction(name = "where")]
fn where_(mask: PyRef<'_, PyMask>, if_true: &Bound<'_, PyAny>, if_false: &Bound<'_, PyAny>) -> PyResult<PyTracer> {
    Ok(PyTracer(Tracer::select(mask.0, tr(if_true)?, tr(if_false)?)))
}

/// Joins tracers along `axis` (`numpy.concatenate`).
/// A backend's error: wrong shapes are the caller's (ValueError), the rest the environment's.
fn flux_error(e: autodyne::flux::FluxError) -> PyErr {
    match e {
        autodyne::flux::FluxError::Shape(_) => value_error(e),
        _ => runtime_error(e),
    }
}

#[pyfunction]
#[pyo3(signature = (parts, axis=0))]
fn concatenate(parts: Vec<Bound<'_, PyAny>>, axis: usize) -> PyResult<PyTracer> {
    let parts = parts.iter().map(tr).collect::<PyResult<Vec<_>>>()?;
    Ok(PyTracer(Tracer::concatenate(&parts, axis)))
}

/// The inverse real FFT along the last axis from `(real, imaginary)` bins: `n` samples, scaled by `1 / n`.
#[pyfunction]
fn irfft(re: &Bound<'_, PyAny>, im: &Bound<'_, PyAny>, n: usize) -> PyResult<PyTracer> {
    Ok(PyTracer(Tracer::irfft(tr(re)?, tr(im)?, n)))
}

/// `n` real samples from complex bins (`n / 2 + 1` along the last axis).
#[pyfunction]
fn irfft_complex(spectrum: &Bound<'_, PyAny>, n: usize) -> PyResult<PyTracer> {
    Ok(PyTracer(Tracer::irfft_complex(tr(spectrum)?, n)))
}

/// `re + i im`.
#[pyfunction]
fn complex(re: &Bound<'_, PyAny>, im: &Bound<'_, PyAny>) -> PyResult<PyTracer> {
    Ok(PyTracer(Tracer::complex(tr(re)?, tr(im)?)))
}

/// The complex DFT along the last axis of `re + i im`: `(real parts, imaginary parts)`.
#[pyfunction]
#[pyo3(signature = (re, im, inverse=false))]
fn fft(re: &Bound<'_, PyAny>, im: &Bound<'_, PyAny>, inverse: bool) -> PyResult<(PyTracer, PyTracer)> {
    let (a, b) = if inverse { Tracer::ifft_parts(tr(re)?, tr(im)?) } else { Tracer::fft_parts(tr(re)?, tr(im)?) };
    Ok((PyTracer(a), PyTracer(b)))
}

/// Full linear convolution along the last axis (`numpy.convolve` mode `full`), batched over the leading axes.
#[pyfunction]
fn convolve(x: &Bound<'_, PyAny>, kernel: &Bound<'_, PyAny>) -> PyResult<PyTracer> {
    Ok(PyTracer(autodyne::signal::convolve(tr(x)?, tr(kernel)?)))
}

/// Windows of `length` samples, `hop` apart, along the last axis: `[..., n]` becomes `[..., count, length]`.
#[pyfunction]
fn frames(x: &Bound<'_, PyAny>, length: usize, hop: usize) -> PyResult<PyTracer> {
    Ok(PyTracer(autodyne::signal::frames(tr(x)?, length, hop)))
}

fn resolutions(r: Option<Vec<(usize, usize, usize)>>) -> Vec<StftResolution> {
    r.map(|r| r.into_iter().map(|(n, h, w)| StftResolution::new(n, h, w)).collect()).unwrap_or_else(|| StftResolution::DEFAULT.to_vec())
}

/// STFT magnitudes along the last axis: Hann frames of `window` samples, `hop` apart, zero-padded to `n_fft`.
#[pyfunction]
fn stft_magnitude(x: &Bound<'_, PyAny>, n_fft: usize, hop: usize, window: usize) -> PyResult<PyTracer> {
    Ok(PyTracer(fx::stft_magnitude(tr(x)?, StftResolution::new(n_fft, hop, window))))
}

/// The multi-resolution STFT loss; `resolutions` are `(n_fft, hop, window)` triples.
#[pyfunction]
#[pyo3(signature = (y, target, resolutions=None))]
fn multi_resolution_stft(y: &Bound<'_, PyAny>, target: &Bound<'_, PyAny>, resolutions: Option<Vec<(usize, usize, usize)>>) -> PyResult<PyTracer> {
    Ok(PyTracer(fx::multi_resolution_stft(tr(y)?, tr(target)?, &self::resolutions(resolutions))))
}

// PROCESSOR STEPS =================================================================================

fn parse<T: Copy>(name: &str, table: &[(&str, T)]) -> PyResult<T> {
    table.iter().find(|(n, _)| *n == name).map(|&(_, v)| v).ok_or_else(|| {
        PyValueError::new_err(format!("unknown kind {name:?}; expected one of {}", table.iter().map(|(n, _)| *n).collect::<Vec<_>>().join(", ")))
    })
}

/// One sample of a one-pole low-pass: `(next state, output)`.
#[pyfunction]
#[pyo3(signature = (cutoff, state, x, sample_rate=48_000.0))]
fn one_pole(cutoff: &Bound<'_, PyAny>, state: &Bound<'_, PyAny>, x: &Bound<'_, PyAny>, sample_rate: f64) -> PyResult<(PyTracer, PyTracer)> {
    let (s, y) = OnePole::lowpass(tr(cutoff)?, Tracer::lit(sample_rate)).tick(tr(state)?, tr(x)?);
    Ok((PyTracer(s), PyTracer(y)))
}

/// One sample of an RBJ-cookbook biquad (`kind`: lowpass, highpass, bandpass, notch, allpass,
/// peaking, lowshelf, highshelf): `((state, state), output)`.
#[pyfunction]
#[pyo3(signature = (kind, frequency, q, gain_db, state, x, sample_rate=48_000.0))]
fn biquad(
    kind: &str,
    frequency: &Bound<'_, PyAny>,
    q: &Bound<'_, PyAny>,
    gain_db: &Bound<'_, PyAny>,
    state: (Bound<'_, PyAny>, Bound<'_, PyAny>),
    x: &Bound<'_, PyAny>,
    sample_rate: f64,
) -> PyResult<((PyTracer, PyTracer), PyTracer)> {
    let kind = parse(
        kind,
        &[
            ("lowpass", BiquadKind::Lowpass),
            ("highpass", BiquadKind::Highpass),
            ("bandpass", BiquadKind::Bandpass),
            ("notch", BiquadKind::Notch),
            ("allpass", BiquadKind::Allpass),
            ("peaking", BiquadKind::Peaking),
            ("lowshelf", BiquadKind::LowShelf),
            ("highshelf", BiquadKind::HighShelf),
        ],
    )?;
    let c = BiquadCoeffs::design(kind, tr(frequency)?, tr(q)?, tr(gain_db)?, Tracer::lit(sample_rate));
    let ([a, b], y) = c.tick([tr(&state.0)?, tr(&state.1)?], tr(x)?);
    Ok(((PyTracer(a), PyTracer(b)), PyTracer(y)))
}

/// One sample of a state-variable filter (`mode`: lowpass, bandpass, highpass, notch, peak,
/// allpass): `((state, state), output)`.
#[pyfunction]
#[pyo3(signature = (mode, cutoff, q, state, x, sample_rate=48_000.0))]
fn svf(mode: &str, cutoff: &Bound<'_, PyAny>, q: &Bound<'_, PyAny>, state: (Bound<'_, PyAny>, Bound<'_, PyAny>), x: &Bound<'_, PyAny>, sample_rate: f64) -> PyResult<((PyTracer, PyTracer), PyTracer)> {
    let mode = parse(
        mode,
        &[("lowpass", SvfMode::Lowpass), ("bandpass", SvfMode::Bandpass), ("highpass", SvfMode::Highpass), ("notch", SvfMode::Notch), ("peak", SvfMode::Peak), ("allpass", SvfMode::Allpass)],
    )?;
    let c = SvfCoeffs::new(tr(cutoff)?, tr(q)?, Tracer::lit(sample_rate));
    let ([a, b], o) = c.tick([tr(&state.0)?, tr(&state.1)?], tr(x)?);
    Ok(((PyTracer(a), PyTracer(b)), PyTracer(mode.output(o, c.k))))
}

/// One sample of the four-pole ladder low-pass (`state`: four values): `(state, output)`.
#[pyfunction]
#[pyo3(signature = (cutoff, resonance, drive, state, x, sample_rate=48_000.0, compensate=true))]
fn ladder(
    cutoff: &Bound<'_, PyAny>,
    resonance: &Bound<'_, PyAny>,
    drive: &Bound<'_, PyAny>,
    state: Vec<Bound<'_, PyAny>>,
    x: &Bound<'_, PyAny>,
    sample_rate: f64,
    compensate: bool,
) -> PyResult<(Vec<PyTracer>, PyTracer)> {
    let s: Vec<Tracer> = state.iter().map(tr).collect::<PyResult<_>>()?;
    let s: [Tracer; 4] = s.try_into().map_err(|_| PyValueError::new_err("the ladder's state is four values"))?;
    let c = LadderCoeffs::new(tr(cutoff)?, tr(resonance)?, tr(drive)?, compensate, Tracer::lit(sample_rate));
    let (next, y) = c.tick(s, tr(x)?);
    Ok((next.into_iter().map(PyTracer).collect(), PyTracer(y)))
}

/// A waveshaper (`kind`: tanh, softclip, hardclip, fold).
#[pyfunction]
fn shape(kind: &str, x: &Bound<'_, PyAny>) -> PyResult<PyTracer> {
    let s = parse(kind, &[("tanh", Shape::Tanh), ("softclip", Shape::SoftClip), ("hardclip", Shape::HardClip), ("fold", Shape::Fold)])?;
    Ok(PyTracer(s.apply(tr(x)?)))
}

/// One sample of a peak envelope follower (attack and release in seconds): the next envelope.
#[pyfunction]
#[pyo3(signature = (attack, release, envelope, x, sample_rate=48_000.0))]
fn envelope(attack: &Bound<'_, PyAny>, release: &Bound<'_, PyAny>, envelope: &Bound<'_, PyAny>, x: &Bound<'_, PyAny>, sample_rate: f64) -> PyResult<PyTracer> {
    let fs = Tracer::lit(sample_rate);
    Ok(PyTracer(envelope_step(time_coeff(tr(attack)?, fs), time_coeff(tr(release)?, fs), tr(envelope)?, tr(x)?)))
}

/// One sample of a compressor's gain computer on a level (`|x|`): `(next gain reduction in dB,
/// linear gain)`.
#[pyfunction]
#[pyo3(signature = (threshold_db, slope, knee_db, attack, release, reduction, level, sample_rate=48_000.0))]
#[allow(clippy::too_many_arguments)]
fn compressor(
    threshold_db: &Bound<'_, PyAny>,
    slope: &Bound<'_, PyAny>,
    knee_db: &Bound<'_, PyAny>,
    attack: &Bound<'_, PyAny>,
    release: &Bound<'_, PyAny>,
    reduction: &Bound<'_, PyAny>,
    level: &Bound<'_, PyAny>,
    sample_rate: f64,
) -> PyResult<(PyTracer, PyTracer)> {
    let fs = Tracer::lit(sample_rate);
    let curve = CompressorCurve { threshold_db: tr(threshold_db)?, slope: tr(slope)?, knee_db: tr(knee_db)?, attack_coeff: time_coeff(tr(attack)?, fs), release_coeff: time_coeff(tr(release)?, fs) };
    let (r, g) = curve.tick(tr(reduction)?, tr(level)?);
    Ok((PyTracer(r), PyTracer(g)))
}

// HOST ARRAYS =====================================================================================

/// Any float array, number or nested list as f64, with the array's own float type if it had one.
fn host(x: &Bound<'_, PyAny>) -> PyResult<(NdArray<f64>, Option<DType>)> {
    if let Ok(v) = x.extract::<f64>() {
        return Ok((NdArray::from_vec(vec![v], &[]).map_err(value_error)?, None));
    }
    let numpy = x.py().import("numpy")?;
    let array = if x.hasattr("__dlpack__")? { x.clone() } else { numpy.call_method1("asarray", (x, "float64"))? };
    let input = input(&array)?;
    crate::with_float_input!(input, T, view => {
        let dtype = if std::mem::size_of::<T>() == 8 { DType::F64 } else { DType::F32 };
        let data: Vec<f64> = view.iter().map(|v| v.to_f64().unwrap_or(f64::NAN)).collect();
        Ok((NdArray::from_vec(data, view.shape()).map_err(value_error)?, Some(dtype)))
    })
}

/// Arrays of one precision: f64 if any array is float64, else f32 if any is float32, else f64.
enum Batch {
    F32(Vec<NdArray<f32>>),
    F64(Vec<NdArray<f64>>),
}

fn batch(items: &[Bound<'_, PyAny>], force: Option<DType>) -> PyResult<Batch> {
    let hosts = items.iter().map(host).collect::<PyResult<Vec<_>>>()?;
    let wide = match force {
        Some(d) => d == DType::F64,
        None => hosts.iter().any(|(_, d)| *d == Some(DType::F64)) || !hosts.iter().any(|(_, d)| *d == Some(DType::F32)),
    };
    Ok(if wide {
        Batch::F64(hosts.into_iter().map(|(a, _)| a).collect())
    } else {
        Batch::F32(hosts.into_iter().map(|(a, _)| a.map(|&v| v as f32)).collect())
    })
}

fn numpy_list<T: FluxFloat + numpy::Element>(py: Python<'_>, arrays: Vec<NdArray<T>>) -> PyResult<Vec<Obj>> {
    arrays.into_iter().map(|a| numpy_array(py, a)).collect()
}

// TRACING =========================================================================================

/// The tracers a Python function returned (one, or a tuple / list).
fn outputs(result: &Bound<'_, PyAny>) -> PyResult<Vec<Tracer>> {
    if let Ok(t) = result.cast::<PyTracer>() {
        return Ok(vec![t.borrow().0]);
    }
    if result.is_instance_of::<pyo3::types::PyString>() {
        return Err(PyTypeError::new_err("a traced function must return tracers, not a string"));
    }
    let items: Vec<Bound<'_, PyAny>> = result.try_iter().map_err(|_| PyTypeError::new_err("a traced function must return tracers"))?.collect::<PyResult<_>>()?;
    items.iter().map(tr).collect()
}

/// Runs `f` inside a trace, keeping the first Python error to raise after the trace ends.
fn traced<R>(error: &RefCell<Option<PyErr>>, f: impl FnOnce() -> PyResult<R>, fallback: R) -> R {
    match f() {
        Ok(r) => r,
        Err(e) => {
            error.borrow_mut().get_or_insert(e);
            fallback
        }
    }
}

fn tracers(py: Python<'_>, ts: &[Tracer]) -> PyResult<Vec<Obj>> {
    ts.iter().map(|&t| wrap(py, t)).collect()
}

/// A traced function: evaluate it, or emit it as a program.
#[pyclass(name = "Graph", module = "autodyne.flux", unsendable)]
struct PyGraph(Graph);

#[pymethods]
impl PyGraph {
    /// Evaluates the graph (in float64 if any input is float64, else float32).
    #[pyo3(signature = (*inputs))]
    fn __call__(&self, py: Python<'_>, inputs: Vec<Bound<'_, PyAny>>) -> PyResult<Vec<Obj>> {
        self.check(&inputs)?;
        match batch(&inputs, None)? {
            Batch::F32(a) => numpy_list(py, self.0.eval(&a)),
            Batch::F64(a) => numpy_list(py, self.0.eval(&a)),
        }
    }
    /// The StableHLO program, in `dtype` ("float32" or "float64"), with FFTs of at most `max_fft`
    /// points if given (see `Backend.max_fft`).
    #[pyo3(signature = (dtype="float32", max_fft=None))]
    fn program(&self, dtype: &str, max_fft: Option<usize>) -> PyResult<PyProgram> {
        Ok(PyProgram(self.0.program_with(&emit(dtype, max_fft)?)))
    }
    #[getter]
    fn input_shapes(&self) -> Vec<Vec<usize>> {
        self.0.inputs().to_vec()
    }
    #[getter]
    fn output_shapes(&self) -> Vec<Vec<usize>> {
        self.0.output_shapes()
    }
    fn __str__(&self) -> String {
        self.0.to_string()
    }
}

impl PyGraph {
    fn check(&self, inputs: &[Bound<'_, PyAny>]) -> PyResult<()> {
        if inputs.len() != self.0.inputs().len() {
            return Err(PyValueError::new_err(format!("the graph takes {} inputs, got {}", self.0.inputs().len(), inputs.len())));
        }
        Ok(())
    }
}

fn emit(dtype: &str, max_fft: Option<usize>) -> PyResult<Emit> {
    let e = if wide(dtype)? { Emit::f64() } else { Emit::f32() };
    match max_fft {
        Some(n) if n < 2 => Err(PyValueError::new_err("max_fft must be at least 2")),
        Some(n) => Ok(e.max_fft(n)),
        None => Ok(e),
    }
}

fn wide(dtype: &str) -> PyResult<bool> {
    match dtype {
        "float32" | "f32" => Ok(false),
        "float64" | "f64" => Ok(true),
        other => Err(PyValueError::new_err(format!("dtype must be float32 or float64, not {other:?}"))),
    }
}

/// Traces `f` (called with one tracer per input shape) into a graph.
#[pyfunction]
fn trace(py: Python<'_>, f: &Bound<'_, PyAny>, shapes: Vec<Vec<usize>>) -> PyResult<PyGraph> {
    let refs: Vec<&[usize]> = shapes.iter().map(Vec::as_slice).collect();
    let error = RefCell::new(None);
    let graph = fx::trace(&refs, |v| traced(&error, || outputs(&f.call1(PyTuple::new(py, tracers(py, v)?)?)?), Vec::new()));
    match error.into_inner() {
        Some(e) => Err(e),
        None => Ok(PyGraph(graph)),
    }
}

/// Traces a scalar function `f` with its gradient: the graph returns `(value, d/d input...)` for
/// the inputs numbered in `argnums` (all by default).
#[pyfunction]
#[pyo3(signature = (f, shapes, argnums=None))]
fn value_and_grad(py: Python<'_>, f: &Bound<'_, PyAny>, shapes: Vec<Vec<usize>>, argnums: Option<Vec<usize>>) -> PyResult<PyGraph> {
    let refs: Vec<&[usize]> = shapes.iter().map(Vec::as_slice).collect();
    let error = RefCell::new(None);
    let graph = fx::trace(&refs, |v| {
        traced(
            &error,
            || {
                let out = outputs(&f.call1(PyTuple::new(py, tracers(py, v)?)?)?)?;
                let [value] = out[..] else { return Err(PyValueError::new_err("value_and_grad: the function must return one scalar")) };
                if !value.shape().is_empty() {
                    return Err(PyValueError::new_err(format!("value_and_grad: the function must return a scalar, not {:?}", value.shape())));
                }
                let wrt: Vec<Tracer> = match &argnums {
                    Some(a) => a.iter().map(|&i| v.get(i).copied().ok_or_else(|| PyIndexError::new_err(format!("argnum {i} out of range")))).collect::<PyResult<_>>()?,
                    None => v.to_vec(),
                };
                let grads = fx::vjp(&[value], &[Tracer::lit(1.0)], &wrt);
                Ok(std::iter::once(value).chain(grads).collect())
            },
            Vec::new(),
        )
    });
    match error.into_inner() {
        Some(e) => Err(e),
        None => Ok(PyGraph(graph)),
    }
}

// SCANS AND LOSSES ================================================================================

/// A loss over a scan's whole output, with its gradient.
#[pyclass(name = "Loss", module = "autodyne.flux", unsendable)]
struct PyLoss(Loss);

#[pymethods]
impl PyLoss {
    /// Mean squared error against a target of `shape`.
    #[staticmethod]
    fn mse(shape: Vec<usize>) -> Self {
        PyLoss(Loss::mse(&shape))
    }
    /// The multi-resolution STFT loss against a target of `shape` (`[len, channels...]`);
    /// `resolutions` are `(n_fft, hop, window)` triples.
    #[staticmethod]
    #[pyo3(signature = (shape, resolutions=None))]
    fn stft(shape: Vec<usize>, resolutions: Option<Vec<(usize, usize, usize)>>) -> Self {
        PyLoss(Loss::stft(&shape, &self::resolutions(resolutions)))
    }
    /// `f(output, *aux)`, returning a scalar tracer.
    #[staticmethod]
    #[pyo3(name = "trace")]
    fn trace_fn(py: Python<'_>, f: &Bound<'_, PyAny>, output: Vec<usize>, aux: Vec<Vec<usize>>) -> PyResult<Self> {
        let refs: Vec<&[usize]> = aux.iter().map(Vec::as_slice).collect();
        let error = RefCell::new(None);
        let loss = Loss::trace(&output, &refs, |y, a| {
            traced(
                &error,
                || {
                    let args: Vec<Obj> = std::iter::once(wrap(py, y)).chain(a.iter().map(|&t| wrap(py, t))).collect::<PyResult<_>>()?;
                    let out = outputs(&f.call1(PyTuple::new(py, args)?)?)?;
                    out.first().copied().ok_or_else(|| PyValueError::new_err("the loss must return a scalar"))
                },
                Tracer::lit(0.0),
            )
        });
        match error.into_inner() {
            Some(e) => Err(e),
            None => Ok(PyLoss(loss)),
        }
    }
    /// `(loss, d loss / d output)`.
    #[pyo3(signature = (output, *aux))]
    fn grad(&self, py: Python<'_>, output: &Bound<'_, PyAny>, aux: Vec<Bound<'_, PyAny>>) -> PyResult<(f64, Obj)> {
        let items: Vec<Bound<'_, PyAny>> = std::iter::once(output.clone()).chain(aux).collect();
        match batch(&items, None)? {
            Batch::F32(mut a) => {
                let out = a.remove(0);
                let (v, d) = self.0.grad(&out, &a);
                Ok((v as f64, numpy_array(py, d)?))
            }
            Batch::F64(mut a) => {
                let out = a.remove(0);
                let (v, d) = self.0.grad(&out, &a);
                Ok((v, numpy_array(py, d)?))
            }
        }
    }
}

/// A recurrence `step(params, state, x) -> (state, y)` traced once and run over signals.
#[pyclass(name = "Scan", module = "autodyne.flux", unsendable)]
struct PyScan(Scan);

fn scan_args<'py>(params: &[Bound<'py, PyAny>], xs: &Bound<'py, PyAny>, s0: &[Bound<'py, PyAny>], extra: &[Bound<'py, PyAny>]) -> Vec<Bound<'py, PyAny>> {
    params.iter().cloned().chain([xs.clone()]).chain(s0.iter().cloned()).chain(extra.iter().cloned()).collect()
}

macro_rules! split {
    ($a:expr, $p:expr, $s:expr) => {{
        let mut a = $a;
        let rest = a.split_off($p);
        let (xs, rest) = rest.split_first().map(|(x, r)| (x.clone(), r.to_vec())).expect("xs");
        let (s0, extra) = rest.split_at($s);
        (a, xs, s0.to_vec(), extra.to_vec())
    }};
}

#[pymethods]
impl PyScan {
    /// `step(params, state, x)` receives lists of tracers (and `x`) and returns `(state list, y)`.
    #[new]
    fn new(py: Python<'_>, params: Vec<Vec<usize>>, states: Vec<Vec<usize>>, sample: Vec<usize>, step: &Bound<'_, PyAny>) -> PyResult<Self> {
        let (p, s): (Vec<&[usize]>, Vec<&[usize]>) = (params.iter().map(Vec::as_slice).collect(), states.iter().map(Vec::as_slice).collect());
        let n = states.len();
        let error = RefCell::new(None);
        let scan = Scan::trace(&p, &s, &sample, |p, s, x| {
            traced(
                &error,
                || {
                    let args = (PyList::new(py, tracers(py, p)?)?, PyList::new(py, tracers(py, s)?)?, wrap(py, x)?);
                    let result = step.call1(args)?;
                    let (next, y): (Bound<'_, PyAny>, Bound<'_, PyAny>) = result.extract().map_err(|_| PyTypeError::new_err("step must return (state, y)"))?;
                    let next = outputs(&next)?;
                    if next.len() != n {
                        return Err(PyValueError::new_err(format!("step returned {} state values, expected {n}", next.len())));
                    }
                    Ok((next, tr(&y)?))
                },
                (s.to_vec(), x),
            )
        });
        match error.into_inner() {
            Some(e) => Err(e),
            None => Ok(PyScan(scan)),
        }
    }

    /// Runs over `xs` (`[len, sample...]`) from `s0`: `(ys, final state list)`.
    fn run(&self, py: Python<'_>, params: Vec<Bound<'_, PyAny>>, xs: &Bound<'_, PyAny>, s0: Vec<Bound<'_, PyAny>>) -> PyResult<(Obj, Vec<Obj>)> {
        let (np, ns) = (params.len(), s0.len());
        match batch(&scan_args(&params, xs, &s0, &[]), None)? {
            Batch::F32(a) => {
                let (p, x, s, _) = split!(a, np, ns);
                let (ys, last) = self.0.run(&p, &x, &s);
                Ok((numpy_array(py, ys)?, numpy_list(py, last)?))
            }
            Batch::F64(a) => {
                let (p, x, s, _) = split!(a, np, ns);
                let (ys, last) = self.0.run(&p, &x, &s);
                Ok((numpy_array(py, ys)?, numpy_list(py, last)?))
            }
        }
    }

    /// The loss of the outputs and its gradient: a dict of `loss`, `params`, `state` and `input`
    /// (the gradients). Without `loss`, the mean squared error against `aux[0]`.
    #[pyo3(signature = (params, xs, s0, aux, loss=None))]
    fn grad(&self, py: Python<'_>, params: Vec<Bound<'_, PyAny>>, xs: &Bound<'_, PyAny>, s0: Vec<Bound<'_, PyAny>>, aux: Vec<Bound<'_, PyAny>>, loss: Option<PyRef<'_, PyLoss>>) -> PyResult<Obj> {
        let (np, ns) = (params.len(), s0.len());
        let dict = PyDict::new(py);
        macro_rules! go {
            ($a:expr) => {{
                let (p, x, s, aux) = split!($a, np, ns);
                let mse;
                let loss = match &loss {
                    Some(l) => &l.0,
                    None => {
                        let target = aux.first().ok_or_else(|| PyValueError::new_err("without a loss, aux must hold the targets"))?;
                        mse = Loss::mse(target.shape());
                        &mse
                    }
                };
                let g = self.0.grad(&p, &x, &s, loss, &aux);
                dict.set_item("loss", g.loss.to_f64().unwrap_or(f64::NAN))?;
                dict.set_item("params", numpy_list(py, g.params)?)?;
                dict.set_item("state", numpy_list(py, g.state)?)?;
                dict.set_item("input", numpy_array(py, g.input)?)?;
            }};
        }
        match batch(&scan_args(&params, xs, &s0, &aux), None)? {
            Batch::F32(a) => go!(a),
            Batch::F64(a) => go!(a),
        }
        Ok(dict.into_any().unbind())
    }

    /// The forward program over `len` steps: `(params..., xs, s0...) -> (ys, final state...)`.
    #[pyo3(signature = (len, dtype="float32", max_fft=None))]
    fn forward_program(&self, len: usize, dtype: &str, max_fft: Option<usize>) -> PyResult<PyProgram> {
        Ok(PyProgram(self.0.forward_program_with(len, &emit(dtype, max_fft)?)))
    }

    /// The same scan with gradients that recompute each step (`True`: the least memory) or save the
    /// step's intermediate values (`False`, the default: faster).
    fn checkpointed(&self, checkpointed: bool) -> Self {
        PyScan(self.0.clone().checkpointed(checkpointed))
    }

    /// The same scan allowed (`True`) to fuse each product read only by a sum into one fused
    /// multiply-add when it runs a scalar scan in process: faster recurrences, results differing
    /// from the step-by-step interpreter's in the last bits (`False`, the default: bit for bit).
    fn contracted(&self, contracted: bool) -> Self {
        PyScan(self.0.clone().contracted(contracted))
    }

    /// The shapes saved per step for gradients besides the state.
    #[getter]
    fn residual_shapes(&self) -> Vec<Vec<usize>> {
        self.0.residual_shapes().to_vec()
    }

    /// The gradient program over `len` steps: `(params..., xs, aux..., s0...) -> (loss, d params...,
    /// d s0..., d xs)`; mean squared error against a target without `loss`.
    #[pyo3(signature = (len, loss=None, dtype="float32", max_fft=None))]
    fn grad_program(&self, len: usize, loss: Option<PyRef<'_, PyLoss>>, dtype: &str, max_fft: Option<usize>) -> PyResult<PyProgram> {
        let mse;
        let loss = match &loss {
            Some(l) => &l.0,
            None => {
                mse = Loss::mse(&[&[len], self.0.output_shape()].concat());
                &mse
            }
        };
        Ok(PyProgram(self.0.grad_program_with(len, loss, &emit(dtype, max_fft)?)))
    }
}

// PROGRAMS AND BACKENDS ===========================================================================

/// A StableHLO program and its signature.
#[pyclass(name = "Program", module = "autodyne.flux")]
struct PyProgram(Program);

#[pymethods]
impl PyProgram {
    #[getter]
    fn text(&self) -> String {
        self.0.text.clone()
    }
    #[getter]
    fn input_shapes(&self) -> Vec<Vec<usize>> {
        self.0.inputs.clone()
    }
    #[getter]
    fn output_shapes(&self) -> Vec<Vec<usize>> {
        self.0.outputs.clone()
    }
    #[getter]
    fn dtype(&self) -> &'static str {
        if self.0.dtype == DType::F64 { "float64" } else { "float32" }
    }
    fn __str__(&self) -> String {
        self.0.text.clone()
    }
}

/// A compiler and runtime: IREE or a PJRT plugin (XLA's, for instance).
#[pyclass(name = "Backend", module = "autodyne.flux", unsendable)]
struct PyBackend(Box<dyn Backend>);

#[pymethods]
impl PyBackend {
    /// IREE's command-line tools (`AUTODYNE_IREE_DIR` or `PATH`) for `target`: "cpu", "vulkan",
    /// "cuda", "rocm" or "metal", optionally with an architecture ("vulkan:ampere", "cuda:sm_80",
    /// "rocm:gfx1100").
    #[staticmethod]
    #[pyo3(signature = (target="cpu"))]
    fn iree(target: &str) -> PyResult<Self> {
        let iree = Iree::find().ok_or_else(|| runtime_error("IREE tools not found (set AUTODYNE_IREE_DIR or put iree-compile on PATH)"))?;
        let (api, arch) = target.split_once(':').map_or((target, None), |(a, b)| (a, Some(b.to_string())));
        let target = match (api, arch) {
            ("cpu", None) => IreeTarget::Cpu,
            ("vulkan", target) => IreeTarget::Vulkan { target },
            ("cuda", None) => IreeTarget::cuda(),
            ("cuda", Some(target)) => IreeTarget::Cuda { target },
            ("rocm", Some(target)) => IreeTarget::Rocm { target },
            ("metal", None) => IreeTarget::Metal,
            _ => return Err(PyValueError::new_err(format!("unknown IREE target {target:?}"))),
        };
        Ok(PyBackend(Box::new(iree.with_target(target))))
    }
    /// A PJRT plugin loaded in-process, with client options (e.g. `{"preallocate": False}`).
    #[staticmethod]
    #[pyo3(signature = (path, options=None))]
    fn pjrt(path: std::path::PathBuf, options: Option<Bound<'_, PyDict>>) -> PyResult<Self> {
        let mut opts = Vec::new();
        for (k, v) in options.iter().flat_map(|d| d.iter()) {
            let value = if let Ok(b) = v.cast::<pyo3::types::PyBool>() {
                PjrtOption::Bool(b.is_true())
            } else if let Ok(i) = v.extract::<i64>() {
                PjrtOption::Int(i)
            } else if let Ok(f) = v.extract::<f32>() {
                PjrtOption::Float(f)
            } else {
                PjrtOption::Str(v.extract()?)
            };
            opts.push((k.extract::<String>()?, value));
        }
        Ok(PyBackend(Box::new(Pjrt::load_with_options(&path, &opts).map_err(flux_error)?)))
    }
    #[getter]
    fn name(&self) -> &'static str {
        self.0.name()
    }
    /// The longest FFT this backend compiles, if limited: pass it as `max_fft` when writing programs.
    #[getter]
    fn max_fft(&self) -> Option<usize> {
        self.0.max_fft()
    }
    fn compile(&self, program: PyRef<'_, PyProgram>) -> PyResult<PyExecutable> {
        Ok(PyExecutable(self.0.compile(&program.0).map_err(flux_error)?))
    }
}

/// A compiled program.
#[pyclass(name = "Executable", module = "autodyne.flux", unsendable)]
struct PyExecutable(Box<dyn Executable>);

#[pymethods]
impl PyExecutable {
    /// Runs the program on host arrays (converted to its precision); returns NumPy arrays.
    #[pyo3(signature = (*inputs))]
    fn __call__(&self, py: Python<'_>, inputs: Vec<Bound<'_, PyAny>>) -> PyResult<Vec<Obj>> {
        match batch(&inputs, Some(self.0.program().dtype))? {
            Batch::F32(a) => numpy_list(py, self.0.run(&a).map_err(flux_error)?),
            Batch::F64(a) => numpy_list(py, self.0.run(&a).map_err(flux_error)?),
        }
    }
}

pub(crate) fn register(parent: &Bound<'_, PyModule>) -> PyResult<()> {
    let m = PyModule::new(parent.py(), "flux")?;
    m.add_class::<PyTracer>()?;
    m.add_class::<PyMask>()?;
    m.add_class::<PyGraph>()?;
    m.add_class::<PyLoss>()?;
    m.add_class::<PyScan>()?;
    m.add_class::<PyProgram>()?;
    m.add_class::<PyBackend>()?;
    m.add_class::<PyExecutable>()?;
    for f in [
        wrap_pyfunction!(trace, &m)?,
        wrap_pyfunction!(value_and_grad, &m)?,
        wrap_pyfunction!(where_, &m)?,
        wrap_pyfunction!(concatenate, &m)?,
        wrap_pyfunction!(irfft, &m)?,
        wrap_pyfunction!(irfft_complex, &m)?,
        wrap_pyfunction!(complex, &m)?,
        wrap_pyfunction!(fft, &m)?,
        wrap_pyfunction!(convolve, &m)?,
        wrap_pyfunction!(frames, &m)?,
        wrap_pyfunction!(stft_magnitude, &m)?,
        wrap_pyfunction!(multi_resolution_stft, &m)?,
        wrap_pyfunction!(one_pole, &m)?,
        wrap_pyfunction!(biquad, &m)?,
        wrap_pyfunction!(svf, &m)?,
        wrap_pyfunction!(ladder, &m)?,
        wrap_pyfunction!(shape, &m)?,
        wrap_pyfunction!(envelope, &m)?,
        wrap_pyfunction!(compressor, &m)?,
    ] {
        m.add_function(f)?;
    }
    parent.add_submodule(&m)?;
    Ok(())
}
