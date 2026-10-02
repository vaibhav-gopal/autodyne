//! `autodyne._autodyne`: the native module behind the `autodyne` Python package.
//!
//! Arrays cross the boundary only through DLPack, so nothing is copied either way: inputs are any
//! object with `__dlpack__` (NumPy, PyTorch, JAX, CuPy-on-host...), viewed in place with whatever
//! strides they have; results are [`Array`]s, which hand their memory to the consumer
//! (`numpy.from_dlpack`) the first time they are exported.

use std::ffi::{c_void, CStr};

use autodyne::dlpack::{DLManagedTensor, DLManagedTensorVersioned, DlpackTensor, DL_CPU};
use autodyne::dynamic::{BinaryOp, CastMode, DynArray, DynElement, Promotion};
use autodyne::filter::{Biquad, BUTTERWORTH_Q};
use autodyne::signal::{NdArray, NdView, Zip};
use autodyne::spectral::RealFft;
use autodyne::units::*;
use pyo3::exceptions::{PyBufferError, PyTypeError, PyValueError};
use pyo3::ffi;
use pyo3::prelude::*;
use pyo3::types::PyDict;
use numpy::ndarray::{ArrayD, IxDyn, ShapeBuilder};
use numpy::{PyArray, PyArrayDyn, PyArrayMethods, PyReadonlyArrayDyn};

const VERSIONED: &CStr = c"dltensor_versioned";
const USED_VERSIONED: &CStr = c"used_dltensor_versioned";
const LEGACY: &CStr = c"dltensor";
const USED_LEGACY: &CStr = c"used_dltensor";

fn value_error(e: impl std::fmt::Display) -> PyErr {
    PyValueError::new_err(e.to_string())
}

/// Takes the DLPack tensor of any object with `__dlpack__` (versioned when the producer offers it).
fn import(obj: &Bound<'_, PyAny>) -> PyResult<DlpackTensor> {
    let py = obj.py();
    let kwargs = PyDict::new(py);
    kwargs.set_item("max_version", (1u32, 0u32))?;
    let capsule = match obj.call_method("__dlpack__", (), Some(&kwargs)) {
        Ok(c) => c,
        // producers that predate DLPack 1.0 don't take max_version
        Err(e) if e.is_instance_of::<PyTypeError>(py) => obj.call_method0("__dlpack__")?,
        Err(e) => return Err(e),
    };
    let cap = capsule.as_ptr();
    // SAFETY: `cap` is a live object (held by `capsule`); a capsule with one of these names holds a
    // DLPack tensor that, once renamed "used", belongs to us (the DLPack Python protocol)
    unsafe {
        if ffi::PyCapsule_IsValid(cap, VERSIONED.as_ptr()) == 1 {
            let tensor = ffi::PyCapsule_GetPointer(cap, VERSIONED.as_ptr()).cast::<DLManagedTensorVersioned>();
            if tensor.is_null() || ffi::PyCapsule_SetName(cap, USED_VERSIONED.as_ptr()) != 0 {
                return Err(PyErr::fetch(py));
            }
            DlpackTensor::from_raw(tensor).map_err(value_error)
        } else if ffi::PyCapsule_IsValid(cap, LEGACY.as_ptr()) == 1 {
            let tensor = ffi::PyCapsule_GetPointer(cap, LEGACY.as_ptr()).cast::<DLManagedTensor>();
            if tensor.is_null() || ffi::PyCapsule_SetName(cap, USED_LEGACY.as_ptr()) != 0 {
                return Err(PyErr::fetch(py));
            }
            DlpackTensor::from_raw_legacy(tensor).map_err(value_error)
        } else {
            Err(PyTypeError::new_err("__dlpack__ did not return a DLPack capsule"))
        }
    }
}

/// Deletes the tensor of a capsule that nobody consumed (still carrying its original name).
unsafe extern "C" fn drop_versioned_capsule(cap: *mut ffi::PyObject) {
    // SAFETY: called by Python with the capsule being destroyed
    unsafe {
        if ffi::PyCapsule_IsValid(cap, VERSIONED.as_ptr()) == 1 {
            let tensor = ffi::PyCapsule_GetPointer(cap, VERSIONED.as_ptr()).cast::<DLManagedTensorVersioned>();
            if let Some(deleter) = (*tensor).deleter {
                deleter(tensor);
            }
        }
    }
}

unsafe extern "C" fn drop_legacy_capsule(cap: *mut ffi::PyObject) {
    // SAFETY: as above
    unsafe {
        if ffi::PyCapsule_IsValid(cap, LEGACY.as_ptr()) == 1 {
            let tensor = ffi::PyCapsule_GetPointer(cap, LEGACY.as_ptr()).cast::<DLManagedTensor>();
            if let Some(deleter) = (*tensor).deleter {
                deleter(tensor);
            }
        }
    }
}

/// A result array. Export it with `numpy.from_dlpack(array)` (or any DLPack consumer): the first
/// export hands over its memory without copying, after which this object is empty.
#[pyclass(module = "autodyne")]
struct Array {
    inner: Option<DynArray>,
    shape: Vec<usize>,
    dtype: DType,
    /// export the stored array through this axis permutation (results computed in an input's
    /// memory order go back with that input's strides, as NumPy's do)
    axes: Option<Vec<usize>>,
}

#[pymethods]
impl Array {
    #[getter]
    fn shape(&self) -> Vec<usize> {
        self.shape.clone()
    }
    #[getter]
    fn dtype(&self) -> &'static str {
        self.dtype.name()
    }
    #[pyo3(signature = (*, stream=None, max_version=None, dl_device=None, copy=None))]
    fn __dlpack__(
        &mut self,
        py: Python<'_>,
        stream: Option<Bound<'_, PyAny>>,
        max_version: Option<(u32, u32)>,
        dl_device: Option<(i32, i32)>,
        copy: Option<bool>,
    ) -> PyResult<Py<PyAny>> {
        let _ = (stream, copy);
        if let Some((device, _)) = dl_device {
            if device != DL_CPU {
                return Err(PyBufferError::new_err("autodyne arrays live in CPU memory"));
            }
        }
        let array = self.inner.take().ok_or_else(|| PyBufferError::new_err("this array was already exported"))?;
        // SAFETY: the capsule takes the exported tensor; its destructor deletes it unless a consumer
        // renamed (took) it first
        let capsule = unsafe {
            if let Some(axes) = &self.axes {
                // permuted strides need a versioned consumer (legacy ones get a contiguous export
                // below only when no permutation is needed)
                ffi::PyCapsule_New(array.into_dlpack_permuted(axes).cast::<c_void>(), VERSIONED.as_ptr(), Some(drop_versioned_capsule))
            } else if max_version.is_some_and(|(major, _)| major >= 1) {
                ffi::PyCapsule_New(array.into_dlpack().cast::<c_void>(), VERSIONED.as_ptr(), Some(drop_versioned_capsule))
            } else {
                ffi::PyCapsule_New(array.into_dlpack_legacy().cast::<c_void>(), LEGACY.as_ptr(), Some(drop_legacy_capsule))
            }
        };
        if capsule.is_null() {
            return Err(PyErr::fetch(py));
        }
        // SAFETY: a new reference returned by PyCapsule_New
        Ok(unsafe { Bound::from_owned_ptr(py, capsule) }.unbind())
    }
    fn __dlpack_device__(&self) -> (i32, i32) {
        (DL_CPU, 0)
    }
    fn __repr__(&self) -> String {
        format!("autodyne.Array(shape={:?}, dtype={}{})", self.shape, self.dtype, if self.inner.is_none() { ", exported" } else { "" })
    }
}

fn axis_index(axis: isize, ndim: usize) -> PyResult<usize> {
    let a = if axis < 0 { axis + ndim as isize } else { axis };
    if a < 0 || a as usize >= ndim {
        return Err(PyValueError::new_err(format!("axis {axis} is out of range for {ndim} dimensions")));
    }
    Ok(a as usize)
}

/// Runs `$body` with `$T` = f32 or f64 for the tensor's dtype.
macro_rules! float_dispatch {
    ($tensor:expr, $T:ident => $body:expr) => {
        match $tensor.dtype() {
            DType::F32 => {
                #[allow(dead_code)]
                type $T = f32;
                $body
            }
            DType::F64 => {
                #[allow(dead_code)]
                type $T = f64;
                $body
            }
            d => Err(PyTypeError::new_err(format!("expected float32 or float64, got {d}"))),
        }
    };
}

fn typed<T: DynElement>(tensor: &DlpackTensor) -> PyResult<NdView<'_, T>> {
    tensor.typed::<T>().map_err(value_error)
}

/// A float input: a NumPy array read through NumPy's C API (a type check, no Python call), or
/// anything else through DLPack. Both are zero-copy views.
enum Input<'py> {
    F32(PyReadonlyArrayDyn<'py, f32>),
    F64(PyReadonlyArrayDyn<'py, f64>),
    Dlpack(DlpackTensor),
}

fn input<'py>(x: &Bound<'py, PyAny>) -> PyResult<Input<'py>> {
    if let Ok(a) = x.cast::<PyArrayDyn<f32>>() {
        if let Ok(r) = a.try_readonly() {
            return Ok(Input::F32(r));
        }
    }
    if let Ok(a) = x.cast::<PyArrayDyn<f64>>() {
        if let Ok(r) = a.try_readonly() {
            return Ok(Input::F64(r));
        }
    }
    Ok(Input::Dlpack(import(x)?))
}

/// Runs `$body` with `$T` = f32 / f64 and `$view` the input's typed view.
macro_rules! with_float_input {
    ($input:expr, $T:ident, $view:ident => $body:expr) => {
        match &$input {
            Input::F32(a) => {
                #[allow(dead_code)]
                type $T = f32;
                let $view: NdView<'_, f32> = NdView::try_from(a.as_array()).map_err(value_error)?;
                $body
            }
            Input::F64(a) => {
                #[allow(dead_code)]
                type $T = f64;
                let $view: NdView<'_, f64> = NdView::try_from(a.as_array()).map_err(value_error)?;
                $body
            }
            Input::Dlpack(tensor) => float_dispatch!(tensor, $T => {
                let $view = typed::<$T>(tensor)?;
                $body
            }),
        }
    };
}

/// A NumPy array that owns `data` (no copy), with `shape`, row-major, or exported through the
/// inverse of `order` (a result computed in an input's memory order goes back with its strides).
fn numpy_out<T: numpy::Element>(py: Python<'_>, data: Vec<T>, shape: &[usize], order: Option<&[usize]>) -> PyResult<Py<PyAny>> {
    let array = match order {
        None => ArrayD::from_shape_vec(IxDyn(shape), data),
        Some(order) => {
            let n = shape.len();
            let mut row_major = vec![0usize; n];
            let mut acc = 1;
            for i in (0..n).rev() {
                row_major[i] = acc;
                acc *= shape[i];
            }
            let mut inverse = vec![0; n];
            for (i, &a) in order.iter().enumerate() {
                inverse[a] = i;
            }
            let exported: Vec<usize> = inverse.iter().map(|&i| shape[i]).collect();
            let strides: Vec<usize> = inverse.iter().map(|&i| row_major[i]).collect();
            ArrayD::from_shape_vec(IxDyn(&exported).strides(IxDyn(&strides)), data)
        }
    }
    .map_err(value_error)?;
    Ok(PyArray::from_owned_array(py, array).into_any().unbind())
}

fn numpy_array<T: numpy::Element>(py: Python<'_>, array: NdArray<T>) -> PyResult<Py<PyAny>> {
    let shape = array.shape().to_vec();
    numpy_out(py, array.into_vec(), &shape, None)
}

/// Complex results: autodyne's and num-complex's `Complex` are both `repr(C) { re, im }`.
fn numpy_complex<T: Float>(py: Python<'_>, array: NdArray<Complex<T>>) -> PyResult<Py<PyAny>>
where
    num_complex::Complex<T>: numpy::Element,
{
    let shape = array.shape().to_vec();
    let mut data = std::mem::ManuallyDrop::new(array.into_vec());
    // SAFETY: identical layout (repr(C) { re: T, im: T }), so the allocation is reused as is
    let data: Vec<num_complex::Complex<T>> = unsafe { Vec::from_raw_parts(data.as_mut_ptr().cast(), data.len(), data.capacity()) };
    numpy_out(py, data, &shape, None)
}

/// Sum of every element (a float), or along `axis` (an array). Pairwise, so accurate.
#[pyfunction]
#[pyo3(signature = (x, axis=None))]
fn sum(py: Python<'_>, x: &Bound<'_, PyAny>, axis: Option<isize>) -> PyResult<Py<PyAny>> {
    let input = input(x)?;
    with_float_input!(input, T, view => match axis {
        None => Ok(view.sum().to_f64().unwrap_or(f64::NAN).into_pyobject(py)?.into_any().unbind()),
        Some(axis) => numpy_array(py, view.sum_axis(axis_index(axis, view.ndim())?).map_err(value_error)?),
    })
}

/// `a * x + b` in one pass (no temporaries), in the input's memory order.
#[pyfunction]
fn axpb(py: Python<'_>, x: &Bound<'_, PyAny>, a: f64, b: f64) -> PyResult<Py<PyAny>> {
    let input = input(x)?;
    with_float_input!(input, T, view => {
        let (a, b) = (T::_lit(a), T::_lit(b));
        let n = view.ndim();
        if view.is_contiguous() {
            let out = Zip::from(view).map_collect(|&v| a * v + b);
            return numpy_array(py, out);
        }
        // compute in the input's memory order (sequential reads and writes), like NumPy's order='K'
        let order = view.memory_order();
        let order = &order[..n];
        let out = Zip::from(view.permute(order).map_err(value_error)?).map_collect(|&v| a * v + b);
        let shape = out.shape().to_vec();
        numpy_out(py, out.into_vec(), &shape, Some(order))
    })
}

/// 2nd-order Butterworth low-pass along `axis` (one filter per lane), like
/// `scipy.signal.sosfilt(butter(2, cutoff, fs=sample_rate, output="sos"), x, axis)`.
#[pyfunction]
#[pyo3(signature = (x, cutoff, sample_rate, axis=-1))]
fn lowpass(py: Python<'_>, x: &Bound<'_, PyAny>, cutoff: f64, sample_rate: f64, axis: isize) -> PyResult<Py<PyAny>> {
    let input = input(x)?;
    with_float_input!(input, T, view => {
        let axis = axis_index(axis, view.ndim())?;
        let mut out = view.to_owned();
        let lanes = out.lanes(axis).map_err(value_error)?.len();
        let mut filters: Vec<Biquad<T>> = (0..lanes).map(|_| Biquad::lowpass(T::_lit(cutoff), T::_lit(sample_rate), T::_lit(BUTTERWORTH_Q))).collect();
        out.process_lanes(axis, &mut filters).map_err(value_error)?;
        numpy_array(py, out)
    })
}

/// Real FFT along the last axis (any length), like `numpy.fft.rfft(x)`.
#[pyfunction]
fn rfft(py: Python<'_>, x: &Bound<'_, PyAny>) -> PyResult<Py<PyAny>> {
    let input = input(x)?;
    with_float_input!(input, T, view => {
        let last = view.ndim().checked_sub(1).ok_or_else(|| PyValueError::new_err("rfft needs at least one axis"))?;
        let n = view.shape()[last];
        if n == 0 {
            return Err(PyValueError::new_err("rfft needs a non-empty last axis"));
        }
        let mut fft = RealFft::<T>::new(n);
        let mut shape = view.shape().to_vec();
        shape[last] = n / 2 + 1;
        let mut out = NdArray::<Complex<T>>::zeros(&shape).map_err(value_error)?;
        let mut scratch = vec![T::_ZERO; n];
        for (input, mut output) in view.lanes(last).map_err(value_error)?.zip(out.lanes_mut(last).map_err(value_error)?) {
            let samples = match input.as_slice() {
                Some(s) => s,
                None => {
                    for (d, &s) in scratch.iter_mut().zip(input.iter()) {
                        *d = s;
                    }
                    &scratch
                }
            };
            let bins = output.as_mut_slice().expect("a fresh array's last-axis lanes are contiguous");
            fft.forward(samples, bins);
        }
        numpy_complex(py, out)
    })
}
/// Wraps a runtime-typed result.
fn wrap_dyn(py: Python<'_>, array: DynArray) -> PyResult<Py<PyAny>> {
    let (shape, dtype) = (array.shape().to_vec(), array.dtype());
    Ok(Py::new(py, Array { inner: Some(array), shape, dtype, axes: None })?.into_any())
}

/// `x op y` for any two numeric dtypes, broadcast, with autodyne's promotion: NumPy's table with
/// every conversion checked (`keep_float=False`), or floats keeping their width against integers.
#[pyfunction]
#[pyo3(signature = (op, x, y, keep_float=false))]
fn binary(py: Python<'_>, op: &str, x: &Bound<'_, PyAny>, y: &Bound<'_, PyAny>, keep_float: bool) -> PyResult<Py<PyAny>> {
    let op = match op {
        "add" => BinaryOp::Add,
        "sub" => BinaryOp::Sub,
        "mul" => BinaryOp::Mul,
        "div" => BinaryOp::Div,
        "min" => BinaryOp::Min,
        "max" => BinaryOp::Max,
        other => return Err(PyValueError::new_err(format!("unknown operation {other:?}"))),
    };
    let (a, b) = (import(x)?, import(y)?);
    let policy = if keep_float { Promotion::KeepFloat } else { Promotion::Standard };
    let result = a.view().map_err(value_error)?.binary(&b.view().map_err(value_error)?, op, policy).map_err(value_error)?;
    wrap_dyn(py, result)
}

/// `x` converted to `dtype` ("f32", "i16", "complex_f64"...): `mode` is "checked", "saturating"
/// or "wrapping".
#[pyfunction]
#[pyo3(signature = (x, dtype, mode="checked"))]
fn cast(py: Python<'_>, x: &Bound<'_, PyAny>, dtype: &str, mode: &str) -> PyResult<Py<PyAny>> {
    let dtype = DType::from_name(dtype).ok_or_else(|| PyValueError::new_err(format!("unknown dtype {dtype:?}")))?;
    let mode = match mode {
        "checked" => CastMode::Checked,
        "saturating" => CastMode::Saturating,
        "wrapping" => CastMode::Wrapping,
        other => return Err(PyValueError::new_err(format!("unknown cast mode {other:?}"))),
    };
    let tensor = import(x)?;
    wrap_dyn(py, tensor.view().map_err(value_error)?.cast(dtype, mode).map_err(value_error)?)
}

#[pymodule]
fn _autodyne(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(binary, m)?)?;
    m.add_function(wrap_pyfunction!(cast, m)?)?;
    m.add_class::<Array>()?;
    m.add_function(wrap_pyfunction!(sum, m)?)?;
    m.add_function(wrap_pyfunction!(axpb, m)?)?;
    m.add_function(wrap_pyfunction!(lowpass, m)?)?;
    m.add_function(wrap_pyfunction!(rfft, m)?)?;
    Ok(())
}
