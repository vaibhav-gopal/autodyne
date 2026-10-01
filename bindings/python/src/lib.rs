//! `autodyne._autodyne`: the native module behind the `autodyne` Python package.
//!
//! Arrays cross the boundary only through DLPack, so nothing is copied either way: inputs are any
//! object with `__dlpack__` (NumPy, PyTorch, JAX, CuPy-on-host...), viewed in place with whatever
//! strides they have; results are [`Array`]s, which hand their memory to the consumer
//! (`numpy.from_dlpack`) the first time they are exported.

use std::ffi::{c_void, CStr};

use autodyne::dlpack::{DLManagedTensor, DLManagedTensorVersioned, DlpackTensor, DL_CPU};
use autodyne::dynamic::{DynArray, DynElement};
use autodyne::filter::{Biquad, BUTTERWORTH_Q};
use autodyne::signal::{NdArray, NdView, Zip};
use autodyne::spectral::RealFft;
use autodyne::units::*;
use pyo3::exceptions::{PyBufferError, PyTypeError, PyValueError};
use pyo3::ffi;
use pyo3::prelude::*;
use pyo3::types::PyDict;

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
}

impl Array {
    fn wrap<T: DynElement>(py: Python<'_>, array: NdArray<T>) -> PyResult<Py<PyAny>> {
        let shape = array.shape().to_vec();
        Ok(Py::new(py, Array { inner: Some(DynArray::from_array(array)), shape, dtype: T::DTYPE })?.into_any())
    }
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
            if max_version.is_some_and(|(major, _)| major >= 1) {
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
                type $T = f32;
                $body
            }
            DType::F64 => {
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

/// Sum of every element (a float), or along `axis` (an Array). Pairwise, so accurate.
#[pyfunction]
#[pyo3(signature = (x, axis=None))]
fn sum(py: Python<'_>, x: &Bound<'_, PyAny>, axis: Option<isize>) -> PyResult<Py<PyAny>> {
    let tensor = import(x)?;
    float_dispatch!(tensor, T => {
        let view = typed::<T>(&tensor)?;
        match axis {
            None => Ok(view.sum().to_f64().unwrap_or(f64::NAN).into_pyobject(py)?.into_any().unbind()),
            Some(axis) => Array::wrap(py, view.sum_axis(axis_index(axis, view.ndim())?).map_err(value_error)?),
        }
    })
}

/// `a * x + b` in one pass (no temporaries).
#[pyfunction]
fn axpb(py: Python<'_>, x: &Bound<'_, PyAny>, a: f64, b: f64) -> PyResult<Py<PyAny>> {
    let tensor = import(x)?;
    float_dispatch!(tensor, T => {
        let view = typed::<T>(&tensor)?;
        let (a, b) = (T::_lit(a), T::_lit(b));
        let out = Zip::from(view).map_collect(|&v| a * v + b);
        Array::wrap(py, out)
    })
}

/// 2nd-order Butterworth low-pass along `axis` (one filter per lane), like
/// `scipy.signal.sosfilt(butter(2, cutoff, fs=sample_rate, output="sos"), x, axis)`.
#[pyfunction]
#[pyo3(signature = (x, cutoff, sample_rate, axis=-1))]
fn lowpass(py: Python<'_>, x: &Bound<'_, PyAny>, cutoff: f64, sample_rate: f64, axis: isize) -> PyResult<Py<PyAny>> {
    let tensor = import(x)?;
    float_dispatch!(tensor, T => {
        let view = typed::<T>(&tensor)?;
        let axis = axis_index(axis, view.ndim())?;
        let mut out = view.to_owned();
        let lanes = out.lanes(axis).map_err(value_error)?.len();
        let mut filters: Vec<Biquad<T>> = (0..lanes).map(|_| Biquad::lowpass(T::_lit(cutoff), T::_lit(sample_rate), T::_lit(BUTTERWORTH_Q))).collect();
        out.process_lanes(axis, &mut filters).map_err(value_error)?;
        Array::wrap(py, out)
    })
}

/// Real FFT along the last axis (power-of-two length), like `numpy.fft.rfft(x)`.
#[pyfunction]
fn rfft(py: Python<'_>, x: &Bound<'_, PyAny>) -> PyResult<Py<PyAny>> {
    let tensor = import(x)?;
    float_dispatch!(tensor, T => {
        let view = typed::<T>(&tensor)?;
        let last = view.ndim().checked_sub(1).ok_or_else(|| PyValueError::new_err("rfft needs at least one axis"))?;
        let n = view.shape()[last];
        if !n.is_power_of_two() || n < 2 {
            return Err(PyValueError::new_err(format!("rfft needs a power-of-two length >= 2, got {n}")));
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
        Array::wrap(py, out)
    })
}

#[pymodule]
fn _autodyne(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<Array>()?;
    m.add_function(wrap_pyfunction!(sum, m)?)?;
    m.add_function(wrap_pyfunction!(axpb, m)?)?;
    m.add_function(wrap_pyfunction!(lowpass, m)?)?;
    m.add_function(wrap_pyfunction!(rfft, m)?)?;
    Ok(())
}
