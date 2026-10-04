//! `autodyne.gpu`: arrays in GPU memory (the `gpu` feature; without it, `available()` is False).

use pyo3::prelude::*;

#[cfg(feature = "gpu")]
mod enabled {
    use autodyne::gpu::{self, GpuArray, GpuError, GpuFloat};
    use autodyne::signal::NdView;
    use autodyne::units::*;
    use pyo3::exceptions::{PyTypeError, PyValueError};
    use pyo3::prelude::*;

    use crate::{axis_index, input, numpy_array, typed, value_error, Input};

    pub(super) enum Inner {
        F32(GpuArray<f32>),
        F64(GpuArray<f64>),
    }

    pub(super) trait Wrap: GpuFloat + numpy::Element {
        fn wrap(g: GpuArray<Self>) -> Inner;
    }
    impl Wrap for f32 {
        fn wrap(g: GpuArray<f32>) -> Inner {
            Inner::F32(g)
        }
    }
    impl Wrap for f64 {
        fn wrap(g: GpuArray<f64>) -> Inner {
            Inner::F64(g)
        }
    }

    /// `$body` on the array, whichever its dtype, keeping that dtype.
    macro_rules! map {
        ($inner:expr, $g:ident => $body:expr) => {
            match $inner {
                Inner::F32($g) => Inner::F32($body),
                Inner::F64($g) => Inner::F64($body),
            }
        };
    }
    macro_rules! with {
        ($inner:expr, $g:ident => $body:expr) => {
            match $inner {
                Inner::F32($g) => $body,
                Inner::F64($g) => $body,
            }
        };
    }

    fn gpu_error(e: GpuError) -> PyErr {
        crate::runtime_error(e)
    }

    #[derive(Clone, Copy)]
    enum Op {
        Add,
        Sub,
        Mul,
        Div,
    }

    fn pair<T: GpuFloat>(a: &GpuArray<T>, b: &GpuArray<T>, op: Op) -> PyResult<GpuArray<T>> {
        if !a.can_combine(b) {
            return Err(PyValueError::new_err(format!(
                "GpuArray: cannot combine shapes {:?} and {:?} (the same shape, or a row or a column of a matrix)",
                a.shape(),
                b.shape()
            )));
        }
        Ok(match op {
            Op::Add => a.add(b),
            Op::Sub => a.sub(b),
            Op::Mul => a.mul(b),
            Op::Div => a.div(b),
        })
    }

    /// `x op s`, or `s op x` when `reflected`.
    fn with_scalar<T: GpuFloat>(x: &GpuArray<T>, s: T, op: Op, reflected: bool) -> GpuArray<T> {
        match (op, reflected) {
            (Op::Add, _) => x.add_scalar(s),
            (Op::Mul, _) => x.mul_scalar(s),
            (Op::Sub, false) => x.sub_scalar(s),
            (Op::Sub, true) => x.scalar_sub(s),
            (Op::Div, false) => x.div_scalar(s),
            (Op::Div, true) => x.scalar_div(s),
        }
    }

    /// An array in GPU memory (float32, or float64 where the GPU has it). Operations run on the GPU
    /// and give new GpuArrays without copying back; `.numpy()` (or `numpy.asarray`) brings the
    /// values to the host. `.T` and `transpose` are views: nothing moves.
    #[pyclass(name = "GpuArray", module = "autodyne.gpu", frozen)]
    pub(super) struct PyGpuArray(pub(super) Inner);

    fn wrap(py: Python<'_>, inner: Inner) -> PyResult<Py<PyAny>> {
        Ok(Py::new(py, PyGpuArray(inner))?.into_any())
    }

    impl PyGpuArray {
        fn combine(&self, py: Python<'_>, other: &Bound<'_, PyAny>, op: Op, reflected: bool) -> PyResult<Py<PyAny>> {
            if let Ok(o) = other.cast::<PyGpuArray>() {
                let inner = match (&self.0, &o.get().0) {
                    (Inner::F32(a), Inner::F32(b)) => Inner::F32(pair(a, b, op)?),
                    (Inner::F64(a), Inner::F64(b)) => Inner::F64(pair(a, b, op)?),
                    _ => return Err(PyTypeError::new_err("GpuArray: the operands have different dtypes (float32 and float64)")),
                };
                return wrap(py, inner);
            }
            if let Ok(s) = other.extract::<f64>() {
                let inner = match &self.0 {
                    Inner::F32(x) => Inner::F32(with_scalar(x, s as f32, op, reflected)),
                    Inner::F64(x) => Inner::F64(with_scalar(x, s, op, reflected)),
                };
                return wrap(py, inner);
            }
            Ok(py.NotImplemented())
        }
    }

    #[pymethods]
    impl PyGpuArray {
        /// Copies a float32 or float64 array (NumPy, or any DLPack producer) to the GPU. A
        /// transposed or otherwise permuted contiguous array goes up as it lies in memory and
        /// stays that view.
        #[new]
        fn new(x: &Bound<'_, PyAny>) -> PyResult<Self> {
            let input = input(x)?;
            crate::with_float_input!(input, T, view => Ok(PyGpuArray(T::wrap(GpuArray::<T>::from_host(&view).map_err(gpu_error)?))))
        }

        #[getter]
        fn shape(&self) -> Vec<usize> {
            with!(&self.0, g => g.shape())
        }
        #[getter]
        fn ndim(&self) -> usize {
            with!(&self.0, g => g.ndim())
        }
        #[getter]
        fn size(&self) -> usize {
            with!(&self.0, g => g.len())
        }
        #[getter]
        fn dtype(&self) -> &'static str {
            match self.0 {
                Inner::F32(_) => "float32",
                Inner::F64(_) => "float64",
            }
        }
        /// Whether the GPU buffer is in this array's row-major order (not a permuted view).
        #[getter]
        fn is_contiguous(&self) -> bool {
            with!(&self.0, g => g.is_standard())
        }
        fn __len__(&self) -> PyResult<usize> {
            self.shape().first().copied().ok_or_else(|| PyTypeError::new_err("len() of a 0-d GpuArray"))
        }

        /// The values as a NumPy array (waits for the GPU).
        fn numpy(&self, py: Python<'_>) -> PyResult<Py<PyAny>> {
            with!(&self.0, g => numpy_array(py, g.to_host()))
        }
        #[pyo3(signature = (dtype=None, copy=None))]
        fn __array__(&self, py: Python<'_>, dtype: Option<&Bound<'_, PyAny>>, copy: Option<bool>) -> PyResult<Py<PyAny>> {
            if copy == Some(false) {
                return Err(PyValueError::new_err("a GpuArray's values are on the GPU: numpy needs a copy"));
            }
            let a = self.numpy(py)?;
            match dtype {
                Some(d) => Ok(a.call_method1(py, "astype", (d,))?),
                None => Ok(a),
            }
        }
        /// The single value of a one-element array.
        fn __float__(&self) -> PyResult<f64> {
            if self.size() != 1 {
                return Err(PyTypeError::new_err("only one-element GpuArrays convert to float"));
            }
            Ok(with!(&self.0, g => g.to_host().as_slice()[0] as f64))
        }
        fn __repr__(&self) -> String {
            format!("autodyne.gpu.GpuArray(shape={:?}, dtype={})", self.shape(), self.dtype())
        }

        /// The axes reversed, or reordered as `axes` (`numpy.transpose`): a view.
        #[pyo3(signature = (*axes))]
        fn transpose(&self, py: Python<'_>, axes: Vec<isize>) -> PyResult<Py<PyAny>> {
            let n = self.ndim();
            if axes.is_empty() {
                return wrap(py, map!(&self.0, g => g.transpose()));
            }
            let axes = axes.iter().map(|&a| axis_index(a, n)).collect::<PyResult<Vec<_>>>()?;
            let mut seen = vec![false; n];
            if axes.len() != n || axes.iter().any(|&a| std::mem::replace(&mut seen[a], true)) {
                return Err(PyValueError::new_err(format!("transpose: {axes:?} is not a permutation of {n} axes")));
            }
            wrap(py, map!(&self.0, g => g.permute(&axes)))
        }
        #[getter(T)]
        fn transposed(&self, py: Python<'_>) -> PyResult<Py<PyAny>> {
            wrap(py, map!(&self.0, g => g.transpose()))
        }
        /// The same values with the GPU buffer in row-major order (a copy only for a view).
        fn contiguous(&self, py: Python<'_>) -> PyResult<Py<PyAny>> {
            wrap(py, map!(&self.0, g => g.contiguous()))
        }

        fn __add__(&self, py: Python<'_>, other: &Bound<'_, PyAny>) -> PyResult<Py<PyAny>> {
            self.combine(py, other, Op::Add, false)
        }
        fn __radd__(&self, py: Python<'_>, other: &Bound<'_, PyAny>) -> PyResult<Py<PyAny>> {
            self.combine(py, other, Op::Add, true)
        }
        fn __sub__(&self, py: Python<'_>, other: &Bound<'_, PyAny>) -> PyResult<Py<PyAny>> {
            self.combine(py, other, Op::Sub, false)
        }
        fn __rsub__(&self, py: Python<'_>, other: &Bound<'_, PyAny>) -> PyResult<Py<PyAny>> {
            self.combine(py, other, Op::Sub, true)
        }
        fn __mul__(&self, py: Python<'_>, other: &Bound<'_, PyAny>) -> PyResult<Py<PyAny>> {
            self.combine(py, other, Op::Mul, false)
        }
        fn __rmul__(&self, py: Python<'_>, other: &Bound<'_, PyAny>) -> PyResult<Py<PyAny>> {
            self.combine(py, other, Op::Mul, true)
        }
        fn __truediv__(&self, py: Python<'_>, other: &Bound<'_, PyAny>) -> PyResult<Py<PyAny>> {
            self.combine(py, other, Op::Div, false)
        }
        fn __rtruediv__(&self, py: Python<'_>, other: &Bound<'_, PyAny>) -> PyResult<Py<PyAny>> {
            self.combine(py, other, Op::Div, true)
        }
        fn __neg__(&self, py: Python<'_>) -> PyResult<Py<PyAny>> {
            wrap(py, map!(&self.0, g => g.neg()))
        }
        fn __abs__(&self, py: Python<'_>) -> PyResult<Py<PyAny>> {
            wrap(py, map!(&self.0, g => g.abs()))
        }

        /// `a * x + b` in one pass.
        fn axpb(&self, py: Python<'_>, a: f64, b: f64) -> PyResult<Py<PyAny>> {
            wrap(
                py,
                match &self.0 {
                    Inner::F32(g) => Inner::F32(g.axpb(a as f32, b as f32)),
                    Inner::F64(g) => Inner::F64(g.axpb(a, b)),
                },
            )
        }
        fn exp(&self, py: Python<'_>) -> PyResult<Py<PyAny>> {
            wrap(py, map!(&self.0, g => g.exp()))
        }
        fn log(&self, py: Python<'_>) -> PyResult<Py<PyAny>> {
            wrap(py, map!(&self.0, g => g.ln()))
        }
        fn tanh(&self, py: Python<'_>) -> PyResult<Py<PyAny>> {
            wrap(py, map!(&self.0, g => g.tanh()))
        }
        fn sin(&self, py: Python<'_>) -> PyResult<Py<PyAny>> {
            wrap(py, map!(&self.0, g => g.sin()))
        }
        fn cos(&self, py: Python<'_>) -> PyResult<Py<PyAny>> {
            wrap(py, map!(&self.0, g => g.cos()))
        }
        fn sqrt(&self, py: Python<'_>) -> PyResult<Py<PyAny>> {
            wrap(py, map!(&self.0, g => g.sqrt()))
        }
        fn abs(&self, py: Python<'_>) -> PyResult<Py<PyAny>> {
            wrap(py, map!(&self.0, g => g.abs()))
        }

        /// The sum of every element (a 0-d GpuArray: `float()` it), or along `axis` of a vector or
        /// a matrix. Stays on the GPU.
        #[pyo3(signature = (axis=None))]
        fn sum(&self, py: Python<'_>, axis: Option<isize>) -> PyResult<Py<PyAny>> {
            let n = self.ndim();
            match axis {
                None => wrap(py, map!(&self.0, g => g.sum())),
                Some(a) => {
                    let a = axis_index(a, n)?;
                    match n {
                        1 => wrap(py, map!(&self.0, g => g.sum())),
                        2 => wrap(py, map!(&self.0, g => g.sum_axis(a))),
                        _ => Err(PyValueError::new_err("GpuArray.sum: an axis of a vector or a matrix (or no axis)")),
                    }
                }
            }
        }

        /// A causal FIR filter with `taps` along the last axis (each lane starts from silence).
        fn fir(&self, py: Python<'_>, taps: Vec<f64>) -> PyResult<Py<PyAny>> {
            wrap(
                py,
                match &self.0 {
                    Inner::F32(g) => Inner::F32(g.fir(&taps.iter().map(|&t| t as f32).collect::<Vec<_>>())),
                    Inner::F64(g) => Inner::F64(g.fir(&taps)),
                },
            )
        }
    }

    /// Whether a GPU adapter is available.
    #[pyfunction]
    pub(super) fn available() -> bool {
        gpu::available()
    }

    /// Whether the GPU computes in `dtype` ("float32", "float64").
    #[pyfunction]
    pub(super) fn supports(dtype: &Bound<'_, PyAny>) -> PyResult<bool> {
        let name: String = match dtype.extract::<String>() {
            Ok(s) => s,
            Err(_) => dtype.getattr("__name__")?.extract()?,
        };
        Ok(match name.as_str() {
            "float32" | "f32" => gpu::supports::<f32>(),
            "float64" | "f64" | "float" => gpu::supports::<f64>(),
            _ => false,
        })
    }

    /// Waits until the GPU has finished all submitted work.
    #[pyfunction]
    pub(super) fn sync() {
        gpu::sync()
    }
}

#[cfg(not(feature = "gpu"))]
mod disabled {
    use pyo3::prelude::*;

    /// Whether a GPU adapter is available (this build has no GPU support).
    #[pyfunction]
    pub(super) fn available() -> bool {
        false
    }

    #[pyfunction]
    pub(super) fn supports(_dtype: &Bound<'_, PyAny>) -> bool {
        false
    }

    #[pyfunction]
    pub(super) fn sync() {}
}

pub(crate) fn register(parent: &Bound<'_, PyModule>) -> PyResult<()> {
    let m = PyModule::new(parent.py(), "gpu")?;
    #[cfg(feature = "gpu")]
    {
        use enabled::*;
        m.add_class::<PyGpuArray>()?;
        m.add_function(wrap_pyfunction!(available, &m)?)?;
        m.add_function(wrap_pyfunction!(supports, &m)?)?;
        m.add_function(wrap_pyfunction!(sync, &m)?)?;
    }
    #[cfg(not(feature = "gpu"))]
    {
        use disabled::*;
        m.add_function(wrap_pyfunction!(available, &m)?)?;
        m.add_function(wrap_pyfunction!(supports, &m)?)?;
        m.add_function(wrap_pyfunction!(sync, &m)?)?;
    }
    parent.add_submodule(&m)
}
