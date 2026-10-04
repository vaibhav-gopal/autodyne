//! `autodyne.random`: a seeded [`Rng`] with NumPy's `Generator` methods (the Python layer adds the
//! `size` handling), and the random processes.

use autodyne::random::{process, Beta, Binomial, ChiSquared, Distribution, Exponential, Gamma, Geometric, Laplace, LogNormal, Normal, Poisson, Rng, StudentT, Uniform};
use autodyne::signal::NdArray;
use pyo3::prelude::*;

use crate::{numpy_array, numpy_out, value_error};

type Obj = Py<PyAny>;

fn vec_out(py: Python<'_>, v: Vec<f64>) -> PyResult<Obj> {
    let n = v.len();
    numpy_out(py, v, &[n], None)
}

/// A random generator (xoshiro256++). Draws come back as NumPy arrays of the requested shape.
#[pyclass(name = "Generator", module = "autodyne.random")]
pub(crate) struct PyGenerator(Rng);

/// Fills an array of `shape` from `$dist`, as float32 when `single`, else float64.
macro_rules! draw {
    ($self:ident, $py:ident, $dist:expr, $shape:expr, $single:expr) => {{
        let dist = $dist.map_err(value_error)?;
        if $single {
            numpy_array($py, $self.0.array::<f32, _>(&dist, &$shape).map_err(value_error)?)
        } else {
            numpy_array($py, $self.0.array::<f64, _>(&dist, &$shape).map_err(value_error)?)
        }
    }};
}

fn counts<D: Distribution<u64>>(py: Python<'_>, rng: &mut Rng, dist: &D, shape: &[usize]) -> PyResult<Obj> {
    let a = rng.array::<u64, _>(dist, shape).map_err(value_error)?;
    numpy_array(py, a.map(|&v| v as i64))
}

#[pymethods]
impl PyGenerator {
    /// From `seed` (the same seed gives the same draws), or from entropy when `None`.
    #[new]
    #[pyo3(signature = (seed=None))]
    fn new(seed: Option<u64>) -> Self {
        PyGenerator(seed.map_or_else(Rng::from_entropy, Rng::new))
    }

    /// Floats of a continuous distribution: `name` with parameters `a` and `b` (see the Python
    /// wrapper for which), float32 when `single`.
    fn continuous(&mut self, py: Python<'_>, name: &str, a: f64, b: f64, shape: Vec<usize>, single: bool) -> PyResult<Obj> {
        match name {
            "uniform" => draw!(self, py, Uniform::new(a, b), shape, single),
            "normal" => draw!(self, py, Normal::new(a, b), shape, single),
            "lognormal" => draw!(self, py, LogNormal::new(a, b), shape, single),
            "exponential" => draw!(self, py, Exponential::new(a), shape, single),
            "gamma" => draw!(self, py, Gamma::new(a, b), shape, single),
            "beta" => draw!(self, py, Beta::new(a, b), shape, single),
            "chisquare" => draw!(self, py, ChiSquared::new(a), shape, single),
            "standard_t" => draw!(self, py, StudentT::new(a), shape, single),
            "laplace" => draw!(self, py, Laplace::new(a, b), shape, single),
            other => Err(value_error(format!("unknown distribution {other:?}"))),
        }
    }

    /// Integers of a discrete distribution (`poisson` with mean `a`, `binomial` with `a` trials of
    /// probability `b`, `geometric` with probability `a`), as int64.
    fn discrete(&mut self, py: Python<'_>, name: &str, a: f64, b: f64, shape: Vec<usize>) -> PyResult<Obj> {
        match name {
            "poisson" => counts(py, &mut self.0, &Poisson::new(a).map_err(value_error)?, &shape),
            "binomial" => {
                if a < 0.0 || a.fract() != 0.0 {
                    return Err(value_error(format!("n must be a non-negative integer, got {a}")));
                }
                counts(py, &mut self.0, &Binomial::new(a as u64, b).map_err(value_error)?, &shape)
            }
            "geometric" => counts(py, &mut self.0, &Geometric::new(a).map_err(value_error)?, &shape),
            other => Err(value_error(format!("unknown distribution {other:?}"))),
        }
    }

    /// Uniform int64 in [`low`, `high`).
    fn integers(&mut self, py: Python<'_>, low: i64, high: i64, shape: Vec<usize>) -> PyResult<Obj> {
        if low >= high {
            return Err(value_error(format!("low ({low}) must be below high ({high})")));
        }
        let mut a = NdArray::<i64>::zeros(&shape).map_err(value_error)?;
        a.as_mut_slice().iter_mut().for_each(|v| *v = self.0.integers(low, high));
        numpy_array(py, a)
    }

    /// `0..n` in random order (int64).
    fn permutation(&mut self, py: Python<'_>, n: usize) -> PyResult<Obj> {
        numpy_out(py, self.0.permutation(n).into_iter().map(|i| i as i64).collect(), &[n], None)
    }

    /// `k` distinct indices of `0..n` (int64).
    fn sample_indices(&mut self, py: Python<'_>, n: usize, k: usize) -> PyResult<Obj> {
        if k > n {
            return Err(value_error(format!("cannot take {k} distinct samples from {n}")));
        }
        numpy_out(py, self.0.sample_indices(n, k).into_iter().map(|i| i as i64).collect(), &[k], None)
    }

    /// A generator for another stream: this one continues 2^128 draws ahead.
    fn spawn(&mut self) -> PyGenerator {
        PyGenerator(self.0.fork())
    }

    /// Gaussian noise with power spectrum `1 / f^beta`, unit variance.
    fn colored_noise(&mut self, py: Python<'_>, beta: f64, n: usize) -> PyResult<Obj> {
        vec_out(py, process::colored_noise(&mut self.0, beta, n).map_err(value_error)?)
    }

    fn brownian_motion(&mut self, py: Python<'_>, n: usize, dt: f64, sigma: f64) -> PyResult<Obj> {
        vec_out(py, process::brownian_motion(&mut self.0, n, dt, sigma).map_err(value_error)?)
    }

    #[allow(clippy::too_many_arguments)]
    fn geometric_brownian_motion(&mut self, py: Python<'_>, n: usize, dt: f64, mu: f64, sigma: f64, s0: f64) -> PyResult<Obj> {
        vec_out(py, process::geometric_brownian_motion(&mut self.0, n, dt, mu, sigma, s0).map_err(value_error)?)
    }

    #[allow(clippy::too_many_arguments)]
    fn ornstein_uhlenbeck(&mut self, py: Python<'_>, n: usize, dt: f64, theta: f64, mu: f64, sigma: f64, x0: f64) -> PyResult<Obj> {
        vec_out(py, process::ornstein_uhlenbeck(&mut self.0, n, dt, theta, mu, sigma, x0).map_err(value_error)?)
    }

    fn arma(&mut self, py: Python<'_>, ar: Vec<f64>, ma: Vec<f64>, n: usize, sigma: f64, burnin: usize) -> PyResult<Obj> {
        vec_out(py, process::arma(&mut self.0, &ar, &ma, n, sigma, burnin).map_err(value_error)?)
    }

    fn poisson_process(&mut self, py: Python<'_>, rate: f64, duration: f64) -> PyResult<Obj> {
        vec_out(py, process::poisson_process(&mut self.0, rate, duration).map_err(value_error)?)
    }
}

pub(crate) fn register(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<PyGenerator>()
}
