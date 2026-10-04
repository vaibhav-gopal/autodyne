"""Random numbers with ``numpy.random.Generator``'s methods, and random processes.

:func:`default_rng` gives a :class:`Generator` (xoshiro256++, ziggurat normal and exponential
draws). Distribution parameters are scalars; ``size`` is ``None`` (one value), an int or a shape.
The streams differ from NumPy's (another generator), but the distributions are the same.

    >>> from autodyne import random
    >>> rng = random.default_rng(42)
    >>> x = rng.normal(0.0, 2.0, size=(3, 4))
    >>> pink = rng.colored_noise(1.0, 4096)
"""

import numpy as _np

from ._autodyne import Generator as _Native

__all__ = ["Generator", "default_rng"]


def _shape(size):
    if size is None:
        return None
    return [int(size)] if _np.ndim(size) == 0 else [int(s) for s in size]


def _single(dtype):
    dtype = _np.dtype(dtype)
    if dtype not in (_np.float32, _np.float64):
        raise TypeError(f"dtype must be float32 or float64, got {dtype}")
    return dtype == _np.float32


class Generator:
    """A seeded random generator with NumPy's ``Generator`` methods."""

    def __init__(self, seed=None):
        self._rng = seed if isinstance(seed, _Native) else _Native(None if seed is None else int(seed))

    def _cont(self, name, a, b, size, dtype=_np.float64):
        shape = _shape(size)
        out = self._rng.continuous(name, float(a), float(b), [1] if shape is None else shape, _single(dtype))
        return out[0] if shape is None else out

    def _disc(self, name, a, b, size):
        shape = _shape(size)
        out = self._rng.discrete(name, float(a), float(b), [1] if shape is None else shape)
        return int(out[0]) if shape is None else out

    def random(self, size=None, dtype=_np.float64):
        """Uniform floats in [0, 1)."""
        return self._cont("uniform", 0.0, 1.0, size, dtype)

    def uniform(self, low=0.0, high=1.0, size=None):
        """Uniform floats in [low, high)."""
        return self._cont("uniform", low, high, size)

    def standard_normal(self, size=None, dtype=_np.float64):
        """Normal, mean 0 and standard deviation 1."""
        return self._cont("normal", 0.0, 1.0, size, dtype)

    def normal(self, loc=0.0, scale=1.0, size=None):
        """Normal with mean ``loc`` and standard deviation ``scale``."""
        return self._cont("normal", loc, scale, size)

    def lognormal(self, mean=0.0, sigma=1.0, size=None):
        """``exp`` of a normal with this mean and standard deviation."""
        return self._cont("lognormal", mean, sigma, size)

    def standard_exponential(self, size=None, dtype=_np.float64):
        """Exponential with mean 1."""
        return self._cont("exponential", 1.0, 0.0, size, dtype)

    def exponential(self, scale=1.0, size=None):
        """Exponential with mean ``scale``."""
        return self._cont("exponential", scale, 0.0, size)

    def standard_gamma(self, shape, size=None, dtype=_np.float64):
        """Gamma with this shape and scale 1."""
        return self._cont("gamma", shape, 1.0, size, dtype)

    def gamma(self, shape, scale=1.0, size=None):
        """Gamma with ``shape`` and ``scale``."""
        return self._cont("gamma", shape, scale, size)

    def beta(self, a, b, size=None):
        """Beta on [0, 1]."""
        return self._cont("beta", a, b, size)

    def chisquare(self, df, size=None):
        """Chi-squared with ``df`` degrees of freedom."""
        return self._cont("chisquare", df, 0.0, size)

    def standard_t(self, df, size=None):
        """Student's t with ``df`` degrees of freedom."""
        return self._cont("standard_t", df, 0.0, size)

    def laplace(self, loc=0.0, scale=1.0, size=None):
        """Laplace (double exponential)."""
        return self._cont("laplace", loc, scale, size)

    def poisson(self, lam=1.0, size=None):
        """Poisson counts with mean ``lam`` (int64)."""
        return self._disc("poisson", lam, 0.0, size)

    def binomial(self, n, p, size=None):
        """Successes in ``n`` trials of probability ``p`` (int64)."""
        return self._disc("binomial", n, p, size)

    def geometric(self, p, size=None):
        """Trials up to the first success, at least 1 (int64)."""
        return self._disc("geometric", p, 0.0, size)

    def integers(self, low, high=None, size=None, dtype=_np.int64, endpoint=False):
        """Uniform integers in [low, high) ([0, low) with one argument; ``endpoint`` includes high)."""
        if high is None:
            low, high = 0, low
        if endpoint:
            high += 1
        shape = _shape(size)
        out = self._rng.integers(int(low), int(high), [1] if shape is None else shape).astype(dtype, copy=False)
        return out[0] if shape is None else out

    def permutation(self, x):
        """A shuffled copy of ``x`` (along its first axis), or of ``arange(x)`` for an int."""
        if _np.ndim(x) == 0:
            return self._rng.permutation(int(x))
        x = _np.asarray(x)
        return x[self._rng.permutation(len(x))]

    def shuffle(self, x):
        """Shuffles ``x`` in place along its first axis."""
        x[...] = x[self._rng.permutation(len(x))]

    def choice(self, a, size=None, replace=True):
        """Random elements of ``a`` (or of ``arange(a)``), with or without replacement."""
        pool = _np.arange(a) if _np.ndim(a) == 0 else _np.asarray(a)
        shape = _shape(size)
        count = 1 if shape is None else int(_np.prod(shape))
        if replace:
            idx = self._rng.integers(0, len(pool), [count])
        else:
            idx = self._rng.sample_indices(len(pool), count)
        out = pool[idx]
        return out[0] if shape is None else out.reshape(shape)

    def spawn(self, n_children):
        """``n_children`` generators on non-overlapping streams."""
        return [Generator(self._rng.spawn()) for _ in range(n_children)]

    # ---- random processes ------------------------------------------------------------------------

    def colored_noise(self, beta, n):
        """``n`` samples of Gaussian noise with power spectrum ``1 / f**beta`` (0 white, 1 pink, 2 brown,
        -1 blue), unit variance."""
        return self._rng.colored_noise(float(beta), int(n))

    def brownian_motion(self, n, dt=1.0, sigma=1.0):
        """``n`` samples of a Wiener process from 0, ``dt`` apart, volatility ``sigma``."""
        return self._rng.brownian_motion(int(n), float(dt), float(sigma))

    def geometric_brownian_motion(self, n, dt, mu, sigma, s0=1.0):
        """``n`` samples of ``dS = mu S dt + sigma S dW`` from ``s0``, stepped exactly."""
        return self._rng.geometric_brownian_motion(int(n), float(dt), float(mu), float(sigma), float(s0))

    def ornstein_uhlenbeck(self, n, dt, theta, mu=0.0, sigma=1.0, x0=0.0):
        """``n`` samples of ``dx = theta (mu - x) dt + sigma dW`` from ``x0``, stepped exactly."""
        return self._rng.ornstein_uhlenbeck(int(n), float(dt), float(theta), float(mu), float(sigma), float(x0))

    def arma(self, ar, ma, nsample, scale=1.0, burnin=0):
        """An ARMA sample like ``statsmodels``' ``arma_generate_sample`` (lag polynomials with their
        leading 1: ``ar = [1, -phi1, ...]``)."""
        return self._rng.arma([float(v) for v in ar], [float(v) for v in ma], int(nsample), float(scale), int(burnin))

    def poisson_process(self, rate, duration):
        """Event times in [0, duration) of a Poisson process with ``rate`` events per unit time."""
        return self._rng.poisson_process(float(rate), float(duration))


def default_rng(seed=None):
    """A :class:`Generator` from ``seed`` (an int), or from fresh entropy."""
    return Generator(seed)
