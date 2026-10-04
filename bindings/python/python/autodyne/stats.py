"""Statistics with NumPy's, ``scipy.stats``' and ``statsmodels``' names and defaults: quantiles and
medians, moments, histograms, covariance and correlation, and time-series estimates
(autocovariance, autocorrelation, partial autocorrelation, Yule-Walker and Burg fits)."""

import numpy as _np

from ._autodyne import stats as _native

__all__ = ["median", "quantile", "percentile", "skew", "kurtosis", "moment", "zscore", "histogram", "histogram_bin_edges",
           "cov", "corrcoef", "acovf", "acf", "pacf", "yule_walker", "burg", "levinson_durbin"]


def _data(x):
    x = _np.asarray(x)
    return x if x.dtype in (_np.float32, _np.float64) else x.astype(_np.float64)


def _flat_or(x, axis):
    """``x`` and the axis to reduce: flattened for ``axis=None``, as NumPy does."""
    x = _data(x)
    return (x.ravel(), 0) if axis is None else (x, axis)


def quantile(a, q, axis=None, method="linear", keepdims=False):
    """Quantiles ``q`` (in [0, 1]) along ``axis``, like ``numpy.quantile``: the quantiles come first."""
    x, ax = _flat_or(a, axis)
    qs = _np.atleast_1d(_np.asarray(q, dtype=float))
    out = _np.moveaxis(_native.quantile(x, qs.ravel().tolist(), ax, method), ax % x.ndim, 0)
    if keepdims and axis is not None:
        out = _np.expand_dims(out, axis % x.ndim + 1)
    return out[0] if _np.ndim(q) == 0 else out.reshape(qs.shape + out.shape[1:])


def percentile(a, q, axis=None, method="linear", keepdims=False):
    """Percentiles ``q`` (in [0, 100]), like ``numpy.percentile``."""
    return quantile(a, _np.asarray(q, dtype=float) / 100.0, axis, method, keepdims)


def median(a, axis=None, keepdims=False):
    """The median along ``axis``, like ``numpy.median`` (NaN where a lane holds NaN)."""
    x, ax = _flat_or(a, axis)
    out = _native.median(x, ax)
    if keepdims and axis is not None:
        out = _np.expand_dims(out, axis)
    return out[()] if out.ndim == 0 else out


def skew(a, axis=0, bias=True):
    """Sample skewness, like ``scipy.stats.skew``."""
    x, ax = _flat_or(a, axis)
    return _native.skew(x, ax, bias)


def kurtosis(a, axis=0, fisher=True, bias=True):
    """Kurtosis (excess with ``fisher``), like ``scipy.stats.kurtosis``."""
    x, ax = _flat_or(a, axis)
    return _native.kurtosis(x, ax, fisher, bias)


def moment(a, order=1, axis=0):
    """The central moment of ``order``, like ``scipy.stats.moment``."""
    x, ax = _flat_or(a, axis)
    return _native.moment(x, int(order), ax)


def zscore(a, axis=0, ddof=0):
    """Standard scores, like ``scipy.stats.zscore``."""
    x, ax = _flat_or(a, axis)
    out = _native.zscore(x, ax, ddof)
    return out.reshape(_np.shape(a)) if axis is None else out


def _bins(bins):
    return bins if isinstance(bins, (str, int, _np.integer)) else [float(v) for v in bins]


def histogram(a, bins=10, range=None, weights=None, density=False):
    """``(hist, bin_edges)``, like ``numpy.histogram`` (``bins``: a count, edges, or a rule name)."""
    w = None if weights is None else _np.asarray(weights, dtype=float).ravel().tolist()
    h, e = _native.histogram(_data(a).ravel(), _bins(bins), range, w, density)
    if weights is None and not density:
        h = h.astype(_np.int64)
    return h, e


def histogram_bin_edges(a, bins=10, range=None):
    """The edges :func:`histogram` uses, like ``numpy.histogram_bin_edges``."""
    return _native.histogram_bin_edges(_data(a).ravel(), _bins(bins), range)


def cov(m, rowvar=True, bias=False, ddof=None):
    """The covariance matrix, like ``numpy.cov``."""
    if ddof is None:
        ddof = 0 if bias else 1
    out = _native.cov(_data(m), rowvar, ddof)
    return out[0, 0] if out.shape == (1, 1) else out


def corrcoef(x, rowvar=True):
    """The Pearson correlation matrix, like ``numpy.corrcoef``."""
    out = _native.corrcoef(_data(x), rowvar)
    return out[0, 0] if out.shape == (1, 1) else out


def acovf(x, adjusted=False, demean=True, nlag=None):
    """Autocovariances (by FFT), like ``statsmodels.tsa.stattools.acovf``."""
    x = _data(x).ravel()
    return _native.acovf(x, len(x) - 1 if nlag is None else nlag, adjusted, demean)


def _default_lags(n):
    return min(int(10 * _np.log10(n)), n - 1)


def acf(x, adjusted=False, nlags=None):
    """Autocorrelations at lags 0..nlags, like ``statsmodels.tsa.stattools.acf``."""
    x = _data(x).ravel()
    return _native.acf(x, _default_lags(len(x)) if nlags is None else nlags, adjusted)


def pacf(x, nlags=None, method="ywadjusted"):
    """Partial autocorrelations (Yule-Walker), like ``statsmodels.tsa.stattools.pacf`` with ``method``
    yw / ywadjusted or ywm / ywmle."""
    x = _data(x).ravel()
    if nlags is None:
        nlags = min(int(10 * _np.log10(len(x))), len(x) // 2 - 1)
    return _native.pacf(x, nlags, method)


def yule_walker(x, order=1, method="adjusted"):
    """``(rho, sigma)``: AR coefficients and the innovations' standard deviation, like statsmodels'
    ``yule_walker`` (``method`` adjusted or mle)."""
    ar, s2 = _native.yule_walker(_data(x).ravel(), order, method)
    return ar, _np.sqrt(s2)


def burg(endog, order=1):
    """``(ar, sigma2)``: AR coefficients and the innovation variance by Burg's method, like statsmodels'
    ``burg``."""
    return _native.burg(_data(endog).ravel(), order)


def levinson_durbin(s, nlags=10):
    """``(sigma_v, arcoefs, pacf)`` from autocovariances ``s``, like statsmodels' ``levinson_durbin``
    with ``isacov=True`` (``sigma_v`` the innovation variance)."""
    return _native.levinson_durbin([float(v) for v in s], nlags)
