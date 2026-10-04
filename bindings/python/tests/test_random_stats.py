"""autodyne.random (statistically, against SciPy's distributions) and autodyne.stats (against NumPy,
SciPy and, when installed, statsmodels)."""

import numpy as np
import pytest
from scipy import stats as ss

from autodyne import random as ar
from autodyne import stats as ast

rng = np.random.default_rng(3)


def close(a, b, tol=1e-10):
    np.testing.assert_allclose(np.asarray(a), np.asarray(b), rtol=tol, atol=tol)


# ---- random --------------------------------------------------------------------------------------

def test_seeds_shapes_and_dtypes():
    a, b = ar.default_rng(5), ar.default_rng(5)
    np.testing.assert_array_equal(a.normal(size=10), b.normal(size=10))
    assert a.normal(size=(2, 3)).shape == (2, 3)
    assert isinstance(a.normal(), float)
    assert a.standard_normal(4, dtype=np.float32).dtype == np.float32
    assert a.poisson(3.0, size=5).dtype == np.int64
    assert isinstance(a.binomial(10, 0.5), int)
    x = a.integers(3, 7, size=1000)
    assert x.min() >= 3 and x.max() <= 6
    assert sorted(a.permutation(10)) == list(range(10))
    c = a.choice(100, 20, replace=False)
    assert len(set(c.tolist())) == 20
    kids = a.spawn(2)
    assert not np.array_equal(kids[0].random(5), kids[1].random(5))
    with pytest.raises(ValueError):
        a.normal(0.0, -1.0)


@pytest.mark.parametrize("draw,dist", [
    (lambda g, n: g.standard_normal(n), ss.norm()),
    (lambda g, n: g.normal(2.0, 3.0, n), ss.norm(2.0, 3.0)),
    (lambda g, n: g.exponential(2.0, n), ss.expon(scale=2.0)),
    (lambda g, n: g.gamma(0.4, 1.5, n), ss.gamma(0.4, scale=1.5)),
    (lambda g, n: g.gamma(7.0, 1.0, n), ss.gamma(7.0)),
    (lambda g, n: g.beta(2.0, 3.0, n), ss.beta(2.0, 3.0)),
    (lambda g, n: g.chisquare(3.0, n), ss.chi2(3.0)),
    (lambda g, n: g.standard_t(5.0, n), ss.t(5.0)),
    (lambda g, n: g.laplace(1.0, 0.5, n), ss.laplace(1.0, 0.5)),
    (lambda g, n: g.lognormal(0.0, 0.7, n), ss.lognorm(0.7)),
    (lambda g, n: g.uniform(-1.0, 4.0, n), ss.uniform(-1.0, 5.0)),
])
def test_continuous_distributions_pass_kolmogorov_smirnov(draw, dist):
    x = draw(ar.default_rng(11), 50_000)
    assert ss.kstest(x, dist.cdf).pvalue > 1e-3


@pytest.mark.parametrize("draw,dist", [
    (lambda g, n: g.poisson(3.5, n), ss.poisson(3.5)),
    (lambda g, n: g.poisson(80.0, n), ss.poisson(80.0)),
    (lambda g, n: g.binomial(30, 0.2, n), ss.binom(30, 0.2)),
    (lambda g, n: g.binomial(500, 0.6, n), ss.binom(500, 0.6)),
    (lambda g, n: g.geometric(0.3, n), ss.geom(0.3)),
])
def test_discrete_distributions_pass_chi_squared(draw, dist):
    x = draw(ar.default_rng(12), 50_000)
    lo, hi = int(dist.ppf(0.001)), int(dist.ppf(0.999))
    edges = np.arange(lo, hi + 2)
    observed = np.histogram(np.clip(x, lo, hi), edges)[0]
    expected = np.diff(dist.cdf(edges - 1))
    expected[0] += dist.cdf(lo - 1)
    expected[-1] += dist.sf(hi)
    assert ss.chisquare(observed, expected * len(x) / expected.sum()).pvalue > 1e-3


def test_processes():
    g = ar.default_rng(4)
    assert g.colored_noise(1.0, 1000).shape == (1000,)
    assert g.brownian_motion(10, 0.1)[0] == 0.0
    ou = g.ornstein_uhlenbeck(100_000, 0.1, 0.5, 2.0, 1.0, 2.0)
    assert abs(ou.mean() - 2.0) < 0.05 and abs(ou.var() - 1.0) < 0.1
    y = g.arma([1.0, -0.5], [1.0], 50_000, burnin=100)
    assert abs(np.corrcoef(y[:-1], y[1:])[0, 1] - 0.5) < 0.02
    t = g.poisson_process(10.0, 100.0)
    assert abs(len(t) - 1000) < 120 and np.all(np.diff(t) > 0)


# ---- stats ---------------------------------------------------------------------------------------

@pytest.mark.parametrize("dtype", [np.float32, np.float64])
def test_order_statistics_and_moments(dtype):
    x = rng.standard_normal((4, 301)).astype(dtype)
    tol = 1e-5 if dtype == np.float32 else 1e-12
    for method in ["linear", "lower", "higher", "nearest", "midpoint"]:
        close(ast.quantile(x, [0.05, 0.5, 0.99], axis=1, method=method), np.quantile(x, [0.05, 0.5, 0.99], axis=1, method=method), tol)
    close(ast.quantile(x, 0.3), np.quantile(x, 0.3), tol)
    close(ast.percentile(x.T, 90, axis=0), np.percentile(x.T, 90, axis=0), tol)
    close(ast.median(x, axis=1), np.median(x, axis=1), tol)
    close(ast.median(x), np.median(x), tol)
    close(ast.median(x[:, :300], axis=1, keepdims=True), np.median(x[:, :300], axis=1, keepdims=True), tol)
    close(ast.skew(x, axis=1), ss.skew(x, axis=1), tol * 10)
    close(ast.skew(x.T, bias=False), ss.skew(x.T, bias=False), tol * 10)
    close(ast.kurtosis(x, axis=1), ss.kurtosis(x, axis=1), tol * 10)
    close(ast.kurtosis(x.T, fisher=False, bias=False), ss.kurtosis(x.T, fisher=False, bias=False), tol * 10)
    close(ast.moment(x, 3, axis=1), ss.moment(x, 3, axis=1), tol * 10)
    close(ast.zscore(x, axis=1, ddof=1), ss.zscore(x, axis=1, ddof=1), tol * 10)


def test_histograms_and_correlation():
    x = rng.standard_normal(1000) ** 3
    for bins in [10, "auto", "fd", "sturges", "scott", "rice", "sqrt", [-5.0, -1.0, 0.0, 0.5, 3.0]]:
        h, e = ast.histogram(x, bins)
        hr, er = np.histogram(x, bins)
        np.testing.assert_array_equal(h, hr)
        close(e, er, 1e-14)
    h, _ = ast.histogram(x, 6, range=(-2, 2), weights=np.abs(x), density=True)
    close(h, np.histogram(x, 6, range=(-2, 2), weights=np.abs(x), density=True)[0])
    m = rng.standard_normal((5, 200))
    close(ast.cov(m), np.cov(m))
    close(ast.cov(m.T, rowvar=False, bias=True), np.cov(m.T, rowvar=False, bias=True))
    close(ast.corrcoef(m), np.corrcoef(m))
    close(ast.cov(m[0]), np.cov(m[0]))


def test_time_series_against_statsmodels():
    stattools = pytest.importorskip("statsmodels.tsa.stattools")
    from statsmodels.regression.linear_model import burg, yule_walker
    from scipy import signal
    y = signal.lfilter([1.0], [1.0, -0.7, 0.2], rng.standard_normal(2000))
    close(ast.acovf(y, nlag=20), stattools.acovf(y, nlag=20))
    close(ast.acf(y), stattools.acf(y))
    close(ast.acf(y, adjusted=True, nlags=15), stattools.acf(y, adjusted=True, nlags=15))
    close(ast.pacf(y, nlags=12), stattools.pacf(y, nlags=12), 1e-9)
    close(ast.pacf(y, nlags=12, method="ywm"), stattools.pacf(y, nlags=12, method="ywm"), 1e-9)
    rho, sigma = ast.yule_walker(y, 3)
    rho_r, sigma_r = yule_walker(y, 3, result_object=False)
    close(rho, rho_r, 1e-9)
    close(sigma, sigma_r, 1e-9)
    ar, s2 = ast.burg(y, 3)
    ar_r, s2_r = burg(y, 3)
    close(ar, ar_r, 1e-9)
    close(s2, s2_r, 1e-9)
