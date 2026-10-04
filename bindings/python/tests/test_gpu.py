import numpy as np
import pytest

from autodyne import gpu

pytestmark = pytest.mark.skipif(not gpu.available(), reason="no GPU adapter")

DTYPES = [np.float32] + ([np.float64] if gpu.available() and gpu.supports("float64") else [])


def tolerance(dtype):
    return dict(rtol=2e-6, atol=1e-6) if dtype == np.float32 else dict(rtol=1e-14, atol=1e-300)


@pytest.mark.parametrize("dtype", DTYPES)
def test_arithmetic_and_functions(dtype):
    rng = np.random.default_rng(0)
    x, y = rng.standard_normal((37, 20)).astype(dtype), rng.standard_normal((37, 20)).astype(dtype)
    gx, gy = gpu.asarray(x), gpu.asarray(y)
    assert gx.shape == [37, 20] and gx.dtype == np.dtype(dtype).name
    tol = tolerance(dtype)
    for got, want in [
        (gx + gy, x + y), (gx - gy, x - y), (gx * gy, x * y), (gx / gy, x / y),
        (gx + 2.5, x + 2.5), (2.5 - gx, 2.5 - x), (gx * 3, x * 3), (gx / 3, x / 3), (3 / gx, 3 / x),
        (-gx, -x), (abs(gx), np.abs(x)), (gx.axpb(2.0, 0.5), 2.0 * x + 0.5),
        (gx.exp(), np.exp(x)), (abs(gx).log(), np.log(np.abs(x))), (gx.tanh(), np.tanh(x)),
        (gx.sin(), np.sin(x)), (gx.cos(), np.cos(x)), (abs(gx).sqrt(), np.sqrt(np.abs(x))),
    ]:
        np.testing.assert_allclose(got.numpy(), want, **tol)
    row, col = rng.standard_normal(20).astype(dtype), rng.standard_normal((37, 1)).astype(dtype)
    np.testing.assert_allclose((gx * gpu.asarray(row)).numpy(), x * row, **tol)
    np.testing.assert_allclose((gx - gpu.asarray(col)).numpy(), x - col, **tol)
    with pytest.raises(ValueError):
        gx + gpu.asarray(np.ones(7, dtype))


@pytest.mark.parametrize("dtype", DTYPES)
def test_views_and_sums(dtype):
    rng = np.random.default_rng(1)
    x = rng.standard_normal((37, 20)).astype(dtype)
    g = gpu.asarray(x)
    t = g.T
    assert t.shape == [20, 37] and not t.is_contiguous
    np.testing.assert_array_equal(np.asarray(t), x.T)
    # a transposed NumPy array goes up as it lies, and comes back as the transpose
    up = gpu.asarray(x.T)
    assert not up.is_contiguous
    np.testing.assert_array_equal(up.numpy(), x.T)
    tol = dict(rtol=1e-5, atol=1e-5) if dtype == np.float32 else dict(rtol=1e-12, atol=1e-12)
    np.testing.assert_allclose((t + gpu.asarray(np.ascontiguousarray(x.T))).numpy(), 2 * x.T, **tol)
    np.testing.assert_allclose(t.sum(axis=0).numpy(), x.T.sum(axis=0), **tol)
    np.testing.assert_allclose(t.sum(axis=-1).numpy(), x.T.sum(axis=1), **tol)
    np.testing.assert_allclose(float(g.sum()), x.sum(), **tol)
    np.testing.assert_array_equal(t.contiguous().numpy(), x.T)
    z = rng.standard_normal((2, 3, 4)).astype(dtype)
    np.testing.assert_array_equal(gpu.asarray(z).transpose(2, 0, 1).numpy(), z.transpose(2, 0, 1))


def test_fir_and_conversions():
    rng = np.random.default_rng(2)
    x = rng.standard_normal((3, 500)).astype(np.float32)
    taps = np.hanning(31) / np.hanning(31).sum()
    want = np.stack([np.convolve(lane, taps)[:500] for lane in x.astype(np.float64)])
    np.testing.assert_allclose(gpu.asarray(x).fir(taps).numpy(), want, rtol=1e-4, atol=1e-5)
    # integers and lists become float64 (where supported); dtypes don't mix
    if gpu.supports("float64"):
        assert gpu.asarray([1, 2, 3]).dtype == "float64"
        with pytest.raises(TypeError):
            gpu.asarray(x) + gpu.asarray(x.astype(np.float64))
