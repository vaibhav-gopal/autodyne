import gc

import numpy as np
import pytest
from scipy import signal

import autodyne


@pytest.mark.parametrize("dtype", [np.float32, np.float64])
def test_sum_on_any_layout(dtype):
    x = np.arange(24, dtype=dtype).reshape(2, 3, 4)
    for view in [x, x.T, x[:, ::-1, 1::2], x[1]]:
        assert np.isclose(autodyne.sum(view), view.sum())
        for axis in range(view.ndim):
            np.testing.assert_allclose(autodyne.sum(view, axis=axis), view.sum(axis=axis), rtol=1e-6)


def test_results_are_numpy_arrays_that_own_the_memory():
    x = np.linspace(-1, 1, 1001)
    y = autodyne.axpb(x, 2.0, 0.5)
    assert isinstance(y, np.ndarray) and y.dtype == np.float64 and y.flags.writeable
    np.testing.assert_allclose(y, 2.0 * x + 0.5)
    y += 1.0  # writable, and stays valid after the producing object is gone
    gc.collect()
    np.testing.assert_allclose(y, 2.0 * x + 1.5)


def test_inputs_are_read_in_place():
    x = np.arange(12.0).reshape(3, 4)
    # a reversed, strided view: no copy is made, and the values match
    np.testing.assert_allclose(autodyne.axpb(x[::-1, ::2], 1.0, 0.0), x[::-1, ::2])
    with pytest.raises(TypeError):
        autodyne.sum(np.arange(4, dtype=np.int16))


@pytest.mark.parametrize("axis", [0, 1])
def test_lowpass_matches_scipy(axis):
    rng = np.random.default_rng(1)
    x = rng.standard_normal((8, 4096))
    if axis == 0:
        x = np.ascontiguousarray(x.T)  # time along axis 0: strided lanes
    sos = signal.butter(2, 1000.0, fs=48000.0, output="sos")
    np.testing.assert_allclose(autodyne.lowpass(x, 1000.0, 48000.0, axis=axis), signal.sosfilt(sos, x, axis=axis), atol=1e-9)


def test_rfft_matches_numpy():
    x = np.random.default_rng(2).standard_normal((4, 1024))
    np.testing.assert_allclose(autodyne.rfft(x), np.fft.rfft(x), atol=1e-9)
    np.testing.assert_allclose(autodyne.rfft(x.astype(np.float32)), np.fft.rfft(x.astype(np.float32)), atol=1e-3)


def test_arrays_export_once():
    a = autodyne._autodyne.axpb(np.ones(3), 1.0, 0.0)
    np.from_dlpack(a)
    with pytest.raises(BufferError):
        np.from_dlpack(a)
