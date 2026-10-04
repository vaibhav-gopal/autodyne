"""FFTs of any length along the last axis (``numpy.fft`` names)."""

import numpy as _np

from . import _autodyne as _native

__all__ = ["fft", "ifft", "rfft", "irfft"]


def fft(x):
    """Complex FFT along the last axis, like ``numpy.fft.fft`` (real input is made complex)."""
    return _native.fft(_np.asarray(x), False)


def ifft(x):
    """Inverse complex FFT along the last axis, scaled by ``1 / n``, like ``numpy.fft.ifft``."""
    return _native.fft(_np.asarray(x), True)


def rfft(x):
    """Real FFT along the last axis: ``n // 2 + 1`` bins, like ``numpy.fft.rfft``."""
    return _native.rfft(x)


def irfft(x, n=None):
    """Inverse real FFT along the last axis: ``n`` samples (default ``2 * (bins - 1)``), like ``numpy.fft.irfft``."""
    return _native.irfft(_np.asarray(x, dtype=_np.complex128), n)
