"""FFTs of any length along the last axis (``numpy.fft`` names)."""

import numpy as _np

from . import _autodyne as _native

__all__ = ["fft", "ifft", "rfft", "irfft"]


def fft(x):
    return _native.fft(_np.asarray(x), False)


def ifft(x):
    return _native.fft(_np.asarray(x), True)


def rfft(x):
    return _native.rfft(x)


def irfft(x, n=None):
    return _native.irfft(_np.asarray(x, dtype=_np.complex128), n)
