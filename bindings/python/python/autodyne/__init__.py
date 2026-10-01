"""autodyne: signals, processes and systems, from Python.

Arrays are exchanged with NumPy (and PyTorch, JAX...) through DLPack, so nothing is copied:
inputs are read in place with whatever strides they have, and results come back as NumPy
arrays that own autodyne's memory.
"""

import numpy as _np

from . import _autodyne as _native
from ._autodyne import Array

__all__ = ["Array", "sum", "axpb", "lowpass", "rfft"]


def _numpy(result):
    return _np.from_dlpack(result)


def sum(x, axis=None):
    """Sum of every element, or along ``axis`` (pairwise, so accurate for long arrays)."""
    result = _native.sum(x, axis)
    return result if axis is None else _numpy(result)


def axpb(x, a, b):
    """``a * x + b`` in a single pass, without temporaries."""
    return _numpy(_native.axpb(x, a, b))


def lowpass(x, cutoff, sample_rate, axis=-1):
    """2nd-order Butterworth low-pass along ``axis``, one filter per lane."""
    return _numpy(_native.lowpass(x, cutoff, sample_rate, axis))


def rfft(x):
    """Real FFT along the last axis (power-of-two length)."""
    return _numpy(_native.rfft(x))
