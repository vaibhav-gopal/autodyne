"""autodyne: signals, processes and systems, from Python.

Arrays are exchanged with NumPy (and PyTorch, JAX...) through DLPack, so nothing is copied:
inputs are read in place with whatever strides they have, and results come back as NumPy
arrays that own autodyne's memory.
"""

import numpy as _np

from . import _autodyne as _native
from ._autodyne import Array

__all__ = ["Array", "sum", "axpb", "lowpass", "rfft", "add", "sub", "mul", "div", "minimum", "maximum", "cast"]


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


_DTYPES = {"float32": "f32", "float64": "f64", "int8": "i8", "int16": "i16", "int32": "i32", "int64": "i64",
           "uint8": "u8", "uint16": "u16", "uint32": "u32", "uint64": "u64",
           "complex64": "complex_f32", "complex128": "complex_f64"}


def _binary(op, x, y, keep_float):
    return _numpy(_native.binary(op, _np.asarray(x), _np.asarray(y), keep_float))


def add(x, y, keep_float=False):
    """``x + y`` for any numeric dtypes, broadcast. Promotion follows NumPy's table, but a value
    that wouldn't convert exactly raises instead of silently rounding, and a signed integer with
    uint64 stays int64. ``keep_float=True`` lets a float keep its width against integers."""
    return _binary("add", x, y, keep_float)


def sub(x, y, keep_float=False):
    return _binary("sub", x, y, keep_float)


def mul(x, y, keep_float=False):
    return _binary("mul", x, y, keep_float)


def div(x, y, keep_float=False):
    """True division (integers give float64)."""
    return _binary("div", x, y, keep_float)


def minimum(x, y, keep_float=False):
    return _binary("min", x, y, keep_float)


def maximum(x, y, keep_float=False):
    return _binary("max", x, y, keep_float)


def cast(x, dtype, mode="checked"):
    """Convert to ``dtype`` (a NumPy dtype); ``mode`` is "checked", "saturating" or "wrapping"."""
    name = _DTYPES[_np.dtype(dtype).name]
    return _numpy(_native.cast(_np.asarray(x), name, mode))