"""Arrays in GPU memory (wgpu: Vulkan, Metal, DirectX 12).

A :class:`GpuArray` keeps float32 or float64 values on the GPU between operations: arithmetic with
other GpuArrays and with Python numbers, ``exp``/``log``/``tanh``/``sin``/``cos``/``sqrt``/``abs``,
sums (all, or along an axis of a matrix) and FIR filtering all give new GpuArrays without a round
trip; ``.numpy()`` (or ``numpy.asarray``) brings the values back. ``.T`` and ``transpose`` are
views, and a transposed NumPy array goes up as it lies in memory.

    >>> import numpy as np
    >>> from autodyne import gpu
    >>> if gpu.available():
    ...     x = gpu.asarray(np.random.rand(1000, 64).astype(np.float32))
    ...     y = (x.T * 2.0 + 0.5).tanh().sum(axis=1).numpy()

float64 needs a GPU that computes in it (``gpu.supports("float64")``: most desktop GPUs under
Vulkan; Metal has none). Its exp/log/tanh/sin/cos are computed in software from f64 arithmetic,
since shader languages have them for 32-bit floats only.
"""

import numpy as _np

from ._autodyne import gpu as _gpu

available = _gpu.available
supports = _gpu.supports
sync = _gpu.sync
GpuArray = getattr(_gpu, "GpuArray", None)

__all__ = ["GpuArray", "asarray", "available", "supports", "sync"]


def asarray(x, dtype=None):
    """``x`` on the GPU: float32 stays float32, anything else becomes float64 (or ``dtype``)."""
    if GpuArray is not None and isinstance(x, GpuArray):
        if dtype is None or _np.dtype(dtype).name == x.dtype:
            return x
        x = x.numpy()
    if not available():
        raise RuntimeError("no GPU adapter is available")
    a = _np.asarray(x)
    if dtype is None:
        dtype = _np.float32 if a.dtype == _np.float32 else _np.float64
    # order='K' keeps a transposed array's memory layout, which the GPU copy keeps too
    return GpuArray(a.astype(dtype, order="K", copy=False))
