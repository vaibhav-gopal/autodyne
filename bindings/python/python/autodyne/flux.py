"""Differentiable signal processing: trace a function once, differentiate it, run it, or compile it
for IREE, XLA or a PJRT plugin.

Functions are written with :class:`Tracer` values, which record what is done to them: NumPy-style
operators and methods (``x * 2``, ``x.sin()``, ``x @ w``, ``x[1:]``, ``x.sum(axis=0)``), the
functions here (``where``, ``concatenate``, ``fft``, ``convolve``...) and the processors' sample
steps (``biquad``, ``svf``, ``ladder``, ``shape``, ``compressor``...). Tracing, the derivatives and
the compiled programs are all built in Rust.

    >>> import numpy as np
    >>> from autodyne import flux
    >>> g = flux.value_and_grad(lambda x: (x.sin() * x).sum(), [[3]])
    >>> value, grad = g(np.array([0.1, 0.2, 0.3], np.float32))

A :class:`Scan` runs a per-sample step over a signal and differentiates through it, with any
:class:`Loss` (mean squared error, the multi-resolution STFT loss, or a traced function)::

    chain = flux.Scan([[], [], []], [[], []], [], lambda p, s, x: step(p, s, x))
    g = chain.grad(params, xs, [0.0, 0.0], [target], loss=flux.Loss.stft([len(xs)]))
"""

import numpy as _np

from ._autodyne import flux as _flux

Tracer = _flux.Tracer
Mask = _flux.Mask
Graph = _flux.Graph
Loss = _flux.Loss
Scan = _flux.Scan
Program = _flux.Program
Backend = _flux.Backend
Executable = _flux.Executable

trace = _flux.trace
value_and_grad = _flux.value_and_grad
where = _flux.where
concatenate = _flux.concatenate
irfft = _flux.irfft
fft = _flux.fft
convolve = _flux.convolve
frames = _flux.frames
stft_magnitude = _flux.stft_magnitude
multi_resolution_stft = _flux.multi_resolution_stft
one_pole = _flux.one_pole
biquad = _flux.biquad
svf = _flux.svf
ladder = _flux.ladder
shape = _flux.shape
envelope = _flux.envelope
compressor = _flux.compressor

__all__ = ["Tracer", "Mask", "Graph", "Loss", "Scan", "Program", "Backend", "Executable", "Adam", "trace", "value_and_grad",
           "where", "concatenate", "irfft", "fft", "convolve", "frames", "stft_magnitude", "multi_resolution_stft",
           "one_pole", "biquad", "svf", "ladder", "shape", "envelope", "compressor"]


class Adam:
    """Adam (Kingma & Ba) on a list of parameter arrays: ``params = adam.step(params, grads)``."""

    def __init__(self, lr, beta1=0.9, beta2=0.999, eps=1e-8):
        self.lr, self.beta1, self.beta2, self.eps = lr, beta1, beta2, eps
        self.t, self.m, self.v = 0, None, None

    def step(self, params, grads):
        params = [_np.asarray(p, dtype=_np.float64) for p in params]
        grads = [_np.asarray(g, dtype=_np.float64) for g in grads]
        if self.m is None:
            self.m = [_np.zeros_like(p) for p in params]
            self.v = [_np.zeros_like(p) for p in params]
        self.t += 1
        c1, c2 = 1 - self.beta1 ** self.t, 1 - self.beta2 ** self.t
        out = []
        for p, g, m, v in zip(params, grads, self.m, self.v):
            m *= self.beta1
            m += (1 - self.beta1) * g
            v *= self.beta2
            v += (1 - self.beta2) * g * g
            out.append(p - self.lr * (m / c1) / (_np.sqrt(v / c2) + self.eps))
        return out
