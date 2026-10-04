"""Control analysis with ``python-control``'s names: controllability and observability matrices,
Gramians, DC gain, damping, linear quadratic regulators and stability margins. Systems are given
as state-space matrices (``dt`` for a discrete system) or, for margins, as a transfer function."""

import numpy as _np

from ._autodyne import control as _native

__all__ = ["ctrb", "obsv", "gram", "dcgain", "damp", "lqr", "dlqr", "stability_margins", "margin"]


def _m(x):
    return _np.atleast_2d(_np.asarray(x, dtype=float))


def ctrb(A, B):
    """The controllability matrix ``[B, AB, ..., A^(n-1) B]``."""
    return _native.ctrb(_m(A), _m(B))


def obsv(A, C):
    """The observability matrix ``[C; CA; ...; C A^(n-1)]``."""
    return _native.obsv(_m(A), _m(C))


def gram(A, B, C, D, kind, dt=None):
    """The controllability (``kind='c'``) or observability (``'o'``) Gramian of a stable system."""
    return _native.gram(_m(A), _m(B), _m(C), _m(D), kind, dt)


def dcgain(A, B, C, D, dt=None):
    """The steady-state gain (outputs x inputs)."""
    return _native.dcgain(_m(A), _m(B), _m(C), _m(D), dt)


def damp(A, dt=None):
    """``(wn, zeta, poles)``: natural frequencies (rad/s) and damping ratios of the eigenvalues of ``A``."""
    return _native.damp(_m(A), dt)


def lqr(A, B, Q, R):
    """``(K, S, E)``: the continuous-time LQR gain, Riccati solution and closed-loop poles."""
    return _native.lqr(_m(A), _m(B), _m(Q), _m(R), False)


def dlqr(A, B, Q, R):
    """``(K, S, E)``: the discrete-time LQR gain, Riccati solution and closed-loop poles."""
    return _native.lqr(_m(A), _m(B), _m(Q), _m(R), True)


def stability_margins(num, den, dt=None):
    """``(gm, pm, wpc, wgc)`` of the open loop ``num / den`` (polynomials, highest power first; ``dt``
    for a discrete loop): the gain margin (a ratio), the phase margin (degrees), and the phase and
    gain crossover frequencies (rad/s)."""
    return _native.stability_margins([float(v) for v in _np.atleast_1d(num)], [float(v) for v in _np.atleast_1d(den)], dt)


def margin(num, den, dt=None):
    """``(gm, pm, wpc, wgc)``, like ``control.margin`` (see :func:`stability_margins`)."""
    return stability_margins(num, den, dt)
