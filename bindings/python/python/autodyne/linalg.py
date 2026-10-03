"""Linear algebra (``numpy.linalg`` / ``scipy.linalg`` names), on faer: inputs are read in place
with any strides; float32 inputs stay float32."""

import numpy as _np

from . import _autodyne as _native

__all__ = ["set_threads", "matmul", "solve", "inv", "det", "lstsq", "eig", "eigvals", "eigh", "svd", "pinv", "qr", "cholesky",
           "expm", "roots"]


def set_threads(n):
    """Threads for large matrix operations (0: all cores, 1: single-threaded)."""
    _native.set_threads(n)


def _float(a):
    # dtype.char rather than comparing dtypes with types (several times faster, for small calls)
    if type(a) is _np.ndarray and a.dtype.char in "fd":
        return a
    a = _np.asarray(a)
    return a if a.dtype.char in "fd" else a.astype(_np.float64)


def matmul(a, b):
    """Matrix product of 1-D or 2-D arrays."""
    try:
        # float arrays go straight in (the common case, and the one where a call's overhead shows)
        return _native.matmul(a, b)
    except (TypeError, ValueError, AttributeError):
        return _native.matmul(_float(a), _float(b))


def solve(a, b):
    """The solution of ``a @ x = b`` (``b`` a vector or a matrix of right-hand sides)."""
    return _native.solve(_float(a), _float(b))


def inv(a):
    return _native.inv(_float(a))


def det(a):
    return _native.det(_float(a))


def lstsq(a, b):
    """``(x, rank, singular_values)``: the minimum-norm least-squares solution."""
    return _native.lstsq(_float(a), _float(b))


def eig(a):
    """``(w, v)``: eigenvalues and right eigenvectors (columns), complex."""
    return _native.eig(_float(a))


def eigvals(a):
    return _native.eigvals(_float(a))


def eigh(a):
    """``(w, v)`` of a symmetric matrix: ascending eigenvalues, orthonormal eigenvectors."""
    return _native.eigh(_float(a))


def svd(a, full_matrices=True, compute_uv=True):
    """``(u, s, vt)``, or ``s`` alone when ``compute_uv`` is false."""
    if not compute_uv:
        return _native.svdvals(_float(a))
    return _native.svd(_float(a), full_matrices)


def pinv(a):
    return _native.pinv(_float(a))


def qr(a):
    """Reduced ``(q, r)``."""
    return _native.qr(_float(a))


def cholesky(a, lower=True):
    """The Cholesky factor of a symmetric positive definite matrix (lower, or upper if ``lower`` is false)."""
    l = _native.cholesky(_float(a))
    return l if lower else l.T.copy()


def expm(a):
    """The matrix exponential (Pade scaling and squaring)."""
    return _native.expm(_float(a))


def roots(p):
    """Polynomial roots (coefficients highest power first), like ``numpy.roots``."""
    return _native.roots(_np.asarray(p, dtype=_np.float64))
