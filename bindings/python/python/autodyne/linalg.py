"""Linear algebra (``numpy.linalg`` / ``scipy.linalg`` names), on faer: inputs are read in place
with any strides; float32 inputs stay float32."""

import numpy as _np

from . import _autodyne as _native

__all__ = ["set_threads", "matmul", "solve", "inv", "det", "lstsq", "eig", "eigvals", "eigh", "svd", "pinv", "qr", "cholesky",
           "expm", "roots", "schur", "solve_sylvester", "solve_continuous_lyapunov", "solve_discrete_lyapunov",
           "solve_continuous_are", "solve_discrete_are"]


def set_threads(n):
    """Threads for large matrix operations (0: all cores, 1: single-threaded)."""
    _native.set_threads(n)


def _float(a):
    # dtype.char rather than comparing dtypes with types (several times faster, for small calls)
    if type(a) is _np.ndarray and a.dtype.char in "fd":
        return a
    a = _np.asarray(a)
    return a if a.dtype.char in "fd" else a.astype(_np.float64)


# Matrix product of 1-D or 2-D arrays: the native function itself (no Python frame in between, the
# call's overhead shows on small matrices); it converts lists and integer arrays to float64.
matmul = _native.matmul


def solve(a, b):
    """The solution of ``a @ x = b`` (``b`` a vector or a matrix of right-hand sides)."""
    return _native.solve(_float(a), _float(b))


def inv(a):
    """The inverse of a square matrix (``LinAlgError``-style ``ValueError`` if singular)."""
    return _native.inv(_float(a))


def det(a):
    """The determinant of a square matrix."""
    return _native.det(_float(a))


def lstsq(a, b):
    """``(x, rank, singular_values)``: the minimum-norm least-squares solution."""
    return _native.lstsq(_float(a), _float(b))


def eig(a):
    """``(w, v)``: eigenvalues and right eigenvectors (columns), complex."""
    return _native.eig(_float(a))


def eigvals(a):
    """The eigenvalues of a general square matrix, complex, in no particular order."""
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
    """The Moore-Penrose pseudo-inverse (through the SVD)."""
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


# ---- matrix equations (scipy.linalg) -------------------------------------------------------------

def schur(a, output="real"):
    """``(T, Z)`` with ``a = Z @ T @ Z.T``, ``T`` quasi-upper-triangular (``scipy.linalg.schur``, real)."""
    if output != "real":
        raise NotImplementedError("only output='real'")
    return _native.schur(_float(a))


def solve_sylvester(a, b, q):
    """``x`` with ``a @ x + x @ b = q`` (Bartels-Stewart), like ``scipy.linalg.solve_sylvester``."""
    return _native.solve_sylvester(_float(a), _float(b), _float(q))


def solve_continuous_lyapunov(a, q):
    """``x`` with ``a @ x + x @ a.T = q``, like ``scipy.linalg.solve_continuous_lyapunov``."""
    return _native.solve_continuous_lyapunov(_float(a), _float(q))


def solve_discrete_lyapunov(a, q, method=None):
    """``x`` with ``a @ x @ a.T - x + q = 0``, like ``scipy.linalg.solve_discrete_lyapunov``."""
    return _native.solve_discrete_lyapunov(_float(a), _float(q))


def solve_continuous_are(a, b, q, r):
    """The stabilizing solution of the continuous algebraic Riccati equation, like
    ``scipy.linalg.solve_continuous_are`` (matrix sign function, polished by Newton steps)."""
    return _native.solve_continuous_are(_float(a), _float(_np.atleast_2d(b)), _float(q), _float(_np.atleast_2d(r)))


def solve_discrete_are(a, b, q, r):
    """The stabilizing solution of the discrete algebraic Riccati equation, like
    ``scipy.linalg.solve_discrete_are`` (structure-preserving doubling, polished by Newton steps)."""
    return _native.solve_discrete_are(_float(a), _float(_np.atleast_2d(b)), _float(q), _float(_np.atleast_2d(r)))