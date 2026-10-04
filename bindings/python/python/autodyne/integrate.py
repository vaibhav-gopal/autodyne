"""Initial value problems with ``scipy.integrate.solve_ivp``'s interface: RK45 (the default), RK23
and, for stiff problems, Rosenbrock23 (MATLAB's ode23s). The step logic matches SciPy's, so RK45 and
RK23 take the same steps; the loop runs in Rust, calling ``fun`` from Python only for derivatives."""

import types as _types

import numpy as _np

from ._autodyne import integrate as _native

__all__ = ["solve_ivp"]


def solve_ivp(fun, t_span, y0, method="RK45", t_eval=None, events=None, args=None, rtol=1e-3, atol=1e-6,
              first_step=None, max_step=_np.inf):
    """Solves ``y' = fun(t, y)`` from ``y(t_span[0]) = y0``, like ``scipy.integrate.solve_ivp``. Returns
    an object with ``t``, ``y`` (states x times), ``t_events``, ``y_events``, ``nfev``, ``njev``,
    ``nlu``, ``status``, ``message`` and ``success``. ``events`` are callables ``g(t, y)``, with
    optional ``terminal`` and ``direction`` attributes as in SciPy."""
    if args is not None:
        f = fun
        fun = lambda t, y: f(t, y, *args)  # noqa: E731
    events = [] if events is None else (list(events) if isinstance(events, (list, tuple)) else [events])
    terminal = [bool(getattr(e, "terminal", False)) for e in events]
    direction = [float(getattr(e, "direction", 0.0)) for e in events]
    t0, t1 = map(float, t_span)
    out = _native.solve_ivp(fun, t0, t1, [float(v) for v in _np.atleast_1d(y0)], method, float(rtol), float(atol),
                            None if first_step is None else float(first_step), float(max_step),
                            None if t_eval is None else [float(v) for v in t_eval], events, terminal, direction)
    out["success"] = out["status"] >= 0
    return _types.SimpleNamespace(**out)
