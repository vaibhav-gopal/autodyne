"""autodyne.linalg matrix equations, autodyne.control and autodyne.integrate against SciPy (and
python-control when installed)."""

import numpy as np
import pytest
import scipy.linalg as sla
from scipy import integrate as si

from autodyne import control as ac
from autodyne import integrate as ai
from autodyne import linalg as al

rng = np.random.default_rng(5)


def close(a, b, tol=1e-9):
    np.testing.assert_allclose(np.asarray(a), np.asarray(b), rtol=tol, atol=tol)


def test_matrix_equations():
    n = 8
    a = rng.standard_normal((n, n))
    stable = a - (np.max(np.linalg.eigvals(a).real) + 1.0) * np.eye(n)
    q = (lambda m: m @ m.T + n * np.eye(n))(rng.standard_normal((n, n)))
    b = rng.standard_normal((n, 2))
    r = np.array([[1.0, 0.1], [0.1, 2.0]])
    t, z = al.schur(a)
    close(z @ t @ z.T, a, 1e-12)
    close(np.tril(t, -2), 0.0)
    c = rng.standard_normal((n, 3))
    bb = rng.standard_normal((3, 3))
    close(al.solve_sylvester(a, bb, c), sla.solve_sylvester(a, bb, c))
    close(al.solve_continuous_lyapunov(stable, -q), sla.solve_continuous_lyapunov(stable, -q))
    d = a / (1.2 * np.max(np.abs(np.linalg.eigvals(a))))
    close(al.solve_discrete_lyapunov(d, q), sla.solve_discrete_lyapunov(d, q))
    close(al.solve_continuous_are(stable, b, q, r), sla.solve_continuous_are(stable, b, q, r), 1e-8)
    close(al.solve_discrete_are(d, b, q, r), sla.solve_discrete_are(d, b, q, r), 1e-8)


def test_control_against_python_control():
    control = pytest.importorskip("control")
    A = np.array([[-1.0, 2.0, 0.0], [-2.0, -1.5, 1.0], [0.0, 0.3, -0.8]])
    B = np.array([[1.0], [0.0], [0.5]])
    C = np.array([[1.0, 0.0, 1.0]])
    D = np.array([[0.0]])
    close(ac.ctrb(A, B), control.ctrb(A, B))
    close(ac.obsv(A, C), control.obsv(A, C))
    close(ac.dcgain(A, B, C, D), control.dcgain(control.ss(A, B, C, D)))
    k, s, e = ac.lqr(A, B, np.eye(3), np.array([[0.5]]))
    kr, sr, er = control.lqr(A, B, np.eye(3), np.array([[0.5]]))
    close(k, kr, 1e-8)
    close(s, sr, 1e-8)
    gm, pm, wpc, wgc = ac.stability_margins([2.5], [1.0, 3.0, 2.0, 0.0])
    gr, pr, _, wpcr, wgcr, _ = control.stability_margins(control.tf([2.5], [1.0, 3.0, 2.0, 0.0]))
    close([gm, pm, wpc, wgc], [gr, pr, wpcr, wgcr], 1e-8)
    wc = ac.gram(A, B, C, D, "c")
    close(A @ wc + wc @ A.T + B @ B.T, 0.0, 1e-10)
    wn, zeta, poles = ac.damp(A)
    close(np.sort(wn), np.sort(np.abs(np.linalg.eigvals(A))))


def lotka(t, y):
    return np.array([1.5 * y[0] - y[0] * y[1], -3.0 * y[1] + y[0] * y[1]])


@pytest.mark.parametrize("method", ["RK45", "RK23"])
def test_solve_ivp_takes_scipys_steps(method):
    ours = ai.solve_ivp(lotka, (0.0, 10.0), [10.0, 5.0], method=method, rtol=1e-6, atol=1e-9)
    ref = si.solve_ivp(lotka, (0.0, 10.0), [10.0, 5.0], method=method, rtol=1e-6, atol=1e-9)
    assert ours.nfev == ref.nfev
    close(ours.t, ref.t, 1e-10)  # the same steps, summed in another order
    close(ours.y, ref.y, 1e-10)
    te = np.linspace(0, 10, 21)
    close(ai.solve_ivp(lotka, (0, 10), [10.0, 5.0], method=method, t_eval=te, rtol=1e-6, atol=1e-9).y,
          si.solve_ivp(lotka, (0, 10), [10.0, 5.0], method=method, t_eval=te, rtol=1e-6, atol=1e-9).y, 1e-10)


def test_events_args_stiffness_and_errors():
    def hit(t, y):
        return y[0]
    hit.terminal, hit.direction = True, -1

    sol = ai.solve_ivp(lambda t, y, g: [y[1], -g], (0, 10), [0.0, 10.0], events=hit, args=(9.81,), rtol=1e-10, atol=1e-12)
    assert sol.status == 1 and sol.success
    close(sol.t_events[0], [2 * 10 / 9.81])
    stiff = ai.solve_ivp(lambda t, y: [y[1], 1000 * (1 - y[0] ** 2) * y[1] - y[0]], (0, 3000), [2.0, 0.0], method="Rosenbrock23", rtol=1e-4)
    assert stiff.success and stiff.t.size < 2000

    def boom(t, y):
        raise RuntimeError("from the right-hand side")
    with pytest.raises(RuntimeError, match="right-hand side"):
        ai.solve_ivp(boom, (0, 1), [1.0])
    with pytest.raises(ValueError):
        ai.solve_ivp(lotka, (0, 1), [1.0, 1.0], method="nope")
