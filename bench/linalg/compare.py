"""Linear algebra across languages and libraries: the same matrices, the same operations, each
timed natively, single-threaded, in f64, and every result checked against NumPy's.

Rust (autodyne, faer, nalgebra, Burn) runs from ``bench/rust/examples/linalg.rs``; Python times
NumPy (OpenBLAS's LAPACK), PyTorch (its LAPACK, MKL in the Windows / x86 wheels) and JAX (XLA)
in-process. Timings are the best of repeated runs (at least 5, up to a second), like the Rust side.

    python bench/linalg/compare.py [--out bench/linalg/RESULTS.md]

Run it on one core so no library can use more than one thread (some ignore thread settings): on
Windows, `start /affinity 1 python compare.py`; on Linux, `taskset -c 0 python compare.py`.
"""

import argparse
import json
import os
import platform
import subprocess
import sys
import tempfile
import time
from pathlib import Path

for var in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "RAYON_NUM_THREADS"):
    os.environ[var] = "1"
os.environ["XLA_FLAGS"] = "--xla_cpu_multi_thread_eigen=false intra_op_parallelism_threads=1"

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
LAPACK_PHASES = {}
OPS = ["matmul 64", "matmul 512", "solve 512", "det 300", "svdvals 300", "svd 300", "eigvals 300", "eigh 300", "eigvalsh 300"]


def generate(d):
    rng = np.random.default_rng(1)
    for n in (64, 512):
        np.save(d / f"matmul_{n}_a.npy", rng.standard_normal((n, n)))
        np.save(d / f"matmul_{n}_b.npy", rng.standard_normal((n, n)))
    np.save(d / "solve_512_a.npy", rng.standard_normal((512, 512)) + 512 ** 0.5 * np.eye(512))
    np.save(d / "solve_512_b.npy", rng.standard_normal(512))
    np.save(d / "square_300.npy", rng.standard_normal((300, 300)) / 300 ** 0.5)  # a moderate determinant
    m = rng.standard_normal((300, 300))
    np.save(d / "symmetric_300.npy", (m + m.T) / 2)


def best(f):
    times, start = [], time.perf_counter()
    while len(times) < 5 or (time.perf_counter() - start < 1.0 and len(times) < 300):
        t = time.perf_counter()
        r = f()
        times.append(time.perf_counter() - t)
    return min(times), r


def svd_residual(a, u, s, vt):
    return np.linalg.norm(a - (u * s) @ vt) / np.linalg.norm(a)


def eigh_residual(s, w, v):
    return np.linalg.norm(s @ v - v * w) / np.linalg.norm(s)


def eig_key(w):
    return np.concatenate([np.sort(w.real), np.sort(w.imag)])


def lapack_phases(a, s):
    """LAPACK's phases, called in NumPy's own OpenBLAS (64-bit integers): the reductions (dgebrd,
    dsytrd) and the iterations NumPy's values-only routines then run (dlasq1's dqds for singular
    values, dsterf for symmetric eigenvalues)."""
    import ctypes
    import glob
    libs = glob.glob(os.path.join(os.path.dirname(np.__file__), os.pardir, "numpy.libs", "*openblas64*"))
    libs += glob.glob(os.path.join(os.path.dirname(np.__file__), ".libs", "*openblas64*"))
    if not libs:
        print("no OpenBLAS (ILP64) library found next to NumPy: skipping LAPACK's phases", file=sys.stderr)
        return {}
    lib = ctypes.CDLL(libs[0])
    i64, p = ctypes.c_int64, ctypes.POINTER
    ref = lambda v: ctypes.byref(i64(v))
    ptr = lambda arr: arr.ctypes.data_as(p(ctypes.c_double))
    n = a.shape[0]
    info = i64(0)

    def gebrd(m):
        d, e, tq, tp = np.empty(n), np.empty(n - 1), np.empty(n), np.empty(n)
        query = np.empty(1)
        lib.scipy_dgebrd_64_(ref(n), ref(n), ptr(m), ref(n), ptr(d), ptr(e), ptr(tq), ptr(tp), ptr(query), ref(-1), ctypes.byref(info))
        work = np.empty(int(query[0]))
        lib.scipy_dgebrd_64_(ref(n), ref(n), ptr(m), ref(n), ptr(d), ptr(e), ptr(tq), ptr(tp), ptr(work), ref(work.size), ctypes.byref(info))
        return d, e

    def sytrd(m):
        d, e, tau = np.empty(n), np.empty(n - 1), np.empty(n - 1)
        query, lower = np.empty(1), ctypes.c_char_p(b"L")
        lib.scipy_dsytrd_64_(lower, ref(n), ptr(m), ref(n), ptr(d), ptr(e), ptr(tau), ptr(query), ref(-1), ctypes.byref(info), ctypes.c_size_t(1))
        work = np.empty(int(query[0]))
        lib.scipy_dsytrd_64_(lower, ref(n), ptr(m), ref(n), ptr(d), ptr(e), ptr(tau), ptr(work), ref(work.size), ctypes.byref(info), ctypes.c_size_t(1))
        return d, e

    def lasq1(d, e):
        d, e, work = d.copy(), np.append(e, 0.0), np.empty(4 * n)
        lib.scipy_dlasq1_64_(ref(n), ptr(d), ptr(e), ptr(work), ctypes.byref(info))
        return d

    def sterf(d, e):
        d, e = d.copy(), e.copy()
        lib.scipy_dsterf_64_(ref(n), ptr(d), ptr(e), ctypes.byref(info))
        return d

    out = {}
    out["bidiagonalization 300"], (bd, be) = best(lambda: gebrd(np.asfortranarray(a).copy(order="F")))
    out["bidiagonal singular values 300"], sv = best(lambda: lasq1(bd, be))
    out["tridiagonalization 300"], (td, te) = best(lambda: sytrd(np.asfortranarray(s).copy(order="F")))
    out["tridiagonal eigenvalues 300"], ev = best(lambda: sterf(td, te))
    # the phases compute what NumPy does
    assert np.allclose(np.sort(sv), np.sort(np.linalg.svd(a, compute_uv=False)), rtol=1e-10, atol=1e-12)
    assert np.allclose(np.sort(ev), np.linalg.eigvalsh(s), rtol=1e-10, atol=1e-12)
    return out


def python_libraries(d):
    """{library: {op: (seconds, result vector)}}."""
    inputs = {name: np.load(d / f"{name}.npy") for name in
              ["matmul_64_a", "matmul_64_b", "matmul_512_a", "matmul_512_b", "solve_512_a", "solve_512_b", "square_300", "symmetric_300"]}
    a, s = inputs["square_300"], inputs["symmetric_300"]
    out = {}

    numpy = {}
    for n in (64, 512):
        x, y = inputs[f"matmul_{n}_a"], inputs[f"matmul_{n}_b"]
        numpy[f"matmul {n}"] = best(lambda: x @ y)
    numpy["solve 512"] = best(lambda: np.linalg.solve(inputs["solve_512_a"], inputs["solve_512_b"]))
    numpy["det 300"] = best(lambda: np.array([np.linalg.det(a)]))
    numpy["svdvals 300"] = best(lambda: np.linalg.svd(a, compute_uv=False))
    t, (u, sv, vt) = best(lambda: np.linalg.svd(a))
    numpy["svd 300"] = (t, np.append(sv, svd_residual(a, u, sv, vt)))
    t, w = best(lambda: np.linalg.eigvals(a))
    numpy["eigvals 300"] = (t, eig_key(w))
    t, (w, v) = best(lambda: np.linalg.eigh(s))
    numpy["eigh 300"] = (t, np.append(w, eigh_residual(s, w, v)))
    numpy["eigvalsh 300"] = best(lambda: np.linalg.eigvalsh(s))
    out[f"NumPy {np.__version__}"] = numpy
    LAPACK_PHASES.update(lapack_phases(a, s))

    try:
        import torch
        torch.set_num_threads(1)
        T = {k: torch.from_numpy(v) for k, v in inputs.items()}
        ta, ts = T["square_300"], T["symmetric_300"]
        lib = {}
        for n in (64, 512):
            x, y = T[f"matmul_{n}_a"], T[f"matmul_{n}_b"]
            lib[f"matmul {n}"] = best(lambda: x @ y)
        lib["solve 512"] = best(lambda: torch.linalg.solve(T["solve_512_a"], T["solve_512_b"]))
        lib["det 300"] = best(lambda: torch.linalg.det(ta).reshape(1))
        lib["svdvals 300"] = best(lambda: torch.linalg.svdvals(ta))
        t, (u, sv, vt) = best(lambda: torch.linalg.svd(ta))
        lib["svd 300"] = (t, np.append(sv.numpy(), svd_residual(a, u.numpy(), sv.numpy(), vt.numpy())))
        t, w = best(lambda: torch.linalg.eigvals(ta))
        lib["eigvals 300"] = (t, eig_key(w.numpy()))
        t, (w, v) = best(lambda: torch.linalg.eigh(ts))
        lib["eigh 300"] = (t, np.append(w.numpy(), eigh_residual(s, w.numpy(), v.numpy())))
        lib["eigvalsh 300"] = best(lambda: torch.linalg.eigvalsh(ts))
        out[f"PyTorch {torch.__version__.split('+')[0]}"] = {k: (t, np.asarray(r)) for k, (t, r) in lib.items()}
    except ImportError:
        print("skipping PyTorch: not installed", file=sys.stderr)

    try:
        import jax
        jax.config.update("jax_enable_x64", True)
        import jax.numpy as jnp
        J = {k: jax.device_put(v) for k, v in inputs.items()}
        ja, js = J["square_300"], J["symmetric_300"]

        def timed(f, *args):
            g = jax.jit(f)
            jax.block_until_ready(g(*args))  # compile first
            return best(lambda: jax.block_until_ready(g(*args)))

        lib = {}
        for n in (64, 512):
            lib[f"matmul {n}"] = timed(jnp.matmul, J[f"matmul_{n}_a"], J[f"matmul_{n}_b"])
        lib["solve 512"] = timed(jnp.linalg.solve, J["solve_512_a"], J["solve_512_b"])
        lib["det 300"] = timed(lambda x: jnp.linalg.det(x).reshape(1), ja)
        lib["svdvals 300"] = timed(lambda x: jnp.linalg.svd(x, compute_uv=False), ja)
        t, (u, sv, vt) = timed(jnp.linalg.svd, ja)
        lib["svd 300"] = (t, np.append(sv, svd_residual(a, np.asarray(u), np.asarray(sv), np.asarray(vt))))
        try:
            t, w = timed(jnp.linalg.eigvals, ja)
            lib["eigvals 300"] = (t, eig_key(np.asarray(w)))
        except Exception as e:  # not on every platform
            print(f"JAX eigvals: {e}", file=sys.stderr)
        t, (w, v) = timed(jnp.linalg.eigh, js)
        lib["eigh 300"] = (t, np.append(w, eigh_residual(s, np.asarray(w), np.asarray(v))))
        lib["eigvalsh 300"] = timed(jnp.linalg.eigvalsh, js)
        out[f"JAX {jax.__version__}"] = {k: (t, np.asarray(r)) for k, (t, r) in lib.items()}
    except ImportError:
        print("skipping JAX: not installed", file=sys.stderr)
    return out


def rust_libraries(d):
    cmd = ["cargo", "run", "--quiet", "--release", "--example", "linalg", "--manifest-path", str(ROOT / "bench" / "rust" / "Cargo.toml"), "--", str(d)]
    lines = subprocess.run(cmd, check=True, capture_output=True, text=True, env=os.environ).stdout.splitlines()
    out, phases = {}, {}
    for line in lines:
        row = json.loads(line)
        if row["op"].startswith("phase: "):
            phases[row["op"][7:]] = row["seconds"]
            continue
        out.setdefault(row["library"], {})[row["op"]] = (row["seconds"], np.load(d / row["result"]))
    return out, phases


def check(op, got, want):
    """The largest difference relative to the reference's scale; residual tails are compared to 1e-10."""
    got, want = np.asarray(got, float).ravel(), np.asarray(want, float).ravel()
    if op in ("svd 300", "eigh 300"):
        if got[-1] > 1e-10:
            return float("inf")
        got, want = got[:-1], want[:-1]
    if op.startswith("svd") or op == "eigvalsh 300":
        got, want = np.sort(got), np.sort(want)
    if got.shape != want.shape:
        return float("inf")
    return float(np.abs(got - want).max() / max(np.abs(want).max(), 1e-300))


def fmt(seconds):
    return f"{seconds * 1e3:.2f} ms" if seconds >= 1e-3 else f"{seconds * 1e6:.1f} µs"


def main():
    parser = argparse.ArgumentParser(description="Linear algebra across languages and libraries")
    parser.add_argument("--out", type=Path, help="also write the Markdown table to this file")
    args = parser.parse_args()
    with tempfile.TemporaryDirectory() as tmp:
        d = Path(tmp)
        generate(d)
        results = python_libraries(d)
        rust, phases = rust_libraries(d)
        results.update(rust)
    reference = results[next(k for k in results if k.startswith("NumPy"))]
    libraries = list(results)
    lines = [
        "# Linear algebra across languages and libraries",
        "",
        f"Generated by `bench/linalg/compare.py`: {platform.processor() or platform.machine()}, one thread, f64. The same",
        "matrices (n x n, random normal; the symmetric one `(m + mᵀ) / 2`) go through each library, timed natively (best of",
        "repeated runs) and checked against NumPy: `!` marks a result off by more than 1e-8 relative (or a residual above 1e-10).",
        "Relative = the fastest time / this one.",
        "",
        "| Operation | " + " | ".join(libraries) + " |",
        "|---|" + "---:|" * len(libraries),
    ]
    for op in OPS:
        times = {lib: results[lib][op][0] for lib in libraries if op in results[lib]}
        fastest = min(times.values())
        cells = []
        for lib in libraries:
            if op not in results[lib]:
                cells.append("—")
                continue
            t, r = results[lib][op]
            bad = check(op, r, reference[op][1]) > 1e-8
            cells.append(f"{fmt(t)} ({fastest / t:.2f}){' !' if bad else ''}")
        lines.append(f"| {op} | " + " | ".join(cells) + " |")
    if phases:
        faer = results["faer"]
        lines += [
            "",
            "## faer's phases",
            "",
            "The reduction to bidiagonal (SVD) or tridiagonal (symmetric eigenvalues) form, timed alone; the rest of each",
            "decomposition is the iteration on that form (and, with vectors, accumulating them).",
            "",
            "| Decomposition | faer total | faer reduction | faer rest | LAPACK total | LAPACK reduction | LAPACK rest |",
            "|---|---:|---:|---:|---:|---:|---:|",
        ]
        numpy = reference
        for op, phase in [("svdvals 300", "bidiagonalization 300"), ("svd 300", "bidiagonalization 300"), ("eigvalsh 300", "tridiagonalization 300"), ("eigh 300", "tridiagonalization 300")]:
            total, reduce = faer[op][0], phases[phase]
            lt, lr = numpy[op][0], LAPACK_PHASES.get(phase, float("nan"))
            lines.append(f"| {op} | {fmt(total)} | {fmt(reduce)} | {fmt(total - reduce)} | {fmt(lt)} | {fmt(lr)} | {fmt(lt - lr)} |")
        lines += [
            "",
            "LAPACK: OpenBLAS's (NumPy's own library): the totals through NumPy, the reductions by calling `dgebrd` / `dsytrd`",
            "directly. Its values-only iterations, timed alone: "
            f"`dlasq1` (dqds, singular values of the bidiagonal) {fmt(LAPACK_PHASES.get('bidiagonal singular values 300', float('nan')))}, "
            f"`dsterf` (tridiagonal eigenvalues) {fmt(LAPACK_PHASES.get('tridiagonal eigenvalues 300', float('nan')))}.",
        ]
    text = "\n".join(lines) + "\n"
    print(text)
    if args.out:
        args.out.write_text(text, encoding="utf-8", newline="\n")


if __name__ == "__main__":
    main()
