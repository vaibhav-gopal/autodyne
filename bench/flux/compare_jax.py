"""flux against JAX on the same XLA: compile time and run time of the same models.

flux's programs (written by `cargo run --release --features flux --example flux_programs -- DIR`)
are compiled from their StableHLO text by JAX's XLA backend; the same models written in
`jax.numpy` are traced, lowered and compiled by `jax.jit`. Inputs are on the device before timing,
both outputs are checked against flux's interpreter, and run times are medians.

    python compare_jax.py DIR [RESULTS.md] [--platform cpu|cuda]
"""

import argparse
import platform
import statistics
import time
from pathlib import Path

import numpy as np
import jax
import jax.extend
import jax.numpy as jnp
from jax import lax

jax.config.update("jax_default_matmul_precision", "highest")  # flux asks XLA for full f32 products

FS = 48_000.0


# THE MODELS, IN JAX ==============================================================================

def one_pole_scan(cutoff, xs, s0):
    a = 1.0 - jnp.exp(-2.0 * jnp.pi * cutoff / FS)

    def step(s, x):
        y = s + a * (x - s)
        return y, y

    last, ys = lax.scan(step, s0, xs)
    return ys, last


def one_pole_forward(cutoff, xs, s0):
    return one_pole_scan(cutoff, xs, s0)


def one_pole_grad(cutoff, xs, targets, s0):
    def loss(c, s, x):
        ys, _ = one_pole_scan(c, x, s)
        return jnp.mean((ys - targets) ** 2)

    value, (dc, ds, dx) = jax.value_and_grad(loss, argnums=(0, 1, 2))(cutoff, s0, xs)
    return value, dc, ds, dx


def peaking(f, gain_db, q=1.0):
    w = 2.0 * jnp.pi * f / FS
    cos_w, alpha = jnp.cos(w), jnp.sin(w) / (2.0 * q)
    a = jnp.sqrt(10.0 ** (gain_db / 20.0))
    a0 = 1.0 + alpha / a
    return (1.0 + alpha * a) / a0, -2.0 * cos_w / a0, (1.0 - alpha * a) / a0, -2.0 * cos_w / a0, (1.0 - alpha / a) / a0


def eq_drive(logf, g10, drive, xs, s0a, s0b):
    b0, b1, b2, a1, a2 = peaking(jnp.exp(logf), g10 * 10.0)

    def step(s, x):
        y = b0 * x + s[0]
        return jnp.stack([b1 * x - a1 * y + s[1], b2 * x - a2 * y]), jnp.tanh(y * drive)

    _, ys = lax.scan(step, jnp.stack([s0a, s0b]), xs)
    return ys


def hann(n):
    return 0.5 - 0.5 * np.cos(2.0 * np.pi * np.arange(n) / n)


def stft_magnitude(x, n_fft, hop, window):
    n = x.shape[-1]
    low = high = window // 2
    while (n + low + high - window) % hop:
        high += 1
    padded = jnp.pad(x, (low, high))
    count = 1 + (padded.shape[0] - window) // hop
    frames = padded[np.arange(count)[:, None] * hop + np.arange(window)[None, :]] * hann(window)
    spectrum = jnp.fft.rfft(jnp.pad(frames, ((0, 0), (0, n_fft - window))), axis=-1)
    return jnp.sqrt(jnp.maximum(spectrum.real ** 2 + spectrum.imag ** 2, 1e-8))


RESOLUTIONS = [(512, 128, 512), (128, 32, 128), (32, 8, 32)]


def multi_resolution_stft(y, t):
    total = 0.0
    for n_fft, hop, window in RESOLUTIONS:
        s, m = stft_magnitude(y, n_fft, hop, window), stft_magnitude(t, n_fft, hop, window)
        total = total + jnp.sqrt(jnp.sum((m - s) ** 2)) / jnp.sqrt(jnp.sum(m ** 2)) + jnp.mean(jnp.abs(jnp.log(s) - jnp.log(m)))
    return total / len(RESOLUTIONS)


def eq_drive_stft_grad(logf, g10, drive, xs, target, s0a, s0b):
    def loss(logf, g10, drive, s0a, s0b, xs):
        return multi_resolution_stft(eq_drive(logf, g10, drive, xs, s0a, s0b), target)

    value, grads = jax.value_and_grad(loss, argnums=(0, 1, 2, 3, 4, 5))(logf, g10, drive, s0a, s0b, xs)
    return (value, *grads)


def spectral_model_grad(x, gain, w):
    n = x.shape[-1]

    def loss(gain, w):
        spectrum = jnp.fft.rfft(x, axis=-1)
        y = jnp.tanh(jnp.fft.irfft(spectrum * gain, n=n, axis=-1) @ w)
        return jnp.mean(y * y)

    value, (dg, dw) = jax.value_and_grad(loss, argnums=(0, 1))(gain, w)
    return value, dg, dw


MODELS = {
    "one_pole_forward": ("one-pole low-pass, 48k samples", one_pole_forward),
    "one_pole_grad": ("one-pole MSE gradient, 48k samples", one_pole_grad),
    "eq_drive_stft_grad": ("EQ + drive, multi-resolution STFT loss gradient, 2048 samples", eq_drive_stft_grad),
    "spectral_model_grad": ("rfft -> gain -> irfft -> dense -> tanh gradient, 256 x 1024", spectral_model_grad),
}


# TIMING ==========================================================================================

def median_time(run, repeats, budget=3.0):
    jax.block_until_ready(run())  # warm up
    times = []
    start = time.perf_counter()
    while len(times) < repeats or (time.perf_counter() - start < budget and len(times) < 200):
        t = time.perf_counter()
        jax.block_until_ready(run())
        times.append(time.perf_counter() - t)
        if len(times) >= repeats and time.perf_counter() - start > budget:
            break
    return statistics.median(times)


def max_relative_error(got, want):
    got, want = np.asarray(got, dtype=np.float64), np.asarray(want, dtype=np.float64)
    scale = np.maximum(np.abs(want).max(), 1e-12)
    return float(np.abs(got - want).max() / scale)


def fmt(seconds):
    if seconds >= 1.0:
        return f"{seconds:.2f} s"
    if seconds >= 1e-3:
        return f"{seconds * 1e3:.2f} ms"
    return f"{seconds * 1e6:.1f} µs"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("dir")
    parser.add_argument("results", nargs="?")
    parser.add_argument("--platform", default="cpu")
    parser.add_argument("--repeats", type=int, default=20)
    args = parser.parse_args()
    root = Path(args.dir)
    backend = jax.extend.backend.get_backend(args.platform)
    device = backend.local_devices()[0]
    # JAX's first compilation pays for its own start-up; do that before measuring
    jax.jit(lambda x: jnp.sin(x) + 1.0).lower(jax.device_put(np.ones(4, np.float32), device)).compile()

    rows = []
    for line in (root / "cases.tsv").read_text().splitlines():
        name, n_in, n_out, build_ms = line.split("\t")
        title, model = MODELS[name]
        inputs = [np.load(root / f"{name}_in{k}.npy") for k in range(int(n_in))]
        want = [np.load(root / f"{name}_out{k}.npy") for k in range(int(n_out))]
        on_device = [jax.device_put(x, device) for x in inputs]

        # flux: the emitted StableHLO, compiled by XLA
        text = (root / f"{name}.mlir").read_text()
        t = time.perf_counter()
        exe = backend.compile_and_load(text, [device])
        flux_compile = time.perf_counter() - t
        run_flux = lambda: [o[0] for o in exe.execute_sharded(on_device).disassemble_into_single_device_arrays()]
        flux_out = run_flux()
        flux_run = median_time(run_flux, args.repeats)

        # JAX: trace, lower and compile the same model
        t = time.perf_counter()
        compiled = jax.jit(model).lower(*on_device).compile()
        jax_compile = time.perf_counter() - t
        run_jax = lambda: compiled(*on_device)
        jax_out = jax.tree_util.tree_leaves(run_jax())
        jax_run = median_time(run_jax, args.repeats)

        flux_err = max(max_relative_error(g, w) for g, w in zip(flux_out, want))
        agree = max(max_relative_error(g, w) for g, w in zip(jax_out, flux_out))
        rows.append((title, float(build_ms) / 1e3, flux_compile, jax_compile, flux_run, jax_run, flux_err, agree))
        print(f"{name}: flux compile {fmt(flux_compile)} run {fmt(flux_run)} | jax compile {fmt(jax_compile)} run {fmt(jax_run)}"
              f" | flux vs interpreter {flux_err:.1e}, jax vs flux {agree:.1e}")

    out = [
        "# flux against JAX on the same XLA",
        "",
        f"Generated by `compare_jax.py` from `examples/flux_programs.rs`: JAX {jax.__version__}, XLA {args.platform}"
        f" ({device.device_kind}), {platform.processor() or platform.machine()}.",
        "",
        "- flux build: tracing and emitting StableHLO in Rust. flux compile: XLA compiling that text.",
        "- JAX compile: `jax.jit(f).lower(...).compile()` (tracing, lowering and XLA compilation).",
        "- Run: median wall time with the inputs already on the device.",
        "- Agreement: the largest difference between JAX's and flux's outputs, relative to the largest output.",
        "",
        "| Model | flux build | flux compile | JAX compile | flux run | JAX run | JAX / flux run | Agreement |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for title, build, fc, jc, fr, jr, _, agree in rows:
        out.append(f"| {title} | {fmt(build)} | {fmt(fc)} | {fmt(jc)} | {fmt(fr)} | {fmt(jr)} | {jr / fr:.2f}x | {agree:.1e} |")
    out += [
        "",
        "Where flux differs:",
        "",
        "- Its graphs hold real arrays, so a complex spectrum travels as separate real and imaginary arrays,",
        "  with more elementwise work and memory traffic than JAX's complex arrays. That is the gap on the",
        "  spectral model.",
        "- Its scan gradients save only each step's starting state and recompute the step in the reverse loop",
        "  (checkpointing, which uses less memory), while JAX saves every step's intermediate values.",
        "- It compiles sooner: its programs come straight from the trace, while `jax.jit` traces Python and",
        "  lowers it first. Both compiles include XLA's own.",
    ]
    text = "\n".join(out) + "\n"
    if args.results:
        Path(args.results).write_text(text, encoding="utf-8")
    print(text)


if __name__ == "__main__":
    main()
