"""flux's benchmark models in PyTorch, on the inputs `examples/flux_programs.rs` writes, checked
against flux's interpreter: autograd eager, and compiled (`torch.compile` of the loss, its backward
through AOTAutograd; the recurrences through PyTorch's `scan` operator so they are not unrolled).

    python bench/flux/compare_torch.py DIR [--out FILE] [--no-compile]

Prints one Markdown row per model and mode: median run time and the largest difference from flux's
outputs relative to the largest output. `torch.compile` needs a C++ compiler (gcc on Linux).
"""

import argparse
import statistics
import sys
import time
from pathlib import Path

import numpy as np
import torch

torch.set_num_threads(1)
FS = 48_000.0

try:
    from torch._higher_order_ops.scan import scan as hop_scan
except Exception:  # noqa: BLE001
    hop_scan = None


# THE MODELS ======================================================================================

def coefficient(cutoff):
    return 1.0 - torch.exp(-2.0 * torch.pi * cutoff / FS)


def one_pole_loop(cutoff, xs, s0):
    a = coefficient(cutoff)
    s, ys = s0, []
    for x in xs:
        s = s + a * (x - s)
        ys.append(s)
    return torch.stack(ys), s


def one_pole_scan(cutoff, xs, s0):
    a = coefficient(cutoff)

    def step(s, x):
        y = s + a * (x - s)
        return y, y.clone()

    last, ys = hop_scan(step, s0, xs)
    return ys, last


def peaking(f, gain_db, q=1.0):
    w = 2.0 * torch.pi * f / FS
    cos_w, alpha = torch.cos(w), torch.sin(w) / (2.0 * q)
    a = torch.sqrt(10.0 ** (gain_db / 20.0))
    a0 = 1.0 + alpha / a
    return (1.0 + alpha * a) / a0, -2.0 * cos_w / a0, (1.0 - alpha * a) / a0, -2.0 * cos_w / a0, (1.0 - alpha / a) / a0


def eq_drive_loop(logf, g10, drive, xs, s0a, s0b):
    b0, b1, b2, a1, a2 = peaking(torch.exp(logf), g10 * 10.0)
    sa, sb, ys = s0a, s0b, []
    for x in xs:
        y = b0 * x + sa
        sa, sb = b1 * x - a1 * y + sb, b2 * x - a2 * y
        ys.append(torch.tanh(y * drive))
    return torch.stack(ys)


def eq_drive_scan(logf, g10, drive, xs, s0a, s0b):
    b0, b1, b2, a1, a2 = peaking(torch.exp(logf), g10 * 10.0)

    def step(state, x):
        sa, sb = state
        y = b0 * x + sa
        return (b1 * x - a1 * y + sb, b2 * x - a2 * y), torch.tanh(y * drive)

    _, ys = hop_scan(step, (s0a, s0b), xs)
    return ys


def stft_magnitude(x, n_fft, hop, window):
    n = x.shape[-1]
    low = high = window // 2
    while (n + low + high - window) % hop:
        high += 1
    padded = torch.nn.functional.pad(x, (low, high))
    frames = padded.unfold(0, window, hop) * torch.hann_window(window, periodic=True, dtype=x.dtype)
    spectrum = torch.fft.rfft(frames, n=n_fft, dim=-1)
    return torch.sqrt(torch.clamp(spectrum.real ** 2 + spectrum.imag ** 2, min=1e-8))


RESOLUTIONS = [(512, 128, 512), (128, 32, 128), (32, 8, 32)]


def multi_resolution_stft(y, t):
    total = 0.0
    for n_fft, hop, window in RESOLUTIONS:
        s, m = stft_magnitude(y, n_fft, hop, window), stft_magnitude(t, n_fft, hop, window)
        total = total + torch.sqrt(torch.sum((m - s) ** 2)) / torch.sqrt(torch.sum(m ** 2)) + torch.mean(torch.abs(torch.log(s) - torch.log(m)))
    return total / len(RESOLUTIONS)


def spectral_loss(x, gain, w):
    n = x.shape[-1]
    y = torch.tanh(torch.fft.irfft(torch.fft.rfft(x, dim=-1) * gain, n=n, dim=-1) @ w)
    return torch.mean(y * y)


# RUNNERS (one per model; `fns` holds the eager or compiled pieces) ===============================

def one_pole_forward(fns, cutoff, xs, s0):
    with torch.no_grad():
        return fns["one_pole"](cutoff, xs, s0)


def one_pole_grad(fns, cutoff, xs, targets, s0):
    cutoff, xs, s0 = (t.clone().requires_grad_() for t in (cutoff, xs, s0))
    loss = fns["one_pole_loss"](cutoff, xs, targets, s0)
    loss.backward()
    return loss.detach(), cutoff.grad, s0.grad, xs.grad


def eq_drive_stft_grad(fns, logf, g10, drive, xs, target, s0a, s0b):
    leaves = [t.clone().requires_grad_() for t in (logf, g10, drive, s0a, s0b, xs)]
    logf, g10, drive, s0a, s0b, xs = leaves
    loss = fns["eq_loss"](logf, g10, drive, xs, target, s0a, s0b)
    loss.backward()
    return (loss.detach(), *(t.grad for t in leaves))


def spectral_model_grad(fns, x, gain, w):
    gain, w = gain.clone().requires_grad_(), w.clone().requires_grad_()
    loss = fns["spectral_loss"](x, gain, w)
    loss.backward()
    return loss.detach(), gain.grad, w.grad


def pieces(scan_fn, eq_fn):
    return {
        "one_pole": scan_fn,
        "one_pole_loss": lambda c, x, t, s: torch.mean((scan_fn(c, x, s)[0] - t) ** 2),
        "eq_loss": lambda lf, g, d, x, t, sa, sb: multi_resolution_stft(eq_fn(lf, g, d, x, sa, sb), t),
        "spectral_loss": spectral_loss,
    }


def compiled_pieces():
    fns = pieces(one_pole_scan, eq_drive_scan)
    return {name: torch.compile(f) for name, f in fns.items()}


MODELS = {
    "one_pole_forward": ("one-pole low-pass, 48k samples", one_pole_forward),
    "one_pole_grad": ("one-pole MSE gradient, 48k samples", one_pole_grad),
    "eq_drive_stft_grad": ("EQ + drive, multi-resolution STFT loss gradient, 2048 samples", eq_drive_stft_grad),
    "spectral_model_grad": ("rfft -> gain -> irfft -> dense -> tanh gradient, 256 x 1024", spectral_model_grad),
}


def median_time(run, budget=2.0, repeats=5):
    run()
    times, start = [], time.perf_counter()
    while len(times) < repeats or (time.perf_counter() - start < budget and len(times) < 100):
        t = time.perf_counter()
        run()
        times.append(time.perf_counter() - t)
        if len(times) >= 3 and time.perf_counter() - start > 4 * budget:
            break
    return statistics.median(times)


def error(got, want):
    got = np.asarray(torch.as_tensor(got).detach(), dtype=np.float64).ravel()
    want = np.asarray(want, dtype=np.float64).ravel()
    return float(np.abs(got - want).max() / max(np.abs(want).max(), 1e-12))


def fmt(seconds):
    return f"{seconds:.2f} s" if seconds >= 1 else f"{seconds * 1e3:.2f} ms" if seconds >= 1e-3 else f"{seconds * 1e6:.1f} µs"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("dir")
    parser.add_argument("--out", type=Path, help="also write the Markdown table to this file")
    parser.add_argument("--no-compile", action="store_true", help="eager mode only")
    args = parser.parse_args()
    root = Path(args.dir)
    print(f"PyTorch {torch.__version__}, one thread, scan operator {'available' if hop_scan else 'missing'}", file=sys.stderr)
    modes = [("eager", pieces(one_pole_loop, eq_drive_loop))]
    if not args.no_compile and hop_scan is not None:
        modes.append(("torch.compile", compiled_pieces()))
    rows = []

    def emit(line):
        print(line, flush=True)
        rows.append(line)

    emit("| Model | Mode | Time | Error vs flux |")
    emit("|---|---|---:|---:|")
    for name, (title, model) in MODELS.items():
        n_in = sum(1 for _ in root.glob(f"{name}_in*.npy"))
        n_out = sum(1 for _ in root.glob(f"{name}_out*.npy"))
        inputs = [torch.from_numpy(np.load(root / f"{name}_in{k}.npy")) for k in range(n_in)]
        want = [np.load(root / f"{name}_out{k}.npy") for k in range(n_out)]
        for mode, fns in modes:
            try:
                start = time.perf_counter()
                out = model(fns, *inputs)
                first = time.perf_counter() - start
                err = max(error(g, w) for g, w in zip(out, want))
                run = median_time(lambda: model(fns, *inputs))
                note = f" (first call {fmt(first)})" if mode != "eager" else ""
                emit(f"| {title} | {mode}{note} | {fmt(run)} | {err:.1e} |")
            except Exception as e:  # noqa: BLE001
                emit(f"| {title} | {mode} | failed: {type(e).__name__} | — |")
                print(f"{name} {mode}: {str(e)[:400]}", file=sys.stderr)
    if args.out:
        args.out.write_text("\n".join(rows) + "\n", encoding="utf-8", newline="\n")


if __name__ == "__main__":
    main()
