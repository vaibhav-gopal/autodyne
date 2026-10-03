"""flux's benchmark models in PyTorch (autograd, eager; `torch.compile` when it works here), on the
inputs `examples/flux_programs.rs` writes, checked against flux's interpreter.

    python compare_torch.py DIR

Prints one Markdown row per model and mode: median run time and the largest difference from flux's
outputs relative to the largest output.
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


def one_pole(cutoff, xs, s0):
    a = 1.0 - torch.exp(-2.0 * torch.pi * cutoff / FS)
    s, ys = s0, []
    for x in xs:
        s = s + a * (x - s)
        ys.append(s)
    return torch.stack(ys), s


def one_pole_forward(cutoff, xs, s0):
    with torch.no_grad():
        return one_pole(cutoff, xs, s0)


def one_pole_grad(cutoff, xs, targets, s0):
    cutoff, xs, s0 = (t.clone().requires_grad_() for t in (cutoff, xs, s0))
    ys, _ = one_pole(cutoff, xs, s0)
    loss = torch.mean((ys - targets) ** 2)
    loss.backward()
    return loss.detach(), cutoff.grad, s0.grad, xs.grad


def peaking(f, gain_db, q=1.0):
    w = 2.0 * torch.pi * f / FS
    cos_w, alpha = torch.cos(w), torch.sin(w) / (2.0 * q)
    a = torch.sqrt(10.0 ** (gain_db / 20.0))
    a0 = 1.0 + alpha / a
    return (1.0 + alpha * a) / a0, -2.0 * cos_w / a0, (1.0 - alpha * a) / a0, -2.0 * cos_w / a0, (1.0 - alpha / a) / a0


def eq_drive(logf, g10, drive, xs, s0a, s0b):
    b0, b1, b2, a1, a2 = peaking(torch.exp(logf), g10 * 10.0)
    sa, sb, ys = s0a, s0b, []
    for x in xs:
        y = b0 * x + sa
        sa, sb = b1 * x - a1 * y + sb, b2 * x - a2 * y
        ys.append(torch.tanh(y * drive))
    return torch.stack(ys)


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


def eq_drive_stft_grad(logf, g10, drive, xs, target, s0a, s0b):
    leaves = [t.clone().requires_grad_() for t in (logf, g10, drive, s0a, s0b, xs)]
    logf, g10, drive, s0a, s0b, xs = leaves
    loss = multi_resolution_stft(eq_drive(logf, g10, drive, xs, s0a, s0b), target)
    loss.backward()
    return (loss.detach(), *(t.grad for t in leaves))


def spectral_model_grad(x, gain, w):
    gain, w = gain.clone().requires_grad_(), w.clone().requires_grad_()
    n = x.shape[-1]
    y = torch.tanh(torch.fft.irfft(torch.fft.rfft(x, dim=-1) * gain, n=n, dim=-1) @ w)
    loss = torch.mean(y * y)
    loss.backward()
    return loss.detach(), gain.grad, w.grad


MODELS = {
    "one_pole_forward": ("one-pole low-pass, 48k samples", one_pole_forward),
    "one_pole_grad": ("one-pole MSE gradient, 48k samples", one_pole_grad),
    "eq_drive_stft_grad": ("EQ + drive, multi-resolution STFT loss gradient, 2048 samples", eq_drive_stft_grad),
    "spectral_model_grad": ("rfft -> gain -> irfft -> dense -> tanh gradient, 256 x 1024", spectral_model_grad),
}


COMPILED = {"spectral_model_grad"}


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
    args = parser.parse_args()
    root = Path(args.dir)
    print(f"PyTorch {torch.__version__}, one thread", file=sys.stderr)
    print("| Model | Mode | Time | Error vs flux |")
    print("|---|---|---:|---:|")
    for name, (title, model) in MODELS.items():
        n_in = sum(1 for _ in root.glob(f"{name}_in*.npy"))
        n_out = sum(1 for _ in root.glob(f"{name}_out*.npy"))
        inputs = [torch.from_numpy(np.load(root / f"{name}_in{k}.npy")) for k in range(n_in)]
        want = [np.load(root / f"{name}_out{k}.npy") for k in range(n_out)]
        modes = [("eager", model)]
        # dynamo unrolls Python loops: the scans (48k and 2048 steps) would trace one op per sample
        if name in COMPILED:
          try:
            modes.append(("torch.compile", torch.compile(model)))
          except Exception as e:  # noqa: BLE001
            print(f"{name}: torch.compile unavailable: {e}", file=sys.stderr)
        for mode, fn in modes:
            try:
                out = fn(*inputs)
                err = max(error(g, w) for g, w in zip(out, want))
                print(f"| {title} | {mode} | {fmt(median_time(lambda: fn(*inputs)))} | {err:.1e} |", flush=True)
            except Exception as e:  # noqa: BLE001
                print(f"| {title} | {mode} | failed: {type(e).__name__} | — |", flush=True)
                print(f"{name} {mode}: {e}", file=sys.stderr)


if __name__ == "__main__":
    main()
