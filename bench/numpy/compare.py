"""autodyne vs NumPy / SciPy on shared axes of functionality.

    python bench/numpy/compare.py [--out bench/numpy/RESULTS.md]

Needs the bindings installed (`maturin develop --release` in bindings/python).

Every case runs the same work on the same arrays in both libraries, checks the results agree,
and reports the best-of-5 time per call. autodyne's times include crossing into Rust and back
(DLPack import of the inputs, export of the result as a NumPy array), so they are end to end.
Single-threaded on both sides.
"""

import os

for var in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(var, "1")

import argparse
import pathlib
import platform
import timeit

import numpy as np
import scipy
from scipy import signal

import autodyne
from autodyne import fft as afft
from autodyne import linalg as al
from autodyne import signal as asg

parser = argparse.ArgumentParser(description="autodyne vs NumPy / SciPy")
parser.add_argument("--out", type=pathlib.Path, help="also write the Markdown table to this file")
args = parser.parse_args()

al.set_threads(1)  # single-threaded, like NumPy with OMP_NUM_THREADS=1

rng = np.random.default_rng(0)


def best(fn, min_time=0.2):
    """Best time per call, in seconds."""
    timer = timeit.Timer(fn)
    number, _ = timer.autorange()
    number = max(1, int(number * min_time / 0.2))
    return min(timer.repeat(repeat=5, number=number)) / number


def check(a, b, tol):
    a, b = np.asarray(a), np.asarray(b)
    assert a.shape == b.shape, (a.shape, b.shape)
    np.testing.assert_allclose(a, b, rtol=tol, atol=tol)


cases = []


def case(group, name, numpy_fn, autodyne_fn, tol=1e-6):
    check(autodyne_fn(), numpy_fn(), tol)
    t_np, t_ad = best(numpy_fn), best(autodyne_fn)
    cases.append((group, name, t_np, t_ad))
    print(f"{group:<22} {name:<46} numpy {t_np * 1e6:10.1f} us   autodyne {t_ad * 1e6:10.1f} us   {t_np / t_ad:5.2f}x")


# element-wise: a * x + b (NumPy makes a temporary per operation; autodyne fuses one pass)
for dtype in (np.float32, np.float64):
    for n in (1_000, 100_000, 10_000_000):
        x = rng.standard_normal(n).astype(dtype)
        case("fused a*x+b", f"{np.dtype(dtype).name}, n={n:,}", lambda: 2.0 * x + 0.5, lambda: autodyne.axpb(x, 2.0, 0.5), 1e-5)

# strided inputs: views are read in place on both sides
m = rng.standard_normal((2_000, 2_000))
case("strided a*x+b", "transposed 2000x2000 f64", lambda: 2.0 * m.T + 0.5, lambda: autodyne.axpb(m.T, 2.0, 0.5))
case("strided a*x+b", "every other column, 2000x1000", lambda: 2.0 * m[:, ::2] + 0.5, lambda: autodyne.axpb(m[:, ::2], 2.0, 0.5))
case("strided a*x+b", "reversed rows", lambda: 2.0 * m[::-1] + 0.5, lambda: autodyne.axpb(m[::-1], 2.0, 0.5))

# reductions
for dtype in (np.float32, np.float64):
    v = rng.standard_normal(10_000_000).astype(dtype)
    case("sum", f"{np.dtype(dtype).name}, n=10,000,000", lambda: np.sum(v), lambda: autodyne.sum(v), 1e-3 if dtype == np.float32 else 1e-9)
case("sum", "axis 0 of 2000x2000 f64", lambda: m.sum(axis=0), lambda: autodyne.sum(m, axis=0), 1e-9)
case("sum", "axis 1 of 2000x2000 f64", lambda: m.sum(axis=1), lambda: autodyne.sum(m, axis=1), 1e-9)
case("sum", "transposed 2000x2000 f64, axis 1", lambda: m.T.sum(axis=1), lambda: autodyne.sum(m.T, axis=1), 1e-9)

# runtime-typed arithmetic: mixed dtypes promote (int16 matrix + float32 row -> float32)
pcm = rng.integers(-32768, 32767, (2_000, 2_000), dtype=np.int16)
row = rng.standard_normal(2_000).astype(np.float32)
case("mixed dtypes", "int16 2000x2000 + float32 row", lambda: pcm + row, lambda: autodyne.add(pcm, row), 1e-6)
a64 = rng.standard_normal((2_000, 2_000))
case("mixed dtypes", "float64 + float64, same shape", lambda: a64 + m, lambda: autodyne.add(a64, m), 0)

# small arrays: per-call overhead (dispatch, crossing the language boundary)
tiny = rng.standard_normal(64)
case("small calls", "sum of 64 f64", lambda: np.sum(tiny), lambda: autodyne.sum(tiny), 1e-12)
case("small calls", "a*x+b on 64 f64", lambda: 2.0 * tiny + 0.5, lambda: autodyne.axpb(tiny, 2.0, 0.5))

# IIR filtering (2nd-order Butterworth low-pass), contiguous and strided time axis
sos = signal.butter(2, 1_000.0, fs=48_000.0, output="sos")
audio = rng.standard_normal((16, 480_000)).astype(np.float32)
case("IIR (scipy sosfilt)", "16 x 480,000 f32, along contiguous time", lambda: signal.sosfilt(sos, audio, axis=-1),
     lambda: autodyne.lowpass(audio, 1_000.0, 48_000.0, axis=-1), 1e-3)
interleaved = np.ascontiguousarray(audio.T)
case("IIR (scipy sosfilt)", "480,000 x 16 f32, along strided time", lambda: signal.sosfilt(sos, interleaved, axis=0),
     lambda: autodyne.lowpass(interleaved, 1_000.0, 48_000.0, axis=0), 1e-3)

# FFT
frames = rng.standard_normal((256, 4_096))
case("FFT (numpy.fft)", "rfft, 256 x 4096 f64", lambda: np.fft.rfft(frames), lambda: autodyne.rfft(frames), 1e-8)
case("FFT (numpy.fft)", "rfft, 4096 x 64 f64", lambda: np.fft.rfft(frames.reshape(-1, 64)), lambda: autodyne.rfft(frames.reshape(-1, 64)), 1e-8)

# FFT of awkward lengths (Bluestein / mixed radix)
for n in (1000, 1031, 4095):
    z = rng.standard_normal((64, n)) + 1j * rng.standard_normal((64, n))
    case("FFT (numpy.fft)", f"complex fft, 64 x {n}", lambda: np.fft.fft(z), lambda: afft.fft(z), 1e-8)

# linear algebra (single-threaded on both sides)
for n in (64, 512):
    a, b = rng.standard_normal((n, n)), rng.standard_normal((n, n))
    case("linalg (numpy)", f"matmul {n}x{n}", lambda: a @ b, lambda: al.matmul(a, b), 1e-9)
    case("linalg (numpy)", f"solve {n}x{n}", lambda: np.linalg.solve(a, b), lambda: al.solve(a, b), 1e-6)
a = rng.standard_normal((300, 300))
case("linalg (numpy)", "eigvals 300x300", lambda: np.sort_complex(np.linalg.eigvals(a)), lambda: np.sort_complex(al.eigvals(a)), 1e-6)
case("linalg (numpy)", "svd 300x300", lambda: np.linalg.svd(a, compute_uv=False), lambda: al.svd(a, compute_uv=False), 1e-8)
sym = a + a.T
case("linalg (numpy)", "eigh 300x300", lambda: np.linalg.eigh(sym)[0], lambda: al.eigh(sym)[0], 1e-8)

# signal processing vs scipy.signal
sig = rng.standard_normal((16, 48_000))
sos8 = signal.ellip(8, 0.5, 60, [500, 4000], btype="band", fs=48_000, output="sos")
case("filtering (scipy.signal)", "sosfilt, 8th-order ellip, 16 x 48000", lambda: signal.sosfilt(sos8, sig), lambda: asg.sosfilt(sos8, sig), 1e-8)
case("filtering (scipy.signal)", "sosfiltfilt, 16 x 48000", lambda: signal.sosfiltfilt(sos8, sig), lambda: asg.sosfiltfilt(sos8, sig), 1e-7)
b4, a4 = signal.butter(4, 0.1)
case("filtering (scipy.signal)", "lfilter, 4th-order, 16 x 48000", lambda: signal.lfilter(b4, a4, sig), lambda: asg.lfilter(b4, a4, sig), 1e-8)
case("filter design (scipy.signal)", "ellip(8) to sos", lambda: signal.ellip(8, 0.5, 60, [500, 4000], btype="band", fs=48_000, output="sos"),
     lambda: asg.ellip(8, 0.5, 60, [500, 4000], btype="band", fs=48_000, output="sos"), 1e-8)
case("filter design (scipy.signal)", "remez, 101 taps", lambda: signal.remez(101, [0, 0.2, 0.25, 0.5], [1, 0]),
     lambda: asg.remez(101, [0, 0.2, 0.25, 0.5], [1, 0]), 1e-8)
case("spectral (scipy.signal)", "welch, 16 x 48000, nperseg 1024", lambda: signal.welch(sig, nperseg=1024)[1], lambda: asg.welch(sig, nperseg=1024)[1], 1e-8)
case("spectral (scipy.signal)", "stft, 16 x 48000, nperseg 512", lambda: signal.stft(sig, nperseg=512)[2], lambda: asg.stft(sig, nperseg=512)[2], 1e-8)
case("spectral (scipy.signal)", "hilbert, 16 x 48000", lambda: signal.hilbert(sig), lambda: asg.hilbert(sig), 1e-8)

# convolution: each library's automatic choice, then the explicit methods
long_x, taps32, taps1k = rng.standard_normal(1_000_000), rng.standard_normal(32), rng.standard_normal(1_000)
case("convolution (scipy.signal)", "convolve 48000 x 32 taps (auto)", lambda: signal.convolve(sig[0], taps32), lambda: asg.convolve(sig[0], taps32), 1e-9)
case("convolution (scipy.signal)", "convolve 1M x 1000 taps (auto)", lambda: signal.convolve(long_x, taps1k), lambda: asg.convolve(long_x, taps1k), 1e-8)
case("convolution (scipy.signal)", "oaconvolve 1M x 1000 taps", lambda: signal.oaconvolve(long_x, taps1k), lambda: asg.oaconvolve(long_x, taps1k), 1e-8)
case("convolution (scipy.signal)", "fftconvolve 100k x 50k", lambda: signal.fftconvolve(long_x[:100_000], long_x[:50_000]),
     lambda: asg.fftconvolve(long_x[:100_000], long_x[:50_000]), 1e-8)
case("convolution (scipy.signal)", "correlate 48000 x 4800 (auto)", lambda: signal.correlate(sig[0], sig[1, :4_800]),
     lambda: asg.correlate(sig[0], sig[1, :4_800]), 1e-8)

# smoothing, resampling, peaks
case("smoothing (scipy.signal)", "savgol_filter 16 x 48000, window 31, order 3", lambda: signal.savgol_filter(sig, 31, 3),
     lambda: asg.savgol_filter(sig, 31, 3), 1e-8)
case("resampling (scipy.signal)", "resample_poly 48k -> 44.1k, 16 x 48000", lambda: signal.resample_poly(sig, 147, 160, axis=-1),
     lambda: asg.resample_poly(sig, 147, 160, axis=-1), 1e-8)
case("resampling (scipy.signal)", "upfirdn 4/3, 64 taps, 16 x 48000", lambda: signal.upfirdn(taps1k[:64], sig, 4, 3),
     lambda: asg.upfirdn(taps1k[:64], sig, 4, 3), 1e-8)
wave = np.sin(np.arange(1_000_000) * 0.01) + 0.1 * rng.standard_normal(1_000_000)
case("peaks (scipy.signal)", "find_peaks 1M samples", lambda: signal.find_peaks(wave)[0], lambda: asg.find_peaks(wave)[0], 0)
case("peaks (scipy.signal)", "find_peaks 1M, prominence + width", lambda: signal.find_peaks(wave, prominence=1.0, width=10)[0],
     lambda: asg.find_peaks(wave, prominence=1.0, width=10)[0], 0)

lines = [
    "# autodyne vs NumPy / SciPy",
    "",
    f"Generated by `bench/numpy/compare.py`: best of 5, single-threaded, {platform.processor() or platform.machine()},",
    f"Python {platform.python_version()}, NumPy {np.__version__}, SciPy {scipy.__version__}. autodyne times are end to end",
    "(inputs cross into Rust through DLPack, results come back as NumPy arrays). Speedup > 1 means autodyne is faster.",
    "",
    "| Axis | Case | NumPy / SciPy | autodyne | Speedup |",
    "|---|---|---:|---:|---:|",
]
for group, name, t_np, t_ad in cases:
    lines.append(f"| {group} | {name} | {t_np * 1e6:,.1f} µs | {t_ad * 1e6:,.1f} µs | {t_np / t_ad:.2f}x |")
text = "\n".join(lines) + "\n"
print("\n" + text)
if args.out:
    args.out.write_text(text, encoding="utf-8")
