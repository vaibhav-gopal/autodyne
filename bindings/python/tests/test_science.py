"""autodyne.linalg / .signal / .fft against NumPy and SciPy."""

import numpy as np
import pytest
import scipy.linalg
from scipy import signal as ss

import autodyne
from autodyne import fft as afft
from autodyne import linalg as al
from autodyne import signal as asg

rng = np.random.default_rng(7)


def close(a, b, tol=1e-9):
    np.testing.assert_allclose(np.asarray(a), np.asarray(b), rtol=tol, atol=tol)


# ---- linear algebra ----------------------------------------------------------------------------

@pytest.mark.parametrize("dtype", [np.float32, np.float64])
def test_matmul_and_solve(dtype):
    tol = 1e-4 if dtype == np.float32 else 1e-10
    a = rng.standard_normal((6, 6)).astype(dtype)
    b = rng.standard_normal((6, 3)).astype(dtype)
    out = al.matmul(a, b)
    assert out.dtype == dtype
    close(out, a @ b, tol)
    close(al.matmul(a.T, b), a.T @ b, tol)  # transposed input read in place
    close(al.matmul(a[::-1], b), a[::-1] @ b, tol)
    close(al.solve(a, b), np.linalg.solve(a, b), tol * 100)
    close(al.inv(a), np.linalg.inv(a), tol * 100)
    close(al.det(a), np.linalg.det(a), tol * 100)


def test_decompositions():
    a = rng.standard_normal((5, 3))
    x, rank, s = al.lstsq(a, rng.standard_normal(5))
    assert rank == 3
    close(s, np.linalg.svd(a, compute_uv=False))
    sq = rng.standard_normal((4, 4))
    w = al.eigvals(sq)
    close(np.sort_complex(w), np.sort_complex(np.linalg.eigvals(sq)))
    w, v = al.eig(sq)
    close(sq @ v, v * w)
    sym = sq + sq.T
    w, v = al.eigh(sym)
    close(w, np.linalg.eigh(sym)[0])
    u, s, vt = al.svd(a, full_matrices=False)
    close((u * s) @ vt, a)
    close(al.pinv(a), np.linalg.pinv(a))
    q, r = al.qr(a)
    close(q @ r, a)
    spd = a.T @ a
    close(al.cholesky(spd), np.linalg.cholesky(spd))
    close(al.expm(sq), scipy.linalg.expm(sq))
    close(np.sort_complex(al.roots([1, -6, 11, -6])), [1, 2, 3])


# ---- filter design -----------------------------------------------------------------------------

@pytest.mark.parametrize("make", [
    lambda m: m.butter(5, [300, 3400], btype="bandpass", fs=16000, output="sos"),
    lambda m: m.cheby1(4, 1, 1000, btype="highpass", fs=8000, output="sos"),
    lambda m: m.cheby2(6, 40, 0.3, output="sos"),
    lambda m: m.ellip(6, 0.5, 60, [0.1, 0.3], btype="bandstop", output="sos"),
    lambda m: m.bessel(4, 1000, fs=8000, output="sos", norm="mag"),
    lambda m: np.concatenate(m.butter(3, 0.25, output="ba")),
])
def test_iir_designs(make):
    close(make(asg), make(ss), 1e-8)


def test_iir_zpk_and_analog():
    z, p, k = asg.ellip(5, 1, 40, 2 * np.pi * 100, analog=True, output="zpk")
    zr, pr, kr = ss.ellip(5, 1, 40, 2 * np.pi * 100, analog=True, output="zpk")
    close(np.sort_complex(z), np.sort_complex(zr))
    close(np.sort_complex(p), np.sort_complex(pr))
    close(k, kr)


def test_fir_designs_and_windows():
    close(asg.firwin(65, [1000, 2000], window=("kaiser", 8.6), pass_zero=False, fs=8000),
          ss.firwin(65, [1000, 2000], window=("kaiser", 8.6), pass_zero=False, fs=8000), 1e-12)
    # floats: given integers, SciPy keeps an integer array and its nudge of the repeated frequency truncates away
    freq, gain = [0.0, 1000.0, 1000.0, 4000.0], [1.0, 1.0, 0.0, 0.0]
    close(asg.firwin2(31, freq, gain, fs=8000), ss.firwin2(31, freq, gain, fs=8000), 1e-12)
    close(asg.firls(31, [0, 1000, 1500, 4000], [1, 1, 0, 0], fs=8000), ss.firls(31, [0, 1000, 1500, 4000], [1, 1, 0, 0], fs=8000), 1e-9)
    close(asg.remez(41, [0, 0.2, 0.3, 0.5], [1, 0]), ss.remez(41, [0, 0.2, 0.3, 0.5], [1, 0]), 1e-9)
    assert asg.kaiserord(65, 0.05) == ss.kaiserord(65, 0.05)
    for w in ["hann", "blackmanharris", ("tukey", 0.3), ("chebwin", 80)]:
        close(asg.get_window(w, 33), ss.get_window(w, 33), 1e-12)
        close(asg.get_window(w, 32, fftbins=False), ss.get_window(w, 32, fftbins=False), 1e-12)


def test_responses():
    b, a = ss.butter(4, 0.3)
    sos = ss.butter(6, [0.1, 0.4], btype="band", output="sos")
    w, h = asg.freqz(b, a, 64)
    close(h, ss.freqz(b, a, 64)[1])
    close(asg.sosfreqz(sos, 64)[1], ss.sosfreqz(sos, 64)[1])
    close(asg.group_delay((b, a), 16)[1][1:], ss.group_delay((b, a), 16)[1][1:], 1e-8)
    z, p, k = asg.tf2zpk(b, a)
    close(np.sort_complex(p), np.sort_complex(ss.tf2zpk(b, a)[1]))
    close(asg.zpk2sos(z, p, k), ss.zpk2sos(z, p, k), 1e-9)


# ---- filtering ---------------------------------------------------------------------------------

@pytest.mark.parametrize("dtype", [np.float32, np.float64])
def test_filtering_along_axes(dtype):
    tol = 1e-4 if dtype == np.float32 else 1e-10
    x = rng.standard_normal((4, 300)).astype(dtype)
    b, a = ss.cheby1(4, 1, 0.2)
    sos = ss.ellip(6, 0.5, 50, [0.1, 0.4], btype="band", output="sos")
    close(asg.lfilter(b, a, x), ss.lfilter(b, a, x), tol)
    close(asg.lfilter(b, a, x.T, axis=0), ss.lfilter(b, a, x.T, axis=0), tol)
    zi = ss.lfilter_zi(b, a)
    close(asg.lfilter_zi(b, a), zi)
    y, zf = asg.lfilter(b, a, x, zi=np.outer(x[:, 0], zi))
    yr, zfr = ss.lfilter(b, a, x, zi=np.outer(x[:, 0], zi))
    close(y, yr, tol)
    close(zf, zfr, tol)
    close(asg.sosfilt(sos, x), ss.sosfilt(sos, x), tol)
    close(asg.sosfilt_zi(sos), ss.sosfilt_zi(sos))
    close(asg.filtfilt(b, a, x), ss.filtfilt(b, a, x), tol * 10)
    close(asg.filtfilt(b, a, x, padtype="even", padlen=30), ss.filtfilt(b, a, x, padtype="even", padlen=30), tol * 10)
    close(asg.sosfiltfilt(sos, x[:, ::2]), ss.sosfiltfilt(sos, x[:, ::2]), tol * 10)  # strided input


# ---- spectral estimation -----------------------------------------------------------------------

def test_spectral_estimates():
    x = rng.standard_normal((2, 2000))
    y = np.roll(x, 5, axis=1) + 0.2 * rng.standard_normal((2, 2000))
    for kw in [{}, dict(nperseg=100, noverlap=40, nfft=256, detrend="linear", scaling="spectrum", average="median"),
               dict(window=("kaiser", 6.0), return_onesided=False)]:
        f, p = asg.welch(x, fs=1000, **kw)
        fr, pr = ss.welch(x, fs=1000, **kw)
        close(f, fr)
        close(p, pr)
    close(asg.periodogram(x, fs=1000, window="hann", nfft=1500)[1], ss.periodogram(x, fs=1000, window="hann", nfft=1500)[1])
    close(asg.csd(x, y, fs=1000, nperseg=128)[1], ss.csd(x, y, fs=1000, nperseg=128)[1])
    close(asg.coherence(x, y, fs=1000, nperseg=128)[1], ss.coherence(x, y, fs=1000, nperseg=128)[1])
    for mode in ["psd", "magnitude", "angle", "phase", "complex"]:
        f, t, s = asg.spectrogram(x, fs=1000, nperseg=64, mode=mode)
        fr, tr, sr = ss.spectrogram(x, fs=1000, nperseg=64, mode=mode)
        close(t, tr)
        close(s, sr, 1e-8)
    f, t, z = asg.stft(x, fs=1000, nperseg=64)
    fr, tr, zr = ss.stft(x, fs=1000, nperseg=64)
    close(t, tr)
    close(z, zr)
    t, back = asg.istft(z, fs=1000, nperseg=64)
    close(back[:, :2000], x)
    close(back, ss.istft(zr, fs=1000, nperseg=64)[1])


# ---- FFT ---------------------------------------------------------------------------------------

@pytest.mark.parametrize("n", [1, 7, 100, 1000, 4096])
def test_ffts_of_any_length(n):
    x = rng.standard_normal((3, n))
    z = x + 1j * rng.standard_normal((3, n))
    close(afft.fft(z), np.fft.fft(z), 1e-9 * n)
    close(afft.ifft(z), np.fft.ifft(z), 1e-9)
    close(afft.rfft(x), np.fft.rfft(x), 1e-9 * n)
    close(afft.irfft(np.fft.rfft(x), n), x, 1e-9)
    close(autodyne.rfft(x), np.fft.rfft(x), 1e-9 * n)
