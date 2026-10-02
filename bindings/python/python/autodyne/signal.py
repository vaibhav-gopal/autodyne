"""Signal processing with ``scipy.signal``'s names, arguments and defaults: IIR and FIR design,
filtering along any axis, frequency responses, windows and spectral estimation."""

import numpy as _np

from . import _autodyne as _native

__all__ = ["iirfilter", "butter", "cheby1", "cheby2", "ellip", "bessel", "firwin", "firwin2", "firls", "remez",
           "kaiserord", "get_window", "zpk2sos", "tf2zpk", "freqz", "sosfreqz", "group_delay", "lfilter", "lfilter_zi",
           "sosfilt", "sosfilt_zi", "filtfilt", "sosfiltfilt", "welch", "periodogram", "csd", "coherence", "spectrogram",
           "stft", "istft"]


# ---- IIR design ------------------------------------------------------------------------------

def iirfilter(N, Wn, rp=None, rs=None, btype="band", analog=False, ftype="butter", output="ba", fs=None):
    btype = {"band": "bandpass", "pass": "bandpass", "stop": "bandstop", "low": "lowpass", "high": "highpass"}.get(btype, btype)
    return _native.iirfilter(N, _np.atleast_1d(_np.asarray(Wn, dtype=float)), rp, rs, btype, analog, ftype, output, fs)


def butter(N, Wn, btype="low", analog=False, output="ba", fs=None):
    return iirfilter(N, Wn, btype=btype, analog=analog, output=output, ftype="butter", fs=fs)


def cheby1(N, rp, Wn, btype="low", analog=False, output="ba", fs=None):
    return iirfilter(N, Wn, rp=rp, btype=btype, analog=analog, output=output, ftype="cheby1", fs=fs)


def cheby2(N, rs, Wn, btype="low", analog=False, output="ba", fs=None):
    return iirfilter(N, Wn, rs=rs, btype=btype, analog=analog, output=output, ftype="cheby2", fs=fs)


def ellip(N, rp, rs, Wn, btype="low", analog=False, output="ba", fs=None):
    return iirfilter(N, Wn, rp=rp, rs=rs, btype=btype, analog=analog, output=output, ftype="ellip", fs=fs)


def bessel(N, Wn, btype="low", analog=False, output="ba", norm="phase", fs=None):
    btype = {"low": "lowpass", "high": "highpass", "band": "bandpass", "stop": "bandstop"}.get(btype, btype)
    return _native.iirfilter(N, _np.atleast_1d(_np.asarray(Wn, dtype=float)), None, None, btype, analog, "bessel", output, fs, norm)


# ---- FIR design ------------------------------------------------------------------------------

def _pass_zero(pass_zero):
    return {"lowpass": True, "bandstop": True, "highpass": False, "bandpass": False}.get(pass_zero, pass_zero)


def firwin(numtaps, cutoff, *, window="hamming", pass_zero=True, scale=True, fs=2.0):
    return _native.firwin(numtaps, _np.atleast_1d(_np.asarray(cutoff, dtype=float)), window, bool(_pass_zero(pass_zero)), scale, fs)


def firwin2(numtaps, freq, gain, *, nfreqs=None, window="hamming", antisymmetric=False, fs=2.0):
    return _native.firwin2(numtaps, _np.asarray(freq, dtype=float), _np.asarray(gain, dtype=float), nfreqs, window, antisymmetric, fs)


def firls(numtaps, bands, desired, *, weight=None, fs=2.0):
    w = None if weight is None else _np.asarray(weight, dtype=float)
    return _native.firls(numtaps, _np.asarray(bands, dtype=float).ravel(), _np.asarray(desired, dtype=float).ravel(), w, fs)


def remez(numtaps, bands, desired, *, weight=None, type="bandpass", maxiter=25, grid_density=16, fs=1.0):
    w = None if weight is None else _np.asarray(weight, dtype=float)
    return _native.remez(numtaps, _np.asarray(bands, dtype=float).ravel(), _np.asarray(desired, dtype=float), w, type,
                         maxiter, grid_density, fs)


def kaiserord(ripple, width):
    return _native.kaiserord(ripple, width)


def get_window(window, Nx, fftbins=True):
    return _native.get_window(window, Nx, fftbins)


# ---- conversions and responses -----------------------------------------------------------------

def zpk2sos(z, p, k, pairing=None, *, analog=False):
    pairing = pairing or ("minimal" if analog else "nearest")
    return _native.zpk2sos(_np.asarray(z, dtype=complex), _np.asarray(p, dtype=complex), float(k), pairing, analog)


def tf2zpk(b, a):
    return _native.tf2zpk(_np.asarray(b, dtype=float), _np.asarray(a, dtype=float))


def _grid(worN, whole, fs):
    if _np.ndim(worN) == 0:
        n = int(worN)
        return _np.linspace(0, fs if whole else fs / 2, n, endpoint=False)
    return _np.asarray(worN, dtype=float)


def freqz(b, a=1, worN=512, whole=False, fs=2 * _np.pi):
    """``(w, h)``; ``w`` in the units of ``fs`` (radians/sample by default)."""
    w = _grid(worN, whole, fs)
    return w, _native.freqz(_np.atleast_1d(_np.asarray(b, dtype=float)), _np.atleast_1d(_np.asarray(a, dtype=float)), w, fs)


def sosfreqz(sos, worN=512, whole=False, fs=2 * _np.pi):
    w = _grid(worN, whole, fs)
    return w, _native.sosfreqz(_np.asarray(sos, dtype=float), w, fs)


def group_delay(system, w=512, whole=False, fs=2 * _np.pi):
    b, a = system
    freqs = _grid(w, whole, fs)
    return freqs, _native.group_delay(_np.atleast_1d(_np.asarray(b, dtype=float)), _np.atleast_1d(_np.asarray(a, dtype=float)), freqs, fs)


# ---- filtering -------------------------------------------------------------------------------

def _data(x):
    x = _np.asarray(x)
    return x if x.dtype in (_np.float32, _np.float64) else x.astype(_np.float64)


def lfilter(b, a, x, axis=-1, zi=None):
    b, a = _np.atleast_1d(_np.asarray(b, dtype=float)), _np.atleast_1d(_np.asarray(a, dtype=float))
    return _native.lfilter(b, a, _data(x), axis, None if zi is None else _data(zi))


def lfilter_zi(b, a):
    return _native.lfilter_zi(_np.atleast_1d(_np.asarray(b, dtype=float)), _np.atleast_1d(_np.asarray(a, dtype=float)))


def sosfilt(sos, x, axis=-1, zi=None):
    return _native.sosfilt(_np.asarray(sos, dtype=float), _data(x), axis, None if zi is None else _data(zi))


def sosfilt_zi(sos):
    return _native.sosfilt_zi(_np.asarray(sos, dtype=float))


def filtfilt(b, a, x, axis=-1, padtype="odd", padlen=None, method="pad"):
    if method != "pad":
        raise NotImplementedError("only method='pad'")
    b, a = _np.atleast_1d(_np.asarray(b, dtype=float)), _np.atleast_1d(_np.asarray(a, dtype=float))
    return _native.filtfilt(b, a, _data(x), axis, padtype, padlen)


def sosfiltfilt(sos, x, axis=-1, padtype="odd", padlen=None):
    return _native.sosfiltfilt(_np.asarray(sos, dtype=float), _data(x), axis, padtype, padlen)


# ---- spectral estimation -----------------------------------------------------------------------

def _detrend(d):
    return "none" if d is False or d is None else d


def welch(x, fs=1.0, window="hann", nperseg=None, noverlap=None, nfft=None, detrend="constant",
          return_onesided=True, scaling="density", axis=-1, average="mean"):
    return _native.welch(_data(x), fs, window, nperseg, noverlap, nfft, _detrend(detrend), return_onesided, scaling, axis, average)


def periodogram(x, fs=1.0, window="boxcar", nfft=None, detrend="constant", return_onesided=True, scaling="density", axis=-1):
    x = _data(x)
    n = x.shape[axis]
    if nfft is not None and nfft < n:
        x = _np.take(x, _np.arange(nfft), axis=axis)
        n = nfft
    return _native.welch(x, fs, window or "boxcar", n, 0, nfft if nfft is None or nfft > n else None, _detrend(detrend),
                         return_onesided, scaling, axis, "mean")


def csd(x, y, fs=1.0, window="hann", nperseg=None, noverlap=None, nfft=None, detrend="constant",
        return_onesided=True, scaling="density", axis=-1, average="mean"):
    return _native.csd(_data(x), _data(y), fs, window, nperseg, noverlap, nfft, _detrend(detrend), return_onesided, scaling, axis, average)


def coherence(x, y, fs=1.0, window="hann", nperseg=None, noverlap=None, nfft=None, detrend="constant", axis=-1):
    return _native.coherence(_data(x), _data(y), fs, window, nperseg, noverlap, nfft, _detrend(detrend), axis)


def spectrogram(x, fs=1.0, window=("tukey", 0.25), nperseg=None, noverlap=None, nfft=None, detrend="constant",
                return_onesided=True, scaling="density", axis=-1, mode="psd"):
    if mode == "complex":
        nper = nperseg or 256
        return stft(x, fs, window, nper, noverlap if noverlap is not None else nper // 8, nfft, detrend, return_onesided,
                    None, False, axis, "psd" if scaling == "density" else "spectrum")
    return _native.spectrogram(_data(x), fs, window, nperseg, noverlap, nfft, _detrend(detrend), return_onesided, scaling, axis, mode)


def stft(x, fs=1.0, window="hann", nperseg=256, noverlap=None, nfft=None, detrend=False, return_onesided=True,
         boundary="zeros", padded=True, axis=-1, scaling="spectrum"):
    scaling = "density" if scaling == "psd" else scaling
    return _native.stft(_data(x), fs, window, nperseg, noverlap, nfft, _detrend(detrend), return_onesided, boundary, padded, axis, scaling)


def istft(Zxx, fs=1.0, window="hann", nperseg=None, noverlap=None, nfft=None, input_onesided=True, boundary=True,
          time_axis=-1, freq_axis=-2, scaling="spectrum"):
    Z = _np.asarray(Zxx)
    if (time_axis % Z.ndim, freq_axis % Z.ndim) != (Z.ndim - 1, Z.ndim - 2):
        Z = _np.moveaxis(Z, (freq_axis, time_axis), (-2, -1))
    return _native.istft(_np.ascontiguousarray(Z, dtype=_np.complex128), fs, window, nperseg, noverlap, nfft, input_onesided, boundary, scaling)
