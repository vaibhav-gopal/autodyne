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
    """An IIR filter of order ``N`` designed like ``scipy.signal.iirfilter``: ``ftype`` butter, cheby1,
    cheby2, ellip or bessel; ``output`` ba, zpk or sos; digital (``fs``, edges in its units) or analog."""
    btype = {"band": "bandpass", "pass": "bandpass", "stop": "bandstop", "low": "lowpass", "high": "highpass"}.get(btype, btype)
    return _native.iirfilter(N, _np.atleast_1d(_np.asarray(Wn, dtype=float)), rp, rs, btype, analog, ftype, output, fs)


def butter(N, Wn, btype="low", analog=False, output="ba", fs=None):
    """A Butterworth filter, like ``scipy.signal.butter``."""
    return iirfilter(N, Wn, btype=btype, analog=analog, output=output, ftype="butter", fs=fs)


def cheby1(N, rp, Wn, btype="low", analog=False, output="ba", fs=None):
    """A Chebyshev type I filter (passband ripple ``rp`` dB), like ``scipy.signal.cheby1``."""
    return iirfilter(N, Wn, rp=rp, btype=btype, analog=analog, output=output, ftype="cheby1", fs=fs)


def cheby2(N, rs, Wn, btype="low", analog=False, output="ba", fs=None):
    """A Chebyshev type II filter (stopband ``rs`` dB down), like ``scipy.signal.cheby2``."""
    return iirfilter(N, Wn, rs=rs, btype=btype, analog=analog, output=output, ftype="cheby2", fs=fs)


def ellip(N, rp, rs, Wn, btype="low", analog=False, output="ba", fs=None):
    """An elliptic filter (``rp`` dB ripple, ``rs`` dB stopband), like ``scipy.signal.ellip``."""
    return iirfilter(N, Wn, rp=rp, rs=rs, btype=btype, analog=analog, output=output, ftype="ellip", fs=fs)


def bessel(N, Wn, btype="low", analog=False, output="ba", norm="phase", fs=None):
    """A Bessel filter (``norm`` phase, delay or mag), like ``scipy.signal.bessel``."""
    btype = {"low": "lowpass", "high": "highpass", "band": "bandpass", "stop": "bandstop"}.get(btype, btype)
    return _native.iirfilter(N, _np.atleast_1d(_np.asarray(Wn, dtype=float)), None, None, btype, analog, "bessel", output, fs, norm)


# ---- FIR design ------------------------------------------------------------------------------

def _pass_zero(pass_zero):
    return {"lowpass": True, "bandstop": True, "highpass": False, "bandpass": False}.get(pass_zero, pass_zero)


def firwin(numtaps, cutoff, *, window="hamming", pass_zero=True, scale=True, fs=2.0):
    """Windowed-sinc FIR taps, like ``scipy.signal.firwin``."""
    return _native.firwin(numtaps, _np.atleast_1d(_np.asarray(cutoff, dtype=float)), window, bool(_pass_zero(pass_zero)), scale, fs)


def firwin2(numtaps, freq, gain, *, nfreqs=None, window="hamming", antisymmetric=False, fs=2.0):
    """FIR taps for an arbitrary gain curve (frequency sampling), like ``scipy.signal.firwin2``."""
    return _native.firwin2(numtaps, _np.asarray(freq, dtype=float), _np.asarray(gain, dtype=float), nfreqs, window, antisymmetric, fs)


def firls(numtaps, bands, desired, *, weight=None, fs=2.0):
    """Least-squares FIR taps, like ``scipy.signal.firls``."""
    w = None if weight is None else _np.asarray(weight, dtype=float)
    return _native.firls(numtaps, _np.asarray(bands, dtype=float).ravel(), _np.asarray(desired, dtype=float).ravel(), w, fs)


def remez(numtaps, bands, desired, *, weight=None, type="bandpass", maxiter=25, grid_density=16, fs=1.0):
    """Equiripple FIR taps (Parks-McClellan), like ``scipy.signal.remez``."""
    w = None if weight is None else _np.asarray(weight, dtype=float)
    return _native.remez(numtaps, _np.asarray(bands, dtype=float).ravel(), _np.asarray(desired, dtype=float), w, type,
                         maxiter, grid_density, fs)


def kaiserord(ripple, width):
    """``(numtaps, beta)`` of a Kaiser-window FIR for ``ripple`` dB and a transition ``width``
    (as a fraction of Nyquist), like ``scipy.signal.kaiserord``."""
    return _native.kaiserord(ripple, width)


def get_window(window, Nx, fftbins=True):
    """``Nx`` samples of ``window`` (a name or a ``(name, parameter)`` tuple), periodic when ``fftbins``,
    like ``scipy.signal.get_window``."""
    return _native.get_window(window, Nx, fftbins)


# ---- conversions and responses -----------------------------------------------------------------

def zpk2sos(z, p, k, pairing=None, *, analog=False):
    """Second-order sections from zeros, poles and gain, like ``scipy.signal.zpk2sos``."""
    pairing = pairing or ("minimal" if analog else "nearest")
    return _native.zpk2sos(_np.asarray(z, dtype=complex), _np.asarray(p, dtype=complex), float(k), pairing, analog)


def tf2zpk(b, a):
    """``(z, p, k)`` of a transfer function ``b / a``, like ``scipy.signal.tf2zpk``."""
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
    """``(w, h)``: the response of second-order sections, like ``scipy.signal.sosfreqz``."""
    w = _grid(worN, whole, fs)
    return w, _native.sosfreqz(_np.asarray(sos, dtype=float), w, fs)


def group_delay(system, w=512, whole=False, fs=2 * _np.pi):
    """``(w, gd)``: the group delay of ``(b, a)`` in samples, like ``scipy.signal.group_delay``."""
    b, a = system
    freqs = _grid(w, whole, fs)
    return freqs, _native.group_delay(_np.atleast_1d(_np.asarray(b, dtype=float)), _np.atleast_1d(_np.asarray(a, dtype=float)), freqs, fs)


# ---- filtering -------------------------------------------------------------------------------

def _data(x):
    x = _np.asarray(x)
    return x if x.dtype in (_np.float32, _np.float64) else x.astype(_np.float64)


def lfilter(b, a, x, axis=-1, zi=None):
    """Filters along ``axis`` with ``b / a``, like ``scipy.signal.lfilter``; with ``zi``, returns ``(y, zf)``."""
    b, a = _np.atleast_1d(_np.asarray(b, dtype=float)), _np.atleast_1d(_np.asarray(a, dtype=float))
    return _native.lfilter(b, a, _data(x), axis, None if zi is None else _data(zi))


def lfilter_zi(b, a):
    """Initial state of :func:`lfilter` for a step response's steady state, like ``scipy.signal.lfilter_zi``."""
    return _native.lfilter_zi(_np.atleast_1d(_np.asarray(b, dtype=float)), _np.atleast_1d(_np.asarray(a, dtype=float)))


def sosfilt(sos, x, axis=-1, zi=None):
    """Filters along ``axis`` with second-order sections, like ``scipy.signal.sosfilt``; with ``zi``, returns
    ``(y, zf)``."""
    return _native.sosfilt(_np.asarray(sos, dtype=float), _data(x), axis, None if zi is None else _data(zi))


def sosfilt_zi(sos):
    """Initial state of :func:`sosfilt` for a step response's steady state, like ``scipy.signal.sosfilt_zi``."""
    return _native.sosfilt_zi(_np.asarray(sos, dtype=float))


def filtfilt(b, a, x, axis=-1, padtype="odd", padlen=None, method="pad"):
    """Forward-backward (zero-phase) filtering with ``b / a``, like ``scipy.signal.filtfilt`` (``method='pad'``)."""
    if method != "pad":
        raise NotImplementedError("only method='pad'")
    b, a = _np.atleast_1d(_np.asarray(b, dtype=float)), _np.atleast_1d(_np.asarray(a, dtype=float))
    return _native.filtfilt(b, a, _data(x), axis, padtype, padlen)


def sosfiltfilt(sos, x, axis=-1, padtype="odd", padlen=None):
    """Forward-backward (zero-phase) filtering with second-order sections, like ``scipy.signal.sosfiltfilt``."""
    return _native.sosfiltfilt(_np.asarray(sos, dtype=float), _data(x), axis, padtype, padlen)


# ---- spectral estimation -----------------------------------------------------------------------

def _detrend(d):
    return "none" if d is False or d is None else d


def welch(x, fs=1.0, window="hann", nperseg=None, noverlap=None, nfft=None, detrend="constant",
          return_onesided=True, scaling="density", axis=-1, average="mean"):
    """``(f, Pxx)``: Welch's power spectral density estimate, like ``scipy.signal.welch``."""
    return _native.welch(_data(x), fs, window, nperseg, noverlap, nfft, _detrend(detrend), return_onesided, scaling, axis, average)


def periodogram(x, fs=1.0, window="boxcar", nfft=None, detrend="constant", return_onesided=True, scaling="density", axis=-1):
    """``(f, Pxx)``: the periodogram (one segment), like ``scipy.signal.periodogram``."""
    x = _data(x)
    n = x.shape[axis]
    if nfft is not None and nfft < n:
        x = _np.take(x, _np.arange(nfft), axis=axis)
        n = nfft
    return _native.welch(x, fs, window or "boxcar", n, 0, nfft if nfft is None or nfft > n else None, _detrend(detrend),
                         return_onesided, scaling, axis, "mean")


def csd(x, y, fs=1.0, window="hann", nperseg=None, noverlap=None, nfft=None, detrend="constant",
        return_onesided=True, scaling="density", axis=-1, average="mean"):
    """``(f, Pxy)``: the cross power spectral density, like ``scipy.signal.csd``."""
    return _native.csd(_data(x), _data(y), fs, window, nperseg, noverlap, nfft, _detrend(detrend), return_onesided, scaling, axis, average)


def coherence(x, y, fs=1.0, window="hann", nperseg=None, noverlap=None, nfft=None, detrend="constant", axis=-1):
    """``(f, Cxy)``: magnitude-squared coherence, like ``scipy.signal.coherence``."""
    return _native.coherence(_data(x), _data(y), fs, window, nperseg, noverlap, nfft, _detrend(detrend), axis)


def spectrogram(x, fs=1.0, window=("tukey", 0.25), nperseg=None, noverlap=None, nfft=None, detrend="constant",
                return_onesided=True, scaling="density", axis=-1, mode="psd"):
    """``(f, t, Sxx)``, like ``scipy.signal.spectrogram`` (``mode`` psd, complex, magnitude, angle or phase)."""
    if mode == "complex":
        nper = nperseg or 256
        return stft(x, fs, window, nper, noverlap if noverlap is not None else nper // 8, nfft, detrend, return_onesided,
                    None, False, axis, "psd" if scaling == "density" else "spectrum")
    return _native.spectrogram(_data(x), fs, window, nperseg, noverlap, nfft, _detrend(detrend), return_onesided, scaling, axis, mode)


def stft(x, fs=1.0, window="hann", nperseg=256, noverlap=None, nfft=None, detrend=False, return_onesided=True,
         boundary="zeros", padded=True, axis=-1, scaling="spectrum"):
    """``(f, t, Zxx)``: the short-time Fourier transform, like ``scipy.signal.stft``."""
    scaling = "density" if scaling == "psd" else scaling
    return _native.stft(_data(x), fs, window, nperseg, noverlap, nfft, _detrend(detrend), return_onesided, boundary, padded, axis, scaling)


def istft(Zxx, fs=1.0, window="hann", nperseg=None, noverlap=None, nfft=None, input_onesided=True, boundary=True,
          time_axis=-1, freq_axis=-2, scaling="spectrum"):
    """``(t, x)``: the inverse STFT (overlap-add), like ``scipy.signal.istft``."""
    Z = _np.asarray(Zxx)
    if (time_axis % Z.ndim, freq_axis % Z.ndim) != (Z.ndim - 1, Z.ndim - 2):
        Z = _np.moveaxis(Z, (freq_axis, time_axis), (-2, -1))
    return _native.istft(_np.ascontiguousarray(Z, dtype=_np.complex128), fs, window, nperseg, noverlap, nfft, input_onesided, boundary, scaling)
