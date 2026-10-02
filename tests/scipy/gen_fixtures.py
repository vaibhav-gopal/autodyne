"""Reference outputs from SciPy for tests/scipy_parity.rs.

    python tests/scipy/gen_fixtures.py      # writes tests/scipy/fixtures.json

Complex numbers are [re, im] pairs. Regenerate when adding cases; the JSON is committed so the
Rust tests need neither Python nor SciPy.
"""

import json
import pathlib

import numpy as np
import scipy
from scipy import signal
from scipy.linalg import expm

rng = np.random.default_rng(1234)


def c(z):
    z = np.asarray(z, dtype=complex)
    return [[float(v.real), float(v.imag)] for v in z.ravel()]


def r(x):
    return np.asarray(x, dtype=float).tolist()


fx = {"scipy": scipy.__version__}

# ---- IIR design (digital: second-order sections; analog: zpk) -------------------------------
iir = []


def add_iir(name, kind, order, band, edges, fs, **kw):
    btype = {"lowpass": "lowpass", "highpass": "highpass", "bandpass": "bandpass", "bandstop": "bandstop"}[band]
    wn = edges if len(edges) > 1 else edges[0]
    ftype = {"butter": "butter", "cheby1": "cheby1", "cheby2": "cheby2", "ellip": "ellip", "bessel": "bessel"}[kind]
    args = dict(btype=btype, ftype=ftype, rp=kw.get("rp"), rs=kw.get("rs"))
    if kind == "bessel":
        z, p, k = signal.bessel(order, wn, btype=btype, norm=kw["norm"], fs=fs, output="zpk")
        sos = signal.bessel(order, wn, btype=btype, norm=kw["norm"], fs=fs, output="sos")
    else:
        z, p, k = signal.iirfilter(order, wn, fs=fs, output="zpk", **args)
        sos = signal.iirfilter(order, wn, fs=fs, output="sos", **args)
    iir.append(dict(name=name, kind=kind, order=order, band=band, edges=edges, fs=fs, params=kw,
                    zeros=c(z), poles=c(p), gain=float(k), sos=r(sos)))


add_iir("butter4 lp", "butter", 4, "lowpass", [1000.0], 48000.0)
add_iir("butter5 hp", "butter", 5, "highpass", [300.0], 8000.0)
add_iir("butter3 bp", "butter", 3, "bandpass", [300.0, 3400.0], 16000.0)
add_iir("butter4 bs", "butter", 4, "bandstop", [50.0, 70.0], 1000.0)
add_iir("cheby1 5 lp", "cheby1", 5, "lowpass", [2000.0], 44100.0, rp=1.0)
add_iir("cheby1 4 bp", "cheby1", 4, "bandpass", [1000.0, 2000.0], 8000.0, rp=0.5)
add_iir("cheby2 6 hp", "cheby2", 6, "highpass", [500.0], 8000.0, rs=40.0)
add_iir("cheby2 5 lp", "cheby2", 5, "lowpass", [3000.0], 48000.0, rs=50.0)
add_iir("ellip4 lp", "ellip", 4, "lowpass", [1000.0], 8000.0, rp=0.5, rs=60.0)
add_iir("ellip5 hp", "ellip", 5, "highpass", [200.0], 2000.0, rp=1.0, rs=40.0)
add_iir("ellip6 bp", "ellip", 6, "bandpass", [300.0, 3400.0], 16000.0, rp=0.5, rs=60.0)
add_iir("ellip3 bs", "ellip", 3, "bandstop", [100.0, 300.0], 2000.0, rp=1.0, rs=50.0)
add_iir("bessel4 phase lp", "bessel", 4, "lowpass", [1000.0], 8000.0, norm="phase")
add_iir("bessel5 mag hp", "bessel", 5, "highpass", [500.0], 8000.0, norm="mag")
add_iir("bessel3 delay lp", "bessel", 3, "lowpass", [1000.0], 8000.0, norm="delay")
fx["iir"] = iir

analog = []
for name, f in [("butter4", lambda: signal.butter(4, 2 * np.pi * 1000, analog=True, output="zpk")),
                ("cheby1 3", lambda: signal.cheby1(3, 1.0, 2 * np.pi * 500, analog=True, output="zpk")),
                ("ellip4", lambda: signal.ellip(4, 0.5, 40, 100.0, analog=True, output="zpk")),
                ("cheby2 4 bp", lambda: signal.cheby2(4, 40, [100.0, 400.0], btype="bandpass", analog=True, output="zpk"))]:
    z, p, k = f()
    analog.append(dict(name=name, zeros=c(z), poles=c(p), gain=float(k)))
fx["analog"] = analog

# ---- FIR design -------------------------------------------------------------------------------
fx["firwin"] = [
    dict(numtaps=51, cutoff=[1000.0], window="hamming", pass_zero=True, scale=True, fs=8000.0,
         taps=r(signal.firwin(51, 1000.0, fs=8000.0))),
    dict(numtaps=65, cutoff=[1000.0, 2000.0], window=["kaiser", 8.6], pass_zero=False, scale=True, fs=8000.0,
         taps=r(signal.firwin(65, [1000.0, 2000.0], window=("kaiser", 8.6), pass_zero=False, fs=8000.0))),
    dict(numtaps=63, cutoff=[3000.0], window="hann", pass_zero=False, scale=True, fs=8000.0,
         taps=r(signal.firwin(63, 3000.0, window="hann", pass_zero=False, fs=8000.0))),
    dict(numtaps=61, cutoff=[1000.0, 2000.0], window="blackman", pass_zero=True, scale=False, fs=8000.0,
         taps=r(signal.firwin(61, [1000.0, 2000.0], window="blackman", pass_zero=True, scale=False, fs=8000.0))),
]
fx["firwin2"] = [
    dict(numtaps=31, freq=[0.0, 1000.0, 1000.0, 4000.0], gain=[1.0, 1.0, 0.0, 0.0], antisymmetric=False, fs=8000.0,
         taps=r(signal.firwin2(31, [0.0, 1000.0, 1000.0, 4000.0], [1.0, 1.0, 0.0, 0.0], fs=8000.0))),
    dict(numtaps=30, freq=[0.0, 2000.0, 4000.0], gain=[1.0, 0.5, 0.0], antisymmetric=False, fs=8000.0,
         taps=r(signal.firwin2(30, [0.0, 2000.0, 4000.0], [1.0, 0.5, 0.0], fs=8000.0))),
    dict(numtaps=31, freq=[0.0, 1000.0, 3000.0, 4000.0], gain=[0.0, 1.0, 1.0, 0.0], antisymmetric=True, fs=8000.0,
         taps=r(signal.firwin2(31, [0.0, 1000.0, 3000.0, 4000.0], [0.0, 1.0, 1.0, 0.0], antisymmetric=True, fs=8000.0))),
]
fx["firls"] = [
    dict(numtaps=31, bands=[[0.0, 1000.0], [1500.0, 4000.0]], desired=[[1.0, 1.0], [0.0, 0.0]], weight=None, fs=8000.0,
         taps=r(signal.firls(31, [0.0, 1000.0, 1500.0, 4000.0], [1.0, 1.0, 0.0, 0.0], fs=8000.0))),
    dict(numtaps=41, bands=[[0.0, 500.0], [800.0, 1800.0], [2100.0, 4000.0]], desired=[[0.0, 0.0], [1.0, 0.5], [0.0, 0.0]],
         weight=[10.0, 1.0, 10.0], fs=8000.0,
         taps=r(signal.firls(41, [0.0, 500.0, 800.0, 1800.0, 2100.0, 4000.0], [0.0, 0.0, 1.0, 0.5, 0.0, 0.0],
                             weight=[10.0, 1.0, 10.0], fs=8000.0))),
]
fx["remez"] = [
    dict(numtaps=31, bands=[[0.0, 1000.0], [1500.0, 4000.0]], desired=[1.0, 0.0], weight=None, type="bandpass", fs=8000.0,
         taps=r(signal.remez(31, [0.0, 1000.0, 1500.0, 4000.0], [1.0, 0.0], fs=8000.0))),
    dict(numtaps=40, bands=[[0.0, 500.0], [800.0, 1800.0], [2100.0, 4000.0]], desired=[0.0, 1.0, 0.0], weight=[1.0, 2.0, 1.0],
         type="bandpass", fs=8000.0,
         taps=r(signal.remez(40, [0.0, 500.0, 800.0, 1800.0, 2100.0, 4000.0], [0.0, 1.0, 0.0], weight=[1.0, 2.0, 1.0], fs=8000.0))),
    dict(numtaps=31, bands=[[0.05, 0.45]], desired=[1.0], weight=None, type="hilbert", fs=1.0,
         taps=r(signal.remez(31, [0.05, 0.45], [1.0], type="hilbert"))),
    dict(numtaps=32, bands=[[0.0, 0.4]], desired=[1.0], weight=None, type="differentiator", fs=1.0,
         taps=r(signal.remez(32, [0.0, 0.4], [1.0], type="differentiator"))),
]
fx["kaiserord"] = [dict(ripple=a, width=w, numtaps=int(signal.kaiserord(a, w)[0]), beta=float(signal.kaiserord(a, w)[1]))
                   for a, w in [(60.0, 0.05), (30.0, 0.1), (100.0, 0.02)]]

# ---- windows ---------------------------------------------------------------------------------
windows = []
for spec in ["boxcar", "triang", "bartlett", "hann", "hamming", "blackman", "blackmanharris", "nuttall", "flattop",
             "bohman", "parzen", "cosine", ("kaiser", 6.0), ("gaussian", 2.0), ("tukey", 0.3), ("exponential", None, 2.0),
             ("chebwin", 60.0)]:
    for n in [7, 8]:
        for periodic in [True, False]:
            if spec == ("exponential", None, 2.0) and periodic:
                continue
            w = signal.get_window(spec, n, fftbins=periodic)
            windows.append(dict(spec=spec if isinstance(spec, str) else list(spec), n=n, periodic=periodic, w=r(w)))
fx["windows"] = windows

# ---- conversions -----------------------------------------------------------------------------
b = [1.0, -0.5, 0.25, 0.1]
a = [1.0, -1.2, 0.8, -0.2]
z, p, k = signal.tf2zpk(b, a)
A, B, C, D = signal.tf2ss([2.0, 3.0, 1.0], [1.0, 4.0, 5.0, 2.0])
num, den = signal.ss2tf(np.array([[0.0, 1.0], [-2.0, -3.0]]), np.array([[0.0, 1.0], [1.0, 0.0]]),
                        np.array([[1.0, 0.0], [1.0, 1.0]]), np.array([[0.0, 0.0], [0.5, 0.0]]), input=0)
zs = np.array([-1, -1, 0.5 + 0.5j, 0.5 - 0.5j, 0.9])
ps = np.array([0.8 + 0.1j, 0.8 - 0.1j, 0.3, -0.2 + 0.6j, -0.2 - 0.6j])
fx["convert"] = dict(
    b=b, a=a, zeros=c(z), poles=c(p), gain=float(k),
    tf2ss=dict(num=[2.0, 3.0, 1.0], den=[1.0, 4.0, 5.0, 2.0], A=r(A), B=r(B), C=r(C), D=r(D)),
    ss2tf=dict(num=r(num), den=r(den)),
    zpk2sos=dict(zeros=c(zs), poles=c(ps), gain=2.0,
                 nearest=r(signal.zpk2sos(zs, ps, 2.0, pairing="nearest")),
                 keep_odd=r(signal.zpk2sos(zs, ps, 2.0, pairing="keep_odd")),
                 minimal=r(signal.zpk2sos([-3.0], [-1.0 + 2j, -1.0 - 2j, -4.0], 1.5, pairing="minimal", analog=True))),
)

# ---- frequency responses ---------------------------------------------------------------------
sos = signal.butter(6, [500.0, 1500.0], btype="bandpass", fs=8000.0, output="sos")
bz, az = signal.butter(4, 1000.0, fs=8000.0)
freqs = np.linspace(0, 4000.0, 9)
_, h = signal.freqz(bz, az, worN=freqs, fs=8000.0)
_, hs = signal.sosfreqz(sos, worN=freqs, fs=8000.0)
_, gd = signal.group_delay((bz, az), w=freqs[1:-1], fs=8000.0)
w = np.array([0.0, 100.0, 1000.0, 10000.0])
_, ha = signal.freqs([1.0], [1.0, 2.0, 1.0], worN=w)
fx["freq"] = dict(b=r(bz), a=r(az), sos=r(sos), freqs=r(freqs), h=c(h), hs=c(hs), gd_freqs=r(freqs[1:-1]), gd=r(gd),
                  w=r(w), ha=c(ha))

# ---- filtering -------------------------------------------------------------------------------
x = rng.standard_normal((3, 200))
bf, af = signal.cheby1(4, 1.0, 0.2, output="ba")
sosf = signal.ellip(6, 0.5, 50, [0.1, 0.4], btype="bandpass", output="sos")
zi = signal.lfilter_zi(bf, af)
y1, zf1 = signal.lfilter(bf, af, x, axis=1, zi=np.outer(x[:, 0], zi))
szi = signal.sosfilt_zi(sosf)
y2, zf2 = signal.sosfilt(sosf, x.T, axis=0, zi=szi[:, :, None] * x.T[0][None, None, :])
fx["filter"] = dict(
    x=r(x), b=r(bf), a=r(af), sos=r(sosf),
    lfilter=r(signal.lfilter(bf, af, x, axis=1)), lfilter_axis0=r(signal.lfilter(bf, af, x.T, axis=0)),
    lfilter_zi=r(zi), lfilter_state_y=r(y1), lfilter_state_zf=r(zf1),
    sosfilt=r(signal.sosfilt(sosf, x, axis=1)), sosfilt_zi=r(szi), sosfilt_state_y=r(y2), sosfilt_state_zf=r(zf2),
    filtfilt=r(signal.filtfilt(bf, af, x, axis=1)), filtfilt_even=r(signal.filtfilt(bf, af, x, axis=1, padtype="even", padlen=20)),
    filtfilt_none=r(signal.filtfilt(bf, af, x, axis=1, padtype=None)),
    sosfiltfilt=r(signal.sosfiltfilt(sosf, x, axis=1)), sosfiltfilt_const=r(signal.sosfiltfilt(sosf, x, axis=1, padtype="constant")),
)

# ---- systems: discretization, expm, responses -------------------------------------------------
Ac = np.array([[0.0, 1.0], [-4.0, -0.8]])
Bc = np.array([[0.0], [1.0]])
Cc = np.array([[1.0, 0.0]])
Dc = np.array([[0.0]])
disc = {}
for method in ["zoh", "bilinear", "euler", "backward_diff"]:
    Ad, Bd, Cd, Dd, _ = signal.cont2discrete((Ac, Bc, Cc, Dc), 0.1, method=method)
    disc[method] = dict(A=r(Ad), B=r(Bd), C=r(Cd), D=r(Dd))
M = rng.standard_normal((4, 4)) * 2.0
t = np.arange(50) * 0.1
_, y_step = signal.step((Ac, Bc, Cc, Dc), T=t)
_, y_imp = signal.impulse((Ac, Bc, Cc, Dc), T=t)
u = np.sin(np.arange(40) * 0.3)
_, y_dl, x_dl = signal.dlsim((disc["zoh"]["A"], disc["zoh"]["B"], disc["zoh"]["C"], disc["zoh"]["D"], 0.1), u)
bb, ab = signal.bilinear([1.0, 2.0], [1.0, 3.0, 2.0], fs=10.0)
fx["systems"] = dict(A=r(Ac), B=r(Bc), C=r(Cc), D=r(Dc), discretized=disc, M=r(M), expm=r(expm(M)),
                     expm_big=r(expm(M * 10.0)), step=r(y_step), impulse=r(y_imp), u=r(u), dlsim_y=r(y_dl), dlsim_x=r(x_dl),
                     bilinear_b=r(bb), bilinear_a=r(ab))

out = pathlib.Path(__file__).with_name("fixtures.json")
out.write_text(json.dumps(fx))
print(f"wrote {out} ({out.stat().st_size // 1024} KiB), SciPy {scipy.__version__}")
