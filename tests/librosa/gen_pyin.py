"""Reference pYIN outputs from librosa for tests/librosa_parity.rs (analysis::PitchTracker).

    uv run --no-project --with librosa python tests/librosa/gen_pyin.py   # writes tests/librosa/pyin.json

Synthetic voices with known pitch tracks, silence, breath noise and breathy phonation, stored as
16-bit PCM (sample = pcm / 32768) so both sides read exactly the same input. librosa runs with its
defaults except center=False: frame t then starts at sample t * hop_length, as a stream sees it.
Regenerate when adding cases; the JSON is committed so the Rust tests need neither Python nor librosa.
"""

import json
import pathlib

import librosa
import numpy as np
from scipy import signal as sps

SR = 16_000
FMIN, FMAX = 65.0, 1000.0
FRAME, HOP = 1024, 128
rng = np.random.default_rng(2026)


def harmonic(f0, amp, rolloff=1.0, n_harm=12):
    """A harmonic tone with instantaneous frequency f0 (Hz, per sample) and amplitude amp."""
    phase = 2 * np.pi * np.cumsum(f0) / SR
    out = np.zeros_like(f0)
    for h in range(1, n_harm + 1):
        alias = h * f0 < SR / 2
        out += alias * np.sin(h * phase) / h**rolloff
    return amp * out / np.max(np.abs(out) + 1e-12)


def breath(n, level):
    """Band-limited noise like breath (300 Hz to 4 kHz)."""
    sos = sps.butter(4, [300, 4000], btype="bandpass", fs=SR, output="sos")
    x = sps.sosfilt(sos, rng.standard_normal(n))
    return level * x / np.std(x)


def env(n, attack=0.03, release=0.05):
    e = np.ones(n)
    a, r = int(attack * SR), int(release * SR)
    e[:a] = np.linspace(0, 1, a)
    e[n - r :] = np.linspace(1, 0, r)
    return e


def seconds(s):
    return int(round(s * SR))


def glide_vibrato():
    parts = [np.zeros(seconds(0.2))]
    n = seconds(0.9)
    t = np.arange(n) / SR
    f0 = 196.0 * 2 ** (np.log2(294.0 / 196.0) * t / t[-1]) * 2 ** (0.3 / 12 * np.sin(2 * np.pi * 5.5 * t))
    parts.append(harmonic(f0, 0.5) * env(n))
    parts.append(breath(seconds(0.3), 0.05))
    n = seconds(0.8)
    tone = harmonic(np.full(n, 110.0), 0.4, rolloff=1.6)
    parts.append((tone + breath(n, 0.08)) * env(n))
    parts.append(breath(seconds(0.2), 0.001))
    return np.concatenate(parts)


def low_breathy():
    n = seconds(1.5)
    t = np.arange(n) / SR
    f0 = 100.0 * 2 ** (np.log2(82.0 / 100.0) * t / t[-1])
    amp = 0.3 * (0.6 + 0.4 * np.sin(2 * np.pi * 1.3 * t))
    tone = harmonic(f0, 1.0, rolloff=2.0) * amp
    return (tone + breath(n, 0.06)) * env(n, 0.08, 0.2)


def melody():
    notes = [261.63, 329.63, 392.0, 523.25, 220.0]
    f0 = []
    for i, f in enumerate(notes):
        seg = np.full(seconds(0.3), f)
        if i:
            glide = seconds(0.03)  # portamento from the previous note
            seg[:glide] = np.geomspace(notes[i - 1], f, glide)
        f0.append(seg)
    f0 = np.concatenate(f0)
    tone = harmonic(f0, 0.5, rolloff=1.5) * env(len(f0))
    gap = breath(seconds(0.15), 0.02)
    return np.concatenate([tone[: seconds(0.9)], gap, tone[seconds(0.9) :]])


fx = {
    "librosa": librosa.__version__,
    "sr": SR,
    "fmin": FMIN,
    "fmax": FMAX,
    "frame_length": FRAME,
    "hop_length": HOP,
    "signals": [],
}
for name, make in [("glide_vibrato", glide_vibrato), ("low_breathy", low_breathy), ("melody", melody)]:
    pcm = np.clip(np.round(make() * 32768), -32768, 32767).astype(np.int16)
    y = pcm.astype(np.float64) / 32768
    f0, voiced, prob = librosa.pyin(y, fmin=FMIN, fmax=FMAX, sr=SR, frame_length=FRAME, hop_length=HOP, center=False)
    fx["signals"].append(
        {
            "name": name,
            "pcm": pcm.tolist(),
            "f0": [None if np.isnan(v) else float(v) for v in f0],
            "voiced": [bool(v) for v in voiced],
            "voiced_prob": [float(v) for v in prob],
        }
    )
    print(f"{name}: {len(pcm)} samples, {len(f0)} frames, {int(np.sum(voiced))} voiced")

out = pathlib.Path(__file__).with_name("pyin.json")
out.write_text(json.dumps(fx, separators=(",", ":")) + "\n")
print(f"wrote {out} ({out.stat().st_size // 1024} KB)")
