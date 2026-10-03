import os

import numpy as np
import pytest

import autodyne
from autodyne import flux

FS = 48_000.0


def noise(n, seed):
    return np.random.default_rng(seed).uniform(-1, 1, n)


def test_trace_matches_numpy_and_keeps_precision():
    g = flux.trace(lambda x, w: [(x.sin() * 2.0 + 1.0) @ w, x[1:, ::2].T, x.max(axis=1), flux.where(x > 0.0, x, -x)], [[3, 4], [4, 2]])
    x, w = noise(12, 1).reshape(3, 4), noise(8, 2).reshape(4, 2)
    for dtype in (np.float32, np.float64):
        out = g(x.astype(dtype), w.astype(dtype))
        assert all(o.dtype == dtype for o in out)
        tol = 1e-5 if dtype == np.float32 else 1e-12
        np.testing.assert_allclose(out[0], (np.sin(x) * 2 + 1) @ w, rtol=tol, atol=tol)
        np.testing.assert_allclose(out[1], x[1:, ::2].T, rtol=tol)
        np.testing.assert_allclose(out[2], x.max(axis=1), rtol=tol)
        np.testing.assert_allclose(out[3], np.abs(x), rtol=tol)


def test_value_and_grad():
    g = flux.value_and_grad(lambda x, y: (x.sin() * y).sum(), [[5], [5]])
    x, y = noise(5, 3), noise(5, 4)
    value, dx, dy = g(x, y)
    np.testing.assert_allclose(value, np.sum(np.sin(x) * y), rtol=1e-12)
    np.testing.assert_allclose(dx, np.cos(x) * y, rtol=1e-12)
    np.testing.assert_allclose(dy, np.sin(x), rtol=1e-12)
    only_x = flux.value_and_grad(lambda x, y: (x * y).sum(), [[5], [5]], argnums=[0])
    assert len(only_x(x, y)) == 2
    with pytest.raises(ValueError):
        flux.value_and_grad(lambda x: x * 2.0, [[3]])


def test_errors_in_traced_functions_propagate():
    with pytest.raises(ZeroDivisionError):
        flux.trace(lambda x: 1 / 0, [[2]])
    with pytest.raises(TypeError):
        flux.trace(lambda x: "not a tracer", [[2]])


def test_a_scan_of_the_biquad_matches_the_filter():
    # the Butterworth low-pass autodyne.lowpass applies, as a traced biquad
    xs = noise(512, 5)
    scan = flux.Scan([[]], [[], []], [], lambda p, s, x: flux.biquad("lowpass", p[0], 2 ** -0.5, 0.0, (s[0], s[1]), x, sample_rate=FS))
    ys, state = scan.run([1_000.0], xs, [0.0, 0.0])
    np.testing.assert_allclose(ys, autodyne.lowpass(xs, 1_000.0, FS), rtol=1e-9, atol=1e-12)
    assert len(state) == 2


def test_fitting_an_eq_and_drive_with_the_stft_loss():
    def step(p, s, x):
        (a, b), y = flux.biquad("peaking", p[0].exp(), 1.0, p[1] * 10.0, (s[0], s[1]), x, sample_rate=FS)
        return [a, b], flux.shape("tanh", y * p[2])

    n = 1024
    chain = flux.Scan([[], [], []], [[], []], [], step)
    xs = (0.5 * noise(n, 6)).astype(np.float32)
    target, _ = chain.run([np.log(3_000.0), 0.9, 2.0], xs, [0.0, 0.0])
    loss = flux.Loss.stft([n], resolutions=[(256, 64, 256), (64, 16, 64)])
    params, adam = [np.log(1_000.0), 0.3, 1.0], flux.Adam(0.03)
    first = None
    for _ in range(250):
        g = chain.grad([np.float32(p) for p in params], xs, [0.0, 0.0], [target], loss=loss)
        first = first or g["loss"]
        params = adam.step(params, g["params"])
    assert g["loss"] < first * 0.05
    assert abs(np.exp(params[0]) / 3_000.0 - 1.0) < 0.05
    assert g["input"].shape == (n,)


def test_programs_and_a_backend():
    scan = flux.Scan([[]], [[]], [], lambda p, s, x: flux.one_pole(p[0], s[0], x))
    program = scan.grad_program(64)
    assert "stablehlo.while" in program.text and program.dtype == "float32"
    assert scan.forward_program(64, dtype="float64").dtype == "float64"
    if not os.environ.get("AUTODYNE_IREE_DIR"):
        pytest.skip("IREE tools not configured")
    xs = noise(64, 7).astype(np.float32)
    targets, _ = scan.run([1_500.0], xs, [0.0])
    exe = flux.Backend.iree().compile(program)
    loss, d_cutoff, d_state, d_xs = exe(1_000.0, xs, targets, 0.0)
    want = scan.grad([np.float32(1_000.0)], xs, [0.0], [targets])
    np.testing.assert_allclose(loss, want["loss"], rtol=1e-5)
    np.testing.assert_allclose(d_cutoff, want["params"][0], rtol=1e-4)
    np.testing.assert_allclose(d_xs, want["input"], rtol=1e-4, atol=1e-9)


def test_complex_spectra_and_checkpointing():
    n = 64
    g = flux.value_and_grad(lambda x, gain: flux.irfft_complex(x.rfft_complex() * flux.complex(gain, gain * 0.5), n).sum() + x.fft().conj().real.sum(), [[3, n], [n // 2 + 1]])
    x, gain = noise(3 * n, 8).reshape(3, n), noise(n // 2 + 1, 9)
    value, dx, dgain = g(x, gain)
    spectrum = np.fft.rfft(x, axis=-1)
    want = np.fft.irfft(spectrum * (gain + 0.5j * gain), n=n, axis=-1).sum() + np.fft.fft(x, axis=-1).real.sum()
    np.testing.assert_allclose(value, want, rtol=1e-10)
    scan = flux.Scan([[]], [[]], [], lambda p, s, x: (lambda st, y: ([st], flux.shape("tanh", y * 2.0)))(*flux.one_pole(p[0], s[0], x)))
    xs, target = noise(128, 10), noise(128, 11)
    saved = scan.grad([900.0], xs, [0.0], [target])
    recomputed = scan.checkpointed(True).grad([900.0], xs, [0.0], [target])
    assert not scan.checkpointed(True).residual_shapes
    fused = scan.contracted(True).grad([900.0], xs, [0.0], [target])
    np.testing.assert_allclose(fused["params"][0], saved["params"][0], rtol=1e-6)
    np.testing.assert_allclose(saved["params"][0], recomputed["params"][0], rtol=1e-12)
    program = flux.trace(lambda x: x.rfft()[0], [[4, 2048]]).program(max_fft=64)
    assert "length = [2048]" not in program.text