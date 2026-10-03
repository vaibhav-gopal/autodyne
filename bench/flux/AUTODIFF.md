# flux against other automatic differentiation

The models of `bench/flux`, written in each library and run on the inputs `examples/flux_programs.rs` writes:
AMD Ryzen 9 7900, one core (process affinity / `taskset`), f32, median run times. Every library's outputs are
checked against flux's interpreter; all agree to 3e-6 relative or better, except the STFT-loss gradients
(2e-5 to 8e-4: float32 rounding through three FFT resolutions and logarithms).

| Model | flux (XLA) | flux interpreter | JAX (XLA) | PyTorch eager | Enzyme (C) | candle | Burn flex | Burn ndarray |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| one-pole low-pass, 48k samples | 153.6 µs | 111.80 ms | 155.8 µs | 303.09 ms | 64.1 µs | 29.75 ms | 44.51 ms | 53.36 ms |
| one-pole MSE gradient, 48k samples | 501.8 µs | 350.28 ms | 491.5 µs | 1.26 s | 160.4 µs | 853.43 ms | 3.51 s | 4.56 s |
| EQ + drive, multi-resolution STFT loss gradient, 2048 samples | 274.1 µs | — | 390.1 µs | 149.29 ms | 3.54 ms | — | — | — |
| rfft -> gain -> irfft -> dense -> tanh gradient, 256 x 1024 | 3.36 ms | 8.52 ms | 3.34 ms | 1.68 ms | 14.50 ms | 19.89 ms | 13.68 ms | 23.75 ms |

autodyne's core `OnePole` filter (no tracing) runs the forward pass in 119.2 µs.

- flux (XLA) and JAX: `compare_jax.py` (XLA's CPU client from Python; run times include Python dispatch).
  PyTorch 2.14 (MKL): `compare_torch.py`, autograd with Python loops for the scans; `torch.compile` needs a
  C++ compiler (MSVC on Windows) and was not available. Enzyme 0.0.217 with clang 21 -O3 in WSL:
  `enzyme/models.c`, plain C loops and a radix-2 FFT that Enzyme differentiates. candle 0.9 and Burn 0.21:
  `bench/rust/examples/flux_autodiff.rs`.
- Burn and candle have no compiled loop: a recurrence is a few tensor ops per sample on their tapes. Neither has
  FFT gradients (Burn's `rfft` / `irfft` are `todo!()` under autodiff; candle has no FFT), so their spectral
  models use DFT matrix products, and the STFT-loss model was not ported.
- Enzyme differentiates compiled code: the fastest on the one-pole, where the whole model is a scalar loop, and
  limited by the FFT and loops it is given on the spectral models.
- flux on XLA pays a compile once (31 ms for the one-pole, 0.1-0.5 s for the others).