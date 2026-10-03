# flux against other automatic differentiation

The models of `bench/flux`, written in each library and run on the inputs `examples/flux_programs.rs` writes:
AMD Ryzen 9 7900, one core (process affinity / `taskset`), f32, median run times, mimalloc as the allocator of
the Rust processes. Every library's outputs are checked against flux's interpreter; all agree to 3e-6 relative or
better, except the STFT-loss gradients (2e-5 to 8e-4: float32 rounding through three FFT resolutions and
logarithms).

| Model | flux in process | flux in process, contracted | flux (XLA) | JAX (XLA) | PyTorch eager | Enzyme (C) | candle | Burn flex | Burn ndarray |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| one-pole low-pass, 48k samples | 53.2 µs | 35.6 µs | 153.6 µs | 155.8 µs | 303.09 ms | 64.1 µs | 14.23 ms | 24.63 ms | 34.10 ms |
| one-pole MSE gradient, 48k samples | 171.8 µs | 145.8 µs | 501.8 µs | 491.5 µs | 1.26 s | 160.4 µs | 597.09 ms | 3.07 s | 4.18 s |
| EQ + drive, multi-resolution STFT loss gradient, 2048 samples | 352.2 µs | — | 274.1 µs | 390.1 µs | 149.29 ms | 3.54 ms | — | — | — |
| rfft -> gain -> irfft -> dense -> tanh gradient, 256 x 1024 | 1.43 ms | — | 3.36 ms | 3.34 ms | 1.68 ms | 14.50 ms | 22.65 ms | 12.39 ms | 19.85 ms |

autodyne's core `OnePole` filter (no tracing, `process` on a block) runs the forward pass in 57.3 µs.

- flux in process: `Scan::run` / `Scan::grad` / `Graph::eval` with the `jit` feature. Scalar scans (per-sample
  recurrences) are compiled to machine code with Cranelift (the step's arithmetic on registers, parameter-only
  values hoisted out of the loop), bit for bit the interpreter's; "contracted" (`Scan::contracted(true)`) also
  fuses products into sums (one rounding instead of two, results differing in the last bits). Array graphs are
  interpreted with element-wise chains fused into single passes, matrix products reading operands in place and
  FFTs in the arrays' precision.
- flux (XLA) and JAX: `compare_jax.py` (XLA's CPU client from Python; run times include Python dispatch).
  PyTorch 2.14 (MKL): `compare_torch.py`, autograd with Python loops for the scans; `torch.compile` needs a
  C++ compiler (MSVC on Windows) and was not available. Enzyme 0.0.217 with clang 21 -O3 in WSL:
  `enzyme/models.c`, plain C loops (clang contracts multiply-adds) and a radix-2 FFT that Enzyme differentiates.
  candle 0.9 and Burn 0.21: `bench/rust/examples/flux_autodiff.rs`.
- Burn and candle have no compiled loop: a recurrence is a few tensor ops per sample on their tapes. Neither has
  FFT gradients (Burn's `rfft` / `irfft` are `todo!()` under autodiff; candle has no FFT), so their spectral
  models use DFT matrix products, and the STFT-loss model was not ported.
- flux on XLA pays a compile once (31 ms for the one-pole, 0.1-0.5 s for the others); in process, a scalar
  scan's machine code is compiled on first use in about a millisecond.