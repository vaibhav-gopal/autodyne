# flux against other automatic differentiation

The models of `bench/flux`, written in each library and run on the inputs `examples/flux_programs.rs` writes:
AMD Ryzen 9 7900, one core (process affinity from start / `taskset`), f32, median run times, mimalloc as the
allocator of the Rust processes. Every library's outputs are checked against flux's interpreter; all agree to
3e-6 relative or better, except the STFT-loss gradients (2e-5 to 7e-4: float32 rounding through three FFT
resolutions and logarithms).

| Model | flux in process | flux in process, contracted | flux (XLA) | JAX (XLA) | PyTorch eager | torch.compile | Enzyme (C) | candle | dfdx | Burn flex | Burn ndarray |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| one-pole low-pass, 48k samples | 53.2 µs | 35.6 µs | 126.6 µs | 156.5 µs | 213.66 ms | fails | 64.1 µs | 26.79 ms | 14.79 ms | 25.33 ms | 35.51 ms |
| one-pole MSE gradient, 48k samples | 162.5 µs | 134.7 µs | 483.8 µs | 501.9 µs | 1.04 s | fails | 160.4 µs | 592.81 ms | 699.04 ms | 3.41 s | 4.34 s |
| EQ + drive, multi-resolution STFT loss gradient, 2048 samples | 255.3 µs | — | 278.2 µs | 410.2 µs | 107.84 ms | fails | 3.54 ms | — | — | — | — |
| rfft -> gain -> irfft -> dense -> tanh gradient, 256 x 1024 | 1.40 ms | — | 4.84 ms | 4.83 ms | 1.73 ms | 1.79 ms | 14.50 ms | 23.45 ms | 65.36 ms | 12.96 ms | 24.25 ms |

autodyne's core `OnePole` filter (no tracing, `process` on a block) runs the forward pass in 55.2 µs.

- flux in process: `Scan::run` / `Scan::grad` / `Graph::eval` with the `jit` feature. Scalar scans (per-sample
  recurrences) are compiled to machine code with Cranelift (the step's arithmetic on registers, parameter-only
  values hoisted out of the loop), bit for bit the interpreter's; "contracted" (`Scan::contracted(true)`) also
  fuses products into sums (one rounding instead of two, results differing in the last bits). Array graphs are
  interpreted with element-wise chains fused into single passes (complex arrays' real and imaginary parts read
  in place), frames and overlap-adds as single copies, matrix products reading operands in place and FFTs in the
  arrays' precision with plans kept per thread.
- flux (XLA) and JAX: `compare_jax.py` (XLA's CPU client from Python; run times include Python dispatch).
- PyTorch 2.14 (CPU wheel, Linux): `compare_torch.py`. Eager mode records Python loops for the scans.
  `torch.compile` compiles the losses (backward through AOTAutograd) with the recurrences written with PyTorch's
  `scan` operator so they aren't unrolled; it fails on both recurrences: the one-pole with
  `DataDependentOutputException`, the EQ chain with "scan might be aliasing the input or the output". The spectral
  model compiles (14.2 s on first call) and then runs at 1.79 ms.
- Enzyme 0.0.217 with clang 21 -O3 in WSL: `enzyme/models.c`, plain C loops (clang contracts multiply-adds) and
  a radix-2 FFT that Enzyme differentiates.
- candle 0.9, dfdx 0.13 and Burn 0.21: `bench/rust/examples/flux_autodiff.rs`. None has a compiled loop: a
  recurrence is a few tensor ops per sample on their tapes (dfdx's tensors carry the tape through each
  operation; a value used twice is `retaped`). None has FFT gradients (Burn's `rfft` / `irfft` are `todo!()`
  under autodiff; candle and dfdx have no FFT), so their spectral models use DFT matrix products, and the
  STFT-loss model was not ported.
- flux on XLA pays a compile once (40-75 ms for the one-pole, 0.1-0.6 s for the others); in process, a scalar
  scan's machine code is compiled on first use in about a millisecond.
