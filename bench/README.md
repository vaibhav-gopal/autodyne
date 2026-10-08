# Benchmarks

Every suite runs the same work on the same inputs in autodyne and in the libraries it is compared with,
checks that the results agree, and times them. Each comparison script prints a Markdown table; `--out FILE`
also writes it (the `RESULTS.md` next to the script is that table from the last run). Time on one core
where the suite says so: `taskset -c 0 ...` on Linux, `start /affinity 1 ...` on Windows.

| Suite | Compares | Run | Results |
|---|---|---|---|
| `dsp.rs` | autodyne's own processors and kernels (criterion), no peers | `cargo bench --bench dsp` | criterion's report in `target/criterion` |
| `numpy/` | the Python bindings against NumPy / SciPy: element-wise maths, reductions, FFTs, linear algebra, filtering, filter design, spectral estimates (end to end, inputs and results crossing the boundary) | `python bench/numpy/compare.py --out bench/numpy/RESULTS.md` (bindings installed: `maturin develop --release` in `bindings/python`) | [`numpy/RESULTS.md`](numpy/RESULTS.md) |
| `rust/` | Rust array, tensor and GPU libraries: ndarray, Burn (flex, ndarray, wgpu) and CubeCL against autodyne's arrays and `GpuArray` | `cd bench/rust && cargo bench --bench compare && python results.py --out RESULTS.md` | [`rust/RESULTS.md`](rust/RESULTS.md) |
| `geometry/` | autodyne's `geometry` against glam: `Mat4` and `Mat3` products and inverses (f32 and f64) and `compose_world` against glam in the same loop, alternating rounds, each side's best | `cd bench/geometry && cargo run --release -- --out RESULTS.md` | [`geometry/RESULTS.md`](geometry/RESULTS.md) |
| `linalg/` | linear algebra in NumPy (OpenBLAS LAPACK), PyTorch, JAX, autodyne, faer, nalgebra and Burn, one thread, f64 | `python bench/linalg/compare.py --out bench/linalg/RESULTS.md` (builds and runs `bench/rust/examples/linalg.rs` itself) | [`linalg/RESULTS.md`](linalg/RESULTS.md) |
| `flux/` (XLA) | flux's StableHLO against JAX's own lowering of the same models, on the same XLA: compile and run times | `cargo run --release --features flux --example flux_programs -- DIR`, then `python bench/flux/compare_jax.py DIR --out bench/flux/RESULTS.md` | [`flux/RESULTS.md`](flux/RESULTS.md) |
| `flux/` (autodiff) | flux in process against PyTorch (eager, `torch.compile`), Enzyme, candle and Burn on the same models | the `flux_programs` step above, then `python bench/flux/compare_torch.py DIR`, the C models (`bench/flux/enzyme/models.c`, its header has the command), and `cd bench/rust && cargo run --release --example flux_autodiff -- DIR` | [`flux/AUTODIFF.md`](flux/AUTODIFF.md) (collected by hand from the runs) |

Notes:

- `DIR` is any directory: `flux_programs` writes the models' StableHLO and their inputs and outputs (`.npy`)
  there, and every flux comparison reads them, so all libraries see the same data and are checked against
  flux's interpreter.
- `bench/rust` is a workspace of its own (Burn, candle and CubeCL are heavy; the core crate's builds don't
  pay for them). It uses mimalloc, as the Python bindings do. `bench/geometry` is one too (glam), small and
  quick to build.
- In criterion code, wrap inputs in `black_box` so the work isn't optimized away.
