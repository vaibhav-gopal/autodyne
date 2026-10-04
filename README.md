# Autodyne - DSP and Numerical Library

## Goals
- should be as performant as possible
  - SIMD/vectorization
  - OPTIONAL: GPU compute via OpenCL (or CUDA) after that as an exercise (should be interesting)
- should support realtime processing
    - should support acceptable real-time processing on posix and windows with as small as possible buffer sizes (hard since non realtime os)
- should have a respectable selection of useful and common DSP operations
  - see the JUCE library documentation for their DSP related functions and try to get most of the functionality they have
- should support extending the crate with user-given DSP functions that are **easy** to create and work with
- should allow creation of modulation and demodulation DSP operations that can be user defined **easily** (for future me: remember modulation is process of **encoding** information into a carrier signal to be transmitted)
  - should have a basic implementation of IQ modulation and demodulation (enables AM, PM and FM)
- OPTIONAL: creating a type of "signal" graph / pipeline (a system...), and passing in audio (basis of most audio plugins / engines)
  - should easily represent sampling rate differences between stages and account for it
  - implementation details to be determined ; probably not scoped to this library

## What works today

Everything is generic over `f32` / `f64`. Processors are constructed once (that's where any allocation
happens) and then run in place on `&mut [T]` blocks, so they are safe to call from an audio callback.

Modules, from the base up: each uses only modules above it in the table. `flux` (differentiable programs) and the plugins sit on top; see below.

| module | contents |
|---|---|
| `units` | number traits (`Float`, `Real`, `Integer`, `Trig`, casts, ...), `Complex<T>`, decibels (`db_to_gain` / `gain_to_db`), and reflection: `DType` (runtime element type) and `Reflection` |
| `simd` | vectorized `dot` kernel on stable Rust; AVX2 chosen at runtime on x86-64 (used by `Fir`, `Resampler` and the `Signal` reductions) |
| `fft` | `Fft` (forward, inverse, real input) and `RealFft` (real signals) on `rustfft` / `realfft` kernels (AVX / SSE / NEON, default feature `rustfft`; a portable radix-2 / Bluestein path otherwise), planned with their scratch so transforms never allocate; a reference `dft` |
| `processor` | the `Processor` trait all effects share; tuples are zero-cost chains, `Vec<Box<dyn Processor>>` is a runtime chain |
| `wav` | dependency-free WAV `read` / `write`: 8-32-bit PCM and 32/64-bit float, `WAVE_FORMAT_EXTENSIBLE`, sampler metadata (`smpl` root key and loops) |
| `signal` | the core abstraction. Any sample slice, `Vec`, array or `NdArray` is a signal: `Signal` (levels, norms, statistics, argmax, inner/angle/distance, convolved/correlated), `SignalMut` (gain, normalize, fades, cumsum/diff, projection, pointwise math with `Broadcast` policies), `ComplexSignal`. Capability tiers `SignalOwned` / `SignalResizable` unlock `SigOwnedOps` / `SigResizeOps`. `Source`s compose lazily (`scaled`, `mix`, `through`). Streams: `SignalRead` / `SignalWrite` / `SignalSeek` with `SampleReader` / `SampleWriter` at the byte boundary. `NdArray` (over `Vec`, `Box<[T]>`, shared copy-on-write `Arc<[T]>` or foreign storage) + zero-copy `NdView` / `NdViewMut` (O(1) slicing, steps and reversal, permute / transpose, broadcasting, reshape without copying when the layout allows it and `to_shape` when it does not; disjoint mutable lanes and sub-views; layouts checked once, mutable ones proven injective) whose lanes are signals, with axis labels; `Zip` (fused element-wise passes over up to 5 views of any layout, NumPy broadcasting, as fast as a hand-written slice loop), arithmetic operators, reductions (`sum` pairwise and vectorized, `mean`, `rms`, `min` / `max`, `sum_axis`, and `fold_into` / `sum_into`: reducing into any shape that broadcasts to the input), and allocation-free `process_lanes` along strided axes |
| `channels` | planar `AudioBuffer` (+ interleave conversion, `[channel, time]` n-d view, linked per-frame gain), `MultiProcessor`, `PerChannel`, `Linked` (one detector for every channel: `Linked(compressor)`) |
| `linalg` | (feature `faer`, default) on `NdView`s of any strides, read in place: `matmul` (a register-blocked AVX2 kernel for small row-major products, faer above), `solve`, `lstsq`, `inv`, `pinv`, `det`, `cond`, `matrix_rank`, `qr`, `cholesky`, `eigh` / `eigvalsh` and `svd` / `svdvals` (faer's reductions, then LAPACK-style divide and conquer and dqds ported to Rust), `eig` / `eigvals`, `expm`, polynomials (`roots`, `poly`, `polyval`, `polymul`, `polyadd`) |
| `systems` | (feature `faer`) LTI systems as `scipy.signal` has them: `TransferFunction`, `Zpk` and `StateSpace` (continuous or sampled; poles, zeros, stability, series / parallel / feedback), the conversions (`tf2zpk`, `zpk2sos`, `tf2ss`, ...), responses (`freqz`, `sosfreqz`, `freqs`, `group_delay`, impulse, step, `lsim`), and discretization (ZOH, bilinear with or without prewarping, Euler, backward difference, generalized bilinear) |
| `params` | `ParamInfo` (range, unit, scale, kind: continuous / integer / toggle / choice, smoothing, normalized 0..1, formatting and parsing) and `Parameterized` (each processor module declares its own, in its `params.rs`; tuple / `Vec` chains and linked `PerChannel` combine them); `Smoothed` makes any processor's automation click-free (control-rate ramps for coefficients, per-sample smoothing for gains), and `ParamEvent` applies changes at exact sample offsets |
| `gain` | levels: smoothed `Gain`, `Panner` (constant-power), `StereoWidth` (mid / side), and `SmoothedValue` (click-free parameter ramps) |
| `osc` | band-limited `Oscillator` (saw / pulse with PolyBLEP, triangle, sine), `Sine`, `Phasor` (complex oscillator), `Noise` (seeded), `Impulse`, and `WavetableOsc` (mip-mapped `Wavetable`s built from single-cycle frames or harmonics, alias-free to below -80 dB, smooth morphing across frames) |
| `distortion` | `Waveshaper` (tanh, soft clip, hard clip, fold) with smoothed drive / output / mix; `Bitcrusher` (fractional bit depth, sample-and-hold rate reduction, optional dither) |
| `spectral` | windows (`get_window`), estimates as `scipy.signal` makes them (periodogram, Welch, CSD, coherence, spectrogram, STFT / ISTFT, Lomb-Scargle), and a phase vocoder with identity phase locking: `PitchShifter` (real time, fixed latency) and `time_stretch` (offline, any factor 0.1-10) |
| `filter` | `Fir` + windowed-sinc `design_lowpass`, `Biquad` (low/high/band-pass, notch, all-pass, peaking, low/high shelf), `MultiBiquad` (one per channel, 4 channels in lockstep: ~3.8x faster on 8 channels), `ParametricEq` (up to 8 bands: bells, shelves, 12-48 dB/oct cuts, notch, band-pass; its curve for drawing), `LinkwitzRiley` and `Crossover` (2-4 bands that sum flat), and two filters built for modulation: `Svf` (zero-delay-feedback state-variable filter, stable under per-sample cutoff changes, six simultaneous responses) and `Ladder` (4-pole Moog-style, zero-delay feedback, self-oscillates at the cutoff, level-compensated drive); `magnitude_at` for analytic responses |
| `analysis` | `SpectrumAnalyzer` (sliding FFT with analyzer ballistics, peak trace, log-spaced bands for drawing; Hann / Blackman-Harris windows), `LoudnessMeter` (ITU-R BS.1770 / EBU R128: momentary, short-term and gated integrated LUFS, loudness range, true peak; fixed memory for any duration), `TruePeak` (oversampled inter-sample peaks), `PitchDetector` (YIN via FFT correlation), `OnsetDetector` (SuperFlux-style spectral flux with adaptive peak picking) |
| `iq` | `IqModulator` / `IqDemodulator`, `envelope` (AM), `phase` (PM), `FmModulator` / `FmDiscriminator` |
| `delay` | `DelayLine` (integer and interpolated reads), `Echo` (feedback delay with smoothed parameters) |
| `dynamics` | `EnvelopeFollower`, `Compressor` (soft knee, attack/release, makeup), `LookaheadLimiter` (linked brickwall with a smooth lookahead ramp, true-peak detection, never over the ceiling), `Gate` (gate and downward expander: range, hysteresis, hold), `TransientShaper` (level-independent attack / sustain), `MultibandCompressor` (2-4 Linkwitz-Riley bands, linked); `Linked` runs any of them across channels |
| `envelope` | `Adsr`: exact linear segments, click-free retrigger / release; a VCA `Processor` and a modulation `Source` |
| `modulation` | `ModulatedDelay` (`chorus` / `flanger` presets), `Phaser` (swept all-pass stages) |
| `reverb` | `Reverb` (8-line feedback delay network: exact RT60, modulated delays with all-pass interpolation, damping, pre-delay, diffusion, decorrelated stereo) and `Convolver` (partitioned FFT convolution with any impulse response; `synthetic_ir`) |
| `resample` | streaming rational `Resampler` (polyphase, e.g. 48 kHz <-> 44.1 kHz), `Resample` (`x.resampled(48_000, 44_100)` on any signal, aligned to the input) and `Oversampled<P>` (runs any processor at 2x / 4x / 8x; 4x cuts tanh saturation aliasing by ~39 dB) |
| `synth` | `MidiMessage` parsing (including aftertouch), the `Voice` trait, `Poly` (voice allocation and stealing, sustain pedal, pitch bend, sample-accurate events, poly / mono / legato modes with last-note priority, MPE and per-note expression; mono or stereo rendering) and `SynthVoice` (up to 7 detuned unison oscillators -> SVF or ladder filter swept by its envelope, pressure and timbre -> ADSR, with glide; classic waveforms or a morphing wavetable) and `FmVoice` (4-operator phase modulation: 8 algorithms, ratios, detune, feedback, an envelope per operator, timbre and pressure on modulation depth) |
| `sampler` | `Sample` (mono / stereo, root key, forward / ping-pong / sustain loops with crossfades, optional mipmaps for notes far above the root; loads WAV with its loop metadata), `Zone` / `SampleMap` (key and velocity ranges, tuning, gain, pan, round robin) and `SamplerVoice` (64-tap Kaiser-sinc, cubic or linear interpolation, band-limited when pitched up, amplitude envelope, stereo) for `Poly` |
| `control` | `Lfo` (sine, triangle, saws, square, sample & hold, smooth random; free-running or locked to the host's beat grid; fade-in, retrigger), `Modulated` (a modulation matrix around any processor: routes from sources to any parameter, depths in normalized units with "via" scaling, automatable depths), `Transport` and `Division` (tempo, beat position, dotted and triplet note lengths) |
| `dynamic` | runtime-typed data and processing: `DynArray` (any dtype, bytes), zero-copy `DynView` over external memory, runtime-typed math (`binary` with broadcasting, scalars with NumPy's weak typing, `unary` functions, `sum` / `mean` / `min` / `max`, `cast` checked / saturating / wrapping) under explicit `Promotion` rules: NumPy's table with every conversion checked, so nothing is lost silently (or `KeepFloat`, floats keeping their width); `DynProcessor` built at runtime with `build_dyn` |
| `dlpack` | DLPack 1.x (and legacy) export and import: arrays go to NumPy / PyTorch / JAX / CuPy without copying (shared arrays read-only), foreign tensors of any strides come in as views or, when contiguous, as `NdArray`s over the foreign memory, released exactly once |
| `interop` | zero-copy conversions with the `ndarray` crate (feature `ndarray`): views both ways for any strides, owned arrays move their `Vec`; Arrow buffers are slices already |
| `gpu` | (feature `gpu`) `GpuArray<f32 / f64>` kept in GPU memory, CubeCL kernels through wgpu (Vulkan, Metal, DirectX 12): fused `a * x + b`, element-wise ops with arrays (row / column broadcasts) and scalars, `exp` / `ln` / `tanh` / `sin` / `cos` / `sqrt` / `abs` (f64 ones computed from f64 arithmetic, within 4 ulps), sums along any axis, per-lane FIR; transposes and permutations are views; `available()` / `supports::<T>()`; in Python as `autodyne.gpu` |
| `prelude` | `use autodyne::prelude::*` brings every trait and the common types into scope |

```rust
use autodyne::prelude::*;
use autodyne::filter::{Biquad, BUTTERWORTH_Q};
use autodyne::osc::Sine;

let mut block = [0.0f32; 512];
let mut tone = Sine::new(440.0, 48_000.0);
let mut lowpass = Biquad::lowpass(1_000.0, BUTTERWORTH_Q as f32, 48_000.0);

tone.fill(&mut block);       // generate
lowpass.process(&mut block); // filter in place
let level = block.rms_db();  // analyse: every slice is a signal
```

### Examples
`cargo run --release --example live [seconds] [--demo]` is a playable 8-voice synth: plug in a MIDI keyboard (or it
plays a built-in chord sequence), through oversampled saturation and a stereo chain (chorus, linked compressor, echo,
width, limiter), rendered live in the audio callback. MIDI reaches the audio thread through a lock-free ring buffer.

The others write WAV files to `target/examples-out/`:

- `cargo run --example simple`: the smallest program, a mix of two sines low-passed as it is generated, levels printed
- `cargo run --release --example tone`: a 440 Hz tone, the same tone with noise, and the noisy one low-passed; prints the noise reduction and the FFT peak
- `cargo run --release --example fm_radio`: an FM radio link (modulate, 12 kHz carrier, noisy channel, demodulate); prints the audio SNR
- `cargo run --release --example effects`: a melody through a processor chain (EQ, compressor, echo, limiter), rendered at 48 kHz and resampled to 44.1 kHz
- `cargo run --release --example reverb`: a synth phrase dry, through the FDN reverb, and through convolution reverb
- `cargo run --release --example sampler`: records two FM notes as looped WAV files, loads them back as velocity layers, and plays a stereo phrase
- `cargo run --release --example host -- [f32|f64] [id=value ...]`: a host picking the sample type at runtime, listing and setting parameters by id, processing, and printing a preset
- `cargo run --release --example interop`: a `[batch, channel, time]` tensor filtered along time, shared as raw memory without copying, and streamed through 16-bit PCM

### Python
`bindings/python` builds the `autodyne` Python package (PyO3 + maturin). Arrays cross between NumPy (or PyTorch,
JAX...) and autodyne through DLPack in both directions, so nothing is copied: inputs are read in place with any strides,
results come back as NumPy arrays that own autodyne's memory. Besides the core functions, `autodyne.linalg`,
`autodyne.signal` and `autodyne.fft` follow `numpy.linalg` / `scipy.signal` / `numpy.fft` names and defaults (solve,
eig, svd, expm; butter / cheby / ellip / bessel, firwin, remez, lfilter, sosfilt, filtfilt, welch, stft, spectrogram...),
tested against NumPy and SciPy.

```sh
cd bindings/python
uv venv .venv && uv pip install --python .venv maturin numpy scipy pytest
VIRTUAL_ENV=$PWD/.venv .venv/bin/maturin develop --release --uv   # (Scripts\ on Windows)
.venv/bin/python -m pytest tests
```
### Benchmarks against NumPy / SciPy
`bench/numpy/compare.py` runs the same work in NumPy / SciPy and autodyne on the same arrays (checking the
results agree), single-threaded, end to end: autodyne's times include crossing into Rust and back through DLPack. Latest
results are in [`bench/numpy/RESULTS.md`](bench/numpy/RESULTS.md). On a Ryzen 9 7900:

- the extension module allocates with mimalloc (autodyne's `mimalloc` feature), which reuses freed pages: a large new
  array costs no page faults on first touch, most of the time of a large element-wise operation otherwise
- fused `a * x + b`: 4.8-5.6x faster than NumPy's two passes with a temporary on 10M elements, 4-5.4x on transposed,
  strided or reversed inputs (results keep the input's memory order, as NumPy's do), 1.9-2.7x on small arrays
- sums: 2.5x (f32) to 3.3x (f64) faster, row sums 6.7x, column sums (contiguous or transposed) 1.4x
- mixed dtypes (`int16` matrix + `float32` row, promoted with checking): 3.6x; same-dtype `+`: 2x
- FFT vs `numpy.fft` (pocketfft): 2-2.8x faster (`rustfft` / `realfft`; any length)
- `scipy.signal`: `lfilter` 3.3x, `sosfiltfilt` 2.4x, `sosfilt` 1.7-2.1x, `welch` 3.4x, `stft` 2.1x, filter design
  ~130x (SciPy designs in Python), `remez` on par
- linear algebra (one thread): `solve` 1.5-2x, `matmul` 1.1x at 512 x 512 and on par at 64 x 64 (a register-blocked
  AVX2 kernel for small products), `eigvals` 1.1x, `eigh` 1.17x, singular values on par. faer does the reductions and products;
  the iterations are autodyne's own: dqds and Pal-Walker-Kahan for values alone (LAPACK's `dlasq1` / `dsterf`),
  divide and conquer with vectors (`dstedc` / `dbdsdc`). `bench/linalg` adds PyTorch, JAX, faer, nalgebra and Burn on
  the same inputs: fastest of all on `svd` (1.11x JAX), `eigh` (1.15x JAX) and `eigvalsh`, tied elsewhere
### Benchmarks against Rust libraries (ndarray, Burn, CubeCL)
`bench/rust` (its own workspace) compares autodyne with the `ndarray` crate, Burn 0.21 (CPU backend `flex`, and `wgpu`
on the GPU) and a hand-written CubeCL kernel on shared axes: element-wise, transposed, broadcast, reductions, FIR vs
`conv1d`; every case first checks that the libraries agree. `cargo bench --bench compare` then `python results.py`
writes `RESULTS.md` (run it on a quiet machine: single-threaded CPU timings swing with background load). Observed
on a Ryzen 9 7900 + RTX 5070 Ti:

- vs `ndarray`: on par for element-wise, transposed and broadcast work; sums 1.5-2x faster (full, row and column),
  while pairwise summation is also more accurate than its running sums
- vs Burn's CPU backend: 3-30x faster on element-wise, transposed, broadcast and full / row reductions, 1.5x on column
  sums, 2.4x on FIR vs `conv1d`
- on the GPU (feature `gpu`: `autodyne::gpu::GpuArray`, CubeCL kernels through wgpu), data already there: the fastest
  of the three on every case, element-wise on par with a hand-written CubeCL kernel (115 µs for 4M elements, Burn
  130), reductions 1.4-5x Burn's (full, row and column sums 113-126 µs), FIR 1.9x Burn's `conv1d`; a round trip
  from CPU memory (upload, compute, download) costs about what Burn's does
### flux: differentiable programs (feature `flux`)
Plain Rust, no DSL. Code is written once over two traits and runs eagerly or traced:

- `Real` (`f32` / `f64` samples, the real-time path) and `ArrayMath` / `RealArrayMath` (`NdArray`s, NumPy-style:
  element-wise maths with broadcasting, reshape, transpose, slices, padding, concatenation, reversal, sums, products,
  maxima, `dot_general`, real and complex FFTs, `take` gathers) compute directly; `signal::convolve` and
  `signal::frames` are written over them;
- on `flux::Tracer` the same code records a graph, which flux differentiates in reverse mode (`vjp`; the backward pass
  is more graph, so derivatives nest) and forward mode (`jvp`), evaluates in `f32` or `f64`, and emits as textual
  StableHLO. Spectra are complex values in the graph (`rfft_complex`, `irfft_complex`, complex products and FFTs),
  so they stay one array from one FFT to the next.

The audio processors' sample steps are generic, so biquads (every cookbook response), the SVF, the ladder, the
waveshapers, the envelope follower and the compressor trace and differentiate unchanged. A `Scan` runs a step over
whole signals, with gradients for the parameters, the initial state and the input signal, under any `Loss`: mean
squared error, the multi-resolution STFT loss usual for audio, or a traced function. The gradient saves each step's
intermediate values (or, `checkpointed(true)`, only its state, recomputing the step on the way back). `flux::optim`
has SGD and Adam.
Fitting a peaking EQ into a tanh drive to a target recording from its spectrogram alone recovers 3000 Hz / 9 dB /
drive 2 to within 1% on every backend.

Backends, all found at run time (nothing is linked at build time, and the real-time path never touches a trace):

- IREE: `iree-compile` / `iree-run-module`, for the CPU, Vulkan, CUDA, ROCm or Metal; modules can be saved (`.vmfb`)
  and loaded elsewhere. Backends state their limits and programs are written for them (`Emit::for_backend`): on
  Vulkan, long FFTs are built from 64-point ones
- PJRT: a plugin library (XLA CPU, CUDA, ...) loaded in-process through the PJRT C API, no Python; arrays can stay
  on the device between runs (`upload` / `run_resident` / `download`). The test suite passes on XLA's CUDA plugin
- XLA through JAX in a long-lived Python process, for platforms without a plugin (Windows)

Against JAX on the same XLA (`bench/flux`, [`RESULTS.md`](bench/flux/RESULTS.md)), flux's programs compile 20-40%
sooner and run at 0.96-1.24x JAX's speed: on par for a one-pole's gradient, ahead on spectral models and on an EQ
chain's STFT-loss gradient. Against other automatic differentiation ([`AUTODIFF.md`](bench/flux/AUTODIFF.md), one core), in process
(feature `jit`: scalar scans compiled with Cranelift, array graphs interpreted with fused element-wise chains): a
one-pole's gradient over 48k samples in 163 µs (135 µs with `Scan::contracted`; Enzyme 160 µs, XLA 0.48 ms, candle
0.59 s, dfdx 0.70 s, PyTorch 1.04 s, `torch.compile` fails on the recurrence), its forward pass in 53 µs (36 µs;
C 64 µs), a spectral model's gradient in 1.40 ms (PyTorch 1.73 ms eager, 1.79 ms compiled, XLA 4.8 ms), the EQ
chain's STFT-loss gradient in 0.26 ms (flux on XLA 0.28 ms, JAX 0.41 ms, Enzyme 3.5 ms). From Python,
`autodyne.flux` traces functions written with NumPy-style operators on tracers, with the same scans, losses,
processors and backends.

```sh
pip install iree-base-compiler iree-base-runtime jax   # tools on PATH, or AUTODYNE_IREE_DIR / AUTODYNE_XLA_PYTHON
export AUTODYNE_PJRT_PLUGIN=/path/to/libpjrt_cpu.so    # e.g. from github.com/zml/pjrt-artifacts (Linux, macOS)
cargo test --features flux --test flux                  # each missing backend is reported and skipped
```
### Plugins
`plugins/` turns autodyne processors into CLAP and VST3 plugins with [nice-plug](https://codeberg.org/RustAudio/nice-plug),
the community-maintained continuation of NIH-plug (ISC, with MIT-licensed VST3 bindings):

- `autodyne-plug`: `ParamBridge` exposes any `Parameterized` processor's parameters to the host (ranges, log scaling,
  defaults, display text and typed input in units, grouped by stage for chains), with no hand-written parameter struct
- `autodyne-reverb`: the FDN reverb as a stereo/mono effect
- `autodyne-synth`: a 16-voice subtractive / wavetable synth (MIDI notes on their exact sample, sustain pedal, pitch bend, MPE) into
  the FDN reverb; its parameters are the whole chain's, grouped by stage

```sh
cargo xtask bundle autodyne-reverb --release                       # target/bundled/autodyne-reverb.{clap,vst3}
cargo xtask bundle -p autodyne-synth -p autodyne-reverb --release  # several at once
```

Copy the `.clap` into your CLAP folder (`%COMMONPROGRAMFILES%\CLAP` on Windows, `~/.clap` on Linux,
`~/Library/Audio/Plug-Ins/CLAP` on macOS) or the `.vst3` into your VST3 folder (`%COMMONPROGRAMFILES%\VST3`,
`~/.vst3`, `~/Library/Audio/Plug-Ins/VST3`). Both pass their format's validator (clap-validator, Steinberg's
VST3 validator).

### Tests and benchmarks
- `cargo test`: every processor is checked against a known answer (closed-form signals, cookbook frequency responses, FFT vs DFT, modulation round trips), and `tests/no_alloc.rs` proves processing never allocates
- `cargo bench`: throughput per 512-sample block. The SIMD pass made FIR filtering 4.5-13x faster
  (more taps, bigger win) and resampling 6-11x faster than the scalar versions.

## License
autodyne is licensed under the [GNU General Public License v3.0 only](LICENSE) (`GPL-3.0-only`): you may use, study,
change and share it, and software you distribute that is built on it must also be released under the GPLv3, with
source.

Commercial licenses for closed-source use are available from the author: open an issue or contact
[@vaibhav-gopal](https://github.com/vaibhav-gopal).

Every third-party dependency is permissively licensed (MIT, Apache-2.0, ISC, BSD, Zlib, ...), so products built on
autodyne carry no other copyleft obligations. Those licenses do require shipping each dependency's copyright and
license text, which is automated:

- **Policy:** [cargo-deny](deny.toml) runs on every push and pull request: a dependency under any other license, a
  known vulnerability (RustSec) or an unexpected source fails CI.
- **Notices:** each plugin has a `THIRD-PARTY-LICENSES.html` generated from its own dependency graph by
  [cargo-about](about.toml) (`cargo xtask notices`). The Notices workflow regenerates and commits them whenever
  dependencies change on main, and `cargo xtask bundle` ships them next to and inside every bundle.
- **Updates:** Dependabot opens weekly dependency update pull requests, which go through the same checks.

## Contributing
Contributions are welcome. Pull requests need a one-time signature of the [Contributor License Agreement](CLA.md);
see [CONTRIBUTING.md](CONTRIBUTING.md).