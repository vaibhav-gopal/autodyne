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

| module | contents |
|---|---|
| `units` | number traits (`Float`, `Integer`, `Trig`, casts, ...), `Complex<T>`, and reflection: `DType` (runtime element type) and `Reflection` |
| `osc` | band-limited `Oscillator` (saw / pulse with PolyBLEP, triangle, sine), `Sine`, `Phasor` (complex oscillator), `Noise` (seeded), `Impulse`, and `WavetableOsc` (mip-mapped `Wavetable`s built from single-cycle frames or harmonics, alias-free to below -80 dB, smooth morphing across frames) |
| `filter` | `convolve`, `Fir` + windowed-sinc `design_lowpass`, `Biquad` (low/high/band-pass, notch, all-pass, peaking, low/high shelf), `MultiBiquad` (one per channel, 4 channels in lockstep: ~3.8x faster on 8 channels), `ParametricEq` (up to 8 bands: bells, shelves, 12-48 dB/oct cuts, notch, band-pass; its curve for drawing), `LinkwitzRiley` and `Crossover` (2-4 bands that sum flat), and two filters built for modulation: `Svf` (zero-delay-feedback state-variable filter, stable under per-sample cutoff changes, six simultaneous responses) and `Ladder` (4-pole Moog-style, zero-delay feedback, self-oscillates at the cutoff, level-compensated drive); `magnitude_at` for analytic responses |
| `spectral` | `Fft` (forward, inverse, real input) and `RealFft` (real signals) on `rustfft` / `realfft` kernels (AVX / SSE / NEON, default feature `rustfft`; a portable radix-2 otherwise), planned with their scratch so transforms never allocate, a reference `dft`, and a phase vocoder with identity phase locking: `PitchShifter` (real time, fixed latency) and `time_stretch` (offline, any factor 0.1-10) |
| `iq` | `IqModulator` / `IqDemodulator`, `envelope` (AM), `phase` (PM), `FmModulator` / `FmDiscriminator` |
| `gain` | `db_to_gain` / `gain_to_db`, `SmoothedValue` (click-free parameter ramps), smoothed `Gain` |
| `delay` | `DelayLine` (integer and interpolated reads), `Echo` (feedback delay with smoothed parameters) |
| `synth` | `MidiMessage` parsing (including aftertouch), the `Voice` trait, `Poly` (voice allocation and stealing, sustain pedal, pitch bend, sample-accurate events, poly / mono / legato modes with last-note priority, MPE and per-note expression; mono or stereo rendering) and `SynthVoice` (up to 7 detuned unison oscillators -> SVF or ladder filter swept by its envelope, pressure and timbre -> ADSR, with glide; classic waveforms or a morphing wavetable) and `FmVoice` (4-operator phase modulation: 8 algorithms, ratios, detune, feedback, an envelope per operator, timbre and pressure on modulation depth) |
| `sampler` | `Sample` (mono / stereo, root key, forward / ping-pong / sustain loops with crossfades, optional mipmaps for notes far above the root; loads WAV with its loop metadata), `Zone` / `SampleMap` (key and velocity ranges, tuning, gain, pan, round robin) and `SamplerVoice` (64-tap Kaiser-sinc, cubic or linear interpolation, band-limited when pitched up, amplitude envelope, stereo) for `Poly` |
| `reverb` | `Reverb` (8-line feedback delay network: exact RT60, modulated delays with all-pass interpolation, damping, pre-delay, diffusion, decorrelated stereo) and `Convolver` (partitioned FFT convolution with any impulse response; `synthetic_ir`) |
| `envelope` | `Adsr`: exact linear segments, click-free retrigger / release; a VCA `Processor` and a modulation `Source` |
| `distortion` | `Waveshaper` (tanh, soft clip, hard clip, fold) with smoothed drive / output / mix; `Bitcrusher` (fractional bit depth, sample-and-hold rate reduction, optional dither) |
| `dynamics` | `EnvelopeFollower`, `Compressor` (soft knee, attack/release, makeup), `LookaheadLimiter` (linked brickwall with a smooth lookahead ramp, true-peak detection, never over the ceiling), `Gate` (gate and downward expander: range, hysteresis, hold), `TransientShaper` (level-independent attack / sustain), `MultibandCompressor` (2-4 Linkwitz-Riley bands, linked); `Linked` runs any of them across channels |
| `modulation` | `ModulatedDelay` (`chorus` / `flanger` presets), `Phaser` (swept all-pass stages) |
| `analysis` | `SpectrumAnalyzer` (sliding FFT with analyzer ballistics, peak trace, log-spaced bands for drawing; Hann / Blackman-Harris windows), `LoudnessMeter` (ITU-R BS.1770 / EBU R128: momentary, short-term and gated integrated LUFS, loudness range, true peak; fixed memory for any duration), `TruePeak` (oversampled inter-sample peaks), `PitchDetector` (YIN via FFT correlation), `OnsetDetector` (SuperFlux-style spectral flux with adaptive peak picking) |
| `control` | `Lfo` (sine, triangle, saws, square, sample & hold, smooth random; free-running or locked to the host's beat grid; fade-in, retrigger), `Modulated` (a modulation matrix around any processor: routes from sources to any parameter, depths in normalized units with "via" scaling, automatable depths), `Transport` and `Division` (tempo, beat position, dotted and triplet note lengths) |
| `processor` | the `Processor` trait all effects share; tuples are zero-cost chains, `Vec<Box<dyn Processor>>` is a runtime chain |
| `channels` | planar `AudioBuffer` (+ interleave conversion, `[channel, time]` n-d view), `MultiProcessor`, `PerChannel`, `Linked(compressor)`, `StereoWidth`, `Panner` |
| `resample` | streaming rational `Resampler` (polyphase, e.g. 48 kHz <-> 44.1 kHz) and `Oversampled<P>` (runs any processor at 2x / 4x / 8x; 4x cuts tanh saturation aliasing by ~39 dB) |
| `simd` | vectorized `dot` kernel on stable Rust; AVX2 chosen at runtime on x86-64 (used by `Fir` and `Resampler`) |
| `signal` | the core abstraction. Any sample slice, `Vec`, array or `NdArray` is a signal: `Signal` (levels, norms, statistics, argmax, inner/angle/distance, convolved/correlated/resampled), `SignalMut` (gain, normalize, fades, cumsum/diff, projection, pointwise math with `Broadcast` policies), `ComplexSignal`. Capability tiers `SignalOwned` / `SignalResizable` unlock `SigOwnedOps` / `SigResizeOps`. `Source`s compose lazily (`scaled`, `mix`, `through`). Streams: `SignalRead` / `SignalWrite` / `SignalSeek` with `SampleReader` / `SampleWriter` at the byte boundary. `NdArray` (over `Vec`, `Box<[T]>`, shared copy-on-write `Arc<[T]>` or foreign storage) + zero-copy `NdView` / `NdViewMut` (O(1) slicing, steps and reversal, permute / transpose, broadcasting, reshape without copying when the layout allows it and `to_shape` when it does not; disjoint mutable lanes and sub-views; layouts checked once, mutable ones proven injective) whose lanes are signals, with axis labels; `Zip` (fused element-wise passes over up to 5 views of any layout, NumPy broadcasting, as fast as a hand-written slice loop), arithmetic operators, reductions (`sum` pairwise and vectorized, `mean`, `rms`, `min` / `max`, `sum_axis`, and `fold_into` / `sum_into`: reducing into any shape that broadcasts to the input), and allocation-free `process_lanes` along strided axes |
| `wav` | dependency-free WAV `read` / `write`: 8-32-bit PCM and 32/64-bit float, `WAVE_FORMAT_EXTENSIBLE`, sampler metadata (`smpl` root key and loops) |
| `params` | `ParamInfo` (range, unit, scale, kind: continuous / integer / toggle / choice, smoothing, normalized 0..1, formatting and parsing) and `Parameterized` for every processor, tuple / `Vec` chains and linked `PerChannel`; `Smoothed` makes any processor's automation click-free (control-rate ramps for coefficients, per-sample smoothing for gains), and `ParamEvent` applies changes at exact sample offsets |
| `dlpack` | DLPack 1.x (and legacy) export and import: arrays go to NumPy / PyTorch / JAX / CuPy without copying (shared arrays read-only), foreign tensors of any strides come in as views or, when contiguous, as `NdArray`s over the foreign memory, released exactly once |
| `interop` | zero-copy conversions with the `ndarray` crate (feature `ndarray`): views both ways for any strides, owned arrays move their `Vec`; Arrow buffers are slices already |
| `prelude` | `use autodyne::prelude::*` brings every trait and the common types into scope |
| `dynamic` | runtime-typed data and processing: `DynArray` (any dtype, bytes), zero-copy `DynView` over external memory, runtime-typed math (`binary` with broadcasting, scalars with NumPy's weak typing, `unary` functions, `sum` / `mean` / `min` / `max`, `cast` checked / saturating / wrapping) under explicit `Promotion` rules: NumPy's table with every conversion checked, so nothing is lost silently (or `KeepFloat`, floats keeping their width); `DynProcessor` built at runtime with `build_dyn` |

```rust
use autodyne::prelude::*;
use autodyne::filter::{Biquad, BUTTERWORTH_Q};
use autodyne::osc::Sine;

let mut block = [0.0f32; 512];
let mut tone = Sine::new(440.0, 48_000.0);
let mut lowpass = Biquad::lowpass(1_000.0, 48_000.0, BUTTERWORTH_Q as f32);

tone.fill(&mut block);       // generate
lowpass.process(&mut block); // filter in place
let level = block.rms_db();  // analyse: every slice is a signal
```

### Examples
`cargo run --release --example live [seconds] [--demo]` is a playable 8-voice synth: plug in a MIDI keyboard (or it
plays a built-in chord sequence), through oversampled saturation and a stereo chain (chorus, linked compressor, echo,
width, limiter), rendered live in the audio callback. MIDI reaches the audio thread through a lock-free ring buffer.

The others write WAV files to `target/examples-out/`:

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
`bindings/python/bench/compare.py` runs the same work in NumPy / SciPy and autodyne on the same arrays (checking the
results agree), single-threaded, end to end: autodyne's times include crossing into Rust and back through DLPack. Latest
results are in [`bindings/python/bench/RESULTS.md`](bindings/python/bench/RESULTS.md). On a Ryzen 9 7900:

- fused `a * x + b`: ~2x faster than NumPy's two passes with a temporary, on contiguous, transposed, strided or
  reversed inputs (results keep the input's memory order, as NumPy's do); 1.6-2.4x on small arrays (64-1,000 elements),
  where NumPy arrays cross through NumPy's C API rather than a Python-level DLPack call
- sums: 1.4x (f32) to 3.4x (f64) faster, row sums 6.5x; column sums on par (both at memory bandwidth)
- mixed dtypes (`int16` matrix + `float32` row, promoted with checking): 1.45x faster
- IIR filtering vs `scipy.signal.sosfilt`: 1.27x faster on contiguous lanes, on par on strided ones
- FFT vs `numpy.fft` (pocketfft): 1.3-1.5x faster, 2x at prime lengths (`rustfft` / `realfft` kernels; any length)
- `scipy.signal`: `sosfilt` 1.1x, `sosfiltfilt` 1.5x, `lfilter` on par (lanes filtered four at a time), `welch` 3x,
  `stft` 1.1x, filter design ~100x (SciPy designs in Python); `remez` 0.4x
- linear algebra (faer vs OpenBLAS, one thread): `solve` 1.2-1.6x, `eigvals` 1.1x; `matmul` 0.6-0.9x, `eigh` 0.8x, `svd`
  0.6x
### Benchmarks against Rust libraries (ndarray, Burn, CubeCL)
`bench/rust` (its own workspace) compares autodyne with the `ndarray` crate, Burn 0.21 (CPU backend `flex`, and `wgpu`
on the GPU) and a hand-written CubeCL kernel on shared axes: element-wise, transposed, broadcast, reductions, FIR vs
`conv1d`; every case first checks that the libraries agree. `cargo bench --bench compare` then `python results.py`
writes `RESULTS.md` (run it on a quiet machine: single-threaded CPU timings swing with background load). Observed
on a Ryzen 9 7900 + RTX 5070 Ti:

- vs `ndarray`: on par for element-wise, transposed and broadcast work; reductions currently ~5-15% behind (pairwise
  summation is more accurate than its running sums)
- vs Burn's CPU backend: 2-7x faster on element-wise, transposed, broadcast and full / row reductions; on par for
  column sums and FIR vs `conv1d`
- GPU (Burn wgpu, CubeCL, data already on the GPU): ~10-15x faster than one CPU core on element-wise work, slower on
  these reductions; uploading and downloading from CPU memory costs ~5x the CPU computation, so the GPU pays off
  only for data that lives there (batches, fitting)
### flux: differentiable programs (feature `flux`)
Plain Rust, no DSL. Code is written once over two traits and runs eagerly or traced:

- `Real` (`f32` / `f64` samples, the real-time path) and `ArrayMath` (`NdArray`s, NumPy-style: element-wise maths
  with broadcasting, reshape, transpose, sums, `dot_general`, real FFTs) compute directly;
- on `flux::Tracer` the same code records a graph of primitives (comparisons become `select`), which flux
  differentiates in reverse mode (the backward pass is just more graph, so derivatives compose and nest), runs over
  whole signals as a `Scan` with array-valued parameters, state and samples (gradient = a reverse scan), and emits as
  textual StableHLO.

Three backends run the programs, all found at run time (nothing is linked at build time, and the real-time path never
touches a trace):

- IREE: `iree-compile` / `iree-run-module`
- PJRT: a plugin library (XLA CPU / CUDA / ROCm, ...) loaded in-process through the PJRT C API, no Python
- XLA through JAX's client in a long-lived Python process, for platforms without a plugin (Windows)

Every primitive and its gradient agree with flux's reference interpreter on all three; one generic model gives the same
result run on `NdArray`s and compiled; a traced `OnePole` matches the f32 filter, its gradient matches finite
differences, and gradient descent fits its cutoff (300 Hz -> 1200 Hz in 30 steps) and recovers per-bin gains through
`rfft` / `irfft` frame by frame.

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