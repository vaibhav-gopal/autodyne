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
| `filter` | `convolve`, `Fir` + windowed-sinc `design_lowpass`, `Biquad` (low/high/band-pass, notch, all-pass, peaking, low/high shelf), `MultiBiquad` (one per channel, 4 channels in lockstep: ~3.8x faster on 8 channels), and two filters built for modulation: `Svf` (zero-delay-feedback state-variable filter, stable under per-sample cutoff changes, six simultaneous responses) and `Ladder` (4-pole Moog-style, zero-delay feedback, self-oscillates at the cutoff, level-compensated drive); `magnitude_at` for analytic responses |
| `spectral` | radix-2 `Fft` (forward, inverse, real input), `RealFft` (real signals, ~1.7x faster) and a reference `dft` |
| `iq` | `IqModulator` / `IqDemodulator`, `envelope` (AM), `phase` (PM), `FmModulator` / `FmDiscriminator` |
| `gain` | `db_to_gain` / `gain_to_db`, `SmoothedValue` (click-free parameter ramps), smoothed `Gain` |
| `delay` | `DelayLine` (integer and interpolated reads), `Echo` (feedback delay with smoothed parameters) |
| `synth` | `MidiMessage` parsing (including aftertouch), the `Voice` trait, `Poly` (voice allocation and stealing, sustain pedal, pitch bend, sample-accurate events, poly / mono / legato modes with last-note priority, MPE and per-note expression) and `SynthVoice` (up to 7 detuned unison oscillators -> SVF or ladder filter swept by its envelope, pressure and timbre -> ADSR, with glide; classic waveforms or a morphing wavetable) and `FmVoice` (4-operator phase modulation: 8 algorithms, ratios, detune, feedback, an envelope per operator, timbre and pressure on modulation depth) |
| `reverb` | `Reverb` (8-line feedback delay network: exact RT60, modulated delays with all-pass interpolation, damping, pre-delay, diffusion, decorrelated stereo) and `Convolver` (partitioned FFT convolution with any impulse response; `synthetic_ir`) |
| `envelope` | `Adsr`: exact linear segments, click-free retrigger / release; a VCA `Processor` and a modulation `Source` |
| `distortion` | `Waveshaper` (tanh, soft clip, hard clip, fold) with smoothed drive / output / mix |
| `dynamics` | `EnvelopeFollower`, `Compressor` (soft knee, attack/release, makeup), `Compressor::limiter` |
| `modulation` | `ModulatedDelay` (`chorus` / `flanger` presets), `Phaser` (swept all-pass stages) |
| `control` | `Lfo` (sine, triangle, saws, square, sample & hold, smooth random; free-running or locked to the host's beat grid; fade-in, retrigger), `Modulated` (a modulation matrix around any processor: routes from sources to any parameter, depths in normalized units with "via" scaling, automatable depths), `Transport` and `Division` (tempo, beat position, dotted and triplet note lengths) |
| `processor` | the `Processor` trait all effects share; tuples are zero-cost chains, `Vec<Box<dyn Processor>>` is a runtime chain |
| `channels` | planar `AudioBuffer` (+ interleave conversion, `[channel, time]` n-d view), `MultiProcessor`, `PerChannel`, `Linked(compressor)`, `StereoWidth`, `Panner` |
| `resample` | streaming rational `Resampler` (polyphase, e.g. 48 kHz <-> 44.1 kHz) and `Oversampled<P>` (runs any processor at 2x / 4x / 8x; 4x cuts tanh saturation aliasing by ~39 dB) |
| `simd` | vectorized `dot` kernel on stable Rust; AVX2 chosen at runtime on x86-64 (used by `Fir` and `Resampler`) |
| `signal` | the core abstraction. Any sample slice, `Vec`, array or `NdArray` is a signal: `Signal` (levels, norms, statistics, argmax, inner/angle/distance, convolved/correlated/resampled), `SignalMut` (gain, normalize, fades, cumsum/diff, projection, pointwise math with `Broadcast` policies), `ComplexSignal`. Capability tiers `SignalOwned` / `SignalResizable` unlock `SigOwnedOps` / `SigResizeOps`. `Source`s compose lazily (`scaled`, `mix`, `through`). Streams: `SignalRead` / `SignalWrite` / `SignalSeek` with `SampleReader` / `SampleWriter` at the byte boundary. `NdArray` + zero-copy `NdView`s whose lanes are signals, with axis labels |
| `params` | `ParamInfo` (range, unit, scale, kind: continuous / integer / toggle / choice, smoothing, normalized 0..1, formatting and parsing) and `Parameterized` for every processor, tuple / `Vec` chains and linked `PerChannel`; `Smoothed` makes any processor's automation click-free (control-rate ramps for coefficients, per-sample smoothing for gains), and `ParamEvent` applies changes at exact sample offsets |
| `prelude` | `use autodyne::prelude::*` brings every trait and the common types into scope |
| `dynamic` | runtime-typed data and processing: `DynArray` (any dtype, casts, bytes), zero-copy `DynView` over external memory, `DynProcessor` built at runtime with `build_dyn` |

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
- `cargo run --release --example host -- [f32|f64] [id=value ...]`: a host picking the sample type at runtime, listing and setting parameters by id, processing, and printing a preset
- `cargo run --release --example interop`: a `[batch, channel, time]` tensor filtered along time, shared as raw memory without copying, and streamed through 16-bit PCM

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

## Not here
The `flux` IR/compiler experiment (a JAX/XLA-style tracer and compiler) lives on the `flux` branch.

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