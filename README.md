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
| `units` | number traits (`Float`, `Integer`, `Trig`, casts, ...) and `Complex<T>` |
| `osc` | `Sine`, `Phasor` (complex oscillator), `Noise` (seeded), `Impulse`; each is also an infinite iterator |
| `filter` | `convolve`, `Fir` + windowed-sinc `design_lowpass`, `Biquad` (low/high/band-pass, notch, all-pass, peaking, low/high shelf), `magnitude_at` for analytic responses |
| `spectral` | radix-2 `Fft` (forward, inverse, real input) and a reference `dft` |
| `iq` | `IqModulator` / `IqDemodulator`, `envelope` (AM), `phase` (PM), `FmModulator` / `FmDiscriminator` |
| `gain` | `db_to_gain` / `gain_to_db`, `SmoothedValue` (click-free parameter ramps), smoothed `Gain` |
| `delay` | `DelayLine` (integer and interpolated reads), `Echo` (feedback delay with smoothed parameters) |
| `dynamics` | `EnvelopeFollower`, `Compressor` (soft knee, attack/release, makeup), `Compressor::limiter` |
| `processor` | the `Processor` trait all effects share; tuples are zero-cost chains, `Vec<Box<dyn Processor>>` is a runtime chain |
| `resample` | streaming rational `Resampler` (polyphase, e.g. 48 kHz <-> 44.1 kHz) |
| `simd` | vectorized `dot` kernel on stable Rust; AVX2 chosen at runtime on x86-64 (used by `Fir` and `Resampler`) |
| `signal` | the planned `Signal` trait hierarchy (design only, not implemented yet) |

```rust
use autodyne::filter::{Biquad, BUTTERWORTH_Q};
use autodyne::osc::Sine;

let mut block = [0.0f32; 512];
let mut tone = Sine::new(440.0, 48_000.0);
let mut lowpass = Biquad::lowpass(1_000.0, 48_000.0, BUTTERWORTH_Q as f32);

tone.fill(&mut block);      // generate
lowpass.process(&mut block); // filter in place
```

### Examples
Both write WAV files to `target/examples-out/`:

- `cargo run --release --example tone`: a 440 Hz tone, the same tone with noise, and the noisy one low-passed; prints the noise reduction and the FFT peak
- `cargo run --release --example fm_radio`: an FM radio link (modulate, 12 kHz carrier, noisy channel, demodulate); prints the audio SNR
- `cargo run --release --example effects`: a melody through a processor chain (EQ, compressor, echo, limiter), rendered at 48 kHz and resampled to 44.1 kHz

### Tests and benchmarks
- `cargo test`: every processor is checked against a known answer (closed-form signals, cookbook frequency responses, FFT vs DFT, modulation round trips)
- `cargo bench`: throughput per 512-sample block. The SIMD pass made FIR filtering 4.5-13x faster
  (more taps, bigger win) and resampling 6-11x faster than the scalar versions.

## Not here
The `flux` IR/compiler experiment (a JAX/XLA-style tracer and compiler) lives on the `flux` branch.
