# DSP Library

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

## Units
- defines traits that represent real number operations
- defines a complex unit type
- also defines a new type that works on fixed point arithmetic (todo)

## Buffers
- defines an extension trait that represents operations on fixed-size signals/data (buffers)
- iterator adaptors define lazy operations on buffers