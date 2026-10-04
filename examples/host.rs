//! A host driving processors it only knows at runtime, the way a DAW, game engine or scripting layer
//! would: the sample type comes from the command line, parameters are discovered and set by id.
//!
//!     cargo run --release --example host                                  # f32, defaults
//!     cargo run --release --example host -- f64 threshold_db=-30 mix=0.2  # f64, two settings
//!
//! Steps: pick a DType -> build a chain with `build_dyn` -> list its parameters (grouped by stage)
//! -> apply `id=value` settings -> render a test signal in that dtype, process it, report levels ->
//! print a preset snapshot.

use autodyne::delay::Echo;
use autodyne::dynamic::{build_dyn, DynArray, FloatElement, ProcessorFactory};
use autodyne::dynamics::Compressor;
use autodyne::filter::{Biquad, BUTTERWORTH_Q};
use autodyne::gain::Gain;
use autodyne::osc::{Noise, Sine};
use autodyne::params::Parameterized;
use autodyne::signal::{NdArray, Signal, Source};
use autodyne::units::DType;

const SAMPLE_RATE: f64 = 48_000.0;

/// The chain this host offers: low shelf -> compressor -> echo -> output gain.
struct VoiceChain;

impl ProcessorFactory for VoiceChain {
    type Output<T: FloatElement> = (Biquad<T>, Compressor<T>, Echo<T>, Gain<T>);
    fn build<T: FloatElement>(&self, fs: T) -> Self::Output<T> {
        let mut echo = Echo::new(T::_lit(1.0), fs);
        echo.set_immediate(T::_lit(0.25), T::_lit(0.35), T::_lit(0.3));
        (
            Biquad::low_shelf(T::_lit(200.0), T::_lit(BUTTERWORTH_Q), T::_lit(3.0), fs),
            Compressor::new(fs),
            echo,
            Gain::new(T::_ONE, T::_lit(0.02), fs),
        )
    }
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let mut args = std::env::args().skip(1).peekable();
    let dtype = match args.peek().and_then(|a| DType::from_name(a)) {
        Some(d) => {
            args.next();
            d
        }
        None => DType::F32,
    };

    let mut chain = build_dyn(&VoiceChain, dtype, SAMPLE_RATE)?;
    println!("chain built for {} samples at {SAMPLE_RATE} Hz", chain.dtype());

    for setting in args {
        let (id, value) = setting.split_once('=').ok_or_else(|| format!("expected id=value, got {setting:?}"))?;
        let applied = chain.set_param_by_id(id, value.parse()?)?;
        println!("set {id} = {applied}");
    }

    println!("\n{:<18} {:<14} {:>12}   range", "stage", "parameter", "value");
    for i in 0..chain.param_count() {
        let info = chain.param_info(i).unwrap();
        let value = chain.get_param(i).unwrap();
        println!(
            "{:<18} {:<14} {:>12}   {} .. {}",
            chain.param_group(i).unwrap_or("-"),
            info.id,
            info.format(value),
            info.format(info.min),
            info.format(info.max),
        );
    }

    // One second of a loud-ish tone with noise, generated in f64 and converted to the host's dtype.
    let mut signal = vec![0.0f64; SAMPLE_RATE as usize];
    Sine::new(220.0, SAMPLE_RATE).with_amplitude(0.8).mix(Noise::new(1).scaled(0.05)).fill(&mut signal);
    let before = signal.rms_db().unwrap();
    let mut buffer = DynArray::from_array(NdArray::from_vec(signal, &[SAMPLE_RATE as usize])?).cast(dtype)?;

    chain.process_dyn(buffer.as_block_mut()?)?;

    let processed = buffer.cast(DType::F64)?.into_array::<f64>().map_err(|_| "cast produced f64")?;
    println!(
        "\nprocessed {} {} samples: rms {before:.1} dB -> {:.1} dB, peak {:.1} dB",
        processed.len(),
        dtype,
        processed.rms_db().unwrap(),
        processed.peak_db(),
    );
    println!("preset snapshot: {:?}", chain.snapshot().iter().map(|v| (v * 1000.0).round() / 1000.0).collect::<Vec<_>>());
    Ok(())
}
