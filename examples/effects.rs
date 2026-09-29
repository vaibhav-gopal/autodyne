//! A small effects chain: a melody -> EQ (low shelf + presence bell) -> compressor -> echo -> limiter,
//! rendered at 48 kHz and then resampled to 44.1 kHz. Writes both versions as WAV.
//!
//!     cargo run --release --example effects               # writes to target/examples-out/
//!     cargo run --release --example effects -- <out_dir>
//!
//! The whole chain is one tuple of processors (see `autodyne::processor`); notes are gated by the
//! `Gain` at its front so each note fades in and out instead of clicking.

use std::path::Path;
use autodyne::delay::Echo;
use autodyne::dynamics::Compressor;
use autodyne::filter::{Biquad, BUTTERWORTH_Q};
use autodyne::gain::Gain;
use autodyne::osc::Sine;
use autodyne::processor::Processor;
use autodyne::resample::Resampler;

const SAMPLE_RATE: u32 = 48_000;
const OUTPUT_RATE: u32 = 44_100;
const BLOCK_SIZE: usize = 480; // 10 ms

/// (frequency Hz, length in blocks) — a short C major arpeggio, then a rest for the echoes to ring out
const MELODY: [(f32, usize); 6] = [(261.63, 25), (329.63, 25), (392.00, 25), (523.25, 50), (0.0, 100), (0.0, 50)];

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let out_dir = std::env::args().nth(1).unwrap_or_else(|| "target/examples-out".into());
    std::fs::create_dir_all(&out_dir)?;
    let fs = SAMPLE_RATE as f32;

    let mut osc = Sine::new(440.0, fs).with_amplitude(0.4);
    let mut compressor = Compressor::new(fs);
    compressor.set_threshold_db(-20.0);
    compressor.set_makeup_db(4.0);
    let mut echo = Echo::new(1.0, fs);
    echo.set_immediate(0.3, 0.45, 0.35);
    let mut chain = (
        Gain::new(0.0, 0.01, fs), // note gate, 10 ms fades
        Biquad::low_shelf(200.0, fs, BUTTERWORTH_Q as f32, 4.0),
        Biquad::peaking(2_500.0, fs, 1.0, 3.0),
        compressor,
        echo,
        Compressor::limiter(-1.0, 0.05, fs), // keep the result under -1 dBFS
    );

    let mut rendered = Vec::new();
    let mut block = [0.0f32; BLOCK_SIZE];
    for &(freq, blocks) in &MELODY {
        if freq > 0.0 {
            osc.set_frequency(freq, fs);
            chain.0.set_gain(1.0);
        }
        for b in 0..blocks {
            if b + 5 == blocks {
                chain.0.set_gain(0.0); // start the release 50 ms before the note ends
            }
            osc.fill(&mut block);
            chain.process(&mut block);
            rendered.extend_from_slice(&block);
        }
    }

    // Stream the render through the resampler block by block, as a real-time converter would.
    let mut resampler = Resampler::new(SAMPLE_RATE, OUTPUT_RATE);
    let mut converted = Vec::with_capacity(resampler.max_output_len(rendered.len()));
    let mut out = vec![0.0f32; resampler.max_output_len(BLOCK_SIZE)];
    for chunk in rendered.chunks(BLOCK_SIZE) {
        let n = resampler.process(chunk, &mut out);
        converted.extend_from_slice(&out[..n]);
    }
    let (l, m) = resampler.ratio();
    println!(
        "rendered {} samples at {SAMPLE_RATE} Hz -> {} samples at {OUTPUT_RATE} Hz (ratio {l}/{m}, filter delay {:.1} samples)",
        rendered.len(),
        converted.len(),
        resampler.delay(),
    );

    write_wav(Path::new(&out_dir).join("effects_48k.wav"), &rendered, SAMPLE_RATE)?;
    write_wav(Path::new(&out_dir).join("effects_44k1.wav"), &converted, OUTPUT_RATE)?;
    println!("wrote effects_48k.wav and effects_44k1.wav to {out_dir}");
    Ok(())
}

fn write_wav(path: impl AsRef<Path>, samples: &[f32], sample_rate: u32) -> Result<(), hound::Error> {
    let spec = hound::WavSpec { channels: 1, sample_rate, bits_per_sample: 32, sample_format: hound::SampleFormat::Float };
    let mut writer = hound::WavWriter::create(path, spec)?;
    for &s in samples {
        writer.write_sample(s)?;
    }
    writer.finalize()
}
