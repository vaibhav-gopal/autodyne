//! Generates two seconds of a 440 Hz tone, clean and with added noise, and writes both as WAV.
//!
//!     cargo run --example tone               # writes to target/examples-out/
//!     cargo run --example tone -- <out_dir>
//!
//! Samples are produced block by block, the way an audio callback would request them.

use std::path::Path;
use autodyne::osc::{Noise, Sine};

const SAMPLE_RATE: u32 = 48_000;
const BLOCK_SIZE: usize = 512;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let out_dir = std::env::args().nth(1).unwrap_or_else(|| "target/examples-out".into());
    std::fs::create_dir_all(&out_dir)?;

    let mut tone = Sine::new(440.0, SAMPLE_RATE as f32).with_amplitude(0.5);
    let mut noise = Noise::new(1).with_amplitude(0.05);

    let mut clean = vec![0.0f32; 2 * SAMPLE_RATE as usize];
    let mut noisy = clean.clone();
    for (clean_block, noisy_block) in clean.chunks_mut(BLOCK_SIZE).zip(noisy.chunks_mut(BLOCK_SIZE)) {
        tone.fill(clean_block);
        noisy_block.copy_from_slice(clean_block);
        noise.add_to(noisy_block);
    }

    write_wav(Path::new(&out_dir).join("tone_clean.wav"), &clean)?;
    write_wav(Path::new(&out_dir).join("tone_noisy.wav"), &noisy)?;
    println!("wrote tone_clean.wav and tone_noisy.wav to {out_dir}");
    Ok(())
}

fn write_wav(path: impl AsRef<Path>, samples: &[f32]) -> Result<(), hound::Error> {
    let spec = hound::WavSpec {
        channels: 1,
        sample_rate: SAMPLE_RATE,
        bits_per_sample: 32,
        sample_format: hound::SampleFormat::Float,
    };
    let mut writer = hound::WavWriter::create(path, spec)?;
    for &s in samples {
        writer.write_sample(s)?;
    }
    writer.finalize()
}
