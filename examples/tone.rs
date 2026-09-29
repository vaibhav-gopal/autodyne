//! Generates two seconds of a 440 Hz tone, adds noise, then low-passes the noisy version.
//! Writes all three as WAV and reports the noise reduction and the FFT peak.
//!
//!     cargo run --example tone               # writes to target/examples-out/
//!     cargo run --example tone -- <out_dir>
//!
//! Samples are produced and filtered block by block, the way an audio callback would request them.

use std::path::Path;
use autodyne::filter::{Biquad, BUTTERWORTH_Q};
use autodyne::osc::{Noise, Sine};
use autodyne::signal::{ComplexSignal, Signal};
use autodyne::spectral::{bin_frequency, Fft};
use autodyne::units::Complex;

const SAMPLE_RATE: u32 = 48_000;
const BLOCK_SIZE: usize = 512;
const NOISE_SEED: u64 = 1;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let out_dir = std::env::args().nth(1).unwrap_or_else(|| "target/examples-out".into());
    std::fs::create_dir_all(&out_dir)?;
    let fs = SAMPLE_RATE as f32;

    let mut tone = Sine::new(440.0, fs).with_amplitude(0.5);
    let mut noise = Noise::new(NOISE_SEED).with_amplitude(0.05);
    let mut lowpass = Biquad::lowpass(1_000.0, fs, BUTTERWORTH_Q as f32);

    let len = 2 * SAMPLE_RATE as usize;
    let (mut clean, mut noisy, mut filtered) = (vec![0.0f32; len], vec![0.0f32; len], vec![0.0f32; len]);
    for ((c, n), f) in clean.chunks_mut(BLOCK_SIZE).zip(noisy.chunks_mut(BLOCK_SIZE)).zip(filtered.chunks_mut(BLOCK_SIZE)) {
        tone.fill(c);
        n.copy_from_slice(c);
        noise.add_to(n);
        f.copy_from_slice(n);
        lowpass.process(f);
    }

    // The filter is linear, so the noise left in `filtered` is exactly the filtered noise on its own.
    let mut noise_only = vec![0.0f32; len];
    Noise::new(NOISE_SEED).with_amplitude(0.05).fill(&mut noise_only);
    let noise_before = noise_only.rms().unwrap();
    Biquad::lowpass(1_000.0, fs, BUTTERWORTH_Q as f32).process(&mut noise_only);
    let noise_after = noise_only.rms().unwrap();
    println!(
        "noise rms {noise_before:.4} -> {noise_after:.4} ({:.1} dB less noise; the 440 Hz tone keeps {:.1}% of its level)",
        20.0 * (noise_before / noise_after).log10(),
        100.0 * lowpass.magnitude_at(440.0, fs),
    );

    // Where is the energy? Transform a 4096-sample window (after the filter settles) and find the peak.
    let fft = Fft::new(4096);
    let mut spectrum = vec![Complex::zero(); fft.len()];
    fft.forward_real(&filtered[SAMPLE_RATE as usize..][..fft.len()], &mut spectrum);
    let peak = spectrum[..fft.len() / 2].argmax_magnitude().unwrap(); // positive frequencies only
    println!(
        "FFT peak: {:.1} Hz (bins are {:.1} Hz wide)",
        bin_frequency(peak, fft.len(), fs),
        bin_frequency(1, fft.len(), fs),
    );

    for (name, data) in [("tone_clean", &clean), ("tone_noisy", &noisy), ("tone_filtered", &filtered)] {
        write_wav(Path::new(&out_dir).join(format!("{name}.wav")), data)?;
    }
    println!("wrote tone_clean.wav, tone_noisy.wav and tone_filtered.wav to {out_dir}");
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
