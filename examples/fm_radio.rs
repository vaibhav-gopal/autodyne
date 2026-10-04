//! A software FM radio link: an audio message is FM-modulated onto a 12 kHz carrier, noise is added
//! in the "air", and a receiver demodulates it back to audio. Writes the message, the transmitted
//! carrier and the received audio as WAV so you can listen to each stage.
//!
//!     cargo run --example fm_radio               # writes to target/examples-out/
//!     cargo run --example fm_radio -- <out_dir>
//!
//! Chain: message -> FmModulator -> IqModulator -> (+ noise) -> IqDemodulator -> FmDiscriminator -> audio low-pass

use std::path::Path;
use autodyne::filter::{Biquad, BUTTERWORTH_Q};
use autodyne::iq::{FmDiscriminator, FmModulator, IqDemodulator, IqModulator};
use autodyne::osc::{Noise, Sine};
use autodyne::signal::Signal;
use autodyne::units::Complex;

const SAMPLE_RATE: u32 = 48_000;
const BLOCK_SIZE: usize = 512;
const CARRIER: f64 = 12_000.0;
/// message value 1.0 shifts the carrier by this many Hz
const DEVIATION: f64 = 3_000.0;
/// receiver bandwidth per Carson's rule: deviation + highest message frequency, with margin
const BANDWIDTH: f64 = 4_500.0;
/// FM noise rises with frequency after the discriminator, so keep only the audio band we need.
const AUDIO_CUTOFF: f64 = 1_500.0;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let out_dir = std::env::args().nth(1).unwrap_or_else(|| "target/examples-out".into());
    std::fs::create_dir_all(&out_dir)?;
    let fs = SAMPLE_RATE as f64;
    let len = 3 * SAMPLE_RATE as usize;

    // Message: an A major chord (A4, C#5, E5).
    let mut message = vec![0.0f64; len];
    for f in [440.0, 554.37, 659.25] {
        Sine::new(f, fs).with_amplitude(0.3).add_to(&mut message);
    }

    // Transmitter, channel and receiver, all running block by block.
    let (mut fm_tx, mut iq_tx) = (FmModulator::new(DEVIATION, fs), IqModulator::new(CARRIER, fs));
    let mut air_noise = Noise::new(7).with_amplitude(0.1);
    let (mut iq_rx, mut fm_rx) = (IqDemodulator::new(CARRIER, BANDWIDTH, fs), FmDiscriminator::new(DEVIATION, fs));
    let mut audio_lp = Biquad::lowpass(AUDIO_CUTOFF, BUTTERWORTH_Q, fs);

    let mut rf = vec![0.0f64; len];
    let mut received = vec![0.0f64; len];
    let mut baseband = vec![Complex::zero(); BLOCK_SIZE];
    for ((msg, air), out) in message.chunks(BLOCK_SIZE).zip(rf.chunks_mut(BLOCK_SIZE)).zip(received.chunks_mut(BLOCK_SIZE)) {
        let bb = &mut baseband[..msg.len()];
        fm_tx.process(msg, bb);
        iq_tx.process(bb, air);
        air_noise.add_to(air);
        iq_rx.process(air, bb);
        fm_rx.process(bb, out);
        audio_lp.process(out);
    }

    // Compare message and received audio after the receiver settles, allowing for the filter delay.
    let skip = SAMPLE_RATE as usize / 10;
    let delay = best_delay(&message[skip..], &received[skip..], 200);
    let (msg, rx) = (&message[skip..len - delay], &received[skip + delay..]);
    let error: Vec<f64> = msg.iter().zip(rx).map(|(m, r)| r - m).collect();
    println!("receiver delay: {delay} samples ({:.2} ms)", 1_000.0 * delay as f64 / fs);
    println!("audio SNR after the link: {:.1} dB", msg.rms_db().unwrap() - error.rms_db().unwrap());

    for (name, data) in [("fm_message", &message), ("fm_carrier", &rf), ("fm_received", &received)] {
        write_wav(Path::new(&out_dir).join(format!("{name}.wav")), data, 0.9 / data.peak())?;
    }
    println!("wrote fm_message.wav, fm_carrier.wav and fm_received.wav to {out_dir}");
    Ok(())
}

/// Lag (0..max_lag) at which `delayed` best lines up with `reference`, by cross-correlation.
fn best_delay(reference: &[f64], delayed: &[f64], max_lag: usize) -> usize {
    let n = reference.len() - max_lag;
    (0..max_lag)
        .max_by(|&a, &b| {
            let corr = |lag: usize| reference[..n].iter().zip(&delayed[lag..]).map(|(x, y)| x * y).sum::<f64>();
            corr(a).total_cmp(&corr(b))
        })
        .unwrap()
}

/// Writes 32-bit float WAV, scaled by `gain` so each file plays at a similar level.
fn write_wav(path: impl AsRef<Path>, samples: &[f64], gain: f64) -> Result<(), hound::Error> {
    let spec = hound::WavSpec {
        channels: 1,
        sample_rate: SAMPLE_RATE,
        bits_per_sample: 32,
        sample_format: hound::SampleFormat::Float,
    };
    let mut writer = hound::WavWriter::create(path, spec)?;
    for &s in samples {
        writer.write_sample((s * gain) as f32)?;
    }
    writer.finalize()
}
