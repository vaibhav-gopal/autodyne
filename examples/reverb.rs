//! The same short phrase dry, through the algorithmic reverb, and through convolution reverb.
//!
//!     cargo run --release --example reverb               # writes to target/examples-out/
//!     cargo run --release --example reverb -- <out_dir>
//!
//! The phrase is played by the polyphonic synth (plucky saw notes), then processed three ways and
//! written as stereo WAV: `reverb_dry`, `reverb_fdn` (8-line feedback delay network, 2.5 s decay) and
//! `reverb_convolution` (a 2 s synthetic impulse response per channel, partitioned FFT convolution).
//! Prints how loud each tail still is one second after the last note.

use std::path::Path;

use autodyne::channels::{AudioBuffer, MultiProcessor, PerChannel};
use autodyne::reverb::{synthetic_ir, Convolver, Reverb};
use autodyne::signal::Signal;
use autodyne::synth::{MidiMessage, Poly, SynthVoice, TimedEvent};

const FS: u32 = 48_000;
/// (start in seconds, MIDI note): a rising figure, then a chord
const PHRASE: [(f64, u8); 7] = [(0.0, 60), (0.25, 64), (0.5, 67), (0.75, 72), (1.0, 60), (1.0, 64), (1.0, 67)];
const NOTE_LENGTH: f64 = 0.2;
const TOTAL_SECONDS: f64 = 4.5;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let out_dir = std::env::args().nth(1).unwrap_or_else(|| "target/examples-out".into());
    std::fs::create_dir_all(&out_dir)?;
    let fs = FS as f32;
    let frames = (TOTAL_SECONDS * FS as f64) as usize;

    // Render the phrase: every note on and off is an event at its exact sample.
    let mut events: Vec<TimedEvent> = PHRASE
        .iter()
        .flat_map(|&(start, note)| {
            let on = (start * FS as f64) as usize;
            let off = ((start + NOTE_LENGTH) * FS as f64) as usize;
            [
                TimedEvent { offset: on, message: MidiMessage::NoteOn { channel: 0, note, velocity: 100 } },
                TimedEvent { offset: off, message: MidiMessage::NoteOff { channel: 0, note, velocity: 0 } },
            ]
        })
        .collect();
    events.sort_by_key(|e| e.offset);
    let mut synth = Poly::new(8, 1_024, |_| SynthVoice::new(fs));
    let mut mono = vec![0.0f32; frames];
    synth.render_events(&mut mono, &events);
    mono.iter_mut().for_each(|s| *s *= 0.25);

    let stereo_copy = || {
        let mut buf = AudioBuffer::new(2, frames);
        buf.channel_mut(0).copy_from_slice(&mono);
        buf.channel_mut(1).copy_from_slice(&mono);
        buf
    };

    let dry = stereo_copy();

    let mut fdn = stereo_copy();
    let mut reverb = Reverb::new(fs);
    reverb.set_decay(2.5);
    reverb.set_mix(0.35);
    reverb.process(&mut fdn);

    let mut convolution = stereo_copy();
    let mut convolvers = PerChannel::new(2, |ch| {
        let mut c = Convolver::new(&synthetic_ir(2.0, fs, ch as u64 + 1), 512, fs);
        c.set_mix(0.35);
        c
    });
    convolvers.process(&mut convolution);

    // how much is left one second after the last note ends
    let last_note_end = PHRASE.iter().map(|&(s, _)| s).fold(0.0, f64::max) + NOTE_LENGTH;
    let tail = |buf: &AudioBuffer<f32>| {
        let start = ((last_note_end + 1.0) * FS as f64) as usize;
        buf.channel(0)[start..start + FS as usize / 10].rms_db().unwrap_or(f32::NEG_INFINITY)
    };
    for (name, buf) in [("reverb_dry", &dry), ("reverb_fdn", &fdn), ("reverb_convolution", &convolution)] {
        println!("{name:<20} level 1 s after the last note: {:>7.1} dB", tail(buf));
        write_stereo_wav(Path::new(&out_dir).join(format!("{name}.wav")), buf)?;
    }
    println!("wrote reverb_dry.wav, reverb_fdn.wav and reverb_convolution.wav to {out_dir}");
    Ok(())
}

fn write_stereo_wav(path: impl AsRef<Path>, buf: &AudioBuffer<f32>) -> Result<(), Box<dyn std::error::Error>> {
    let spec = hound::WavSpec { channels: 2, sample_rate: FS, bits_per_sample: 32, sample_format: hound::SampleFormat::Float };
    let mut interleaved = vec![0.0f32; buf.frames() * 2];
    buf.copy_to_interleaved(&mut interleaved);
    let mut writer = hound::WavWriter::create(path, spec)?;
    for s in interleaved {
        writer.write_sample(s)?;
    }
    writer.finalize()?;
    Ok(())
}
