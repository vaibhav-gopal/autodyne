//! A tiny multisampled instrument: record a sound, save it as a looped WAV, load it back, and play.
//!
//!     cargo run --release --example sampler               # writes to target/examples-out/
//!     cargo run --release --example sampler -- <out_dir>
//!
//! Two "recordings" are made with the FM voice (a soft one and a bright one, at C4), each with a
//! sustain loop, and written as WAV files with `smpl` metadata (root key and loop points). They are
//! loaded back as two velocity layers spread across the keyboard, and a phrase is played in stereo
//! (the layers panned apart) to `sampler_phrase.wav`.

use std::path::Path;
use std::sync::Arc;

use autodyne::sampler::{LoopMode, Sample, SampleMap, SamplerVoice, Zone};
use autodyne::synth::{FmVoice, MidiMessage, Poly, TimedEvent, Voice};
use autodyne::wav::{self, Wav, WavFormat, WavLoop, WavLoopKind};

const FS: u32 = 48_000;
/// (start in seconds, MIDI note, velocity)
const PHRASE: [(f64, u8, u8); 8] = [(0.0, 48, 50), (0.0, 60, 50), (0.5, 64, 70), (1.0, 67, 90), (1.5, 72, 120), (2.0, 76, 110), (2.0, 79, 110), (2.0, 84, 127)];
const NOTE_LENGTH: f64 = 0.9;
const TOTAL_SECONDS: f64 = 4.5;

/// Renders one note of an FM patch as a "recording" with a sustain loop of whole cycles.
fn record(brightness: f32) -> Wav<f32> {
    let fs = FS as f32;
    let mut voice = FmVoice::new(fs);
    voice.set_algorithm(0);
    for op in 0..4 {
        voice.env_mut(op).set_sustain(0.7);
        voice.env_mut(op).set_decay(0.4);
    }
    voice.set_level(1, 0.3 * brightness);
    voice.set_ratio(1, 2.0);
    voice.note_on(60, 1.0);
    let mut audio = vec![0.0f32; FS as usize * 2];
    voice.render(&mut audio);
    // the loop spans whole periods of C4 once the envelope has settled, so it is seamless
    let period = FS as f64 / 261.625_565;
    let start = (1.2 * FS as f64 / period).round() * period;
    let end = start + (40.0 * period).round();
    Wav {
        sample_rate: FS,
        channels: vec![audio],
        root_key: Some(60.0),
        loops: vec![WavLoop { start: start as usize, end: end as usize, kind: WavLoopKind::Forward }],
    }
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let out_dir = std::env::args().nth(1).unwrap_or_else(|| "target/examples-out".into());
    std::fs::create_dir_all(&out_dir)?;

    // record and save two layers, then load them back as a sampler would
    let mut layers = Vec::new();
    for (name, brightness) in [("sampler_soft.wav", 0.4), ("sampler_bright.wav", 1.6)] {
        let path = Path::new(&out_dir).join(name);
        std::fs::write(&path, wav::write(&record(brightness), WavFormat::Pcm24))?;
        let sample = Sample::<f32>::from_wav(&std::fs::read(&path)?)?;
        let lp = sample.loop_points().expect("the smpl loop came back");
        // loop while the key is held, with a short crossfade, then play the recording's end
        let sample = sample.with_loop(lp.start, lp.end, LoopMode::Sustain, 256).with_mipmaps(3);
        println!("{name}: {} frames, root {}, loop {}..{}", sample.frames(), sample.root_key(), lp.start, lp.end);
        layers.push(Arc::new(sample));
    }
    let map = Arc::new(SampleMap::new(vec![
        Zone::new(layers[0].clone()).velocities(0..=79).pan(-0.4),
        Zone::new(layers[1].clone()).velocities(80..=127).pan(0.4).gain_db(-3.0),
    ]));

    let mut events: Vec<TimedEvent> = PHRASE
        .iter()
        .flat_map(|&(start, note, velocity)| {
            let on = (start * FS as f64) as usize;
            let off = ((start + NOTE_LENGTH) * FS as f64) as usize;
            [
                TimedEvent { offset: on, message: MidiMessage::NoteOn { channel: 0, note, velocity } },
                TimedEvent { offset: off, message: MidiMessage::NoteOff { channel: 0, note, velocity: 0 } },
            ]
        })
        .collect();
    events.sort_by_key(|e| e.offset);
    let mut sampler = Poly::new(16, 1_024, |_| {
        let mut v = SamplerVoice::new(map.clone(), FS as f32);
        v.amp_env_mut().set_release(0.6);
        v
    });
    let frames = (TOTAL_SECONDS * FS as f64) as usize;
    let (mut left, mut right) = (vec![0.0f32; frames], vec![0.0f32; frames]);
    let started = std::time::Instant::now();
    sampler.render_stereo_events(&mut left, &mut right, &events);
    let elapsed = started.elapsed().as_secs_f64();
    for s in left.iter_mut().chain(right.iter_mut()) {
        *s *= 0.3;
    }
    let phrase = Wav { sample_rate: FS, channels: vec![left, right], ..Default::default() };
    std::fs::write(Path::new(&out_dir).join("sampler_phrase.wav"), wav::write(&phrase, WavFormat::Float32))?;
    println!("rendered {TOTAL_SECONDS} s in {:.1} ms ({:.0}x real time); wrote sampler_phrase.wav to {out_dir}", elapsed * 1e3, TOTAL_SECONDS / elapsed);
    Ok(())
}