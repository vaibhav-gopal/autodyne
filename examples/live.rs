//! A playable polyphonic synth on your default audio output.
//!
//!     cargo run --release --example live                # MIDI keyboard if one is connected, else a demo; 10 s
//!     cargo run --release --example live -- 60          # play for 60 seconds
//!     cargo run --release --example live -- 30 --demo   # always use the demo sequence
//!
//! Signal path: MIDI -> 8-voice `Poly<SynthVoice>` (band-limited saw -> enveloped resonant low-pass ->
//! ADSR) -> gain -> 4x oversampled tanh saturation -> panner (mono -> stereo) -> per-channel EQ +
//! chorus (LFOs half a cycle apart) -> linked compressor -> per-channel echo -> FDN reverb -> stereo
//! width -> linked limiter at -3 dBFS.
//!
//! MIDI arrives on midir's thread and reaches the audio thread through a lock-free ring buffer
//! (rtrb), so the audio callback never locks or allocates. Without a MIDI input (or with --demo), a
//! built-in sequencer plays chords under an arpeggio, with every event on its exact sample.

use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::Arc;
use std::time::Duration;

use autodyne::channels::{AudioBuffer, Linked, MultiProcessor, PerChannel};
use autodyne::delay::Echo;
use autodyne::distortion::{Shape, Waveshaper};
use autodyne::dynamics::Compressor;
use autodyne::filter::{Biquad, MultiBiquad, BUTTERWORTH_Q};
use autodyne::gain::{Gain, Panner, StereoWidth};
use autodyne::modulation::ModulatedDelay;
use autodyne::params::Parameterized;
use autodyne::resample::Oversampled;
use autodyne::reverb::Reverb;
use autodyne::synth::{MidiMessage, Poly, SynthVoice, TimedEvent};
use cpal::traits::{DeviceTrait, HostTrait, StreamTrait};
use cpal::{FromSample, SampleFormat, SizedSample, StreamConfig};
use rtrb::{Consumer, RingBuffer};

/// Largest block rendered at once; bigger device buffers are handled in pieces.
const MAX_FRAMES: usize = 1024;
const VOICES: usize = 8;

type StereoChain = (
    MultiBiquad<f32>,
    PerChannel<ModulatedDelay<f32>>,
    Linked<Compressor<f32>>,
    PerChannel<Echo<f32>>,
    Reverb<f32>,
    StereoWidth<f32>,
    Linked<Compressor<f32>>,
);

// DEMO SEQUENCE ===================================================================================

/// C - Am - F - G, one chord per 8 steps, with an arpeggio over it.
const CHORDS: [[u8; 4]; 4] = [[48, 60, 64, 67], [45, 57, 60, 64], [41, 57, 60, 65], [43, 55, 59, 62]];
const ARP: [u8; 8] = [72, 76, 79, 84, 79, 76, 72, 67];

/// Emits note events with sample offsets inside each block.
struct Sequencer {
    step_len: usize,
    gate_len: usize,
    pos: usize,
    step: usize,
}

impl Sequencer {
    fn new(sample_rate: f32) -> Self {
        let step_len = (0.16 * sample_rate) as usize;
        Self { step_len, gate_len: step_len * 2 / 3, pos: 0, step: 0 }
    }

    fn events(&mut self, frames: usize, out: &mut Vec<TimedEvent>) {
        let note = |offset, on: bool, note| TimedEvent {
            offset,
            message: if on {
                MidiMessage::NoteOn { channel: 0, note, velocity: 90 }
            } else {
                MidiMessage::NoteOff { channel: 0, note, velocity: 0 }
            },
        };
        let mut t = 0;
        while t < frames {
            if self.pos == 0 {
                if self.step.is_multiple_of(8) {
                    let chord = (self.step / 8) % CHORDS.len();
                    let previous = (chord + CHORDS.len() - 1) % CHORDS.len();
                    if self.step > 0 {
                        CHORDS[previous].iter().for_each(|&n| out.push(note(t, false, n)));
                    }
                    CHORDS[chord].iter().for_each(|&n| out.push(note(t, true, n)));
                }
                out.push(note(t, true, ARP[self.step % ARP.len()]));
            } else if self.pos == self.gate_len {
                out.push(note(t, false, ARP[self.step % ARP.len()]));
            }
            let next = if self.pos < self.gate_len { self.gate_len } else { self.step_len };
            let n = (next - self.pos).min(frames - t);
            t += n;
            self.pos += n;
            if self.pos == self.step_len {
                self.pos = 0;
                self.step += 1;
            }
        }
    }
}

// ENGINE ==========================================================================================

struct Engine {
    poly: Poly<SynthVoice<f32>>,
    level: Gain<f32>,
    drive: Oversampled<Waveshaper<f32>, f32>,
    pan: Panner<f32>,
    chain: StereoChain,
    midi: Option<Consumer<MidiMessage>>,
    demo: Option<Sequencer>,
    events: Vec<TimedEvent>,
    mono: Vec<f32>,
    stereo: AudioBuffer<f32>,
}

impl Engine {
    fn new(sample_rate: f32, midi: Option<Consumer<MidiMessage>>) -> Self {
        let mut poly = Poly::new(VOICES, MAX_FRAMES, |_| SynthVoice::new(sample_rate));
        // a warmer patch than the defaults, set through the same parameter API a host would use
        for (id, value) in [("cutoff_hz", 700.0), ("resonance", 0.35), ("env_amount", 2.5), ("amp_release_s", 0.4)] {
            poly.set_param_by_id(id, value).expect("known parameter");
        }
        let mut shaper = Waveshaper::new(Shape::Tanh, 4.0 * sample_rate); // runs at 4x the rate
        shaper.set_drive_db(6.0);
        shaper.set_output_db(-3.0);
        let mut compressor = Compressor::new(sample_rate);
        compressor.set_threshold_db(-18.0);
        compressor.set_makeup_db(3.0);
        let chain = (
            MultiBiquad::new(2, |_| Biquad::high_shelf(6_000.0, sample_rate, BUTTERWORTH_Q as f32, -4.0)),
            PerChannel::new(2, |ch| ModulatedDelay::chorus(sample_rate).with_lfo_phase(ch as f32 * 0.5)),
            Linked(compressor),
            PerChannel::new(2, |ch| {
                let mut echo = Echo::new(1.0, sample_rate);
                // slightly different times per side give a wider echo
                echo.set_immediate(if ch == 0 { 0.36 } else { 0.27 }, 0.3, 0.15);
                echo
            }),
            {
                let mut reverb = Reverb::new(sample_rate);
                reverb.set_decay(2.2);
                reverb.set_mix(0.25);
                reverb
            },
            StereoWidth::new(1.3, sample_rate),
            Linked(Compressor::limiter(-3.0, 0.05, sample_rate)),
        );
        let demo = midi.is_none().then(|| Sequencer::new(sample_rate));
        Self {
            poly,
            level: Gain::new(0.2, 0.0, sample_rate), // several voices sum; leave headroom
            drive: Oversampled::new(shaper, 4, MAX_FRAMES),
            pan: Panner::new(0.0, sample_rate),
            chain,
            midi,
            demo,
            events: Vec::with_capacity(64),
            mono: vec![0.0; MAX_FRAMES],
            stereo: AudioBuffer::new(2, MAX_FRAMES),
        }
    }

    /// Renders `frames` (<= MAX_FRAMES) stereo frames. Called on the audio thread: no allocation.
    fn render(&mut self, frames: usize) -> &AudioBuffer<f32> {
        if let Some(midi) = &mut self.midi {
            while let Ok(message) = midi.pop() {
                self.poly.handle(message); // applied at the start of the block
            }
        }
        self.events.clear();
        if let Some(demo) = &mut self.demo {
            demo.events(frames, &mut self.events); // a handful per block, within the reserved capacity
        }
        let mono = &mut self.mono[..frames];
        self.poly.render_events(mono, &self.events);
        self.level.process(mono);
        self.drive.process(mono);
        self.pan.process(mono, &mut self.stereo);
        self.chain.process(&mut self.stereo);
        &self.stereo
    }
}

// MIDI + AUDIO ====================================================================================

/// Connects to the first MIDI input port, forwarding parsed messages into a ring buffer.
/// Returns the connection (keep it alive) and the audio-side consumer, or None without MIDI.
fn open_midi() -> Option<(midir::MidiInputConnection<()>, Consumer<MidiMessage>)> {
    let input = match midir::MidiInput::new("autodyne live") {
        Ok(input) => input,
        Err(e) => {
            eprintln!("MIDI unavailable: {e}");
            return None;
        }
    };
    let ports = input.ports();
    let Some(port) = ports.first() else {
        println!("no MIDI input ports found");
        return None;
    };
    let name = input.port_name(port).unwrap_or_else(|_| "unnamed port".into());
    let (mut producer, consumer) = RingBuffer::new(1024);
    let connection = input
        .connect(
            port,
            "autodyne-live-in",
            move |_timestamp, bytes, _| {
                if let Some(message) = MidiMessage::parse(bytes) {
                    let _ = producer.push(message); // if the audio thread falls behind, drop rather than block
                }
            },
            (),
        )
        .map_err(|e| eprintln!("could not open MIDI input {name}: {e}"))
        .ok()?;
    println!("MIDI input: {name}");
    Some((connection, consumer))
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let args: Vec<String> = std::env::args().skip(1).collect();
    let seconds: u64 = args.iter().find_map(|a| a.parse().ok()).unwrap_or(10);
    let force_demo = args.iter().any(|a| a == "--demo");

    let midi = if force_demo { None } else { open_midi() };
    let (_connection, consumer) = match midi {
        Some((c, rx)) => (Some(c), Some(rx)),
        None => {
            println!("no MIDI input in use: playing the demo sequence");
            (None, None)
        }
    };

    let host = cpal::default_host();
    let device = host.default_output_device().ok_or("no output device found")?;
    let config = device.default_output_config()?;
    println!("device: {}", device.id()?);
    println!("format: {} Hz, {} channels, {}", config.sample_rate(), config.channels(), config.sample_format());

    let played = Arc::new(AtomicUsize::new(0));
    match config.sample_format() {
        SampleFormat::F32 => run::<f32>(&device, config.into(), seconds, consumer, played.clone())?,
        SampleFormat::F64 => run::<f64>(&device, config.into(), seconds, consumer, played.clone())?,
        SampleFormat::I16 => run::<i16>(&device, config.into(), seconds, consumer, played.clone())?,
        SampleFormat::I32 => run::<i32>(&device, config.into(), seconds, consumer, played.clone())?,
        SampleFormat::U16 => run::<u16>(&device, config.into(), seconds, consumer, played.clone())?,
        other => return Err(format!("unsupported sample format {other}").into()),
    }
    println!("played {} frames", played.load(Ordering::Relaxed));
    Ok(())
}

fn run<T>(
    device: &cpal::Device,
    config: StreamConfig,
    seconds: u64,
    midi: Option<Consumer<MidiMessage>>,
    played: Arc<AtomicUsize>,
) -> Result<(), Box<dyn std::error::Error>>
where
    T: SizedSample + FromSample<f32>,
{
    let channels = config.channels as usize;
    let mut engine = Engine::new(config.sample_rate as f32, midi);

    let stream = device.build_output_stream(
        config,
        move |data: &mut [T], _: &cpal::OutputCallbackInfo| {
            for chunk in data.chunks_mut(MAX_FRAMES * channels) {
                let frames = chunk.len() / channels;
                let stereo = engine.render(frames);
                let (left, right) = (stereo.channel(0), stereo.channel(1));
                for (f, frame) in chunk.chunks_mut(channels).enumerate() {
                    // map stereo onto whatever the device has: mono gets the sum, extra channels silence
                    match frame {
                        [mono] => *mono = T::from_sample(0.5 * (left[f] + right[f])),
                        [l, r, rest @ ..] => {
                            *l = T::from_sample(left[f]);
                            *r = T::from_sample(right[f]);
                            rest.iter_mut().for_each(|s| *s = T::from_sample(0.0f32));
                        }
                        [] => {}
                    }
                }
                played.fetch_add(frames, Ordering::Relaxed);
            }
        },
        |err| eprintln!("stream error: {err}"),
        None,
    )?;
    stream.play()?;
    std::thread::sleep(Duration::from_secs(seconds));
    Ok(())
}
