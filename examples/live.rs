//! Real-time playback: a looping arpeggio through a stereo effects chain, rendered inside the audio
//! callback of the default output device.
//!
//!     cargo run --release --example live             # plays for 10 seconds
//!     cargo run --release --example live -- 30       # plays for 30 seconds
//!
//! Chain: sine voice -> note gate -> auto-panner (mono -> stereo) -> per-channel EQ + chorus (LFOs half a
//! cycle apart) -> linked compressor -> per-channel echo -> stereo width -> linked limiter at -3 dBFS.
//! Everything is allocated before the stream starts; the callback only processes.

use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::Arc;
use std::time::Duration;

use autodyne::channels::{AudioBuffer, Linked, MultiProcessor, Panner, PerChannel, StereoWidth};
use autodyne::delay::Echo;
use autodyne::dynamics::Compressor;
use autodyne::filter::{Biquad, BUTTERWORTH_Q};
use autodyne::gain::Gain;
use autodyne::modulation::ModulatedDelay;
use autodyne::osc::Sine;
use cpal::traits::{DeviceTrait, HostTrait, StreamTrait};
use cpal::{FromSample, SampleFormat, SizedSample, StreamConfig};

/// Largest block rendered at once; bigger device buffers are handled in pieces.
const MAX_FRAMES: usize = 1024;
/// C major arpeggio, up and back down (Hz)
const NOTES: [f32; 6] = [261.63, 329.63, 392.00, 523.25, 392.00, 329.63];
const STEP_SECONDS: f32 = 0.18;
/// fraction of each step the note sounds for
const GATE_FRACTION: f32 = 0.7;

type StereoChain = (
    PerChannel<(Biquad<f32>, ModulatedDelay<f32>)>,
    Linked<Compressor<f32>>,
    PerChannel<Echo<f32>>,
    StereoWidth<f32>,
    Linked<Compressor<f32>>,
);

struct Synth {
    sample_rate: f32,
    voice: Sine<f32>,
    gate: Gain<f32>,
    step_len: usize,
    gate_len: usize,
    pos_in_step: usize,
    note: usize,
    pan: Panner<f32>,
    pan_phase: f32,
    chain: StereoChain,
    mono: Vec<f32>,
    stereo: AudioBuffer<f32>,
}

impl Synth {
    fn new(sample_rate: f32) -> Self {
        let step_len = (STEP_SECONDS * sample_rate) as usize;
        let mut compressor = Compressor::new(sample_rate);
        compressor.set_threshold_db(-18.0);
        compressor.set_makeup_db(3.0);
        let chain = (
            PerChannel::new(2, |ch| {
                (
                    Biquad::high_shelf(6_000.0, sample_rate, BUTTERWORTH_Q as f32, -4.0),
                    ModulatedDelay::chorus(sample_rate).with_lfo_phase(ch as f32 * 0.5),
                )
            }),
            Linked(compressor),
            PerChannel::new(2, |ch| {
                let mut echo = Echo::new(1.0, sample_rate);
                // slightly different times per side give a wider echo
                echo.set_immediate(if ch == 0 { 0.36 } else { 0.27 }, 0.4, 0.3);
                echo
            }),
            StereoWidth::new(1.3, sample_rate),
            Linked(Compressor::limiter(-3.0, 0.05, sample_rate)),
        );
        Self {
            sample_rate,
            voice: Sine::new(NOTES[0], sample_rate).with_amplitude(0.3),
            gate: Gain::new(0.0, 0.005, sample_rate),
            step_len,
            gate_len: (step_len as f32 * GATE_FRACTION) as usize,
            pos_in_step: 0,
            note: 0,
            pan: Panner::new(0.0, sample_rate),
            pan_phase: 0.0,
            chain,
            mono: vec![0.0; MAX_FRAMES],
            stereo: AudioBuffer::new(2, MAX_FRAMES),
        }
    }

    /// Renders `frames` (<= MAX_FRAMES) stereo frames into `self.stereo`.
    fn render(&mut self, frames: usize) -> &AudioBuffer<f32> {
        // Note on/off events land on exact samples: split the block at each event.
        let mut done = 0;
        while done < frames {
            if self.pos_in_step == 0 {
                self.voice.set_frequency(NOTES[self.note], self.sample_rate);
                self.gate.set_gain(1.0);
            } else if self.pos_in_step == self.gate_len {
                self.gate.set_gain(0.0);
            }
            let next_event = if self.pos_in_step < self.gate_len { self.gate_len } else { self.step_len };
            let n = (next_event - self.pos_in_step).min(frames - done);
            let part = &mut self.mono[done..done + n];
            self.voice.fill(part);
            self.gate.process(part);
            done += n;
            self.pos_in_step += n;
            if self.pos_in_step == self.step_len {
                self.pos_in_step = 0;
                self.note = (self.note + 1) % NOTES.len();
            }
        }

        // slow auto-pan, 0.1 Hz; the panner smooths between block-rate position updates
        self.pan_phase = (self.pan_phase + frames as f32 * 0.1 / self.sample_rate).fract();
        self.pan.set_position(0.6 * (std::f32::consts::TAU * self.pan_phase).sin());
        self.pan.process(&self.mono[..frames], &mut self.stereo);
        self.chain.process(&mut self.stereo);
        &self.stereo
    }
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let seconds: u64 = std::env::args().nth(1).map(|s| s.parse()).transpose()?.unwrap_or(10);

    let host = cpal::default_host();
    let device = host.default_output_device().ok_or("no output device found")?;
    let config = device.default_output_config()?;
    println!("device: {}", device.id()?);
    println!("format: {} Hz, {} channels, {}", config.sample_rate(), config.channels(), config.sample_format());

    let played = Arc::new(AtomicUsize::new(0));
    match config.sample_format() {
        SampleFormat::F32 => run::<f32>(&device, config.into(), seconds, played.clone())?,
        SampleFormat::F64 => run::<f64>(&device, config.into(), seconds, played.clone())?,
        SampleFormat::I16 => run::<i16>(&device, config.into(), seconds, played.clone())?,
        SampleFormat::I32 => run::<i32>(&device, config.into(), seconds, played.clone())?,
        SampleFormat::U16 => run::<u16>(&device, config.into(), seconds, played.clone())?,
        other => return Err(format!("unsupported sample format {other}").into()),
    }
    println!("played {} frames", played.load(Ordering::Relaxed));
    Ok(())
}

fn run<T>(device: &cpal::Device, config: StreamConfig, seconds: u64, played: Arc<AtomicUsize>) -> Result<(), Box<dyn std::error::Error>>
where
    T: SizedSample + FromSample<f32>,
{
    let channels = config.channels as usize;
    let mut synth = Synth::new(config.sample_rate as f32);

    let stream = device.build_output_stream(
        config,
        move |data: &mut [T], _: &cpal::OutputCallbackInfo| {
            for chunk in data.chunks_mut(MAX_FRAMES * channels) {
                let frames = chunk.len() / channels;
                let stereo = synth.render(frames);
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
