//! Proves the real-time promise: after construction (and one warm-up call), processing never
//! allocates. A counting global allocator records allocations made on the current thread while
//! inside `assert_no_alloc`.

use std::alloc::{GlobalAlloc, Layout, System};
use std::cell::Cell;
use std::sync::Arc;

use autodyne::analysis::{LoudnessMeter, OnsetDetector, PitchDetector, SpectrumAnalyzer, TruePeak};
use autodyne::channels::{AudioBuffer, Linked, Panner, PerChannel, StereoWidth};
use autodyne::control::{Lfo, LfoShape, Modulated, Route, Transport};
use autodyne::delay::Echo;
use autodyne::dynamic::{build_dyn, DynBlock, FloatElement, ProcessorFactory};
use autodyne::dynamics::{Compressor, EnvelopeFollower};
use autodyne::filter::{Biquad, Fir, MultiBiquad, BUTTERWORTH_Q};
use autodyne::gain::Gain;
use autodyne::iq::{FmDiscriminator, FmModulator, IqDemodulator, IqModulator};
use autodyne::modulation::{ModulatedDelay, Phaser};
use autodyne::distortion::{Shape, Waveshaper};
use autodyne::envelope::Adsr;
use autodyne::osc::{Impulse, Noise, Oscillator, Phasor, Sine, Waveform, Wavetable};
use autodyne::params::{process_buffer_events, process_events, ParamEvent, Parameterized, Smoothed};
use autodyne::prelude::*;
use autodyne::resample::{Oversampled, Resampler};
use autodyne::sampler::{Interpolation, LoopMode, Sample, SampleMap, SamplerVoice, Zone};
use autodyne::reverb::{synthetic_ir, Convolver, Reverb};
use autodyne::spectral::{Fft, RealFft};
use autodyne::synth::{FmVoice, MidiMessage, Poly, SynthVoice, TimedEvent};
use autodyne::units::{Complex, DType};

struct Counting;

thread_local! {
    static ACTIVE: Cell<bool> = const { Cell::new(false) };
    static COUNT: Cell<usize> = const { Cell::new(0) };
}

fn note() {
    // try_with: the thread-locals may already be gone while a thread shuts down
    let _ = ACTIVE.try_with(|a| {
        if a.get() {
            let _ = COUNT.try_with(|c| c.set(c.get() + 1));
        }
    });
}

unsafe impl GlobalAlloc for Counting {
    unsafe fn alloc(&self, layout: Layout) -> *mut u8 {
        note();
        unsafe { System.alloc(layout) }
    }
    unsafe fn alloc_zeroed(&self, layout: Layout) -> *mut u8 {
        note();
        unsafe { System.alloc_zeroed(layout) }
    }
    unsafe fn realloc(&self, ptr: *mut u8, layout: Layout, new_size: usize) -> *mut u8 {
        note();
        unsafe { System.realloc(ptr, layout, new_size) }
    }
    unsafe fn dealloc(&self, ptr: *mut u8, layout: Layout) {
        unsafe { System.dealloc(ptr, layout) }
    }
}

#[global_allocator]
static ALLOC: Counting = Counting;

/// Runs `f` twice: once to warm up (lazy buffers may size themselves), then counting.
fn assert_no_alloc(what: &str, mut f: impl FnMut()) {
    f();
    ACTIVE.with(|a| a.set(true));
    COUNT.with(|c| c.set(0));
    f();
    ACTIVE.with(|a| a.set(false));
    let n = COUNT.with(|c| c.get());
    assert_eq!(n, 0, "{what} allocated {n} time(s) while processing");
}

const FS: f64 = 48_000.0;

#[test]
fn processors_do_not_allocate() {
    let mut block: Vec<f64> = Noise::new(1).take(512).collect();

    let mut biquad = Biquad::peaking(1_000.0, FS, 1.0, 6.0);
    assert_no_alloc("Biquad", || biquad.process(&mut block));
    let mut fir = Fir::lowpass(2_000.0, FS, 63);
    assert_no_alloc("Fir", || fir.process(&mut block));
    let mut gain = Gain::new(1.0, 0.01, FS);
    assert_no_alloc("Gain (ramping)", || {
        gain.set_gain(0.5);
        gain.process(&mut block)
    });
    let mut echo = Echo::new(0.5, FS);
    assert_no_alloc("Echo", || echo.process(&mut block));
    let mut comp = Compressor::new(FS);
    assert_no_alloc("Compressor", || comp.process(&mut block));
    let mut env = EnvelopeFollower::new(0.01, 0.1, FS);
    let mut copy = block.clone();
    assert_no_alloc("EnvelopeFollower", || env.process(&mut copy));
    let mut chorus = ModulatedDelay::chorus(FS);
    assert_no_alloc("ModulatedDelay", || chorus.process(&mut block));
    let mut phaser = Phaser::new(6, FS);
    assert_no_alloc("Phaser", || phaser.process(&mut block));
    let mut chain = (Biquad::lowpass(5_000.0, FS, BUTTERWORTH_Q), Compressor::new(FS), Echo::new(0.2, FS));
    assert_no_alloc("tuple chain", || chain.process(&mut block));
    let mut adsr = Adsr::new(0.01, 0.1, 0.5, 0.2, FS);
    assert_no_alloc("Adsr", || {
        adsr.note_on();
        adsr.process(&mut block);
        adsr.note_off();
    });
    let mut shaper = Oversampled::new(Waveshaper::new(Shape::Tanh, 4.0 * FS), 4, 256);
    assert_no_alloc("Oversampled<Waveshaper>", || shaper.process(&mut block));
}

#[test]
fn generators_sources_and_streams_do_not_allocate() {
    let mut block = vec![0.0f64; 512];
    let mut sine = Sine::new(440.0, FS);
    assert_no_alloc("Sine::fill", || sine.fill(&mut block));
    let mut saw = Oscillator::new(Waveform::Saw, 440.0, FS);
    assert_no_alloc("Oscillator::fill", || saw.fill(&mut block));
    let mut noise = Noise::new(3);
    assert_no_alloc("Noise::add_to", || noise.add_to(&mut block));
    let mut imp = Impulse::new();
    assert_no_alloc("Impulse::fill", || imp.fill(&mut block));
    let mut phasor = Phasor::new(1_000.0, FS);
    let mut zs = vec![Complex::zero(); 512];
    assert_no_alloc("Phasor::fill", || phasor.fill(&mut zs));
    let mut lazy = Sine::new(220.0, FS).mix(Noise::new(2).scaled(0.1)).through(Biquad::lowpass(1_000.0, FS, BUTTERWORTH_Q));
    assert_no_alloc("Source::through", || lazy.fill(&mut block));
    let mut stream = Sine::new(220.0, FS).stream().through(Compressor::new(FS));
    assert_no_alloc("SignalRead::through", || {
        stream.read_samples(&mut block).unwrap();
    });
}

#[test]
fn analysis_and_in_place_transforms_do_not_allocate() {
    let mut x: Vec<f64> = Noise::new(4).take(4_096).collect();
    let y: Vec<f64> = Noise::new(5).take(4_096).collect();
    assert_no_alloc("Signal analysis", || {
        std::hint::black_box((x.rms(), x.peak(), x.argmax(), x.variance(), x.zero_crossings(), x.inner(&y).ok(), x.angle(&y).ok()));
    });
    assert_no_alloc("SignalMut transforms", || {
        x.normalize_peak(0.9);
        x.remove_dc();
        x.fade_in(64);
        x.cumsum();
        x.diff();
        x.zip_apply(&y[..100], Broadcast::Tile, |a, b| a + b).unwrap();
    });
}

#[test]
fn spectral_iq_and_resampling_do_not_allocate() {
    let fft = Fft::<f64>::new(1_024);
    let mut buf = vec![Complex::new(1.0, 0.0); 1_024];
    assert_no_alloc("Fft forward + inverse", || {
        fft.forward(&mut buf);
        fft.inverse(&mut buf);
    });

    let baseband = vec![Complex::new(0.5, -0.2); 512];
    let (mut rf, mut out) = (vec![0.0; 512], vec![Complex::zero(); 512]);
    let (mut tx, mut rx) = (IqModulator::new(12_000.0, FS), IqDemodulator::new(12_000.0, 3_000.0, FS));
    assert_no_alloc("IQ modulate + demodulate", || {
        tx.process(&baseband, &mut rf);
        rx.process(&rf, &mut out);
    });
    let msg = vec![0.3; 512];
    let (mut fm, mut disc, mut audio) = (FmModulator::new(1_000.0, FS), FmDiscriminator::new(1_000.0, FS), vec![0.0; 512]);
    assert_no_alloc("FM modulate + discriminate", || {
        fm.process(&msg, &mut out);
        disc.process(&out, &mut audio);
    });

    let mut rs = Resampler::<f64>::new(48_000, 44_100);
    let input = vec![0.1; 480];
    let mut res_out = vec![0.0; rs.max_output_len(480)];
    assert_no_alloc("Resampler", || {
        rs.process(&input, &mut res_out);
    });
}

#[test]
fn multichannel_does_not_allocate() {
    let mut buf = AudioBuffer::new(2, 512);
    let interleaved: Vec<f64> = Noise::new(6).take(1_024).collect();
    let mut back = vec![0.0; 1_024];
    let mut chain = (
        PerChannel::new(2, |ch| ModulatedDelay::chorus(FS).with_lfo_phase(ch as f64 * 0.5)),
        Linked(Compressor::new(FS)),
        StereoWidth::new(1.2, FS),
    );
    assert_no_alloc("interleave + multichannel chain", || {
        buf.copy_from_interleaved(&interleaved);
        chain.process(&mut buf);
        buf.copy_to_interleaved(&mut back);
    });
    let mut eq = MultiBiquad::new(2, |ch| Biquad::peaking(1_000.0 + 500.0 * ch as f64, FS, 1.0, 3.0));
    assert_no_alloc("MultiBiquad", || eq.process(&mut buf));
    let mono = vec![0.5; 512];
    let mut pan = Panner::new(-0.3, FS);
    assert_no_alloc("Panner", || pan.process(&mono, &mut buf));
    assert_no_alloc("contiguous lanes of an AudioBuffer view", || {
        buf.as_nd_view_mut().for_each_lane(1, |ch| ch.iter_mut().for_each(|s| *s *= 0.5)).unwrap();
    });
}

struct Comp;

impl ProcessorFactory for Comp {
    type Output<T: FloatElement> = (Biquad<T>, Compressor<T>);
    fn build<T: FloatElement>(&self, fs: T) -> Self::Output<T> {
        (Biquad::lowpass(T::_lit(3_000.0), fs, T::_lit(BUTTERWORTH_Q)), Compressor::new(fs))
    }
}

#[test]
fn parameters_and_dynamic_processing_do_not_allocate() {
    let mut chain = (Gain::new(1.0, 0.0, FS), Compressor::new(FS), Biquad::peaking(1_000.0, FS, 1.0, 0.0));
    assert_no_alloc("setting parameters (automation)", || {
        chain.set_param(2, -24.0).unwrap();
        chain.set_normalized(7, 0.5).unwrap();
        std::hint::black_box(chain.get_param(4));
    });
    let mut stages = vec![Gain::new(1.0, 0.0, FS), Gain::new(1.0, 0.0, FS)];
    assert_no_alloc("setting parameters on a Vec chain", || {
        stages.set_param(1, -6.0).unwrap();
        std::hint::black_box(stages.get_param(1));
    });

    let mut dynamic = build_dyn(&Comp, DType::F32, FS).unwrap();
    let mut block = vec![0.25f32; 512];
    assert_no_alloc("DynProcessor::process_dyn", || dynamic.process_dyn(DynBlock::F32(&mut block)).unwrap());

    // ramped automation with sample-accurate events, mono and multichannel
    let mut smoothed = Smoothed::new((Biquad::lowpass(1_000.0, FS, BUTTERWORTH_Q), Compressor::new(FS)), 0.02, FS);
    let mut audio = vec![0.25f64; 512];
    let mut flip = false;
    assert_no_alloc("Smoothed + process_events", || {
        flip = !flip;
        let events = [ParamEvent { offset: 100, index: 0, value: if flip { 8_000.0 } else { 200.0 } }];
        process_events(&mut smoothed, &mut audio, &events).unwrap();
    });
    let mut stereo = Smoothed::new(Linked(Compressor::new(FS)), 0.02, FS);
    let mut buffer = AudioBuffer::new(2, 512);
    assert_no_alloc("Smoothed multichannel + process_buffer_events", || {
        flip = !flip;
        let events = [ParamEvent { offset: 64, index: 0, value: if flip { -30.0 } else { -10.0 } }];
        process_buffer_events(&mut stereo, &mut buffer, &events).unwrap();
    });

    // an LFO driving a modulation matrix, tempo-synced, with routes changed while running
    let mut voices = Modulated::new(Biquad::lowpass(1_000.0, FS, BUTTERWORTH_Q), 2, 4);
    let cutoff = voices.param_index("frequency_hz").unwrap();
    voices.set_route(0, Some(Route { source: 0, destination: cutoff, via: Some(1) })).unwrap();
    voices.set_depth(0, 0.3).unwrap();
    let mut lfo = Lfo::new(FS).with_shape(LfoShape::SmoothRandom);
    lfo.set_sync(true);
    let mut transport = Transport { playing: true, ..Transport::new(128.0) };
    assert_no_alloc("Lfo + Modulated::run", || {
        voices.set_route(1, Some(Route { source: 0, destination: cutoff, via: None })).unwrap();
        voices.run(
            audio.len(),
            |sources, n| {
                sources[0] = lfo.advance(n, Some(&transport));
                sources[1] = 0.5;
                transport.advance(n, FS);
            },
            |p, range| p.process(&mut audio[range]),
        );
    });
}

#[test]
fn polyphonic_synth_does_not_allocate() {
    let mut poly = Poly::new(8, 256, |_| SynthVoice::new(FS));
    let mut out = vec![0.0; 512];
    let on = |offset, note| TimedEvent { offset, message: MidiMessage::NoteOn { channel: 0, note, velocity: 100 } };
    let off = |offset, note| TimedEvent { offset, message: MidiMessage::NoteOff { channel: 0, note, velocity: 0 } };
    let events = [on(0, 60), on(10, 64), on(20, 67), off(300, 60), on(400, 72)];
    assert_no_alloc("Poly::render_events with note on/off", || poly.render_events(&mut out, &events));
    assert_no_alloc("Poly::handle + render (stealing, pedal, bend)", || {
        for n in 40..60 {
            poly.handle(MidiMessage::NoteOn { channel: 0, note: n, velocity: 90 });
        }
        poly.handle(MidiMessage::ControlChange { channel: 0, controller: 64, value: 127 });
        poly.handle(MidiMessage::PitchBend { channel: 0, value: 3000 });
        poly.render(&mut out);
        poly.set_param_by_id("cutoff_hz", 2_000.0).unwrap();
    });
    // unison, the ladder, glide, MPE expression and the mono modes
    for (id, value) in [("unison", 7.0), ("filter", 3.0), ("glide_s", 0.05), ("mpe", 1.0)] {
        poly.set_param_by_id(id, value).unwrap();
    }
    assert_no_alloc("Poly with unison, ladder, glide and MPE", || {
        poly.handle(MidiMessage::NoteOn { channel: 3, note: 62, velocity: 90 });
        poly.handle(MidiMessage::PitchBend { channel: 3, value: -2_000 });
        poly.handle(MidiMessage::ChannelPressure { channel: 3, value: 100 });
        poly.handle(MidiMessage::ControlChange { channel: 3, controller: 74, value: 20 });
        poly.handle(MidiMessage::PolyPressure { channel: 3, note: 62, value: 60 });
        poly.set_note_tuning(62, 0.3);
        poly.render(&mut out);
    });
    poly.set_param_by_id("waveform", 4.0).unwrap(); // the wavetable source
    assert_no_alloc("Poly on the wavetable source, morphing", || {
        poly.set_param_by_id("wt_position", 0.4).unwrap();
        poly.handle(MidiMessage::NoteOn { channel: 0, note: 70, velocity: 90 });
        poly.render(&mut out);
        let table = Wavetable::shared_classic();
        poly.voices_mut()[0].set_wavetable(table);
    });
    let mut fm = Poly::new(4, 256, |_| FmVoice::new(FS));
    assert_no_alloc("Poly<FmVoice>: notes, parameters, render", || {
        fm.note_on(60, 0.8);
        fm.set_param_by_id("algorithm", 2.0).unwrap();
        fm.set_param_by_id("feedback", 0.7).unwrap();
        fm.set_param_by_id("op4_ratio", 3.5).unwrap();
        fm.render(&mut out);
    });    poly.set_param_by_id("voice_mode", 2.0).unwrap(); // legato
    assert_no_alloc("Poly in legato mode", || {
        for n in [60, 64, 67] {
            poly.note_on(n, 0.7);
        }
        poly.note_off(67);
        poly.render(&mut out);
    });
}

#[test]
fn reverbs_and_real_fft_do_not_allocate() {
    let mut rfft = RealFft::<f64>::new(1_024);
    let signal = vec![0.1; 1_024];
    let mut spectrum = vec![Complex::zero(); rfft.spectrum_len()];
    let mut back = vec![0.0; 1_024];
    assert_no_alloc("RealFft forward + inverse", || {
        rfft.forward(&signal, &mut spectrum);
        rfft.inverse(&spectrum, &mut back);
    });

    let mut conv = Convolver::new(&synthetic_ir(0.5, FS, 1), 256, FS);
    let mut block: Vec<f64> = Noise::new(8).take(700).collect();
    assert_no_alloc("Convolver (0.5 s IR, uneven block)", || conv.process(&mut block));

    let mut reverb = Reverb::new(FS);
    let mut stereo = AudioBuffer::new(2, 512);
    assert_no_alloc("Reverb", || {
        reverb.process(&mut stereo);
        reverb.set_param_by_id("decay_s", 3.0).unwrap();
    });
}

#[test]
fn analyzers_do_not_allocate() {
    let tone: Vec<f64> = Oscillator::new(Waveform::Saw, 220.0, FS).take(4_096).collect();
    let mut spectrum = SpectrumAnalyzer::new(2_048, FS);
    let mut bands = [0.0; 64];
    assert_no_alloc("SpectrumAnalyzer push + bands", || {
        spectrum.push(&tone);
        spectrum.bands_db(20.0, 20_000.0, &mut bands);
        spectrum.peak_bands_db(20.0, 20_000.0, &mut bands);
    });
    let mut meter = LoudnessMeter::new(2, FS);
    let mut stereo = AudioBuffer::new(2, 4_096);
    stereo.channel_mut(0).copy_from_slice(&tone);
    assert_no_alloc("LoudnessMeter (one 4096-frame buffer, many steps)", || {
        meter.process_buffer(&stereo);
        meter.process(&[&tone, &tone]);
        let _ = (meter.momentary(), meter.integrated(), meter.loudness_range(), meter.true_peak_db());
    });
    let mut peak = TruePeak::new(FS);
    assert_no_alloc("TruePeak", || peak.push(&tone));
    let mut pitch = PitchDetector::new(FS, 50.0, 1_000.0);
    assert_no_alloc("PitchDetector", || {
        pitch.process(&tone);
    });
    assert!(pitch.pitch().is_some());
    let mut onsets = OnsetDetector::new(FS);
    let mut count = 0;
    assert_no_alloc("OnsetDetector", || onsets.process(&tone, |_| count += 1));
}

#[test]
fn sampler_does_not_allocate() {
    let tone: Vec<f64> = Oscillator::new(Waveform::Saw, 220.0, FS).take(48_000).collect();
    let looped = Arc::new(Sample::new(vec![tone.clone(), tone.clone()], FS).with_loop(10_000, 20_000, LoopMode::Forward, 500).with_mipmaps(3));
    let ping = Arc::new(Sample::from_mono(tone, FS).with_root_key(57.0).with_loop(1_000, 1_500, LoopMode::PingPong, 0));
    let map = Arc::new(SampleMap::new(vec![Zone::new(looped.clone()).keys(0..=63), Zone::new(ping.clone()).keys(64..=127).pan(0.5), Zone::new(looped).keys(64..=127)]));
    let mut poly = Poly::new(8, 256, |_| SamplerVoice::new(map.clone(), FS));
    let (mut left, mut right) = (vec![0.0; 1_000], vec![0.0; 1_000]);
    let events = [
        TimedEvent { offset: 10, message: MidiMessage::NoteOn { channel: 0, note: 40, velocity: 100 } },
        TimedEvent { offset: 300, message: MidiMessage::NoteOn { channel: 0, note: 100, velocity: 60 } },
        TimedEvent { offset: 600, message: MidiMessage::NoteOff { channel: 0, note: 40, velocity: 0 } },
    ];
    assert_no_alloc("Poly<SamplerVoice>: zones, round robin, loops, mipmaps, stereo", || {
        poly.render_stereo_events(&mut left, &mut right, &events);
        poly.set_param_by_id("tune", 3.5).unwrap();
        poly.set_param_by_id("interpolation", 1.0).unwrap();
        poly.handle(MidiMessage::NoteOn { channel: 0, note: 90, velocity: 127 });
        poly.handle(MidiMessage::PitchBend { channel: 0, value: 4_000 });
        poly.render(&mut left);
    });
    assert_eq!(poly.voices()[0].interpolation(), Interpolation::Cubic);
}