//! Throughput of autodyne's own processors, oscillators and transforms, a 512-sample block at a time
//! (10.7 ms of audio at 48 kHz). `cargo bench --bench dsp`; criterion keeps the history in
//! `target/criterion` and reports changes from the last run. Comparisons with other libraries are in
//! `bench/<suite>` (`bench/README.md`).

use std::hint::black_box;
use criterion::{criterion_group, criterion_main, BenchmarkId, Criterion, Throughput};
use autodyne::delay::Echo;
use autodyne::distortion::{Shape, Waveshaper};
use autodyne::envelope::Adsr;
use autodyne::dynamics::Compressor;
use autodyne::channels::PerChannel;
use autodyne::filter::{Biquad, Fir, MultiBiquad, BUTTERWORTH_Q};
use autodyne::gain::Gain;
use autodyne::resample::{Oversampled, Resampler};
use autodyne::iq::{IqDemodulator, IqModulator};
use autodyne::modulation::{ModulatedDelay, Phaser};
use autodyne::osc::{Noise, Oscillator, Sine, Waveform};
use autodyne::channels::{AudioBuffer, MultiProcessor};
use autodyne::reverb::{synthetic_ir, Convolver, Reverb};
use autodyne::fft::{Fft, RealFft};
use autodyne::units::Complex;

const FS: f32 = 48_000.0;
const BLOCK: usize = 512;

fn noise_block() -> Vec<f32> {
    Noise::new(1).take(BLOCK).collect()
}

fn oscillators(c: &mut Criterion) {
    let mut g = c.benchmark_group("osc");
    g.throughput(Throughput::Elements(BLOCK as u64));
    let mut buf = vec![0.0f32; BLOCK];
    let mut sine = Sine::new(440.0, FS);
    g.bench_function("sine", |b| b.iter(|| sine.fill(black_box(&mut buf))));
    let mut noise = Noise::new(1);
    g.bench_function("noise", |b| b.iter(|| noise.fill(black_box(&mut buf))));
    let mut saw = Oscillator::new(Waveform::Saw, 440.0, FS);
    g.bench_function("saw (polyblep)", |b| b.iter(|| saw.fill(black_box(&mut buf))));
    g.finish();
}

fn filters(c: &mut Criterion) {
    let mut g = c.benchmark_group("filter");
    g.throughput(Throughput::Elements(BLOCK as u64));
    let input = noise_block();
    let mut buf = input.clone();

    let mut bq = Biquad::lowpass(1_000.0, BUTTERWORTH_Q as f32, FS);
    g.bench_function("biquad", |b| b.iter(|| {
        buf.copy_from_slice(&input);
        bq.process(black_box(&mut buf));
    }));

    // 8 channels: one biquad per channel, one channel at a time vs 4 channels in lockstep
    let mut eight = AudioBuffer::new(8, BLOCK);
    for ch in 0..8 {
        eight.channel_mut(ch).copy_from_slice(&input);
    }
    let mut per_channel = PerChannel::new(8, |ch| Biquad::peaking(500.0 * (ch + 1) as f32, 1.0, 3.0, FS));
    g.throughput(Throughput::Elements(8 * BLOCK as u64));
    g.bench_function("biquad x8 channels, per channel", |b| b.iter(|| per_channel.process(black_box(&mut eight))));
    let mut multi = MultiBiquad::new(8, |ch| Biquad::peaking(500.0 * (ch + 1) as f32, 1.0, 3.0, FS));
    g.bench_function("biquad x8 channels, MultiBiquad", |b| b.iter(|| multi.process(black_box(&mut eight))));
    g.throughput(Throughput::Elements(BLOCK as u64));

    for taps in [16, 64, 256] {
        let mut fir = Fir::lowpass(1_000.0, taps, FS);
        g.bench_with_input(BenchmarkId::new("fir", taps), &taps, |b, _| b.iter(|| {
            buf.copy_from_slice(&input);
            fir.process(black_box(&mut buf));
        }));
    }
    g.finish();
}

fn fft(c: &mut Criterion) {
    let mut g = c.benchmark_group("fft");
    for len in [256, 1024, 4096] {
        let mut fft = Fft::<f32>::new(len);
        let input: Vec<Complex<f32>> = Noise::<f32>::new(2).take(len).map(Complex::from).collect();
        let mut buf = input.clone();
        g.throughput(Throughput::Elements(len as u64));
        g.bench_with_input(BenchmarkId::from_parameter(len), &len, |b, _| b.iter(|| {
            buf.copy_from_slice(&input);
            fft.forward(black_box(&mut buf));
        }));
    }
    g.finish();
}

fn iq(c: &mut Criterion) {
    let mut g = c.benchmark_group("iq");
    g.throughput(Throughput::Elements(BLOCK as u64));
    let baseband: Vec<Complex<f32>> = vec![Complex::new(0.6, -0.3); BLOCK];
    let mut rf = vec![0.0f32; BLOCK];
    let mut out = vec![Complex::zero(); BLOCK];
    let mut tx = IqModulator::new(12_000.0, FS);
    let mut rx = IqDemodulator::new(12_000.0, 3_000.0, FS);
    g.bench_function("modulate", |b| b.iter(|| tx.process(black_box(&baseband), &mut rf)));
    g.bench_function("demodulate", |b| b.iter(|| rx.process(black_box(&rf), &mut out)));
    g.finish();
}

fn effects(c: &mut Criterion) {
    let mut g = c.benchmark_group("effects");
    g.throughput(Throughput::Elements(BLOCK as u64));
    let input = noise_block();
    let mut buf = input.clone();

    let mut echo = Echo::new(1.0, FS);
    echo.set_immediate(0.3, 0.5, 0.5);
    g.bench_function("echo", |b| b.iter(|| {
        buf.copy_from_slice(&input);
        echo.process(black_box(&mut buf));
    }));

    let mut compressor = Compressor::new(FS);
    g.bench_function("compressor", |b| b.iter(|| {
        buf.copy_from_slice(&input);
        compressor.process(black_box(&mut buf));
    }));

    let mut gain = Gain::new(1.0, 0.02, FS);
    g.bench_function("gain", |b| b.iter(|| {
        buf.copy_from_slice(&input);
        gain.process(black_box(&mut buf));
    }));

    for (from, to) in [(48_000, 44_100), (44_100, 48_000), (48_000, 16_000)] {
        let mut rs = Resampler::<f32>::new(from, to);
        let mut out = vec![0.0f32; rs.max_output_len(BLOCK)];
        g.bench_function(format!("resample {from}->{to}"), |b| b.iter(|| rs.process(black_box(&input), &mut out)));
    }
    g.finish();
}

fn modulation(c: &mut Criterion) {
    let mut g = c.benchmark_group("modulation");
    g.throughput(Throughput::Elements(BLOCK as u64));
    let input = noise_block();
    let mut buf = input.clone();
    let mut chorus = ModulatedDelay::chorus(FS);
    g.bench_function("chorus", |b| b.iter(|| {
        buf.copy_from_slice(&input);
        chorus.process(black_box(&mut buf));
    }));
    let mut phaser = Phaser::new(6, FS);
    g.bench_function("phaser (6 stages)", |b| b.iter(|| {
        buf.copy_from_slice(&input);
        phaser.process(black_box(&mut buf));
    }));
    g.finish();
}

fn synth(c: &mut Criterion) {
    let mut g = c.benchmark_group("synth");
    g.throughput(Throughput::Elements(BLOCK as u64));
    let input = noise_block();
    let mut buf = input.clone();
    let mut adsr = Adsr::new(0.01, 0.1, 0.7, 0.3, FS);
    adsr.note_on();
    g.bench_function("adsr", |b| b.iter(|| {
        buf.copy_from_slice(&input);
        adsr.process(black_box(&mut buf));
    }));
    let mut shaper = Waveshaper::new(Shape::Tanh, FS);
    g.bench_function("waveshaper tanh", |b| b.iter(|| {
        buf.copy_from_slice(&input);
        shaper.process(black_box(&mut buf));
    }));
    let mut over = Oversampled::new(Waveshaper::new(Shape::Tanh, 4.0 * FS), 4, BLOCK);
    g.bench_function("waveshaper tanh, 4x oversampled", |b| b.iter(|| {
        buf.copy_from_slice(&input);
        over.process(black_box(&mut buf));
    }));
    g.finish();
}

fn reverb(c: &mut Criterion) {
    let mut g = c.benchmark_group("reverb");
    g.throughput(Throughput::Elements(BLOCK as u64));
    let input = noise_block();
    let mut stereo = AudioBuffer::new(2, BLOCK);
    let mut fdn = Reverb::new(FS);
    g.bench_function("fdn (stereo)", |b| b.iter(|| {
        stereo.channel_mut(0).copy_from_slice(&input);
        stereo.channel_mut(1).copy_from_slice(&input);
        fdn.process(black_box(&mut stereo));
    }));
    let mut buf = input.clone();
    let mut conv = Convolver::new(&synthetic_ir(1.0f32, FS, 1), 256, FS);
    g.bench_function("convolver, 1 s IR, block 256", |b| b.iter(|| {
        buf.copy_from_slice(&input);
        conv.process(black_box(&mut buf));
    }));
    g.finish();

    let mut g = c.benchmark_group("real fft vs complex fft");
    let real: Vec<f32> = Noise::new(3).take(4_096).collect();
    let mut rfft = RealFft::<f32>::new(4_096);
    let mut spectrum = vec![Complex::zero(); rfft.spectrum_len()];
    g.bench_function("real 4096", |b| b.iter(|| rfft.forward(black_box(&real), &mut spectrum)));
    let mut fft = Fft::<f32>::new(4_096);
    let mut full = vec![Complex::zero(); 4_096];
    g.bench_function("complex 4096 (real input)", |b| b.iter(|| fft.forward_real(black_box(&real), &mut full)));
    g.finish();
}

/// The n-d engine against the plain-slice loop it should match, and strided / broadcast /
/// reduction paths (1M f32 elements, so memory bandwidth matters as it does on real data).
fn nd(c: &mut Criterion) {
    use autodyne::signal::{NdArray, Zip};
    let (rows, cols) = (1_000, 1_000);
    let x = NdArray::<f32>::from_fn(&[rows, cols], |i| (i[0] * 7 + i[1]) as f32 * 1e-3).unwrap();
    let y = NdArray::<f32>::full(&[rows, cols], 0.25).unwrap();
    let row = NdArray::<f32>::full(&[cols], 2.0).unwrap();
    let mut out = NdArray::<f32>::zeros(&[rows, cols]).unwrap();
    let mut g = c.benchmark_group("nd (1M f32)");
    g.throughput(criterion::Throughput::Elements((rows * cols) as u64));
    g.bench_function("slice loop: out = x * y + 1", |b| b.iter(|| {
        for ((o, &a), &b) in out.as_mut_slice().iter_mut().zip(x.as_slice()).zip(y.as_slice()) {
            *o = a * b + 1.0;
        }
    }));
    g.bench_function("Zip: out = x * y + 1", |b| b.iter(|| {
        Zip::from(out.view_mut()).and(x.view()).unwrap().and(y.view()).unwrap().for_each(|o, &a, &b| *o = a * b + 1.0);
    }));
    g.bench_function("Zip, x transposed", |b| b.iter(|| {
        Zip::from(out.view_mut()).and(x.view().transpose()).unwrap().and(y.view()).unwrap().for_each(|o, &a, &b| *o = a * b + 1.0);
    }));
    g.bench_function("Zip, row broadcast: out = x * row", |b| b.iter(|| {
        Zip::from(out.view_mut()).and(x.view()).unwrap().and_broadcast(row.view()).unwrap().for_each(|o, &a, &r| *o = a * r);
    }));
    let x64 = NdArray::<f64>::from_fn(&[2_000, 2_000], |i| (i[0] + i[1]) as f64).unwrap();
    g.bench_function("map (allocating), f64 2000x2000", |b| b.iter(|| black_box(x64.view().map(|&v| 2.0 * v + 0.5))));
    g.bench_function("map (allocating), f64 2000x2000 transposed", |b| b.iter(|| black_box(x64.view().transpose().map(|&v| 2.0 * v + 0.5))));
    g.bench_function("sum (pairwise)", |b| b.iter(|| black_box(x.view().sum())));
    g.bench_function("sum, transposed view", |b| b.iter(|| black_box(x.view().transpose().sum())));
    g.bench_function("sum_axis(0) (column sums)", |b| b.iter(|| black_box(x.sum_axis(0).unwrap())));
    g.bench_function("sum_axis(1) (row sums)", |b| b.iter(|| black_box(x.sum_axis(1).unwrap())));
    g.finish();

    // processing along a strided axis: [time, channel] with 16 channels
    let (frames, channels) = (16_384, 16);
    let mut interleaved = NdArray::<f32>::from_fn(&[frames, channels], |i| ((i[0] * 31 + i[1]) % 97) as f32 * 0.01).unwrap();
    let mut planar = NdArray::<f32>::from_fn(&[channels, frames], |i| ((i[1] * 31 + i[0]) % 97) as f32 * 0.01).unwrap();
    let mut filters: Vec<Biquad<f32>> = (0..channels).map(|_| Biquad::lowpass(1_000.0, BUTTERWORTH_Q as f32, FS)).collect();
    let mut g = c.benchmark_group("lanes (16 ch x 16384)");
    g.throughput(criterion::Throughput::Elements((frames * channels) as u64));
    g.bench_function("biquad per lane, contiguous lanes", |b| b.iter(|| planar.process_lanes(1, &mut filters).unwrap()));
    g.bench_function("biquad per lane, strided lanes (chunked)", |b| b.iter(|| interleaved.process_lanes(0, &mut filters).unwrap()));
    g.finish();
}

criterion_group!(benches, oscillators, filters, fft, iq, effects, modulation, synth, reverb, nd);
criterion_main!(benches);
