//! Baseline throughput for the core processors, before any SIMD work.
//! Run with `cargo bench`; criterion reports time per block (512 samples = 10.7 ms of audio at 48 kHz).

use std::hint::black_box;
use criterion::{criterion_group, criterion_main, BenchmarkId, Criterion, Throughput};
use autodyne::delay::Echo;
use autodyne::filter::{Biquad, Fir, BUTTERWORTH_Q};
use autodyne::gain::Gain;
use autodyne::resample::Resampler;
use autodyne::iq::{IqDemodulator, IqModulator};
use autodyne::osc::{Noise, Sine};
use autodyne::spectral::Fft;
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
    g.finish();
}

fn filters(c: &mut Criterion) {
    let mut g = c.benchmark_group("filter");
    g.throughput(Throughput::Elements(BLOCK as u64));
    let input = noise_block();
    let mut buf = input.clone();

    let mut bq = Biquad::lowpass(1_000.0, FS, BUTTERWORTH_Q as f32);
    g.bench_function("biquad", |b| b.iter(|| {
        buf.copy_from_slice(&input);
        bq.process(black_box(&mut buf));
    }));

    for taps in [16, 64, 256] {
        let mut fir = Fir::lowpass(1_000.0, FS, taps);
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
        let fft = Fft::<f32>::new(len);
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

criterion_group!(benches, oscillators, filters, fft, iq, effects);
criterion_main!(benches);
