//! The audio processors' generic kernels traced through `Scan`: each must match the concrete
//! `f32` processor sample for sample, and its parameters must have usable gradients.

use super::optim::Optimizer;
use super::*;
use crate::distortion::Shape;
use crate::dynamics::{envelope_step, time_coeff, Compressor, CompressorCurve, EnvelopeFollower};
use crate::filter::{Biquad, BiquadCoeffs, BiquadKind, Ladder, LadderCoeffs, Svf, SvfCoeffs, SvfMode};
use crate::signal::NdArray;
use crate::units::{Elementwise, RealValued};

const FS: f64 = 48_000.0;

fn noise(n: usize, seed: u32) -> Vec<f32> {
    let mut state = seed;
    (0..n)
        .map(|_| {
            state = state.wrapping_mul(1_664_525).wrapping_add(1_013_904_223);
            (state >> 8) as f32 / (1u32 << 23) as f32 - 1.0
        })
        .collect()
}

fn close(a: &[f32], b: &[f32], tol: f32) {
    assert_eq!(a.len(), b.len());
    for (i, (x, y)) in a.iter().zip(b).enumerate() {
        assert!((x - y).abs() <= tol * (1.0 + y.abs()), "sample {i}: {x} vs {y}");
    }
}

fn run(scan: &Scan, params: &[f32], xs: &[f32], states: usize) -> Vec<f32> {
    let p: Vec<NdArray<f32>> = params.iter().map(|&v| scalar(v)).collect();
    let s0: Vec<NdArray<f32>> = (0..states).map(|_| scalar(0.0)).collect();
    scan.run(&p, &vector(xs), &s0).0.into_vec()
}

#[test]
fn every_biquad_kind_traces_like_the_filter() {
    let xs = noise(400, 1);
    for kind in [BiquadKind::Lowpass, BiquadKind::Highpass, BiquadKind::Bandpass, BiquadKind::Notch, BiquadKind::Allpass, BiquadKind::Peaking, BiquadKind::LowShelf, BiquadKind::HighShelf] {
        // params: frequency, q, gain_db
        let scan = Scan::trace(&[&[], &[], &[]], &[&[], &[]], &[], |p, s, x| {
            let c = BiquadCoeffs::design(kind, p[0], p[1], p[2], Tracer::lit(FS));
            let (next, y) = c.tick([s[0], s[1]], x);
            (next.to_vec(), y)
        });
        let traced = run(&scan, &[1_200.0, 0.9, 4.0], &xs, 2);
        let mut f = Biquad::from_design(crate::filter::BiquadDesign { kind, frequency: 1_200.0f32, q: 0.9, gain_db: 4.0, sample_rate: FS as f32 });
        let concrete: Vec<f32> = xs.iter().map(|&x| f.process_sample(x)).collect();
        close(&traced, &concrete, 1e-5);
    }
}

#[test]
fn svf_ladder_shapes_and_dynamics_trace_like_the_processors() {
    let xs: Vec<f32> = noise(600, 2).iter().map(|v| 1.5 * v).collect();
    // SVF, every mode
    for mode in SvfMode::ALL {
        let scan = Scan::trace(&[&[], &[]], &[&[], &[]], &[], |p, s, x| {
            let c = SvfCoeffs::new(p[0], p[1], Tracer::lit(FS));
            let (next, o) = c.tick([s[0], s[1]], x);
            (next.to_vec(), mode.output(o, c.k))
        });
        let mut f = Svf::new(mode, 2_000.0f32, 3.0, FS as f32);
        let concrete: Vec<f32> = xs.iter().map(|&x| f.process_sample(x)).collect();
        close(&run(&scan, &[2_000.0, 3.0], &xs, 2), &concrete, 1e-4);
    }
    // ladder with drive (saturation)
    let scan = Scan::trace(&[&[], &[], &[]], &[&[], &[], &[], &[]], &[], |p, s, x| {
        let c = LadderCoeffs::new(p[0], p[1], p[2], true, Tracer::lit(FS));
        let (next, y) = c.tick([s[0], s[1], s[2], s[3]], x);
        (next.to_vec(), y)
    });
    let mut ladder = Ladder::new(3_000.0f32, 0.6, FS as f32);
    ladder.set_drive(4.0);
    let concrete: Vec<f32> = xs.iter().map(|&x| ladder.process_sample(x)).collect();
    close(&run(&scan, &[3_000.0, 0.6, 4.0], &xs, 4), &concrete, 1e-4);
    // every waveshaper shape (no state)
    for shape in [Shape::Tanh, Shape::SoftClip, Shape::HardClip, Shape::Fold] {
        let g = trace(&[&[600]], |v| vec![shape.apply(v[0] * Tracer::lit(1.7))]);
        let traced = g.eval(&[vector(&xs)])[0].clone().into_vec();
        let concrete: Vec<f32> = xs.iter().map(|&x| shape.apply(x * 1.7)).collect();
        close(&traced, &concrete, 1e-6);
    }
    // envelope follower
    let scan = Scan::trace(&[&[], &[]], &[&[]], &[], |p, s, x| {
        let e = envelope_step(time_coeff(p[0], Tracer::lit(FS)), time_coeff(p[1], Tracer::lit(FS)), s[0], x);
        (vec![e], e)
    });
    let mut env = EnvelopeFollower::new(0.002f32, 0.05, FS as f32);
    let concrete: Vec<f32> = xs.iter().map(|&x| env.process_sample(x)).collect();
    close(&run(&scan, &[0.002, 0.05], &xs, 1), &concrete, 1e-5);
    // compressor: threshold, slope, knee, attack, release (no makeup)
    let scan = Scan::trace(&[&[], &[], &[], &[], &[]], &[&[]], &[], |p, s, x| {
        let curve = CompressorCurve { threshold_db: p[0], slope: p[1], knee_db: p[2], attack_coeff: time_coeff(p[3], Tracer::lit(FS)), release_coeff: time_coeff(p[4], Tracer::lit(FS)) };
        let (reduction, gain) = curve.tick(s[0], x.abs());
        (vec![reduction], x * gain)
    });
    let mut comp = Compressor::new(FS as f32);
    comp.set_threshold_db(-12.0);
    comp.set_slope(0.25);
    comp.set_knee_db(6.0);
    comp.set_attack(0.005);
    comp.set_release(0.08);
    let concrete: Vec<f32> = xs.iter().map(|&x| comp.process_sample(x)).collect();
    close(&run(&scan, &[-12.0, 0.25, 6.0, 0.005, 0.08], &xs, 1), &concrete, 1e-4);
}

#[test]
fn a_filter_chain_fits_its_parameters_by_gradient() {
    // biquad low-pass then tanh drive: recover cutoff (log Hz), Q and drive from a target
    let chain = |p: &[Tracer], s: &[Tracer], x: Tracer| {
        let c = BiquadCoeffs::design(BiquadKind::Lowpass, p[0].exp(), p[1], Tracer::lit(0.0), Tracer::lit(FS));
        let ([s0, s1], y) = c.tick([s[0], s[1]], x);
        (vec![s0, s1], Shape::Tanh.apply(y * p[2]))
    };
    let scan = Scan::trace(&[&[], &[], &[]], &[&[], &[]], &[], chain);
    let xs = vector(&noise(512, 3));
    let truth = [scalar(1_500f32.ln()), scalar(0.9), scalar(2.5)];
    let s0 = [scalar(0.0), scalar(0.0)];
    let (targets, _) = scan.run(&truth, &xs, &s0);
    let mut params = vec![scalar(400f32.ln()), scalar(0.6), scalar(1.0)];
    let mut opt = optim::Adam::new(0.05);
    let first = scan.loss_grad(&params, &xs, &targets, &s0).loss;
    for _ in 0..400 {
        let g = scan.loss_grad(&params, &xs, &targets, &s0);
        opt.step(&mut params, &g.params);
    }
    let last = scan.loss_grad(&params, &xs, &targets, &s0).loss;
    assert!(last < first * 1e-3, "loss {first} -> {last}");
    let cutoff = params[0].as_slice()[0].exp();
    assert!((cutoff / 1_500.0 - 1.0).abs() < 0.05, "cutoff {cutoff}");
    assert!((params[2].as_slice()[0] - 2.5).abs() < 0.1, "drive {}", params[2].as_slice()[0]);
}
