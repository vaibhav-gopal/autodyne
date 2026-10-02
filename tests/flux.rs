//! flux end to end: a one-pole low-pass traced over `Real`, emitted as StableHLO, compiled and run
//! by IREE, then differentiated and fitted.
//!
//! Needs `iree-compile` and `iree-run-module` (`pip install iree-base-compiler iree-base-runtime`)
//! in `AUTODYNE_IREE_DIR` or on `PATH`; each test says so and passes without running when they
//! are missing.

#![cfg(feature = "flux")]

use autodyne::filter::OnePole;
use autodyne::flux::{Iree, Module, Scan, Tensor};
use autodyne::units::Real;

const FS: f64 = 48_000.0;
const N: usize = 512;

fn iree() -> Option<Iree> {
    let found = Iree::find();
    if found.is_none() {
        eprintln!("skipping: IREE tools not found (set AUTODYNE_IREE_DIR or put iree-compile / iree-run-module on PATH)");
    }
    found
}

/// The step, with the cutoff as its parameter.
fn one_pole() -> Scan {
    Scan::trace(1, 1, |p, s, x| {
        let (s, y) = OnePole::lowpass(p[0], Real::lit(FS)).tick(s[0], x);
        (vec![s], y)
    })
}

fn noise(n: usize, seed: u32) -> Vec<f32> {
    let mut state = seed;
    (0..n)
        .map(|_| {
            state = state.wrapping_mul(1_664_525).wrapping_add(1_013_904_223);
            (state >> 8) as f32 / (1u32 << 23) as f32 - 1.0
        })
        .collect()
}

fn concrete(cutoff: f32, xs: &[f32]) -> Vec<f32> {
    let mut lp = OnePole::lowpass(cutoff, FS as f32);
    let mut block = xs.to_vec();
    lp.process(&mut block);
    block
}

fn loss_f64(cutoff: f64, xs: &[f32], targets: &[f32]) -> f64 {
    let lp = OnePole::lowpass(cutoff, FS);
    let (mut s, mut sum) = (0.0, 0.0);
    for (&x, &t) in xs.iter().zip(targets) {
        let (next, y) = lp.tick(s, x as f64);
        s = next;
        sum += (y - t as f64).powi(2);
    }
    sum / xs.len() as f64
}

/// `(loss, d loss / d param, d loss / d s0)` from a compiled `loss_grad_hlo` module.
fn loss_grad(module: &Module, param: f32, xs: &[f32], targets: &[f32]) -> (f32, f32, f32) {
    let out = module
        .call("main", &[Tensor::scalar(param), Tensor::vector(xs), Tensor::vector(targets), Tensor::scalar(0.0)], 3)
        .unwrap();
    (out[0].data[0], out[1].data[0], out[2].data[0])
}

#[test]
fn forward_matches_the_concrete_path() {
    let Some(iree) = iree() else { return };
    let module = iree.compile(&one_pole().forward_hlo(N)).unwrap();
    let xs = noise(N, 1);
    for cutoff in [200.0f32, 1_000.0, 8_000.0] {
        let out = module.call("main", &[Tensor::scalar(cutoff), Tensor::vector(&xs), Tensor::scalar(0.0)], 2).unwrap();
        assert_eq!(out[0].shape, vec![N]);
        let expected = concrete(cutoff, &xs);
        for (i, (got, want)) in out[0].data.iter().zip(&expected).enumerate() {
            assert!((got - want).abs() <= 1e-5, "cutoff {cutoff}, sample {i}: {got} vs {want}");
        }
        assert!((out[1].data[0] - expected[N - 1]).abs() <= 1e-5);
    }
}

#[test]
fn gradient_matches_finite_differences() {
    let Some(iree) = iree() else { return };
    let module = iree.compile(&one_pole().loss_grad_hlo(N)).unwrap();
    let xs = noise(N, 2);
    let targets = concrete(1_500.0, &xs);
    for cutoff in [300.0f32, 1_000.0, 4_000.0] {
        let (loss, grad, _) = loss_grad(&module, cutoff, &xs, &targets);
        let h = cutoff as f64 * 1e-4;
        let fd = (loss_f64(cutoff as f64 + h, &xs, &targets) - loss_f64(cutoff as f64 - h, &xs, &targets)) / (2.0 * h);
        assert!(((grad as f64 - fd) / fd).abs() < 1e-3, "cutoff {cutoff}: {grad} vs {fd}");
        assert!(((loss as f64 - loss_f64(cutoff as f64, &xs, &targets)) / loss as f64).abs() < 1e-4);
        // the interpreter computes the same program
        let reference = one_pole().loss_grad(&[cutoff], &xs, &targets, &[0.0]);
        assert!(((grad - reference.params[0]) / reference.params[0]).abs() < 1e-4);
    }
}

#[test]
fn fits_the_cutoff_by_gradient_descent() {
    let Some(iree) = iree() else { return };
    // optimise the log of the cutoff, so steps are relative
    let scan = Scan::trace(1, 1, |p, s, x| {
        let (s, y) = OnePole::lowpass(p[0].exp(), Real::lit(FS)).tick(s[0], x);
        (vec![s], y)
    });
    let module = iree.compile(&scan.loss_grad_hlo(N)).unwrap();
    let xs = noise(N, 3);
    let target_cutoff = 1_200.0f32;
    let targets = concrete(target_cutoff, &xs);

    // Adam
    let (mut u, mut m, mut v) = (300.0f32.ln(), 0.0f32, 0.0f32);
    let (lr, b1, b2) = (0.1f32, 0.9f32, 0.999f32);
    let mut first_loss = None;
    let mut steps = 0;
    for step in 1..=300 {
        steps = step;
        let (loss, g, _) = loss_grad(&module, u, &xs, &targets);
        first_loss.get_or_insert(loss);
        if ((u.exp() - target_cutoff) / target_cutoff).abs() < 1e-3 {
            break;
        }
        m = b1 * m + (1.0 - b1) * g;
        v = b2 * v + (1.0 - b2) * g * g;
        let (mh, vh) = (m / (1.0 - b1.powi(step)), v / (1.0 - b2.powi(step)));
        u -= lr * mh / (vh.sqrt() + 1e-12);
    }
    let fitted = u.exp();
    let (loss, _, _) = loss_grad(&module, u, &xs, &targets);
    eprintln!("fitted {fitted:.1} Hz (target {target_cutoff} Hz, start 300 Hz) in {steps} steps, loss {} -> {loss:e}", first_loss.unwrap());
    assert!(((fitted - target_cutoff) / target_cutoff).abs() < 1e-2, "fitted {fitted} Hz, wanted {target_cutoff} Hz");
    assert!(loss < first_loss.unwrap() * 1e-3, "loss {loss} from {}", first_loss.unwrap());
}
