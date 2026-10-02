use super::*;
use crate::filter::OnePole;
use crate::units::Real;

const FS: f64 = 48_000.0;

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

/// The one-pole's mean squared error in f64 through the concrete path, for finite differences.
fn loss_f64(cutoff: f64, xs: &[f32], targets: &[f32]) -> f64 {
    let lp = OnePole::lowpass(cutoff, FS);
    let mut s = 0.0;
    let mut sum = 0.0;
    for (&x, &t) in xs.iter().zip(targets) {
        let (next, y) = lp.tick(s, x as f64);
        s = next;
        sum += (y - t as f64).powi(2);
    }
    sum / xs.len() as f64
}

#[test]
fn derivatives_of_each_primitive() {
    // (traced function, its derivative, where to evaluate)
    type Case = (fn(Tracer) -> Tracer, fn(f32) -> f32, f32);
    let cases: [Case; 10] = [
        (|x| x.exp(), |x| x.exp(), 0.3),
        (|x| x.ln(), |x| 1.0 / x, 1.7),
        (|x| x.sin(), |x| x.cos(), 0.4),
        (|x| x.cos(), |x| -x.sin(), 0.4),
        (|x| x.tanh(), |x| 1.0 - x.tanh().powi(2), 0.6),
        (|x| x.sqrt(), |x| 0.5 / x.sqrt(), 2.0),
        (|x| x.abs(), |_| -1.0, -1.5),
        (|x| x.powf(Real::lit(3.0)), |x| 3.0 * x * x, 1.2),
        (|x| Tracer::lit(2.0).powf(x), |x| 2f32.powf(x) * 2f32.ln(), 1.2),
        (|x| Tracer::lit(1.0) / x - x * x, |x| -1.0 / (x * x) - 2.0 * x, 0.8),
    ];
    for (k, (f, df, x)) in cases.into_iter().enumerate() {
        let g = trace(1, |v| vjp(&[f(v[0])], &[Real::lit(1.0)], &[v[0]]));
        let got = g.eval(&[x])[0];
        assert!((got - df(x)).abs() < 1e-5 * df(x).abs().max(1.0), "case {k}: {got} vs {}", df(x));
    }
}

#[test]
fn min_max_select_route_the_gradient() {
    let g = trace(2, |v| {
        let (a, b) = (v[0], v[1]);
        let y = a.min(b) + Tracer::lit(2.0) * a.max(b) + Tracer::select(a.less(b), a * a, b);
        vjp(&[y], &[Real::lit(1.0)], &[a, b])
    });
    // a < b: min = a, max = b, select = a²  ->  d/da = 1 + 2a, d/db = 2
    assert_eq!(g.eval(&[1.0, 3.0]), vec![3.0, 2.0]);
    // a > b: min = b, max = a, select = b  ->  d/da = 2, d/db = 1 + 1
    assert_eq!(g.eval(&[3.0, 1.0]), vec![2.0, 2.0]);
}

#[test]
fn unused_inputs_get_zero_gradient() {
    let g = trace(2, |v| vjp(&[v[0] * v[0]], &[Real::lit(1.0)], &[v[0], v[1]]));
    assert_eq!(g.eval(&[3.0, 5.0]), vec![6.0, 0.0]);
}

#[test]
fn second_derivative() {
    // d²/dx² x³ = 6x, by differentiating the traced derivative again
    let g = trace(1, |v| {
        let x = v[0];
        let d = vjp(&[x * x * x], &[Real::lit(1.0)], &[x])[0];
        vjp(&[d], &[Real::lit(1.0)], &[x])
    });
    assert_eq!(g.eval(&[2.0]), vec![12.0]);
}

#[test]
fn scan_matches_the_concrete_filter_exactly() {
    let xs = noise(512, 1);
    let (ys, last) = one_pole().run(&[1_000.0], &xs, &[0.0]);
    let mut lp = OnePole::lowpass(1_000.0f32, FS as f32);
    let mut block = xs.clone();
    lp.process(&mut block);
    assert_eq!(ys, block);
    assert_eq!(last, vec![block[511]]);
}

#[test]
fn scan_gradient_matches_finite_differences() {
    let xs = noise(512, 2);
    let (targets, _) = one_pole().run(&[1_500.0], &xs, &[0.0]);
    for cutoff in [300.0f32, 1_000.0, 4_000.0] {
        let g = one_pole().loss_grad(&[cutoff], &xs, &targets, &[0.0]);
        let h = cutoff as f64 * 1e-4;
        let fd = (loss_f64(cutoff as f64 + h, &xs, &targets) - loss_f64(cutoff as f64 - h, &xs, &targets)) / (2.0 * h);
        assert!(((g.params[0] as f64 - fd) / fd).abs() < 1e-3, "cutoff {cutoff}: {} vs {fd}", g.params[0]);
        assert!(((g.loss as f64 - loss_f64(cutoff as f64, &xs, &targets)) / g.loss as f64).abs() < 1e-4);
    }
}

#[test]
fn initial_state_gradient() {
    // y = s0 * p^n decays from the initial state; d loss / d s0 against finite differences
    let scan = Scan::trace(0, 1, |_, s, x| {
        let y = s[0] * Real::lit(0.9) + x;
        (vec![y], y)
    });
    let xs = noise(64, 3);
    let targets = vec![0.0; 64];
    let g = scan.loss_grad(&[], &xs, &targets, &[0.5]);
    let loss = |s0: f32| scan.loss_grad(&[], &xs, &targets, &[s0]).loss;
    let fd = (loss(0.51) - loss(0.49)) / 0.02;
    assert!((g.state[0] - fd).abs() < 1e-3 * fd.abs().max(1.0), "{} vs {fd}", g.state[0]);
}

#[test]
fn emits_one_while_loop_per_pass() {
    let scan = one_pole();
    let forward = scan.forward_hlo(512);
    assert_eq!(forward.matches("stablehlo.while").count(), 1);
    assert!(forward.starts_with("func.func @main(%v0: tensor<f32>, %v1: tensor<512xf32>, %v2: tensor<f32>) -> (tensor<512xf32>, tensor<f32>)"));
    let grad = scan.loss_grad_hlo(512);
    assert_eq!(grad.matches("stablehlo.while").count(), 2);
    assert!(grad.contains("-> (tensor<f32>, tensor<f32>, tensor<f32>)"));
}

#[test]
#[should_panic(expected = "outside flux::trace")]
fn tracers_do_not_escape_their_trace() {
    let mut escaped = None;
    trace(1, |v| {
        escaped = Some(v[0]);
        vec![v[0]]
    });
    let _ = escaped.unwrap().exp();
}

#[test]
#[should_panic(expected = "the trace that created it")]
fn tracers_from_another_trace_are_rejected() {
    let mut escaped = None;
    trace(1, |v| {
        escaped = Some(v[0]);
        vec![v[0]]
    });
    trace(1, |v| vec![v[0] + escaped.unwrap()]);
}
