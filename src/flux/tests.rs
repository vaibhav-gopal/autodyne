use super::*;
use crate::filter::OnePole;
use crate::signal::{ArrayMath, NdArray};
use crate::units::Elementwise;

const FS: f64 = 48_000.0;

fn arr(data: &[f32], shape: &[usize]) -> NdArray<f32> {
    NdArray::from_vec(data.to_vec(), shape).unwrap()
}

/// Deterministic values in [-1, 1).
fn noise(n: usize, seed: u32) -> Vec<f32> {
    let mut state = seed;
    (0..n)
        .map(|_| {
            state = state.wrapping_mul(1_664_525).wrapping_add(1_013_904_223);
            (state >> 8) as f32 / (1u32 << 23) as f32 - 1.0
        })
        .collect()
}

fn random(shape: &[usize], seed: u32) -> NdArray<f32> {
    arr(&noise(shape.iter().product(), seed), shape)
}

fn close(a: &[f32], b: &[f32], tol: f32) -> bool {
    a.len() == b.len() && a.iter().zip(b).all(|(x, y)| (x - y).abs() <= tol * (1.0 + y.abs()))
}

/// Checks `vjp` of `f` against central differences of `Σ f(inputs) · w` (w fixed, random), for every
/// element of every input.
fn check_gradient(shapes: &[&[usize]], h: f32, tol: f32, f: impl Fn(&[Tracer]) -> Tracer) {
    let inputs: Vec<NdArray<f32>> = shapes.iter().enumerate().map(|(k, s)| random(s, 11 + k as u32)).collect();
    let forward = trace(shapes, |v| vec![f(v)]);
    let w = random(&forward.output_shapes()[0], 99);
    let loss_graph = trace(shapes, |v| vec![(f(v) * Tracer::constant(&w)).sum_all()]);
    let grads = trace(shapes, |v| {
        let y = f(v);
        vjp(&[y], &[Tracer::constant(&w)], v)
    })
    .eval(&inputs);
    let loss = |xs: &[NdArray<f32>]| loss_graph.eval(xs)[0].as_slice()[0] as f64;
    for (k, x) in inputs.iter().enumerate() {
        assert_eq!(grads[k].shape(), x.shape());
        for i in 0..x.len() {
            let mut moved = inputs.clone();
            moved[k].as_mut_slice()[i] += h;
            let up = loss(&moved);
            moved[k].as_mut_slice()[i] -= 2.0 * h;
            let down = loss(&moved);
            let fd = ((up - down) / (2.0 * h as f64)) as f32;
            let got = grads[k].as_slice()[i];
            assert!((got - fd).abs() <= tol * (1.0 + fd.abs()), "input {k} element {i}: vjp {got} vs finite difference {fd}");
        }
    }
}

// ELEMENT-WISE ====================================================================================

#[test]
fn derivatives_of_each_function() {
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
        (|x| x.powf(Tracer::lit(3.0)), |x| 3.0 * x * x, 1.2),
        (|x| Tracer::lit(2.0).powf(x), |x| 2f32.powf(x) * 2f32.ln(), 1.2),
        (|x| Tracer::lit(1.0) / x - x * x, |x| -1.0 / (x * x) - 2.0 * x, 0.8),
    ];
    for (k, (f, df, x)) in cases.into_iter().enumerate() {
        let g = trace(&[&[]], |v| vjp(&[f(v[0])], &[Tracer::lit(1.0)], &[v[0]]));
        let got = g.eval(&[scalar(x)])[0].as_slice()[0];
        assert!((got - df(x)).abs() < 1e-5 * df(x).abs().max(1.0), "case {k}: {got} vs {}", df(x));
    }
}

#[test]
fn min_max_select_route_the_gradient() {
    let g = trace(&[&[], &[]], |v| {
        let (a, b) = (v[0], v[1]);
        let y = a.minimum(b) + Tracer::lit(2.0) * a.maximum(b) + Tracer::select(a.less(b), a * a, b);
        vjp(&[y], &[Tracer::lit(1.0)], &[a, b])
    });
    let at = |a: f32, b: f32| g.eval(&[scalar(a), scalar(b)]).iter().map(|x| x.as_slice()[0]).collect::<Vec<_>>();
    // a < b: min = a, max = b, select = a²  ->  d/da = 1 + 2a, d/db = 2
    assert_eq!(at(1.0, 3.0), vec![3.0, 2.0]);
    // a > b: min = b, max = a, select = b  ->  d/da = 2, d/db = 1 + 1
    assert_eq!(at(3.0, 1.0), vec![2.0, 2.0]);
}

#[test]
fn element_wise_on_arrays_with_broadcasting() {
    // [2, 3] * [3] + scalar, tanh, select against a [2, 1] threshold
    check_gradient(&[&[2, 3], &[3], &[2, 1]], 1e-2, 2e-3, |v| {
        let y = (v[0] * v[1] + Tracer::lit(0.5)).tanh();
        Tracer::select(y.greater(v[2]), y * y, y * v[2])
    });
}

#[test]
fn unused_inputs_get_zero_gradient() {
    let g = trace(&[&[], &[2]], |v| vjp(&[v[0] * v[0]], &[Tracer::lit(1.0)], &[v[0], v[1]]));
    let out = g.eval(&[scalar(3.0), vector(&[1.0, 2.0])]);
    assert_eq!(out[0].as_slice(), &[6.0]);
    assert_eq!(out[1].shape(), &[2]);
    assert_eq!(out[1].as_slice(), &[0.0, 0.0]);
}

#[test]
fn second_derivative() {
    // d²/dx² x³ = 6x, by differentiating the traced derivative again
    let g = trace(&[&[]], |v| {
        let x = v[0];
        let d = vjp(&[x * x * x], &[Tracer::lit(1.0)], &[x])[0];
        vjp(&[d], &[Tracer::lit(1.0)], &[x])
    });
    assert_eq!(g.eval(&[scalar(2.0)])[0].as_slice(), &[12.0]);
}

// ARRAY OPERATIONS ================================================================================

#[test]
fn shape_operations_forward() {
    let x = arr(&[1.0, 2.0, 3.0, 4.0, 5.0, 6.0], &[2, 3]);
    let column = vector(&[10.0, 20.0, 30.0]);
    let g = trace(&[&[2, 3]], |v| {
        vec![
            v[0].transpose(&[1, 0]),
            v[0].reshape(&[3, 2]),
            v[0].sum_axes(&[0]),
            v[0].sum_axes(&[1]),
            v[0].sum_all(),
            Tracer::constant(&column).broadcast_in_dim(&[3, 2], &[0]),
            v[0].sum_axes(&[1]).broadcast_in_dim(&[2, 2], &[0]),
        ]
    });
    let out = g.eval(&[x]);
    assert_eq!(out[0].shape(), &[3, 2]);
    assert_eq!(out[0].as_slice(), &[1.0, 4.0, 2.0, 5.0, 3.0, 6.0]);
    assert_eq!(out[1].as_slice(), &[1.0, 2.0, 3.0, 4.0, 5.0, 6.0]);
    assert_eq!(out[2].as_slice(), &[5.0, 7.0, 9.0]);
    assert_eq!(out[3].as_slice(), &[6.0, 15.0]);
    assert_eq!(out[4].as_slice(), &[21.0]);
    assert_eq!(out[5].as_slice(), &[10.0, 10.0, 20.0, 20.0, 30.0, 30.0]);
    assert_eq!(out[6].as_slice(), &[6.0, 6.0, 15.0, 15.0]);
}

#[test]
fn shape_operation_gradients() {
    check_gradient(&[&[3]], 0.1, 1e-4, |v| v[0].broadcast_to(&[2, 3]));
    check_gradient(&[&[2, 1]], 0.1, 1e-4, |v| v[0].broadcast_to(&[3, 2, 4]));
    check_gradient(&[&[2, 3]], 0.1, 1e-4, |v| v[0].reshape(&[3, 2]));
    check_gradient(&[&[2, 3, 4]], 0.1, 1e-4, |v| v[0].transpose(&[2, 0, 1]));
    check_gradient(&[&[2, 3, 4]], 0.1, 1e-4, |v| v[0].sum_axes(&[0, 2]));
    check_gradient(&[&[2, 3]], 1e-2, 1e-3, |v| v[0].mean_all() * v[0]);
}

#[test]
fn dot_forward() {
    let m = arr(&[1.0, 2.0, 3.0, 4.0, 5.0, 6.0], &[2, 3]);
    let x = vector(&[1.0, 0.0, -1.0]);
    let g = trace(&[&[2, 3], &[3]], |v| vec![v[0].dot(v[1]), v[1].dot(v[1]), v[0].dot(v[0].transpose(&[1, 0])), v[0].dot_general(v[0], &[0], &[0])]);
    let out = g.eval(&[m, x]);
    assert_eq!(out[0].as_slice(), &[-2.0, -2.0]);
    assert_eq!(out[1].as_slice(), &[2.0]);
    assert_eq!(out[2].as_slice(), &[14.0, 32.0, 32.0, 77.0]);
    assert_eq!(out[3].shape(), &[3, 3]);
    assert_eq!(out[3].as_slice(), &[17.0, 22.0, 27.0, 22.0, 29.0, 36.0, 27.0, 36.0, 45.0]);
}

#[test]
fn dot_gradients() {
    check_gradient(&[&[3, 4], &[4]], 0.1, 1e-4, |v| v[0].dot(v[1]));
    check_gradient(&[&[4], &[4]], 0.1, 1e-4, |v| v[0].dot(v[1]));
    check_gradient(&[&[2, 3], &[3, 5]], 0.1, 1e-4, |v| v[0].dot(v[1]));
    check_gradient(&[&[3, 2], &[4, 3]], 0.1, 1e-4, |v| v[0].dot_general(v[1], &[0], &[1]));
    // two contracted axes, paired out of order
    check_gradient(&[&[2, 3, 4], &[4, 5, 3]], 0.1, 1e-4, |v| v[0].dot_general(v[1], &[1, 2], &[2, 0]));
    // outer product
    check_gradient(&[&[2], &[3]], 0.1, 1e-4, |v| v[0].dot_general(v[1], &[], &[]));
}

/// The DFT of a real signal by definition, in f64.
fn naive_rfft(x: &[f32]) -> (Vec<f32>, Vec<f32>) {
    let n = x.len();
    (0..n / 2 + 1)
        .map(|k| {
            x.iter().enumerate().fold((0.0f64, 0.0f64), |(re, im), (t, &v)| {
                let phase = -std::f64::consts::TAU * (k * t) as f64 / n as f64;
                (re + v as f64 * phase.cos(), im + v as f64 * phase.sin())
            })
        })
        .map(|(re, im)| (re as f32, im as f32))
        .unzip()
}

#[test]
fn fft_forward_and_round_trip() {
    for n in [8usize, 7, 6, 1] {
        let x = random(&[3, n], n as u32);
        let g = trace(&[&[3, n]], |v| {
            let (re, im) = v[0].rfft();
            vec![re, im, Tracer::irfft(re, im, n)]
        });
        let out = g.eval(std::slice::from_ref(&x));
        let m = n / 2 + 1;
        for row in 0..3 {
            let (re, im) = naive_rfft(&x.as_slice()[row * n..(row + 1) * n]);
            assert!(close(&out[0].as_slice()[row * m..(row + 1) * m], &re, 1e-5), "n = {n}");
            assert!(close(&out[1].as_slice()[row * m..(row + 1) * m], &im, 1e-5), "n = {n}");
        }
        assert!(close(out[2].as_slice(), x.as_slice(), 1e-5), "irfft(rfft(x)) == x for n = {n}");
    }
}

#[test]
fn fft_gradients() {
    for n in [8usize, 7] {
        check_gradient(&[&[2, n]], 0.1, 1e-3, |v| v[0].rfft().0);
        check_gradient(&[&[2, n]], 0.1, 1e-3, |v| v[0].rfft().1);
        let m = n / 2 + 1;
        check_gradient(&[&[2, m], &[2, m]], 0.1, 1e-3, |v| Tracer::irfft(v[0], v[1], n));
        // a spectral gain: irfft(rfft(x) * g)
        check_gradient(&[&[n], &[m]], 1e-2, 2e-3, |v| {
            let (re, im) = v[0].rfft();
            Tracer::irfft(re * v[1], im * v[1], n)
        });
    }
}

// SCAN ============================================================================================

fn one_pole() -> Scan {
    Scan::trace(&[&[]], &[&[]], &[], |p, s, x| {
        let (s, y) = OnePole::lowpass(p[0], Tracer::lit(FS)).tick(s[0], x);
        (vec![s], y)
    })
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
fn scan_matches_the_concrete_filter_exactly() {
    let xs = noise(512, 1);
    let (ys, last) = one_pole().run(&[scalar(1_000.0)], &vector(&xs), &[scalar(0.0)]);
    let mut lp = OnePole::lowpass(1_000.0f32, FS as f32);
    let mut block = xs.clone();
    lp.process(&mut block);
    assert_eq!(ys.as_slice(), block);
    assert_eq!(last[0].as_slice(), &[block[511]]);
}

#[test]
fn scan_gradient_matches_finite_differences() {
    let xs = noise(512, 2);
    let (targets, _) = one_pole().run(&[scalar(1_500.0)], &vector(&xs), &[scalar(0.0)]);
    let t = targets.as_slice();
    for cutoff in [300.0f32, 1_000.0, 4_000.0] {
        let g = one_pole().loss_grad(&[scalar(cutoff)], &vector(&xs), &targets, &[scalar(0.0)]);
        let h = cutoff as f64 * 1e-4;
        let fd = (loss_f64(cutoff as f64 + h, &xs, t) - loss_f64(cutoff as f64 - h, &xs, t)) / (2.0 * h);
        let got = g.params[0].as_slice()[0] as f64;
        assert!(((got - fd) / fd).abs() < 1e-3, "cutoff {cutoff}: {got} vs {fd}");
        assert!(((g.loss as f64 - loss_f64(cutoff as f64, &xs, t)) / g.loss as f64).abs() < 1e-4);
    }
}

#[test]
fn multichannel_scan_runs_each_channel_like_the_scalar_filter() {
    // four one-poles at once: parameters, state and samples are [4]
    let bank = Scan::trace(&[&[4]], &[&[4]], &[4], |p, s, x| {
        let (s, y) = OnePole::lowpass(p[0], Tracer::lit(FS)).tick(s[0], x);
        (vec![s], y)
    });
    let cutoffs = [200.0f32, 1_000.0, 3_000.0, 9_000.0];
    let xs = random(&[128, 4], 5);
    let (ys, _) = bank.run(&[vector(&cutoffs)], &xs, &[vector(&[0.0; 4])]);
    assert_eq!(ys.shape(), &[128, 4]);
    for (c, &fc) in cutoffs.iter().enumerate() {
        let mut lp = OnePole::lowpass(fc, FS as f32);
        let mut channel: Vec<f32> = (0..128).map(|i| xs.as_slice()[i * 4 + c]).collect();
        lp.process(&mut channel);
        let got: Vec<f32> = (0..128).map(|i| ys.as_slice()[i * 4 + c]).collect();
        assert_eq!(got, channel, "channel {c}");
    }
}

/// `s' = A·s + B x`, `y = C·s'`: dot products in the step.
fn state_space() -> Scan {
    Scan::trace(&[&[2, 2], &[2], &[2]], &[&[2]], &[], |p, s, x| {
        let next = p[0].dot(s[0]) + p[1] * x;
        (vec![next], p[2].dot(next))
    })
}

#[test]
fn state_space_gradient_matches_finite_differences() {
    let scan = state_space();
    let params = [arr(&[0.6, -0.3, 0.2, 0.5], &[2, 2]), vector(&[1.0, 0.5]), vector(&[0.3, -0.7])];
    let xs = random(&[64], 6);
    let targets = random(&[64], 7);
    let s0 = [vector(&[0.1, -0.2])];
    let g = scan.loss_grad(&params, &xs, &targets, &s0);
    let loss = |params: &[NdArray<f32>], s0: &[NdArray<f32>]| scan.loss_grad(params, &xs, &targets, s0).loss as f64;
    let h = 1e-2f32;
    for k in 0..3 {
        for i in 0..params[k].len() {
            let mut p = params.to_vec();
            p[k].as_mut_slice()[i] += h;
            let up = loss(&p, &s0);
            p[k].as_mut_slice()[i] -= 2.0 * h;
            let fd = ((up - loss(&p, &s0)) / (2.0 * h as f64)) as f32;
            let got = g.params[k].as_slice()[i];
            assert!((got - fd).abs() < 2e-3 * (1.0 + fd.abs()), "param {k}[{i}]: {got} vs {fd}");
        }
    }
    for i in 0..2 {
        let mut s = s0.to_vec();
        s[0].as_mut_slice()[i] += h;
        let up = loss(&params, &s);
        s[0].as_mut_slice()[i] -= 2.0 * h;
        let fd = ((up - loss(&params, &s)) / (2.0 * h as f64)) as f32;
        let got = g.state[0].as_slice()[i];
        assert!((got - fd).abs() < 2e-3 * (1.0 + fd.abs()), "state[{i}]: {got} vs {fd}");
    }
}

#[test]
fn programs_have_the_expected_signatures() {
    let forward = one_pole().forward_program(512);
    assert_eq!(forward.text.matches("stablehlo.while").count(), 1);
    assert!(forward.text.starts_with("func.func @main(%v0: tensor<f32>, %v1: tensor<512xf32>, %v2: tensor<f32>) -> (tensor<512xf32>, tensor<f32>)"));
    assert_eq!(forward.inputs, vec![vec![], vec![512], vec![]]);
    let grad = state_space().loss_grad_program(64);
    assert_eq!(grad.text.matches("stablehlo.while").count(), 2);
    assert_eq!(grad.inputs, vec![vec![2, 2], vec![2], vec![2], vec![64], vec![64], vec![2]]);
    assert_eq!(grad.outputs, vec![vec![], vec![2, 2], vec![2], vec![2], vec![2]]);
    let g = trace(&[&[4, 8]], |v| vec![v[0].rfft().0.sum_axes(&[1])]).program();
    assert!(g.text.contains("stablehlo.fft") && g.text.contains("stablehlo.reduce"));
}

// TRACING RULES ===================================================================================

#[test]
#[should_panic(expected = "outside flux::trace")]
fn tracers_do_not_escape_their_trace() {
    let mut escaped = None;
    trace(&[&[]], |v| {
        escaped = Some(v[0]);
        vec![v[0]]
    });
    let _ = escaped.unwrap().exp();
}

#[test]
#[should_panic(expected = "the trace that created it")]
fn tracers_from_another_trace_are_rejected() {
    let mut escaped = None;
    trace(&[&[]], |v| {
        escaped = Some(v[0]);
        vec![v[0]]
    });
    trace(&[&[]], |v| vec![v[0] + escaped.unwrap()]);
}

#[test]
#[should_panic(expected = "do not broadcast")]
fn mismatched_shapes_are_rejected() {
    trace(&[&[2], &[3]], |v| vec![v[0] + v[1]]);
}
