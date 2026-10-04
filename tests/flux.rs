//! flux end to end: traced programs emitted as StableHLO, compiled and run by each backend found
//! (IREE, PJRT, XLA), and compared with the interpreter and with the concrete f32 code.
//!
//! IREE needs `iree-compile` and `iree-run-module` (`pip install iree-base-compiler
//! iree-base-runtime`) in `AUTODYNE_IREE_DIR` or on `PATH`; XLA needs Python with `jax`
//! (`AUTODYNE_XLA_PYTHON`, else `python3` / `python` on `PATH`); PJRT needs a plugin library in
//! `AUTODYNE_PJRT_PLUGIN` (client options in `AUTODYNE_PJRT_OPTIONS`, e.g. `preallocate=false` for
//! XLA's GPU plugin, as every test makes a client). A missing backend is reported and skipped; with none, the tests pass
//! without running. `AUTODYNE_IREE_GPU` adds IREE on GPUs: a comma-separated list of `vulkan`,
//! `cuda` or `rocm`, each with an optional architecture (`vulkan=ampere,cuda=sm_80`; the FFTs need
//! one for Vulkan, and ROCm always does).

#![cfg(feature = "flux")]

use autodyne::distortion::Shape;
use autodyne::filter::{BiquadCoeffs, BiquadKind, OnePole};
use autodyne::flux::optim::{Adam, Optimizer};
use autodyne::signal::frames;
use autodyne::flux::{multi_resolution_stft, scalar, trace, vector, vjp, Backend, Emit, Executable, ExecutableExt, Iree, IreeTarget, Loss, Pjrt, Program, Scan, StftResolution, Tracer, Xla};
use autodyne::signal::{ArrayMath, ComplexArrayMath, NdArray, RealArrayMath};
use autodyne::units::{Elementwise, RealValued};

const FS: f64 = 48_000.0;
const N: usize = 512;

fn backends() -> Vec<Box<dyn Backend>> {
    let mut found: Vec<Box<dyn Backend>> = Vec::new();
    match Iree::find() {
        Some(iree) => {
            for gpu in std::env::var("AUTODYNE_IREE_GPU").unwrap_or_default().split(',').map(str::trim).filter(|g| !g.is_empty()) {
                let (api, arch) = gpu.split_once('=').map_or((gpu, None), |(a, b)| (a, Some(b.to_string())));
                let target = match (api, arch) {
                    ("vulkan", target) => IreeTarget::Vulkan { target },
                    ("cuda", None) => IreeTarget::cuda(),
                    ("cuda", Some(target)) => IreeTarget::Cuda { target },
                    ("rocm", Some(target)) => IreeTarget::Rocm { target },
                    _ => panic!("AUTODYNE_IREE_GPU: cannot use {gpu:?} (vulkan[=arch], cuda[=arch], rocm=chip)"),
                };
                found.push(Box::new(iree.clone().with_target(target)));
            }
            found.push(Box::new(iree));
        }
        None => eprintln!("skipping IREE: tools not found (set AUTODYNE_IREE_DIR or put iree-compile / iree-run-module on PATH)"),
    }
    match std::env::var_os("AUTODYNE_PJRT_PLUGIN") {
        Some(path) => match Pjrt::load_with_options(std::path::Path::new(&path), &Pjrt::options_from_env()) {
            Ok(pjrt) => {
                eprintln!("PJRT: {}", pjrt.description());
                found.push(Box::new(pjrt));
            }
            Err(e) => panic!("AUTODYNE_PJRT_PLUGIN is set but the plugin fails to load: {e}"),
        },
        None => eprintln!("skipping PJRT: set AUTODYNE_PJRT_PLUGIN to a plugin library (e.g. libpjrt_cpu.so)"),
    }
    match Xla::start() {
        Ok(xla) => {
            eprintln!("XLA: {}", xla.description());
            found.push(Box::new(xla));
        }
        Err(e) => eprintln!("skipping XLA: {e}"),
    }
    found
}

fn compile(backend: &dyn Backend, program: &Program) -> Box<dyn Executable> {
    backend.compile(program).unwrap_or_else(|e| panic!("{}: {e}\n{}", backend.name(), program.text))
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

fn random(shape: &[usize], seed: u32) -> NdArray<f32> {
    NdArray::from_vec(noise(shape.iter().product(), seed), shape).unwrap()
}

fn assert_close(name: &str, got: &NdArray<f32>, want: &NdArray<f32>, tol: f32) {
    assert_eq!(got.shape(), want.shape(), "{name}: shape");
    for (i, (g, w)) in got.as_slice().iter().zip(want.as_slice()).enumerate() {
        assert!((g - w).abs() <= tol * (1.0 + w.abs()), "{name}[{i}]: {g} vs {w}");
    }
}

/// The step, with the cutoff as its parameter.
fn one_pole() -> Scan {
    Scan::trace(&[&[]], &[&[]], &[], |p, s, x| {
        let (s, y) = OnePole::lowpass(p[0], Elementwise::lit(FS)).tick(s[0], x);
        (vec![s], y)
    })
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

#[test]
fn every_primitive_matches_the_interpreter() {
    let shapes: &[&[usize]] = &[&[3, 8], &[8], &[8, 5], &[3, 1]];
    let graph = trace(shapes, |v| {
        let (a, b, m, c) = (v[0], v[1], v[2], v[3]);
        let e = (a * b + c).tanh() - (a.abs() + Tracer::lit(0.5)).sqrt().ln() / (b.exp() + Tracer::lit(1.0));
        let f = e.sin() * e.cos() + e.powf(Tracer::lit(2.0)).minimum(b.maximum(c));
        let g = Tracer::select(f.greater(a), f, -a) + Tracer::select(f.less(c), c, a);
        let (re, im) = g.rfft();
        let spectrum = Tracer::irfft(re * Tracer::lit(0.5), im, 8);
        vec![
            g,
            spectrum,
            re,
            im,
            g.dot(m),
            g.transpose(&[1, 0]).reshape(&[4, 6]),
            g.sum_axes(&[1]),
            g.mean_all(),
            b.broadcast_in_dim(&[8, 2], &[0]),
            a.dot_general(m, &[1], &[0]).dot_general(a, &[0], &[0]),
            a.slice(&[1, 1], &[3, 8], &[1, 3]),
            m.pad(&[1, 0], &[0, 2], &[2, 1]),
            Tracer::concatenate(&[a, c, a.slice_axis(1, 2, 4)], 1),
            frames(m.transpose(&[1, 0]), 3, 2),
            g.max_axes(&[1]),
            g.min_axes(&[0, 1]),
            c.prod_axes(&[0]),
            m.reverse(&[0, 1]),
            m.take((b * Tracer::lit(6.0)).abs()),
            b.take(Tracer::lit(3.5)),
            Tracer::fft_parts(a, g).0,
            Tracer::ifft_parts(g, a).1,
            (Tracer::complex(a, g).exp() * g.to_complex().conj() / Tracer::complex(g, a + Tracer::lit(2.0))).abs(),
            Tracer::irfft_complex(a.rfft_complex() * Tracer::complex(b.slice_axis(0, 0, 5), b.slice_axis(0, 3, 8)), 8),
            Tracer::complex(a, g).fft().dot(m.to_complex()).re(),
        ]
    });
    let inputs: Vec<NdArray<f32>> = shapes.iter().enumerate().map(|(k, s)| random(s, k as u32 + 1)).collect();
    let want = graph.eval(&inputs);
    for backend in backends() {
        let got = compile(&*backend, &graph.program()).run(&inputs).unwrap();
        for (k, (g, w)) in got.iter().zip(&want).enumerate() {
            assert_close(&format!("{} output {k}", backend.name()), g, w, 1e-4);
        }
    }
}

#[test]
fn gradients_of_array_programs_match_the_interpreter() {
    // gradient graphs are graphs: emit the backward pass of a program with every array primitive
    let shapes: &[&[usize]] = &[&[4, 8], &[8, 3], &[5]];
    let graph = trace(shapes, |v| {
        let (x, w, g) = (v[0], v[1], v[2]);
        let (re, im) = x.rfft();
        let y = Tracer::irfft(re * g, im * g, 8).dot(w).tanh();
        let framed = frames(Tracer::concatenate(&[x, x.pad(&[0, 1], &[0, 0], &[0, 1])], 1), 6, 4);
        let loss = (y * y).sum_all() + x.transpose(&[1, 0]).sum_axes(&[0]).mean_all() + (framed * framed).slice(&[1, 0, 0], &[4, 3, 6], &[2, 1, 2]).sum_all();
        let rows = Tracer::constant(&NdArray::from_vec(vec![0.0, 3.5, 3.0, 9.0, 1.2, 0.0], &[2, 3]).unwrap());
        let loss = loss + x.take(rows).max_axes(&[2]).sum_all() + w.reverse(&[0]).min_axes(&[1]).prod_axes(&[0]) * g.take(Tracer::lit(2.0));
        let z = Tracer::irfft_complex(x.rfft_complex() * Tracer::complex(g, g.sin()), 8);
        let loss = loss + (z * z).sum_all() + (Tracer::complex(x, x.cos()).ifft().conj() * x).abs().sum_all();
        let (fr, fi) = Tracer::fft_parts(x, x.reverse(&[1]));
        let (ir, ii) = Tracer::ifft_parts(fr * fi, fi);
        let loss = loss + (ir * ir + ii).sum_all();
        autodyne::flux::vjp(&[loss], &[Elementwise::lit(1.0)], v)
    });
    let inputs: Vec<NdArray<f32>> = shapes.iter().enumerate().map(|(k, s)| random(s, k as u32 + 10)).collect();
    let want = graph.eval(&inputs);
    for backend in backends() {
        let got = compile(&*backend, &graph.program()).run(&inputs).unwrap();
        for (k, (g, w)) in got.iter().zip(&want).enumerate() {
            assert_close(&format!("{} d input {k}", backend.name()), g, w, 1e-4);
        }
    }
}

#[test]
fn forward_scan_matches_the_concrete_filter() {
    let xs = noise(N, 1);
    for backend in backends() {
        let exe = compile(&*backend, &one_pole().forward_program(N));
        for cutoff in [200.0f32, 1_000.0, 8_000.0] {
            let out = exe.run(&[scalar(cutoff), vector(&xs), scalar(0.0)]).unwrap();
            let want = vector(&concrete(cutoff, &xs));
            assert_close(&format!("{} cutoff {cutoff}", backend.name()), &out[0], &want, 1e-5);
            assert!((out[1].as_slice()[0] - want.as_slice()[N - 1]).abs() <= 1e-5);
        }
    }
}

#[test]
fn scan_gradient_matches_finite_differences() {
    let xs = noise(N, 2);
    let targets = concrete(1_500.0, &xs);
    for backend in backends() {
        let exe = compile(&*backend, &one_pole().loss_grad_program(N));
        for cutoff in [300.0f32, 1_000.0, 4_000.0] {
            let out = exe.run(&[scalar(cutoff), vector(&xs), vector(&targets), scalar(0.0)]).unwrap();
            let (loss, grad) = (out[0].as_slice()[0], out[1].as_slice()[0]);
            let h = cutoff as f64 * 1e-4;
            let fd = (loss_f64(cutoff as f64 + h, &xs, &targets) - loss_f64(cutoff as f64 - h, &xs, &targets)) / (2.0 * h);
            assert!(((grad as f64 - fd) / fd).abs() < 1e-3, "{} cutoff {cutoff}: {grad} vs {fd}", backend.name());
            assert!(((loss as f64 - loss_f64(cutoff as f64, &xs, &targets)) / loss as f64).abs() < 1e-4);
        }
    }
}

#[test]
fn unrolled_scans_match_one_step_per_iteration() {
    // 7 steps per iteration over 512: 73 iterations and 1 step left over after the loop
    let xs = noise(N, 3);
    let targets = concrete(1_200.0, &xs);
    let inputs = [scalar(700.0), vector(&xs), vector(&targets), scalar(0.0)];
    for backend in backends() {
        let one = compile(&*backend, &one_pole().grad_program_with(N, &Loss::mse(&[N]), &Emit::f32().scan_unroll(1)));
        let seven = compile(&*backend, &one_pole().grad_program_with(N, &Loss::mse(&[N]), &Emit::f32().scan_unroll(7)));
        let (a, b) = (one.run(&inputs).unwrap(), seven.run(&inputs).unwrap());
        for (k, (x, y)) in a.iter().zip(&b).enumerate() {
            assert_close(&format!("{} output {k}", backend.name()), y, x, 1e-6);
        }
        let forward = compile(&*backend, &one_pole().forward_program_with(N, &Emit::f32().scan_unroll(7)));
        let out = forward.run(&[scalar(700.0), vector(&xs), scalar(0.0)]).unwrap();
        assert_close(&format!("{} unrolled forward", backend.name()), &out[0], &vector(&concrete(700.0, &xs)), 1e-5);
    }
}

#[test]
fn shaped_scans_match_the_interpreter() {
    // a bank of four one-poles, and a state space model with dot products
    let bank = Scan::trace(&[&[4]], &[&[4]], &[4], |p, s, x| {
        let (s, y) = OnePole::lowpass(p[0], Elementwise::lit(FS)).tick(s[0], x);
        (vec![s], y)
    });
    let ss = Scan::trace(&[&[2, 2], &[2], &[2]], &[&[2]], &[], |p, s, x| {
        let next = p[0].dot(s[0]) + p[1] * x;
        (vec![next], p[2].dot(next))
    });
    let cases = [
        (bank, vec![vector(&[200.0, 1_000.0, 3_000.0, 9_000.0])], random(&[64, 4], 3), random(&[64, 4], 4), vec![vector(&[0.0; 4])]),
        (
            ss,
            vec![NdArray::from_vec(vec![0.6, -0.3, 0.2, 0.5], &[2, 2]).unwrap(), vector(&[1.0, 0.5]), vector(&[0.3, -0.7])],
            random(&[64], 5),
            random(&[64], 6),
            vec![vector(&[0.1, -0.2])],
        ),
    ];
    for backend in backends() {
        for (k, (scan, params, xs, targets, s0)) in cases.iter().enumerate() {
            let name = format!("{} scan {k}", backend.name());
            let (ys, last) = scan.run(params, xs, s0);
            let forward = compile(&*backend, &scan.forward_program(xs.shape()[0]));
            let out = forward.run(&[params.clone(), vec![xs.clone()], s0.clone()].concat()).unwrap();
            assert_close(&format!("{name} ys"), &out[0], &ys, 1e-5);
            assert_close(&format!("{name} final state"), &out[1], &last[0], 1e-5);

            let want = scan.loss_grad(params, xs, targets, s0);
            let grad = compile(&*backend, &scan.loss_grad_program(xs.shape()[0]));
            let out = grad.run(&[params.clone(), vec![xs.clone(), targets.clone()], s0.clone()].concat()).unwrap();
            assert!((out[0].as_slice()[0] - want.loss).abs() <= 1e-5 * (1.0 + want.loss), "{name} loss");
            for (p, w) in out[1..1 + params.len()].iter().zip(&want.params) {
                assert_close(&format!("{name} d param"), p, w, 1e-4);
            }
            assert_close(&format!("{name} d state"), &out[1 + params.len()], &want.state[0], 1e-4);
            assert_close(&format!("{name} d xs"), &out[2 + params.len()], &want.input, 1e-4);
        }
    }
}

/// One generic function: a spectral gain, then a dense layer, then a soft clip. Written once over
/// `RealArrayMath`, run eagerly on `NdArray` and traced for the backends.
fn model<A: RealArrayMath>(x: A, gain: A, w: A) -> A {
    let n = *x.shape().last().unwrap();
    let (re, im) = x.rfft();
    let y = A::irfft(re * gain.clone(), im * gain, n).dot(w);
    (y.clone() * A::lit(0.5)).tanh() + y.minimum(A::lit(0.25))
}

#[test]
fn one_generic_function_runs_eagerly_and_compiled() {
    let (x, gain, w) = (random(&[4, 16], 21), random(&[9], 22), random(&[16, 3], 23));
    let eager = model(x.clone(), gain.clone(), w.clone());
    let graph = trace(&[&[4, 16], &[9], &[16, 3]], |v| vec![model(v[0], v[1], v[2])]);
    let inputs = [x, gain, w];
    assert_close("interpreter", &graph.eval(&inputs)[0], &eager, 1e-6);
    for backend in backends() {
        let got = compile(&*backend, &graph.program()).run(&inputs).unwrap();
        assert_close(backend.name(), &got[0], &eager, 1e-4);
    }
}

/// Adam on `params` (flattened), `steps` at most, until `done`.
fn adam(mut params: Vec<f32>, lr: f32, steps: i32, mut grad: impl FnMut(&[f32]) -> (f32, Vec<f32>), done: impl Fn(&[f32]) -> bool) -> (Vec<f32>, f32, f32, i32) {
    let (b1, b2) = (0.9f32, 0.999f32);
    let (mut m, mut v) = (vec![0.0; params.len()], vec![0.0; params.len()]);
    let mut first = None;
    let mut last = 0.0;
    let mut taken = 0;
    for step in 1..=steps {
        let (loss, g) = grad(&params);
        first.get_or_insert(loss);
        last = loss;
        taken = step;
        if done(&params) {
            break;
        }
        for i in 0..params.len() {
            m[i] = b1 * m[i] + (1.0 - b1) * g[i];
            v[i] = b2 * v[i] + (1.0 - b2) * g[i] * g[i];
            let (mh, vh) = (m[i] / (1.0 - b1.powi(step)), v[i] / (1.0 - b2.powi(step)));
            params[i] -= lr * mh / (vh.sqrt() + 1e-12);
        }
    }
    (params, first.unwrap(), last, taken)
}

#[test]
fn fits_the_cutoff_by_gradient_descent() {
    // optimise the log of the cutoff, so steps are relative
    let scan = Scan::trace(&[&[]], &[&[]], &[], |p, s, x| {
        let (s, y) = OnePole::lowpass(p[0].exp(), Elementwise::lit(FS)).tick(s[0], x);
        (vec![s], y)
    });
    let xs = vector(&noise(N, 3));
    let target_cutoff = 1_200.0f32;
    let targets = vector(&concrete(target_cutoff, xs.as_slice()));
    for backend in backends() {
        let exe = compile(&*backend, &scan.loss_grad_program(N));
        let (u, first, last, steps) = adam(
            vec![300.0f32.ln()],
            0.1,
            300,
            |u| {
                let out = exe.run(&[scalar(u[0]), xs.clone(), targets.clone(), scalar(0.0)]).unwrap();
                (out[0].as_slice()[0], vec![out[1].as_slice()[0]])
            },
            |u| ((u[0].exp() - target_cutoff) / target_cutoff).abs() < 1e-3,
        );
        let fitted = u[0].exp();
        eprintln!("{}: fitted {fitted:.1} Hz (target {target_cutoff} Hz, start 300 Hz) in {steps} steps, loss {first} -> {last:e}", backend.name());
        assert!(((fitted - target_cutoff) / target_cutoff).abs() < 1e-2, "{}: fitted {fitted} Hz", backend.name());
        assert!(last < first * 1e-3);
    }
}

#[test]
fn fits_a_spectral_gain_frame_by_frame() {
    // frames of 16 samples through rfft -> per-bin gain -> irfft; recover the gains
    let (n, m, frames) = (16usize, 9usize, 32usize);
    let scan = Scan::trace(&[&[m]], &[], &[n], |p, _, x| {
        let (re, im) = x.rfft();
        (vec![], Tracer::irfft(re * p[0], im * p[0], n))
    });
    let xs = random(&[frames, n], 8);
    let true_gains: Vec<f32> = (0..m).map(|k| 1.0 / (1.0 + k as f32 * 0.4)).collect();
    let (targets, _) = scan.run(&[vector(&true_gains)], &xs, &[]);
    for backend in backends() {
        let exe = compile(&*backend, &scan.loss_grad_program(frames));
        let (gains, first, last, steps) = adam(
            vec![0.5; m],
            0.05,
            400,
            |g| {
                let out = exe.run(&[vector(g), xs.clone(), targets.clone()]).unwrap();
                (out[0].as_slice()[0], out[1].as_slice().to_vec())
            },
            |g| g.iter().zip(&true_gains).all(|(a, b)| (a - b).abs() < 1e-3),
        );
        eprintln!("{}: spectral gains in {steps} steps, loss {first} -> {last:e}", backend.name());
        for (k, (g, t)) in gains.iter().zip(&true_gains).enumerate() {
            assert!((g - t).abs() < 1e-2, "{} bin {k}: {g} vs {t}", backend.name());
        }
    }
}

#[test]
fn spectral_loss_gradient_program_matches_the_interpreter() {
    // a filter chain scored by a multi-resolution STFT loss plus MSE; every gradient, d xs included
    let scan = Scan::trace(&[&[], &[]], &[&[]], &[], |p, s, x| {
        let (s, y) = OnePole::lowpass(p[0].exp(), Elementwise::lit(FS)).tick(s[0], x);
        (vec![s], (y * p[1]).tanh())
    });
    let len = 256;
    let resolutions = [StftResolution::overlapping(64), StftResolution::new(32, 8, 24)];
    let loss = Loss::trace(&[len], &[&[len]], |y, aux| {
        let d = y - aux[0];
        multi_resolution_stft(y, aux[0], &resolutions) + (d * d).mean_all()
    });
    let xs = random(&[len], 31);
    let (targets, _) = scan.run(&[scalar(2_000f32.ln()), scalar(2.0)], &xs, &[scalar(0.0)]);
    let params = [scalar(600f32.ln()), scalar(1.0)];
    let want = scan.grad(&params, &xs, &[scalar(0.0)], &loss, std::slice::from_ref(&targets));
    for backend in backends() {
        let exe = compile(&*backend, &scan.grad_program(len, &loss));
        let out = exe.run(&[params[0].clone(), params[1].clone(), xs.clone(), targets.clone(), scalar(0.0)]).unwrap();
        let name = backend.name();
        assert!((out[0].as_slice()[0] - want.loss).abs() <= 1e-4 * (1.0 + want.loss), "{name} loss {} vs {}", out[0].as_slice()[0], want.loss);
        assert_close(&format!("{name} d cutoff"), &out[1], &want.params[0], 1e-3);
        assert_close(&format!("{name} d gain"), &out[2], &want.params[1], 1e-3);
        assert_close(&format!("{name} d state"), &out[3], &want.state[0], 1e-3);
        assert_close(&format!("{name} d xs"), &out[4], &want.input, 1e-3);
    }
}
#[test]
fn fits_an_eq_and_drive_to_a_recording_with_the_stft_loss() {
    // a peaking EQ into tanh drive: recover the centre frequency (log Hz), the gain (in units of
    // 10 dB) and the drive from the target's spectrogram alone (no sample-by-sample error)
    let scan = Scan::trace(&[&[], &[], &[]], &[&[], &[]], &[], |p, s, x| {
        let c = BiquadCoeffs::design(BiquadKind::Peaking, p[0].exp(), Tracer::lit(1.0), p[1] * Tracer::lit(10.0), Tracer::lit(FS));
        let ([a, b], y) = c.tick([s[0], s[1]], x);
        (vec![a, b], Shape::Tanh.apply(y * p[2]))
    });
    let len = 2048;
    let loss = Loss::stft(&[len], &[StftResolution::overlapping(512), StftResolution::overlapping(128), StftResolution::overlapping(32)]);
    let xs = NdArray::from_vec(noise(len, 41).iter().map(|v| 0.5 * v).collect(), &[len]).unwrap();
    let s0 = [scalar(0.0), scalar(0.0)];
    let truth = [3_000f32.ln(), 0.9, 2.0];
    let (target, _) = scan.run(&truth.map(scalar), &xs, &s0);
    for backend in backends() {
        // with the backend's limits: IREE's Vulkan backend gets its long FFTs built from short ones
        let exe = compile(&*backend, &scan.grad_program_with(len, &loss, &Emit::f32().for_backend(&*backend)));
        let mut params = vec![scalar(1_000f32.ln()), scalar(0.3), scalar(1.0)];
        if matches!(backend.name(), "iree-vulkan" | "iree-cuda" | "iree-rocm" | "iree-metal") {
            // IREE drives a scan from the host on GPUs (seconds per gradient here): check one
            let out = exe.run(&[params.clone(), vec![xs.clone(), target.clone()], s0.to_vec()].concat()).unwrap();
            let want = scan.grad(&params, &xs, &s0, &loss, std::slice::from_ref(&target));
            assert!((out[0].as_slice()[0] - want.loss).abs() < 1e-3 * want.loss, "{}: loss", backend.name());
            for (k, (g, w)) in out[1..4].iter().zip(&want.params).enumerate() {
                assert_close(&format!("{} d param {k}", backend.name()), g, w, 1e-2);
            }
            continue;
        }
        let mut adam = Adam::new(0.03);
        let mut losses = Vec::new();
        for _ in 0..300 {
            let out = exe.run(&[params.clone(), vec![xs.clone(), target.clone()], s0.to_vec()].concat()).unwrap();
            losses.push(out[0].as_slice()[0]);
            adam.step(&mut params, &out[1..4]);
        }
        let [f, g, d] = [0, 1, 2].map(|k| params[k].as_slice()[0]);
        eprintln!("{}: {:.0} Hz, {:.2} dB, drive {d:.3}; loss {} -> {}", backend.name(), f.exp(), 10.0 * g, losses[0], losses[losses.len() - 1]);
        assert!((f.exp() / 3_000.0 - 1.0).abs() < 0.05, "{}: centre {} Hz", backend.name(), f.exp());
        assert!((10.0 * g - 9.0).abs() < 0.5, "{}: gain {} dB", backend.name(), 10.0 * g);
        assert!((d - 2.0).abs() < 0.1, "{}: drive {d}", backend.name());
    }
}
#[test]
fn iree_modules_are_saved_and_loaded() {
    // ahead of time: compile to a .vmfb file, then load and run it without the program text
    let Some(iree) = Iree::find() else {
        eprintln!("skipping: IREE tools not found");
        return;
    };
    let program = one_pole().forward_program(N);
    let path = std::env::temp_dir().join(format!("autodyne-one-pole-{}.vmfb", std::process::id()));
    iree.compile_to(&program, &path).unwrap();
    let exe = iree.load(&path, &program).unwrap();
    let xs = noise(N, 9);
    let out = exe.run(&[scalar(700.0), vector(&xs), scalar(0.0)]).unwrap();
    assert_close("loaded module", &out[0], &vector(&concrete(700.0, &xs)), 1e-5);
    assert!(exe.run(&[scalar(700.0), vector(&xs[..N - 1]), scalar(0.0)]).is_err(), "shapes are checked");
    std::fs::remove_file(&path).unwrap();
    assert!(iree.load(&path, &program).is_err());
}
#[test]
fn resident_arrays_run_like_host_arrays() {
    let scan = one_pole();
    let len = 1 << 12;
    let xs = vector(&noise(len, 51));
    let targets = vector(&concrete(1_500.0, xs.as_slice()));
    for backend in backends() {
        let name = backend.name();
        let grad = compile(&*backend, &scan.loss_grad_program(len));
        // the signal and target are uploaded once; the parameter every run
        let (dxs, dtargets, ds0) = (grad.upload(&xs).unwrap(), grad.upload(&targets).unwrap(), grad.upload(&scalar(0.0)).unwrap());
        assert_eq!(dxs.shape(), [len]);
        let start = std::time::Instant::now();
        for cutoff in [300.0f32, 1_000.0, 4_000.0] {
            let host = grad.run(&[scalar(cutoff), xs.clone(), targets.clone(), scalar(0.0)]).unwrap();
            let p = grad.upload(&scalar(cutoff)).unwrap();
            let out = grad.run_resident(&[&p, &dxs, &dtargets, &ds0]).unwrap();
            assert_eq!(out.len(), host.len());
            for (k, (o, h)) in out.iter().zip(&host).enumerate() {
                assert_close(&format!("{name} cutoff {cutoff} output {k}"), &grad.download(o).unwrap(), h, 1e-6);
            }
        }
        let both = start.elapsed();
        // outputs feed later runs without leaving the device: filter twice
        let forward = compile(&*backend, &scan.forward_program(len));
        let (p, s0) = (forward.upload(&scalar(2_000.0)).unwrap(), forward.upload(&scalar(0.0)).unwrap());
        let x = forward.upload(&xs).unwrap();
        let once = forward.run_resident(&[&p, &x, &s0]).unwrap();
        let twice = forward.run_resident(&[&p, &once[0], &s0]).unwrap();
        let want = concrete(2_000.0, &concrete(2_000.0, xs.as_slice()));
        assert_close(&format!("{name} chained"), &forward.download(&twice[0]).unwrap(), &vector(&want), 1e-5);
        // shapes are checked, and arrays belong to their backend
        assert!(forward.run_resident(&[&p, &s0, &s0]).is_err());
        eprintln!("{name}: resident arrays {} (3 host + 3 resident gradient runs of {len} samples in {both:?})", if dxs.is_on_device() { "on the device" } else { "on the host" });
    }
}
#[test]
fn double_precision_programs_run_on_every_backend() {
    let xs: Vec<f64> = noise(N, 61).iter().map(|&v| v as f64).collect();
    let mut lp = OnePole::lowpass(900.0f64, FS);
    let mut want = xs.clone();
    lp.process(&mut want);
    let f64s = |v: &[f64], shape: &[usize]| NdArray::<f64>::array(v, shape);
    let (p, x, s0) = (f64s(&[900.0], &[]), f64s(&xs, &[N]), f64s(&[0.0], &[]));
    let targets = f64s(&want.iter().map(|v| v * 0.9).collect::<Vec<_>>(), &[N]);
    let interpreted = one_pole().loss_grad(std::slice::from_ref(&p), &x, &targets, std::slice::from_ref(&s0));
    for backend in backends() {
        let name = backend.name();
        // written for the backend: IREE's CPU and Vulkan targets get f64 maths out of arithmetic
        let emit = Emit::f64().for_backend(&*backend);
        let forward = compile(&*backend, &one_pole().forward_program_with(N, &emit));
        let out = forward.run(&[p.clone(), x.clone(), s0.clone()]).unwrap();
        for (i, (g, w)) in out[0].as_slice().iter().zip(&want).enumerate() {
            assert!((g - w).abs() < 1e-12, "{name} sample {i}: {g} vs {w}");
        }
        // f32 data is refused by an f64 program
        assert!(forward.run(&[scalar(900.0), vector(&[0.0; N]), scalar(0.0)]).is_err(), "{name}");
        let grad = compile(&*backend, &one_pole().grad_program_with(N, &Loss::mse(&[N]), &emit));
        let out = grad.run(&[p.clone(), x.clone(), targets.clone(), s0.clone()]).unwrap();
        assert!((out[0].as_slice()[0] - interpreted.loss).abs() < 1e-12 * (1.0 + interpreted.loss), "{name} loss");
        assert!((out[1].as_slice()[0] - interpreted.params[0].as_slice()[0]).abs() < 1e-9 * (1.0 + interpreted.params[0].as_slice()[0].abs()), "{name} gradient");
        let resident = grad.upload(&p).unwrap();
        assert_eq!(resident.dtype(), autodyne::units::DType::F64);
    }
}
#[test]
fn long_ffts_built_from_short_ones_match_the_interpreter() {
    // 8-point FFTs at most: 16 = 8 x 2, 64 = 8 x 8, 256 = 8 x 32 (32 = 8 x 4), and, where FFTs of
    // any length compile (not IREE: powers of two only), 12 = 6 x 2, 30 = 6 x 5 and 11 (prime, whole)
    for n in [16usize, 64, 256, 12, 30, 11] {
        let m = n / 2 + 1;
        let graph = trace(&[&[3, n], &[3, m]], |v| {
            let z = Tracer::complex(v[0], v[0].sin());
            let spectrum = v[0].rfft_complex();
            let shaped = Tracer::irfft_complex(spectrum * Tracer::complex(v[1], v[1].cos()), n);
            let loss = (shaped * shaped).sum_all() + z.fft().abs().sum_all();
            let mut out = vec![spectrum.re(), spectrum.im(), shaped, z.fft().re(), z.ifft().im()];
            out.extend(vjp(&[loss], &[Tracer::lit(1.0)], v));
            out
        });
        let program = graph.program_with(&Emit::f32().max_fft(8));
        if n != 11 {
            assert!(!program.text.contains(&format!("length = [{n}]")), "n = {n} still has a whole FFT");
        }
        let inputs = [random(&[3, n], n as u32), random(&[3, m], n as u32 + 1)];
        let want = graph.eval(&inputs);
        for backend in backends() {
            if !n.is_power_of_two() && backend.name().starts_with("iree") {
                continue;
            }
            let got = compile(&*backend, &program).run(&inputs).unwrap();
            for (k, (g, w)) in got.iter().zip(&want).enumerate() {
                assert_close(&format!("{} n {n} output {k}", backend.name()), g, w, 2e-4);
            }
        }
    }
}
#[test]
fn f64_maths_written_out_of_arithmetic_match_the_cpu() {
    // exp, log, sin, cos, tanh and pow over wide ranges, specials included, on every backend, with
    // and without the arithmetic versions (a backend with its own f64 maths runs both)
    let n = 64;
    let x: Vec<f64> = (0..n).map(|i| -30.0 + 60.0 * i as f64 / (n - 1) as f64 + 0.123).collect();
    let mut pos: Vec<f64> = (0..n - 6).map(|i| 10f64.powf(-300.0 + 600.0 * i as f64 / (n - 7) as f64)).collect();
    // (no subnormals: IREE's CPU runtime flushes them to zero)
    pos.extend([0.0, -1.0, f64::INFINITY, 1.0, 1e-307, 2.2250738585072014e-308]);
    let big: Vec<f64> = (0..n).map(|i| (i as f64 - 32.0) * 1234.567).collect();
    let graph = trace(&[&[n], &[n], &[n]], |v| {
        let (x, p, b) = (v[0], v[1], v[2]);
        vec![x.exp(), p.ln(), b.sin(), b.cos(), x.tanh(), p.powf(x * Tracer::lit(0.1))]
    });
    let inputs = [NdArray::<f64>::array(&x, &[n]), NdArray::<f64>::array(&pos, &[n]), NdArray::<f64>::array(&big, &[n])];
    let want: [Vec<f64>; 6] = [
        x.iter().map(|v| v.exp()).collect(),
        pos.iter().map(|v| v.ln()).collect(),
        big.iter().map(|v| v.sin()).collect(),
        big.iter().map(|v| v.cos()).collect(),
        x.iter().map(|v| v.tanh()).collect(),
        pos.iter().zip(&x).map(|(p, x)| p.powf(x * 0.1)).collect(),
    ];
    // the arguments each function scales its error by on Vulkan (see below)
    let args = [&x, &pos, &big, &big, &x, &pos];
    for backend in backends() {
        // Vulkan drivers reassociate f64 arithmetic, folding the parts of the range reductions
        // together: sin and cos are then accurate to ulps of their argument, not of their value
        let reordered = backend.name() == "iree-vulkan";
        for soft in [true, false] {
            let emit = Emit::f64().for_backend(&*backend).soft_f64(soft || backend.soft_f64());
            let program = graph.program_with(&emit);
            let got = compile(&*backend, &program).run(&inputs).unwrap();
            for (k, (g, w)) in got.iter().zip(&want).enumerate() {
                // the worst error in units of the last place (specials must match exactly)
                let (mut worst, mut at) = (0.0f64, 0);
                for (i, (g, w)) in g.as_slice().iter().zip(w).enumerate() {
                    if g.is_nan() || w.is_nan() || w.is_infinite() || *w == 0.0 {
                        assert!((g.is_nan() && w.is_nan()) || g == w || (w.abs() < 1e-300 && g.abs() < 1e-300), "{} (soft {soft}) function {k} element {i}: {g:e} vs {w:e}", backend.name());
                        continue;
                    }
                    // (pow's double-double steps collapse too: its error grows with |ln result|)
                    let scale = match k {
                        2 | 3 if reordered => w.abs().max(args[k][i].abs()),
                        5 if reordered => w.abs() * (1.0 + w.abs().ln().abs()),
                        _ => w.abs(),
                    };
                    let ulp = f64::EPSILON * scale.max(f64::MIN_POSITIVE);
                    if (g - w).abs() / ulp > worst {
                        (worst, at) = ((g - w).abs() / ulp, i);
                    }
                }
                eprintln!("{} (soft {soft}) function {k}: within {worst:.1} ulp", backend.name());
                // GPUs may contract or reorder the range reduction (StableHLO has no fused multiply-add)
                assert!(worst <= 16.0, "{} (soft {soft}) function {k}: {worst} ulp at element {at}: {:e} vs {:e}", backend.name(), g.as_slice()[at], w[at]);
            }
        }
    }
}
