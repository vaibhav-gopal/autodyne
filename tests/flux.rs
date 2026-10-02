//! flux end to end: traced programs emitted as StableHLO, compiled and run by each backend found
//! (IREE, PJRT, XLA), and compared with the interpreter and with the concrete f32 code.
//!
//! IREE needs `iree-compile` and `iree-run-module` (`pip install iree-base-compiler
//! iree-base-runtime`) in `AUTODYNE_IREE_DIR` or on `PATH`; XLA needs Python with `jax`
//! (`AUTODYNE_XLA_PYTHON`, else `python3` / `python` on `PATH`); PJRT needs a plugin library in
//! `AUTODYNE_PJRT_PLUGIN`. A missing backend is reported and skipped; with none, the tests pass
//! without running. `AUTODYNE_IREE_GPU` adds IREE on GPUs: a comma-separated list of `vulkan`,
//! `cuda` or `rocm`, each with an optional architecture (`vulkan=ampere,cuda=sm_80`; the FFTs need
//! one for Vulkan, and ROCm always does).

#![cfg(feature = "flux")]

use autodyne::distortion::Shape;
use autodyne::filter::{BiquadCoeffs, BiquadKind, OnePole};
use autodyne::flux::optim::{Adam, Optimizer};
use autodyne::flux::{frames, multi_resolution_stft, scalar, trace, vector, Backend, Executable, Iree, IreeTarget, Loss, Pjrt, Program, Scan, StftResolution, Tracer, Xla};
use autodyne::signal::{ArrayMath, NdArray, RealArrayMath};
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
        Some(path) => match Pjrt::load(std::path::Path::new(&path)) {
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
        let c = BiquadCoeffs::design(BiquadKind::Peaking, p[0].exp(), Tracer::lit(FS), Tracer::lit(1.0), p[1] * Tracer::lit(10.0));
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
        let program = scan.grad_program(len, &loss);
        let exe = match backend.compile(&program) {
            // IREE 3.11 cannot compile FFTs of 128 points or more for Vulkan
            Err(e) if backend.name() == "iree-vulkan" && e.to_string().contains("fft") => {
                eprintln!("skipping iree-vulkan: {}", e.to_string().lines().next().unwrap_or_default());
                continue;
            }
            result => result.unwrap_or_else(|e| panic!("{}: {e}", backend.name())),
        };
        let mut params = vec![scalar(1_000f32.ln()), scalar(0.3), scalar(1.0)];
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
    let exe = iree.load(&path, &program.inputs, &program.outputs).unwrap();
    let xs = noise(N, 9);
    let out = exe.run(&[scalar(700.0), vector(&xs), scalar(0.0)]).unwrap();
    assert_close("loaded module", &out[0], &vector(&concrete(700.0, &xs)), 1e-5);
    assert!(exe.run(&[scalar(700.0), vector(&xs[..N - 1]), scalar(0.0)]).is_err(), "shapes are checked");
    std::fs::remove_file(&path).unwrap();
    assert!(iree.load(&path, &program.inputs, &program.outputs).is_err());
}