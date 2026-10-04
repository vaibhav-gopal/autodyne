//! Writes the flux benchmark models for `bench/flux/compare_jax.py`: for each, the StableHLO
//! program, its inputs and the interpreter's outputs (`.npy`), and how long tracing and emitting
//! took.
//!
//! ```text
//! cargo run --release --features flux --example flux_programs -- <out dir> [scan steps per loop iteration]
//! ```

use std::io::Write;
use std::path::Path;
use std::time::Instant;

use autodyne::distortion::Shape;
use autodyne::filter::{BiquadCoeffs, BiquadKind, OnePole};
use autodyne::flux::{scalar, trace, vector, vjp, Emit, Loss, Program, Scan, StftResolution, Tracer};
use autodyne::signal::{ArrayMath, NdArray, RealArrayMath};
use autodyne::units::Elementwise;

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

fn npy(path: &Path, a: &NdArray<f32>) {
    let shape = match a.shape() {
        [] => "()".to_string(),
        [n] => format!("({n},)"),
        dims => format!("({})", dims.iter().map(usize::to_string).collect::<Vec<_>>().join(", ")),
    };
    let mut header = format!("{{'descr': '<f4', 'fortran_order': False, 'shape': {shape}, }}");
    let total = (10 + header.len() + 1).div_ceil(64) * 64;
    header.extend(std::iter::repeat_n(' ', total - 10 - header.len() - 1));
    header.push('\n');
    let mut out = b"\x93NUMPY\x01\x00".to_vec();
    out.extend_from_slice(&(header.len() as u16).to_le_bytes());
    out.extend_from_slice(header.as_bytes());
    a.as_slice().iter().for_each(|v| out.extend_from_slice(&v.to_le_bytes()));
    std::fs::write(path, out).unwrap();
}

struct Case {
    name: &'static str,
    program: Program,
    inputs: Vec<NdArray<f32>>,
    outputs: Vec<NdArray<f32>>,
    build_ms: f64,
}

fn write(dir: &Path, case: &Case, index: &mut impl Write) {
    std::fs::write(dir.join(format!("{}.mlir", case.name)), &case.program.text).unwrap();
    for (k, x) in case.inputs.iter().enumerate() {
        npy(&dir.join(format!("{}_in{k}.npy", case.name)), x);
    }
    for (k, y) in case.outputs.iter().enumerate() {
        npy(&dir.join(format!("{}_out{k}.npy", case.name)), y);
    }
    writeln!(index, "{}\t{}\t{}\t{:.3}", case.name, case.inputs.len(), case.outputs.len(), case.build_ms).unwrap();
}

fn one_pole() -> Scan {
    Scan::trace(&[&[]], &[&[]], &[], |p, s, x| {
        let (s, y) = OnePole::lowpass(p[0], Elementwise::lit(FS)).tick(s[0], x);
        (vec![s], y)
    })
}

fn main() {
    let dir = std::env::args().nth(1).expect("usage: flux_programs <out dir>");
    let dir = Path::new(&dir);
    // scans: steps per loop iteration (1: one)
    let unroll: usize = std::env::args().nth(2).map_or(1, |u| u.parse().expect("a number of steps"));
    let emit = Emit::f32().scan_unroll(unroll);
    std::fs::create_dir_all(dir).unwrap();
    let mut index = std::fs::File::create(dir.join("cases.tsv")).unwrap();
    let mut cases = Vec::new();

    // 1. a one-pole low-pass over a second of audio
    let n = 48_000;
    let xs = vector(&noise(n, 1));
    let start = Instant::now();
    let scan = one_pole();
    let program = scan.forward_program_with(n, &emit);
    let build_ms = start.elapsed().as_secs_f64() * 1e3;
    let inputs = vec![scalar(1_000.0), xs.clone(), scalar(0.0)];
    let (ys, last) = scan.run(&[scalar(1_000.0)], &xs, &[scalar(0.0)]);
    cases.push(Case { name: "one_pole_forward", program, inputs, outputs: vec![ys, last[0].clone()], build_ms });

    // 2. its MSE gradient (cutoff, initial state and input)
    let (targets, _) = scan.run(&[scalar(1_500.0)], &xs, &[scalar(0.0)]);
    let start = Instant::now();
    let program = one_pole().grad_program_with(n, &Loss::mse(&[n]), &emit);
    let build_ms = start.elapsed().as_secs_f64() * 1e3;
    let g = scan.loss_grad(&[scalar(1_000.0)], &xs, &targets, &[scalar(0.0)]);
    let outputs = vec![scalar(g.loss), g.params[0].clone(), g.state[0].clone(), g.input.clone()];
    let inputs = vec![scalar(1_000.0), xs.clone(), targets, scalar(0.0)];
    cases.push(Case { name: "one_pole_grad", program, inputs: inputs.clone(), outputs: outputs.clone(), build_ms });
    // the same gradient recomputing each step instead of saving residuals
    let start = Instant::now();
    let program = one_pole().checkpointed(true).grad_program_with(n, &Loss::mse(&[n]), &emit);
    let build_ms = start.elapsed().as_secs_f64() * 1e3;
    cases.push(Case { name: "one_pole_grad_checkpointed", program, inputs, outputs, build_ms });

    // 3. peaking EQ into tanh drive, multi-resolution STFT loss gradient
    let n = 2_048;
    let xs = NdArray::from_vec(noise(n, 41).iter().map(|v| 0.5 * v).collect(), &[n]).unwrap();
    let start = Instant::now();
    let chain = Scan::trace(&[&[], &[], &[]], &[&[], &[]], &[], |p, s, x| {
        let c = BiquadCoeffs::design(BiquadKind::Peaking, p[0].exp(), Tracer::lit(1.0), p[1] * Tracer::lit(10.0), Tracer::lit(FS));
        let ([a, b], y) = c.tick([s[0], s[1]], x);
        (vec![a, b], Shape::Tanh.apply(y * p[2]))
    });
    let loss = Loss::stft(&[n], &[StftResolution::overlapping(512), StftResolution::overlapping(128), StftResolution::overlapping(32)]);
    let program = chain.grad_program_with(n, &loss, &emit);
    let build_ms = start.elapsed().as_secs_f64() * 1e3;
    let s0 = [scalar(0.0), scalar(0.0)];
    let (target, _) = chain.run(&[scalar(3_000f32.ln()), scalar(0.9), scalar(2.0)], &xs, &s0);
    let params = [scalar(1_000f32.ln()), scalar(0.3), scalar(1.0)];
    let g = chain.grad(&params, &xs, &s0, &loss, std::slice::from_ref(&target));
    let mut outputs = vec![scalar(g.loss)];
    outputs.extend(g.params);
    outputs.extend(g.state);
    outputs.push(g.input);
    let inputs = [params.to_vec(), vec![xs, target], s0.to_vec()].concat();
    cases.push(Case { name: "eq_drive_stft_grad", program, inputs: inputs.clone(), outputs: outputs.clone(), build_ms });
    let start = Instant::now();
    let program = chain.clone().checkpointed(true).grad_program_with(n, &loss, &emit);
    let build_ms = start.elapsed().as_secs_f64() * 1e3;
    cases.push(Case { name: "eq_drive_stft_grad_checkpointed", program, inputs, outputs, build_ms });

    // 4. a batch through rfft -> per-bin gain -> irfft -> dense -> tanh; gradient of the energy
    let (batch, len, width) = (256usize, 1_024usize, 64usize);
    let start = Instant::now();
    let model = trace(&[&[batch, len], &[len / 2 + 1], &[len, width]], |v| {
        let (x, gain, w) = (v[0], v[1], v[2]);
        let y = Tracer::irfft_complex(x.rfft_complex() * gain, len).dot(w).tanh();
        let loss = (y * y).mean_all();
        let d = vjp(&[loss], &[Tracer::lit(1.0)], &[gain, w]);
        vec![loss, d[0], d[1]]
    });
    let program = model.program();
    let build_ms = start.elapsed().as_secs_f64() * 1e3;
    let inputs = vec![
        NdArray::from_vec(noise(batch * len, 7), &[batch, len]).unwrap(),
        NdArray::from_vec(noise(len / 2 + 1, 8).iter().map(|v| 1.0 + 0.5 * v).collect(), &[len / 2 + 1]).unwrap(),
        NdArray::from_vec(noise(len * width, 9).iter().map(|v| v * 0.05).collect(), &[len, width]).unwrap(),
    ];
    let outputs = model.eval(&inputs);
    cases.push(Case { name: "spectral_model_grad", program, inputs, outputs, build_ms });

    for case in &cases {
        write(dir, case, &mut index);
        println!("{}: traced and emitted in {:.2} ms ({} lines of StableHLO)", case.name, case.build_ms, case.program.text.lines().count());
    }
}
