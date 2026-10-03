//! flux against Burn's autodiff on the models of `bench/flux`: the same inputs (the `.npy` files
//! `examples/flux_programs.rs` writes), the results checked against flux's interpreter, times printed
//! as a Markdown table.
//!
//! ```text
//! cargo run --release --example flux_programs --features flux -- DIR     (in the autodyne root)
//! cargo run --release --example flux_burn -- DIR                          (here)
//! ```
//!
//! Burn has no compiled loop (scan): a recurrence is one small tensor op per sample, recorded on its
//! autodiff tape. Its `rfft` / `irfft` have no autodiff rules yet (`todo!`), so the spectral model's
//! transforms are matrix products with DFT matrices, which is what a Burn user writes today.

use std::f64::consts::TAU;
use std::path::Path;
use std::time::{Duration, Instant};

use autodyne::filter::OnePole;
use autodyne::flux::{scalar, trace, vector, vjp, Scan, Tracer};
use autodyne::signal::{ArrayMath, NdArray, RealArrayMath};
use autodyne::units::Elementwise;
use burn::backend::{Autodiff, Flex, NdArray as BurnNdArray};
use burn::tensor::backend::{AutodiffBackend, Backend};
use burn::tensor::{Tensor, TensorData};

const FS: f64 = 48_000.0;

fn read_npy(path: &Path) -> Vec<f32> {
    let bytes = std::fs::read(path).unwrap_or_else(|e| panic!("{}: {e}", path.display()));
    let header = u16::from_le_bytes([bytes[8], bytes[9]]) as usize;
    assert!(std::str::from_utf8(&bytes[10..10 + header]).unwrap().contains("'<f4'"), "f32 arrays");
    bytes[10 + header..].as_chunks::<4>().0.iter().map(|&c| f32::from_le_bytes(c)).collect()
}

/// The median of runs filling about a second (at least 3, at most 50).
fn time<R>(mut f: impl FnMut() -> R) -> Duration {
    std::hint::black_box(f());
    let mut times = Vec::new();
    let start = Instant::now();
    while times.len() < 3 || (start.elapsed() < Duration::from_secs(1) && times.len() < 50) {
        let t = Instant::now();
        std::hint::black_box(f());
        times.push(t.elapsed());
    }
    times.sort();
    times[times.len() / 2]
}

fn fmt(d: Duration) -> String {
    let s = d.as_secs_f64();
    if s >= 1.0 {
        format!("{s:.2} s")
    } else if s >= 1e-3 {
        format!("{:.2} ms", s * 1e3)
    } else {
        format!("{:.1} µs", s * 1e6)
    }
}

/// The largest difference relative to the largest value of `want`.
fn error(got: &[f32], want: &[f32]) -> f64 {
    assert_eq!(got.len(), want.len());
    let top = want.iter().fold(0f64, |m, &v| m.max(v.abs() as f64)).max(1e-30);
    got.iter().zip(want).fold(0f64, |m, (&g, &w)| m.max((g as f64 - w as f64).abs())) / top
}

fn tensor<B: Backend, const D: usize>(v: &[f32], shape: [usize; D], device: &B::Device) -> Tensor<B, D> {
    Tensor::from_data(TensorData::new(v.to_vec(), shape), device)
}

fn values<B: Backend, const D: usize>(t: Tensor<B, D>) -> Vec<f32> {
    t.into_data().to_vec::<f32>().expect("f32 data")
}

// THE MODELS, IN BURN =============================================================================

/// The one-pole low-pass, one tensor op chain per sample: the outputs, `[n]`.
fn burn_one_pole<B: Backend>(cutoff: Tensor<B, 1>, xs: Tensor<B, 1>, s0: Tensor<B, 1>) -> Tensor<B, 1> {
    let n = xs.dims()[0];
    // a = 1 - exp(-2π fc / fs)
    let a = cutoff.mul_scalar(-TAU / FS).exp().neg().add_scalar(1.0);
    let mut s = s0;
    let mut ys = Vec::with_capacity(n);
    for x in xs.chunk(n, 0) {
        s = s.clone() + a.clone() * (x - s);
        ys.push(s.clone());
    }
    Tensor::cat(ys, 0)
}

/// The one-pole's MSE against `targets` and its gradient: (loss, d cutoff, d s0, d xs).
fn burn_one_pole_grad<B: AutodiffBackend>(cutoff: &[f32], xs: &[f32], targets: &[f32], s0: &[f32], device: &B::Device) -> [Vec<f32>; 4] {
    let n = xs.len();
    let cutoff = tensor::<B, 1>(cutoff, [1], device).require_grad();
    let xs = tensor::<B, 1>(xs, [n], device).require_grad();
    let s0 = tensor::<B, 1>(s0, [1], device).require_grad();
    let targets = tensor::<B, 1>(targets, [n], device);
    let e = burn_one_pole(cutoff.clone(), xs.clone(), s0.clone()) - targets;
    let loss = (e.clone() * e).mean();
    let grads = loss.backward();
    let grad = |t: &Tensor<B, 1>| values(t.grad(&grads).expect("a gradient"));
    [values(loss.inner()), grad(&cutoff), grad(&s0), grad(&xs)]
}

/// DFT matrices for real signals of length `len`: forward `[len, bins]` (cos, -sin) and inverse
/// `[bins, len]` (with irfft's 1/N and doubled interior bins).
fn dft_matrices(len: usize) -> [Vec<f32>; 4] {
    let bins = len / 2 + 1;
    let angle = |a: usize, b: usize| TAU * ((a * b) % len) as f64 / len as f64;
    let mut fwd_re = vec![0f32; len * bins];
    let mut fwd_im = vec![0f32; len * bins];
    let mut inv_re = vec![0f32; bins * len];
    let mut inv_im = vec![0f32; bins * len];
    for n in 0..len {
        for k in 0..bins {
            let w = angle(n, k);
            fwd_re[n * bins + k] = w.cos() as f32;
            fwd_im[n * bins + k] = -w.sin() as f32;
            let c = if k == 0 || 2 * k == len { 1.0 } else { 2.0 } / len as f64;
            inv_re[k * len + n] = (c * w.cos()) as f32;
            inv_im[k * len + n] = (-c * w.sin()) as f32;
        }
    }
    [fwd_re, fwd_im, inv_re, inv_im]
}

/// rfft -> per-bin gain -> irfft -> dense -> tanh, the mean energy and its gradient with respect to
/// the gain and the dense weights: (loss, d gain, d w).
fn burn_spectral_grad<B: AutodiffBackend>(x: &[f32], gain: &[f32], w: &[f32], dft: &[Tensor<B, 2>; 4], dims: [usize; 3], device: &B::Device) -> [Vec<f32>; 3] {
    let [batch, len, width] = dims;
    let bins = len / 2 + 1;
    let x = tensor::<B, 2>(x, [batch, len], device);
    let gain = tensor::<B, 1>(gain, [bins], device).require_grad();
    let w = tensor::<B, 2>(w, [len, width], device).require_grad();
    let [fwd_re, fwd_im, inv_re, inv_im] = dft.clone();
    let g = gain.clone().unsqueeze_dim::<2>(0);
    let re = x.clone().matmul(fwd_re) * g.clone();
    let im = x.matmul(fwd_im) * g;
    let y = re.matmul(inv_re) + im.matmul(inv_im);
    let h = y.matmul(w.clone()).tanh();
    let loss = (h.clone() * h).mean();
    let grads = loss.backward();
    [values(loss.inner()), values(gain.grad(&grads).expect("a gradient")), values(w.grad(&grads).expect("a gradient"))]
}

// THE COMPARISON ==================================================================================

struct Row {
    model: &'static str,
    library: String,
    time: Duration,
    error: Option<f64>,
}

fn burn_rows<B: AutodiffBackend>(name: &str, dir: &Path, rows: &mut Vec<Row>) {
    let device = B::Device::default();
    let load = |case: &str, k: &str| read_npy(&dir.join(format!("{case}_{k}.npy")));

    // the one-pole forward (no tape: the inner backend)
    let (cutoff, xs, s0) = (load("one_pole_forward", "in0"), load("one_pole_forward", "in1"), load("one_pole_forward", "in2"));
    let want = load("one_pole_forward", "out0");
    let n = xs.len();
    let run = || values(burn_one_pole::<B::InnerBackend>(tensor(&cutoff, [1], &device), tensor(&xs, [n], &device), tensor(&s0, [1], &device)));
    let error = error(&run(), &want);
    rows.push(Row { model: "one-pole low-pass, 48k samples", library: name.to_string(), time: time(run), error: Some(error) });

    // its MSE gradient
    let ins: Vec<Vec<f32>> = (0..4).map(|k| load("one_pole_grad", &format!("in{k}"))).collect();
    let want: Vec<Vec<f32>> = (0..4).map(|k| load("one_pole_grad", &format!("out{k}"))).collect();
    let run = || burn_one_pole_grad::<B>(&ins[0], &ins[1], &ins[2], &ins[3], &device);
    let error = run().iter().zip(&want).map(|(g, w)| self::error(g, w)).fold(0.0, f64::max);
    rows.push(Row { model: "one-pole MSE gradient, 48k samples", library: name.to_string(), time: time(run), error: Some(error) });

    // the spectral model's gradient, transforms as DFT matrix products
    let ins: Vec<Vec<f32>> = (0..3).map(|k| load("spectral_model_grad", &format!("in{k}"))).collect();
    let want: Vec<Vec<f32>> = (0..3).map(|k| load("spectral_model_grad", &format!("out{k}"))).collect();
    let bins = ins[1].len();
    let len = (bins - 1) * 2;
    let dims = [ins[0].len() / len, len, ins[2].len() / len];
    let [a, b, c, d] = dft_matrices(len);
    let dft = [tensor(&a, [len, bins], &device), tensor(&b, [len, bins], &device), tensor(&c, [bins, len], &device), tensor(&d, [bins, len], &device)];
    let run = || burn_spectral_grad::<B>(&ins[0], &ins[1], &ins[2], &dft, dims, &device);
    let error = run().iter().zip(&want).map(|(g, w)| self::error(g, w)).fold(0.0, f64::max);
    rows.push(Row { model: "rfft -> gain -> irfft -> dense -> tanh gradient, 256 x 1024", library: name.to_string(), time: time(run), error: Some(error) });
}

fn main() {
    let dir = std::env::args().nth(1).expect("usage: flux_burn <dir written by flux_programs>");
    let dir = Path::new(&dir);
    let load = |case: &str, k: &str| read_npy(&dir.join(format!("{case}_{k}.npy")));
    let mut rows = Vec::new();

    // autodyne's core: the filter itself, no tracing
    let (cutoff, xs) = (load("one_pole_forward", "in0")[0], load("one_pole_forward", "in1"));
    let want = load("one_pole_forward", "out0");
    let run = || {
        let mut f = OnePole::lowpass(cutoff, FS as f32);
        xs.iter().map(|&x| f.process_sample(x)).collect::<Vec<f32>>()
    };
    let e = error(&run(), &want);
    rows.push(Row { model: "one-pole low-pass, 48k samples", library: "autodyne core (OnePole)".into(), time: time(run), error: Some(e) });

    // flux's interpreter: the traced graphs evaluated in Rust, no compiler
    let scan = Scan::trace(&[&[]], &[&[]], &[], |p, s, x| {
        let (s, y) = OnePole::lowpass(p[0], Elementwise::lit(FS)).tick(s[0], x);
        (vec![s], y)
    });
    let xs_nd = vector(&xs);
    rows.push(Row { model: "one-pole low-pass, 48k samples", library: "flux interpreter".into(), time: time(|| scan.run(&[scalar(cutoff)], &xs_nd, &[scalar(0.0)])), error: None });
    let targets = NdArray::from_vec(load("one_pole_grad", "in2"), &[xs.len()]).unwrap();
    rows.push(Row { model: "one-pole MSE gradient, 48k samples", library: "flux interpreter".into(), time: time(|| scan.loss_grad(&[scalar(cutoff)], &xs_nd, &targets, &[scalar(0.0)])), error: None });
    let (batch, len, width) = (256usize, 1_024usize, 64usize);
    let model = trace(&[&[batch, len], &[len / 2 + 1], &[len, width]], |v| {
        let (x, gain, w) = (v[0], v[1], v[2]);
        let y = Tracer::irfft_complex(x.rfft_complex() * gain, len).dot(w).tanh();
        let loss = (y * y).mean_all();
        let d = vjp(&[loss], &[Tracer::lit(1.0)], &[gain, w]);
        vec![loss, d[0], d[1]]
    });
    let inputs = vec![
        NdArray::from_vec(load("spectral_model_grad", "in0"), &[batch, len]).unwrap(),
        NdArray::from_vec(load("spectral_model_grad", "in1"), &[len / 2 + 1]).unwrap(),
        NdArray::from_vec(load("spectral_model_grad", "in2"), &[len, width]).unwrap(),
    ];
    rows.push(Row { model: "rfft -> gain -> irfft -> dense -> tanh gradient, 256 x 1024", library: "flux interpreter".into(), time: time(|| model.eval(&inputs)), error: None });

    burn_rows::<Autodiff<Flex>>("burn flex (autodiff)", dir, &mut rows);
    burn_rows::<Autodiff<BurnNdArray>>("burn ndarray (autodiff)", dir, &mut rows);

    println!("| Model | Library | Time | Error vs flux |");
    println!("|---|---|---:|---:|");
    for r in &rows {
        let e = r.error.map_or("—".to_string(), |e| format!("{e:.1e}"));
        println!("| {} | {} | {} | {} |", r.model, r.library, fmt(r.time), e);
    }
}
