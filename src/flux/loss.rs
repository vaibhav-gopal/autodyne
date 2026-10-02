//! Losses for fitting: a traced function of a scan's whole output (and of extra inputs, such as a
//! target), with its gradient. [`Scan::grad`](super::Scan::grad) runs the scan forward, the loss
//! and its vector-Jacobian product, then the scan backward from the loss's cotangent.
//!
//! The spectral pieces ([`frames`], [`stft_magnitude`], [`multi_resolution_stft`]) are generic over
//! [`RealArrayMath`]: they run eagerly on `NdArray`s and trace into losses alike.

use super::ad::vjp;
use super::graph::{trace, Graph, Tracer};
use crate::signal::{frames, ArrayMath, NdArray, RealArrayMath};
use crate::units::Elementwise;

/// A scalar function of an output signal and extra inputs, traced with its gradient.
///
/// ```
/// use autodyne::flux::{vector, Loss};
/// use autodyne::signal::ArrayMath;
/// use autodyne::units::RealValued;
///
/// // mean absolute error
/// let mae = Loss::trace(&[4], &[&[4]], |y, aux| (y - aux[0]).abs().mean_all());
/// let (loss, dy) = mae.grad(&vector(&[1.0, 2.0, 3.0, 4.0]), &[vector(&[0.0, 1.0, 5.0, 4.5])]);
/// assert_eq!(loss, 1.125);
/// assert_eq!(dy.as_slice(), &[0.25, 0.25, -0.25, -0.25]);
/// ```
#[derive(Clone, Debug)]
pub struct Loss {
    output: Vec<usize>,
    aux: Vec<Vec<usize>>,
    /// inputs: output, aux; outputs: loss
    pub(crate) value: Graph,
    /// inputs: output, aux; outputs: loss, d loss / d output
    pub(crate) grad: Graph,
}

impl Loss {
    /// Traces `f(output, aux)`, which must return a scalar, for an output of shape `output` and
    /// extra inputs of shapes `aux`.
    pub fn trace(output: &[usize], aux: &[&[usize]], f: impl FnOnce(Tracer, &[Tracer]) -> Tracer) -> Loss {
        let shapes: Vec<&[usize]> = std::iter::once(output).chain(aux.iter().copied()).collect();
        let value = trace(&shapes, |v| {
            let l = f(v[0], &v[1..]);
            assert!(l.shape().is_empty(), "Loss::trace: the loss must be a scalar, not {:?}", l.shape());
            vec![l]
        });
        let grad = trace(&shapes, |v| {
            let l = value.call(v)[0];
            let d = vjp(&[l], &[Tracer::lit(1.0)], &[v[0]]);
            vec![l, d[0]]
        });
        Loss { output: output.to_vec(), aux: aux.iter().map(|s| s.to_vec()).collect(), value, grad }
    }

    /// Mean squared error against a target of the output's shape (the one extra input).
    pub fn mse(output: &[usize]) -> Loss {
        Loss::trace(output, &[output], |y, aux| {
            let d = y - aux[0];
            (d * d).mean_all()
        })
    }

    /// [`multi_resolution_stft`] against a target of the output's shape (the one extra input). The
    /// output is `[len, channels...]` (a scan's output): the transforms run along the first axis.
    pub fn stft(output: &[usize], resolutions: &[StftResolution]) -> Loss {
        let time_last: Vec<usize> = (1..output.len()).chain([0]).collect();
        Loss::trace(output, &[output], |y, aux| multi_resolution_stft(y.transpose(&time_last), aux[0].transpose(&time_last), resolutions))
    }

    /// The shape of the output it scores.
    pub fn output_shape(&self) -> &[usize] {
        &self.output
    }
    /// The shapes of the extra inputs.
    pub fn aux_shapes(&self) -> &[Vec<usize>] {
        &self.aux
    }
    /// The traced loss: inputs output, aux...; output the loss.
    pub fn graph(&self) -> &Graph {
        &self.value
    }

    /// The loss of `output`.
    pub fn eval(&self, output: &NdArray<f32>, aux: &[NdArray<f32>]) -> f32 {
        self.value.eval(&self.args(output, aux))[0].as_slice()[0]
    }

    /// The loss of `output` and its gradient with respect to `output`.
    pub fn grad(&self, output: &NdArray<f32>, aux: &[NdArray<f32>]) -> (f32, NdArray<f32>) {
        let mut out = self.grad.eval(&self.args(output, aux));
        let d = out.pop().expect("two outputs");
        (out[0].as_slice()[0], d)
    }

    fn args(&self, output: &NdArray<f32>, aux: &[NdArray<f32>]) -> Vec<NdArray<f32>> {
        assert_eq!(aux.len(), self.aux.len(), "Loss: wrong number of extra inputs");
        std::iter::once(output).chain(aux).cloned().collect()
    }
}

/// One resolution of [`multi_resolution_stft`]: an FFT of `n_fft` points over Hann-windowed frames
/// of `window` samples (`window <= n_fft`, zero-padded), `hop` samples apart.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct StftResolution {
    pub n_fft: usize,
    pub hop: usize,
    pub window: usize,
}

impl StftResolution {
    pub const fn new(n_fft: usize, hop: usize, window: usize) -> Self {
        StftResolution { n_fft, hop, window }
    }

    /// The resolutions usual for audio (Parallel WaveGAN, auraloss): FFTs of 1024, 2048 and 512
    /// points, hops of 120, 240 and 50, windows of 600, 1200 and 240.
    pub const DEFAULT: [StftResolution; 3] = [StftResolution::new(1024, 120, 600), StftResolution::new(2048, 240, 1200), StftResolution::new(512, 50, 240)];

    /// A resolution with a 75% overlapping window as long as the FFT.
    pub const fn overlapping(n_fft: usize) -> Self {
        StftResolution::new(n_fft, n_fft / 4, n_fft)
    }
}

/// The periodic Hann window of `n` points.
fn hann(n: usize) -> Vec<f64> {
    (0..n).map(|i| 0.5 - 0.5 * (2.0 * std::f64::consts::PI * i as f64 / n as f64).cos()).collect()
}

/// STFT magnitudes along the last axis: `[..., n]` becomes `[..., frames, n_fft / 2 + 1]`.
///
/// The signal is zero-padded by half a window at the start, and at the end by at least as much (to
/// a whole number of hops), so every sample is in as many frames as any other. Magnitudes are
/// `sqrt(max(re² + im², 1e-8))`, which keeps the gradient finite at silent bins.
pub fn stft_magnitude<A: RealArrayMath>(x: A, resolution: StftResolution) -> A {
    let StftResolution { n_fft, hop, window } = resolution;
    assert!(window >= 1 && window <= n_fft && hop >= 1, "stft_magnitude: needs 1 <= window <= n_fft and hop >= 1");
    let shape = x.shape();
    let (&n, lead) = shape.split_last().expect("stft_magnitude: needs an axis");
    let k = lead.len();
    let low = window / 2;
    let mut high = window / 2;
    while !(n + low + high - window).is_multiple_of(hop) {
        high += 1;
    }
    let (mut lo, mut hi) = (vec![0; k + 1], vec![0; k + 1]);
    (lo[k], hi[k]) = (low, high);
    let framed = frames(x.pad(&lo, &hi, &vec![0; k + 1]), window, hop);
    let windowed = framed * A::array(&hann(window), &[window]);
    let fs = windowed.shape();
    let mut hi = vec![0; fs.len()];
    *hi.last_mut().unwrap() = n_fft - window;
    let (re, im) = windowed.pad(&vec![0; fs.len()], &hi, &vec![0; fs.len()]).rfft();
    (re.clone() * re + im.clone() * im).maximum(A::lit(1e-8)).sqrt()
}

/// The multi-resolution STFT loss usual for fitting audio (Yamamoto et al., Parallel WaveGAN):
/// for each resolution, spectral convergence `‖|T| - |Y|‖ / ‖|T|‖` (Frobenius norms) plus the mean
/// `|ln |Y| - ln |T||`, averaged over the resolutions. Transforms run along the last axis of `y`
/// and `target` (same shapes).
pub fn multi_resolution_stft<A: RealArrayMath>(y: A, target: A, resolutions: &[StftResolution]) -> A {
    assert!(!resolutions.is_empty(), "multi_resolution_stft: needs a resolution");
    assert_eq!(y.shape(), target.shape(), "multi_resolution_stft: output and target differ in shape");
    let mut total: Option<A> = None;
    for &r in resolutions {
        let (s, t) = (stft_magnitude(y.clone(), r), stft_magnitude(target.clone(), r));
        let d = t.clone() - s.clone();
        let convergence = (d.clone() * d).sum_all().sqrt() / (t.clone() * t.clone()).sum_all().sqrt();
        let log_magnitude = (s.ln() - t.ln()).abs().mean_all();
        let term = convergence + log_magnitude;
        total = Some(match total {
            Some(sum) => sum + term,
            None => term,
        });
    }
    total.expect("at least one resolution") / A::lit(resolutions.len() as f64)
}
