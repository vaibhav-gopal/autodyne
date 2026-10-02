//! A reference interpreter: evaluates a graph in f32, the precision the emitted programs use.

use super::graph::{Cmp, Graph, Op, Part};
use crate::signal::NdArray;
use crate::spectral::RealFft;
use crate::units::Complex;

impl Graph {
    /// Evaluates the graph on `inputs` (one array per input, of its shape) and returns its outputs.
    /// Masks come back as 1.0 / 0.0.
    pub fn eval(&self, inputs: &[NdArray<f32>]) -> Vec<NdArray<f32>> {
        let values = self.eval_all(inputs);
        self.outputs
            .iter()
            .map(|&o| NdArray::from_vec(values[o as usize].clone(), &self.nodes[o as usize].shape).expect("node shapes are valid"))
            .collect()
    }

    /// Evaluates every node (row-major data; shapes are the nodes').
    pub(crate) fn eval_all(&self, inputs: &[NdArray<f32>]) -> Vec<Vec<f32>> {
        assert_eq!(inputs.len(), self.inputs.len(), "Graph::eval: wrong number of inputs");
        for (k, (x, s)) in inputs.iter().zip(&self.inputs).enumerate() {
            assert_eq!(x.shape(), s.as_slice(), "Graph::eval: input {k} has the wrong shape");
        }
        let mut values: Vec<Vec<f32>> = Vec::with_capacity(self.nodes.len());
        for node in &self.nodes {
            let v = |i: u32| values[i as usize].as_slice();
            let shape_of = |i: u32| self.nodes[i as usize].shape.as_slice();
            let map = |a: u32, f: fn(f32) -> f32| v(a).iter().map(|&x| f(x)).collect::<Vec<f32>>();
            let zip = |a: u32, b: u32, f: fn(f32, f32) -> f32| v(a).iter().zip(v(b)).map(|(&x, &y)| f(x, y)).collect::<Vec<f32>>();
            let r = match node.op {
                Op::Input(n) => inputs[n as usize].as_slice().to_vec(),
                Op::Const(c) => vec![c as f32],
                Op::Literal(ref data) => data.to_vec(),
                Op::Add(a, b) => zip(a, b, |x, y| x + y),
                Op::Sub(a, b) => zip(a, b, |x, y| x - y),
                Op::Mul(a, b) => zip(a, b, |x, y| x * y),
                Op::Div(a, b) => zip(a, b, |x, y| x / y),
                Op::Pow(a, b) => zip(a, b, f32::powf),
                Op::Min(a, b) => zip(a, b, f32::min),
                Op::Max(a, b) => zip(a, b, f32::max),
                Op::Compare(Cmp::Lt, a, b) => zip(a, b, |x, y| f32::from(x < y)),
                Op::Compare(Cmp::Gt, a, b) => zip(a, b, |x, y| f32::from(x > y)),
                Op::Neg(a) => map(a, |x| -x),
                Op::Exp(a) => map(a, f32::exp),
                Op::Log(a) => map(a, f32::ln),
                Op::Sin(a) => map(a, f32::sin),
                Op::Cos(a) => map(a, f32::cos),
                Op::Tanh(a) => map(a, f32::tanh),
                Op::Sqrt(a) => map(a, f32::sqrt),
                Op::Abs(a) => map(a, f32::abs),
                Op::Select(c, a, b) => v(c).iter().zip(v(a).iter().zip(v(b))).map(|(&m, (&x, &y))| if m != 0.0 { x } else { y }).collect(),
                Op::Broadcast(a, ref dims) => {
                    let (from, from_strides) = (shape_of(a), strides(shape_of(a)));
                    let mut src = vec![0isize; node.shape.len()];
                    for (i, &d) in dims.iter().enumerate() {
                        src[d] = if from[i] == 1 { 0 } else { from_strides[i] };
                    }
                    gather(&node.shape, &src, v(a))
                }
                Op::Reshape(a) => v(a).to_vec(),
                Op::Transpose(a, ref perm) => {
                    let from_strides = strides(shape_of(a));
                    let src: Vec<isize> = perm.iter().map(|&p| from_strides[p]).collect();
                    gather(&node.shape, &src, v(a))
                }
                Op::Sum(a, ref axes) => {
                    // scatter-add each operand element into its output position
                    let from = shape_of(a);
                    let out_strides = strides(&node.shape);
                    let mut dst = vec![0isize; from.len()];
                    let mut k = 0;
                    for (axis, d) in dst.iter_mut().enumerate() {
                        if !axes.contains(&axis) {
                            *d = out_strides[k];
                            k += 1;
                        }
                    }
                    let mut out = vec![0.0f32; node.shape.iter().product()];
                    for_each_offset(from, &dst, |i, o| out[o] += v(a)[i]);
                    out
                }
                Op::Dot { a, b, ref ca, ref cb } => dot(v(a), shape_of(a), v(b), shape_of(b), ca, cb),
                Op::Rfft(a, part) => {
                    let n = *shape_of(a).last().unwrap();
                    v(a).chunks(n.max(1))
                        .flat_map(|row| {
                            rfft(row).into_iter().map(move |z| match part {
                                Part::Re => z.re as f32,
                                Part::Im => z.im as f32,
                            })
                        })
                        .collect()
                }
                Op::Irfft { re, im, n } => {
                    let m = n / 2 + 1;
                    v(re).chunks(m).zip(v(im).chunks(m)).flat_map(|(r, i)| irfft(r, i, n)).collect()
                }
            };
            values.push(r);
        }
        values
    }
}

/// Row-major strides (in elements).
fn strides(shape: &[usize]) -> Vec<isize> {
    let mut s = vec![1isize; shape.len()];
    for i in (0..shape.len().saturating_sub(1)).rev() {
        s[i] = s[i + 1] * shape[i + 1] as isize;
    }
    s
}

/// Calls `f(position, offset)` for each index of `shape` in row-major order, where `offset` moves
/// by `step[axis]` along each axis.
fn for_each_offset(shape: &[usize], step: &[isize], mut f: impl FnMut(usize, usize)) {
    let count: usize = shape.iter().product();
    let mut index = vec![0usize; shape.len()];
    let mut offset = 0isize;
    for i in 0..count {
        f(i, offset as usize);
        for axis in (0..shape.len()).rev() {
            index[axis] += 1;
            offset += step[axis];
            if index[axis] < shape[axis] {
                break;
            }
            offset -= step[axis] * shape[axis] as isize;
            index[axis] = 0;
        }
    }
}

/// The array of `shape` whose element at each index is `src[Σ index * step]`.
fn gather(shape: &[usize], step: &[isize], src: &[f32]) -> Vec<f32> {
    let mut out = Vec::with_capacity(shape.iter().product());
    for_each_offset(shape, step, |_, o| out.push(src[o]));
    out
}

/// `dot_general` without batch axes: permute to `[free, contracted]` and `[contracted, free]`, then
/// a matrix product (accumulated in f64).
fn dot(a: &[f32], sa: &[usize], b: &[f32], sb: &[usize], ca: &[usize], cb: &[usize]) -> Vec<f32> {
    let fa: Vec<usize> = (0..sa.len()).filter(|x| !ca.contains(x)).collect();
    let fb: Vec<usize> = (0..sb.len()).filter(|x| !cb.contains(x)).collect();
    let permute = |data: &[f32], shape: &[usize], perm: &[usize]| {
        let st = strides(shape);
        let out_shape: Vec<usize> = perm.iter().map(|&p| shape[p]).collect();
        gather(&out_shape, &perm.iter().map(|&p| st[p]).collect::<Vec<_>>(), data)
    };
    let a = permute(a, sa, &fa.iter().chain(ca).copied().collect::<Vec<_>>());
    let b = permute(b, sb, &cb.iter().chain(&fb).copied().collect::<Vec<_>>());
    let m: usize = fa.iter().map(|&x| sa[x]).product();
    let k: usize = ca.iter().map(|&x| sa[x]).product();
    let n: usize = fb.iter().map(|&x| sb[x]).product();
    let mut out = vec![0.0f32; m * n];
    for i in 0..m {
        for j in 0..n {
            out[i * n + j] = (0..k).map(|p| a[i * k + p] as f64 * b[p * n + j] as f64).sum::<f64>() as f32;
        }
    }
    out
}

/// Bins 0..=n/2 of a real signal's DFT (f64 internally).
fn rfft(x: &[f32]) -> Vec<Complex<f64>> {
    let n = x.len();
    if n >= 2 && n.is_power_of_two() {
        let mut fft = RealFft::<f64>::new(n);
        let mut out = vec![Complex::new(0.0, 0.0); n / 2 + 1];
        fft.forward(&x.iter().map(|&v| v as f64).collect::<Vec<_>>(), &mut out);
        return out;
    }
    (0..n / 2 + 1)
        .map(|k| {
            x.iter().enumerate().fold(Complex::new(0.0, 0.0), |acc, (t, &v)| {
                let phase = -std::f64::consts::TAU * ((k * t) % n) as f64 / n as f64;
                acc + Complex::new(v as f64 * phase.cos(), v as f64 * phase.sin())
            })
        })
        .collect()
}

/// `n` samples from bins 0..=n/2, scaled by 1/n; imaginary parts of bins 0 and n/2 ignored.
fn irfft(re: &[f32], im: &[f32], n: usize) -> Vec<f32> {
    let spectrum: Vec<Complex<f64>> = re.iter().zip(im).map(|(&r, &i)| Complex::new(r as f64, i as f64)).collect();
    if n >= 2 && n.is_power_of_two() {
        let mut fft = RealFft::<f64>::new(n);
        let mut out = vec![0.0f64; n];
        fft.inverse(&spectrum, &mut out);
        return out.into_iter().map(|v| v as f32).collect();
    }
    (0..n)
        .map(|t| {
            let mut sum = spectrum[0].re;
            for (k, z) in spectrum.iter().enumerate().skip(1) {
                let phase = std::f64::consts::TAU * ((k * t) % n) as f64 / n as f64;
                let term = z.re * phase.cos() - z.im * phase.sin();
                sum += if n.is_multiple_of(2) && k == n / 2 { term } else { 2.0 * term };
            }
            (sum / n as f64) as f32
        })
        .collect()
}
