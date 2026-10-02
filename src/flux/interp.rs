//! The reference interpreter: each node evaluated by the eager [`ArrayMath`] implementation on
//! `NdArray<f32>`, so a traced function and the same function run on arrays agree by construction.

use super::graph::{Cmp, FluxFloat, Graph, Op, Part, Reduction};
use crate::signal::{ArrayMath, NdArray, RealArrayMath};
use crate::units::{Elementwise, RealValued};

/// A node's value.
#[derive(Clone)]
enum Value<T> {
    Real(NdArray<T>),
    Mask(NdArray<bool>),
}

impl<T: FluxFloat> Value<T> {
    fn real(&self) -> NdArray<T> {
        match self {
            Value::Real(a) => a.clone(),
            Value::Mask(_) => panic!("flux: a mask used as a number"),
        }
    }
    fn mask(&self) -> NdArray<bool> {
        match self {
            Value::Mask(m) => m.clone(),
            Value::Real(_) => panic!("flux: a number used as a mask"),
        }
    }
}

impl Graph {
    /// Evaluates the graph in `T` (`f32` or `f64`) on `inputs` (one array per input, of its
    /// shape) and returns its outputs. Masks come back as 1.0 / 0.0.
    pub fn eval<T: FluxFloat>(&self, inputs: &[NdArray<T>]) -> Vec<NdArray<T>> {
        assert_eq!(inputs.len(), self.inputs.len(), "Graph::eval: wrong number of inputs");
        for (k, (x, s)) in inputs.iter().zip(&self.inputs).enumerate() {
            assert_eq!(x.shape(), s.as_slice(), "Graph::eval: input {k} has the wrong shape");
        }
        let mut values: Vec<Value<T>> = Vec::with_capacity(self.nodes.len());
        for node in &self.nodes {
            let r = |i: u32| values[i as usize].real();
            let value = match node.op {
                Op::Input(n) => Value::Real(inputs[n as usize].clone()),
                Op::Const(c) => Value::Real(NdArray::lit(c)),
                Op::Literal(ref data) => Value::Real(NdArray::array(data, &node.shape)),
                Op::Add(a, b) => Value::Real(r(a) + r(b)),
                Op::Sub(a, b) => Value::Real(r(a) - r(b)),
                Op::Mul(a, b) => Value::Real(r(a) * r(b)),
                Op::Div(a, b) => Value::Real(r(a) / r(b)),
                Op::Neg(a) => Value::Real(-r(a)),
                Op::Exp(a) => Value::Real(r(a).exp()),
                Op::Log(a) => Value::Real(r(a).ln()),
                Op::Sin(a) => Value::Real(r(a).sin()),
                Op::Cos(a) => Value::Real(r(a).cos()),
                Op::Tanh(a) => Value::Real(r(a).tanh()),
                Op::Sqrt(a) => Value::Real(r(a).sqrt()),
                Op::Abs(a) => Value::Real(r(a).abs()),
                Op::Floor(a) => Value::Real(r(a).floor()),
                Op::Pow(a, b) => Value::Real(r(a).powf(r(b))),
                Op::Min(a, b) => Value::Real(r(a).minimum(r(b))),
                Op::Max(a, b) => Value::Real(r(a).maximum(r(b))),
                Op::Compare(Cmp::Lt, a, b) => Value::Mask(r(a).less(r(b))),
                Op::Compare(Cmp::Gt, a, b) => Value::Mask(r(a).greater(r(b))),
                Op::Compare(Cmp::Eq, a, b) => {
                    let (x, y) = (r(a), r(b));
                    Value::Mask(NdArray::from_vec(x.as_slice().iter().zip(y.as_slice()).map(|(p, q)| p == q).collect(), x.shape()).expect("same shape"))
                }
                Op::Select(c, a, b) => Value::Real(NdArray::select(values[c as usize].mask(), r(a), r(b))),
                Op::Broadcast(a, ref dims) => match &values[a as usize] {
                    Value::Real(x) => Value::Real(crate::signal::broadcast_in_dim(x, &node.shape, dims)),
                    Value::Mask(m) => Value::Mask(crate::signal::broadcast_in_dim(m, &node.shape, dims)),
                },
                Op::Reshape(a) => Value::Real(ArrayMath::reshape(r(a), &node.shape)),
                Op::Transpose(a, ref perm) => Value::Real(ArrayMath::transpose(r(a), perm)),
                Op::Sum(a, ref axes) => Value::Real(r(a).sum_axes(axes)),
                Op::Dot { a, b, ref ca, ref cb } => Value::Real(r(a).dot_general(r(b), ca, cb)),
                Op::Rfft(a, part) => {
                    let (re, im) = r(a).rfft();
                    Value::Real(if part == Part::Re { re } else { im })
                }
                Op::Irfft { re, im, n } => Value::Real(NdArray::irfft(r(re), r(im), n)),
                Op::Slice { a, ref start, ref limit, ref stride } => Value::Real(r(a).slice(start, limit, stride)),
                Op::Pad { a, ref low, ref high, ref interior } => Value::Real(r(a).pad(low, high, interior)),
                Op::Fft { re, im, inverse, part } => {
                    let (a, b) = if inverse { NdArray::ifft_parts(r(re), r(im)) } else { NdArray::fft_parts(r(re), r(im)) };
                    Value::Real(if part == Part::Re { a } else { b })
                }
                Op::Reduce(a, ref axes, Reduction::Max) => Value::Real(r(a).max_axes(axes)),
                Op::Reduce(a, ref axes, Reduction::Min) => Value::Real(r(a).min_axes(axes)),
                Op::Reduce(a, ref axes, Reduction::Prod) => Value::Real(r(a).prod_axes(axes)),
                Op::Reverse(a, ref axes) => Value::Real(r(a).reverse(axes)),
                Op::Take { table, indices } => Value::Real(r(table).take(r(indices))),
                Op::ScatterAdd { indices, updates } => Value::Real(scatter_add(&node.shape, &r(indices), &r(updates))),
                Op::Concat(ref parts, axis) => Value::Real(NdArray::concatenate(&parts.iter().map(|&p| r(p)).collect::<Vec<_>>(), axis)),
            };
            values.push(value);
        }
        self.outputs
            .iter()
            .map(|&o| match &values[o as usize] {
                Value::Real(x) => x.clone(),
                Value::Mask(m) => m.map(|&b| if b { T::_ONE } else { T::_ZERO }),
            })
            .collect()
    }
}

/// Zeros of `shape` with each row of `updates` added at its index's row (rounded down, clamped).
fn scatter_add<T: FluxFloat>(shape: &[usize], indices: &NdArray<T>, updates: &NdArray<T>) -> NdArray<T> {
    let row: usize = shape[1..].iter().product();
    let mut out = vec![T::_ZERO; shape.iter().product()];
    if row > 0 {
        for (&i, u) in indices.as_slice().iter().zip(updates.as_slice().chunks(row)) {
            let k = crate::signal::clamp_index(i.to_f64().unwrap_or(f64::NAN), shape[0]);
            out[k * row..(k + 1) * row].iter_mut().zip(u).for_each(|(o, &v)| *o = *o + v);
        }
    }
    NdArray::from_vec(out, shape).expect("valid shape")
}