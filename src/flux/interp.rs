//! The reference interpreter: each node evaluated by the eager [`ArrayMath`] implementation on
//! `NdArray<f32>`, so a traced function and the same function run on arrays agree by construction.

use super::graph::{Cmp, Graph, Op, Part};
use crate::signal::{ArrayMath, NdArray};
use crate::units::Elementwise;

/// A node's value.
#[derive(Clone)]
enum Value {
    Real(NdArray<f32>),
    Mask(NdArray<bool>),
}

impl Value {
    fn real(&self) -> NdArray<f32> {
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
    /// Evaluates the graph on `inputs` (one array per input, of its shape) and returns its outputs.
    /// Masks come back as 1.0 / 0.0.
    pub fn eval(&self, inputs: &[NdArray<f32>]) -> Vec<NdArray<f32>> {
        assert_eq!(inputs.len(), self.inputs.len(), "Graph::eval: wrong number of inputs");
        for (k, (x, s)) in inputs.iter().zip(&self.inputs).enumerate() {
            assert_eq!(x.shape(), s.as_slice(), "Graph::eval: input {k} has the wrong shape");
        }
        let mut values: Vec<Value> = Vec::with_capacity(self.nodes.len());
        for node in &self.nodes {
            let r = |i: u32| values[i as usize].real();
            let value = match node.op {
                Op::Input(n) => Value::Real(inputs[n as usize].clone()),
                Op::Const(c) => Value::Real(NdArray::lit(c)),
                Op::Literal(ref data) => Value::Real(NdArray::from_vec(data.to_vec(), &node.shape).expect("literal shape")),
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
                Op::Pow(a, b) => Value::Real(r(a).powf(r(b))),
                Op::Min(a, b) => Value::Real(r(a).minimum(r(b))),
                Op::Max(a, b) => Value::Real(r(a).maximum(r(b))),
                Op::Compare(Cmp::Lt, a, b) => Value::Mask(r(a).less(r(b))),
                Op::Compare(Cmp::Gt, a, b) => Value::Mask(r(a).greater(r(b))),
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
            };
            values.push(value);
        }
        self.outputs
            .iter()
            .map(|&o| match &values[o as usize] {
                Value::Real(x) => x.clone(),
                Value::Mask(m) => m.map(|&b| f32::from(b)),
            })
            .collect()
    }
}
