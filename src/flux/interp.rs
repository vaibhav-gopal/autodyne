//! A reference interpreter: evaluates a graph in f32, the same precision the emitted programs use.

use super::graph::{Cmp, Graph, Op};

impl Graph {
    /// Evaluates the graph on `inputs` and returns its outputs (in f32).
    pub fn eval(&self, inputs: &[f32]) -> Vec<f32> {
        let mut values = Vec::new();
        self.eval_into(inputs, &mut values);
        self.outputs.iter().map(|&o| values[o as usize]).collect()
    }

    /// Evaluates every node into `values` (reused between calls); masks are 1.0 / 0.0.
    pub(crate) fn eval_into(&self, inputs: &[f32], values: &mut Vec<f32>) {
        assert_eq!(inputs.len(), self.inputs, "Graph::eval: wrong number of inputs");
        values.clear();
        for op in &self.nodes {
            let v = |i: u32| values[i as usize];
            let r = match *op {
                Op::Input(n) => inputs[n as usize],
                Op::Const(c) => c as f32,
                Op::Add(a, b) => v(a) + v(b),
                Op::Sub(a, b) => v(a) - v(b),
                Op::Mul(a, b) => v(a) * v(b),
                Op::Div(a, b) => v(a) / v(b),
                Op::Neg(a) => -v(a),
                Op::Exp(a) => v(a).exp(),
                Op::Log(a) => v(a).ln(),
                Op::Sin(a) => v(a).sin(),
                Op::Cos(a) => v(a).cos(),
                Op::Tanh(a) => v(a).tanh(),
                Op::Sqrt(a) => v(a).sqrt(),
                Op::Abs(a) => v(a).abs(),
                Op::Pow(a, b) => v(a).powf(v(b)),
                Op::Min(a, b) => v(a).min(v(b)),
                Op::Max(a, b) => v(a).max(v(b)),
                Op::Compare(Cmp::Lt, a, b) => f32::from(v(a) < v(b)),
                Op::Compare(Cmp::Gt, a, b) => f32::from(v(a) > v(b)),
                Op::Select(c, a, b) => {
                    if v(c) != 0.0 {
                        v(a)
                    } else {
                        v(b)
                    }
                }
            };
            values.push(r);
        }
    }
}
