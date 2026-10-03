//! The reference interpreter: each node evaluated by the eager [`ArrayMath`] implementation on
//! `NdArray<f32>`, so a traced function and the same function run on arrays agree by construction.

use super::graph::{Cmp, FluxFloat, Graph, Kind, Op, Reduction};
use crate::signal::{ArrayMath, ComplexArrayMath, NdArray, RealArrayMath};
use crate::units::{Complex, Elementwise, RealValued};

/// A node's value.
#[derive(Clone)]
enum Value<T: FluxFloat> {
    Real(NdArray<T>),
    Complex(NdArray<Complex<T>>),
    Mask(NdArray<bool>),
}

impl<T: FluxFloat> Value<T> {
    fn real(self) -> NdArray<T> {
        match self {
            Value::Real(a) => a,
            _ => panic!("flux: expected real values"),
        }
    }
    fn complex(self) -> NdArray<Complex<T>> {
        match self {
            Value::Complex(a) => a,
            _ => panic!("flux: expected complex values"),
        }
    }
    fn mask(self) -> NdArray<bool> {
        match self {
            Value::Mask(m) => m,
            _ => panic!("flux: a number used as a mask"),
        }
    }
}

/// The values computed so far, each handed to its last reader instead of copied, and dropped once
/// nothing else reads it.
struct Values<T: FluxFloat> {
    values: Vec<Option<Value<T>>>,
    /// reads still to come, per node (outputs count as one each)
    remaining: Vec<u32>,
}

impl<T: FluxFloat> Values<T> {
    /// Node `i`'s value for one read: moved out at the last read, copied before it.
    fn get(&mut self, i: u32) -> Value<T> {
        let i = i as usize;
        self.remaining[i] -= 1;
        if self.remaining[i] == 0 {
            self.values[i].take().expect("a value read after its last use")
        } else {
            self.values[i].clone().expect("a value read after its last use")
        }
    }
}

/// `$e` on a real or complex value (the same expression for both).
macro_rules! each {
    ($v:expr, $x:ident => $e:expr) => {
        match $v {
            Value::Real($x) => Value::Real($e),
            Value::Complex($x) => Value::Complex($e),
            Value::Mask(_) => panic!("flux: a mask used as a number"),
        }
    };
}

/// `$e` on two values of one kind.
macro_rules! each2 {
    ($a:expr, $b:expr, $x:ident, $y:ident => $e:expr) => {
        match ($a, $b) {
            (Value::Real($x), Value::Real($y)) => Value::Real($e),
            (Value::Complex($x), Value::Complex($y)) => Value::Complex($e),
            _ => panic!("flux: operands of different kinds"),
        }
    };
}

impl Graph {
    /// Evaluates the graph in `T` (`f32` or `f64`) on `inputs` (one array per input, of its
    /// shape) and returns its outputs. Masks come back as 1.0 / 0.0.
    pub fn eval<T: FluxFloat>(&self, inputs: &[NdArray<T>]) -> Vec<NdArray<T>> {
        self.eval_owned(inputs.to_vec())
    }

    /// [`eval`](Self::eval) taking the inputs by value: each moves into the graph instead of being
    /// copied.
    pub fn eval_owned<T: FluxFloat>(&self, inputs: Vec<NdArray<T>>) -> Vec<NdArray<T>> {
        assert_eq!(inputs.len(), self.inputs.len(), "Graph::eval: wrong number of inputs");
        for (k, (x, s)) in inputs.iter().zip(&self.inputs).enumerate() {
            assert_eq!(x.shape(), s.as_slice(), "Graph::eval: input {k} has the wrong shape");
        }
        let plan = self.fusion();
        let mut remaining = vec![0u32; self.nodes.len()];
        for (i, node) in self.nodes.iter().enumerate() {
            plan.reads(i, node).into_iter().for_each(|a| remaining[a as usize] += 1);
        }
        self.outputs.iter().for_each(|&o| remaining[o as usize] += 1);
        let mut env = Values { values: Vec::with_capacity(self.nodes.len()), remaining };
        let mut inputs: Vec<Option<NdArray<T>>> = inputs.into_iter().map(Some).collect();
        for (i, node) in self.nodes.iter().enumerate() {
            if env.remaining[i] == 0 {
                // nothing reads it
                env.values.push(None);
                continue;
            }
            let e = &mut env;
            if let Some(group) = &plan.groups[i] {
                // a fused element-wise group: single values, then arrays of the node's shape
                let as_numbers = |v: Value<T>| match v {
                    Value::Real(x) if x.view().is_contiguous() => x,
                    Value::Real(x) => x.view().to_owned(),
                    Value::Mask(m) => m.map(|&b| if b { T::_ONE } else { T::_ZERO }),
                    Value::Complex(_) => unreachable!("complex values are not fused"),
                };
                let uniform: Vec<T> = group.inputs[..group.uniform].iter().map(|&a| as_numbers(e.get(a)).as_slice()[0]).collect();
                let arrays: Vec<NdArray<T>> = group.inputs[group.uniform..].iter().map(|&a| as_numbers(e.get(a))).collect();
                let slices: Vec<&[T]> = arrays.iter().map(|a| a.as_slice()).collect();
                let len = node.shape.iter().product();
                let values = NdArray::from_vec(group.run(len, &uniform, &slices), &node.shape).expect("the node's shape");
                let value = if node.kind == Kind::Mask { Value::Mask(values.map(|&v| v != T::_ZERO)) } else { Value::Real(values) };
                env.values.push(Some(value));
                continue;
            }
            macro_rules! v {
                ($i:expr) => {
                    e.get($i)
                };
            }
            macro_rules! r {
                ($i:expr) => {
                    e.get($i).real()
                };
            }
            macro_rules! c {
                ($i:expr) => {
                    e.get($i).complex()
                };
            }
            let value = match node.op {
                Op::Input(n) => Value::Real(inputs[n as usize].take().expect("each input is read by one node")),
                Op::Const(k) => Value::Real(NdArray::lit(k)),
                Op::Literal(ref data) => Value::Real(NdArray::array(data, &node.shape)),
                Op::Add(a, b) => each2!(v!(a), v!(b), x, y => x + y),
                Op::Sub(a, b) => each2!(v!(a), v!(b), x, y => x - y),
                Op::Mul(a, b) => each2!(v!(a), v!(b), x, y => x * y),
                Op::Div(a, b) => each2!(v!(a), v!(b), x, y => x / y),
                Op::Neg(a) => each!(v!(a), x => -x),
                Op::Exp(a) => each!(v!(a), x => x.exp()),
                Op::Log(a) => each!(v!(a), x => x.ln()),
                Op::Sin(a) => each!(v!(a), x => x.sin()),
                Op::Cos(a) => each!(v!(a), x => x.cos()),
                Op::Tanh(a) => each!(v!(a), x => x.tanh()),
                Op::Sqrt(a) => each!(v!(a), x => x.sqrt()),
                Op::Abs(a) => match v!(a) {
                    Value::Complex(z) => Value::Real(z.map(|w| (w.re * w.re + w.im * w.im)._sqrt())),
                    other => Value::Real(other.real().abs()),
                },
                Op::Floor(a) => Value::Real(r!(a).floor()),
                Op::Pow(a, b) => Value::Real(r!(a).powf(r!(b))),
                Op::Min(a, b) => Value::Real(r!(a).minimum(r!(b))),
                Op::Max(a, b) => Value::Real(r!(a).maximum(r!(b))),
                Op::Compare(Cmp::Lt, a, b) => Value::Mask(r!(a).less(r!(b))),
                Op::Compare(Cmp::Gt, a, b) => Value::Mask(r!(a).greater(r!(b))),
                Op::Compare(Cmp::Eq, a, b) => {
                    let (x, y) = (r!(a), r!(b));
                    Value::Mask(NdArray::from_vec(x.as_slice().iter().zip(y.as_slice()).map(|(p, q)| p == q).collect(), x.shape()).expect("same shape"))
                }
                Op::Select(m, a, b) => {
                    let m = v!(m).mask();
                    each2!(v!(a), v!(b), x, y => crate::signal::select_any(&m, &x, &y))
                }
                Op::Broadcast(a, ref dims) if plan.lazy[i] => {
                    // the operand with its axes placed, length 1 elsewhere: its readers broadcast it
                    let mut placed = vec![1; node.shape.len()];
                    let x = v!(a);
                    let operand_shape: Vec<usize> = match &x {
                        Value::Real(x) => x.shape().to_vec(),
                        Value::Complex(x) => x.shape().to_vec(),
                        Value::Mask(x) => x.shape().to_vec(),
                    };
                    for (&d, &len) in dims.iter().zip(&operand_shape) {
                        placed[d] = len;
                    }
                    each!(x, x => ArrayMath::reshape(x, &placed))
                }
                Op::Broadcast(a, ref dims) => match v!(a) {
                    Value::Mask(m) => Value::Mask(crate::signal::broadcast_in_dim(&m, &node.shape, dims)),
                    other => each!(other, x => crate::signal::broadcast_in_dim(&x, &node.shape, dims)),
                },
                Op::Reshape(a) => each!(v!(a), x => ArrayMath::reshape(x, &node.shape)),
                Op::Transpose(a, ref perm) => each!(v!(a), x => ArrayMath::transpose(x, perm)),
                Op::Sum(a, ref axes) => each!(v!(a), x => x.sum_axes(axes)),
                Op::Dot { a, b, ref ca, ref cb } => each2!(v!(a), v!(b), x, y => x.dot_general(y, ca, cb)),
                Op::Rfft(a) => Value::Complex(r!(a).rfft_complex()),
                Op::Irfft(a, n) => Value::Real(NdArray::irfft_complex(c!(a), n)),
                Op::Fft(a, inverse) => Value::Complex(if inverse { c!(a).ifft() } else { c!(a).fft() }),
                Op::Complex(a, b) => Value::Complex(NdArray::complex(r!(a), r!(b))),
                Op::Re(a) => Value::Real(NdArray::real_part(c!(a))),
                Op::Im(a) => Value::Real(NdArray::imag_part(c!(a))),
                Op::Conj(a) => Value::Complex(c!(a).conj()),
                Op::ToComplex(a) => Value::Complex(r!(a).to_complex()),
                Op::Slice { a, ref start, ref limit, ref stride } => each!(v!(a), x => x.slice(start, limit, stride)),
                Op::Pad { a, ref low, ref high, ref interior } => each!(v!(a), x => x.pad(low, high, interior)),
                Op::Reduce(a, ref axes, Reduction::Max) => Value::Real(r!(a).max_axes(axes)),
                Op::Reduce(a, ref axes, Reduction::Min) => Value::Real(r!(a).min_axes(axes)),
                Op::Reduce(a, ref axes, Reduction::Prod) => Value::Real(r!(a).prod_axes(axes)),
                Op::Reverse(a, ref axes) => each!(v!(a), x => x.reverse(axes)),
                Op::Take { table, indices } => Value::Real(r!(table).take(r!(indices))),
                Op::ScatterAdd { indices, updates } => Value::Real(scatter_add(&node.shape, &r!(indices), &r!(updates))),
                Op::Concat(ref parts, axis) => match self.nodes[parts[0] as usize].kind {
                    Kind::Complex => Value::Complex(NdArray::concatenate(&parts.iter().map(|&p| c!(p)).collect::<Vec<_>>(), axis)),
                    _ => Value::Real(NdArray::concatenate(&parts.iter().map(|&p| r!(p)).collect::<Vec<_>>(), axis)),
                },
            };
            env.values.push(Some(value));
        }
        self.outputs
            .iter()
            .map(|&o| match env.get(o) {
                Value::Real(x) => x,
                Value::Mask(m) => m.map(|&b| if b { T::_ONE } else { T::_ZERO }),
                Value::Complex(_) => panic!("Graph::eval: outputs must be real (take the real and imaginary parts)"),
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