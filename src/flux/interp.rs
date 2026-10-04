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
        let remaining = plan.uses.clone();
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
                // each value it reads, once, borrowed where it can be (a group only reads): real (masks
                // as 1 / 0) or complex, contiguous
                enum In<'a, T: FluxFloat> {
                    Real(&'a [T]),
                    Complex(&'a [Complex<T>]),
                    Owned(Vec<T>),
                    OwnedComplex(Vec<Complex<T>>),
                }
                let fetched: Vec<In<'_, T>> = group
                    .fetch
                    .iter()
                    .map(|&a| match e.values[a as usize].as_ref().expect("a value read after its last use") {
                        Value::Real(x) => match x.view().as_slice() {
                            Some(s) => In::Real(s),
                            None => In::Owned(x.view().to_vec()),
                        },
                        Value::Mask(m) => In::Owned(m.as_slice().iter().map(|&b| if b { T::_ONE } else { T::_ZERO }).collect()),
                        Value::Complex(z) => match z.view().as_slice() {
                            Some(s) => In::Complex(s),
                            None => In::OwnedComplex(z.view().to_vec()),
                        },
                    })
                    .collect();
                // as numbers: a real array itself, or a part of a complex one (every other number)
                let slice = |&(at, part): &(usize, u8)| -> (&[T], usize) {
                    let complex = |z: &[Complex<T>]| {
                        // SAFETY: Complex<T> is repr(C) { re, im }: n of them are 2n T's
                        let flat = unsafe { std::slice::from_raw_parts(z.as_ptr().cast::<T>(), 2 * z.len()) };
                        (&flat[usize::from(part) - 1..], 2)
                    };
                    match &fetched[at] {
                        In::Real(x) => (*x, 1),
                        In::Owned(x) => (x.as_slice(), 1),
                        In::Complex(z) => complex(z),
                        In::OwnedComplex(z) => complex(z),
                    }
                };
                let uniform: Vec<T> = group.inputs[..group.uniform].iter().map(|k| slice(k).0[0]).collect();
                let slices: Vec<(&[T], usize)> = group.inputs[group.uniform..].iter().map(slice).collect();
                let len: usize = node.shape.iter().product();
                let out = group.run(len, &uniform, &slices);
                drop(slices);
                drop(fetched);
                // the reads done: values nothing else reads are freed
                for &a in &group.fetch {
                    let a = a as usize;
                    e.remaining[a] -= 1;
                    if e.remaining[a] == 0 {
                        e.values[a] = None;
                    }
                }
                let values = NdArray::from_vec(out, &node.shape).expect("the node's shape");
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
                Op::Literal(ref data) => {
                    Value::Real(NdArray::from_vec(plan.literals.get::<T>(i, data).as_ref().clone(), &node.shape).expect("the literal's shape"))
                }
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
                Op::Frames { a, length, hop } => each!(v!(a), x => x.frames(length, hop)),
                Op::OverlapAdd { a, n, hop } => each!(v!(a), x => x.overlap_add(n, hop)),
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