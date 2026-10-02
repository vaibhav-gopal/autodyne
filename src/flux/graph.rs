//! The traced graph: a flat list of primitive operations on f32 arrays, and the `Tracer` handle that
//! records into it.

use std::cell::{Cell, RefCell};
use std::fmt;
use std::ops::{Add, Div, Mul, Neg, Sub};
use std::sync::Arc;

use crate::signal::{NdArray, MAX_DIMS};
use crate::signal::{ArrayMath, RealArrayMath};
use crate::units::{Elementwise, RealValued};

/// Index of a node in its [`Graph`].
pub(crate) type Id = u32;

/// Comparison directions (the result is a mask).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Cmp {
    Lt,
    Gt,
}

/// Which half of a complex spectrum an [`Op::Rfft`] node holds.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Part {
    Re,
    Im,
}

/// A primitive operation. Operands are earlier nodes, so a graph is always in topological order.
///
/// Element-wise operations take operands of the node's own shape (the [`Tracer`] operators insert
/// [`Broadcast`](Op::Broadcast)s, NumPy style, so the graph itself never broadcasts implicitly).
#[derive(Clone, Debug, PartialEq)]
pub enum Op {
    /// The graph's `n`-th input.
    Input(u32),
    /// A scalar constant; stored as f64, evaluated and emitted at the graph's precision (f32).
    Const(f64),
    /// A constant array (row-major), of the node's shape.
    Literal(Arc<[f32]>),
    Add(Id, Id),
    Sub(Id, Id),
    Mul(Id, Id),
    Div(Id, Id),
    Neg(Id),
    Exp(Id),
    Log(Id),
    Sin(Id),
    Cos(Id),
    Tanh(Id),
    Sqrt(Id),
    Abs(Id),
    Pow(Id, Id),
    Min(Id, Id),
    Max(Id, Id),
    /// A mask.
    Compare(Cmp, Id, Id),
    /// `select(mask, if_true, if_false)`.
    Select(Id, Id, Id),
    /// Operand axis `i` becomes result axis `dims[i]` (`dims` increasing); operand axes of length 1
    /// stretch, and the other result axes repeat the operand (`stablehlo.broadcast_in_dim`).
    Broadcast(Id, Vec<usize>),
    /// The same elements (row-major) in the node's shape.
    Reshape(Id),
    /// Result axis `i` is operand axis `perm[i]`.
    Transpose(Id, Vec<usize>),
    /// Sum over `axes` (increasing), which are removed.
    Sum(Id, Vec<usize>),
    /// Contracts axes `ca` of the first operand with axes `cb` of the second (pairwise); the result
    /// has the first operand's other axes, then the second's (`stablehlo.dot_general`).
    Dot { a: Id, b: Id, ca: Vec<usize>, cb: Vec<usize> },
    /// One part of the real FFT along the last axis: length `n` becomes `n / 2 + 1` bins.
    Rfft(Id, Part),
    /// The inverse real FFT along the last axis from real and imaginary parts (`n / 2 + 1` bins)
    /// to `n` samples, scaled by `1 / n`. The imaginary parts of bins 0 and `n / 2` are ignored.
    Irfft { re: Id, im: Id, n: usize },
}

impl Op {
    /// The operands, in order.
    pub fn operands(&self) -> impl Iterator<Item = Id> {
        let (a, b, c) = match *self {
            Op::Input(_) | Op::Const(_) | Op::Literal(_) => (None, None, None),
            Op::Neg(a) | Op::Exp(a) | Op::Log(a) | Op::Sin(a) | Op::Cos(a) | Op::Tanh(a) | Op::Sqrt(a) | Op::Abs(a) => (Some(a), None, None),
            Op::Broadcast(a, _) | Op::Reshape(a) | Op::Transpose(a, _) | Op::Sum(a, _) | Op::Rfft(a, _) => (Some(a), None, None),
            Op::Add(a, b) | Op::Sub(a, b) | Op::Mul(a, b) | Op::Div(a, b) | Op::Pow(a, b) | Op::Min(a, b) | Op::Max(a, b) | Op::Compare(_, a, b) => {
                (Some(a), Some(b), None)
            }
            Op::Dot { a, b, .. } | Op::Irfft { re: a, im: b, .. } => (Some(a), Some(b), None),
            Op::Select(c, a, b) => (Some(c), Some(a), Some(b)),
        };
        [a, b, c].into_iter().flatten()
    }

    /// The same operation on other operands (`f` maps each operand id).
    pub(crate) fn map(&self, f: impl Fn(Id) -> Id) -> Op {
        match self.clone() {
            op @ (Op::Input(_) | Op::Const(_) | Op::Literal(_)) => op,
            Op::Add(a, b) => Op::Add(f(a), f(b)),
            Op::Sub(a, b) => Op::Sub(f(a), f(b)),
            Op::Mul(a, b) => Op::Mul(f(a), f(b)),
            Op::Div(a, b) => Op::Div(f(a), f(b)),
            Op::Neg(a) => Op::Neg(f(a)),
            Op::Exp(a) => Op::Exp(f(a)),
            Op::Log(a) => Op::Log(f(a)),
            Op::Sin(a) => Op::Sin(f(a)),
            Op::Cos(a) => Op::Cos(f(a)),
            Op::Tanh(a) => Op::Tanh(f(a)),
            Op::Sqrt(a) => Op::Sqrt(f(a)),
            Op::Abs(a) => Op::Abs(f(a)),
            Op::Pow(a, b) => Op::Pow(f(a), f(b)),
            Op::Min(a, b) => Op::Min(f(a), f(b)),
            Op::Max(a, b) => Op::Max(f(a), f(b)),
            Op::Compare(c, a, b) => Op::Compare(c, f(a), f(b)),
            Op::Select(c, a, b) => Op::Select(f(c), f(a), f(b)),
            Op::Broadcast(a, dims) => Op::Broadcast(f(a), dims),
            Op::Reshape(a) => Op::Reshape(f(a)),
            Op::Transpose(a, perm) => Op::Transpose(f(a), perm),
            Op::Sum(a, axes) => Op::Sum(f(a), axes),
            Op::Dot { a, b, ca, cb } => Op::Dot { a: f(a), b: f(b), ca, cb },
            Op::Rfft(a, part) => Op::Rfft(f(a), part),
            Op::Irfft { re, im, n } => Op::Irfft { re: f(re), im: f(im), n },
        }
    }
}

/// A node: its operation, the shape of its value, and whether that value is a mask (booleans)
/// rather than real numbers.
#[derive(Clone, Debug, PartialEq)]
pub struct Node {
    pub op: Op,
    pub shape: Vec<usize>,
    pub mask: bool,
}

/// A traced program: nodes in topological order, the inputs' shapes, and the output nodes.
///
/// Built by [`trace`]. Values are f32 arrays of up to [`MAX_DIMS`] axes (shape `[]` is a scalar).
#[derive(Clone, Debug, PartialEq, Default)]
pub struct Graph {
    pub(crate) nodes: Vec<Node>,
    pub(crate) inputs: Vec<Vec<usize>>,
    pub(crate) outputs: Vec<Id>,
}

impl Graph {
    pub fn nodes(&self) -> &[Node] {
        &self.nodes
    }
    /// The shape of each input.
    pub fn inputs(&self) -> &[Vec<usize>] {
        &self.inputs
    }
    /// The shape of each output.
    pub fn output_shapes(&self) -> Vec<Vec<usize>> {
        self.outputs.iter().map(|&o| self.nodes[o as usize].shape.clone()).collect()
    }
    pub fn outputs(&self) -> usize {
        self.outputs.len()
    }

    /// Replays this graph into the current trace with `args` as its inputs; returns its outputs.
    /// This is how one traced function calls another (the call is inlined).
    pub fn call(&self, args: &[Tracer]) -> Vec<Tracer> {
        assert_eq!(args.len(), self.inputs.len(), "Graph::call: wrong number of arguments");
        for (k, (a, s)) in args.iter().zip(&self.inputs).enumerate() {
            assert_eq!(&a.shape(), s, "Graph::call: argument {k} has the wrong shape");
        }
        let mut ids: Vec<Id> = Vec::with_capacity(self.nodes.len());
        for node in &self.nodes {
            let id = match node.op {
                Op::Input(n) => args[n as usize].check(),
                ref op => push(op.map(|i| ids[i as usize]), node.shape.clone(), node.mask),
            };
            ids.push(id);
        }
        let trace = current_trace();
        self.outputs.iter().map(|&o| Tracer { id: ids[o as usize], trace }).collect()
    }
}

impl fmt::Display for Graph {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        for (i, n) in self.nodes.iter().enumerate() {
            writeln!(f, "%{i}: {:?}{} = {:?}", n.shape, if n.mask { " mask" } else { "" }, n.op)?;
        }
        write!(f, "return {:?}", self.outputs)
    }
}

/// A traced array of real numbers: a handle to a node in the graph being traced on this thread.
///
/// Implements [`Real`](crate::units::Real) (element-wise, so per-sample DSP code runs on it unchanged)
/// and [`ArrayMath`] (shapes, reductions, products, FFTs, so array code written for `NdArray` runs
/// on it too), recording what it computes. Operators broadcast NumPy style. Only valid inside the
/// [`trace`] that created it.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Tracer {
    pub(crate) id: Id,
    trace: u32,
}

/// A traced array of booleans (the result of a comparison), consumed by [`RealValued::select`].
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Mask {
    pub(crate) id: Id,
    trace: u32,
}

thread_local! {
    static GRAPH: RefCell<Option<(u32, Graph)>> = const { RefCell::new(None) };
    static NEXT_TRACE: Cell<u32> = const { Cell::new(0) };
}

/// Traces `f` on inputs of the given shapes and returns the graph of what it computed.
///
/// ```
/// use autodyne::flux::{scalar, trace};
/// use autodyne::units::Elementwise;
///
/// let g = trace(&[&[], &[]], |v| vec![(v[0] * v[1]).exp()]);
/// assert_eq!(g.eval(&[scalar(1.0), scalar(2.0)])[0].as_slice(), &[2.0f32.exp()]);
/// ```
pub fn trace(inputs: &[&[usize]], f: impl FnOnce(&[Tracer]) -> Vec<Tracer>) -> Graph {
    for s in inputs {
        assert!(s.len() <= MAX_DIMS, "flux::trace: at most {MAX_DIMS} axes");
    }
    let nodes = (0..inputs.len()).map(|n| Node { op: Op::Input(n as u32), shape: inputs[n].to_vec(), mask: false }).collect();
    let graph = Graph { nodes, inputs: inputs.iter().map(|s| s.to_vec()).collect(), outputs: Vec::new() };
    let trace = NEXT_TRACE.with(|n| {
        let t = n.get();
        n.set(t.wrapping_add(1));
        t
    });
    GRAPH.with(|g| {
        let mut g = g.borrow_mut();
        assert!(g.is_none(), "flux::trace: traces cannot be nested");
        *g = Some((trace, graph));
    });
    // take the graph back even if `f` panics, so the thread can trace again
    struct Reset;
    impl Drop for Reset {
        fn drop(&mut self) {
            GRAPH.with(|g| g.borrow_mut().take());
        }
    }
    let reset = Reset;
    let args: Vec<Tracer> = (0..inputs.len() as Id).map(|id| Tracer { id, trace }).collect();
    let outs = f(&args);
    for t in &outs {
        assert_eq!(t.trace, trace, "flux::trace: output belongs to another trace");
    }
    let (_, mut graph) = GRAPH.with(|g| g.borrow_mut().take()).expect("flux::trace: graph missing");
    drop(reset);
    graph.outputs = outs.iter().map(|t| t.id).collect();
    graph
}

/// An array constant for use in a trace (or as an input to [`Graph::eval`]).
pub fn scalar(v: f32) -> NdArray<f32> {
    NdArray::from_vec(vec![v], &[]).expect("a scalar always fits")
}

/// A 1-D array.
pub fn vector(v: &[f32]) -> NdArray<f32> {
    NdArray::from_vec(v.to_vec(), &[v.len()]).expect("a vector always fits")
}

fn with_graph<R>(f: impl FnOnce(&mut Graph) -> R) -> R {
    GRAPH.with(|g| f(&mut g.borrow_mut().as_mut().expect("flux: tracer used outside flux::trace").1))
}

fn current_trace() -> u32 {
    GRAPH.with(|g| g.borrow().as_ref().map(|(t, _)| *t).expect("flux: tracer used outside flux::trace"))
}

/// Node `id` of the current trace.
pub(crate) fn node(id: Id) -> Node {
    with_graph(|g| g.nodes[id as usize].clone())
}

/// The number of nodes in the current trace so far.
pub(crate) fn len() -> usize {
    with_graph(|g| g.nodes.len())
}

fn push(op: Op, shape: Vec<usize>, mask: bool) -> Id {
    with_graph(|g| {
        g.nodes.push(Node { op, shape, mask });
        (g.nodes.len() - 1) as Id
    })
}

/// NumPy broadcasting: the shape both operands stretch to, if any.
fn broadcast_shapes(a: &[usize], b: &[usize]) -> Option<Vec<usize>> {
    let n = a.len().max(b.len());
    let at = |s: &[usize], i: usize| if i + s.len() >= n { s[i + s.len() - n] } else { 1 };
    (0..n)
        .map(|i| match (at(a, i), at(b, i)) {
            (x, y) if x == y => Some(x),
            (1, y) => Some(y),
            (x, 1) => Some(x),
            _ => None,
        })
        .collect()
}

impl Tracer {
    /// The tracer for node `id` of the current trace.
    pub(crate) fn node(id: Id) -> Tracer {
        Tracer { id, trace: current_trace() }
    }

    fn check(self) -> Id {
        assert_eq!(self.trace, current_trace(), "flux: tracer used outside the trace that created it");
        self.id
    }

    fn new(op: Op, shape: Vec<usize>) -> Tracer {
        assert!(shape.len() <= MAX_DIMS, "flux: at most {MAX_DIMS} axes");
        Tracer { id: push(op, shape, false), trace: current_trace() }
    }

    /// A constant array.
    pub fn constant(value: &NdArray<f32>) -> Tracer {
        Tracer::new(Op::Literal(value.as_slice().into()), value.shape().to_vec())
    }

    /// Zeros of `shape` (initial states, padding).
    pub fn zeros(shape: &[usize]) -> Tracer {
        Tracer::lit(0.0).broadcast_to(shape)
    }

    fn unary(self, f: fn(Id) -> Op) -> Tracer {
        let shape = self.shape();
        Tracer::new(f(self.check()), shape)
    }

    /// Both operands broadcast to a common shape.
    fn align(self, rhs: Tracer) -> (Tracer, Tracer, Vec<usize>) {
        let (sa, sb) = (self.shape(), rhs.shape());
        let shape = broadcast_shapes(&sa, &sb).unwrap_or_else(|| panic!("flux: shapes {sa:?} and {sb:?} do not broadcast"));
        (self.broadcast_to(&shape), rhs.broadcast_to(&shape), shape)
    }

    fn binary(self, rhs: Tracer, f: fn(Id, Id) -> Op) -> Tracer {
        let (a, b, shape) = self.align(rhs);
        Tracer::new(f(a.check(), b.check()), shape)
    }

    fn compare(self, rhs: Tracer, c: Cmp) -> Mask {
        let (a, b, shape) = self.align(rhs);
        Mask { id: push(Op::Compare(c, a.check(), b.check()), shape, true), trace: self.trace }
    }
}

/// Array operations record nodes; see [`ArrayMath`] for what each computes.
impl ArrayMath for Tracer {
    fn shape(&self) -> Vec<usize> {
        let id = self.check();
        with_graph(|g| g.nodes[id as usize].shape.clone())
    }

    fn broadcast_to(self, shape: &[usize]) -> Tracer {
        let own = self.shape();
        if own == shape {
            return self;
        }
        assert!(own.len() <= shape.len(), "flux: cannot broadcast {own:?} to {shape:?}");
        let offset = shape.len() - own.len();
        self.broadcast_in_dim(shape, &(offset..shape.len()).collect::<Vec<_>>())
    }

    fn broadcast_in_dim(self, shape: &[usize], dims: &[usize]) -> Tracer {
        let own = self.shape();
        assert_eq!(own.len(), dims.len(), "broadcast_in_dim: one result axis per operand axis");
        assert!(dims.windows(2).all(|w| w[0] < w[1]), "broadcast_in_dim: dims must increase");
        for (&d, &n) in dims.iter().zip(&own) {
            assert!(d < shape.len() && (n == 1 || n == shape[d]), "flux: cannot broadcast {own:?} to {shape:?} along {dims:?}");
        }
        Tracer::new(Op::Broadcast(self.check(), dims.to_vec()), shape.to_vec())
    }

    fn reshape(self, shape: &[usize]) -> Tracer {
        let own = self.shape();
        assert_eq!(own.iter().product::<usize>(), shape.iter().product::<usize>(), "reshape: {own:?} to {shape:?} changes the element count");
        if own == shape {
            return self;
        }
        Tracer::new(Op::Reshape(self.check()), shape.to_vec())
    }

    fn transpose(self, perm: &[usize]) -> Tracer {
        let own = self.shape();
        let mut seen = vec![false; own.len()];
        assert_eq!(perm.len(), own.len(), "transpose: one entry per axis");
        for &p in perm {
            assert!(p < own.len() && !std::mem::replace(&mut seen[p], true), "transpose: {perm:?} is not a permutation");
        }
        if perm.iter().enumerate().all(|(i, &p)| i == p) {
            return self;
        }
        Tracer::new(Op::Transpose(self.check(), perm.to_vec()), perm.iter().map(|&p| own[p]).collect())
    }

    fn sum_axes(self, axes: &[usize]) -> Tracer {
        let own = self.shape();
        let mut axes = axes.to_vec();
        axes.sort_unstable();
        axes.dedup();
        assert!(axes.iter().all(|&a| a < own.len()), "sum_axes: axis out of range for {own:?}");
        if axes.is_empty() {
            return self;
        }
        let shape = (0..own.len()).filter(|a| !axes.contains(a)).map(|a| own[a]).collect();
        Tracer::new(Op::Sum(self.check(), axes), shape)
    }

    fn dot_general(self, rhs: Tracer, ca: &[usize], cb: &[usize]) -> Tracer {
        let (sa, sb) = (self.shape(), rhs.shape());
        assert_eq!(ca.len(), cb.len(), "dot_general: pair each contracted axis");
        for (&i, &j) in ca.iter().zip(cb) {
            assert!(i < sa.len() && j < sb.len() && sa[i] == sb[j], "dot_general: cannot contract {sa:?} axis {i} with {sb:?} axis {j}");
        }
        let distinct = |c: &[usize]| c.iter().enumerate().all(|(k, x)| !c[..k].contains(x));
        assert!(distinct(ca) && distinct(cb), "dot_general: an axis is contracted twice");
        let shape = (0..sa.len()).filter(|a| !ca.contains(a)).map(|a| sa[a]).chain((0..sb.len()).filter(|b| !cb.contains(b)).map(|b| sb[b])).collect();
        Tracer::new(Op::Dot { a: self.check(), b: rhs.check(), ca: ca.to_vec(), cb: cb.to_vec() }, shape)
    }

}

/// Real FFTs record nodes; the spectrum is two real arrays (real and imaginary parts).
impl RealArrayMath for Tracer {
    fn rfft(self) -> (Tracer, Tracer) {
        let mut shape = self.shape();
        let n = *shape.last().expect("rfft: needs an axis");
        assert!(n >= 1, "rfft: empty axis");
        *shape.last_mut().unwrap() = n / 2 + 1;
        let id = self.check();
        (Tracer::new(Op::Rfft(id, Part::Re), shape.clone()), Tracer::new(Op::Rfft(id, Part::Im), shape))
    }

    fn irfft(re: Tracer, im: Tracer, n: usize) -> Tracer {
        let (sr, si) = (re.shape(), im.shape());
        assert_eq!(sr, si, "irfft: real and imaginary parts differ in shape");
        assert!(n >= 1 && sr.last() == Some(&(n / 2 + 1)), "irfft: {n} samples need {} bins, got {sr:?}", n / 2 + 1);
        let mut shape = sr;
        *shape.last_mut().unwrap() = n;
        Tracer::new(Op::Irfft { re: re.check(), im: im.check(), n }, shape)
    }
}

impl Mask {
    pub(crate) fn node(id: Id) -> Mask {
        Mask { id, trace: current_trace() }
    }

    fn broadcast_to(self, shape: &[usize]) -> Mask {
        let own = with_graph(|g| g.nodes[self.id as usize].shape.clone());
        if own == shape {
            return self;
        }
        let offset = shape.len() - own.len();
        let dims = (offset..shape.len()).collect();
        Mask { id: push(Op::Broadcast(self.id, dims), shape.to_vec(), true), trace: self.trace }
    }
}

macro_rules! impl_operator {
    ($($Trait:ident $method:ident $Op:ident),+) => {$(
        impl $Trait for Tracer {
            type Output = Tracer;
            fn $method(self, rhs: Tracer) -> Tracer {
                self.binary(rhs, Op::$Op)
            }
        }
    )+};
}

impl_operator!(Add add Add, Sub sub Sub, Mul mul Mul, Div div Div);

impl Neg for Tracer {
    type Output = Tracer;
    fn neg(self) -> Tracer {
        self.unary(Op::Neg)
    }
}

impl Elementwise for Tracer {
    fn lit(v: f64) -> Self {
        Tracer::new(Op::Const(v), Vec::new())
    }
    fn exp(self) -> Self {
        self.unary(Op::Exp)
    }
    fn ln(self) -> Self {
        self.unary(Op::Log)
    }
    fn sin(self) -> Self {
        self.unary(Op::Sin)
    }
    fn cos(self) -> Self {
        self.unary(Op::Cos)
    }
    fn tanh(self) -> Self {
        self.unary(Op::Tanh)
    }
    fn sqrt(self) -> Self {
        self.unary(Op::Sqrt)
    }
    fn powf(self, e: Self) -> Self {
        self.binary(e, Op::Pow)
    }
}

impl RealValued for Tracer {
    type Mask = Mask;
    fn abs(self) -> Self {
        self.unary(Op::Abs)
    }
    fn minimum(self, other: Self) -> Self {
        self.binary(other, Op::Min)
    }
    fn maximum(self, other: Self) -> Self {
        self.binary(other, Op::Max)
    }
    fn less(self, other: Self) -> Mask {
        self.compare(other, Cmp::Lt)
    }
    fn greater(self, other: Self) -> Mask {
        self.compare(other, Cmp::Gt)
    }
    fn select(mask: Mask, if_true: Self, if_false: Self) -> Self {
        assert_eq!(mask.trace, current_trace(), "flux: mask used outside the trace that created it");
        let ms = with_graph(|g| g.nodes[mask.id as usize].shape.clone());
        let (a, b, shape) = if_true.align(if_false);
        let shape = broadcast_shapes(&shape, &ms).unwrap_or_else(|| panic!("flux: mask {ms:?} does not broadcast with {shape:?}"));
        let (a, b, m) = (a.broadcast_to(&shape), b.broadcast_to(&shape), mask.broadcast_to(&shape));
        Tracer::new(Op::Select(m.id, a.check(), b.check()), shape)
    }
}