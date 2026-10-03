//! The traced graph: a flat list of primitive operations on arrays, and the `Tracer` handle that
//! records into it.

use std::cell::{Cell, RefCell};
use std::fmt;
use std::ops::{Add, Div, Mul, Neg, Sub};
use std::sync::Arc;

use super::runtime::{HostArray, HostRef};
use super::FluxError;
use crate::signal::{NdArray, MAX_DIMS};
use crate::signal::{ArrayMath, RealArrayMath};
use crate::units::{DType, Elementwise, Float, RealValued};

/// The element types traced programs compute in: `f32` and `f64`. A trace has no precision of its
/// own (constants are kept as `f64`); it is chosen when the graph is evaluated or emitted.
pub trait FluxFloat: Float + Default + sealed::Sealed {
    const DTYPE: DType;
    #[doc(hidden)]
    fn host_ref(a: &NdArray<Self>) -> HostRef<'_>;
    #[doc(hidden)]
    fn from_host(a: HostArray) -> Result<NdArray<Self>, FluxError>;
}

impl FluxFloat for f32 {
    const DTYPE: DType = DType::F32;
    fn host_ref(a: &NdArray<f32>) -> HostRef<'_> {
        HostRef::F32(a)
    }
    fn from_host(a: HostArray) -> Result<NdArray<f32>, FluxError> {
        match a {
            HostArray::F32(a) => Ok(a),
            HostArray::F64(_) => Err(FluxError::Shape("expected f32 data, got f64".into())),
        }
    }
}

impl FluxFloat for f64 {
    const DTYPE: DType = DType::F64;
    fn host_ref(a: &NdArray<f64>) -> HostRef<'_> {
        HostRef::F64(a)
    }
    fn from_host(a: HostArray) -> Result<NdArray<f64>, FluxError> {
        match a {
            HostArray::F64(a) => Ok(a),
            HostArray::F32(_) => Err(FluxError::Shape("expected f64 data, got f32".into())),
        }
    }
}

mod sealed {
    pub trait Sealed {}
    impl Sealed for f32 {}
    impl Sealed for f64 {}
}

/// Index of a node in its [`Graph`].
pub(crate) type Id = u32;

/// Comparison directions (the result is a mask).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Cmp {
    Lt,
    Gt,
    Eq,
}

/// Reductions other than sums.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Reduction {
    Max,
    Min,
    Prod,
}

/// What a node's elements are.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Kind {
    Real,
    /// Complex numbers (spectra), carried as such so XLA keeps them in one array.
    Complex,
    /// Booleans (comparisons), consumed by `select`.
    Mask,
}

/// A primitive operation. Operands are earlier nodes, so a graph is always in topological order.
///
/// Element-wise operations take operands of the node's own shape (the [`Tracer`] operators insert
/// [`Broadcast`](Op::Broadcast)s, NumPy style, so the graph itself never broadcasts implicitly).
#[derive(Clone, Debug, PartialEq)]
pub enum Op {
    /// The graph's `n`-th input.
    Input(u32),
    /// A scalar constant; stored as f64, rounded to the precision the graph is run at.
    Const(f64),
    /// A constant array (row-major), of the node's shape; stored as f64 like [`Const`](Op::Const).
    Literal(Arc<[f64]>),
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
    /// Rounds down; its derivative is zero.
    Floor(Id),
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
    /// The real FFT along the last axis: `n` real samples become `n / 2 + 1` complex bins.
    Rfft(Id),
    /// The inverse real FFT along the last axis: `n / 2 + 1` complex bins become `n` real samples,
    /// scaled by `1 / n`. The imaginary parts of bins 0 and `n / 2` are ignored.
    Irfft(Id, usize),
    /// The complex DFT along the last axis, or its inverse (scaled by `1 / n`).
    Fft(Id, bool),
    /// `re + i im` from two real operands.
    Complex(Id, Id),
    /// The real part of a complex operand.
    Re(Id),
    /// The imaginary part of a complex operand.
    Im(Id),
    /// The complex conjugate.
    Conj(Id),
    /// A real operand as complex numbers.
    ToComplex(Id),
    /// Indices `start..limit` by `stride` along each axis (`stablehlo.slice`).
    Slice { a: Id, start: Vec<usize>, limit: Vec<usize>, stride: Vec<usize> },
    /// Zero padding before, after and between the elements of each axis (`stablehlo.pad`).
    Pad { a: Id, low: Vec<usize>, high: Vec<usize>, interior: Vec<usize> },
    /// The operands joined along an axis (`stablehlo.concatenate`).
    Concat(Vec<Id>, usize),
    /// Max / min / product over `axes` (increasing), which are removed.
    Reduce(Id, Vec<usize>, Reduction),
    /// Reverses each of `axes`.
    Reverse(Id, Vec<usize>),
    /// Rows of `table` (first axis) at `indices`, rounded down and clamped (`stablehlo.gather`).
    Take { table: Id, indices: Id },
    /// Zeros of the node's shape with each row of `updates` added at the row `indices` names (rounded
    /// down and clamped): the transpose of [`Take`](Op::Take) (`stablehlo.scatter`).
    ScatterAdd { indices: Id, updates: Id },
}

impl Op {
    /// The operands, in order.
    pub fn operands(&self) -> impl Iterator<Item = Id> {
        if let Op::Concat(parts, _) = self {
            return parts.clone().into_iter();
        }
        let (a, b, c) = match *self {
            Op::Input(_) | Op::Const(_) | Op::Literal(_) => (None, None, None),
            Op::Neg(a) | Op::Exp(a) | Op::Log(a) | Op::Sin(a) | Op::Cos(a) | Op::Tanh(a) | Op::Sqrt(a) | Op::Abs(a) | Op::Floor(a) => (Some(a), None, None),
            Op::Rfft(a) | Op::Irfft(a, _) | Op::Fft(a, _) | Op::Re(a) | Op::Im(a) | Op::Conj(a) | Op::ToComplex(a) => (Some(a), None, None),
            Op::Broadcast(a, _) | Op::Reshape(a) | Op::Transpose(a, _) | Op::Sum(a, _) | Op::Slice { a, .. } | Op::Pad { a, .. } | Op::Reduce(a, ..) | Op::Reverse(a, _) => {
                (Some(a), None, None)
            }
            Op::Add(a, b) | Op::Sub(a, b) | Op::Mul(a, b) | Op::Div(a, b) | Op::Pow(a, b) | Op::Min(a, b) | Op::Max(a, b) | Op::Compare(_, a, b) => {
                (Some(a), Some(b), None)
            }
            Op::Dot { a, b, .. } | Op::Complex(a, b) | Op::Take { table: a, indices: b } | Op::ScatterAdd { indices: a, updates: b } => {
                (Some(a), Some(b), None)
            }
            Op::Select(c, a, b) => (Some(c), Some(a), Some(b)),
            Op::Concat(..) => unreachable!("handled above"),
        };
        [a, b, c].into_iter().flatten().collect::<Vec<_>>().into_iter()
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
            Op::Floor(a) => Op::Floor(f(a)),
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
            Op::Rfft(a) => Op::Rfft(f(a)),
            Op::Irfft(a, n) => Op::Irfft(f(a), n),
            Op::Fft(a, inverse) => Op::Fft(f(a), inverse),
            Op::Complex(a, b) => Op::Complex(f(a), f(b)),
            Op::Re(a) => Op::Re(f(a)),
            Op::Im(a) => Op::Im(f(a)),
            Op::Conj(a) => Op::Conj(f(a)),
            Op::ToComplex(a) => Op::ToComplex(f(a)),
            Op::Slice { a, start, limit, stride } => Op::Slice { a: f(a), start, limit, stride },
            Op::Pad { a, low, high, interior } => Op::Pad { a: f(a), low, high, interior },
            Op::Concat(parts, axis) => Op::Concat(parts.into_iter().map(f).collect(), axis),
            Op::Reduce(a, axes, r) => Op::Reduce(f(a), axes, r),
            Op::Reverse(a, axes) => Op::Reverse(f(a), axes),
            Op::Take { table, indices } => Op::Take { table: f(table), indices: f(indices) },
            Op::ScatterAdd { indices, updates } => Op::ScatterAdd { indices: f(indices), updates: f(updates) },
        }
    }
}

/// A node: its operation, and the shape and kind of its value.
#[derive(Clone, Debug, PartialEq)]
pub struct Node {
    pub op: Op,
    pub shape: Vec<usize>,
    pub kind: Kind,
}

/// A traced program: nodes in topological order, the inputs' shapes, and the output nodes.
///
/// Built by [`trace`]. Values are arrays of up to [`MAX_DIMS`] axes (shape `[]` is a scalar) of real
/// numbers, complex numbers (spectra) or booleans (masks); inputs and outputs are real.
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
                ref op => push(op.map(|i| ids[i as usize]), node.shape.clone(), node.kind),
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
            writeln!(f, "%{i}: {:?} {:?} = {:?}", n.shape, n.kind, n.op)?;
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
    let mut graph = trace_unpruned(inputs, f);
    graph.prune();
    graph
}

/// [`trace`] keeping every recorded node, so ids match what the closure saw.
pub(crate) fn trace_unpruned(inputs: &[&[usize]], f: impl FnOnce(&[Tracer]) -> Vec<Tracer>) -> Graph {
    for s in inputs {
        assert!(s.len() <= MAX_DIMS, "flux::trace: at most {MAX_DIMS} axes");
    }
    let nodes = (0..inputs.len()).map(|n| Node { op: Op::Input(n as u32), shape: inputs[n].to_vec(), kind: Kind::Real }).collect();
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

/// Records `node` again in the current trace, its operands mapped by `map`.
pub(crate) fn replay(node: &Node, map: impl Fn(Id) -> Id) -> Tracer {
    Tracer { id: push(node.op.map(map), node.shape.clone(), node.kind), trace: current_trace() }
}

impl Graph {
    /// Drops the nodes no output depends on (the inputs stay): reverse mode records cotangents for
    /// constants and the like that nothing reads.
    fn prune(&mut self) {
        let mut needed = vec![false; self.nodes.len()];
        for &o in &self.outputs {
            needed[o as usize] = true;
        }
        for i in (0..self.nodes.len()).rev() {
            if needed[i] || matches!(self.nodes[i].op, Op::Input(_)) {
                needed[i] = true;
                for a in self.nodes[i].op.operands() {
                    needed[a as usize] = true;
                }
            }
        }
        if needed.iter().all(|&n| n) {
            return;
        }
        let mut new_id = vec![Id::MAX; self.nodes.len()];
        let mut nodes = Vec::with_capacity(needed.iter().filter(|&&n| n).count());
        for (i, node) in std::mem::take(&mut self.nodes).into_iter().enumerate() {
            if needed[i] {
                new_id[i] = nodes.len() as Id;
                nodes.push(Node { op: node.op.map(|a| new_id[a as usize]), ..node });
            }
        }
        self.nodes = nodes;
        self.outputs.iter_mut().for_each(|o| *o = new_id[*o as usize]);
    }
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

fn push(op: Op, shape: Vec<usize>, kind: Kind) -> Id {
    with_graph(|g| {
        g.nodes.push(Node { op, shape, kind });
        (g.nodes.len() - 1) as Id
    })
}

/// The kind of value `op` makes, from its operands'.
fn kind_of(op: &Op) -> Kind {
    let of = |i: Id| with_graph(|g| g.nodes[i as usize].kind);
    match *op {
        Op::Input(_) | Op::Const(_) | Op::Literal(_) | Op::Abs(_) | Op::Re(_) | Op::Im(_) | Op::Irfft(..) => Kind::Real,
        Op::Compare(..) => Kind::Mask,
        Op::Rfft(_) | Op::ToComplex(_) | Op::Complex(..) => Kind::Complex,
        Op::Select(_, a, _) | Op::Take { table: a, .. } | Op::ScatterAdd { updates: a, .. } => of(a),
        Op::Concat(ref parts, _) => of(parts[0]),
        ref op => of(op.operands().next().expect("an operand")),
    }
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
        let kind = kind_of(&op);
        Tracer { id: push(op, shape, kind), trace: current_trace() }
    }

    /// What the elements are: real or complex.
    pub fn kind(&self) -> Kind {
        let id = self.check();
        with_graph(|g| g.nodes[id as usize].kind)
    }

    pub fn is_complex(&self) -> bool {
        self.kind() == Kind::Complex
    }

    fn real_only(self, what: &str) -> Tracer {
        assert!(!self.is_complex(), "flux: {what} needs real values, not complex ones");
        self
    }

    /// `re + i im` (real operands, broadcast together).
    pub fn complex(re: Tracer, im: Tracer) -> Tracer {
        let (re, im) = (re.real_only("complex"), im.real_only("complex"));
        let (a, b, shape) = re.align(im);
        Tracer::new(Op::Complex(a.check(), b.check()), shape)
    }

    /// The real part (a real value is its own).
    pub fn re(self) -> Tracer {
        if self.is_complex() { self.unary(Op::Re) } else { self }
    }

    /// The imaginary part (zero for a real value).
    pub fn im(self) -> Tracer {
        if self.is_complex() { self.unary(Op::Im) } else { Tracer::zeros(&self.shape()) }
    }

    /// The value as complex numbers (a complex value is unchanged).
    pub fn to_complex(self) -> Tracer {
        if self.is_complex() { self } else { self.unary(Op::ToComplex) }
    }

    /// Both operands of one kind: a real one becomes complex beside a complex one.
    fn promote(self, rhs: Tracer) -> (Tracer, Tracer) {
        if self.is_complex() == rhs.is_complex() { (self, rhs) } else { (self.to_complex(), rhs.to_complex()) }
    }

    /// A constant array (of `f32` or `f64`).
    pub fn constant<T: FluxFloat>(value: &NdArray<T>) -> Tracer {
        Tracer::new(Op::Literal(value.as_slice().iter().map(|v| v.to_f64().unwrap_or(f64::NAN)).collect()), value.shape().to_vec())
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

    fn complex_fft(self, inverse: bool) -> Tracer {
        let z = self.to_complex();
        let shape = z.shape();
        assert!(shape.last().is_some_and(|&n| n >= 1), "fft: needs a non-empty axis");
        Tracer::new(Op::Fft(z.check(), inverse), shape)
    }

    fn reduce(self, axes: &[usize], r: Reduction) -> Tracer {
        let _ = self.real_only("max / min / prod");
        let own = self.shape();
        let mut axes = axes.to_vec();
        axes.sort_unstable();
        axes.dedup();
        assert!(axes.iter().all(|&a| a < own.len()), "reduction: axis out of range for {own:?}");
        if axes.is_empty() {
            return self;
        }
        let shape = (0..own.len()).filter(|a| !axes.contains(a)).map(|a| own[a]).collect();
        Tracer::new(Op::Reduce(self.check(), axes, r), shape)
    }

    /// Zeros of `shape` (`[n, rest...]`) with each row of `updates` (`[indices..., rest...]`) added
    /// at the row its index names (rounded down, clamped): the transpose of [`take`](RealArrayMath::take).
    pub fn scatter_add(shape: &[usize], indices: Tracer, updates: Tracer) -> Tracer {
        let (is, us) = (indices.shape(), updates.shape());
        assert!(!shape.is_empty() && shape[0] > 0, "scatter_add: needs a non-empty first axis");
        assert_eq!(us, [is.as_slice(), &shape[1..]].concat(), "scatter_add: updates must be [indices..., rows...]");
        Tracer::new(Op::ScatterAdd { indices: indices.check(), updates: updates.check() }, shape.to_vec())
    }

    fn binary(self, rhs: Tracer, f: fn(Id, Id) -> Op) -> Tracer {
        let (a, b) = self.promote(rhs);
        let (a, b, shape) = a.align(b);
        Tracer::new(f(a.check(), b.check()), shape)
    }

    /// A binary operation defined for real values only.
    fn real_binary(self, rhs: Tracer, f: fn(Id, Id) -> Op, what: &str) -> Tracer {
        self.real_only(what).binary(rhs.real_only(what), f)
    }

    pub(crate) fn compare(self, rhs: Tracer, c: Cmp) -> Mask {
        let (a, b, shape) = self.real_only("comparison").align(rhs.real_only("comparison"));
        Mask { id: push(Op::Compare(c, a.check(), b.check()), shape, Kind::Mask), trace: self.trace }
    }
}

/// Array operations record nodes; see [`ArrayMath`] for what each computes.
impl ArrayMath for Tracer {
    fn array(values: &[f64], shape: &[usize]) -> Tracer {
        assert_eq!(values.len(), shape.iter().product::<usize>(), "array: {} values do not fill {shape:?}", values.len());
        Tracer::new(Op::Literal(values.into()), shape.to_vec())
    }
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
        let (a, b) = self.promote(rhs);
        Tracer::new(Op::Dot { a: a.check(), b: b.check(), ca: ca.to_vec(), cb: cb.to_vec() }, shape)
    }

    fn slice(self, start: &[usize], limit: &[usize], stride: &[usize]) -> Tracer {
        let own = self.shape();
        let shape = crate::signal::slice_shape(&own, start, limit, stride);
        if shape == own {
            return self;
        }
        Tracer::new(Op::Slice { a: self.check(), start: start.to_vec(), limit: limit.to_vec(), stride: stride.to_vec() }, shape)
    }

    fn pad(self, low: &[usize], high: &[usize], interior: &[usize]) -> Tracer {
        let own = self.shape();
        let shape = crate::signal::pad_shape(&own, low, high, interior);
        if shape == own {
            return self;
        }
        Tracer::new(Op::Pad { a: self.check(), low: low.to_vec(), high: high.to_vec(), interior: interior.to_vec() }, shape)
    }

    fn prod_axes(self, axes: &[usize]) -> Tracer {
        self.reduce(axes, Reduction::Prod)
    }

    fn reverse(self, axes: &[usize]) -> Tracer {
        let own = self.shape();
        let mut axes = axes.to_vec();
        axes.sort_unstable();
        axes.dedup();
        assert!(axes.iter().all(|&a| a < own.len()), "reverse: axis out of range for {own:?}");
        if axes.iter().all(|&a| own[a] <= 1) {
            return self;
        }
        Tracer::new(Op::Reverse(self.check(), axes), own)
    }

    fn concatenate(parts: &[Tracer], axis: usize) -> Tracer {
        let shape = crate::signal::concat_shape(&parts.iter().map(|p| p.shape()).collect::<Vec<_>>(), axis);
        if parts.len() == 1 {
            return parts[0];
        }
        let complex = parts.iter().any(Tracer::is_complex);
        Tracer::new(Op::Concat(parts.iter().map(|p| if complex { p.to_complex().check() } else { p.check() }).collect(), axis), shape)
    }
}

/// Spectra are complex tracers: one array, as XLA keeps them.
impl RealArrayMath for Tracer {
    type Complex = Tracer;

    fn rfft_complex(self) -> Tracer {
        let mut shape = self.real_only("rfft").shape();
        let n = *shape.last().expect("rfft: needs an axis");
        assert!(n >= 1, "rfft: empty axis");
        *shape.last_mut().unwrap() = n / 2 + 1;
        Tracer::new(Op::Rfft(self.check()), shape)
    }

    fn irfft_complex(spectrum: Tracer, n: usize) -> Tracer {
        let z = spectrum.to_complex();
        let mut shape = z.shape();
        assert!(n >= 1 && shape.last() == Some(&(n / 2 + 1)), "irfft: {n} samples need {} bins, got {shape:?}", n / 2 + 1);
        *shape.last_mut().unwrap() = n;
        Tracer::new(Op::Irfft(z.check(), n), shape)
    }

    fn to_complex(self) -> Tracer {
        Tracer::to_complex(self)
    }

    fn complex(re: Tracer, im: Tracer) -> Tracer {
        Tracer::complex(re, im)
    }

    fn real_part(z: Tracer) -> Tracer {
        z.re()
    }

    fn imag_part(z: Tracer) -> Tracer {
        z.im()
    }

    fn max_axes(self, axes: &[usize]) -> Tracer {
        self.reduce(axes, Reduction::Max)
    }

    fn min_axes(self, axes: &[usize]) -> Tracer {
        self.reduce(axes, Reduction::Min)
    }

    fn take(self, indices: Tracer) -> Tracer {
        let (table, is) = (self.real_only("take").shape(), indices.real_only("take").shape());
        assert!(!table.is_empty() && table[0] > 0, "take: the table needs a non-empty first axis");
        assert!(is.len() + table.len() - 1 <= MAX_DIMS, "flux: at most {MAX_DIMS} axes");
        Tracer::new(Op::Take { table: self.check(), indices: indices.check() }, [is.as_slice(), &table[1..]].concat())
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
        Mask { id: push(Op::Broadcast(self.id, dims), shape.to_vec(), Kind::Mask), trace: self.trace }
    }
}

/// Complex FFTs record nodes on complex tracers (a real tracer is promoted).
impl crate::signal::ComplexArrayMath for Tracer {
    fn fft(self) -> Tracer {
        self.complex_fft(false)
    }
    fn ifft(self) -> Tracer {
        self.complex_fft(true)
    }
    fn conj(self) -> Tracer {
        if self.is_complex() { self.unary(Op::Conj) } else { self }
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
        self.real_binary(e, Op::Pow, "powf")
    }
}

impl RealValued for Tracer {
    type Mask = Mask;
    fn abs(self) -> Self {
        self.unary(Op::Abs)
    }
    fn minimum(self, other: Self) -> Self {
        self.real_binary(other, Op::Min, "minimum")
    }
    fn maximum(self, other: Self) -> Self {
        self.real_binary(other, Op::Max, "maximum")
    }
    fn less(self, other: Self) -> Mask {
        self.compare(other, Cmp::Lt)
    }
    fn greater(self, other: Self) -> Mask {
        self.compare(other, Cmp::Gt)
    }
    fn floor(self) -> Self {
        self.real_only("floor").unary(Op::Floor)
    }
    fn select(mask: Mask, if_true: Self, if_false: Self) -> Self {
        assert_eq!(mask.trace, current_trace(), "flux: mask used outside the trace that created it");
        let ms = with_graph(|g| g.nodes[mask.id as usize].shape.clone());
        let (a, b) = if_true.promote(if_false);
        let (a, b, shape) = a.align(b);
        let shape = broadcast_shapes(&shape, &ms).unwrap_or_else(|| panic!("flux: mask {ms:?} does not broadcast with {shape:?}"));
        let (a, b, m) = (a.broadcast_to(&shape), b.broadcast_to(&shape), mask.broadcast_to(&shape));
        Tracer::new(Op::Select(m.id, a.check(), b.check()), shape)
    }
}