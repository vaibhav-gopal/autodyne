//! The traced graph: a flat list of primitive operations on scalars, and the `Tracer` handle that
//! records into it.

use std::cell::{Cell, RefCell};
use std::fmt;
use std::ops::{Add, Div, Mul, Neg, Sub};

use crate::units::Real;

/// Index of a node in its [`Graph`].
pub(crate) type Id = u32;

/// Comparison directions (the result is a boolean node).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Cmp {
    Lt,
    Gt,
}

/// A primitive operation. Operands are earlier nodes, so a graph is always in topological order.
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum Op {
    /// The graph's `n`-th input.
    Input(u32),
    /// A constant; stored as f64, evaluated and emitted at the graph's precision (f32).
    Const(f64),
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
    /// A boolean node.
    Compare(Cmp, Id, Id),
    /// `select(mask, if_true, if_false)`.
    Select(Id, Id, Id),
}

impl Op {
    /// Whether this node is boolean (a mask) rather than real.
    pub fn is_mask(&self) -> bool {
        matches!(self, Op::Compare(..))
    }
}

/// A traced program: nodes in topological order, the number of inputs, and the output nodes.
///
/// Built by [`trace`]. Scalar f32 for now; shaped values (broadcast, dot, FFT) come later.
#[derive(Clone, Debug, PartialEq, Default)]
pub struct Graph {
    pub(crate) nodes: Vec<Op>,
    pub(crate) inputs: usize,
    pub(crate) outputs: Vec<Id>,
}

impl Graph {
    pub fn nodes(&self) -> &[Op] {
        &self.nodes
    }
    pub fn inputs(&self) -> usize {
        self.inputs
    }
    pub fn outputs(&self) -> usize {
        self.outputs.len()
    }

    /// Replays this graph into the current trace with `args` as its inputs; returns its outputs.
    /// This is how one traced function calls another (the call is inlined).
    pub fn call(&self, args: &[Tracer]) -> Vec<Tracer> {
        assert_eq!(args.len(), self.inputs, "Graph::call: wrong number of arguments");
        let mut ids: Vec<Id> = Vec::with_capacity(self.nodes.len());
        for op in &self.nodes {
            let m = |i: &Id| ids[*i as usize];
            let id = match *op {
                Op::Input(n) => args[n as usize].check(),
                Op::Const(v) => push(Op::Const(v)),
                Op::Add(a, b) => push(Op::Add(m(&a), m(&b))),
                Op::Sub(a, b) => push(Op::Sub(m(&a), m(&b))),
                Op::Mul(a, b) => push(Op::Mul(m(&a), m(&b))),
                Op::Div(a, b) => push(Op::Div(m(&a), m(&b))),
                Op::Neg(a) => push(Op::Neg(m(&a))),
                Op::Exp(a) => push(Op::Exp(m(&a))),
                Op::Log(a) => push(Op::Log(m(&a))),
                Op::Sin(a) => push(Op::Sin(m(&a))),
                Op::Cos(a) => push(Op::Cos(m(&a))),
                Op::Tanh(a) => push(Op::Tanh(m(&a))),
                Op::Sqrt(a) => push(Op::Sqrt(m(&a))),
                Op::Abs(a) => push(Op::Abs(m(&a))),
                Op::Pow(a, b) => push(Op::Pow(m(&a), m(&b))),
                Op::Min(a, b) => push(Op::Min(m(&a), m(&b))),
                Op::Max(a, b) => push(Op::Max(m(&a), m(&b))),
                Op::Compare(c, a, b) => push(Op::Compare(c, m(&a), m(&b))),
                Op::Select(c, a, b) => push(Op::Select(m(&c), m(&a), m(&b))),
            };
            ids.push(id);
        }
        let trace = current_trace();
        self.outputs.iter().map(|&o| Tracer { id: ids[o as usize], trace }).collect()
    }
}

impl fmt::Display for Graph {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        for (i, op) in self.nodes.iter().enumerate() {
            writeln!(f, "%{i} = {op:?}")?;
        }
        write!(f, "return {:?}", self.outputs)
    }
}

/// A traced real number: a handle to a node in the graph being traced on this thread.
///
/// Implements [`Real`], so generic DSP code runs on it unchanged and records what it computes.
/// Only valid inside the [`trace`] that created it.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Tracer {
    pub(crate) id: Id,
    trace: u32,
}

/// A traced boolean (the result of a comparison), consumed by [`Real::select`].
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Mask {
    pub(crate) id: Id,
    trace: u32,
}

thread_local! {
    static GRAPH: RefCell<Option<(u32, Graph)>> = const { RefCell::new(None) };
    static NEXT_TRACE: Cell<u32> = const { Cell::new(0) };
}

/// Traces `f` with `inputs` fresh inputs and returns the graph of what it computed.
///
/// ```
/// use autodyne::flux::trace;
/// use autodyne::units::Real;
///
/// let g = trace(2, |v| vec![(v[0] * v[1]).exp()]);
/// assert_eq!(g.eval(&[1.0, 2.0]), vec![2.0f32.exp()]);
/// ```
pub fn trace(inputs: usize, f: impl FnOnce(&[Tracer]) -> Vec<Tracer>) -> Graph {
    let graph = Graph { nodes: (0..inputs as u32).map(Op::Input).collect(), inputs, outputs: Vec::new() };
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
    let args: Vec<Tracer> = (0..inputs as Id).map(|id| Tracer { id, trace }).collect();
    let outs = f(&args);
    let (_, mut graph) = GRAPH.with(|g| g.borrow_mut().take()).expect("flux::trace: graph missing");
    drop(reset);
    graph.outputs = outs
        .iter()
        .map(|t| {
            assert_eq!(t.trace, trace, "flux::trace: output belongs to another trace");
            t.id
        })
        .collect();
    graph
}

fn current_trace() -> u32 {
    GRAPH.with(|g| g.borrow().as_ref().map(|(t, _)| *t).expect("flux: tracer used outside flux::trace"))
}

/// The operation of node `id` in the current trace.
pub(crate) fn op(id: Id) -> Op {
    GRAPH.with(|g| g.borrow().as_ref().expect("flux: tracer used outside flux::trace").1.nodes[id as usize])
}

/// The number of nodes in the current trace so far.
pub(crate) fn len() -> usize {
    GRAPH.with(|g| g.borrow().as_ref().expect("flux: tracer used outside flux::trace").1.nodes.len())
}

fn push(op: Op) -> Id {
    GRAPH.with(|g| {
        let mut g = g.borrow_mut();
        let (_, graph) = g.as_mut().expect("flux: tracer used outside flux::trace");
        graph.nodes.push(op);
        (graph.nodes.len() - 1) as Id
    })
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

    fn unary(self, f: fn(Id) -> Op) -> Tracer {
        Tracer { id: push(f(self.check())), trace: self.trace }
    }

    fn binary(self, rhs: Tracer, f: fn(Id, Id) -> Op) -> Tracer {
        Tracer { id: push(f(self.check(), rhs.check())), trace: self.trace }
    }

    fn compare(self, rhs: Tracer, c: Cmp) -> Mask {
        Mask { id: push(Op::Compare(c, self.check(), rhs.check())), trace: self.trace }
    }
}

impl Mask {
    pub(crate) fn node(id: Id) -> Mask {
        Mask { id, trace: current_trace() }
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

impl Real for Tracer {
    type Mask = Mask;
    fn lit(v: f64) -> Self {
        Tracer { id: push(Op::Const(v)), trace: current_trace() }
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
    fn abs(self) -> Self {
        self.unary(Op::Abs)
    }
    fn powf(self, e: Self) -> Self {
        self.binary(e, Op::Pow)
    }
    fn min(self, other: Self) -> Self {
        self.binary(other, Op::Min)
    }
    fn max(self, other: Self) -> Self {
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
        Tracer { id: push(Op::Select(mask.id, if_true.check(), if_false.check())), trace: if_true.trace }
    }
}
