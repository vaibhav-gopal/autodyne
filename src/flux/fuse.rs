//! Element-wise fusion for the interpreter: a chain of element-wise operations on arrays of one
//! shape runs as one pass over blocks of elements, instead of one new array per operation (and per
//! broadcast scalar).
//!
//! An element-wise node with a single reader that is itself element-wise and of the same shape is
//! folded into that reader; a broadcast of a single value, or one cheap operation on values from
//! outside, is folded into every element-wise reader (recomputed in each).
//! Each remaining element-wise node (a root) is computed with what folded into it as a register
//! program ([`Program`]) over single elements, run over blocks of 256: one tight loop per
//! instruction and block. The scalar functions are the interpreter's, applied in the same order, so
//! the results are bit for bit the unfused ones.

use super::graph::{FluxFloat, Graph, Id, Kind, Node, Op};
use super::scalar::{Binary, Inst, Program, Unary};

const BLOCK: usize = 256;

/// The fusion plan of a graph.
#[derive(Debug, Default)]
pub(crate) struct Fusion {
    /// per node: a group computing it (roots), or nothing
    pub groups: Vec<Option<Group>>,
    /// per node: computed inside a group, never on its own
    pub absorbed: Vec<bool>,
    /// per node: a broadcast only arithmetic reads (`+ - * /`, which broadcast themselves): kept
    /// as its operand with axes of length 1, never stretched into an array of its own
    pub lazy: Vec<bool>,
}

/// A root and the nodes folded into it.
#[derive(Debug)]
pub(crate) struct Group {
    pub program: Program,
    /// the values it reads: single values first (`uniform` of them), then arrays of the root's shape
    pub inputs: Vec<Id>,
    pub uniform: usize,
}

fn single(shape: &[usize]) -> bool {
    shape.iter().product::<usize>() == 1
}

/// An element-wise operation, given its operands' shapes equal its own.
fn elementwise(op: &Op) -> bool {
    matches!(
        op,
        Op::Add(..)
            | Op::Sub(..)
            | Op::Mul(..)
            | Op::Div(..)
            | Op::Pow(..)
            | Op::Min(..)
            | Op::Max(..)
            | Op::Compare(..)
            | Op::Select(..)
            | Op::Neg(_)
            | Op::Exp(_)
            | Op::Log(_)
            | Op::Sin(_)
            | Op::Cos(_)
            | Op::Tanh(_)
            | Op::Sqrt(_)
            | Op::Abs(_)
            | Op::Floor(_)
    )
}

impl Fusion {
    pub fn plan(graph: &Graph) -> Fusion {
        let nodes = &graph.nodes;
        let n = nodes.len();
        // fusible: element-wise on arrays (more than one element) of its own shape, real or mask
        let fusible: Vec<bool> = nodes
            .iter()
            .map(|node| {
                node.kind != Kind::Complex
                    && !single(&node.shape)
                    && elementwise(&node.op)
                    && node.op.operands().all(|a| nodes[a as usize].shape == node.shape && nodes[a as usize].kind != Kind::Complex)
            })
            .collect();
        // a broadcast of one real value: folded into element-wise readers
        let splat: Vec<bool> = nodes
            .iter()
            .map(|node| matches!(node.op, Op::Broadcast(a, _) if single(&nodes[a as usize].shape) && nodes[a as usize].kind == Kind::Real) && !single(&node.shape))
            .collect();
        let mut readers: Vec<Vec<Id>> = vec![Vec::new(); n];
        for (i, node) in nodes.iter().enumerate() {
            let mut ops: Vec<Id> = node.op.operands().collect();
            ops.dedup();
            for a in ops {
                if !readers[a as usize].contains(&(i as Id)) {
                    readers[a as usize].push(i as Id);
                }
            }
        }
        let mut output = vec![false; n];
        graph.outputs.iter().for_each(|&o| output[o as usize] = true);
        let absorbed: Vec<bool> = (0..n)
            .map(|i| {
                let all_fusible = |rs: &[Id]| !rs.is_empty() && rs.iter().all(|&r| fusible[r as usize] && nodes[r as usize].shape == nodes[i].shape);
                if output[i] {
                    false
                } else if splat[i] {
                    all_fusible(&readers[i])
                } else if readers[i].len() == 1 {
                    fusible[i] && all_fusible(&readers[i])
                } else {
                    // read several times: recomputed in each reader when it is one cheap operation
                    // on values from outside (cheaper than a pass and an array of its own)
                    let cheap = matches!(nodes[i].op, Op::Add(..) | Op::Sub(..) | Op::Mul(..) | Op::Neg(_) | Op::Abs(_));
                    let leaf = nodes[i].op.operands().all(|a| !fusible[a as usize] || splat[a as usize]);
                    fusible[i] && cheap && leaf && all_fusible(&readers[i])
                }
            })
            .collect();
        let mut groups: Vec<Option<Group>> = (0..n).map(|_| None).collect();
        for i in 0..n {
            if fusible[i] && !absorbed[i] {
                groups[i] = group(nodes, i, &absorbed, &splat);
            }
        }
        // a node folded into a root that could not be compiled must be computed after all
        let mut absorbed_ok = vec![false; n];
        for (i, g) in groups.iter().enumerate() {
            if g.is_some() {
                mark(nodes, i, &absorbed, &splat, &mut absorbed_ok);
            }
        }
        let lazy = (0..n)
            .map(|i| {
                matches!(nodes[i].op, Op::Broadcast(..))
                    && !output[i]
                    && !absorbed_ok[i]
                    && groups[i].is_none()
                    && !readers[i].is_empty()
                    && readers[i].iter().all(|&r| {
                        let r = r as usize;
                        groups[r].is_none() && !absorbed_ok[r] && matches!(nodes[r].op, Op::Add(..) | Op::Sub(..) | Op::Mul(..) | Op::Div(..)) && nodes[r].shape == nodes[i].shape
                    })
            })
            .collect();
        Fusion { groups, absorbed: absorbed_ok, lazy }
    }

    /// The values node `i` reads when the graph is evaluated with this plan (none if absorbed).
    pub fn reads(&self, i: usize, node: &Node) -> Vec<Id> {
        if self.absorbed[i] {
            Vec::new()
        } else if let Some(g) = &self.groups[i] {
            g.inputs.clone()
        } else {
            node.op.operands().collect()
        }
    }
}

/// Marks the nodes folded into root `i`.
fn mark(nodes: &[Node], i: usize, absorbed: &[bool], splat: &[bool], out: &mut [bool]) {
    for a in nodes[i].op.operands() {
        let a = a as usize;
        if absorbed[a] && !out[a] {
            out[a] = true;
            if !splat[a] {
                mark(nodes, a, absorbed, splat, out);
            }
        }
    }
}

/// Root `i`'s group: the nodes folded into it, in order, as a program over single elements.
fn group(nodes: &[Node], root: usize, absorbed: &[bool], splat: &[bool]) -> Option<Group> {
    // members (folded nodes and the root), external inputs
    let mut members = Vec::new();
    let mut external: Vec<Id> = Vec::new();
    let mut stack = vec![root];
    let mut seen = std::collections::HashSet::new();
    while let Some(i) = stack.pop() {
        if !seen.insert(i) {
            continue;
        }
        members.push(i);
        if splat[i] {
            continue;
        }
        for a in nodes[i].op.operands() {
            let a = a as usize;
            if absorbed[a] {
                stack.push(a);
            } else if !external.contains(&(a as Id)) {
                external.push(a as Id);
            }
        }
    }
    // a splat reads its single value from outside
    for &m in &members {
        if splat[m] {
            if let Op::Broadcast(a, _) = nodes[m].op {
                if !external.contains(&a) {
                    external.push(a);
                }
            }
        }
    }
    members.sort_unstable();
    let (uniform, arrays): (Vec<Id>, Vec<Id>) = external.into_iter().partition(|&a| single(&nodes[a as usize].shape));
    let inputs: Vec<Id> = uniform.iter().chain(&arrays).copied().collect();
    // the group as a graph over single elements: inputs, then the members
    let mut map = std::collections::HashMap::new();
    let mut sub = Graph::default();
    for (k, &a) in inputs.iter().enumerate() {
        map.insert(a, k as Id);
        sub.inputs.push(Vec::new());
        let kind = if nodes[a as usize].kind == Kind::Mask { Kind::Mask } else { Kind::Real };
        sub.nodes.push(Node { op: Op::Input(k as u32), shape: Vec::new(), kind });
    }
    for &m in &members {
        let id = sub.nodes.len() as Id;
        let op = if splat[m] {
            let Op::Broadcast(a, _) = nodes[m].op else { unreachable!() };
            Op::Reshape(map[&a])
        } else {
            nodes[m].op.map(|a| map[&a])
        };
        sub.nodes.push(Node { op, shape: Vec::new(), kind: nodes[m].kind });
        map.insert(m as Id, id);
    }
    sub.outputs = vec![map[&(root as Id)]];
    let program = Program::compile(&sub, uniform.len())?;
    Some(Group { program, inputs, uniform: uniform.len() })
}

/// An operand of one block: varying (a slice) or the same for every element.
#[derive(Clone, Copy)]
enum Src<T> {
    Varying(*const T),
    Uniform(T),
}

impl<T: Copy> Src<T> {
    #[inline(always)]
    fn at(self, i: usize) -> T {
        match self {
            // SAFETY: varying registers hold a block of `m <= BLOCK` values, i < m
            Src::Varying(p) => unsafe { *p.add(i) },
            Src::Uniform(v) => v,
        }
    }
}

impl<T: Copy> Src<T> {
    /// The block a varying operand points at.
    #[inline(always)]
    fn slice<'a>(p: *const T, m: usize) -> &'a [T] {
        // SAFETY: varying operands point at `m` readable values (an input array's or a register's
        // block), not overlapping the destination
        unsafe { std::slice::from_raw_parts(p, m) }
    }
}

#[inline(always)]
fn unary<T: Copy>(dst: &mut [T], a: Src<T>, f: impl Fn(T) -> T) {
    match a {
        Src::Varying(p) => {
            let m = dst.len();
            dst.iter_mut().zip(Src::slice(p, m)).for_each(|(d, &x)| *d = f(x))
        }
        Src::Uniform(x) => dst.fill(f(x)),
    }
}

#[inline(always)]
fn binary<T: Copy>(dst: &mut [T], a: Src<T>, b: Src<T>, f: impl Fn(T, T) -> T) {
    let m = dst.len();
    match (a, b) {
        (Src::Varying(p), Src::Varying(q)) => dst.iter_mut().zip(Src::slice(p, m)).zip(Src::slice(q, m)).for_each(|((d, &x), &y)| *d = f(x, y)),
        (Src::Varying(p), Src::Uniform(y)) => dst.iter_mut().zip(Src::slice(p, m)).for_each(|(d, &x)| *d = f(x, y)),
        (Src::Uniform(x), Src::Varying(q)) => dst.iter_mut().zip(Src::slice(q, m)).for_each(|(d, &y)| *d = f(x, y)),
        (Src::Uniform(x), Src::Uniform(y)) => dst.fill(f(x, y)),
    }
}

impl Group {
    /// Runs the group over `len` elements: `uniform` holds the single-value inputs, `arrays` the
    /// others (each `len` long, masks as 1 / 0). Returns the root's values.
    pub fn run<T: FluxFloat>(&self, len: usize, uniform: &[T], arrays: &[&[T]]) -> Vec<T> {
        let p = &self.program;
        let mut scalars = vec![T::_ZERO; p.registers];
        scalars[..uniform.len()].copy_from_slice(uniform);
        Program::exec(&p.prologue, &mut scalars);
        // which registers vary from element to element
        let mut varying = vec![false; p.registers];
        (self.uniform..p.inputs).for_each(|k| varying[k] = true);
        for inst in &p.body {
            varying[inst.dst() as usize] = true;
        }
        let out_reg = p.outputs[0] as usize;
        let mut out = Vec::with_capacity(len);
        if !varying[out_reg] {
            out.resize(len, scalars[out_reg]);
            return out;
        }
        let mut block = vec![T::_ZERO; (p.registers - p.inputs) * BLOCK + p.inputs * BLOCK];
        let mut start = 0;
        while start < len {
            let m = BLOCK.min(len - start);
            let base = block.as_mut_ptr();
            let src = |r: u32| {
                let r = r as usize;
                if !varying[r] {
                    Src::Uniform(scalars[r])
                } else if r < p.inputs {
                    // an input array, read in place
                    Src::Varying(arrays[r - self.uniform][start..].as_ptr())
                } else {
                    // SAFETY: register r's block lies inside `block`
                    Src::Varying(unsafe { base.add(r * BLOCK) } as *const T)
                }
            };
            for inst in &p.body {
                let (dst, op) = (inst.dst() as usize, inst);
                // SAFETY: the destination register differs from every operand register (each node has
                // its own), so this block does not overlap the ones read
                let d = unsafe { std::slice::from_raw_parts_mut(base.add(dst * BLOCK), m) };
                match *op {
                    Inst::Const { value, .. } => d.fill(T::_lit(value)),
                    Inst::Unary { op, a, .. } => {
                        let a = src(a);
                        match op {
                            Unary::Neg => unary(d, a, |x| -x),
                            Unary::Exp => unary(d, a, |x| x.exp()),
                            Unary::Log => unary(d, a, |x| x.ln()),
                            Unary::Sin => unary(d, a, |x| x.sin()),
                            Unary::Cos => unary(d, a, |x| x.cos()),
                            Unary::Tanh => unary(d, a, |x| x.tanh()),
                            Unary::Sqrt => unary(d, a, |x| x.sqrt()),
                            Unary::Abs => unary(d, a, |x| x.abs()),
                            Unary::Floor => unary(d, a, |x| x.floor()),
                        }
                    }
                    Inst::Binary { op, a, b, .. } => {
                        let (a, b) = (src(a), src(b));
                        let mask = |m: bool| if m { T::_ONE } else { T::_ZERO };
                        match op {
                            Binary::Add => binary(d, a, b, |x, y| x + y),
                            Binary::Sub => binary(d, a, b, |x, y| x - y),
                            Binary::Mul => binary(d, a, b, |x, y| x * y),
                            Binary::Div => binary(d, a, b, |x, y| x / y),
                            Binary::Pow => binary(d, a, b, |x, y| x.powf(y)),
                            Binary::Min => binary(d, a, b, |x, y| x.minimum(y)),
                            Binary::Max => binary(d, a, b, |x, y| x.maximum(y)),
                            Binary::Lt => binary(d, a, b, |x, y| mask(x < y)),
                            Binary::Gt => binary(d, a, b, |x, y| mask(x > y)),
                            Binary::Eq => binary(d, a, b, |x, y| mask(x == y)),
                        }
                    }
                    Inst::Select { m: c, a, b, .. } => {
                        let (c, a, b) = (src(c), src(a), src(b));
                        d.iter_mut().enumerate().for_each(|(i, v)| *v = if c.at(i) != T::_ZERO { a.at(i) } else { b.at(i) });
                    }
                    Inst::MulAdd { a, b, c, .. } => {
                        let (a, b, c) = (src(a), src(b), src(c));
                        d.iter_mut().enumerate().for_each(|(i, v)| *v = super::scalar::mul_add(a.at(i), b.at(i), c.at(i)));
                    }
                }
            }
            if out_reg < p.inputs {
                out.extend_from_slice(&arrays[out_reg - self.uniform][start..start + m]);
            } else {
                out.extend_from_slice(&block[out_reg * BLOCK..out_reg * BLOCK + m]);
            }
            start += m;
        }
        out
    }
}
