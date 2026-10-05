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
//!
//! tend: flux / in process

use super::graph::{FluxFloat, Graph, Id, Kind, Node, Op};
use super::scalar::{Binary, Inst, Program, Unary};

const BLOCK: usize = 256;

/// Constant arrays converted to an element type, per node: made on first use.
#[derive(Default)]
pub(crate) struct Literals(std::sync::Mutex<std::collections::HashMap<(usize, std::any::TypeId), std::sync::Arc<dyn std::any::Any + Send + Sync>>>);

impl std::fmt::Debug for Literals {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str("Literals")
    }
}

impl Literals {
    /// Node `i`'s constant `data` in `T`, converted once.
    pub fn get<T: FluxFloat>(&self, i: usize, data: &[f64]) -> std::sync::Arc<Vec<T>> {
        let key = (i, std::any::TypeId::of::<T>());
        let mut map = self.0.lock().expect("the literal cache is not poisoned");
        let entry = map.entry(key).or_insert_with(|| std::sync::Arc::new(data.iter().map(|&v| T::_lit(v)).collect::<Vec<T>>()));
        entry.clone().downcast::<Vec<T>>().expect("cached under its own type")
    }
}

/// The fusion plan of a graph.
#[derive(Debug, Default)]
pub(crate) struct Fusion {
    /// constant arrays already converted, per element type
    pub literals: Literals,
    /// per node: the values its evaluation reads (none if absorbed)
    pub reads: Vec<Vec<Id>>,
    /// per node: how many times evaluating the graph reads it (outputs count once each)
    pub uses: Vec<u32>,
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
    /// the values it reads, each once
    pub fetch: Vec<Id>,
    /// its inputs as (index into `fetch`, part: 0 the value itself, 1 / 2 the real / imaginary part
    /// of a complex value): single values first (`uniform` of them), then arrays of the root's shape
    pub inputs: Vec<(usize, u8)>,
    pub uniform: usize,
    /// per register: whether it changes from element to element
    varying: Vec<bool>,
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
        // the real or imaginary part of a complex array: folded into element-wise readers too, which
        // read it in place from the complex values (1: real, 2: imaginary)
        let part: Vec<u8> = nodes
            .iter()
            .map(|node| match node.op {
                Op::Re(a) if !single(&node.shape) && nodes[a as usize].shape == node.shape => 1,
                Op::Im(a) if !single(&node.shape) && nodes[a as usize].shape == node.shape => 2,
                _ => 0,
            })
            .collect();
        let leaf: Vec<bool> = (0..n).map(|i| splat[i] || part[i] != 0).collect();
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
                } else if leaf[i] {
                    all_fusible(&readers[i])
                } else if readers[i].len() == 1 {
                    fusible[i] && all_fusible(&readers[i])
                } else {
                    // read several times: recomputed in each reader when it is one cheap operation
                    // on values from outside (cheaper than a pass and an array of its own)
                    let cheap = matches!(nodes[i].op, Op::Add(..) | Op::Sub(..) | Op::Mul(..) | Op::Neg(_) | Op::Abs(_));
                    let outside = nodes[i].op.operands().all(|a| !fusible[a as usize] || leaf[a as usize]);
                    fusible[i] && cheap && outside && all_fusible(&readers[i])
                }
            })
            .collect();
        let mut groups: Vec<Option<Group>> = (0..n).map(|_| None).collect();
        for i in 0..n {
            if fusible[i] && !absorbed[i] {
                groups[i] = group(nodes, i, &absorbed, &splat, &part);
            }
        }
        // a node folded into a root that could not be compiled must be computed after all
        let mut absorbed_ok = vec![false; n];
        for (i, g) in groups.iter().enumerate() {
            if g.is_some() {
                mark(nodes, i, &absorbed, &leaf, &mut absorbed_ok);
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
        let mut plan = Fusion { groups, absorbed: absorbed_ok, lazy, literals: Literals::default(), reads: Vec::new(), uses: vec![0; n] };
        plan.reads = (0..n).map(|i| plan.reads_of(i, &nodes[i])).collect();
        for r in &plan.reads {
            r.iter().for_each(|&a| plan.uses[a as usize] += 1);
        }
        graph.outputs.iter().for_each(|&o| plan.uses[o as usize] += 1);
        plan
    }

    /// The values node `i` reads when the graph is evaluated with this plan (none if absorbed).
    fn reads_of(&self, i: usize, node: &Node) -> Vec<Id> {
        if self.absorbed[i] {
            Vec::new()
        } else if let Some(g) = &self.groups[i] {
            g.fetch.clone()
        } else {
            node.op.operands().collect()
        }
    }
}

/// Marks the nodes folded into root `i`.
fn mark(nodes: &[Node], i: usize, absorbed: &[bool], leaf: &[bool], out: &mut [bool]) {
    for a in nodes[i].op.operands() {
        let a = a as usize;
        if absorbed[a] && !out[a] {
            out[a] = true;
            if !leaf[a] {
                mark(nodes, a, absorbed, leaf, out);
            }
        }
    }
}

/// Root `i`'s group: the nodes folded into it, in order, as a program over single elements.
fn group(nodes: &[Node], root: usize, absorbed: &[bool], splat: &[bool], part: &[u8]) -> Option<Group> {
    // members (folded nodes and the root), external inputs as (node, part)
    let mut members = Vec::new();
    let mut external: Vec<(Id, u8)> = Vec::new();
    let add = |key: (Id, u8), external: &mut Vec<(Id, u8)>| {
        if !external.contains(&key) {
            external.push(key);
        }
    };
    let mut stack = vec![root];
    let mut seen = std::collections::HashSet::new();
    while let Some(i) = stack.pop() {
        if !seen.insert(i) {
            continue;
        }
        members.push(i);
        if splat[i] || part[i] != 0 {
            continue;
        }
        for a in nodes[i].op.operands() {
            if absorbed[a as usize] {
                stack.push(a as usize);
            } else {
                add((a, 0), &mut external);
            }
        }
    }
    // a splat reads its single value from outside, a part its complex array
    for &m in &members {
        match nodes[m].op {
            Op::Broadcast(a, _) if splat[m] => add((a, 0), &mut external),
            Op::Re(a) | Op::Im(a) if part[m] != 0 => add((a, part[m]), &mut external),
            _ => {}
        }
    }
    members.sort_unstable();
    type Keys = Vec<(Id, u8)>;
    let (uniform, arrays): (Keys, Keys) = external.into_iter().partition(|&(a, _)| single(&nodes[a as usize].shape));
    let keys: Vec<(Id, u8)> = uniform.iter().chain(&arrays).copied().collect();
    // the group as a graph over single elements: inputs, then the members
    let mut map: std::collections::HashMap<(Id, u8), Id> = std::collections::HashMap::new();
    let mut sub = Graph::default();
    for (k, &(a, p)) in keys.iter().enumerate() {
        map.insert((a, p), k as Id);
        sub.inputs.push(Vec::new());
        let kind = if p == 0 && nodes[a as usize].kind == Kind::Mask { Kind::Mask } else { Kind::Real };
        sub.nodes.push(Node { op: Op::Input(k as u32), shape: Vec::new(), kind });
    }
    for &m in &members {
        let id = sub.nodes.len() as Id;
        let op = match nodes[m].op {
            Op::Broadcast(a, _) if splat[m] => Op::Reshape(map[&(a, 0)]),
            Op::Re(a) | Op::Im(a) if part[m] != 0 => Op::Reshape(map[&(a, part[m])]),
            ref op => op.map(|a| map[&(a, 0)]),
        };
        sub.nodes.push(Node { op, shape: Vec::new(), kind: nodes[m].kind });
        map.insert((m as Id, 0), id);
    }
    sub.outputs = vec![map[&(root as Id, 0)]];
    let program = Program::compile(&sub, uniform.len())?;
    let mut varying = vec![false; program.registers];
    (uniform.len()..program.inputs).for_each(|k| varying[k] = true);
    for inst in &program.body {
        varying[inst.dst() as usize] = true;
    }
    let mut fetch: Vec<Id> = Vec::new();
    let inputs = keys
        .iter()
        .map(|&(a, p)| {
            let at = fetch.iter().position(|&f| f == a).unwrap_or_else(|| {
                fetch.push(a);
                fetch.len() - 1
            });
            (at, p)
        })
        .collect();
    Some(Group { program, fetch, inputs, uniform: uniform.len(), varying })
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
    pub fn run<T: FluxFloat>(&self, len: usize, uniform: &[T], arrays: &[(&[T], usize)]) -> Vec<T> {
        #[cfg(target_arch = "x86_64")]
        if crate::simd::avx2_available() {
            // SAFETY: AVX2 support was just checked
            return unsafe { self.run_avx2(len, uniform, arrays) };
        }
        self.run_body(len, uniform, arrays)
    }

    /// The same loops compiled for AVX2 (wider vectors, and the transcendentals vectorize there).
    /// Plain IEEE operations only, nothing contracted: the results are the baseline build's.
    #[cfg(target_arch = "x86_64")]
    #[target_feature(enable = "avx2")]
    unsafe fn run_avx2<T: FluxFloat>(&self, len: usize, uniform: &[T], arrays: &[(&[T], usize)]) -> Vec<T> {
        self.run_body(len, uniform, arrays)
    }

    #[inline(always)]
    fn run_body<T: FluxFloat>(&self, len: usize, uniform: &[T], arrays: &[(&[T], usize)]) -> Vec<T> {
        let p = &self.program;
        let mut scalars = vec![T::_ZERO; p.registers];
        scalars[..uniform.len()].copy_from_slice(uniform);
        Program::exec(&p.prologue, &mut scalars);
        let varying = &self.varying;
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
            // strided inputs (parts of complex arrays) gathered into their registers' blocks
            for (k, &(a, stride)) in arrays.iter().enumerate() {
                if stride != 1 {
                    let r = self.uniform + k;
                    for (slot, &v) in block[r * BLOCK..r * BLOCK + m].iter_mut().zip(a[start * stride..].iter().step_by(stride)) {
                        *slot = v;
                    }
                }
            }
            let base = block.as_mut_ptr();
            let src = |r: u32| {
                let r = r as usize;
                if !varying[r] {
                    Src::Uniform(scalars[r])
                } else if r < p.inputs && arrays[r - self.uniform].1 == 1 {
                    // an input array, read in place
                    Src::Varying(arrays[r - self.uniform].0[start..].as_ptr())
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
            if out_reg < p.inputs && arrays[out_reg - self.uniform].1 == 1 {
                out.extend_from_slice(&arrays[out_reg - self.uniform].0[start..start + m]);
            } else {
                out.extend_from_slice(&block[out_reg * BLOCK..out_reg * BLOCK + m]);
            }
            start += m;
        }
        out
    }
}
