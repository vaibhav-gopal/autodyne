//! Scalar scans, compiled: when every parameter, state value, sample and output of a scan is a single
//! number (a per-sample filter, an envelope, an oscillator), its step graphs become flat register
//! programs. Values that depend on the parameters alone (filter coefficients) are computed once,
//! before the loop; each step runs only the rest, on registers, without allocating.
//!
//! Every instruction is the same scalar function the interpreter applies element by element, in the
//! same order, so a compiled scan's results are bit for bit the interpreter's.

use std::sync::Arc;

use super::graph::{Cmp, FluxFloat, Graph, Kind, Op};

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum Unary {
    Neg,
    Exp,
    Log,
    Sin,
    Cos,
    Tanh,
    Sqrt,
    Abs,
    Floor,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum Binary {
    Add,
    Sub,
    Mul,
    Div,
    Pow,
    Min,
    Max,
    /// Masks are 1 or 0.
    Lt,
    Gt,
    Eq,
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub(crate) enum Inst {
    Const { dst: u32, value: f64 },
    Unary { dst: u32, op: Unary, a: u32 },
    Binary { dst: u32, op: Binary, a: u32, b: u32 },
    /// `if m != 0 { a } else { b }`.
    Select { dst: u32, m: u32, a: u32, b: u32 },
    /// `a * b + c`, rounded once (from [`Program::contract`]).
    MulAdd { dst: u32, a: u32, b: u32, c: u32 },
}

impl Inst {
    pub fn dst(&self) -> u32 {
        match *self {
            Inst::Const { dst, .. } | Inst::Unary { dst, .. } | Inst::Binary { dst, .. } | Inst::Select { dst, .. } | Inst::MulAdd { dst, .. } => dst,
        }
    }
    fn operands(&self) -> Vec<u32> {
        match *self {
            Inst::Const { .. } => vec![],
            Inst::Unary { a, .. } => vec![a],
            Inst::Binary { a, b, .. } => vec![a, b],
            Inst::Select { m, a, b, .. } => vec![m, a, b],
            Inst::MulAdd { a, b, c, .. } => vec![a, b, c],
        }
    }
}

/// A step graph as register code: registers `0..inputs` hold the inputs; `prologue` computes what
/// depends on the first `invariant` inputs alone, `body` the rest.
#[derive(Clone, Debug, PartialEq)]
pub(crate) struct Program {
    pub inputs: usize,
    pub invariant: usize,
    pub registers: usize,
    pub prologue: Vec<Inst>,
    pub body: Vec<Inst>,
    pub outputs: Vec<u32>,
}

impl Program {
    /// The graph as a program, if all its values are single real numbers (or masks) and its
    /// operations scalar ones; the first `invariant` inputs stay fixed from step to step.
    pub fn compile(graph: &Graph, invariant: usize) -> Option<Program> {
        let single = |shape: &[usize]| shape.iter().product::<usize>() == 1;
        if !graph.inputs.iter().all(|s| single(s)) {
            return None;
        }
        let nodes = &graph.nodes;
        let mut reg = vec![u32::MAX; nodes.len()];
        let mut fixed = vec![false; nodes.len()];
        let mut next = graph.inputs.len() as u32;
        let (mut prologue, mut body) = (Vec::new(), Vec::new());
        for (i, node) in nodes.iter().enumerate() {
            if !single(&node.shape) || node.kind == Kind::Complex {
                return None;
            }
            let un = |op, a: u32| (Inst::Unary { dst: 0, op, a: reg[a as usize] }, fixed[a as usize]);
            let bin = |op, a: u32, b: u32| (Inst::Binary { dst: 0, op, a: reg[a as usize], b: reg[b as usize] }, fixed[a as usize] && fixed[b as usize]);
            let (inst, invariant_value) = match node.op {
                Op::Input(k) => {
                    reg[i] = k;
                    fixed[i] = (k as usize) < invariant;
                    continue;
                }
                // one element in, the same element out
                Op::Broadcast(a, _)
                | Op::Reshape(a)
                | Op::Transpose(a, _)
                | Op::Sum(a, _)
                | Op::Reduce(a, _, _)
                | Op::Reverse(a, _)
                | Op::Slice { a, .. }
                | Op::Pad { a, .. } => {
                    reg[i] = reg[a as usize];
                    fixed[i] = fixed[a as usize];
                    continue;
                }
                Op::Concat(ref parts, _) if parts.len() == 1 => {
                    reg[i] = reg[parts[0] as usize];
                    fixed[i] = fixed[parts[0] as usize];
                    continue;
                }
                Op::Const(value) => (Inst::Const { dst: 0, value }, true),
                Op::Literal(ref data) => (Inst::Const { dst: 0, value: data[0] }, true),
                Op::Neg(a) => un(Unary::Neg, a),
                Op::Exp(a) => un(Unary::Exp, a),
                Op::Log(a) => un(Unary::Log, a),
                Op::Sin(a) => un(Unary::Sin, a),
                Op::Cos(a) => un(Unary::Cos, a),
                Op::Tanh(a) => un(Unary::Tanh, a),
                Op::Sqrt(a) => un(Unary::Sqrt, a),
                Op::Abs(a) => un(Unary::Abs, a),
                Op::Floor(a) => un(Unary::Floor, a),
                Op::Add(a, b) => bin(Binary::Add, a, b),
                Op::Sub(a, b) => bin(Binary::Sub, a, b),
                Op::Mul(a, b) => bin(Binary::Mul, a, b),
                Op::Div(a, b) => bin(Binary::Div, a, b),
                Op::Pow(a, b) => bin(Binary::Pow, a, b),
                Op::Min(a, b) => bin(Binary::Min, a, b),
                Op::Max(a, b) => bin(Binary::Max, a, b),
                Op::Compare(Cmp::Lt, a, b) => bin(Binary::Lt, a, b),
                Op::Compare(Cmp::Gt, a, b) => bin(Binary::Gt, a, b),
                Op::Compare(Cmp::Eq, a, b) => bin(Binary::Eq, a, b),
                Op::Select(m, a, b) => (
                    Inst::Select { dst: 0, m: reg[m as usize], a: reg[a as usize], b: reg[b as usize] },
                    fixed[m as usize] && fixed[a as usize] && fixed[b as usize],
                ),
                _ => return None,
            };
            let dst = next;
            next += 1;
            let inst = match inst {
                Inst::Const { value, .. } => Inst::Const { dst, value },
                Inst::Unary { op, a, .. } => Inst::Unary { dst, op, a },
                Inst::Binary { op, a, b, .. } => Inst::Binary { dst, op, a, b },
                Inst::Select { m, a, b, .. } => Inst::Select { dst, m, a, b },
                Inst::MulAdd { .. } => unreachable!("compiled graphs have no fused instructions"),
            };
            reg[i] = dst;
            fixed[i] = invariant_value;
            if invariant_value { prologue.push(inst) } else { body.push(inst) }
        }
        Some(Program {
            inputs: graph.inputs.len(),
            invariant,
            registers: next as usize,
            prologue,
            body,
            outputs: graph.outputs.iter().map(|&o| reg[o as usize]).collect(),
        })
    }

    /// This program with each product read only by one sum fused into it (`a * b + c` and
    /// `a * b - c` rounded once): a shorter chain of dependent operations, and results that differ
    /// from the separately rounded ones in the last bits. Within the prologue and within the body.
    pub fn contract(&self) -> Program {
        let mut uses = vec![0u32; self.registers];
        for inst in self.prologue.iter().chain(&self.body) {
            inst.operands().into_iter().for_each(|r| uses[r as usize] += 1);
        }
        self.outputs.iter().for_each(|&r| uses[r as usize] += 1);
        let mut registers = self.registers as u32;
        let mut fuse = |code: &[Inst]| -> Vec<Inst> {
            let product = |r: u32| code.iter().find_map(|i| match *i {
                Inst::Binary { dst, op: Binary::Mul, a, b } if dst == r && uses[r as usize] == 1 => Some((a, b)),
                _ => None,
            });
            let mut fused = vec![false; registers as usize];
            let mut out = Vec::with_capacity(code.len());
            for inst in code {
                match *inst {
                    Inst::Binary { dst, op: Binary::Add, a, b } => {
                        if let Some((x, y)) = product(a) {
                            fused[a as usize] = true;
                            out.push(Inst::MulAdd { dst, a: x, b: y, c: b });
                        } else if let Some((x, y)) = product(b) {
                            fused[b as usize] = true;
                            out.push(Inst::MulAdd { dst, a: x, b: y, c: a });
                        } else {
                            out.push(*inst);
                        }
                    }
                    Inst::Binary { dst, op: Binary::Sub, a, b } if product(a).is_some() => {
                        // a * b - c = a * b + (-c), the negation exact
                        let (x, y) = product(a).expect("checked");
                        fused[a as usize] = true;
                        let neg = registers;
                        registers += 1;
                        fused.push(false);
                        out.push(Inst::Unary { dst: neg, op: Unary::Neg, a: b });
                        out.push(Inst::MulAdd { dst, a: x, b: y, c: neg });
                    }
                    _ => out.push(*inst),
                }
            }
            // drop the products now inside a fused instruction
            out.retain(|i| !matches!(*i, Inst::Binary { dst, op: Binary::Mul, .. } if fused.get(dst as usize).copied().unwrap_or(false)));
            out
        };
        let prologue = fuse(&self.prologue);
        let body = fuse(&self.body);
        Program { inputs: self.inputs, invariant: self.invariant, registers: registers as usize, prologue, body, outputs: self.outputs.clone() }
    }

    /// Runs `code` on the registers.
    #[inline]
    pub fn exec<T: FluxFloat>(code: &[Inst], r: &mut [T]) {
        for inst in code {
            match *inst {
                Inst::Const { dst, value } => r[dst as usize] = T::_lit(value),
                Inst::Unary { dst, op, a } => {
                    let x = r[a as usize];
                    r[dst as usize] = match op {
                        Unary::Neg => -x,
                        Unary::Exp => x.exp(),
                        Unary::Log => x.ln(),
                        Unary::Sin => x.sin(),
                        Unary::Cos => x.cos(),
                        Unary::Tanh => x.tanh(),
                        Unary::Sqrt => x.sqrt(),
                        Unary::Abs => x.abs(),
                        Unary::Floor => x.floor(),
                    };
                }
                Inst::Binary { dst, op, a, b } => {
                    let (x, y) = (r[a as usize], r[b as usize]);
                    let mask = |m: bool| if m { T::_ONE } else { T::_ZERO };
                    r[dst as usize] = match op {
                        Binary::Add => x + y,
                        Binary::Sub => x - y,
                        Binary::Mul => x * y,
                        Binary::Div => x / y,
                        Binary::Pow => x.powf(y),
                        Binary::Min => x.minimum(y),
                        Binary::Max => x.maximum(y),
                        Binary::Lt => mask(x < y),
                        Binary::Gt => mask(x > y),
                        Binary::Eq => mask(x == y),
                    };
                }
                Inst::Select { dst, m, a, b } => r[dst as usize] = if r[m as usize] != T::_ZERO { r[a as usize] } else { r[b as usize] },
                Inst::MulAdd { dst, a, b, c } => r[dst as usize] = mul_add(r[a as usize], r[b as usize], r[c as usize]),
            }
        }
    }
}

/// `a * b + c` rounded once, as `f32::mul_add` / `f64::mul_add` compute it.
#[inline]
pub(crate) fn mul_add<T: FluxFloat>(a: T, b: T, c: T) -> T {
    use std::any::Any;
    if let (Some(&a), Some(&b), Some(&c)) = ((&a as &dyn Any).downcast_ref::<f32>(), (&b as &dyn Any).downcast_ref::<f32>(), (&c as &dyn Any).downcast_ref::<f32>()) {
        return T::_lit(a.mul_add(b, c) as f64);
    }
    let (a, b, c) = (a.to_f64().unwrap_or(f64::NAN), b.to_f64().unwrap_or(f64::NAN), c.to_f64().unwrap_or(f64::NAN));
    T::_lit(a.mul_add(b, c))
}

/// A scalar scan's programs: the step, the step saving its residuals and the reverse step reading
/// them, and the reverse step recomputing the step (checkpointed).
#[derive(Clone, Debug)]
pub(crate) struct ScalarScan {
    pub params: usize,
    pub states: usize,
    pub residuals: usize,
    pub step: Program,
    pub step_fwd: Program,
    pub step_bwd: Program,
    pub step_vjp: Program,
    /// the same four with products fused into sums (`Scan::contracted`)
    pub contracted: [Program; 4],
    /// the programs as machine code, per element type and contraction, compiled on first use
    #[cfg(feature = "jit")]
    jit: [std::sync::OnceLock<Option<Arc<super::jit::Jit>>>; 4],
}

impl ScalarScan {
    pub fn compile(step: &Graph, step_fwd: &Graph, step_bwd: &Graph, step_vjp: &Graph, params: usize, states: usize, residuals: usize) -> Option<Arc<ScalarScan>> {
        let (step, step_fwd, step_bwd, step_vjp) =
            (Program::compile(step, params)?, Program::compile(step_fwd, params)?, Program::compile(step_bwd, params)?, Program::compile(step_vjp, params)?);
        let contracted = [step.contract(), step_fwd.contract(), step_bwd.contract(), step_vjp.contract()];
        Some(Arc::new(ScalarScan {
            params,
            states,
            residuals,
            step,
            step_fwd,
            step_bwd,
            step_vjp,
            contracted,
            #[cfg(feature = "jit")]
            jit: Default::default(),
        }))
    }

    /// The step, the step saving residuals, the reverse step reading them, and the reverse step
    /// recomputing the step: as written, or with products fused into sums.
    pub(crate) fn programs(&self, contract: bool) -> [&Program; 4] {
        if contract {
            let [a, b, c, d] = &self.contracted;
            [a, b, c, d]
        } else {
            [&self.step, &self.step_fwd, &self.step_bwd, &self.step_vjp]
        }
    }

    /// The compiled loops for `T` (`None` without the `jit` feature, or if Cranelift cannot
    /// target this machine).
    #[cfg(feature = "jit")]
    pub(crate) fn native<T: FluxFloat>(&self, contract: bool) -> Option<&super::jit::Jit> {
        let cell = &self.jit[usize::from(size_of::<T>() == 8) * 2 + usize::from(contract)];
        cell.get_or_init(|| super::jit::Jit::compile::<T>(self, contract).map(Arc::new)).as_deref()
    }

    /// Registers holding the parameters, with the prologue run.
    fn registers<T: FluxFloat>(program: &Program, params: &[T]) -> Vec<T> {
        let mut r = vec![T::_ZERO; program.registers];
        r[..params.len()].copy_from_slice(params);
        Program::exec(&program.prologue, &mut r);
        r
    }

    /// The outputs and the final state.
    pub fn run<T: FluxFloat>(&self, params: &[T], xs: &[T], s0: &[T], contract: bool) -> (Vec<T>, Vec<T>) {
        #[cfg(feature = "jit")]
        if let Some(jit) = self.native::<T>(contract) {
            let mut state = s0.to_vec();
            let mut ys = Vec::with_capacity(xs.len());
            // SAFETY: the buffers have the sizes the compiled loop reads and writes, and it writes
            // every output step
            unsafe {
                jit.run::<T>()(params.as_ptr(), state.as_mut_ptr(), xs.as_ptr(), ys.as_mut_ptr(), std::ptr::null_mut(), xs.len());
                ys.set_len(xs.len());
            }
            return (ys, state);
        }
        let (p, s) = (self.params, self.states);
        let program = self.programs(contract)[0];
        let mut r = Self::registers(program, params);
        let mut state = s0.to_vec();
        let mut ys = Vec::with_capacity(xs.len());
        for &x in xs {
            r[p..p + s].copy_from_slice(&state);
            r[p + s] = x;
            Program::exec(&program.body, &mut r);
            for (k, v) in state.iter_mut().enumerate() {
                *v = r[program.outputs[k] as usize];
            }
            ys.push(r[program.outputs[s] as usize]);
        }
        (ys, state)
    }

    /// The outputs, and for each step its input state followed by its residuals (none when
    /// `checkpointed`).
    pub fn forward<T: FluxFloat>(&self, params: &[T], xs: &[T], s0: &[T], checkpointed: bool, contract: bool) -> (Vec<T>, Vec<T>) {
        #[cfg(feature = "jit")]
        if let Some(jit) = self.native::<T>(contract) {
            let per = self.states + if checkpointed { 0 } else { self.residuals };
            let mut state = s0.to_vec();
            let mut ys = Vec::with_capacity(xs.len());
            let mut saved = Vec::with_capacity(xs.len() * per);
            // SAFETY: as in `run`; every step's saved values are written too
            unsafe {
                jit.forward::<T>(checkpointed)(params.as_ptr(), state.as_mut_ptr(), xs.as_ptr(), ys.as_mut_ptr(), saved.as_mut_ptr(), xs.len());
                ys.set_len(xs.len());
                saved.set_len(xs.len() * per);
            }
            return (ys, saved);
        }
        let (p, s) = (self.params, self.states);
        let [step, step_fwd, ..] = self.programs(contract);
        let (program, r_count) = if checkpointed { (step, 0) } else { (step_fwd, self.residuals) };
        let mut r = Self::registers(program, params);
        let mut state = s0.to_vec();
        let mut ys = Vec::with_capacity(xs.len());
        let mut saved = Vec::with_capacity(xs.len() * (s + r_count));
        for &x in xs {
            saved.extend_from_slice(&state);
            r[p..p + s].copy_from_slice(&state);
            r[p + s] = x;
            Program::exec(&program.body, &mut r);
            for (k, v) in state.iter_mut().enumerate() {
                *v = r[program.outputs[k] as usize];
            }
            ys.push(r[program.outputs[s] as usize]);
            saved.extend(program.outputs[s + 1..s + 1 + r_count].iter().map(|&o| r[o as usize]));
        }
        (ys, saved)
    }

    /// The reverse scan: the cotangents of the parameters, the initial state and the samples.
    pub fn backward<T: FluxFloat>(&self, params: &[T], xs: &[T], saved: &[T], dys: &[T], checkpointed: bool, contract: bool) -> (Vec<T>, Vec<T>, Vec<T>) {
        #[cfg(feature = "jit")]
        if let Some(jit) = self.native::<T>(contract) {
            let mut d_params = vec![T::_ZERO; self.params];
            let mut d_state = vec![T::_ZERO; self.states];
            let mut d_xs = Vec::with_capacity(xs.len());
            // SAFETY: as in `run`; every step's d x is written
            unsafe {
                jit.backward::<T>(checkpointed)(params.as_ptr(), xs.as_ptr(), saved.as_ptr(), dys.as_ptr(), d_params.as_mut_ptr(), d_state.as_mut_ptr(), d_xs.as_mut_ptr(), xs.len());
                d_xs.set_len(xs.len());
            }
            return (d_params, d_state, d_xs);
        }
        let (p, s) = (self.params, self.states);
        let [_, _, step_bwd, step_vjp] = self.programs(contract);
        let (program, r_count) = if checkpointed { (step_vjp, 0) } else { (step_bwd, self.residuals) };
        let mut r = Self::registers(program, params);
        let mut d_params = vec![T::_ZERO; p];
        let mut d_state = vec![T::_ZERO; s];
        let mut d_xs = vec![T::_ZERO; xs.len()];
        let per = s + r_count;
        // inputs: params, state, x, residuals, d state', d y
        let (at_x, at_res, at_ds) = (p + s, p + s + 1, p + s + 1 + r_count);
        for i in (0..xs.len()).rev() {
            let row = &saved[i * per..(i + 1) * per];
            r[p..p + s].copy_from_slice(&row[..s]);
            r[at_x] = xs[i];
            r[at_res..at_res + r_count].copy_from_slice(&row[s..]);
            r[at_ds..at_ds + s].copy_from_slice(&d_state);
            r[at_ds + s] = dys[i];
            Program::exec(&program.body, &mut r);
            for (k, d) in d_params.iter_mut().enumerate() {
                *d = *d + r[program.outputs[k] as usize];
            }
            for (k, d) in d_state.iter_mut().enumerate() {
                *d = r[program.outputs[p + k] as usize];
            }
            d_xs[i] = r[program.outputs[p + s] as usize];
        }
        (d_params, d_state, d_xs)
    }
}
