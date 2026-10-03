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
            }
        }
    }
}

/// A scalar scan's programs: the step, the step saving its residuals and the reverse step reading
/// them, and the reverse step recomputing the step (checkpointed).
#[derive(Clone, Debug, PartialEq)]
pub(crate) struct ScalarScan {
    pub params: usize,
    pub states: usize,
    pub residuals: usize,
    pub step: Program,
    pub step_fwd: Program,
    pub step_bwd: Program,
    pub step_vjp: Program,
}

impl ScalarScan {
    pub fn compile(step: &Graph, step_fwd: &Graph, step_bwd: &Graph, step_vjp: &Graph, params: usize, states: usize, residuals: usize) -> Option<Arc<ScalarScan>> {
        Some(Arc::new(ScalarScan {
            params,
            states,
            residuals,
            step: Program::compile(step, params)?,
            step_fwd: Program::compile(step_fwd, params)?,
            step_bwd: Program::compile(step_bwd, params)?,
            step_vjp: Program::compile(step_vjp, params)?,
        }))
    }

    /// Registers holding the parameters, with the prologue run.
    fn registers<T: FluxFloat>(program: &Program, params: &[T]) -> Vec<T> {
        let mut r = vec![T::_ZERO; program.registers];
        r[..params.len()].copy_from_slice(params);
        Program::exec(&program.prologue, &mut r);
        r
    }

    /// The outputs and the final state.
    pub fn run<T: FluxFloat>(&self, params: &[T], xs: &[T], s0: &[T]) -> (Vec<T>, Vec<T>) {
        let (p, s) = (self.params, self.states);
        let program = &self.step;
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
    pub fn forward<T: FluxFloat>(&self, params: &[T], xs: &[T], s0: &[T], checkpointed: bool) -> (Vec<T>, Vec<T>) {
        let (p, s) = (self.params, self.states);
        let (program, r_count) = if checkpointed { (&self.step, 0) } else { (&self.step_fwd, self.residuals) };
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
    pub fn backward<T: FluxFloat>(&self, params: &[T], xs: &[T], saved: &[T], dys: &[T], checkpointed: bool) -> (Vec<T>, Vec<T>, Vec<T>) {
        let (p, s) = (self.params, self.states);
        let (program, r_count) = if checkpointed { (&self.step_vjp, 0) } else { (&self.step_bwd, self.residuals) };
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
