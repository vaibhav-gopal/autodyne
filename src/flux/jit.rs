//! Scalar scans as machine code (feature `jit`): Cranelift compiles a [`ScalarScan`]'s register
//! programs into native loops, in process: the step's arithmetic on machine registers, the
//! parameter-only prologue before the loop, nothing allocated or dispatched per step.
//!
//! Arithmetic the IEEE standard rounds exactly (add, subtract, multiply, divide, square root,
//! negation, absolute value, floor, comparisons) is emitted as instructions; transcendentals, powers
//! and min / max call the same Rust functions the interpreter applies, so results stay bit for bit
//! the interpreter's (unless the scan is contracted: then products fused into sums are one ma).

use cranelift_codegen::ir::condcodes::{FloatCC, IntCC};
use cranelift_codegen::ir::{types, AbiParam, FuncRef, InstBuilder, MemFlagsData, Signature, Type, UserFuncName, Value};
use cranelift_codegen::settings::{self, Configurable};
use cranelift_frontend::{FunctionBuilder, FunctionBuilderContext, Variable};
use cranelift_jit::{JITBuilder, JITModule};
use cranelift_module::{default_libcall_names, FuncId, Linkage, Module};

use super::graph::FluxFloat;
use super::scalar::{Binary, Inst, Program, ScalarScan, Unary};

/// `(params, state in / out, xs, ys, saved, n)`: runs `n` steps; with `saved` non-null, writes each
/// step's input state and residuals there.
pub(crate) type ForwardFn<T> = unsafe extern "C" fn(*const T, *mut T, *const T, *mut T, *mut T, usize);
/// `(params, xs, saved, dys, d params out, d state out, d xs out, n)`: the reverse scan.
pub(crate) type BackwardFn<T> = unsafe extern "C" fn(*const T, *const T, *const T, *const T, *mut T, *mut T, *mut T, usize);

/// A scalar scan's compiled loops, for one element type.
pub(crate) struct Jit {
    module: Option<JITModule>,
    run: *const u8,
    forward: *const u8,
    forward_checkpointed: *const u8,
    backward: *const u8,
    backward_checkpointed: *const u8,
}

// SAFETY: the code is immutable once finalized and the functions keep no state; the module is only
// touched again to free it, on drop
unsafe impl Send for Jit {}
unsafe impl Sync for Jit {}

impl Drop for Jit {
    fn drop(&mut self) {
        if let Some(module) = self.module.take() {
            // SAFETY: nothing holds the function pointers past the Jit that owns them
            unsafe { module.free_memory() };
        }
    }
}

impl std::fmt::Debug for Jit {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str("Jit")
    }
}

impl Jit {
    /// The step's run loop: no residuals saved.
    pub fn run<T>(&self) -> ForwardFn<T> {
        // SAFETY: compiled with this signature for T (the Jit is cached per element type)
        unsafe { std::mem::transmute::<*const u8, ForwardFn<T>>(self.run) }
    }
    pub fn forward<T>(&self, checkpointed: bool) -> ForwardFn<T> {
        let f = if checkpointed { self.forward_checkpointed } else { self.forward };
        // SAFETY: as in `run`
        unsafe { std::mem::transmute::<*const u8, ForwardFn<T>>(f) }
    }
    pub fn backward<T>(&self, checkpointed: bool) -> BackwardFn<T> {
        let f = if checkpointed { self.backward_checkpointed } else { self.backward };
        // SAFETY: as in `run`
        unsafe { std::mem::transmute::<*const u8, BackwardFn<T>>(f) }
    }

    /// Compiles the scan's loops for `T`; `None` if Cranelift cannot target this machine.
    pub fn compile<T: FluxFloat>(scan: &ScalarScan, contract: bool) -> Option<Jit> {
        let mut flags = settings::builder();
        flags.set("use_colocated_libcalls", "false").ok()?;
        flags.set("is_pic", "false").ok()?;
        flags.set("opt_level", "speed").ok()?;
        let isa = cranelift_native::builder().ok()?.finish(settings::Flags::new(flags)).ok()?;
        let mut builder = JITBuilder::with_isa(isa, default_libcall_names());
        for (name, f) in helpers::<T>() {
            builder.symbol(name, f);
        }
        let mut module = JITModule::new(builder);
        let ty = if size_of::<T>() == 4 { types::F32 } else { types::F64 };
        let (p, s, r) = (scan.params, scan.states, scan.residuals);
        let mut cx = Codegen { module: &mut module, ty, helpers: Vec::new(), names: 0 };
        let [step, step_fwd, step_bwd, step_vjp] = scan.programs(contract);
        let run = cx.forward::<T>(step, p, s, 0, false)?;
        let forward = cx.forward::<T>(step_fwd, p, s, r, true)?;
        let forward_checkpointed = cx.forward::<T>(step, p, s, 0, true)?;
        let backward = cx.backward::<T>(step_bwd, p, s, r)?;
        let backward_checkpointed = cx.backward::<T>(step_vjp, p, s, 0)?;
        module.finalize_definitions().ok()?;
        let get = |id| module.get_finalized_function(id);
        Some(Jit {
            run: get(run),
            forward: get(forward),
            forward_checkpointed: get(forward_checkpointed),
            backward: get(backward),
            backward_checkpointed: get(backward_checkpointed),
            module: Some(module),
        })
    }
}

extern "C" fn exp<T: FluxFloat>(x: T) -> T {
    x.exp()
}
extern "C" fn ln<T: FluxFloat>(x: T) -> T {
    x.ln()
}
extern "C" fn sin<T: FluxFloat>(x: T) -> T {
    x.sin()
}
extern "C" fn cos<T: FluxFloat>(x: T) -> T {
    x.cos()
}
extern "C" fn tanh<T: FluxFloat>(x: T) -> T {
    x.tanh()
}
extern "C" fn pow<T: FluxFloat>(x: T, y: T) -> T {
    x.powf(y)
}
extern "C" fn min<T: FluxFloat>(x: T, y: T) -> T {
    x.minimum(y)
}
extern "C" fn max<T: FluxFloat>(x: T, y: T) -> T {
    x.maximum(y)
}

/// The Rust functions compiled code calls, by symbol.
fn helpers<T: FluxFloat>() -> [(&'static str, *const u8); 8] {
    [
        ("flux_exp", exp::<T> as extern "C" fn(T) -> T as *const u8),
        ("flux_ln", ln::<T> as extern "C" fn(T) -> T as *const u8),
        ("flux_sin", sin::<T> as extern "C" fn(T) -> T as *const u8),
        ("flux_cos", cos::<T> as extern "C" fn(T) -> T as *const u8),
        ("flux_tanh", tanh::<T> as extern "C" fn(T) -> T as *const u8),
        ("flux_pow", pow::<T> as extern "C" fn(T, T) -> T as *const u8),
        ("flux_min", min::<T> as extern "C" fn(T, T) -> T as *const u8),
        ("flux_max", max::<T> as extern "C" fn(T, T) -> T as *const u8),
    ]
}

struct Codegen<'m> {
    module: &'m mut JITModule,
    ty: Type,
    /// helper functions declared in the module: (symbol, id)
    helpers: Vec<(&'static str, FuncId)>,
    names: u32,
}

/// Emits one function's instructions, reading and writing `vals` (a value per register).
struct Emitter<'a, 'b> {
    b: &'a mut FunctionBuilder<'b>,
    ty: Type,
    /// helper calls declared in this function: (symbol, reference)
    refs: Vec<(&'static str, FuncRef)>,
}

impl Codegen<'_> {
    fn helper(&mut self, name: &'static str, arity: usize) -> Option<FuncId> {
        if let Some(&(_, id)) = self.helpers.iter().find(|(n, _)| *n == name) {
            return Some(id);
        }
        let mut sig = self.module.make_signature();
        sig.params.extend(std::iter::repeat_n(AbiParam::new(self.ty), arity));
        sig.returns.push(AbiParam::new(self.ty));
        let id = self.module.declare_function(name, Linkage::Import, &sig).ok()?;
        self.helpers.push((name, id));
        Some(id)
    }

    /// The helpers `code` calls, declared in the module.
    fn declare_helpers(&mut self, code: &[&[Inst]]) -> Option<Vec<(&'static str, FuncId)>> {
        let mut out = Vec::new();
        for inst in code.iter().flat_map(|c| c.iter()) {
            let (name, arity) = match *inst {
                Inst::Unary { op: Unary::Exp, .. } => ("flux_exp", 1),
                Inst::Unary { op: Unary::Log, .. } => ("flux_ln", 1),
                Inst::Unary { op: Unary::Sin, .. } => ("flux_sin", 1),
                Inst::Unary { op: Unary::Cos, .. } => ("flux_cos", 1),
                Inst::Unary { op: Unary::Tanh, .. } => ("flux_tanh", 1),
                Inst::Binary { op: Binary::Pow, .. } => ("flux_pow", 2),
                Inst::Binary { op: Binary::Min, .. } => ("flux_min", 2),
                Inst::Binary { op: Binary::Max, .. } => ("flux_max", 2),
                _ => continue,
            };
            if !out.iter().any(|(n, _)| *n == name) {
                out.push((name, self.helper(name, arity)?));
            }
        }
        Some(out)
    }

    fn signature(&self, pointers: usize) -> Signature {
        let ptr = self.module.target_config().pointer_type();
        let mut sig = self.module.make_signature();
        sig.params.extend(std::iter::repeat_n(AbiParam::new(ptr), pointers + 1));
        sig
    }

    fn define(&mut self, sig: Signature, build: impl FnOnce(&mut FunctionBuilder, &[(&'static str, FuncId)], &mut JITModule), helpers: Vec<(&'static str, FuncId)>) -> Option<FuncId> {
        self.names += 1;
        let id = self.module.declare_function(&format!("flux_scan_{}", self.names), Linkage::Local, &sig).ok()?;
        let mut ctx = self.module.make_context();
        ctx.func.signature = sig;
        ctx.func.name = UserFuncName::user(0, id.as_u32());
        let mut fcx = FunctionBuilderContext::new();
        let target = self.module.target_config();
        {
            let mut b = FunctionBuilder::new(&mut ctx.func, &mut fcx);
            build(&mut b, &helpers, self.module);
            b.seal_all_blocks();
            b.finalize(target);
        }
        if std::env::var_os("FLUX_JIT_DUMP").is_some() {
            eprintln!("{}", ctx.func.display());
        }
        self.module.define_function(id, &mut ctx).ok()?;
        self.module.clear_context(&mut ctx);
        Some(id)
    }

    /// The forward loop: `(params, state, xs, ys, saved, n)`.
    fn forward<T: FluxFloat>(&mut self, program: &Program, p: usize, s: usize, r: usize, save: bool) -> Option<FuncId> {
        let helpers = self.declare_helpers(&[&program.prologue, &program.body])?;
        let (ty, size) = (self.ty, size_of::<T>() as i64);
        let ptr = self.module.target_config().pointer_type();
        let sig = self.signature(5);
        self.define(sig, |b, helpers, module| {
            let entry = b.create_block();
            b.append_block_params_for_function_params(entry);
            b.switch_to_block(entry);
            let [params, state, xs, ys, saved, n] = b.block_params(entry)[..] else { unreachable!() };
            let refs = helpers.iter().map(|&(name, id)| (name, module.declare_func_in_func(id, b.func))).collect();
            let flags = MemFlagsData::trusted();
            let mut vals: Vec<Option<Value>> = vec![None; program.registers];
            for (k, v) in vals.iter_mut().enumerate().take(p) {
                *v = Some(b.ins().load(ty, flags, params, (k as i64 * size) as i32));
            }
            let mut e = Emitter { b, ty, refs };
            e.emit::<T>(&program.prologue, &mut vals);
            let b = e.b;
            let state_vars: Vec<Variable> = (0..s)
                .map(|k| {
                    let v = b.declare_var(ty);
                    let x = b.ins().load(ty, flags, state, (k as i64 * size) as i32);
                    b.def_var(v, x);
                    v
                })
                .collect();
            let i_var = b.declare_var(ptr);
            let zero = b.ins().iconst(ptr, 0);
            b.def_var(i_var, zero);
            let (header, body, exit) = (b.create_block(), b.create_block(), b.create_block());
            b.ins().jump(header, &[]);

            b.switch_to_block(header);
            let i = b.use_var(i_var);
            let more = b.ins().icmp(IntCC::UnsignedLessThan, i, n);
            b.ins().brif(more, body, &[], exit, &[]);

            b.switch_to_block(body);
            let i = b.use_var(i_var);
            let offset = b.ins().imul_imm_s(i, size);
            let x_at = b.ins().iadd(xs, offset);
            let x = b.ins().load(ty, flags, x_at, 0);
            let current: Vec<Value> = state_vars.iter().map(|&v| b.use_var(v)).collect();
            let base = if save {
                let row = b.ins().imul_imm_s(i, (s + r) as i64 * size);
                let base = b.ins().iadd(saved, row);
                for (k, &v) in current.iter().enumerate() {
                    b.ins().store(flags, v, base, (k as i64 * size) as i32);
                }
                Some(base)
            } else {
                None
            };
            let mut step = vals.clone();
            for (k, &v) in current.iter().enumerate() {
                step[p + k] = Some(v);
            }
            step[p + s] = Some(x);
            let mut e = Emitter { b, ty, refs: std::mem::take(&mut e.refs) };
            e.emit::<T>(&program.body, &mut step);
            let b = e.b;
            let out = |k: usize| step[program.outputs[k] as usize].expect("an output value");
            let y_at = b.ins().iadd(ys, offset);
            b.ins().store(flags, out(s), y_at, 0);
            if let Some(base) = base {
                for j in 0..r {
                    b.ins().store(flags, out(s + 1 + j), base, ((s + j) as i64 * size) as i32);
                }
            }
            for (k, &v) in state_vars.iter().enumerate() {
                b.def_var(v, out(k));
            }
            let next = b.ins().iadd_imm_s(i, 1);
            b.def_var(i_var, next);
            b.ins().jump(header, &[]);

            b.switch_to_block(exit);
            for (k, &v) in state_vars.iter().enumerate() {
                let x = b.use_var(v);
                b.ins().store(flags, x, state, (k as i64 * size) as i32);
            }
            b.ins().return_(&[]);
        }, helpers)
    }

    /// The reverse loop: `(params, xs, saved, dys, d params, d state, d xs, n)`.
    fn backward<T: FluxFloat>(&mut self, program: &Program, p: usize, s: usize, r: usize) -> Option<FuncId> {
        let helpers = self.declare_helpers(&[&program.prologue, &program.body])?;
        let (ty, size) = (self.ty, size_of::<T>() as i64);
        let ptr = self.module.target_config().pointer_type();
        let sig = self.signature(7);
        self.define(sig, |b, helpers, module| {
            let entry = b.create_block();
            b.append_block_params_for_function_params(entry);
            b.switch_to_block(entry);
            let [params, xs, saved, dys, d_params, d_state, d_xs, n] = b.block_params(entry)[..] else { unreachable!() };
            let refs = helpers.iter().map(|&(name, id)| (name, module.declare_func_in_func(id, b.func))).collect();
            let flags = MemFlagsData::trusted();
            let mut vals: Vec<Option<Value>> = vec![None; program.registers];
            for (k, v) in vals.iter_mut().enumerate().take(p) {
                *v = Some(b.ins().load(ty, flags, params, (k as i64 * size) as i32));
            }
            let mut e = Emitter { b, ty, refs };
            e.emit::<T>(&program.prologue, &mut vals);
            let b = e.b;
            let zero = e_zero(b, ty);
            let var = |b: &mut FunctionBuilder| {
                let v = b.declare_var(ty);
                b.def_var(v, zero);
                v
            };
            let dp_vars: Vec<Variable> = (0..p).map(|_| var(b)).collect();
            let ds_vars: Vec<Variable> = (0..s).map(|_| var(b)).collect();
            let i_var = b.declare_var(ptr);
            b.def_var(i_var, n);
            let (header, body, exit) = (b.create_block(), b.create_block(), b.create_block());
            b.ins().jump(header, &[]);

            b.switch_to_block(header);
            let i = b.use_var(i_var);
            let more = b.ins().icmp_imm_s(IntCC::NotEqual, i, 0);
            b.ins().brif(more, body, &[], exit, &[]);

            b.switch_to_block(body);
            let i = b.use_var(i_var);
            let k = b.ins().iadd_imm_s(i, -1);
            let offset = b.ins().imul_imm_s(k, size);
            let row = b.ins().imul_imm_s(k, (s + r) as i64 * size);
            let base = b.ins().iadd(saved, row);
            let mut step = vals.clone();
            for j in 0..s {
                step[p + j] = Some(b.ins().load(ty, flags, base, (j as i64 * size) as i32));
            }
            let x_at = b.ins().iadd(xs, offset);
            step[p + s] = Some(b.ins().load(ty, flags, x_at, 0));
            for j in 0..r {
                step[p + s + 1 + j] = Some(b.ins().load(ty, flags, base, ((s + j) as i64 * size) as i32));
            }
            let at_ds = p + s + 1 + r;
            for (j, &v) in ds_vars.iter().enumerate() {
                step[at_ds + j] = Some(b.use_var(v));
            }
            let dy_at = b.ins().iadd(dys, offset);
            step[at_ds + s] = Some(b.ins().load(ty, flags, dy_at, 0));
            let mut e = Emitter { b, ty, refs: std::mem::take(&mut e.refs) };
            e.emit::<T>(&program.body, &mut step);
            let b = e.b;
            let out = |j: usize| step[program.outputs[j] as usize].expect("an output value");
            for (j, &v) in dp_vars.iter().enumerate() {
                let acc = b.use_var(v);
                let sum = b.ins().fadd(acc, out(j));
                b.def_var(v, sum);
            }
            for (j, &v) in ds_vars.iter().enumerate() {
                b.def_var(v, out(p + j));
            }
            let dx_at = b.ins().iadd(d_xs, offset);
            b.ins().store(flags, out(p + s), dx_at, 0);
            b.def_var(i_var, k);
            b.ins().jump(header, &[]);

            b.switch_to_block(exit);
            for (j, &v) in dp_vars.iter().enumerate() {
                let x = b.use_var(v);
                b.ins().store(flags, x, d_params, (j as i64 * size) as i32);
            }
            for (j, &v) in ds_vars.iter().enumerate() {
                let x = b.use_var(v);
                b.ins().store(flags, x, d_state, (j as i64 * size) as i32);
            }
            b.ins().return_(&[]);
        }, helpers)
    }
}

fn e_zero(b: &mut FunctionBuilder, ty: Type) -> Value {
    if ty == types::F32 { b.ins().f32const(0.0f32) } else { b.ins().f64const(0.0f64) }
}

impl Emitter<'_, '_> {
    fn call(&mut self, name: &str, args: &[Value]) -> Value {
        let f = self.refs.iter().find(|(n, _)| *n == name).expect("a declared helper").1;
        let call = self.b.ins().call(f, args);
        self.b.inst_results(call)[0]
    }

    fn constant<T: FluxFloat>(&mut self, value: f64) -> Value {
        let v = T::_lit(value).to_f64().expect("a float");
        if self.ty == types::F32 { self.b.ins().f32const(v as f32) } else { self.b.ins().f64const(v) }
    }

    fn emit<T: FluxFloat>(&mut self, code: &[Inst], vals: &mut [Option<Value>]) {
        let get = |vals: &[Option<Value>], r: u32| vals[r as usize].expect("a register written before it is read");
        for inst in code {
            let (dst, v) = match *inst {
                Inst::Const { dst, value } => (dst, self.constant::<T>(value)),
                Inst::Unary { dst, op, a } => {
                    let x = get(vals, a);
                    let v = match op {
                        Unary::Neg => self.b.ins().fneg(x),
                        Unary::Sqrt => self.b.ins().sqrt(x),
                        Unary::Abs => self.b.ins().fabs(x),
                        Unary::Floor => self.b.ins().floor(x),
                        Unary::Exp => self.call("flux_exp", &[x]),
                        Unary::Log => self.call("flux_ln", &[x]),
                        Unary::Sin => self.call("flux_sin", &[x]),
                        Unary::Cos => self.call("flux_cos", &[x]),
                        Unary::Tanh => self.call("flux_tanh", &[x]),
                    };
                    (dst, v)
                }
                Inst::Binary { dst, op, a, b } => {
                    let (x, y) = (get(vals, a), get(vals, b));
                    let v = match op {
                        Binary::Add => self.b.ins().fadd(x, y),
                        Binary::Sub => self.b.ins().fsub(x, y),
                        Binary::Mul => self.b.ins().fmul(x, y),
                        Binary::Div => self.b.ins().fdiv(x, y),
                        Binary::Pow => self.call("flux_pow", &[x, y]),
                        Binary::Min => self.call("flux_min", &[x, y]),
                        Binary::Max => self.call("flux_max", &[x, y]),
                        Binary::Lt | Binary::Gt | Binary::Eq => {
                            let cc = match op {
                                Binary::Lt => FloatCC::LessThan,
                                Binary::Gt => FloatCC::GreaterThan,
                                _ => FloatCC::Equal,
                            };
                            let c = self.b.ins().fcmp(cc, x, y);
                            let (one, zero) = (self.constant::<T>(1.0), self.constant::<T>(0.0));
                            self.b.ins().select(c, one, zero)
                        }
                    };
                    (dst, v)
                }
                Inst::MulAdd { dst, a, b, c } => (dst, self.b.ins().fma(get(vals, a), get(vals, b), get(vals, c))),
                Inst::Select { dst, m, a, b } => {
                    let zero = self.constant::<T>(0.0);
                    let c = self.b.ins().fcmp(FloatCC::NotEqual, get(vals, m), zero);
                    (dst, self.b.ins().select(c, get(vals, a), get(vals, b)))
                }
            };
            vals[dst as usize] = Some(v);
        }
    }
}
