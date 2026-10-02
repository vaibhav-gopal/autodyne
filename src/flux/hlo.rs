//! Emits traced programs as textual StableHLO (MLIR), the input format of XLA (through PJRT) and
//! IREE. Values are `tensor<...xf32>` or `tensor<...xf64>` (masks `xi1`), the precision chosen at
//! emission; a scan is a `stablehlo.while` loop that slices one step per iteration out of the
//! signal's first axis.

use std::collections::HashMap;
use std::fmt::Write;

use super::graph::{Cmp, FluxFloat, Graph, Op, Part, Reduction};
use super::loss::Loss;
use super::scan::Scan;
use crate::units::DType;

/// A StableHLO module with one function, `@main`, the shapes of its inputs and outputs, and their
/// element type (`DType::F32` or `DType::F64`).
#[derive(Clone, Debug, PartialEq)]
pub struct Program {
    /// The module text (MLIR, StableHLO dialect).
    pub text: String,
    pub inputs: Vec<Vec<usize>>,
    pub outputs: Vec<Vec<usize>>,
    pub dtype: DType,
}

const INDEX: &str = "tensor<i32>";

/// The tensor type of a shape: `elem` is `f32`, `i1` or `complex<f32>`.
fn ty(shape: &[usize], elem: &str) -> String {
    let mut s = String::from("tensor<");
    for d in shape {
        write!(s, "{d}x").unwrap();
    }
    s.push_str(elem);
    s.push('>');
    s
}


fn list(xs: &[usize]) -> String {
    xs.iter().map(usize::to_string).collect::<Vec<_>>().join(", ")
}

/// An `array<i64: ...>` attribute.
fn i64s(xs: &[usize]) -> String {
    if xs.is_empty() {
        "array<i64>".into()
    } else {
        format!("array<i64: {}>", list(xs))
    }
}

/// A function body being written: SSA names are numbered once per function, so nested regions never
/// shadow a name.
struct Writer {
    out: String,
    next: usize,
    depth: usize,
    dtype: DType,
}

impl Writer {
    fn new(dtype: DType) -> Self {
        assert!(matches!(dtype, DType::F32 | DType::F64), "flux: programs are f32 or f64, not {dtype:?}");
        Writer { out: String::new(), next: 0, depth: 1, dtype }
    }

    fn elem(&self) -> &'static str {
        if self.dtype == DType::F64 { "f64" } else { "f32" }
    }

    /// The real tensor type of `shape`.
    fn real(&self, shape: &[usize]) -> String {
        ty(shape, self.elem())
    }

    fn complex(&self, shape: &[usize]) -> String {
        ty(shape, &format!("complex<{}>", self.elem()))
    }

    /// `v` rounded to the element type, as the bit pattern MLIR reads exactly.
    fn hex(&self, v: f64) -> String {
        if self.dtype == DType::F64 { format!("0x{:016X}", v.to_bits()) } else { format!("0x{:08X}", (v as f32).to_bits()) }
    }

    fn fresh(&mut self) -> String {
        self.next += 1;
        format!("%v{}", self.next - 1)
    }

    fn line(&mut self, text: &str) {
        for _ in 0..self.depth {
            self.out.push_str("  ");
        }
        self.out.push_str(text);
        self.out.push('\n');
    }

    /// `%name = <rhs>`; returns the name.
    fn emit(&mut self, rhs: &str) -> String {
        let name = self.fresh();
        self.line(&format!("{name} = {rhs}"));
        name
    }

    /// `v` (bit-exact, as hex) splatted over `shape`.
    fn splat(&mut self, v: f64, shape: &[usize]) -> String {
        let (hex, t) = (self.hex(v), self.real(shape));
        self.emit(&format!("stablehlo.constant dense<{hex}> : {t}"))
    }

    fn index(&mut self, v: i32) -> String {
        self.emit(&format!("stablehlo.constant dense<{v}> : {INDEX}"))
    }

    /// Writes `g` with `args` as its inputs; returns the names of its outputs.
    fn graph(&mut self, g: &Graph, args: &[String]) -> Vec<String> {
        assert_eq!(args.len(), g.inputs.len());
        let mut names: Vec<String> = Vec::with_capacity(g.nodes.len());
        // complex spectra by operand, so the real and imaginary parts share one FFT
        let mut spectra: HashMap<u32, String> = HashMap::new();
        let mut complex_spectra: HashMap<(u32, u32, bool), String> = HashMap::new();
        let elem = self.elem();
        let scalar = self.real(&[]);
        for node in &g.nodes {
            let n = |i: u32| names[i as usize].clone();
            let t = |i: u32| {
                let node = &g.nodes[i as usize];
                ty(&node.shape, if node.mask { "i1" } else { elem })
            };
            let out = ty(&node.shape, if node.mask { "i1" } else { elem });
            let unary = |name: &str, a: u32| format!("stablehlo.{name} {} : {out}", n(a));
            let binary = |name: &str, a: u32, b: u32| format!("stablehlo.{name} {}, {} : {out}", n(a), n(b));
            let rhs = match node.op {
                Op::Input(i) => {
                    names.push(args[i as usize].clone());
                    continue;
                }
                Op::Const(c) => format!("stablehlo.constant dense<{}> : {out}", self.hex(c)),
                Op::Literal(ref data) => {
                    let mut hex = String::with_capacity(2 + 16 * data.len());
                    hex.push_str("0x");
                    for &v in data.iter() {
                        let bytes = if self.dtype == DType::F64 { v.to_le_bytes().to_vec() } else { (v as f32).to_le_bytes().to_vec() };
                        for b in bytes {
                            write!(hex, "{b:02X}").unwrap();
                        }
                    }
                    format!("stablehlo.constant dense<\"{hex}\"> : {out}")
                }
                Op::Add(a, b) => binary("add", a, b),
                Op::Sub(a, b) => binary("subtract", a, b),
                Op::Mul(a, b) => binary("multiply", a, b),
                Op::Div(a, b) => binary("divide", a, b),
                Op::Pow(a, b) => binary("power", a, b),
                Op::Min(a, b) => binary("minimum", a, b),
                Op::Max(a, b) => binary("maximum", a, b),
                Op::Neg(a) => unary("negate", a),
                Op::Exp(a) => unary("exponential", a),
                Op::Log(a) => unary("log", a),
                Op::Sin(a) => unary("sine", a),
                Op::Cos(a) => unary("cosine", a),
                Op::Tanh(a) => unary("tanh", a),
                Op::Sqrt(a) => unary("sqrt", a),
                Op::Abs(a) => unary("abs", a),
                Op::Floor(a) => unary("floor", a),
                Op::Compare(c, a, b) => {
                    let dir = match c {
                        Cmp::Lt => "LT",
                        Cmp::Gt => "GT",
                        Cmp::Eq => "EQ",
                    };
                    format!("stablehlo.compare {dir}, {}, {}, FLOAT : ({}, {}) -> {out}", n(a), n(b), t(a), t(b))
                }
                Op::Select(c, a, b) => format!("stablehlo.select {}, {}, {} : ({}, {}, {}) -> {out}", n(c), n(a), n(b), t(c), t(a), t(b)),
                Op::Broadcast(a, ref dims) => format!("stablehlo.broadcast_in_dim {}, dims = [{}] : ({}) -> {out}", n(a), list(dims), t(a)),
                Op::Reshape(a) => format!("stablehlo.reshape {} : ({}) -> {out}", n(a), t(a)),
                Op::Transpose(a, ref perm) => format!("stablehlo.transpose {}, dims = [{}] : ({}) -> {out}", n(a), list(perm), t(a)),
                Op::Sum(a, ref axes) => {
                    let zero = self.splat(0.0, &[]);
                    format!("stablehlo.reduce({} init: {zero}) applies stablehlo.add across dimensions = [{}] : ({}, {scalar}) -> {out}", n(a), list(axes), t(a))
                }
                Op::Dot { a, b, ref ca, ref cb } => {
                    // full precision: GPUs would otherwise multiply f32 in TF32 (10-bit mantissas)
                    format!("stablehlo.dot_general {}, {}, contracting_dims = [{}] x [{}], precision = [HIGHEST, HIGHEST] : ({}, {}) -> {out}", n(a), n(b), list(ca), list(cb), t(a), t(b))
                }
                Op::Rfft(a, part) => {
                    let from = &g.nodes[a as usize].shape;
                    let spectrum = self.complex(&node.shape);
                    let c = match spectra.get(&a) {
                        Some(c) => c.clone(),
                        None => {
                            let len = from.last().unwrap();
                            let c = self.emit(&format!("stablehlo.fft {}, type = RFFT, length = [{len}] : ({}) -> {spectrum}", n(a), t(a)));
                            spectra.insert(a, c.clone());
                            c
                        }
                    };
                    let op = if part == Part::Re { "real" } else { "imag" };
                    format!("stablehlo.{op} {c} : ({spectrum}) -> {out}")
                }
                Op::Irfft { re, im, n: len } => {
                    let spectrum = self.complex(&g.nodes[re as usize].shape);
                    let c = self.emit(&format!("stablehlo.complex {}, {} : {spectrum}", n(re), n(im)));
                    format!("stablehlo.fft {c}, type = IRFFT, length = [{len}] : ({spectrum}) -> {out}")
                }
                // generic syntax for these three: it parses the same across StableHLO versions
                Op::Slice { a, ref start, ref limit, ref stride } => format!(
                    "\"stablehlo.slice\"({}) {{start_indices = {}, limit_indices = {}, strides = {}}} : ({}) -> {out}",
                    n(a),
                    i64s(start),
                    i64s(limit),
                    i64s(stride),
                    t(a)
                ),
                Op::Pad { a, ref low, ref high, ref interior } => {
                    let zero = self.splat(0.0, &[]);
                    format!(
                        "\"stablehlo.pad\"({}, {zero}) {{edge_padding_low = {}, edge_padding_high = {}, interior_padding = {}}} : ({}, {scalar}) -> {out}",
                        n(a),
                        i64s(low),
                        i64s(high),
                        i64s(interior),
                        t(a)
                    )
                }
                Op::Fft { re, im, inverse, part } => {
                    let spectrum = self.complex(&node.shape);
                    let c = match complex_spectra.get(&(re, im, inverse)) {
                        Some(c) => c.clone(),
                        None => {
                            let z = self.emit(&format!("stablehlo.complex {}, {} : {spectrum}", n(re), n(im)));
                            let kind = if inverse { "IFFT" } else { "FFT" };
                            let len = node.shape.last().unwrap();
                            let c = self.emit(&format!("stablehlo.fft {z}, type = {kind}, length = [{len}] : ({spectrum}) -> {spectrum}"));
                            complex_spectra.insert((re, im, inverse), c.clone());
                            c
                        }
                    };
                    let op = if part == Part::Re { "real" } else { "imag" };
                    format!("stablehlo.{op} {c} : ({spectrum}) -> {out}")
                }
                Op::Reduce(a, ref axes, r) => {
                    let (init, op) = match r {
                        Reduction::Max => (f64::NEG_INFINITY, "maximum"),
                        Reduction::Min => (f64::INFINITY, "minimum"),
                        Reduction::Prod => (1.0, "multiply"),
                    };
                    let init = self.splat(init, &[]);
                    format!("stablehlo.reduce({} init: {init}) applies stablehlo.{op} across dimensions = [{}] : ({}, {scalar}) -> {out}", n(a), list(axes), t(a))
                }
                Op::Reverse(a, ref axes) => format!("stablehlo.reverse {}, dims = [{}] : {out}", n(a), list(axes)),
                Op::Take { table, indices } => {
                    let (ts, is) = (&g.nodes[table as usize].shape, &g.nodes[indices as usize].shape);
                    let ix = self.row_indices(&n(indices), is, ts[0]);
                    let offset: Vec<usize> = (is.len()..is.len() + ts.len() - 1).collect();
                    let mut dims = Vec::new();
                    if !offset.is_empty() {
                        dims.push(format!("offset_dims = [{}]", list(&offset)));
                    }
                    dims.extend(["collapsed_slice_dims = [0]".to_string(), "start_index_map = [0]".into(), format!("index_vector_dim = {}", is.len())]);
                    let sizes: Vec<usize> = std::iter::once(1).chain(ts[1..].iter().copied()).collect();
                    format!(
                        "\"stablehlo.gather\"({}, {ix}) {{dimension_numbers = #stablehlo.gather<{}>, slice_sizes = {}, indices_are_sorted = false}} : ({}, {}) -> {out}",
                        n(table),
                        dims.join(", "),
                        i64s(&sizes),
                        t(table),
                        ty(is, "i32")
                    )
                }
                Op::ScatterAdd { indices, updates } => {
                    let (is, us) = (&g.nodes[indices as usize].shape, &g.nodes[updates as usize].shape);
                    let ix = self.row_indices(&n(indices), is, node.shape[0]);
                    let zeros = self.splat(0.0, &node.shape);
                    let window: Vec<usize> = (is.len()..us.len()).collect();
                    let mut dims = Vec::new();
                    if !window.is_empty() {
                        dims.push(format!("update_window_dims = [{}]", list(&window)));
                    }
                    dims.extend(["inserted_window_dims = [0]".to_string(), "scatter_dims_to_operand_dims = [0]".into(), format!("index_vector_dim = {}", is.len())]);
                    let (x, y, s) = (self.fresh(), self.fresh(), self.fresh());
                    format!(
                        "\"stablehlo.scatter\"({zeros}, {ix}, {}) ({{\n^bb0({x}: {scalar}, {y}: {scalar}):\n  {s} = stablehlo.add {x}, {y} : {scalar}\n  stablehlo.return {s} : {scalar}\n}}) {{scatter_dimension_numbers = #stablehlo.scatter<{}>, indices_are_sorted = false, unique_indices = false}} : ({out}, {}, {}) -> {out}",
                        n(updates),
                        dims.join(", "),
                        ty(is, "i32"),
                        t(updates)
                    )
                }
                Op::Concat(ref parts, axis) => {
                    let names: Vec<String> = parts.iter().map(|&p| n(p)).collect();
                    let types: Vec<String> = parts.iter().map(|&p| t(p)).collect();
                    format!("\"stablehlo.concatenate\"({}) {{dimension = {axis} : i64}} : ({}) -> {out}", names.join(", "), types.join(", "))
                }
            };
            let name = self.emit(&rhs);
            names.push(name);
        }
        g.outputs.iter().map(|&o| names[o as usize].clone()).collect()
    }

    /// Real indices into `rows` rows as `i32`s: rounded down and clamped (the gather would clamp
    /// anyway, but the scatter that is its gradient drops out-of-range rows instead).
    fn row_indices(&mut self, v: &str, shape: &[usize], rows: usize) -> String {
        let real_t = self.real(shape);
        let floor = self.emit(&format!("stablehlo.floor {v} : {real_t}"));
        let (lo, hi) = (self.splat(0.0, shape), self.splat((rows - 1) as f64, shape));
        let clamped = self.emit(&format!("\"stablehlo.clamp\"({lo}, {floor}, {hi}) : ({real_t}, {real_t}, {real_t}) -> {real_t}"));
        self.emit(&format!("stablehlo.convert {clamped} : ({real_t}) -> {}", ty(shape, "i32")))
    }

    /// Step `i` of `v` (shape `[len, rest...]`), shape `rest`.
    fn step_of(&mut self, v: &str, i: &str, shape: &[usize]) -> String {
        let rest = &shape[1..];
        let zero = if rest.is_empty() { String::new() } else { self.index(0) };
        let indices: Vec<&str> = std::iter::once(i).chain(rest.iter().map(|_| zero.as_str())).collect();
        let index_types = vec![INDEX; indices.len()].join(", ");
        let one = [&[1], rest].concat();
        let sliced = self.emit(&format!(
            "stablehlo.dynamic_slice {v}, {}, sizes = [{}] : ({}, {index_types}) -> {}",
            indices.join(", "),
            list(&one),
            self.real(shape),
            self.real(&one)
        ));
        self.emit(&format!("stablehlo.reshape {sliced} : ({}) -> {}", self.real(&one), self.real(rest)))
    }

    /// `v` (shape `[len, rest...]`) with step `i` replaced by `x` (shape `rest`).
    fn store(&mut self, v: &str, x: &str, i: &str, shape: &[usize]) -> String {
        let rest = &shape[1..];
        let one = [&[1], rest].concat();
        let x1 = self.emit(&format!("stablehlo.reshape {x} : ({}) -> {}", self.real(rest), self.real(&one)));
        let zero = if rest.is_empty() { String::new() } else { self.index(0) };
        let indices: Vec<&str> = std::iter::once(i).chain(rest.iter().map(|_| zero.as_str())).collect();
        let index_types = vec![INDEX; indices.len()].join(", ");
        self.emit(&format!(
            "stablehlo.dynamic_update_slice {v}, {x1}, {} : ({}, {}, {index_types}) -> {}",
            indices.join(", "),
            self.real(shape),
            self.real(&one),
            self.real(shape)
        ))
    }

    /// A `stablehlo.while` carrying `init` (value, type) pairs. Iterates `for i in 0..len` with the
    /// counter as the first carried value; `body` gets the counter and the carried values and returns
    /// the next carried values. Returns the final carried values (without the counter).
    fn for_loop(&mut self, len: usize, init: &[(String, String)], body: impl FnOnce(&mut Self, &str, &[String]) -> Vec<String>) -> Vec<String> {
        let zero = self.index(0);
        let counter = self.fresh();
        let args: Vec<String> = init.iter().map(|_| self.fresh()).collect();
        let types: Vec<&str> = std::iter::once(INDEX).chain(init.iter().map(|(_, t)| t.as_str())).collect();
        let types = types.join(", ");
        let result = self.fresh();
        let bindings: Vec<String> = std::iter::once(format!("{counter} = {zero}"))
            .chain(args.iter().zip(init).map(|(a, (v, _))| format!("{a} = {v}")))
            .collect();
        self.line(&format!("{result}:{} = stablehlo.while({}) : {types}", init.len() + 1, bindings.join(", ")));
        self.line("cond {");
        self.depth += 1;
        let n = self.index(len as i32);
        let more = self.emit(&format!("stablehlo.compare LT, {counter}, {n}, SIGNED : ({INDEX}, {INDEX}) -> tensor<i1>"));
        self.line(&format!("stablehlo.return {more} : tensor<i1>"));
        self.depth -= 1;
        self.line("} do {");
        self.depth += 1;
        let next = body(self, &counter, &args);
        assert_eq!(next.len(), init.len());
        let one = self.index(1);
        let step = self.emit(&format!("stablehlo.add {counter}, {one} : {INDEX}"));
        let values: Vec<&str> = std::iter::once(step.as_str()).chain(next.iter().map(String::as_str)).collect();
        self.line(&format!("stablehlo.return {} : {types}", values.join(", ")));
        self.depth -= 1;
        self.line("}");
        (1..=init.len()).map(|k| format!("{result}#{k}")).collect()
    }

    /// Wraps the body into `func.func @main`.
    fn function(self, params: &[(String, Vec<usize>)], results: &[(String, Vec<usize>)]) -> Program {
        let mut text = String::new();
        let args: Vec<String> = params.iter().map(|(n, s)| format!("{n}: {}", self.real(s))).collect();
        let types: Vec<String> = results.iter().map(|(_, s)| self.real(s)).collect();
        let names: Vec<&str> = results.iter().map(|(n, _)| n.as_str()).collect();
        writeln!(text, "func.func @main({}) -> ({}) {{", args.join(", "), types.join(", ")).unwrap();
        text.push_str(&self.out);
        writeln!(text, "  return {} : {}", names.join(", "), types.join(", ")).unwrap();
        text.push_str("}\n");
        Program { text, inputs: params.iter().map(|(_, s)| s.clone()).collect(), outputs: results.iter().map(|(_, s)| s.clone()).collect(), dtype: self.dtype }
    }
}

impl Graph {
    /// This graph as an f32 program: `@main(inputs...) -> (outputs...)`.
    pub fn program(&self) -> Program {
        self.program_as::<f32>()
    }

    /// This graph as a program computing in `T` (`f32` or `f64`).
    pub fn program_as<T: FluxFloat>(&self) -> Program {
        assert!(self.outputs.iter().all(|&o| !self.nodes[o as usize].mask), "Graph::program: outputs must be real, not masks");
        let mut w = Writer::new(T::DTYPE);
        let args: Vec<String> = self.inputs.iter().map(|_| w.fresh()).collect();
        let outs = w.graph(self, &args);
        let params: Vec<(String, Vec<usize>)> = args.into_iter().zip(self.inputs.iter().cloned()).collect();
        let results: Vec<(String, Vec<usize>)> = outs.into_iter().zip(self.output_shapes()).collect();
        w.function(&params, &results)
    }
}

fn check_len(len: usize) {
    assert!(len > 0 && len <= i32::MAX as usize, "Scan: the signal length must be in 1..=i32::MAX");
}

/// `[len, shape...]`.
fn stacked(len: usize, shape: &[usize]) -> Vec<usize> {
    [&[len], shape].concat()
}

impl Scan {
    /// The scan over `len` steps: `@main(params..., xs: [len, sample...], s0...) -> (ys: [len,
    /// output...], final state...)`.
    pub fn forward_program(&self, len: usize) -> Program {
        self.forward_program_as::<f32>(len)
    }

    /// [`forward_program`](Self::forward_program) computing in `T` (`f32` or `f64`).
    pub fn forward_program_as<T: FluxFloat>(&self, len: usize) -> Program {
        check_len(len);
        let (p, s) = (self.params.len(), self.states.len());
        let (xs_shape, ys_shape) = (stacked(len, &self.sample), stacked(len, &self.output));
        let mut w = Writer::new(T::DTYPE);
        let params: Vec<String> = (0..p).map(|_| w.fresh()).collect();
        let xs = w.fresh();
        let s0: Vec<String> = (0..s).map(|_| w.fresh()).collect();
        let ys = w.splat(0.0, &ys_shape);

        let mut init: Vec<(String, String)> = params.iter().zip(&self.params).map(|(v, sh)| (v.clone(), w.real(sh))).collect();
        init.push((xs.clone(), w.real(&xs_shape)));
        init.push((ys, w.real(&ys_shape)));
        init.extend(s0.iter().zip(&self.states).map(|(v, sh)| (v.clone(), w.real(sh))));
        let out = w.for_loop(len, &init, |w, i, c| {
            let (params, xs, ys, state) = (&c[..p], &c[p], &c[p + 1], &c[p + 2..]);
            let x = w.step_of(xs, i, &xs_shape);
            let args: Vec<String> = params.iter().chain(state).cloned().chain([x]).collect();
            let next = w.graph(&self.step, &args);
            let ys = w.store(ys, &next[s], i, &ys_shape);
            params.iter().cloned().chain([xs.clone(), ys]).chain(next[..s].iter().cloned()).collect()
        });

        let mut signature: Vec<(String, Vec<usize>)> = params.into_iter().zip(self.params.iter().cloned()).collect();
        signature.push((xs, xs_shape));
        signature.extend(s0.into_iter().zip(self.states.iter().cloned()));
        let mut results = vec![(out[p + 1].clone(), ys_shape)];
        results.extend(out[p + 2..].iter().cloned().zip(self.states.iter().cloned()));
        w.function(&signature, &results)
    }

    /// `loss(outputs, aux...)` and its gradient over `len` steps:
    /// `@main(params..., xs, aux..., s0...) -> (loss, d params..., d s0..., d xs)`.
    ///
    /// Two loops around the loss: the forward one stores the outputs and the state each step starts
    /// from; the loss and its vector-Jacobian product run on the whole output; the backward loop
    /// walks the steps in reverse from the loss's cotangent, applying the step's vector-Jacobian
    /// product.
    pub fn grad_program(&self, len: usize, loss: &Loss) -> Program {
        self.grad_program_as::<f32>(len, loss)
    }

    /// [`grad_program`](Self::grad_program) computing in `T` (`f32` or `f64`).
    pub fn grad_program_as<T: FluxFloat>(&self, len: usize, loss: &Loss) -> Program {
        check_len(len);
        let (p, s) = (self.params.len(), self.states.len());
        let (xs_shape, ys_shape) = (stacked(len, &self.sample), stacked(len, &self.output));
        assert_eq!(loss.output_shape(), ys_shape.as_slice(), "Scan::grad_program: the loss scores outputs of another shape");
        let saved_shapes: Vec<Vec<usize>> = self.states.iter().map(|sh| stacked(len, sh)).collect();
        let mut w = Writer::new(T::DTYPE);
        let params: Vec<String> = (0..p).map(|_| w.fresh()).collect();
        let xs = w.fresh();
        let aux: Vec<String> = loss.aux_shapes().iter().map(|_| w.fresh()).collect();
        let s0: Vec<String> = (0..s).map(|_| w.fresh()).collect();

        // forward: carry params, xs, ys, state, saved states
        let mut init: Vec<(String, String)> = params.iter().zip(&self.params).map(|(v, sh)| (v.clone(), w.real(sh))).collect();
        init.push((xs.clone(), w.real(&xs_shape)));
        let ys = w.splat(0.0, &ys_shape);
        init.push((ys, w.real(&ys_shape)));
        init.extend(s0.iter().zip(&self.states).map(|(v, sh)| (v.clone(), w.real(sh))));
        for sh in &saved_shapes {
            let z = w.splat(0.0, sh);
            init.push((z, w.real(sh)));
        }
        let fwd = w.for_loop(len, &init, |w, i, c| {
            let (params, xs, ys) = (&c[..p], &c[p], &c[p + 1]);
            let (state, saved) = (&c[p + 2..p + 2 + s], &c[p + 2 + s..]);
            let x = w.step_of(xs, i, &xs_shape);
            let saved: Vec<String> = saved.iter().zip(state).zip(&saved_shapes).map(|((v, st), sh)| w.store(v, st, i, sh)).collect();
            let args: Vec<String> = params.iter().chain(state).cloned().chain([x]).collect();
            let out = w.graph(&self.step, &args);
            let ys = w.store(ys, &out[s], i, &ys_shape);
            params.iter().cloned().chain([xs.clone(), ys]).chain(out[..s].iter().cloned()).chain(saved).collect()
        });
        let saved = &fwd[p + 2 + s..];

        // the loss and its cotangent
        let args: Vec<String> = std::iter::once(fwd[p + 1].clone()).chain(aux.iter().cloned()).collect();
        let out = w.graph(&loss.grad, &args);
        let (value, dys) = (out[0].clone(), out[1].clone());

        // backward: carry params, xs, dys, saved states, d state, d params, d xs
        let mut init: Vec<(String, String)> = params.iter().zip(&self.params).map(|(v, sh)| (v.clone(), w.real(sh))).collect();
        init.push((xs.clone(), w.real(&xs_shape)));
        init.push((dys, w.real(&ys_shape)));
        init.extend(saved.iter().zip(&saved_shapes).map(|(v, sh)| (v.clone(), w.real(sh))));
        for sh in self.states.iter().chain(&self.params).chain([&xs_shape]) {
            let z = w.splat(0.0, sh);
            init.push((z, w.real(sh)));
        }
        let bwd = w.for_loop(len, &init, |w, j, c| {
            let (params, xs, dys, saved) = (&c[..p], &c[p], &c[p + 1], &c[p + 2..p + 2 + s]);
            let (d_state, d_params, d_xs) = (&c[p + 2 + s..p + 2 + 2 * s], &c[p + 2 + 2 * s..p + 2 + 2 * s + p], &c[p + 2 + 2 * s + p]);
            let last = w.index(len as i32 - 1);
            let i = w.emit(&format!("stablehlo.subtract {last}, {j} : {INDEX}"));
            let x = w.step_of(xs, &i, &xs_shape);
            let dy = w.step_of(dys, &i, &ys_shape);
            let state: Vec<String> = saved.iter().zip(&saved_shapes).map(|(v, sh)| w.step_of(v, &i, sh)).collect();
            let args: Vec<String> = params.iter().chain(&state).cloned().chain([x]).chain(d_state.iter().cloned()).chain([dy]).collect();
            let out = w.graph(&self.step_vjp, &args);
            let d_params: Vec<String> = d_params
                .iter()
                .zip(&out[..p])
                .zip(&self.params)
                .map(|((acc, g), sh)| w.emit(&format!("stablehlo.add {acc}, {g} : {}", w.real(sh))))
                .collect();
            let d_xs = w.store(d_xs, &out[p + s], &i, &xs_shape);
            params.iter().chain([xs, dys]).chain(saved).cloned().chain(out[p..p + s].iter().cloned()).chain(d_params).chain([d_xs]).collect()
        });

        let mut signature: Vec<(String, Vec<usize>)> = params.into_iter().zip(self.params.iter().cloned()).collect();
        signature.push((xs, xs_shape.clone()));
        signature.extend(aux.into_iter().zip(loss.aux_shapes().iter().cloned()));
        signature.extend(s0.into_iter().zip(self.states.iter().cloned()));
        let mut results = vec![(value, Vec::new())];
        results.extend(bwd[p + 2 + 2 * s..p + 2 + 2 * s + p].iter().cloned().zip(self.params.iter().cloned()));
        results.extend(bwd[p + 2 + s..p + 2 + 2 * s].iter().cloned().zip(self.states.iter().cloned()));
        results.push((bwd[p + 2 + 2 * s + p].clone(), xs_shape));
        w.function(&signature, &results)
    }

    /// The mean squared error and its gradient over `len` steps: [`grad_program`](Self::grad_program)
    /// with [`Loss::mse`], `@main(params..., xs, targets, s0...) -> (loss, d params..., d s0...,
    /// d xs)`.
    pub fn loss_grad_program(&self, len: usize) -> Program {
        self.grad_program(len, &Loss::mse(&stacked(len, &self.output)))
    }
}
