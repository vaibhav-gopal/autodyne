//! Emits traced programs as textual StableHLO (MLIR), the input format of XLA (through PJRT) and
//! IREE. Scalars are `tensor<f32>`; signals are `tensor<Nxf32>`; a scan is a `stablehlo.while`
//! loop that slices one sample per iteration.

use std::fmt::Write;

use super::graph::{Cmp, Graph, Op};
use super::scan::Scan;

const SCALAR: &str = "tensor<f32>";
const INDEX: &str = "tensor<i32>";

/// A function body being written: SSA names are numbered once per function, so nested regions never
/// shadow a name.
struct Writer {
    out: String,
    next: usize,
    depth: usize,
}

impl Writer {
    fn new() -> Self {
        Writer { out: String::new(), next: 0, depth: 1 }
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

    /// An f32 constant (bit-exact, as hex) of type `ty` (a scalar, or a splat for a vector).
    fn real(&mut self, v: f32, ty: &str) -> String {
        self.emit(&format!("stablehlo.constant dense<0x{:08X}> : {ty}", v.to_bits()))
    }

    fn index(&mut self, v: i32) -> String {
        self.emit(&format!("stablehlo.constant dense<{v}> : {INDEX}"))
    }

    /// Writes `g` with `args` as its inputs; returns the names of its outputs.
    fn graph(&mut self, g: &Graph, args: &[String]) -> Vec<String> {
        assert_eq!(args.len(), g.inputs);
        let mut names: Vec<String> = Vec::with_capacity(g.nodes.len());
        for op in &g.nodes {
            let n = |i: u32| names[i as usize].as_str();
            let unary = |name: &str, a: u32| format!("stablehlo.{name} {} : {SCALAR}", n(a));
            let binary = |name: &str, a: u32, b: u32| format!("stablehlo.{name} {}, {} : {SCALAR}", n(a), n(b));
            let rhs = match *op {
                Op::Input(i) => {
                    names.push(args[i as usize].clone());
                    continue;
                }
                Op::Const(c) => {
                    let name = self.real(c as f32, SCALAR);
                    names.push(name);
                    continue;
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
                Op::Compare(c, a, b) => {
                    let dir = match c {
                        Cmp::Lt => "LT",
                        Cmp::Gt => "GT",
                    };
                    format!("stablehlo.compare {dir}, {}, {}, FLOAT : ({SCALAR}, {SCALAR}) -> tensor<i1>", n(a), n(b))
                }
                Op::Select(c, a, b) => {
                    format!("stablehlo.select {}, {}, {} : (tensor<i1>, {SCALAR}, {SCALAR}) -> {SCALAR}", n(c), n(a), n(b))
                }
            };
            let name = self.emit(&rhs);
            names.push(name);
        }
        g.outputs.iter().map(|&o| names[o as usize].clone()).collect()
    }

    /// Sample `i` of the vector `v` (of type `ty`) as a scalar.
    fn sample(&mut self, v: &str, i: &str, ty: &str) -> String {
        let one = self.emit(&format!("stablehlo.dynamic_slice {v}, {i}, sizes = [1] : ({ty}, {INDEX}) -> tensor<1xf32>"));
        self.emit(&format!("stablehlo.reshape {one} : (tensor<1xf32>) -> {SCALAR}"))
    }

    /// `v` with sample `i` replaced by the scalar `x`.
    fn store(&mut self, v: &str, x: &str, i: &str, ty: &str) -> String {
        let one = self.emit(&format!("stablehlo.reshape {x} : ({SCALAR}) -> tensor<1xf32>"));
        self.emit(&format!("stablehlo.dynamic_update_slice {v}, {one}, {i} : ({ty}, tensor<1xf32>, {INDEX}) -> {ty}"))
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
    fn function(self, params: &[(String, String)], results: &[(String, String)]) -> String {
        let mut text = String::new();
        let args: Vec<String> = params.iter().map(|(n, t)| format!("{n}: {t}")).collect();
        let types: Vec<&str> = results.iter().map(|(_, t)| t.as_str()).collect();
        let names: Vec<&str> = results.iter().map(|(n, _)| n.as_str()).collect();
        writeln!(text, "func.func @main({}) -> ({}) {{", args.join(", "), types.join(", ")).unwrap();
        text.push_str(&self.out);
        writeln!(text, "  return {} : {}", names.join(", "), types.join(", ")).unwrap();
        text.push_str("}\n");
        text
    }
}

fn signal(len: usize) -> String {
    format!("tensor<{len}xf32>")
}

fn check_len(len: usize) {
    assert!(len > 0 && len <= i32::MAX as usize, "Scan: the signal length must be in 1..=i32::MAX");
}

impl Scan {
    /// The scan over `len` samples as StableHLO:
    /// `@main(params..., xs: tensor<len x f32>, s0...) -> (ys: tensor<len x f32>, final state...)`,
    /// every parameter and state value a `tensor<f32>`.
    pub fn forward_hlo(&self, len: usize) -> String {
        check_len(len);
        let (p, s, sig) = (self.params, self.states, signal(len));
        let mut w = Writer::new();
        let params: Vec<String> = (0..p).map(|_| w.fresh()).collect();
        let xs = w.fresh();
        let s0: Vec<String> = (0..s).map(|_| w.fresh()).collect();
        let ys = w.real(0.0, &sig);

        let mut init: Vec<(String, String)> = params.iter().map(|v| (v.clone(), SCALAR.into())).collect();
        init.push((xs.clone(), sig.clone()));
        init.push((ys, sig.clone()));
        init.extend(s0.iter().map(|v| (v.clone(), SCALAR.into())));
        let out = w.for_loop(len, &init, |w, i, c| {
            let (params, xs, ys, state) = (&c[..p], &c[p], &c[p + 1], &c[p + 2..]);
            let x = w.sample(xs, i, &sig);
            let args: Vec<String> = params.iter().chain(state).cloned().chain([x]).collect();
            let next = w.graph(&self.step, &args);
            let ys = w.store(ys, &next[s], i, &sig);
            params.iter().cloned().chain([xs.clone(), ys]).chain(next[..s].iter().cloned()).collect()
        });

        let mut signature: Vec<(String, String)> = params.into_iter().map(|v| (v, SCALAR.into())).collect();
        signature.push((xs, sig.clone()));
        signature.extend(s0.into_iter().map(|v| (v, SCALAR.into())));
        let mut results = vec![(out[p + 1].clone(), sig)];
        results.extend(out[p + 2..].iter().map(|v| (v.clone(), SCALAR.into())));
        w.function(&signature, &results)
    }

    /// The mean squared error and its gradient over `len` samples as StableHLO:
    /// `@main(params..., xs, targets, s0...) -> (loss, d params..., d s0...)`.
    ///
    /// Two loops: the forward one accumulates the loss and saves the state each step starts from;
    /// the backward one walks the samples in reverse, applying the step's vector-Jacobian product.
    pub fn loss_grad_hlo(&self, len: usize) -> String {
        check_len(len);
        let (p, s, sig) = (self.params, self.states, signal(len));
        let mut w = Writer::new();
        let params: Vec<String> = (0..p).map(|_| w.fresh()).collect();
        let xs = w.fresh();
        let ts = w.fresh();
        let s0: Vec<String> = (0..s).map(|_| w.fresh()).collect();

        // forward: carry params, xs, ts, state, saved states, loss
        let zero_sig = w.real(0.0, &sig);
        let zero = w.real(0.0, SCALAR);
        let mut init: Vec<(String, String)> = params.iter().map(|v| (v.clone(), SCALAR.into())).collect();
        init.push((xs.clone(), sig.clone()));
        init.push((ts.clone(), sig.clone()));
        init.extend(s0.iter().map(|v| (v.clone(), SCALAR.into())));
        init.extend((0..s).map(|_| (zero_sig.clone(), sig.clone())));
        init.push((zero.clone(), SCALAR.into()));
        let fwd = w.for_loop(len, &init, |w, i, c| {
            let (params, xs, ts) = (&c[..p], &c[p], &c[p + 1]);
            let (state, saved, loss) = (&c[p + 2..p + 2 + s], &c[p + 2 + s..p + 2 + 2 * s], &c[p + 2 + 2 * s]);
            let x = w.sample(xs, i, &sig);
            let t = w.sample(ts, i, &sig);
            let saved: Vec<String> = saved.iter().zip(state).map(|(v, st)| w.store(v, st, i, &sig)).collect();
            let args: Vec<String> = params.iter().chain(state).cloned().chain([x, t]).collect();
            let out = w.graph(&self.step_loss, &args);
            let loss = w.emit(&format!("stablehlo.add {loss}, {} : {SCALAR}", out[s + 1]));
            params.iter().cloned().chain([xs.clone(), ts.clone()]).chain(out[..s].iter().cloned()).chain(saved).chain([loss]).collect()
        });
        let saved = &fwd[p + 2 + s..p + 2 + 2 * s];
        let scale = w.real(1.0 / len as f32, SCALAR);
        let loss = w.emit(&format!("stablehlo.multiply {}, {scale} : {SCALAR}", fwd[p + 2 + 2 * s]));

        // backward: carry params, xs, ts, saved states, d state, d params
        let mut init: Vec<(String, String)> = params.iter().map(|v| (v.clone(), SCALAR.into())).collect();
        init.push((xs.clone(), sig.clone()));
        init.push((ts.clone(), sig.clone()));
        init.extend(saved.iter().map(|v| (v.clone(), sig.clone())));
        init.extend((0..s + p).map(|_| (zero.clone(), SCALAR.into())));
        let bwd = w.for_loop(len, &init, |w, j, c| {
            let (params, xs, ts, saved) = (&c[..p], &c[p], &c[p + 1], &c[p + 2..p + 2 + s]);
            let (d_state, d_params) = (&c[p + 2 + s..p + 2 + 2 * s], &c[p + 2 + 2 * s..]);
            let last = w.index(len as i32 - 1);
            let i = w.emit(&format!("stablehlo.subtract {last}, {j} : {INDEX}"));
            let x = w.sample(xs, &i, &sig);
            let t = w.sample(ts, &i, &sig);
            let state: Vec<String> = saved.iter().map(|v| w.sample(v, &i, &sig)).collect();
            // the cotangent of each squared error (the loss is their mean); made in the region
            // rather than captured from the function body
            let dl = w.real(1.0 / len as f32, SCALAR);
            let args: Vec<String> = params.iter().chain(&state).cloned().chain([x, t]).chain(d_state.iter().cloned()).chain([dl]).collect();
            let out = w.graph(&self.step_vjp, &args);
            let d_params: Vec<String> = d_params.iter().zip(&out[..p]).map(|(acc, g)| w.emit(&format!("stablehlo.add {acc}, {g} : {SCALAR}"))).collect();
            params.iter().chain([xs, ts]).chain(saved).cloned().chain(out[p..].iter().cloned()).chain(d_params).collect()
        });

        let mut signature: Vec<(String, String)> = params.into_iter().map(|v| (v, SCALAR.into())).collect();
        signature.push((xs, sig.clone()));
        signature.push((ts, sig));
        signature.extend(s0.into_iter().map(|v| (v, SCALAR.into())));
        let mut results = vec![(loss, SCALAR.to_string())];
        results.extend(bwd[p + 2 + 2 * s..].iter().map(|v| (v.clone(), SCALAR.into())));
        results.extend(bwd[p + 2 + s..p + 2 + 2 * s].iter().map(|v| (v.clone(), SCALAR.into())));
        w.function(&signature, &results)
    }
}
