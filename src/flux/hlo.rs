//! Emits traced programs as textual StableHLO (MLIR), the input format of XLA (through PJRT) and
//! IREE. Values are `tensor<...xf32>` or `tensor<...xf64>` (masks `xi1`), the precision chosen at
//! emission; a scan is a `stablehlo.while` loop that slices one step per iteration out of the
//! signal's first axis.

use std::fmt::Write;

use super::graph::{Cmp, FluxFloat, Graph, Kind, Op, Reduction};
use super::loss::Loss;
use super::scan::Scan;
use crate::units::DType;

/// A StableHLO module with one function, `@main`, the shapes of its inputs and outputs, and their
/// element type (`DType::F32` or `DType::F64`).
#[derive(Clone, Debug, PartialEq)]
pub struct Program {
    /// The module text (MLIR, StableHLO dialect).
    pub text: String,
    /// The shape of each input.
    pub inputs: Vec<Vec<usize>>,
    /// The shape of each output.
    pub outputs: Vec<Vec<usize>>,
    /// The element type it computes in.
    pub dtype: DType,
}

/// How a program is written: its precision, and limits a backend needs.
///
/// ```
/// use autodyne::flux::{trace, Emit};
/// use autodyne::signal::RealArrayMath;
///
/// let g = trace(&[&[4, 2048]], |v| vec![v[0].rfft().0]);
/// // FFTs of at most 64 points, as IREE's Vulkan backend needs: 2048 = 64 x 32
/// let program = g.program_with(&Emit::f32().max_fft(64));
/// assert!(program.text.contains("length = [64]") && !program.text.contains("length = [2048]"));
/// ```
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Emit {
    /// `DType::F32` or `DType::F64`.
    pub dtype: DType,
    /// FFTs longer than this are built from shorter ones (a four-step decomposition), for backends
    /// that cannot compile long FFTs; `None` emits every FFT as one operation.
    pub max_fft: Option<usize>,
}

impl Default for Emit {
    fn default() -> Self {
        Emit::f32()
    }
}

impl Emit {
    /// Single precision, FFTs emitted whole.
    pub fn f32() -> Self {
        Emit { dtype: DType::F32, max_fft: None }
    }
    /// Double precision, FFTs emitted whole.
    pub fn f64() -> Self {
        Emit { dtype: DType::F64, max_fft: None }
    }
    /// Computing in `T`.
    pub fn of<T: FluxFloat>() -> Self {
        Emit { dtype: T::DTYPE, max_fft: None }
    }
    /// FFTs of at most `n` points (at least 2).
    pub fn max_fft(self, n: usize) -> Self {
        assert!(n >= 2, "Emit::max_fft: at least 2 points");
        Emit { max_fft: Some(n), ..self }
    }
    /// With the limits `backend` needs (see [`Backend::max_fft`](super::Backend::max_fft)).
    pub fn for_backend(self, backend: &dyn super::Backend) -> Self {
        match backend.max_fft() {
            Some(n) => self.max_fft(self.max_fft.map_or(n, |m| m.min(n))),
            None => self,
        }
    }
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


fn gcd(a: usize, b: usize) -> usize {
    if b == 0 { a } else { gcd(b, a % b) }
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
    max_fft: Option<usize>,
}

impl Writer {
    fn new(emit: &Emit) -> Self {
        let dtype = emit.dtype;
        assert!(matches!(dtype, DType::F32 | DType::F64), "flux: programs are f32 or f64, not {dtype:?}");
        Writer { out: String::new(), next: 0, depth: 1, dtype, max_fft: emit.max_fft }
    }

    /// A real constant array (bit-exact hex).
    fn literal(&mut self, data: &[f64], shape: &[usize]) -> String {
        let mut hex = String::with_capacity(2 + 16 * data.len());
        hex.push_str("0x");
        for &v in data {
            let bytes = if self.dtype == DType::F64 { v.to_le_bytes().to_vec() } else { (v as f32).to_le_bytes().to_vec() };
            for b in bytes {
                write!(hex, "{b:02X}").unwrap();
            }
        }
        let t = self.real(shape);
        self.emit(&format!("stablehlo.constant dense<\"{hex}\"> : {t}"))
    }

    /// `v` with its last two axes swapped.
    fn swap_last(&mut self, v: &str, shape: &[usize], complex: bool) -> String {
        let r = shape.len();
        let mut perm: Vec<usize> = (0..r).collect();
        perm.swap(r - 2, r - 1);
        let mut to = shape.to_vec();
        to.swap(r - 2, r - 1);
        let (a, b) = if complex { (self.complex(shape), self.complex(&to)) } else { (self.real(shape), self.real(&to)) };
        self.emit(&format!("stablehlo.transpose {v}, dims = [{}] : ({a}) -> {b}", list(&perm)))
    }

    /// The split of an `n`-point FFT for the length limit: `(n1, n2)`, `n1` the largest factor
    /// within the limit; `None` when `n` fits (or has no such factor, and is emitted whole).
    fn fft_split(&self, n: usize) -> Option<(usize, usize)> {
        let max = self.max_fft.filter(|&m| n > m)?;
        (2..=max.min(n - 1)).rev().find(|d| n.is_multiple_of(*d)).map(|d| (d, n / d))
    }

    /// The complex FFT (or its inverse, scaled by 1/n) of `z` along the last axis of `shape`. When
    /// it is too long: `n = n1 n2` points as n2 transforms of n1 points, twiddles, then n1 of n2.
    fn fft(&mut self, z: &str, shape: &[usize], inverse: bool) -> String {
        let n = *shape.last().expect("an axis");
        let ct = self.complex(shape);
        let Some((n1, n2)) = self.fft_split(n) else {
            let kind = if inverse { "IFFT" } else { "FFT" };
            return self.emit(&format!("stablehlo.fft {z}, type = {kind}, length = [{n}] : ({ct}) -> {ct}"));
        };
        let lead = &shape[..shape.len() - 1];
        let r = lead.len();
        // x[t1 n2 + t2] as [.., t1, t2], then the n1-point transforms along t1
        let grid = [lead, &[n1, n2]].concat();
        let gt = self.complex(&grid);
        let a = self.emit(&format!("stablehlo.reshape {z} : ({ct}) -> {gt}"));
        let a = self.swap_last(&a, &grid, true);
        let rows = [lead, &[n2, n1]].concat();
        let a = self.fft(&a, &rows, inverse);
        // twiddles exp(∓2πi t2 k1 / n), from real constants (some backends lack complex ones)
        let sign = if inverse { 1.0 } else { -1.0 };
        let angles: Vec<f64> = (0..n2).flat_map(|t2| (0..n1).map(move |k1| sign * std::f64::consts::TAU * (t2 * k1) as f64 / n as f64)).collect();
        let cos = self.literal(&angles.iter().map(|a| a.cos()).collect::<Vec<_>>(), &[n2, n1]);
        let sin = self.literal(&angles.iter().map(|a| a.sin()).collect::<Vec<_>>(), &[n2, n1]);
        let tt = self.complex(&[n2, n1]);
        let twiddle = self.emit(&format!("stablehlo.complex {cos}, {sin} : {tt}"));
        let rt = self.complex(&rows);
        let twiddle = if r == 0 { twiddle } else { self.emit(&format!("stablehlo.broadcast_in_dim {twiddle}, dims = [{}, {}] : ({tt}) -> {rt}", r, r + 1)) };
        let a = self.emit(&format!("stablehlo.multiply {a}, {twiddle} : {rt}"));
        // the n2-point transforms along t2, then X[k1 + n1 k2] in order
        let a = self.swap_last(&a, &rows, true);
        let a = self.fft(&a, &grid, inverse);
        let a = self.swap_last(&a, &grid, true);
        self.emit(&format!("stablehlo.reshape {a} : ({rt}) -> {ct}"))
    }

    /// The real FFT of `x` along the last axis of `shape`: `n / 2 + 1` complex bins.
    fn rfft(&mut self, x: &str, shape: &[usize]) -> String {
        let n = *shape.last().expect("an axis");
        let bins = [&shape[..shape.len() - 1], &[n / 2 + 1]].concat();
        let (rt, bt) = (self.real(shape), self.complex(&bins));
        if self.fft_split(n).is_none() {
            return self.emit(&format!("stablehlo.fft {x}, type = RFFT, length = [{n}] : ({rt}) -> {bt}"));
        }
        let zeros = self.splat(0.0, shape);
        let ct = self.complex(shape);
        let z = self.emit(&format!("stablehlo.complex {x}, {zeros} : {ct}"));
        let spectrum = self.fft(&z, shape, false);
        let limit: Vec<usize> = bins.clone();
        let line = format!("\"stablehlo.slice\"({spectrum}) {{start_indices = {}, limit_indices = {}, strides = {}}} : ({ct}) -> {bt}", i64s(&vec![0; shape.len()]), i64s(&limit), i64s(&vec![1; shape.len()]));
        self.emit(&line)
    }

    /// The inverse real FFT of `z` (`n / 2 + 1` bins along the last axis of `bins`) to `n` samples.
    fn irfft(&mut self, z: &str, bins: &[usize], n: usize) -> String {
        let r = bins.len();
        let m = n / 2 + 1;
        let out_shape = [&bins[..r - 1], &[n]].concat();
        let (bt, ot) = (self.complex(bins), self.real(&out_shape));
        if self.fft_split(n).is_none() {
            return self.emit(&format!("stablehlo.fft {z}, type = IRFFT, length = [{n}] : ({bt}) -> {ot}"));
        }
        // the whole Hermitian spectrum: bins 0..m (the imaginary parts of 0 and n/2 ignored), then
        // conj of bins n - m .. 1
        let rt = self.real(bins);
        let re = self.emit(&format!("stablehlo.real {z} : ({bt}) -> {rt}"));
        let im = self.emit(&format!("stablehlo.imag {z} : ({bt}) -> {rt}"));
        let keep: Vec<f64> = (0..m).map(|k| if k == 0 || (n.is_multiple_of(2) && k == n / 2) { 0.0 } else { 1.0 }).collect();
        let keep = self.literal(&keep, &[m]);
        let keep = if r == 1 { keep } else { self.emit(&format!("stablehlo.broadcast_in_dim {keep}, dims = [{}] : ({}) -> {rt}", r - 1, self.real(&[m]))) };
        let im = self.emit(&format!("stablehlo.multiply {im}, {keep} : {rt}"));
        let neg = self.emit(&format!("stablehlo.negate {im} : {rt}"));
        let head = self.emit(&format!("stablehlo.complex {re}, {im} : {bt}"));
        let conj = self.emit(&format!("stablehlo.complex {re}, {neg} : {bt}"));
        let tail_shape = [&bins[..r - 1], &[n - m]].concat();
        let tt = self.complex(&tail_shape);
        let (mut start, mut limit) = (vec![0; r], bins.to_vec());
        (start[r - 1], limit[r - 1]) = (1, n - m + 1);
        let tail = self.emit(&format!("\"stablehlo.slice\"({conj}) {{start_indices = {}, limit_indices = {}, strides = {}}} : ({bt}) -> {tt}", i64s(&start), i64s(&limit), i64s(&vec![1; r])));
        let tail = self.emit(&format!("stablehlo.reverse {tail}, dims = [{}] : {tt}", r - 1));
        let ct = self.complex(&out_shape);
        let full = self.emit(&format!("\"stablehlo.concatenate\"({head}, {tail}) {{dimension = {} : i64}} : ({bt}, {tt}) -> {ct}", r - 1));
        let samples = self.fft(&full, &out_shape, true);
        self.emit(&format!("stablehlo.real {samples} : ({ct}) -> {ot}"))
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

    /// A scalar zero of a real or complex kind.
    fn zero(&mut self, kind: Kind) -> String {
        match kind {
            Kind::Complex => {
                let t = self.complex(&[]);
                self.emit(&format!("stablehlo.constant dense<(0.000000e+00,0.000000e+00)> : {t}"))
            }
            _ => self.splat(0.0, &[]),
        }
    }

    fn index(&mut self, v: i32) -> String {
        self.emit(&format!("stablehlo.constant dense<{v}> : {INDEX}"))
    }

    /// Writes `g` with `args` as its inputs; returns the names of its outputs.
    fn graph(&mut self, g: &Graph, args: &[String]) -> Vec<String> {
        assert_eq!(args.len(), g.inputs.len());
        let mut names: Vec<String> = Vec::with_capacity(g.nodes.len());
        let elem = self.elem();
        let complex = format!("complex<{elem}>");
        let of_kind = |kind: Kind| match kind {
            Kind::Real => elem.to_string(),
            Kind::Complex => complex.clone(),
            Kind::Mask => "i1".to_string(),
        };
        for node in &g.nodes {
            let n = |i: u32| names[i as usize].clone();
            let t = |i: u32| {
                let node = &g.nodes[i as usize];
                ty(&node.shape, &of_kind(node.kind))
            };
            let out = ty(&node.shape, &of_kind(node.kind));
            // the element type as a scalar (reduction and padding values)
            let scalar = ty(&[], &of_kind(node.kind));
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
                Op::Abs(a) => format!("\"stablehlo.abs\"({}) : ({}) -> {out}", n(a), t(a)),
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
                    let zero = self.zero(node.kind);
                    format!("stablehlo.reduce({} init: {zero}) applies stablehlo.add across dimensions = [{}] : ({}, {scalar}) -> {out}", n(a), list(axes), t(a))
                }
                Op::Dot { a, b, ref ca, ref cb } => {
                    // full precision: GPUs would otherwise multiply f32 in TF32 (10-bit mantissas)
                    format!("stablehlo.dot_general {}, {}, contracting_dims = [{}] x [{}], precision = [HIGHEST, HIGHEST] : ({}, {}) -> {out}", n(a), n(b), list(ca), list(cb), t(a), t(b))
                }
                Op::Rfft(a) => {
                    let name = self.rfft(&n(a), &g.nodes[a as usize].shape);
                    names.push(name);
                    continue;
                }
                Op::Irfft(a, len) => {
                    let name = self.irfft(&n(a), &g.nodes[a as usize].shape, len);
                    names.push(name);
                    continue;
                }
                Op::Fft(a, inverse) => {
                    let name = self.fft(&n(a), &node.shape, inverse);
                    names.push(name);
                    continue;
                }
                Op::Complex(a, b) => format!("stablehlo.complex {}, {} : {out}", n(a), n(b)),
                Op::Re(a) => format!("stablehlo.real {} : ({}) -> {out}", n(a), t(a)),
                Op::Im(a) => format!("stablehlo.imag {} : ({}) -> {out}", n(a), t(a)),
                // `complex(x, 0)` rather than a conversion: IREE mishandles converting a constant inside
                // a loop, and its Vulkan backend has no complex constants
                Op::ToComplex(a) => {
                    let zeros = self.splat(0.0, &node.shape);
                    format!("stablehlo.complex {}, {zeros} : {out}", n(a))
                }
                Op::Conj(a) => {
                    let parts = ty(&node.shape, elem);
                    let re = self.emit(&format!("stablehlo.real {} : ({out}) -> {parts}", n(a)));
                    let im = self.emit(&format!("stablehlo.imag {} : ({out}) -> {parts}", n(a)));
                    let negated = self.emit(&format!("stablehlo.negate {im} : {parts}"));
                    format!("stablehlo.complex {re}, {negated} : {out}")
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
                    let zero = self.zero(node.kind);
                    format!(
                        "\"stablehlo.pad\"({}, {zero}) {{edge_padding_low = {}, edge_padding_high = {}, interior_padding = {}}} : ({}, {scalar}) -> {out}",
                        n(a),
                        i64s(low),
                        i64s(high),
                        i64s(interior),
                        t(a)
                    )
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
                Op::Frames { a, length, hop } => {
                    // as `frames_by_slices`: blocks of gcd(length, hop) samples, one strided slice of
                    // them per position in the frame, concatenated
                    let (shape, el) = (&g.nodes[a as usize].shape, of_kind(g.nodes[a as usize].kind));
                    let k = shape.len() - 1;
                    let lead = &shape[..k];
                    let count = 1 + (shape[k] - length) / hop;
                    let used = (count - 1) * hop + length;
                    let gg = gcd(length, hop);
                    let (per_frame, per_hop) = (length / gg, hop / gg);
                    let mut cur = n(a);
                    if used != shape[k] {
                        let mut limit = shape.clone();
                        limit[k] = used;
                        cur = self.emit(&format!(
                            "\"stablehlo.slice\"({cur}) {{start_indices = {}, limit_indices = {}, strides = {}}} : ({}) -> {}",
                            i64s(&vec![0; k + 1]),
                            i64s(&limit),
                            i64s(&vec![1; k + 1]),
                            t(a),
                            ty(&limit, &el)
                        ));
                    }
                    let blocks = [lead, &[used / gg, gg]].concat();
                    let b = self.emit(&format!("stablehlo.reshape {cur} : ({}) -> {}", ty(&[lead, &[used]].concat(), &el), ty(&blocks, &el)));
                    let part = [lead, &[count, 1, gg]].concat();
                    let parts: Vec<String> = (0..per_frame)
                        .map(|r| {
                            let (mut start, mut limit, mut stride) = (vec![0; k + 2], blocks.clone(), vec![1; k + 2]);
                            (start[k], limit[k], stride[k]) = (r, r + (count - 1) * per_hop + 1, per_hop);
                            let s = self.emit(&format!(
                                "\"stablehlo.slice\"({b}) {{start_indices = {}, limit_indices = {}, strides = {}}} : ({}) -> {}",
                                i64s(&start),
                                i64s(&limit),
                                i64s(&stride),
                                ty(&blocks, &el),
                                ty(&[lead, &[count, gg]].concat(), &el)
                            ));
                            self.emit(&format!("stablehlo.reshape {s} : ({}) -> {}", ty(&[lead, &[count, gg]].concat(), &el), ty(&part, &el)))
                        })
                        .collect();
                    let joined = [lead, &[count, per_frame, gg]].concat();
                    let cat = if per_frame == 1 {
                        parts[0].clone()
                    } else {
                        self.emit(&format!(
                            "\"stablehlo.concatenate\"({}) {{dimension = {} : i64}} : ({}) -> {}",
                            parts.join(", "),
                            k + 1,
                            vec![ty(&part, &el); per_frame].join(", "),
                            ty(&joined, &el)
                        ))
                    };
                    format!("stablehlo.reshape {cat} : ({}) -> {out}", ty(&joined, &el))
                }
                Op::OverlapAdd { a, n: total, hop } => {
                    // the transpose: each position's blocks padded into place, summed, padded to n
                    let (shape, el) = (&g.nodes[a as usize].shape, of_kind(g.nodes[a as usize].kind));
                    let k = shape.len() - 2;
                    let lead = &shape[..k];
                    let (count, length) = (shape[k], shape[k + 1]);
                    let used = (count - 1) * hop + length;
                    let gg = gcd(length, hop);
                    let (per_frame, per_hop) = (length / gg, hop / gg);
                    let split = [lead, &[count, per_frame, gg]].concat();
                    let b = self.emit(&format!("stablehlo.reshape {} : ({}) -> {}", n(a), t(a), ty(&split, &el)));
                    let zero = self.zero(g.nodes[a as usize].kind);
                    let placed_shape = [lead, &[used / gg, gg]].concat();
                    let mut sum: Option<String> = None;
                    for r in 0..per_frame {
                        let (mut start, mut limit) = (vec![0; k + 3], split.clone());
                        (start[k + 1], limit[k + 1]) = (r, r + 1);
                        let s = self.emit(&format!(
                            "\"stablehlo.slice\"({b}) {{start_indices = {}, limit_indices = {}, strides = {}}} : ({}) -> {}",
                            i64s(&start),
                            i64s(&limit),
                            i64s(&vec![1; k + 3]),
                            ty(&split, &el),
                            ty(&[lead, &[count, 1, gg]].concat(), &el)
                        ));
                        let rs = self.emit(&format!("stablehlo.reshape {s} : ({}) -> {}", ty(&[lead, &[count, 1, gg]].concat(), &el), ty(&[lead, &[count, gg]].concat(), &el)));
                        let (mut low, mut high, mut interior) = (vec![0; k + 2], vec![0; k + 2], vec![0; k + 2]);
                        (low[k], interior[k]) = (r, per_hop - 1);
                        high[k] = used / gg - (r + (count - 1) * per_hop + 1);
                        let p = self.emit(&format!(
                            "\"stablehlo.pad\"({rs}, {zero}) {{edge_padding_low = {}, edge_padding_high = {}, interior_padding = {}}} : ({}, {scalar}) -> {}",
                            i64s(&low),
                            i64s(&high),
                            i64s(&interior),
                            ty(&[lead, &[count, gg]].concat(), &el),
                            ty(&placed_shape, &el)
                        ));
                        sum = Some(match sum {
                            None => p,
                            Some(prev) => self.emit(&format!("stablehlo.add {prev}, {p} : {}", ty(&placed_shape, &el))),
                        });
                    }
                    let sum = sum.expect("at least one frame position");
                    let flat_shape = [lead, &[used]].concat();
                    if total == used {
                        format!("stablehlo.reshape {sum} : ({}) -> {out}", ty(&placed_shape, &el))
                    } else {
                        let flat = self.emit(&format!("stablehlo.reshape {sum} : ({}) -> {}", ty(&placed_shape, &el), ty(&flat_shape, &el)));
                        let mut high = vec![0; k + 1];
                        high[k] = total - used;
                        format!(
                            "\"stablehlo.pad\"({flat}, {zero}) {{edge_padding_low = {}, edge_padding_high = {}, interior_padding = {}}} : ({}, {scalar}) -> {out}",
                            i64s(&vec![0; k + 1]),
                            i64s(&high),
                            i64s(&vec![0; k + 1]),
                            ty(&flat_shape, &el)
                        )
                    }
                }
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
        self.program_with(&Emit::of::<T>())
    }

    /// This graph as a program, written as `emit` says.
    pub fn program_with(&self, emit: &Emit) -> Program {
        assert!(self.outputs.iter().all(|&o| self.nodes[o as usize].kind == Kind::Real), "Graph::program: outputs must be real (not masks or complex values)");
        let mut w = Writer::new(emit);
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
        self.forward_program_with(len, &Emit::of::<T>())
    }

    /// [`forward_program`](Self::forward_program) written as `emit` says.
    pub fn forward_program_with(&self, len: usize, emit: &Emit) -> Program {
        check_len(len);
        let (p, s) = (self.params.len(), self.states.len());
        let (xs_shape, ys_shape) = (stacked(len, &self.sample), stacked(len, &self.output));
        let mut w = Writer::new(emit);
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
        self.grad_program_with(len, loss, &Emit::of::<T>())
    }

    /// [`grad_program`](Self::grad_program) written as `emit` says.
    pub fn grad_program_with(&self, len: usize, loss: &Loss, emit: &Emit) -> Program {
        check_len(len);
        let (p, s) = (self.params.len(), self.states.len());
        let (xs_shape, ys_shape) = (stacked(len, &self.sample), stacked(len, &self.output));
        assert_eq!(loss.output_shape(), ys_shape.as_slice(), "Scan::grad_program: the loss scores outputs of another shape");
        // per step: the state it starts from, then the residuals (when not checkpointed)
        let (forward, reverse) = self.passes();
        let saved_shapes: Vec<Vec<usize>> = self.states.iter().chain(self.residual_shapes()).map(|sh| stacked(len, sh)).collect();
        let k = saved_shapes.len();
        let mut w = Writer::new(emit);
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
            let args: Vec<String> = params.iter().chain(state).cloned().chain([x]).collect();
            let out = w.graph(forward, &args);
            let values = state.iter().chain(&out[s + 1..]);
            let saved: Vec<String> = saved.iter().zip(values).zip(&saved_shapes).map(|((v, st), sh)| w.store(v, st, i, sh)).collect();
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
            let (params, xs, dys, saved) = (&c[..p], &c[p], &c[p + 1], &c[p + 2..p + 2 + k]);
            let (d_state, d_params, d_xs) = (&c[p + 2 + k..p + 2 + k + s], &c[p + 2 + k + s..p + 2 + k + s + p], &c[p + 2 + k + s + p]);
            let last = w.index(len as i32 - 1);
            let i = w.emit(&format!("stablehlo.subtract {last}, {j} : {INDEX}"));
            let x = w.step_of(xs, &i, &xs_shape);
            let dy = w.step_of(dys, &i, &ys_shape);
            let at_step: Vec<String> = saved.iter().zip(&saved_shapes).map(|(v, sh)| w.step_of(v, &i, sh)).collect();
            let (state, residuals) = at_step.split_at(s);
            let args: Vec<String> = params.iter().chain(state).cloned().chain([x]).chain(residuals.iter().cloned()).chain(d_state.iter().cloned()).chain([dy]).collect();
            let out = w.graph(reverse, &args);
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
        results.extend(bwd[p + 2 + k + s..p + 2 + k + s + p].iter().cloned().zip(self.params.iter().cloned()));
        results.extend(bwd[p + 2 + k..p + 2 + k + s].iter().cloned().zip(self.states.iter().cloned()));
        results.push((bwd[p + 2 + k + s + p].clone(), xs_shape));
        w.function(&signature, &results)
    }

    /// The mean squared error and its gradient over `len` steps: [`grad_program`](Self::grad_program)
    /// with [`Loss::mse`], `@main(params..., xs, targets, s0...) -> (loss, d params..., d s0...,
    /// d xs)`.
    pub fn loss_grad_program(&self, len: usize) -> Program {
        self.grad_program(len, &Loss::mse(&stacked(len, &self.output)))
    }
}
