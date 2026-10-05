//! Double-precision `exp`, `log`, `sin`, `cos`, `tanh` and `pow` written out of StableHLO
//! arithmetic, for backends that compile f64 programs but not their transcendentals (IREE 3.11 on
//! the CPU, with no libm in its modules, and on Vulkan, whose SPIR-V maths is 32-bit). The same
//! algorithms as the GPU module's: Cody-Waite range reduction, Taylor or atanh series to below an
//! ulp, and powers of two (and exponents) through the IEEE bit layout, with 64-bit integers.
//!
//! tend: flux / emission

// full-precision constants are the point here
#![allow(clippy::excessive_precision, clippy::approx_constant)]

use super::{ty, Writer};

const LN2_HI: f64 = 6.93147180369123816490e-1;
const LN2_LO: f64 = 1.90821492927058770002e-10;
const LOG2E: f64 = 1.4426950408889634;
/// pi / 2 in three parts (Cody-Waite), for sin and cos
const PIO2: [f64; 3] = [1.57079632673412561417e0, 6.07710050633881403649e-11, 2.02226624879595063154e-21];
const TWO_OVER_PI: f64 = 0.6366197723675814;
/// 1/14! .. 1/2!: e^r - 1 = r + r² p(r)
const EXPM1: [f64; 13] = [
    1.1470745597729725e-11, 1.6059043836821613e-10, 2.08767569878681e-9, 2.505210838544172e-8, 2.755731922398589e-7, 2.7557319223985893e-6, 2.48015873015873e-5,
    1.984126984126984e-4, 1.3888888888888889e-3, 8.333333333333333e-3, 4.1666666666666664e-2, 0.16666666666666666, 0.5,
];
/// 1/23 .. 1/3: atanh series for ln
const ATANH: [f64; 11] =
    [0.043478260869565216, 0.047619047619047616, 0.05263157894736842, 0.058823529411764705, 0.06666666666666667, 0.07692307692307693, 0.09090909090909091, 0.1111111111111111, 0.14285714285714285, 0.2, 0.3333333333333333];
/// sin r = r + r z s(z), cos r = 1 + z c(z), z = r², on |r| <= pi/4
const SIN: [f64; 8] = [2.8114572543455206e-15, -7.647163731819816e-13, 1.6059043836821613e-10, -2.505210838544172e-8, 2.7557319223985893e-6, -1.984126984126984e-4, 8.333333333333333e-3, -0.16666666666666666];
const COS: [f64; 8] = [4.779477332387385e-14, -1.1470745597729725e-11, 2.08767569878681e-9, -2.755731922398589e-7, 2.48015873015873e-5, -1.3888888888888889e-3, 4.1666666666666664e-2, -0.5];

impl Writer {
    fn f(&self, shape: &[usize]) -> String {
        self.real(shape)
    }

    fn op2(&mut self, name: &str, a: &str, b: &str, shape: &[usize]) -> String {
        let t = self.f(shape);
        self.emit(&format!("stablehlo.{name} {a}, {b} : {t}"))
    }

    fn op1(&mut self, name: &str, a: &str, shape: &[usize]) -> String {
        let t = self.f(shape);
        self.emit(&format!("stablehlo.{name} {a} : {t}"))
    }

    /// The nearest integer, as `floor(v + 1/2)` (ties up): Vulkan's f64 `round` is not to be
    /// trusted, and the reductions only need some nearby integer.
    fn round(&mut self, v: &str, shape: &[usize]) -> String {
        let h = self.op_c("add", v, 0.5, shape);
        self.op1("floor", &h, shape)
    }

    fn c(&mut self, v: f64, shape: &[usize]) -> String {
        self.splat(v, shape)
    }

    /// `a op v` for a constant `v`.
    fn op_c(&mut self, name: &str, a: &str, v: f64, shape: &[usize]) -> String {
        let k = self.c(v, shape);
        self.op2(name, a, &k, shape)
    }

    /// Horner's rule: `sum coeffs[i] z^(n - 1 - i)`.
    fn horner(&mut self, z: &str, coeffs: &[f64], shape: &[usize]) -> String {
        let mut p = self.c(coeffs[0], shape);
        for &k in &coeffs[1..] {
            let pz = self.op2("multiply", &p, z, shape);
            p = self.op_c("add", &pz, k, shape);
        }
        p
    }

    fn cmp(&mut self, dir: &str, a: &str, b: &str, shape: &[usize]) -> String {
        let (ft, bt) = (self.f(shape), ty(shape, "i1"));
        self.emit(&format!("stablehlo.compare {dir}, {a}, {b}, FLOAT : ({ft}, {ft}) -> {bt}"))
    }

    fn cmp_c(&mut self, dir: &str, a: &str, v: f64, shape: &[usize]) -> String {
        let k = self.c(v, shape);
        self.cmp(dir, a, &k, shape)
    }

    fn select(&mut self, mask: &str, a: &str, b: &str, shape: &[usize]) -> String {
        let (ft, bt) = (self.f(shape), ty(shape, "i1"));
        self.emit(&format!("stablehlo.select {mask}, {a}, {b} : ({bt}, {ft}, {ft}) -> {ft}"))
    }

    fn and(&mut self, a: &str, b: &str, shape: &[usize]) -> String {
        let t = ty(shape, "i1");
        self.emit(&format!("stablehlo.and {a}, {b} : {t}"))
    }

    fn i64_const(&mut self, v: i64, shape: &[usize]) -> String {
        let t = ty(shape, "i64");
        self.emit(&format!("stablehlo.constant dense<{v}> : {t}"))
    }

    fn iop2(&mut self, name: &str, a: &str, b: &str, shape: &[usize]) -> String {
        let t = ty(shape, "i64");
        self.emit(&format!("stablehlo.{name} {a}, {b} : {t}"))
    }

    /// `2^k` for integer-valued `k` (f64) in [-1022, 1023], exact: `(k + 1023) << 52` read as f64.
    fn pow2(&mut self, k: &str, shape: &[usize]) -> String {
        let (ft, it) = (self.f(shape), ty(shape, "i64"));
        let ki = self.emit(&format!("stablehlo.convert {k} : ({ft}) -> {it}"));
        let bias = self.i64_const(1023, shape);
        let biased = self.iop2("add", &ki, &bias, shape);
        let s = self.i64_const(52, shape);
        let bits = self.iop2("shift_left", &biased, &s, shape);
        self.emit(&format!("stablehlo.bitcast_convert {bits} : ({it}) -> {ft}"))
    }

    /// `e^r - 1` for |r| <= 0.35.
    fn expm1_small(&mut self, r: &str, shape: &[usize]) -> String {
        let p = self.horner(r, &EXPM1, shape);
        let pr = self.op2("multiply", &p, r, shape);
        let prr = self.op2("multiply", &pr, r, shape);
        self.op2("add", &prr, r, shape)
    }

    /// `e^x`: `x = q ln 2 + r`, `e^r` by its series, `2^q` as two exact halves (so a subnormal
    /// result rounds once).
    pub(super) fn soft_exp(&mut self, x: &str, shape: &[usize]) -> String {
        self.exp_with_low(x, None, shape)
    }

    /// `e^(x + low)` for a small `low` (the low part of a double-double argument), added into the
    /// reduced argument.
    fn exp_with_low(&mut self, x: &str, low: Option<&str>, shape: &[usize]) -> String {
        // beyond ±800, e^x is 0 or infinite anyway
        let hi = self.c(800.0, shape);
        let lo = self.c(-800.0, shape);
        let y = self.op2("minimum", x, &hi, shape);
        let y = self.op2("maximum", &y, &lo, shape);
        let scaled = self.op_c("multiply", &y, LOG2E, shape);
        let q = self.round(&scaled, shape);
        let qh = self.op_c("multiply", &q, LN2_HI, shape);
        let r = self.op2("subtract", &y, &qh, shape);
        let ql = self.op_c("multiply", &q, LN2_LO, shape);
        let mut r = self.op2("subtract", &r, &ql, shape);
        if let Some(low) = low {
            r = self.op2("add", &r, low, shape);
        }
        let em1 = self.expm1_small(&r, shape);
        let er = self.op_c("add", &em1, 1.0, shape);
        let half_q = self.op_c("multiply", &q, 0.5, shape);
        let h = self.op1("floor", &half_q, shape);
        let rest = self.op2("subtract", &q, &h, shape);
        let (p1, p2) = (self.pow2(&h, shape), self.pow2(&rest, shape));
        let out = self.op2("multiply", &er, &p1, shape);
        let out = self.op2("multiply", &out, &p2, shape);
        // NaN in, NaN out (the clamps let it through; the integer path would not)
        let nan = self.cmp("NE", x, x, shape);
        self.select(&nan, x, &out, shape)
    }

    /// `ln x`: the exponent and mantissa from the bit layout (subnormals scaled up first), the
    /// mantissa within [sqrt(1/2), sqrt 2), then `2 atanh((m - 1) / (m + 1))` by its series.
    pub(super) fn soft_log(&mut self, x: &str, shape: &[usize]) -> String {
        let (k, m) = self.log_reduce(x, shape);
        let num = self.op_c("subtract", &m, 1.0, shape);
        let den = self.op_c("add", &m, 1.0, shape);
        let s = self.op2("divide", &num, &den, shape);
        let z = self.op2("multiply", &s, &s, shape);
        let p = self.horner(&z, &ATANH, shape);
        let sz = self.op2("multiply", &s, &z, shape);
        let tail = self.op2("multiply", &sz, &p, shape);
        let tail = self.op_c("multiply", &tail, 2.0, shape);
        let kl = self.op_c("multiply", &k, LN2_LO, shape);
        let small = self.op2("add", &kl, &tail, shape);
        let two_s = self.op_c("multiply", &s, 2.0, shape);
        let body = self.op2("add", &two_s, &small, shape);
        let kh = self.op_c("multiply", &k, LN2_HI, shape);
        let out = self.op2("add", &kh, &body, shape);
        self.log_specials(x, &out, shape)
    }

    /// `ln x` for special `x`: 0 gives -inf, negatives NaN, inf and NaN themselves.
    fn log_specials(&mut self, x: &str, out: &str, shape: &[usize]) -> String {
        let is_zero = self.cmp_c("EQ", x, 0.0, shape);
        let ninf = self.c(f64::NEG_INFINITY, shape);
        let out = self.select(&is_zero, &ninf, out, shape);
        let negative = self.cmp_c("LT", x, 0.0, shape);
        let nan = self.c(f64::NAN, shape);
        let out = self.select(&negative, &nan, &out, shape);
        let inf = self.cmp_c("GT", x, f64::MAX, shape);
        let out = self.select(&inf, x, &out, shape);
        let is_nan = self.cmp("NE", x, x, shape);
        self.select(&is_nan, x, &out, shape)
    }

    /// `x = 2^k m` with `m` in [sqrt(1/2), sqrt 2): the exponent from the bit layout (subnormals
    /// scaled up first), the mantissa halved above sqrt 2. Returns `(k, m)`.
    fn log_reduce(&mut self, x: &str, shape: &[usize]) -> (String, String) {
        let (ft, it) = (self.f(shape), ty(shape, "i64"));
        // subnormals: scale by 2^54 (exact), and take 54 off the exponent
        let tiny = self.cmp_c("LT", x, 2.2250738585072014e-308, shape);
        let up = self.op_c("multiply", x, 18014398509481984.0, shape);
        let v = self.select(&tiny, &up, x, shape);
        let zero = self.c(0.0, shape);
        let minus54 = self.c(-54.0, shape);
        let adjust = self.select(&tiny, &minus54, &zero, shape);
        let bits = self.emit(&format!("stablehlo.bitcast_convert {v} : ({ft}) -> {it}"));
        let s52 = self.i64_const(52, shape);
        let shifted = self.iop2("shift_right_logical", &bits, &s52, shape);
        let mask = self.i64_const(0x7FF, shape);
        let biased = self.iop2("and", &shifted, &mask, shape);
        let e = self.emit(&format!("stablehlo.convert {biased} : ({it}) -> {ft}"));
        let e = self.op_c("subtract", &e, 1023.0, shape);
        let e = self.op2("add", &e, &adjust, shape);
        // m = mantissa with exponent 0: in [1, 2)
        let mmask = self.i64_const(0x000F_FFFF_FFFF_FFFF, shape);
        let mbits = self.iop2("and", &bits, &mmask, shape);
        let one_bits = self.i64_const(0x3FF0_0000_0000_0000, shape);
        let mbits = self.iop2("or", &mbits, &one_bits, shape);
        let m = self.emit(&format!("stablehlo.bitcast_convert {mbits} : ({it}) -> {ft}"));
        // [sqrt(1/2), sqrt 2): halve m above sqrt 2
        let big = self.cmp_c("GT", &m, std::f64::consts::SQRT_2, shape);
        let mh = self.op_c("multiply", &m, 0.5, shape);
        let m = self.select(&big, &mh, &m, shape);
        let e1 = self.op_c("add", &e, 1.0, shape);
        let k = self.select(&big, &e1, &e, shape);
        (k, m)
    }

    /// `a + b` exactly, as `(sum, error)` (Knuth's two-sum).
    fn two_sum(&mut self, a: &str, b: &str, shape: &[usize]) -> (String, String) {
        let s = self.op2("add", a, b, shape);
        let bb = self.op2("subtract", &s, a, shape);
        let sbb = self.op2("subtract", &s, &bb, shape);
        let ea = self.op2("subtract", a, &sbb, shape);
        let eb = self.op2("subtract", b, &bb, shape);
        (s, self.op2("add", &ea, &eb, shape))
    }

    /// `v` as two halves of 26 bits (Veltkamp's split), whose products are exact.
    fn split(&mut self, v: &str, shape: &[usize]) -> (String, String) {
        let c = self.op_c("multiply", v, 134217729.0, shape);
        let cv = self.op2("subtract", &c, v, shape);
        let hi = self.op2("subtract", &c, &cv, shape);
        let lo = self.op2("subtract", v, &hi, shape);
        (hi, lo)
    }

    /// `a b` exactly, as `(product, error)` (Dekker's product, no fused multiply-add needed).
    fn two_prod(&mut self, a: &str, b: &str, shape: &[usize]) -> (String, String) {
        let p = self.op2("multiply", a, b, shape);
        let (ah, al) = self.split(a, shape);
        let (bh, bl) = self.split(b, shape);
        let hh = self.op2("multiply", &ah, &bh, shape);
        let e = self.op2("subtract", &hh, &p, shape);
        let hl = self.op2("multiply", &ah, &bl, shape);
        let e = self.op2("add", &e, &hl, shape);
        let lh = self.op2("multiply", &al, &bh, shape);
        let e = self.op2("add", &e, &lh, shape);
        let ll = self.op2("multiply", &al, &bl, shape);
        (p, self.op2("add", &e, &ll, shape))
    }

    /// `ln x` for finite positive `x` as a double-double `(hi, lo)`: the series' leading term
    /// `2 s` with `s = (m - 1) / (m + 1)` carried with the division's exact residual.
    fn log_double(&mut self, x: &str, shape: &[usize]) -> (String, String) {
        let (k, m) = self.log_reduce(x, shape);
        // m - 1 is exact for m in [0.7, 1.42]; m + 1 as a sum and its error
        let num = self.op_c("subtract", &m, 1.0, shape);
        let one = self.c(1.0, shape);
        let (den, den_e) = self.two_sum(&m, &one, shape);
        let s = self.op2("divide", &num, &den, shape);
        // s_lo = (num - s den - s den_e) / den, with s den exact
        let (p, pe) = self.two_prod(&s, &den, shape);
        let r = self.op2("subtract", &num, &p, shape);
        let r = self.op2("subtract", &r, &pe, shape);
        let sde = self.op2("multiply", &s, &den_e, shape);
        let r = self.op2("subtract", &r, &sde, shape);
        let s_lo = self.op2("divide", &r, &den, shape);
        let z = self.op2("multiply", &s, &s, shape);
        let poly = self.horner(&z, &ATANH, shape);
        let sz = self.op2("multiply", &s, &z, shape);
        let tail = self.op2("multiply", &sz, &poly, shape);
        let tail = self.op_c("multiply", &tail, 2.0, shape);
        // k ln2_hi (exact) + 2 s, then everything small
        let kh = self.op_c("multiply", &k, LN2_HI, shape);
        let two_s = self.op_c("multiply", &s, 2.0, shape);
        let (h, he) = self.two_sum(&kh, &two_s, shape);
        let kl = self.op_c("multiply", &k, LN2_LO, shape);
        let two_slo = self.op_c("multiply", &s_lo, 2.0, shape);
        let small = self.op2("add", &kl, &two_slo, shape);
        let small = self.op2("add", &small, &tail, shape);
        let lo = self.op2("add", &he, &small, shape);
        // renormalized
        let hi = self.op2("add", &h, &lo, shape);
        let back = self.op2("subtract", &hi, &h, shape);
        (hi.clone(), self.op2("subtract", &lo, &back, shape))
    }

    /// `sin x` (or `cos x`): `x = q pi/2 + r` with a three-part pi/2, both series on `r`, the
    /// quadrant (`q mod 4`, shifted by one for the cosine) choosing which and its sign.
    pub(super) fn soft_sin_cos(&mut self, x: &str, shape: &[usize], cosine: bool) -> String {
        let scaled = self.op_c("multiply", x, TWO_OVER_PI, shape);
        let q = self.round(&scaled, shape);
        let mut r = x.to_string();
        for p in PIO2 {
            let qp = self.op_c("multiply", &q, p, shape);
            r = self.op2("subtract", &r, &qp, shape);
        }
        let z = self.op2("multiply", &r, &r, shape);
        let sp = self.horner(&z, &SIN, shape);
        let rz = self.op2("multiply", &r, &z, shape);
        let rzs = self.op2("multiply", &rz, &sp, shape);
        let sin_r = self.op2("add", &r, &rzs, shape);
        let cp = self.horner(&z, &COS, shape);
        let zc = self.op2("multiply", &z, &cp, shape);
        let cos_r = self.op_c("add", &zc, 1.0, shape);
        // quadrant = (q + cosine) mod 4, in f64 (exact for |q| < 2^51)
        let qq = if cosine { self.op_c("add", &q, 1.0, shape) } else { q };
        let quarter = self.op_c("multiply", &qq, 0.25, shape);
        let fl = self.op1("floor", &quarter, shape);
        let four_fl = self.op_c("multiply", &fl, 4.0, shape);
        let quadrant = self.op2("subtract", &qq, &four_fl, shape);
        let neg_sin = self.op1("negate", &sin_r, shape);
        let neg_cos = self.op1("negate", &cos_r, shape);
        let is1 = self.cmp_c("EQ", &quadrant, 1.0, shape);
        let is2 = self.cmp_c("EQ", &quadrant, 2.0, shape);
        let is3 = self.cmp_c("EQ", &quadrant, 3.0, shape);
        let out = self.select(&is1, &cos_r, &sin_r, shape);
        let out = self.select(&is2, &neg_sin, &out, shape);
        self.select(&is3, &neg_cos, &out, shape)
    }

    /// `tanh x = e / (e + 2)` with `e = e^(2|x|) - 1` (its series near 0), the sign restored.
    pub(super) fn soft_tanh(&mut self, x: &str, shape: &[usize]) -> String {
        let a = self.op1("abs", x, shape);
        let y = self.op_c("multiply", &a, 2.0, shape);
        let ey = self.soft_exp(&y, shape);
        let e_big = self.op_c("subtract", &ey, 1.0, shape);
        let e_small = self.expm1_small(&y, shape);
        let small = self.cmp_c("LT", &y, 0.35, shape);
        let e = self.select(&small, &e_small, &e_big, shape);
        let e2 = self.op_c("add", &e, 2.0, shape);
        let t = self.op2("divide", &e, &e2, shape);
        let one = self.c(1.0, shape);
        let large = self.cmp_c("GT", &a, 22.0, shape);
        let t = self.select(&large, &one, &t, shape);
        let neg = self.cmp_c("LT", x, 0.0, shape);
        let nt = self.op1("negate", &t, shape);
        self.select(&neg, &nt, &t, shape)
    }

    /// `a^b = e^(b ln |a|)`, negative bases taking the sign of an integer exponent (NaN for a
    /// fractional one), `x^0 = 1`. `ln |a|` is carried in double-double and `b ln |a|` formed
    /// exactly, so the result stays within a few ulps however large `b ln |a|` is (in plain
    /// doubles its rounding would be multiplied by it).
    pub(super) fn soft_pow(&mut self, a: &str, b: &str, shape: &[usize]) -> String {
        let abs = self.op1("abs", a, shape);
        let (l, l_lo) = self.log_double(&abs, shape);
        let (w, we) = self.two_prod(b, &l, shape);
        let bl_lo = self.op2("multiply", b, &l_lo, shape);
        let low = self.op2("add", &we, &bl_lo, shape);
        let precise = self.exp_with_low(&w, Some(&low), shape);
        // zero, infinite and NaN bases (and huge exponents) through the plain path
        let plain_log = self.log_specials(&abs, &l, shape);
        let plain = self.op2("multiply", b, &plain_log, shape);
        let plain = self.soft_exp(&plain, shape);
        let ordinary = self.cmp_c("GT", &abs, 0.0, shape);
        let finite = self.cmp_c("LT", &abs, f64::INFINITY, shape);
        let ok = self.and(&ordinary, &finite, shape);
        let bmax = self.c(1e290, shape);
        let babs = self.op1("abs", b, shape);
        let moderate = self.cmp("LT", &babs, &bmax, shape);
        let ok = self.and(&ok, &moderate, shape);
        let mag = self.select(&ok, &precise, &plain, shape);
        // the sign for a negative base: integer exponents only, odd ones negative
        let fb = self.op1("floor", b, shape);
        let integral = self.cmp("EQ", &fb, b, shape);
        let half = self.op_c("multiply", b, 0.5, shape);
        let fh = self.op1("floor", &half, shape);
        let even = self.cmp("EQ", &fh, &half, shape);
        let neg_mag = self.op1("negate", &mag, shape);
        let signed = self.select(&even, &mag, &neg_mag, shape);
        let nan = self.c(f64::NAN, shape);
        let negative_base = self.select(&integral, &signed, &nan, shape);
        let neg = self.cmp_c("LT", a, 0.0, shape);
        let out = self.select(&neg, &negative_base, &mag, shape);
        let zero_exp = self.cmp_c("EQ", b, 0.0, shape);
        let one = self.c(1.0, shape);
        self.select(&zero_exp, &one, &out, shape)
    }
}
