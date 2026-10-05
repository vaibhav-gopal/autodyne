//! The real Schur decomposition (`scipy.linalg.schur`, `output="real"`): `a = z t zᵀ` with `z`
//! orthogonal and `t` quasi-upper-triangular (1 x 1 blocks for real eigenvalues, 2 x 2 blocks for
//! complex pairs).
//!
//! - Hessenberg form: faer's reduction (LAPACK's `dgehrd` / `dorghr`) from 64 x 64, EISPACK's
//!   `orthes` below, where faer's set-up costs more than the work.
//! - Then Francis double-shift QR with the transformations accumulated: the Schur phase of EISPACK's
//!   `hqr2` (by way of JAMA), vectorized, below 200 x 200; from there LAPACK's `dlaqr0` structure,
//!   whose aggressive early deflation (`dlaqr3`, with the Schur-form reordering in
//!   [`reorder`](super::reorder)) finds converged eigenvalues in a window at the bottom of the active
//!   block and feeds the others back as shifts. LAPACK chases those shifts as many bulges at once
//!   (`dlaqr5`, its updates in matrix products); here each pair is its own sweep, which leaves
//!   LAPACK about 10% ahead from 300 x 300.
//!
//! tend: Numerics / linalg / schur

use std::ops::Range;

use super::dense::Mat;
use super::reorder;
use super::{LinalgError, LinalgFloat};
use crate::signal::{NdArray, NdView};

/// A real Schur form: `a = z t zᵀ`.
pub(crate) struct Schur {
    pub(crate) t: Mat,
    pub(crate) z: Mat,
}

/// Below this size [`hessenberg`] runs [`hessenberg_small`].
const SMALL: usize = 64;

/// Reduces `a` to upper Hessenberg form, returning `(h, vᵀ)` with `a = v h vᵀ` and `v`
/// orthogonal: faer's reduction (blocked from 256 x 256, with matrix products) and its block
/// Householder sequence applied to the identity for `v`, as LAPACK's `dgehrd` and `dorghr` do.
#[inline(always)]
fn hessenberg(a: &Mat) -> (Mat, Mat) {
    use faer::dyn_stack::{MemBuffer, MemStack, StackReq};
    use faer::linalg::evd::hessenberg as hess;
    use faer::linalg::householder;
    let n = a.rows;
    if n < SMALL {
        let mut h = a.clone();
        let v = hessenberg_small(&mut h);
        return (h, v.t());
    }
    // one thread: faer's threaded reduction was slower at every size tried
    let par = faer::Par::Seq;
    let mut h = faer::Mat::<f64>::from_fn(n, n, |i, j| a.at(i, j));
    let bs = faer::linalg::qr::no_pivoting::factor::recommended_block_size::<f64>(n - 1, n - 1);
    let mut factors = faer::Mat::<f64>::zeros(bs, n - 1);
    let mut v = faer::Mat::<f64>::identity(n, n);
    let mut buf = MemBuffer::new(StackReq::any_of(&[
        hess::hessenberg_in_place_scratch::<f64>(n, bs, par, Default::default()),
        householder::apply_block_householder_sequence_on_the_right_in_place_scratch::<f64>(n - 1, bs, n - 1),
    ]));
    let stack = MemStack::new(&mut buf);
    hess::hessenberg_in_place(h.as_mut(), factors.as_mut(), par, stack, Default::default());
    householder::apply_block_householder_sequence_on_the_right_in_place_with_conj(
        h.as_ref().submatrix(1, 0, n - 1, n - 1),
        factors.as_ref(),
        faer::Conj::No,
        v.as_mut().submatrix_mut(1, 1, n - 1, n - 1),
        par,
        stack,
    );
    // h row-major, without the Householder vectors stored below its subdiagonal; v is column-major,
    // so its columns read in order are the rows of vᵀ
    let h = Mat::from_fn(n, n, |i, j| if i > j + 1 { 0.0 } else { h[(i, j)] });
    let vt = Mat::from_fn(n, n, |i, j| v[(j, i)]);
    (h, vt)
}

/// [`hessenberg`] for small matrices, where faer's set-up costs more than the reduction: in place,
/// returning the orthogonal `v` with `a = v h vᵀ` (EISPACK `orthes` + `ortran`).
// indexed loops, as in EISPACK, read more clearly here than iterator chains
#[allow(clippy::needless_range_loop)]
#[inline(always)]
fn hessenberg_small(h: &mut Mat) -> Mat {
    let n = h.rows;
    let mut v = Mat::identity(n);
    if n < 3 {
        return v;
    }
    let (low, high) = (0, n - 1);
    let mut ort = vec![0.0; n];
    for m in low + 1..high {
        let scale: f64 = (m..=high).map(|i| h.at(i, m - 1).abs()).sum();
        if scale == 0.0 {
            continue;
        }
        let mut hh = 0.0;
        for i in (m..=high).rev() {
            ort[i] = h.at(i, m - 1) / scale;
            hh += ort[i] * ort[i];
        }
        let mut g = hh.sqrt();
        if ort[m] > 0.0 {
            g = -g;
        }
        hh -= ort[m] * g;
        ort[m] -= g;
        // H = (I - u uᵀ / hh) H (I - u uᵀ / hh), a row at a time (the matrix is row-major): fᵀ = uᵀ H
        // in one pass, then each row's update from the left and, while it is in cache, from the
        // right (the two sides commute)
        let u = &ort[m..=high];
        let mut f = vec![0.0; n - m];
        for (o, i) in u.iter().zip(m..=high) {
            for (fj, &hij) in f.iter_mut().zip(&h.data[i * n + m..(i + 1) * n]) {
                *fj += o * hij;
            }
        }
        f.iter_mut().for_each(|v| *v /= hh);
        let right = |row: &mut [f64]| {
            let g = crate::simd::dot_kernel(row, u) / hh;
            row.iter_mut().zip(u).for_each(|(hij, o)| *hij -= g * o);
        };
        for i in 0..m {
            right(&mut h.data[i * n + m..i * n + high + 1]);
        }
        for (o, i) in u.iter().zip(m..=high) {
            let row = &mut h.data[i * n + m..(i + 1) * n];
            for (hij, fj) in row.iter_mut().zip(&f) {
                *hij -= fj * o;
            }
            right(&mut row[..high + 1 - m]);
        }
        ort[m] *= scale;
        h.set(m, m - 1, scale * g);
    }
    // accumulate the transformations
    for m in (low + 1..high).rev() {
        if h.at(m, m - 1) == 0.0 {
            continue;
        }
        for i in m + 1..=high {
            ort[i] = h.at(i, m - 1);
        }
        // g_j = sum_i ort[i] v[i][j], accumulated a row at a time
        let mut g = vec![0.0; high + 1 - m];
        for i in m..=high {
            let o = ort[i];
            for (gj, &vij) in g.iter_mut().zip(&v.data[i * n + m..i * n + high + 1]) {
                *gj += o * vij;
            }
        }
        // double division avoids possible underflow
        let denom = h.at(m, m - 1);
        g.iter_mut().for_each(|gj| *gj = (*gj / ort[m]) / denom);
        for i in m..=high {
            let o = ort[i];
            for (vij, gj) in v.data[i * n + m..i * n + high + 1].iter_mut().zip(&g) {
                *vij += gj * o;
            }
        }
    }
    // the Householder vectors stored below the subdiagonal are done with
    for i in 2..n {
        for j in 0..i - 1 {
            h.set(i, j, 0.0);
        }
    }
    v
}

/// One Householder reflector of the double-shift QR sweep, on two or three consecutive indices
/// (`hqr2`'s `x, y, z` and `q, r`).
struct Reflector {
    x: f64,
    y: f64,
    z: f64,
    q: f64,
    r: f64,
    three: bool,
}

impl Reflector {
    /// From the left: rows `k`, `k + 1` (and `k + 2`) of a row-major matrix with `cols` columns,
    /// from column `from` on.
    #[inline]
    fn left(&self, data: &mut [f64], cols: usize, k: usize, from: usize) {
        let (rows, rest) = data[k * cols..].split_at_mut(cols);
        let (next, after) = rest.split_at_mut(cols);
        let (r0, r1) = (&mut rows[from..], &mut next[from..]);
        let Reflector { x, y, z, q, r, three } = *self;
        if three {
            for ((a, b), c) in r0.iter_mut().zip(r1.iter_mut()).zip(after[from..cols].iter_mut()) {
                let p = *a + q * *b + r * *c;
                *c -= p * z;
                *a -= p * x;
                *b -= p * y;
            }
        } else {
            for (a, b) in r0.iter_mut().zip(r1.iter_mut()) {
                let p = *a + q * *b;
                *a -= p * x;
                *b -= p * y;
            }
        }
    }

    /// From the right on one row's two or three entries.
    #[inline]
    fn right(&self, e: &mut [f64]) {
        let Reflector { x, y, z, q, r, three } = *self;
        let mut p = x * e[0] + y * e[1];
        if three {
            p += z * e[2];
            e[2] -= p * r;
        }
        e[0] -= p;
        e[1] -= p * q;
    }

    /// From the right on columns `k..` of rows `rows` of the row-major `h` (`nn` columns): three
    /// entries a row, strided, so four rows at a time with the arithmetic across rows.
    #[inline(always)]
    fn right_cols(&self, h: &mut [f64], nn: usize, k: usize, rows: Range<usize>) {
        let Reflector { x, y, z, q, r, three } = *self;
        let mut i = rows.start;
        if three {
            while i + 4 <= rows.end {
                let (mut a, mut b, mut c) = ([0.0; 4], [0.0; 4], [0.0; 4]);
                for g in 0..4 {
                    let e = &h[(i + g) * nn + k..(i + g) * nn + k + 3];
                    (a[g], b[g], c[g]) = (e[0], e[1], e[2]);
                }
                for g in 0..4 {
                    let p = x * a[g] + y * b[g] + z * c[g];
                    c[g] -= p * r;
                    a[g] -= p;
                    b[g] -= p * q;
                }
                for g in 0..4 {
                    h[(i + g) * nn + k..(i + g) * nn + k + 3].copy_from_slice(&[a[g], b[g], c[g]]);
                }
                i += 4;
            }
        }
        for i in i..rows.end {
            self.right(&mut h[i * nn + k..i * nn + k + if three { 3 } else { 2 }]);
        }
    }

    /// `self` then `next` (three-index reflectors on rows `k..k + 3` and `k + 1..k + 4`) from the right on
    /// every row of a matrix, applied to its transpose `t` (`cols` columns): each column's four
    /// entries loaded and stored once for both.
    #[inline(always)]
    fn right_rows_pair(&self, next: &Reflector, t: &mut [f64], cols: usize, k: usize) {
        let mut rows = t[k * cols..(k + 4) * cols].chunks_exact_mut(cols);
        let mut row = || rows.next().expect("four rows");
        let (row0, row1, row2, row3) = (row(), row(), row(), row());
        let Reflector { x, y, z, q, r, .. } = *self;
        let Reflector { x: x2, y: y2, z: z2, q: q2, r: r2, three } = *next;
        let _ = three;
        debug_assert!(three, "pairs of three-index reflectors");
        for (((a, b), c), d) in row0.iter_mut().zip(row1.iter_mut()).zip(row2.iter_mut()).zip(row3.iter_mut()) {
            let p = x * *a + y * *b + z * *c;
            let (b1, c1) = (*b - p * q, *c - p * r);
            *a -= p;
            let p2 = x2 * b1 + y2 * c1 + z2 * *d;
            *d -= p2 * r2;
            *b = b1 - p2;
            *c = c1 - p2 * q2;
        }
    }

    /// From the right on every row of a matrix, applied to its transpose: rows `k`, `k + 1` (and
    /// `k + 2`) of `t` (`cols` columns), contiguous.
    #[inline]
    fn right_rows(&self, t: &mut [f64], cols: usize, k: usize) {
        let (rows, rest) = t[k * cols..].split_at_mut(cols);
        let (next, after) = rest.split_at_mut(cols);
        let Reflector { x, y, z, q, r, three } = *self;
        if three {
            for ((a, b), c) in rows.iter_mut().zip(next.iter_mut()).zip(after[..cols].iter_mut()) {
                let p = x * *a + y * *b + z * *c;
                *c -= p * r;
                *a -= p;
                *b -= p * q;
            }
        } else {
            for (a, b) in rows.iter_mut().zip(next.iter_mut()) {
                let p = x * *a + y * *b;
                *a -= p;
                *b -= p * q;
            }
        }
    }
}

/// The real Schur form of `a` (n x n): `a = z t zᵀ`. Errors if the QR iteration does not converge
/// (30 n double-shift steps).
pub(crate) fn real_schur(a: &Mat) -> Result<Schur, LinalgError> {
    // entries far from 1 scaled by a power of two first (exact), as LAPACK's dgees scales: the
    // iteration multiplies entries together
    let big = a.data.iter().fold(0.0f64, |m, v| m.max(v.abs()));
    if big.is_finite() && big > 0.0 && !(2f64.powi(-400)..=2f64.powi(400)).contains(&big) {
        let e = big.log2().round() as i32;
        let mut s = real_schur(&a.scale(2f64.powi(-e)))?;
        s.t = s.t.scale(2f64.powi(e));
        return Ok(s);
    }
    #[cfg(target_arch = "x86_64")]
    if crate::simd::avx2_available() {
        // SAFETY: the CPU was just checked for AVX2, the only feature real_schur_avx2 is compiled with.
        return unsafe { real_schur_avx2(a) };
    }
    real_schur_body(a)
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
unsafe fn real_schur_avx2(a: &Mat) -> Result<Schur, LinalgError> {
    real_schur_body(a)
}

/// From this size on the QR iteration deflates aggressively ([`aed_qr`], LAPACK's `dlaqr0`);
/// below, [`hqr_block`] runs on the whole matrix (`dlahqr`'s place). LAPACK switches at 75, where
/// its sweeps chase many bulges at once in matrix products; with this crate's one-bulge sweeps
/// aggressive deflation starts paying at about 200 (and at 400 takes a quarter off).
const AED_FROM: usize = 200;

/// [`aed_qr`] finishes active blocks of up to this many rows whole ([`finish_block`]).
const FINISH: usize = 75;

/// AED rounds that deflate more than this percentage of the window skip the sweep and look again
/// (LAPACK's `nibble`).
const NIBBLE: usize = 14;

/// [`real_schur`]'s work, inlined into each instruction-set version.
#[inline(always)]
#[allow(clippy::needless_range_loop)]
fn real_schur_body(a: &Mat) -> Result<Schur, LinalgError> {
    let nn = a.rows;
    // the Schur vectors kept transposed, so their updates touch rows instead of columns
    let (mut h, mut vt) = hessenberg(a);
    // complex pairs found, by their upper index (their 2 x 2 blocks stay)
    let mut pair = vec![false; nn];
    let mut steps = Steps { done: 0, limit: 30 * nn.max(10) };
    if nn >= AED_FROM {
        aed_qr(&mut h, &mut vt, &mut pair, &mut steps)?;
    } else if nn > 0 {
        hqr_block(&mut h, &mut vt, 0, nn - 1, &mut pair, &mut steps)?;
    }
    // a clean quasi-triangular form: only the complex pairs' subdiagonals remain
    for i in 1..nn {
        for j in 0..i {
            if !(j + 1 == i && pair[j]) {
                h.set(i, j, 0.0);
            }
        }
    }
    Ok(Schur { t: h, z: vt.t() })
}

/// Double-shift QR steps taken, and the most allowed.
struct Steps {
    done: usize,
    limit: usize,
}

impl Steps {
    fn take(&mut self) -> Result<(), LinalgError> {
        self.done += 1;
        if self.done > self.limit {
            return Err(LinalgError::NoConvergence);
        }
        Ok(())
    }
}

/// EISPACK `hqr2`'s iteration on the diagonal block `lo..=hi` of the Hessenberg `h` until the
/// block is quasi-triangular: every transformation applied to all of `h` (the rows above the block
/// and the columns right of it too) and to the transposed Schur vectors `vt`. Marks the complex
/// pairs found in `pair`, by their upper index.
#[inline(always)]
#[allow(clippy::needless_range_loop)]
fn hqr_block(h: &mut Mat, vt: &mut Mat, lo: usize, hi: usize, pair: &mut [bool], steps: &mut Steps) -> Result<(), LinalgError> {
    let nn = h.rows;
    let eps = f64::EPSILON;
    let mut exshift = 0.0;
    let (mut p, mut q, mut r, mut s, mut z);
    let (mut w, mut x, mut y);
    let norm: f64 = (lo..=hi).map(|i| (i.saturating_sub(1).max(lo)..=hi).map(|j| h.at(i, j).abs()).sum::<f64>()).sum();
    let low = lo as isize;
    let mut n = hi as isize;
    let mut iter = 0;
    while n >= low {
        let nu = n as usize;
        // look for a single small subdiagonal element
        let mut l = n;
        while l > low {
            let lu = l as usize;
            s = h.at(lu - 1, lu - 1).abs() + h.at(lu, lu).abs();
            if s == 0.0 {
                s = norm;
            }
            // (<=: a zero matrix splits everywhere)
            if h.at(lu, lu - 1).abs() <= eps * s {
                break;
            }
            l -= 1;
        }
        if l == n {
            // one root found
            h.set(nu, nu, h.at(nu, nu) + exshift);
            if nu > lo {
                h.set(nu, nu - 1, 0.0);
            }
            n -= 1;
            iter = 0;
        } else if l == n - 1 {
            // two roots found
            w = h.at(nu, nu - 1) * h.at(nu - 1, nu);
            p = (h.at(nu - 1, nu - 1) - h.at(nu, nu)) / 2.0;
            q = p * p + w;
            z = q.abs().sqrt();
            h.set(nu, nu, h.at(nu, nu) + exshift);
            h.set(nu - 1, nu - 1, h.at(nu - 1, nu - 1) + exshift);
            if q >= 0.0 {
                // a real pair: rotate the block to upper triangular
                z = if p >= 0.0 { p + z } else { p - z };
                x = h.at(nu, nu - 1);
                s = x.abs() + z.abs();
                if s == 0.0 {
                    // already triangular: no rotation
                    (p, q) = (0.0, 1.0);
                } else {
                    p = x / s;
                    q = z / s;
                    r = (p * p + q * q).sqrt();
                    p /= r;
                    q /= r;
                }
                for j in nu - 1..nn {
                    let zz = h.at(nu - 1, j);
                    h.set(nu - 1, j, q * zz + p * h.at(nu, j));
                    h.set(nu, j, q * h.at(nu, j) - p * zz);
                }
                for i in 0..=nu {
                    let zz = h.at(i, nu - 1);
                    h.set(i, nu - 1, q * zz + p * h.at(i, nu));
                    h.set(i, nu, q * h.at(i, nu) - p * zz);
                }
                for i in 0..vt.cols {
                    let zz = vt.at(nu - 1, i);
                    vt.set(nu - 1, i, q * zz + p * vt.at(nu, i));
                    vt.set(nu, i, q * vt.at(nu, i) - p * zz);
                }
                h.set(nu, nu - 1, 0.0);
            } else {
                pair[nu - 1] = true;
            }
            if nu >= lo + 2 {
                h.set(nu - 1, nu - 2, 0.0);
            }
            n -= 2;
            iter = 0;
        } else {
            // no convergence yet: form the shift
            x = h.at(nu, nu);
            y = h.at(nu - 1, nu - 1);
            w = h.at(nu, nu - 1) * h.at(nu - 1, nu);
            // Wilkinson's original ad hoc shift
            if iter == 10 {
                exshift += x;
                for i in lo..=nu {
                    h.set(i, i, h.at(i, i) - x);
                }
                s = h.at(nu, nu - 1).abs() + h.at(nu - 1, nu - 2).abs();
                x = 0.75 * s;
                y = x;
                w = -0.4375 * s * s;
            }
            // MATLAB's ad hoc shift
            if iter == 30 {
                s = (y - x) / 2.0;
                s = s * s + w;
                if s > 0.0 {
                    s = s.sqrt();
                    if y < x {
                        s = -s;
                    }
                    s = x - w / ((y - x) / 2.0 + s);
                    for i in lo..=nu {
                        h.set(i, i, h.at(i, i) - s);
                    }
                    exshift += s;
                    x = 0.964;
                    y = x;
                    w = x;
                }
            }
            iter += 1;
            steps.take()?;
            double_step(h, vt, l as usize, nu, x, y, w);
        }
    }
    // the block quasi-triangular: no bulge remnants, subdiagonals only under complex pairs
    for i in lo + 1..=hi {
        if !pair[i - 1] {
            h.set(i, i - 1, 0.0);
        }
        for j in i.saturating_sub(3).max(lo)..i - 1 {
            h.set(i, j, 0.0);
        }
    }
    Ok(())
}

/// One double-shift (Francis) QR step on rows and columns `l..=nu` of `h` (`nu >= l + 2`), with
/// the two shifts given as `hqr2` gives them: the eigenvalues of `[[y, ·], [·, x]]` whose
/// off-diagonal entries multiply to `w`. It starts below two consecutive small subdiagonal entries
/// if it finds them; every reflector is applied to all of `h` and to the transposed Schur vectors
/// `vt`. Leaves the bulge's remnants below the subdiagonal (the next step's start clears them).
#[inline(always)]
fn double_step(h: &mut Mat, vt: &mut Mat, l: usize, nu: usize, x: f64, y: f64, w: f64) {
    let nn = h.rows;
    let eps = f64::EPSILON;
    let (mut p, mut q, mut r, mut s, mut z);
    // look for two consecutive small subdiagonal elements
    let mut m = nu - 2;
    loop {
        z = h.at(m, m);
        r = x - z;
        s = y - z;
        p = (r * s - w) / h.at(m + 1, m) + h.at(m, m + 1);
        q = h.at(m + 1, m + 1) - z - r - s;
        r = h.at(m + 2, m + 1);
        s = p.abs() + q.abs() + r.abs();
        p /= s;
        q /= s;
        r /= s;
        if m == l {
            break;
        }
        if h.at(m, m - 1).abs() * (q.abs() + r.abs()) < eps * (p.abs() * (h.at(m - 1, m - 1).abs() + z.abs() + h.at(m + 1, m + 1).abs())) {
            break;
        }
        m -= 1;
    }
    for i in m + 2..=nu {
        h.set(i, i - 2, 0.0);
        if i > m + 2 {
            h.set(i, i - 3, 0.0);
        }
    }
    // the bulge chased from row m to the bottom
    let mut pending: Option<(usize, Reflector)> = None;
    let mut scale = 0.0;
    for k in m..nu {
        let notlast = k != nu - 1;
        if k != m {
            p = h.at(k, k - 1);
            q = h.at(k + 1, k - 1);
            r = if notlast { h.at(k + 2, k - 1) } else { 0.0 };
            scale = p.abs() + q.abs() + r.abs();
            if scale == 0.0 {
                continue;
            }
            p /= scale;
            q /= scale;
            r /= scale;
        }
        s = (p * p + q * q + r * r).sqrt();
        if p < 0.0 {
            s = -s;
        }
        if s != 0.0 {
            if k != m {
                h.set(k, k - 1, -s * scale);
            } else if l != m {
                h.set(k, k - 1, -h.at(k, k - 1));
            }
            p += s;
            let reflector = Reflector { x: p / s, y: q / s, z: r / s, q: q / p, r: r / p, three: notlast };
            // from the left on rows k.. of h (columns k..), from the right on its columns
            // k.. (rows up to k + 3) and on the Schur vectors (rows of vt)
            reflector.left(&mut h.data, nn, k, k);
            reflector.right_cols(&mut h.data, nn, k, 0..nu.min(k + 3) + 1);
            // the Schur vectors two reflectors at a time when the step before had one
            match pending.take() {
                Some((j, prev)) if j + 1 == k && notlast => prev.right_rows_pair(&reflector, &mut vt.data, vt.cols, j),
                prev => {
                    if let Some((j, prev)) = prev {
                        prev.right_rows(&mut vt.data, vt.cols, j);
                    }
                    if notlast {
                        pending = Some((k, reflector));
                    } else {
                        reflector.right_rows(&mut vt.data, vt.cols, k);
                    }
                }
            }
        }
    }
    if let Some((j, prev)) = pending {
        prev.right_rows(&mut vt.data, vt.cols, j);
    }
}

/// The number of shifts per round for an active block of `nh` rows (LAPACK's `iparmq`).
fn shift_count(nh: usize) -> usize {
    let ns = match nh {
        0..30 => 2,
        30..60 => 4,
        60..150 => 10,
        150..590 => (nh / (nh as f64).log2().round() as usize).max(10),
        590..3000 => 64,
        3000..6000 => 128,
        _ => 256,
    };
    (ns - ns % 2).max(2)
}

/// LAPACK's `dlaqr0` on the Hessenberg `h`: each round looks for converged eigenvalues at the
/// bottom of the active block with aggressive early deflation ([`aed`]); unless that deflated
/// plenty, the window's unconverged eigenvalues then go through the block as shifts, a pair per
/// double-shift sweep. Active blocks of up to [`FINISH`] rows are finished whole.
#[inline(always)]
fn aed_qr(h: &mut Mat, vt: &mut Mat, pair: &mut [bool], steps: &mut Steps) -> Result<(), LinalgError> {
    let nn = h.rows;
    let mut kbot = nn - 1;
    // rounds since the last deflation, and the last window size
    let (mut stalled, mut nw) = (0usize, 0usize);
    loop {
        let mut ktop = kbot;
        while ktop > 0 && h.at(ktop, ktop - 1) != 0.0 {
            ktop -= 1;
        }
        let nh = kbot + 1 - ktop;
        if nh <= FINISH {
            finish_block(h, vt, pair, ktop, kbot, steps)?;
            if ktop == 0 {
                return Ok(());
            }
            (kbot, stalled) = (ktop - 1, 0);
            continue;
        }
        // LAPACK's window: as many rows as shifts, doubled each round from the fifth without a
        // deflation, the whole block if it would leave a row
        let ns = shift_count(nh);
        let nwr = ns.min((nn - 1) / 3).min(nh).max(2);
        nw = if stalled < 5 { nwr } else { (2 * nw).min(nh) };
        if nw + 1 >= nh {
            nw = nh;
        }
        let (nd, shifts) = aed(h, vt, pair, ktop, kbot, nw);
        if nd == nh {
            if ktop == 0 {
                return Ok(());
            }
            (kbot, stalled) = (ktop - 1, 0);
            continue;
        }
        kbot -= nd;
        stalled = if nd > 0 { 0 } else { stalled + 1 };
        if nd > 0 && (100 * nd > nw * NIBBLE || kbot + 1 - ktop <= FINISH) {
            // deflated plenty: look again before sweeping
            continue;
        }
        let pairs = if stalled > 0 && stalled % 6 == 0 { exceptional_shifts(h, ktop, kbot, ns) } else { shift_pairs(h, ktop, kbot, &shifts, ns) };
        for (x, y, w) in pairs {
            steps.take()?;
            double_step(h, vt, ktop, kbot, x, y, w);
            for i in ktop + 2..=kbot {
                h.set(i, i - 2, 0.0);
                if i >= ktop + 3 {
                    h.set(i, i - 3, 0.0);
                }
            }
            if split(h, ktop, kbot) {
                break;
            }
        }
    }
}

/// Sets the negligible subdiagonal entries of the block `ktop..=kbot` to zero; whether there were.
fn split(h: &mut Mat, ktop: usize, kbot: usize) -> bool {
    let mut found = false;
    for i in ktop + 1..=kbot {
        let mut tst = h.at(i - 1, i - 1).abs() + h.at(i, i).abs();
        if tst == 0.0 {
            if i >= ktop + 2 {
                tst += h.at(i - 1, i - 2).abs();
            }
            if i < kbot {
                tst += h.at(i + 1, i).abs();
            }
        }
        if h.at(i, i - 1).abs() <= f64::MIN_POSITIVE.max(f64::EPSILON * tst) {
            h.set(i, i - 1, 0.0);
            found = true;
        }
    }
    found
}

/// Finishes the active block `ktop..=kbot`: its Schur form on a copy, applied to the rest of the
/// matrix at once ([`aed`] with the whole block as its window), or, should that copy's iteration
/// fail, [`hqr_block`] in place.
fn finish_block(h: &mut Mat, vt: &mut Mat, pair: &mut [bool], ktop: usize, kbot: usize, steps: &mut Steps) -> Result<(), LinalgError> {
    let nh = kbot + 1 - ktop;
    if aed(h, vt, pair, ktop, kbot, nh).0 < nh {
        hqr_block(h, vt, ktop, kbot, pair, steps)?;
    }
    Ok(())
}

/// The eigenvalues `(re, im)` of the quasi-triangular `t`'s first `ns` rows, top to bottom.
fn eigenvalues(t: &Mat, ns: usize) -> Vec<(f64, f64)> {
    let mut out = Vec::with_capacity(ns);
    let mut j = 0;
    while j < ns {
        if j + 1 < ns && t.at(j + 1, j) != 0.0 {
            let (a, b, c, d) = (t.at(j, j), t.at(j, j + 1), t.at(j + 1, j), t.at(j + 1, j + 1));
            let p = 0.5 * (a - d);
            let disc = p * p + b * c;
            if disc < 0.0 {
                let (re, im) = (0.5 * (a + d), (-disc).sqrt());
                out.extend([(re, im), (re, -im)]);
            } else {
                let r = disc.sqrt().copysign(p);
                let big = d + p + r;
                let small = if big != 0.0 { (a * d - b * c) / big } else { 0.0 };
                out.extend([(big, 0.0), (small, 0.0)]);
            }
            j += 2;
        } else {
            out.push((t.at(j, j), 0.0));
            j += 1;
        }
    }
    out
}

/// Aggressive early deflation (LAPACK's `dlaqr3`) on the window of the bottom `nw` rows of the
/// active block `ktop..=kbot`: the window's Schur form, computed on a copy, decouples from the
/// rest of the block wherever the matching entry of the spike (the window's link to the rows
/// above, `h[kwtop][kwtop - 1]` times the first row of the window's Schur vectors) is negligible.
/// Those eigenvalues are moved to the bottom and deflated; the others are moved to the top and
/// returned as shifts. If any deflated, the window goes back as the deflated blocks below the rest
/// returned to Hessenberg form, its transformation applied to the rest of `h` and to `vt`.
/// Returns the number deflated (all of them with the whole block as the window) and the shifts.
fn aed(h: &mut Mat, vt: &mut Mat, pair: &mut [bool], ktop: usize, kbot: usize, nw: usize) -> (usize, Vec<(f64, f64)>) {
    let nn = h.rows;
    let eps = f64::EPSILON;
    let smlnum = f64::MIN_POSITIVE * (nn as f64 / eps);
    let jw = nw.min(kbot + 1 - ktop);
    let kwtop = kbot + 1 - jw;
    let s = if kwtop == ktop { 0.0 } else { h.at(kwtop, kwtop - 1) };
    if jw == 1 {
        if s.abs() <= smlnum.max(eps * h.at(kwtop, kwtop).abs()) {
            if kwtop > ktop {
                h.set(kwtop, kwtop - 1, 0.0);
            }
            return (1, Vec::new());
        }
        return (0, vec![(h.at(kwtop, kwtop), 0.0)]);
    }
    // the window's Schur form on a copy, its Schur vectors transposed in qt
    let mut t = Mat::from_fn(jw, jw, |i, j| if i > j + 1 { 0.0 } else { h.at(kwtop + i, kwtop + j) });
    let mut qt = Mat::identity(jw);
    let mut steps = Steps { done: 0, limit: 30 * jw.max(10) };
    if hqr_block(&mut t, &mut qt, 0, jw - 1, &mut vec![false; jw], &mut steps).is_err() {
        // (rare) nothing learned this round
        return (0, Vec::new());
    }
    // converged blocks off the bottom; the others moved up out of the way
    let (mut ns, mut ilst) = (jw, 0);
    while ilst < ns {
        let two = ns >= 2 && t.at(ns - 1, ns - 2) != 0.0;
        // (LAPACK's foo: the size of the block's eigenvalues)
        let (size, magnitude, spike) = if two {
            let magnitude = t.at(ns - 1, ns - 1).abs() + t.at(ns - 1, ns - 2).abs().sqrt() * t.at(ns - 2, ns - 1).abs().sqrt();
            (2, magnitude, (s * qt.at(ns - 1, 0)).abs().max((s * qt.at(ns - 2, 0)).abs()))
        } else {
            (1, t.at(ns - 1, ns - 1).abs(), (s * qt.at(ns - 1, 0)).abs())
        };
        let magnitude = if magnitude == 0.0 { s.abs() } else { magnitude };
        if spike <= smlnum.max(eps * magnitude) {
            ns -= size;
        } else {
            reorder::move_up(&mut t, &mut qt, ns - size, ilst);
            ilst += size;
        }
    }
    let s = if ns == 0 { 0.0 } else { s };
    let shifts = eigenvalues(&t, ns);
    if ns == jw && s != 0.0 {
        // nothing converged: the window stays as it was
        return (0, shifts);
    }
    if ns > 1 && s != 0.0 {
        // the spike reflected to a multiple of the first unit vector, then the unconverged part
        // back to Hessenberg form
        let spike: Vec<f64> = (0..ns).map(|j| s * qt.at(j, 0)).collect();
        let (tau, u) = reorder::householder(&spike);
        reorder::reflect_rows(&mut t, &u, tau, 0, 0..jw);
        reorder::reflect_cols(&mut t, &u, tau, 0, 0..ns);
        reorder::reflect_rows(&mut qt, &u, tau, 0, 0..jw);
        let (h11, q11t) = hessenberg(&Mat::from_fn(ns, ns, |i, j| t.at(i, j)));
        let t12 = q11t.mul(&Mat::from_fn(ns, jw - ns, |i, j| t.at(i, ns + j)));
        let q1 = q11t.mul(&Mat::from_fn(ns, jw, |i, j| qt.at(i, j)));
        for i in 0..ns {
            for j in 0..jw {
                t.set(i, j, if j < ns { h11.at(i, j) } else { t12.at(i, j - ns) });
                qt.set(i, j, q1.at(i, j));
            }
        }
    }
    // back into place, the converged complex pairs marked
    if kwtop > 0 {
        h.set(kwtop, kwtop - 1, s * qt.at(0, 0));
    }
    for i in 0..jw {
        for j in 0..jw {
            h.set(kwtop + i, kwtop + j, t.at(i, j));
        }
    }
    for j in ns..jw - 1 {
        if t.at(j + 1, j) != 0.0 {
            pair[kwtop + j] = true;
        }
    }
    apply_window(h, vt, &qt, kwtop, kbot);
    (jw - ns, shifts)
}

/// A window's orthogonal transformation `v` (given transposed, `qt`) applied to the rest of `h`
/// (the rows above the window and the columns right of it) and to the transposed Schur vectors
/// `vt`, as three matrix products.
fn apply_window(h: &mut Mat, vt: &mut Mat, qt: &Mat, kwtop: usize, kbot: usize) {
    use faer::linalg::matmul::matmul;
    use faer::{Accum, MatMut, MatRef};
    let (nn, jw) = (h.rows, qt.rows);
    let q = MatRef::from_row_major_slice(&qt.data, jw, jw);
    if kwtop > 0 {
        let a = Mat::from_fn(kwtop, jw, |i, j| h.at(i, kwtop + j));
        let dst = block_mut(h, 0, kwtop, kwtop, jw);
        matmul(dst, Accum::Replace, MatRef::from_row_major_slice(&a.data, kwtop, jw), q.transpose(), 1.0, super::product_parallelism(kwtop, jw, jw));
    }
    if kbot + 1 < nn {
        let c = nn - kbot - 1;
        let b = Mat::from_fn(jw, c, |i, j| h.at(kwtop + i, kbot + 1 + j));
        let dst = block_mut(h, kwtop, kbot + 1, jw, c);
        matmul(dst, Accum::Replace, q, MatRef::from_row_major_slice(&b.data, jw, c), 1.0, super::product_parallelism(jw, c, jw));
    }
    let z = Mat::from_fn(jw, vt.cols, |i, j| vt.at(kwtop + i, j));
    let cols = vt.cols;
    let dst = MatMut::from_row_major_slice_mut(&mut vt.data[kwtop * cols..(kbot + 1) * cols], jw, cols);
    matmul(dst, Accum::Replace, q, MatRef::from_row_major_slice(&z.data, jw, cols), 1.0, super::product_parallelism(jw, cols, jw));
}

/// The `rows` x `cols` block of `m` from `(r0, c0)` as a faer view. (faer 0.24's
/// `from_row_major_slice_with_stride_mut` swaps the two strides; the column-major view, transposed.)
fn block_mut(m: &mut Mat, r0: usize, c0: usize, rows: usize, cols: usize) -> faer::MatMut<'_, f64> {
    let n = m.cols;
    faer::MatMut::from_column_major_slice_with_stride_mut(&mut m.data[r0 * n + c0..], cols, rows, n).transpose_mut()
}

/// Up to `ns` of the shifts from [`aed`], the bottom ones, as `(x, y, w)` pairs for
/// [`double_step`]: complex conjugate pairs together, real shifts two by two. With fewer than two,
/// the eigenvalues of the block's trailing `ns` x `ns` corner, or failing that its trailing 2 x 2
/// one (Wilkinson's double shift).
fn shift_pairs(h: &Mat, ktop: usize, kbot: usize, shifts: &[(f64, f64)], ns: usize) -> Vec<(f64, f64, f64)> {
    let mut shifts = shifts.to_vec();
    if shifts.len() < 2 {
        let k = ns.min(kbot + 1 - ktop);
        let lo = kbot + 1 - k;
        let mut t = Mat::from_fn(k, k, |i, j| if i > j + 1 { 0.0 } else { h.at(lo + i, lo + j) });
        let mut steps = Steps { done: 0, limit: 30 * k.max(10) };
        shifts = if hqr_block(&mut t, &mut Mat::identity(k), 0, k - 1, &mut vec![false; k], &mut steps).is_ok() { eigenvalues(&t, k) } else { Vec::new() };
        if shifts.len() < 2 {
            return vec![(h.at(kbot, kbot), h.at(kbot - 1, kbot - 1), h.at(kbot, kbot - 1) * h.at(kbot - 1, kbot))];
        }
    }
    // the bottom ns, a complex pair not split
    let mut from = shifts.len().saturating_sub(ns);
    if shifts[from].1 < 0.0 {
        from += 1;
    }
    let (mut reals, mut pairs) = (Vec::new(), Vec::new());
    for &(re, im) in &shifts[from..] {
        if im > 0.0 {
            pairs.push((re, re, -im * im));
        } else if im == 0.0 {
            reals.push(re);
        }
    }
    for two in reals.chunks(2) {
        pairs.push((two[0], *two.get(1).unwrap_or(&two[0]), 0.0));
    }
    pairs
}

/// LAPACK's exceptional shifts for a round that has stalled: `ns / 2` ad hoc pairs from the
/// subdiagonal entries near the bottom of the block.
fn exceptional_shifts(h: &Mat, ktop: usize, kbot: usize, ns: usize) -> Vec<(f64, f64, f64)> {
    (0..ns / 2)
        .map(|k| kbot - 2 * k)
        .take_while(|&i| i >= ktop + 2)
        .map(|i| {
            let ss = h.at(i, i - 1).abs() + h.at(i - 1, i - 2).abs();
            let aa = 0.75 * ss + h.at(i, i);
            (aa, aa, -0.4375 * ss * ss)
        })
        .collect()
}

/// The real Schur decomposition `a = z t zᵀ` (`scipy.linalg.schur`): `z` orthogonal, `t` upper
/// triangular but for 2 x 2 blocks on the diagonal, one per complex eigenvalue pair. Returns
/// `(t, z)`. Computed in f64.
pub fn schur<T: LinalgFloat>(a: NdView<'_, T>) -> Result<(NdArray<T>, NdArray<T>), LinalgError> {
    let s = real_schur(&Mat::square(a)?)?;
    Ok((s.t.to_array(), s.z.to_array()))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::testing::noise_f64;

    /// `a = z t zᵀ` to rounding, `z` orthogonal, `t` quasi-triangular with only complex pairs in
    /// its 2 x 2 blocks.
    fn check(a: &Mat, what: &str) {
        let n = a.rows;
        let s = real_schur(a).unwrap_or_else(|e| panic!("{what}: {e}"));
        // (relative to the largest entry, so that huge matrices' norms do not overflow)
        let big = a.data.iter().fold(f64::MIN_POSITIVE, |m, v| m.max(v.abs()));
        let res = s.z.mul(&s.t.scale(1.0 / big)).mul(&s.z.t()).sub(&a.scale(1.0 / big)).norm() / a.scale(1.0 / big).norm().max(f64::MIN_POSITIVE);
        let orth = s.z.t().mul(&s.z).sub(&Mat::identity(n)).norm();
        let tol = 1e-14 * (n as f64).max(4.0);
        assert!(res < tol && orth < tol, "{what}: residual {res:e}, orthogonality {orth:e}");
        for i in 1..n {
            for j in 0..i - 1 {
                assert_eq!(s.t.at(i, j), 0.0, "{what}: t[{i}][{j}]");
            }
            if s.t.at(i, i - 1) != 0.0 {
                assert!(i + 1 == n || s.t.at(i + 1, i) == 0.0, "{what}: 2 x 2 blocks overlap at {i}");
                let (a, b, c, d) = (s.t.at(i - 1, i - 1) / big, s.t.at(i - 1, i) / big, s.t.at(i, i - 1) / big, s.t.at(i, i) / big);
                assert!(((a - d) / 2.0).powi(2) + b * c < 0.0, "{what}: a 2 x 2 block with real eigenvalues at {i}");
            }
        }
    }

    fn random(n: usize, seed: u64) -> Mat {
        let r = noise_f64(n * n, seed);
        Mat::from_fn(n, n, |i, j| r[i * n + j])
    }

    #[test]
    fn hard_matrices_on_every_path() {
        // either side of the small Hessenberg reduction (64), finishing blocks whole (75) and
        // aggressive early deflation (200)
        for n in [1, 2, 3, 7, 63, 64, 76, 200, 230] {
            let a = random(n, n as u64);
            check(&a, &format!("random {n}"));
            check(&a.add(&a.t()), &format!("symmetric {n}"));
            // graded: entries spanning twenty orders of magnitude
            let g = Mat::from_fn(n, n, |i, j| a.at(i, j) * 10f64.powf(10.0 * (j as f64 - i as f64) / n as f64));
            check(&g, &format!("graded {n}"));
            // a companion matrix (roots spread around a circle: complex pairs throughout)
            let c = Mat::from_fn(n, n, |i, j| if i == 0 { a.at(0, j) } else if i == j + 1 { 1.0 } else { 0.0 });
            check(&c, &format!("companion {n}"));
            // a Jordan block, slightly perturbed (eigenvalues that are very sensitive)
            let jb = Mat::from_fn(n, n, |i, j| if i == j { 2.0 } else if j == i + 1 { 1.0 } else { 1e-14 * a.at(i, j) });
            check(&jb, &format!("jordan {n}"));
            // rotations: every eigenvalue in a complex pair of modulus one
            let rot = Mat::from_fn(n, n, |i, j| match (i / 2 == j / 2, i % 2, j % 2) {
                (true, 0, 1) if i + 1 < n => -0.6,
                (true, 1, 0) => 0.6,
                (true, _, _) if i == j => 0.8,
                _ => 0.0,
            });
            check(&rot.add(&a.scale(1e-3)), &format!("rotations {n}"));
            check(&Mat::zeros(n, n), &format!("zero {n}"));
            check(&Mat::from_fn(n, n, |i, j| if j >= i { a.at(i, j) } else { 0.0 }), &format!("triangular {n}"));
            check(&a.scale(1e250), &format!("huge {n}"));
            check(&a.scale(1e-250), &format!("tiny {n}"));
        }
    }

    #[test]
    fn window_updates_write_only_their_blocks() {
        // (faer 0.24's from_row_major_slice_with_stride_mut swaps its strides; apply_window must
        // not depend on it)
        let n = 76;
        let mut h = random(n, 3);
        let h0 = h.clone();
        let mut vt = Mat::identity(n);
        apply_window(&mut h, &mut vt, &Mat::identity(69), 0, 68);
        apply_window(&mut h, &mut vt, &Mat::identity(10), 30, 39);
        assert_eq!(h, h0);
        assert_eq!(vt, Mat::identity(n));
    }
}