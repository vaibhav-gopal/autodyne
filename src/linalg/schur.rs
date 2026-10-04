//! The real Schur decomposition (`scipy.linalg.schur`, `output="real"`): `a = z t zᵀ` with `z`
//! orthogonal and `t` quasi-upper-triangular (1 x 1 blocks for real eigenvalues, 2 x 2 blocks for
//! complex pairs). Householder reduction to Hessenberg form, then Francis double-shift QR with the
//! transformations accumulated (EISPACK's `orthes` and the Schur phase of `hqr2`, by way of JAMA).

use super::dense::Mat;
use super::{LinalgError, LinalgFloat};
use crate::signal::{NdArray, NdView};

/// A real Schur form: `a = z t zᵀ`.
pub(crate) struct Schur {
    pub(crate) t: Mat,
    pub(crate) z: Mat,
}

/// Reduces `h` to upper Hessenberg form in place, returning the orthogonal `v` with
/// `a = v h vᵀ` (EISPACK `orthes` + `ortran`).
// indexed loops, as in EISPACK, read more clearly here than iterator chains
#[allow(clippy::needless_range_loop)]
fn hessenberg(h: &mut Mat) -> Mat {
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
        // H = (I - u uᵀ / hh) H (I - u uᵀ / hh), both sides a row at a time (the matrix is row-major)
        let u = &ort[m..=high];
        let mut f = vec![0.0; n - m];
        for (o, i) in u.iter().zip(m..=high) {
            for (fj, &hij) in f.iter_mut().zip(&h.data[i * n + m..(i + 1) * n]) {
                *fj += o * hij;
            }
        }
        f.iter_mut().for_each(|v| *v /= hh);
        for (o, i) in u.iter().zip(m..=high) {
            for (hij, fj) in h.data[i * n + m..(i + 1) * n].iter_mut().zip(&f) {
                *hij -= fj * o;
            }
        }
        for i in 0..=high {
            let row = &mut h.data[i * n + m..i * n + high + 1];
            let g = row.iter().zip(u).map(|(a, b)| a * b).sum::<f64>() / hh;
            row.iter_mut().zip(u).for_each(|(hij, o)| *hij -= g * o);
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
/// (30 n iterations).
pub(crate) fn real_schur(a: &Mat) -> Result<Schur, LinalgError> {
    #[cfg(target_arch = "x86_64")]
    if crate::simd::avx2_available() {
        // SAFETY: the CPU was just checked for AVX2, the only feature eal_schur_avx2 is compiled with.
        return unsafe { real_schur_avx2(a) };
    }
    real_schur_body(a)
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
unsafe fn real_schur_avx2(a: &Mat) -> Result<Schur, LinalgError> {
    real_schur_body(a)
}

/// [`real_schur`]'s work, inlined into each instruction-set version.
#[inline(always)]
#[allow(clippy::needless_range_loop)]
fn real_schur_body(a: &Mat) -> Result<Schur, LinalgError> {
    let nn = a.rows;
    let mut h = a.clone();
    // the Schur vectors kept transposed, so their updates touch rows instead of columns
    let mut vt = hessenberg(&mut h).t();
    if nn == 0 {
        return Ok(Schur { t: h, z: vt });
    }
    let eps = f64::EPSILON;
    let mut exshift = 0.0;
    let (mut p, mut q, mut r, mut s, mut z);
    let (mut w, mut x, mut y);
    let norm: f64 = (0..nn).map(|i| (i.saturating_sub(1)..nn).map(|j| h.at(i, j).abs()).sum::<f64>()).sum();
    // complex pairs found, by their upper index (their 2 x 2 blocks stay)
    let mut pair = vec![false; nn];
    let mut n = nn as isize - 1;
    let low = 0isize;
    let mut iter = 0;
    let mut total = 0;
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
            if h.at(lu, lu - 1).abs() < eps * s {
                break;
            }
            l -= 1;
        }
        if l == n {
            // one root found
            h.set(nu, nu, h.at(nu, nu) + exshift);
            if nu > 0 {
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
                p = x / s;
                q = z / s;
                r = (p * p + q * q).sqrt();
                p /= r;
                q /= r;
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
                for i in 0..nn {
                    let zz = vt.at(nu - 1, i);
                    vt.set(nu - 1, i, q * zz + p * vt.at(nu, i));
                    vt.set(nu, i, q * vt.at(nu, i) - p * zz);
                }
                h.set(nu, nu - 1, 0.0);
            } else {
                pair[nu - 1] = true;
            }
            if nu >= 2 {
                h.set(nu - 1, nu - 2, 0.0);
            }
            n -= 2;
            iter = 0;
        } else {
            // no convergence yet: form the shift
            x = h.at(nu, nu);
            y = 0.0;
            w = 0.0;
            if l < n {
                y = h.at(nu - 1, nu - 1);
                w = h.at(nu, nu - 1) * h.at(nu - 1, nu);
            }
            // Wilkinson's original ad hoc shift
            if iter == 10 {
                exshift += x;
                for i in low as usize..=nu {
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
                    for i in low as usize..=nu {
                        h.set(i, i, h.at(i, i) - s);
                    }
                    exshift += s;
                    x = 0.964;
                    y = x;
                    w = x;
                }
            }
            iter += 1;
            total += 1;
            if total > 30 * nn.max(10) {
                return Err(LinalgError::NoConvergence);
            }
            // look for two consecutive small subdiagonal elements
            let mut m = n - 2;
            loop {
                let mu = m as usize;
                z = h.at(mu, mu);
                r = x - z;
                s = y - z;
                p = (r * s - w) / h.at(mu + 1, mu) + h.at(mu, mu + 1);
                q = h.at(mu + 1, mu + 1) - z - r - s;
                r = h.at(mu + 2, mu + 1);
                s = p.abs() + q.abs() + r.abs();
                p /= s;
                q /= s;
                r /= s;
                if m == l {
                    break;
                }
                if h.at(mu, mu - 1).abs() * (q.abs() + r.abs()) < eps * (p.abs() * (h.at(mu - 1, mu - 1).abs() + z.abs() + h.at(mu + 1, mu + 1).abs())) {
                    break;
                }
                m -= 1;
            }
            let mu = m as usize;
            for i in mu + 2..=nu {
                h.set(i, i - 2, 0.0);
                if i > mu + 2 {
                    h.set(i, i - 3, 0.0);
                }
            }
            // a double QR step on rows l..n and columns m..n
            for k in mu..nu {
                let notlast = k != nu - 1;
                if k != mu {
                    p = h.at(k, k - 1);
                    q = h.at(k + 1, k - 1);
                    r = if notlast { h.at(k + 2, k - 1) } else { 0.0 };
                    x = p.abs() + q.abs() + r.abs();
                    if x == 0.0 {
                        continue;
                    }
                    p /= x;
                    q /= x;
                    r /= x;
                }
                s = (p * p + q * q + r * r).sqrt();
                if p < 0.0 {
                    s = -s;
                }
                if s != 0.0 {
                    if k != mu {
                        h.set(k, k - 1, -s * x);
                    } else if l != m {
                        h.set(k, k - 1, -h.at(k, k - 1));
                    }
                    p += s;
                    x = p / s;
                    y = q / s;
                    z = r / s;
                    q /= p;
                    r /= p;
                    let reflector = Reflector { x, y, z, q, r, three: notlast };
                    // from the left on rows k.. of h (columns k..), from the right on its columns
                    // k.. (rows up to k + 3) and on the Schur vectors (rows of vt)
                    reflector.left(&mut h.data, nn, k, k);
                    for i in 0..=nu.min(k + 3) {
                        reflector.right(&mut h.data[i * nn + k..i * nn + k + if notlast { 3 } else { 2 }]);
                    }
                    reflector.right_rows(&mut vt.data, nn, k);
                }
            }
        }
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

/// The real Schur decomposition `a = z t zᵀ` (`scipy.linalg.schur`): `z` orthogonal, `t` upper
/// triangular but for 2 x 2 blocks on the diagonal, one per complex eigenvalue pair. Returns
/// `(t, z)`. Computed in f64.
pub fn schur<T: LinalgFloat>(a: NdView<'_, T>) -> Result<(NdArray<T>, NdArray<T>), LinalgError> {
    let s = real_schur(&Mat::square(a)?)?;
    Ok((s.t.to_array(), s.z.to_array()))
}

