//! Reordering a real Schur form: adjacent 1 x 1 and 2 x 2 diagonal blocks swapped by orthogonal
//! similarity (LAPACK's `dlaexc` and `dtrexc`, with `dlasy2`, `dlanv2`, `dlarfg` and `dlartg`), for
//! the aggressive early deflation of [`schur`](super::schur). Matrices are row-major [`Mat`]s; the
//! Schur vectors are kept transposed (`qt`), so updating a column of them updates a row of `qt`.
//!
//! tend: Numerics / linalg / schur

use std::ops::Range;

use super::dense::Mat;

const EPS: f64 = f64::EPSILON;
const SMLNUM: f64 = f64::MIN_POSITIVE / f64::EPSILON;

/// The Householder reflector `I - tau u uᵀ` (`u[0] = 1`) taking `x` to a multiple of the first
/// unit vector (`dlarfg`); `tau = 0` (the identity) if `x` already is one.
pub(super) fn householder(x: &[f64]) -> (f64, Vec<f64>) {
    let alpha = x[0];
    let xnorm = x[1..].iter().fold(0.0f64, |a, &v| a.hypot(v));
    let mut u = vec![0.0; x.len()];
    u[0] = 1.0;
    if xnorm == 0.0 {
        return (0.0, u);
    }
    let beta = -alpha.hypot(xnorm).copysign(alpha);
    let scale = 1.0 / (alpha - beta);
    for (ui, &xi) in u[1..].iter_mut().zip(&x[1..]) {
        *ui = xi * scale;
    }
    ((beta - alpha) / beta, u)
}

/// Rows `r0..r0 + u.len()` of `m`, in columns `cols`, multiplied from the left by `I - tau u uᵀ`.
pub(super) fn reflect_rows(m: &mut Mat, u: &[f64], tau: f64, r0: usize, cols: Range<usize>) {
    if tau == 0.0 {
        return;
    }
    let n = m.cols;
    let mut w = vec![0.0; cols.len()];
    for (k, &uk) in u.iter().enumerate() {
        let row = &m.data[(r0 + k) * n + cols.start..(r0 + k) * n + cols.end];
        w.iter_mut().zip(row).for_each(|(w, &v)| *w += uk * v);
    }
    for (k, &uk) in u.iter().enumerate() {
        let f = tau * uk;
        let row = &mut m.data[(r0 + k) * n + cols.start..(r0 + k) * n + cols.end];
        row.iter_mut().zip(&w).for_each(|(v, &w)| *v -= f * w);
    }
}

/// Columns `c0..c0 + u.len()` of `m`, in rows `rows`, multiplied from the right by `I - tau u uᵀ`.
pub(super) fn reflect_cols(m: &mut Mat, u: &[f64], tau: f64, c0: usize, rows: Range<usize>) {
    if tau == 0.0 {
        return;
    }
    let n = m.cols;
    for i in rows {
        let row = &mut m.data[i * n + c0..i * n + c0 + u.len()];
        let s = tau * row.iter().zip(u).map(|(a, b)| a * b).sum::<f64>();
        row.iter_mut().zip(u).for_each(|(v, &uk)| *v -= s * uk);
    }
}

/// Rows `i` and `j` of `m`, in columns `cols`, rotated: `(x, y) <- (c x + s y, c y - s x)` (`drot`).
fn rot_rows(m: &mut Mat, i: usize, j: usize, cols: Range<usize>, c: f64, s: f64) {
    for k in cols {
        let (x, y) = (m.at(i, k), m.at(j, k));
        m.set(i, k, c * x + s * y);
        m.set(j, k, c * y - s * x);
    }
}

/// Columns `i` and `j` of `m`, in rows `rows`, rotated as [`rot_rows`] rotates rows.
fn rot_cols(m: &mut Mat, i: usize, j: usize, rows: Range<usize>, c: f64, s: f64) {
    for k in rows {
        let (x, y) = (m.at(k, i), m.at(k, j));
        m.set(k, i, c * x + s * y);
        m.set(k, j, c * y - s * x);
    }
}

/// A rotation `(c, s)` with `-s f + c g = 0` (`dlartg`).
fn givens(f: f64, g: f64) -> (f64, f64) {
    if g == 0.0 {
        (1.0, 0.0)
    } else if f == 0.0 {
        (0.0, 1.0)
    } else {
        let r = f.hypot(g).copysign(f);
        (f / r, g / r)
    }
}

/// The standard form of the 2 x 2 block `[[a, b], [c, d]]` (`dlanv2`): upper triangular if its
/// eigenvalues are real, else equal diagonal entries and off-diagonal ones of opposite signs.
/// Returns the new block and the rotation `(cs, sn)` that makes it.
fn standardize(mut a: f64, mut b: f64, mut c: f64, mut d: f64) -> ([f64; 4], f64, f64) {
    const MULTPL: f64 = 4.0;
    let (mut cs, mut sn);
    if c == 0.0 {
        (cs, sn) = (1.0, 0.0);
    } else if b == 0.0 {
        // swap rows and columns
        (cs, sn) = (0.0, 1.0);
        (a, d) = (d, a);
        b = -c;
        c = 0.0;
    } else if a - d == 0.0 && b.signum() != c.signum() {
        (cs, sn) = (1.0, 0.0);
    } else {
        let temp = a - d;
        let mut p = 0.5 * temp;
        let bcmax = b.abs().max(c.abs());
        let bcmis = b.abs().min(c.abs()) * b.signum() * c.signum();
        let scale = p.abs().max(bcmax);
        let mut z = (p / scale) * p + (bcmax / scale) * bcmis;
        if z >= MULTPL * EPS {
            // real eigenvalues: a and d, triangular
            z = p + (scale.sqrt() * z.sqrt()).copysign(p);
            a = d + z;
            d -= (bcmax / z) * bcmis;
            let tau = c.hypot(z);
            (cs, sn) = (z / tau, c / tau);
            b -= c;
            c = 0.0;
        } else {
            // complex or nearly equal real eigenvalues: equal diagonal entries
            let sigma = b + c;
            let tau = sigma.hypot(temp);
            cs = (0.5 * (1.0 + sigma.abs() / tau)).sqrt();
            sn = -(p / (tau * cs)) * sigma.signum();
            let (aa, bb) = (a * cs + b * sn, -a * sn + b * cs);
            let (cc, dd) = (c * cs + d * sn, -c * sn + d * cs);
            a = aa * cs + cc * sn;
            b = bb * cs + dd * sn;
            c = -aa * sn + cc * cs;
            d = -bb * sn + dd * cs;
            let mid = 0.5 * (a + d);
            (a, d) = (mid, mid);
            if c != 0.0 {
                if b != 0.0 {
                    if b.signum() == c.signum() {
                        // real eigenvalues after all: triangular
                        let (sab, sac) = (b.abs().sqrt(), c.abs().sqrt());
                        p = (sab * sac).copysign(c);
                        let tau = 1.0 / (b + c).abs().sqrt();
                        a = mid + p;
                        d = mid - p;
                        b -= c;
                        c = 0.0;
                        let (cs1, sn1) = (sab * tau, sac * tau);
                        (cs, sn) = (cs * cs1 - sn * sn1, cs * sn1 + sn * cs1);
                    }
                } else {
                    b = -c;
                    c = 0.0;
                    (cs, sn) = (-sn, cs);
                }
            }
        }
    }
    ([a, b, c, d], cs, sn)
}

/// `X` with `T11 X - X T22 = T12` for the diagonal blocks `T11` (`n1` x `n1`) and `T22` (`n2` x
/// `n2`) and the block `T12` of `d` (`dlasy2`): the Kronecker system, by Gaussian elimination with
/// complete pivoting, tiny pivots raised to `eps` times the blocks' size.
fn sylvester(d: &[[f64; 4]; 4], n1: usize, n2: usize) -> [[f64; 2]; 2] {
    let m = n1 * n2;
    let (mut k, mut rhs) = ([[0.0f64; 4]; 4], [0.0f64; 4]);
    for j in 0..n2 {
        for i in 0..n1 {
            let row = i + j * n1;
            rhs[row] = d[i][n1 + j];
            for l in 0..n2 {
                for p in 0..n1 {
                    let mut v = 0.0;
                    if j == l {
                        v += d[i][p];
                    }
                    if i == p {
                        v -= d[n1 + l][n1 + j];
                    }
                    k[row][p + l * n1] = v;
                }
            }
        }
    }
    let big = (0..n1 + n2).flat_map(|i| (0..n1 + n2).map(move |j| (i, j))).filter(|&(i, j)| (i < n1) == (j < n1)).map(|(i, j)| d[i][j].abs()).fold(0.0, f64::max);
    let smin = (EPS * big).max(SMLNUM);
    let mut perm = [0, 1, 2, 3];
    for s in 0..m {
        let (mut pr, mut pc) = (s, s);
        for r in s..m {
            for c in s..m {
                if k[r][c].abs() > k[pr][pc].abs() {
                    (pr, pc) = (r, c);
                }
            }
        }
        k.swap(s, pr);
        rhs.swap(s, pr);
        for row in k.iter_mut() {
            row.swap(s, pc);
        }
        perm.swap(s, pc);
        if k[s][s].abs() < smin {
            k[s][s] = smin;
        }
        let pivot = k[s];
        for r in s + 1..m {
            let f = k[r][s] / pivot[s];
            k[r][s..m].iter_mut().zip(&pivot[s..m]).for_each(|(v, p)| *v -= f * p);
            rhs[r] -= f * rhs[s];
        }
    }
    let mut y = [0.0f64; 4];
    for s in (0..m).rev() {
        let tail: f64 = (s + 1..m).map(|c| k[s][c] * y[c]).sum();
        y[s] = (rhs[s] - tail) / k[s][s];
    }
    let mut x = [[0.0f64; 2]; 2];
    for s in 0..m {
        let v = perm[s];
        x[v % n1][v / n1] = y[s];
    }
    x
}

/// `I - tau u uᵀ` on rows `r0..r0 + 3` (columns `cols`) of a 4 x 4 array.
fn reflect_rows4(d: &mut [[f64; 4]; 4], u: &[f64], tau: f64, r0: usize, cols: Range<usize>) {
    for c in cols {
        let s = tau * (0..3).map(|k| u[k] * d[r0 + k][c]).sum::<f64>();
        (0..3).for_each(|k| d[r0 + k][c] -= s * u[k]);
    }
}

/// `I - tau u uᵀ` on columns `c0..c0 + 3` (rows `rows`) of a 4 x 4 array.
fn reflect_cols4(d: &mut [[f64; 4]; 4], u: &[f64], tau: f64, c0: usize, rows: Range<usize>) {
    for r in rows {
        let s = tau * (0..3).map(|k| u[k] * d[r][c0 + k]).sum::<f64>();
        (0..3).for_each(|k| d[r][c0 + k] -= s * u[k]);
    }
}

/// Swaps the adjacent diagonal blocks of `t` at `j1` (`n1` x `n1`) and `j1 + n1` (`n2` x `n2`)
/// (`dlaexc`), updating the transposed Schur vectors `qt`. Swaps involving a 2 x 2 block are
/// rejected (`false`, nothing changed) if they would perturb the blocks by more than ten ulps of
/// their size.
fn swap(t: &mut Mat, qt: &mut Mat, j1: usize, n1: usize, n2: usize) -> bool {
    let (n, nq) = (t.rows, qt.cols);
    let (j2, j3, j4) = (j1 + 1, j1 + 2, j1 + 3);
    if n1 == 1 && n2 == 1 {
        let (t11, t22) = (t.at(j1, j1), t.at(j2, j2));
        let (cs, sn) = givens(t.at(j1, j2), t22 - t11);
        rot_rows(t, j1, j2, j3.min(n)..n, cs, sn);
        rot_cols(t, j1, j2, 0..j1, cs, sn);
        t.set(j1, j1, t22);
        t.set(j2, j2, t11);
        rot_rows(qt, j1, j2, 0..nq, cs, sn);
        return true;
    }
    let nd = n1 + n2;
    let mut d = [[0.0f64; 4]; 4];
    for (i, row) in d.iter_mut().enumerate().take(nd) {
        for (j, v) in row.iter_mut().enumerate().take(nd) {
            *v = t.at(j1 + i, j1 + j);
        }
    }
    let dnorm = d.iter().flatten().fold(0.0f64, |m, v| m.max(v.abs()));
    let thresh = (10.0 * EPS * dnorm).max(SMLNUM);
    let x = sylvester(&d, n1, n2);
    match (n1, n2) {
        (1, 2) => {
            // ( 1, x11, x12 ) H = ( 0, 0, * )
            let (tau, w) = householder(&[x[0][1], 1.0, x[0][0]]);
            let u = [w[1], w[2], 1.0];
            let t11 = t.at(j1, j1);
            reflect_rows4(&mut d, &u, tau, 0, 0..3);
            reflect_cols4(&mut d, &u, tau, 0, 0..3);
            if d[2][0].abs().max(d[2][1].abs()).max((d[2][2] - t11).abs()) > thresh {
                return false;
            }
            reflect_rows(t, &u, tau, j1, j1..n);
            reflect_cols(t, &u, tau, j1, 0..j3);
            t.set(j3, j1, 0.0);
            t.set(j3, j2, 0.0);
            t.set(j3, j3, t11);
            reflect_rows(qt, &u, tau, j1, 0..nq);
        }
        (2, 1) => {
            // H ( -x11, -x21, 1 ) = ( *, 0, 0 )
            let (tau, u) = householder(&[-x[0][0], -x[1][0], 1.0]);
            let t33 = t.at(j3, j3);
            reflect_rows4(&mut d, &u, tau, 0, 0..3);
            reflect_cols4(&mut d, &u, tau, 0, 0..3);
            if d[1][0].abs().max(d[2][0].abs()).max((d[0][0] - t33).abs()) > thresh {
                return false;
            }
            reflect_cols(t, &u, tau, j1, 0..j4);
            reflect_rows(t, &u, tau, j1, j2..n);
            t.set(j1, j1, t33);
            t.set(j2, j1, 0.0);
            t.set(j3, j1, 0.0);
            reflect_rows(qt, &u, tau, j1, 0..nq);
        }
        _ => {
            // H2 H1 ( -X; I ) upper triangular
            let (tau1, u1) = householder(&[-x[0][0], -x[1][0], 1.0]);
            let temp = -tau1 * (x[0][1] + u1[1] * x[1][1]);
            let (tau2, u2) = householder(&[-temp * u1[1] - x[1][1], -temp * u1[2], 1.0]);
            reflect_rows4(&mut d, &u1, tau1, 0, 0..4);
            reflect_cols4(&mut d, &u1, tau1, 0, 0..4);
            reflect_rows4(&mut d, &u2, tau2, 1, 0..4);
            reflect_cols4(&mut d, &u2, tau2, 1, 0..4);
            if d[2][0].abs().max(d[2][1].abs()).max(d[3][0].abs()).max(d[3][1].abs()) > thresh {
                return false;
            }
            reflect_rows(t, &u1, tau1, j1, j1..n);
            reflect_cols(t, &u1, tau1, j1, 0..j4 + 1);
            reflect_rows(t, &u2, tau2, j2, j1..n);
            reflect_cols(t, &u2, tau2, j2, 0..j4 + 1);
            for (i, j) in [(j3, j1), (j3, j2), (j4, j1), (j4, j2)] {
                t.set(i, j, 0.0);
            }
            reflect_rows(qt, &u1, tau1, j1, 0..nq);
            reflect_rows(qt, &u2, tau2, j2, 0..nq);
        }
    }
    // the 2 x 2 blocks in standard form
    for (size, at) in [(n2, j1), (n1, j1 + n2)] {
        if size == 2 {
            let ([a, b, c, dd], cs, sn) = standardize(t.at(at, at), t.at(at, at + 1), t.at(at + 1, at), t.at(at + 1, at + 1));
            t.set(at, at, a);
            t.set(at, at + 1, b);
            t.set(at + 1, at, c);
            t.set(at + 1, at + 1, dd);
            rot_rows(t, at, at + 1, (at + 2).min(n)..n, cs, sn);
            rot_cols(t, at, at + 1, 0..at, cs, sn);
            rot_rows(qt, at, at + 1, 0..nq, cs, sn);
        }
    }
    true
}

/// Moves the diagonal block of `t` starting at row `ifst` up to start at row `ilst` (`dtrexc`,
/// `ifst > ilst`), one swap at a time, stopping early where a swap is rejected; `qt` follows.
pub(super) fn move_up(t: &mut Mat, qt: &mut Mat, mut ifst: usize, mut ilst: usize) {
    let n = t.rows;
    if ifst > 0 && t.at(ifst, ifst - 1) != 0.0 {
        ifst -= 1;
    }
    if ilst > 0 && t.at(ilst, ilst - 1) != 0.0 {
        ilst -= 1;
    }
    // 1, 2, or 3 for a 2 x 2 block split into two 1 x 1 ones on the way
    let mut nbf = if ifst + 1 < n && t.at(ifst + 1, ifst) != 0.0 { 2 } else { 1 };
    let mut here = ifst;
    while here > ilst {
        let nbnext = if here >= 2 && t.at(here - 1, here - 2) != 0.0 { 2 } else { 1 };
        if nbf != 3 {
            if !swap(t, qt, here - nbnext, nbnext, nbf) {
                return;
            }
            here -= nbnext;
            if nbf == 2 && t.at(here + 1, here) == 0.0 {
                nbf = 3;
            }
        } else {
            // the two 1 x 1 blocks one at a time
            if !swap(t, qt, here - nbnext, nbnext, 1) {
                return;
            }
            if nbnext == 1 {
                swap(t, qt, here, 1, 1);
                here -= 1;
            } else if t.at(here, here - 1) != 0.0 {
                // the 2 x 2 block above stayed whole
                if !swap(t, qt, here - 1, 2, 1) {
                    return;
                }
                here -= 2;
            } else {
                swap(t, qt, here, 1, 1);
                swap(t, qt, here - 1, 1, 1);
                here -= 2;
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn random(n: usize, seed: u64) -> Vec<f64> {
        crate::testing::noise_f64(n, seed)
    }

    /// A random quasi-triangular matrix with the given block sizes (2 x 2 blocks with complex
    /// eigenvalues).
    fn quasi(blocks: &[usize], seed: u64) -> Mat {
        let n: usize = blocks.iter().sum();
        let r = random(n * n, seed);
        let mut t = Mat::from_fn(n, n, |i, j| if j >= i { r[i * n + j] } else { 0.0 });
        let mut at = 0;
        for &b in blocks {
            if b == 2 {
                t.set(at, at + 1, 1.0 + t.at(at, at + 1).abs());
                t.set(at + 1, at, -0.5 - t.at(at + 1, at + 1).abs());
            }
            at += b;
        }
        t
    }

    fn eigenvalues(t: &Mat, at: usize, size: usize) -> (f64, f64) {
        if size == 1 {
            (t.at(at, at), 0.0)
        } else {
            let (a, b, c, d) = (t.at(at, at), t.at(at, at + 1), t.at(at + 1, at), t.at(at + 1, at + 1));
            ((a + d) / 2.0, (((a - d) / 2.0).powi(2) + b * c).abs().sqrt())
        }
    }

    #[test]
    fn swaps_keep_the_similarity_and_move_the_eigenvalues() {
        for (blocks, seed) in [(vec![1, 1, 1], 1), (vec![1, 2, 1], 2), (vec![2, 1, 1], 3), (vec![1, 1, 2, 2], 4), (vec![2, 2, 1, 2], 5), (vec![1, 2, 2, 1, 1, 2], 6)] {
            let t0 = quasi(&blocks, seed);
            let n = t0.rows;
            let (mut t, mut qt) = (t0.clone(), Mat::identity(n));
            // the last block to the top
            let last = *blocks.last().unwrap();
            let want = eigenvalues(&t0, n - last, last);
            move_up(&mut t, &mut qt, n - last, 0);
            let got = eigenvalues(&t, 0, last);
            assert!((got.0 - want.0).abs() < 1e-12 && (got.1 - want.1).abs() < 1e-12, "{blocks:?}: {got:?} vs {want:?}");
            // t0 = q t qᵀ, q orthogonal, t quasi-triangular
            let back = qt.t().mul(&t).mul(&qt);
            assert!(back.sub(&t0).norm() < 1e-12 * t0.norm(), "{blocks:?}: not similar");
            assert!(qt.mul(&qt.t()).sub(&Mat::identity(n)).norm() < 1e-13, "{blocks:?}: not orthogonal");
            for i in 2..n {
                for j in 0..i - 1 {
                    assert_eq!(t.at(i, j), 0.0, "{blocks:?}: ({i}, {j})");
                }
            }
        }
    }

    #[test]
    fn standard_form_of_two_by_two_blocks() {
        for [a, b, c, d] in [[1.0, 2.0, -3.0, 4.0], [1.0, 2.0, 3.0, 4.0], [2.0, 1.0, -1.0, 2.0], [1.0, 0.0, 5.0, 3.0], [0.0, 1e-20, -1e-20, 0.0]] {
            let ([na, nb, nc, nd], cs, sn) = standardize(a, b, c, d);
            // the same eigenvalues: trace and determinant
            assert!((na + nd - (a + d)).abs() < 1e-12 * (1.0 + (a + d).abs()));
            assert!((na * nd - nb * nc - (a * d - b * c)).abs() < 1e-12 * (1.0 + (a * d - b * c).abs()));
            assert!(nc == 0.0 || (na == nd && nb.signum() != nc.signum()), "{a} {b} {c} {d}: {na} {nb} {nc} {nd}");
            assert!((cs * cs + sn * sn - 1.0).abs() < 1e-14);
        }
    }
}
