//! Singular values and vectors of an upper bidiagonal matrix by divide and conquer (LAPACK's
//! `dbdsdc` / `dlasd0`-`dlasd4`): the matrix splits at a middle row into two bidiagonal halves (the
//! upper one with one extra column) and that row; each half is solved recursively, and in the
//! halves' bases the whole is a "broken arrow" `M` (the middle row `z` on top of a diagonal `D`,
//! the halves' singular values, with a 0 for the combined null direction), whose SVD comes from the
//! secular equation of `M^T M = D^2 + z z^T`: the same as the tridiagonal one with poles `d_j^2`,
//! so the same root finder serves, the pole differences computed as `(d_j - d_o)(d_j + d_o)`.
//! Deflation, Gu & Eisenstat's recomputed `z` and the products back to the full space follow
//! `dlasd2` / `dlasd3`. Small blocks use faer's dense SVD.

use faer::{Accum, Mat, MatMut, MatRef, Par};

use super::dc::secular;
use super::values::NoConvergence;

/// Blocks with at most this many rows are solved densely.
const SMALL: usize = 8;
const EPS: f64 = f64::EPSILON / 2.0;

/// A solved block: `B = U diag(s) [I 0] V^T`, `U` n x n and `V` m x m column-major (m = n + sqre;
/// with sqre, V's last column spans B's null space).
struct Solved {
    s: Vec<f64>,
    u: Vec<f64>,
    v: Vec<f64>,
}

/// The SVD of the n x n upper bidiagonal matrix with diagonal `d` and superdiagonal `e` (n - 1):
/// singular values descending into `d`, left vectors into `u` and right vectors into `v` (both
/// n x n, column-major, one vector per column), so `B = U diag(d) V^T`.
pub(crate) fn bidiagonal_svd(d: &mut [f64], e: &[f64], u: &mut [f64], v: &mut [f64]) -> Result<(), NoConvergence> {
    let n = d.len();
    assert!(e.len() + 1 == n.max(1) && u.len() == n * n && v.len() == n * n, "bidiagonal_svd: n diagonal, n - 1 superdiagonal, n x n vectors");
    if n == 0 {
        return Ok(());
    }
    // scaled so the largest entry is 1
    let scale = d.iter().chain(e).fold(0.0f64, |m, x| m.max(x.abs()));
    if !scale.is_finite() {
        return Err(NoConvergence);
    }
    let solved = if scale == 0.0 {
        let eye: Vec<f64> = (0..n * n).map(|k| if k % (n + 1) == 0 { 1.0 } else { 0.0 }).collect();
        Solved { s: vec![0.0; n], u: eye.clone(), v: eye }
    } else {
        let ds: Vec<f64> = d.iter().map(|x| x / scale).collect();
        let es: Vec<f64> = e.iter().map(|x| x / scale).collect();
        solve(&ds, &es, 0)?
    };
    // descending, vectors along
    let mut order: Vec<usize> = (0..n).collect();
    order.sort_by(|&a, &b| solved.s[b].total_cmp(&solved.s[a]));
    for (j, &o) in order.iter().enumerate() {
        d[j] = solved.s[o] * scale;
        u[j * n..(j + 1) * n].copy_from_slice(&solved.u[o * n..(o + 1) * n]);
        v[j * n..(j + 1) * n].copy_from_slice(&solved.v[o * n..(o + 1) * n]);
    }
    Ok(())
}

/// The SVD of the n x (n + sqre) upper bidiagonal block with diagonal `d` and superdiagonal `e`
/// (n - 1 + sqre).
fn solve(d: &[f64], e: &[f64], sqre: usize) -> Result<Solved, NoConvergence> {
    let n = d.len();
    let m = n + sqre;
    if n <= SMALL {
        return dense(d, e, sqre);
    }
    let nl = n / 2;
    let nr = n - nl - 1;
    // the upper half has one extra column (shared with the middle row), the lower keeps sqre
    let top = solve(&d[..nl], &e[..nl], 1)?;
    let bottom = solve(&d[nl + 1..], &e[nl + 1..], sqre)?;
    let (alpha, beta) = (d[nl], e[nl]);
    merge(n, m, nl, nr, sqre, alpha, beta, top, bottom)
}

/// A small block by faer's dense SVD.
fn dense(d: &[f64], e: &[f64], sqre: usize) -> Result<Solved, NoConvergence> {
    let n = d.len();
    let m = n + sqre;
    let b = Mat::<f64>::from_fn(n, m, |i, j| if i == j { d[i] } else if j == i + 1 { e[i] } else { 0.0 });
    let svd = b.svd().map_err(|_| NoConvergence)?;
    let (u, v) = (svd.U(), svd.V());
    let s = (0..n).map(|i| svd.S()[i]).collect();
    let u = (0..n).flat_map(|j| (0..n).map(move |i| u[(i, j)])).collect();
    let v = (0..m).flat_map(|j| (0..m).map(move |i| v[(i, j)])).collect();
    Ok(Solved { s, u, v })
}

/// Where a basis vector of the merge is nonzero: the upper half's rows, a mix (rotated pairs), the
/// lower half's rows, or the middle row alone (a left vector; sorts last).
#[derive(Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Debug)]
enum Block {
    Top,
    Mixed,
    Bottom,
    Middle,
}

/// The kept entries ordered by block, each entry's position in that order, and the counts of Top,
/// Mixed and Bottom entries.
fn by_block(kept: &[usize], blocks: &[Block]) -> (Vec<usize>, Vec<usize>, [usize; 3]) {
    let k = kept.len();
    let mut idx: Vec<usize> = (0..k).collect();
    idx.sort_by_key(|&i| blocks[kept[i]]);
    let mut pos = vec![0; k];
    for (p, &i) in idx.iter().enumerate() {
        pos[i] = p;
    }
    let count = |b: Block| idx.iter().filter(|&&i| blocks[kept[i]] == b).count();
    let counts = [count(Block::Top), count(Block::Mixed), count(Block::Bottom)];
    (idx, pos, counts)
}

#[allow(clippy::too_many_arguments)]
fn merge(n: usize, m: usize, nl: usize, nr: usize, sqre: usize, alpha: f64, beta: f64, top: Solved, bottom: Solved) -> Result<Solved, NoConvergence> {
    let (m1, m2) = (nl + 1, nr + sqre);
    // the broken arrow's entries: value, coupling in the middle row, and the left (n) and right
    // (m) basis vectors they stand for, as columns of `lw` and `rw`
    let mut d = vec![0.0; n];
    let mut z = vec![0.0; n];
    let mut lw = vec![0.0; n * n];
    let mut rw = vec![0.0; m * m];
    let mut lblock = vec![Block::Top; n];
    let mut rblock = vec![Block::Top; n];
    // entry 0: the null directions (V1's last column, and V2's last with sqre) combined into the
    // one carrying all their coupling, value 0; the other combination couples nothing (B's null)
    let (za, zb) = (alpha * top.v[nl * m1 + nl], if sqre == 1 { beta * bottom.v[nr * m2] } else { 0.0 });
    let r = za.hypot(zb);
    let (c, s) = if r == 0.0 { (1.0, 0.0) } else { (za / r, zb / r) };
    z[0] = r;
    lw[nl] = 1.0;
    lblock[0] = Block::Middle;
    for (r, &x) in rw[..m1].iter_mut().zip(&top.v[nl * m1..(nl + 1) * m1]) {
        *r = c * x;
    }
    let mut null = None;
    if sqre == 1 {
        for i in 0..m2 {
            rw[m1 + i] = s * bottom.v[nr * m2 + i];
        }
        rblock[0] = if s == 0.0 {
            Block::Top
        } else if c == 0.0 {
            Block::Bottom
        } else {
            Block::Mixed
        };
        let mut x = vec![0.0; m];
        for (r, &t) in x[..m1].iter_mut().zip(&top.v[nl * m1..(nl + 1) * m1]) {
            *r = -s * t;
        }
        for i in 0..m2 {
            x[m1 + i] = c * bottom.v[nr * m2 + i];
        }
        null = Some(x);
    }
    for j in 0..nl {
        let e = 1 + j;
        d[e] = top.s[j];
        z[e] = alpha * top.v[j * m1 + nl];
        lw[e * n..e * n + nl].copy_from_slice(&top.u[j * nl..(j + 1) * nl]);
        rw[e * m..e * m + m1].copy_from_slice(&top.v[j * m1..(j + 1) * m1]);
    }
    for j in 0..nr {
        let e = 1 + nl + j;
        d[e] = bottom.s[j];
        z[e] = beta * bottom.v[j * m2];
        lw[e * n + nl + 1..(e + 1) * n].copy_from_slice(&bottom.u[j * nr..(j + 1) * nr]);
        rw[e * m + m1..(e + 1) * m].copy_from_slice(&bottom.v[j * m2..(j + 1) * m2]);
        lblock[e] = Block::Bottom;
        rblock[e] = Block::Bottom;
    }
    // ascending values after entry 0
    let mut order: Vec<usize> = (1..n).collect();
    order.sort_by(|&a, &b| d[a].total_cmp(&d[b]));

    // deflation (dlasd2): negligible couplings, and values too close to tell apart rotated so one
    // coupling vanishes (rows and columns together: the block is d I); entry 0 keeps its coupling
    let dmax = d.iter().fold(0.0f64, |a, x| a.max(x.abs()));
    let tol = 64.0 * EPS * dmax.max(alpha.abs()).max(beta.abs());
    if z[0].abs() <= tol {
        z[0] = tol;
    }
    let rotate = |w: &mut [f64], len: usize, p: usize, j: usize, c: f64, s: f64| {
        for i in 0..len {
            let (x, y) = (w[p * len + i], w[j * len + i]);
            w[p * len + i] = c * x + s * y;
            w[j * len + i] = c * y - s * x;
        }
    };
    let mut kept = vec![0usize];
    let mut deflated: Vec<usize> = Vec::new();
    let mut pending: Option<usize> = None;
    for &j in &order {
        if z[j].abs() <= tol {
            deflated.push(j);
            continue;
        }
        let Some(p) = pending else {
            pending = Some(j);
            continue;
        };
        if (d[j] - d[p]).abs() <= tol {
            let tau = z[p].hypot(z[j]);
            let (c, s) = (z[j] / tau, -z[p] / tau);
            rotate(&mut lw, n, p, j, c, s);
            rotate(&mut rw, m, p, j, c, s);
            for blocks in [&mut lblock, &mut rblock] {
                if blocks[p] != blocks[j] {
                    blocks[p] = Block::Mixed;
                    blocks[j] = Block::Mixed;
                }
            }
            z[j] = tau;
            z[p] = 0.0;
            deflated.push(p);
        } else {
            kept.push(p);
        }
        pending = Some(j);
    }
    if let Some(p) = pending {
        kept.push(p);
    }

    let k = kept.len();
    let mut dk: Vec<f64> = kept.iter().map(|&j| d[j]).collect();
    // the smallest kept value kept apart from the zero one (dlasd2)
    if k > 1 && dk[1] <= tol {
        dk[1] = tol;
    }
    let zk: Vec<f64> = kept.iter().map(|&j| z[j]).collect();
    let rho: f64 = zk.iter().map(|v| v * v).sum();
    let z2: Vec<f64> = zk.iter().map(|v| v * v / rho).collect();
    // roots in the squared values: delta[i * k + j] = d_j^2 - sigma_i^2
    let mut delta = vec![0.0; k * k];
    let mut sigma = vec![0.0; k];
    for i in 0..k {
        let (o, tau) = secular(k, &z2, rho, i, |o| dk.iter().map(|&v| (v - dk[o]) * (v + dk[o])).collect(), |i| (dk[i + 1] - dk[i]) * (dk[i + 1] + dk[i]), &mut delta[i * k..(i + 1) * k])?;
        sigma[i] = (dk[o] * dk[o] + tau).max(0.0).sqrt();
    }
    // Gu & Eisenstat: the coupling for which the computed values are exact (where rounding leaves
    // no positive square, the coupling as it was)
    let mut w: Vec<f64> = (0..k).map(|i| delta[i * k + i]).collect();
    for j in 0..k {
        let row = &delta[j * k..(j + 1) * k];
        for (range_w, range_d) in [(0..j, 0..j), (j + 1..k, j + 1..k)] {
            for ((w, &dl), &di) in w[range_w].iter_mut().zip(&row[range_d.clone()]).zip(&dk[range_d]) {
                *w *= dl / ((di - dk[j]) * (di + dk[j]));
            }
        }
    }
    let zhat: Vec<f64> = (0..k).map(|i| if -w[i] > 0.0 && w[i].is_finite() { (-w[i]).sqrt().copysign(zk[i]) } else { zk[i] }).collect();

    // the kept entries in block order, separately for the left and right bases, so each half's rows
    // come from a contiguous run of them
    let (left_idx, left_pos, [left_top, left_mixed, left_bottom]) = by_block(&kept, &lblock);
    let (right_idx, right_pos, [right_top, right_mixed, right_bottom]) = by_block(&kept, &rblock);
    // the arrow's vectors (k x k, column-major, rows in those orders): right v_i = zhat_i / (d_i^2 -
    // sigma^2), left (M v) = (-1, d_i v_i), each normalized
    let mut vm = vec![0.0; k * k];
    let mut um = vec![0.0; k * k];
    let (mut vc, mut uc) = (vec![0.0; k], vec![0.0; k]);
    for j in 0..k {
        let row = &delta[j * k..(j + 1) * k];
        for i in 0..k {
            vc[i] = zhat[i] / row[i];
            uc[i] = if i == 0 { -1.0 } else { dk[i] * vc[i] };
        }
        let nearest = (0..k).min_by(|&a, &b| row[a].abs().total_cmp(&row[b].abs())).unwrap_or(0);
        normalize(&mut vc, nearest);
        normalize(&mut uc, nearest);
        for i in 0..k {
            vm[j * k + right_pos[i]] = vc[i];
            um[j * k + left_pos[i]] = uc[i];
        }
    }
    // back to the full spaces, block by block: a half's rows come from its own and the mixed
    // vectors; the middle row of the left vectors from entry 0 alone
    let gather = |w: &[f64], len: usize, idx: &[usize], cols: &std::ops::Range<usize>, rows: &std::ops::Range<usize>| -> Vec<f64> {
        idx[cols.clone()].iter().flat_map(|&i| w[kept[i] * len + rows.start..kept[i] * len + rows.end].iter().copied()).collect()
    };
    let mut u_out = vec![0.0; n * k];
    let mut v_out = vec![0.0; m * k];
    for (rows, cols) in [(0..nl, 0..left_top + left_mixed), (nl + 1..n, left_top..left_top + left_mixed + left_bottom)] {
        let a = gather(&lw, n, &left_idx, &cols, &rows);
        product_into(&a, rows.len(), &um, k, cols.start, cols.len(), &mut u_out, n, rows.start);
    }
    let middle = left_pos[0];
    for j in 0..k {
        u_out[j * n + nl] = um[j * k + middle];
    }
    for (rows, cols) in [(0..m1, 0..right_top + right_mixed), (m1..m, right_top..right_top + right_mixed + right_bottom)] {
        let a = gather(&rw, m, &right_idx, &cols, &rows);
        product_into(&a, rows.len(), &vm, k, cols.start, cols.len(), &mut v_out, m, rows.start);
    }

    let mut s = sigma;
    let mut u = u_out;
    let mut v = v_out;
    s.reserve(n - k);
    u.reserve(n * (n - k));
    v.reserve(m * (m - k));
    for &j in &deflated {
        s.push(d[j]);
        u.extend_from_slice(&lw[j * n..(j + 1) * n]);
        v.extend_from_slice(&rw[j * m..(j + 1) * m]);
    }
    if let Some(null) = null {
        v.extend_from_slice(&null);
    }
    Ok(Solved { s, u, v })
}

/// `out[r0..r0 + rows, :] = a (rows x nc) * b[c0..c0 + nc, :]`, `b` k x k column-major and `out`
/// column-major with leading dimension `ld`.
#[allow(clippy::too_many_arguments)]
fn product_into(a: &[f64], rows: usize, b: &[f64], k: usize, c0: usize, nc: usize, out: &mut [f64], ld: usize, r0: usize) {
    if rows == 0 || nc == 0 {
        return;
    }
    let a = MatRef::from_column_major_slice(a, rows, nc);
    let b = MatRef::from_column_major_slice(b, k, k).subrows(c0, nc);
    // SAFETY: rows r0..r0 + rows of the ld x k column-major `out`
    let c = unsafe { MatMut::from_raw_parts_mut(out.as_mut_ptr().add(r0), rows, k, 1, ld as isize) };
    faer::linalg::matmul::matmul(c, Accum::Replace, a, b, 1.0, Par::Seq);
}

/// Scales `x` to unit length; a vector that collapsed (zero or not finite) becomes `e_fallback`.
fn normalize(x: &mut [f64], fallback: usize) {
    let norm = x.iter().map(|v| v * v).sum::<f64>().sqrt();
    if norm > 0.0 && norm.is_finite() {
        let scale = 1.0 / norm;
        x.iter_mut().for_each(|v| *v *= scale);
    } else {
        x.fill(0.0);
        x[fallback] = 1.0;
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn random(n: usize, seed: u64) -> Vec<f64> {
        let mut state = seed;
        (0..n)
            .map(|_| {
                state = state.wrapping_mul(6_364_136_223_846_793_005).wrapping_add(1_442_695_040_888_963_407);
                (state >> 11) as f64 / (1u64 << 53) as f64 * 2.0 - 1.0
            })
            .collect()
    }

    /// Checks B = U diag(s) V^T: values against dqds, U and V orthogonal, B v = s u.
    fn check(d: &[f64], e: &[f64], what: &str) {
        let n = d.len();
        let (mut s, mut u, mut v) = (d.to_vec(), vec![0.0; n * n], vec![0.0; n * n]);
        bidiagonal_svd(&mut s, e, &mut u, &mut v).unwrap_or_else(|_| panic!("{what}: no convergence"));
        let mut want = d.to_vec();
        super::super::values::bidiagonal_singular_values(&mut want, e).unwrap();
        let norm = d.iter().chain(e).fold(0.0f64, |m, x| m.max(x.abs())).max(f64::MIN_POSITIVE);
        let tol = 1e-13 * (n as f64).max(1.0);
        for (a, b) in s.iter().zip(&want) {
            assert!((a - b).abs() <= tol * norm, "{what}: singular value {a} vs {b}");
        }
        assert!(s.windows(2).all(|w| w[0] >= w[1]), "{what}: not descending");
        for (name, q) in [("U", &u), ("V", &v)] {
            for i in 0..n {
                for j in 0..n {
                    let dot: f64 = (0..n).map(|r| q[i * n + r] * q[j * n + r]).sum();
                    let want = if i == j { 1.0 } else { 0.0 };
                    assert!((dot - want).abs() <= tol, "{what}: {name} columns {i}, {j}: {dot}");
                }
            }
        }
        for j in 0..n {
            for r in 0..n {
                let mut bv = d[r] * v[j * n + r];
                if r + 1 < n {
                    bv += e[r] * v[j * n + r + 1];
                }
                assert!((bv - s[j] * u[j * n + r]).abs() <= tol * norm, "{what}: residual at {j}, {r}");
            }
        }
    }

    #[test]
    fn random_bidiagonals_of_many_sizes() {
        for (k, n) in [1, 2, 3, 24, 25, 26, 27, 40, 51, 52, 53, 77, 100, 150].into_iter().enumerate() {
            check(&random(n, 10 + k as u64), &random(n.saturating_sub(1), 100 + k as u64), &format!("random {n}"));
        }
    }

    #[test]
    fn rank_deficient() {
        // a 64 x 64 matrix of two repeated columns, reduced by faer
        let n = 64;
        let r = random(n * 2, 77);
        let a = faer::Mat::<f64>::from_fn(n, n, |i, j| r[i * 2 + j % 2]);
        let bs = faer::linalg::qr::no_pivoting::factor::recommended_block_size::<f64>(n, n);
        let (mut bid, mut hl, mut hr) = (a.clone(), faer::Mat::<f64>::zeros(bs, n), faer::Mat::<f64>::zeros(bs, n - 1));
        let mut buf = faer::dyn_stack::MemBuffer::new(faer::linalg::svd::bidiag::bidiag_in_place_scratch::<f64>(n, n, Par::Seq, Default::default()));
        faer::linalg::svd::bidiag::bidiag_in_place(bid.as_mut(), hl.as_mut(), hr.as_mut(), Par::Seq, faer::dyn_stack::MemStack::new(&mut buf), Default::default());
        let d: Vec<f64> = (0..n).map(|i| bid[(i, i)]).collect();
        let e: Vec<f64> = (0..n - 1).map(|i| bid[(i, i + 1)]).collect();
        check(&d, &e, "rank 2");
    }

    #[test]
    fn hard_bidiagonals() {
        let n = 120;
        check(&vec![1.0; n], &vec![1e-12; n - 1], "cluster");
        check(&random(n, 3), &vec![0.0; n - 1], "diagonal");
        check(&vec![1.0; n], &vec![1.0; n - 1], "ones");
        let d: Vec<f64> = (0..n).map(|i| (i % 3) as f64).collect();
        let e: Vec<f64> = (0..n - 1).map(|i| if i % 10 == 9 { 0.0 } else { 1e-3 }).collect();
        check(&d, &e, "repeated with zeros");
        let d: Vec<f64> = (0..60).map(|i| 10f64.powi(-i / 4)).collect();
        let e: Vec<f64> = (0..59).map(|i| 10f64.powi(-i / 4 - 1)).collect();
        check(&d, &e, "graded");
        let mut d = random(80, 5);
        d[40] = 0.0;
        check(&d, &random(79, 6), "a zero on the diagonal");
        check(&random(80, 7).iter().map(|x| x * 1e300).collect::<Vec<_>>(), &random(79, 8).iter().map(|x| x * 1e300).collect::<Vec<_>>(), "huge");
    }
}
