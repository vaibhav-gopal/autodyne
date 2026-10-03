//! Eigenvalues and eigenvectors of a symmetric tridiagonal matrix by divide and conquer (Cuppen's
//! method as LAPACK's `dstedc` runs it): the matrix is torn into two halves plus a rank-one
//! correction, each half solved recursively, and the halves' solutions merged by solving the
//! secular equation of the rank-one update. Eigenvalues that the update barely moves are deflated
//! (taken as they are); the others are found by a rational iteration that converges in a few
//! steps, each root computed relative to its nearer pole so the differences `d_j - λ` stay
//! accurate; the eigenvectors come from Gu & Eisenstat's recomputed update vector, which keeps them
//! orthogonal. Small blocks use the implicit QL iteration.

use faer::{Accum, MatMut, MatRef, Par};

use super::values::NoConvergence;

/// Blocks this small are solved by QL iteration.
const SMALL: usize = 25;
const EPS: f64 = f64::EPSILON / 2.0;

/// The eigenvalues (ascending, into `d`) and orthonormal eigenvectors (the columns of `q`, n x n,
/// column-major) of the symmetric tridiagonal matrix with diagonal `d` (n) and off-diagonal `e`
/// (n - 1).
pub(crate) fn tridiagonal_eigen(d: &mut [f64], e: &[f64], q: &mut [f64]) -> Result<(), NoConvergence> {
    let n = d.len();
    assert!(e.len() + 1 == n.max(1) && q.len() == n * n, "tridiagonal_eigen: n diagonal, n - 1 off-diagonal, n x n vectors");
    q.fill(0.0);
    if n == 0 {
        return Ok(());
    }
    // scaled so the largest entry is 1
    let scale = d.iter().chain(e).fold(0.0f64, |m, v| m.max(v.abs()));
    if !scale.is_finite() {
        return Err(NoConvergence);
    }
    if scale == 0.0 {
        (0..n).for_each(|i| q[i * n + i] = 1.0);
        return Ok(());
    }
    let mut e: Vec<f64> = e.iter().map(|v| v / scale).collect();
    d.iter_mut().for_each(|v| *v /= scale);
    solve(d, &mut e, q, n)?;
    d.iter_mut().for_each(|v| *v *= scale);
    Ok(())
}

/// Solves the block whose vectors are `q` (n x n at leading dimension `ld`, zero on entry).
fn solve(d: &mut [f64], e: &mut [f64], q: &mut [f64], ld: usize) -> Result<(), NoConvergence> {
    let n = d.len();
    if n <= SMALL {
        return ql(d, e, q, ld);
    }
    let m = n / 2;
    // T = diag(T1, T2) + beta v v^T, v = e_m + s e_{m+1}: the off-diagonal entry is beta s
    let coupling = e[m - 1];
    let beta = coupling.abs();
    d[m - 1] -= beta;
    d[m] -= beta;
    let (d1, d2) = d.split_at_mut(m);
    let (e1, e2) = e.split_at_mut(m);
    solve(d1, &mut e1[..m - 1], q, ld)?;
    solve(d2, e2, &mut q[m * ld + m..], ld)?;
    merge(d, q, ld, m, beta, if coupling < 0.0 { -1.0 } else { 1.0 })
}

/// Merges two solved halves (eigenvalues `d[..m]`, `d[m..]` ascending; vectors in the diagonal
/// blocks of `q`) under the rank-one update `beta v v^T`.
fn merge(d: &mut [f64], q: &mut [f64], ld: usize, m: usize, beta: f64, sign: f64) -> Result<(), NoConvergence> {
    let n = d.len();
    let col = |q: &[f64], j: usize| q[j * ld..j * ld + n].to_vec();
    // z = Q^T v: the last row of Q1 and (signed) the first row of Q2; |v| = sqrt 2, so z / sqrt 2 and
    // rho = 2 beta
    let s2 = std::f64::consts::FRAC_1_SQRT_2;
    let z: Vec<f64> = (0..n).map(|j| if j < m { q[j * ld + m - 1] * s2 } else { sign * q[j * ld + m] * s2 }).collect();
    let rho = 2.0 * beta;
    // the two halves' eigenvalues in one ascending order
    let mut order: Vec<usize> = (0..n).collect();
    order.sort_by(|&a, &b| d[a].total_cmp(&d[b]));
    let mut ds: Vec<f64> = order.iter().map(|&i| d[i]).collect();
    let mut zs: Vec<f64> = order.iter().map(|&i| z[i]).collect();
    let mut qs: Vec<Vec<f64>> = order.iter().map(|&i| col(q, i)).collect();

    // deflation: negligible components of z, and nearly equal eigenvalues rotated so one component
    // vanishes
    let dmax = ds.iter().fold(0.0f64, |a, v| a.max(v.abs()));
    let zmax = zs.iter().fold(0.0f64, |a, v| a.max(v.abs()));
    let tol = 8.0 * EPS * dmax.max(zmax);
    let mut kept: Vec<usize> = Vec::with_capacity(n);
    let mut deflated: Vec<usize> = Vec::new();
    let mut pending: Option<usize> = None;
    for j in 0..n {
        if rho * zs[j].abs() <= tol {
            deflated.push(j);
            continue;
        }
        let Some(p) = pending else {
            pending = Some(j);
            continue;
        };
        let tau = zs[p].hypot(zs[j]);
        // as LAPACK's dlaed2: c z_p + s z_j = 0, so the rotation zeroes z_p
        let (c, s) = (zs[j] / tau, -zs[p] / tau);
        if ((ds[j] - ds[p]) * c * s).abs() <= tol {
            // rotate columns p, j so z_p vanishes: p deflates
            zs[j] = tau;
            zs[p] = 0.0;
            let (x, y) = (std::mem::take(&mut qs[p]), std::mem::take(&mut qs[j]));
            qs[p] = x.iter().zip(&y).map(|(&a, &b)| c * a + s * b).collect();
            qs[j] = x.iter().zip(&y).map(|(&a, &b)| c * b - s * a).collect();
            let t = ds[p] * c * c + ds[j] * s * s;
            ds[j] = ds[p] * s * s + ds[j] * c * c;
            ds[p] = t;
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
    let mut pairs: Vec<(f64, Vec<f64>)> = deflated.iter().map(|&j| (ds[j], std::mem::take(&mut qs[j]))).collect();
    if k > 0 {
        let dk: Vec<f64> = kept.iter().map(|&j| ds[j]).collect();
        let zk: Vec<f64> = kept.iter().map(|&j| zs[j]).collect();
        let z2k: Vec<f64> = zk.iter().map(|v| v * v).collect();
        // delta[i * k + j] = dk[j] - lambda_i
        let mut delta = vec![0.0; k * k];
        let mut lambda = vec![0.0; k];
        for i in 0..k {
            let (origin, tau) = secular(k, &z2k, rho, i, |o| dk.iter().map(|&v| v - dk[o]).collect(), |i| dk[i + 1] - dk[i], &mut delta[i * k..(i + 1) * k])?;
            lambda[i] = dk[origin] + tau;
        }
        // Gu & Eisenstat: the z for which the computed eigenvalues are exact
        let mut w: Vec<f64> = (0..k).map(|i| delta[i * k + i]).collect();
        for j in 0..k {
            let row = &delta[j * k..(j + 1) * k];
            // i != j, as two branch-free runs
            for (range_w, range_d) in [(0..j, 0..j), (j + 1..k, j + 1..k)] {
                for ((w, &dl), &di) in w[range_w].iter_mut().zip(&row[range_d.clone()]).zip(&dk[range_d]) {
                    *w *= dl / (di - dk[j]);
                }
            }
        }
        // (where rounding leaves no positive square, the coupling as it was)
        let zhat: Vec<f64> = (0..k).map(|i| if -w[i] > 0.0 && w[i].is_finite() { (-w[i]).sqrt().copysign(zk[i]) } else { zk[i] }).collect();
        // A column from the first half is zero in the second half's rows and the other way round
        // (only rotated columns mix them): the kept columns in the order first half only, mixed,
        // second half only, so the top rows of the result come from a leading block of them and
        // the bottom rows from a trailing one
        let class = |j: usize| {
            if qs[j][m..].iter().all(|&v| v == 0.0) {
                0
            } else if qs[j][..m].iter().all(|&v| v == 0.0) {
                2
            } else {
                1
            }
        };
        let classes: Vec<u8> = kept.iter().map(|&j| class(j)).collect();
        let mut by_class: Vec<usize> = (0..k).collect();
        by_class.sort_by_key(|&i| classes[i]);
        let mut pos = vec![0; k];
        for (p, &i) in by_class.iter().enumerate() {
            pos[i] = p;
        }
        let first = classes.iter().filter(|&&c| c == 0).count();
        let second = classes.iter().filter(|&&c| c == 2).count();
        // the update's eigenvectors (k x k, column-major, rows in that order), normalized
        let mut u = vec![0.0; k * k];
        for j in 0..k {
            let column = &mut u[j * k..(j + 1) * k];
            let row = &delta[j * k..(j + 1) * k];
            let mut norm = 0.0;
            for i in 0..k {
                let v = zhat[i] / row[i];
                column[pos[i]] = v;
                norm += v * v;
            }
            if norm > 0.0 && norm.is_finite() {
                let scale = 1.0 / norm.sqrt();
                column.iter_mut().for_each(|v| *v *= scale);
            } else {
                // collapsed: the nearest pole's direction
                let nearest = (0..k).min_by(|&a, &b| row[a].abs().total_cmp(&row[b].abs())).unwrap_or(0);
                column.fill(0.0);
                column[pos[nearest]] = 1.0;
            }
        }
        // back to the full space: top rows from the first k - second columns, bottom rows from the
        // last k - first
        let mut out = vec![0.0; n * k];
        let ordered: Vec<usize> = by_class.iter().map(|&i| kept[i]).collect();
        for (rows, cols) in [(0..m, 0..k - second), (m..n, first..k)] {
            let (r0, nr, nc) = (rows.start, rows.len(), cols.len());
            if nr == 0 || nc == 0 {
                continue;
            }
            let qa: Vec<f64> = ordered[cols.clone()].iter().flat_map(|&j| qs[j][rows.clone()].iter().copied()).collect();
            let a = MatRef::from_column_major_slice(&qa, nr, nc);
            let b = MatRef::from_column_major_slice(&u, k, k).subrows(cols.start, nc);
            // SAFETY: rows r0..r0 + nr of the n x k column-major `out`
            let c = unsafe { MatMut::from_raw_parts_mut(out.as_mut_ptr().add(r0), nr, k, 1, n as isize) };
            faer::linalg::matmul::matmul(c, Accum::Replace, a, b, 1.0, Par::Seq);
        }        for (j, &l) in lambda.iter().enumerate() {
            pairs.push((l, out[j * n..(j + 1) * n].to_vec()));
        }
    }
    pairs.sort_by(|a, b| a.0.total_cmp(&b.0));
    for (j, (value, vector)) in pairs.into_iter().enumerate() {
        d[j] = value;
        q[j * ld..j * ld + n].copy_from_slice(&vector);
    }
    Ok(())
}

/// Root `i` of the secular equation `1/rho + sum_j z2_j / (p_j - x) = 0` over `k` poles `p` (ascending
/// and distinct, `rho > 0`): in `(p_i, p_{i+1})`, the last in `(p_{k-1}, p_{k-1} + rho sum z2]`.
/// The poles enter through `offsets(o)`, every `p_j - p_o` (computed accurately by the caller), and
/// `gap(i) = p_{i+1} - p_i`. Returns the origin `o`, the pole nearer the root, and `tau = x - p_o`;
/// writes `p_j - x` into `delta`, so the differences keep their relative accuracy.
pub(crate) fn secular(k: usize, z2: &[f64], rho: f64, i: usize, offsets: impl Fn(usize) -> Vec<f64>, gap: impl Fn(usize) -> f64, delta: &mut [f64]) -> Result<(usize, f64), NoConvergence> {
    let rho_inv = 1.0 / rho;
    let last = i + 1 == k;
    if k == 1 {
        let shift = rho * z2[0];
        delta[0] = -shift;
        return Ok((0, shift));
    }
    // the origin: the pole nearer the root, and the root's bracket around it
    let (origin, mut lo, mut hi, dd) = if last {
        (k - 1, 0.0, rho * z2.iter().sum::<f64>(), offsets(k - 1))
    } else {
        let half = gap(i) / 2.0;
        let from_i = offsets(i);
        let f: f64 = rho_inv + (0..k).map(|j| z2[j] / (from_i[j] - half)).sum::<f64>();
        if f >= 0.0 { (i, 0.0, half, from_i) } else { (i + 1, -half, 0.0, offsets(i + 1)) }
    };
    // the poles either side of the root (for the last root: the last two, both to its left)
    let (p, r) = if last { (k - 2, k - 1) } else { (i, i + 1) };
    let mut tau = (lo + hi) / 2.0;
    for _ in 0..200 {
        // psi: the terms of the poles up to p (left of the root, all negative), phi: the others
        // (positive); two branch-free loops, one division a term
        let side = |range: std::ops::Range<usize>, delta: &mut [f64]| {
            let (mut sum, mut slope) = (0.0, 0.0);
            for ((dl, &d), &w) in delta[range.clone()].iter_mut().zip(&dd[range.clone()]).zip(&z2[range]) {
                let dj = d - tau;
                *dl = dj;
                let inv = 1.0 / dj;
                let t = w * inv;
                sum += t;
                slope += t * inv;
            }
            (sum, slope)
        };
        let (psi, dpsi) = side(0..p + 1, delta);
        let (phi, dphi) = side(p + 1..k, delta);
        let bound = phi - psi;
        let f = rho_inv + psi + phi;
        // f's rounding error, as dlaed4 bounds it (the terms, 1 / rho, and tau's own error)
        if f == 0.0 || f.abs() <= EPS * (8.0 * bound + 2.0 * rho_inv + 3.0 * tau.abs() * (dpsi + dphi)) {
            return Ok((origin, tau));
        }
        if f < 0.0 {
            lo = tau;
        } else {
            hi = tau;
        }
        // each side as a constant plus one pole at its nearest (matching value and slope here; the
        // last root's right side is its own single term, exact), then the root of
        // C + b / (P - eta) + c / (R - eta) = 0 where the model crosses zero
        let (pp, rr) = (delta[p], delta[r]);
        let b = dpsi * pp * pp;
        let c = dphi * rr * rr;
        let cst = rho_inv + (psi - b / pp) + (phi - c / rr);
        let (a2, a1, a0) = (cst, cst * (pp + rr) + b + c, cst * pp * rr + b * rr + c * pp);
        let eta = if a2 == 0.0 {
            a0 / a1
        } else {
            let sq = (a1 * a1 - 4.0 * a2 * a0).max(0.0).sqrt();
            let (r1, r2) = if a1 >= 0.0 { ((a1 + sq) / (2.0 * a2), 2.0 * a0 / (a1 + sq)) } else { (2.0 * a0 / (a1 - sq), (a1 - sq) / (2.0 * a2)) };
            // between the two poles, or past both for the last root
            let fits = |eta: f64| if last { eta > rr } else { eta > pp && eta < rr };
            if fits(r1) { r1 } else { r2 }
        };
        let mut next = tau + eta;
        if !(next > lo && next < hi) || !next.is_finite() {
            next = (lo + hi) / 2.0;
        }
        if (next - tau).abs() <= 2.0 * EPS * next.abs().max(f64::MIN_POSITIVE) || hi - lo <= 4.0 * EPS * tau.abs().max(f64::MIN_POSITIVE) {
            for j in 0..k {
                delta[j] = dd[j] - next;
            }
            return Ok((origin, next));
        }
        tau = next;
    }
    Err(NoConvergence)
}

/// Eigenvalues (ascending) and vectors (columns of `q`, at leading dimension `ld`) of a small
/// tridiagonal block by the implicit QL iteration with Wilkinson-type shifts (EISPACK's `tql2`).
fn ql(d: &mut [f64], e: &[f64], q: &mut [f64], ld: usize) -> Result<(), NoConvergence> {
    let n = d.len();
    for i in 0..n {
        q[i * ld..i * ld + n].fill(0.0);
        q[i * ld + i] = 1.0;
    }
    if n == 1 {
        return Ok(());
    }
    // e shifted: e[i] couples i and i + 1, with e[n - 1] = 0
    let mut e: Vec<f64> = e.iter().copied().chain([0.0]).collect();
    let mut f = 0.0;
    let mut tst1 = 0.0f64;
    for l in 0..n {
        let mut iterations = 0;
        tst1 = tst1.max(d[l].abs() + e[l].abs());
        let mut m = l;
        while m < n - 1 && e[m].abs() > EPS * 2.0 * tst1 {
            m += 1;
        }
        if m > l {
            loop {
                iterations += 1;
                if iterations > 60 {
                    return Err(NoConvergence);
                }
                // the shift
                let g = d[l];
                let mut p = (d[l + 1] - g) / (2.0 * e[l]);
                let mut r = p.hypot(1.0);
                if p < 0.0 {
                    r = -r;
                }
                d[l] = e[l] / (p + r);
                d[l + 1] = e[l] * (p + r);
                let dl1 = d[l + 1];
                let h = g - d[l];
                for v in d.iter_mut().skip(l + 2) {
                    *v -= h;
                }
                f += h;
                // the implicit QL sweep
                p = d[m];
                let (mut c, mut c2, mut c3) = (1.0, 1.0, 1.0);
                let el1 = e[l + 1];
                let (mut s, mut s2) = (0.0, 0.0);
                for i in (l..m).rev() {
                    c3 = c2;
                    c2 = c;
                    s2 = s;
                    let g = c * e[i];
                    let h = c * p;
                    r = p.hypot(e[i]);
                    e[i + 1] = s * r;
                    s = e[i] / r;
                    c = p / r;
                    p = c * d[i] - s * g;
                    d[i + 1] = h + s * (c * g + s * d[i]);
                    for row in 0..n {
                        let (a, b) = (q[i * ld + row], q[(i + 1) * ld + row]);
                        q[(i + 1) * ld + row] = s * a + c * b;
                        q[i * ld + row] = c * a - s * b;
                    }
                }
                p = -s * s2 * c3 * el1 * e[l] / dl1;
                e[l] = s * p;
                d[l] = c * p;
                if e[l].abs() <= EPS * 2.0 * tst1 {
                    break;
                }
            }
        }
        d[l] += f;
        e[l] = 0.0;
    }
    // ascending, vectors along
    let mut order: Vec<usize> = (0..n).collect();
    order.sort_by(|&a, &b| d[a].total_cmp(&d[b]));
    let values: Vec<f64> = order.iter().map(|&i| d[i]).collect();
    let vectors: Vec<Vec<f64>> = order.iter().map(|&i| q[i * ld..i * ld + n].to_vec()).collect();
    for (j, (v, x)) in values.into_iter().zip(vectors).enumerate() {
        d[j] = v;
        q[j * ld..j * ld + n].copy_from_slice(&x);
    }
    Ok(())
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

    /// Checks the decomposition: eigenvalues against dsterf, Q^T Q = I, T q = lambda q.
    fn check(d: &[f64], e: &[f64], what: &str) {
        let n = d.len();
        let mut values = d.to_vec();
        let mut q = vec![0.0; n * n];
        tridiagonal_eigen(&mut values, e, &mut q).unwrap_or_else(|_| panic!("{what}: no convergence"));
        let mut want = d.to_vec();
        super::super::values::tridiagonal_eigenvalues(&mut want, e).unwrap();
        let norm = d.iter().chain(e).fold(0.0f64, |m, v| m.max(v.abs())).max(f64::MIN_POSITIVE);
        let tol = 1e-13 * (n as f64).max(1.0);
        for (a, b) in values.iter().zip(&want) {
            assert!((a - b).abs() <= tol * norm, "{what}: eigenvalue {a} vs {b}");
        }
        assert!(values.windows(2).all(|w| w[0] <= w[1]), "{what}: not ascending");
        for i in 0..n {
            for j in 0..n {
                let dot: f64 = (0..n).map(|r| q[i * n + r] * q[j * n + r]).sum();
                let want = if i == j { 1.0 } else { 0.0 };
                assert!((dot - want).abs() <= tol, "{what}: q{i} . q{j} = {dot}");
            }
            // T q_i - lambda_i q_i
            let x = &q[i * n..(i + 1) * n];
            for r in 0..n {
                let mut t = d[r] * x[r];
                if r > 0 {
                    t += e[r - 1] * x[r - 1];
                }
                if r + 1 < n {
                    t += e[r] * x[r + 1];
                }
                assert!((t - values[i] * x[r]).abs() <= tol * norm, "{what}: residual {} at {i}, {r}", t - values[i] * x[r]);
            }
        }
    }

    #[test]
    #[ignore]
    fn timing() {
        let n = 300;
        let (d, e) = (random(n, 1), random(n - 1, 2));
        let mut best = f64::MAX;
        for _ in 0..20 {
            let (mut v, mut q) = (d.clone(), vec![0.0; n * n]);
            let t = std::time::Instant::now();
            tridiagonal_eigen(&mut v, &e, &mut q).unwrap();
            best = best.min(t.elapsed().as_secs_f64() * 1e3);
        }
        let (mut v, mut q) = (d.clone(), vec![0.0; n * n]);
        let t = std::time::Instant::now();
        ql(&mut v, &e, &mut q, n).unwrap();
        eprintln!("TIMING d&c {best:.3} ms | ql on all {:.3} ms", t.elapsed().as_secs_f64() * 1e3);
    }

    #[test]
    fn random_tridiagonals_of_many_sizes() {
        for (k, n) in [1, 2, 3, 5, 24, 25, 26, 27, 50, 51, 64, 100, 131, 200].into_iter().enumerate() {
            check(&random(n, 10 + k as u64), &random(n.saturating_sub(1), 100 + k as u64), &format!("random {n}"));
        }
    }

    #[test]
    fn hard_spectra() {
        let n = 120;
        // a constant diagonal with tiny couplings: one tight cluster
        check(&vec![1.0; n], &vec![1e-12; n - 1], "cluster");
        // decoupled: every eigenvalue deflates
        check(&random(n, 3), &vec![0.0; n - 1], "diagonal");
        // repeated eigenvalues (the 1-2-1 Laplacian's spectrum is simple; this one is not)
        let d: Vec<f64> = (0..n).map(|i| (i % 3) as f64).collect();
        let e: Vec<f64> = (0..n - 1).map(|i| if i % 10 == 9 { 0.0 } else { 1e-3 }).collect();
        check(&d, &e, "repeated");
        // the Laplacian
        check(&vec![2.0; n], &vec![-1.0; n - 1], "laplacian");
        // Wilkinson's W+ (pairs of nearly equal eigenvalues)
        let m = 101;
        let d: Vec<f64> = (0..m).map(|i| (50.0 - i as f64).abs()).collect();
        check(&d, &vec![1.0; m - 1], "wilkinson");
        // graded
        let d: Vec<f64> = (0..60).map(|i| 10f64.powi(-i / 4)).collect();
        let e: Vec<f64> = (0..59).map(|i| 10f64.powi(-i / 4 - 1)).collect();
        check(&d, &e, "graded");
        // huge and tiny scales
        check(&random(80, 7).iter().map(|v| v * 1e300).collect::<Vec<_>>(), &random(79, 8).iter().map(|v| v * 1e300).collect::<Vec<_>>(), "huge");
        check(&random(80, 9).iter().map(|v| v * 1e-300).collect::<Vec<_>>(), &random(79, 10).iter().map(|v| v * 1e-300).collect::<Vec<_>>(), "tiny");
    }
}