//! Matrix equations of control theory, as `scipy.linalg` solves them: Sylvester (`a x + x b = q`),
//! continuous and discrete Lyapunov, and continuous and discrete algebraic Riccati equations.
//!
//! Sylvester and Lyapunov equations go through real Schur forms (Bartels-Stewart: `O(n³)`). The
//! continuous Riccati equation is solved from the matrix sign function of its Hamiltonian, then
//! polished by Newton steps (each a Lyapunov solve); the discrete one by the structure-preserving
//! doubling algorithm, which converges quadratically.
//!
//! tend: Numerics / linalg / matrix equations

use super::dense::Mat;
use super::schur::real_schur;
use super::{LinalgError, LinalgFloat};
use crate::signal::{NdArray, NdView};

/// The diagonal blocks of a quasi-triangular matrix: `(start, size)`, size 1 or 2.
fn blocks(t: &Mat) -> Vec<(usize, usize)> {
    let n = t.rows;
    let mut out = Vec::new();
    let mut i = 0;
    while i < n {
        let size = if i + 1 < n && t.at(i + 1, i) != 0.0 { 2 } else { 1 };
        out.push((i, size));
        i += size;
    }
    out
}

/// Solves the dense system `m y = rhs` (at most 4 x 4) by Gaussian elimination with partial
/// pivoting.
fn solve_small(mut m: Vec<f64>, mut rhs: Vec<f64>, n: usize) -> Result<Vec<f64>, LinalgError> {
    let scale = m.iter().fold(0.0f64, |a, v| a.max(v.abs())).max(f64::MIN_POSITIVE);
    for col in 0..n {
        let pivot = (col..n).max_by(|&a, &b| m[a * n + col].abs().total_cmp(&m[b * n + col].abs())).expect("rows remain");
        if m[pivot * n + col].abs() <= scale * f64::EPSILON * 4.0 {
            return Err(LinalgError::Singular);
        }
        for j in 0..n {
            m.swap(col * n + j, pivot * n + j);
        }
        rhs.swap(col, pivot);
        for row in col + 1..n {
            let f = m[row * n + col] / m[col * n + col];
            for j in col..n {
                m[row * n + j] -= f * m[col * n + j];
            }
            rhs[row] -= f * rhs[col];
        }
    }
    for row in (0..n).rev() {
        let s: f64 = (row + 1..n).map(|j| m[row * n + j] * rhs[j]).sum();
        rhs[row] = (rhs[row] - s) / m[row * n + row];
    }
    Ok(rhs)
}

/// Solves `s y + y t = f` for quasi-upper-triangular `s` (n x n) and `t` (m x m), block by block
/// (LAPACK's `trsyl`): `t`'s block columns left to right, each first reduced by the columns already
/// solved, then `s`'s blocks bottom to top. `y` is kept by columns, so every inner product runs
/// over contiguous memory.
fn quasi_triangular_sylvester(s: &Mat, t: &Mat, f: &Mat) -> Result<Mat, LinalgError> {
    let (n, m) = (s.rows, t.rows);
    let (sb, tb) = (blocks(s), blocks(t));
    let mut cols: Vec<Vec<f64>> = (0..m).map(|j| (0..n).map(|i| f.at(i, j)).collect()).collect();
    for &(j0, q) in &tb {
        // less the solved columns: f_c - sum_{l < j0} y_l t[l][c]
        let (done, rest) = cols.split_at_mut(j0);
        for (c, target) in rest[..q].iter_mut().enumerate() {
            for (l, yl) in done.iter().enumerate() {
                let tlc = t.at(l, j0 + c);
                if tlc != 0.0 {
                    target.iter_mut().zip(yl).for_each(|(v, &y)| *v -= tlc * y);
                }
            }
        }
        for &(i0, p) in sb.iter().rev() {
            // the right-hand side less the rows of this block column already solved
            let mut rhs = vec![0.0; p * q];
            for r in 0..p {
                let srow = &s.data[(i0 + r) * n + i0 + p..(i0 + r + 1) * n];
                for c in 0..q {
                    let col = &cols[j0 + c];
                    rhs[r * q + c] = col[i0 + r] - srow.iter().zip(&col[i0 + p..]).map(|(a, b)| a * b).sum::<f64>();
                }
            }
            // (I ⊗ S_ii + T_jjᵀ ⊗ I) vec(Y_ij) = vec(rhs), row-major unknowns (r, c)
            let size = p * q;
            let mut mat = vec![0.0; size * size];
            for r in 0..p {
                for c in 0..q {
                    let row = r * q + c;
                    for k in 0..p {
                        mat[row * size + k * q + c] += s.at(i0 + r, i0 + k);
                    }
                    for l in 0..q {
                        mat[row * size + r * q + l] += t.at(j0 + l, j0 + c);
                    }
                }
            }
            let sol = solve_small(mat, rhs, size)?;
            for r in 0..p {
                for c in 0..q {
                    cols[j0 + c][i0 + r] = sol[r * q + c];
                }
            }
        }
    }
    let mut y = Mat::zeros(n, m);
    for (j, col) in cols.iter().enumerate() {
        for (i, &v) in col.iter().enumerate() {
            y.set(i, j, v);
        }
    }
    Ok(y)
}

/// Bartels-Stewart on f64 matrices: `a x + x b = q`.
pub(crate) fn sylvester(a: &Mat, b: &Mat, q: &Mat) -> Result<Mat, LinalgError> {
    if q.rows != a.rows || q.cols != b.rows {
        return Err(LinalgError::Mismatch(vec![a.rows, b.rows], vec![q.rows, q.cols]));
    }
    let (sa, sb) = (real_schur(a)?, real_schur(b)?);
    let f = sa.z.t().mul(q).mul(&sb.z);
    let y = quasi_triangular_sylvester(&sa.t, &sb.t, &f)?;
    Ok(sa.z.mul(&y).mul(&sb.z.t()))
}

/// `a x + x aᵀ = q` from one Schur form: with `a = u s uᵀ`, `s y + y sᵀ = uᵀ q u`. Reversing the
/// order of the columns turns `sᵀ` (lower) into an upper quasi-triangular `t`, so the same block
/// solver applies: `s (y p) + (y p)(p sᵀ p) = (f p)`.
pub(crate) fn lyapunov(a: &Mat, q: &Mat) -> Result<Mat, LinalgError> {
    if q.rows != a.rows || q.cols != a.rows {
        return Err(LinalgError::Mismatch(vec![a.rows, a.cols], vec![q.rows, q.cols]));
    }
    let n = a.rows;
    let sa = real_schur(a)?;
    let f = sa.z.t().mul(q).mul(&sa.z);
    let mut t = Mat::zeros(n, n);
    let mut fr = Mat::zeros(n, n);
    for i in 0..n {
        for j in 0..n {
            t.set(i, j, sa.t.at(n - 1 - j, n - 1 - i));
            fr.set(i, j, f.at(i, n - 1 - j));
        }
    }
    let yr = quasi_triangular_sylvester(&sa.t, &t, &fr)?;
    let mut y = Mat::zeros(n, n);
    for i in 0..n {
        for j in 0..n {
            y.set(i, j, yr.at(i, n - 1 - j));
        }
    }
    Ok(sa.z.mul(&y).mul(&sa.z.t()).symmetrized_if(q))
}

/// `a x aᵀ - x + q = 0`, through the bilinear transform to a continuous equation (as SciPy does
/// for larger matrices): with `b = (aᵀ - I)(aᵀ + I)⁻¹` and `c = 2 (a + I)⁻¹ q (aᵀ + I)⁻¹`,
/// `bᵀ x + x b = -c`.
pub(crate) fn discrete_lyapunov(a: &Mat, q: &Mat) -> Result<Mat, LinalgError> {
    let eye = Mat::identity(a.rows);
    let at = a.t();
    let at_plus_inv = at.add(&eye).inv()?;
    let b = at.sub(&eye).mul(&at_plus_inv);
    let c = a.add(&eye).inv()?.mul(q).mul(&at_plus_inv).scale(2.0);
    Ok(lyapunov(&b.t(), &c.scale(-1.0))?.symmetrized_if(q))
}

impl Mat {
    /// Symmetrized when `q` is symmetric (so the solution of a symmetric equation is exactly
    /// symmetric), else as is.
    fn symmetrized_if(self, q: &Mat) -> Mat {
        if q.rows == q.cols && (0..q.rows).all(|i| (0..i).all(|j| q.at(i, j) == q.at(j, i))) { self.symmetrized() } else { self }
    }
}

fn riccati_inputs<T: LinalgFloat>(a: NdView<'_, T>, b: NdView<'_, T>, q: NdView<'_, T>, r: NdView<'_, T>) -> Result<(Mat, Mat, Mat, Mat), LinalgError> {
    let (a, b, q, r) = (Mat::square(a)?, Mat::from_view(b)?, Mat::square(q)?, Mat::square(r)?);
    if b.rows != a.rows || q.rows != a.rows || r.rows != b.cols {
        return Err(LinalgError::Mismatch(vec![a.rows, a.cols, b.rows, b.cols], vec![q.rows, r.rows]));
    }
    Ok((a, b, q, r))
}

/// The continuous algebraic Riccati equation's residual `aᵀ x + x a - x g x + q`.
fn care_residual(a: &Mat, g: &Mat, q: &Mat, x: &Mat) -> Mat {
    a.t().mul(x).add(&x.mul(a)).sub(&x.mul(g).mul(x)).add(q)
}

/// Solves the continuous algebraic Riccati equation `aᵀ x + x a - x b r⁻¹ bᵀ x + q = 0` for the
/// stabilizing `x` (`scipy.linalg.solve_continuous_are`): `a - b r⁻¹ bᵀ x` has its eigenvalues in
/// the left half-plane. The matrix sign function of the Hamiltonian `[[a, -g], [-q, -aᵀ]]`
/// (Newton's iteration with determinant scaling) gives its stable subspace; Newton-Kleinman steps
/// (Lyapunov solves) then polish `x` to working precision. Errors when no stabilizing solution
/// exists (the iteration meets a singular matrix or does not converge).
pub fn solve_continuous_are<T: LinalgFloat>(a: NdView<'_, T>, b: NdView<'_, T>, q: NdView<'_, T>, r: NdView<'_, T>) -> Result<NdArray<T>, LinalgError> {
    let (a, b, q, r) = riccati_inputs(a, b, q, r)?;
    let n = a.rows;
    let g = b.mul(&r.inv()?).mul(&b.t()).symmetrized();
    // the Hamiltonian
    let mut z = Mat::zeros(2 * n, 2 * n);
    for i in 0..n {
        for j in 0..n {
            z.set(i, j, a.at(i, j));
            z.set(i, n + j, -g.at(i, j));
            z.set(n + i, j, -q.at(i, j));
            z.set(n + i, n + j, -a.at(j, i));
        }
    }
    let mut converged = false;
    for _ in 0..100 {
        let c = z.det_root();
        let next = z.scale(0.5 / c).add(&z.inv()?.scale(0.5 * c));
        let change = next.sub(&z).norm();
        z = next;
        if change <= 1e-13 * z.norm() {
            converged = true;
            break;
        }
    }
    if !converged {
        return Err(LinalgError::NoConvergence);
    }
    // (sign + I) [I; x] = 0: [w12; w22 + I] x = -[w11 + I; w21]
    let mut lhs = Mat::zeros(2 * n, n);
    let mut rhs = Mat::zeros(2 * n, n);
    for i in 0..2 * n {
        for j in 0..n {
            lhs.set(i, j, z.at(i, n + j) + if i == n + j { 1.0 } else { 0.0 });
            rhs.set(i, j, -(z.at(i, j) + if i == j { 1.0 } else { 0.0 }));
        }
    }
    let mut x = lhs.lstsq(&rhs)?.symmetrized();
    // Newton-Kleinman: (a - g x)ᵀ dx + dx (a - g x) = -residual
    for _ in 0..3 {
        let res = care_residual(&a, &g, &q, &x);
        if res.norm() <= 1e-15 * x.norm().max(1.0) {
            break;
        }
        let closed = a.sub(&g.mul(&x));
        let dx = lyapunov(&closed.t(), &res.scale(-1.0))?;
        x = x.add(&dx).symmetrized();
    }
    Ok(x.to_array())
}

/// Solves the discrete algebraic Riccati equation
/// `aᵀ x a - x - aᵀ x b (r + bᵀ x b)⁻¹ bᵀ x a + q = 0` for the stabilizing `x`
/// (`scipy.linalg.solve_discrete_are`), by the structure-preserving doubling algorithm: with
/// `g = b r⁻¹ bᵀ`, `w = I + g h`, `a ← a w⁻¹ a`, `g ← g + a w⁻¹ g aᵀ`, `h ← h + aᵀ h w⁻¹ a`, `h`
/// converging quadratically to `x`. Errors when it does not converge.
pub fn solve_discrete_are<T: LinalgFloat>(a: NdView<'_, T>, b: NdView<'_, T>, q: NdView<'_, T>, r: NdView<'_, T>) -> Result<NdArray<T>, LinalgError> {
    let (a0, b, q, r) = riccati_inputs(a, b, q, r)?;
    let n = a0.rows;
    let eye = Mat::identity(n);
    let (mut ak, mut gk, mut hk) = (a0.clone(), b.mul(&r.inv()?).mul(&b.t()).symmetrized(), q.symmetrized());
    for _ in 0..100 {
        let w_inv = eye.add(&gk.mul(&hk)).inv()?;
        let wa = w_inv.mul(&ak);
        let next_h = hk.add(&ak.t().mul(&hk).mul(&wa)).symmetrized();
        let next_g = gk.add(&ak.mul(&w_inv).mul(&gk).mul(&ak.t())).symmetrized();
        ak = ak.mul(&wa);
        let change = next_h.sub(&hk).norm();
        hk = next_h;
        gk = next_g;
        if change <= 1e-14 * hk.norm().max(f64::MIN_POSITIVE) {
            return Ok(dare_newton(&a0, &b, &q, &r, hk)?.to_array());
        }
    }
    Err(LinalgError::NoConvergence)
}

/// The discrete Riccati residual `aᵀ x a - x - aᵀ x b (r + bᵀ x b)⁻¹ bᵀ x a + q`, and the
/// closed-loop `a - b k` with `k = (r + bᵀ x b)⁻¹ bᵀ x a`.
fn dare_residual(a: &Mat, b: &Mat, q: &Mat, r: &Mat, x: &Mat) -> Result<(Mat, Mat), LinalgError> {
    let xa = x.mul(a);
    let k = r.add(&b.t().mul(x).mul(b)).inv()?.mul(&b.t().mul(&xa));
    let res = a.t().mul(&xa).sub(x).sub(&a.t().mul(x).mul(b).mul(&k)).add(q);
    Ok((res, a.sub(&b.mul(&k))))
}

/// Newton (Hewer) steps polishing a discrete Riccati solution: each solves
/// `acᵀ dx ac - dx + residual = 0` for the closed loop `ac`, kept while the residual shrinks.
fn dare_newton(a: &Mat, b: &Mat, q: &Mat, r: &Mat, mut x: Mat) -> Result<Mat, LinalgError> {
    let (mut res, mut closed) = dare_residual(a, b, q, r, &x)?;
    for _ in 0..3 {
        let Ok(dx) = discrete_lyapunov(&closed.t(), &res) else { break };
        let candidate = x.add(&dx).symmetrized();
        let (next_res, next_closed) = dare_residual(a, b, q, r, &candidate)?;
        if next_res.norm() >= res.norm() {
            break;
        }
        (x, res, closed) = (candidate, next_res, next_closed);
    }
    Ok(x)
}

/// Solves the Sylvester equation `a x + x b = q` (`scipy.linalg.solve_sylvester`): `a` n x n, `b`
/// m x m, `q` n x m, by Bartels-Stewart (both matrices to real Schur form, then block
/// substitution). Errors when `a` and `-b` share an eigenvalue (no unique solution).
pub fn solve_sylvester<T: LinalgFloat>(a: NdView<'_, T>, b: NdView<'_, T>, q: NdView<'_, T>) -> Result<NdArray<T>, LinalgError> {
    Ok(sylvester(&Mat::square(a)?, &Mat::square(b)?, &Mat::from_view(q)?)?.to_array())
}

/// Solves the continuous Lyapunov equation `a x + x aᵀ = q`
/// (`scipy.linalg.solve_continuous_lyapunov`). For a stable `a` and `q = -b bᵀ`, `x` is the
/// controllability Gramian.
pub fn solve_continuous_lyapunov<T: LinalgFloat>(a: NdView<'_, T>, q: NdView<'_, T>) -> Result<NdArray<T>, LinalgError> {
    Ok(lyapunov(&Mat::square(a)?, &Mat::square(q)?)?.to_array())
}

/// Solves the discrete Lyapunov (Stein) equation `a x aᵀ - x + q = 0`
/// (`scipy.linalg.solve_discrete_lyapunov`). Errors if `a` has the eigenvalue -1.
pub fn solve_discrete_lyapunov<T: LinalgFloat>(a: NdView<'_, T>, q: NdView<'_, T>) -> Result<NdArray<T>, LinalgError> {
    Ok(discrete_lyapunov(&Mat::square(a)?, &Mat::square(q)?)?.to_array())
}
