//! Eigenvalues and singular values without vectors, in O(n²) after the O(n³) reduction:
//!
//! - [`tridiagonal_eigenvalues`]: a symmetric tridiagonal matrix's eigenvalues by the
//!   Pal-Walker-Kahan square-root-free QL / QR iteration (LAPACK's `dsterf`);
//! - [`bidiagonal_singular_values`]: a bidiagonal matrix's singular values by the dqds algorithm
//!   (Fernando & Parlett; LAPACK's `dlasq1` to `dlasq6`), which computes every singular value,
//!   however small, to high relative accuracy.
//!
//! Both are ports of the LAPACK routines (keeping their 1-based indexing of the qd array, so the
//! code can be read against the reference), in f64.

/// Relative machine precision (LAPACK's `dlamch('E')`).
const EPS: f64 = f64::EPSILON / 2.0;
/// `dlamch('P')`: precision times the base.
const PREC: f64 = f64::EPSILON;
/// `dlamch('S')`: the smallest number whose reciprocal does not overflow.
const SAFMIN: f64 = f64::MIN_POSITIVE;

/// The iteration did not converge.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) struct NoConvergence;

// TRIDIAGONAL EIGENVALUES (dsterf) ===================================================================

/// The eigenvalues of the symmetric tridiagonal matrix with diagonal `d` (n) and off-diagonal `e`
/// (n - 1): ascending, into `values` (which holds the diagonal on entry).
pub(crate) fn tridiagonal_eigenvalues(values: &mut [f64], off: &[f64]) -> Result<(), NoConvergence> {
    let n = values.len();
    if n <= 1 {
        return Ok(());
    }
    assert_eq!(off.len(), n - 1, "tridiagonal_eigenvalues: n - 1 off-diagonal values");
    // 1-based, as in LAPACK: d[1..=n], e[1..=n-1]
    let mut d = vec![0.0; n + 1];
    let mut e = vec![0.0; n + 1];
    d[1..].copy_from_slice(values);
    e[1..n].copy_from_slice(off);

    let eps2 = EPS * EPS;
    let safmax = 1.0 / SAFMIN;
    let ssfmax = safmax.sqrt() / 3.0;
    let ssfmin = SAFMIN.sqrt() / eps2;
    let nmaxit = n * 30;
    let mut jtot = 0;
    let mut l1 = 1;

    'outer: loop {
        if l1 > n {
            break;
        }
        if l1 > 1 {
            e[l1 - 1] = 0.0;
        }
        // a split: the first negligible off-diagonal value from l1
        let mut m = l1;
        while m < n {
            if e[m].abs() <= d[m].abs().sqrt() * d[m + 1].abs().sqrt() * EPS {
                e[m] = 0.0;
                break;
            }
            m += 1;
        }
        let mut l = l1;
        let lsv = l;
        let mut lend = m;
        let lendsv = lend;
        l1 = m + 1;
        if lend == l {
            continue;
        }
        // scale the block
        let anorm = (l..=lend).map(|i| d[i].abs()).chain((l..lend).map(|i| e[i].abs())).fold(0.0, f64::max);
        if anorm == 0.0 {
            continue;
        }
        let scale = if anorm > ssfmax {
            ssfmax / anorm
        } else if anorm < ssfmin {
            ssfmin / anorm
        } else {
            1.0
        };
        if scale != 1.0 {
            (l..=lend).for_each(|i| d[i] *= scale);
            (l..lend).for_each(|i| e[i] *= scale);
        }
        e[l..lend].iter_mut().for_each(|v| *v *= *v);
        // QL when the top is the smaller end, else QR
        if d[lend].abs() < d[l].abs() {
            lend = lsv;
            l = lendsv;
        }
        if lend >= l {
            // QL iteration: look for a small subdiagonal element
            loop {
                let mut m = l;
                while m < lend {
                    if e[m].abs() <= eps2 * (d[m] * d[m + 1]).abs() {
                        break;
                    }
                    m += 1;
                }
                if m < lend {
                    e[m] = 0.0;
                }
                let p = d[l];
                if m == l {
                    // an eigenvalue found
                    d[l] = p;
                    l += 1;
                    if l <= lend {
                        continue;
                    }
                    break;
                }
                if m == l + 1 {
                    // a 2 x 2 block
                    let rte = e[l].sqrt();
                    let (rt1, rt2) = lae2(d[l], rte, d[l + 1]);
                    d[l] = rt1;
                    d[l + 1] = rt2;
                    e[l] = 0.0;
                    l += 2;
                    if l <= lend {
                        continue;
                    }
                    break;
                }
                if jtot == nmaxit {
                    break;
                }
                jtot += 1;
                // the shift
                let rte = e[l].sqrt();
                let mut sigma = (d[l + 1] - p) / (2.0 * rte);
                let r = sigma.hypot(1.0);
                sigma = p - rte / (sigma + r.copysign(sigma));
                let (mut c, mut s) = (1.0, 0.0);
                let mut gamma = d[m] - sigma;
                let mut p = gamma * gamma;
                // the inner loop
                let mut i = m - 1;
                loop {
                    let bb = e[i];
                    let r = p + bb;
                    if i != m - 1 {
                        e[i + 1] = s * r;
                    }
                    let oldc = c;
                    c = p / r;
                    s = bb / r;
                    let oldgam = gamma;
                    let alpha = d[i];
                    gamma = c * (alpha - sigma) - s * oldgam;
                    d[i + 1] = oldgam + (alpha - gamma);
                    p = if c != 0.0 { gamma * gamma / c } else { oldc * bb };
                    if i == l {
                        break;
                    }
                    i -= 1;
                }
                e[l] = s * p;
                d[l] = sigma + gamma;
            }
        } else {
            // QR iteration: look for a small superdiagonal element
            loop {
                let mut m = l;
                while m > lend {
                    if e[m - 1].abs() <= eps2 * (d[m] * d[m - 1]).abs() {
                        break;
                    }
                    m -= 1;
                }
                if m > lend {
                    e[m - 1] = 0.0;
                }
                let p = d[l];
                if m == l {
                    d[l] = p;
                    if l == lend {
                        break;
                    }
                    l -= 1;
                    if l >= lend {
                        continue;
                    }
                    break;
                }
                if m + 1 == l {
                    let rte = e[l - 1].sqrt();
                    let (rt1, rt2) = lae2(d[l], rte, d[l - 1]);
                    d[l] = rt1;
                    d[l - 1] = rt2;
                    e[l - 1] = 0.0;
                    if l < lend + 2 {
                        break;
                    }
                    l -= 2;
                    continue;
                }
                if jtot == nmaxit {
                    break;
                }
                jtot += 1;
                let rte = e[l - 1].sqrt();
                let mut sigma = (d[l - 1] - p) / (2.0 * rte);
                let r = sigma.hypot(1.0);
                sigma = p - rte / (sigma + r.copysign(sigma));
                let (mut c, mut s) = (1.0, 0.0);
                let mut gamma = d[m] - sigma;
                let mut p = gamma * gamma;
                for i in m..l {
                    let bb = e[i];
                    let r = p + bb;
                    if i != m {
                        e[i - 1] = s * r;
                    }
                    let oldc = c;
                    c = p / r;
                    s = bb / r;
                    let oldgam = gamma;
                    let alpha = d[i + 1];
                    gamma = c * (alpha - sigma) - s * oldgam;
                    d[i] = oldgam + (alpha - gamma);
                    p = if c != 0.0 { gamma * gamma / c } else { oldc * bb };
                }
                e[l - 1] = s * p;
                d[l] = sigma + gamma;
            }
        }
        // undo the scaling
        if scale != 1.0 {
            (lsv..=lendsv).for_each(|i| d[i] /= scale);
        }
        if jtot >= nmaxit {
            if (1..n).any(|i| e[i] != 0.0) {
                return Err(NoConvergence);
            }
            break 'outer;
        }
    }
    values.copy_from_slice(&d[1..]);
    values.sort_by(f64::total_cmp);
    Ok(())
}

/// The eigenvalues of `[[a, b], [b, c]]`, the larger in magnitude first (LAPACK's `dlae2`).
fn lae2(a: f64, b: f64, c: f64) -> (f64, f64) {
    let (sm, df) = (a + c, a - c);
    let (adf, tb) = (df.abs(), b + b);
    let ab = tb.abs();
    let (acmx, acmn) = if a.abs() > c.abs() { (a, c) } else { (c, a) };
    let rt = if adf > ab {
        adf * (1.0 + (ab / adf).powi(2)).sqrt()
    } else if adf < ab {
        ab * (1.0 + (adf / ab).powi(2)).sqrt()
    } else {
        ab * std::f64::consts::SQRT_2
    };
    if sm < 0.0 {
        let rt1 = 0.5 * (sm - rt);
        (rt1, (acmx / rt1) * acmn - (b / rt1) * b)
    } else if sm > 0.0 {
        let rt1 = 0.5 * (sm + rt);
        (rt1, (acmx / rt1) * acmn - (b / rt1) * b)
    } else {
        (0.5 * rt, -0.5 * rt)
    }
}

// BIDIAGONAL SINGULAR VALUES (dqds: dlasq1 - dlasq6) =================================================

/// The singular values of the upper bidiagonal matrix with diagonal `d` (n) and superdiagonal `e`
/// (n - 1): descending, into `values` (which holds the diagonal on entry). Every singular value is
/// computed to high relative accuracy.
pub(crate) fn bidiagonal_singular_values(values: &mut [f64], off: &[f64]) -> Result<(), NoConvergence> {
    let n = values.len();
    if n == 0 {
        return Ok(());
    }
    assert_eq!(off.len(), n.saturating_sub(1), "bidiagonal_singular_values: n - 1 superdiagonal values");
    if n == 1 {
        values[0] = values[0].abs();
        return Ok(());
    }
    if n == 2 {
        let (smin, smax) = las2(values[0], off[0], values[1]);
        values[0] = smax;
        values[1] = smin;
        return Ok(());
    }
    let mut sigmx = off.iter().fold(0.0f64, |m, &e| m.max(e.abs()));
    values.iter_mut().for_each(|d| *d = d.abs());
    if sigmx == 0.0 {
        values.sort_by(|a, b| b.total_cmp(a));
        return Ok(());
    }
    sigmx = values.iter().fold(sigmx, |m, &d| m.max(d));
    // scale to avoid over- and underflow, square, and run dqds on the qd array
    let scale = (PREC / SAFMIN).sqrt();
    let factor = scale / sigmx;
    let mut z = vec![0.0; 4 * n + 1];
    for i in 0..n {
        z[2 * i + 1] = (values[i] * factor).powi(2);
        if i + 1 < n {
            z[2 * i + 2] = (off[i] * factor).powi(2);
        }
    }
    z[2 * n] = 0.0;
    lasq2(n, &mut z)?;
    for i in 0..n {
        values[i] = z[i + 1].sqrt() / factor;
    }
    Ok(())
}

/// The singular values of `[[f, g], [0, h]]`: `(smaller, larger)` (LAPACK's `dlas2`).
fn las2(f: f64, g: f64, h: f64) -> (f64, f64) {
    let (fa, ga, ha) = (f.abs(), g.abs(), h.abs());
    let (fhmn, fhmx) = (fa.min(ha), fa.max(ha));
    if fhmn == 0.0 {
        let smax = if fhmx == 0.0 { ga } else { fhmx.max(ga) * (1.0 + (fhmx.min(ga) / fhmx.max(ga)).powi(2)).sqrt() };
        (0.0, smax)
    } else if ga < fhmx {
        let as_ = 1.0 + fhmn / fhmx;
        let at = (fhmx - fhmn) / fhmx;
        let au = (ga / fhmx).powi(2);
        let c = 2.0 / ((as_ * as_ + au).sqrt() + (at * at + au).sqrt());
        (fhmn * c, fhmx / c)
    } else {
        let au = fhmx / ga;
        if au == 0.0 {
            ((fhmn * fhmx) / ga, ga)
        } else {
            let as_ = 1.0 + fhmn / fhmx;
            let at = (fhmx - fhmn) / fhmx;
            let c = 1.0 / ((1.0 + (as_ * au).powi(2)).sqrt() + (1.0 + (at * au).powi(2)).sqrt());
            let smin = (fhmn * c) * au;
            (smin + smin, ga / (c + c))
        }
    }
}

/// `min` that propagates NaN, as the IEEE path of dqds relies on to detect a failed step.
fn nmin(a: f64, b: f64) -> f64 {
    if a.is_nan() || b.is_nan() { f64::NAN } else { a.min(b) }
}

/// The state LAPACK's dqds routines pass to one another.
#[derive(Default)]
struct Qd {
    dmin: f64,
    dmin1: f64,
    dmin2: f64,
    dn: f64,
    dn1: f64,
    dn2: f64,
    g: f64,
    tau: f64,
    sigma: f64,
    desig: f64,
    qmax: f64,
    ttype: i32,
    nfail: usize,
    iter: usize,
    ndiv: usize,
}

/// dqds on the qd array `z` (1-based, length 4n + 1, holding q1, e1, q2, e2, ... in z[1..2n]):
/// the squares of the singular values, descending, into z[1..=n] (LAPACK's `dlasq2`).
fn lasq2(n: usize, z: &mut [f64]) -> Result<(), NoConvergence> {
    const CBIAS: f64 = 1.5;
    let tol = PREC * 100.0;
    let tol2 = tol * tol;
    // sums of the q's and e's, and a check for negative data
    z[2 * n] = 0.0;
    let (mut d, mut e) = (0.0, 0.0);
    let mut k = 1;
    while k <= 2 * (n - 1) {
        if z[k] < 0.0 || z[k + 1] < 0.0 {
            return Err(NoConvergence);
        }
        d += z[k];
        e += z[k + 1];
        k += 2;
    }
    if z[2 * n - 1] < 0.0 {
        return Err(NoConvergence);
    }
    d += z[2 * n - 1];
    // diagonal: the q's are the answer
    if e == 0.0 {
        for k in 2..=n {
            z[k] = z[2 * k - 1];
        }
        z[1..=n].sort_by(|a, b| b.total_cmp(a));
        return Ok(());
    }
    let trace = d + e;
    if trace == 0.0 {
        z[2 * n - 1] = 0.0;
        return Ok(());
    }
    // rearrange for locality: z = (q1, qq1, e1, ee1, q2, qq2, e2, ee2, ...)
    let mut k = 2 * n;
    while k >= 2 {
        z[2 * k] = 0.0;
        z[2 * k - 1] = z[k];
        z[2 * k - 2] = 0.0;
        z[2 * k - 3] = z[k - 1];
        k -= 2;
    }
    let mut i0 = 1usize;
    let mut n0 = n;
    // reverse the qd-array, if warranted
    if CBIAS * z[4 * i0 - 3] < z[4 * n0 - 3] {
        let ipn4 = 4 * (i0 + n0);
        let mut i4 = 4 * i0;
        while i4 <= 2 * (i0 + n0 - 1) {
            z.swap(i4 - 3, ipn4 - i4 - 3);
            z.swap(i4 - 1, ipn4 - i4 - 5);
            i4 += 4;
        }
    }
    // initial split checking via dqd and Li's test
    let mut pp = 0usize;
    let mut qd = Qd::default();
    for _ in 0..2 {
        let mut d = z[4 * n0 + pp - 3];
        let mut i4 = 4 * (n0 - 1) + pp;
        while i4 >= 4 * i0 + pp {
            if z[i4 - 1] <= tol2 * d {
                z[i4 - 1] = -0.0;
                d = z[i4 - 3];
            } else {
                d = z[i4 - 3] * (d / (d + z[i4 - 1]));
            }
            i4 -= 4;
        }
        // dqd maps z to zz plus Li's test
        let mut d = z[4 * i0 + pp - 3];
        let mut i4 = 4 * i0 + pp;
        while i4 <= 4 * (n0 - 1) + pp {
            z[i4 - 2 * pp - 2] = d + z[i4 - 1];
            if z[i4 - 1] <= tol2 * d {
                z[i4 - 1] = -0.0;
                z[i4 - 2 * pp - 2] = d;
                z[i4 - 2 * pp] = 0.0;
                d = z[i4 + 1];
            } else if SAFMIN * z[i4 + 1] < z[i4 - 2 * pp - 2] && SAFMIN * z[i4 - 2 * pp - 2] < z[i4 + 1] {
                let temp = z[i4 + 1] / z[i4 - 2 * pp - 2];
                z[i4 - 2 * pp] = z[i4 - 1] * temp;
                d *= temp;
            } else {
                z[i4 - 2 * pp] = z[i4 + 1] * (z[i4 - 1] / z[i4 - 2 * pp - 2]);
                d = z[i4 + 1] * (d / z[i4 - 2 * pp - 2]);
            }
            i4 += 4;
        }
        z[4 * n0 - pp - 2] = d;
        // the largest q
        qd.qmax = z[4 * i0 - pp - 2];
        let mut i4 = 4 * i0 - pp + 2;
        while i4 <= 4 * n0 - pp - 2 {
            qd.qmax = qd.qmax.max(z[i4]);
            i4 += 4;
        }
        pp = 1 - pp;
    }
    qd.iter = 2;
    qd.ndiv = 2 * (n0 - i0);

    for _ in 0..=n {
        if n0 < 1 {
            break;
        }
        // e(n0) holds sigma (negated) where the block i0..n0 split from the rest
        qd.desig = 0.0;
        qd.sigma = if n0 == n { 0.0 } else { -z[4 * n0 - 1] };
        if qd.sigma < 0.0 {
            return Err(NoConvergence);
        }
        // the last unreduced block's top i0, its largest q and a Gershgorin-type bound
        let mut emax = 0.0f64;
        let mut qmin = z[4 * n0 - 3];
        qd.qmax = qmin;
        let mut i4 = 4 * n0;
        let mut top = 4;
        while i4 >= 8 {
            if z[i4 - 5] <= 0.0 {
                top = i4;
                break;
            }
            if qmin >= 4.0 * emax {
                qmin = qmin.min(z[i4 - 3]);
                emax = emax.max(z[i4 - 5]);
            }
            qd.qmax = qd.qmax.max(z[i4 - 7] + z[i4 - 5]);
            i4 -= 4;
        }
        i0 = top / 4;
        pp = 0;
        if n0 - i0 > 1 {
            let mut dee = z[4 * i0 - 3];
            let mut deemin = dee;
            let mut kmin = i0;
            let mut i4 = 4 * i0 + 1;
            while i4 <= 4 * n0 - 3 {
                dee = z[i4] * (dee / (dee + z[i4 - 2]));
                if dee <= deemin {
                    deemin = dee;
                    kmin = i4.div_ceil(4);
                }
                i4 += 4;
            }
            if (kmin - i0) * 2 < n0 - kmin && deemin <= 0.5 * z[4 * n0 - 3] {
                let ipn4 = 4 * (i0 + n0);
                pp = 2;
                let mut i4 = 4 * i0;
                while i4 <= 2 * (i0 + n0 - 1) {
                    z.swap(i4 - 3, ipn4 - i4 - 3);
                    z.swap(i4 - 2, ipn4 - i4 - 2);
                    z.swap(i4 - 1, ipn4 - i4 - 5);
                    z.swap(i4, ipn4 - i4 - 4);
                    i4 += 4;
                }
            }
        }
        // -(the initial shift)
        qd.dmin = -(0.0f64).max(qmin - 2.0 * qmin.sqrt() * emax.sqrt());
        let nbig = 100 * (n0 - i0 + 1);
        let mut done = false;
        for _ in 0..nbig {
            if i0 > n0 {
                done = true;
                break;
            }
            lasq3(i0, &mut n0, z, &mut pp, &mut qd);
            pp = 1 - pp;
            // when emin is very small, check for splits
            if pp == 0 && n0 >= i0 + 3 && (z[4 * n0] <= tol2 * qd.qmax || z[4 * n0 - 1] <= tol2 * qd.sigma) {
                let mut splt = i0 - 1;
                qd.qmax = z[4 * i0 - 3];
                let mut emin = z[4 * i0 - 1];
                let mut oldemn = z[4 * i0];
                let mut i4 = 4 * i0;
                while i4 <= 4 * (n0 - 3) {
                    if z[i4] <= tol2 * z[i4 - 3] || z[i4 - 1] <= tol2 * qd.sigma {
                        z[i4 - 1] = -qd.sigma;
                        splt = i4 / 4;
                        qd.qmax = 0.0;
                        emin = z[i4 + 3];
                        oldemn = z[i4 + 4];
                    } else {
                        qd.qmax = qd.qmax.max(z[i4 + 1]);
                        emin = emin.min(z[i4 - 1]);
                        oldemn = oldemn.min(z[i4]);
                    }
                    i4 += 4;
                }
                z[4 * n0 - 1] = emin;
                z[4 * n0] = oldemn;
                i0 = splt + 1;
            }
        }
        if !done && i0 <= n0 {
            return Err(NoConvergence);
        }
    }
    if n0 >= 1 {
        return Err(NoConvergence);
    }
    // the q's, in front, descending
    for k in 2..=n {
        z[k] = z[4 * k - 3];
    }
    z[1..=n].sort_by(|a, b| b.total_cmp(a));
    Ok(())
}

/// Deflation checks, then one dqds step with a good shift (LAPACK's `dlasq3`).
fn lasq3(i0: usize, n0: &mut usize, z: &mut [f64], pp: &mut usize, qd: &mut Qd) {
    const CBIAS: f64 = 1.5;
    let n0in = *n0;
    let tol = PREC * 100.0;
    let tol2 = tol * tol;
    // deflation
    loop {
        if *n0 < i0 {
            return;
        }
        let nn = 4 * *n0 + *pp;
        let one = if *n0 == i0 {
            true
        } else if *n0 == i0 + 1 {
            false
        } else if z[nn - 5] > tol2 * (qd.sigma + z[nn - 3]) && z[nn - 2 * *pp - 4] > tol2 * z[nn - 7] {
            // E(n0 - 1) not negligible: check E(n0 - 2)
            if z[nn - 9] > tol2 * qd.sigma && z[nn - 2 * *pp - 8] > tol2 * z[nn - 11] {
                break;
            }
            false
        } else {
            true
        };
        if one {
            z[4 * *n0 - 3] = z[4 * *n0 + *pp - 3] + qd.sigma;
            *n0 -= 1;
            continue;
        }
        // two eigenvalues
        if z[nn - 3] > z[nn - 7] {
            z.swap(nn - 3, nn - 7);
        }
        let t = 0.5 * ((z[nn - 7] - z[nn - 3]) + z[nn - 5]);
        if z[nn - 5] > z[nn - 3] * tol2 && t != 0.0 {
            let mut s = z[nn - 3] * (z[nn - 5] / t);
            if s <= t {
                s = z[nn - 3] * (z[nn - 5] / (t * (1.0 + (1.0 + s / t).sqrt())));
            } else {
                s = z[nn - 3] * (z[nn - 5] / (t + t.sqrt() * (t + s).sqrt()));
            }
            let t = z[nn - 7] + (s + z[nn - 5]);
            z[nn - 3] *= z[nn - 7] / t;
            z[nn - 7] = t;
        }
        z[4 * *n0 - 7] = z[nn - 7] + qd.sigma;
        z[4 * *n0 - 3] = z[nn - 3] + qd.sigma;
        if *n0 < 2 {
            *n0 = 0;
            return;
        }
        *n0 -= 2;
    }
    if *pp == 2 {
        *pp = 0;
    }
    let (n0v, ppv) = (*n0, *pp);
    // reverse the qd-array, if warranted
    if (qd.dmin <= 0.0 || n0v < n0in) && CBIAS * z[4 * i0 + ppv - 3] < z[4 * n0v + ppv - 3] {
        let ipn4 = 4 * (i0 + n0v);
        let mut j4 = 4 * i0;
        while j4 <= 2 * (i0 + n0v - 1) {
            z.swap(j4 - 3, ipn4 - j4 - 3);
            z.swap(j4 - 2, ipn4 - j4 - 2);
            z.swap(j4 - 1, ipn4 - j4 - 5);
            z.swap(j4, ipn4 - j4 - 4);
            j4 += 4;
        }
        if n0v - i0 <= 4 {
            z[4 * n0v + ppv - 1] = z[4 * i0 + ppv - 1];
            z[4 * n0v - ppv] = z[4 * i0 - ppv];
        }
        qd.dmin2 = qd.dmin2.min(z[4 * n0v + ppv - 1]);
        z[4 * n0v + ppv - 1] = z[4 * n0v + ppv - 1].min(z[4 * i0 + ppv - 1]).min(z[4 * i0 + ppv + 3]);
        z[4 * n0v - ppv] = z[4 * n0v - ppv].min(z[4 * i0 - ppv]).min(z[4 * i0 - ppv + 4]);
        qd.qmax = qd.qmax.max(z[4 * i0 + ppv - 3]).max(z[4 * i0 + ppv + 1]);
        qd.dmin = -0.0;
    }
    // a shift, then dqds until dmin > 0
    lasq4(i0, n0v, z, ppv, n0in, qd);
    loop {
        lasq5(i0, n0v, z, ppv, qd);
        qd.ndiv += n0v - i0 + 2;
        qd.iter += 1;
        if qd.dmin >= 0.0 && qd.dmin1 >= 0.0 {
            break; // success
        } else if qd.dmin < 0.0 && qd.dmin1 > 0.0 && z[4 * (n0v - 1) - ppv] < tol * (qd.sigma + qd.dn1) && qd.dn.abs() < tol * qd.sigma {
            // convergence hidden by a negative dn
            z[4 * (n0v - 1) - ppv + 2] = 0.0;
            qd.dmin = 0.0;
            break;
        } else if qd.dmin < 0.0 {
            // tau too big: a new tau
            qd.nfail += 1;
            if qd.ttype < -22 {
                qd.tau = 0.0;
            } else if qd.dmin1 > 0.0 {
                qd.tau = (qd.tau + qd.dmin) * (1.0 - 2.0 * PREC);
                qd.ttype -= 11;
            } else {
                qd.tau *= 0.25;
                qd.ttype -= 12;
            }
        } else if qd.dmin.is_nan() {
            if qd.tau == 0.0 {
                lasq6(i0, n0v, z, ppv, qd);
                qd.ndiv += n0v - i0 + 2;
                qd.iter += 1;
                qd.tau = 0.0;
                break;
            }
            qd.tau = 0.0;
        } else {
            // possible underflow: play it safe
            lasq6(i0, n0v, z, ppv, qd);
            qd.ndiv += n0v - i0 + 2;
            qd.iter += 1;
            qd.tau = 0.0;
            break;
        }
    }
    // sigma += tau, with compensated summation
    let t;
    if qd.tau < qd.sigma {
        qd.desig += qd.tau;
        t = qd.sigma + qd.desig;
        qd.desig -= t - qd.sigma;
    } else {
        t = qd.sigma + qd.tau;
        qd.desig = qd.sigma + (qd.desig - (t - qd.tau));
    }
    qd.sigma = t;
}

/// The shift (LAPACK's `dlasq4`): from the last dqds step's minima, an estimate of the smallest
/// eigenvalue of the block. Leaves `tau` unchanged when the estimate is unreliable.
fn lasq4(i0: usize, n0: usize, z: &[f64], pp: usize, n0in: usize, qd: &mut Qd) {
    const CNST1: f64 = 0.563;
    const CNST2: f64 = 1.010;
    const CNST3: f64 = 1.050;
    const QURTR: f64 = 0.25;
    const THIRD: f64 = 0.333;
    if qd.dmin <= 0.0 {
        qd.tau = -qd.dmin;
        qd.ttype = -1;
        return;
    }
    let nn = 4 * n0 + pp;
    let (dmin, dmin1, dmin2, dn, dn1, dn2) = (qd.dmin, qd.dmin1, qd.dmin2, qd.dn, qd.dn1, qd.dn2);
    let s;
    if n0in == n0 {
        // no eigenvalues deflated
        if dmin == dn || dmin == dn1 {
            let b1 = z[nn - 3].sqrt() * z[nn - 5].sqrt();
            let b2 = z[nn - 7].sqrt() * z[nn - 9].sqrt();
            let a2 = z[nn - 7] + z[nn - 5];
            if dmin == dn && dmin1 == dn1 {
                // cases 2 and 3
                let gap2 = dmin2 - a2 - dmin2 * QURTR;
                let gap1 = if gap2 > 0.0 && gap2 > b2 { a2 - dn - (b2 / gap2) * b2 } else { a2 - dn - (b1 + b2) };
                if gap1 > 0.0 && gap1 > b1 {
                    s = (dn - (b1 / gap1) * b1).max(0.5 * dmin);
                    qd.ttype = -2;
                } else {
                    let mut t = 0.0;
                    if dn > b1 {
                        t = dn - b1;
                    }
                    if a2 > b1 + b2 {
                        t = t.min(a2 - (b1 + b2));
                    }
                    s = t.max(THIRD * dmin);
                    qd.ttype = -3;
                }
            } else {
                // case 4
                qd.ttype = -4;
                let mut t = QURTR * dmin;
                let (gam, mut a2, mut b2, np);
                if dmin == dn {
                    gam = dn;
                    a2 = 0.0;
                    if z[nn - 5] > z[nn - 7] {
                        return;
                    }
                    b2 = z[nn - 5] / z[nn - 7];
                    np = nn as isize - 9;
                } else {
                    let npp = nn - 2 * pp;
                    gam = dn1;
                    if z[npp - 4] > z[npp - 2] {
                        return;
                    }
                    a2 = z[npp - 4] / z[npp - 2];
                    if z[nn - 9] > z[nn - 11] {
                        return;
                    }
                    b2 = z[nn - 9] / z[nn - 11];
                    np = nn as isize - 13;
                }
                // approximate contribution to the norm squared from i < nn - 1
                a2 += b2;
                let mut i4 = np;
                while i4 >= (4 * i0 + pp) as isize - 1 {
                    if b2 == 0.0 {
                        break;
                    }
                    let b1 = b2;
                    let k = i4 as usize;
                    if z[k] > z[k - 2] {
                        return;
                    }
                    b2 *= z[k] / z[k - 2];
                    a2 += b2;
                    if 100.0 * b2.max(b1) < a2 || CNST1 < a2 {
                        break;
                    }
                    i4 -= 4;
                }
                a2 *= CNST3;
                // Rayleigh quotient residual bound
                if a2 < CNST1 {
                    t = gam * (1.0 - a2.sqrt()) / (1.0 + a2);
                }
                s = t;
            }
        } else if dmin == dn2 {
            // case 5
            qd.ttype = -5;
            let mut t = QURTR * dmin;
            // contribution to the norm squared from i > nn - 2
            let np = nn - 2 * pp;
            let (mut b1, mut b2) = (z[np - 2], z[np - 6]);
            let gam = dn2;
            if z[np - 8] > b2 || z[np - 4] > b1 {
                return;
            }
            let mut a2 = (z[np - 8] / b2) * (1.0 + z[np - 4] / b1);
            // approximate contribution to the norm squared from i < nn - 2
            if n0 - i0 > 2 {
                b2 = z[nn - 13] / z[nn - 15];
                a2 += b2;
                let mut i4 = nn as isize - 17;
                while i4 >= (4 * i0 + pp) as isize - 1 {
                    if b2 == 0.0 {
                        break;
                    }
                    b1 = b2;
                    let k = i4 as usize;
                    if z[k] > z[k - 2] {
                        return;
                    }
                    b2 *= z[k] / z[k - 2];
                    a2 += b2;
                    if 100.0 * b2.max(b1) < a2 || CNST1 < a2 {
                        break;
                    }
                    i4 -= 4;
                }
                a2 *= CNST3;
            }
            if a2 < CNST1 {
                t = gam * (1.0 - a2.sqrt()) / (1.0 + a2);
            }
            s = t;
        } else {
            // case 6: no information to guide us
            if qd.ttype == -6 {
                qd.g += THIRD * (1.0 - qd.g);
            } else if qd.ttype == -18 {
                qd.g = QURTR * THIRD;
            } else {
                qd.g = QURTR;
            }
            s = qd.g * dmin;
            qd.ttype = -6;
        }
    } else if n0in == n0 + 1 {
        // one eigenvalue just deflated: dmin1, dn1 stand for dmin, dn
        if dmin1 == dn1 && dmin2 == dn2 {
            // cases 7 and 8
            qd.ttype = -7;
            let mut t = THIRD * dmin1;
            if z[nn - 5] > z[nn - 7] {
                return;
            }
            let mut b1 = z[nn - 5] / z[nn - 7];
            let mut b2 = b1;
            if b2 != 0.0 {
                let mut i4 = (4 * n0 + pp) as isize - 9;
                while i4 >= (4 * i0 + pp) as isize - 1 {
                    let a2 = b1;
                    let k = i4 as usize;
                    if z[k] > z[k - 2] {
                        return;
                    }
                    b1 *= z[k] / z[k - 2];
                    b2 += b1;
                    if 100.0 * b1.max(a2) < b2 {
                        break;
                    }
                    i4 -= 4;
                }
            }
            let b2 = (CNST3 * b2).sqrt();
            let a2 = dmin1 / (1.0 + b2 * b2);
            let gap2 = 0.5 * dmin2 - a2;
            if gap2 > 0.0 && gap2 > b2 * a2 {
                t = t.max(a2 * (1.0 - CNST2 * a2 * (b2 / gap2) * b2));
            } else {
                t = t.max(a2 * (1.0 - CNST2 * b2));
                qd.ttype = -8;
            }
            s = t;
        } else {
            // case 9
            s = if dmin1 == dn1 { 0.5 * dmin1 } else { QURTR * dmin1 };
            qd.ttype = -9;
        }
    } else if n0in == n0 + 2 {
        // two eigenvalues deflated: dmin2, dn2 stand for dmin, dn (cases 10 and 11)
        if dmin2 == dn2 && 2.0 * z[nn - 5] < z[nn - 7] {
            qd.ttype = -10;
            let mut t = THIRD * dmin2;
            if z[nn - 5] > z[nn - 7] {
                return;
            }
            let mut b1 = z[nn - 5] / z[nn - 7];
            let mut b2 = b1;
            if b2 != 0.0 {
                let mut i4 = (4 * n0 + pp) as isize - 9;
                while i4 >= (4 * i0 + pp) as isize - 1 {
                    let k = i4 as usize;
                    if z[k] > z[k - 2] {
                        return;
                    }
                    b1 *= z[k] / z[k - 2];
                    b2 += b1;
                    if 100.0 * b1 < b2 {
                        break;
                    }
                    i4 -= 4;
                }
            }
            let b2 = (CNST3 * b2).sqrt();
            let a2 = dmin2 / (1.0 + b2 * b2);
            let gap2 = z[nn - 7] + z[nn - 9] - z[nn - 11].sqrt() * z[nn - 9].sqrt() - a2;
            if gap2 > 0.0 && gap2 > b2 * a2 {
                t = t.max(a2 * (1.0 - CNST2 * a2 * (b2 / gap2) * b2));
            } else {
                t = t.max(a2 * (1.0 - CNST2 * b2));
            }
            s = t;
        } else {
            s = QURTR * dmin2;
            qd.ttype = -11;
        }
    } else {
        // case 12: more than two eigenvalues deflated, no information
        s = 0.0;
        qd.ttype = -12;
    }
    qd.tau = s;
}

/// One dqds step with shift `tau` (LAPACK's `dlasq5`, IEEE arithmetic).
fn lasq5(i0: usize, n0: usize, z: &mut [f64], pp: usize, qd: &mut Qd) {
    if n0 <= i0 + 1 {
        return;
    }
    let dthresh = EPS * (qd.sigma + qd.tau);
    if qd.tau < dthresh * 0.5 {
        qd.tau = 0.0;
    }
    let tau = qd.tau;
    let mut j4 = 4 * i0 + pp - 3;
    let mut emin = z[j4 + 4];
    let mut d = z[j4] - tau;
    let mut dmin = d;
    qd.dmin1 = -z[j4];
    // tau = 0: small d's are set to zero
    let flush = tau == 0.0;
    j4 = 4 * i0;
    while j4 <= 4 * (n0 - 3) {
        if pp == 0 {
            z[j4 - 2] = d + z[j4 - 1];
            let temp = z[j4 + 1] / z[j4 - 2];
            d = d * temp - tau;
            if flush && d < dthresh {
                d = 0.0;
            }
            dmin = nmin(dmin, d);
            z[j4] = z[j4 - 1] * temp;
            emin = emin.min(z[j4]);
        } else {
            z[j4 - 3] = d + z[j4];
            let temp = z[j4 + 2] / z[j4 - 3];
            d = d * temp - tau;
            if flush && d < dthresh {
                d = 0.0;
            }
            dmin = nmin(dmin, d);
            z[j4 - 1] = z[j4] * temp;
            emin = emin.min(z[j4 - 1]);
        }
        j4 += 4;
    }
    // the last two steps, unrolled
    let dnm2 = d;
    qd.dmin2 = dmin;
    let mut j4 = 4 * (n0 - 2) - pp;
    let mut j4p2 = j4 + 2 * pp - 1;
    z[j4 - 2] = dnm2 + z[j4p2];
    z[j4] = z[j4p2 + 2] * (z[j4p2] / z[j4 - 2]);
    let dnm1 = z[j4p2 + 2] * (dnm2 / z[j4 - 2]) - tau;
    dmin = nmin(dmin, dnm1);
    qd.dmin1 = dmin;
    j4 += 4;
    j4p2 = j4 + 2 * pp - 1;
    z[j4 - 2] = dnm1 + z[j4p2];
    z[j4] = z[j4p2 + 2] * (z[j4p2] / z[j4 - 2]);
    let dn = z[j4p2 + 2] * (dnm1 / z[j4 - 2]) - tau;
    dmin = nmin(dmin, dn);
    z[j4 + 2] = dn;
    z[4 * n0 - pp] = emin;
    (qd.dmin, qd.dn, qd.dn1, qd.dn2) = (dmin, dn, dnm1, dnm2);
}

/// One dqd step without shift, guarded against underflow (LAPACK's `dlasq6`).
fn lasq6(i0: usize, n0: usize, z: &mut [f64], pp: usize, qd: &mut Qd) {
    if n0 <= i0 + 1 {
        return;
    }
    let mut j4 = 4 * i0 + pp - 3;
    let mut emin = z[j4 + 4];
    let mut d = z[j4];
    let mut dmin = d;
    j4 = 4 * i0;
    // (qq, ee) slots and the next (q, e) for ping (pp = 0) and pong (pp = 1)
    let (q_out, e_in, e_out, q_next) = if pp == 0 { (2usize, 1usize, 0usize, 1usize) } else { (3, 0, 1, 2) };
    while j4 <= 4 * (n0 - 3) {
        let (qo, ei, eo, qn) = (j4 - q_out, j4 - e_in, j4 - e_out, j4 + q_next);
        z[qo] = d + z[ei];
        if z[qo] == 0.0 {
            z[eo] = 0.0;
            d = z[qn];
            dmin = d;
            emin = 0.0;
        } else if SAFMIN * z[qn] < z[qo] && SAFMIN * z[qo] < z[qn] {
            let temp = z[qn] / z[qo];
            z[eo] = z[ei] * temp;
            d *= temp;
        } else {
            z[eo] = z[qn] * (z[ei] / z[qo]);
            d = z[qn] * (d / z[qo]);
        }
        dmin = dmin.min(d);
        emin = emin.min(z[eo]);
        j4 += 4;
    }
    // the last two steps, unrolled
    let dnm2 = d;
    qd.dmin2 = dmin;
    let mut j4 = 4 * (n0 - 2) - pp;
    let step = |z: &mut [f64], j4: usize, dprev: f64, dmin: &mut f64, emin: &mut f64| -> f64 {
        let j4p2 = j4 + 2 * pp - 1;
        z[j4 - 2] = dprev + z[j4p2];
        if z[j4 - 2] == 0.0 {
            z[j4] = 0.0;
            let d = z[j4p2 + 2];
            *dmin = d;
            *emin = 0.0;
            d
        } else if SAFMIN * z[j4p2 + 2] < z[j4 - 2] && SAFMIN * z[j4 - 2] < z[j4p2 + 2] {
            let temp = z[j4p2 + 2] / z[j4 - 2];
            z[j4] = z[j4p2] * temp;
            dprev * temp
        } else {
            z[j4] = z[j4p2 + 2] * (z[j4p2] / z[j4 - 2]);
            z[j4p2 + 2] * (dprev / z[j4 - 2])
        }
    };
    let dnm1 = step(z, j4, dnm2, &mut dmin, &mut emin);
    dmin = dmin.min(dnm1);
    qd.dmin1 = dmin;
    j4 += 4;
    let dn = step(z, j4, dnm1, &mut dmin, &mut emin);
    dmin = dmin.min(dn);
    z[j4 + 2] = dn;
    z[4 * n0 - pp] = emin;
    (qd.dmin, qd.dn, qd.dn1, qd.dn2) = (dmin, dn, dnm1, dnm2);
}