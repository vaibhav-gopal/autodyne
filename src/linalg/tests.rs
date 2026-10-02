use super::*;

fn arr(data: &[f64], shape: &[usize]) -> NdArray<f64> {
    NdArray::from_vec(data.to_vec(), shape).unwrap()
}

fn close(a: &[f64], b: &[f64], tol: f64) -> bool {
    a.len() == b.len() && a.iter().zip(b).all(|(x, y)| (x - y).abs() <= tol * (1.0 + y.abs()))
}

/// Deterministic values in [-1, 1).
fn random(shape: &[usize], seed: u64) -> NdArray<f64> {
    let mut state = seed;
    let n = shape.iter().product();
    let data = (0..n)
        .map(|_| {
            state = state.wrapping_mul(6_364_136_223_846_793_005).wrapping_add(1_442_695_040_888_963_407);
            (state >> 11) as f64 / (1u64 << 53) as f64 * 2.0 - 1.0
        })
        .collect();
    NdArray::from_vec(data, shape).unwrap()
}

fn naive(a: &NdArray<f64>, b: &NdArray<f64>) -> Vec<f64> {
    let (m, k, n) = (a.shape()[0], a.shape()[1], b.shape()[1]);
    let mut out = vec![0.0; m * n];
    for i in 0..m {
        for j in 0..n {
            out[i * n + j] = (0..k).map(|p| a.as_slice()[i * k + p] * b.as_slice()[p * n + j]).sum();
        }
    }
    out
}

#[test]
fn matmul_shapes_and_strided_views() {
    let a = random(&[5, 7], 1);
    let b = random(&[7, 3], 2);
    let c = matmul(a.view(), b.view()).unwrap();
    assert_eq!(c.shape(), &[5, 3]);
    assert!(close(c.as_slice(), &naive(&a, &b), 1e-12));
    // a transposed view is read in place: (bᵀ aᵀ)ᵀ = a b
    let ct = matmul(b.view().transpose(), a.view().transpose()).unwrap();
    assert!(close(ct.view().transpose().to_vec().as_slice(), &naive(&a, &b), 1e-12));
    // reversed rows (negative stride)
    let rev = matmul(a.view().flip(0).unwrap(), b.view()).unwrap();
    let mut expected = naive(&a, &b);
    expected = expected.chunks(3).rev().flatten().copied().collect();
    assert!(close(rev.as_slice(), &expected, 1e-12));
    // matrix-vector, vector-matrix, vector-vector
    let x = random(&[7], 3);
    assert_eq!(matmul(a.view(), x.view()).unwrap().shape(), &[5]);
    assert_eq!(matmul(x.view(), b.view()).unwrap().shape(), &[3]);
    let dot = matmul(x.view(), x.view()).unwrap();
    assert_eq!(dot.shape(), &[] as &[usize]);
    assert!((dot.as_slice()[0] - x.as_slice().iter().map(|v| v * v).sum::<f64>()).abs() < 1e-12);
    assert!(matches!(matmul(a.view(), a.view()), Err(LinalgError::Mismatch(..))));
}

#[test]
fn solve_inverse_determinant() {
    let a = arr(&[4.0, -2.0, 1.0, -2.0, 4.0, -2.0, 1.0, -2.0, 4.0], &[3, 3]);
    let b = arr(&[11.0, -16.0, 17.0], &[3]);
    let x = solve(a.view(), b.view()).unwrap();
    assert!(close(x.as_slice(), &[1.0, -2.0, 3.0], 1e-12));
    let ai = inv(a.view()).unwrap();
    let eye = matmul(a.view(), ai.view()).unwrap();
    assert!(close(eye.as_slice(), &[1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0], 1e-12));
    assert!((det(a.view()).unwrap() - 36.0).abs() < 1e-10);
    // several right-hand sides at once
    let bb = arr(&[11.0, 1.0, -16.0, 2.0, 17.0, 3.0], &[3, 2]);
    let xx = solve(a.view(), bb.view()).unwrap();
    assert_eq!(xx.shape(), &[3, 2]);
    assert!(close(&[xx.as_slice()[0], xx.as_slice()[2], xx.as_slice()[4]], &[1.0, -2.0, 3.0], 1e-12));
    let singular = arr(&[1.0, 2.0, 2.0, 4.0], &[2, 2]);
    assert_eq!(solve(singular.view(), arr(&[1.0, 2.0], &[2]).view()), Err(LinalgError::Singular));
    assert_eq!(inv(arr(&[1.0, 2.0, 3.0, 4.0, 5.0, 6.0], &[2, 3]).view()), Err(LinalgError::NotSquare(vec![2, 3])));
}

#[test]
fn least_squares_fits_a_line_and_handles_rank_deficiency() {
    // y = 2x + 1 sampled exactly
    let xs = [0.0, 1.0, 2.0, 3.0, 4.0];
    let a = arr(&xs.iter().flat_map(|&x| [x, 1.0]).collect::<Vec<_>>(), &[5, 2]);
    let y = arr(&xs.iter().map(|x| 2.0 * x + 1.0).collect::<Vec<_>>(), &[5]);
    let fit = lstsq(a.view(), y.view()).unwrap();
    assert!(close(fit.solution.as_slice(), &[2.0, 1.0], 1e-12));
    assert_eq!(fit.rank, 2);
    // a duplicated column: rank 1, the minimum-norm solution splits the weight evenly
    let dup = arr(&[1.0, 1.0, 2.0, 2.0, 3.0, 3.0], &[3, 2]);
    let fit = lstsq(dup.view(), arr(&[2.0, 4.0, 6.0], &[3]).view()).unwrap();
    assert_eq!(fit.rank, 1);
    assert!(close(fit.solution.as_slice(), &[1.0, 1.0], 1e-12));
}

#[test]
fn qr_cholesky_reconstruct() {
    let a = random(&[6, 4], 7);
    let (q, r) = qr(a.view()).unwrap();
    assert_eq!((q.shape(), r.shape()), (&[6usize, 4][..], &[4usize, 4][..]));
    assert!(close(matmul(q.view(), r.view()).unwrap().as_slice(), a.as_slice(), 1e-12));
    let qtq = matmul(q.view().transpose(), q.view()).unwrap();
    let eye: Vec<f64> = (0..16).map(|i| if i % 5 == 0 { 1.0 } else { 0.0 }).collect();
    assert!(close(qtq.as_slice(), &eye, 1e-12));
    assert!((0..4).all(|i| (0..i).all(|j| r.as_slice()[i * 4 + j] == 0.0)), "r is upper triangular");

    let spd = matmul(a.view().transpose(), a.view()).unwrap();
    let l = cholesky(spd.view()).unwrap();
    assert!(close(matmul(l.view(), l.view().transpose()).unwrap().as_slice(), spd.as_slice(), 1e-12));
    assert_eq!(cholesky(arr(&[1.0, 2.0, 2.0, 1.0], &[2, 2]).view()), Err(LinalgError::NotPositiveDefinite));
}

#[test]
fn eigenvalues_general_and_symmetric() {
    // a rotation by 90 degrees: eigenvalues ±i
    let rot = arr(&[0.0, -1.0, 1.0, 0.0], &[2, 2]);
    let mut vals = eigvals(rot.view()).unwrap();
    vals.sort_by(|a, b| a.im.partial_cmp(&b.im).unwrap());
    assert!((vals[0] - Complex::new(0.0, -1.0)).norm() < 1e-12 && (vals[1] - Complex::new(0.0, 1.0)).norm() < 1e-12);
    // A v = λ v for each eigenpair of a random matrix
    let a = random(&[5, 5], 9);
    let e = eig(a.view()).unwrap();
    for (k, &lambda) in e.values.iter().enumerate() {
        for i in 0..5 {
            let av: Complex<f64> = (0..5).fold(Complex::zero(), |s, j| s + e.vectors.as_slice()[j * 5 + k] * a.as_slice()[i * 5 + j]);
            assert!((av - e.vectors.as_slice()[i * 5 + k] * lambda).norm() < 1e-10, "eigenpair {k}");
        }
    }
    // symmetric: real eigenvalues 1 and 3 for [[2, 1], [1, 2]]
    let (w, v) = eigh(arr(&[2.0, 1.0, 1.0, 2.0], &[2, 2]).view()).unwrap();
    assert!(close(&w, &[1.0, 3.0], 1e-12));
    assert!((v.as_slice()[0].abs() - 0.5f64.sqrt()).abs() < 1e-12);
}

#[test]
fn svd_reconstructs_and_pinv_inverts() {
    let a = random(&[4, 6], 11);
    let Svd { u, s, vt } = svd(a.view(), false).unwrap();
    assert_eq!((u.shape(), s.len(), vt.shape()), (&[4usize, 4][..], 4, &[4usize, 6][..]));
    let mut us = u.clone();
    for row in us.as_mut_slice().chunks_mut(4) {
        for (x, sj) in row.iter_mut().zip(&s) {
            *x *= sj;
        }
    }
    assert!(close(matmul(us.view(), vt.view()).unwrap().as_slice(), a.as_slice(), 1e-12));
    assert!(s.windows(2).all(|w| w[0] >= w[1]));
    assert!(close(&svdvals(a.view()).unwrap(), &s, 1e-12));
    let full = svd(a.view(), true).unwrap();
    assert_eq!((full.u.shape(), full.vt.shape()), (&[4usize, 4][..], &[6usize, 6][..]));
    // a · pinv(a) · a = a
    let p = pinv(a.view()).unwrap();
    let apa = matmul(matmul(a.view(), p.view()).unwrap().view(), a.view()).unwrap();
    assert!(close(apa.as_slice(), a.as_slice(), 1e-12));
    assert_eq!(matrix_rank(a.view()).unwrap(), 4);
    assert!((cond(arr(&[2.0, 0.0, 0.0, 0.5], &[2, 2]).view()).unwrap() - 4.0).abs() < 1e-12);
}

#[test]
fn polynomial_roots_and_expansion() {
    // (x - 1)(x - 2)(x + 3) = x³ - 7x + 6
    let mut r = roots(&[1.0, 0.0, -7.0, 6.0]).unwrap();
    r.sort_by(|a, b| a.re.partial_cmp(&b.re).unwrap());
    assert!(close(&r.iter().map(|z| z.re).collect::<Vec<_>>(), &[-3.0, 1.0, 2.0], 1e-12));
    assert!(r.iter().all(|z| z.im.abs() < 1e-12));
    // x² + 1: ±i; leading and trailing zeros
    let r = roots(&[0.0, 1.0, 0.0, 1.0, 0.0]).unwrap();
    assert_eq!(r.len(), 3);
    assert!(r.iter().filter(|z| (z.norm() - 1.0).abs() < 1e-12 && z.re.abs() < 1e-12).count() == 2);
    assert!(r.iter().any(|z| z.norm() < 1e-15));
    // round trip through the roots
    let p = [2.0, -3.0, 0.5, 4.0];
    let back: Vec<f64> = poly(&roots(&p).unwrap()).into_iter().map(|c| c * 2.0).collect();
    assert!(close(&back, &p, 1e-12));
    assert_eq!(polyval(&[1.0, -3.0, 2.0], 3.0), 2.0);
    assert_eq!(polymul(&[1.0, 1.0], &[1.0, -1.0]), vec![1.0, 0.0, -1.0]);
    assert_eq!(polyadd(&[1.0, 2.0, 3.0], &[1.0, 1.0]), vec![1.0, 3.0, 4.0]);
}
