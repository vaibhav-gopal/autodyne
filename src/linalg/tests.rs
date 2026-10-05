//! Tests for linear algebra: reconstructions, residuals and known answers.

use super::*;
use crate::testing::random_f64 as random;

fn arr(data: &[f64], shape: &[usize]) -> NdArray<f64> {
    NdArray::from_vec(data.to_vec(), shape).unwrap()
}

fn close(a: &[f64], b: &[f64], tol: f64) -> bool {
    a.len() == b.len() && a.iter().zip(b).all(|(x, y)| (x - y).abs() <= tol * (1.0 + y.abs()))
}

/// Deterministic values in [-1, 1).
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

#[test]
fn polynomial_fits_and_derivatives() {
    // numpy.polyfit(arange(6), [1, 2.5, 2, 4.5, 7, 9.5], 2)
    let x = [0.0, 1.0, 2.0, 3.0, 4.0, 5.0];
    let fit = polyfit(&x, &[1.0, 2.5, 2.0, 4.5, 7.0, 9.5], 2).unwrap();
    assert!(close(&fit, &[0.3035714285714289, 0.1535714285714276, 1.250000000000002], 1e-12));
    // an exact cubic is recovered, also far from the origin
    let p = [0.5, -2.0, 0.25, 3.0];
    let xs: Vec<f64> = (0..20).map(|i| 1000.0 + i as f64 * 0.1).collect();
    let ys: Vec<f64> = xs.iter().map(|&x| polyval(&p, x)).collect();
    let fit = polyfit(&xs, &ys, 3).unwrap();
    assert!(xs.iter().all(|&x| (polyval(&fit, x) - polyval(&p, x)).abs() < 1e-6 * polyval(&p, x).abs()));
    assert!(polyfit(&x, &[1.0], 1).is_err());
    assert_eq!(polyder(&[1.0, -3.0, 2.0, 5.0], 1), vec![3.0, -6.0, 2.0]);
    assert_eq!(polyder(&[1.0, -3.0, 2.0, 5.0], 2), vec![6.0, -6.0]);
    assert_eq!(polyder(&[4.0], 1), vec![0.0]);
}

/// The largest difference relative to the largest value.
fn spread(a: &[f64], b: &[f64]) -> f64 {
    assert_eq!(a.len(), b.len());
    let top = b.iter().fold(0.0f64, |m, v| m.max(v.abs())).max(f64::MIN_POSITIVE);
    a.iter().zip(b).fold(0.0f64, |m, (x, y)| m.max((x - y).abs())) / top
}

#[test]
fn values_only_paths_match_the_full_decompositions() {
    for (k, &(m, n)) in [(1, 1), (2, 2), (3, 3), (4, 7), (7, 4), (17, 17), (60, 33), (33, 60), (150, 150)].iter().enumerate() {
        let a = random(&[m, n], 100 + k as u64);
        let s = svdvals(a.view()).unwrap();
        let full = svd(a.view(), false).unwrap().s;
        assert!(s.windows(2).all(|w| w[0] >= w[1]), "{m}x{n}: not descending");
        assert!(spread(&s, &full) < 1e-13, "{m}x{n}: {}", spread(&s, &full));
        if m == n {
            let w = eigvalsh(a.view()).unwrap();
            let (want, _) = eigh(a.view()).unwrap();
            assert!(w.windows(2).all(|w| w[0] <= w[1]), "{n}: not ascending");
            assert!(spread(&w, &want) < 1e-13, "{n}: {}", spread(&w, &want));
        }
    }
    // f32 through the same path
    let a = random(&[40, 40], 7);
    let a32 = NdArray::from_vec(a.as_slice().iter().map(|&v| v as f32).collect(), &[40, 40]).unwrap();
    let s32: Vec<f64> = svdvals(a32.view()).unwrap().into_iter().map(f64::from).collect();
    assert!(spread(&s32, &svdvals(a.view()).unwrap()) < 1e-5);
    let w32: Vec<f64> = eigvalsh(a32.view()).unwrap().into_iter().map(f64::from).collect();
    assert!(spread(&w32, &eigvalsh(a.view()).unwrap()) < 1e-5);
    // strided input: the transpose
    assert!(spread(&svdvals(a.view().transpose()).unwrap(), &svdvals(a.view()).unwrap()) < 1e-14);
    assert!(svdvals(NdArray::<f64>::zeros(&[0, 3]).unwrap().view()).unwrap().is_empty());
    assert!(eigvalsh(NdArray::<f64>::zeros(&[0, 0]).unwrap().view()).unwrap().is_empty());
}

#[test]
fn values_only_paths_on_hard_inputs() {
    // diagonal (no off-diagonal work) and zero
    let d = [3.0, -1e-200, 0.0, 7.5, -2.0];
    let mut a = vec![0.0; 25];
    for i in 0..5 {
        a[i * 6] = d[i];
    }
    let a = arr(&a, &[5, 5]);
    assert_eq!(svdvals(a.view()).unwrap(), vec![7.5, 3.0, 2.0, 1e-200, 0.0]);
    assert_eq!(eigvalsh(a.view()).unwrap(), vec![-2.0, -1e-200, 0.0, 3.0, 7.5]);
    assert_eq!(svdvals(NdArray::<f64>::zeros(&[4, 4]).unwrap().view()).unwrap(), vec![0.0; 4]);
    // the identity (one repeated value), and a cluster
    let mut eye = vec![0.0; 36];
    (0..6).for_each(|i| eye[i * 7] = 1.0);
    assert!(close(&svdvals(arr(&eye, &[6, 6]).view()).unwrap(), &[1.0; 6], 1e-15));
    assert!(close(&eigvalsh(arr(&eye, &[6, 6]).view()).unwrap(), &[1.0; 6], 1e-15));
    // tiny and huge scales: the same values, scaled (faer's own SVD loses the tiny ones: its
    // reduction underflows without the scaling LAPACK applies first)
    let a = random(&[30, 30], 9);
    let base = svdvals(a.view()).unwrap();
    let base_w = eigvalsh(a.view()).unwrap();
    for scale in [1e-280, 1e-150, 1e150, 1e280] {
        let scaled = NdArray::from_vec(a.as_slice().iter().map(|v| v * scale).collect(), &[30, 30]).unwrap();
        let s: Vec<f64> = svdvals(scaled.view()).unwrap().iter().map(|v| v / scale).collect();
        assert!(spread(&s, &base) < 1e-13, "{scale}: {}", spread(&s, &base));
        let w: Vec<f64> = eigvalsh(scaled.view()).unwrap().iter().map(|v| v / scale).collect();
        assert!(spread(&w, &base_w) < 1e-13, "{scale}: {}", spread(&w, &base_w));
    }
    // a strongly graded bidiagonal: dqds finds every singular value to high relative accuracy,
    // so the product of them all matches |det| = product of the diagonal
    let n = 12;
    let mut diag: Vec<f64> = (0..n).map(|i| 10f64.powi(-(i as i32) * 5) * (1.0 + 0.1 * i as f64)).collect();
    let off: Vec<f64> = (0..n - 1).map(|i| 10f64.powi(-(i as i32) * 5 - 2)).collect();
    let det: f64 = diag.iter().map(|v| v.ln()).sum();
    values::bidiagonal_singular_values(&mut diag, &off).unwrap();
    let product: f64 = diag.iter().map(|v| v.ln()).sum();
    assert!((product - det).abs() < 1e-12, "{product} vs {det}");
    // the smallest is close to its diagonal entry (the off-diagonal coupling is tiny)
    assert!((diag[n - 1] / (10f64.powi(-(n as i32 - 1) * 5) * 2.1) - 1.0).abs() < 1e-3, "{}", diag[n - 1]);
    // 2 x 2: closed form
    let mut two = [4.0, 3.0];
    values::bidiagonal_singular_values(&mut two, &[2.0]).unwrap();
    // the eigenvalues of BᵀB for B = [[4, 2], [0, 3]]
    let gram = [[16.0, 8.0], [8.0, 13.0f64]];
    let (tr, det) = (gram[0][0] + gram[1][1], gram[0][0] * gram[1][1] - gram[0][1] * gram[1][0]);
    let disc = (tr * tr / 4.0 - det).sqrt();
    assert!(close(&two, &[(tr / 2.0 + disc).sqrt(), (tr / 2.0 - disc).sqrt()], 1e-15));
    // a Wilkinson matrix W21+ (pairs of nearly equal eigenvalues)
    let n = 21;
    let mut w = vec![0.0; n * n];
    for i in 0..n {
        w[i * n + i] = (10.0 - i as f64).abs();
        if i + 1 < n {
            w[i * n + i + 1] = 1.0;
            w[(i + 1) * n + i] = 1.0;
        }
    }
    let w = arr(&w, &[n, n]);
    let (want, _) = eigh(w.view()).unwrap();
    assert!(spread(&eigvalsh(w.view()).unwrap(), &want) < 1e-14);
    assert!((eigvalsh(w.view()).unwrap()[n - 1] - 10.746194182903322).abs() < 1e-13);
}
#[test]
fn values_only_paths_on_many_structures() {
    let mut seed = 1000;
    for m in [1, 2, 3, 5, 8, 13, 31, 64] {
        for n in [1, 2, 4, 9, 31, 64] {
            for kind in 0..4 {
                seed += 1;
                let mut a = random(&[m, n], seed);
                let data = a.as_mut_slice();
                match kind {
                    // graded rows
                    1 => (0..m).for_each(|i| (0..n).for_each(|j| data[i * n + j] *= 10f64.powi(-3 * i as i32))),
                    // rank one
                    2 => (0..m).for_each(|i| (0..n).for_each(|j| data[i * n + j] = (i as f64 + 1.0) * (j as f64 - 0.5))),
                    // repeated columns
                    3 => (0..m).for_each(|i| (1..n).for_each(|j| data[i * n + j] = data[i * n + j % 2])),
                    _ => {}
                }
                let s = svdvals(a.view()).unwrap();
                assert!(spread(&s, &svd(a.view(), false).unwrap().s) < 1e-13, "{m}x{n} kind {kind}");
                if m == n {
                    let (want, _) = eigh(a.view()).unwrap();
                    assert!(spread(&eigvalsh(a.view()).unwrap(), &want) < 1e-13, "{n} kind {kind}");
                }
            }
        }
    }
}
#[test]
fn small_products_match_the_naive_product() {
    // shapes the small kernel takes (multiples of 4 x 8 / 16) and shapes it leaves to faer
    for (m, k, n) in [(4, 1, 8), (4, 3, 16), (8, 64, 16), (64, 64, 64), (128, 128, 128), (12, 7, 32), (5, 5, 8), (64, 64, 63)] {
        let a = random(&[m, k], 3);
        let b = random(&[k, n], 4);
        let want = naive(&a, &b);
        let got = matmul(a.view(), b.view()).unwrap();
        assert!(close(got.as_slice(), &want, 1e-12), "f64 {m}x{k}x{n}");
        let a32 = NdArray::from_vec(a.as_slice().iter().map(|&v| v as f32).collect(), &[m, k]).unwrap();
        let b32 = NdArray::from_vec(b.as_slice().iter().map(|&v| v as f32).collect(), &[k, n]).unwrap();
        let got32 = matmul(a32.view(), b32.view()).unwrap();
        let back: Vec<f64> = got32.as_slice().iter().map(|&v| v as f64).collect();
        assert!(close(&back, &want, 1e-4), "f32 {m}x{k}x{n}");
    }
}