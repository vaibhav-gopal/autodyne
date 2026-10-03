//! The Rust side of `bench/linalg/compare.py`: the same matrices (read from `.npy`), the same
//! operations, timed in each Rust library, single-threaded and in f64. Each result is written back
//! as `.npy` for the script to check, and each timing is printed as one JSON line.
//!
//! ```text
//! cargo run --release --example linalg -- <dir with the inputs>
//! ```

use std::path::{Path, PathBuf};
use std::time::Instant;

use burn::backend::NdArray as BurnNd;
use burn::tensor::{linalg as burn_linalg, Tensor, TensorData};
use faer::linalg::solvers::Solve;
use faer::{linalg::matmul::matmul as faer_matmul, Accum, Mat, Par, Side};
use nalgebra::{DMatrix, DVector, SymmetricEigen};

type Burn = BurnNd<f64>;

// .npy (little-endian f64, C order) =================================================================

fn read_npy(path: &Path) -> (Vec<f64>, Vec<usize>) {
    let bytes = std::fs::read(path).unwrap_or_else(|e| panic!("{}: {e}", path.display()));
    let len = u16::from_le_bytes([bytes[8], bytes[9]]) as usize;
    let header = std::str::from_utf8(&bytes[10..10 + len]).unwrap();
    assert!(header.contains("'<f8'") && header.contains("'fortran_order': False"), "{}: expected C-order f64", path.display());
    let dims = header.split("'shape': (").nth(1).unwrap().split(')').next().unwrap();
    let shape: Vec<usize> = dims.split(',').map(str::trim).filter(|d| !d.is_empty()).map(|d| d.parse().unwrap()).collect();
    let data = bytes[10 + len..].chunks_exact(8).map(|b| f64::from_le_bytes(b.try_into().unwrap())).collect();
    (data, shape)
}

fn write_npy(path: &Path, data: &[f64]) {
    let mut header = format!("{{'descr': '<f8', 'fortran_order': False, 'shape': ({},), }}", data.len());
    let total = (10 + header.len() + 1).div_ceil(64) * 64;
    header.extend(std::iter::repeat_n(' ', total - 10 - header.len() - 1));
    header.push('\n');
    let mut out = b"\x93NUMPY\x01\x00".to_vec();
    out.extend_from_slice(&(header.len() as u16).to_le_bytes());
    out.extend_from_slice(header.as_bytes());
    data.iter().for_each(|v| out.extend_from_slice(&v.to_le_bytes()));
    std::fs::write(path, out).unwrap();
}

// timing and checks ==================================================================================

/// The best of repeated runs: at least 5, until a second has passed (at most 300).
fn best<R>(mut f: impl FnMut() -> R) -> (f64, R) {
    let start = Instant::now();
    let (mut best, mut last) = (f64::INFINITY, None);
    let mut runs = 0;
    while runs < 5 || (start.elapsed().as_secs_f64() < 1.0 && runs < 300) {
        let t = Instant::now();
        let r = f();
        best = best.min(t.elapsed().as_secs_f64());
        last = Some(r);
        runs += 1;
    }
    (best, last.unwrap())
}

struct Run {
    dir: PathBuf,
}

impl Run {
    fn record(&self, library: &str, op: &str, seconds: f64, result: &[f64]) {
        let file = format!("rust__{}__{}.npy", library, op.replace(' ', "_"));
        write_npy(&self.dir.join(&file), result);
        println!("{{\"library\": \"{library}\", \"op\": \"{op}\", \"seconds\": {seconds:e}, \"result\": \"{file}\"}}");
    }
}

/// `a b` for row-major n x n matrices (for residuals; not timed).
fn product(a: &[f64], b: &[f64], n: usize, k: usize, m: usize) -> Vec<f64> {
    let mut c = vec![0.0; n * m];
    for i in 0..n {
        for p in 0..k {
            let x = a[i * k + p];
            for j in 0..m {
                c[i * m + j] += x * b[p * m + j];
            }
        }
    }
    c
}

fn norm(v: &[f64]) -> f64 {
    v.iter().map(|x| x * x).sum::<f64>().sqrt()
}

/// `‖a - u diag(s) vt‖ / ‖a‖` (u n x n, vt n x n, row-major).
fn svd_residual(a: &[f64], u: &[f64], s: &[f64], vt: &[f64], n: usize) -> f64 {
    let us: Vec<f64> = (0..n * n).map(|k| u[k] * s[k % n]).collect();
    let r = product(&us, vt, n, n, n);
    norm(&a.iter().zip(&r).map(|(x, y)| x - y).collect::<Vec<_>>()) / norm(a)
}

/// `‖s v - v diag(w)‖ / ‖s‖` (v's columns are eigenvectors).
fn eigh_residual(s: &[f64], w: &[f64], v: &[f64], n: usize) -> f64 {
    let sv = product(s, v, n, n, n);
    let diff: Vec<f64> = (0..n * n).map(|k| sv[k] - v[k] * w[k % n]).collect();
    norm(&diff) / norm(s)
}

fn sorted(mut v: Vec<f64>) -> Vec<f64> {
    v.sort_by(|a, b| a.partial_cmp(b).unwrap());
    v
}

/// Complex eigenvalues as their sorted real parts then their sorted imaginary parts.
fn eig_key(re: Vec<f64>, im: Vec<f64>) -> Vec<f64> {
    [sorted(re), sorted(im)].concat()
}

// conversions ========================================================================================

fn faer_mat(data: &[f64], n: usize, m: usize) -> Mat<f64> {
    Mat::from_fn(n, m, |i, j| data[i * m + j])
}

fn faer_rows(m: faer::MatRef<'_, f64>) -> Vec<f64> {
    (0..m.nrows()).flat_map(|i| (0..m.ncols()).map(move |j| m[(i, j)])).collect()
}

fn nalgebra_mat(data: &[f64], n: usize, m: usize) -> DMatrix<f64> {
    DMatrix::from_row_slice(n, m, data)
}

fn nalgebra_rows(m: &DMatrix<f64>) -> Vec<f64> {
    (0..m.nrows()).flat_map(|i| (0..m.ncols()).map(move |j| m[(i, j)])).collect()
}

fn burn_mat(data: &[f64], shape: &[usize]) -> Tensor<Burn, 2> {
    Tensor::from_data(TensorData::new(data.to_vec(), shape.to_vec()), &Default::default())
}

fn main() {
    let dir = PathBuf::from(std::env::args().nth(1).expect("usage: linalg <dir>"));
    let run = Run { dir: dir.clone() };
    faer::set_global_parallelism(Par::Seq);
    autodyne::linalg::set_threads(1);
    let load = |name: &str| read_npy(&dir.join(format!("{name}.npy")));

    // matrix products
    for n in [64usize, 512] {
        let ((a, _), (b, _)) = (load(&format!("matmul_{n}_a")), load(&format!("matmul_{n}_b")));
        let op = format!("matmul {n}");
        let (av, bv) = (autodyne::signal::NdArray::from_vec(a.clone(), &[n, n]).unwrap(), autodyne::signal::NdArray::from_vec(b.clone(), &[n, n]).unwrap());
        let (t, c) = best(|| autodyne::linalg::matmul(av.view(), bv.view()).unwrap());
        run.record("autodyne", &op, t, c.as_slice());
        let (fa, fb) = (faer_mat(&a, n, n), faer_mat(&b, n, n));
        let mut fc = Mat::<f64>::zeros(n, n);
        let (t, _) = best(|| faer_matmul(fc.as_mut(), Accum::Replace, fa.as_ref(), fb.as_ref(), 1.0, Par::Seq));
        run.record("faer", &op, t, &faer_rows(fc.as_ref()));
        let (na, nb) = (nalgebra_mat(&a, n, n), nalgebra_mat(&b, n, n));
        let (t, c) = best(|| &na * &nb);
        run.record("nalgebra", &op, t, &nalgebra_rows(&c));
        let (ba, bb) = (burn_mat(&a, &[n, n]), burn_mat(&b, &[n, n]));
        let (t, c) = best(|| ba.clone().matmul(bb.clone()).into_data());
        run.record("burn (ndarray)", &op, t, &c.to_vec::<f64>().unwrap());
    }

    // a linear system and a determinant
    let ((a, _), (b, _)) = (load("solve_512_a"), load("solve_512_b"));
    let n = 512;
    let (av, bv) = (autodyne::signal::NdArray::from_vec(a.clone(), &[n, n]).unwrap(), autodyne::signal::NdArray::from_vec(b.clone(), &[n]).unwrap());
    let (t, x) = best(|| autodyne::linalg::solve(av.view(), bv.view()).unwrap());
    run.record("autodyne", "solve 512", t, x.as_slice());
    let (fa, fb) = (faer_mat(&a, n, n), Mat::from_fn(n, 1, |i, _| b[i]));
    let (t, x) = best(|| fa.partial_piv_lu().solve(&fb));
    run.record("faer", "solve 512", t, &faer_rows(x.as_ref()));
    let (na, nb) = (nalgebra_mat(&a, n, n), DVector::from_column_slice(&b));
    let (t, x) = best(|| na.clone().lu().solve(&nb).unwrap());
    run.record("nalgebra", "solve 512", t, x.as_slice());

    let (a, _) = load("square_300");
    let n = 300;
    let av = autodyne::signal::NdArray::from_vec(a.clone(), &[n, n]).unwrap();
    let (fa, na) = (faer_mat(&a, n, n), nalgebra_mat(&a, n, n));
    let (t, d) = best(|| autodyne::linalg::det(av.view()).unwrap());
    run.record("autodyne", "det 300", t, &[d]);
    let (t, d) = best(|| fa.determinant());
    run.record("faer", "det 300", t, &[d]);
    let (t, d) = best(|| na.clone().lu().determinant());
    run.record("nalgebra", "det 300", t, &[d]);
    let ba: Tensor<Burn, 3> = burn_mat(&a, &[n, n]).unsqueeze();
    let (t, d) = best(|| burn_linalg::det::<Burn, 3, 2, 1>(ba.clone()).into_data());
    run.record("burn (ndarray)", "det 300", t, &d.to_vec::<f64>().unwrap());

    // singular values, the full SVD, and the general eigenvalues
    let (t, s) = best(|| autodyne::linalg::svdvals(av.view()).unwrap());
    run.record("autodyne", "svdvals 300", t, &s);
    let (t, s) = best(|| fa.singular_values().unwrap());
    run.record("faer", "svdvals 300", t, &s);
    let (t, s) = best(|| na.clone().singular_values());
    run.record("nalgebra", "svdvals 300", t, &sorted(s.as_slice().to_vec()).into_iter().rev().collect::<Vec<_>>());

    let (t, r) = best(|| autodyne::linalg::svd(av.view(), true).unwrap());
    let res = svd_residual(&a, r.u.as_slice(), &r.s, r.vt.as_slice(), n);
    run.record("autodyne", "svd 300", t, &[r.s.clone(), vec![res]].concat());
    let (t, r) = best(|| fa.svd().unwrap());
    let s: Vec<f64> = r.S().column_vector().iter().copied().collect();
    let res = svd_residual(&a, &faer_rows(r.U()), &s, &faer_rows(r.V().transpose()), n);
    run.record("faer", "svd 300", t, &[s, vec![res]].concat());
    let (t, r) = best(|| na.clone().svd(true, true));
    let (u, vt) = (r.u.as_ref().unwrap(), r.v_t.as_ref().unwrap());
    let res = svd_residual(&a, &nalgebra_rows(u), r.singular_values.as_slice(), &nalgebra_rows(vt), n);
    run.record("nalgebra", "svd 300", t, &[sorted(r.singular_values.as_slice().to_vec()).into_iter().rev().collect(), vec![res]].concat());

    let (t, w) = best(|| autodyne::linalg::eigvals(av.view()).unwrap());
    run.record("autodyne", "eigvals 300", t, &eig_key(w.iter().map(|z| z.re).collect(), w.iter().map(|z| z.im).collect()));
    let (t, w) = best(|| fa.eigenvalues().unwrap());
    run.record("faer", "eigvals 300", t, &eig_key(w.iter().map(|z| z.re).collect(), w.iter().map(|z| z.im).collect()));
    let (t, w) = best(|| na.clone().complex_eigenvalues());
    run.record("nalgebra", "eigvals 300", t, &eig_key(w.iter().map(|z| z.re).collect(), w.iter().map(|z| z.im).collect()));

    // a symmetric matrix: eigenvalues with and without vectors
    let (s, _) = load("symmetric_300");
    let sv = autodyne::signal::NdArray::from_vec(s.clone(), &[n, n]).unwrap();
    let (fs, ns) = (faer_mat(&s, n, n), nalgebra_mat(&s, n, n));
    let (t, (w, v)) = best(|| autodyne::linalg::eigh(sv.view()).unwrap());
    let res = eigh_residual(&s, &w, v.as_slice(), n);
    run.record("autodyne", "eigh 300", t, &[w, vec![res]].concat());
    let (t, e) = best(|| fs.self_adjoint_eigen(Side::Lower).unwrap());
    let w: Vec<f64> = e.S().column_vector().iter().copied().collect();
    let res = eigh_residual(&s, &w, &faer_rows(e.U()), n);
    run.record("faer", "eigh 300", t, &[w, vec![res]].concat());
    let (t, e) = best(|| SymmetricEigen::new(ns.clone()));
    let mut order: Vec<usize> = (0..n).collect();
    order.sort_by(|&i, &j| e.eigenvalues[i].partial_cmp(&e.eigenvalues[j]).unwrap());
    let w: Vec<f64> = order.iter().map(|&i| e.eigenvalues[i]).collect();
    let vectors = &e.eigenvectors;
    let v: Vec<f64> = (0..n).flat_map(|r| order.iter().map(move |&c| vectors[(r, c)])).collect();
    run.record("nalgebra", "eigh 300", t, &[w, vec![eigh_residual(&s, &order.iter().map(|&i| e.eigenvalues[i]).collect::<Vec<_>>(), &v, n)]].concat());

    let (t, w) = best(|| autodyne::linalg::eigvalsh(sv.view()).unwrap());
    run.record("autodyne", "eigvalsh 300", t, &w);
    let (t, w) = best(|| fs.self_adjoint_eigenvalues(Side::Lower).unwrap());
    run.record("faer", "eigvalsh 300", t, &w);
    let (t, w) = best(|| ns.clone().symmetric_eigenvalues());
    run.record("nalgebra", "eigvalsh 300", t, &sorted(w.as_slice().to_vec()));

    // faer's phases: the reductions to bidiagonal / tridiagonal form, the rest is the iteration
    phases(&run, &fa, &fs);
}

/// Times faer's reduction steps on their own, for the breakdown of its SVD and symmetric
/// eigendecomposition.
fn phases(run: &Run, a: &Mat<f64>, s: &Mat<f64>) {
    use dyn_stack::{MemBuffer, MemStack, StackReq};
    use faer::linalg::evd::tridiag::{tridiag_in_place, tridiag_in_place_scratch};
    use faer::linalg::qr::no_pivoting::factor::recommended_block_size;
    use faer::linalg::svd::bidiag::{bidiag_in_place, bidiag_in_place_scratch};
    let n = a.nrows();
    let bs = recommended_block_size::<f64>(n, n);
    let mut buf = MemBuffer::new(StackReq::any_of(&[bidiag_in_place_scratch::<f64>(n, n, Par::Seq, Default::default()), tridiag_in_place_scratch::<f64>(n, Par::Seq, Default::default())]));
    let (t, _) = best(|| {
        let (mut m, mut hl, mut hr) = (a.clone(), Mat::<f64>::zeros(bs, n), Mat::<f64>::zeros(bs, n - 1));
        bidiag_in_place(m.as_mut(), hl.as_mut(), hr.as_mut(), Par::Seq, MemStack::new(&mut buf), Default::default());
        m
    });
    run.record("faer", "phase: bidiagonalization 300", t, &[]);
    let (t, _) = best(|| {
        let (mut m, mut h) = (s.clone(), Mat::<f64>::zeros(bs, n - 1));
        tridiag_in_place(m.as_mut(), h.as_mut(), Par::Seq, MemStack::new(&mut buf), Default::default());
        m
    });
    run.record("faer", "phase: tridiagonalization 300", t, &[]);
}
