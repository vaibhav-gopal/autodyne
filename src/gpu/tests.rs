use super::*;

fn data(shape: &[usize], seed: u32) -> NdArray<f32> {
    let n = shape.iter().product();
    let mut s = seed;
    NdArray::from_vec(
        (0..n)
            .map(|_| {
                s = s.wrapping_mul(1_664_525).wrapping_add(1_013_904_223);
                (s >> 8) as f32 / (1u32 << 24) as f32 * 2.0 - 1.0
            })
            .collect(),
        shape,
    )
    .unwrap()
}

fn up(x: &NdArray<f32>) -> GpuArray<f32> {
    GpuArray::from_host(&x.view()).unwrap()
}

fn close(got: &NdArray<f32>, want: &[f32], tol: f32, what: &str) {
    assert_eq!(got.len(), want.len(), "{what}: length");
    for (i, (a, b)) in got.as_slice().iter().zip(want).enumerate() {
        assert!((a - b).abs() <= tol * (1.0 + b.abs()), "{what}: element {i}: {a} vs {b}");
    }
}

#[test]
fn no_gpu_is_an_error() {
    if !available() {
        let x = data(&[4], 0);
        assert_eq!(GpuArray::from_host(&x.view()).unwrap_err(), GpuError::NoDevice);
    }
}

#[test]
fn element_wise_matches_the_cpu() {
    if !available() {
        return;
    }
    for shape in [vec![7], vec![3, 4], vec![5, 7], vec![64, 33]] {
        let (x, y) = (data(&shape, 1), data(&shape, 2));
        let (gx, gy) = (up(&x), up(&y));
        let xs = x.as_slice();
        close(&gx.axpb(2.0, 0.5).to_host(), &xs.iter().map(|v| 2.0 * v + 0.5).collect::<Vec<_>>(), 1e-6, "axpb");
        let ys = y.as_slice();
        for (name, got, f) in [
            ("add", gx.add(&gy), (|a: f32, b: f32| a + b) as fn(f32, f32) -> f32),
            ("sub", gx.sub(&gy), |a, b| a - b),
            ("mul", gx.mul(&gy), |a, b| a * b),
            ("div", gx.div(&gy), |a, b| a / b),
        ] {
            close(&got.to_host(), &xs.iter().zip(ys).map(|(&a, &b)| f(a, b)).collect::<Vec<_>>(), 1e-5, name);
        }
        for (name, got, f) in [
            ("exp", gx.exp(), (|a: f32| a.exp()) as fn(f32) -> f32),
            ("tanh", gx.tanh(), |a| a.tanh()),
            ("sin", gx.sin(), |a| a.sin()),
            ("cos", gx.cos(), |a| a.cos()),
            ("abs", gx.abs(), |a| a.abs()),
            ("neg", gx.neg(), |a| -a),
            ("sqrt", gx.abs().sqrt(), |a| a.abs().sqrt()),
            ("ln", gx.abs().ln(), |a| a.abs().ln()),
        ] {
            close(&got.to_host(), &xs.iter().map(|&a| f(a)).collect::<Vec<_>>(), 1e-4, name);
        }
        for (name, got, f) in [
            ("x + s", gx.add_scalar(1.5), (|a: f32| a + 1.5) as fn(f32) -> f32),
            ("x - s", gx.sub_scalar(1.5), |a| a - 1.5),
            ("x * s", gx.mul_scalar(1.5), |a| a * 1.5),
            ("x / s", gx.div_scalar(3.0), |a| a / 3.0),
            ("s - x", gx.scalar_sub(1.5), |a| 1.5 - a),
            ("s / x", gx.scalar_div(3.0), |a| 3.0 / a),
        ] {
            // (Vulkan's f32 division is good to 2.5 ulps, not correctly rounded)
            close(&got.to_host(), &xs.iter().map(|&a| f(a)).collect::<Vec<_>>(), 1e-6, name);
        }
        assert!(gx.can_combine(&gy) && !gx.can_combine(&up(&data(&[3], 9))) == (shape.last() != Some(&3)));
        if shape.len() == 2 {
            let (rows, cols) = (shape[0], shape[1]);
            let row = data(&[cols], 3);
            let col = data(&[rows, 1], 4);
            close(&gx.mul(&up(&row)).to_host(), &(0..rows * cols).map(|k| xs[k] * row.as_slice()[k % cols]).collect::<Vec<_>>(), 1e-6, "row broadcast");
            close(&gx.sub(&up(&col)).to_host(), &(0..rows * cols).map(|k| xs[k] - col.as_slice()[k / cols]).collect::<Vec<_>>(), 1e-6, "column broadcast");
        }
    }
}

#[test]
fn views_match_the_cpu() {
    if !available() {
        return;
    }
    for (rows, cols) in [(37, 20), (16, 12), (5, 5)] {
        let x = data(&[rows, cols], 11);
        let t = up(&x).transpose();
        assert_eq!(t.shape(), [cols, rows]);
        assert!(!t.is_standard());
        let xt = x.view().transpose().to_owned();
        assert_eq!(t.to_host(), xt);
        let xs = xt.as_slice();
        // element-wise work keeps the view; a transposed host view goes up as it lies
        close(&t.axpb(2.0, 1.0).to_host(), &xs.iter().map(|v| 2.0 * v + 1.0).collect::<Vec<_>>(), 1e-6, "transposed axpb");
        let lifted = GpuArray::from_host(&x.view().transpose()).unwrap();
        assert!(!lifted.is_standard());
        assert_eq!(lifted.to_host(), xt);
        // mixed layouts
        let y = data(&[cols, rows], 12);
        let sum: Vec<f32> = xs.iter().zip(y.as_slice()).map(|(a, b)| a + b).collect();
        close(&t.add(&up(&y)).to_host(), &sum, 1e-6, "transposed + standard");
        close(&up(&y).add(&t).to_host(), &sum, 1e-6, "standard + transposed");
        // broadcasts and sums on the transposed view (cols x rows)
        let row = data(&[rows], 13);
        let col = data(&[cols, 1], 14);
        close(&t.mul(&up(&row)).to_host(), &(0..rows * cols).map(|k| xs[k] * row.as_slice()[k % rows]).collect::<Vec<_>>(), 1e-6, "row on transposed");
        close(&t.sub(&up(&col)).to_host(), &(0..rows * cols).map(|k| xs[k] - col.as_slice()[k / rows]).collect::<Vec<_>>(), 1e-6, "column on transposed");
        close(&t.mul(&up(&row).permute(&[0])).to_host(), &(0..rows * cols).map(|k| xs[k] * row.as_slice()[k % rows]).collect::<Vec<_>>(), 1e-6, "row on transposed again");
        let row_sums: Vec<f32> = (0..cols).map(|i| xs[i * rows..(i + 1) * rows].iter().sum()).collect();
        let col_sums: Vec<f32> = (0..rows).map(|j| (0..cols).map(|i| xs[i * rows + j]).sum()).collect();
        close(&t.sum_axis(1).to_host(), &row_sums, 1e-4, "row sums of transposed");
        close(&t.sum_axis(0).to_host(), &col_sums, 1e-4, "column sums of transposed");
        let c = t.contiguous();
        assert!(c.is_standard());
        assert_eq!(c.to_host(), xt);
        assert_eq!(t.transpose().to_host(), x);
    }
    // permuting a 3-D array
    let z = data(&[2, 3, 4], 15);
    let want = z.view().permute(&[2, 0, 1]).unwrap().to_owned();
    let gz = up(&z).permute(&[2, 0, 1]);
    assert_eq!(gz.shape(), [4, 2, 3]);
    assert_eq!(gz.to_host(), want);
    assert_eq!(gz.contiguous().to_host(), want);
    close(&gz.exp().to_host(), &want.as_slice().iter().map(|v| v.exp()).collect::<Vec<_>>(), 1e-6, "exp of a permuted view");
}

#[test]
fn reductions_match_the_cpu() {
    if !available() {
        return;
    }
    for n in [1, 5, 1000, 1023, 300_000, 4_000_001] {
        let x = data(&[n], 5);
        let want: f64 = x.as_slice().iter().map(|&v| v as f64).sum();
        let got = up(&x).sum().to_host().as_slice()[0] as f64;
        assert!((got - want).abs() <= 1e-4 * (n as f64).sqrt().max(1.0), "sum of {n}: {got} vs {want}");
    }
    for (rows, cols) in [(1, 1), (3, 5), (100, 64), (2000, 2000), (513, 7)] {
        let x = data(&[rows, cols], 6);
        let g = up(&x);
        let xs = x.as_slice();
        let row_sums: Vec<f32> = (0..rows).map(|i| xs[i * cols..(i + 1) * cols].iter().sum()).collect();
        let col_sums: Vec<f32> = (0..cols).map(|j| (0..rows).map(|i| xs[i * cols + j]).sum()).collect();
        close(&g.sum_axis(1).to_host(), &row_sums, 1e-4, "row sums");
        close(&g.sum_axis(0).to_host(), &col_sums, 1e-4, "column sums");
    }
}

#[test]
fn fir_matches_the_cpu() {
    if !available() {
        return;
    }
    let (lanes, len) = (3, 1000);
    let x = data(&[lanes, len], 7);
    let taps = crate::filter::design_lowpass(2_000.0f32, 63, 48_000.0);
    let mut want = Vec::new();
    for l in 0..lanes {
        let mut f = crate::filter::Fir::new(taps.clone());
        let mut lane = x.as_slice()[l * len..(l + 1) * len].to_vec();
        f.process(&mut lane);
        want.extend(lane);
    }
    close(&up(&x).fir(&taps).to_host(), &want, 1e-5, "fir");
    // lanes along the last axis of a transposed view
    let xt = x.view().transpose().to_owned();
    close(&up(&xt).transpose().fir(&taps).to_host(), &want, 1e-5, "fir of a transposed view");
}

#[test]
fn double_precision_where_supported() {
    if !available() {
        return;
    }
    let x = NdArray::from_vec((0..1000).map(|i| (i as f64 * 0.1).sin()).collect(), &[10, 100]).unwrap();
    match GpuArray::from_host(&x.view()) {
        Ok(g) => {
            assert!(supports::<f64>());
            let want: f64 = x.as_slice().iter().map(|v| 2.0 * v + 0.5f64).sum();
            let got = g.axpb(2.0, 0.5).sum().to_host().as_slice()[0];
            assert!((got - want).abs() < 1e-9 * want.abs().max(1.0), "{got} vs {want}");
            // the transcendentals built from arithmetic, to within a few ulps
            let wide = NdArray::from_vec((0..4000).map(|i| (i as f64 - 2000.0) * 0.37 + 0.001).collect(), &[4000]).unwrap();
            let gw = GpuArray::from_host(&wide.view()).unwrap();
            let positive = NdArray::from_vec([1e-310, 1e-300, 1e-20, 0.5, 0.75, 1.0, 1.5, 2.0, 3.0, 10.0, 1e20, 1e300].to_vec(), &[12]).unwrap();
            let gp = GpuArray::from_host(&positive.view()).unwrap();
            for (name, got, input, f) in [
                ("exp", gw.axpb(0.002, 0.0).exp(), wide.as_slice().iter().map(|v| v * 0.002).collect::<Vec<_>>(), f64::exp as fn(f64) -> f64),
                ("exp wide", gw.exp(), wide.as_slice().to_vec(), f64::exp),
                ("tanh", gw.axpb(0.01, 0.0).tanh(), wide.as_slice().iter().map(|v| v * 0.01).collect(), f64::tanh),
                ("sin", gw.sin(), wide.as_slice().to_vec(), f64::sin),
                ("cos", gw.cos(), wide.as_slice().to_vec(), f64::cos),
                ("ln", gw.abs().ln(), wide.as_slice().iter().map(|v| v.abs()).collect(), f64::ln),
                ("ln of extremes", gp.ln(), positive.as_slice().to_vec(), f64::ln),
            ] {
                for (i, (a, &v)) in got.to_host().as_slice().iter().zip(&input).enumerate() {
                    let b = f(v);
                    let ok = if b.is_finite() { (a - b).abs() <= 4.0 * f64::EPSILON * b.abs().max(f64::MIN_POSITIVE) + 1e-300 } else { *a == b };
                    assert!(ok, "{name}: element {i} ({v}): {a} vs {b}");
                }
            }
            let special = NdArray::from_vec(vec![0.0, -1.0, f64::INFINITY, 1000.0, -1000.0], &[5]).unwrap();
            let gs = GpuArray::from_host(&special.view()).unwrap();
            let ln = gs.ln().to_host();
            assert!(ln.as_slice()[0] == f64::NEG_INFINITY && ln.as_slice()[1].is_nan() && ln.as_slice()[2] == f64::INFINITY);
            let exp = gs.exp().to_host();
            assert!(exp.as_slice()[2] == f64::INFINITY && exp.as_slice()[3] == f64::INFINITY && exp.as_slice()[4] == 0.0);
            let t = g.transpose();
            let col_sums: Vec<f64> = (0..100).map(|j| (0..10).map(|i| x.as_slice()[i * 100 + j]).sum()).collect();
            let got = t.sum_axis(1).to_host();
            for (a, b) in got.as_slice().iter().zip(&col_sums) {
                assert!((a - b).abs() < 1e-12, "{a} vs {b}");
            }
        }
        Err(e) => assert_eq!(e, GpuError::Unsupported("f64")),
    }
}
