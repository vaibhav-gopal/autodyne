//! autodyne against other Rust array / tensor libraries on shared axes of functionality:
//! the `ndarray` crate, Burn (its CPU backend `flex`, and `wgpu` on the GPU) and a hand-written
//! CubeCL kernel. Every case first checks that the libraries agree.
//!
//!     cargo bench --bench compare            (from bench/rust)
//!     python results.py                      writes RESULTS.md from criterion's estimates
//!
//! GPU cases come in two forms: "resident" (data already on the GPU, timed until the device is
//! idle) and "round trip" (upload, compute, download: what a CPU caller pays).

use std::hint::black_box;

use autodyne::filter::{design_lowpass, Fir};
use autodyne::gpu::GpuArray;
use autodyne::signal::{NdArray, NdView, Zip};
use burn::backend::{Flex, Wgpu};
use burn::tensor::backend::Backend;
use burn::tensor::module::conv1d;
use burn::tensor::ops::ConvOptions;
use burn::tensor::{Tensor, TensorData};
use criterion::{criterion_group, criterion_main, Criterion, Throughput};
use cubecl::bytes::Bytes;
use cubecl::prelude::*;
use cubecl::wgpu::{WgpuDevice, WgpuRuntime};

const ROWS: usize = 2_000;
const COLS: usize = 2_000;
const N: usize = ROWS * COLS;

fn data() -> Vec<f32> {
    (0..N).map(|i| ((i * 7_919) % 1_000) as f32 * 1e-3 - 0.5).collect()
}

fn close(a: &[f32], b: &[f32], tol: f32, what: &str) {
    assert_eq!(a.len(), b.len(), "{what}: lengths");
    let worst = a.iter().zip(b).map(|(x, y)| (x - y).abs() / y.abs().max(1.0)).fold(0.0f32, f32::max);
    assert!(worst <= tol, "{what}: worst relative difference {worst}");
}

fn burn_tensor<B: Backend>(v: &[f32], shape: [usize; 2], device: &B::Device) -> Tensor<B, 2> {
    Tensor::from_data(TensorData::new(v.to_vec(), shape), device)
}

fn to_vec<B: Backend, const D: usize>(t: Tensor<B, D>) -> Vec<f32> {
    t.into_data().to_vec::<f32>().expect("f32 data")
}

#[cube(launch_unchecked)]
fn axpb_kernel<N: Size>(x: &Array<Vector<f32, N>>, out: &mut Array<Vector<f32, N>>, a: f32, b: f32) {
    if ABSOLUTE_POS < x.len() {
        out[ABSOLUTE_POS] = x[ABSOLUTE_POS] * Vector::new(a) + Vector::new(b);
    }
}

/// Launches the CubeCL kernel over `len` elements already on the GPU.
fn launch_axpb(client: &ComputeClient<WgpuRuntime>, x: &cubecl::server::Handle, out: &cubecl::server::Handle, len: usize) {
    let vectorization = 4usize;
    let threads = 256u32;
    let units = (len / vectorization) as u32;
    let cubes = units.div_ceil(threads);
    unsafe {
        axpb_kernel::launch_unchecked::<WgpuRuntime>(
            client,
            CubeCount::Static(cubes, 1, 1),
            CubeDim::new_1d(threads),
            vectorization,
            ArrayArg::from_raw_parts(x.clone(), len),
            ArrayArg::from_raw_parts(out.clone(), len),
            2.0f32,
            0.5f32,
        )
    };
}

fn elementwise(c: &mut Criterion) {
    let v = data();
    let expected: Vec<f32> = v.iter().map(|x| 2.0 * x + 0.5).collect();
    let ours = NdArray::from_vec(v.clone(), &[ROWS, COLS]).unwrap();
    let nd = ndarray::Array2::from_shape_vec((ROWS, COLS), v.clone()).unwrap();
    let flex_device = Default::default();
    let flex = burn_tensor::<Flex>(&v, [ROWS, COLS], &flex_device);
    let gpu_device = Default::default();
    let gpu = burn_tensor::<Wgpu>(&v, [ROWS, COLS], &gpu_device);
    let client = WgpuRuntime::client(&WgpuDevice::default());
    let cube_in = client.create(Bytes::from_elems(v.clone()));
    let cube_out = client.empty(N * 4);

    // agreement
    close(Zip::from(ours.view()).map_collect(|&x| 2.0 * x + 0.5).as_slice(), &expected, 1e-6, "autodyne");
    close(nd.mapv(|x| 2.0 * x + 0.5).as_slice().unwrap(), &expected, 1e-6, "ndarray");
    close(&to_vec(flex.clone().mul_scalar(2.0).add_scalar(0.5)), &expected, 1e-6, "burn flex");
    close(&to_vec(gpu.clone().mul_scalar(2.0).add_scalar(0.5)), &expected, 1e-6, "burn wgpu");
    launch_axpb(&client, &cube_in, &cube_out, N);
    close(f32::from_bytes(&client.read_one(cube_out.clone()).unwrap()), &expected, 1e-6, "cubecl");

    let mut g = c.benchmark_group("a*x+b, 2000x2000 f32");
    g.throughput(Throughput::Elements(N as u64));
    g.bench_function("autodyne (CPU, 1 thread)", |b| b.iter(|| black_box(Zip::from(ours.view()).map_collect(|&x| 2.0 * x + 0.5))));
    g.bench_function("ndarray (CPU, 1 thread)", |b| b.iter(|| black_box(nd.mapv(|x| 2.0 * x + 0.5))));
    g.bench_function("burn flex (CPU)", |b| b.iter(|| black_box(to_vec(flex.clone().mul_scalar(2.0).add_scalar(0.5)))));
    g.bench_function("burn wgpu (GPU, resident)", |b| {
        b.iter(|| {
            let out = gpu.clone().mul_scalar(2.0).add_scalar(0.5);
            <Wgpu as Backend>::sync(&gpu_device).unwrap();
            black_box(out)
        })
    });
    g.bench_function("burn wgpu (GPU, round trip)", |b| b.iter(|| black_box(to_vec(burn_tensor::<Wgpu>(&v, [ROWS, COLS], &gpu_device).mul_scalar(2.0).add_scalar(0.5)))));
    g.bench_function("cubecl kernel (GPU, resident)", |b| {
        b.iter(|| {
            launch_axpb(&client, &cube_in, &cube_out, N);
            cubecl::future::block_on(client.sync()).unwrap();
        })
    });
    let ag = GpuArray::from_host(&ours.view()).unwrap();
    close(ag.axpb(2.0, 0.5).to_host().as_slice(), &expected, 1e-6, "autodyne gpu");
    g.bench_function("autodyne gpu (GPU, resident)", |b| {
        b.iter(|| {
            let out = ag.axpb(2.0, 0.5);
            autodyne::gpu::sync();
            black_box(out)
        })
    });
    g.bench_function("autodyne gpu (GPU, round trip)", |b| b.iter(|| black_box(GpuArray::from_host(&ours.view()).unwrap().axpb(2.0, 0.5).to_host())));
    g.bench_function("cubecl kernel (GPU, round trip)", |b| {
        b.iter(|| {
            let input = client.create(Bytes::from_elems(v.clone()));
            let out = client.empty(N * 4);
            launch_axpb(&client, &input, &out, N);
            black_box(client.read_one(out).unwrap())
        })
    });
    g.finish();

    // transposed input: each library keeps the input's memory order where it can
    let t_expected: Vec<f32> = nd.t().iter().map(|x| 2.0 * x + 0.5).collect();
    let in_order = |view: NdView<'_, f32>| {
        let order = view.memory_order();
        Zip::from(view.permute(&order[..view.ndim()]).unwrap()).map_collect(|&x| 2.0 * x + 0.5)
    };
    let mine = in_order(ours.view().transpose());
    close(&mine.view().transpose().to_vec(), &t_expected, 1e-6, "autodyne transposed");
    close(&nd.t().mapv(|x| 2.0 * x + 0.5).iter().copied().collect::<Vec<_>>(), &t_expected, 1e-6, "ndarray transposed");
    let mut g = c.benchmark_group("a*x+b on a transposed 2000x2000 f32");
    g.throughput(Throughput::Elements(N as u64));
    g.bench_function("autodyne (CPU, 1 thread)", |b| b.iter(|| black_box(in_order(ours.view().transpose()))));
    g.bench_function("ndarray (CPU, 1 thread)", |b| b.iter(|| black_box(nd.t().mapv(|x| 2.0 * x + 0.5))));
    g.bench_function("burn flex (CPU)", |b| b.iter(|| black_box(to_vec(flex.clone().transpose().mul_scalar(2.0).add_scalar(0.5)))));
    g.bench_function("burn wgpu (GPU, resident)", |b| {
        b.iter(|| {
            let out = gpu.clone().transpose().mul_scalar(2.0).add_scalar(0.5);
            <Wgpu as Backend>::sync(&gpu_device).unwrap();
            black_box(out)
        })
    });
    g.finish();

    // broadcasting a row
    let row: Vec<f32> = (0..COLS).map(|i| i as f32 * 1e-3).collect();
    let b_expected: Vec<f32> = v.iter().enumerate().map(|(i, x)| x + row[i % COLS]).collect();
    let our_row = NdArray::from_vec(row.clone(), &[COLS]).unwrap();
    let nd_row = ndarray::Array1::from_vec(row.clone());
    let flex_row: Tensor<Flex, 2> = Tensor::<Flex, 1>::from_data(TensorData::new(row.clone(), [COLS]), &flex_device).unsqueeze();
    let gpu_row: Tensor<Wgpu, 2> = Tensor::<Wgpu, 1>::from_data(TensorData::new(row.clone(), [COLS]), &gpu_device).unsqueeze();
    let add_row = || Zip::from(ours.view()).and_broadcast(our_row.view()).unwrap().map_collect(|&x, &r| x + r);
    close(add_row().as_slice(), &b_expected, 1e-6, "autodyne broadcast");
    close((&nd + &nd_row).as_slice().unwrap(), &b_expected, 1e-6, "ndarray broadcast");
    close(&to_vec(flex.clone() + flex_row.clone()), &b_expected, 1e-6, "burn broadcast");
    let mut g = c.benchmark_group("matrix + row (broadcast), 2000x2000 f32");
    g.throughput(Throughput::Elements(N as u64));
    g.bench_function("autodyne (CPU, 1 thread)", |b| b.iter(|| black_box(add_row())));
    g.bench_function("ndarray (CPU, 1 thread)", |b| b.iter(|| black_box(&nd + &nd_row)));
    g.bench_function("burn flex (CPU)", |b| b.iter(|| black_box(to_vec(flex.clone() + flex_row.clone()))));
    g.bench_function("burn wgpu (GPU, resident)", |b| {
        b.iter(|| {
            let out = gpu.clone() + gpu_row.clone();
            <Wgpu as Backend>::sync(&gpu_device).unwrap();
            black_box(out)
        })
    });
    let (ag, ag_row) = (GpuArray::from_host(&ours.view()).unwrap(), GpuArray::from_host(&our_row.view()).unwrap());
    close(ag.add(&ag_row).to_host().as_slice(), &b_expected, 1e-6, "autodyne gpu broadcast");
    g.bench_function("autodyne gpu (GPU, resident)", |b| {
        b.iter(|| {
            let out = ag.add(&ag_row);
            autodyne::gpu::sync();
            black_box(out)
        })
    });
    g.finish();
}

fn reductions(c: &mut Criterion) {
    let v = data();
    let ours = NdArray::from_vec(v.clone(), &[ROWS, COLS]).unwrap();
    let nd = ndarray::Array2::from_shape_vec((ROWS, COLS), v.clone()).unwrap();
    let flex_device = Default::default();
    let flex = burn_tensor::<Flex>(&v, [ROWS, COLS], &flex_device);
    let gpu_device = Default::default();
    let gpu = burn_tensor::<Wgpu>(&v, [ROWS, COLS], &gpu_device);

    let total: f64 = v.iter().map(|&x| x as f64).sum();
    assert!((ours.view().sum() as f64 - total).abs() < 1e-2 * total.abs().max(1.0));
    assert!((nd.sum() as f64 - total).abs() < 1.0 * total.abs().max(1.0));
    close(ours.sum_axis(0).unwrap().as_slice(), nd.sum_axis(ndarray::Axis(0)).as_slice().unwrap(), 1e-3, "column sums");
    close(&to_vec(flex.clone().sum_dim(0)), nd.sum_axis(ndarray::Axis(0)).as_slice().unwrap(), 1e-3, "burn column sums");
    close(&to_vec(gpu.clone().sum_dim(1)), ours.sum_axis(1).unwrap().as_slice(), 1e-3, "burn wgpu row sums");

    let ag = GpuArray::from_host(&ours.view()).unwrap();
    assert!((ag.sum().to_host().as_slice()[0] as f64 - total).abs() < 1e-2 * total.abs().max(1.0));
    close(ag.sum_axis(0).to_host().as_slice(), ours.sum_axis(0).unwrap().as_slice(), 1e-3, "autodyne gpu column sums");
    close(ag.sum_axis(1).to_host().as_slice(), ours.sum_axis(1).unwrap().as_slice(), 1e-3, "autodyne gpu row sums");
    for (name, axis) in [("sum of everything", None), ("column sums", Some(0usize)), ("row sums", Some(1usize))] {
        let mut g = c.benchmark_group(format!("{name}, 2000x2000 f32"));
        g.throughput(Throughput::Elements(N as u64));
        match axis {
            None => {
                g.bench_function("autodyne (CPU, 1 thread)", |b| b.iter(|| black_box(ours.view().sum())));
                g.bench_function("ndarray (CPU, 1 thread)", |b| b.iter(|| black_box(nd.sum())));
                g.bench_function("burn flex (CPU)", |b| b.iter(|| black_box(flex.clone().sum().into_scalar())));
                g.bench_function("burn wgpu (GPU, resident)", |b| b.iter(|| black_box(gpu.clone().sum().into_scalar())));
                // read back like Burn's into_scalar
                g.bench_function("autodyne gpu (GPU, resident)", |b| b.iter(|| black_box(ag.sum().to_host())));
            }
            Some(axis) => {
                g.bench_function("autodyne (CPU, 1 thread)", |b| b.iter(|| black_box(ours.sum_axis(axis).unwrap())));
                g.bench_function("ndarray (CPU, 1 thread)", |b| b.iter(|| black_box(nd.sum_axis(ndarray::Axis(axis)))));
                g.bench_function("burn flex (CPU)", |b| b.iter(|| black_box(to_vec(flex.clone().sum_dim(axis)))));
                g.bench_function("burn wgpu (GPU, resident)", |b| {
                    b.iter(|| {
                        let out = gpu.clone().sum_dim(axis);
                        <Wgpu as Backend>::sync(&gpu_device).unwrap();
                        black_box(out)
                    })
                });
                g.bench_function("autodyne gpu (GPU, resident)", |b| {
                    b.iter(|| {
                        let out = ag.sum_axis(axis);
                        autodyne::gpu::sync();
                        black_box(out)
                    })
                });
            }
        }
        g.finish();
    }
}

/// 16 channels x 48,000 samples through a 63-tap FIR each (Burn: a depthwise conv1d).
fn convolution(c: &mut Criterion) {
    let (channels, len) = (16usize, 48_000usize);
    let taps: Vec<f32> = design_lowpass(4_000.0, 63, 48_000.0);
    let signal: Vec<f32> = (0..channels * len).map(|i| (((i * 7_919) % 1_000) as f32 * 1e-3) - 0.5).collect();
    let mut ours = NdArray::from_vec(signal.clone(), &[channels, len]).unwrap();
    let mut firs: Vec<Fir<f32>> = (0..channels).map(|_| Fir::new(taps.clone())).collect();

    // Burn's conv1d is a cross-correlation: reverse the taps to compute the same convolution
    let reversed: Vec<f32> = taps.iter().rev().copied().collect();
    let flex_device = Default::default();
    let gpu_device = Default::default();
    let weight = |d| Tensor::<Flex, 3>::from_data(TensorData::new(reversed.repeat(channels), [channels, 1, taps.len()]), d);
    let input = |d| Tensor::<Flex, 3>::from_data(TensorData::new(signal.clone(), [1, channels, len]), d);
    let (flex_w, flex_x) = (weight(&flex_device), input(&flex_device));
    let gpu_w = Tensor::<Wgpu, 3>::from_data(TensorData::new(reversed.repeat(channels), [channels, 1, taps.len()]), &gpu_device);
    let gpu_x = Tensor::<Wgpu, 3>::from_data(TensorData::new(signal.clone(), [1, channels, len]), &gpu_device);
    let options = ConvOptions::new([1], [taps.len() - 1], [1], channels);

    // agreement on the first channel's fully overlapped samples
    ours.process_lanes(1, &mut firs).unwrap();
    let burn_out = to_vec(conv1d(flex_x.clone(), flex_w.clone(), None, options.clone()));
    let burn_first = &burn_out[..len]; // output length len + taps - 1 with this padding: the first len match a causal FIR
    close(&ours.as_slice()[100..len], &burn_first[100..len], 1e-4, "FIR vs conv1d");

    let mut g = c.benchmark_group("FIR / conv1d, 16 x 48000 f32, 63 taps");
    g.throughput(Throughput::Elements((channels * len) as u64));
    g.bench_function("autodyne Fir per lane (CPU, 1 thread)", |b| {
        b.iter(|| {
            ours.as_mut_slice().copy_from_slice(&signal);
            firs.iter_mut().for_each(|f| f.reset());
            ours.process_lanes(1, &mut firs).unwrap();
        })
    });
    g.bench_function("burn flex conv1d (CPU)", |b| b.iter(|| black_box(to_vec(conv1d(flex_x.clone(), flex_w.clone(), None, options.clone())))));
    let ag = GpuArray::from_host(&NdArray::from_vec(signal.clone(), &[channels, len]).unwrap().view()).unwrap();
    close(&ag.fir(&taps).to_host().as_slice()[..len], &ours.as_slice()[..len], 1e-4, "autodyne gpu fir");
    g.bench_function("autodyne gpu fir (GPU, resident)", |b| {
        b.iter(|| {
            let out = ag.fir(&taps);
            autodyne::gpu::sync();
            black_box(out)
        })
    });
    g.bench_function("burn wgpu conv1d (GPU, resident)", |b| {
        b.iter(|| {
            let out = conv1d(gpu_x.clone(), gpu_w.clone(), None, options.clone());
            <Wgpu as Backend>::sync(&gpu_device).unwrap();
            black_box(out)
        })
    });
    g.finish();
}

criterion_group! {
    name = benches;
    config = Criterion::default().sample_size(30).warm_up_time(std::time::Duration::from_secs(1)).measurement_time(std::time::Duration::from_secs(3));
    targets = elementwise, reductions, convolution
}
criterion_main!(benches);
