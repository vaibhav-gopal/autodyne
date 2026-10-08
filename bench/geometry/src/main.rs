//! autodyne's geometry against glam, on what parda-2d calls: single-value products and inverses
//! (f32 and f64) and the batched world-matrix pass.
//!
//!     cd bench/geometry && cargo run --release [-- --out RESULTS.md]
//!
//! Every case first checks that the two libraries agree, then times each over the same inputs in
//! alternating rounds and keeps each side's best (background load on a desktop swings single runs
//! by tens of percent; the best of many rounds is the machine's speed).

use std::hint::black_box;
use std::time::Instant;

use autodyne::geometry::{compose_world, Mat3, Mat4, Vec3, Vec4, NO_PARENT};

/// Elements per timed pass (independent operations: throughput, as a transform or hit-test pass).
const N: usize = 1024;
/// Alternating rounds per case.
const ROUNDS: usize = 101;

/// A deterministic stream in [-1, 1).
struct Rng(u64);
impl Rng {
    fn next(&mut self) -> f64 {
        self.0 ^= self.0 << 13;
        self.0 ^= self.0 >> 7;
        self.0 ^= self.0 << 17;
        (self.0 >> 11) as f64 / (1u64 << 53) as f64 * 2.0 - 1.0
    }
}

/// Transforms as a UI or scene has them: translation, rotation about a random axis order, scale,
/// and a small projective bottom row so every cofactor matters. Column-major.
fn transforms(n: usize, seed: u64) -> Vec<[f64; 16]> {
    let mut r = Rng(seed);
    (0..n)
        .map(|_| {
            let m = Mat4::from_translation(Vec3::new(r.next() * 100.0, r.next() * 100.0, r.next()))
                * Mat4::from_rotation_z(r.next() * 3.0)
                * Mat4::from_rotation_x(r.next() * 0.5)
                * Mat4::from_scale(Vec3::new(0.5 + r.next().abs() * 2.0, 0.5 + r.next().abs() * 2.0, 1.0));
            let mut a = m.to_cols_array();
            a[3] = r.next() * 0.01;
            a[7] = r.next() * 0.01;
            a
        })
        .collect()
}

fn ad4<T: autodyne::units::Float>(a: &[f64; 16]) -> Mat4<T> {
    let v = |i: usize| Vec4::new(T::lit(a[i]), T::lit(a[i + 1]), T::lit(a[i + 2]), T::lit(a[i + 3]));
    Mat4::from_cols(v(0), v(4), v(8), v(12))
}

fn ad3<T: autodyne::units::Float>(a: &[f64; 16]) -> Mat3<T> {
    // the 2D affine part: columns x, y and translation, as parda's homographies
    let v = |i: usize| Vec3::new(T::lit(a[i]), T::lit(a[i + 1]), T::lit(a[i + 3]));
    Mat3::from_cols(v(0), v(4), v(12))
}

/// Best seconds per pass of `f` (which does one pass over the inputs), timing `reps` passes per sample.
fn sample(f: &mut dyn FnMut(), reps: usize) -> f64 {
    let t = Instant::now();
    for _ in 0..reps {
        f();
    }
    t.elapsed().as_secs_f64() / reps as f64
}

/// Times two passes alternately; returns each side's best time per pass.
fn race(mut ours: impl FnMut(), mut theirs: impl FnMut()) -> (f64, f64) {
    // passes per sample: about 4 ms of work
    let one = sample(&mut ours, 3).max(1e-9);
    let reps = ((4e-3 / one) as usize).clamp(1, 100_000);
    let (mut a, mut b) = (f64::MAX, f64::MAX);
    for round in 0..ROUNDS {
        if round % 2 == 0 {
            a = a.min(sample(&mut ours, reps));
            b = b.min(sample(&mut theirs, reps));
        } else {
            b = b.min(sample(&mut theirs, reps));
            a = a.min(sample(&mut ours, reps));
        }
    }
    (a, b)
}

/// The largest difference between two results, relative to the largest magnitude in them.
fn rel_diff(a: &[f64], b: &[f64]) -> f64 {
    let scale = a.iter().chain(b).fold(f64::MIN_POSITIVE, |s, x| s.max(x.abs()));
    a.iter().zip(b).map(|(x, y)| (x - y).abs()).fold(0.0, f64::max) / scale
}

struct Row {
    case: String,
    ours: f64,
    theirs: f64,
    agree: f64,
}

fn check(case: &str, agree: f64, tol: f64) -> f64 {
    assert!(agree <= tol, "{case}: autodyne and glam disagree by {agree:e} (relative), above {tol:e}");
    agree
}

fn main() {
    let out = std::env::args().skip_while(|a| a != "--out").nth(1);
    let src = transforms(N, 7);
    let src2 = transforms(N, 11);
    let mut rows = Vec::new();

    macro_rules! single {
        ($t:ty, $gm4:ty, $gv4:ty, $gm3:ty, $tol:expr) => {{
            let t = stringify!($t);
            let a: Vec<Mat4<$t>> = src.iter().map(ad4).collect();
            let b: Vec<Mat4<$t>> = src2.iter().map(ad4).collect();
            let ga: Vec<$gm4> = a.iter().map(|m| <$gm4>::from_cols_array(&m.to_cols_array())).collect();
            let gb: Vec<$gm4> = b.iter().map(|m| <$gm4>::from_cols_array(&m.to_cols_array())).collect();
            let wide = |x: &[$t]| x.iter().map(|&v| v as f64).collect::<Vec<f64>>();

            // Mat4 * Mat4
            let mut o = vec![Mat4::<$t>::identity(); N];
            let mut go = vec![<$gm4>::IDENTITY; N];
            let mut run_o = || for i in 0..N { o[i] = black_box(a[i]) * b[i] };
            let mut run_g = || for i in 0..N { go[i] = black_box(ga[i]) * gb[i] };
            run_o();
            run_g();
            let agree = rel_diff(&wide(&o.iter().flat_map(|m| m.to_cols_array()).collect::<Vec<_>>()), &wide(&go.iter().flat_map(|m| m.to_cols_array()).collect::<Vec<_>>()));
            let agree = check("mat4 * mat4", agree, $tol);
            let (x, y) = race(|| for i in 0..N { o[i] = black_box(a[i]) * b[i] }, || for i in 0..N { go[i] = black_box(ga[i]) * gb[i] });
            black_box((&o, &go));
            rows.push(Row { case: format!("Mat4 * Mat4, {t}"), ours: x, theirs: y, agree });

            // Mat4 * Vec4
            let v: Vec<Vec4<$t>> = b.iter().map(|m| m.cols[3]).collect();
            let gv: Vec<$gv4> = v.iter().map(|x| <$gv4>::from_array(x.to_array())).collect();
            let mut o = vec![Vec4::<$t>::zero(); N];
            let mut go = vec![<$gv4>::ZERO; N];
            for i in 0..N { o[i] = a[i] * v[i]; go[i] = ga[i] * gv[i]; }
            let agree = rel_diff(&wide(&o.iter().flat_map(|x| x.to_array()).collect::<Vec<_>>()), &wide(&go.iter().flat_map(|x| x.to_array()).collect::<Vec<_>>()));
            let agree = check("mat4 * vec4", agree, $tol);
            let (x, y) = race(|| for i in 0..N { o[i] = black_box(a[i]) * v[i] }, || for i in 0..N { go[i] = black_box(ga[i]) * gv[i] });
            black_box((&o, &go));
            rows.push(Row { case: format!("Mat4 * Vec4, {t}"), ours: x, theirs: y, agree });

            // Mat4 inverse
            let mut o = vec![Mat4::<$t>::identity(); N];
            let mut go = vec![<$gm4>::IDENTITY; N];
            for i in 0..N { o[i] = a[i].inverse(); go[i] = ga[i].inverse(); }
            let agree = rel_diff(&wide(&o.iter().flat_map(|m| m.to_cols_array()).collect::<Vec<_>>()), &wide(&go.iter().flat_map(|m| m.to_cols_array()).collect::<Vec<_>>()));
            let agree = check("mat4 inverse", agree, $tol * 100.0);
            let (x, y) = race(|| for i in 0..N { o[i] = black_box(a[i]).inverse() }, || for i in 0..N { go[i] = black_box(ga[i]).inverse() });
            black_box((&o, &go));
            rows.push(Row { case: format!("Mat4 inverse, {t}"), ours: x, theirs: y, agree });

            // Mat3 inverse
            let a3: Vec<Mat3<$t>> = src.iter().map(ad3).collect();
            let ga3: Vec<$gm3> = a3.iter().map(|m| <$gm3>::from_cols_array(&m.to_cols_array())).collect();
            let mut o = vec![Mat3::<$t>::identity(); N];
            let mut go = vec![<$gm3>::IDENTITY; N];
            for i in 0..N { o[i] = a3[i].inverse(); go[i] = ga3[i].inverse(); }
            let agree = rel_diff(&wide(&o.iter().flat_map(|m| m.to_cols_array()).collect::<Vec<_>>()), &wide(&go.iter().flat_map(|m| m.to_cols_array()).collect::<Vec<_>>()));
            let agree = check("mat3 inverse", agree, $tol * 100.0);
            let (x, y) = race(|| for i in 0..N { o[i] = black_box(a3[i]).inverse() }, || for i in 0..N { go[i] = black_box(ga3[i]).inverse() });
            black_box((&o, &go));
            rows.push(Row { case: format!("Mat3 inverse, {t}"), ours: x, theirs: y, agree });
        }};
    }
    single!(f32, glam::Mat4, glam::Vec4, glam::Mat3, 1e-6);
    single!(f64, glam::DMat4, glam::DVec4, glam::DMat3, 1e-14);

    // compose_world over a 16-ary tree, breadth first (parents before children)
    for n in [10_000usize, 100_000] {
        let parents: Vec<u32> = (0..n).map(|i| if i == 0 { NO_PARENT } else { ((i - 1) / 16) as u32 }).collect();
        let locals: Vec<Mat4<f32>> = (0..n).map(|i| ad4(&src[i % N])).collect();
        let glocals: Vec<glam::Mat4> = locals.iter().map(|m| glam::Mat4::from_cols_array(&m.to_cols_array())).collect();
        let mut world = vec![Mat4::identity(); n];
        let mut gworld = vec![glam::Mat4::IDENTITY; n];
        let glam_pass = |gworld: &mut [glam::Mat4]| {
            for i in 0..n {
                gworld[i] = if parents[i] == NO_PARENT { glocals[i] } else { gworld[parents[i] as usize] * glocals[i] };
            }
        };
        compose_world(&parents, &locals, &mut world);
        glam_pass(&mut gworld);
        let a: Vec<f64> = world.iter().flat_map(|m| m.to_cols_array()).map(f64::from).collect();
        let b: Vec<f64> = gworld.iter().flat_map(|m| m.to_cols_array()).map(f64::from).collect();
        // deep chains of products: rounding differs by summation order, so a looser bound
        let agree = check("compose_world", rel_diff(&a, &b), 1e-4);
        let (x, y) = race(|| compose_world(&parents, &locals, black_box(&mut world)), || glam_pass(black_box(&mut gworld)));
        rows.push(Row { case: format!("compose_world, {n} nodes, f32 (glam: the same loop)"), ours: x, theirs: y, agree });
    }

    let fmt = |s: f64| if s >= 1e-3 { format!("{:.3} ms", s * 1e3) } else { format!("{:.2} µs", s * 1e6) };
    let mut lines = vec![
        "# autodyne geometry vs glam".to_string(),
        String::new(),
        format!(
            "Generated by `cargo run --release` in `bench/geometry` (glam 0.30, {}). Each case checks that the two agree, then times both over",
            std::env::consts::ARCH
        ),
        format!("the same {N} inputs (independent operations) in {ROUNDS} alternating rounds; times are each side's best round per pass."),
        String::new(),
        "- Relative = glam's time / autodyne's (above 1: autodyne is faster).".to_string(),
        "- Agree = the largest difference between the two results, relative to the largest value.".to_string(),
        "- Across builds, where the code lands alone moves the single-value cases by up to ~8% (glam's own Mat4 product measured".to_string(),
        "  3.14 and 3.40 µs in two builds of this harness), so within that read as parity.".to_string(),
        String::new(),
        "| Case | autodyne | glam | Relative | Agree |".to_string(),
        "|---|---:|---:|---:|---:|".to_string(),
    ];
    for r in &rows {
        lines.push(format!("| {} | {} | {} | {:.2}x | {:.0e} |", r.case, fmt(r.ours), fmt(r.theirs), r.theirs / r.ours, r.agree));
    }
    let text = lines.join("\n") + "\n";
    print!("{text}");
    if let Some(path) = out {
        std::fs::write(path, text).expect("write the results");
    }
}
