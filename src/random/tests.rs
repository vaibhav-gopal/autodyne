//! Tests for the random number generator and its distributions (KS and chi-squared tests).

use super::*;
use crate::special::ndtr;

/// Mean and variance (population).
fn moments(x: &[f64]) -> (f64, f64) {
    let n = x.len() as f64;
    let mean = x.iter().sum::<f64>() / n;
    (mean, x.iter().map(|v| (v - mean) * (v - mean)).sum::<f64>() / n)
}

/// The Kolmogorov-Smirnov distance between the samples and `cdf`.
fn ks(mut x: Vec<f64>, cdf: impl Fn(f64) -> f64) -> f64 {
    x.sort_by(f64::total_cmp);
    let n = x.len() as f64;
    x.iter().enumerate().fold(0.0f64, |d, (i, &v)| {
        let f = cdf(v);
        d.max((f - i as f64 / n).abs()).max(((i + 1) as f64 / n - f).abs())
    })
}

/// The KS distance a sample of `n` from the right distribution stays below 99.9% of the time.
fn ks_bound(n: usize) -> f64 {
    1.95 / (n as f64).sqrt()
}

fn draws<D: Distribution<f64>>(dist: &D, n: usize, seed: u64) -> Vec<f64> {
    let mut rng = Rng::new(seed);
    (0..n).map(|_| rng.sample(dist)).collect()
}

#[test]
fn xoshiro_matches_its_reference_and_seeds_are_reproducible() {
    // xoshiro256++ from the state (1, 2, 3, 4): rotl(1 + 4, 23) + 1
    let mut rng = Rng { s: [1, 2, 3, 4] };
    assert_eq!(rng.next_u64(), 41_943_041);
    let (mut a, mut b, mut c) = (Rng::new(7), Rng::new(7), Rng::new(8));
    let first: Vec<u64> = (0..5).map(|_| a.next_u64()).collect();
    assert_eq!(first, (0..5).map(|_| b.next_u64()).collect::<Vec<_>>());
    assert_ne!(first, (0..5).map(|_| c.next_u64()).collect::<Vec<_>>());
    // a fork continues the stream; the parent jumps away from it
    let mut parent = Rng::new(3);
    let mut child = parent.fork();
    let mut fresh = Rng::new(3);
    assert_eq!(child.next_u64(), fresh.next_u64());
    assert_ne!(parent.next_u64(), fresh.next_u64());
}

#[test]
fn uniform_integers_are_unbiased() {
    let mut rng = Rng::new(11);
    let mut counts = [0u32; 7];
    for _ in 0..70_000 {
        counts[rng.below(7) as usize] += 1;
    }
    // chi-squared with 6 degrees of freedom: 22.46 is its 99.9% point
    let chi2: f64 = counts.iter().map(|&c| (c as f64 - 10_000.0).powi(2) / 10_000.0).sum();
    assert!(chi2 < 22.46, "chi2 {chi2}");
    assert!((0..1000).all(|_| (-3..4).contains(&rng.integers(-3, 4))));
    let u = draws(&Uniform::new(-2.0, 3.0).unwrap(), 100_000, 2);
    assert!(ks(u, |x| (x + 2.0) / 5.0) < ks_bound(100_000));
}

#[test]
fn normal_ziggurat_matches_the_normal_distribution() {
    let n = 400_000;
    let x = draws(&Normal::standard(), n, 5);
    let (mean, var) = moments(&x);
    assert!(mean.abs() < 0.006 && (var - 1.0).abs() < 0.008, "mean {mean} var {var}");
    // the tail beyond the base strip, about 2.6e-4 of the draws
    let tail = x.iter().filter(|v| v.abs() > NORMAL_R).count() as f64 / n as f64;
    assert!((tail - 2.0 * (1.0 - ndtr(NORMAL_R))).abs() < 1e-4, "tail {tail}");
    assert!(ks(x, ndtr) < ks_bound(n));
    let y = draws(&Normal::new(3.0, 0.5).unwrap(), 100_000, 6);
    assert!(ks(y, |v| ndtr((v - 3.0) / 0.5)) < ks_bound(100_000));
}

#[test]
fn exponential_ziggurat_matches_the_exponential_distribution() {
    let n = 400_000;
    let x = draws(&Exponential::new(1.0).unwrap(), n, 9);
    let (mean, var) = moments(&x);
    assert!((mean - 1.0).abs() < 0.008 && (var - 1.0).abs() < 0.02, "mean {mean} var {var}");
    assert!(ks(x, |v| 1.0 - (-v).exp()) < ks_bound(n));
}

#[test]
fn gamma_family_has_its_moments() {
    for (shape, scale) in [(0.3, 1.0), (1.0, 2.0), (2.5, 0.5), (40.0, 1.0)] {
        let x = draws(&Gamma::new(shape, scale).unwrap(), 200_000, 13);
        let (mean, var) = moments(&x);
        let (m, v) = (shape * scale, shape * scale * scale);
        assert!((mean - m).abs() < 0.01 * m.max(1.0) && (var - v).abs() < 0.03 * v.max(1.0), "gamma({shape}, {scale}): {mean} {var}");
    }
    let (mean, var) = moments(&draws(&Beta::new(2.0, 5.0).unwrap(), 200_000, 14));
    assert!((mean - 2.0 / 7.0).abs() < 0.003 && (var - 10.0 / (49.0 * 8.0)).abs() < 0.001, "beta {mean} {var}");
    let (mean, var) = moments(&draws(&ChiSquared::new(4.0).unwrap(), 200_000, 15));
    assert!((mean - 4.0).abs() < 0.03 && (var - 8.0).abs() < 0.15, "chi2 {mean} {var}");
    let (mean, var) = moments(&draws(&StudentT::new(10.0).unwrap(), 200_000, 16));
    assert!(mean.abs() < 0.01 && (var - 1.25).abs() < 0.03, "t {mean} {var}");
    let (mean, var) = moments(&draws(&Laplace::new(1.0, 2.0).unwrap(), 200_000, 17));
    assert!((mean - 1.0).abs() < 0.02 && (var - 8.0).abs() < 0.15, "laplace {mean} {var}");
    let x = draws(&LogNormal::new(0.0, 0.5).unwrap(), 200_000, 18);
    assert!(ks(x, |v| if v <= 0.0 { 0.0 } else { ndtr(v.ln() / 0.5) }) < ks_bound(200_000));
}

#[test]
fn discrete_distributions_have_their_moments() {
    let count = |d: &dyn Fn(&mut Rng) -> u64, seed| {
        let mut rng = Rng::new(seed);
        moments(&(0..200_000).map(|_| d(&mut rng) as f64).collect::<Vec<_>>())
    };
    for lam in [0.5, 3.0, 9.9, 10.0, 50.0, 1e4] {
        let (mean, var) = count(&|r| r.poisson(lam), 21);
        assert!((mean - lam).abs() < 0.01 * lam.max(1.0) && (var - lam).abs() < 0.03 * lam.max(1.0), "poisson({lam}): {mean} {var}");
    }
    for (n, p) in [(20, 0.3), (5, 0.9), (1000, 0.4), (100_000, 0.02)] {
        let (mean, var) = count(&|r| r.binomial(n, p), 22);
        let (m, v) = (n as f64 * p, n as f64 * p * (1.0 - p));
        assert!((mean - m).abs() < 0.01 * m && (var - v).abs() < 0.03 * v, "binomial({n}, {p}): {mean} {var}");
        let mut rng = Rng::new(23);
        assert!((0..10_000).all(|_| rng.binomial(n, p) <= n));
    }
    let (mean, _) = count(&|r| Geometric::new(0.25).unwrap().sample(r), 24);
    assert!((mean - 4.0).abs() < 0.05, "geometric {mean}");
    let mut rng = Rng::new(25);
    let heads = (0..100_000).filter(|_| Bernoulli::new(0.3).unwrap().sample(&mut rng)).count();
    assert!((heads as f64 / 100_000.0 - 0.3).abs() < 0.005);
}

#[test]
fn shuffles_and_samples_are_permutations() {
    let mut rng = Rng::new(31);
    let mut p = rng.permutation(100);
    p.sort_unstable();
    assert_eq!(p, (0..100).collect::<Vec<_>>());
    let s = rng.sample_indices(1000, 50);
    let mut distinct = s.clone();
    distinct.sort_unstable();
    distinct.dedup();
    assert_eq!(distinct.len(), 50);
    assert!(s.iter().all(|&i| i < 1000));
    // every position equally likely for the first element
    let mut first = [0u32; 4];
    for _ in 0..40_000 {
        let mut v = [0, 1, 2, 3];
        rng.shuffle(&mut v);
        first[v[0]] += 1;
    }
    assert!(first.iter().all(|&c| (c as f64 - 10_000.0).abs() < 400.0), "{first:?}");
}

#[test]
fn arrays_and_parameters() {
    let mut rng = Rng::new(41);
    let a = rng.array::<f32, _>(&Normal::standard(), &[3, 4]).unwrap();
    assert_eq!(a.shape(), [3, 4]);
    let counts = rng.array::<u64, _>(&Poisson::new(2.0).unwrap(), &[10]).unwrap();
    assert_eq!(counts.shape(), [10]);
    assert!(Normal::new(0.0, -1.0).is_err());
    assert!(Gamma::new(0.0, 1.0).is_err());
    assert!(Binomial::new(10, 1.5).is_err());
    assert!(Uniform::new(2.0, 1.0).is_err());
    assert!(matches!(Poisson::new(-1.0), Err(RandomError::Invalid(_))));
}

#[test]
fn log_factorials_match_gammaln() {
    for k in [0.0, 1.0, 5.0, 100.0, 255.0, 256.0, 257.0, 1000.0, 1e6, 1e12] {
        let want = crate::special::gammaln(k + 1.0);
        assert!((ln_factorial(k) - want).abs() <= 1e-14 * want.abs().max(1.0), "ln({k}!) = {} vs {want}", ln_factorial(k));
    }
}
