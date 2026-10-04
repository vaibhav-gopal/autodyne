//! Random numbers and random processes, as `numpy.random.Generator` has them: [`Rng`]
//! (xoshiro256++, seedable, with independent streams from [`Rng::fork`]), distributions
//! ([`Normal`], [`Uniform`], [`Exponential`], [`Gamma`], [`Beta`], [`ChiSquared`], [`StudentT`],
//! [`LogNormal`], [`Laplace`], [`Poisson`], [`Binomial`], [`Bernoulli`], [`Geometric`]) that sample
//! one value, fill a slice or make an array, and random processes in [`process`] (coloured noise,
//! Brownian motion, Ornstein-Uhlenbeck, ARMA, Poisson events).
//!
//! Normal and exponential variates come from 256-layer ziggurats (one 64-bit draw, a table look-up
//! and a multiply 99% of the time). Not for cryptography.
//!
//! ```
//! use autodyne::random::{Normal, Rng};
//!
//! let mut rng = Rng::new(42);
//! let x = rng.array::<f64, _>(&Normal::new(1.0, 2.0).unwrap(), &[1000]).unwrap();
//! let mean = x.as_slice().iter().sum::<f64>() / 1000.0;
//! assert!((mean - 1.0).abs() < 0.2);
//! ```

use std::sync::OnceLock;

use thiserror::Error;

use crate::signal::{NdArray, NdError};
use crate::units::*;

pub mod process;

/// Errors from random number generation.
#[derive(Debug, Clone, PartialEq, Eq, Error)]
pub enum RandomError {
    /// A distribution parameter is out of range (a negative scale, a probability above 1).
    #[error("invalid distribution: {0}")]
    Invalid(String),
    /// An n-d layout error (a shape too large).
    #[error(transparent)]
    Nd(#[from] NdError),
}

impl RandomError {
    pub(crate) fn invalid(message: impl Into<String>) -> Self {
        RandomError::Invalid(message.into())
    }
}

// GENERATOR =======================================================================================

/// A pseudo-random generator: xoshiro256++ (period 2^256 - 1, passes BigCrush), seeded through
/// splitmix64 so nearby seeds give unrelated streams. Deterministic per seed on every platform.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Rng {
    s: [u64; 4],
}

fn splitmix64(state: &mut u64) -> u64 {
    *state = state.wrapping_add(0x9E37_79B9_7F4A_7C15);
    let mut z = *state;
    z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
    z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
    z ^ (z >> 31)
}

impl Rng {
    /// The generator for `seed` (the same seed gives the same numbers).
    pub fn new(seed: u64) -> Self {
        let mut state = seed;
        let s = [splitmix64(&mut state), splitmix64(&mut state), splitmix64(&mut state), splitmix64(&mut state)];
        // splitmix64 never yields four zeros in a row, the one state xoshiro can't leave
        Self { s }
    }

    /// A generator seeded from the process's hash randomness and the clock: different on every
    /// call and every run.
    pub fn from_entropy() -> Self {
        use std::hash::{BuildHasher, Hasher};
        let mut h = std::collections::hash_map::RandomState::new().build_hasher();
        h.write_u128(std::time::SystemTime::now().duration_since(std::time::UNIX_EPOCH).map_or(0, |d| d.as_nanos()));
        Self::new(h.finish())
    }

    /// The next 64 random bits.
    #[inline]
    pub fn next_u64(&mut self) -> u64 {
        let s = &mut self.s;
        let result = s[0].wrapping_add(s[3]).rotate_left(23).wrapping_add(s[0]);
        let t = s[1] << 17;
        s[2] ^= s[0];
        s[3] ^= s[1];
        s[1] ^= s[2];
        s[0] ^= s[3];
        s[2] ^= t;
        s[3] = s[3].rotate_left(45);
        result
    }

    /// Advances the generator by 2^128 draws: from one seed, `jump`ed copies give 2^128
    /// non-overlapping streams (for parallel work).
    pub fn jump(&mut self) {
        const JUMP: [u64; 4] = [0x180E_C6D3_3CFD_0ABA, 0xD5A6_1266_F0C9_392C, 0xA958_2618_E03F_C9AA, 0x39AB_DC45_29B1_661C];
        let mut acc = [0u64; 4];
        for word in JUMP {
            for bit in 0..64 {
                if word & (1 << bit) != 0 {
                    for (a, s) in acc.iter_mut().zip(self.s) {
                        *a ^= s;
                    }
                }
                self.next_u64();
            }
        }
        self.s = acc;
    }

    /// A copy of this generator for another stream: the copy continues from here, and this one
    /// jumps 2^128 draws ahead, so the two never overlap.
    pub fn fork(&mut self) -> Rng {
        let child = self.clone();
        self.jump();
        child
    }

    /// Uniform in [0, 1) with 53 random bits.
    #[inline]
    pub fn random(&mut self) -> f64 {
        // through i64: 53 bits convert exactly with one instruction (u64 -> f64 needs a fix-up)
        ((self.next_u64() >> 11) as i64) as f64 * (1.0 / (1u64 << 53) as f64)
    }

    /// Uniform in [`low`, `high`).
    #[inline]
    pub fn uniform(&mut self, low: f64, high: f64) -> f64 {
        low + (high - low) * self.random()
    }

    /// Uniform in `0..n` without bias (Lemire's multiply-and-reject). Panics if `n` is 0.
    #[inline]
    pub fn below(&mut self, n: u64) -> u64 {
        assert!(n > 0, "below(0) has no values");
        let mut m = self.next_u64() as u128 * n as u128;
        if (m as u64) < n {
            let threshold = n.wrapping_neg() % n;
            while (m as u64) < threshold {
                m = self.next_u64() as u128 * n as u128;
            }
        }
        (m >> 64) as u64
    }

    /// Uniform integer in [`low`, `high`) (`Generator.integers`). Panics unless `low < high`.
    pub fn integers(&mut self, low: i64, high: i64) -> i64 {
        assert!(low < high, "integers needs low < high");
        low.wrapping_add(self.below(high.wrapping_sub(low) as u64) as i64)
    }

    /// A standard normal variate (mean 0, variance 1), from the ziggurat: the rectangle test inline
    /// (99% of draws), the wedges and the tail out of line.
    #[inline]
    pub fn standard_normal(&mut self) -> f64 {
        let z = ziggurat_normal();
        let r = self.next_u64();
        let i = (r & 0xff) as usize;
        let mag = r >> 12; // 52 bits
        if mag < z.k[i] {
            return with_sign_of_bit8((mag as i64) as f64 * z.w[i], r);
        }
        self.normal_rejected(r)
    }

    /// The rest of [`standard_normal`](Self::standard_normal) after draw `r` missed its rectangle.
    #[cold]
    #[inline(never)]
    fn normal_rejected(&mut self, first: u64) -> f64 {
        let z = ziggurat_normal();
        let mut r = first;
        loop {
            let i = (r & 0xff) as usize;
            let negative = r & 0x100 != 0;
            let mag = r >> 12;
            let x = (mag as i64) as f64 * z.w[i];
            if mag < z.k[i] {
                return if negative { -x } else { x };
            }
            if i == 0 {
                // the tail beyond r, by Marsaglia's method
                loop {
                    let tx = -(-self.random()).ln_1p() / NORMAL_R;
                    let ty = -(-self.random()).ln_1p();
                    if ty + ty > tx * tx {
                        let v = NORMAL_R + tx;
                        return if negative { -v } else { v };
                    }
                }
            }
            if z.f[i] + self.random() * (z.f[i - 1] - z.f[i]) < (-0.5 * x * x).exp() {
                return if negative { -x } else { x };
            }
            r = self.next_u64();
        }
    }

    /// A standard exponential variate (rate 1), from the ziggurat (the rectangle test inline).
    #[inline]
    pub fn standard_exponential(&mut self) -> f64 {
        let z = ziggurat_exponential();
        let r = self.next_u64();
        let i = (r & 0xff) as usize;
        let mag = r >> 11; // 53 bits
        if mag < z.k[i] {
            return (mag as i64) as f64 * z.w[i];
        }
        self.exponential_rejected(r)
    }

    /// The rest of [`standard_exponential`](Self::standard_exponential) after a missed rectangle.
    #[cold]
    #[inline(never)]
    fn exponential_rejected(&mut self, first: u64) -> f64 {
        let z = ziggurat_exponential();
        let mut r = first;
        loop {
            let i = (r & 0xff) as usize;
            let mag = r >> 11;
            let x = (mag as i64) as f64 * z.w[i];
            if mag < z.k[i] {
                return x;
            }
            if i == 0 {
                return EXPONENTIAL_R - (-self.random()).ln_1p();
            }
            if z.f[i] + self.random() * (z.f[i - 1] - z.f[i]) < (-x).exp() {
                return x;
            }
            r = self.next_u64();
        }
    }

    /// A standard gamma variate of `shape > 0` (scale 1): Marsaglia and Tsang's method, with
    /// `Gamma(a) = Gamma(a + 1) U^(1/a)` below 1.
    pub fn standard_gamma(&mut self, shape: f64) -> f64 {
        if shape == 1.0 {
            return self.standard_exponential();
        }
        if shape < 1.0 {
            let u = self.random();
            return self.standard_gamma(shape + 1.0) * u.powf(1.0 / shape);
        }
        let d = shape - 1.0 / 3.0;
        let c = 1.0 / (9.0 * d).sqrt();
        loop {
            let x = self.standard_normal();
            let v = 1.0 + c * x;
            if v <= 0.0 {
                continue;
            }
            let v = v * v * v;
            let u = self.random();
            if u < 1.0 - 0.0331 * x * x * x * x || u.ln() < 0.5 * x * x + d * (1.0 - v + v.ln()) {
                return d * v;
            }
        }
    }

    /// A Poisson variate with mean `lam >= 0`: inversion (one uniform, a walk up the cumulative
    /// distribution) below 10, Hörmann's transformed rejection (PTRS) above.
    pub fn poisson(&mut self, lam: f64) -> u64 {
        if lam <= 0.0 {
            return 0;
        }
        if lam < 10.0 {
            let mut p = (-lam).exp();
            let (mut k, mut cdf, u) = (0, p, self.random());
            // the cumulative sum stops short of 1 by rounding; the cap keeps the walk finite
            while u > cdf && k < 1000 {
                k += 1;
                p *= lam / k as f64;
                cdf += p;
            }
            return k;
        }
        Ptrs::new(lam).sample(self)
    }

    /// A binomial variate: successes in `n` trials of probability `p`. Inversion when `n p < 10`,
    /// Hörmann's transformed rejection (BTRS) above; `p > 1/2` by symmetry.
    pub fn binomial(&mut self, n: u64, p: f64) -> u64 {
        if n == 0 || p <= 0.0 {
            return 0;
        }
        if p >= 1.0 {
            return n;
        }
        if p > 0.5 {
            return n - self.binomial(n, 1.0 - p);
        }
        let q = 1.0 - p;
        if (n as f64) * p < 10.0 {
            // walk the cumulative distribution from 0
            let s = p / q;
            let a = (n as f64 + 1.0) * s;
            loop {
                let mut r = q.powf(n as f64);
                let mut u = self.random();
                let mut k = 0;
                while u > r {
                    u -= r;
                    k += 1;
                    if k > n {
                        break;
                    }
                    r *= a / k as f64 - s;
                }
                if k <= n {
                    return k;
                }
            }
        }
        let nf = n as f64;
        let spq = (nf * p * q).sqrt();
        let b = 1.15 + 2.53 * spq;
        let a = -0.0873 + 0.0248 * b + 0.01 * p;
        let c = nf * p + 0.5;
        let alpha = (2.83 + 5.1 / b) * spq;
        let vr = 0.92 - 4.2 / b;
        let m = ((nf + 1.0) * p).floor();
        let h = ln_factorial(m) + ln_factorial(nf - m);
        let lpq = (p / q).ln();
        loop {
            let u = self.random() - 0.5;
            let v = self.random();
            let us = 0.5 - u.abs();
            let k = ((2.0 * a / us + b) * u + c).floor();
            if k < 0.0 || k > nf {
                continue;
            }
            if us >= 0.07 && v <= vr {
                return k as u64;
            }
            let v = (v * alpha / (a / (us * us) + b)).ln();
            if v <= h - ln_factorial(k) - ln_factorial(nf - k) + (k - m) * lpq {
                return k as u64;
            }
        }
    }

    /// Shuffles `items` in place (Fisher-Yates): every order equally likely.
    pub fn shuffle<T>(&mut self, items: &mut [T]) {
        for i in (1..items.len()).rev() {
            let j = self.below(i as u64 + 1) as usize;
            items.swap(i, j);
        }
    }

    /// `0..n` in random order.
    pub fn permutation(&mut self, n: usize) -> Vec<usize> {
        let mut p: Vec<usize> = (0..n).collect();
        self.shuffle(&mut p);
        p
    }

    /// `k` distinct indices from `0..n` in random order (a partial Fisher-Yates shuffle). Panics if
    /// `k > n`.
    pub fn sample_indices(&mut self, n: usize, k: usize) -> Vec<usize> {
        assert!(k <= n, "cannot take {k} distinct indices from {n}");
        let mut p: Vec<usize> = (0..n).collect();
        for i in 0..k {
            let j = i + self.below((n - i) as u64) as usize;
            p.swap(i, j);
        }
        p.truncate(k);
        p
    }

    /// One value of `dist`.
    #[inline]
    pub fn sample<T, D: Distribution<T>>(&mut self, dist: &D) -> T {
        dist.sample(self)
    }

    /// Fills `out` with values of `dist`.
    pub fn fill<T, D: Distribution<T>>(&mut self, dist: &D, out: &mut [T]) {
        dist.fill(self, out);
    }

    /// Standard normal draws into `out` through `map`, 64 at a time: the raw words first (the
    /// generator's dependency chain on its own), then the ziggurat over them, so the two overlap
    /// instead of alternating. Deterministic per seed, though not the same values as repeated
    /// [`standard_normal`](Self::standard_normal) calls (rejections draw from later in the stream).
    fn normals_into<T>(&mut self, out: &mut [T], map: impl Fn(f64) -> T) {
        let z = ziggurat_normal();
        let mut raw = [0u64; 64];
        for chunk in out.chunks_mut(64) {
            for r in raw[..chunk.len()].iter_mut() {
                *r = self.next_u64();
            }
            for (o, &r) in chunk.iter_mut().zip(&raw) {
                let (i, mag) = ((r & 0xff) as usize, r >> 12);
                let v = if mag < z.k[i] {
                    with_sign_of_bit8((mag as i64) as f64 * z.w[i], r)
                } else {
                    self.normal_rejected(r)
                };
                *o = map(v);
            }
        }
    }

    /// An array of `shape` filled with values of `dist` (row-major order).
    pub fn array<T: Copy + Default, D: Distribution<T>>(&mut self, dist: &D, shape: &[usize]) -> Result<NdArray<T>, RandomError> {
        let mut out = NdArray::<T>::zeros(shape)?;
        self.fill(dist, out.as_mut_slice());
        Ok(out)
    }
}

/// `x` negated when bit 8 of `r` is set, without a branch: a random sign would mispredict half the
/// time (it cost the ziggurat more than everything else together).
#[inline(always)]
fn with_sign_of_bit8(x: f64, r: u64) -> f64 {
    f64::from_bits(x.to_bits() ^ ((r & 0x100) << 55))
}

// ZIGGURATS =======================================================================================

/// Layers of each ziggurat.
const LAYERS: usize = 256;
/// Where the normal ziggurat's base strip ends, and each layer's area (Marsaglia and Tsang).
const NORMAL_R: f64 = 3.654_152_885_361_009;
const NORMAL_V: f64 = 0.004_928_673_233_99;
/// The same for the exponential ziggurat.
const EXPONENTIAL_R: f64 = 7.697_117_470_131_05;
const EXPONENTIAL_V: f64 = 0.003_949_659_822_581_557;

/// One ziggurat: `k[i]` accepts a draw outright, `w[i]` scales it to `x`, `f[i]` is the density at
/// the layer's edge.
struct Ziggurat {
    k: [u64; LAYERS],
    w: [f64; LAYERS],
    f: [f64; LAYERS],
}

/// Builds a ziggurat for the (unnormalized) density `pdf` with inverse `inv`, tail start `r`, layer
/// area `v`, and draws of `bits` bits (Marsaglia and Tsang's set-up).
fn ziggurat(pdf: fn(f64) -> f64, inv: impl Fn(f64) -> f64, r: f64, v: f64, bits: u32) -> Ziggurat {
    let m = (1u64 << bits) as f64;
    let (mut k, mut w, mut f) = ([0u64; LAYERS], [0.0; LAYERS], [0.0; LAYERS]);
    let mut d = r;
    let mut t = d;
    let q = v / pdf(d);
    k[0] = ((d / q) * m) as u64;
    k[1] = 0;
    w[0] = q / m;
    w[LAYERS - 1] = d / m;
    f[0] = 1.0;
    f[LAYERS - 1] = pdf(d);
    for i in (1..LAYERS - 1).rev() {
        d = inv(v / d + pdf(d));
        k[i + 1] = ((d / t) * m) as u64;
        t = d;
        f[i] = pdf(d);
        w[i] = d / m;
    }
    Ziggurat { k, w, f }
}

fn ziggurat_normal() -> &'static Ziggurat {
    static Z: OnceLock<Ziggurat> = OnceLock::new();
    Z.get_or_init(|| ziggurat(|x| (-0.5 * x * x).exp(), |y| (-2.0 * y.ln()).sqrt(), NORMAL_R, NORMAL_V, 52))
}

fn ziggurat_exponential() -> &'static Ziggurat {
    static Z: OnceLock<Ziggurat> = OnceLock::new();
    Z.get_or_init(|| ziggurat(|x| (-x).exp(), |y| -y.ln(), EXPONENTIAL_R, EXPONENTIAL_V, 53))
}

// DISTRIBUTIONS ===================================================================================

/// Something to draw values of type `T` from.
pub trait Distribution<T> {
    /// One value.
    fn sample(&self, rng: &mut Rng) -> T;
    /// Fills `out` with values (one `sample` each, unless a distribution has a faster batch).
    fn fill(&self, rng: &mut Rng, out: &mut [T]) {
        for o in out {
            *o = self.sample(rng);
        }
    }
}

fn positive(name: &str, v: f64) -> Result<f64, RandomError> {
    if v > 0.0 && v.is_finite() { Ok(v) } else { Err(RandomError::invalid(format!("{name} must be positive and finite, got {v}"))) }
}

fn probability(p: f64) -> Result<f64, RandomError> {
    if (0.0..=1.0).contains(&p) { Ok(p) } else { Err(RandomError::invalid(format!("p must be in [0, 1], got {p}"))) }
}

macro_rules! continuous {
    ($D:ty, |$s:ident, $rng:ident| $body:expr) => {
        impl<T: Float> Distribution<T> for $D {
            #[inline]
            fn sample(&self, $rng: &mut Rng) -> T {
                let $s = self;
                T::_lit($body)
            }
        }
    };
}

/// Uniform on [`low`, `high`).
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Uniform {
    low: f64,
    high: f64,
}

impl Uniform {
    /// Errors unless `low <= high`, both finite.
    pub fn new(low: f64, high: f64) -> Result<Self, RandomError> {
        if low <= high && low.is_finite() && high.is_finite() { Ok(Self { low, high }) } else { Err(RandomError::invalid(format!("uniform needs finite low <= high, got [{low}, {high})"))) }
    }
}
continuous!(Uniform, |s, rng| rng.uniform(s.low, s.high));

/// Normal (Gaussian) with mean `mean` and standard deviation `std`.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Normal {
    mean: f64,
    std: f64,
}

impl Normal {
    /// Errors unless `std >= 0` and both are finite.
    pub fn new(mean: f64, std: f64) -> Result<Self, RandomError> {
        if std >= 0.0 && std.is_finite() && mean.is_finite() { Ok(Self { mean, std }) } else { Err(RandomError::invalid(format!("normal needs a finite mean and std >= 0, got {mean}, {std}"))) }
    }
    /// Mean 0, standard deviation 1.
    pub fn standard() -> Self {
        Self { mean: 0.0, std: 1.0 }
    }
}
impl<T: Float> Distribution<T> for Normal {
    #[inline]
    fn sample(&self, rng: &mut Rng) -> T {
        T::_lit(self.mean + self.std * rng.standard_normal())
    }
    fn fill(&self, rng: &mut Rng, out: &mut [T]) {
        let (mean, std) = (self.mean, self.std);
        rng.normals_into(out, |v| T::_lit(mean + std * v));
    }
}

/// `exp(Normal(mean, sigma))`.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct LogNormal {
    normal: Normal,
}

impl LogNormal {
    /// The log of the variate has this mean and standard deviation; errors as [`Normal::new`].
    pub fn new(mean: f64, sigma: f64) -> Result<Self, RandomError> {
        Ok(Self { normal: Normal::new(mean, sigma)? })
    }
}
continuous!(LogNormal, |s, rng| (s.normal.mean + s.normal.std * rng.standard_normal()).exp());

/// Exponential with mean `scale` (rate `1 / scale`).
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Exponential {
    scale: f64,
}

impl Exponential {
    /// Errors unless `scale > 0`.
    pub fn new(scale: f64) -> Result<Self, RandomError> {
        Ok(Self { scale: positive("scale", scale)? })
    }
}
continuous!(Exponential, |s, rng| s.scale * rng.standard_exponential());

/// Gamma with `shape` (k) and `scale` (θ): mean `k θ`.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Gamma {
    shape: f64,
    scale: f64,
}

impl Gamma {
    /// Errors unless both are positive.
    pub fn new(shape: f64, scale: f64) -> Result<Self, RandomError> {
        Ok(Self { shape: positive("shape", shape)?, scale: positive("scale", scale)? })
    }
}
continuous!(Gamma, |s, rng| s.scale * rng.standard_gamma(s.shape));

/// Beta on [0, 1] with shapes `a` and `b` (from two gamma variates).
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Beta {
    a: f64,
    b: f64,
}

impl Beta {
    /// Errors unless both are positive.
    pub fn new(a: f64, b: f64) -> Result<Self, RandomError> {
        Ok(Self { a: positive("a", a)?, b: positive("b", b)? })
    }
}
continuous!(Beta, |s, rng| {
    let x = rng.standard_gamma(s.a);
    x / (x + rng.standard_gamma(s.b))
});

/// Chi-squared with `df` degrees of freedom (`2 Gamma(df / 2)`).
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct ChiSquared {
    df: f64,
}

impl ChiSquared {
    /// Errors unless `df > 0`.
    pub fn new(df: f64) -> Result<Self, RandomError> {
        Ok(Self { df: positive("df", df)? })
    }
}
continuous!(ChiSquared, |s, rng| 2.0 * rng.standard_gamma(s.df / 2.0));

/// Student's t with `df` degrees of freedom.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct StudentT {
    df: f64,
}

impl StudentT {
    /// Errors unless `df > 0`.
    pub fn new(df: f64) -> Result<Self, RandomError> {
        Ok(Self { df: positive("df", df)? })
    }
}
continuous!(StudentT, |s, rng| {
    let z = rng.standard_normal();
    z / (2.0 * rng.standard_gamma(s.df / 2.0) / s.df).sqrt()
});

/// Laplace (double exponential) centred on `loc` with scale `scale`.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Laplace {
    loc: f64,
    scale: f64,
}

impl Laplace {
    /// Errors unless `scale > 0`.
    pub fn new(loc: f64, scale: f64) -> Result<Self, RandomError> {
        Ok(Self { loc, scale: positive("scale", scale)? })
    }
}
continuous!(Laplace, |s, rng| {
    let e = rng.standard_exponential();
    s.loc + with_sign_of_bit8(s.scale * e, rng.next_u64())
});

/// `ln k!`: a table below 256, Stirling's series above (relative error below 1e-15).
fn ln_factorial(k: f64) -> f64 {
    static TABLE: OnceLock<[f64; 256]> = OnceLock::new();
    if k < 256.0 {
        let t = TABLE.get_or_init(|| {
            let mut t = [0.0; 256];
            for i in 1..256 {
                t[i] = t[i - 1] + (i as f64).ln();
            }
            t
        });
        return t[k as usize];
    }
    let (n, inv) = (k + 1.0, 1.0 / (k + 1.0));
    let inv2 = inv * inv;
    (n - 0.5) * n.ln() - n + 0.5 * std::f64::consts::TAU.ln() + inv * (1.0 / 12.0 - inv2 * (1.0 / 360.0 - inv2 / 1260.0))
}

/// Hörmann's transformed rejection with squeeze (PTRS) for Poisson means of 10 or more, its
/// constants computed once.
#[derive(Clone, Copy, Debug, PartialEq)]
struct Ptrs {
    lam: f64,
    loglam: f64,
    a: f64,
    b: f64,
    ln_inv_alpha: f64,
    vr: f64,
}

impl Ptrs {
    fn new(lam: f64) -> Self {
        let b = 0.931 + 2.53 * lam.sqrt();
        Self { lam, loglam: lam.ln(), a: -0.059 + 0.02483 * b, b, ln_inv_alpha: (1.1239 + 1.1328 / (b - 3.4)).ln(), vr: 0.9277 - 3.6224 / (b - 2.0) }
    }

    fn sample(&self, rng: &mut Rng) -> u64 {
        let Ptrs { lam, loglam, a, b, ln_inv_alpha, vr } = *self;
        loop {
            let u = rng.random() - 0.5;
            let v = rng.random();
            let us = 0.5 - u.abs();
            let k = ((2.0 * a / us + b) * u + lam + 0.43).floor();
            if us >= 0.07 && v <= vr {
                return k as u64;
            }
            if k < 0.0 || (us < 0.013 && v > us) {
                continue;
            }
            if v.ln() + ln_inv_alpha - (a / (us * us) + b).ln() <= -lam + k * loglam - ln_factorial(k) {
                return k as u64;
            }
        }
    }
}

/// Poisson with mean `lam`: counts of events. Below a mean of 10 its cumulative distribution is
/// tabulated once and each draw is one uniform and a short search; above, transformed rejection
/// (PTRS) with its constants precomputed.
#[derive(Clone, Debug, PartialEq)]
pub struct Poisson {
    method: PoissonMethod,
}

#[derive(Clone, Debug, PartialEq)]
enum PoissonMethod {
    /// `cdf[k] = P(X <= k)`, up to where it reaches 1 at double precision
    Table(Vec<f64>),
    Ptrs(Ptrs),
}

impl Poisson {
    /// Errors unless `0 <= lam < 1e15` (finite).
    pub fn new(lam: f64) -> Result<Self, RandomError> {
        if !(0.0..1e15).contains(&lam) {
            return Err(RandomError::invalid(format!("lam must be in [0, 1e15), got {lam}")));
        }
        if lam >= 10.0 {
            return Ok(Self { method: PoissonMethod::Ptrs(Ptrs::new(lam)) });
        }
        let mut p = (-lam).exp();
        let mut cdf = vec![p];
        let mut k = 0.0;
        while *cdf.last().expect("starts with P(0)") < 1.0 - f64::EPSILON && cdf.len() < 200 {
            k += 1.0;
            p *= lam / k;
            cdf.push(cdf.last().expect("non-empty") + p);
        }
        Ok(Self { method: PoissonMethod::Table(cdf) })
    }
}

impl Distribution<u64> for Poisson {
    fn sample(&self, rng: &mut Rng) -> u64 {
        match &self.method {
            PoissonMethod::Table(cdf) => {
                let u = rng.random();
                // the first k with u <= P(X <= k); past the table's end only by rounding
                cdf.iter().position(|&c| u <= c).unwrap_or(cdf.len() - 1) as u64
            }
            PoissonMethod::Ptrs(p) => p.sample(rng),
        }
    }
}

/// Binomial: successes in `n` trials of probability `p`.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Binomial {
    n: u64,
    p: f64,
}

impl Binomial {
    /// Errors unless `p` is in [0, 1].
    pub fn new(n: u64, p: f64) -> Result<Self, RandomError> {
        Ok(Self { n, p: probability(p)? })
    }
}

impl Distribution<u64> for Binomial {
    fn sample(&self, rng: &mut Rng) -> u64 {
        rng.binomial(self.n, self.p)
    }
}

/// Bernoulli: `true` with probability `p`.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Bernoulli {
    p: f64,
}

impl Bernoulli {
    /// Errors unless `p` is in [0, 1].
    pub fn new(p: f64) -> Result<Self, RandomError> {
        Ok(Self { p: probability(p)? })
    }
}

impl Distribution<bool> for Bernoulli {
    fn sample(&self, rng: &mut Rng) -> bool {
        rng.random() < self.p
    }
}

/// Geometric: the number of trials up to and including the first success (at least 1).
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Geometric {
    p: f64,
}

impl Geometric {
    /// Errors unless `0 < p <= 1`.
    pub fn new(p: f64) -> Result<Self, RandomError> {
        if p > 0.0 && p <= 1.0 { Ok(Self { p }) } else { Err(RandomError::invalid(format!("p must be in (0, 1], got {p}"))) }
    }
}

impl Distribution<u64> for Geometric {
    fn sample(&self, rng: &mut Rng) -> u64 {
        if self.p == 1.0 {
            return 1;
        }
        // inversion: ceil(ln(1 - U) / ln(1 - p))
        ((-rng.random()).ln_1p() / (-self.p).ln_1p()).ceil().max(1.0) as u64
    }
}

#[cfg(test)]
mod tests;
