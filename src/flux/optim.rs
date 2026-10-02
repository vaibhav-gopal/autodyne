//! Gradient-based optimizers for fitting: they update parameter arrays in place from gradients of
//! the same shapes (as [`Scan::loss_grad`](super::Scan::loss_grad) returns them).
//!
//! ```
//! use autodyne::flux::{optim::{Adam, Optimizer}, scalar};
//!
//! // minimize (p - 3)²: the gradient is 2 (p - 3)
//! let mut p = vec![scalar(0.0)];
//! let mut adam = Adam::new(0.1);
//! for _ in 0..500 {
//!     let g = vec![scalar(2.0 * (p[0].as_slice()[0] - 3.0))];
//!     adam.step(&mut p, &g);
//! }
//! assert!((p[0].as_slice()[0] - 3.0).abs() < 1e-3);
//! ```

use super::graph::FluxFloat;
use crate::signal::NdArray;

/// Updates parameters from their gradients. Hyperparameters and state are `f64`; parameters are
/// `f32` or `f64`.
pub trait Optimizer {
    /// One update of `params` (in place) against `grads` (same count and shapes).
    fn step<T: FluxFloat>(&mut self, params: &mut [NdArray<T>], grads: &[NdArray<T>]);
}

fn check<T: FluxFloat>(params: &[NdArray<T>], grads: &[NdArray<T>]) {
    assert_eq!(params.len(), grads.len(), "one gradient per parameter");
    for (k, (p, g)) in params.iter().zip(grads).enumerate() {
        assert_eq!(p.shape(), g.shape(), "gradient {k} has the wrong shape");
    }
}

/// Per-parameter state, created on the first step.
fn zeros_like<T: FluxFloat>(params: &[NdArray<T>]) -> Vec<Vec<f64>> {
    params.iter().map(|p| vec![0.0; p.len()]).collect()
}

/// Stochastic gradient descent with optional momentum (`v = μ v + g`, `p -= lr v`).
#[derive(Debug, Clone)]
pub struct Sgd {
    pub lr: f64,
    pub momentum: f64,
    velocity: Vec<Vec<f64>>,
}

impl Sgd {
    pub fn new(lr: f64) -> Self {
        Sgd { lr, momentum: 0.0, velocity: Vec::new() }
    }
    pub fn with_momentum(mut self, momentum: f64) -> Self {
        self.momentum = momentum;
        self
    }
}

impl Optimizer for Sgd {
    fn step<T: FluxFloat>(&mut self, params: &mut [NdArray<T>], grads: &[NdArray<T>]) {
        check(params, grads);
        if self.velocity.is_empty() {
            self.velocity = zeros_like(params);
        }
        for ((p, g), v) in params.iter_mut().zip(grads).zip(&mut self.velocity) {
            for ((p, &g), v) in p.as_mut_slice().iter_mut().zip(g.as_slice()).zip(v.iter_mut()) {
                *v = self.momentum * *v + wide(g);
                *p = T::_lit(wide(*p) - self.lr * *v);
            }
        }
    }
}

/// Adam (Kingma & Ba): per-element step sizes from running estimates of the gradient's first and
/// second moments, bias-corrected. Defaults β₁ = 0.9, β₂ = 0.999, ε = 1e-8.
#[derive(Debug, Clone)]
pub struct Adam {
    pub lr: f64,
    pub beta1: f64,
    pub beta2: f64,
    pub eps: f64,
    t: i32,
    m: Vec<Vec<f64>>,
    v: Vec<Vec<f64>>,
}

impl Adam {
    pub fn new(lr: f64) -> Self {
        Adam { lr, beta1: 0.9, beta2: 0.999, eps: 1e-8, t: 0, m: Vec::new(), v: Vec::new() }
    }
    pub fn with_betas(mut self, beta1: f64, beta2: f64) -> Self {
        (self.beta1, self.beta2) = (beta1, beta2);
        self
    }
    /// The number of steps taken.
    pub fn steps(&self) -> i32 {
        self.t
    }
}

impl Optimizer for Adam {
    fn step<T: FluxFloat>(&mut self, params: &mut [NdArray<T>], grads: &[NdArray<T>]) {
        check(params, grads);
        if self.m.is_empty() {
            (self.m, self.v) = (zeros_like(params), zeros_like(params));
        }
        self.t += 1;
        let (c1, c2) = (1.0 - self.beta1.powi(self.t), 1.0 - self.beta2.powi(self.t));
        for (((p, g), m), v) in params.iter_mut().zip(grads).zip(&mut self.m).zip(&mut self.v) {
            for (((p, &g), m), v) in p.as_mut_slice().iter_mut().zip(g.as_slice()).zip(m.iter_mut()).zip(v.iter_mut()) {
                let g = wide(g);
                *m = self.beta1 * *m + (1.0 - self.beta1) * g;
                *v = self.beta2 * *v + (1.0 - self.beta2) * g * g;
                *p = T::_lit(wide(*p) - self.lr * (*m / c1) / ((*v / c2).sqrt() + self.eps));
            }
        }
    }
}

/// Scales `grads` in place so their combined L2 norm is at most `max_norm`; returns the norm before
/// clipping.
pub fn clip_grad_norm<T: FluxFloat>(grads: &mut [NdArray<T>], max_norm: f64) -> f64 {
    let norm = grads.iter().flat_map(|g| g.as_slice()).map(|&x| wide(x) * wide(x)).sum::<f64>().sqrt();
    if norm > max_norm && norm > 0.0 {
        let s = max_norm / norm;
        for g in grads.iter_mut() {
            g.as_mut_slice().iter_mut().for_each(|x| *x = T::_lit(wide(*x) * s));
        }
    }
    norm
}

fn wide<T: FluxFloat>(x: T) -> f64 {
    x.to_f64().unwrap_or(f64::NAN)
}
