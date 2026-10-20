//! Damped springs, many at once: the exact motion of `x'' = -ω² (x - target) - 2ζω x'` over any
//! time step, for lanes kept as columns (structure of arrays), with a settle test that reports the
//! lanes that came to rest.
//!
//! tend: Geometry and lanes / springs

use crate::alloc_prelude::*;
use crate::units::*;

/// How a spring moves: its natural angular frequency `omega` (rad/s) and damping ratio `zeta`
/// (below 1 it overshoots and rings, 1 is the fastest return without overshoot, above 1 slower).
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct SpringConfig<T> {
    /// Natural angular frequency, rad/s.
    pub omega: T,
    /// Damping ratio.
    pub zeta: T,
}

impl<T: Float> SpringConfig<T> {
    /// A spring that completes an undamped oscillation in `response` seconds (`omega = 2π /
    /// response`), with `damping_ratio` (SwiftUI's and Framer's parameters).
    pub fn from_response(response: T, damping_ratio: T) -> Self {
        SpringConfig { omega: T::_TAU / response, zeta: damping_ratio }
    }

    /// A mass on a spring with a damper: `omega = sqrt(k / m)`, `zeta = c / (2 sqrt(k m))`.
    pub fn from_physical(stiffness: T, damping: T, mass: T) -> Self {
        SpringConfig { omega: (stiffness / mass)._sqrt(), zeta: damping / (T::_lit(2.0) * (stiffness * mass)._sqrt()) }
    }

    /// The exact state transition over `dt`: with `x` the displacement from the target and `v` the
    /// velocity, `[x', v'] = [[m0, m1], [m2, m3]] [x, v]`, returned as `[m0, m1, m2, m3]`.
    ///
    /// All three regimes in one form: `e^(-a t) [[c + a s, s], [-ω² s, c - a s]]` with `a = ζω`,
    /// and `c`, `s` = `cos(w t)`, `sin(w t) / w` (under-damped, `w² = ω² - a²`) or `cosh`, `sinh / w`
    /// (over-damped); near critical damping their Taylor series, so there is no division by a
    /// vanishing `w`.
    pub fn transition(self, dt: T) -> [T; 4] {
        let a = self.zeta * self.omega;
        let q = a * a - self.omega * self.omega; // > 0 over-damped, < 0 under-damped
        let qt2 = q * dt * dt;
        let (c, s) = if qt2._abs() < T::_lit(1e-2) {
            // c = Σ (q t²)^k / (2k)!, s = t Σ (q t²)^k / (2k + 1)!: four terms, error below 3e-13
            let c = T::_ONE + qt2 * (T::_lit(0.5) + qt2 * (T::_lit(1.0 / 24.0) + qt2 * (T::_lit(1.0 / 720.0) + qt2 * T::_lit(1.0 / 40_320.0))));
            let s = dt * (T::_ONE + qt2 * (T::_lit(1.0 / 6.0) + qt2 * (T::_lit(1.0 / 120.0) + qt2 * (T::_lit(1.0 / 5_040.0) + qt2 * T::_lit(1.0 / 362_880.0)))));
            (c, s)
        } else if q > T::_ZERO {
            let w = q._sqrt();
            // cosh and sinh from one exponential (core has neither)
            let e = (w * dt)._exp();
            let inv = T::_ONE / e;
            (T::_lit(0.5) * (e + inv), T::_lit(0.5) * (e - inv) / w)
        } else {
            let w = (-q)._sqrt();
            let (sin, cos) = (w * dt)._sin_cos();
            (cos, sin / w)
        };
        let decay = (-a * dt)._exp();
        [decay * (c + a * s), decay * s, -decay * self.omega * self.omega * s, decay * (c - a * s)]
    }
}

/// Springs as columns: positions, velocities and targets, each lane with one of a few shared
/// [`SpringConfig`]s.
///
/// [`step`](Self::step) moves every lane exactly (any `dt`, no integration error): it computes each
/// config's transition once, then each lane is two multiply-adds per coordinate. Retargeting keeps
/// position and velocity. A moving lane within the rest tolerances of its target is snapped to it,
/// stopped, and reported in [`settled`](Self::settled) for the step it came to rest. Lanes are 1-D:
/// a 2-D spring is two lanes. Allocates only to add lanes (and configs).
#[derive(Clone, Debug)]
pub struct Springs<T> {
    position: Vec<T>,
    velocity: Vec<T>,
    target: Vec<T>,
    config: Vec<u32>,
    moving: Vec<bool>,
    configs: Vec<SpringConfig<T>>,
    matrices: Vec<[T; 4]>,
    rest_distance: T,
    rest_speed: T,
    settled: Vec<usize>,
}

impl<T: Float> Springs<T> {
    /// No lanes. Lanes rest within `rest_distance` of their target moving slower than `rest_speed`
    /// (in position units, and units per second).
    pub fn new(rest_distance: T, rest_speed: T) -> Self {
        Springs {
            position: Vec::new(),
            velocity: Vec::new(),
            target: Vec::new(),
            config: Vec::new(),
            moving: Vec::new(),
            configs: Vec::new(),
            matrices: Vec::new(),
            rest_distance,
            rest_speed,
            settled: Vec::new(),
        }
    }

    /// Adds a lane at `position`, at rest, heading for `target`; returns its index. Lanes sharing a
    /// config (equal parameters) share its transition.
    pub fn add(&mut self, position: T, target: T, config: SpringConfig<T>) -> usize {
        let id = match self.configs.iter().position(|c| *c == config) {
            Some(i) => i,
            None => {
                self.configs.push(config);
                self.matrices.push([T::_ONE, T::_ZERO, T::_ZERO, T::_ONE]);
                self.configs.len() - 1
            }
        };
        self.position.push(position);
        self.velocity.push(T::_ZERO);
        self.target.push(target);
        self.config.push(id as u32);
        self.moving.push(true);
        self.settled.reserve(1);
        self.position.len() - 1
    }

    /// Removes lane `i`; the last lane takes its index (as `Vec::swap_remove`). Panics if out of range.
    pub fn remove(&mut self, i: usize) {
        self.position.swap_remove(i);
        self.velocity.swap_remove(i);
        self.target.swap_remove(i);
        self.config.swap_remove(i);
        self.moving.swap_remove(i);
    }

    /// Lanes.
    pub fn len(&self) -> usize {
        self.position.len()
    }
    /// Whether there are no lanes.
    pub fn is_empty(&self) -> bool {
        self.position.is_empty()
    }
    /// Positions, by lane.
    pub fn positions(&self) -> &[T] {
        &self.position
    }
    /// Velocities, by lane.
    pub fn velocities(&self) -> &[T] {
        &self.velocity
    }
    /// Targets, by lane.
    pub fn targets(&self) -> &[T] {
        &self.target
    }
    /// Whether lane `i` is still moving (not settled since it was last given a target or a push).
    pub fn is_moving(&self, i: usize) -> bool {
        self.moving[i]
    }

    /// Sends lane `i` toward `target` from where it is, at its current velocity.
    pub fn set_target(&mut self, i: usize, target: T) {
        self.target[i] = target;
        self.moving[i] = true;
    }
    /// Adds `dv` to lane `i`'s velocity (a flick).
    pub fn push_velocity(&mut self, i: usize, dv: T) {
        self.velocity[i] = self.velocity[i] + dv;
        self.moving[i] = true;
    }
    /// Moves lane `i` to `position` and stops it there (its target unchanged).
    pub fn jump(&mut self, i: usize, position: T) {
        self.position[i] = position;
        self.velocity[i] = T::_ZERO;
        self.moving[i] = true;
    }

    /// Advances every lane by `dt` seconds, exactly. Returns how many lanes came to rest; they are
    /// [`settled`](Self::settled).
    pub fn step(&mut self, dt: T) -> usize {
        for (m, c) in self.matrices.iter_mut().zip(&self.configs) {
            *m = c.transition(dt);
        }
        for i in 0..self.position.len() {
            let m = self.matrices[self.config[i] as usize];
            let (x, v, t) = (self.position[i] - self.target[i], self.velocity[i], self.target[i]);
            self.position[i] = t + (m[0] * x + m[1] * v);
            self.velocity[i] = m[2] * x + m[3] * v;
        }
        self.settled.clear();
        for i in 0..self.position.len() {
            if self.moving[i] && (self.position[i] - self.target[i])._abs() <= self.rest_distance && self.velocity[i]._abs() <= self.rest_speed {
                self.position[i] = self.target[i];
                self.velocity[i] = T::_ZERO;
                self.moving[i] = false;
                self.settled.push(i);
            }
        }
        self.settled.len()
    }

    /// The lanes that came to rest in the last [`step`](Self::step), ascending.
    pub fn settled(&self) -> &[usize] {
        &self.settled
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// x'' = -ω² x - 2ζω x' by classic Runge-Kutta, `n` substeps over `t`.
    fn rk4(c: SpringConfig<f64>, (mut x, mut v): (f64, f64), t: f64, n: usize) -> (f64, f64) {
        let f = |x: f64, v: f64| (v, -c.omega * c.omega * x - 2.0 * c.zeta * c.omega * v);
        let h = t / n as f64;
        for _ in 0..n {
            let k1 = f(x, v);
            let k2 = f(x + h / 2.0 * k1.0, v + h / 2.0 * k1.1);
            let k3 = f(x + h / 2.0 * k2.0, v + h / 2.0 * k2.1);
            let k4 = f(x + h * k3.0, v + h * k3.1);
            x += h / 6.0 * (k1.0 + 2.0 * k2.0 + 2.0 * k3.0 + k4.0);
            v += h / 6.0 * (k1.1 + 2.0 * k2.1 + 2.0 * k3.1 + k4.1);
        }
        (x, v)
    }

    fn apply(m: [f64; 4], (x, v): (f64, f64)) -> (f64, f64) {
        (m[0] * x + m[1] * v, m[2] * x + m[3] * v)
    }

    #[test]
    fn every_regime_matches_the_differential_equation() {
        for zeta in [0.0, 0.1, 0.5, 0.9, 0.999, 1.0, 1.001, 1.5, 4.0] {
            let c = SpringConfig { omega: 12.0, zeta };
            for (t, start) in [(0.016, (1.0, 0.0)), (0.25, (-0.3, 5.0)), (1.3, (2.0, -8.0))] {
                let want = rk4(c, start, t, 20_000);
                let got = apply(c.transition(t), start);
                let err = (got.0 - want.0).abs().max((got.1 - want.1).abs() / c.omega);
                assert!(err < 1e-9, "zeta {zeta}, t {t}: {got:?} vs {want:?}");
            }
        }
    }

    #[test]
    fn closed_forms_and_continuity_at_critical_damping() {
        // critically damped from rest at x0: x(t) = x0 (1 + ω t) e^(-ω t)
        let c = SpringConfig { omega: 8.0, zeta: 1.0 };
        for t in [0.01, 0.1, 0.5, 2.0] {
            let (x, _) = apply(c.transition(t), (1.0, 0.0));
            assert!((x - (1.0 + 8.0 * t) * (-8.0 * t).exp()).abs() < 1e-14, "t {t}");
        }
        // undamped: a pure cosine
        let (x, v) = apply(SpringConfig { omega: 3.0, zeta: 0.0 }.transition(0.7), (1.0, 0.0));
        assert!((x - (2.1f64).cos()).abs() < 1e-14 && (v + 3.0 * (2.1f64).sin()).abs() < 1e-13);
        // the series and the closed forms agree where they meet
        for zeta in [1.0 - 1e-7, 1.0 + 1e-7, 1.0 - 0.05, 1.0 + 0.05] {
            let m = SpringConfig { omega: 10.0, zeta }.transition(0.2);
            let crit = SpringConfig { omega: 10.0, zeta: 1.0 }.transition(0.2);
            let gap = m.iter().zip(&crit).map(|(a, b)| (a - b).abs()).fold(0.0, f64::max);
            assert!(gap < 2.0 * (zeta - 1.0f64).abs(), "zeta {zeta}: {gap}");
        }
    }

    #[test]
    fn steps_compose_exactly() {
        // the transition over 0.5 s equals fifty transitions over 0.01 s: no integration error
        let c = SpringConfig::from_response(0.4, 0.6);
        let whole = apply(c.transition(0.5), (1.0, 2.0));
        let mut s = (1.0, 2.0);
        for _ in 0..50 {
            s = apply(c.transition(0.01), s);
        }
        assert!((s.0 - whole.0).abs() < 1e-13 && (s.1 - whole.1).abs() < 1e-12, "{s:?} vs {whole:?}");
    }

    #[test]
    fn parameterizations() {
        let c = SpringConfig::from_physical(400.0, 20.0, 1.0);
        assert!((c.omega - 20.0).abs() < 1e-12 && (c.zeta - 0.5).abs() < 1e-12);
        let c = SpringConfig::<f64>::from_response(0.5, 1.0);
        assert!((c.omega - 4.0 * core::f64::consts::PI).abs() < 1e-12);
    }

    #[test]
    fn lanes_retarget_settle_and_report() {
        let mut s = Springs::new(1e-3, 1e-2);
        let snappy = SpringConfig::from_response(0.2, 1.0);
        let bouncy = SpringConfig::from_response(0.3, 0.3);
        let a = s.add(0.0, 100.0, snappy);
        let b = s.add(0.0, -50.0, bouncy);
        let c = s.add(5.0, 5.0, snappy); // already there: settles on the first step
        assert_eq!(s.configs.len(), 2);
        assert_eq!(s.step(1.0 / 60.0), 1);
        assert_eq!(s.settled(), &[c]);
        // halfway, retarget a: position and velocity carry over
        for _ in 0..6 {
            s.step(1.0 / 60.0);
        }
        let (p, v) = (s.positions()[a], s.velocities()[a]);
        assert!(p > 10.0 && v > 0.0);
        s.set_target(a, 0.0);
        assert_eq!((s.positions()[a], s.velocities()[a]), (p, v));
        // run until both rest, each reported once, snapped onto its target
        let mut reported = Vec::new();
        for _ in 0..600 {
            s.step(1.0 / 60.0);
            reported.extend_from_slice(s.settled());
        }
        reported.sort();
        assert_eq!(reported, [a, b]);
        assert_eq!((s.positions()[a], s.positions()[b]), (0.0, -50.0));
        assert!(!s.is_moving(a) && s.velocities()[b] == 0.0);
        // a flick wakes a lane; removal moves the last lane into the gap
        s.push_velocity(b, 30.0);
        assert!(s.is_moving(b));
        s.remove(a);
        assert_eq!((s.len(), s.targets()[a]), (2, 5.0));
    }

    #[test]
    fn single_precision_lanes() {
        let mut s = Springs::<f32>::new(1e-3, 1e-2);
        let i = s.add(0.0, 1.0, SpringConfig::from_response(0.25, 0.8));
        let mut steps = 0;
        while s.is_moving(i) {
            s.step(1.0 / 120.0);
            steps += 1;
            assert!(steps < 1_000, "never settled");
        }
        // overshoot is bounded by the damping: about e^(-πζ/√(1-ζ²)) = 1.5% for ζ = 0.8
        assert_eq!(s.positions()[i], 1.0);
    }
}
