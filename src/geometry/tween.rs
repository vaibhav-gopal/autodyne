//! Tweens: values moving from a start to an end over a duration along an easing curve, one at a
//! time ([`Ease::apply`]) or many as columns ([`Tweens`]).
//!
//! tend: Geometry and lanes / batched kernels

use crate::alloc_prelude::*;
use crate::units::*;

/// An easing curve: progress 0..1 in, eased progress out (0 at 0, 1 at 1; Bézier curves may
/// overshoot in between). The named curves are the usual ones (easings.net); `CubicBezier` is CSS's
/// `cubic-bezier(x1, y1, x2, y2)`.
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum Ease<T> {
    /// No easing.
    Linear,
    /// `t²`.
    InQuad,
    /// `1 - (1 - t)²`.
    OutQuad,
    /// Quadratic in, then out.
    InOutQuad,
    /// `t³`.
    InCubic,
    /// `1 - (1 - t)³`.
    OutCubic,
    /// Cubic in, then out.
    InOutCubic,
    /// `2^(10 t - 10)` (0 at 0).
    InExpo,
    /// `1 - 2^(-10 t)` (1 at 1).
    OutExpo,
    /// Exponential in, then out.
    InOutExpo,
    /// A cubic Bézier from (0, 0) to (1, 1) with control points (x1, y1) and (x2, y2); x1 and x2
    /// within 0..=1 (clamped), so the curve is a function of x.
    CubicBezier {
        /// First control point, x.
        x1: T,
        /// First control point, y.
        y1: T,
        /// Second control point, x.
        x2: T,
        /// Second control point, y.
        y2: T,
    },
}

impl<T: Float> Ease<T> {
    /// CSS's `ease`: `cubic-bezier(0.25, 0.1, 0.25, 1)`.
    pub fn css_ease() -> Self {
        Ease::CubicBezier { x1: T::_lit(0.25), y1: T::_lit(0.1), x2: T::_lit(0.25), y2: T::_ONE }
    }

    /// The eased progress at `t` (clamped to 0..=1).
    pub fn apply(&self, t: T) -> T {
        let t = t._clamp(T::_ZERO, T::_ONE);
        let (one, two, half) = (T::_ONE, T::_lit(2.0), T::_lit(0.5));
        match *self {
            Ease::Linear => t,
            Ease::InQuad => t * t,
            Ease::OutQuad => one - (one - t) * (one - t),
            Ease::InOutQuad => {
                if t < half {
                    two * t * t
                } else {
                    let u = two - two * t;
                    one - u * u * half
                }
            }
            Ease::InCubic => t * t * t,
            Ease::OutCubic => {
                let u = one - t;
                one - u * u * u
            }
            Ease::InOutCubic => {
                if t < half {
                    T::_lit(4.0) * t * t * t
                } else {
                    let u = two - two * t;
                    one - u * u * u * half
                }
            }
            Ease::InExpo => {
                if t == T::_ZERO { t } else { (T::_lit(10.0) * t - T::_lit(10.0))._exp2() }
            }
            Ease::OutExpo => {
                if t == one { t } else { one - (T::_lit(-10.0) * t)._exp2() }
            }
            Ease::InOutExpo => {
                if t == T::_ZERO || t == one {
                    t
                } else if t < half {
                    (T::_lit(20.0) * t - T::_lit(10.0))._exp2() * half
                } else {
                    (two - (T::_lit(-20.0) * t + T::_lit(10.0))._exp2()) * half
                }
            }
            Ease::CubicBezier { x1, y1, x2, y2 } => bezier(x1._clamp(T::_ZERO, one), y1, x2._clamp(T::_ZERO, one), y2, t),
        }
    }
}

/// The y of the CSS cubic Bézier at x = `t`: the curve parameter by Newton's method from `t`, or by
/// bisection where the slope is too flat for Newton to be trusted.
fn bezier<T: Float>(x1: T, y1: T, x2: T, y2: T, t: T) -> T {
    // B(u) = 3(1-u)² u p1 + 3(1-u) u² p2 + u³ = ((a u + b) u + c) u with these coefficients
    let coeffs = |p1: T, p2: T| {
        let c = T::_lit(3.0) * p1;
        let b = T::_lit(3.0) * (p2 - p1) - c;
        (T::_ONE - c - b, b, c)
    };
    let (ax, bx, cx) = coeffs(x1, x2);
    let (ay, by, cy) = coeffs(y1, y2);
    let x_at = |u: T| ((ax * u + bx) * u + cx) * u;
    let tolerance = T::_EPSILON * T::_lit(4.0);
    let mut u = t;
    let mut solved = false;
    for _ in 0..8 {
        let err = x_at(u) - t;
        if err._abs() <= tolerance {
            solved = true;
            break;
        }
        let slope = (T::_lit(3.0) * ax * u + T::_lit(2.0) * bx) * u + cx;
        if slope._abs() < T::_lit(1e-6) {
            break;
        }
        u = u - err / slope;
    }
    if !solved || !(T::_ZERO..=T::_ONE).contains(&u) {
        // x(u) increases on 0..=1 (x1, x2 within it): bisect
        let (mut lo, mut hi) = (T::_ZERO, T::_ONE);
        u = t;
        for _ in 0..64 {
            let x = x_at(u);
            if (x - t)._abs() <= tolerance {
                break;
            }
            if x < t {
                lo = u;
            } else {
                hi = u;
            }
            u = (lo + hi) * T::_lit(0.5);
        }
    }
    ((ay * u + by) * u + cy) * u
}

/// Tweens as columns: each lane goes from `from` to `to` over `duration` seconds starting at
/// `start`, along its [`Ease`]. [`sample`](Self::sample) writes every lane's value at a time and
/// reports the lanes that finished. Allocates only to add lanes.
#[derive(Clone, Debug)]
pub struct Tweens<T> {
    from: Vec<T>,
    to: Vec<T>,
    start: Vec<T>,
    /// 1 / duration (the largest finite value for a zero duration)
    rate: Vec<T>,
    ease: Vec<Ease<T>>,
    value: Vec<T>,
    running: Vec<bool>,
    finished: Vec<usize>,
}

impl<T: Float> Default for Tweens<T> {
    fn default() -> Self {
        Self::new()
    }
}

impl<T: Float> Tweens<T> {
    /// No lanes.
    pub fn new() -> Self {
        Tweens { from: Vec::new(), to: Vec::new(), start: Vec::new(), rate: Vec::new(), ease: Vec::new(), value: Vec::new(), running: Vec::new(), finished: Vec::new() }
    }

    /// Adds a lane from `from` to `to`, starting at time `start` and lasting `duration` seconds
    /// (0 jumps to `to` as soon as time passes `start`); returns its index.
    pub fn add(&mut self, from: T, to: T, start: T, duration: T, ease: Ease<T>) -> usize {
        self.from.push(from);
        self.to.push(to);
        self.start.push(start);
        self.rate.push(Self::rate(duration));
        self.ease.push(ease);
        self.value.push(from);
        self.running.push(true);
        self.finished.reserve(1);
        self.from.len() - 1
    }

    /// Removes lane `i`; the last lane takes its index (as `Vec::swap_remove`).
    pub fn remove(&mut self, i: usize) {
        for column in [&mut self.from, &mut self.to, &mut self.start, &mut self.rate, &mut self.value] {
            column.swap_remove(i);
        }
        self.ease.swap_remove(i);
        self.running.swap_remove(i);
    }

    /// Restarts lane `i` from its current value toward `to`, at `start`, over `duration`.
    pub fn retarget(&mut self, i: usize, to: T, start: T, duration: T) {
        self.from[i] = self.value[i];
        self.to[i] = to;
        self.start[i] = start;
        self.rate[i] = Self::rate(duration);
        self.running[i] = true;
    }

    /// A duration's reciprocal, the largest finite value for zero (or less): any time past the start
    /// is then the end.
    fn rate(duration: T) -> T {
        if duration > T::_ZERO { (T::_ONE / duration)._min(T::_MAX) } else { T::_MAX }
    }

    /// Lanes.
    pub fn len(&self) -> usize {
        self.from.len()
    }
    /// Whether there are no lanes.
    pub fn is_empty(&self) -> bool {
        self.from.is_empty()
    }
    /// Every lane's value at the last [`sample`](Self::sample).
    pub fn values(&self) -> &[T] {
        &self.value
    }

    /// Every lane's value at time `now`. Returns how many lanes finished (reached their end) at this
    /// sample; they are [`finished`](Self::finished).
    pub fn sample(&mut self, now: T) -> usize {
        self.finished.clear();
        // zipped, not indexed: the columns' lengths are then known equal (no bounds checks)
        let lanes = self.from.iter().zip(&self.to).zip(&self.start).zip(&self.rate).zip(&self.ease);
        for (i, (((((&from, &to), &start), &rate), ease), (value, running))) in lanes.zip(self.value.iter_mut().zip(self.running.iter_mut())).enumerate() {
            let t = (now - start) * rate;
            if *running && t >= T::_ONE {
                *value = to;
                *running = false;
                self.finished.push(i);
            } else {
                *value = from + (to - from) * ease.apply(t);
            }
        }
        self.finished.len()
    }

    /// The lanes that finished at the last [`sample`](Self::sample), ascending.
    pub fn finished(&self) -> &[usize] {
        &self.finished
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn named_curves_hit_their_formulas_and_ends() {
        let all = [
            Ease::Linear,
            Ease::InQuad,
            Ease::OutQuad,
            Ease::InOutQuad,
            Ease::InCubic,
            Ease::OutCubic,
            Ease::InOutCubic,
            Ease::InExpo,
            Ease::OutExpo,
            Ease::InOutExpo,
            Ease::css_ease(),
        ];
        for e in all {
            assert_eq!((e.apply(0.0f64), e.apply(1.0)), (0.0, 1.0), "{e:?}");
            assert_eq!((e.apply(-1.0), e.apply(2.0)), (0.0, 1.0), "{e:?} clamps");
        }
        let t = 0.3f64;
        let cases = [
            (Ease::InQuad, 0.09),
            (Ease::OutQuad, 0.51),
            (Ease::InOutQuad, 0.18),
            (Ease::InCubic, 0.027),
            (Ease::OutCubic, 0.657),
            (Ease::InOutCubic, 0.108),
            (Ease::InExpo, 2f64.powf(-7.0)),
            (Ease::OutExpo, 1.0 - 2f64.powf(-3.0)),
            (Ease::InOutExpo, 2f64.powf(-4.0) / 2.0),
        ];
        for (e, want) in cases {
            assert!((e.apply(t) - want).abs() < 1e-15, "{e:?}({t}) = {}, want {want}", e.apply(t));
        }
        // in-out curves are point-symmetric about (0.5, 0.5)
        for e in [Ease::InOutQuad, Ease::InOutCubic, Ease::InOutExpo] {
            assert!((e.apply(0.5f64) - 0.5).abs() < 1e-15 && (e.apply(0.8) + e.apply(0.2) - 1.0).abs() < 1e-15, "{e:?}");
        }
    }

    #[test]
    fn cubic_beziers_match_high_precision_references() {
        // (curve, x, y) from mpmath at 50 digits
        let ease = Ease::css_ease();
        let back = Ease::CubicBezier { x1: 0.68, y1: -0.55, x2: 0.265, y2: 1.55 };
        for (e, x, y) in [(ease, 0.1, 0.09479630571604325), (ease, 0.5, 0.802403387584857), (ease, 0.9, 0.9943164774845563), (back, 0.2, -0.09270169482331352), (back, 0.7, 1.0692605606108994)] {
            assert!((e.apply(x) - y).abs() < 1e-13, "{e:?}({x}) = {}, want {y}", e.apply(x));
        }
        // the diagonal is the identity; a curve with flat ends (slope 0 at both) still solves
        let diagonal = Ease::CubicBezier { x1: 0.0, y1: 0.0, x2: 1.0, y2: 1.0 };
        let flat = Ease::CubicBezier { x1: 1.0, y1: 0.0, x2: 0.0, y2: 1.0 };
        for x in [0.001, 0.25, 0.5, 0.75, 0.999] {
            assert!((diagonal.apply(x) - x).abs() < 1e-14);
            assert!((flat.apply(x) + flat.apply(1.0 - x) - 1.0).abs() < 1e-12, "flat at {x}");
        }
        let f = Ease::<f32>::css_ease();
        assert!((f.apply(0.5) - 0.802_403_4).abs() < 1e-6);
    }

    #[test]
    fn lanes_run_retarget_and_report() {
        let mut t = Tweens::new();
        let a = t.add(0.0, 10.0, 0.0, 1.0, Ease::Linear);
        let b = t.add(100.0, 0.0, 0.5, 0.5, Ease::OutQuad);
        let c = t.add(3.0, 7.0, 0.2, 0.0, Ease::InExpo); // zero duration: jumps at its start
        assert_eq!(t.sample(0.1), 0);
        assert_eq!(t.values(), &[1.0, 100.0, 3.0]);
        assert_eq!(t.sample(0.5), 1);
        assert_eq!(t.finished(), &[c]);
        assert_eq!(t.values()[a], 5.0);
        // retargeting starts from where the lane is
        t.retarget(a, 0.0, 0.5, 0.5);
        assert_eq!(t.sample(0.75), 0);
        assert_eq!(t.values()[a], 2.5);
        assert_eq!(t.values()[b], 100.0 * 0.25);
        assert_eq!(t.sample(1.0), 2);
        assert_eq!(t.finished(), &[a, b]);
        assert_eq!(t.values(), &[0.0, 0.0, 7.0]);
        t.remove(a);
        assert_eq!((t.len(), t.values()[0]), (2, 7.0));
    }
}
