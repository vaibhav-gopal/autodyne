//! Reverse-mode differentiation as a graph transformation: the backward pass is recorded into the
//! same trace as the forward pass, so it is evaluated, emitted and compiled like any other code.

use super::graph::{self, Id, Mask, Op, Tracer};
use crate::units::Real;

/// Vector-Jacobian product inside a trace.
///
/// Given `outputs` and a cotangent for each, returns `Σ cotangent · ∂output/∂w` for every `w` in
/// `wrt` (zero where nothing depends on `w`). The backward pass is recorded into the current trace,
/// reusing forward values where the rules need them.
///
/// ```
/// use autodyne::flux::{trace, vjp};
/// use autodyne::units::Real;
///
/// // d/dx sin(x) * x = cos(x) * x + sin(x)
/// let g = trace(1, |v| {
///     let y = v[0].sin() * v[0];
///     vjp(&[y], &[Real::lit(1.0)], &[v[0]])
/// });
/// let x = 0.7f32;
/// assert!((g.eval(&[x])[0] - (x.cos() * x + x.sin())).abs() < 1e-6);
/// ```
pub fn vjp(outputs: &[Tracer], cotangents: &[Tracer], wrt: &[Tracer]) -> Vec<Tracer> {
    assert_eq!(outputs.len(), cotangents.len(), "vjp: one cotangent per output");
    let n = graph::len();

    // only nodes that depend on something in `wrt` need an adjoint
    let mut live = vec![false; n];
    for w in wrt {
        live[w.id as usize] = true;
    }
    let lo = wrt.iter().map(|w| w.id as usize).min().unwrap_or(n);
    for id in lo..n {
        if !live[id] {
            live[id] = operands(graph::op(id as Id)).any(|a| live[a as usize]);
        }
    }

    let mut adj: Vec<Option<Tracer>> = vec![None; n];
    let mut acc = |adj: &mut Vec<Option<Tracer>>, id: Id, g: Tracer| {
        if live[id as usize] {
            let slot = &mut adj[id as usize];
            *slot = Some(match *slot {
                Some(prev) => prev + g,
                None => g,
            });
        }
    };
    for (o, c) in outputs.iter().zip(cotangents) {
        acc(&mut adj, o.id, *c);
    }

    let t = Tracer::node;
    let zero = || Tracer::lit(0.0);
    for id in (lo..n).rev() {
        let Some(g) = adj[id] else { continue };
        let id = id as Id;
        match graph::op(id) {
            Op::Input(_) | Op::Const(_) | Op::Compare(..) => {}
            Op::Add(a, b) => {
                acc(&mut adj, a, g);
                acc(&mut adj, b, g);
            }
            Op::Sub(a, b) => {
                acc(&mut adj, a, g);
                acc(&mut adj, b, -g);
            }
            Op::Mul(a, b) => {
                acc(&mut adj, a, g * t(b));
                acc(&mut adj, b, g * t(a));
            }
            Op::Div(a, b) => {
                // y = a / b: da = g / b, db = -g y / b
                let gb = g / t(b);
                acc(&mut adj, a, gb);
                acc(&mut adj, b, -(gb * t(id)));
            }
            Op::Neg(a) => acc(&mut adj, a, -g),
            Op::Exp(a) => acc(&mut adj, a, g * t(id)),
            Op::Log(a) => acc(&mut adj, a, g / t(a)),
            Op::Sin(a) => acc(&mut adj, a, g * t(a).cos()),
            Op::Cos(a) => acc(&mut adj, a, -(g * t(a).sin())),
            Op::Tanh(a) => {
                let y = t(id);
                acc(&mut adj, a, g * (Tracer::lit(1.0) - y * y));
            }
            Op::Sqrt(a) => acc(&mut adj, a, Tracer::lit(0.5) * g / t(id)),
            Op::Abs(a) => {
                let negative = t(a).less(zero());
                acc(&mut adj, a, Tracer::select(negative, -g, g));
            }
            Op::Pow(a, b) => {
                // y = a^b: da = g b a^(b-1), db = g y ln(a) (only when the exponent is live)
                if live[a as usize] {
                    acc(&mut adj, a, g * t(b) * t(a).powf(t(b) - Tracer::lit(1.0)));
                }
                if live[b as usize] {
                    acc(&mut adj, b, g * t(id) * t(a).ln());
                }
            }
            Op::Min(a, b) => route(&mut adj, &mut acc, t(a).less(t(b)), a, b, g),
            Op::Max(a, b) => route(&mut adj, &mut acc, t(a).greater(t(b)), a, b, g),
            Op::Select(c, a, b) => route(&mut adj, &mut acc, Mask::node(c), a, b, g),
        }
    }
    wrt.iter().map(|w| adj[w.id as usize].unwrap_or_else(zero)).collect()
}

/// Sends `g` to `a` where `mask` holds and to `b` elsewhere.
fn route(adj: &mut Vec<Option<Tracer>>, acc: &mut impl FnMut(&mut Vec<Option<Tracer>>, Id, Tracer), mask: Mask, a: Id, b: Id, g: Tracer) {
    let zero = Tracer::lit(0.0);
    acc(adj, a, Tracer::select(mask, g, zero));
    acc(adj, b, Tracer::select(mask, zero, g));
}

fn operands(op: Op) -> impl Iterator<Item = Id> {
    let (a, b, c) = match op {
        Op::Input(_) | Op::Const(_) => (None, None, None),
        Op::Neg(a) | Op::Exp(a) | Op::Log(a) | Op::Sin(a) | Op::Cos(a) | Op::Tanh(a) | Op::Sqrt(a) | Op::Abs(a) => (Some(a), None, None),
        Op::Add(a, b) | Op::Sub(a, b) | Op::Mul(a, b) | Op::Div(a, b) | Op::Pow(a, b) | Op::Min(a, b) | Op::Max(a, b) | Op::Compare(_, a, b) => {
            (Some(a), Some(b), None)
        }
        Op::Select(c, a, b) => (Some(c), Some(a), Some(b)),
    };
    [a, b, c].into_iter().flatten()
}
