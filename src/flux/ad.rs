//! Reverse-mode differentiation as a graph transformation: the backward pass is recorded into the
//! same trace as the forward pass, so it is evaluated, emitted and compiled like any other code.

use super::graph::{self, Id, Mask, Op, Part, Tracer};
use crate::signal::NdArray;
use crate::units::Real;

/// Vector-Jacobian product inside a trace.
///
/// Given `outputs` and a cotangent for each (of the same shape), returns `Σ cotangent · ∂output/∂w`
/// for every `w` in `wrt` (zero where nothing depends on `w`). The backward pass is recorded into
/// the current trace, reusing forward values where the rules need them.
///
/// ```
/// use autodyne::flux::{scalar, trace, vjp};
/// use autodyne::units::Real;
///
/// // d/dx sin(x) * x = cos(x) * x + sin(x)
/// let g = trace(&[&[]], |v| {
///     let y = v[0].sin() * v[0];
///     vjp(&[y], &[Real::lit(1.0)], &[v[0]])
/// });
/// let x = 0.7f32;
/// assert!((g.eval(&[scalar(x)])[0].as_slice()[0] - (x.cos() * x + x.sin())).abs() < 1e-6);
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
            live[id] = graph::node(id as Id).op.operands().any(|a| live[a as usize]);
        }
    }

    let mut adj: Vec<Option<Tracer>> = vec![None; n];
    let mut acc = |adj: &mut Vec<Option<Tracer>>, id: Id, g: Tracer| {
        if live[id as usize] {
            debug_assert_eq!(g.shape(), graph::node(id).shape, "vjp: cotangent shape");
            let slot = &mut adj[id as usize];
            *slot = Some(match *slot {
                Some(prev) => prev + g,
                None => g,
            });
        }
    };
    for (o, c) in outputs.iter().zip(cotangents) {
        acc(&mut adj, o.id, c.broadcast_to(&o.shape()));
    }

    let t = Tracer::node;
    let zero = || Tracer::lit(0.0);
    for id in (lo..n).rev() {
        let Some(g) = adj[id] else { continue };
        let id = id as Id;
        let shape_of = |i: Id| graph::node(i).shape;
        match graph::node(id).op {
            Op::Input(_) | Op::Const(_) | Op::Literal(_) | Op::Compare(..) => {}
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
            Op::Broadcast(a, dims) => {
                // sum over the axes that were added or stretched
                let (from, to) = (shape_of(a), shape_of(id));
                let axes: Vec<usize> = (0..to.len())
                    .filter(|j| match dims.iter().position(|d| d == j) {
                        None => true,
                        Some(i) => from[i] == 1 && to[*j] != 1,
                    })
                    .collect();
                acc(&mut adj, a, g.sum(&axes).reshape(&from));
            }
            Op::Reshape(a) => acc(&mut adj, a, g.reshape(&shape_of(a))),
            Op::Transpose(a, perm) => {
                let mut inverse = vec![0; perm.len()];
                for (i, &p) in perm.iter().enumerate() {
                    inverse[p] = i;
                }
                acc(&mut adj, a, g.transpose(&inverse));
            }
            Op::Sum(a, axes) => {
                let from = shape_of(a);
                let kept: Vec<usize> = (0..from.len()).filter(|x| !axes.contains(x)).collect();
                acc(&mut adj, a, g.broadcast_in_dim(&from, &kept));
            }
            Op::Dot { a, b, ca, cb } => {
                let (sa, sb) = (shape_of(a), shape_of(b));
                let fa: Vec<usize> = (0..sa.len()).filter(|x| !ca.contains(x)).collect();
                let fb: Vec<usize> = (0..sb.len()).filter(|x| !cb.contains(x)).collect();
                if live[a as usize] {
                    // g [fa, fb] · b over fb -> [fa, b's contracted axes ascending], then a's order
                    let g_fb: Vec<usize> = (fa.len()..fa.len() + fb.len()).collect();
                    let r = g.dot_general(t(b), &g_fb, &fb);
                    let mut src = fa.clone();
                    let mut cb_sorted = cb.clone();
                    cb_sorted.sort_unstable();
                    src.extend(cb_sorted.iter().map(|j| ca[cb.iter().position(|x| x == j).unwrap()]));
                    acc(&mut adj, a, r.transpose(&order(&src)));
                }
                if live[b as usize] {
                    // a · g over fa -> [a's contracted axes ascending, fb], then b's order
                    let g_fa: Vec<usize> = (0..fa.len()).collect();
                    let r = t(a).dot_general(g, &fa, &g_fa);
                    let mut ca_sorted = ca.clone();
                    ca_sorted.sort_unstable();
                    let mut src: Vec<usize> = ca_sorted.iter().map(|i| cb[ca.iter().position(|x| x == i).unwrap()]).collect();
                    src.extend(&fb);
                    acc(&mut adj, b, r.transpose(&order(&src)));
                }
            }
            Op::Rfft(a, part) => {
                // transpose of the real DFT: n * irfft(w * cotangent), with w = 1 on bins 0 and n/2
                // (they appear once in the full spectrum) and 1/2 on the others (twice)
                let n = *shape_of(a).last().unwrap();
                let w = Tracer::constant(&weights(n, 1.0, 0.5));
                let (re, im) = match part {
                    Part::Re => (g * w, zero().broadcast_to(&g.shape())),
                    Part::Im => (zero().broadcast_to(&g.shape()), g * w),
                };
                acc(&mut adj, a, Tracer::lit(n as f64) * Tracer::irfft(re, im, n));
            }
            Op::Irfft { re, im, n } => {
                // transpose of the inverse: (s / n) * rfft(cotangent), s = 1 on bins 0 and n/2, else 2
                let (gr, gi) = g.rfft();
                let s = Tracer::constant(&weights(n, 1.0 / n as f64, 2.0 / n as f64));
                acc(&mut adj, re, gr * s);
                acc(&mut adj, im, gi * s);
            }
        }
    }
    wrt.iter().map(|w| adj[w.id as usize].unwrap_or_else(|| zero().broadcast_to(&w.shape()))).collect()
}

/// Per-bin weights for an `n`-point real FFT: `ends` on bins 0 and n/2 (when n is even), `middle`
/// elsewhere.
fn weights(n: usize, ends: f64, middle: f64) -> NdArray<f32> {
    let m = n / 2 + 1;
    let w = (0..m).map(|k| if k == 0 || (n.is_multiple_of(2) && k == n / 2) { ends } else { middle } as f32).collect();
    NdArray::from_vec(w, &[m]).expect("1-D")
}

/// The permutation that puts axes labelled `src` (a permutation of 0..n) into ascending order.
fn order(src: &[usize]) -> Vec<usize> {
    let mut perm = vec![0; src.len()];
    for (r, &s) in src.iter().enumerate() {
        perm[s] = r;
    }
    perm
}

/// Sends `g` to `a` where `mask` holds and to `b` elsewhere.
fn route(adj: &mut Vec<Option<Tracer>>, acc: &mut impl FnMut(&mut Vec<Option<Tracer>>, Id, Tracer), mask: Mask, a: Id, b: Id, g: Tracer) {
    let zero = Tracer::lit(0.0);
    acc(adj, a, Tracer::select(mask, g, zero));
    acc(adj, b, Tracer::select(mask, zero, g));
}
