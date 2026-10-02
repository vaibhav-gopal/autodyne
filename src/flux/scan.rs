//! `scan`: a per-sample step traced once and run over a whole signal, forward and backward.

use super::ad::vjp;
use super::graph::{trace, Graph, Tracer};

/// A recurrence `step(params, state, x) -> (state', y)`, traced once, run over a signal.
///
/// The step is traced into a [`Graph`] a single time; [`run`](Self::run) and
/// [`loss_grad`](Self::loss_grad) interpret it, and [`forward_hlo`](Self::forward_hlo) /
/// [`loss_grad_hlo`](Self::loss_grad_hlo) emit it as a `stablehlo.while` loop. The gradient is a
/// reverse scan: the forward loop saves each step's input state, the backward loop walks the
/// signal from the end, applying the step's vector-Jacobian product.
///
/// ```
/// use autodyne::filter::OnePole;
/// use autodyne::flux::Scan;
/// use autodyne::units::Real;
///
/// // one parameter (the cutoff), one state value
/// let scan = Scan::trace(1, 1, |p, s, x| {
///     let (s, y) = OnePole::lowpass(p[0], Real::lit(48_000.0)).tick(s[0], x);
///     (vec![s], y)
/// });
/// let (ys, _) = scan.run(&[1_000.0], &[1.0; 64], &[0.0]);
///
/// let mut lp = OnePole::lowpass(1_000.0f32, 48_000.0);
/// let mut block = [1.0f32; 64];
/// lp.process(&mut block);
/// assert_eq!(ys, block);
/// ```
#[derive(Clone, Debug)]
pub struct Scan {
    pub(crate) params: usize,
    pub(crate) states: usize,
    /// inputs: params, state, x; outputs: state', y
    pub(crate) step: Graph,
    /// inputs: params, state, x, target; outputs: state', y, (y - target)²
    pub(crate) step_loss: Graph,
    /// inputs: params, state, x, target, cotangents of state' and of the loss term;
    /// outputs: cotangents of params and state
    pub(crate) step_vjp: Graph,
}

/// The loss and its gradient from [`Scan::loss_grad`].
#[derive(Clone, Debug, PartialEq)]
pub struct LossGrad {
    /// Mean squared error between the outputs and the targets.
    pub loss: f32,
    /// d loss / d params.
    pub params: Vec<f32>,
    /// d loss / d initial state.
    pub state: Vec<f32>,
}

impl Scan {
    /// Traces `step` with `params` parameters and `states` state values. The closure receives the
    /// parameters, the state and the input sample and returns the next state and the output.
    pub fn trace(params: usize, states: usize, step: impl FnOnce(&[Tracer], &[Tracer], Tracer) -> (Vec<Tracer>, Tracer)) -> Scan {
        let (p, s) = (params, states);
        let step = trace(p + s + 1, |v| {
            let (next, y) = step(&v[..p], &v[p..p + s], v[p + s]);
            assert_eq!(next.len(), s, "Scan::trace: the step returned the wrong number of state values");
            next.into_iter().chain([y]).collect()
        });
        let step_loss = trace(p + s + 2, |v| {
            let out = step.call(&v[..p + s + 1]);
            let d = out[s] - v[p + s + 1];
            out.into_iter().chain([d * d]).collect()
        });
        let step_vjp = trace(p + 2 * s + 3, |v| {
            let out = step_loss.call(&v[..p + s + 2]);
            let outputs: Vec<Tracer> = out[..s].iter().copied().chain([out[s + 1]]).collect();
            vjp(&outputs, &v[p + s + 2..], &v[..p + s])
        });
        Scan { params, states, step, step_loss, step_vjp }
    }

    pub fn params(&self) -> usize {
        self.params
    }
    pub fn states(&self) -> usize {
        self.states
    }
    /// The traced step: inputs params, state, x; outputs state', y.
    pub fn step(&self) -> &Graph {
        &self.step
    }

    /// Runs the scan over `xs` from state `s0`: returns the outputs and the final state.
    pub fn run(&self, params: &[f32], xs: &[f32], s0: &[f32]) -> (Vec<f32>, Vec<f32>) {
        self.check(params, s0);
        let mut state = s0.to_vec();
        let mut args = Vec::with_capacity(self.step.inputs);
        let mut values = Vec::new();
        let ys = xs
            .iter()
            .map(|&x| {
                args.clear();
                args.extend_from_slice(params);
                args.extend_from_slice(&state);
                args.push(x);
                self.step.eval_into(&args, &mut values);
                for (k, s) in state.iter_mut().enumerate() {
                    *s = values[self.step.outputs[k] as usize];
                }
                values[self.step.outputs[self.states] as usize]
            })
            .collect();
        (ys, state)
    }

    /// The mean squared error between the outputs and `targets`, and its gradient with respect to
    /// the parameters and the initial state (interpreted; the same computation as
    /// [`loss_grad_hlo`](Self::loss_grad_hlo)).
    pub fn loss_grad(&self, params: &[f32], xs: &[f32], targets: &[f32], s0: &[f32]) -> LossGrad {
        self.check(params, s0);
        assert_eq!(xs.len(), targets.len(), "Scan::loss_grad: one target per input");
        let (p, s, n) = (self.params, self.states, xs.len());

        // forward, saving the state each step starts from
        let mut saved = Vec::with_capacity(n * s);
        let mut state = s0.to_vec();
        let mut loss = 0.0f32;
        let mut args = Vec::new();
        for (&x, &t) in xs.iter().zip(targets) {
            saved.extend_from_slice(&state);
            args.clear();
            args.extend_from_slice(params);
            args.extend_from_slice(&state);
            args.extend_from_slice(&[x, t]);
            let out = self.step_loss.eval(&args);
            state.copy_from_slice(&out[..s]);
            loss += out[s + 1];
        }

        // backward
        let dl = 1.0 / n as f32;
        let mut d_params = vec![0.0f32; p];
        let mut d_state = vec![0.0f32; s];
        for i in (0..n).rev() {
            args.clear();
            args.extend_from_slice(params);
            args.extend_from_slice(&saved[i * s..(i + 1) * s]);
            args.extend_from_slice(&[xs[i], targets[i]]);
            args.extend_from_slice(&d_state);
            args.push(dl);
            let out = self.step_vjp.eval(&args);
            for (d, g) in d_params.iter_mut().zip(&out[..p]) {
                *d += g;
            }
            d_state.copy_from_slice(&out[p..]);
        }
        LossGrad { loss: loss * dl, params: d_params, state: d_state }
    }

    fn check(&self, params: &[f32], s0: &[f32]) {
        assert_eq!(params.len(), self.params, "Scan: wrong number of parameters");
        assert_eq!(s0.len(), self.states, "Scan: wrong number of state values");
    }
}
