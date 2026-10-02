//! `scan`: a step traced once and run over a whole signal, forward and backward.

use super::ad::vjp;
use super::graph::{trace, Graph, Tracer};
use crate::signal::NdArray;
use crate::units::Real;

/// A recurrence `step(params, state, x) -> (state', y)`, traced once, run over a signal.
///
/// Parameters, state values, input steps and output steps are arrays of any fixed shape (scalars
/// for a per-sample filter; vectors for a multichannel one or a frame-by-frame process). The signal
/// stacks the steps along a new first axis: `xs` is `[len, sample...]`, `ys` is `[len, output...]`.
///
/// The step is traced into a [`Graph`] a single time; [`run`](Self::run) and
/// [`loss_grad`](Self::loss_grad) interpret it, and [`forward_program`](Self::forward_program) /
/// [`loss_grad_program`](Self::loss_grad_program) emit it as a `stablehlo.while` loop. The gradient
/// is a reverse scan: the forward loop saves each step's input state, the backward loop walks the
/// signal from the end, applying the step's vector-Jacobian product.
///
/// ```
/// use autodyne::filter::OnePole;
/// use autodyne::flux::{scalar, vector, Scan};
/// use autodyne::units::Real;
///
/// // one scalar parameter (the cutoff), one scalar state value, scalar samples
/// let scan = Scan::trace(&[&[]], &[&[]], &[], |p, s, x| {
///     let (s, y) = OnePole::lowpass(p[0], Real::lit(48_000.0)).tick(s[0], x);
///     (vec![s], y)
/// });
/// let (ys, _) = scan.run(&[scalar(1_000.0)], &vector(&[1.0; 64]), &[scalar(0.0)]);
///
/// let mut lp = OnePole::lowpass(1_000.0f32, 48_000.0);
/// let mut block = [1.0f32; 64];
/// lp.process(&mut block);
/// assert_eq!(ys.as_slice(), block);
/// ```
#[derive(Clone, Debug)]
pub struct Scan {
    pub(crate) params: Vec<Vec<usize>>,
    pub(crate) states: Vec<Vec<usize>>,
    pub(crate) sample: Vec<usize>,
    pub(crate) output: Vec<usize>,
    /// inputs: params, state, x; outputs: state', y
    pub(crate) step: Graph,
    /// inputs: params, state, x, target; outputs: state', y, Σ (y - target)²
    pub(crate) step_loss: Graph,
    /// inputs: params, state, x, target, cotangents of state' and of the loss term;
    /// outputs: cotangents of params and state
    pub(crate) step_vjp: Graph,
}

/// The loss and its gradient from [`Scan::loss_grad`].
#[derive(Clone, Debug, PartialEq)]
pub struct LossGrad {
    /// Mean squared error between the outputs and the targets (over every element).
    pub loss: f32,
    /// d loss / d each parameter.
    pub params: Vec<NdArray<f32>>,
    /// d loss / d each initial state value.
    pub state: Vec<NdArray<f32>>,
}

impl Scan {
    /// Traces `step` on parameters, state values and an input step of the given shapes. The
    /// closure receives the parameters, the state and the input step, and returns the next state
    /// (same shapes) and the output step.
    pub fn trace(params: &[&[usize]], states: &[&[usize]], sample: &[usize], step: impl FnOnce(&[Tracer], &[Tracer], Tracer) -> (Vec<Tracer>, Tracer)) -> Scan {
        let (p, s) = (params.len(), states.len());
        let shapes: Vec<&[usize]> = params.iter().chain(states).copied().chain([sample]).collect();
        let step = trace(&shapes, |v| {
            let (next, y) = step(&v[..p], &v[p..p + s], v[p + s]);
            assert_eq!(next.len(), s, "Scan::trace: the step returned the wrong number of state values");
            for (k, (n, sh)) in next.iter().zip(states).enumerate() {
                assert_eq!(&n.shape(), sh, "Scan::trace: state value {k} changed shape");
            }
            next.into_iter().chain([y]).collect()
        });
        let output = step.output_shapes()[s].clone();

        let mut shapes = shapes;
        shapes.push(&output);
        let step_loss = trace(&shapes, |v| {
            let out = step.call(&v[..p + s + 1]);
            let d = out[s] - v[p + s + 1];
            out.into_iter().chain([(d * d).sum_all()]).collect()
        });

        shapes.extend(states.iter().copied());
        shapes.push(&[]);
        let step_vjp = trace(&shapes, |v| {
            let out = step_loss.call(&v[..p + s + 2]);
            let outputs: Vec<Tracer> = out[..s].iter().copied().chain([out[s + 1]]).collect();
            vjp(&outputs, &v[p + s + 2..], &v[..p + s])
        });
        Scan {
            params: params.iter().map(|x| x.to_vec()).collect(),
            states: states.iter().map(|x| x.to_vec()).collect(),
            sample: sample.to_vec(),
            output,
            step,
            step_loss,
            step_vjp,
        }
    }

    pub fn param_shapes(&self) -> &[Vec<usize>] {
        &self.params
    }
    pub fn state_shapes(&self) -> &[Vec<usize>] {
        &self.states
    }
    /// The shape of one input step.
    pub fn sample_shape(&self) -> &[usize] {
        &self.sample
    }
    /// The shape of one output step.
    pub fn output_shape(&self) -> &[usize] {
        &self.output
    }
    /// The traced step: inputs params, state, x; outputs state', y.
    pub fn step(&self) -> &Graph {
        &self.step
    }

    /// Runs the scan over `xs` (`[len, sample...]`) from state `s0`: returns the outputs
    /// (`[len, output...]`) and the final state.
    pub fn run(&self, params: &[NdArray<f32>], xs: &NdArray<f32>, s0: &[NdArray<f32>]) -> (NdArray<f32>, Vec<NdArray<f32>>) {
        let len = self.check(params, xs, s0);
        let s = self.states.len();
        let mut state = s0.to_vec();
        let mut ys = Vec::with_capacity(len * self.output.iter().product::<usize>());
        for x in steps(xs, &self.sample) {
            let args: Vec<NdArray<f32>> = params.iter().chain(&state).cloned().chain([x]).collect();
            let mut out = self.step.eval(&args);
            ys.extend_from_slice(out[s].as_slice());
            out.truncate(s);
            state = out;
        }
        (NdArray::from_vec(ys, &[&[len], self.output.as_slice()].concat()).expect("output shape"), state)
    }

    /// The mean squared error between the outputs and `targets` (`[len, output...]`), and its
    /// gradient with respect to the parameters and the initial state (interpreted; the same
    /// computation as [`loss_grad_program`](Self::loss_grad_program)).
    pub fn loss_grad(&self, params: &[NdArray<f32>], xs: &NdArray<f32>, targets: &NdArray<f32>, s0: &[NdArray<f32>]) -> LossGrad {
        let len = self.check(params, xs, s0);
        assert_eq!(targets.shape(), [&[len], self.output.as_slice()].concat(), "Scan::loss_grad: targets must be [len, output...]");
        let (p, s) = (self.params.len(), self.states.len());
        let inputs: Vec<NdArray<f32>> = steps(xs, &self.sample).collect();
        let targets: Vec<NdArray<f32>> = steps(targets, &self.output).collect();

        // forward, saving the state each step starts from
        let mut saved = Vec::with_capacity(len);
        let mut state = s0.to_vec();
        let mut loss = 0.0f32;
        for (x, t) in inputs.iter().zip(&targets) {
            let args: Vec<NdArray<f32>> = params.iter().chain(&state).cloned().chain([x.clone(), t.clone()]).collect();
            let mut out = self.step_loss.eval(&args);
            loss += out[s + 1].as_slice()[0];
            out.truncate(s);
            saved.push(std::mem::replace(&mut state, out));
        }

        // backward
        let dl = 1.0 / (len * self.output.iter().product::<usize>()) as f32;
        let mut d_params: Vec<NdArray<f32>> = self.params.iter().map(|sh| NdArray::zeros(sh).expect("shape")).collect();
        let mut d_state: Vec<NdArray<f32>> = self.states.iter().map(|sh| NdArray::zeros(sh).expect("shape")).collect();
        for i in (0..len).rev() {
            let args: Vec<NdArray<f32>> =
                params.iter().chain(&saved[i]).cloned().chain([inputs[i].clone(), targets[i].clone()]).chain(d_state).chain([super::scalar(dl)]).collect();
            let mut out = self.step_vjp.eval(&args);
            for (d, g) in d_params.iter_mut().zip(&out[..p]) {
                d.as_mut_slice().iter_mut().zip(g.as_slice()).for_each(|(d, g)| *d += g);
            }
            d_state = out.split_off(p);
        }
        LossGrad { loss: loss * dl, params: d_params, state: d_state }
    }

    /// Checks the argument shapes; returns the number of steps.
    fn check(&self, params: &[NdArray<f32>], xs: &NdArray<f32>, s0: &[NdArray<f32>]) -> usize {
        assert_eq!(params.len(), self.params.len(), "Scan: wrong number of parameters");
        assert_eq!(s0.len(), self.states.len(), "Scan: wrong number of state values");
        for (k, (a, sh)) in params.iter().zip(&self.params).enumerate() {
            assert_eq!(a.shape(), sh.as_slice(), "Scan: parameter {k} has the wrong shape");
        }
        for (k, (a, sh)) in s0.iter().zip(&self.states).enumerate() {
            assert_eq!(a.shape(), sh.as_slice(), "Scan: state value {k} has the wrong shape");
        }
        assert!(xs.ndim() == self.sample.len() + 1 && &xs.shape()[1..] == self.sample.as_slice(), "Scan: xs must be [len, sample...]");
        xs.shape()[0]
    }
}

/// The steps of a stacked signal (`[len, shape...]`), each of `shape`.
fn steps<'a>(stacked: &'a NdArray<f32>, shape: &'a [usize]) -> impl Iterator<Item = NdArray<f32>> + 'a {
    let size = shape.iter().product::<usize>();
    stacked.as_slice().chunks(size.max(1)).take(stacked.shape()[0]).map(move |c| NdArray::from_vec(c[..size].to_vec(), shape).expect("step shape"))
}

impl Tracer {
    /// A zero of `shape` (a convenience for initial states and padding).
    pub fn zeros(shape: &[usize]) -> Tracer {
        Tracer::lit(0.0).broadcast_to(shape)
    }
}
