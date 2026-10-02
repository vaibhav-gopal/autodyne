//! `scan`: a step traced once and run over a whole signal, forward and backward.

use super::ad::vjp;
use super::graph::{trace, FluxFloat, Graph, Tracer};
use super::loss::Loss;
use crate::signal::{ArrayMath, NdArray};

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
/// use autodyne::units::Elementwise;
///
/// // one scalar parameter (the cutoff), one scalar state value, scalar samples
/// let scan = Scan::trace(&[&[]], &[&[]], &[], |p, s, x| {
///     let (s, y) = OnePole::lowpass(p[0], Elementwise::lit(48_000.0)).tick(s[0], x);
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
    /// inputs: params, state, x, cotangents of state' and of y; outputs: cotangents of params,
    /// state and x
    pub(crate) step_vjp: Graph,
}

/// A loss and its gradient, from [`Scan::grad`] / [`Scan::loss_grad`].
#[derive(Clone, Debug, PartialEq)]
pub struct LossGrad<T = f32> {
    pub loss: T,
    /// d loss / d each parameter.
    pub params: Vec<NdArray<T>>,
    /// d loss / d each initial state value.
    pub state: Vec<NdArray<T>>,
    /// d loss / d the input signal (`[len, sample...]`).
    pub input: NdArray<T>,
}

/// Cotangents from [`Scan::vjp`].
#[derive(Clone, Debug, PartialEq)]
pub struct ScanVjp<T = f32> {
    pub params: Vec<NdArray<T>>,
    pub state: Vec<NdArray<T>>,
    pub input: NdArray<T>,
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
        shapes.extend(states.iter().copied());
        shapes.push(&output);
        let step_vjp = trace(&shapes, |v| {
            let out = step.call(&v[..p + s + 1]);
            vjp(&out, &v[p + s + 1..], &v[..p + s + 1])
        });
        Scan {
            params: params.iter().map(|x| x.to_vec()).collect(),
            states: states.iter().map(|x| x.to_vec()).collect(),
            sample: sample.to_vec(),
            output,
            step,
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
    pub fn run<T: FluxFloat>(&self, params: &[NdArray<T>], xs: &NdArray<T>, s0: &[NdArray<T>]) -> (NdArray<T>, Vec<NdArray<T>>) {
        let len = self.check(params, xs, s0);
        let s = self.states.len();
        let mut state = s0.to_vec();
        let mut ys = Vec::with_capacity(len * self.output.iter().product::<usize>());
        for x in steps(xs, &self.sample) {
            let args: Vec<NdArray<T>> = params.iter().chain(&state).cloned().chain([x]).collect();
            let mut out = self.step.eval(&args);
            ys.extend_from_slice(out[s].as_slice());
            out.truncate(s);
            state = out;
        }
        (NdArray::from_vec(ys, &[&[len], self.output.as_slice()].concat()).expect("output shape"), state)
    }

    /// The outputs, and the state each step starts from.
    fn forward<T: FluxFloat>(&self, params: &[NdArray<T>], inputs: &[NdArray<T>], s0: &[NdArray<T>]) -> (NdArray<T>, Vec<Vec<NdArray<T>>>) {
        let s = self.states.len();
        let mut saved = Vec::with_capacity(inputs.len());
        let mut state = s0.to_vec();
        let mut ys = Vec::with_capacity(inputs.len() * self.output.iter().product::<usize>());
        for x in inputs {
            let args: Vec<NdArray<T>> = params.iter().chain(&state).cloned().chain([x.clone()]).collect();
            let mut out = self.step.eval(&args);
            ys.extend_from_slice(out[s].as_slice());
            out.truncate(s);
            saved.push(std::mem::replace(&mut state, out));
        }
        (NdArray::from_vec(ys, &[&[inputs.len()], self.output.as_slice()].concat()).expect("output shape"), saved)
    }

    /// The reverse scan: from the cotangents of every output step, those of the parameters, the
    /// initial state and the input steps.
    fn backward<T: FluxFloat>(&self, params: &[NdArray<T>], inputs: &[NdArray<T>], saved: &[Vec<NdArray<T>>], dys: &NdArray<T>) -> ScanVjp<T> {
        let (p, s) = (self.params.len(), self.states.len());
        let dys: Vec<NdArray<T>> = steps(dys, &self.output).collect();
        let mut d_params: Vec<NdArray<T>> = self.params.iter().map(|sh| NdArray::zeros(sh).expect("shape")).collect();
        let mut d_state: Vec<NdArray<T>> = self.states.iter().map(|sh| NdArray::zeros(sh).expect("shape")).collect();
        let mut d_xs = vec![NdArray::zeros(&self.sample).expect("shape"); inputs.len()];
        for i in (0..inputs.len()).rev() {
            let args: Vec<NdArray<T>> = params.iter().chain(&saved[i]).cloned().chain([inputs[i].clone()]).chain(d_state).chain([dys[i].clone()]).collect();
            let mut out = self.step_vjp.eval(&args);
            d_xs[i] = out.pop().expect("d x");
            for (d, g) in d_params.iter_mut().zip(&out[..p]) {
                d.as_mut_slice().iter_mut().zip(g.as_slice()).for_each(|(d, &g)| *d = *d + g);
            }
            d_state = out.split_off(p);
            debug_assert_eq!(d_state.len(), s);
        }
        let input = d_xs.iter().flat_map(|d| d.as_slice().iter().copied()).collect();
        ScanVjp { params: d_params, state: d_state, input: NdArray::from_vec(input, &[&[inputs.len()], self.sample.as_slice()].concat()).expect("input shape") }
    }

    /// The vector-Jacobian product of the scan's outputs: given a cotangent for each output step
    /// (`dys`, `[len, output...]`), returns `Σ dys · ∂ys/∂w` for the parameters, the initial state
    /// and the input signal. This is how a scan's gradient composes with anything after it.
    pub fn vjp<T: FluxFloat>(&self, params: &[NdArray<T>], xs: &NdArray<T>, s0: &[NdArray<T>], dys: &NdArray<T>) -> ScanVjp<T> {
        let len = self.check(params, xs, s0);
        assert_eq!(dys.shape(), [&[len], self.output.as_slice()].concat(), "Scan::vjp: dys must be [len, output...]");
        let inputs: Vec<NdArray<T>> = steps(xs, &self.sample).collect();
        let (_, saved) = self.forward(params, &inputs, s0);
        self.backward(params, &inputs, &saved, dys)
    }

    /// `loss(outputs, aux...)` and its gradient with respect to the parameters, the initial state and
    /// the input signal (interpreted; the same computation as [`grad_program`](Self::grad_program)).
    ///
    /// ```
    /// use autodyne::filter::OnePole;
    /// use autodyne::flux::{scalar, vector, Loss, Scan, StftResolution};
    /// use autodyne::units::Elementwise;
    ///
    /// let scan = Scan::trace(&[&[]], &[&[]], &[], |p, s, x| {
    ///     let (s, y) = OnePole::lowpass(p[0], Elementwise::lit(48_000.0)).tick(s[0], x);
    ///     (vec![s], y)
    /// });
    /// let xs = vector(&(0..256).map(|i| ((i * 37 % 101) as f32 / 50.0) - 1.0).collect::<Vec<_>>());
    /// let (target, _) = scan.run(&[scalar(2_000.0)], &xs, &[scalar(0.0)]);
    /// let loss = Loss::stft(&[256], &[StftResolution::overlapping(64), StftResolution::overlapping(32)]);
    /// let g = scan.grad(&[scalar(500.0)], &xs, &[scalar(0.0)], &loss, &[target]);
    /// assert!(g.loss > 0.0 && g.params[0].as_slice()[0] < 0.0); // raising the cutoff helps
    /// assert_eq!(g.input.shape(), [256]);
    /// ```
    pub fn grad<T: FluxFloat>(&self, params: &[NdArray<T>], xs: &NdArray<T>, s0: &[NdArray<T>], loss: &Loss, aux: &[NdArray<T>]) -> LossGrad<T> {
        let len = self.check(params, xs, s0);
        assert_eq!(loss.output_shape(), [&[len], self.output.as_slice()].concat(), "Scan::grad: the loss scores outputs of another shape");
        let inputs: Vec<NdArray<T>> = steps(xs, &self.sample).collect();
        let (ys, saved) = self.forward(params, &inputs, s0);
        let (value, dys) = loss.grad(&ys, aux);
        let ScanVjp { params, state, input } = self.backward(params, &inputs, &saved, &dys);
        LossGrad { loss: value, params, state, input }
    }

    /// The mean squared error between the outputs and `targets` (`[len, output...]`), and its
    /// gradient: [`grad`](Self::grad) with [`Loss::mse`].
    pub fn loss_grad<T: FluxFloat>(&self, params: &[NdArray<T>], xs: &NdArray<T>, targets: &NdArray<T>, s0: &[NdArray<T>]) -> LossGrad<T> {
        self.grad(params, xs, s0, &Loss::mse(targets.shape()), std::slice::from_ref(targets))
    }
    /// Checks the argument shapes; returns the number of steps.
    fn check<T: FluxFloat>(&self, params: &[NdArray<T>], xs: &NdArray<T>, s0: &[NdArray<T>]) -> usize {
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
fn steps<'a, T: FluxFloat>(stacked: &'a NdArray<T>, shape: &'a [usize]) -> impl Iterator<Item = NdArray<T>> + 'a {
    let size = shape.iter().product::<usize>();
    stacked.as_slice().chunks(size.max(1)).take(stacked.shape()[0]).map(move |c| NdArray::from_vec(c[..size].to_vec(), shape).expect("step shape"))
}

