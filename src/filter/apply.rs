//! Filtering n-d data along an axis with designed coefficients, as in `scipy.signal`: direct form
//! II transposed (`lfilter`), cascaded second-order sections (`sosfilt`), zero-phase forward-backward
//! filtering (`filtfilt`, `sosfiltfilt`), and steady-state initial conditions (`lfilter_zi`,
//! `sosfilt_zi`). Inputs are views with any strides; computation is in f64.

use crate::signal::{NdArray, NdView};
use crate::systems::SystemError;
use crate::units::*;

fn invalid(m: impl Into<String>) -> SystemError {
    SystemError::Invalid(m.into())
}

/// `b` and `a` divided by `a[0]` and padded to the same length.
fn normalized(b: &[f64], a: &[f64]) -> Result<(Vec<f64>, Vec<f64>), SystemError> {
    let a0 = *a.first().ok_or_else(|| invalid("the denominator is empty"))?;
    if a0 == 0.0 {
        return Err(invalid("a[0] must not be zero"));
    }
    if b.is_empty() {
        return Err(invalid("the numerator is empty"));
    }
    let n = a.len().max(b.len());
    let pad = |c: &[f64]| c.iter().map(|x| x / a0).chain(std::iter::repeat_n(0.0, n - c.len())).collect::<Vec<f64>>();
    Ok((pad(b), pad(a)))
}

/// Direct form II transposed on one lane, updating the state `z` (length n - 1) in place.
fn df2t(b: &[f64], a: &[f64], x: &[f64], z: &mut [f64], y: &mut [f64]) {
    let order = z.len();
    for (xi, yi) in x.iter().zip(y.iter_mut()) {
        let out = b[0] * xi + if order > 0 { z[0] } else { 0.0 };
        for k in 0..order {
            let next = if k + 1 < order { z[k + 1] } else { 0.0 };
            z[k] = b[k + 1] * xi - a[k + 1] * out + next;
        }
        *yi = out;
    }
}

/// One second-order section on one lane, updating its two states.
fn biquad(s: &[f64; 6], x: &mut [f64], z: &mut [f64; 2]) {
    let (b0, b1, b2, a1, a2) = (s[0] / s[3], s[1] / s[3], s[2] / s[3], s[4] / s[3], s[5] / s[3]);
    for v in x.iter_mut() {
        let xi = *v;
        let y = b0 * xi + z[0];
        z[0] = b1 * xi - a1 * y + z[1];
        z[1] = b2 * xi - a2 * y;
        *v = y;
    }
}

fn check_axis(shape: &[usize], axis: usize) -> Result<(), SystemError> {
    if axis >= shape.len() {
        return Err(invalid(format!("axis {axis} is out of range for shape {shape:?}")));
    }
    Ok(())
}

/// Visits every lane of `x` along `axis` (as f64), handing out the lane's index and an output
/// buffer of the same length; collects the outputs into an array of `x`'s shape.
fn map_lanes<T: Float + Default>(x: NdView<'_, T>, axis: usize, mut f: impl FnMut(usize, &[f64], &mut [f64])) -> Result<NdArray<T>, SystemError> {
    check_axis(x.shape(), axis)?;
    let mut out = NdArray::<T>::zeros(x.shape()).map_err(|e| invalid(e.to_string()))?;
    let n = x.shape()[axis];
    let (mut input, mut output) = (vec![0.0; n], vec![0.0; n]);
    let lanes = x.lanes(axis).map_err(|e| invalid(e.to_string()))?;
    for (i, (lane, mut dst)) in lanes.zip(out.lanes_mut(axis).map_err(|e| invalid(e.to_string()))?).enumerate() {
        for (d, s) in input.iter_mut().zip(lane.iter()) {
            *d = s.to_f64().unwrap_or(f64::NAN);
        }
        f(i, &input, &mut output);
        for (d, &s) in dst.iter_mut().zip(&output) {
            *d = T::_lit(s);
        }
    }
    Ok(out)
}

/// Filters `x` along `axis` with `b(z⁻¹) / a(z⁻¹)` from rest (`scipy.signal.lfilter`).
pub fn lfilter<T: Float + Default>(b: &[f64], a: &[f64], x: NdView<'_, T>, axis: usize) -> Result<NdArray<T>, SystemError> {
    let (b, a) = normalized(b, a)?;
    let mut z = vec![0.0; b.len() - 1];
    map_lanes(x, axis, |_, input, output| {
        z.iter_mut().for_each(|v| *v = 0.0);
        df2t(&b, &a, input, &mut z, output);
    })
}

/// [`lfilter`] from initial state `zi` (the shape of `x` with `axis` of length
/// `max(len(a), len(b)) - 1`): returns the output and the final state, the same shape as `zi`.
pub fn lfilter_with_state<T: Float + Default>(b: &[f64], a: &[f64], x: NdView<'_, T>, axis: usize, zi: NdView<'_, T>) -> Result<(NdArray<T>, NdArray<T>), SystemError> {
    let (b, a) = normalized(b, a)?;
    let order = b.len() - 1;
    let mut expected = x.shape().to_vec();
    check_axis(&expected, axis)?;
    expected[axis] = order;
    if zi.shape() != expected.as_slice() {
        return Err(invalid(format!("zi should have shape {expected:?}, got {:?}", zi.shape())));
    }
    let states: Vec<Vec<f64>> = zi.lanes(axis).map_err(|e| invalid(e.to_string()))?.map(|l| l.iter().map(|v| v.to_f64().unwrap_or(f64::NAN)).collect()).collect();
    let mut finals = states.clone();
    let y = map_lanes(x, axis, |i, input, output| df2t(&b, &a, input, &mut finals[i], output))?;
    let mut zf = NdArray::<T>::zeros(&expected).map_err(|e| invalid(e.to_string()))?;
    for (mut dst, src) in zf.lanes_mut(axis).map_err(|e| invalid(e.to_string()))?.zip(&finals) {
        for (d, &s) in dst.iter_mut().zip(src) {
            *d = T::_lit(s);
        }
    }
    Ok((y, zf))
}

/// The initial state of [`lfilter_with_state`] for the steady state of a unit step: scale it by
/// the first input value to start without a transient (`scipy.signal.lfilter_zi`).
pub fn lfilter_zi(b: &[f64], a: &[f64]) -> Result<Vec<f64>, SystemError> {
    let (b, a) = normalized(b, a)?;
    let n = b.len() - 1;
    if n == 0 {
        return Ok(Vec::new());
    }
    // (I - companion(a)ᵀ) zi = b[1:] - a[1:] b[0]
    let m = NdArray::from_fn(&[n, n], |i| {
        let (r, c) = (i[0], i[1]);
        let companion_t = if c == 0 { -a[r + 1] } else if r + 1 == c { 1.0 } else { 0.0 };
        (if r == c { 1.0 } else { 0.0 }) - companion_t
    })
    .expect("n x n");
    let rhs = NdArray::from_vec((0..n).map(|i| b[i + 1] - a[i + 1] * b[0]).collect(), &[n]).expect("n");
    Ok(crate::linalg::solve(m.view(), rhs.view())?.into_vec())
}

/// Filters `x` along `axis` through second-order sections in series, from rest
/// (`scipy.signal.sosfilt`).
pub fn sosfilt<T: Float + Default>(sos: &[[f64; 6]], x: NdView<'_, T>, axis: usize) -> Result<NdArray<T>, SystemError> {
    check_sos(sos)?;
    map_lanes(x, axis, |_, input, output| {
        output.copy_from_slice(input);
        for s in sos {
            biquad(s, output, &mut [0.0; 2]);
        }
    })
}

fn check_sos(sos: &[[f64; 6]]) -> Result<(), SystemError> {
    if sos.is_empty() || sos.iter().any(|s| s[3] == 0.0) {
        return Err(invalid("sections need a0 != 0 (and there must be at least one)"));
    }
    Ok(())
}

/// [`sosfilt`] from initial state `zi`, shape `[sections, ...x's shape with axis of length 2]`:
/// returns the output and the final state.
pub fn sosfilt_with_state<T: Float + Default>(sos: &[[f64; 6]], x: NdView<'_, T>, axis: usize, zi: NdView<'_, T>) -> Result<(NdArray<T>, NdArray<T>), SystemError> {
    check_sos(sos)?;
    check_axis(x.shape(), axis)?;
    let mut expected = vec![sos.len()];
    expected.extend_from_slice(x.shape());
    expected[axis + 1] = 2;
    if zi.shape() != expected.as_slice() {
        return Err(invalid(format!("zi should have shape {expected:?}, got {:?}", zi.shape())));
    }
    let lanes_per_section: usize = x.shape().iter().enumerate().filter(|&(i, _)| i != axis).map(|(_, &n)| n).product();
    let mut state: Vec<[f64; 2]> = zi
        .lanes(axis + 1)
        .map_err(|e| invalid(e.to_string()))?
        .map(|l| {
            let v: Vec<f64> = l.iter().map(|v| v.to_f64().unwrap_or(f64::NAN)).collect();
            [v[0], v[1]]
        })
        .collect();
    let y = map_lanes(x, axis, |i, input, output| {
        output.copy_from_slice(input);
        for (si, s) in sos.iter().enumerate() {
            biquad(s, output, &mut state[si * lanes_per_section + i]);
        }
    })?;
    let mut zf = NdArray::<T>::zeros(&expected).map_err(|e| invalid(e.to_string()))?;
    for (mut dst, src) in zf.lanes_mut(axis + 1).map_err(|e| invalid(e.to_string()))?.zip(&state) {
        for (d, &s) in dst.iter_mut().zip(src) {
            *d = T::_lit(s);
        }
    }
    Ok((y, zf))
}

/// The steady-state step initial conditions of each section, scaled by the gain of the sections
/// before it (`scipy.signal.sosfilt_zi`).
pub fn sosfilt_zi(sos: &[[f64; 6]]) -> Result<Vec<[f64; 2]>, SystemError> {
    check_sos(sos)?;
    let mut scale = 1.0;
    let mut out = Vec::with_capacity(sos.len());
    for s in sos {
        let zi = lfilter_zi(&s[..3], &s[3..])?;
        out.push([scale * zi.first().copied().unwrap_or(0.0), scale * zi.get(1).copied().unwrap_or(0.0)]);
        scale *= s[..3].iter().sum::<f64>() / s[3..].iter().sum::<f64>();
    }
    Ok(out)
}

/// How [`filtfilt`] extends the signal at both ends before filtering (`padtype`), and by how many
/// samples (`None`: SciPy's default, three times the filter length).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Pad {
    /// Point-symmetric about the end value (SciPy's default).
    Odd(Option<usize>),
    /// Mirrored.
    Even(Option<usize>),
    /// The end value repeated.
    Constant(Option<usize>),
    None,
}

/// `x` extended by `n` samples at both ends.
fn extend(x: &[f64], n: usize, pad: Pad) -> Vec<f64> {
    let len = x.len();
    let (first, last) = (x[0], x[len - 1]);
    let left = (1..=n).rev().map(|i| match pad {
        Pad::Odd(_) => 2.0 * first - x[i],
        Pad::Even(_) => x[i],
        _ => first,
    });
    let right = (1..=n).map(|i| match pad {
        Pad::Odd(_) => 2.0 * last - x[len - 1 - i],
        Pad::Even(_) => x[len - 1 - i],
        _ => last,
    });
    left.chain(x.iter().copied()).chain(right).collect()
}

fn pad_len(pad: Pad, default: usize, len: usize) -> Result<usize, SystemError> {
    let n = match pad {
        Pad::Odd(n) | Pad::Even(n) | Pad::Constant(n) => n.unwrap_or(default),
        Pad::None => 0,
    };
    if n > 0 && n >= len {
        return Err(invalid(format!("the signal (length {len}) must be longer than the padding ({n})")));
    }
    Ok(n)
}

/// Zero-phase filtering (`scipy.signal.filtfilt`, `method="pad"`): forward, then backward, each pass
/// starting from the steady state of the padded end, so the result has no phase shift and the
/// magnitude response squared.
pub fn filtfilt<T: Float + Default>(b: &[f64], a: &[f64], x: NdView<'_, T>, axis: usize, pad: Pad) -> Result<NdArray<T>, SystemError> {
    check_axis(x.shape(), axis)?;
    let (bn, an) = normalized(b, a)?;
    let edge = pad_len(pad, 3 * bn.len(), x.shape()[axis])?;
    let zi = lfilter_zi(&bn, &an)?;
    let mut z = vec![0.0; zi.len()];
    map_lanes(x, axis, |_, input, output| {
        let ext = if edge > 0 { extend(input, edge, pad) } else { input.to_vec() };
        let mut y = vec![0.0; ext.len()];
        z.iter_mut().zip(&zi).for_each(|(s, &v)| *s = v * ext[0]);
        df2t(&bn, &an, &ext, &mut z, &mut y);
        y.reverse();
        let back = y.clone();
        z.iter_mut().zip(&zi).for_each(|(s, &v)| *s = v * back[0]);
        df2t(&bn, &an, &back, &mut z, &mut y);
        y.reverse();
        output.copy_from_slice(&y[edge..edge + input.len()]);
    })
}

/// Zero-phase filtering through second-order sections (`scipy.signal.sosfiltfilt`).
pub fn sosfiltfilt<T: Float + Default>(sos: &[[f64; 6]], x: NdView<'_, T>, axis: usize, pad: Pad) -> Result<NdArray<T>, SystemError> {
    check_axis(x.shape(), axis)?;
    let zi = sosfilt_zi(sos)?;
    // SciPy's default padding: 3 (2 sections + 1), less the trivial second orders
    let trivial = sos.iter().filter(|s| s[2] == 0.0).count().min(sos.iter().filter(|s| s[5] == 0.0).count());
    let edge = pad_len(pad, 3 * (2 * sos.len() + 1 - trivial), x.shape()[axis])?;
    map_lanes(x, axis, |_, input, output| {
        let mut ext = if edge > 0 { extend(input, edge, pad) } else { input.to_vec() };
        let x0 = ext[0];
        for (s, z) in sos.iter().zip(&zi) {
            biquad(s, &mut ext, &mut [z[0] * x0, z[1] * x0]);
        }
        ext.reverse();
        let y0 = ext[0];
        for (s, z) in sos.iter().zip(&zi) {
            biquad(s, &mut ext, &mut [z[0] * y0, z[1] * y0]);
        }
        ext.reverse();
        output.copy_from_slice(&ext[edge..edge + input.len()]);
    })
}
