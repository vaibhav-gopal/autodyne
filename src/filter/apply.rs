//! Filtering n-d data along an axis with designed coefficients, as in `scipy.signal`: direct form
//! II transposed (`lfilter`), cascaded second-order sections (`sosfilt`), zero-phase forward-backward
//! filtering (`filtfilt`, `sosfiltfilt`), and steady-state initial conditions (`lfilter_zi`,
//! `sosfilt_zi`). Inputs are views with any strides; computation is in f64.
//!
//! Lanes are filtered [`LANES`] at a time, interleaved sample by sample, so the independent
//! recursions of different lanes (and of a cascade's sections) overlap instead of each waiting on
//! its own previous output.

use crate::signal::{NdArray, NdView};
use crate::systems::SystemError;
use crate::units::*;

/// Lanes filtered together.
const LANES: usize = 4;

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

/// Direct form II transposed over interleaved lanes (`x[i * LANES + k]`), updating the states
/// `z[j][k]` (order entries per lane) in place. Orders up to 8 run a fixed-size kernel the compiler
/// unrolls and vectorizes.
fn df2t(b: &[f64], a: &[f64], x: &[f64], z: &mut [[f64; LANES]], y: &mut [f64]) {
    macro_rules! fixed {
        ($($n:literal),+) => {
            match z.len() {
                $($n => return df2t_fixed::<$n>(b, a, x, z, y),)+
                _ => {}
            }
        };
    }
    fixed!(1, 2, 3, 4, 5, 6, 7, 8);
    df2t_dynamic(b, a, x, z, y)
}

fn df2t_fixed<const N: usize>(b: &[f64], a: &[f64], x: &[f64], zs: &mut [[f64; LANES]], y: &mut [f64]) {
    let bf: [f64; N] = std::array::from_fn(|j| b[j + 1]);
    let af: [f64; N] = std::array::from_fn(|j| a[j + 1]);
    let b0 = b[0];
    let mut z: [[f64; LANES]; N] = std::array::from_fn(|j| zs[j]);
    for (xi, yi) in x.as_chunks::<LANES>().0.iter().zip(y.as_chunks_mut::<LANES>().0.iter_mut()) {
        let xi: [f64; LANES] = std::array::from_fn(|k| xi[k]);
        let out: [f64; LANES] = std::array::from_fn(|k| b0 * xi[k] + z[0][k]);
        for j in 0..N {
            let next = if j + 1 < N { z[j + 1] } else { [0.0; LANES] };
            z[j] = std::array::from_fn(|k| bf[j] * xi[k] - af[j] * out[k] + next[k]);
        }
        yi.copy_from_slice(&out);
    }
    zs.copy_from_slice(&z);
}

fn df2t_dynamic(b: &[f64], a: &[f64], x: &[f64], z: &mut [[f64; LANES]], y: &mut [f64]) {
    let order = z.len();
    for (xi, yi) in x.as_chunks::<LANES>().0.iter().zip(y.as_chunks_mut::<LANES>().0.iter_mut()) {
        let mut out = [0.0; LANES];
        for k in 0..LANES {
            out[k] = b[0] * xi[k] + if order > 0 { z[0][k] } else { 0.0 };
        }
        for j in 0..order {
            let (bj, aj) = (b[j + 1], a[j + 1]);
            if j + 1 < order {
                let next = z[j + 1];
                for k in 0..LANES {
                    z[j][k] = bj * xi[k] - aj * out[k] + next[k];
                }
            } else {
                for k in 0..LANES {
                    z[j][k] = bj * xi[k] - aj * out[k];
                }
            }
        }
        yi.copy_from_slice(&out);
    }
}

/// Normalized section coefficients `(b0, b1, b2, a1, a2)`.
fn section(s: &[f64; 6]) -> [f64; 5] {
    [s[0] / s[3], s[1] / s[3], s[2] / s[3], s[4] / s[3], s[5] / s[3]]
}

/// Cascaded sections over interleaved lanes, in place, sample by sample (so the sections pipeline),
/// updating each section's states `z[s][state][lane]`.
fn cascade(sections: &[[f64; 5]], x: &mut [f64], z: &mut [[[f64; LANES]; 2]]) {
    for v in x.as_chunks_mut::<LANES>().0 {
        let mut cur = [0.0; LANES];
        cur.copy_from_slice(v);
        for (c, zs) in sections.iter().zip(z.iter_mut()) {
            for k in 0..LANES {
                let xi = cur[k];
                let y = c[0] * xi + zs[0][k];
                zs[0][k] = c[1] * xi - c[3] * y + zs[1][k];
                zs[1][k] = c[2] * xi - c[4] * y;
                cur[k] = y;
            }
        }
        v.copy_from_slice(&cur);
    }
}

fn check_axis(shape: &[usize], axis: usize) -> Result<(), SystemError> {
    if axis >= shape.len() {
        return Err(invalid(format!("axis {axis} is out of range for shape {shape:?}")));
    }
    Ok(())
}

fn nd(e: crate::signal::NdError) -> SystemError {
    invalid(e.to_string())
}

/// The lanes of `x` along `axis` as f64 vectors.
fn lanes_f64<T: Float>(x: &NdView<'_, T>, axis: usize) -> Result<Vec<Vec<f64>>, SystemError> {
    check_axis(x.shape(), axis)?;
    let f = |v: &T| v.to_f64().unwrap_or(f64::NAN);
    // contiguous lanes copy as slices; strided ones through the n-d iterator
    Ok(x.lanes(axis).map_err(nd)?.map(|l| match l.as_slice() { Some(s) => s.iter().map(f).collect(), None => l.iter().map(f).collect() }).collect())
}

/// Interleaves up to [`LANES`] equal-length lanes (missing ones are zero).
fn interleave(group: &[&[f64]], n: usize) -> Vec<f64> {
    let mut out = vec![0.0; n * LANES];
    for (k, lane) in group.iter().enumerate() {
        for (i, &v) in lane.iter().enumerate().take(n) {
            out[i * LANES + k] = v;
        }
    }
    out
}

/// Runs `f(first lane index, lanes in the group, interleaved input) -> interleaved output of
/// `out_len` samples` over groups of lanes, and assembles an array shaped like `x` with `axis` of
/// length `out_len`.
fn map_groups<T: Float + Default>(
    x: NdView<'_, T>,
    axis: usize,
    out_len: usize,
    mut f: impl FnMut(usize, &[&[f64]]) -> Vec<f64>,
) -> Result<NdArray<T>, SystemError> {
    let lanes = lanes_f64(&x, axis)?;
    let mut shape = x.shape().to_vec();
    shape[axis] = out_len;
    let mut out = NdArray::<T>::zeros(&shape).map_err(nd)?;
    let mut dst: Vec<_> = out.lanes_mut(axis).map_err(nd)?.collect();
    for (g, chunk) in lanes.chunks(LANES).enumerate() {
        let group: Vec<&[f64]> = chunk.iter().map(Vec::as_slice).collect();
        let y = f(g * LANES, &group);
        for (k, d) in dst[g * LANES..g * LANES + chunk.len()].iter_mut().enumerate() {
            match d.as_mut_slice() {
                Some(s) => {
                    for (i, v) in s.iter_mut().enumerate() {
                        *v = T::_lit(y[i * LANES + k]);
                    }
                }
                None => {
                    for (i, v) in d.iter_mut().enumerate() {
                        *v = T::_lit(y[i * LANES + k]);
                    }
                }
            }
        }
    }
    Ok(out)
}

/// Filters `x` along `axis` with `b(z⁻¹) / a(z⁻¹)` from rest (`scipy.signal.lfilter`).
pub fn lfilter<T: Float + Default>(b: &[f64], a: &[f64], x: NdView<'_, T>, axis: usize) -> Result<NdArray<T>, SystemError> {
    let (b, a) = normalized(b, a)?;
    check_axis(x.shape(), axis)?;
    let n = x.shape()[axis];
    map_groups(x, axis, n, |_, group| {
        let xin = interleave(group, n);
        let mut y = vec![0.0; xin.len()];
        let mut z = vec![[0.0; LANES]; b.len() - 1];
        df2t(&b, &a, &xin, &mut z, &mut y);
        y
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
    let states = lanes_f64(&zi, axis)?;
    let mut finals = states.clone();
    let n = x.shape()[axis];
    let y = map_groups(x, axis, n, |first, group| {
        let mut z = vec![[0.0; LANES]; order];
        for k in 0..group.len() {
            for (j, zj) in z.iter_mut().enumerate() {
                zj[k] = states[first + k][j];
            }
        }
        let xin = interleave(group, n);
        let mut y = vec![0.0; xin.len()];
        df2t(&b, &a, &xin, &mut z, &mut y);
        for k in 0..group.len() {
            for (j, zj) in z.iter().enumerate() {
                finals[first + k][j] = zj[k];
            }
        }
        y
    })?;
    Ok((y, assemble_states(&expected, axis, &finals)?))
}

/// An array of `shape` whose lanes along `axis` are `lanes`.
fn assemble_states<T: Float + Default>(shape: &[usize], axis: usize, lanes: &[Vec<f64>]) -> Result<NdArray<T>, SystemError> {
    let mut out = NdArray::<T>::zeros(shape).map_err(nd)?;
    for (mut dst, src) in out.lanes_mut(axis).map_err(nd)?.zip(lanes) {
        for (d, &s) in dst.iter_mut().zip(src) {
            *d = T::_lit(s);
        }
    }
    Ok(out)
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

fn check_sos(sos: &[[f64; 6]]) -> Result<Vec<[f64; 5]>, SystemError> {
    if sos.is_empty() || sos.iter().any(|s| s[3] == 0.0) {
        return Err(invalid("sections need a0 != 0 (and there must be at least one)"));
    }
    Ok(sos.iter().map(section).collect())
}

/// Filters `x` along `axis` through second-order sections in series, from rest
/// (`scipy.signal.sosfilt`).
pub fn sosfilt<T: Float + Default>(sos: &[[f64; 6]], x: NdView<'_, T>, axis: usize) -> Result<NdArray<T>, SystemError> {
    let sections = check_sos(sos)?;
    check_axis(x.shape(), axis)?;
    let n = x.shape()[axis];
    map_groups(x, axis, n, |_, group| {
        let mut data = interleave(group, n);
        cascade(&sections, &mut data, &mut vec![[[0.0; LANES]; 2]; sections.len()]);
        data
    })
}

/// [`sosfilt`] from initial state `zi`, shape `[sections, ...x's shape with axis of length 2]`:
/// returns the output and the final state.
pub fn sosfilt_with_state<T: Float + Default>(sos: &[[f64; 6]], x: NdView<'_, T>, axis: usize, zi: NdView<'_, T>) -> Result<(NdArray<T>, NdArray<T>), SystemError> {
    let sections = check_sos(sos)?;
    check_axis(x.shape(), axis)?;
    let mut expected = vec![sos.len()];
    expected.extend_from_slice(x.shape());
    expected[axis + 1] = 2;
    if zi.shape() != expected.as_slice() {
        return Err(invalid(format!("zi should have shape {expected:?}, got {:?}", zi.shape())));
    }
    let per_section: usize = x.shape().iter().enumerate().filter(|&(i, _)| i != axis).map(|(_, &n)| n).product();
    // zi lanes along axis + 1, section-major
    let states = lanes_f64(&zi, axis + 1)?;
    let mut finals = states.clone();
    let n = x.shape()[axis];
    let y = map_groups(x, axis, n, |first, group| {
        let mut z = vec![[[0.0; LANES]; 2]; sections.len()];
        for (s, zs) in z.iter_mut().enumerate() {
            for k in 0..group.len() {
                let st = &states[s * per_section + first + k];
                zs[0][k] = st[0];
                zs[1][k] = st[1];
            }
        }
        let mut data = interleave(group, n);
        cascade(&sections, &mut data, &mut z);
        for (s, zs) in z.iter().enumerate() {
            for k in 0..group.len() {
                finals[s * per_section + first + k] = vec![zs[0][k], zs[1][k]];
            }
        }
        data
    })?;
    Ok((y, assemble_states(&expected, axis + 1, &finals)?))
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
    if n == 0 {
        return x.to_vec();
    }
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

/// Reverses interleaved lanes in time.
fn reverse_samples(x: &mut [f64]) {
    let n = x.len() / LANES;
    for i in 0..n / 2 {
        for k in 0..LANES {
            x.swap(i * LANES + k, (n - 1 - i) * LANES + k);
        }
    }
}

/// The group's lanes extended at both ends, interleaved.
fn extended_group(group: &[&[f64]], edge: usize, pad: Pad) -> (Vec<f64>, usize) {
    let ext: Vec<Vec<f64>> = group.iter().map(|lane| extend(lane, edge, pad)).collect();
    let len = ext[0].len();
    let refs: Vec<&[f64]> = ext.iter().map(Vec::as_slice).collect();
    (interleave(&refs, len), len)
}

/// Zero-phase filtering (`scipy.signal.filtfilt`, `method="pad"`): forward, then backward, each pass
/// starting from the steady state of the padded end, so the result has no phase shift and the
/// magnitude response squared.
pub fn filtfilt<T: Float + Default>(b: &[f64], a: &[f64], x: NdView<'_, T>, axis: usize, pad: Pad) -> Result<NdArray<T>, SystemError> {
    check_axis(x.shape(), axis)?;
    let (bn, an) = normalized(b, a)?;
    let n = x.shape()[axis];
    let edge = pad_len(pad, 3 * bn.len(), n)?;
    let zi = lfilter_zi(&bn, &an)?;
    map_groups(x, axis, n, |_, group| {
        let (ext, _) = extended_group(group, edge, pad);
        let start = |data: &[f64], at: usize| -> Vec<[f64; LANES]> { zi.iter().map(|&v| std::array::from_fn(|k| v * data[at * LANES + k])).collect() };
        let mut y = vec![0.0; ext.len()];
        let mut z = start(&ext, 0);
        df2t(&bn, &an, &ext, &mut z, &mut y);
        reverse_samples(&mut y);
        let back = y.clone();
        let mut z = start(&back, 0);
        df2t(&bn, &an, &back, &mut z, &mut y);
        reverse_samples(&mut y);
        y[edge * LANES..(edge + n) * LANES].to_vec()
    })
}

/// Zero-phase filtering through second-order sections (`scipy.signal.sosfiltfilt`).
pub fn sosfiltfilt<T: Float + Default>(sos: &[[f64; 6]], x: NdView<'_, T>, axis: usize, pad: Pad) -> Result<NdArray<T>, SystemError> {
    check_axis(x.shape(), axis)?;
    let sections = check_sos(sos)?;
    let zi = sosfilt_zi(sos)?;
    // SciPy's default padding: 3 (2 sections + 1), less the trivial second orders
    let trivial = sos.iter().filter(|s| s[2] == 0.0).count().min(sos.iter().filter(|s| s[5] == 0.0).count());
    let n = x.shape()[axis];
    let edge = pad_len(pad, 3 * (2 * sos.len() + 1 - trivial), n)?;
    map_groups(x, axis, n, |_, group| {
        let (mut data, _) = extended_group(group, edge, pad);
        let start = |data: &[f64]| -> Vec<[[f64; LANES]; 2]> {
            zi.iter().map(|z| [std::array::from_fn(|k| z[0] * data[k]), std::array::from_fn(|k| z[1] * data[k])]).collect()
        };
        let mut z = start(&data);
        cascade(&sections, &mut data, &mut z);
        reverse_samples(&mut data);
        let mut z = start(&data);
        cascade(&sections, &mut data, &mut z);
        reverse_samples(&mut data);
        data[edge * LANES..(edge + n) * LANES].to_vec()
    })
}
