//! Conversions between transfer functions, zeros-poles-gain, state space and second-order sections
//! (the algorithms of `scipy.signal`, so results match it).

use super::{real, Domain, Sos, StateSpace, SystemError, C64};
use crate::linalg::{eigvals, poly, roots};
use crate::signal::NdArray;

/// Drops leading zeros of `den` (and of `num`, below 1e-14) and divides both by `den[0]`
/// (`scipy.signal.normalize`).
pub fn normalize(num: &[f64], den: &[f64]) -> Result<(Vec<f64>, Vec<f64>), SystemError> {
    let start = den.iter().position(|&d| d != 0.0).ok_or_else(|| SystemError::invalid("the denominator is zero"))?;
    let den = &den[start..];
    let lead = num.iter().position(|&b| b.abs() > 1e-14).unwrap_or(num.len().saturating_sub(1));
    let num = if num.is_empty() { &[0.0][..] } else { &num[lead..] };
    let d0 = den[0];
    Ok((num.iter().map(|b| b / d0).collect(), den.iter().map(|a| a / d0).collect()))
}

/// Zeros, poles and gain of `num / den` (`scipy.signal.tf2zpk`).
pub fn tf2zpk(num: &[f64], den: &[f64]) -> Result<(Vec<C64>, Vec<C64>, f64), SystemError> {
    let (b, a) = normalize(num, den)?;
    let k = b[0];
    let zeros = if k == 0.0 { Vec::new() } else { roots(&b.iter().map(|x| x / k).collect::<Vec<_>>())? };
    Ok((zeros, roots(&a)?, k))
}

/// Numerator and denominator of `k Π(x - z) / Π(x - p)` (`scipy.signal.zpk2tf`); complex roots must
/// come in conjugate pairs.
pub fn zpk2tf(zeros: &[C64], poles: &[C64], gain: f64) -> (Vec<f64>, Vec<f64>) {
    (poly(zeros).into_iter().map(|b| b * gain).collect(), poly(poles))
}

/// The controllable canonical state space of `num / den` (`scipy.signal.tf2ss`).
#[allow(clippy::type_complexity)]
pub fn tf2ss(num: &[f64], den: &[f64]) -> Result<(NdArray<f64>, NdArray<f64>, NdArray<f64>, NdArray<f64>), SystemError> {
    let (num, den) = normalize(num, den)?;
    let (m, k) = (num.len(), den.len());
    if m > k {
        return Err(SystemError::invalid("improper transfer function: the numerator is longer than the denominator"));
    }
    let mut padded = vec![0.0; k - m];
    padded.extend_from_slice(&num);
    let d = NdArray::from_vec(vec![padded[0]], &[1, 1]).expect("1 x 1");
    if k == 1 {
        let zeros = |r, c| NdArray::<f64>::zeros(&[r, c]).expect("shape");
        return Ok((zeros(1, 1), zeros(1, 1), zeros(1, 1), d));
    }
    let n = k - 1;
    let a = NdArray::from_fn(&[n, n], |i| if i[0] == 0 { -den[1 + i[1]] } else if i[0] == i[1] + 1 { 1.0 } else { 0.0 }).expect("n x n");
    let b = NdArray::from_fn(&[n, 1], |i| if i[0] == 0 { 1.0 } else { 0.0 }).expect("n x 1");
    let c = NdArray::from_fn(&[1, n], |i| padded[1 + i[1]] - padded[0] * den[1 + i[1]]).expect("1 x n");
    Ok((a, b, c, d))
}

/// The transfer function from `input` to each output of a state-space system: numerators (one
/// per output) and the shared denominator (`scipy.signal.ss2tf`).
pub fn ss2tf(sys: &StateSpace, input: usize) -> Result<(Vec<Vec<f64>>, Vec<f64>), SystemError> {
    let (n, inputs) = (sys.order(), sys.b.shape()[1]);
    let outputs = sys.c.shape()[0];
    if input >= inputs {
        return Err(SystemError::invalid(format!("the system has {inputs} inputs, not {}", input + 1)));
    }
    let char_poly = |m: &NdArray<f64>| -> Result<Vec<f64>, SystemError> { if n == 0 { Ok(vec![1.0]) } else { Ok(poly(&eigvals(m.view())?)) } };
    let den = char_poly(&sys.a)?;
    let mut nums = Vec::with_capacity(outputs);
    for k in 0..outputs {
        // det(sI - A + B C_k) - det(sI - A) + D det(sI - A), from the matrix determinant lemma
        let d = sys.d.as_slice()[k * inputs + input];
        let a_bc = NdArray::from_fn(&[n, n], |i| sys.a.as_slice()[i[0] * n + i[1]] - sys.b.as_slice()[i[0] * inputs + input] * sys.c.as_slice()[k * n + i[1]]).expect("n x n");
        let p = char_poly(&a_bc)?;
        nums.push(p.iter().zip(&den).map(|(x, y)| x + (d - 1.0) * y).collect());
    }
    Ok((nums, den))
}

/// The complex roots (one of each conjugate pair, positive imaginary part, sorted by real part)
/// and the real roots of a set closed under conjugation (`scipy.signal._cplxreal`).
pub fn cplxreal(z: &[C64]) -> Result<(Vec<C64>, Vec<f64>), SystemError> {
    let tol = 100.0 * f64::EPSILON;
    let mut z = z.to_vec();
    // by real part, then by |imaginary part|
    z.sort_by(|a, b| a.re.total_cmp(&b.re).then(a.im.abs().total_cmp(&b.im.abs())));
    let mut reals = Vec::new();
    let (mut pos, mut neg) = (Vec::new(), Vec::new());
    for x in z {
        if x.im.abs() <= tol * x.norm() {
            reals.push(x.re);
        } else if x.im > 0.0 {
            pos.push(x);
        } else {
            neg.push(x);
        }
    }
    if pos.len() != neg.len() {
        return Err(SystemError::invalid("a complex value has no matching conjugate"));
    }
    // within runs of (nearly) equal real parts, order by |imaginary part|
    let mut start = 0;
    while start < pos.len() {
        let mut stop = start + 1;
        while stop < pos.len() && pos[stop].re - pos[stop - 1].re <= tol * pos[stop - 1].norm() {
            stop += 1;
        }
        pos[start..stop].sort_by(|a, b| a.im.abs().total_cmp(&b.im.abs()));
        neg[start..stop].sort_by(|a, b| a.im.abs().total_cmp(&b.im.abs()));
        start = stop;
    }
    let mut pairs = Vec::with_capacity(pos.len());
    for (p, n) in pos.iter().zip(&neg) {
        if (*p - n.conj()).norm() > tol * n.norm() {
            return Err(SystemError::invalid("a complex value has no matching conjugate"));
        }
        // average out rounding between the pair
        pairs.push((*p + n.conj()) * 0.5);
    }
    Ok((pairs, reals))
}

/// How [`zpk2sos`] pairs poles with zeros (`scipy.signal.zpk2sos`).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Pairing {
    /// Each pole with its nearest zeros; an odd order gets an extra pole and zero at the origin.
    Nearest,
    /// As `Nearest`, but an odd order keeps one first-order section.
    KeepOdd,
    /// Fewest sections, no added poles or zeros (for analog systems).
    Minimal,
}

/// Second-order sections of a zeros-poles-gain system (`scipy.signal.zpk2sos`): the poles closest to
/// the stability boundary go last, each paired with its nearest zeros; the gain goes in the first
/// section. Complex roots must come in conjugate pairs.
pub fn zpk2sos(zeros: &[C64], poles: &[C64], gain: f64, pairing: Pairing, domain: Domain) -> Result<Sos, SystemError> {
    let analog = !domain.is_discrete();
    if zeros.is_empty() && poles.is_empty() {
        return Ok(vec![if analog { [0.0, 0.0, gain, 0.0, 0.0, 1.0] } else { [gain, 0.0, 0.0, 1.0, 0.0, 0.0] }]);
    }
    let (mut z, mut p) = (zeros.to_vec(), poles.to_vec());
    let n_sections;
    if pairing != Pairing::Minimal {
        // as many zeros as poles (extra ones at the origin)
        let (nz, np) = (z.len(), p.len());
        p.extend(std::iter::repeat_n(C64::zero(), nz.saturating_sub(np)));
        z.extend(std::iter::repeat_n(C64::zero(), np.saturating_sub(nz)));
        n_sections = p.len().max(z.len()).div_ceil(2);
        if p.len() % 2 == 1 && pairing == Pairing::Nearest {
            p.push(C64::zero());
            z.push(C64::zero());
        }
    } else {
        if p.len() < z.len() {
            return Err(SystemError::invalid("minimal pairing needs at least as many poles as zeros"));
        }
        n_sections = p.len().div_ceil(2);
    }
    // one of each conjugate pair, then the real roots
    let join = |(c, r): (Vec<C64>, Vec<f64>)| c.into_iter().chain(r.into_iter().map(real)).collect::<Vec<C64>>();
    let mut z = join(cplxreal(&z)?);
    let mut p = join(cplxreal(&p)?);
    let is_real = |x: &C64| x.im == 0.0;
    // the "worst" pole: closest to the unit circle (digital) or the imaginary axis (analog)
    let worst = |p: &[C64]| -> usize {
        let score = |x: &C64| if analog { x.re.abs() } else { (1.0 - x.norm()).abs() };
        (0..p.len()).fold(0, |best, i| if score(&p[i]) < score(&p[best]) { i } else { best })
    };
    #[derive(PartialEq)]
    enum Which {
        Real,
        Complex,
        Any,
    }
    let nearest = |from: &[C64], to: C64, which: Which| -> Option<usize> {
        let mut order: Vec<usize> = (0..from.len()).collect();
        order.sort_by(|&a, &b| (from[a] - to).norm().total_cmp(&(from[b] - to).norm()));
        order.into_iter().find(|&i| match which {
            Which::Any => true,
            Which::Real => is_real(&from[i]),
            Which::Complex => !is_real(&from[i]),
        })
    };
    let section = |zs: &[C64], ps: &[C64]| -> [f64; 6] {
        let (b, a) = (poly(zs), poly(ps));
        let mut s = [0.0; 6];
        s[3 - b.len()..3].copy_from_slice(&b);
        s[6 - a.len()..6].copy_from_slice(&a);
        s
    };
    let missing = || SystemError::invalid("could not pair the poles with the zeros");
    let mut sos = vec![[0.0; 6]; n_sections];
    for si in (0..n_sections).rev() {
        let p1 = p.remove(worst(&p));
        let p_reals = p.iter().filter(|x| is_real(x)).count();
        let z_reals = z.iter().filter(|x| is_real(x)).count();
        if is_real(&p1) && p_reals == 0 {
            // the last remaining real pole
            if pairing != Pairing::Minimal {
                let z1 = z.remove(nearest(&z, p1, Which::Real).ok_or_else(missing)?);
                sos[si] = section(&[z1, C64::zero()], &[p1, C64::zero()]);
            } else if !z.is_empty() {
                let z1 = z.remove(nearest(&z, p1, Which::Real).ok_or_else(missing)?);
                sos[si] = section(&[z1], &[p1]);
            } else {
                sos[si] = section(&[], &[p1]);
            }
        } else if p.len() + 1 == z.len() && !is_real(&p1) && p_reals == 1 && z_reals == 1 {
            // one real pole and one real zero left: this complex pole must take a complex zero
            let z1 = z.remove(nearest(&z, p1, Which::Complex).ok_or_else(missing)?);
            sos[si] = section(&[z1, z1.conj()], &[p1, p1.conj()]);
        } else {
            let p2 = if is_real(&p1) {
                let reals: Vec<usize> = (0..p.len()).filter(|&i| is_real(&p[i])).collect();
                let candidates: Vec<C64> = reals.iter().map(|&i| p[i]).collect();
                p.remove(reals[worst(&candidates)])
            } else {
                p1.conj()
            };
            if z.is_empty() {
                sos[si] = section(&[], &[p1, p2]);
            } else {
                let z1 = z.remove(nearest(&z, p1, Which::Any).ok_or_else(missing)?);
                if !is_real(&z1) {
                    sos[si] = section(&[z1, z1.conj()], &[p1, p2]);
                } else if let Some(i) = nearest(&z, p1, Which::Real) {
                    let z2 = z.remove(i);
                    sos[si] = section(&[z1, z2], &[p1, p2]);
                } else {
                    sos[si] = section(&[z1], &[p1, p2]);
                }
            }
        }
    }
    for b in &mut sos[0][..3] {
        *b *= gain;
    }
    Ok(sos)
}

/// Zeros, poles and gain of second-order sections (`scipy.signal.sos2zpk`): two of each per
/// section (a first-order section contributes a root at the origin).
pub fn sos2zpk(sos: &[[f64; 6]]) -> Result<(Vec<C64>, Vec<C64>, f64), SystemError> {
    let mut z = vec![C64::zero(); 2 * sos.len()];
    let mut p = vec![C64::zero(); 2 * sos.len()];
    let mut k = 1.0;
    for (i, s) in sos.iter().enumerate() {
        let (zs, ps, g) = tf2zpk(&s[..3], &s[3..])?;
        z[2 * i..2 * i + zs.len()].copy_from_slice(&zs);
        p[2 * i..2 * i + ps.len()].copy_from_slice(&ps);
        k *= g;
    }
    Ok((z, p, k))
}

/// The transfer function of second-order sections in series (`scipy.signal.sos2tf`).
pub fn sos2tf(sos: &[[f64; 6]]) -> (Vec<f64>, Vec<f64>) {
    let mut b = vec![1.0];
    let mut a = vec![1.0];
    for s in sos {
        b = crate::linalg::polymul(&b, &s[..3]);
        a = crate::linalg::polymul(&a, &s[3..]);
    }
    (b, a)
}

/// Second-order sections of a digital transfer function (`scipy.signal.tf2sos`).
pub fn tf2sos(num: &[f64], den: &[f64], pairing: Pairing) -> Result<Sos, SystemError> {
    let (z, p, k) = tf2zpk(num, den)?;
    zpk2sos(&z, &p, k, pairing, Domain::Discrete { dt: 1.0 })
}
