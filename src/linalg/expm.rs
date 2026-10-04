//! The matrix exponential.

use super::{matmul, solve, LinalgError, LinalgFloat};
use crate::signal::{NdArray, NdView};

/// `e^a` for a square matrix, by scaling and squaring with Padé approximants (Higham 2005, the
/// method behind `scipy.linalg.expm`): the degree (3 to 13) is picked from the 1-norm, larger norms
/// are scaled down by a power of two and squared back.
pub fn expm<T: LinalgFloat>(a: NdView<'_, T>) -> Result<NdArray<T>, LinalgError> {
    let shape = a.shape().to_vec();
    if shape.len() != 2 || shape[0] != shape[1] {
        return Err(LinalgError::NotSquare(shape));
    }
    let n = shape[0];
    let a = a.to_owned();
    let norm = one_norm(&a);
    // the largest 1-norm each Padé degree handles to double precision without scaling
    const THETA: [(usize, f64); 4] = [(3, 1.495585217958292e-2), (5, 2.539_398_330_063_23e-1), (7, 9.504178996162932e-1), (9, 2.097847961257068)];
    for (degree, theta) in THETA {
        if norm <= theta {
            return pade(&a, degree, n);
        }
    }
    let theta13 = 5.371920351148152;
    let s = if norm > theta13 { (norm / theta13).log2().ceil() as i32 } else { 0 };
    let scaled = a.map(|&x| x * T::_lit(2f64.powi(-s)));
    let mut r = pade(&scaled, 13, n)?;
    for _ in 0..s {
        r = matmul(r.view(), r.view())?;
    }
    Ok(r)
}

fn one_norm<T: LinalgFloat>(a: &NdArray<T>) -> f64 {
    let n = a.shape()[1];
    (0..n).map(|j| a.as_slice().iter().skip(j).step_by(n).map(|x| x.abs().to_f64().unwrap_or(f64::NAN)).sum::<f64>()).fold(0.0, f64::max)
}

fn identity<T: LinalgFloat>(n: usize) -> NdArray<T> {
    NdArray::from_fn(&[n, n], |i| if i[0] == i[1] { T::_ONE } else { T::_ZERO }).expect("n x n")
}

/// `Σ c_k m_k` over matrices of one shape.
fn combine<T: LinalgFloat>(terms: &[(f64, &NdArray<T>)]) -> NdArray<T> {
    let mut out = NdArray::<T>::zeros(terms[0].1.shape()).expect("same shape");
    for (c, m) in terms {
        let c = T::_lit(*c);
        for (o, &x) in out.as_mut_slice().iter_mut().zip(m.as_slice()) {
            *o += c * x;
        }
    }
    out
}

/// The [degree/degree] Padé approximant of `e^a`: `(V - U)⁻¹ (V + U)`, where U holds the odd terms
/// and V the even ones.
fn pade<T: LinalgFloat>(a: &NdArray<T>, degree: usize, n: usize) -> Result<NdArray<T>, LinalgError> {
    let b: &[f64] = match degree {
        3 => &[120.0, 60.0, 12.0, 1.0],
        5 => &[30240.0, 15120.0, 3360.0, 420.0, 30.0, 1.0],
        7 => &[17297280.0, 8648640.0, 1995840.0, 277200.0, 25200.0, 1512.0, 56.0, 1.0],
        9 => &[17643225600.0, 8821612800.0, 2075673600.0, 302702400.0, 30270240.0, 2162160.0, 110880.0, 3960.0, 90.0, 1.0],
        _ => &[
            64764752532480000.0,
            32382376266240000.0,
            7771770303897600.0,
            1187353796428800.0,
            129060195264000.0,
            10559470521600.0,
            670442572800.0,
            33522128640.0,
            1323241920.0,
            40840800.0,
            960960.0,
            16380.0,
            182.0,
            1.0,
        ],
    };
    let eye = identity::<T>(n);
    let a2 = matmul(a.view(), a.view())?;
    let (u, v) = if degree < 13 {
        // powers A^0, A^2, A^4, ... up to A^(degree - 1)
        let mut evens = vec![eye.clone(), a2.clone()];
        while evens.len() < degree.div_ceil(2) {
            let next = matmul(evens.last().expect("non-empty").view(), a2.view())?;
            evens.push(next);
        }
        let odd: Vec<(f64, &NdArray<T>)> = evens.iter().enumerate().map(|(k, m)| (b[2 * k + 1], m)).collect();
        let even: Vec<(f64, &NdArray<T>)> = evens.iter().enumerate().map(|(k, m)| (b[2 * k], m)).collect();
        (matmul(a.view(), combine(&odd).view())?, combine(&even))
    } else {
        let a4 = matmul(a2.view(), a2.view())?;
        let a6 = matmul(a4.view(), a2.view())?;
        let inner_u = matmul(a6.view(), combine(&[(b[13], &a6), (b[11], &a4), (b[9], &a2)]).view())?;
        let u = matmul(a.view(), combine(&[(1.0, &inner_u), (b[7], &a6), (b[5], &a4), (b[3], &a2), (b[1], &eye)]).view())?;
        let inner_v = matmul(a6.view(), combine(&[(b[12], &a6), (b[10], &a4), (b[8], &a2)]).view())?;
        let v = combine(&[(1.0, &inner_v), (b[6], &a6), (b[4], &a4), (b[2], &a2), (b[0], &eye)]);
        (u, v)
    };
    let p = combine(&[(1.0, &v), (1.0, &u)]);
    let q = combine(&[(1.0, &v), (-1.0, &u)]);
    solve(q.view(), p.view())
}
