//! Small row-major matrix products by a register-blocked AVX2 + FMA kernel: a block of up to six
//! rows by 8 (`f64`) or 16 (`f32`) columns of the result in twelve vector accumulators, each step
//! broadcasting one value of each of `a`'s rows against one contiguous run of a row of `b`.
//! Row-major operands need no packing (the run of `b` is contiguous), which is what makes this
//! faster than a general kernel for matrices that fit in cache, where packing and dispatch are a
//! measurable share of the work.

/// Products up to this many multiply-adds take the small kernel (beyond, packing pays off).
const LIMIT: usize = 96 * 96 * 96;

/// `c = a · b` for row-major contiguous `a` (m x k), `b` (k x n) and `c` (m x n), if the shapes
/// suit the kernel and the CPU has AVX2 and FMA; `false` (nothing written) otherwise.
pub(crate) fn matmul<T: 'static>(a: *const T, b: *const T, c: *mut T, m: usize, k: usize, n: usize) -> bool {
    #[cfg(target_arch = "x86_64")]
    {
        use std::any::TypeId;
        if m == 0 || n == 0 || k == 0 || m * n * k > LIMIT || !crate::simd::avx2_fma_available() {
            return false;
        }
        if TypeId::of::<T>() == TypeId::of::<f64>() && n.is_multiple_of(8) {
            // SAFETY: AVX2 and FMA were checked; the caller's buffers hold the matrices
            unsafe { rows_f64(a.cast(), b.cast(), c.cast(), m, k, n) };
            return true;
        }
        if TypeId::of::<T>() == TypeId::of::<f32>() && n.is_multiple_of(16) {
            // SAFETY: as above
            unsafe { rows_f32(a.cast(), b.cast(), c.cast(), m, k, n) };
            return true;
        }
    }
    let _ = (a, b, c, m, k, n);
    false
}

/// Blocks of six rows, then the remaining one to five.
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2,fma")]
unsafe fn rows_f64(a: *const f64, b: *const f64, c: *mut f64, m: usize, k: usize, n: usize) {
    let mut i = 0;
    while i + 6 <= m {
        block_f64::<6>(a.add(i * k), b, c.add(i * n), k, n);
        i += 6;
    }
    match m - i {
        5 => block_f64::<5>(a.add(i * k), b, c.add(i * n), k, n),
        4 => block_f64::<4>(a.add(i * k), b, c.add(i * n), k, n),
        3 => block_f64::<3>(a.add(i * k), b, c.add(i * n), k, n),
        2 => block_f64::<2>(a.add(i * k), b, c.add(i * n), k, n),
        1 => block_f64::<1>(a.add(i * k), b, c.add(i * n), k, n),
        _ => {}
    }
}

/// `R` rows of the result (all n columns, 8 at a time).
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2,fma")]
#[inline]
unsafe fn block_f64<const R: usize>(a: *const f64, b: *const f64, c: *mut f64, k: usize, n: usize) {
    use std::arch::x86_64::*;
    for j in (0..n).step_by(8) {
        let mut lo = [_mm256_setzero_pd(); R];
        let mut hi = [_mm256_setzero_pd(); R];
        let mut bp = b.add(j);
        for p in 0..k {
            let (b0, b1) = (_mm256_loadu_pd(bp), _mm256_loadu_pd(bp.add(4)));
            for r in 0..R {
                let x = _mm256_broadcast_sd(&*a.add(r * k + p));
                lo[r] = _mm256_fmadd_pd(x, b0, lo[r]);
                hi[r] = _mm256_fmadd_pd(x, b1, hi[r]);
            }
            bp = bp.add(n);
        }
        for r in 0..R {
            let out = c.add(r * n + j);
            _mm256_storeu_pd(out, lo[r]);
            _mm256_storeu_pd(out.add(4), hi[r]);
        }
    }
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2,fma")]
unsafe fn rows_f32(a: *const f32, b: *const f32, c: *mut f32, m: usize, k: usize, n: usize) {
    let mut i = 0;
    while i + 6 <= m {
        block_f32::<6>(a.add(i * k), b, c.add(i * n), k, n);
        i += 6;
    }
    match m - i {
        5 => block_f32::<5>(a.add(i * k), b, c.add(i * n), k, n),
        4 => block_f32::<4>(a.add(i * k), b, c.add(i * n), k, n),
        3 => block_f32::<3>(a.add(i * k), b, c.add(i * n), k, n),
        2 => block_f32::<2>(a.add(i * k), b, c.add(i * n), k, n),
        1 => block_f32::<1>(a.add(i * k), b, c.add(i * n), k, n),
        _ => {}
    }
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2,fma")]
#[inline]
unsafe fn block_f32<const R: usize>(a: *const f32, b: *const f32, c: *mut f32, k: usize, n: usize) {
    use std::arch::x86_64::*;
    for j in (0..n).step_by(16) {
        let mut lo = [_mm256_setzero_ps(); R];
        let mut hi = [_mm256_setzero_ps(); R];
        let mut bp = b.add(j);
        for p in 0..k {
            let (b0, b1) = (_mm256_loadu_ps(bp), _mm256_loadu_ps(bp.add(8)));
            for r in 0..R {
                let x = _mm256_broadcast_ss(&*a.add(r * k + p));
                lo[r] = _mm256_fmadd_ps(x, b0, lo[r]);
                hi[r] = _mm256_fmadd_ps(x, b1, hi[r]);
            }
            bp = bp.add(n);
        }
        for r in 0..R {
            let out = c.add(r * n + j);
            _mm256_storeu_ps(out, lo[r]);
            _mm256_storeu_ps(out.add(8), hi[r]);
        }
    }
}
