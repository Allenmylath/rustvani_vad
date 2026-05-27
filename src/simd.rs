//! SIMD-accelerated dot products.
//!
//! AVX2+FMA path on x86_64, scalar fallback everywhere else.

#[cfg(target_arch = "x86_64")]
use std::arch::x86_64::*;

#[inline]
pub fn dot(a: &[f32], b: &[f32]) -> f32 {
    #[cfg(target_arch = "x86_64")]
    {
        if is_x86_feature_detected!("avx2") && is_x86_feature_detected!("fma") {
            return unsafe { dot_avx2(a, b) };
        }
    }
    dot_scalar(a, b)
}

#[inline]
pub fn dot_relu(a: &[f32], b: &[f32]) -> f32 {
    #[cfg(target_arch = "x86_64")]
    {
        if is_x86_feature_detected!("avx2") && is_x86_feature_detected!("fma") {
            return unsafe { dot_relu_avx2(a, b) };
        }
    }
    let mut s = 0.0f32;
    for i in 0..a.len() {
        s += a[i] * b[i].max(0.0);
    }
    s
}

fn dot_scalar(a: &[f32], b: &[f32]) -> f32 {
    let n = a.len();
    let (mut s0, mut s1, mut s2, mut s3) = (0.0f32, 0.0f32, 0.0f32, 0.0f32);
    let chunks = n / 4;
    for i in 0..chunks {
        let j = i * 4;
        unsafe {
            s0 += *a.get_unchecked(j) * *b.get_unchecked(j);
            s1 += *a.get_unchecked(j + 1) * *b.get_unchecked(j + 1);
            s2 += *a.get_unchecked(j + 2) * *b.get_unchecked(j + 2);
            s3 += *a.get_unchecked(j + 3) * *b.get_unchecked(j + 3);
        }
    }
    for i in (chunks * 4)..n {
        unsafe {
            s0 += *a.get_unchecked(i) * *b.get_unchecked(i);
        }
    }
    s0 + s1 + s2 + s3
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2,fma")]
unsafe fn dot_avx2(a: &[f32], b: &[f32]) -> f32 {
    let n = a.len();
    let ap = a.as_ptr();
    let bp = b.as_ptr();
    let mut acc0 = _mm256_setzero_ps();
    let mut acc1 = _mm256_setzero_ps();
    let mut acc2 = _mm256_setzero_ps();
    let mut acc3 = _mm256_setzero_ps();
    let chunks32 = n / 32;
    for i in 0..chunks32 {
        let j = i * 32;
        acc0 = _mm256_fmadd_ps(
            _mm256_loadu_ps(ap.add(j)),
            _mm256_loadu_ps(bp.add(j)),
            acc0,
        );
        acc1 = _mm256_fmadd_ps(
            _mm256_loadu_ps(ap.add(j + 8)),
            _mm256_loadu_ps(bp.add(j + 8)),
            acc1,
        );
        acc2 = _mm256_fmadd_ps(
            _mm256_loadu_ps(ap.add(j + 16)),
            _mm256_loadu_ps(bp.add(j + 16)),
            acc2,
        );
        acc3 = _mm256_fmadd_ps(
            _mm256_loadu_ps(ap.add(j + 24)),
            _mm256_loadu_ps(bp.add(j + 24)),
            acc3,
        );
    }
    let done = chunks32 * 32;
    let chunks8 = (n - done) / 8;
    for i in 0..chunks8 {
        let j = done + i * 8;
        acc0 = _mm256_fmadd_ps(
            _mm256_loadu_ps(ap.add(j)),
            _mm256_loadu_ps(bp.add(j)),
            acc0,
        );
    }
    acc0 = _mm256_add_ps(acc0, acc1);
    acc2 = _mm256_add_ps(acc2, acc3);
    acc0 = _mm256_add_ps(acc0, acc2);
    let hi = _mm256_extractf128_ps::<1>(acc0);
    let lo = _mm256_castps256_ps128(acc0);
    let sum128 = _mm_add_ps(lo, hi);
    let shuf = _mm_movehdup_ps(sum128);
    let sums = _mm_add_ps(sum128, shuf);
    let shuf2 = _mm_movehl_ps(sums, sums);
    let result = _mm_add_ss(sums, shuf2);
    let mut total = _mm_cvtss_f32(result);
    let tail = done + chunks8 * 8;
    for i in tail..n {
        total += *ap.add(i) * *bp.add(i);
    }
    total
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2,fma")]
unsafe fn dot_relu_avx2(a: &[f32], b: &[f32]) -> f32 {
    let n = a.len();
    let ap = a.as_ptr();
    let bp = b.as_ptr();
    let zero = _mm256_setzero_ps();
    let mut acc = _mm256_setzero_ps();
    let chunks = n / 8;
    for i in 0..chunks {
        let j = i * 8;
        let vb = _mm256_max_ps(_mm256_loadu_ps(bp.add(j)), zero);
        acc = _mm256_fmadd_ps(_mm256_loadu_ps(ap.add(j)), vb, acc);
    }
    let hi = _mm256_extractf128_ps::<1>(acc);
    let lo = _mm256_castps256_ps128(acc);
    let sum128 = _mm_add_ps(lo, hi);
    let shuf = _mm_movehdup_ps(sum128);
    let sums = _mm_add_ps(sum128, shuf);
    let shuf2 = _mm_movehl_ps(sums, sums);
    let result = _mm_add_ss(sums, shuf2);
    let mut total = _mm_cvtss_f32(result);
    for i in (chunks * 8)..n {
        total += *ap.add(i) * (*bp.add(i)).max(0.0);
    }
    total
}
