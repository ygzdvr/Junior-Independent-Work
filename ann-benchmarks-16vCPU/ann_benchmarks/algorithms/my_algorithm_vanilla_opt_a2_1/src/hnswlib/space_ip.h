//  -----------------------------------------------------------------------------
//  inner_product_optimized.hpp – a drop‑in replacement for the legacy inner‑product
//  kernels found in hnswlib.  The implementation focusses on three goals:
//    1. *Single‑source* – every optimisation lives in one file, so there is no
//       code duplication.
//    2. *Runtime dispatch* – the best kernel is selected on the fly with
//       __builtin_cpu_supports / CPUID, avoiding multiple binaries.
//    3. *Micro‑architectural tuning* – explicit FMA, unrolling and pre‑fetching
//       reduce latency on modern Intel/AMD cores while retaining a clean scalar
//       fall‑back for every platform.
//  -----------------------------------------------------------------------------
//  Copyright (c) 2025  Your Name <you@example.com>
//  SPDX‑License‑Identifier: MIT
//  -----------------------------------------------------------------------------
#pragma once

#include <cstddef>
#include <cstdint>
#include <immintrin.h>
#include <limits>
#include "hnswlib.h"

namespace hnswlib {

namespace detail {
// ------------------------------------------------------------
//  Scalar reference kernel – small enough to be always inlined.
// ------------------------------------------------------------
inline float dot_scalar(const float * __restrict a,
                        const float * __restrict b,
                        std::size_t dim) noexcept {
    float acc = 0.f;
    for (std::size_t i = 0; i < dim; ++i)
        acc += a[i] * b[i];
    return acc;
}

// ------------------------------------------------------------
//  SSE2 kernel – handles any dimension, 4‑wide reduction.
// ------------------------------------------------------------
#if defined(__SSE2__)
inline float dot_sse(const float * __restrict a,
                     const float * __restrict b,
                     std::size_t dim) noexcept {
    const std::size_t kBlock = 16; // process 16 floats (4 × 128 bit) per loop
    const float *aEnd = a + (dim & ~(kBlock - 1));

    __m128 acc0 = _mm_setzero_ps();
    __m128 acc1 = _mm_setzero_ps();
    __m128 acc2 = _mm_setzero_ps();
    __m128 acc3 = _mm_setzero_ps();

    while (a < aEnd) {
        acc0 = _mm_add_ps(acc0, _mm_mul_ps(_mm_loadu_ps(a + 0), _mm_loadu_ps(b + 0)));
        acc1 = _mm_add_ps(acc1, _mm_mul_ps(_mm_loadu_ps(a + 4), _mm_loadu_ps(b + 4)));
        acc2 = _mm_add_ps(acc2, _mm_mul_ps(_mm_loadu_ps(a + 8), _mm_loadu_ps(b + 8)));
        acc3 = _mm_add_ps(acc3, _mm_mul_ps(_mm_loadu_ps(a + 12), _mm_loadu_ps(b + 12)));
        a += kBlock;
        b += kBlock;
    }

    // final horizontal add of the 4 accumulators
    __m128 acc01 = _mm_add_ps(acc0, acc1);
    __m128 acc23 = _mm_add_ps(acc2, acc3);
    __m128 acc   = _mm_add_ps(acc01, acc23);

    float tmp[4];
    _mm_storeu_ps(tmp, acc);
    float res = tmp[0] + tmp[1] + tmp[2] + tmp[3];

    // tail – fewer than 16 remaining elements
    if (std::size_t tail = dim & (kBlock - 1))
        res += dot_scalar(a, b, tail);

    return res;
}
#endif // __SSE2__

// ------------------------------------------------------------
//  AVX2 + FMA kernel – 16‑wide, reduced in 256‑bit lanes, then 128.
// ------------------------------------------------------------
#if defined(__AVX2__)
inline float dot_avx2(const float * __restrict a,
                      const float * __restrict b,
                      std::size_t dim) noexcept {
    const std::size_t kBlock = 32; // 32 floats (4 × 256 bit) per iteration
    const float *aEnd = a + (dim & ~(kBlock - 1));

    __m256 acc0 = _mm256_setzero_ps();
    __m256 acc1 = _mm256_setzero_ps();
    __m256 acc2 = _mm256_setzero_ps();
    __m256 acc3 = _mm256_setzero_ps();

    while (a < aEnd) {
        acc0 = _mm256_fmadd_ps(_mm256_loadu_ps(a + 0),  _mm256_loadu_ps(b + 0),  acc0);
        acc1 = _mm256_fmadd_ps(_mm256_loadu_ps(a + 8),  _mm256_loadu_ps(b + 8),  acc1);
        acc2 = _mm256_fmadd_ps(_mm256_loadu_ps(a + 16), _mm256_loadu_ps(b + 16), acc2);
        acc3 = _mm256_fmadd_ps(_mm256_loadu_ps(a + 24), _mm256_loadu_ps(b + 24), acc3);
        a += kBlock;
        b += kBlock;
    }
    // pairwise add 256‑bit lanes → 128‑bit, then scalar.
    __m256 acc01 = _mm256_add_ps(acc0, acc1);
    __m256 acc23 = _mm256_add_ps(acc2, acc3);
    __m256 acc   = _mm256_add_ps(acc01, acc23);

    __m128 hi = _mm256_extractf128_ps(acc, 1);
    __m128 lo = _mm256_castps256_ps128(acc);
    __m128 sum128 = _mm_add_ps(hi, lo);

    float tmp[4];
    _mm_storeu_ps(tmp, sum128);
    float res = tmp[0] + tmp[1] + tmp[2] + tmp[3];

    if (std::size_t tail = dim & (kBlock - 1))
        res += dot_scalar(a, b, tail);
    return res;
}
#endif // __AVX2__

// ------------------------------------------------------------
//  AVX‑512F kernel – 64‑wide, reduce with _mm512_reduce_add_ps.
// ------------------------------------------------------------
#if defined(__AVX512F__)
inline float dot_avx512(const float * __restrict a,
                        const float * __restrict b,
                        std::size_t dim) noexcept {
    const std::size_t kBlock = 64; // 64 floats (4 × 512 bit) per iteration
    const float *aEnd = a + (dim & ~(kBlock - 1));

    __m512 acc0 = _mm512_setzero_ps();
    __m512 acc1 = _mm512_setzero_ps();
    __m512 acc2 = _mm512_setzero_ps();
    __m512 acc3 = _mm512_setzero_ps();

    while (a < aEnd) {
        acc0 = _mm512_fmadd_ps(_mm512_loadu_ps(a + 0),  _mm512_loadu_ps(b + 0),  acc0);
        acc1 = _mm512_fmadd_ps(_mm512_loadu_ps(a + 16), _mm512_loadu_ps(b + 16), acc1);
        acc2 = _mm512_fmadd_ps(_mm512_loadu_ps(a + 32), _mm512_loadu_ps(b + 32), acc2);
        acc3 = _mm512_fmadd_ps(_mm512_loadu_ps(a + 48), _mm512_loadu_ps(b + 48), acc3);
        a += kBlock;
        b += kBlock;
    }

    __m512 acc = _mm512_add_ps(_mm512_add_ps(acc0, acc1), _mm512_add_ps(acc2, acc3));
    float res = _mm512_reduce_add_ps(acc);

    if (std::size_t tail = dim & (kBlock - 1))
        res += dot_scalar(a, b, tail);
    return res;
}
#endif // __AVX512F__

//--------------------------------------------------------------------
//  Runtime dispatch – pick the fastest kernel available *once*.
//--------------------------------------------------------------------
using dotfun_t = float (*)(const float *, const float *, std::size_t);

inline dotfun_t select_kernel() noexcept {
#if defined(__AVX512F__)
    if (__builtin_cpu_supports("avx512f")) return &dot_avx512;
#endif
#if defined(__AVX2__)
    if (__builtin_cpu_supports("avx2")) return &dot_avx2;
#endif
#if defined(__SSE2__)
    if (__builtin_cpu_supports("sse2")) return &dot_sse;
#endif
    return &dot_scalar;
}

static const dotfun_t DOT_KERNEL = select_kernel();

} // namespace detail

// -----------------------------------------------------------------------------
//  Public interface compatible with hnswlib::SpaceInterface<float>.
// -----------------------------------------------------------------------------
class InnerProductSpace final : public SpaceInterface<float> {
    std::size_t dim_       {0};
    std::size_t data_size_ {0};

    // NB: we store a lambda pointer for clarity; could also expose directly.
    static float ip_distance(const void *v1, const void *v2, const void *dim_ptr) noexcept {
        const auto dim  = *static_cast<const std::size_t *>(dim_ptr);
        const auto *a   = static_cast<const float *>(v1);
        const auto *b   = static_cast<const float *>(v2);
        return 1.f - detail::DOT_KERNEL(a, b, dim);
    }

public:
    explicit InnerProductSpace(std::size_t dim) : dim_{dim}, data_size_{dim * sizeof(float)} {}

    [[nodiscard]] std::size_t get_data_size() override { return data_size_; }
    [[nodiscard]] DISTFUNC<float> get_dist_func() override { return &ip_distance; }
    [[nodiscard]] void * get_dist_func_param() override { return &dim_; }
};

} // namespace hnswlib
