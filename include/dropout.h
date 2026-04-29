// ======================================================================================
// * Copyright (c) 2026, D.Skryabin / tg @ai_bond007 SPDX-License: BSD-3-Clause
// ======================================================================================
#pragma once

#include <cuda.h>
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include "philox.h"

// ======================================================================================
// WMMA_GEMM_DROPOUT: Forward dropout application & GMEM sign-encoding
// FA2 MATH: P_drop = P_orig * mask / (1-p)
//           GMEM: sign-bit flip for dropped elements (validation)
// ======================================================================================
template<typename Config, int BLOCK_M, int BLOCK_N, int SCORE_STRIDE, bool IS_DROPOUT>
__device__ __forceinline__ void WMMA_GEMM_DROPOUT(
    __half* __restrict__ SMEM_P,
    __half* __restrict__ GMEM_P,
    int VALID_Q,
    int VALID_KV,
    int GLOBAL_Q_OFFSET,
    int GLOBAL_KV_OFFSET,
    int GLOBAL_N,
    float    P_DROPOUT,
    uint64_t DROPOUT_SEED,
    uint64_t DROPOUT_OFFSET,
    int THREAD_ID)
{
    if (!IS_DROPOUT && GMEM_P == nullptr) return;

    constexpr int THREADS_PER_BLOCK = Config::THREADS_PER_BLOCK;

    const int chunks_per_row = VALID_KV >> 2;
    const int total_chunks   = VALID_Q * chunks_per_row;

    if (total_chunks == 0) return;

    const float rp_dropout = IS_DROPOUT ? (1.0f / (1.0f - P_DROPOUT)) : 1.0f;
    const uint32_t drop_thr = IS_DROPOUT ? static_cast<uint32_t>((1.0f - P_DROPOUT) * 4294967295.0f) : 0;

    uint32_t smem_base = static_cast<uint32_t>(__cvta_generic_to_shared(SMEM_P));
    uint64_t gmem_base = (GMEM_P != nullptr) ? static_cast<uint64_t>(__cvta_generic_to_global(GMEM_P)) : 0;

    #pragma unroll 1
    for (int i = THREAD_ID; i < total_chunks; i += THREADS_PER_BLOCK) {
        const int row   = i / chunks_per_row;
        const int chunk = i % chunks_per_row;
        const int col   = chunk << 2;

        uint64_t flat_idx = static_cast<uint64_t>(GLOBAL_Q_OFFSET + row) * GLOBAL_N + (GLOBAL_KV_OFFSET + col);
        PhiloxState philox = init_philox(DROPOUT_SEED, DROPOUT_OFFSET + (flat_idx >> 2));
        uint4 rng = philox.next();

        uint32_t m0 = IS_DROPOUT ? (rng.x <= drop_thr) : 1;
        uint32_t m1 = IS_DROPOUT ? (rng.y <= drop_thr) : 1;
        uint32_t m2 = IS_DROPOUT ? (rng.z <= drop_thr) : 1;
        uint32_t m3 = IS_DROPOUT ? (rng.w <= drop_thr) : 1;

        uint32_t smem_addr = smem_base + (row * SCORE_STRIDE + col) * 2;
        uint2 h_bits;
        asm volatile(
            "{\n"
            "  mov.b32 %0, 0; mov.b32 %1, 0;\n"
            "  ld.shared.v2.u32 {%0, %1}, [%2];\n"
            "}\n"
            : "=r"(h_bits.x), "=r"(h_bits.y)
            : "r"(smem_addr)
            : "memory"
        );

        float p0 = __half2float(__ushort_as_half(h_bits.x & 0xFFFF));
        float p1 = __half2float(__ushort_as_half(h_bits.x >> 16));
        float p2 = __half2float(__ushort_as_half(h_bits.y & 0xFFFF));
        float p3 = __half2float(__ushort_as_half(h_bits.y >> 16));

        p0 = m0 ? (p0 * rp_dropout) : 0.0f;
        p1 = m1 ? (p1 * rp_dropout) : 0.0f;
        p2 = m2 ? (p2 * rp_dropout) : 0.0f;
        p3 = m3 ? (p3 * rp_dropout) : 0.0f;

        ushort u0 = __half_as_ushort(__float2half_rn(p0));
        ushort u1 = __half_as_ushort(__float2half_rn(p1));
        ushort u2 = __half_as_ushort(__float2half_rn(p2));
        ushort u3 = __half_as_ushort(__float2half_rn(p3));

        if constexpr (IS_DROPOUT) {
            u0 ^= (m0 ? 0 : 0x8000);
            u1 ^= (m1 ? 0 : 0x8000);
            u2 ^= (m2 ? 0 : 0x8000);
            u3 ^= (m3 ? 0 : 0x8000);
        }

        uint2 smem_out;
        smem_out.x = (u1 << 16) | u0;
        smem_out.y = (u3 << 16) | u2;

        asm volatile(
            "st.shared.v2.u32 [%0], {%1, %2};\n"
            : : "r"(smem_addr), "r"(smem_out.x), "r"(smem_out.y)
            : "memory"
        );

        if (GMEM_P != nullptr) {
            uint64_t gmem_addr = gmem_base + (static_cast<uint64_t>(row) * GLOBAL_N + col) * 2;
            asm volatile(
                "{\n"
                "  .reg .pred p;\n"
                "  setp.ne.b32 p, %4, 0;\n"
                "  @p st.global.v4.u16 [%5], {%0, %1, %2, %3};\n"
                "}\n"
                : : "h"(u0), "h"(u1), "h"(u2), "h"(u3), "r"(1), "l"(gmem_addr)
                : "memory"
            );
        }
    }

    const int tail_start = chunks_per_row << 2;

    if (tail_start < VALID_KV) {
        const int row = THREAD_ID / (THREADS_PER_BLOCK / ((VALID_KV - tail_start + 3) >> 2));
        const int thr = THREAD_ID % (THREADS_PER_BLOCK / ((VALID_KV - tail_start + 3) >> 2));
        if (row < VALID_Q) {
            for (int c = tail_start + thr; c < VALID_KV; c += (THREADS_PER_BLOCK / ((VALID_KV - tail_start + 3) >> 2))) {

                uint64_t flat_idx = static_cast<uint64_t>(GLOBAL_Q_OFFSET + row) * GLOBAL_N + (GLOBAL_KV_OFFSET + c);
                PhiloxState philox = init_philox(DROPOUT_SEED, DROPOUT_OFFSET + (flat_idx >> 2));

                uint4 rng = philox.next();
                uint32_t r_val = (flat_idx & 3) == 0 ? rng.x : (flat_idx & 3) == 1 ? rng.y : (flat_idx & 3) == 2 ? rng.z : rng.w;

                float p = __half2float(SMEM_P[row * SCORE_STRIDE + c]);
                uint32_t keep = IS_DROPOUT ? (r_val <= drop_thr) : 1;
                p = keep ? (p * rp_dropout) : 0.0f;

                ushort u = __half_as_ushort(__float2half_rn(p));
                if constexpr (IS_DROPOUT) u ^= (keep ? 0 : 0x8000);

                SMEM_P[row * SCORE_STRIDE + c] = __ushort_as_half(u);

                if (GMEM_P != nullptr) {
                    GMEM_P[(size_t)row * GLOBAL_N + c] = __ushort_as_half(u);
                }
            }
        }
    }
}
