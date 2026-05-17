// ======================================================================================
// * Copyright (c) 2026, D.Skryabin / tg @ai_bond007 SPDX-License: BSD-3-Clause
// ======================================================================================
#pragma once
#include <cuda.h>
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include "philox.h"

// ======================================================================================
// WMMA_GEMM_DROPOUT
// ======================================================================================
template<typename Config, int BLOCK_M, int BLOCK_N, int STRIDE_SCORE, bool IS_DROPOUT>
__device__ __forceinline__ void WMMA_GEMM_DROPOUT(
    __half* __restrict__ SMEM_P,
    __half* __restrict__ GMEM_P,
    int      VALID_Q,
    int      VALID_KV,
    int      GLOBAL_Q_OFFSET,
    int      GLOBAL_KV_OFFSET,
    int      STRIDE_RNG,
    int      STRIDE_GMEM,
    float    DROPOUT_P,
    uint64_t DROPOUT_SEED,
    uint64_t RNG_OFFSET,
    int      THREAD_ID
) {
    if (!IS_DROPOUT && GMEM_P == nullptr) return;

    constexpr int THREADS_PER_BLOCK = Config::THREADS_PER_BLOCK;

    const int chunks      = VALID_KV >> 2;
    const int total_iters = VALID_Q * chunks;

    if (total_iters == 0) return;

    const float    rp_drop  = IS_DROPOUT ? (1.0f / (1.0f - DROPOUT_P)) : 1.0f;
    const uint32_t keep_thr = IS_DROPOUT ? static_cast<uint32_t>((1.0f - DROPOUT_P) * 4294967295.0f) : 0;

    uint32_t smem_base = static_cast<uint32_t>(__cvta_generic_to_shared(SMEM_P));
    uint64_t gmem_base = (GMEM_P != nullptr) ? static_cast<uint64_t>(__cvta_generic_to_global(GMEM_P)) : 0;

    // ==================================================================================
    // MAIN LOOP: Vectorized processing (4 elements per iteration)
    // ==================================================================================
    #pragma unroll 2
    for (int i = THREAD_ID; i < total_iters; i += THREADS_PER_BLOCK) {
        const int row =  i / chunks;
        const int col = (i % chunks) << 2;

        const uint64_t gmem_offset = static_cast<uint64_t>(row) * STRIDE_GMEM;
        const uint32_t smem_addr   = smem_base + (row * STRIDE_SCORE + col) * 2;

        uint64_t flat_idx  = static_cast<uint64_t>(GLOBAL_Q_OFFSET + row) * STRIDE_RNG + (GLOBAL_KV_OFFSET + col);
        PhiloxState philox = init_philox(DROPOUT_SEED, RNG_OFFSET + (flat_idx >> 2));
        uint4 rng = philox.next();

        uint32_t k0 = IS_DROPOUT ? (rng.x <= keep_thr) : 1;
        uint32_t k1 = IS_DROPOUT ? (rng.y <= keep_thr) : 1;
        uint32_t k2 = IS_DROPOUT ? (rng.z <= keep_thr) : 1;
        uint32_t k3 = IS_DROPOUT ? (rng.w <= keep_thr) : 1;

        uint2 h_bits;
        asm volatile(
            "ld.shared.v2.u32 {%0, %1}, [%2];\n"
            : "=r"(h_bits.x), "=r"(h_bits.y)
            : "r"(smem_addr)
            : "memory"
        );

        float p0 = __half2float(__ushort_as_half(h_bits.x & 0xFFFF));
        float p1 = __half2float(__ushort_as_half(h_bits.x >> 16));
        float p2 = __half2float(__ushort_as_half(h_bits.y & 0xFFFF));
        float p3 = __half2float(__ushort_as_half(h_bits.y >> 16));

        p0 = k0 ? p0 * rp_drop : 0.0f;
        p1 = k1 ? p1 * rp_drop : 0.0f;
        p2 = k2 ? p2 * rp_drop : 0.0f;
        p3 = k3 ? p3 * rp_drop : 0.0f;

        ushort u0 = __half_as_ushort(__float2half_rn(p0));
        ushort u1 = __half_as_ushort(__float2half_rn(p1));
        ushort u2 = __half_as_ushort(__float2half_rn(p2));
        ushort u3 = __half_as_ushort(__float2half_rn(p3));

        uint2 smem_out;
        smem_out.x = (u1 << 16) | u0;
        smem_out.y = (u3 << 16) | u2;

        asm volatile(
            "st.shared.v2.u32 [%0], {%1, %2};\n"
            : : "r"(smem_addr), "r"(smem_out.x), "r"(smem_out.y)
            : "memory"
        );

        if (GMEM_P != nullptr) {
            ushort g0 = 0x3C00 | (k0 ? 0 : 0x8000);
            ushort g1 = 0x3C00 | (k1 ? 0 : 0x8000);
            ushort g2 = 0x3C00 | (k2 ? 0 : 0x8000);
            ushort g3 = 0x3C00 | (k3 ? 0 : 0x8000);

            uint64_t gmem_addr = gmem_base + (gmem_offset + col) * 2;
            asm volatile(
                "{\n"
                "  .reg .pred p;\n"
                "  setp.ne.b32 p, %4, 0;\n"
                "  @p st.global.v4.u16 [%5], {%0, %1, %2, %3};\n"
                "}\n"
                : : "h"(g0), "h"(g1), "h"(g2), "h"(g3), "r"(1), "l"(gmem_addr)
                : "memory"
            );
        }
    }

    // ==================================================================================
    // TAIL LOOP
    // ==================================================================================
    if ((chunks << 2) < VALID_KV) {
        const int work = THREADS_PER_BLOCK / (((VALID_KV - (chunks << 2)) + 3) >> 2);
        const int row  = THREAD_ID / work;
        const int thr  = THREAD_ID % work;

        if (row < VALID_Q) {
            const uint64_t gmem_offset = (GMEM_P != nullptr) ? static_cast<uint64_t>(row) * STRIDE_GMEM : 0;

            #pragma unroll 1
            for (int c = (chunks << 2) + thr; c < VALID_KV; c += work) {

                uint64_t flat_idx  = static_cast<uint64_t>(GLOBAL_Q_OFFSET + row) * STRIDE_RNG + (GLOBAL_KV_OFFSET + c);
                PhiloxState philox = init_philox(DROPOUT_SEED, RNG_OFFSET + (flat_idx >> 2));
                uint4 rng = philox.next();

                uint32_t r_val = (flat_idx & 3) == 0 ? rng.x : (flat_idx & 3) == 1 ? rng.y : (flat_idx & 3) == 2 ? rng.z : rng.w;

                float    p    = __half2float(SMEM_P[row * STRIDE_SCORE + c]);
                uint32_t keep = IS_DROPOUT ? (r_val <= keep_thr) : 1;
                         p    = keep ? p * rp_drop : 0.0f;
                ushort   u    = __half_as_ushort(__float2half_rn(p));

                SMEM_P[row * STRIDE_SCORE + c] = __ushort_as_half(u);

                if (GMEM_P != nullptr) {
                    ushort g = 0x3C00 | (keep ? 0 : 0x8000);
                    GMEM_P[gmem_offset + c] = __ushort_as_half(g);
                }
            }
        }
    }
}
