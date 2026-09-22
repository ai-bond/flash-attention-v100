// ======================================================================================
// * Copyright (c) 2025, D.Skryabin / tg @ai_bond007 SPDX-License: BSD-3-Clause
// ======================================================================================
#pragma once

#include <cuda.h>
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include "swizzle.h"
#include "philox.h"

// ======================================================================================
// WMMA_GEMM_SOFTMAX: Online softmax with O-scaling
// ======================================================================================
// FA2 MATH: m_new   = max(m_old, rowmax(S))
//           P       = exp(S - m_new)    [clamped > -80]
//           l_new   = exp(m_old - m_new) * l_old + rowsum(P)
//           O_new   = exp(m_old - m_new) * O_old
// LAYOUT:   S [float], O [float] row-major, swizzled via ld_float4/st_float4(addr, row)
//           P [half ]            row-major, swizzled via st_half4/st_half(addr, row)
//           sS[row,k]            sP[row,2k], sP[row,2k+1]
// ======================================================================================
template <typename Config, int BLOCK_M, int BLOCK_N, int SCORE_STRIDE, int HEAD_STRIDE, bool IS_DROPOUT>
__device__ __forceinline__ void WMMA_GEMM_SOFTMAX(
    float*   __restrict__ SMEM_S,
    __half*  __restrict__ SMEM_P,
    float*   __restrict__ SMEM_O,
    float*   __restrict__ SMEM_MAX,
    float*   __restrict__ SMEM_SUM,
    __half*  __restrict__ GMEM_MASK,
    int      VALID_Q,
    int      VALID_KV,
    int      THREAD_ID,
    int      BLOCK_ID,
    int      GLOBAL_ROW_OFFSET,
    int      GLOBAL_COL_OFFSET,
    int      GLOBAL_N,
    float    P_DROPOUT,
    uint64_t DROPOUT_SEED,
    uint64_t DROPOUT_OFFSET,
    int      STRIDE_GMEM_MASK
) {
    if (VALID_Q == 0 || VALID_KV == 0) return;

    constexpr int THREADS_PER_ROW = Config::DO::THREADS_PER_ROW;
    constexpr int MAX_ITERS       = ((BLOCK_N >> 2) + THREADS_PER_ROW - 1) / THREADS_PER_ROW;
    constexpr int MAX_TAILS       = (4 + THREADS_PER_ROW - 1) / THREADS_PER_ROW;

    const int row     = THREAD_ID / THREADS_PER_ROW;
    const int thread  = THREAD_ID % THREADS_PER_ROW;
    const int cols    = VALID_KV >> 2;
    const int tail    = (VALID_KV >> 2) << 2;

    float thread_max = NEG_INF, new_max  = NEG_INF;
    float thread_sum = 0.0f,    exp_diff = 1.0f;

    float4 sS[MAX_ITERS];
    float  sS_tail[MAX_TAILS];

    if (row < VALID_Q) {
        uint32_t sS_base = __cvta_generic_to_shared(SMEM_S + row * SCORE_STRIDE);

        #pragma unroll
        for (int idx = 0; idx < MAX_ITERS; ++idx) {
            const int addr = thread + idx * THREADS_PER_ROW;
            if (addr < cols) {
                sS[idx] = ld_float4(sS_base + addr * 16, row);
                thread_max = fmaxf(thread_max, fmaxf(fmaxf(sS[idx].x, sS[idx].y), fmaxf(sS[idx].z, sS[idx].w)));
            }
        }

        if (tail < VALID_KV) {
            #pragma unroll
            for (int idx = 0; idx < MAX_TAILS; ++idx) {
                const int addr = tail + thread + idx * THREADS_PER_ROW;
                if (addr < VALID_KV) {
                    sS_tail[idx] = ld_float(sS_base + addr * 4, row);
                    thread_max = fmaxf(thread_max, sS_tail[idx]);
                }
            }
        }
    }

    #pragma unroll
    for (int offset = THREADS_PER_ROW / 2; offset > 0; offset >>= 1) {
        thread_max = fmaxf(thread_max, __shfl_xor_sync(0xFFFFFFFFU, thread_max, offset, THREADS_PER_ROW));
    }

    if (row < VALID_Q) {
        uint32_t sP_base = __cvta_generic_to_shared(SMEM_P + row * SCORE_STRIDE * 2);

        new_max  =  fmaxf(SMEM_MAX[row],  thread_max);
        exp_diff = __expf(SMEM_MAX[row] - new_max);

        #pragma unroll
        for (int idx = 0; idx < MAX_ITERS; ++idx) {
            const int addr = thread + idx * THREADS_PER_ROW;
            if (addr < cols) {
                float e0 = __expf(fmaxf(sS[idx].x - new_max, -80.0f));
                float e1 = __expf(fmaxf(sS[idx].y - new_max, -80.0f));
                float e2 = __expf(fmaxf(sS[idx].z - new_max, -80.0f));
                float e3 = __expf(fmaxf(sS[idx].w - new_max, -80.0f));

                thread_sum += (e0 + e1) + (e2 + e3);

                if constexpr (IS_DROPOUT) {
                    const float  rp_dropout = (1.0f / (1.0f - P_DROPOUT));
                    const uint32_t drop_thr = static_cast<uint32_t>((1.0f - P_DROPOUT) * 4294967295.0f);
                    uint64_t addr_plx = static_cast<uint64_t>(GLOBAL_ROW_OFFSET + row) * GLOBAL_N + (GLOBAL_COL_OFFSET + addr * 4);

                    PhiloxState philox = init_philox(DROPOUT_SEED, DROPOUT_OFFSET + (addr_plx >> 2));
                    uint4 rng = philox.next();

                    uint32_t r0 = ((addr_plx + 0) & 3) == 0 ? rng.x : ((addr_plx + 0) & 3) == 1 ? rng.y : ((addr_plx + 0) & 3) == 2 ? rng.z : rng.w;
                    uint32_t r1 = ((addr_plx + 1) & 3) == 0 ? rng.x : ((addr_plx + 1) & 3) == 1 ? rng.y : ((addr_plx + 1) & 3) == 2 ? rng.z : rng.w;
                    uint32_t r2 = ((addr_plx + 2) & 3) == 0 ? rng.x : ((addr_plx + 2) & 3) == 1 ? rng.y : ((addr_plx + 2) & 3) == 2 ? rng.z : rng.w;
                    uint32_t r3 = ((addr_plx + 3) & 3) == 0 ? rng.x : ((addr_plx + 3) & 3) == 1 ? rng.y : ((addr_plx + 3) & 3) == 2 ? rng.z : rng.w;

                    uint32_t k0 = (r0 <= drop_thr);
                    uint32_t k1 = (r1 <= drop_thr);
                    uint32_t k2 = (r2 <= drop_thr);
                    uint32_t k3 = (r3 <= drop_thr);

                    e0 = k0 ? (e0 * rp_dropout) : 0.0f;
                    e1 = k1 ? (e1 * rp_dropout) : 0.0f;
                    e2 = k2 ? (e2 * rp_dropout) : 0.0f;
                    e3 = k3 ? (e3 * rp_dropout) : 0.0f;

                    if (GMEM_MASK != nullptr) {
                        ushort gmem0 = 0x3C00 | (k0 ? 0 : 0x8000);
                        ushort gmem1 = 0x3C00 | (k1 ? 0 : 0x8000);
                        ushort gmem2 = 0x3C00 | (k2 ? 0 : 0x8000);
                        ushort gmem3 = 0x3C00 | (k3 ? 0 : 0x8000);
                        uint64_t gmem_addr = __cvta_generic_to_global(GMEM_MASK + row * STRIDE_GMEM_MASK + addr * 4);
                        asm volatile("st.global.v4.u16 [%0], {%1, %2, %3, %4};\n"
                                     :: "l"(gmem_addr), "h"(gmem0), "h"(gmem1), "h"(gmem2), "h"(gmem3) : "memory");
                    }
                }
                st_half4(sP_base + addr * 8, __float22half2_rn(make_float2(e0, e1)), __float22half2_rn(make_float2(e2, e3)), row);
            }
        }

        if (tail < VALID_KV) {
            #pragma unroll
            for (int idx = 0; idx < MAX_TAILS; ++idx) {
                const int addr = tail + thread + idx * THREADS_PER_ROW;
                if (addr < VALID_KV) {
                    float e = __expf(fmaxf(sS_tail[idx] - new_max, -80.0f));
                    thread_sum += e;

                    if constexpr (IS_DROPOUT) {
                        const float  rp_dropout = (1.0f / (1.0f - P_DROPOUT));
                        const uint32_t drop_thr = static_cast<uint32_t>((1.0f - P_DROPOUT) * 4294967295.0f);
                        uint64_t addr_plx_tail  = static_cast<uint64_t>(GLOBAL_ROW_OFFSET + row) * GLOBAL_N + (GLOBAL_COL_OFFSET + addr);

                        PhiloxState philox = init_philox(DROPOUT_SEED, DROPOUT_OFFSET + (addr_plx_tail >> 2));
                        uint4 rng  = philox.next();
                        uint32_t r = (addr_plx_tail & 3) == 0 ? rng.x : (addr_plx_tail & 3) == 1 ? rng.y : (addr_plx_tail & 3) == 2 ? rng.z : rng.w;

                        uint32_t kk = (r <= drop_thr);
                        e = kk ? (e * rp_dropout) : 0.0f;

                        if (GMEM_MASK != nullptr) {
                            ushort gmem = 0x3C00 | (kk ? 0 : 0x8000);
                            uint64_t gmem_addr = __cvta_generic_to_global(GMEM_MASK + row * STRIDE_GMEM_MASK + addr);
                            asm volatile("st.global.u16 [%0], %1;\n" :: "l"(gmem_addr), "h"(gmem) : "memory");
                        }
                    }
                    st_half(sP_base + addr * 2, __float2half_rn(e), row);
                }
            }
        }

        if (VALID_KV < BLOCK_N) {
            #pragma unroll
            for (int idx = VALID_KV + thread; idx < BLOCK_N; idx += THREADS_PER_ROW) {
                st_half(sP_base + idx * 2, __float2half(0.0f), row);
            }
        }
    }

    #pragma unroll
    for (int offset = THREADS_PER_ROW / 2; offset > 0; offset >>= 1) {
        thread_sum += __shfl_xor_sync(0xFFFFFFFFU, thread_sum, offset, THREADS_PER_ROW);
    }

    if (thread == 0) {
        SMEM_SUM[row] = exp_diff * SMEM_SUM[row] + thread_sum;
        SMEM_MAX[row] = new_max;
    }

    if (row < VALID_Q && BLOCK_ID > 0) {
        uint32_t sO_base = __cvta_generic_to_shared(SMEM_O + row * HEAD_STRIDE);
        #pragma unroll 4
        for (int idx = thread; idx < ((HEAD_STRIDE + 3) >> 2); idx += THREADS_PER_ROW) {
            float4 sO = ld_float4(sO_base + idx * 16, row);
            sO.x *= exp_diff;
            sO.y *= exp_diff;
            sO.z *= exp_diff;
            sO.w *= exp_diff;
            st_float4(sO_base + idx * 16, sO, row);
        }
    }
}

// ======================================================================================
// WMMA_GEMM_SOFTMAX_GRADIENT: Recompute P & dS for backward pass
// ======================================================================================
// FA2 MATH: P_orig = exp(S - lse)
//           P_drop = P_orig * mask / (1-p)
//           dS     = (P_drop * dOV - P_orig * D) * softmax_scale
//           if softcap: dS *= (1 - (S/c)^2)   [FMA form]
//
// LAYOUT:   S [float], dOV[float] row-major, stride SMEM_LDS_STRIDE (floats)
//           P [half ], dS [half ] row-major, stride SMEM_LDO_STRIDE (halves)
//           sS[row,k]             sP[row,2k],sP[row,2k+1]
// ======================================================================================
template<typename Config, GemmType TYPE, bool IS_SOFTCAP, bool IS_DROPOUT, int SMEM_LDS_STRIDE, int SMEM_LDO_STRIDE, int TILE_X, int TILE_Y>
__device__ __forceinline__ void WMMA_GEMM_SOFTMAX_GRADIENT(
    const float* __restrict__ SMEM_S,
    const float* __restrict__ SMEM_DOV,
    const float* __restrict__ SMEM_LSE,
    const float* __restrict__ SMEM_DOT,
         __half* __restrict__ SMEM_P,
         __half* __restrict__ SMEM_DS,
    int      VALID_Q_ROWS,
    int      VALID_KV_ROWS,
    float    SOFTMAX_SCALE,
    float    SOFTCAP,
    float    P_DROPOUT,
    uint64_t DROPOUT_SEED,
    uint64_t DROPOUT_OFFSET,
    int      GLOBAL_ROW_OFFSET,
    int      GLOBAL_COL_OFFSET,
    int      GLOBAL_N,
    int      THREAD_ID
) {
    constexpr bool PHASE             = static_cast<uint8_t>(TYPE) & 0x1;
    constexpr int  THREADS_PER_BLOCK = Config::THREADS_PER_BLOCK;

    const float    softcap_inv = IS_SOFTCAP ? (1.0f / SOFTCAP) : 0.0f;
    const float    rp_dropout  = IS_DROPOUT ? (1.0f / (1.0f - P_DROPOUT)) : 1.0f;
    const uint32_t drop_thr    = IS_DROPOUT ? static_cast<uint32_t>((1.0f - P_DROPOUT) * 4294967295.0f) : 0u;

    float4  prev_ds  = make_float4(0.0f, 0.0f, 0.0f, 0.0f);
    int     prev_row  = -1;
    int     prev_col  =  0;
    bool    prev_has  = false;

    #pragma unroll 1
    for (int i = THREAD_ID; i < (TILE_X * TILE_Y >> 2); i += THREADS_PER_BLOCK) {
        const int idx = i << 2;
        const int row = idx / TILE_Y;
        const int col = idx % TILE_Y;

        const bool is_valid = (row < VALID_Q_ROWS);
        const bool in0      = is_valid && (col       < VALID_KV_ROWS);
        const bool in1      = is_valid && ((col + 1) < VALID_KV_ROWS);
        const bool in2      = is_valid && ((col + 2) < VALID_KV_ROWS);
        const bool in3      = is_valid && ((col + 3) < VALID_KV_ROWS);

        const float lse = is_valid ? SMEM_LSE[row] : 0.0f;
        const float dot = is_valid ? SMEM_DOT[row] : 0.0f;

        float4 sS, sdOV;

        if (is_valid) {
            const uint32_t sS_base   = __cvta_generic_to_shared(SMEM_S   + row * SMEM_LDS_STRIDE);
            const uint32_t sDOV_base = __cvta_generic_to_shared(SMEM_DOV + row * SMEM_LDS_STRIDE);
            sS   = ld_float4(sS_base   + col * 4, row);
            sdOV = ld_float4(sDOV_base + col * 4, row);
        } else {
            sS   = make_float4(NEG_INF, NEG_INF, NEG_INF, NEG_INF);
            sdOV = make_float4(0.0f,    0.0f,    0.0f,    0.0f);
        }

        if (!in0) { sS.x = NEG_INF; sdOV.x = 0.0f; }
        if (!in1) { sS.y = NEG_INF; sdOV.y = 0.0f; }
        if (!in2) { sS.z = NEG_INF; sdOV.z = 0.0f; }
        if (!in3) { sS.w = NEG_INF; sdOV.w = 0.0f; }

        float p0 = (sS.x == NEG_INF || (sS.x - lse) < -80.0f) ? 0.0f : __expf(sS.x - lse);
        float p1 = (sS.y == NEG_INF || (sS.y - lse) < -80.0f) ? 0.0f : __expf(sS.y - lse);
        float p2 = (sS.z == NEG_INF || (sS.z - lse) < -80.0f) ? 0.0f : __expf(sS.z - lse);
        float p3 = (sS.w == NEG_INF || (sS.w - lse) < -80.0f) ? 0.0f : __expf(sS.w - lse);

        float pd0 = p0, pd1 = p1, pd2 = p2, pd3 = p3;

        if constexpr (IS_DROPOUT) {
            const uint64_t addr_plx = static_cast<uint64_t>(GLOBAL_ROW_OFFSET + row) * GLOBAL_N + (GLOBAL_COL_OFFSET + col);
            const uint4         rng = init_philox(DROPOUT_SEED, DROPOUT_OFFSET + (addr_plx >> 2)).next();

            if (in0) pd0 = (rng.x <= drop_thr) ? (p0 * rp_dropout) : 0.0f;
            if (in1) pd1 = (rng.y <= drop_thr) ? (p1 * rp_dropout) : 0.0f;
            if (in2) pd2 = (rng.z <= drop_thr) ? (p2 * rp_dropout) : 0.0f;
            if (in3) pd3 = (rng.w <= drop_thr) ? (p3 * rp_dropout) : 0.0f;
        }

        float ds0 = __fmaf_rn(pd0, sdOV.x, -p0 * dot) * SOFTMAX_SCALE;
        float ds1 = __fmaf_rn(pd1, sdOV.y, -p1 * dot) * SOFTMAX_SCALE;
        float ds2 = __fmaf_rn(pd2, sdOV.z, -p2 * dot) * SOFTMAX_SCALE;
        float ds3 = __fmaf_rn(pd3, sdOV.w, -p3 * dot) * SOFTMAX_SCALE;

        if constexpr (IS_SOFTCAP) {
            if (sS.x > NEG_INF) { const float n = __fmul_rn(sS.x, softcap_inv); ds0 = __fmul_rn(ds0, __fmaf_rn(-n, n, 1.0f)); }
            if (sS.y > NEG_INF) { const float n = __fmul_rn(sS.y, softcap_inv); ds1 = __fmul_rn(ds1, __fmaf_rn(-n, n, 1.0f)); }
            if (sS.z > NEG_INF) { const float n = __fmul_rn(sS.z, softcap_inv); ds2 = __fmul_rn(ds2, __fmaf_rn(-n, n, 1.0f)); }
            if (sS.w > NEG_INF) { const float n = __fmul_rn(sS.w, softcap_inv); ds3 = __fmul_rn(ds3, __fmaf_rn(-n, n, 1.0f)); }
        }

        if constexpr (!PHASE) {
            if (prev_has) {
                const uint32_t sDS_base = __cvta_generic_to_shared(SMEM_DS + prev_row * SMEM_LDO_STRIDE);
                const uint32_t addr  = sDS_base + prev_col * 2;
                st_half4(addr, __float22half2_rn(make_float2(prev_ds.x, prev_ds.y)), __float22half2_rn(make_float2(prev_ds.z, prev_ds.w)), prev_row);
            }
            prev_ds  = make_float4(ds0, ds1, ds2, ds3);
            prev_row = row;
            prev_col = col;
            prev_has = true;
        } else {
            const uint32_t sDS_base = __cvta_generic_to_shared(SMEM_DS + row * SMEM_LDO_STRIDE);
            const uint32_t sP_base  = __cvta_generic_to_shared(SMEM_P  + row * SMEM_LDO_STRIDE);
            const uint32_t addr_ds  = sDS_base + col * 2;
            const uint32_t addr_p   = sP_base  + col * 2;

            st_half4(addr_ds, __float22half2_rn(make_float2(ds0, ds1)), __float22half2_rn(make_float2(ds2, ds3)), row);
            st_half4(addr_p,  __float22half2_rn(make_float2(pd0, pd1)), __float22half2_rn(make_float2(pd2, pd3)), row);
        }
    }

    if constexpr (!PHASE) {
        if (prev_has) {
            const uint32_t sDS_base = __cvta_generic_to_shared(SMEM_DS + prev_row * SMEM_LDO_STRIDE);
            const uint32_t addr  = sDS_base + prev_col * 2;
            st_half4(addr, __float22half2_rn(make_float2(prev_ds.x, prev_ds.y)), __float22half2_rn(make_float2(prev_ds.z, prev_ds.w)), prev_row);
        }
    }
}