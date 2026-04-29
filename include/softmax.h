// ======================================================================================
// * Copyright (c) 2025, D.Skryabin / tg @ai_bond007 SPDX-License: BSD-3-Clause
// ======================================================================================
#pragma once

#include <cuda.h>
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include "philox.h"

// ======================================================================================
// WMMA_GEMM_SOFTMAX: Online softmax with O-scaling
// ======================================================================================
template<typename Config, int BLOCK_M, int BLOCK_N, int SCORE_STRIDE, int HEAD_STRIDE>
__device__ __forceinline__ void WMMA_GEMM_SOFTMAX(
    float*  __restrict__ SMEM_S,
    __half* __restrict__ SMEM_P,
    float*  __restrict__ SMEM_O,
    float*  __restrict__ SMEM_MAX,
    float*  __restrict__ SMEM_SUM,
    int VALID_Q,
    int VALID_KV,
    int THREAD_ID,
    int BLOCK_ID
) {
    if (VALID_Q == 0 || VALID_KV == 0) return;

    constexpr int  THREADS_PER_ROW = Config::DO::THREADS_PER_ROW;

    const int row      = THREAD_ID / THREADS_PER_ROW;
    const int thread   = THREAD_ID % THREADS_PER_ROW;

    float thread_max = NEG_INF, new_max = NEG_INF;
    float thread_sum = 0.0f, exp_diff = 1.0f;

    __half2 half_buffer[8];

      float* sS_float  = SMEM_S + row * SCORE_STRIDE;
     float4* sS_float4 = reinterpret_cast<float4*>(sS_float);
     __half* sP_half   = SMEM_P + row * SCORE_STRIDE;
    __half2* sP_half2  = reinterpret_cast<__half2*>(sP_half);

    const int cols = VALID_KV >> 2;
    const int tail = cols << 2;

    if (row < VALID_Q) {
        #pragma unroll 4
        for (int idx = thread; idx < cols; idx += THREADS_PER_ROW) {
            float4 buffer = sS_float4[idx];
            thread_max = fmaxf(thread_max, fmaxf(fmaxf(buffer.x, buffer.y), fmaxf(buffer.z, buffer.w)));
        }
    }

    #pragma unroll
    for (int offset = THREADS_PER_ROW / 2; offset > 0; offset >>= 1) {
        thread_max = fmaxf(thread_max, __shfl_xor_sync(0xFFFFFFFFU, thread_max, offset, THREADS_PER_ROW));
    }

    if (row < VALID_Q) {
        const float old_max = SMEM_MAX[row];
        new_max  = fmaxf(old_max, thread_max);
        exp_diff = __expf(old_max - new_max);

        int rb_idx = 0;
        #pragma unroll 4
        for (int idx = thread; idx < cols; idx += THREADS_PER_ROW) {
            float4 buffer = sS_float4[idx];
            float e0 = __expf(fmaxf(buffer.x - new_max, -80.0f));
            float e1 = __expf(fmaxf(buffer.y - new_max, -80.0f));
            float e2 = __expf(fmaxf(buffer.z - new_max, -80.0f));
            float e3 = __expf(fmaxf(buffer.w - new_max, -80.0f));

            thread_sum += (e0 + e1) + (e2 + e3);
            half_buffer[rb_idx++] = __float22half2_rn(make_float2(e0, e1));
            half_buffer[rb_idx++] = __float22half2_rn(make_float2(e2, e3));
        }

        if (tail < VALID_KV) {
            #pragma unroll 4
            for (int idx = tail + thread; idx < BLOCK_N; idx += THREADS_PER_ROW) {
                float v = (idx < VALID_KV) ? sS_float[idx] : NEG_INF;
                float e = __expf(fmaxf(v - new_max, -80.0f));
                thread_sum += (idx < VALID_KV) ? e : 0.0f;
                sP_half[idx] = (idx < VALID_KV) ? __float2half_rn(e) : __float2half(0.f);
            }
        }

        int wb_idx = 0;
        #pragma unroll 4
        for (int idx = thread; idx < cols; idx += THREADS_PER_ROW) {
            int base = idx * 2;
            sP_half2[base]     = half_buffer[wb_idx++];
            sP_half2[base + 1] = half_buffer[wb_idx++];
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
        float4* sO_float4 = reinterpret_cast<float4*>(SMEM_O + row * HEAD_STRIDE);
        #pragma unroll 4
        for (int idx = thread; idx < ((HEAD_STRIDE + 3) >> 2); idx += THREADS_PER_ROW) {
            float4 buffer = sO_float4[idx];
            buffer.x *= exp_diff; buffer.y *= exp_diff;
            buffer.z *= exp_diff; buffer.w *= exp_diff;
            sO_float4[idx] = buffer;
        }
    }
}

// ======================================================================================
// WMMA_GEMM_SOFTMAX_GRADIENT: Recompute P & dS for backward pass
// FA2 MATH: P_orig = exp(S - lse)
//           P_drop = P_orig * mask / (1-p)
//           dS     = (P_drop * dOV - P_orig * D) * softmax_scale
//           if softcap: dS *= (1 - (S/c)^2)  [S is already softcapped]
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
    int GLOBAL_ROW_OFFSET,
    int GLOBAL_COL_OFFSET,
    int GLOBAL_N,
    int THREAD_ID
) {
    constexpr int  TOTAL_ELEMENTS    = TILE_X * TILE_Y;
    constexpr int  TOTAL_PAIRS       = (TOTAL_ELEMENTS + 1) >> 1;
    constexpr bool PHASE             = static_cast<uint8_t>(TYPE) & 0x1;
    constexpr int  THREADS_PER_BLOCK = Config::THREADS_PER_BLOCK;

    const float softcap_inv = IS_SOFTCAP ? (1.0f / SOFTCAP) : 0.0f;
    const float rp_dropout  = IS_DROPOUT ? (1.0f / (1.0f - P_DROPOUT)) : 1.0f;
    const uint32_t drop_thr = IS_DROPOUT ? static_cast<uint32_t>((1.0f - P_DROPOUT) * 4294967295.0f) : 0;

    __half2  prev_ds = __float22half2_rn(make_float2(0.0f, 0.0f));
    int      prev_ldo0 = -1;
    uint32_t prev_aux = 0;

    #pragma unroll 1
    for (int i = THREAD_ID; i < TOTAL_PAIRS; i += THREADS_PER_BLOCK) {
        const int idx0 = i << 1;
        const int idx1 = idx0 + 1;
        const int row0 = idx0 / TILE_Y;
        const int col0 = idx0 % TILE_Y;
        const bool has_pair = (idx1 < TOTAL_ELEMENTS);
        const int row1 = has_pair ? (idx1 / TILE_Y) : row0;
        const int col1 = has_pair ? (idx1 % TILE_Y) : (col0 + 1);

        const bool in0 = (row0 < VALID_Q_ROWS) && (col0 < VALID_KV_ROWS);
        const bool in1 = has_pair && (row1 < VALID_Q_ROWS) && (col1 < VALID_KV_ROWS);

        const float lse0 = (row0 < VALID_Q_ROWS) ? SMEM_LSE[row0] : 0.0f;
        const float lse1 = (row1 < VALID_Q_ROWS) ? SMEM_LSE[row1] : lse0;
        const float dot0 = (row0 < VALID_Q_ROWS) ? SMEM_DOT[row0] : 0.0f;
        const float dot1 = (row1 < VALID_Q_ROWS) ? SMEM_DOT[row1] : dot0;

        const int lds0 = row0 * SMEM_LDS_STRIDE + col0;
        const int lds1 = has_pair ? (row1 * SMEM_LDS_STRIDE + col1) : 0;

        float s0 = in0 ? SMEM_S[lds0] : NEG_INF;
        float s1 = in1 ? SMEM_S[lds1] : NEG_INF;
        float dov0 = (s0 != NEG_INF) ? SMEM_DOV[lds0] : 0.0f;
        float dov1 = (s1 != NEG_INF) ? SMEM_DOV[lds1] : 0.0f;

        float sh0 = s0 - lse0;
        float sh1 = s1 - lse1;
        float p0_orig = (s0 == NEG_INF || sh0 < -80.0f) ? 0.0f : __expf(sh0);
        float p1_orig = (s1 == NEG_INF || sh1 < -80.0f) ? 0.0f : __expf(sh1);

        float p0_drop = p0_orig;
        float p1_drop = p1_orig;

        if constexpr (IS_DROPOUT) {
            if (in0) {
                uint64_t flat_idx0 = static_cast<uint64_t>(GLOBAL_ROW_OFFSET + row0) * GLOBAL_N + (GLOBAL_COL_OFFSET + col0);
                PhiloxState philox0 = init_philox(DROPOUT_SEED, DROPOUT_OFFSET + (flat_idx0 >> 2));
                uint4 rng0 = philox0.next();
                uint32_t r0 = (flat_idx0 & 3) == 0 ? rng0.x : (flat_idx0 & 3) == 1 ? rng0.y : (flat_idx0 & 3) == 2 ? rng0.z : rng0.w;
                p0_drop = (r0 <= drop_thr) ? (p0_orig * rp_dropout) : 0.0f;
            }
            if (in1) {
                uint64_t flat_idx1 = static_cast<uint64_t>(GLOBAL_ROW_OFFSET + row1) * GLOBAL_N + (GLOBAL_COL_OFFSET + col1);
                PhiloxState philox1 = init_philox(DROPOUT_SEED, DROPOUT_OFFSET + (flat_idx1 >> 2));
                uint4 rng1 = philox1.next();
                uint32_t r1 = (flat_idx1 & 3) == 0 ? rng1.x : (flat_idx1 & 3) == 1 ? rng1.y : (flat_idx1 & 3) == 2 ? rng1.z : rng1.w;
                p1_drop = (r1 <= drop_thr) ? (p1_orig * rp_dropout) : 0.0f;
            }
        }

        float ds0 = __fmaf_rn(p0_drop, dov0, -p0_orig * dot0) * SOFTMAX_SCALE;
        float ds1 = __fmaf_rn(p1_drop, dov1, -p1_orig * dot1) * SOFTMAX_SCALE;

        if constexpr (IS_SOFTCAP) {
            if (s0 > NEG_INF) { float s0_norm = __fmul_rn(s0, softcap_inv); ds0 = __fmul_rn(ds0, __fmaf_rn(-s0_norm, s0_norm, 1.0f)); }
            if (s1 > NEG_INF) { float s1_norm = __fmul_rn(s1, softcap_inv); ds1 = __fmul_rn(ds1, __fmaf_rn(-s1_norm, s1_norm, 1.0f)); }
        }

        const int ldo0 = row0 * SMEM_LDO_STRIDE + col0;
        const int ldo1 = has_pair ? (row1 * SMEM_LDO_STRIDE + col1) : 0;

        if constexpr (!PHASE) {
            if (prev_ldo0 >= 0) {
                const int p_ldo1 = prev_aux & 0xFFFF;
                const bool p_has = (prev_aux >> 16) & 1;
                const bool vec = p_has && ((prev_ldo0 & 1) == 0);

                if (vec) {
                    uintptr_t addr = reinterpret_cast<uintptr_t>(SMEM_DS + prev_ldo0);
                    if ((addr & 0x3) == 0) {
                        *reinterpret_cast<__half2*>(SMEM_DS + prev_ldo0) = prev_ds;
                    } else {
                        SMEM_DS[prev_ldo0] = prev_ds.x;
                        if (p_has) SMEM_DS[p_ldo1] = prev_ds.y;
                    }
                } else {
                    SMEM_DS[prev_ldo0] = prev_ds.x;
                    if (p_has) SMEM_DS[p_ldo1] = prev_ds.y;
                }
            }

            prev_ds   = __float22half2_rn(make_float2(ds0, ds1));
            prev_ldo0 = ldo0;
            prev_aux  = (static_cast<uint32_t>(ldo1) & 0xFFFF) | (static_cast<uint32_t>(has_pair) << 16);

        } else {
            const __half2 h2_ds = __float22half2_rn(make_float2(ds0, ds1));
            const __half2 h2_p  = __float22half2_rn(make_float2(p0_drop, p1_drop));

            const bool vec = has_pair && (row1 == row0) && ((ldo0 & 1) == 0);
            if (vec) {
                uintptr_t addr_ds = reinterpret_cast<uintptr_t>(SMEM_DS + ldo0);
                uintptr_t addr_p  = reinterpret_cast<uintptr_t>(SMEM_P  + ldo0);
                if (((addr_ds & 0x3) == 0) && ((addr_p & 0x3) == 0)) {
                    *reinterpret_cast<__half2*>(SMEM_DS + ldo0) = h2_ds;
                    *reinterpret_cast<__half2*>(SMEM_P  + ldo0) = h2_p;
                    continue;
                }
            }
            SMEM_DS[ldo0] = h2_ds.x;
            SMEM_P [ldo0] = h2_p.x;
            if (has_pair) {
                SMEM_DS[ldo1] = h2_ds.y;
                SMEM_P [ldo1] = h2_p.y;
            }
        }
    }

    if constexpr (!PHASE) {
        if (prev_ldo0 >= 0) {
            const int p_ldo1 = prev_aux & 0xFFFF;
            const bool p_has = (prev_aux >> 16) & 1;
            const bool vec   = p_has && ((prev_ldo0 & 1) == 0);

            if (vec) {
                uintptr_t addr = reinterpret_cast<uintptr_t>(SMEM_DS + prev_ldo0);
                if ((addr & 0x3) == 0) {
                    *reinterpret_cast<__half2*>(SMEM_DS + prev_ldo0) = prev_ds;
                } else {
                    SMEM_DS[prev_ldo0] = prev_ds.x;
                    if (p_has) SMEM_DS[p_ldo1] = prev_ds.y;
                }
            } else {
                SMEM_DS[prev_ldo0] = prev_ds.x;
                if (p_has) SMEM_DS[p_ldo1] = prev_ds.y;
            }
        }
    }
}
