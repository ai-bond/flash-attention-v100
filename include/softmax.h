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
// WMMA_GEMM_SOFTMAX: Online softmax, P written to explicit matrix_a output rP
// ======================================================================================
// FA2 MATH: m_new  = max(m_old, rowmax(S))
//           P      = exp2(S·log2e - m_new·log2e)
//           l_new  = exp2(m_old·log2e - m_new·log2e) * l_old + rowsum(P)
//           O_new  = exp2(m_old·log2e - m_new·log2e) * O_old
// LAYOUT:
//   INPUT :  S [float x[8]]  accumulator  layout (rS)
//            O [float x[8]]  accumulator  layout (rO)
//   OUTPUT:  P [half2 x[8]]  matrix_a row layout (rP)
// ======================================================================================
template <typename Config, GemmPhase PHASE, bool IS_EVEN, bool IS_DROPOUT, typename FRAG_S, typename FRAG_O, typename FRAG_P, int TILES_S, int TILES_O, int ROWS>
__device__ __forceinline__ void WMMA_GEMM_SOFTMAX(
    FRAG_S  (&rS)        [TILES_S],
    FRAG_O  (&rO)        [TILES_O],
    FRAG_P  (&rP)        [TILES_S],
    float   (&rRowMax)   [ROWS],
    float   (&rRowSum)   [ROWS],
    float   (&rExpDiff)  [ROWS],
    __half* __restrict__ GMEM_MASK,
    int      VALID_Q,
    int      VALID_KV,
    int      GLOBAL_ROW_OFFSET,
    int      GLOBAL_COL_OFFSET,
    int      GLOBAL_N,
    float    P_DROPOUT,
    uint64_t DROPOUT_SEED,
    uint64_t DROPOUT_OFFSET,
    int      STRIDE_GMEM,
    int      BLOCK_ID,
    int      WARP_ID,
    int      LANE_ID
) {
    const float    rp_dropout = IS_DROPOUT ? (1.0f / (1.0f - P_DROPOUT)) : 1.0f;
    const uint32_t drop_thr   = IS_DROPOUT ? static_cast<uint32_t>((1.0f - P_DROPOUT) * 4294967295.0f) : 0u;
    const uint2    lane       = BASELANE<accumulator>[static_cast<unsigned>(LANE_ID)];
    const auto&    offset     = OFFSETS <accumulator>;

    // ======================================================================================
    // Zero fill register fragment
    // ======================================================================================
    if constexpr (!IS_EVEN) {
        if (VALID_Q == 0 || VALID_KV == 0 || WARP_ID * WMMA_M >= VALID_Q) {
            #pragma unroll
            for (int tile_idx = 0; tile_idx < TILES_S; ++tile_idx) {
                #pragma unroll
                for (int i = 0; i < 8; ++i) rP[tile_idx].x[i] = 0u;
            }
            return;
        }
    }

    // ======================================================================================
    // Thread-local row max over all row-tiles
    // ======================================================================================
    float thread_max[ROWS];
    #pragma unroll
    for (int row = 0; row < ROWS; ++row) thread_max[row] = NEG_INF;

    #pragma unroll
    for (int tile_idx = 0; tile_idx < TILES_S; ++tile_idx) {
        if constexpr (!IS_EVEN) {
            if ((tile_idx * WMMA_N) >= VALID_KV) continue;
        }
        const bool is_full     = IS_EVEN || ((tile_idx * WMMA_N) + WMMA_N <= VALID_KV);
        auto&      rS_fragment = rS[tile_idx];
        #pragma unroll
        for (int row  = 0; row < 2; ++row) {
            int   idx = row * 2;
            float s0  = rS_fragment.x[idx + 0];
            float s1  = rS_fragment.x[idx + 1];
            float s4  = rS_fragment.x[idx + 4];
            float s5  = rS_fragment.x[idx + 5];
            if constexpr (!IS_EVEN) {
                const int col0 = (tile_idx * WMMA_N) + lane.y + offset[idx + 0].y;
                const int col1 = (tile_idx * WMMA_N) + lane.y + offset[idx + 1].y;
                const int col4 = (tile_idx * WMMA_N) + lane.y + offset[idx + 4].y;
                const int col5 = (tile_idx * WMMA_N) + lane.y + offset[idx + 5].y;
                if (!is_full) {
                    if (col0 >= VALID_KV) s0 = NEG_INF;
                    if (col1 >= VALID_KV) s1 = NEG_INF;
                    if (col4 >= VALID_KV) s4 = NEG_INF;
                    if (col5 >= VALID_KV) s5 = NEG_INF;
                }
            }
            const float lane_max = fmaxf(fmaxf(s0, s1), fmaxf(s4, s5));
            thread_max[row] = fmaxf(thread_max[row], lane_max);
        }
    }

    // ======================================================================================
    // Thread-local max merge into rRowMax
    // ======================================================================================
    float neg_max[ROWS];
    #pragma unroll
    for (int row = 0; row < ROWS; ++row) {
        float warp_max = thread_max[row];
        warp_max = fmaxf(warp_max, __shfl_xor_sync(0xFFFFFFFFu, warp_max, 2));
        warp_max = fmaxf(warp_max, __shfl_xor_sync(0xFFFFFFFFu, warp_max, 8));
        const float row_max_old = rRowMax[row];
        const float row_max_new = fmaxf(row_max_old, warp_max);
        rRowMax[row] = row_max_new;
        if (row_max_new == NEG_INF) {
            neg_max[row]  = NEG_INF;
            rExpDiff[row] = 0.0f;
        } else {
            neg_max[row]  = __fmul_rn(-row_max_new, LOG2E);
            rExpDiff[row] = exp2f(__fmul_rn(row_max_old - row_max_new, LOG2E));
        }
    }

    // ======================================================================================
    // Compute P = exp2f(S·log2e + neg_max)
    // ======================================================================================
    float thread_sum[ROWS];
    #pragma unroll
    for (int row = 0; row < ROWS; ++row) thread_sum[row] = 0.0f;

    #pragma unroll
    for (int tile_idx = 0; tile_idx < TILES_S; ++tile_idx) {
        if constexpr (!IS_EVEN) {
            if ((tile_idx * WMMA_N) >= VALID_KV) {
                #pragma unroll
                for (int i = 0; i < 8; ++i) rP[tile_idx].x[i] = 0u;
                continue;
            }
        }
        const bool is_full     = IS_EVEN || ((tile_idx * WMMA_N) + WMMA_N <= VALID_KV);
        auto&      rS_fragment = rS[tile_idx];
        uint32_t   b0, b1, b2, b3;
        #pragma unroll
        for (int row = 0; row < 2; ++row) {
            int   idx = row * 2;
            float p0  = exp2f(__fmaf_rn(rS_fragment.x[idx + 0], LOG2E, neg_max[row]));
            float p1  = exp2f(__fmaf_rn(rS_fragment.x[idx + 1], LOG2E, neg_max[row]));
            float p4  = exp2f(__fmaf_rn(rS_fragment.x[idx + 4], LOG2E, neg_max[row]));
            float p5  = exp2f(__fmaf_rn(rS_fragment.x[idx + 5], LOG2E, neg_max[row]));
            if constexpr (!IS_EVEN) {
                const int col0 = (tile_idx * WMMA_N) + lane.y + offset[idx + 0].y;
                const int col1 = (tile_idx * WMMA_N) + lane.y + offset[idx + 1].y;
                const int col4 = (tile_idx * WMMA_N) + lane.y + offset[idx + 4].y;
                const int col5 = (tile_idx * WMMA_N) + lane.y + offset[idx + 5].y;
                if (!is_full) {
                    if (col0 >= VALID_KV) p0 = 0.0f;
                    if (col1 >= VALID_KV) p1 = 0.0f;
                    if (col4 >= VALID_KV) p4 = 0.0f;
                    if (col5 >= VALID_KV) p5 = 0.0f;
                }
            }

            thread_sum[row] += (p0 + p1) + (p4 + p5);

            if constexpr (IS_DROPOUT) {
                const int      row0  = (WARP_ID * WMMA_M)  + lane.x + offset[idx].x;
                const int      col0  = (tile_idx * WMMA_N) + lane.y + offset[idx + 0].y;
                const int      col1  = (tile_idx * WMMA_N) + lane.y + offset[idx + 1].y;
                const int      col4  = (tile_idx * WMMA_N) + lane.y + offset[idx + 4].y;
                const int      col5  = (tile_idx * WMMA_N) + lane.y + offset[idx + 5].y;
                const uint64_t addr  = (uint64_t)(GLOBAL_ROW_OFFSET + row0) * GLOBAL_N + (GLOBAL_COL_OFFSET + col0);
                const uint64_t addr4 = (uint64_t)(GLOBAL_ROW_OFFSET + row0) * GLOBAL_N + (GLOBAL_COL_OFFSET + col4);

                const uint4 rng  = init_philox(DROPOUT_SEED, DROPOUT_OFFSET + (addr  >> 2)).next();
                const uint4 rng4 = init_philox(DROPOUT_SEED, DROPOUT_OFFSET + (addr4 >> 2)).next();

                const uint32_t r0 = ((addr  + 0) & 3) == 0 ? rng.x :  ((addr  + 0) & 3) == 1 ? rng.y :  ((addr  + 0) & 3) == 2 ? rng.z  : rng.w;
                const uint32_t r1 = ((addr  + 1) & 3) == 0 ? rng.x :  ((addr  + 1) & 3) == 1 ? rng.y :  ((addr  + 1) & 3) == 2 ? rng.z  : rng.w;
                const uint32_t r4 = ((addr4 + 0) & 3) == 0 ? rng4.x : ((addr4 + 0) & 3) == 1 ? rng4.y : ((addr4 + 0) & 3) == 2 ? rng4.z : rng4.w;
                const uint32_t r5 = ((addr4 + 1) & 3) == 0 ? rng4.x : ((addr4 + 1) & 3) == 1 ? rng4.y : ((addr4 + 1) & 3) == 2 ? rng4.z : rng4.w;

                const bool k0 = (r0 <= drop_thr);
                const bool k1 = (r1 <= drop_thr);
                const bool k4 = (r4 <= drop_thr);
                const bool k5 = (r5 <= drop_thr);

                p0 = k0 ? __fmul_rn(p0, rp_dropout) : 0.0f;
                p1 = k1 ? __fmul_rn(p1, rp_dropout) : 0.0f;
                p4 = k4 ? __fmul_rn(p4, rp_dropout) : 0.0f;
                p5 = k5 ? __fmul_rn(p5, rp_dropout) : 0.0f;

                if (GMEM_MASK != nullptr) {
                    const ushort gmem0 = 0x3C00 | (k0 ? 0 : 0x8000);
                    const ushort gmem1 = 0x3C00 | (k1 ? 0 : 0x8000);
                    const ushort gmem4 = 0x3C00 | (k4 ? 0 : 0x8000);
                    const ushort gmem5 = 0x3C00 | (k5 ? 0 : 0x8000);

                    if (is_full || col1 < VALID_KV) {
                        *reinterpret_cast<ushort2*>(GMEM_MASK + row0 * STRIDE_GMEM + col0) = make_ushort2(gmem0, gmem1);
                    } else if (col0 < VALID_KV) {
                        *reinterpret_cast<ushort*> (GMEM_MASK + row0 * STRIDE_GMEM + col0) = gmem0;
                    }

                    if (is_full || col5 < VALID_KV) {
                        *reinterpret_cast<ushort2*>(GMEM_MASK + row0 * STRIDE_GMEM + col4) = make_ushort2(gmem4, gmem5);
                    } else if (col4 < VALID_KV) {
                        *reinterpret_cast<ushort*> (GMEM_MASK + row0 * STRIDE_GMEM + col4) = gmem4;
                    }
                }
            }

            const __half2 h2_lo = __float22half2_rn(make_float2(p0, p1));
            const __half2 h2_hi = __float22half2_rn(make_float2(p4, p5));
            if (row == 0) { 
                b0 = *reinterpret_cast<const uint32_t*>(&h2_lo);
                b1 = *reinterpret_cast<const uint32_t*>(&h2_hi);
            } else {
                b2 = *reinterpret_cast<const uint32_t*>(&h2_lo);
                b3 = *reinterpret_cast<const uint32_t*>(&h2_hi);
            }
        }

        // ======================================================================================
        // Transpose: accumulator -> matrix_a row_major
        // ======================================================================================
        //   Emit  :  s[i]    = lane.y & 0x2u ? b[i]     : b[i ^ 2]
        //   Xor   :  v{k}_*  = shfl_xor(s*, k) for k in {2, 8, 10}
        //   Col   :  c0..c3  = lane.y & 0x8u ? {v8,v10} : {s2,v2}
        //            c4..c7  = lane.y & 0x8u ? {s2,v2}  : {v8,v10}
        //   Swap  :  x[2j]   = lane.y & 0x2u ? c[2j+1]  : c[2j]  (j = 0..3)
        //            x[2j+1] = lane.y & 0x2u ? c[2j]    : c[2j+1]
        // ======================================================================================
        const uint32_t s0 = lane.y & 0x2u ? b0 : b2;
        const uint32_t s1 = lane.y & 0x2u ? b1 : b3;
        const uint32_t s2 = lane.y & 0x2u ? b2 : b0;
        const uint32_t s3 = lane.y & 0x2u ? b3 : b1;

        const uint32_t shfl02 = __shfl_xor_sync(0xFFFFFFFFu, s0, 2);
        const uint32_t shfl12 = __shfl_xor_sync(0xFFFFFFFFu, s1, 2);
        const uint32_t shfl08 = __shfl_xor_sync(0xFFFFFFFFu, s2, 8);
        const uint32_t shfl18 = __shfl_xor_sync(0xFFFFFFFFu, s3, 8);
        const uint32_t shfl10 = __shfl_xor_sync(0xFFFFFFFFu, s0, 10);
        const uint32_t shfl11 = __shfl_xor_sync(0xFFFFFFFFu, s1, 10);

        const uint32_t c0 = lane.y & 0x8u ? shfl08 : s2;
        const uint32_t c1 = lane.y & 0x8u ? shfl10 : shfl02;
        const uint32_t c2 = lane.y & 0x8u ? shfl18 : s3;
        const uint32_t c3 = lane.y & 0x8u ? shfl11 : shfl12;
        const uint32_t c4 = lane.y & 0x8u ? s2     : shfl08;
        const uint32_t c5 = lane.y & 0x8u ? shfl02 : shfl10;
        const uint32_t c6 = lane.y & 0x8u ? s3     : shfl18;
        const uint32_t c7 = lane.y & 0x8u ? shfl12 : shfl11;

        auto& P = rP[tile_idx];

        P.x[0] = lane.y & 0x2u ? c1 : c0;
        P.x[1] = lane.y & 0x2u ? c0 : c1;
        P.x[2] = lane.y & 0x2u ? c3 : c2;
        P.x[3] = lane.y & 0x2u ? c2 : c3;
        P.x[4] = lane.y & 0x2u ? c5 : c4;
        P.x[5] = lane.y & 0x2u ? c4 : c5;
        P.x[6] = lane.y & 0x2u ? c7 : c6;
        P.x[7] = lane.y & 0x2u ? c6 : c7;
    }

    // ======================================================================================
    // rO rescale only when running max grew
    // ======================================================================================
    if (BLOCK_ID > 0) {
        bool is_rescale = false;
        #pragma unroll
        for (int row = 0; row < ROWS; ++row) is_rescale |= (rExpDiff[row] != 1.0f);

        if (is_rescale) {
            #pragma unroll
            for (int idx = 0; idx < TILES_O; ++idx) {
                auto& rO_fragment = rO[idx];
                #pragma unroll
                for (int elem = 0; elem < 8; ++elem) {
                    const int row_pair = (elem >> 1) & 1;
                    rO_fragment.x[elem] = __fmul_rn(rO_fragment.x[elem], rExpDiff[row_pair]);
                }
            }
        }
    }

    // ======================================================================================
    // Thread-local sum merge into rRowSum
    // ======================================================================================
    #pragma unroll
    for (int row = 0; row < ROWS; ++row) {
        float warp_sum = thread_sum[row];
        warp_sum += __shfl_xor_sync(0xFFFFFFFFu, warp_sum, 2);
        warp_sum += __shfl_xor_sync(0xFFFFFFFFu, warp_sum, 8);
        rRowSum[row] = __fmaf_rn(rExpDiff[row], rRowSum[row], warp_sum);
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
// LAYOUT:   S [float], dOV [float] in WMMA accumulator fragments
//           P [half ], dS [half ] row-major in SMEM via st_half2(addr, half2, row)
//           element = idx + 4*cq ; idx = 2*rp ; (Δrow, Δcol) = OFFSETS<accumulator>[element]
// ======================================================================================
template<typename Config, GemmPhase PHASE, bool IS_EVEN, bool IS_SOFTCAP, bool IS_DROPOUT, typename FRAG_S, typename FRAG_DOV, int TILES>
__device__ __forceinline__ void WMMA_GEMM_SOFTMAX_GRADIENT(
    FRAG_S       (&rS)        [TILES],
    FRAG_DOV     (&rdOV)      [TILES],
    const float* __restrict__ GMEM_LSE,
    const float* __restrict__ GMEM_ROWDOT,
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
    int      WARP_ID,
    int      LANE_ID
) {

    if constexpr (!IS_EVEN) {
        if (VALID_Q_ROWS == 0 || VALID_KV_ROWS == 0) return;
    }

    using PTraits = PhaseTraits<PHASE>;
    using TTraits = TileTraits<Config, GemmType::Q_KT, PHASE>;

    constexpr bool     WRITE_P     = PTraits::WRITE_P;
    constexpr uint32_t STRIDE      = static_cast<uint32_t>(TTraits::TILE_N);
    constexpr int      num_tiles_n = TTraits::num_tiles_n;
    constexpr int      total_tiles = TTraits::total_mn;
    constexpr bool     early_exit  = (total_tiles % Config::WARPS_PER_BLOCK == 0);

    const   float    softcap_inv   = IS_SOFTCAP ? (1.0f / SOFTCAP) : 0.0f;
    const   float    rp_dropout    = IS_DROPOUT ? (1.0f / (1.0f - P_DROPOUT)) : 1.0f;
    const uint32_t   drop_thr      = IS_DROPOUT ? uint32_t((1.0f - P_DROPOUT) * 4294967295.0f) : 0u;
    const uint32_t   sp_base       = WRITE_P ? __cvta_generic_to_shared(SMEM_P) : 0u;
    const uint32_t   sds_base      = __cvta_generic_to_shared(SMEM_DS);

    const uint2 lane   = BASELANE<accumulator>[static_cast<unsigned>(LANE_ID)];
    const auto& offset = OFFSETS <accumulator>;

    #pragma unroll
    for (int tile_idx = 0; tile_idx < TILES; ++tile_idx) {

        if constexpr (!early_exit) {
            if ((WARP_ID * TILES + tile_idx) >= total_tiles) break;
        }

        const int tile_m = ((WARP_ID * TILES + tile_idx) / num_tiles_n) * WMMA_M;
        const int tile_n = ((WARP_ID * TILES + tile_idx) % num_tiles_n) * WMMA_N;

        if constexpr (!IS_EVEN) {
            if (tile_m >= VALID_Q_ROWS) continue;
        }

        auto& rS_fragment   = rS[tile_idx];
        auto& rdOV_fragment = rdOV[tile_idx];

        #pragma unroll
        for (int row = 0; row < 2; ++row) {
            const int idx = row * 2;
            const int global_row = tile_m + lane.x + offset[idx].x;
            float ds0 = 0.0f, ds1 = 0.0f, ds4 = 0.0f, ds5 = 0.0f;
            float pd0 = 0.0f, pd1 = 0.0f, pd4 = 0.0f, pd5 = 0.0f;

            const int col0 = tile_n + lane.y + offset[idx + 0].y;
            const int col4 = tile_n + lane.y + offset[idx + 4].y;

            bool row_valid = true;
            if constexpr (!IS_EVEN) {
                row_valid = (global_row < VALID_Q_ROWS);
            }

            if (row_valid) {
                const float lse = GMEM_LSE   [global_row];
                const float dot = GMEM_ROWDOT[global_row];

                const float s0 = rS_fragment  .x[idx + 0];
                const float s1 = rS_fragment  .x[idx + 1];
                const float s4 = rS_fragment  .x[idx + 4];
                const float s5 = rS_fragment  .x[idx + 5];
                const float d0 = rdOV_fragment.x[idx + 0];
                const float d1 = rdOV_fragment.x[idx + 1];
                const float d4 = rdOV_fragment.x[idx + 4];
                const float d5 = rdOV_fragment.x[idx + 5];

                float p0 = (s0 == NEG_INF || (s0 - lse) < -80.0f) ? 0.0f : __expf(s0 - lse);
                float p1 = (s1 == NEG_INF || (s1 - lse) < -80.0f) ? 0.0f : __expf(s1 - lse);
                float p4 = (s4 == NEG_INF || (s4 - lse) < -80.0f) ? 0.0f : __expf(s4 - lse);
                float p5 = (s5 == NEG_INF || (s5 - lse) < -80.0f) ? 0.0f : __expf(s5 - lse);

                if constexpr (!IS_EVEN) {
                    const int col1 = tile_n + lane.y + offset[idx + 1].y;
                    const int col5 = tile_n + lane.y + offset[idx + 5].y;
                    if (col0 >= VALID_KV_ROWS) p0 = 0.0f;
                    if (col1 >= VALID_KV_ROWS) p1 = 0.0f;
                    if (col4 >= VALID_KV_ROWS) p4 = 0.0f;
                    if (col5 >= VALID_KV_ROWS) p5 = 0.0f;
                }

                pd0 = p0; pd1 = p1; pd4 = p4; pd5 = p5;

                if constexpr (IS_DROPOUT) {
                    const uint64_t addr  = (uint64_t)(GLOBAL_ROW_OFFSET + global_row) * GLOBAL_N + (GLOBAL_COL_OFFSET + col0);
                    const uint64_t addr4 = (uint64_t)(GLOBAL_ROW_OFFSET + global_row) * GLOBAL_N + (GLOBAL_COL_OFFSET + col4);

                    const uint4 rng  = init_philox(DROPOUT_SEED, DROPOUT_OFFSET + (addr  >> 2)).next();
                    const uint4 rng4 = init_philox(DROPOUT_SEED, DROPOUT_OFFSET + (addr4 >> 2)).next();

                    const uint32_t r0 = ((addr  + 0) & 3) == 0 ? rng.x  : ((addr  + 0) & 3) == 1 ? rng.y  : ((addr  + 0) & 3) == 2 ? rng.z  : rng.w;
                    const uint32_t r1 = ((addr  + 1) & 3) == 0 ? rng.x  : ((addr  + 1) & 3) == 1 ? rng.y  : ((addr  + 1) & 3) == 2 ? rng.z  : rng.w;
                    const uint32_t r4 = ((addr4 + 0) & 3) == 0 ? rng4.x : ((addr4 + 0) & 3) == 1 ? rng4.y : ((addr4 + 0) & 3) == 2 ? rng4.z : rng4.w;
                    const uint32_t r5 = ((addr4 + 1) & 3) == 0 ? rng4.x : ((addr4 + 1) & 3) == 1 ? rng4.y : ((addr4 + 1) & 3) == 2 ? rng4.z : rng4.w;

                    const bool k0 = (r0 <= drop_thr);
                    const bool k1 = (r1 <= drop_thr);
                    const bool k4 = (r4 <= drop_thr);
                    const bool k5 = (r5 <= drop_thr);

                    pd0 = k0 ? __fmul_rn(p0, rp_dropout) : 0.0f;
                    pd1 = k1 ? __fmul_rn(p1, rp_dropout) : 0.0f;
                    pd4 = k4 ? __fmul_rn(p4, rp_dropout) : 0.0f;
                    pd5 = k5 ? __fmul_rn(p5, rp_dropout) : 0.0f;
                }

                ds0 = __fmaf_rn(pd0, d0, -p0 * dot) * SOFTMAX_SCALE;
                ds1 = __fmaf_rn(pd1, d1, -p1 * dot) * SOFTMAX_SCALE;
                ds4 = __fmaf_rn(pd4, d4, -p4 * dot) * SOFTMAX_SCALE;
                ds5 = __fmaf_rn(pd5, d5, -p5 * dot) * SOFTMAX_SCALE;

                if constexpr (IS_SOFTCAP) {
                    if (s0 > NEG_INF) { const float n = __fmul_rn(s0, softcap_inv); ds0 = __fmul_rn(ds0, __fmaf_rn(-n, n, 1.0f)); }
                    if (s1 > NEG_INF) { const float n = __fmul_rn(s1, softcap_inv); ds1 = __fmul_rn(ds1, __fmaf_rn(-n, n, 1.0f)); }
                    if (s4 > NEG_INF) { const float n = __fmul_rn(s4, softcap_inv); ds4 = __fmul_rn(ds4, __fmaf_rn(-n, n, 1.0f)); }
                    if (s5 > NEG_INF) { const float n = __fmul_rn(s5, softcap_inv); ds5 = __fmul_rn(ds5, __fmaf_rn(-n, n, 1.0f)); }
                }
            }

            const uint32_t offset_lo = (global_row * STRIDE + col0) * 2u;
            const uint32_t offset_hi = (global_row * STRIDE + col4) * 2u;

            st_half2(sds_base + offset_lo, __float22half2_rn(make_float2(ds0, ds1)), global_row);
            st_half2(sds_base + offset_hi, __float22half2_rn(make_float2(ds4, ds5)), global_row);

            if constexpr (WRITE_P) {
                st_half2(sp_base + offset_lo, __float22half2_rn(make_float2(pd0, pd1)), global_row);
                st_half2(sp_base + offset_hi, __float22half2_rn(make_float2(pd4, pd5)), global_row);
            }
        }
    }
}
