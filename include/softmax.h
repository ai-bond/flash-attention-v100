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
// FA2 MATH: m_new  = max(m_old, rowmax(S))
//           P      = exp2(S·log2e - m_new·log2e)
//           l_new  = exp2(m_old·log2e - m_new·log2e) * l_old + rowsum(P)
//           O_new  = exp2(m_old·log2e - m_new·log2e) * O_old
// LAYOUT:   S [float], O [float] in WMMA accumulator fragments;
//           P [half ] row-major in SMEM, written via st_half2(addr, half2, row);
//           m / l / exp_diff thread-local across the 4 lanes of a row.
// ======================================================================================
template<typename Config, GemmPhase PHASE, bool IS_EVEN, bool IS_DROPOUT, typename FRAG_S, typename FRAG_O, int TILES_S, int TILES_O, int ROWS>
__device__ __forceinline__ void WMMA_GEMM_SOFTMAX(
    FRAG_S (&rS)      [TILES_S],
    FRAG_O (&rO)      [TILES_O],
    float  (&rRowMax )[ROWS],
    float  (&rRowSum )[ROWS],
    float  (&rExpDiff)[ROWS],
         __half* __restrict__ SMEM_P,
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
    if constexpr (!IS_EVEN) {
        if (VALID_Q == 0 || VALID_KV == 0) return;
        if (WARP_ID * WMMA_M >= VALID_Q) return;
    }

    using PTraits        = PhaseTraits<PHASE>;
    constexpr int STRIDE = PTraits::template STRIDE<Config>();
    const uint2 lines    = LINE_CHORDS(static_cast<unsigned>(LANE_ID));

    // ============================================================
    // Per-iteration max -> thread-local over all row-tiles
    // ============================================================
    float iter_max[ROWS];
    #pragma unroll
    for (int row = 0; row < ROWS; ++row) iter_max[row] = NEG_INF;

    #pragma unroll
    for (int tile_idx = 0; tile_idx < TILES_S; ++tile_idx) {

        if constexpr (!IS_EVEN) {
            if ((tile_idx * WMMA_N) >= VALID_KV) continue;
        }

        const bool is_full      = IS_EVEN || ((tile_idx * WMMA_N) + WMMA_N <= VALID_KV);
        const int  column_base  = (tile_idx * WMMA_N) + lines.y;
        const int  column_base4 = column_base + 4;
        auto&      rS_fragment  = rS[tile_idx];

        #pragma unroll
        for (int row_pair = 0; row_pair < 2; ++row_pair) {
            float value_0, value_1, value_4, value_5;
            if (is_full) {
                value_0 = rS_fragment.x[row_pair*2];
                value_1 = rS_fragment.x[row_pair*2+1];
                value_4 = rS_fragment.x[row_pair*2+4];
                value_5 = rS_fragment.x[row_pair*2+5];
            } else {
                value_0 = (column_base      < VALID_KV) ? rS_fragment.x[row_pair*2]   : NEG_INF;
                value_1 = (column_base  + 1 < VALID_KV) ? rS_fragment.x[row_pair*2+1] : NEG_INF;
                value_4 = (column_base4     < VALID_KV) ? rS_fragment.x[row_pair*2+4] : NEG_INF;
                value_5 = (column_base4 + 1 < VALID_KV) ? rS_fragment.x[row_pair*2+5] : NEG_INF;
            }

            const float lane_max = fmaxf(fmaxf(value_0, value_1), fmaxf(value_4, value_5));
            iter_max[row_pair] = fmaxf(iter_max[row_pair], lane_max);
        }
    }

    // ============================================================
    // Warp-reduce per row -> merge into rRowMax
    // ============================================================
    float neg_max[ROWS];
    #pragma unroll
    for (int row = 0; row < ROWS; ++row) {

        float warp_max = iter_max[row];

        warp_max = fmaxf(warp_max, __shfl_xor_sync(0xFFFFFFFFu, warp_max, 2));
        warp_max = fmaxf(warp_max, __shfl_xor_sync(0xFFFFFFFFu, warp_max, 8));

        const float row_max_old = rRowMax[row];
        const float row_max_new = fmaxf(row_max_old, warp_max);

        rRowMax[row] = row_max_new;

        if (row_max_new == NEG_INF) {
            neg_max[row]  = NEG_INF;
            rExpDiff[row] = 0.0f;
        } else {
            neg_max[row]  = -row_max_new * LOG2E;
            rExpDiff[row] = exp2f((row_max_old - row_max_new) * LOG2E);
        }
    }

    // ============================================================
    // P = exp2f(S·log2e - nm·log2e), P -> SMEM, thread-local sum
    // ============================================================
    const uint32_t sp_base    = __cvta_generic_to_shared(SMEM_P);
    const float    rp_dropout = IS_DROPOUT ? (1.0f / (1.0f - P_DROPOUT)) : 1.0f;
    const uint32_t drop_thr   = IS_DROPOUT ? static_cast<uint32_t>((1.0f - P_DROPOUT) * 4294967295.0f) : 0u;

    float iter_sum[ROWS];
    #pragma unroll
    for (int row = 0; row < ROWS; ++row) iter_sum[row] = 0.0f;

    #pragma unroll
    for (int tile_idx = 0; tile_idx < TILES_S; ++tile_idx) {

        if constexpr (!IS_EVEN) {
            if ((tile_idx * WMMA_N) >= VALID_KV) continue;
        }

        const bool is_full      = IS_EVEN || ((tile_idx * WMMA_N) + WMMA_N <= VALID_KV);
        const int  column_base  = (tile_idx * WMMA_N) + lines.y;
        const int  column_base4 = column_base + 4;
        auto&      rS_fragment  = rS[tile_idx];

        #pragma unroll
        for (int row_pair = 0; row_pair < 2; ++row_pair) {
            const int   global_row = (WARP_ID * WMMA_M) + lines.x + row_pair * 2;
            const float nm_log2e   = neg_max[row_pair];

            float p0 = exp2f(fmaf(rS_fragment.x[row_pair*2],   LOG2E, nm_log2e));
            float p1 = exp2f(fmaf(rS_fragment.x[row_pair*2+1], LOG2E, nm_log2e));
            float p4 = exp2f(fmaf(rS_fragment.x[row_pair*2+4], LOG2E, nm_log2e));
            float p5 = exp2f(fmaf(rS_fragment.x[row_pair*2+5], LOG2E, nm_log2e));

            if (!is_full) {
                if (column_base      >= VALID_KV) p0 = 0.0f;
                if (column_base  + 1 >= VALID_KV) p1 = 0.0f;
                if (column_base4     >= VALID_KV) p4 = 0.0f;
                if (column_base4 + 1 >= VALID_KV) p5 = 0.0f;
            }

            iter_sum[row_pair] += (p0 + p1) + (p4 + p5);

            if constexpr (IS_DROPOUT) {
                const uint64_t dropout_addr = (uint64_t)(GLOBAL_ROW_OFFSET + global_row) * GLOBAL_N + (GLOBAL_COL_OFFSET + column_base);
                const uint4    rng0         = init_philox(DROPOUT_SEED, DROPOUT_OFFSET + (dropout_addr >> 2)).next();

                const uint32_t rand0 = ((dropout_addr + 0) & 3) == 0 ? rng0.x : ((dropout_addr + 0) & 3) == 1 ? rng0.y : ((dropout_addr + 0) & 3) == 2 ? rng0.z : rng0.w;
                const uint32_t rand1 = ((dropout_addr + 1) & 3) == 0 ? rng0.x : ((dropout_addr + 1) & 3) == 1 ? rng0.y : ((dropout_addr + 1) & 3) == 2 ? rng0.z : rng0.w;

                const bool keep0 = (rand0 <= drop_thr);
                const bool keep1 = (rand1 <= drop_thr);
                p0 = keep0 ? (p0 * rp_dropout) : 0.0f;
                p1 = keep1 ? (p1 * rp_dropout) : 0.0f;

                if (GMEM_MASK != nullptr) {
                    const ushort   mask0     = 0x3C00 | (keep0 ? 0 : 0x8000);
                    const ushort   mask1     = 0x3C00 | (keep1 ? 0 : 0x8000);
                    const uint64_t gmem_addr = __cvta_generic_to_global(GMEM_MASK + global_row * STRIDE_GMEM + column_base);

                    if (is_full || column_base + 1 < VALID_KV) {
                        asm volatile("st.global.v2.u16 [%0], {%1, %2};\n" :: "l"(gmem_addr), "h"(mask0), "h"(mask1) : "memory");
                    } else if (column_base < VALID_KV) {
                        asm volatile("st.global.u16 [%0], %1;\n" :: "l"(gmem_addr), "h"(mask0) : "memory");
                    }
                }

                const uint64_t dropout_addr4 = (uint64_t)(GLOBAL_ROW_OFFSET + global_row) * GLOBAL_N + (GLOBAL_COL_OFFSET + column_base4);
                const uint4    rng4          = init_philox(DROPOUT_SEED, DROPOUT_OFFSET + (dropout_addr4 >> 2)).next();

                const uint32_t rand4 = ((dropout_addr4 + 0) & 3) == 0 ? rng4.x : ((dropout_addr4 + 0) & 3) == 1 ? rng4.y : ((dropout_addr4 + 0) & 3) == 2 ? rng4.z : rng4.w;
                const uint32_t rand5 = ((dropout_addr4 + 1) & 3) == 0 ? rng4.x : ((dropout_addr4 + 1) & 3) == 1 ? rng4.y : ((dropout_addr4 + 1) & 3) == 2 ? rng4.z : rng4.w;

                const bool keep4 = (rand4 <= drop_thr);
                const bool keep5 = (rand5 <= drop_thr);
                p4 = keep4 ? (p4 * rp_dropout) : 0.0f;
                p5 = keep5 ? (p5 * rp_dropout) : 0.0f;

                if (GMEM_MASK != nullptr) {
                    const ushort   mask4     = 0x3C00 | (keep4 ? 0 : 0x8000);
                    const ushort   mask5     = 0x3C00 | (keep5 ? 0 : 0x8000);
                    const uint64_t gmem_addr = __cvta_generic_to_global(GMEM_MASK + global_row * STRIDE_GMEM + column_base4);

                    if (is_full || column_base4 + 1 < VALID_KV) {
                        asm volatile("st.global.v2.u16 [%0], {%1, %2};\n" :: "l"(gmem_addr), "h"(mask4), "h"(mask5) : "memory");
                    } else if (column_base4 < VALID_KV) {
                        asm volatile("st.global.u16 [%0], %1;\n" :: "l"(gmem_addr), "h"(mask4) : "memory");
                    }
                }
            }

            const uint32_t sp_row = sp_base + (uint32_t)global_row * (STRIDE * 2);

            st_half2(sp_row + (uint32_t)column_base  * 2, __float22half2_rn(make_float2(p0, p1)), global_row);
            st_half2(sp_row + (uint32_t)column_base4 * 2, __float22half2_rn(make_float2(p4, p5)), global_row);
        }
    }

    // ============================================================
    // rO rescale -> only when running max grew
    // ============================================================
    if (BLOCK_ID > 0) {
        bool is_full = false;

        #pragma unroll
        for (int row = 0; row < ROWS; ++row) {
            is_full |= (rExpDiff[row] != 1.0f);
        }

        if (is_full) {
            #pragma unroll
            for (int idx = 0; idx < TILES_O; ++idx) {
                auto& rO_fragment = rO[idx];
                #pragma unroll
                for (int elem = 0; elem < 8; ++elem) {
                    const int row_pair = (elem >> 1) & 1;
                    rO_fragment.x[elem] *= rExpDiff[row_pair];
                }
            }
        }
    }

    // ============================================================
    // Single warp-reduce of row sums -> merge into rRowSum
    // ============================================================
    #pragma unroll
    for (int row = 0; row < ROWS; ++row) {

        float warp_sum = iter_sum[row];

        warp_sum += __shfl_xor_sync(0xFFFFFFFFu, warp_sum, 2);
        warp_sum += __shfl_xor_sync(0xFFFFFFFFu, warp_sum, 8);

        rRowSum[row] = rExpDiff[row] * rRowSum[row] + warp_sum;
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
//           pair p (0..3), e = 2p; row += (p & 1) * 2, col += (p >> 1) * 4
// ======================================================================================
template<typename Config, GemmPhase PHASE,  bool IS_EVEN, bool IS_SOFTCAP, bool IS_DROPOUT, typename FRAG_S, typename FRAG_DOV, int TILES>
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

    using PTraits               = PhaseTraits<PHASE>;
    constexpr bool WRITE_P      = PTraits::WRITE_P;
    constexpr uint32_t STRIDE   = PTraits::template STRIDE<Config>();
    constexpr int  TILE_M       = PTraits::template BLOCK_X<Config>();
    constexpr int  TILE_N       = PTraits::template BLOCK_Y<Config>();
    constexpr int  num_tiles_m  = (TILE_M + WMMA_M - 1) / WMMA_M;
    constexpr int  num_tiles_n  = (TILE_N + WMMA_N - 1) / WMMA_N;
    constexpr int  total_tiles  = num_tiles_m * num_tiles_n;
    constexpr bool early_exit   = (total_tiles % Config::WARPS_PER_BLOCK == 0);

    const   float softcap_inv   = IS_SOFTCAP ? (1.0f / SOFTCAP) : 0.0f;
    const   float rp_dropout    = IS_DROPOUT ? (1.0f / (1.0f - P_DROPOUT)) : 1.0f;
    const uint32_t drop_thr     = IS_DROPOUT ? uint32_t((1.0f - P_DROPOUT) * 4294967295.0f) : 0u;
    const uint32_t sp_base      = WRITE_P ? __cvta_generic_to_shared(SMEM_P) : 0u;
    const uint32_t sds_base     = __cvta_generic_to_shared(SMEM_DS);

    const uint2    lines        = LINE_CHORDS(LANE_ID);

    #pragma unroll
    for (int tile_index = 0; tile_index < TILES; ++tile_index) {

        if constexpr (!early_exit) {
            if ((WARP_ID * TILES + tile_index) >= total_tiles) break;
        }

        const int tile_m = ((WARP_ID * TILES + tile_index) / num_tiles_n) * WMMA_M;
        const int tile_n = ((WARP_ID * TILES + tile_index) % num_tiles_n) * WMMA_N;

        if constexpr (!IS_EVEN) {
            if (tile_m >= VALID_Q_ROWS) continue;
        }

        auto& rS_fragment   = rS[tile_index];
        auto& rdOV_fragment = rdOV[tile_index];

        #pragma unroll
        for (int pair_index = 0; pair_index < 4; ++pair_index) {
            const int element  = pair_index * 2;
            const int row_pair = pair_index & 1;
            const int col_quad = pair_index >> 1;

            const int global_row = tile_m + int(lines.x) + row_pair * 2;
            const int global_col = tile_n + int(lines.y) + col_quad * 4;

            if constexpr (!IS_EVEN) {
                if (global_row >= VALID_Q_ROWS || global_col >= VALID_KV_ROWS) {
                    st_half2(sds_base + ((global_row * STRIDE + global_col) * 2u), __float22half2_rn(make_float2(0.0f, 0.0f)), global_row);
                    if constexpr (WRITE_P)
                        st_half2(sp_base + ((global_row * STRIDE + global_col) * 2u), __float22half2_rn(make_float2(0.0f, 0.0f)), global_row);
                    continue;
                }
            }

            const float lse = GMEM_LSE   [global_row];
            const float dot = GMEM_ROWDOT[global_row];

            const float s0 = rS_fragment.x[element];
            const float s1 = rS_fragment.x[element + 1];
            const float d0 = rdOV_fragment.x[element];
            const float d1 = rdOV_fragment.x[element + 1];

            float p0 = (s0 == NEG_INF || (s0 - lse) < -80.0f) ? 0.0f : __expf(s0 - lse);
            float p1 = (s1 == NEG_INF || (s1 - lse) < -80.0f) ? 0.0f : __expf(s1 - lse);

            float pd0 = p0;
            float pd1 = p1;

            if constexpr (IS_DROPOUT) {
                const uint32_t dropout_addr = (GLOBAL_ROW_OFFSET + global_row) * GLOBAL_N + (GLOBAL_COL_OFFSET + global_col);
                const uint4 rng = init_philox(DROPOUT_SEED, DROPOUT_OFFSET + (dropout_addr >> 2)).next();
                const int c0 = global_col & 3;
                const int c1 = (global_col + 1) & 3;
                const uint32_t r0 = (c0 == 0) ? rng.x : (c0 == 1) ? rng.y : (c0 == 2) ? rng.z : rng.w;
                const uint32_t r1 = (c1 == 0) ? rng.x : (c1 == 1) ? rng.y : (c1 == 2) ? rng.z : rng.w;
                pd0 = (r0 <= drop_thr) ? (p0 * rp_dropout) : 0.0f;
                pd1 = (r1 <= drop_thr) ? (p1 * rp_dropout) : 0.0f;
            }

            float ds0 = __fmaf_rn(pd0, d0, -p0 * dot) * SOFTMAX_SCALE;
            float ds1 = __fmaf_rn(pd1, d1, -p1 * dot) * SOFTMAX_SCALE;

            if constexpr (IS_SOFTCAP) {
                if (s0 > NEG_INF) { const float n = __fmul_rn(s0, softcap_inv); ds0 = __fmul_rn(ds0, __fmaf_rn(-n, n, 1.0f)); }
                if (s1 > NEG_INF) { const float n = __fmul_rn(s1, softcap_inv); ds1 = __fmul_rn(ds1, __fmaf_rn(-n, n, 1.0f)); }
            }

            const uint32_t offset  = (global_row * STRIDE + global_col) * 2u;
            const uint32_t ds_addr = sds_base + offset;
            st_half2(ds_addr, __float22half2_rn(make_float2(ds0, ds1)), global_row);

            if constexpr (WRITE_P) {
                const uint32_t p_addr = sp_base + offset;
                st_half2(p_addr, __float22half2_rn(make_float2(pd0, pd1)), global_row);
            }
        }
    }
}
