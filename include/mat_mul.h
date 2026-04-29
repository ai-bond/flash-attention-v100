// ======================================================================================
// * Copyright (c) 2026, D.Skryabin / tg @ai_bond007 SPDX-License: BSD-3-Clause
// ======================================================================================
#pragma once

#include <cuda.h>
#include <cuda_runtime.h>
#include <cuda_fp16.h>

#ifdef MMA_NATIVE
    #include <mma.h>
#elif defined(MMA_884)
    #include "mma_m8n8k4.h"
#else
    #include "mma_m16n16k16.h"
#endif

// ======================================================================================
// WMMA_GEMM_SCORES: Compute C = (A @ B) * scale [+(causal/window/alibi/softcap)]
// Use for: Q@K^T (forward), dO@V^T (backward pre-softmax)
// ======================================================================================
template<typename Config, GemmType TYPE, int D, bool IS_CAUSAL, bool IS_ALIBI, bool IS_SOFTCAP, bool IS_WINDOW,
                                                       int BLOCK_X, int BLOCK_Y, int IN_STRIDE, int OUT_STRIDE>
__device__ __forceinline__ void WMMA_GEMM_SCORES(
    const __half* __restrict__ SMEM_A,
    const __half* __restrict__ SMEM_B,
           float* __restrict__ SMEM_C,
    int VALID_M,   int VALID_N,
    int GLOBAL_M,  int GLOBAL_N,
    float SOFTMAX_SCALE,
    float SOFTCAP, float ALIBI_SLOPE,
    int WIN_L,     int WIN_R,
    int WARP_ID,   int LANE_ID
) {
#ifdef MMA_NATIVE
    using namespace nvcuda::wmma;
#else
    using namespace volta::wmma;
#endif
    constexpr bool APPLY_MASK   = static_cast<uint8_t>(TYPE) & 0x1;
    constexpr bool NEED_MASKING = APPLY_MASK && IS_CAUSAL;
    constexpr bool HAS_FEATURES = IS_ALIBI || IS_SOFTCAP || IS_WINDOW;

    constexpr int WARPS_PER_BLOCK = Config::WARPS_PER_BLOCK;

    constexpr int num_tiles_m = (BLOCK_X + WMMA_M - 1) / WMMA_M;
    constexpr int num_tiles_n = (BLOCK_Y + WMMA_N - 1) / WMMA_N;
    constexpr int num_tiles_k = (D + WMMA_K - 1) / WMMA_K;

    constexpr int total_tiles = num_tiles_m * num_tiles_n;
    constexpr int tiles_per_warp = (total_tiles + WARPS_PER_BLOCK - 1) / WARPS_PER_BLOCK;

    using a_layout = std::conditional_t<static_cast<uint8_t>(TYPE) & 0x2, col_major, row_major>;
    using b_layout = std::conditional_t<static_cast<uint8_t>(TYPE) & 0x4, col_major, row_major>;

    const unsigned row_causal =  (LANE_ID & 0b1) + ((LANE_ID >> 2) & 0b1) * 8 + ((LANE_ID >> 4) & 0b1) * 4;
    const unsigned col_causal = ((LANE_ID >> 1) & 0b1) * 2 + ((LANE_ID >> 3) & 0b1) * 8;

    for (int tile_local = 0; tile_local < tiles_per_warp; ++tile_local) {
        const int tile_idx = WARP_ID * tiles_per_warp + tile_local;
        if (tile_idx >= total_tiles) break;

        const int tile_m_idx = tile_idx / num_tiles_n;
        const int tile_n_idx = tile_idx % num_tiles_n;

        const int tile_m = tile_m_idx * WMMA_M;
        const int tile_n = tile_n_idx * WMMA_N;

        if (tile_m >= VALID_M || tile_n >= VALID_N) continue;

        fragment<matrix_a, WMMA_M, WMMA_N, WMMA_K, half, a_layout> a_frag;
        fragment<matrix_b, WMMA_M, WMMA_N, WMMA_K, half, b_layout> b_frag;
        fragment<accumulator, WMMA_M, WMMA_N, WMMA_K, float> acc_frag;
        fill_fragment(acc_frag, 0.0f);

        #pragma unroll
        for (int k_tile = 0; k_tile < num_tiles_k; ++k_tile) {
            const int k_offset = k_tile * WMMA_K;
            if (k_offset >= D) break;
            load_matrix_sync(a_frag, SMEM_A + tile_m * IN_STRIDE + k_offset, IN_STRIDE);
            load_matrix_sync(b_frag, SMEM_B + tile_n * IN_STRIDE + k_offset, IN_STRIDE);
            mma_sync(acc_frag, a_frag, b_frag, acc_frag);
        }

        if constexpr (NEED_MASKING) {
            if constexpr (!HAS_FEATURES) {
                #pragma unroll
                for (int i = 0; i < acc_frag.num_elements; ++i) {
                    const unsigned col = col_causal + (i & 0b1) + ((i >> 2) & 0b1) * 4;
                    const unsigned row = row_causal + ((i >> 1) & 0b1) * 2;
                    const int global_m = GLOBAL_M + tile_m + row;
                    const int global_n = GLOBAL_N + tile_n + col;
                    const bool in_bounds = (global_m < GLOBAL_M + VALID_M) && (global_n < GLOBAL_N + VALID_N);

                    acc_frag.x[i] = in_bounds ? ((global_n > global_m) ? NEG_INF : acc_frag.x[i] * SOFTMAX_SCALE) : NEG_INF;
                }
            } else {
                const float softcap_rcp = IS_SOFTCAP ? (1.0f / SOFTCAP) : 0.0f;
                #pragma unroll
                for (int i = 0; i < acc_frag.num_elements; ++i) {
                    const unsigned col = col_causal + (i & 0b1) + ((i >> 2) & 0b1) * 4;
                    const unsigned row = row_causal + ((i >> 1) & 0b1) * 2;
                    const int global_m = GLOBAL_M + tile_m + row;
                    const int global_n = GLOBAL_N + tile_n + col;
                    const bool in_bounds = (global_m < GLOBAL_M + VALID_M) && (global_n < GLOBAL_N + VALID_N);
                    bool is_masked = !in_bounds || (global_n > global_m);

                    if constexpr (IS_WINDOW) {
                        is_masked = is_masked || (WIN_L >= 0 && global_n < global_m - WIN_L) || (WIN_R >= 0 && global_n > global_m + WIN_R);
                    }

                    float val = acc_frag.x[i] * SOFTMAX_SCALE;
                    if (!is_masked) {
                        if constexpr (IS_ALIBI) {
                            val = __fmaf_rn(ALIBI_SLOPE, static_cast<float>(global_m - global_n), val);
                        }
                        if constexpr (IS_SOFTCAP) {
                            val = __fmul_rn(SOFTCAP, __tanhf(__fmul_rn(val, softcap_rcp)));
                        }
                    } else {
                        val = NEG_INF;
                    }
                    acc_frag.x[i] = val;
                }
            }
        } else {
            #pragma unroll
            for (int i = 0; i < acc_frag.num_elements; ++i) {
                acc_frag.x[i] *= SOFTMAX_SCALE;
            }
        }
        store_matrix_sync(SMEM_C + tile_m * OUT_STRIDE + tile_n, acc_frag, OUT_STRIDE, mem_row_major);
    }
}

// ======================================================================================
// WMMA_GEMM_GRADIENTS: Compute C += A @ B  (Read-Modify-Write accumulation)
// Use for: P@V, dS@K, P^T@dO, dS^T@Q in backward pass
// ======================================================================================
template<typename Config, GemmType TYPE, int D, int BLOCK_X, int BLOCK_Y, int IN_STRIDE, int OUT_STRIDE>
__device__ __forceinline__ void WMMA_GEMM_GRADIENTS(
    const __half* __restrict__ SMEM_A,
    const __half* __restrict__ SMEM_B,
           float* __restrict__ SMEM_C,
    int VALID_M,
    int VALID_K,
    int WARP_ID,
    int LANE_ID
) {
#ifdef MMA_NATIVE
    using namespace nvcuda::wmma;
#else
    using namespace volta::wmma;
#endif
    constexpr bool A_IS_COL    = static_cast<uint8_t>(TYPE) & 0x2;
    constexpr bool B_IS_COL    = static_cast<uint8_t>(TYPE) & 0x4;

    constexpr int WARPS_PER_BLOCK = Config::WARPS_PER_BLOCK;

    constexpr int num_tiles_m = (BLOCK_X + WMMA_M - 1) / WMMA_M;
    constexpr int num_tiles_n = (D + WMMA_N - 1) / WMMA_N;
    constexpr int num_tiles_k = (BLOCK_Y + WMMA_K - 1) / WMMA_K;

    constexpr int total_tiles = num_tiles_m * num_tiles_n;
    constexpr int tiles_per_warp = (total_tiles + WARPS_PER_BLOCK - 1) / WARPS_PER_BLOCK;

    using a_layout = std::conditional_t<A_IS_COL, col_major, row_major>;
    using b_layout = std::conditional_t<B_IS_COL, col_major, row_major>;

    for (int tile_idx = 0; tile_idx < tiles_per_warp; ++tile_idx) {
        const int global_tile_idx = WARP_ID * tiles_per_warp + tile_idx;
        if (global_tile_idx >= total_tiles) break;

        const int tile_m_idx = global_tile_idx / num_tiles_n;
        const int tile_n_idx = global_tile_idx % num_tiles_n;

        const int tile_m = tile_m_idx * WMMA_M;
        const int tile_n = tile_n_idx * WMMA_N;

        if (tile_m >= VALID_M) continue;

        fragment<matrix_a, WMMA_M, WMMA_N, WMMA_K, half, a_layout> a_frag;
        fragment<matrix_b, WMMA_M, WMMA_N, WMMA_K, half, b_layout> b_frag;
        fragment<accumulator, WMMA_M, WMMA_N, WMMA_K, float> acc_frag;

        load_matrix_sync(acc_frag, SMEM_C + tile_m * OUT_STRIDE + tile_n, OUT_STRIDE, mem_row_major);

        #pragma unroll
        for (int k_tile = 0; k_tile < num_tiles_k; ++k_tile) {
            const int k_offset = k_tile * WMMA_K;
            if (k_offset >= VALID_K) break;

            const __half* a_ptr = A_IS_COL
                ? SMEM_A + k_offset * IN_STRIDE + tile_m
                : SMEM_A + tile_m * IN_STRIDE + k_offset;

            const __half* b_ptr = B_IS_COL
                ? SMEM_B + tile_n * OUT_STRIDE + k_offset
                : SMEM_B + k_offset * OUT_STRIDE + tile_n;

            load_matrix_sync(a_frag, a_ptr, IN_STRIDE);
            load_matrix_sync(b_frag, b_ptr, OUT_STRIDE);

            mma_sync(acc_frag, a_frag, b_frag, acc_frag);
        }
        store_matrix_sync(SMEM_C + tile_m * OUT_STRIDE + tile_n, acc_frag, OUT_STRIDE, mem_row_major);
    }
}
