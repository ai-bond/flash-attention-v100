// ======================================================================================
// * Copyright (c) 2026, D.Skryabin / tg @ai_bond007 SPDX-License: BSD-3-Clause
// ======================================================================================
#pragma once

#include <cuda.h>
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <array>
#include <type_traits>
#include "traits.h"

#ifdef MMA_NATIVE
    #include <mma.h>
    using namespace nvcuda::wmma;
#elif defined(MMA_884)
    #include "mma_m8n8k4.h"
    using namespace volta::wmma;
#else
    #include "mma_m16n16k16.h"
    using namespace volta::wmma;
#endif

// ======================================================================================
// BASELANE<Use, Layout>[lane_id] : per-lane base coordinate
// ======================================================================================
template <typename Use, typename Layout = void>
__device__ constexpr std::array<uint2, 32> BASELANE = []() constexpr {
    std::array<uint2, 32> r{};
    for (unsigned lid = 0; lid < 32; ++lid) {
        if constexpr (std::is_same_v<Use, accumulator>) {
            r[lid] = uint2{((lid >> 2) & 1u) * 8u + ((lid >> 4) & 1u) * 4u + (lid & 1u), lid & 10u};
        } else if constexpr (std::is_same_v<Use, matrix_a> && std::is_same_v<Layout, row_major>) {
            r[lid] = uint2{(lid & 3u) | ((lid >> 4) & 1u) * 4u | ((lid >> 2) & 1u) * 8u, 0u};
        } else if constexpr (std::is_same_v<Use, matrix_a> && std::is_same_v<Layout, col_major>) {
            r[lid] = uint2{((lid >> 2) & 1u) * 8u + ((lid >> 4) & 1u) * 4u, lid & 3u};
        } else if constexpr (std::is_same_v<Use, matrix_b> && std::is_same_v<Layout, row_major>) {
            r[lid] = uint2{lid & 3u, ((lid >> 3) & 1u) * 8u + ((lid >> 4) & 1u) * 4u};
        } else if constexpr (std::is_same_v<Use, matrix_b> && std::is_same_v<Layout, col_major>) {
            r[lid] = uint2{0u, (lid & 3u) | (lid & 8u) | ((lid >> 2) & 4u)};
        }
    }
    return r;
}();

// ======================================================================================
// OFFSETS <Use, Layout>[reg_idx] : per-register offset
// ======================================================================================
template <typename Use, typename Layout = void>
__device__ constexpr std::array<uint2, 8> OFFSETS = []() constexpr {
    std::array<uint2, 8> r{};
    if constexpr (std::is_same_v<Use, accumulator>) {
        r = {{ {0,0}, {0,1}, {2,0}, {2,1}, {0,4}, {0,5}, {2,4}, {2,5} }};
    } else if constexpr (std::is_same_v<Use, matrix_a> && std::is_same_v<Layout, row_major>) {
        r = {{ {0,0}, {0,2}, {0,4}, {0,6}, {0,8}, {0,10}, {0,12}, {0,14} }};
    } else if constexpr (std::is_same_v<Use, matrix_a> && std::is_same_v<Layout, col_major>) {
        r = {{ {0,0}, {2,0}, {4,0}, {6,0}, {8,0}, {10,0}, {12,0}, {14,0} }};
    } else if constexpr (std::is_same_v<Use, matrix_b> && std::is_same_v<Layout, col_major>) {
        r = {{ {0,0}, {2,0}, {4,0}, {6,0}, {8,0}, {10,0}, {12,0}, {14,0} }};
    } else if constexpr (std::is_same_v<Use, matrix_b> && std::is_same_v<Layout, row_major>) {
        r = {{ {0,0}, {0,2}, {8,0}, {8,2}, {0,4}, {0,6}, {8,4}, {8,6} }};
    } else {
        static_assert(sizeof(Use) == 0, "OFFSETS: unsupported Use/Layout");
    }
    return r;
}();

// ======================================================================================
// WMMA_GEMM_SCORES: Compute (A @ B) * scale [+(causal/window/alibi/softcap)]
// Use for: Q@K^T (forward), dO@V^T (backward pre-softmax)
// ======================================================================================
template<typename Config, GemmType TYPE, GemmPhase PHASE, bool IS_EVEN, bool IS_CAUSAL, bool IS_ALIBI, bool IS_SOFTCAP, bool IS_WINDOW, typename Fragment, int TILES>
__device__ __forceinline__ void WMMA_GEMM_SCORES(
    const __half* __restrict__ SMEM_A,
    const __half* __restrict__ SMEM_B,
         Fragment (&registers) [TILES],
    int   VALID_M,   int   VALID_N,
    int   GLOBAL_M,  int   GLOBAL_N,
    int   SEQLEN,    float SOFTMAX_SCALE,
    float SOFTCAP,   float ALIBI_SLOPE,
    int   WIN_L,     int   WIN_R,
    int   WARP_ID,   int   LANE_ID
) {
#ifdef MMA_NATIVE
    using namespace nvcuda::wmma;
#else
    using namespace volta::wmma;
#endif
    using GTraits = GemmTraits<TYPE, PHASE>;
    using TTraits = TileTraits<Config, TYPE, PHASE>;

    constexpr bool APPLY_MASK       = GTraits::APPLY_MASK;
    constexpr bool HAS_FEATURES     =  IS_ALIBI  || IS_SOFTCAP || IS_WINDOW;
    const     int  SEQLEN_OFFSET    = (IS_CAUSAL || IS_WINDOW  || IS_ALIBI) ? SEQLEN : 0;
    constexpr int  A_STRIDE         = GTraits::A_IS_COL ? TTraits::TILE_M : TTraits::TILE_K;
    constexpr int  B_STRIDE         = GTraits::B_IS_COL ? TTraits::TILE_K : TTraits::TILE_N;
    constexpr int  num_tiles_n      = TTraits::num_tiles_n;
    constexpr int  num_tiles_k      = TTraits::num_tiles_k;
    constexpr int  total_tiles      = TTraits::total_mn;
    constexpr bool early_exit       = (total_tiles % Config::WARPS_PER_BLOCK == 0);

    using a_layout = std::conditional_t<GTraits::A_IS_COL, col_major, row_major>;
    using b_layout = std::conditional_t<GTraits::B_IS_COL, col_major, row_major>;

    const uint2 lane   = BASELANE<accumulator>[static_cast<unsigned>(LANE_ID)];
    const auto& offset = OFFSETS <accumulator>;

    #pragma unroll
    for (int tile_idx = 0; tile_idx < TILES; ++tile_idx) {
        if constexpr (!early_exit) {
            if ((WARP_ID * TILES + tile_idx) >= total_tiles) break;
        }
        const int tile_m_idx = (WARP_ID * TILES + tile_idx) / num_tiles_n;
        const int tile_n_idx = (WARP_ID * TILES + tile_idx) % num_tiles_n;
        const int tile_m     = tile_m_idx * WMMA_M;
        const int tile_n     = tile_n_idx * WMMA_N;
        if constexpr (!IS_EVEN) {
            if (tile_m >= VALID_M || tile_n >= VALID_N) continue;
        }
        auto& acc_frag = registers[tile_idx];
        fragment<matrix_a, WMMA_M, WMMA_N, WMMA_K, half, a_layout> a_frag;
        fragment<matrix_b, WMMA_M, WMMA_N, WMMA_K, half, b_layout> b_frag;
        fill_fragment(acc_frag, 0.0f);
        #pragma unroll
        for (int k_tile = 0; k_tile < num_tiles_k; ++k_tile) {
            const int k_offset = k_tile * WMMA_K;
            const __half* a_ptr = GTraits::A_IS_COL ? SMEM_A + k_offset * A_STRIDE + tile_m   : SMEM_A + tile_m   * A_STRIDE + k_offset;
            const __half* b_ptr = GTraits::B_IS_COL ? SMEM_B + tile_n   * B_STRIDE + k_offset : SMEM_B + k_offset * B_STRIDE + tile_n;
            load_matrix_sync(a_frag, a_ptr, A_STRIDE);
            load_matrix_sync(b_frag, b_ptr, B_STRIDE);
            mma_sync(acc_frag, a_frag, b_frag, acc_frag);
        }
        if constexpr (APPLY_MASK && IS_CAUSAL) {
            if constexpr (!HAS_FEATURES) {
                #pragma unroll
                for (int i = 0; i < acc_frag.num_elements; ++i) {
                    const int   global_m = GLOBAL_M + tile_m + lane.x + offset[i].x;
                    const int   global_n = GLOBAL_N + tile_n + lane.y + offset[i].y;
                    if constexpr (IS_EVEN) {
                        acc_frag.x[i] = ((global_n - SEQLEN_OFFSET) > global_m) ? NEG_INF : acc_frag.x[i] * SOFTMAX_SCALE;
                    } else {
                        const bool in_bounds = (global_m < GLOBAL_M + VALID_M) && (global_n < GLOBAL_N + VALID_N);
                        acc_frag.x[i] = in_bounds ? (((global_n - SEQLEN_OFFSET) > global_m) ? NEG_INF : acc_frag.x[i] * SOFTMAX_SCALE) : NEG_INF;
                    }
                }
            } else {
                const float softcap_rcp = IS_SOFTCAP ? (1.0f / SOFTCAP) : 0.0f;
                #pragma unroll
                for (int i = 0; i < acc_frag.num_elements; ++i) {
                    const int   global_m = GLOBAL_M + tile_m + lane.x + offset[i].x;
                    const int   global_n = GLOBAL_N + tile_n + lane.y + offset[i].y;
                    bool is_masked;
                    if constexpr (IS_EVEN) {
                        is_masked = ((global_n - SEQLEN_OFFSET) > global_m);
                    } else {
                        const bool in_bounds = (global_m < GLOBAL_M + VALID_M) && (global_n < GLOBAL_N + VALID_N);
                        is_masked = !in_bounds || ((global_n - SEQLEN_OFFSET) > global_m);
                    }
                    if constexpr (IS_WINDOW) {
                        is_masked = is_masked || (WIN_L >= 0 && (global_n - SEQLEN_OFFSET) < global_m - WIN_L)
                                              || (WIN_R >= 0 && (global_n - SEQLEN_OFFSET) > global_m + WIN_R);
                    }
                    float val = acc_frag.x[i] * SOFTMAX_SCALE;
                    if (!is_masked) {
                        if constexpr (IS_ALIBI) {
                            val = __fmaf_rn(-ALIBI_SLOPE, fabsf(static_cast<float>(global_m - (global_n - SEQLEN_OFFSET))), val);
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
        } else if constexpr (APPLY_MASK && !IS_CAUSAL && HAS_FEATURES) {
            const float softcap_rcp = IS_SOFTCAP ? (1.0f / SOFTCAP) : 0.0f;
            #pragma unroll
            for (int i = 0; i < acc_frag.num_elements; ++i) {
                const int   global_m = GLOBAL_M + tile_m + lane.x + offset[i].x;
                const int   global_n = GLOBAL_N + tile_n + lane.y + offset[i].y;
                bool is_masked;
                if constexpr (IS_EVEN) {
                    is_masked = false;
                } else {
                    const bool in_bounds = (global_m < GLOBAL_M + VALID_M) && (global_n < GLOBAL_N + VALID_N);
                    is_masked = !in_bounds;
                }
                if constexpr (IS_WINDOW) {
                    is_masked = is_masked || (WIN_L >= 0 && (global_n - SEQLEN_OFFSET) < global_m - WIN_L)
                                          || (WIN_R >= 0 && (global_n - SEQLEN_OFFSET) > global_m + WIN_R);
                }
                float val = acc_frag.x[i] * SOFTMAX_SCALE;
                if (!is_masked) {
                    if constexpr (IS_ALIBI) {
                        val = __fmaf_rn(-ALIBI_SLOPE, fabsf(static_cast<float>(global_m - (global_n - SEQLEN_OFFSET))), val);
                    }
                    if constexpr (IS_SOFTCAP) {
                        val = __fmul_rn(SOFTCAP, __tanhf(__fmul_rn(val, softcap_rcp)));
                    }
                } else {
                    val = NEG_INF;
                }
                acc_frag.x[i] = val;
            }
        } else {
            #pragma unroll
            for (int i = 0; i < acc_frag.num_elements; ++i) {
                acc_frag.x[i] *= SOFTMAX_SCALE;
            }
        }
    }
}

// ======================================================================================
// WMMA_GEMM_GRADIENTS: Compute C += A @ B  (Read-Modify-Write accumulation)
// Use for: P@V (forward), dS@K, P^T@dO, dS^T@Q (backward)
// ======================================================================================
template<typename Config, GemmType TYPE, GemmPhase PHASE, bool IS_EVEN, typename RTYPE, typename Fragment, int TILES>
__device__ __forceinline__ void WMMA_GEMM_GRADIENTS(
    RTYPE&&                    SMEM_A,
    const __half* __restrict__ SMEM_B,
         Fragment (&registers) [TILES],
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
    using GTraits = GemmTraits<TYPE, PHASE>;
    using PTraits = PhaseTraits<PHASE>;
    using TTraits = TileTraits<Config, TYPE, PHASE>;

    constexpr bool IS_REG      = (static_cast<int>(std::extent_v<std::remove_reference_t<RTYPE>>) > 0);
    constexpr int  A_STRIDE    = GTraits::A_IS_COL ? (TTraits::TILE_M) : (TTraits::TILE_K);
    constexpr int  B_STRIDE    = GTraits::B_IS_COL ? PTraits::template BLOCK_Y<Config>() : Config::DIM;
    constexpr int  num_tiles_n = TTraits::num_tiles_n;
    constexpr int  num_tiles_k = IS_REG ? static_cast<int>(std::extent_v<std::remove_reference_t<RTYPE>>) : TTraits::num_tiles_k;
    constexpr bool early_exit  = (TTraits::total_mn % Config::WARPS_PER_BLOCK == 0);

    using a_layout = std::conditional_t<GTraits::A_IS_COL, col_major, row_major>;
    using b_layout = std::conditional_t<GTraits::B_IS_COL, col_major, row_major>;

    #pragma unroll
    for (int tile_idx = 0; tile_idx < TILES; ++tile_idx) {
        if constexpr (!early_exit) {
            if ((WARP_ID * TILES + tile_idx) >= TTraits::total_mn) break;
        }
        const int tile_m_idx = (WARP_ID * TILES + tile_idx) / num_tiles_n;
        const int tile_n_idx = (WARP_ID * TILES + tile_idx) % num_tiles_n;
        const int tile_m     = tile_m_idx * WMMA_M;
        const int tile_n     = tile_n_idx * WMMA_N;
        if constexpr (!IS_EVEN) {
            if (tile_m >= VALID_M) continue;
        }
        auto& acc_frag = registers[tile_idx];
        #pragma unroll
        for (int k_tile = 0; k_tile < num_tiles_k; ++k_tile) {
            const int k_offset = k_tile * WMMA_K;
            if constexpr (!IS_EVEN) {
                if (k_offset >= VALID_K) break;
            }
             if constexpr (IS_REG) {
                const __half* b_ptr = GTraits::B_IS_COL ? SMEM_B + tile_n   * B_STRIDE + k_offset : SMEM_B + k_offset * B_STRIDE + tile_n;
                fragment<matrix_b, WMMA_M, WMMA_N, WMMA_K, half, b_layout> b_frag;
                load_matrix_sync(b_frag, b_ptr, B_STRIDE);
                mma_sync(acc_frag, SMEM_A[k_tile], b_frag, acc_frag);
            } else {
                const __half* b_ptr = GTraits::B_IS_COL ? SMEM_B + tile_n   * B_STRIDE + k_offset : SMEM_B + k_offset * B_STRIDE + tile_n;
                const __half* a_ptr = GTraits::A_IS_COL ? SMEM_A + k_offset * A_STRIDE + tile_m   : SMEM_A + tile_m   * A_STRIDE + k_offset;
                fragment<matrix_a, WMMA_M, WMMA_N, WMMA_K, half, a_layout> a_frag;
                fragment<matrix_b, WMMA_M, WMMA_N, WMMA_K, half, b_layout> b_frag;
                load_matrix_sync(a_frag, a_ptr, A_STRIDE);
                load_matrix_sync(b_frag, b_ptr, B_STRIDE);
                mma_sync(acc_frag, a_frag, b_frag, acc_frag);
            }
        }
    }
}
