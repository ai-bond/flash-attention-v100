#pragma once

#include <cuda.h>
#include <cuda_runtime.h>
#include <cuda_fp16.h>

// ============================================================================
// GEMM OPERATION (3-bit trait encoding)
// Bit 0 (0x1): APPLY_MASK (1=apply causal mask to output, 0=no causal mask to output)
// Bit 1 (0x2): A_IS_COL   (1=load A as col_major fragment, 0=row_major)
// Bit 2 (0x4): B_IS_COL   (1=load B as col_major fragment, 0=row_major)
// ============================================================================
enum class GemmType : uint8_t {
    sQ_KT     = (1<<0) | (0<<1) | (1<<2),  // sQ_KT:       Q(row) @  K(col)^T, APPLY_MASK=1,  A=row(0), B=col(1) → 0b101 = 5
    dOV_dOVT  = (0<<0) | (0<<1) | (1<<2),  // dOV_dOVT:   dO(row) @  V(col)^T, APPLY_MASK=0,  A=row(0), B=col(1) → 0b100 = 4
};

template<GemmType TYPE, int D, bool IS_CAUSAL, int BLOCK_X, int BLOCK_Y, int IN_STRIDE, int OUT_STRIDE, int WARPS_PER_BLOCK>
__device__ __forceinline__ void WMMA_GEMM_SCORES(
    const __half* __restrict__ SMEM_A,
    const __half* __restrict__ SMEM_B,
           float* __restrict__ SMEM_C,
    int VALID_M,  int VALID_N,
    int GLOBAL_M, int GLOBAL_N,
    float SOFTMAX_SCALE,
    int WARP_ID,  int LANE_ID
) {
    using namespace nvcuda::wmma;

    constexpr uint8_t bits = static_cast<uint8_t>(TYPE);

    constexpr bool APPLY_MASK = bits & 0x1;
    constexpr bool A_IS_COL   = bits & 0x2;
    constexpr bool B_IS_COL   = bits & 0x4;

    constexpr int num_tiles_m = (BLOCK_X + WMMA_M - 1) / WMMA_M;
    constexpr int num_tiles_n = (BLOCK_Y + WMMA_N - 1) / WMMA_N;
    constexpr int num_tiles_k = (D + WMMA_K - 1) / WMMA_K;

    constexpr int total_tiles = num_tiles_m * num_tiles_n;
    constexpr int tiles_per_warp = (total_tiles + WARPS_PER_BLOCK - 1) / WARPS_PER_BLOCK;

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

        using a_layout = std::conditional_t<A_IS_COL, col_major, row_major>;
        using b_layout = std::conditional_t<B_IS_COL, col_major, row_major>;

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

        if constexpr (APPLY_MASK && IS_CAUSAL) {
            #pragma unroll
            for (int i = 0; i < acc_frag.num_elements; ++i) {

                const unsigned col = col_causal + (i & 0b1) + ((i >> 2) & 0b1) * 4;
                const unsigned row = row_causal + ((i >> 1) & 0b1) * 2;

                const int global_m = GLOBAL_M + tile_m + row;
                const int global_n = GLOBAL_N + tile_n + col;

                const bool in_bounds = (global_m < GLOBAL_M + VALID_M) && 
                                       (global_n < GLOBAL_N + VALID_N);

                acc_frag.x[i] = in_bounds 
                    ? ((global_n > global_m) ? NEG_INF : acc_frag.x[i] * SOFTMAX_SCALE) 
                    : NEG_INF;
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
