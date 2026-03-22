#pragma once

#include <cuda.h>
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <mma.h>

// ============================================================================
// WMMA_GEMM_SCORES: Compute C = (A @ B) * scale [+(causal mask)]
// Use for: Q@K^T (forward), dO@V^T (backward pre-softmax)
// ============================================================================
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

    using a_layout = std::conditional_t<A_IS_COL, col_major, row_major>;
    using b_layout = std::conditional_t<B_IS_COL, col_major, row_major>;

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

// ============================================================================
// WMMA_GEMM_GRADIENTS: Compute C += A @ B  (Read-Modify-Write accumulation)
// Use for: P@V, dS@K, P^T@dO, dS^T@Q in backward pass
// ============================================================================
template<GemmType TYPE, int D, int BLOCK_X, int BLOCK_Y, int IN_STRIDE, int OUT_STRIDE, int WARPS_PER_BLOCK>
__device__ __forceinline__ void WMMA_GEMM_GRADIENTS(
    const __half* __restrict__ SMEM_A,
    const __half* __restrict__ SMEM_B,
           float* __restrict__ SMEM_C,
    int VALID_M,
    int VALID_K,
    int WARP_ID,
    int LANE_ID
) {
    using namespace nvcuda::wmma;

    constexpr uint8_t bits = static_cast<uint8_t>(TYPE);

    constexpr bool A_IS_COL   = bits & 0x2;
    constexpr bool B_IS_COL   = bits & 0x4;

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

// ============================================================================
// KERNEL_EPILOGUE
// ============================================================================
template<GemmType TYPE, int D, int D_STRIDE>
__device__ __forceinline__ void WMMA_GEMM_EPILOGUE(
    const float* __restrict__ S0_SRC,
    const float* __restrict__ S1_SRC,
         __half* __restrict__ G0_DST,
         __half* __restrict__ G1_DST,
    const float* __restrict__ ROW_SUM,
    int VALID_ROWS,
    int THREAD_ID,
    int THREADS_PER_BLOCK
) {
    constexpr uint8_t bits = static_cast<uint8_t>(TYPE);

    constexpr bool NORMLZE = bits & 0x1;
    constexpr bool DUAL    = bits & 0x2;

    constexpr int VEC = 4;
    constexpr int COL_BLOCKS = D / VEC;
    const int total_iters = VALID_ROWS * COL_BLOCKS;

    for (int i = THREAD_ID; i < total_iters; i += THREADS_PER_BLOCK) {
        const int row =  i / COL_BLOCKS;
        const int col = (i % COL_BLOCKS) * VEC;

        float norm = 1.0f;
        if constexpr (NORMLZE) {
            norm = 1.0f / fmaxf(ROW_SUM[row], 1e-24f);
        }

        const float* s_row = S0_SRC + row * D_STRIDE;
        const __half h0 = __float2half_rn(s_row[col + 0] * norm);
        const __half h1 = __float2half_rn(s_row[col + 1] * norm);
        const __half h2 = __float2half_rn(s_row[col + 2] * norm);
        const __half h3 = __float2half_rn(s_row[col + 3] * norm);

        asm volatile(
            "st.global.v4.u16 [%0], {%1, %2, %3, %4};"
            : : "l"(G0_DST + row * D + col),
                "h"(__half_as_ushort(h0)),
                "h"(__half_as_ushort(h1)),
                "h"(__half_as_ushort(h2)),
                "h"(__half_as_ushort(h3))
            : "memory"
        );

        if constexpr (DUAL) {
            const float* s_row2 = S1_SRC + row * D_STRIDE;
            const __half k0 = __float2half_rn(s_row2[col + 0] * norm);
            const __half k1 = __float2half_rn(s_row2[col + 1] * norm);
            const __half k2 = __float2half_rn(s_row2[col + 2] * norm);
            const __half k3 = __float2half_rn(s_row2[col + 3] * norm);

            asm volatile(
                "st.global.v4.u16 [%0], {%1, %2, %3, %4};"
                : : "l"(G1_DST + row * D + col),
                    "h"(__half_as_ushort(k0)),
                    "h"(__half_as_ushort(k1)),
                    "h"(__half_as_ushort(k2)),
                    "h"(__half_as_ushort(k3))
                : "memory"
            );
        }
    }
}

// ============================================================================
// COMPUTE_ROW_DOT
// ============================================================================
template<GemmType TYPE, int D, int D_STRIDE>
__device__ __forceinline__ void WMMA_GEMM_DOT_PRODUCT(
    const __half* __restrict__ PTR_O,
    const __half* __restrict__ SMEM_DO,
    const  float* __restrict__ PTR_LSE,
           float* __restrict__ SMEM_LSE,
           float* __restrict__ SMEM_DOT,
    int VALID_ROWS,
    int OFFSET,
    int THREAD_ID,
    int THREADS_PER_ROW,
    int THREADS_PER_BLOCK
) {
    constexpr uint8_t bits = static_cast<uint8_t>(TYPE);

    constexpr bool LSE_OFFSET = bits & 0x1;
    constexpr bool USE_FULL_MASK  = bits & 0x2;

    constexpr int VEC = 4;
    constexpr int COL_BLOCKS = D / VEC;

    const int work_per_thread = (COL_BLOCKS + THREADS_PER_ROW - 1) / THREADS_PER_ROW;

    if (THREAD_ID < VALID_ROWS * THREADS_PER_ROW) {
        const int row = THREAD_ID / THREADS_PER_ROW;
        const int thread_in_row = THREAD_ID % THREADS_PER_ROW;

        const unsigned mask = USE_FULL_MASK ? 0xFFFFFFFFU : __activemask();

        float thread_dot = 0.0f;

        #pragma unroll
        for (int j = 0; j < work_per_thread; ++j) {
            const int chunk_idx = thread_in_row + j * THREADS_PER_ROW;
            if (chunk_idx >= COL_BLOCKS) break;
            const int col = chunk_idx * VEC;

            // Load O from global memory (vectorized 4×half)
            const __half* o_addr = PTR_O + row * D + col;
            ushort o_h0, o_h1, o_h2, o_h3;
            asm volatile(
                "ld.global.v4.u16 {%0, %1, %2, %3}, [%4];"
                : "=h"(o_h0), "=h"(o_h1), "=h"(o_h2), "=h"(o_h3)
                : "l"(o_addr)
                : "memory"
            );

            // Load dO from shared memory (vectorized 4×half)
            const __half* dO_addr = SMEM_DO + row * D_STRIDE + col;
            ushort d_h0, d_h1, d_h2, d_h3;
            const uint32_t ptr_dO = static_cast<uint32_t>(
                reinterpret_cast<uintptr_t>(__cvta_generic_to_shared(dO_addr))
            );
            asm volatile(
                "ld.shared.v4.u16 {%0, %1, %2, %3}, [%4];"
                : "=h"(d_h0), "=h"(d_h1), "=h"(d_h2), "=h"(d_h3)
                : "r"(ptr_dO)
                : "memory"
            );

            thread_dot = __fmaf_rn(__half2float(__ushort_as_half(o_h0)),
                                   __half2float(__ushort_as_half(d_h0)), thread_dot);
            thread_dot = __fmaf_rn(__half2float(__ushort_as_half(o_h1)),
                                   __half2float(__ushort_as_half(d_h1)), thread_dot);
            thread_dot = __fmaf_rn(__half2float(__ushort_as_half(o_h2)),
                                   __half2float(__ushort_as_half(d_h2)), thread_dot);
            thread_dot = __fmaf_rn(__half2float(__ushort_as_half(o_h3)),
                                   __half2float(__ushort_as_half(d_h3)), thread_dot);
        }

        #pragma unroll
        for (int offset = THREADS_PER_ROW / 2; offset > 0; offset >>= 1) {
            thread_dot += __shfl_down_sync(mask, thread_dot, offset, THREADS_PER_ROW);
        }

        if (thread_in_row == 0) {
            SMEM_DOT[row] = thread_dot;
        }
    }

    if (THREAD_ID < VALID_ROWS) {
        if constexpr (LSE_OFFSET) {
            SMEM_LSE[THREAD_ID] = PTR_LSE[OFFSET + THREAD_ID];
        } else {
            SMEM_LSE[THREAD_ID] = PTR_LSE[THREAD_ID];
        }
    }
}
