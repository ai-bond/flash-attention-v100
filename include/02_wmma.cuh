#pragma once

#include <cuda.h>
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <mma.h>

// ======================================================================================
// INIT SMEM LAYOUT
// ======================================================================================
template<typename Config>
__device__ __forceinline__ void WMMA_GEMM_INIT_SMEM(char* smem_raw) {
    constexpr int N_U4 = Config::TOTAL_SMEM / 16;
    const int tid = threadIdx.x;
    const int stride = blockDim.x;

    uint32_t addr = static_cast<uint32_t>(__cvta_generic_to_shared(smem_raw));

    #pragma unroll 4
    for (int i = tid; i < N_U4; i += stride) {
        asm volatile("st.shared.v4.u32 [%0], {0x0, 0x0, 0x0, 0x0};"
                     :: "r"(addr + (i << 4))
                     : "memory");
    }
}

// ======================================================================================
// TILE LOADER UINT4 (Universal: Single or Dual load, with internal casting)
// Loads uint4-vectorized tiles from global memory to shared memory with bounds checking.
// ======================================================================================
template<bool DUAL_LOAD, int SRC_STRIDE, int DST_STRIDE>
__device__ __forceinline__ void WMMA_GEMM_LOAD_TILE(
    const __half* __restrict__ SRC0,
          __half* __restrict__ DST0,
    const __half* __restrict__ SRC1,
          __half* __restrict__ DST1,
    int VALID_ROWS,
    int THREAD_ID,
    int THREADS_TOTAL
) {
    constexpr int src_stride_uint4 = (SRC_STRIDE + 7) >> 3;
    constexpr int dst_stride_uint4 = (DST_STRIDE + 7) >> 3;

    const int total_iters   = VALID_ROWS * src_stride_uint4;

    if (total_iters == 0) return;

    uint64_t src_base0 = static_cast<uint64_t>(__cvta_generic_to_global(SRC0));
    uint32_t dst_base0 = static_cast<uint32_t>(__cvta_generic_to_shared(DST0));

    uint64_t src_base1 = 0;
    uint32_t dst_base1 = 0;
    if constexpr (DUAL_LOAD) {
        src_base1 = static_cast<uint64_t>(__cvta_generic_to_global(SRC1));
        dst_base1 = static_cast<uint32_t>(__cvta_generic_to_shared(DST1));
    }

    #pragma unroll 2
    for (int idx = THREAD_ID; idx < total_iters; idx += THREADS_TOTAL) {
        const int row = idx / src_stride_uint4;
        const int col = idx % src_stride_uint4;

        const int src_offset = row * src_stride_uint4 + col;
        const int dst_offset = row * dst_stride_uint4 + col;

        const bool in_bounds = (row < VALID_ROWS);
        const int pred = in_bounds ? 1 : 0;

        if (in_bounds) {
            uint64_t src_addr0 = src_base0 + (static_cast<uint64_t>(src_offset) << 4);
            uint32_t dst_addr0 = dst_base0 + (static_cast<uint32_t>(dst_offset) << 4);

            if constexpr (DUAL_LOAD) {
                uint64_t src_addr1 = src_base1 + (static_cast<uint64_t>(src_offset) << 4);
                uint32_t dst_addr1 = dst_base1 + (static_cast<uint32_t>(dst_offset) << 4);

                uint32_t r0, r1, r2, r3;
                asm volatile(
                    "{\n"
                    "  .reg .pred p;\n"
                    "  setp.ne.b32 p, %8, 0;\n"
                    "  mov.u32 %0, 0;\n"
                    "  mov.u32 %1, 0;\n"
                    "  mov.u32 %2, 0;\n"
                    "  mov.u32 %3, 0;\n"
                    "  @p ld.global.v4.u32 {%0, %1, %2, %3}, [%4];\n"
                    "  @p st.shared.v4.u32 [%6], {%0, %1, %2, %3};\n"
                    "  @p ld.global.v4.u32 {%0, %1, %2, %3}, [%5];\n"
                    "  @p st.shared.v4.u32 [%7], {%0, %1, %2, %3};\n"
                    "}\n"
                    : "=r"(r0), "=r"(r1), "=r"(r2), "=r"(r3)
                    : "l"(src_addr0), "l"(src_addr1),
                      "r"(dst_addr0), "r"(dst_addr1),
                    "r"(pred)
                    : "memory"
                );
            } else {
                uint32_t r0, r1, r2, r3;
                asm volatile(
                    "{\n"
                    "  .reg .pred p;\n"
                    "  setp.ne.b32 p, %6, 0;\n"
                    "  mov.u32 %0, 0;\n"
                    "  mov.u32 %1, 0;\n"
                    "  mov.u32 %2, 0;\n"
                    "  mov.u32 %3, 0;\n"
                    "  @p ld.global.v4.u32 {%0, %1, %2, %3}, [%4];\n"
                    "  @p st.shared.v4.u32 [%5], {%0, %1, %2, %3};\n"
                    "}\n"
                    : "=r"(r0), "=r"(r1), "=r"(r2), "=r"(r3)
                    : "l"(src_addr0), "r"(dst_addr0), "r"(pred)
                    : "memory"
                );
            }
        } else {
            uint32_t dst_addr0 = dst_base0 + (static_cast<uint32_t>(dst_offset) << 4);
            if constexpr (DUAL_LOAD) {
                uint32_t dst_addr1 = dst_base1 + (static_cast<uint32_t>(dst_offset) << 4);
                asm volatile(
                    "st.shared.v4.u32 [%0], {0x0, 0x0, 0x0, 0x0};\n\t"
                    "st.shared.v4.u32 [%1], {0x0, 0x0, 0x0, 0x0};"
                    : : "r"(dst_addr0), "r"(dst_addr1) : "memory"
                );
            } else {
                asm volatile(
                    "st.shared.v4.u32 [%0], {0x0, 0x0, 0x0, 0x0};"
                    : : "r"(dst_addr0) : "memory"
                );
            }
        }
    }
}

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
template<GemmType TYPE, int GLOBAL_STRIDE, int SMEM_STRIDE, int FULL_ROWS>
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
    constexpr int global_blocks = GLOBAL_STRIDE >> 3;

    const int total_iters = VALID_ROWS * global_blocks;
    if (total_iters == 0) return;

    uint64_t global_base = static_cast<uint64_t>(__cvta_generic_to_global(PTR_O));
    uint32_t shared_base = static_cast<uint32_t>(__cvta_generic_to_shared(SMEM_DO));

    constexpr uint8_t bits = static_cast<uint8_t>(TYPE);
    constexpr bool LSE_OFFSET = bits & 0x1;

    const int work = (global_blocks + THREADS_PER_ROW - 1) / THREADS_PER_ROW;

    if (THREAD_ID < VALID_ROWS * THREADS_PER_ROW) {
        const int row    = THREAD_ID / THREADS_PER_ROW;
        const int thread = THREAD_ID % THREADS_PER_ROW;

        const unsigned mask = (VALID_ROWS == FULL_ROWS) ? 0xFFFFFFFFU : __activemask();

        float thread_dot = 0.0f;

        const uint64_t row_global = global_base + (static_cast<uint64_t>(row) * GLOBAL_STRIDE * 2);
        const uint32_t row_shared = shared_base + (static_cast<uint32_t>(row) * SMEM_STRIDE * 2);

        #pragma unroll
        for (int j = 0; j < work; ++j) {
            const int chunk = thread + j * THREADS_PER_ROW;
            if (chunk >= global_blocks) break;
            const int col  =  chunk << 3;
            const int pred = (chunk < global_blocks) ? 1 : 0;

            uint32_t o_pack[4], d_pack[4];

            const uint64_t global_addr = row_global + (static_cast<uint64_t>(col) * 2);
            const uint32_t shared_addr = row_shared + (static_cast<uint32_t>(col) * 2);

            asm volatile(
                "{\n"
                "  .reg .pred p;\n"
                "  setp.ne.b32 p, %10, 0;\n"
                "  mov.u32 %0, 0; mov.u32 %1, 0; mov.u32 %2, 0; mov.u32 %3, 0;\n"
                "  mov.u32 %4, 0; mov.u32 %5, 0; mov.u32 %6, 0; mov.u32 %7, 0;\n"
                "  @p ld.global.v4.u32 {%0, %1, %2, %3}, [%8];\n"
                "  @p ld.shared.v4.u32 {%4, %5, %6, %7}, [%9];\n"
                "}\n"
                : "=r"(o_pack[0]), "=r"(o_pack[1]), "=r"(o_pack[2]), "=r"(o_pack[3]),
                  "=r"(d_pack[0]), "=r"(d_pack[1]), "=r"(d_pack[2]), "=r"(d_pack[3])
                : "l"(global_addr), "r"(shared_addr), "r"(pred)
                : "memory"
            );

            #define H2F_XY(pack, xy) (__half22float2(reinterpret_cast<const __half2&>(pack)).xy)
              thread_dot = __fmaf_rn(H2F_XY(o_pack[0], x), H2F_XY(d_pack[0], x), thread_dot);
              thread_dot = __fmaf_rn(H2F_XY(o_pack[0], y), H2F_XY(d_pack[0], y), thread_dot);
              thread_dot = __fmaf_rn(H2F_XY(o_pack[1], x), H2F_XY(d_pack[1], x), thread_dot);
              thread_dot = __fmaf_rn(H2F_XY(o_pack[1], y), H2F_XY(d_pack[1], y), thread_dot);
              thread_dot = __fmaf_rn(H2F_XY(o_pack[2], x), H2F_XY(d_pack[2], x), thread_dot);
              thread_dot = __fmaf_rn(H2F_XY(o_pack[2], y), H2F_XY(d_pack[2], y), thread_dot);
              thread_dot = __fmaf_rn(H2F_XY(o_pack[3], x), H2F_XY(d_pack[3], x), thread_dot);
              thread_dot = __fmaf_rn(H2F_XY(o_pack[3], y), H2F_XY(d_pack[3], y), thread_dot);
            #undef H2F_XY
        }

        #pragma unroll
        for (int offset = THREADS_PER_ROW / 2; offset > 0; offset >>= 1) {
            thread_dot += __shfl_down_sync(mask, thread_dot, offset, THREADS_PER_ROW);
        }

        if (thread == 0) {
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

// ============================================================================
// WMMA_GEMM_POST_SOFTMAX_GRADIENT: Unified post-softmax computation
// ============================================================================
template<GemmType TYPE, int SMEM_LDS_STRIDE, int SMEM_LDO_STRIDE, int TILE_X, int TILE_Y>
__device__ __forceinline__ void WMMA_GEMM_POST_SOFTMAX_GRADIENT(
    const float* __restrict__ SMEM_S,
    const float* __restrict__ SMEM_DOV,
    const float* __restrict__ SMEM_LSE,
    const float* __restrict__ SMEM_DOT,
    __half* __restrict__ SMEM_P,
    __half* __restrict__ SMEM_DS,
    int VALID_Q_ROWS,
    int VALID_KV_ROWS,
    float SOFTMAX_SCALE,
    int THREAD_ID,
    int THREADS_PER_ROW,
    int THREADS_PER_BLOCK
) {
    constexpr int TOTAL_ELEMENTS = TILE_X * TILE_Y;
    constexpr int TOTAL_PAIRS = (TOTAL_ELEMENTS + 1) / 2;

    constexpr bool IS_SDS_SP = static_cast<uint8_t>(TYPE) & 0x1;

    __half2 buf_ds[2];

    // ====================================================================
    // PHASE 1: READ + COMPUTE
    // ====================================================================
    #pragma unroll 1
    for (int i = THREAD_ID; i < TOTAL_PAIRS; i += THREADS_PER_BLOCK) {
        const int idx0 = i * 2;
        const int idx1 = idx0 + 1;
        const int row0 = idx0 / TILE_Y;
        const int col0 = idx0 % TILE_Y;
        const bool has_pair = (idx1 < TOTAL_ELEMENTS);
        const int row1 = has_pair ? idx1 / TILE_Y : row0;
        const int col1 = has_pair ? idx1 % TILE_Y : col0 + 1;
        const bool in0 = (row0 < VALID_Q_ROWS) && (col0 < VALID_KV_ROWS);
        const bool in1 = has_pair && (row1 < VALID_Q_ROWS) && (col1 < VALID_KV_ROWS);
        const int lds_offset0 = row0 * SMEM_LDS_STRIDE + col0;
        const int lds_offset1 = in1 ? (row1 * SMEM_LDS_STRIDE + col1) : 0;
        const int ldo_offset0 = row0 * SMEM_LDO_STRIDE + col0;
        const int ldo_offset1 = in1 ? (row1 * SMEM_LDO_STRIDE + col1) : 0;

        float s0 = in0 ? SMEM_S[lds_offset0] : NEG_INF;
        float s1 = in1 ? SMEM_S[lds_offset1] : NEG_INF;
        float dov0 = (s0 != NEG_INF) ? SMEM_DOV[lds_offset0] : 0.0f;
        float dov1 = (s1 != NEG_INF) ? SMEM_DOV[lds_offset1] : 0.0f;

        float lse0 = SMEM_LSE[row0];
        float lse1 = (in1 && row1 != row0) ? SMEM_LSE[row1] : lse0;
        float dot0 = SMEM_DOT[row0];
        float dot1 = (in1 && row1 != row0) ? SMEM_DOT[row1] : dot0;

        float sh0 = s0 - lse0;
        float sh1 = s1 - lse1;
        float p0 = (s0 == NEG_INF || sh0 < -80.0f) ? 0.0f : __expf(sh0);
        float p1 = (s1 == NEG_INF || sh1 < -80.0f) ? 0.0f : __expf(sh1);

        float diff0 = dov0 - dot0;
        float diff1 = dov1 - dot1;
        float ds0 = fmaf(p0, SOFTMAX_SCALE * diff0, 0.0f);
        float ds1 = fmaf(p1, SOFTMAX_SCALE * diff1, 0.0f);

        __half2 h2_p  = __float22half2_rn(make_float2(p0, p1));
        __half2 h2_ds = __float22half2_rn(make_float2(ds0, ds1));

        if constexpr (!IS_SDS_SP) {
            int buf_idx = (i - THREAD_ID) / THREADS_PER_BLOCK;
            if (buf_idx < 2) {
                buf_ds[buf_idx] = h2_ds;
            } else {
                if (in0) SMEM_DS[ldo_offset0] = h2_ds.x;
                if (in1) SMEM_DS[ldo_offset1] = h2_ds.y;
            }
        }
        else {
            bool vectorize = (col1 < TILE_Y) && (row1 == row0);

            if (vectorize) {
                uintptr_t addr_p  = reinterpret_cast<uintptr_t>(SMEM_P  + ldo_offset0);
                uintptr_t addr_ds = reinterpret_cast<uintptr_t>(SMEM_DS + ldo_offset0);
                vectorize = ((addr_p & 0x3) == 0) && ((addr_ds & 0x3) == 0);
            }

            if (vectorize) {
                *reinterpret_cast<__half2*>(SMEM_P  + ldo_offset0) = h2_p;
                *reinterpret_cast<__half2*>(SMEM_DS + ldo_offset0) = h2_ds;
            } else {
                if (in0) {
                    SMEM_P[ldo_offset0]  = h2_p.x;
                    SMEM_DS[ldo_offset0] = h2_ds.x;
                }
                if (in1) {
                    SMEM_P[ldo_offset1]  = h2_p.y;
                    SMEM_DS[ldo_offset1] = h2_ds.y;
                }
            }
        }
    }

    // ====================================================================
    // PHASE 2: WRITE from register buffers to SMEM (dQ path ONLY)
    // ====================================================================
    if constexpr (!IS_SDS_SP) {
        #pragma unroll 1
        for (int i = THREAD_ID; i < TOTAL_PAIRS; i += THREADS_PER_BLOCK) {
            int buf_idx = (i - THREAD_ID) / THREADS_PER_BLOCK;
            if (buf_idx >= 2) continue;

            const int idx0 = i * 2;
            const int idx1 = idx0 + 1;
            const int row0 = idx0 / TILE_Y;
            const int col0 = idx0 % TILE_Y;
            const bool has_pair = (idx1 < TOTAL_ELEMENTS);
            const int row1 = has_pair ? idx1 / TILE_Y : row0;
            const int col1 = has_pair ? idx1 % TILE_Y : col0 + 1;
            const int ldo_offset0 = row0 * SMEM_LDO_STRIDE + col0;
            const int ldo_offset1 = has_pair ? (row1 * SMEM_LDO_STRIDE + col1) : 0;

            bool vectorize = (col1 < TILE_Y) && (row1 == row0);

            if (vectorize) {
                uintptr_t addr_ds = reinterpret_cast<uintptr_t>(SMEM_DS + ldo_offset0);
                vectorize = ((addr_ds & 0x3) == 0);
            }

            if (vectorize) {
                *reinterpret_cast<__half2*>(SMEM_DS + ldo_offset0) = buf_ds[buf_idx];
            } else {
                SMEM_DS[ldo_offset0] = buf_ds[buf_idx].x;
                if (has_pair && (row1 < VALID_Q_ROWS) && (col1 < VALID_KV_ROWS)) {
                    SMEM_DS[ldo_offset1] = buf_ds[buf_idx].y;
                }
            }
        }
    }
}
