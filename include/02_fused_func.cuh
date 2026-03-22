#pragma once

#include <cuda.h>
#include <cuda_runtime.h>
#include <cuda_fp16.h>

// ======================================================================================
// INIT SMEM LAYOUT
// ======================================================================================
template<typename Config>
__device__ __forceinline__ void INIT_SMEM(char* smem_raw) {
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
// TILE LOADER UINT4
// ======================================================================================
template<int SRC_STRIDE, int DST_STRIDE>
__device__ __forceinline__ void LOAD_TILE(
    const uint4* __restrict__ SRC_VEC,
          uint4* __restrict__ DST_VEC,
    int VALID_ROWS,
    int THREAD_ID,
    int THREADS_TOTAL
) {
    constexpr int src_stride_uint4 = (SRC_STRIDE + 7) / 8;
    constexpr int dst_stride_uint4 = (DST_STRIDE + 7) / 8;

    #pragma unroll 2
    for (int idx = THREAD_ID; idx < (VALID_ROWS * src_stride_uint4); idx += THREADS_TOTAL) {
        const int row = idx / src_stride_uint4;
        const int col = idx % src_stride_uint4;

        uint4 val = make_uint4(0, 0, 0, 0);
        if (row < VALID_ROWS) {
            val = __ldg(&SRC_VEC[row * src_stride_uint4 + col]);
        }
        DST_VEC[row * dst_stride_uint4 + col] = val;
    }
}

// ======================================================================================
// TILE DUAL LOADER UINT4
// ======================================================================================
template<int SRC_STRIDE, int DST_STRIDE>
__device__ __forceinline__ void LOAD_TILE_DUAL(
    const uint4* __restrict__ SRC0_VEC,
    const uint4* __restrict__ SRC1_VEC,
          uint4* __restrict__ DST0_VEC,
          uint4* __restrict__ DST1_VEC,
    int VALID_ROWS,
    int THREAD_ID,
    int THREADS_TOTAL
) {
    constexpr int src_stride_uint4 = (SRC_STRIDE + 7) / 8;
    constexpr int dst_stride_uint4 = (DST_STRIDE + 7) / 8;

    #pragma unroll 2
    for (int idx = THREAD_ID; idx < (VALID_ROWS * src_stride_uint4); idx += THREADS_TOTAL) {
        const int row = idx / src_stride_uint4;
        const int col = idx % src_stride_uint4;

        uint4 val0 = make_uint4(0, 0, 0, 0);
        uint4 val1 = make_uint4(0, 0, 0, 0);
        if (row < VALID_ROWS) {
            val0 = __ldg(&SRC0_VEC[row * src_stride_uint4 + col]);
            val1 = __ldg(&SRC1_VEC[row * src_stride_uint4 + col]);
        }
        DST0_VEC[row * dst_stride_uint4 + col] = val0;
        DST1_VEC[row * dst_stride_uint4 + col] = val1;
    }
}

// ============================================================================
// KERNEL_EPILOGUE
// ============================================================================
template<bool NORMLZE, bool DUAL, int D, int D_STRIDE>
__device__ __forceinline__ void KERNEL_EPILOGUE(
    const float* __restrict__ S0_SRC,
    const float* __restrict__ S1_SRC,
         __half* __restrict__ G0_DST,
         __half* __restrict__ G1_DST,
    const float* __restrict__ ROW_SUM,
    int VALID_ROWS,
    int THREAD_ID,
    int THREADS_PER_BLOCK
) {
    constexpr int VEC = 4;
    constexpr int COL_BLOCKS = D / VEC;

    const int total_iters = VALID_ROWS * COL_BLOCKS;

    for (int i = THREAD_ID; i < total_iters; i += THREADS_PER_BLOCK) {
        const int row = i / COL_BLOCKS;
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
