#pragma once

#include <cuda.h>
#include <cuda_runtime.h>
#include <cuda_fp16.h>

// ============================================================================
// INIT SMEM LAYOUT
// ============================================================================
template<typename Config>
__device__ __forceinline__ void init_smem(char* smem_raw) {
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
    __syncthreads();
}

// ============================================================================
// UINT4 TILE LOADER
// ============================================================================
__device__ __forceinline__ void load_tile_uint4(
    const uint4* __restrict__ src_vec,
          uint4* __restrict__ dst_vec,
    int valid_rows,
    int src_stride_uint4,
    int dst_stride_uint4,
    int tid,
    int threads_per_block
) {
    #pragma unroll 2
    for (int idx = tid; idx < (valid_rows * src_stride_uint4); idx += threads_per_block) {
        const int row = idx / src_stride_uint4;
        const int vec_col = idx % src_stride_uint4;
        uint4 val = make_uint4(0, 0, 0, 0);
        if (row < valid_rows) {
            val = __ldg(&src_vec[row * src_stride_uint4 + vec_col]);
        }
        dst_vec[row * dst_stride_uint4 + vec_col] = val;
    }
}

// ============================================================================
// DUAL UINT4 TILE LOADER
// ============================================================================
__device__ __forceinline__ void load_tile_uint4_dual(
    const uint4* __restrict__ src0_vec,
    const uint4* __restrict__ src1_vec,
    uint4* __restrict__ dst0_vec,
    uint4* __restrict__ dst1_vec,
    int valid_rows,
    int src_stride_uint4,
    int dst_stride_uint4,
    int tid,
    int threads_per_block
) {
    #pragma unroll 2
    for (int idx = tid; idx < (valid_rows * src_stride_uint4); idx += threads_per_block) {
        const int row = idx / src_stride_uint4;
        const int vec_col = idx % src_stride_uint4;
        uint4 val0 = make_uint4(0, 0, 0, 0);
        uint4 val1 = make_uint4(0, 0, 0, 0);
        if (row < valid_rows) {
            val0 = __ldg(&src0_vec[row * src_stride_uint4 + vec_col]);
            val1 = __ldg(&src1_vec[row * src_stride_uint4 + vec_col]);
        }
        dst0_vec[row * dst_stride_uint4 + vec_col] = val0;
        dst1_vec[row * dst_stride_uint4 + vec_col] = val1;
    }
}