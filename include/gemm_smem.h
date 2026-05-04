// ======================================================================================
// * Copyright (c) 2026, D.Skryabin / tg @ai_bond007 SPDX-License: BSD-3-Clause
// ======================================================================================
#pragma once

#include <cuda.h>
#include <cuda_runtime.h>
#include <cuda_fp16.h>

// ======================================================================================
// INIT SMEM LAYOUT
// ======================================================================================
template<typename Config>
__device__ __forceinline__ void WMMA_GEMM_INIT_SMEM(char* smem_raw) {
    constexpr int UINT4  = Config::TOTAL_SMEM / 16;
    constexpr int STRIDE = Config::THREADS_PER_BLOCK;
    constexpr int ITERS  = (UINT4 + STRIDE - 1) / STRIDE;

    uint32_t base    = static_cast<uint32_t>(__cvta_generic_to_shared(smem_raw));
    uint32_t str_ptr = base + ((threadIdx.x ^ (threadIdx.x >> 3)) << 4);
    uint32_t end_ptr = base + (UINT4 << 4);
    constexpr uint32_t step = STRIDE << 4;

    #pragma unroll
    for (int i = 0; i < ITERS; ++i) {
        asm volatile(
            "setp.lt.u32 %%p0, %0, %1;\n\t"
            "@%%p0 st.shared.v4.u32 [%0], {0, 0, 0, 0};\n\t"
            :: "r"(str_ptr), "r"(end_ptr)
        );
        str_ptr += step;
    }
}

// ======================================================================================
// TILE LOADER (Single or Dual load, with internal casting)
// Loads uint4-vectorized tiles from global memory to shared memory with bounds checking.
// ======================================================================================
template<typename Config, bool DUAL_LOAD, int DST_STRIDE>
__device__ __forceinline__ void WMMA_GEMM_LOAD_TILE(
    const __half* __restrict__ GMEM0,
          __half* __restrict__ SMEM0,
    const __half* __restrict__ GMEM1,
          __half* __restrict__ SMEM1,
    int SRC_STRIDE,
    int VALID_ROWS,
    int THREAD_ID
) {
    constexpr int THREADS_PER_BLOCK = Config::THREADS_PER_BLOCK;
    constexpr int dst_stride_uint4  = (DST_STRIDE + 7) >> 3;
    const     int src_stride_uint4  = (SRC_STRIDE + 7) >> 3;
    const     int total_iters       = VALID_ROWS * src_stride_uint4;

    if (total_iters == 0) return;

    uint64_t src_base0 = static_cast<uint64_t>(__cvta_generic_to_global(GMEM0));
    uint32_t dst_base0 = static_cast<uint32_t>(__cvta_generic_to_shared(SMEM0));

    uint64_t src_base1 = 0;
    uint32_t dst_base1 = 0;
    if constexpr (DUAL_LOAD) {
        src_base1 = static_cast<uint64_t>(__cvta_generic_to_global(GMEM1));
        dst_base1 = static_cast<uint32_t>(__cvta_generic_to_shared(SMEM1));
    }

    #pragma unroll 2
    for (int idx = THREAD_ID; idx < total_iters; idx += THREADS_PER_BLOCK) {
        const int row = idx / src_stride_uint4;
        const int col = idx % src_stride_uint4;

        const int src_offset = row * src_stride_uint4 + col;
        const int dst_offset = row * dst_stride_uint4 + col;

        uint64_t src_addr0 = src_base0 + (static_cast<uint64_t>(src_offset) << 4);
        uint32_t dst_addr0 = dst_base0 + (static_cast<uint32_t>(dst_offset) << 4);

        uint32_t r0, r1, r2, r3;
        if constexpr (DUAL_LOAD) {
            uint64_t src_addr1 = src_base1 + (static_cast<uint64_t>(src_offset) << 4);
            uint32_t dst_addr1 = dst_base1 + (static_cast<uint32_t>(dst_offset) << 4);

            asm volatile(
                "{\n\t"
                "  ld.global.v4.u32 {%0, %1, %2, %3}, [%4];\n\t"
                "  st.shared.v4.u32 [%6], {%0, %1, %2, %3};\n\t"
                "  ld.global.v4.u32 {%0, %1, %2, %3}, [%5];\n\t"
                "  st.shared.v4.u32 [%7], {%0, %1, %2, %3};\n\t"
                "}\n\t"
                : "=r"(r0), "=r"(r1), "=r"(r2), "=r"(r3)
                : "l"(src_addr0), "l"(src_addr1),
                  "r"(dst_addr0), "r"(dst_addr1)
                : "memory"
            );
        } else {
            asm volatile(
                "{\n\t"
                "  ld.global.v4.u32 {%0, %1, %2, %3}, [%4];\n\t"
                "  st.shared.v4.u32 [%5], {%0, %1, %2, %3};\n\t"
                "}\n\t"
                : "=r"(r0), "=r"(r1), "=r"(r2), "=r"(r3)
                : "l"(src_addr0), "r"(dst_addr0)
                : "memory"
            );
        }
    }
}

// ============================================================================
// KERNEL_EPILOGUE
// ============================================================================
template<typename Config, GemmType TYPE, int SMEM_STRIDE>
__device__ __forceinline__ void WMMA_GEMM_EPILOGUE(
    const float* __restrict__ SMEM0,
         __half* __restrict__ GMEM0,
    const float* __restrict__ SMEM1,
         __half* __restrict__ GMEM1,
    const float* __restrict__ SMEM_DOT,
    int GLOBAL_STRIDE,
    int VALID_ROWS,
    int THREAD_ID
) {

    const int global_chunks = GLOBAL_STRIDE >> 2;
    const int total_iters   = VALID_ROWS * global_chunks;

    if (total_iters == 0) return;

    constexpr int  THREADS_PER_BLOCK = Config::THREADS_PER_BLOCK;
    constexpr bool NORMLZE    = static_cast<uint8_t>(TYPE) & 0x1;
    constexpr bool DUAL_STORE = static_cast<uint8_t>(TYPE) & 0x2;

    for (int i = THREAD_ID; i < total_iters; i += THREADS_PER_BLOCK) {
        const int row =  i / global_chunks;
        const int col = (i % global_chunks) << 2;

        float norm = 1.0f;
        if constexpr (NORMLZE) {
            norm = __frcp_rn(fmaxf(SMEM_DOT[row], 1e-24f));
        }

        const bool in_bounds = (row < VALID_ROWS);
        const int pred = in_bounds ? 1 : 0;

        if (in_bounds) {
            const float4 smem_01 = *reinterpret_cast<const float4*>(SMEM0 + row * SMEM_STRIDE + col);
            const __half2 h0 = __float22half2_rn(make_float2(smem_01.x * norm, smem_01.y * norm));
            const __half2 h1 = __float22half2_rn(make_float2(smem_01.z * norm, smem_01.w * norm));

            ushort v0 = __half_as_ushort(h0.x); ushort v1 = __half_as_ushort(h0.y);
            ushort v2 = __half_as_ushort(h1.x); ushort v3 = __half_as_ushort(h1.y);

            asm volatile(
                "{\n"
                "  .reg .pred p;\n"
                "  setp.ne.b32 p, %8, 0;\n"
                "  mov.u16 %0, %4;\n"
                "  mov.u16 %1, %5;\n"
                "  mov.u16 %2, %6;\n"
                "  mov.u16 %3, %7;\n"
                "  @p st.global.v4.u16 [%9], {%0, %1, %2, %3};\n"
                "}\n"
                : "=h"(v0), "=h"(v1), "=h"(v2), "=h"(v3)
                : "h"(v0),  "h"(v1),  "h"(v2),  "h"(v3),
                  "r"(pred),
                  "l"(GMEM0 + row * GLOBAL_STRIDE + col)
                : "memory"
            );

            if constexpr (DUAL_STORE) {
                const float4 smem_02 = *reinterpret_cast<const float4*>(SMEM1 + row * SMEM_STRIDE + col);
                const __half2 h2 = __float22half2_rn(make_float2(smem_02.x * norm, smem_02.y * norm));
                const __half2 h3 = __float22half2_rn(make_float2(smem_02.z * norm, smem_02.w * norm));

                ushort v4 = __half_as_ushort(h2.x); ushort v5 = __half_as_ushort(h2.y);
                ushort v6 = __half_as_ushort(h3.x); ushort v7 = __half_as_ushort(h3.y);

                asm volatile(
                    "{\n"
                    "  .reg .pred p;\n"
                    "  setp.ne.b32 p, %8, 0;\n"
                    "  mov.u16 %0, %4;\n"
                    "  mov.u16 %1, %5;\n"
                    "  mov.u16 %2, %6;\n"
                    "  mov.u16 %3, %7;\n"
                    "  @p st.global.v4.u16 [%9], {%0, %1, %2, %3};\n"
                    "}\n"
                    : "=h"(v4), "=h"(v5), "=h"(v6), "=h"(v7)
                    : "h"(v4), "h"(v5), "h"(v6), "h"(v7),
                      "r"(pred),
                      "l"(GMEM1 + row * GLOBAL_STRIDE + col)
                    : "memory"
                );
            }
        }
    }
}
