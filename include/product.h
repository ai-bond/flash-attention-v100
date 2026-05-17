// ======================================================================================
// * Copyright (c) 2025, D.Skryabin / tg @ai_bond007 SPDX-License: BSD-3-Clause
// ======================================================================================
#pragma once

#include <cuda.h>
#include <cuda_runtime.h>
#include <cuda_fp16.h>

// ======================================================================================
// COMPUTE_ROW_DOT
// ======================================================================================
template <typename Config, GemmType TYPE, int SMEM_STRIDE, int GLOBAL_WIDTH = -1>
__device__ __forceinline__ void WMMA_GEMM_DOT_PRODUCT(
    const __half* __restrict__ GMEM_O,
    const __half* __restrict__ SMEM_DO,
    const float*  __restrict__ GMEM_LSE,
          float*  __restrict__ SMEM_LSE,
          float*  __restrict__ SMEM_DOT,
    int GLOBAL_STRIDE,
    int VALID_ROWS,
    int OFFSET,
    int THREAD_ID
) {
    constexpr bool PHASE            = static_cast<uint8_t>(TYPE) & 0x1;
    constexpr int THREADS_PER_ROW   = PHASE ? Config::DKV::THREADS_PER_ROW : Config::DQ::THREADS_PER_ROW;

    const int total_iters = (GLOBAL_WIDTH > 0 ? GLOBAL_WIDTH : GLOBAL_STRIDE) + 7 >> 3;

    if (total_iters == 0 || VALID_ROWS == 0) return;

    const int row  = THREAD_ID / THREADS_PER_ROW;
    const int lane = THREAD_ID % THREADS_PER_ROW;
    const int work = (total_iters + THREADS_PER_ROW - 1) / THREADS_PER_ROW;

    float thread_dot = 0.0f;

    if (row < VALID_ROWS) {

        const uint64_t gmem_base = static_cast<uint64_t>(__cvta_generic_to_global(GMEM_O))  + static_cast<uint64_t>(row) * GLOBAL_STRIDE * 2;
        const uint32_t smem_base = static_cast<uint32_t>(__cvta_generic_to_shared(SMEM_DO)) + static_cast<uint32_t>(row) * SMEM_STRIDE   * 2;

        #pragma unroll
        for (int j = 0; j < work; ++j) {
            const int chunk = lane + j * THREADS_PER_ROW;
            if (chunk >= total_iters) break;

            const int col  = chunk << 3;
            const int pred = (col < (GLOBAL_WIDTH > 0 ? GLOBAL_WIDTH : GLOBAL_STRIDE)) ? 1 : 0;

            uint32_t o_pack[4], d_pack[4];
            const uint64_t gmem_addr = gmem_base + static_cast<uint64_t>(col) * 2;
            const uint32_t smem_addr = smem_base + static_cast<uint32_t>(col) * 2;

            asm volatile(
                "{\n\t"
                "  .reg .pred p;\n\t"
                "  setp.ne.b32 p, %10, 0;\n\t"
                "  mov.u32 %0, 0; mov.u32 %1, 0; mov.u32 %2, 0; mov.u32 %3, 0;\n\t"
                "  mov.u32 %4, 0; mov.u32 %5, 0; mov.u32 %6, 0; mov.u32 %7, 0;\n\t"
                "  @p ld.global.v4.u32 {%0, %1, %2, %3}, [%8];\n\t"
                "  @p ld.shared.v4.u32 {%4, %5, %6, %7}, [%9];\n\t"
                "}\n\t"
                : "=r"(o_pack[0]), "=r"(o_pack[1]), "=r"(o_pack[2]), "=r"(o_pack[3]),
                  "=r"(d_pack[0]), "=r"(d_pack[1]), "=r"(d_pack[2]), "=r"(d_pack[3])
                : "l"(gmem_addr), "r"(smem_addr), "r"(pred)
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
    }

    #pragma unroll
    for (int offset = THREADS_PER_ROW / 2; offset > 0; offset >>= 1) {
        thread_dot += __shfl_xor_sync(0xFFFFFFFFU, thread_dot, offset, THREADS_PER_ROW);
    }

    if (lane == 0 && row < VALID_ROWS) {
        SMEM_DOT[row] = thread_dot;
    }

    if (THREAD_ID < VALID_ROWS) {
        SMEM_LSE[THREAD_ID] = GMEM_LSE[OFFSET + THREAD_ID];
    }
}
