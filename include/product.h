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
template<typename Config, GemmType TYPE, int GLOBAL_STRIDE, int SMEM_STRIDE>
__device__ __forceinline__ void WMMA_GEMM_DOT_PRODUCT(
    const __half* __restrict__ GMEM_O,
    const __half* __restrict__ SMEM_DO,
    const  float* __restrict__ GMEM_LSE,
           float* __restrict__ SMEM_LSE,
           float* __restrict__ SMEM_DOT,
    int VALID_ROWS,
    int OFFSET,
    int THREAD_ID
) {
    constexpr int global_blocks = GLOBAL_STRIDE >> 3;

    const int total_iters = VALID_ROWS * global_blocks;
    if (total_iters == 0) return;

    uint64_t global_base = static_cast<uint64_t>(__cvta_generic_to_global(GMEM_O));
    uint32_t shared_base = static_cast<uint32_t>(__cvta_generic_to_shared(SMEM_DO));

    constexpr bool PHASE           = static_cast<uint8_t>(TYPE) & 0x1;
    constexpr int  THREADS_PER_ROW = PHASE ? Config::DKV::THREADS_PER_ROW : Config::DQ::THREADS_PER_ROW;

    const int work   = (global_blocks + THREADS_PER_ROW - 1) / THREADS_PER_ROW;
    const int row    = THREAD_ID / THREADS_PER_ROW;
    const int thread = THREAD_ID % THREADS_PER_ROW;

    float thread_dot = 0.0f;

    if (row < VALID_ROWS) {
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
    }

    #pragma unroll
    for (int offset = THREADS_PER_ROW / 2; offset > 0; offset >>= 1) {
        thread_dot += __shfl_xor_sync(0xFFFFFFFFU, thread_dot, offset, THREADS_PER_ROW);
    }

    if (thread == 0) {
        SMEM_DOT[row] = thread_dot;
    }

    if (THREAD_ID < VALID_ROWS) {
        if constexpr (PHASE) {
            SMEM_LSE[THREAD_ID] = GMEM_LSE[OFFSET + THREAD_ID];
        } else {
            SMEM_LSE[THREAD_ID] = GMEM_LSE[THREAD_ID];
        }
    }
}
