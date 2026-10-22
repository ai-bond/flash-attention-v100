// ======================================================================================
// * Copyright (c) 2025, D.Skryabin / tg @ai_bond007 SPDX-License: BSD-3-Clause
// ======================================================================================
#pragma once

#include <cuda.h>
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include "swizzle.h"
#include "kernel.h"

// ======================================================================================
// BACKWARD ROWDOT KERNEL
// ======================================================================================
template<int D>
__global__ void __launch_bounds__(MAX_THREADS_PER_WARP * 8, 1)
flash_attention_rowdot_kernel(
    const __half* __restrict__ O,
    const __half* __restrict__ dO,
           float* __restrict__ row_dot,
    const int B,
    const int H_Q,
    const int M
) {
    constexpr int BLOCK_M         = 32;
    constexpr int THREADS_PER_ROW = 8;

    // ==================================================================================
    // Grid Mapping: X for M-blocks, Y for batch-head composite (batch * H_Q + head)
    // ==================================================================================
    const int block_m  = blockIdx.x;
    const int bthd_idx = blockIdx.y;

    if (bthd_idx >= B * H_Q) return;
    const int start_m = block_m * BLOCK_M;

    if (start_m >= M) return;
    const int valid_m = min(BLOCK_M, M - start_m);

    // ==================================================================================
    // Init:   thread/row/lane IDs for the 8-lane reduction group
    // ==================================================================================
    const int tid  = threadIdx.x;
    const int row  = tid / THREADS_PER_ROW;
    const int lane = tid % THREADS_PER_ROW;

    if (row >= valid_m) return;

    const int batch_idx  = bthd_idx / H_Q;
    const int head_idx   = bthd_idx % H_Q;
    const int global_row = start_m + row;

    // ==================================================================================
    // Layout:   O/dO:    [B, M, H_Q, D]  (element = ((batch_idx*M + global_row)*H_Q + head_idx)*D + d)
    //           row_dot: [B, H_Q, M]     (element =  bthd_idx * M + global_row)
    // ==================================================================================
    const __half* o_row  = O  + ((size_t)batch_idx * M + global_row) * H_Q * D + (size_t)head_idx * D;
    const __half* do_row = dO + ((size_t)batch_idx * M + global_row) * H_Q * D + (size_t)head_idx * D;

    constexpr int total_iters = (D + 7) >> 3;
    constexpr int work        = (total_iters + THREADS_PER_ROW - 1) / THREADS_PER_ROW;

    // ==================================================================================
    // Compute:  partial = Σ_d O[row, d] · dO[row, d]
    // Layout:   uint4 (8 half per load) per lane
    // Template: per-lane work = ceil(total_iters / THREADS_PER_ROW)
    // ==================================================================================
    float partial = 0.0f;
    #pragma unroll
    for (int idx = 0; idx < work; ++idx) {
        if ((lane + idx * THREADS_PER_ROW) >= total_iters) break;
        const int col = (lane + idx * THREADS_PER_ROW) << 3;

        const uint4 ov = *reinterpret_cast<const uint4*>(o_row  + col);
        const uint4 dv = *reinterpret_cast<const uint4*>(do_row + col);
        const uint32_t op[4] = {ov.x, ov.y, ov.z, ov.w};
        const uint32_t dp[4] = {dv.x, dv.y, dv.z, dv.w};

        // 4 × half2 per uint4 -> 8 FMA per lane per chunk
        #pragma unroll
        for (int jdx = 0; jdx < 4; ++jdx) {
            const float2 of = __half22float2(*reinterpret_cast<const __half2*>(&op[jdx]));
            const float2 df = __half22float2(*reinterpret_cast<const __half2*>(&dp[jdx]));
            partial = __fmaf_rn(of.x, df.x, partial);
            partial = __fmaf_rn(of.y, df.y, partial);
        }
    }

    // ==================================================================================
    // Compute:  warp-reduce partials across the THREADS_PER_ROW = 8 lanes
    // Layout:   xor-shuffle tree, log2(THREADS_PER_ROW) = 3 rounds, group-scoped
    // ==================================================================================
    #pragma unroll
    for (int offset = THREADS_PER_ROW / 2; offset > 0; offset >>= 1) {
        partial += __shfl_xor_sync(0xFFFFFFFFu, partial, offset, THREADS_PER_ROW);
    }

    // ==================================================================================
    // Store:    row_dot[b, h, m] — single fp32 store by lane 0 of each row-group
    // Layout:   row_dot[bthd_idx * M + global_row]
    // ==================================================================================
    if (lane == 0) {
        row_dot[(size_t)bthd_idx * M + global_row] = partial;
    }
}
