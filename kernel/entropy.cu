// ======================================================================================
// * Copyright (c) 2026, D.Skryabin / tg @ai_bond007 SPDX-License: BSD-3-Clause
// ======================================================================================
#include <cuda.h>
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <torch/extension.h>
#include <type_traits>

#include <ATen/ATen.h>
#include <ATen/cuda/CUDAContext.h>
#include <ATen/cuda/CUDAGeneratorImpl.h>
#include <c10/cuda/CUDAGuard.h>

#include "kernel.h"
#include "entropy.h"
#include "swizzle.h"
#include "gemm_smem.h"
#ifdef MMA_NATIVE
    #include <mma.h>
    using namespace nvcuda::wmma;
#else
    #include "mma_m16n16k16.h"
    using namespace volta::wmma;
#endif

// ======================================================================================
// KERNEL 1: PartialMax/PartialSum/PartialSumLogits/TargetLogit = Reduce(Input @ Weightᵀ)
// ======================================================================================
template<typename Config>
__global__ void __launch_bounds__(LogitsConfig::THREADS_PER_BLOCK, 1)
forward_lce_partial(
    const __half* __restrict__ Input,
    const __half* __restrict__ Weight,
    const int*    __restrict__ Target,
          float*  __restrict__ PartialMax,
          float*  __restrict__ PartialSum,
          float*  __restrict__ PartialSumLogits,
          float*  __restrict__ TargetLogit,
    int   M,
    int   V,
    int   D,
    int   stride_v,
    float logit_scale,
    int   ignore_index
) {
    constexpr int BLOCK_M         = Config::BLOCK_M;
    constexpr int BLOCK_N         = Config::BLOCK_N;
    constexpr int DIM             = Config::DIM;
    constexpr int WARPS_PER_BLOCK = Config::WARPS_PER_BLOCK;
    constexpr int THREADS_PER_ROW = Config::THREADS_PER_ROW;
    constexpr int MAX_ITERS       = Config::MAX_ITERS;
    constexpr int MAX_TAILS       = Config::MAX_TAILS;

    // ==================================================================================
    // Grid Mapping: X for M-blocks, Y for V-blocks (vocab tiling)
    // ==================================================================================
    const int start_m = blockIdx.x * BLOCK_M;
    const int v_tile  = blockIdx.y * BLOCK_N;
    if (start_m >= M || v_tile >= V) return;

    const int valid_m = min(BLOCK_M, M - start_m);
    const int valid_v = min(BLOCK_N, V - v_tile);

    // ==================================================================================
    // Init:   thread/warp/lane IDs for WMMA coordination
    // ==================================================================================
    const int tid     = threadIdx.x;
    const int warp_id = tid >> 5;
    const int row     = tid / THREADS_PER_ROW;
    const int thread  = tid % THREADS_PER_ROW;

    // ==================================================================================
    // Layout:
    //   Input:     [M, D]              (row-major, offset via start_m)
    //   Weight:    [V, D]              (row-major, offset via v_tile)
    //   Logits:    [BLOCK_M, BLOCK_N]  (computed tile)
    // ==================================================================================
    const __half* __restrict__ input_ptr  = Input  + (size_t)start_m * D;
    const __half* __restrict__ weight_ptr = Weight + (size_t)v_tile  * D;

    // ==================================================================================
    // Init:   shared memory with zero-fill union regions to avoid stale data
    // ==================================================================================
    extern __shared__ char smem_raw[];

    auto& smem = *reinterpret_cast<typename Config::SmemLayout*>(smem_raw);

    __half* __restrict__ sInput  = smem.input;
    __half* __restrict__ sWeight = smem.weight;
     float* __restrict__ sLogits = smem.logits;

    WMMA_GEMM_INIT_SMEM<Config>(smem.logits, 0.0f);
    __syncthreads();

    // ==================================================================================
    // Compute:  Logits += Input @ Weight^T
    // Layout:   Input[row: BLOCK_M, DIM], Weight[col: BLOCK_N, DIM] -> Logits[row: BLOCK_M, col: BLOCK_N]
    // Template: WMMA_M=16, WMMA_N=16, WMMA_K=16
    // ==================================================================================
    constexpr int num_tiles_m    = (BLOCK_M + WMMA_M - 1) / WMMA_M;
    constexpr int num_tiles_n    = (BLOCK_N + WMMA_N - 1) / WMMA_N;
    constexpr int total_tiles    = num_tiles_m * num_tiles_n;
    constexpr int tiles_per_warp = (total_tiles + WARPS_PER_BLOCK - 1) / WARPS_PER_BLOCK;

    fragment<matrix_a,    WMMA_M, WMMA_N, WMMA_K, half, row_major> a_frag[tiles_per_warp];
    fragment<matrix_b,    WMMA_M, WMMA_N, WMMA_K, half, col_major> b_frag[tiles_per_warp];
    fragment<accumulator, WMMA_M, WMMA_N, WMMA_K, float>         acc_frag[tiles_per_warp];

    #pragma unroll
    for (int idx = 0; idx < tiles_per_warp; ++idx)
        fill_fragment(acc_frag[idx], 0.0f);

    // ==================================================================================
    // MAIN LOOP (iterates over D-tiles)
    // ==================================================================================
    for (int k_tile = 0; k_tile < D; k_tile += DIM) {
        // ==================================================================================
        // Load:     Input and Weight tiles from global to sInput/sWeight shared memory
        // Layout:   Input[Weight]: global[row: valid_m(valid_v), D] -> shared[row: BLOCK_M(BLOCK_N), DIM]
        // Template: GemmPhase::LCDX, Stage::SNGL
        // ==================================================================================
        WMMA_GEMM_LOAD_TILE<Config, GemmPhase::LCDX, Stage::SNGL>(
          input_ptr  + k_tile, sInput,
          nullptr, nullptr,
          D, valid_m, tid);
        WMMA_GEMM_LOAD_TILE<Config, GemmPhase::LCDX, Stage::SNGL>(
          weight_ptr + k_tile, sWeight,
          nullptr, nullptr,
          D, valid_v, tid);
        __syncthreads();

        #pragma unroll
        for (int idx = 0; idx < tiles_per_warp; ++idx) {
            const int tile_idx = warp_id + idx * WARPS_PER_BLOCK;
            if (tile_idx >= total_tiles) break;

            const int tile_m = (tile_idx / num_tiles_n) * WMMA_M;
            const int tile_n = (tile_idx % num_tiles_n) * WMMA_N;
            if (tile_m >= valid_m || tile_n >= valid_v) continue;

            #pragma unroll
            for (int k = 0; k < DIM / WMMA_K; ++k) {
                load_matrix_sync(a_frag[idx], sInput  + tile_m * DIM + (k * WMMA_K), DIM);
                load_matrix_sync(b_frag[idx], sWeight + tile_n * DIM + (k * WMMA_K), DIM);
                mma_sync(acc_frag[idx], a_frag[idx], b_frag[idx], acc_frag[idx]);
            }
        }
        __syncthreads();
    }

    // ==================================================================================
    // Epilogue: Store accumulated fragments -> sLogits
    // Layout:   acc_frag -> sLogits[row: BLOCK_M, col: BLOCK_N]
    // ==================================================================================
    #pragma unroll
    for (int idx = 0; idx < tiles_per_warp; ++idx) {
        const int tile_idx = warp_id + idx * WARPS_PER_BLOCK;
        if (tile_idx >= total_tiles) break;
        const int tile_m = (tile_idx / num_tiles_n) * WMMA_M;
        const int tile_n = (tile_idx % num_tiles_n) * WMMA_N;
        if (tile_m >= valid_m || tile_n >= valid_v) continue;
        store_matrix_sync(sLogits + tile_m * BLOCK_N + tile_n, acc_frag[idx], BLOCK_N, mem_row_major);
    }
    __syncthreads();

    // ==================================================================================
    // Compute:  Partial row-wise reductions (Max, Sum, SumLogits)
    // Layout:   Logits[row: BLOCK_M, BLOCK_N] -> thread_{max, sum, slogits}
    // Features: Vectorized float4 loads, warp-level shuffle reductions
    // ==================================================================================    
    float thread_max     = NEG_INF;
    float thread_sum     = 0.0f;
    float thread_slogits = 0.0f;

    float4 sSL[MAX_ITERS];
    float  sSL_tail[MAX_TAILS];

    if (row < valid_m) {
        const uint32_t sSL_base = __cvta_generic_to_shared(sLogits + row * BLOCK_N);
        #pragma unroll
        for (int idx = 0; idx < MAX_ITERS; ++idx) {
            const int addr = thread + idx * THREADS_PER_ROW;
            if (addr < (valid_v >> 2)) {
                sSL[idx]  = ld_float4(sSL_base + addr * 16, row);
                sSL[idx].x *= logit_scale;
                sSL[idx].y *= logit_scale;
                sSL[idx].z *= logit_scale;
                sSL[idx].w *= logit_scale;
                thread_max  = fmaxf(thread_max, fmaxf(fmaxf(sSL[idx].x, sSL[idx].y), fmaxf(sSL[idx].z, sSL[idx].w)));
                if (PartialSumLogits != nullptr)
                    thread_slogits += (sSL[idx].x + sSL[idx].y) + (sSL[idx].z + sSL[idx].w);
            }
        }

        #pragma unroll
        for (int idx = 0; idx < MAX_TAILS; ++idx) {
            const int addr = ((valid_v >> 2) << 2) + thread + idx * THREADS_PER_ROW;
            if (addr < valid_v) {
                sSL_tail[idx]   = ld_float(sSL_base + addr * 4, row) * logit_scale;
                thread_max      = fmaxf(thread_max, sSL_tail[idx]);
                if (PartialSumLogits != nullptr)
                    thread_slogits += sSL_tail[idx];
            }
        }
    }

    #pragma unroll
    for (int offset = THREADS_PER_ROW / 2; offset > 0; offset >>= 1) {
        thread_max = fmaxf(thread_max, __shfl_xor_sync(0xFFFFFFFFu, thread_max, offset, THREADS_PER_ROW));
    }

    if (row < valid_m) {
        #pragma unroll
        for (int idx = 0; idx < MAX_ITERS; ++idx) {
            const int addr = thread + idx * THREADS_PER_ROW;
            if (addr < (valid_v >> 2)) {
                float e = __expf(fmaxf(sSL[idx].x - thread_max, -80.0f));
                e      += __expf(fmaxf(sSL[idx].y - thread_max, -80.0f));
                e      += __expf(fmaxf(sSL[idx].z - thread_max, -80.0f));
                e      += __expf(fmaxf(sSL[idx].w - thread_max, -80.0f));
                thread_sum += e;
            }
        }

        #pragma unroll
        for (int idx = 0; idx < MAX_TAILS; ++idx) {
            const int addr = ((valid_v >> 2) << 2) + thread + idx * THREADS_PER_ROW;
            if (addr < valid_v) {
                thread_sum += __expf(fmaxf(sSL_tail[idx] - thread_max, -80.0f));
            }
        }
    }

    #pragma unroll
    for (int offset = THREADS_PER_ROW / 2; offset > 0; offset >>= 1) {
        thread_sum     += __shfl_xor_sync(0xFFFFFFFFu, thread_sum, offset, THREADS_PER_ROW);
        if (PartialSumLogits != nullptr)
            thread_slogits += __shfl_xor_sync(0xFFFFFFFFu, thread_slogits, offset, THREADS_PER_ROW);
    }

    // ==================================================================================
    // Store:    Partial reduction results
    // Layout:   thread_{max, sum, slogits} -> Partial*[start_m + row, blockIdx.y]
    // ==================================================================================
    if (thread == 0 && row < valid_m) {
        const size_t idx = (size_t)(start_m + row) * stride_v + blockIdx.y;
        PartialMax[idx] = thread_max;
        PartialSum[idx] = thread_sum;
        if (PartialSumLogits != nullptr)
            PartialSumLogits[idx] = thread_slogits;
    }

    // ==================================================================================
    // Compute & Store: Target logit extraction
    // Layout:   sLogits[row, local_target] -> TargetLogit[start_m + row]
    // ==================================================================================
    if (row < valid_m) {
        const int target_idx   = Target[start_m + row];
        const int local_target = target_idx - v_tile;
        const bool valid_tgt   = (target_idx != ignore_index) && (local_target >= 0) && (local_target < valid_v);
        if (valid_tgt && thread == 0) {
            const uint32_t addr = __cvta_generic_to_shared(&sLogits[row * BLOCK_N + local_target]);
            TargetLogit[start_m + row] = ld_float(addr, row) * logit_scale;
        }
    }
}

// ======================================================================================
// KERNEL 2: Loss / LSE / Z_Loss = FinalReduce(PartialMax, PartialSum, PartialSumLogits, TargetLogit)
// ======================================================================================
template<typename Config>
__global__ void __launch_bounds__(LogitsConfig::MAX_THREADS * MAX_THREADS_PER_WARP, 4)
forward_lce_reduce(
    const float* __restrict__ PartialMax,
    const float* __restrict__ PartialSum,
    const float* __restrict__ PartialSumLogits,
    const float* __restrict__ TargetLogit,
    const int*   __restrict__ Target,
          float* __restrict__ Loss,
          float* __restrict__ Logsumexp,
          float* __restrict__ Z_Loss,
    int   M,
    int   V,
    int   n_vblocks,
    int   stride_v,
    float label_smooth,
    float lse_sscale,
    int   ignore_index
) {
    constexpr int MAX_THREADS = Config::MAX_THREADS;

    // ==================================================================================
    // Init:   thread/warp/lane IDs for WMMA coordination
    // ==================================================================================
    const int warp_id = threadIdx.x >> 5;
    const int lane_id = threadIdx.x & 31;
    const int row     = blockIdx.x * MAX_THREADS + warp_id;

    if (row >= M) return;

    // ==================================================================================
    // Layout:
    //   Partial*:    [M, stride_v]  (row-major, offset via row)
    //   Target:      [M]
    //   TargetLogit: [M]
    // ==================================================================================
    const float* __restrict__ pmax = PartialMax + (size_t)row * stride_v;
    const float* __restrict__ psum = PartialSum + (size_t)row * stride_v;
    const float* __restrict__ psl  = (PartialSumLogits != nullptr) ? PartialSumLogits + (size_t)row * stride_v : nullptr;

    float4 rPmax[MAX_THREADS];
    float thread_max     = NEG_INF;
    float thread_sum     = 0.0f;
    float thread_slogits = 0.0f;

    // ==================================================================================
    // Load & Compute: Final Max (log-sum-exp trick)
    // Layout:   PartialMax[row, n_vblocks] -> thread_max
    // Features: Vectorized float4 loads, warp-level shuffle reductions
    // ==================================================================================
    #pragma unroll
    for (int idx = 0; idx < MAX_THREADS; ++idx) {
        const int addr = lane_id + idx * 32;
        if (addr < (n_vblocks >> 2)) {
            rPmax[idx] = reinterpret_cast<const float4*>(pmax)[addr];
            thread_max = fmaxf(thread_max, fmaxf(fmaxf(rPmax[idx].x, rPmax[idx].y), fmaxf(rPmax[idx].z, rPmax[idx].w)));
        }
    }

    for (int idx = ((n_vblocks >> 2) << 2) + lane_id; idx < n_vblocks; idx += 32)
        thread_max = fmaxf(thread_max, pmax[idx]);

    #pragma unroll
    for (int offset = 16; offset > 0; offset >>= 1)
        thread_max = fmaxf(thread_max, __shfl_xor_sync(0xFFFFFFFFu, thread_max, offset));

    // ==================================================================================
    // Compute:  Final Sum + SumLogits (rescaled by exp(partial_max - thread_max))
    // Layout:   PartialSum(PartialSumLogits)[row, n_vblocks] -> thread_{sum, slogits}
    // ==================================================================================
    #pragma unroll
    for (int idx = 0; idx < MAX_THREADS; ++idx) {
        const int addr = lane_id + idx * 32;
        if (addr < (n_vblocks >> 2)) {
            const float4 ps = reinterpret_cast<const float4*>(psum)[addr];
            thread_sum += ps.x * __expf(rPmax[idx].x - thread_max)
                        + ps.y * __expf(rPmax[idx].y - thread_max)
                        + ps.z * __expf(rPmax[idx].z - thread_max)
                        + ps.w * __expf(rPmax[idx].w - thread_max);
            if (PartialSumLogits != nullptr) {
                const float4 pl = reinterpret_cast<const float4*>(psl)[addr];
                thread_slogits += (pl.x + pl.y) + (pl.z + pl.w);
            }
        }
    }

    for (int idx = ((n_vblocks >> 2) << 2) + lane_id; idx < n_vblocks; idx += 32) {
        thread_sum += psum[idx] * __expf(pmax[idx] - thread_max);
        if (PartialSumLogits != nullptr)
            thread_slogits += psl[idx];
    }

    #pragma unroll
    for (int offset = 16; offset > 0; offset >>= 1) {
        thread_sum     += __shfl_xor_sync(0xFFFFFFFFu, thread_sum, offset);
        if (PartialSumLogits != nullptr)
            thread_slogits += __shfl_xor_sync(0xFFFFFFFFu, thread_slogits, offset);
    }

    if (lane_id != 0) return;

    // ==================================================================================
    // Compute:  Final Cross-Entropy Loss, LSE, and Z-Loss
    // Features: Label smoothing, Z-loss regularization, Ignore index masking
    // ==================================================================================
    const float LSE       = thread_max + __logf(fmaxf(thread_sum, 1e-24f));
    const int   target    = Target[row];
    const float tgt_logit = TargetLogit[row];
    const bool  valid     = (target != ignore_index) && (target >= 0) && (target < V);

    float loss_val = 0.0f, z_loss_val = 0.0f;

    if (valid) {
        const float smooth = (PartialSumLogits != nullptr)
                           ? (label_smooth * thread_slogits / (float)V) : 0.0f;
        loss_val   = LSE - (1.0f - label_smooth) * tgt_logit - smooth;
        z_loss_val = (lse_sscale > 0.0f) ? (lse_sscale * LSE * LSE) : 0.0f;
    }

    // ==================================================================================
    // Store:    Final Loss, LSE, and Z-Loss to global memory
    // ==================================================================================
    Loss[row]      = loss_val + z_loss_val;
    Logsumexp[row] = LSE;
    Z_Loss[row]    = z_loss_val;
}

// ======================================================================================
// KERNEL 1: dLogits[M, V_pad] = SoftmaxDerivative(Input @ Weightᵀ)
// ======================================================================================
template<typename Config>
__global__ void __launch_bounds__(Config::THREADS_PER_BLOCK, 1)
backward_lce_dlogits(
    const __half* __restrict__ Input,
    const __half* __restrict__ Weight,
    const   int*  __restrict__ Target,
    const float*  __restrict__ LSE,
    const float*  __restrict__ GradOut,
          __half* __restrict__ dLogits,
    int M,
    int V,
    int V_pad,
    int D,
    int reduction,
    float logit_scale,
    float label_smooth,
    float lse_sscale,
    int ignore_index
) {
    constexpr int BLOCK_M         = Config::BLOCK_M;
    constexpr int BLOCK_N         = Config::BLOCK_N;
    constexpr int DIM             = Config::DIM;
    constexpr int WARPS_PER_BLOCK = Config::WARPS_PER_BLOCK;
    constexpr int THREADS_PER_ROW = Config::THREADS_PER_ROW;
    constexpr int MAX_ITERS       = Config::MAX_ITERS;
    constexpr int MAX_TAILS       = Config::MAX_TAILS;

    // ==================================================================================
    // Grid Mapping: X for M-blocks, Y for V-blocks
    // ==================================================================================
    const int start_m = blockIdx.x * BLOCK_M;
    const int start_v = blockIdx.y * BLOCK_N;

    if (start_m >= M || start_v >= V) return;

    const int valid_m = min(BLOCK_M, M - start_m);
    const int valid_v = min(BLOCK_N, V - start_v);

    // ==================================================================================
    // Init:   thread/warp/lane IDs for WMMA coordination
    // ==================================================================================
    const int tid     = threadIdx.x;
    const int warp_id = tid >> 5;
    const int row     = tid / THREADS_PER_ROW;
    const int thread  = tid % THREADS_PER_ROW;

    // ==================================================================================
    // Layout:
    //   Input:      [M, D]              (row-major, offset via start_m)
    //   Weight:     [V, D]              (row-major, offset via start_v)
    //   dLogits:    [M, V_pad]          (row-major, offset via start_m + row)
    //   LSE:        [M]
    //   GradOut:    [M] (reduction==0) | [1] (reduction!=0)
    // ==================================================================================
    const __half* __restrict__ input_ptr   = Input   + (size_t)start_m * D;
    const __half* __restrict__ weight_ptr  = Weight  + (size_t)start_v * D;
          __half* __restrict__ dLogits_ptr = dLogits + (size_t)(start_m + row) * V_pad;

    // ==================================================================================
    // Init:   shared memory with zero-fill union regions to avoid stale data
    // ==================================================================================
    extern __shared__ char smem_raw[];

    auto& smem = *reinterpret_cast<typename Config::SmemLayout*>(smem_raw);

    __half* __restrict__ sInput  = smem.input;
    __half* __restrict__ sWeight = smem.weight;
    float*  __restrict__ sLogits = smem.logits;

    constexpr int num_tiles_m    = (BLOCK_M + WMMA_M - 1) / WMMA_M;
    constexpr int num_tiles_n    = (BLOCK_N + WMMA_N - 1) / WMMA_N;
    constexpr int num_tiles_k    = DIM / WMMA_K;    
    constexpr int total_tiles    = num_tiles_m * num_tiles_n;
    constexpr int tiles_per_warp = (total_tiles + WARPS_PER_BLOCK - 1) / WARPS_PER_BLOCK;

    fragment<matrix_a,    WMMA_M, WMMA_N, WMMA_K, half, row_major> a_frag[tiles_per_warp];
    fragment<matrix_b,    WMMA_M, WMMA_N, WMMA_K, half, col_major> b_frag[tiles_per_warp];
    fragment<accumulator, WMMA_M, WMMA_N, WMMA_K, float>         acc_frag[tiles_per_warp];

    #pragma unroll
    for (int idx = 0; idx < tiles_per_warp; ++idx)
        fill_fragment(acc_frag[idx], 0.0f);

    // ==================================================================================
    // MAIN LOOP (iterates over D-tiles)
    // ==================================================================================
    for (int k_tile = 0; k_tile < D; k_tile += DIM) {
        // ==================================================================================
        // Load:     Input and Weight tiles from global to sInput/sWeight shared memory
        // Layout:   Input[Weight]: global[row: valid_m(valid_v), D] -> shared[row: BLOCK_M(BLOCK_N), DIM]
        // Template: GemmPhase::LCDX, Stage::SNGL
        // ==================================================================================
        WMMA_GEMM_LOAD_TILE<Config, GemmPhase::LCDX, Stage::SNGL>(
          input_ptr  + k_tile, sInput,
          nullptr, nullptr,
          D, valid_m, tid);
        WMMA_GEMM_LOAD_TILE<Config, GemmPhase::LCDX, Stage::SNGL>(
          weight_ptr + k_tile, sWeight,
          nullptr, nullptr,
          D, valid_v, tid);
        __syncthreads();

        // ==================================================================================
        // Compute:  Logits += Input @ Weight^T
        // Layout:   Input[row: BLOCK_M, DIM], Weight[col: BLOCK_N, DIM] -> Logits[row: BLOCK_M, col: BLOCK_N]
        // Template: WMMA_M=16, WMMA_N=16, WMMA_K=16
        // ==================================================================================
        #pragma unroll
        for (int idx = 0; idx < tiles_per_warp; ++idx) {
            const int tile_index = warp_id + idx * WARPS_PER_BLOCK;
            if (tile_index >= total_tiles) break;

            const int tile_m = (tile_index / num_tiles_n) * WMMA_M;
            const int tile_n = (tile_index % num_tiles_n) * WMMA_N;
            if (tile_m >= valid_m || tile_n >= valid_v) continue;

            #pragma unroll
            for (int k = 0; k < num_tiles_k; ++k) {
                load_matrix_sync(a_frag[idx], sInput  + tile_m * DIM + (k * WMMA_K), DIM);
                load_matrix_sync(b_frag[idx], sWeight + tile_n * DIM + (k * WMMA_K), DIM);
                mma_sync(acc_frag[idx], a_frag[idx], b_frag[idx], acc_frag[idx]);
            }
        }
        __syncthreads();
    }

    // ==================================================================================
    // Epilogue: Store accumulated fragments -> sLogits
    // Layout:   acc_frag -> sLogits[row: BLOCK_M, col: BLOCK_N]
    // ==================================================================================
    #pragma unroll
    for (int idx = 0; idx < tiles_per_warp; ++idx) {
        const int tile_index = warp_id + idx * WARPS_PER_BLOCK;
        if (tile_index >= total_tiles) break;
        const int tile_m = (tile_index / num_tiles_n) * WMMA_M;
        const int tile_n = (tile_index % num_tiles_n) * WMMA_N;
        if (tile_m >= valid_m || tile_n >= valid_v) continue;
        store_matrix_sync(sLogits + tile_m * BLOCK_N + tile_n, acc_frag[idx], BLOCK_N, mem_row_major);
    }
    __syncthreads();

    if (row >= valid_m) return;

    // ==================================================================================
    // Init:   Row-wise metadata
    // ==================================================================================
    const uint32_t sLogits_base = __cvta_generic_to_shared(sLogits + row * BLOCK_N);
    const int      global_row   = start_m + row;
    const int      target       = Target[global_row];
    const bool     ignore_row   = (target == ignore_index);
    const float    lse          = LSE[global_row];
    const float    grad_scale   = (reduction == 0) ? GradOut[global_row] : GradOut[0];

    // ==================================================================================
    // Compute:  Softmax derivative (Cross-Entropy backward)
    // Layout:    Logits[row: BLOCK_M, BLOCK_N], LSE[row: BLOCK_M] -> dLogits[row: BLOCK_M, V_pad]
    // Features:  Label smoothing, z-loss scaling, ignore_index masking
    // ==================================================================================
    const float uniform_prob  = label_smooth / static_cast<float>(V);
    const float target_adjust = -(1.0f - label_smooth);
    const float z_loss_factor = 1.0f + 2.0f * lse_sscale * lse;
    const float scale_factor  = ignore_row ? 0.0f : grad_scale * logit_scale;

    // ==================================================================================
    // Store:    dLogits to global memory (vectorized v4.u16 body)
    // Layout:   sLogits[row, col: BLOCK_N] -> dLogits[global_row, global_v]
    // ==================================================================================
    #pragma unroll
    for (int idx = 0; idx < MAX_ITERS; ++idx) {
        const int addr = thread + idx * THREADS_PER_ROW;
        if (addr < (valid_v >> 2)) {

            const float4 v        = ld_float4(sLogits_base + addr * 16, row);
            const int    global_v = start_v + addr * 4;

            float p0 = __expf(fmaxf(v.x * logit_scale - lse, -80.0f));
            float p1 = __expf(fmaxf(v.y * logit_scale - lse, -80.0f));
            float p2 = __expf(fmaxf(v.z * logit_scale - lse, -80.0f));
            float p3 = __expf(fmaxf(v.w * logit_scale - lse, -80.0f));

            p0 = (p0 * z_loss_factor - uniform_prob) + ((global_v     == target && !ignore_row) ? target_adjust : 0.0f);
            p1 = (p1 * z_loss_factor - uniform_prob) + ((global_v + 1 == target && !ignore_row) ? target_adjust : 0.0f);
            p2 = (p2 * z_loss_factor - uniform_prob) + ((global_v + 2 == target && !ignore_row) ? target_adjust : 0.0f);
            p3 = (p3 * z_loss_factor - uniform_prob) + ((global_v + 3 == target && !ignore_row) ? target_adjust : 0.0f);

            p0 *= scale_factor;
            p1 *= scale_factor;
            p2 *= scale_factor;
            p3 *= scale_factor;

            if (global_v + 3 < V) {
                const unsigned short h0 = __half_as_ushort(__float2half_rn(p0));
                const unsigned short h1 = __half_as_ushort(__float2half_rn(p1));
                const unsigned short h2 = __half_as_ushort(__float2half_rn(p2));
                const unsigned short h3 = __half_as_ushort(__float2half_rn(p3));
                asm volatile("st.global.v4.u16 [%0], {%1,%2,%3,%4};\n"
                    :: "l"(dLogits_ptr + global_v), "h"(h0), "h"(h1), "h"(h2), "h"(h3) : "memory");
            } else {
                if (global_v     < V) dLogits_ptr[global_v]     = __float2half_rn(p0);
                if (global_v + 1 < V) dLogits_ptr[global_v + 1] = __float2half_rn(p1);
                if (global_v + 2 < V) dLogits_ptr[global_v + 2] = __float2half_rn(p2);
                if (global_v + 3 < V) dLogits_ptr[global_v + 3] = __float2half_rn(p3);
            }
        }
    }

    // ==================================================================================
    // Store:    dLogits tail (scalar half stores, < 4 elements)
    // ==================================================================================
    if (((valid_v >> 2) << 2) < valid_v) {
        #pragma unroll
        for (int idx = 0; idx < MAX_TAILS; ++idx) {
            const int addr = ((valid_v >> 2) << 2) + thread + idx * THREADS_PER_ROW;
            if (addr < valid_v) {
                const float logit_value = ld_float(sLogits_base + addr * 4, row);
                const int   global_v    = start_v + addr;
                float p = __expf(fmaxf(logit_value * logit_scale - lse, -80.0f));
                p = (p * z_loss_factor - uniform_prob) + ((global_v == target && !ignore_row) ? target_adjust : 0.0f);
                dLogits_ptr[global_v] = __float2half_rn(p * scale_factor);
            }
        }
    }
}

// ======================================================================================
// KERNEL 2/3: dX = dLogits @ W      (A_LAYOUT = row_major)
//             dW = dLogitsᵀ @ Input (A_LAYOUT = col_major)
// ======================================================================================
template<GemmType TYPE, typename Config>
__global__ void __launch_bounds__(Config::THREADS_PER_BLOCK, 2)
backward_lce_dxw(
    const __half* __restrict__ SMEM_A,
    const __half* __restrict__ SMEM_B,
          __half* __restrict__ SMEM_C,
    int M,
    int N,
    int K,
    int A_STRIDE,
    int B_STRIDE,
    int C_STRIDE
) {
    constexpr int BLOCK_M           = Config::BLOCK_M;
    constexpr int BLOCK_N           = Config::BLOCK_N;
    constexpr int DIM               = Config::DIM;
    constexpr int GRID_X            = Config::GRID_X;
    constexpr int GRID_Y            = Config::GRID_Y;
    constexpr int THREADS_PER_BLOCK = Config::THREADS_PER_BLOCK;

    // ==================================================================================
    // Grid Mapping: X for M-blocks, Y for N-blocks
    // ==================================================================================
    const int block_m = blockIdx.x * BLOCK_M;
    const int block_n = blockIdx.y * BLOCK_N;

    if (block_m >= M || block_n >= N) return;

    const int valid_m = min(BLOCK_M, M - block_m);
    const int valid_n = min(BLOCK_N, N - block_n);

    if (valid_m <= 0 || valid_n <= 0) return;

    // ==================================================================================
    // Init:   thread/warp/lane IDs for WMMA coordination
    // ==================================================================================
    const int tid     = threadIdx.x;
    const int warp_id = tid >> 5;
    const int warp_m  = warp_id / GRID_Y;
    const int warp_n  = warp_id % GRID_Y;

    // ==================================================================================
    // Init:   shared memory with zero-fill union regions to avoid stale data
    // ==================================================================================
    extern __shared__ char smem_raw[];

    auto& smem = *reinterpret_cast<typename Config::SmemLayout*>(smem_raw);

    __half* __restrict__ sdL = smem.TdLogits;
    __half* __restrict__ swI = smem.TweightInput;

    // ==================================================================================
    // Compute:  WMMA accumulation (sdL @ swI)
    // Layout:   sdL[num_tiles_m, WMMA_M, DIM], swI[DIM, num_tiles_n, WMMA_N] -> Acc
    // Template: WMMA_M=16, WMMA_N=16, WMMA_K=16
    // ==================================================================================
    constexpr int num_tiles_m    = (BLOCK_M / GRID_X) / WMMA_M;
    constexpr int num_tiles_n    = (BLOCK_N / GRID_Y) / WMMA_N;
    constexpr int tiles_per_warp = (DIM + WMMA_K - 1) / WMMA_K;

    using GTraits  = GemmTraits<TYPE, GemmPhase::LCDX>;
    using A_LAYOUT = std::conditional_t<GTraits::A_IS_COL, col_major, row_major>;

    fragment<matrix_a,    WMMA_M, WMMA_N, WMMA_K, half,  A_LAYOUT>  a_frag;
    fragment<matrix_b,    WMMA_M, WMMA_N, WMMA_K, half,  row_major> b_frag;
    fragment<accumulator, WMMA_M, WMMA_N, WMMA_K, float> acc[num_tiles_m][num_tiles_n];

    #pragma unroll
    for (int idx = 0; idx < num_tiles_m; ++idx)
        #pragma unroll
        for (int jdx = 0; jdx < num_tiles_n; ++jdx)
            fill_fragment(acc[idx][jdx], 0.0f);

    // ==================================================================================
    // MAIN LOOP (iterates over K-dimension tiles)
    // ==================================================================================
    for (int k_block = 0; k_block < ((K + DIM - 1) / DIM); ++k_block) {
        const int k_offset = k_block * DIM;
        const int k_valid  = min(DIM, K - k_offset);

        // ==================================================================================
        // Load:     dLogits and weightInput tiles from global to sdL/swI shared memory
        // Layout:   dLogits row_major: [BLOCK_M, DIM]   (smem stride DIM)
        //           dLogits col_major: [DIM, BLOCK_M]   (smem stride BLOCK_M = DIM)
        //           weightInput row_major: [K, BLOCK_N] (smem stride BLOCK_N via GemmPhase::LCDX)
        // Template: GemmPhase::LCDX, Stage::SNGL or Stage::TRNSP
        // ==================================================================================
        if constexpr (GTraits::A_IS_COL) {
            WMMA_GEMM_LOAD_TILE<Config, GemmPhase::LCDX, Stage::SNGL>(
              SMEM_A + k_offset * A_STRIDE + block_m, sdL,
              nullptr, nullptr,
              A_STRIDE, k_valid, tid);
        } else {
            WMMA_GEMM_LOAD_TILE<Config, GemmPhase::LCDX, Stage::SNGL>(
              SMEM_A + block_m * A_STRIDE + k_offset, sdL,
              nullptr, nullptr,
              A_STRIDE, valid_m, tid);
        }
        WMMA_GEMM_LOAD_TILE<Config, GemmPhase::LCDX, Stage::TRNSP>(
          SMEM_B + k_offset * B_STRIDE + block_n, swI,
          nullptr, nullptr,
          B_STRIDE, k_valid, tid);

        // ==================================================================================
        // Compute:  Swizzled zero-fill for K-tails (prevents NaNs in WMMA)
        // Layout:   sdL/swI tail rows k ∈ [k_valid, DIM) are zeroed in-place
        // ==================================================================================
        if (k_valid < DIM) {
            if constexpr (GTraits::A_IS_COL) {
                // sdL layout [DIM][BLOCK_M], row = k, Tail = rows k ∈ [k_valid, DIM)
                const int total_a = (DIM - k_valid) * BLOCK_M;
                for (int idx = tid; idx < total_a; idx += THREADS_PER_BLOCK) {
                    const int k = k_valid + (idx / BLOCK_M);
                    const int m = idx % BLOCK_M;
                    const uint32_t addr = __cvta_generic_to_shared(&sdL[k * BLOCK_M + m]);
                    st_half(addr, __float2half(0.0f), k);
                }
            } else {
                // sdL layout [BLOCK_M][DIM], row = m, Tail = columns k ∈ [k_valid, DIM)
                const int total_a = BLOCK_M * (DIM - k_valid);
                for (int idx = tid; idx < total_a; idx += THREADS_PER_BLOCK) {
                    const int m = idx / (DIM - k_valid);
                    const int k = k_valid + (idx % (DIM - k_valid));
                    const uint32_t addr = __cvta_generic_to_shared(&sdL[m * DIM + k]);
                    st_half(addr, __float2half(0.0f), m);
                }
            }
            // swI layout [DIM][BLOCK_N], row = k, Tail = rows k ∈ [k_valid, DIM)
            const int total_b = (DIM - k_valid) * BLOCK_N;
            for (int idx = tid; idx < total_b; idx += THREADS_PER_BLOCK) {
                const int k = k_valid + (idx / BLOCK_N);
                const int n = idx % BLOCK_N;
                const uint32_t addr = __cvta_generic_to_shared(&swI[k * BLOCK_N + n]);
                st_half(addr, __float2half(0.0f), k);
            }
        }
        __syncthreads();

        #pragma unroll
        for (int k_step = 0; k_step < tiles_per_warp; ++k_step) {
            #pragma unroll
            for (int idx = 0; idx < num_tiles_m; ++idx) {

                const int a_offset = GTraits::A_IS_COL
                    ? (k_step * WMMA_K) * BLOCK_M + ((warp_m * num_tiles_m + idx) * WMMA_M)
                    : ((warp_m * num_tiles_m + idx) * WMMA_M) * DIM + (k_step * WMMA_K);

                load_matrix_sync(a_frag, sdL + a_offset, GTraits::A_IS_COL ? BLOCK_M : DIM);

                #pragma unroll
                for (int jdx = 0; jdx < num_tiles_n; ++jdx) {
                    load_matrix_sync(b_frag, swI + (k_step * WMMA_K) * BLOCK_N + ((warp_n * num_tiles_n + jdx) * WMMA_N), BLOCK_N);
                    mma_sync(acc[idx][jdx], a_frag, b_frag, acc[idx][jdx]);
                }
            }
        }
        __syncthreads();
    }

    // ==================================================================================
    // Epilogue: Store accumulated fragments -> global C
    // Layout:   Acc[idx][jdx] (m16n16k16)   -> C[global_m, global_n] (half2 pairs)
    // ==================================================================================
    const int accum_row = (tid & 1) | (((tid >> 4) & 1) << 2) | (((tid >> 2) & 1) << 3);
    const int accum_col = (((tid >> 1) & 1) << 1) | (((tid >> 3) & 1) << 3);

    #pragma unroll
    for (int idx = 0; idx < num_tiles_m; ++idx) {
        #pragma unroll
        for (int jdx = 0; jdx < num_tiles_n; ++jdx) {
            const int global_m_base = block_m + (warp_m * num_tiles_m + idx) * WMMA_M;
            const int global_n_base = block_n + (warp_n * num_tiles_n + jdx) * WMMA_N;

            if (global_m_base >= M || global_n_base >= N) continue;

            #pragma unroll
            for (int e = 0; e < 8; e += 2) {
                const int global_m = global_m_base + (accum_row | (((e >> 1) & 1) << 1));
                const int global_n = global_n_base + (accum_col | (((e >> 2) & 1) << 2));

                if (global_m >= M) continue;

                const float value_lo = acc[idx][jdx].x[e];
                const float value_hi = acc[idx][jdx].x[e + 1];

                __half* dst = &SMEM_C[(size_t)global_m * C_STRIDE + global_n];

                if (global_n + 1 < N) {
                    *reinterpret_cast<__half2*>(dst) = __float22half2_rn(make_float2(value_lo, value_hi));
                } else {
                    if (global_n     < N) dst[0] = __float2half_rn(value_lo);
                    if (global_n + 1 < N) dst[1] = __float2half_rn(value_hi);
                }
            }
        }
    }
}

// ======================================================================================
// LAUNCHER
// ======================================================================================
void launcher_lce_forward(
    const at::Tensor& input,
    const at::Tensor& weight,
    const at::Tensor& target,
    at::Tensor& loss,
    at::Tensor& lse,
    at::Tensor& zloss,
    double logit_scale,
    double label_smooth,
    double lse_sscale,
    int64_t ignore_index,
    cudaStream_t stream
){
    using Config = LogitsConfig;

    constexpr int BLOCK_M = Config::BLOCK_M;
    constexpr int BLOCK_N = Config::BLOCK_N;
    constexpr size_t SMEM = Config::TOTAL_SMEM;
    constexpr int THREADS_PER_BLOCK  = Config::THREADS_PER_BLOCK;
    TORCH_CHECK(SMEM <= MAX_SMEM_PER_SM, "Shared memory exceeds 96KB for Forward lce_partial kernel: ", SMEM, " bytes (", SMEM / 1024, " KB)");

    const int   M         = static_cast<int>(input.size(0));
    const int   V         = static_cast<int>(weight.size(0));
    const int   D         = static_cast<int>(input.size(1));
    const float lscale    = static_cast<float>(logit_scale);
    const float lsmooth   = static_cast<float>(label_smooth);
    const float lsesscale = static_cast<float>(lse_sscale);
    const int   ignore    = static_cast<int>(ignore_index);

    const int n_vblocks = (V + BLOCK_N - 1) / BLOCK_N;
    const int stride_v     = (n_vblocks + 3) & ~3;

    auto opts = input.options().dtype(at::kFloat);

    auto partial_max  = at::empty({M, stride_v}, opts);
    auto partial_sum  = at::empty({M, stride_v}, opts);
    auto partial_sl   = at::empty({M, stride_v}, opts);
    auto target_logit = at::full({M}, NEG_INF,   opts);

    const float* psl_ptr  = (lsmooth != 0.0f) ? partial_sl.data_ptr<float>() : nullptr;

    // ==================================================================================
    // Launch 1: forward_lce_partial (GEMM + Partial Reductions)
    // ==================================================================================
    dim3 grid_partial((M + BLOCK_M - 1) / BLOCK_M, n_vblocks);
    dim3 threads_partial(THREADS_PER_BLOCK);

    cudaFuncSetAttribute(forward_lce_partial<Config>, cudaFuncAttributeMaxDynamicSharedMemorySize, SMEM);

    forward_lce_partial<Config><<<grid_partial, threads_partial, SMEM, stream>>>(
        reinterpret_cast<const __half*>(input.data_ptr<at::Half>()),
        reinterpret_cast<const __half*>(weight.data_ptr<at::Half>()),
        target.data_ptr<int>(),
        partial_max.data_ptr<float>(),
        partial_sum.data_ptr<float>(),
        partial_sl.data_ptr<float>(),
        target_logit.data_ptr<float>(),
        M, V, D,
        stride_v,
        lscale,
        ignore
    );

    // ==================================================================================
    // Launch 2: forward_lce_reduce (Final Reduction & Loss Computation)
    // ==================================================================================
    dim3 grid_reduce((M + (Config::MAX_THREADS * MAX_THREADS_PER_WARP) - 1) / Config::MAX_THREADS);
    dim3 threads_reduce(Config::MAX_THREADS * MAX_THREADS_PER_WARP);

    forward_lce_reduce<Config><<<grid_reduce, threads_reduce, 0, stream>>>(
        partial_max.data_ptr<float>(),
        partial_sum.data_ptr<float>(),
        psl_ptr,
        target_logit.data_ptr<float>(),
        target.data_ptr<int>(),
        loss.data_ptr<float>(),
        lse.data_ptr<float>(),
        zloss.data_ptr<float>(),
        M, V,
        n_vblocks,
        stride_v,
        lsmooth,
        lsesscale,
        ignore
    );
}

void launcher_lce_backward(
    const at::Tensor& input,
    const at::Tensor& weight,
    const at::Tensor& target,
    const at::Tensor& saved_lse,
    const at::Tensor& grad_output,
          at::Tensor& grad_input,
          at::Tensor& grad_weight,
    int64_t reduction,
    double  logit_scale,
    double  label_smooth,
    double  lse_sscale,
    int64_t ignore_index,
    cudaStream_t stream
) {
    using logits = LogitsConfig;
    using dXW = dXWConfig;

    // ==================================================================================
    // Init:   Dimensions, Shared Memory Checks, and Intermediate Tensors
    // ==================================================================================
    const int M = static_cast<int>(input.size(0));
    const int V = static_cast<int>(weight.size(0));
    const int D = static_cast<int>(input.size(1));

    constexpr int    DL_BLOCK_M           = logits::BLOCK_M;
    constexpr int    DL_BLOCK_N           = logits::BLOCK_N;
    constexpr size_t DL_SMEM              = logits::TOTAL_SMEM;
    TORCH_CHECK(DL_SMEM <= MAX_SMEM_PER_SM, "Shared memory exceeds 96KB for logits kernel: ", DL_SMEM, " bytes (", DL_SMEM / 1024, " KB)");

    constexpr int    GXW_BLOCK_M           = dXW::BLOCK_M;
    constexpr int    GXW_BLOCK_N           = dXW::BLOCK_N;
    constexpr size_t GXW_SMEM              = dXW::TOTAL_SMEM;
    TORCH_CHECK(GXW_SMEM <= MAX_SMEM_PER_SM, "Shared memory exceeds 96KB for dXW kernel: ", GXW_SMEM, " bytes (", GXW_SMEM / 1024, " KB)");

    const int V_pad = ((V + GXW_BLOCK_M - 1) / GXW_BLOCK_M) * GXW_BLOCK_M;
    auto dLogits = at::zeros({M, V_pad}, input.options().dtype(at::kHalf));

    // ==================================================================================
    // Launch 1: backward_lce_logits (GEMM + Softmax derivative)
    // ==================================================================================
    dim3 grid_logits((M + DL_BLOCK_M - 1) / DL_BLOCK_M, (V + DL_BLOCK_N - 1) / DL_BLOCK_N);
    dim3 threads_logits(logits::THREADS_PER_BLOCK);

    cudaFuncSetAttribute(backward_lce_dlogits<logits>, cudaFuncAttributeMaxDynamicSharedMemorySize, DL_SMEM);

    backward_lce_dlogits<logits><<<grid_logits, threads_logits, DL_SMEM, stream>>>(
        reinterpret_cast<const __half*>(input.data_ptr<at::Half>()),
        reinterpret_cast<const __half*>(weight.data_ptr<at::Half>()),
        target.data_ptr<int>(),
        saved_lse.data_ptr<float>(),
        grad_output.data_ptr<float>(),
        reinterpret_cast<__half*>(dLogits.data_ptr<at::Half>()),
        M,
        V,
        V_pad,
        D,
        static_cast<int>(reduction),
        static_cast<float>(logit_scale),
        static_cast<float>(label_smooth),
        static_cast<float>(lse_sscale),
        static_cast<int>(ignore_index)
    );

    // ==================================================================================
    // Launch 2: backward_lce_dxw (dX = dLogits @ Weight, A row-major)
    // ==================================================================================
    dim3 grid_dX((M + GXW_BLOCK_M - 1) / GXW_BLOCK_M, (D + GXW_BLOCK_N - 1) / GXW_BLOCK_N);
    dim3 threads_dX(dXW::THREADS_PER_BLOCK);

    cudaFuncSetAttribute(backward_lce_dxw<GemmType::dL_W, dXW>, cudaFuncAttributeMaxDynamicSharedMemorySize, GXW_SMEM);

    backward_lce_dxw<GemmType::dL_W, dXW><<<grid_dX, threads_dX, GXW_SMEM, stream>>>(
        reinterpret_cast<const __half*>(dLogits.data_ptr<at::Half>()),
        reinterpret_cast<const __half*>(weight.data_ptr<at::Half>()),
        reinterpret_cast<__half*>(grad_input.data_ptr<at::Half>()),
        M,
        D,
        V,
        V_pad,
        D,
        D
    );

    // ==================================================================================
    // Launch 3: backward_lce_dxw (dW = dLogits^T @ Input, A col-major)
    // ==================================================================================
    dim3 grid_dW((V + GXW_BLOCK_M - 1) / GXW_BLOCK_M, (D + GXW_BLOCK_N - 1) / GXW_BLOCK_N);
    dim3 threads_dW(dXW::THREADS_PER_BLOCK);

    cudaFuncSetAttribute(backward_lce_dxw<GemmType::dLT_I, dXW>, cudaFuncAttributeMaxDynamicSharedMemorySize, GXW_SMEM);

    backward_lce_dxw<GemmType::dLT_I, dXW><<<grid_dW, threads_dW, GXW_SMEM, stream>>>(
        reinterpret_cast<const __half*>(dLogits.data_ptr<at::Half>()),
        reinterpret_cast<const __half*>(input.data_ptr<at::Half>()),
        reinterpret_cast<__half*>(grad_weight.data_ptr<at::Half>()),
        V,
        D,
        M,
        V_pad,
        D,
        D
    );
}

// ======================================================================================
// WRAPPER
// ======================================================================================
std::tuple<at::Tensor, at::Tensor, at::Tensor> linear_cross_entropy_forward(
    const at::Tensor& input,
    const at::Tensor& weight,
    const at::Tensor& target,
    int64_t reduction,
    double  label_smoothing,
    double  logit_scale,
    double  lse_sscale,
    int64_t ignore_index
){
    // Device guard for multi-GPU / pipeline-parallelism
    at::cuda::CUDAGuard device_guard{input.device()};

    auto props = at::cuda::getCurrentDeviceProperties();
    TORCH_CHECK(props->major == 7 && props->minor == 0, "Kernel supports only Volta GPUs.");

    // dtype / device / contiguity
    TORCH_CHECK(input.scalar_type() == at::kHalf, "input must be FP16");
    TORCH_CHECK(weight.scalar_type() == at::kHalf, "weight must be FP16");
    TORCH_CHECK(target.scalar_type() == torch::kInt32 || target.scalar_type() == torch::kInt64, "target must be Int32 or Int64");
    TORCH_CHECK(input.is_cuda() && weight.is_cuda() && target.is_cuda(), "Tensors input, weight, target must be on CUDA");
    TORCH_CHECK(input.dim() >= 2, "input must be at least 2D");
    TORCH_CHECK(weight.dim() == 2, "weight must be 2D");
    TORCH_CHECK(input.size(-1) == weight.size(1), "input.size(-1) must match weight.size(1)");
    TORCH_CHECK(input.stride(-1) == 1, "Last dim of input must be contiguous");
    TORCH_CHECK(weight.stride(-1) == 1, "Last dim of weight must be contiguous");

    // dimensions
    const int64_t M = input.numel() / input.size(-1);
    const int64_t D = input.size(-1);
    const int64_t V = weight.size(0);

    TORCH_CHECK(M > 0, "batch size (M) must be positive");
    TORCH_CHECK(V > 0, "vocab size (V) must be positive");
    TORCH_CHECK(D > 0, "hidden dimension (D) must be positive");
    TORCH_CHECK(D % 16 == 0, "D (hidden dimension) must be a multiple of 16.");
    TORCH_CHECK(reduction == 0 || reduction == 1 || reduction == 2, "reduction must be 0 (none), 1 (mean), or 2 (sum)");
    TORCH_CHECK(label_smoothing >= 0.0 && label_smoothing < 1.0, "label_smoothing must be in [0, 1)");
    TORCH_CHECK(logit_scale > 0.0, "logit_scale must be > 0.0");

    auto input_flat  = input.reshape({-1, input.size(-1)}).contiguous();
    auto weight_c    = weight.contiguous();
    auto target_flat = target.reshape({-1}).to(torch::kInt32).contiguous();

    auto loss  = torch::empty({M}, input.options().dtype(torch::kFloat32));
    auto lse   = torch::empty({M}, input.options().dtype(torch::kFloat32));
    auto zloss = torch::empty({M}, input.options().dtype(torch::kFloat32));

    // run kernel
    auto stream = at::cuda::getCurrentCUDAStream().stream();

    launcher_lce_forward(input_flat, weight_c, target_flat, loss, lse, zloss, logit_scale, label_smoothing, lse_sscale, ignore_index, stream);

    at::Tensor final_loss;
    if (reduction == 0) {
        final_loss = loss.reshape(target.sizes());
        zloss = zloss.reshape(target.sizes());
    } else if (reduction == 1) {
        auto valid_mask  = target_flat.ne(ignore_index);
        auto valid_count = valid_mask.sum().to(torch::kFloat32).clamp_min(1.0);
        final_loss = loss.sum() / valid_count;
        zloss = zloss.sum() / valid_count;
    } else { // sum
        final_loss = loss.sum();
        zloss = zloss.sum();
    }

    return std::make_tuple(final_loss, lse, zloss);
}

std::tuple<at::Tensor, at::Tensor> linear_cross_entropy_backward(
    const at::Tensor& grad_output,
    const at::Tensor& saved_lse,
    const at::Tensor& input,
    const at::Tensor& weight,
    const at::Tensor& target,
    int64_t reduction,
    double  label_smoothing,
    double  logit_scale,
    double  lse_sscale,
    int64_t ignore_index
){
    // Device guard for multi-GPU / pipeline-parallelism
    at::cuda::CUDAGuard device_guard{input.device()};

    auto props = at::cuda::getCurrentDeviceProperties();
    TORCH_CHECK(props->major == 7 && props->minor == 0, "Kernel supports only Volta GPUs.");

    // dtype / device / contiguity
    TORCH_CHECK(input.scalar_type()  == at::kHalf,          "input must be FP16");
    TORCH_CHECK(weight.scalar_type() == at::kHalf,          "weight must be FP16");
    TORCH_CHECK(target.scalar_type() == torch::kInt32 || target.scalar_type() == torch::kInt64,
                "target must be Int32 or Int64");
    TORCH_CHECK(saved_lse.scalar_type() == torch::kFloat32, "saved_lse must be FP32");
    TORCH_CHECK(input.is_cuda() && weight.is_cuda() && target.is_cuda()
             && saved_lse.is_cuda() && grad_output.is_cuda(), "All tensors must be on CUDA");
    TORCH_CHECK(input.dim()  >= 2, "input must be at least 2D");
    TORCH_CHECK(weight.dim() == 2, "weight must be 2D");
    TORCH_CHECK(input.size(-1) == weight.size(1), "input.size(-1) must match weight.size(1)");
    TORCH_CHECK(input.stride(-1)     == 1, "Last dim of input must be contiguous");
    TORCH_CHECK(weight.stride(-1)    == 1, "Last dim of weight must be contiguous");
    TORCH_CHECK(saved_lse.stride(-1) == 1, "Last dim of saved_lse must be contiguous");

    // dimensions
    const int64_t M = input.numel() / input.size(-1);
    const int64_t D = input.size(-1);
    const int64_t V = weight.size(0);

    TORCH_CHECK(M > 0, "batch size (M) must be positive");
    TORCH_CHECK(V > 0, "vocab size (V) must be positive");
    TORCH_CHECK(D > 0, "hidden dimension (D) must be positive");
    TORCH_CHECK(D % 16 == 0, "D must be a multiple of 16.");
    TORCH_CHECK(reduction == 0 || reduction == 1 || reduction == 2, "reduction must be 0, 1, or 2");
    TORCH_CHECK(label_smoothing >= 0.0 && label_smoothing < 1.0, "label_smoothing must be in [0, 1)");
    TORCH_CHECK(logit_scale > 0.0, "logit_scale must be > 0.0");

    auto input_flat  = input.reshape({-1, input.size(-1)}).contiguous();
    auto weight_c    = weight.contiguous();
    auto target_flat = target.reshape({-1}).to(torch::kInt32).contiguous();

    auto grad_input  = torch::zeros_like(input_flat);
    auto grad_weight = torch::zeros_like(weight_c);

    int reduction_mode = 0;
    at::Tensor gout_tensor;

    if (reduction == 0) {
        gout_tensor = grad_output.to(torch::kFloat32).contiguous();
        TORCH_CHECK(gout_tensor.numel() == M,
                    "reduction=none: grad_output must have M=", M,
                    " elements, got ", gout_tensor.numel());
    } else {
        reduction_mode = 1;
        float grad_out_val = grad_output.item<float>();
        if (reduction == 1) {
            auto valid_mask    = target_flat.ne(ignore_index);
            double valid_count = valid_mask.sum().item<double>();
            grad_out_val = (valid_count > 0)
                ? grad_out_val / static_cast<float>(valid_count)
                : 0.0f;
        }
        gout_tensor = torch::full({1}, grad_out_val, input.options().dtype(torch::kFloat32));
    }

    // run kernel
    auto stream = at::cuda::getCurrentCUDAStream().stream();

    launcher_lce_backward(input_flat, weight_c, target_flat, saved_lse, gout_tensor,
                          grad_input, grad_weight, reduction_mode, logit_scale,
                          label_smoothing, lse_sscale, ignore_index, stream);

    auto final_grad_input  = grad_input.reshape_as(input);
    auto final_grad_weight = grad_weight;

    return std::make_tuple(final_grad_input, final_grad_weight);
}
