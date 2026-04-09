// ======================================================================================
// * Copyright (c) 2025, D.Skryabin / tg @ai_bond007 SPDX-License: BSD-3-Clause
// ======================================================================================
#include <cuda.h>
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <torch/extension.h>

#include <ATen/ATen.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>

#include "00_volta_const.cuh"
#include "01_backward_config.cuh"
#include "02_wmma.cuh"

// ======================================================================================
// BACKWARD KERNEL
// ======================================================================================
template<int D, bool IS_CAUSAL>
__global__ void __launch_bounds__(KernelConfig<D>::THREADS_PER_BLOCK, 2)
flash_attention_backward_kernel(
    const __half* __restrict__ Q,
    const __half* __restrict__ K,
    const __half* __restrict__ V,
    const __half* __restrict__ O,
    const __half* __restrict__ dO,
    const  float* __restrict__ softmax_lse,
          __half* __restrict__ dQ,
          __half* __restrict__ dK,
          __half* __restrict__ dV,
    const int B,
    const int H,
    const int M,
    const int N,
    const int grid_dq_limit,
    const int grid_dkv_limit,
    const float softmax_scale
) {
    // ===================================================================================
    // PHASE 1: dQ
    // ===================================================================================
    if (blockIdx.y == 0) {
        if (blockIdx.x >= grid_dq_limit) return;

        using Config = KernelConfig<D>;

        constexpr int BLOCK_M            = Config::DQ::BLOCK_M;
        constexpr int BLOCK_N            = Config::DQ::BLOCK_N;
        constexpr int THREADS_PER_BLOCK  = Config::THREADS_PER_BLOCK;
        constexpr int THREADS_PER_ROW    = Config::DQ::THREADS_PER_ROW;
        constexpr int WARPS_PER_BLOCK    = Config::DQ::WARPS_PER_BLOCK;
        constexpr int D_STRIDE           = Config::DQ::D_STRIDE;
        constexpr int N_STRIDE           = Config::DQ::N_STRIDE;

        // head index (batch * num_heads + head)
        const int batch_head_id = blockIdx.z;
        if (batch_head_id >= B * H) return;

        const int block_idx = blockIdx.x;
        const int start_q   = block_idx * BLOCK_M;
        if (start_q  >= M) return;

        int num_kv_tiles = (N + BLOCK_N - 1)  / BLOCK_N;
        const int valid_q_rows  = min(BLOCK_M, M - start_q);

        // ==================================================================================
        // Trim iteration count for causal attention: K/V blocks beyond Q position are skipped
        // Logic:    max_key_pos = start_q + valid_q_rows - 1 (last Q position in this tile)
        //           num_kv_tiles = min(original, ceil((max_key_pos + 1) / BLOCK_N))
        // ==================================================================================
        if constexpr (IS_CAUSAL) {
            const int max_key_pos = start_q + valid_q_rows - 1;
            if (max_key_pos < 0) {
               num_kv_tiles = 0;
            } else {
                num_kv_tiles = min(num_kv_tiles, (max_key_pos + BLOCK_N) / BLOCK_N);
            }
        }

        // ==================================================================================
        // Init:   thread/warp/lane IDs for WMMA coordination
        // ==================================================================================
        const int tid          = threadIdx.x;
        const int warp_id      = tid >> 5;
        const int lane_id      = tid & 31;

        // ==================================================================================
        // Layout: [B, H, M/N, D] linear offset: batch_head_id * (M/N) * D + start_* * D
        // ==================================================================================
        const __half* __restrict__ q_ptr   = Q           + (size_t)batch_head_id * M * D + start_q * D;
        const __half* __restrict__ k_ptr   = K           + (size_t)batch_head_id * N * D;
        const __half* __restrict__ v_ptr   = V           + (size_t)batch_head_id * N * D;
        const __half* __restrict__ o_ptr   = O           + (size_t)batch_head_id * M * D + start_q * D;
        const __half* __restrict__ dO_ptr  = dO          + (size_t)batch_head_id * M * D + start_q * D;
              __half* __restrict__ dQ_ptr  = dQ          + (size_t)batch_head_id * M * D + start_q * D;
        const float*  __restrict__ lse_ptr = softmax_lse + (size_t)batch_head_id * M + start_q;

        // ==================================================================================
        // Init:   shared memory with zero-fill union regions to avoid stale data
        // ==================================================================================
        extern __shared__ char smem_raw[];

        WMMA_GEMM_INIT_SMEM<Config>(smem_raw);

        __syncthreads();

        auto& smem = *reinterpret_cast<typename Config::SmemLayout*>(smem_raw);

        __half* __restrict__ sK            = smem.phase.dq.reuse_kv.k;
        __half* __restrict__ sV            = smem.phase.dq.reuse_kv.v;
        __half* __restrict__ sdO           = smem.phase.dq.dO;
        __half* __restrict__ sQ            = smem.phase.dq.q;
         float* __restrict__ sS            = smem.phase.dq.s;
         float* __restrict__ sdOV          = smem.phase.dq.reuse_sdOVS.dOV;
        __half* __restrict__ sdS           = smem.phase.dq.reuse_sdOVS.dS;
         float* __restrict__ sRowDot       = smem.row_dot;
         float* __restrict__ sLse          = smem.lse;
         float* __restrict__ sdQ           = smem.phase.dq.dQ;

        // ==================================================================================
        // Load:     Q(dO)  tile from global to sQ(sdO) shared memory
        // Layout:   Q[dO]: global[row: BLOCK_M, D] -> shared[row: BLOCK_M, D_STRIDE]
        // Template: DUAL_LOAD=true, SRC_STRIDE=D, DST_STRIDE=D_STRIDE
        // ==================================================================================
        WMMA_GEMM_LOAD_TILE<true, D, D_STRIDE>(
        q_ptr,   sQ,
        dO_ptr,  sdO,
        valid_q_rows, tid,
        THREADS_PER_BLOCK);

        __syncthreads();

        // ==================================================================================
        // Compute:  row_dot = sum(O ⊙ dO) [dQ backward pass]
        // Layout:   O[global: total_q, D], dO[shared: valid_q_rows, D_STRIDE] -> sRowDot[shared: valid_q_rows]
        // Template: TYPE=rowdot_dQ (LSE_OFFSET=0), GLOBAL_STRIDE=D, SMEM_STRIDE=D_STRIDE, FULL_ROWS=BLOCK_Y
        // ==================================================================================
        WMMA_GEMM_DOT_PRODUCT<GemmType::rowdot_dQ, D, D_STRIDE, BLOCK_M>(
        o_ptr,   sdO, lse_ptr, sLse,
        sRowDot, valid_q_rows, 0, tid,
        THREADS_PER_ROW, THREADS_PER_BLOCK);

        __syncthreads();

        // ==================================================================================
        // MAIN LOOP (iterates over K/V blocks for current Q block)
        // ==================================================================================
        for (int block = 0; block < num_kv_tiles; ++block) {
            const int start_kv = block * BLOCK_N;
            if (start_kv >= N) break;
            const int valid_kv_rows = min(BLOCK_N, N - start_kv);

            // Early skip per tile
            if constexpr (IS_CAUSAL) { if (start_kv >= start_q + valid_q_rows) continue; }

            // ==================================================================================
            // Load:     V tile from global to sV(reuse) shared memory
            // Layout:   V: global[row: BLOCK_N, D] -> shared[row: BLOCK_N, D_STRIDE]
            // Template: DUAL_LOAD=false, SRC_STRIDE=D, DST_STRIDE=D_STRIDE
            // ==================================================================================
            WMMA_GEMM_LOAD_TILE<false, D, D_STRIDE>(
            v_ptr + start_kv * D, sV,
            nullptr, nullptr,
            valid_kv_rows, tid,
            THREADS_PER_BLOCK);

            __syncthreads();

            // ==================================================================================
            // Compute:  dOV = dO @ V^T
            // Layout:   dO[row: BLOCK_M, D], V[col: BLOCK_N, D] -> dOV[row: BLOCK_M, col: BLOCK_N]
            // Template: BLOCK_X=BLOCK_M, BLOCK_Y=BLOCK_N
            // ==================================================================================
            WMMA_GEMM_SCORES<GemmType::dOV_dOVT, D, IS_CAUSAL, BLOCK_M, BLOCK_N, D_STRIDE, N_STRIDE, WARPS_PER_BLOCK>(
            sdO, sV, sdOV,
            valid_q_rows, valid_kv_rows,
            0, 0, 1.0f,
            warp_id, lane_id);

            __syncthreads();

            // ==================================================================================
            // Load:     K tile from global to sK(reuse) shared memory
            // Layout:   K: global[row: BLOCK_N, D] -> shared[row: BLOCK_N, D_STRIDE]
            // Template: DUAL_LOAD=false, SRC_STRIDE=D, DST_STRIDE=D_STRIDE
            // ==================================================================================
            WMMA_GEMM_LOAD_TILE<false, D, D_STRIDE>(
            k_ptr + start_kv * D,   sK,
            nullptr, nullptr,
            valid_kv_rows, tid,
            THREADS_PER_BLOCK);

            __syncthreads();

            // ==================================================================================
            // Compute:  S = Q @ K^T
            // Layout:   Q[row: BLOCK_M, D], K[col: BLOCK_N, D] -> S[row: BLOCK_M, col: BLOCK_N]
            // Template: BLOCK_X=BLOCK_M, BLOCK_Y=BLOCK_N
            // ==================================================================================
            WMMA_GEMM_SCORES<GemmType::sQ_KT, D, IS_CAUSAL, BLOCK_M, BLOCK_N, D_STRIDE, N_STRIDE, WARPS_PER_BLOCK>(
            sQ, sK, sS,
            valid_q_rows, valid_kv_rows,
            start_q,      start_kv,
            softmax_scale,
            warp_id,      lane_id);

            __syncthreads();

            // ==================================================================================
            // Compute:  dS = exp(S - lse) * (dOV - row_dot) * scale
            // Layout:   S[row: BLOCK_M, BLOCK_N], dOV[row: BLOCK_M, BLOCK_N],
            //           LSE[row: BLOCK_M], row_dot[row: BLOCK_M] -> dS[row: BLOCK_M, BLOCK_N]
            // Template: LDS_STRIDE=N_STRIDE, LDO_STRIDE=N_STRIDE, TILE_X=BLOCK_M, TILE_Y=BLOCK_N
            // ==================================================================================
            WMMA_GEMM_POST_SOFTMAX_GRADIENT<GemmType::compute_dS, N_STRIDE, N_STRIDE, BLOCK_M, BLOCK_N>(
            sS, sdOV, sLse, sRowDot,
            nullptr, sdS,
            valid_q_rows, valid_kv_rows,
            softmax_scale,
            tid,
            THREADS_PER_ROW, THREADS_PER_BLOCK);

            __syncthreads();

            // ==================================================================================
            // Compute:  dQ += dS @ K
            // Layout:   dS[row: BLOCK_M, BLOCK_N], K[row: BLOCK_N, D] -> dQ[row: BLOCK_M, D]
            // Template: BLOCK_X=BLOCK_M, BLOCK_Y=BLOCK_N
            // ==================================================================================
            WMMA_GEMM_GRADIENTS<GemmType::dQ_dSK, D, BLOCK_M, BLOCK_N, N_STRIDE, D_STRIDE, WARPS_PER_BLOCK>(
            sdS, sK, sdQ,
            valid_q_rows, valid_kv_rows,
            warp_id,      lane_id);

            __syncthreads();

        } // END MAIN LOOP

        // ==================================================================================
        // Compute:  Store gradient dQ without normalization
        // Layout:   sdQ[valid_q_rows, D_STRIDE] -> dQ_ptr[valid_q_rows, D]
        // Template: D, D_STRIDE Head dimension and stride
        // ==================================================================================
        WMMA_GEMM_EPILOGUE<GemmType::write_dQ, D, D_STRIDE>(
        sdQ,     dQ_ptr,
        nullptr, nullptr,
        nullptr,
        valid_q_rows, tid,
        THREADS_PER_BLOCK);
    }
    // ===================================================================================
    // PHASE 2: dKV
    // ===================================================================================
    else if (blockIdx.y == 1) {
        if (blockIdx.x >= grid_dkv_limit) return;

        using Config = KernelConfig<D>;

        constexpr int BLOCK_M            = Config::DKV::BLOCK_M;
        constexpr int BLOCK_N            = Config::DKV::BLOCK_N;
        constexpr int THREADS_PER_BLOCK  = Config::THREADS_PER_BLOCK;
        constexpr int THREADS_PER_ROW    = Config::DKV::THREADS_PER_ROW;
        constexpr int WARPS_PER_BLOCK    = Config::DKV::WARPS_PER_BLOCK;
        constexpr int D_STRIDE           = Config::DKV::D_STRIDE;
        constexpr int M_STRIDE           = Config::DKV::M_STRIDE;

        // head index (batch * num_heads + head)
        const int batch_head_id = blockIdx.z;
        if (batch_head_id >= B * H) return;

        const int block_idx = blockIdx.x;
        const int start_kv  = block_idx * BLOCK_M;
        if (start_kv >= N) return;

        int num_q_tiles  = (M + BLOCK_N - 1) / BLOCK_N;
        const int valid_kv_rows = min(BLOCK_M, N - start_kv);

        // ==================================================================================
        // Init:   thread/warp/lane IDs for WMMA coordination
        // ==================================================================================
        const int tid          = threadIdx.x;
        const int warp_id      = tid >> 5;
        const int lane_id      = tid & 31;

        // ==================================================================================
        // Layout: [B, H, M/N, D] linear offset: batch_head_id * (M/N) * D + start_* * D
        // ==================================================================================
        const __half* __restrict__   q_ptr = Q           + (size_t)batch_head_id * M * D;
        const __half* __restrict__   k_ptr = K           + (size_t)batch_head_id * N * D + start_kv * D;
        const __half* __restrict__   v_ptr = V           + (size_t)batch_head_id * N * D + start_kv * D;
        const __half* __restrict__   o_ptr = O           + (size_t)batch_head_id * M * D;
        const __half* __restrict__  dO_ptr = dO          + (size_t)batch_head_id * M * D;
        const  float* __restrict__ lse_ptr = softmax_lse + (size_t)batch_head_id * M;
              __half* __restrict__  dK_ptr = dK          + (size_t)batch_head_id * N * D + start_kv * D;
              __half* __restrict__  dV_ptr = dV          + (size_t)batch_head_id * N * D + start_kv * D;

        // ==================================================================================
        // Init:   shared memory with zero-fill union regions to avoid stale data
        // ==================================================================================
        extern __shared__ char smem_raw[];

        WMMA_GEMM_INIT_SMEM<Config>(smem_raw);

        __syncthreads();

        auto& smem = *reinterpret_cast<typename Config::SmemLayout*>(smem_raw);

        __half* __restrict__ sK            = smem.phase.dkv.k;
        __half* __restrict__ sV            = smem.phase.dkv.v;
        __half* __restrict__ sdO           = smem.phase.dkv.reuse_qdO.dO;
        __half* __restrict__ sQ            = smem.phase.dkv.reuse_qdO.q;
         float* __restrict__ sS            = smem.phase.dkv.reuse_sp.s;
        __half* __restrict__ sP            = smem.phase.dkv.reuse_sp.p;
         float* __restrict__ sdOV          = smem.phase.dkv.reuse_dOVS.dOV;
        __half* __restrict__ sdS           = smem.phase.dkv.reuse_dOVS.dS;
         float* __restrict__ sRowDot       = smem.row_dot;
         float* __restrict__ sLse          = smem.lse;
         float* __restrict__ sdK           = smem.phase.dkv.dK;
         float* __restrict__ sdV           = smem.phase.dkv.dV;

        // ==================================================================================
        // Load:     K(V)  tile from global to sK(sV) shared memory
        // Layout:   K[V]: global[row: BLOCK_M, D] -> shared[row: BLOCK_M, D_STRIDE]
        // Template: DUAL_LOAD=true, SRC_STRIDE=D, DST_STRIDE=D_STRIDE
        // ==================================================================================
        WMMA_GEMM_LOAD_TILE<true, D, D_STRIDE>(
        k_ptr,   sK,
        v_ptr,   sV,
        valid_kv_rows, tid,
        THREADS_PER_BLOCK);

        __syncthreads();

        // ==================================================================================
        // MAIN LOOP (iterates over Q blocks for current K/V block)
        // ==================================================================================
        for (int block = 0; block < num_q_tiles; ++block) {
            const int start_q = block * BLOCK_N;
            if (start_q >= M) break;
            const int valid_q_rows = min(BLOCK_N, M - start_q);

            // Early skip per tile
            if constexpr (IS_CAUSAL) { if (start_kv >= start_q + valid_q_rows) continue; }

            // ==================================================================================
            // Load:     Q tile from global to sQ(reuse) shared memory
            // Layout:   Q: global[row: BLOCK_N, D] -> shared[row: BLOCK_N, D_STRIDE]
            // Template: DUAL_LOAD=false, SRC_STRIDE=D, DST_STRIDE=D_STRIDE
            // ==================================================================================
            WMMA_GEMM_LOAD_TILE<false, D, D_STRIDE>(
            q_ptr + start_q * D, sQ,
            nullptr, nullptr,
            valid_q_rows, tid,
            THREADS_PER_BLOCK);

            __syncthreads();

            // ==================================================================================
            // Compute:  S = Q @ K^T
            // Layout:   Q[row: BLOCK_N, D], K[col: BLOCK_M, D] -> S[row: BLOCK_N, col: BLOCK_M]
            // Template: BLOCK_X=BLOCK_N, BLOCK_Y=BLOCK_M
            // ==================================================================================
            WMMA_GEMM_SCORES<GemmType::sQ_KT, D, IS_CAUSAL, BLOCK_N, BLOCK_M, D_STRIDE, M_STRIDE, WARPS_PER_BLOCK>(
            sQ, sK, sS,
            valid_q_rows, valid_kv_rows,
            start_q,      start_kv,
            softmax_scale,
            warp_id,      lane_id);

            __syncthreads();

            // ==================================================================================
            // Load:     dO tile from global to sdO(reuse) shared memory
            // Layout:   dO global[row: BLOCK_N, D] -> shared[row: BLOCK_N, D_STRIDE]
            // Template: DUAL_LOAD=false, SRC_STRIDE=D, DST_STRIDE=D_STRIDE
            // ==================================================================================
            WMMA_GEMM_LOAD_TILE<false, D, D_STRIDE>(
            dO_ptr + start_q * D, sdO,
            nullptr, nullptr,
            valid_q_rows, tid,
            THREADS_PER_BLOCK);

            __syncthreads();

            // ==================================================================================
            // Compute:  row_dot = sum(O ⊙ dO) [dK/dV backward pass]
            // Layout:   O[global: valid_q_rows, D] (pre-offset = start_q*D), dO[shared: valid_q_rows, D_STRIDE] -> sRowDot[shared]
            // Template: TYPE=rowdot_dKV (LSE_OFFSET=1), GLOBAL_STRIDE=D, SMEM_STRIDE=D_STRIDE, FULL_ROWS=BLOCK_Y
            // Note:     o_ptr must be pre-offset by caller (o_ptr + start_q*D), lse_ptr loaded with offset
            // ==================================================================================
            WMMA_GEMM_DOT_PRODUCT<GemmType::rowdot_dKV, D, D_STRIDE, BLOCK_N>(
            o_ptr + start_q * D, sdO,
            lse_ptr, sLse, sRowDot,
            valid_q_rows, start_q, tid,
            THREADS_PER_ROW, THREADS_PER_BLOCK);

            __syncthreads();

            // ==================================================================================
            // Compute:  dOV = dO @ V^T
            // Layout:   dO[row: BLOCK_N, D], V[col: BLOCK_M, D] -> dOV[row: BLOCK_N, col: BLOCK_M]
            // Template: BLOCK_X=BLOCK_N, BLOCK_Y=BLOCK_M
            // ==================================================================================
            WMMA_GEMM_SCORES<GemmType::dOV_dOVT, D, IS_CAUSAL, BLOCK_N, BLOCK_M, D_STRIDE, M_STRIDE, WARPS_PER_BLOCK>(
            sdO, sV, sdOV,
            valid_q_rows, valid_kv_rows,
            0, 0, 1.0f,
            warp_id, lane_id);

            __syncthreads();

            // ==================================================================================
            // Compute:  P = exp(S - lse), dS = P * (dOV - row_dot) * scale
            // Layout:   S[row: BLOCK_N, BLOCK_M], dOV[row: BLOCK_N, BLOCK_M],
            //           LSE[row: BLOCK_N], row_dot[row: BLOCK_N] -> P[row: BLOCK_N, BLOCK_M], dS[row: BLOCK_N, BLOCK_M]
            // Template: LDS_STRIDE=M_STRIDE, LDO_STRIDE=BLOCK_M, TILE_X=BLOCK_N, TILE_Y=BLOCK_M
            // ==================================================================================
            WMMA_GEMM_POST_SOFTMAX_GRADIENT<GemmType::compute_P_dS, M_STRIDE, BLOCK_M, BLOCK_N, BLOCK_M>(
            sS, sdOV, sLse, sRowDot,
            sP, sdS,
            valid_q_rows, valid_kv_rows,
            softmax_scale, tid,
            THREADS_PER_ROW, THREADS_PER_BLOCK);

            __syncthreads();

            // ==================================================================================
            // Compute:  dV += P^T @ dO
            // Layout:   P^T[col: BLOCK_M, BLOCK_N], dO[row: BLOCK_N, D] -> dV[row: BLOCK_M, D]
            // Template: BLOCK_X=BLOCK_M, BLOCK_Y=BLOCK_N
            // ==================================================================================
            WMMA_GEMM_GRADIENTS<GemmType::dV_PTdO, D, BLOCK_M, BLOCK_N, BLOCK_M, D_STRIDE, WARPS_PER_BLOCK>(
            sP, sdO, sdV,
            valid_kv_rows, valid_q_rows,
            warp_id,       lane_id);

            __syncthreads();

            // ==================================================================================
            // Load:     Q tile from global to sQ(reuse) shared memory
            // Layout:   Q: global[row: BLOCK_N, D] -> shared[row: BLOCK_N, D_STRIDE]
            // Template: DUAL_LOAD=false, SRC_STRIDE=D, DST_STRIDE=D_STRIDE
            // ==================================================================================
            WMMA_GEMM_LOAD_TILE<false, D, D_STRIDE>(
            q_ptr + start_q * D, sQ,
            nullptr, nullptr,
            valid_q_rows, tid,
            THREADS_PER_BLOCK);

            __syncthreads();

            // ==================================================================================
            // Compute:  dK += dS^T @ Q
            // Layout:   dS^T[col: BLOCK_M, BLOCK_N], Q[row: BLOCK_N, D] -> dK[row: BLOCK_M, D]
            // Template: BLOCK_X=BLOCK_M, BLOCK_Y=BLOCK_N
            // ==================================================================================
            WMMA_GEMM_GRADIENTS<GemmType::dK_dSTQ, D, BLOCK_M, BLOCK_N, BLOCK_M, D_STRIDE, WARPS_PER_BLOCK>(
            sdS, sQ, sdK,
            valid_kv_rows, valid_q_rows,
            warp_id,       lane_id);

            __syncthreads();

        } // END MAIN LOOP

        // ==================================================================================
        // Compute:  Store gradients dK + dV without normalization
        // Layout:
        //   sdK[valid_kv_rows, D_STRIDE] -> dK_ptr[valid_kv_rows, D]
        //   sdV[valid_kv_rows, D_STRIDE] -> dV_ptr[valid_kv_rows, D]
        // Template: D, D_STRIDE Head dimension and stride
        // ==================================================================================
        WMMA_GEMM_EPILOGUE<GemmType::write_dKV, D, D_STRIDE>(
        sdK,    dK_ptr,
        sdV,    dV_ptr,
        nullptr,
        valid_kv_rows, tid,
        THREADS_PER_BLOCK);
    }
}

// ======================================================================================
// LAUNCHER
// ======================================================================================
template<int D>
void launcher_flash_attention_backward(
    const torch::Tensor& Q,
    const torch::Tensor& K,
    const torch::Tensor& V,
    const torch::Tensor& O,
    const torch::Tensor& dO,
    const torch::Tensor& softmax_lse,
    torch::Tensor& dQ,
    torch::Tensor& dK,
    torch::Tensor& dV,
    float softmax_scale,
    bool is_causal,
    cudaStream_t stream
) {
    using Config = KernelConfig<D>;

    const int B = Q.size(0);
    const int H = Q.size(1);
    const int M = Q.size(2);
    const int N = K.size(2);

    const int grid_dq  = (M + Config::DQ::BLOCK_M - 1) /  Config::DQ::BLOCK_M;
    const int grid_dkv = (N + Config::DKV::BLOCK_M - 1) / Config::DKV::BLOCK_M;

    const int grid_max = (grid_dq > grid_dkv) ? grid_dq : grid_dkv;
    const dim3 grid(grid_max, 2, B * H);
    const dim3 block(Config::THREADS_PER_BLOCK);
    const size_t smem = Config::TOTAL_SMEM;

    TORCH_CHECK(smem <= MAX_SMEM_PER_SM, "Shared memory exceeds 96KB for Backward kernel: ", smem, " bytes (", smem / 1024, " KB)");

    auto kernel = is_causal ?
        (void*)flash_attention_backward_kernel<D, true> :
        (void*)flash_attention_backward_kernel<D, false>;

    cudaFuncSetAttribute(kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, smem);

    if (is_causal) {
        flash_attention_backward_kernel<D, true><<<grid, block, smem, stream>>>(
            reinterpret_cast<const __half*>(Q.data_ptr()),
            reinterpret_cast<const __half*>(K.data_ptr()),
            reinterpret_cast<const __half*>(V.data_ptr()),
            reinterpret_cast<const __half*>(O.data_ptr()),
            reinterpret_cast<const __half*>(dO.data_ptr()),
            softmax_lse.data_ptr<float>(),
            reinterpret_cast<__half*>(dQ.data_ptr()),
            reinterpret_cast<__half*>(dK.data_ptr()),
            reinterpret_cast<__half*>(dV.data_ptr()),
            B, H, M, N, grid_dq, grid_dkv, softmax_scale
        );
    } else {
        flash_attention_backward_kernel<D, false><<<grid, block, smem, stream>>>(
            reinterpret_cast<const __half*>(Q.data_ptr()),
            reinterpret_cast<const __half*>(K.data_ptr()),
            reinterpret_cast<const __half*>(V.data_ptr()),
            reinterpret_cast<const __half*>(O.data_ptr()),
            reinterpret_cast<const __half*>(dO.data_ptr()),
            softmax_lse.data_ptr<float>(),
            reinterpret_cast<__half*>(dQ.data_ptr()),
            reinterpret_cast<__half*>(dK.data_ptr()),
            reinterpret_cast<__half*>(dV.data_ptr()),
            B, H, M, N, grid_dq, grid_dkv, softmax_scale
        );
    }
}

// ======================================================================================
// WRAPPER
// ======================================================================================
std::vector<at::Tensor> flash_attention_backward(
    const at::Tensor& dout,
    const at::Tensor& q,
    const at::Tensor& k,
    const at::Tensor& v,
    const at::Tensor& out,
    const at::Tensor& softmax_lse,
    std::optional<at::Tensor>& dq_,
    std::optional<at::Tensor>& dk_,
    std::optional<at::Tensor>& dv_,
    std::optional<at::Tensor>& alibi_slopes_,
    const float p_dropout,
    const float softmax_scale,
    const bool is_causal,
    int window_size_left,
    int window_size_right,
    const float softcap,
    const bool deterministic,
    std::optional<at::Generator> gen_,
    std::optional<at::Tensor>& rng_state
) {
    // Now unsupported functions
    TORCH_CHECK(!alibi_slopes_.has_value(), "alibi_slopes not supported");
    TORCH_CHECK(p_dropout == 0.f, "dropout not supported");
    TORCH_CHECK(window_size_left == -1, "window_size_left not supported");
    TORCH_CHECK(window_size_right == -1 || (is_causal && window_size_right == 0), "window not supported");
    TORCH_CHECK(softcap == 0.f, "softcap not supported");
    TORCH_CHECK(!deterministic, "deterministic mode not supported");
    TORCH_CHECK(!gen_.has_value(), "Generator not supported");
    TORCH_CHECK(!rng_state.has_value() || rng_state->numel() == 0, "rng_state not supported");

    // Check layouts
    TORCH_CHECK(q.dtype() == torch::kFloat16, "q must be fp16");
    TORCH_CHECK(k.dtype() == torch::kFloat16, "k must be fp16");
    TORCH_CHECK(v.dtype() == torch::kFloat16, "v must be fp16");
    TORCH_CHECK(dout.dtype() == torch::kFloat16, "dout must be fp16");
    TORCH_CHECK(out.dtype() == torch::kFloat16, "out must be fp16");
    TORCH_CHECK(softmax_lse.dtype() == torch::kFloat32, "softmax_lse must be fp32");

    const auto sizes = q.sizes();
    const int B = sizes[0], H = sizes[1], M = sizes[2], D = sizes[3];
    const int N = k.size(2);
    TORCH_CHECK(D <= 256 && D % 8 == 0 && D % 2 == 0, "D must be even, <=256, multiple of 8");

    // Internal tensors
    at::Tensor dq_fp16 = dq_.has_value() ? dq_.value() : torch::empty_like(q);
    at::Tensor dk_fp16 = dk_.has_value() ? dk_.value() : torch::empty_like(k);
    at::Tensor dv_fp16 = dv_.has_value() ? dv_.value() : torch::empty_like(v);

    TORCH_CHECK(dq_fp16.dtype() == torch::kFloat16, "dq must be fp16");
    TORCH_CHECK(dk_fp16.dtype() == torch::kFloat16, "dk must be fp16");
    TORCH_CHECK(dv_fp16.dtype() == torch::kFloat16, "dv must be fp16");

    auto dsoftmax_sum = torch::empty({B, H, M}, torch::dtype(torch::kFloat32).device(q.device()));

    auto stream = at::cuda::getCurrentCUDAStream().stream();
    auto props  = at::cuda::getCurrentDeviceProperties();
    bool sm70   = props->major == 7 && props->minor == 0;
    TORCH_CHECK(sm70, "Kernel supports only Volta GPUs.");

    switch (D) {
        case 16:
            launcher_flash_attention_backward<16>(q, k, v, out, const_cast<at::Tensor&>(dout),  softmax_lse, dq_fp16, dk_fp16, dv_fp16, softmax_scale, is_causal, stream);
            break;
        case 32:
            launcher_flash_attention_backward<32>(q, k, v, out, const_cast<at::Tensor&>(dout),  softmax_lse, dq_fp16, dk_fp16, dv_fp16, softmax_scale, is_causal, stream);
            break;
        case 64:
            launcher_flash_attention_backward<64>(q, k, v, out, const_cast<at::Tensor&>(dout),  softmax_lse, dq_fp16, dk_fp16, dv_fp16, softmax_scale, is_causal, stream);
            break;
        case 128:
            launcher_flash_attention_backward<128>(q, k, v, out, const_cast<at::Tensor&>(dout), softmax_lse, dq_fp16, dk_fp16, dv_fp16, softmax_scale, is_causal, stream);
            break;
        case 256:
            launcher_flash_attention_backward<256>(q, k, v, out, const_cast<at::Tensor&>(dout), softmax_lse, dq_fp16, dk_fp16, dv_fp16, softmax_scale, is_causal, stream);
            break;
        default: TORCH_CHECK(false, "Unsupported D: ", D);
    }
    return {dq_fp16, dk_fp16, dv_fp16, dsoftmax_sum};
}
