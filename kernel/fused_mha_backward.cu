// ======================================================================================
// * Copyright (c) 2026, D.Skryabin / tg @ai_bond007 SPDX-License: BSD-3-Clause
// ======================================================================================
#include <cuda.h>
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <torch/extension.h>

#include <ATen/ATen.h>
#include <ATen/cuda/CUDAContext.h>
#include <ATen/cuda/CUDAGeneratorImpl.h>

#include "debug.h"
#include "kernel.h"
#include "backward.h"
#include "gemm_smem.h"
#include "product.h"
#include "mat_mul.h"
#include "softmax.h"
#include "template.h"

// ======================================================================================
// BACKWARD KERNEL
// ======================================================================================
template<int D, bool IS_CAUSAL, bool IS_ALIBI, bool IS_SOFTCAP, bool IS_WINDOW, bool IS_DROPOUT>
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
    const int H_Q,
    const int H_K,
    const int M,
    const int N,
    const int grid_dq,
    const int grid_dkv,
    const float  softmax_scale,
    const float  softcap,
    const float* alibi_slopes,
    int window_left,
    int window_right,
    const float p_dropout,
    const uint64_t dropout_seed,
    const uint64_t dropout_offset
) {
    // ===================================================================================
    // PHASE 1: dQ
    // ===================================================================================
    if (blockIdx.y == 0) {
        if (blockIdx.x >= grid_dq) return;

        using Config = KernelConfig<D>;

        constexpr int BLOCK_M   = Config::DQ::BLOCK_M;
        constexpr int BLOCK_N   = Config::DQ::BLOCK_N;
        constexpr int D_STRIDE  = Config::DQ::D_STRIDE;
        constexpr int N_STRIDE  = Config::DQ::N_STRIDE;

        const int batch_head_id = blockIdx.z;
        if (batch_head_id >= B * H_Q) return;

        const float alibi_slope = (alibi_slopes) ? alibi_slopes[batch_head_id % H_Q] : 0.0f;

        const int block_idx = blockIdx.x;
        const int start_q   = block_idx * BLOCK_M;
        if (start_q  >= M) return;

        const int valid_q_rows  = min(BLOCK_M, M - start_q);
        const int seqlen_offset = N - M;

        // ==================================================================================
        // Trim K/V iteration range for causal and sliding window attention (dQ phase)
        // Logic:    causal restricts K/V blocks beyond Q position (right bound)
        //           window_left restricts blocks before Q - window_left (left bound)
        //           window_right restricts blocks beyond Q + window_right (right bound)
        //           block_min/block_max define valid [start, end) K/V tile index range
        // ==================================================================================
        int block_min = 0;
        int block_max = (N + BLOCK_N - 1) / BLOCK_N;

        if constexpr (IS_CAUSAL) {
            const int max_key_pos = start_q + valid_q_rows - 1 + seqlen_offset;
            block_max = (max_key_pos < 0) ? 0 : min(block_max, (max_key_pos / BLOCK_N) + 1);
         }

        if constexpr (IS_WINDOW) {
            if (window_left >= 0) {
                const int min_key_pos = start_q + seqlen_offset - window_left;
                block_min = max(block_min, (min_key_pos > 0) ? (min_key_pos / BLOCK_N) : 0);
            }

            if (window_right >= 0) {
                const int max_key_pos_win = start_q + valid_q_rows - 1 + seqlen_offset + window_right;
                block_max = min(block_max, (max_key_pos_win >= 0) ? (max_key_pos_win / BLOCK_N) + 1 : 0);
            }
        }

        // ==================================================================================
        // Init:   thread/warp/lane IDs for WMMA coordination
        // ==================================================================================
        const int tid          = threadIdx.x;
        const int warp_id      = tid >> 5;
        const int lane_id      = tid & 31;

        // ==================================================================================
        // Layout:
        //   Q/Out/LSE: [B, H_Q, M, D] offset follows batch_head_id (Q-head space)
        //   K/V:       [B, H_K, N, D] mapped via batch_head_id % H_Q / (H_Q / H_K)
        // ==================================================================================
        const __half* __restrict__ q_ptr   = Q           + (size_t)batch_head_id * M * D + start_q * D;
        const __half* __restrict__ k_ptr   = K           + (size_t)((batch_head_id / H_Q) * H_K + (batch_head_id % H_Q) / (H_Q / H_K)) * N * D;
        const __half* __restrict__ v_ptr   = V           + (size_t)((batch_head_id / H_Q) * H_K + (batch_head_id % H_Q) / (H_Q / H_K)) * N * D;
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

        __half* __restrict__ sQ            = smem.phase.bdq.q;
        __half* __restrict__ sK            = smem.phase.bdq.reuse_kv.k;
        __half* __restrict__ sV            = smem.phase.bdq.reuse_kv.v;
         float* __restrict__ sS            = smem.phase.bdq.s;
        __half* __restrict__ sdO           = smem.phase.bdq.dO;
         float* __restrict__ sdOV          = smem.phase.bdq.reuse_sdOVS.dOV;
        __half* __restrict__ sdS           = smem.phase.bdq.reuse_sdOVS.dS;
         float* __restrict__ sRowDot       = smem.row_dot;
         float* __restrict__ sLse          = smem.lse;
         float* __restrict__ sdQ           = smem.phase.bdq.dQ;

        // ==================================================================================
        // Load:     Q(dO)  tile from global to sQ(sdO) shared memory
        // Layout:   Q[dO]: global[row: BLOCK_M, D] -> shared[row: BLOCK_M, D_STRIDE]
        // Template: DUAL_LOAD=true, SRC_STRIDE=D, DST_STRIDE=D_STRIDE
        // ==================================================================================
        WMMA_GEMM_LOAD_TILE<Config, true, D, D_STRIDE>(
        q_ptr,   sQ,
        dO_ptr,  sdO,
        valid_q_rows, tid);

        __syncthreads();

        // ==================================================================================
        // Compute:  row_dot = sum(O ⊙ dO) [dQ backward pass]
        // Layout:   O[global: total_q, D], dO[shared: valid_q_rows, D_STRIDE] -> sRowDot[shared: valid_q_rows]
        // Template: TYPE=rowdot_dQ (LSE_OFFSET=0), GLOBAL_STRIDE=D, SMEM_STRIDE=D_STRIDE
        // ==================================================================================
        WMMA_GEMM_DOT_PRODUCT<Config, GemmType::rowdot_dQ, D, D_STRIDE>(
        o_ptr,   sdO, lse_ptr, sLse,
        sRowDot, valid_q_rows, 0, tid);

        __syncthreads();

        // ==================================================================================
        // MAIN LOOP (iterates over K/V blocks for current Q block)
        // ==================================================================================
        for (int block = block_min; block < block_max; ++block) {
            const int start_kv = block * BLOCK_N;
            const int valid_kv_rows = min(BLOCK_N, N - start_kv);

            // ==================================================================================
            // Load:     V tile from global to sV(reuse) shared memory
            // Layout:   V: global[row: BLOCK_N, D] -> shared[row: BLOCK_N, D_STRIDE]
            // Template: DUAL_LOAD=false, SRC_STRIDE=D, DST_STRIDE=D_STRIDE
            // ==================================================================================
            WMMA_GEMM_LOAD_TILE<Config, false, D, D_STRIDE>(
            v_ptr + start_kv * D, sV,
            nullptr, nullptr,
            valid_kv_rows, tid);

            __syncthreads();

            // ==================================================================================
            // Compute:  dOV = dO @ V^T
            // Layout:   dO[row: BLOCK_M, D], V[col: BLOCK_N, D] -> dOV[row: BLOCK_M, col: BLOCK_N]
            // Template: BLOCK_X=BLOCK_M, BLOCK_Y=BLOCK_N
            // ==================================================================================
            WMMA_GEMM_SCORES<Config, GemmType::dOV_dOVT, D, IS_CAUSAL, IS_ALIBI, IS_SOFTCAP, IS_WINDOW, BLOCK_M, BLOCK_N, D_STRIDE, N_STRIDE>(
            sdO, sV, sdOV,
            valid_q_rows, valid_kv_rows,
            0, 0, 0, 1.0f, 0.0f, 0.0f, -1, -1,
            warp_id, lane_id);

            __syncthreads();

            // ==================================================================================
            // Load:     K tile from global to sK(reuse) shared memory
            // Layout:   K: global[row: BLOCK_N, D] -> shared[row: BLOCK_N, D_STRIDE]
            // Template: DUAL_LOAD=false, SRC_STRIDE=D, DST_STRIDE=D_STRIDE
            // ==================================================================================
            WMMA_GEMM_LOAD_TILE<Config, false, D, D_STRIDE>(
            k_ptr + start_kv * D, sK,
            nullptr, nullptr,
            valid_kv_rows, tid);

            __syncthreads();

            // ==================================================================================
            // Compute:  S = Q @ K^T
            // Layout:   Q[row: BLOCK_M, D], K[col: BLOCK_N, D] -> S[row: BLOCK_M, col: BLOCK_N]
            // Template: BLOCK_X=BLOCK_M, BLOCK_Y=BLOCK_N
            // ==================================================================================
            WMMA_GEMM_SCORES<Config, GemmType::sQ_KT, D, IS_CAUSAL, IS_ALIBI, IS_SOFTCAP, IS_WINDOW, BLOCK_M, BLOCK_N, D_STRIDE, N_STRIDE>(
            sQ, sK, sS,
            valid_q_rows,  valid_kv_rows,
            start_q,       start_kv,
            seqlen_offset,
            softmax_scale, softcap, alibi_slope, window_left, window_right,
            warp_id,       lane_id);

            __syncthreads();

            // ==================================================================================
            // Compute:  dS = exp(S - lse) * (dOV - row_dot) * scale
            // Layout:   S[row: BLOCK_M, BLOCK_N], dOV[row: BLOCK_M, BLOCK_N],
            //           LSE[row: BLOCK_M], row_dot[row: BLOCK_M] -> dS[row: BLOCK_M, BLOCK_N]
            // Template: LDS_STRIDE=N_STRIDE, LDO_STRIDE=N_STRIDE, TILE_X=BLOCK_M, TILE_Y=BLOCK_N
            // ==================================================================================
            WMMA_GEMM_SOFTMAX_GRADIENT<Config, GemmType::compute_dS, IS_SOFTCAP, IS_DROPOUT, N_STRIDE, N_STRIDE, BLOCK_M, BLOCK_N>(
            sS, sdOV, sLse, sRowDot,
            nullptr, sdS,
            valid_q_rows, valid_kv_rows,
            softmax_scale, softcap,
            p_dropout, dropout_seed, dropout_offset,
            start_q, start_kv, N, tid);

            __syncthreads();

            // ==================================================================================
            // Compute:  dQ += dS @ K
            // Layout:   dS[row: BLOCK_M, BLOCK_N], K[row: BLOCK_N, D] -> dQ[row: BLOCK_M, D]
            // Template: BLOCK_X=BLOCK_M, BLOCK_Y=BLOCK_N
            // ==================================================================================
            WMMA_GEMM_GRADIENTS<Config, GemmType::dQ_dSK, D, BLOCK_M, BLOCK_N, N_STRIDE, D_STRIDE>(
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
        WMMA_GEMM_EPILOGUE<Config, GemmType::write_dQ, D, D_STRIDE>(
        sdQ,     dQ_ptr,
        nullptr, nullptr,
        nullptr,
        valid_q_rows, tid);
    }
    // ===================================================================================
    // PHASE 2: dKV
    // ===================================================================================
    else if (blockIdx.y == 1) {
        if (blockIdx.x >= grid_dkv) return;

        using Config = KernelConfig<D>;

        constexpr int BLOCK_M   = Config::DKV::BLOCK_M;
        constexpr int BLOCK_N   = Config::DKV::BLOCK_N;
        constexpr int D_STRIDE  = Config::DKV::D_STRIDE;
        constexpr int M_STRIDE  = Config::DKV::M_STRIDE;

        const int batch_head_id = blockIdx.z;
        if (batch_head_id >= B * H_K) return;

        const int block_idx = blockIdx.x;
        const int start_kv  = block_idx * BLOCK_M;
        if (start_kv >= N) return;

        const int valid_kv_rows = min(BLOCK_M, N - start_kv);
        const int seqlen_offset = N - M;

        // ==================================================================================
        // Trim Q-tile iteration range for causal and sliding window attention
        // Logic:    in dKV phase, K/V positions are fixed (start_kv), Q tiles iterate
        //           causal restricts Q blocks before K - seqlen_offset (left bound)
        //           window_left restricts Q blocks beyond K + window_left (right bound)
        //           window_right restricts Q blocks before K - window_right (left bound)
        //           q_block_min/q_block_max define valid [start, end) Q-tile index range
        // ==================================================================================
        int q_block_min = 0;
        int q_block_max = (M + BLOCK_N - 1) / BLOCK_N;

        if constexpr (IS_CAUSAL) {
            const int min_q_pos = start_kv - seqlen_offset;
            q_block_min = max(q_block_min, (min_q_pos > 0) ? (min_q_pos / BLOCK_N) : 0);
        }

        if constexpr (IS_WINDOW) {
            if (window_right >= 0) {
                const int min_q_pos_win = start_kv - seqlen_offset - window_right;
                q_block_min = max(q_block_min, (min_q_pos_win > 0) ? (min_q_pos_win / BLOCK_N) : 0);
            }
            if (window_left >= 0) {
                const int max_q_pos_win = start_kv + valid_kv_rows - 1 - seqlen_offset + window_left;
                q_block_max = min(q_block_max, (max_q_pos_win >= 0) ? (max_q_pos_win / BLOCK_N) + 1 : 0);
            }
        }

        // ==================================================================================
        // Init:    thread/warp/lane IDs for WMMA coordination
        // ==================================================================================
        const int tid          = threadIdx.x;
        const int warp_id      = tid >> 5;
        const int lane_id      = tid & 31;

        // ==================================================================================
        // Layout:   [B, H_K, N, D] offset follows batch_head_id (KV-head space)
        // ==================================================================================
        const __half* __restrict__ k_ptr  = K  + ((size_t)((batch_head_id / H_K) * H_K + (batch_head_id % H_K)) * N * D) + start_kv * D;
        const __half* __restrict__ v_ptr  = V  + ((size_t)((batch_head_id / H_K) * H_K + (batch_head_id % H_K)) * N * D) + start_kv * D;
              __half* __restrict__ dK_ptr = dK + ((size_t)((batch_head_id / H_K) * H_K + (batch_head_id % H_K)) * N * D) + start_kv * D;
              __half* __restrict__ dV_ptr = dV + ((size_t)((batch_head_id / H_K) * H_K + (batch_head_id % H_K)) * N * D) + start_kv * D;

        // ==================================================================================
        // Init:   shared memory with zero-fill union regions to avoid stale data
        // ==================================================================================
        extern __shared__ char smem_raw[];

        WMMA_GEMM_INIT_SMEM<Config>(smem_raw);

        __syncthreads();

        auto& smem = *reinterpret_cast<typename Config::SmemLayout*>(smem_raw);

        __half* __restrict__ sQ            = smem.phase.bdkv.reuse_qdO.q;
        __half* __restrict__ sK            = smem.phase.bdkv.k;
        __half* __restrict__ sV            = smem.phase.bdkv.v;
         float* __restrict__ sS            = smem.phase.bdkv.reuse_sp.s;
        __half* __restrict__ sdO           = smem.phase.bdkv.reuse_qdO.dO;
         float* __restrict__ sdOV          = smem.phase.bdkv.reuse_dOVS.dOV;
        __half* __restrict__ sdS           = smem.phase.bdkv.reuse_dOVS.dS;
        __half* __restrict__ sP            = smem.phase.bdkv.reuse_sp.p;
         float* __restrict__ sRowDot       = smem.row_dot;
         float* __restrict__ sLse          = smem.lse;
         float* __restrict__ sdK           = smem.phase.bdkv.dK;
         float* __restrict__ sdV           = smem.phase.bdkv.dV;

        // ==================================================================================
        // Load:     K(V)  tile from global to sK(sV) shared memory
        // Layout:   K[V]: global[row: BLOCK_M, D] -> shared[row: BLOCK_M, D_STRIDE]
        // Template: DUAL_LOAD=true, SRC_STRIDE=D, DST_STRIDE=D_STRIDE
        // ==================================================================================
        WMMA_GEMM_LOAD_TILE<Config, true, D, D_STRIDE>(
        k_ptr,   sK,
        v_ptr,   sV,
        valid_kv_rows, tid);

        __syncthreads();

        // ==================================================================================
        // Q-HEADS LOOP (Iterate over Q-head groups sharing this KV-head)
        // ==================================================================================
        for (int group = 0; group < (H_Q / H_K); ++group) {

            // ==================================================================================
            // Layout:    [B, H_Q, M, D] -> offset computed from KV-head + group index
            // ==================================================================================
            const __half* __restrict__ q_ptr   = Q           + (size_t)((batch_head_id / H_K) * H_Q + (((batch_head_id % H_K) * (H_Q / H_K)) + group)) * M * D;
            const __half* __restrict__ o_ptr   = O           + (size_t)((batch_head_id / H_K) * H_Q + (((batch_head_id % H_K) * (H_Q / H_K)) + group)) * M * D;
            const __half* __restrict__ dO_ptr  = dO          + (size_t)((batch_head_id / H_K) * H_Q + (((batch_head_id % H_K) * (H_Q / H_K)) + group)) * M * D;
            const  float* __restrict__ lse_ptr = softmax_lse + (size_t)((batch_head_id / H_K) * H_Q + (((batch_head_id % H_K) * (H_Q / H_K)) + group)) * M;
            const  float           alibi_slope = (alibi_slopes) ? alibi_slopes[((batch_head_id / H_K) * H_Q + (batch_head_id % H_K) * (H_Q / H_K) + group) % H_Q] : 0.0f;

            // ==================================================================================
            // Q-TILES LOOP (Iterate over Q-tiles for the current Q-head)
            // ==================================================================================
            for (int block = q_block_min; block < q_block_max; ++block) {
                const int start_q = block * BLOCK_N;
                const int valid_q_rows = min(BLOCK_N, M - start_q);

                // ==================================================================================
                // Load:     Q tile from global to sQ(reuse) shared memory
                // Layout:   Q: global[row: BLOCK_N, D] -> shared[row: BLOCK_N, D_STRIDE]
                // Template: DUAL_LOAD=false, SRC_STRIDE=D, DST_STRIDE=D_STRIDE
                // ==================================================================================
                WMMA_GEMM_LOAD_TILE<Config, false, D, D_STRIDE>(
                q_ptr + start_q * D, sQ,
                nullptr, nullptr,
                valid_q_rows, tid);

                __syncthreads();

                // ==================================================================================
                // Compute:  S = Q @ K^T
                // Layout:   Q[row: BLOCK_N, D], K[col: BLOCK_M, D] -> S[row: BLOCK_N, col: BLOCK_M]
                // Template: BLOCK_X=BLOCK_N, BLOCK_Y=BLOCK_M
                // ==================================================================================
                WMMA_GEMM_SCORES<Config, GemmType::sQ_KT, D, IS_CAUSAL, IS_ALIBI, IS_SOFTCAP, IS_WINDOW, BLOCK_N, BLOCK_M, D_STRIDE, M_STRIDE>(
                sQ, sK, sS,
                valid_q_rows,  valid_kv_rows,
                start_q,       start_kv,
                seqlen_offset,
                softmax_scale, softcap, alibi_slope, window_left, window_right,
                warp_id,       lane_id);

                __syncthreads();

                // ==================================================================================
                // Load:     dO tile from global to sdO(reuse) shared memory
                // Layout:   dO global[row: BLOCK_N, D] -> shared[row: BLOCK_N, D_STRIDE]
                // Template: DUAL_LOAD=false, SRC_STRIDE=D, DST_STRIDE=D_STRIDE
                // ==================================================================================
                WMMA_GEMM_LOAD_TILE<Config, false, D, D_STRIDE>(
                dO_ptr + start_q * D, sdO,
                nullptr, nullptr,
                valid_q_rows, tid);

                __syncthreads();

                // ==================================================================================
                // Compute:  row_dot = sum(O ⊙ dO) [dK/dV backward pass]
                // Layout:   O[global: valid_q_rows, D] (pre-offset = start_q*D), dO[shared: valid_q_rows, D_STRIDE] -> sRowDot[shared]
                // Template: TYPE=rowdot_dKV (LSE_OFFSET=1), GLOBAL_STRIDE=D, SMEM_STRIDE=D_STRIDE, FULL_ROWS=BLOCK_Y
                // Note:     o_ptr must be pre-offset by caller (o_ptr + start_q*D), lse_ptr loaded with offset
                // ==================================================================================
                WMMA_GEMM_DOT_PRODUCT<Config, GemmType::rowdot_dKV, D, D_STRIDE>(
                o_ptr + start_q * D, sdO,
                lse_ptr, sLse, sRowDot,
                valid_q_rows, start_q, tid);

                __syncthreads();

                // ==================================================================================
                // Compute:  dOV = dO @ V^T
                // Layout:   dO[row: BLOCK_N, D], V[col: BLOCK_M, D] -> dOV[row: BLOCK_N, col: BLOCK_M]
                // Template: BLOCK_X=BLOCK_N, BLOCK_Y=BLOCK_M
                // ==================================================================================
                WMMA_GEMM_SCORES<Config, GemmType::dOV_dOVT, D, IS_CAUSAL, IS_ALIBI, IS_SOFTCAP, IS_WINDOW, BLOCK_N, BLOCK_M, D_STRIDE, M_STRIDE>(
                sdO, sV, sdOV,
                valid_q_rows, valid_kv_rows,
                0, 0, 0, 1.0f, 0.0f, 0.0f, -1, -1,
                warp_id, lane_id);

                __syncthreads();

                // ==================================================================================
                // Compute:  P = exp(S - lse), dS = P * (dOV - row_dot) * scale
                // Layout:   S[row: BLOCK_N, BLOCK_M], dOV[row: BLOCK_N, BLOCK_M],
                //           LSE[row: BLOCK_N], row_dot[row: BLOCK_N] -> P[row: BLOCK_N, BLOCK_M], dS[row: BLOCK_N, BLOCK_M]
                // Template: LDS_STRIDE=M_STRIDE, LDO_STRIDE=BLOCK_M, TILE_X=BLOCK_N, TILE_Y=BLOCK_M
                // ==================================================================================
                WMMA_GEMM_SOFTMAX_GRADIENT<Config, GemmType::compute_P_dS, IS_SOFTCAP, IS_DROPOUT, M_STRIDE, BLOCK_M, BLOCK_N, BLOCK_M>(
                sS, sdOV, sLse, sRowDot, sP, sdS,
                valid_q_rows,  valid_kv_rows,
                softmax_scale, softcap,
                p_dropout, dropout_seed, dropout_offset,
                start_q, start_kv, N, tid);

                __syncthreads();

                // ==================================================================================
                // Compute:  dV += P^T @ dO
                // Layout:   P^T[col: BLOCK_M, BLOCK_N], dO[row: BLOCK_N, D] -> dV[row: BLOCK_M, D]
                // Template: BLOCK_X=BLOCK_M, BLOCK_Y=BLOCK_N
                // ==================================================================================
                WMMA_GEMM_GRADIENTS<Config, GemmType::dV_PTdO, D, BLOCK_M, BLOCK_N, BLOCK_M, D_STRIDE>(
                sP, sdO, sdV,
                valid_kv_rows, valid_q_rows,
                warp_id,       lane_id);

                __syncthreads();

                // ==================================================================================
                // Load:     Q tile from global to sQ(reuse) shared memory
                // Layout:   Q: global[row: BLOCK_N, D] -> shared[row: BLOCK_N, D_STRIDE]
                // Template: DUAL_LOAD=false, SRC_STRIDE=D, DST_STRIDE=D_STRIDE
                // ==================================================================================
                WMMA_GEMM_LOAD_TILE<Config, false, D, D_STRIDE>(
                q_ptr + start_q * D, sQ,
                nullptr, nullptr,
                valid_q_rows, tid);

                __syncthreads();

                // ==================================================================================
                // Compute:  dK += dS^T @ Q
                // Layout:   dS^T[col: BLOCK_M, BLOCK_N], Q[row: BLOCK_N, D] -> dK[row: BLOCK_M, D]
                // Template: BLOCK_X=BLOCK_M, BLOCK_Y=BLOCK_N
                // ==================================================================================
                WMMA_GEMM_GRADIENTS<Config, GemmType::dK_dSTQ, D, BLOCK_M, BLOCK_N, BLOCK_M, D_STRIDE>(
                sdS, sQ, sdK,
                valid_kv_rows, valid_q_rows,
                warp_id,       lane_id);

                __syncthreads();
            } // END Q-TILES LOOP
        } // END Q-HEADS LOOP
        // ==================================================================================
        // Compute:  Store gradients dK + dV without normalization
        // Layout:
        //   sdK[valid_kv_rows, D_STRIDE] -> dK_ptr[valid_kv_rows, D]
        //   sdV[valid_kv_rows, D_STRIDE] -> dV_ptr[valid_kv_rows, D]
        // Template: D, D_STRIDE Head dimension and stride
        // ==================================================================================
        WMMA_GEMM_EPILOGUE<Config, GemmType::write_dKV, D, D_STRIDE>(
        sdK,    dK_ptr,
        sdV,    dV_ptr,
        nullptr,
        valid_kv_rows, tid);
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
    float softcap,
    float p_dropout,
    const float* alibi_slopes,
    int window_left,
    int window_right,
    uint64_t dropout_seed,
    uint64_t dropout_offset,
    cudaStream_t stream
) {
    using Config = KernelConfig<D>;

    const int B   = Q.size(0);
    const int H_Q = Q.size(1);
    const int H_K = K.size(1);
    const int M   = Q.size(2);
    const int N   = K.size(2);

    const int grid_dq  = (M + Config::DQ::BLOCK_M - 1) /  Config::DQ::BLOCK_M;
    const int grid_dkv = (N + Config::DKV::BLOCK_M - 1) / Config::DKV::BLOCK_M;
    const int grid_max = (grid_dq > grid_dkv) ? grid_dq : grid_dkv;

    const dim3 grid(grid_max, 2, B * H_Q);
    const dim3 block(Config::THREADS_PER_BLOCK);
    const size_t smem = Config::TOTAL_SMEM;

    TORCH_CHECK(smem <= MAX_SMEM_PER_SM, "Shared memory exceeds 96KB for Backward kernel: ", smem, " bytes (", smem / 1024, " KB)");

    bool is_alibi   = (alibi_slopes != nullptr);
    bool is_softcap = (softcap > 0.0f);
    bool is_window  = (window_left >= 0 || window_right >= 0);
    bool is_dropout = (p_dropout > 0.0f);

    dispatch_attention_features(is_causal, is_alibi, is_softcap, is_window, is_dropout,
        [&](auto CAUSAL, auto ALIBI, auto SOFTCAP, auto WINDOW, auto DROPOUT) {
            constexpr bool IS_CAUSAL  = decltype(CAUSAL)::value;
            constexpr bool IS_ALIBI   = decltype(ALIBI)::value;
            constexpr bool IS_SOFTCAP = decltype(SOFTCAP)::value;
            constexpr bool IS_WINDOW  = decltype(WINDOW)::value;
            constexpr bool IS_DROPOUT = decltype(DROPOUT)::value;

            auto kernel = flash_attention_backward_kernel<D, IS_CAUSAL, IS_ALIBI, IS_SOFTCAP, IS_WINDOW, IS_DROPOUT>;
            cudaFuncSetAttribute(kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, smem);

            kernel<<<grid, block, smem, stream>>>(
                reinterpret_cast<const __half*>(Q.data_ptr()),
                reinterpret_cast<const __half*>(K.data_ptr()),
                reinterpret_cast<const __half*>(V.data_ptr()),
                reinterpret_cast<const __half*>(O.data_ptr()),
                reinterpret_cast<const __half*>(dO.data_ptr()),
                softmax_lse.data_ptr<float>(),
                reinterpret_cast<__half*>(dQ.data_ptr()),
                reinterpret_cast<__half*>(dK.data_ptr()),
                reinterpret_cast<__half*>(dV.data_ptr()),
                B, H_Q, H_K, M, N, grid_dq, grid_dkv,
                softmax_scale, softcap, alibi_slopes, window_left, window_right,
                p_dropout, dropout_seed, dropout_offset);
        });
}

// ======================================================================================
// BACKWARD WRAPPER
// ======================================================================================
std::vector<at::Tensor> flash_attention_backward(
    const at::Tensor& dout,
    const at::Tensor& q,
    const at::Tensor& k,
    const at::Tensor& v,
    const at::Tensor& out,
    const at::Tensor& softmax_lse,
    std::optional<at::Tensor>& dq,
    std::optional<at::Tensor>& dk,
    std::optional<at::Tensor>& dv,
    std::optional<at::Tensor>& alibi_slopes,
    const float p_dropout,
    const float softmax_scale,
    const bool is_causal,
    int window_left,
    int window_right,
    const float softcap,
    const bool deterministic,
    std::optional<at::Generator> gen,
    std::optional<at::Tensor>& rng_state
) {
    // Now unsupported functions
    TORCH_CHECK(!deterministic, "Deterministic not supported in this Volta build");

    // Check layouts
    TORCH_CHECK(q.dtype() == torch::kFloat16, "q must be fp16");
    TORCH_CHECK(k.dtype() == torch::kFloat16, "k must be fp16");
    TORCH_CHECK(v.dtype() == torch::kFloat16, "v must be fp16");
    TORCH_CHECK(dout.dtype() == torch::kFloat16, "dout must be fp16");
    TORCH_CHECK(out.dtype() == torch::kFloat16, "out must be fp16");
    TORCH_CHECK(softmax_lse.dtype() == torch::kFloat32, "softmax_lse must be fp32");

    const auto sizes = q.sizes();
    const int B      = sizes[0], H_Q = sizes[1], M = sizes[2], D = sizes[3];
    const int H_K    = k.size(1);
    const int N      = k.size(2);
    TORCH_CHECK(D <= 256 && D % 8 == 0 && D % 2 == 0, "D must be even, <=256, multiple of 8");
    TORCH_CHECK(H_Q % H_K == 0, "H_Q must be divisible by H_K for GQA/MQA");

    const float* alibi = nullptr;
    if (alibi_slopes.has_value()) {
        const auto& slopes = alibi_slopes.value();
        auto sizes = slopes.sizes();
        TORCH_CHECK(slopes.dtype() == torch::kFloat32, "alibi_slopes must be fp32");
        TORCH_CHECK(slopes.is_cuda(), "alibi_slopes must be on CUDA");
        TORCH_CHECK(slopes.stride(-1) == 1, "alibi_slopes last dim must be contiguous");
        TORCH_CHECK(slopes.sizes() == torch::IntArrayRef({H_Q}) || slopes.sizes() == torch::IntArrayRef({B, H_Q}), "alibi_slopes shape must be [H_Q] or [B, H_Q]");
        alibi = slopes.data_ptr<float>();
    }

    TORCH_CHECK(p_dropout >= 0.f && p_dropout < 1.f, "p_dropout must be in [0, 1)");
    TORCH_CHECK(p_dropout == 0.f || (rng_state.has_value() && rng_state->numel() == 2), "rng_state with 2 elements (seed, offset) is required when p_dropout > 0");

    uint64_t dropout_seed   = 0;
    uint64_t dropout_offset = 0;

    if (p_dropout > 0.0f) {
        TORCH_CHECK(rng_state.has_value() && rng_state->numel() == 2, "rng_state must be provided for dropout backward pass");
        auto rng_cpu = rng_state->cpu();
        auto rng_acc = rng_cpu.accessor<int64_t, 1>();
        dropout_seed   = static_cast<uint64_t>(rng_acc[0]);
        dropout_offset = static_cast<uint64_t>(rng_acc[1]);
    }

    // Internal tensors
    at::Tensor dq_fp16 = dq.has_value() ? dq.value() : torch::empty_like(q);
    at::Tensor dk_fp16 = dk.has_value() ? dk.value() : torch::empty_like(k);
    at::Tensor dv_fp16 = dv.has_value() ? dv.value() : torch::empty_like(v);
    TORCH_CHECK(dq_fp16.dtype() == torch::kFloat16, "dq must be fp16");
    TORCH_CHECK(dk_fp16.dtype() == torch::kFloat16, "dk must be fp16");
    TORCH_CHECK(dv_fp16.dtype() == torch::kFloat16, "dv must be fp16");

    auto dsoftmax_sum = torch::empty({B, H_Q, M}, torch::dtype(torch::kFloat32).device(q.device()));

    auto stream = at::cuda::getCurrentCUDAStream().stream();
    auto props  = at::cuda::getCurrentDeviceProperties();
    bool sm70   = props->major == 7 && props->minor == 0;
    TORCH_CHECK(sm70, "Kernel supports only Volta GPUs.");

    #define LAUNCH_KERNEL(DIM) \
        launcher_flash_attention_backward<DIM>(q, k, v, out, dout, softmax_lse, dq_fp16, dk_fp16, dv_fp16, softmax_scale, is_causal, \
                                               softcap, p_dropout, alibi, window_left, window_right, dropout_seed, dropout_offset, stream);
    switch (D) {
        case 16:  LAUNCH_KERNEL(16);  break;
        case 32:  LAUNCH_KERNEL(32);  break;
        case 64:  LAUNCH_KERNEL(64);  break;
        case 128: LAUNCH_KERNEL(128); break;
        case 256: LAUNCH_KERNEL(256); break;
        default: TORCH_CHECK(false, "Unsupported D: ", D);
    }
    #undef LAUNCH_KERNEL

    return {dq_fp16, dk_fp16, dv_fp16, dsoftmax_sum};
}
