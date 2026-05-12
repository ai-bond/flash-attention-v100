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
#include <c10/cuda/CUDAGuard.h>

#include "debug.h"
#include "template.h"
#include "kernel.h"
#include "backward.h"
#include "gemm_smem.h"
#include "product.h"
#include "mat_mul.h"
#include "softmax.h"

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
    const int      grid_dq,
    const int      grid_dkv,
    const float    softmax_scale,
    const float    softcap,
    const float*   alibi_slopes,
    const int      alibi_batch,
    int            window_left,
    int            window_right,
    const float    p_dropout,
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

        // ==================================================================================
        // Grid Mapping: X for Q-blocks, Z for batch-head composite (batch * H_Q + head).
        // ==================================================================================
        const int bthd_idx     = blockIdx.z;
        const int block_idx    = blockIdx.x;

        if (bthd_idx >= B * H_Q) return;

        // ======================================================================================
        // BlockInfo: Unified metadata resolution (Dense Q-centric)
        // ======================================================================================
        BlockInfo<IS_CAUSAL, IS_WINDOW, false> block;
        block.init_q(
            block_idx,       // BLOCK_IDX:      Current Q-block index (grid.x)
            bthd_idx,        // BATCH_HEAD_ID:  Global Q-head index (batch * H_Q + head_q)
            H_Q,             // H_Q:            Number of query heads
            H_K,             // H_K:            Number of KV heads
            M,               // M:              Query sequence length
            N,               // N:              KV sequence length
            0,               // B:              Batch size (0 for dense, unused)
            BLOCK_M,         // BLOCK_M:        Tile size along Q dimension
            BLOCK_N,         // BLOCK_N:        Tile size along KV dimension
            window_left,     // WINDOW_LEFT:    Left sliding window bound (-1 if disabled)
            window_right,    // WINDOW_RIGHT:   Right sliding window bound (-1 if disabled)
            nullptr,         // CU_SEQLENS_Q:   Cumulative Q lengths (nullptr for dense)
            nullptr,         // CU_SEQLENS_K:   Cumulative KV lengths (nullptr for dense)
            nullptr          // SEQUSED_K:      Actual KV lengths override (nullptr for dense)
        );

        if (block.start_q >= M) return;

        // ==================================================================================
        // Init:   thread/warp/lane IDs for WMMA coordination
        // ==================================================================================
        const int tid     = threadIdx.x;
        const int warp_id = tid >> 5;
        const int lane_id = tid & 31;
        // Alibi slope only for valid block + batch
        const int   alibi_idx   = (alibi_batch > 0) ? (block.batch_idx * alibi_batch + block.head_idx) : block.head_idx;
        const float alibi_slope = (alibi_slopes != nullptr) ? alibi_slopes[alibi_idx] : 0.0f;

        // ==================================================================================
        // Layout:
        //   Q/Out/LSE: [B, H_Q, M, D] offset follows bthd_idx (Q-head space)
        //   K/V:       [B, H_K, N, D] mapped via bthd_idx % H_Q / (H_Q / H_K)
        // ==================================================================================
        const __half* __restrict__ q_ptr   = Q           + block.q_offset  (D, H_Q, M);
        const __half* __restrict__ k_ptr   = K           + block.kv_offset (D, H_K, N);
        const __half* __restrict__ v_ptr   = V           + block.kv_offset (D, H_K, N);
        const __half* __restrict__ o_ptr   = O           + block.q_offset  (D, H_Q, M);
        const __half* __restrict__ dO_ptr  = dO          + block.q_offset  (D, H_Q, M);
              __half* __restrict__ dQ_ptr  = dQ          + block.q_offset  (D, H_Q, M);
        const float*  __restrict__ lse_ptr = softmax_lse + block.lse_offset(H_Q, M);

        // ==================================================================================
        // Init:   shared memory with zero-fill union regions to avoid stale data
        // ==================================================================================
        extern __shared__ char smem_raw[];

        WMMA_GEMM_INIT_SMEM<Config>(smem_raw);

        __syncthreads();

        auto& smem = *reinterpret_cast<typename Config::SmemLayout*>(smem_raw);

        __half* __restrict__ sQ      = smem.phase.bdq.q;
        __half* __restrict__ sK      = smem.phase.bdq.reuse_kv.k;
        __half* __restrict__ sV      = smem.phase.bdq.reuse_kv.v;
         float* __restrict__ sS      = smem.phase.bdq.s;
        __half* __restrict__ sdO     = smem.phase.bdq.dO;
         float* __restrict__ sdOV    = smem.phase.bdq.reuse_sdOVS.dOV;
        __half* __restrict__ sdS     = smem.phase.bdq.reuse_sdOVS.dS;
         float* __restrict__ sRowDot = smem.row_dot;
         float* __restrict__ sLse    = smem.lse;
         float* __restrict__ sdQ     = smem.phase.bdq.dQ;

        // ==================================================================================
        // Load:     Q(dO)  tile from global to sQ(sdO) shared memory
        // Layout:   Q[dO]: global[row: BLOCK_M, D] -> shared[row: BLOCK_M, D_STRIDE]
        // Template: DUAL_LOAD=true, SRC_STRIDE=D, DST_STRIDE=D_STRIDE
        // ==================================================================================
        WMMA_GEMM_LOAD_TILE<Config, true, D_STRIDE>(
          q_ptr,   sQ,
          dO_ptr,  sdO,
          D, block.valid_q_rows, tid);
        __syncthreads();

        // ==================================================================================
        // Compute:  row_dot = sum(O ⊙ dO) [dQ backward pass]
        // Layout:   O[global: total_q, D], dO[shared: valid_q_rows, D_STRIDE] -> sRowDot[shared: valid_q_rows]
        // Template: TYPE=rowdot_dQ (LSE_OFFSET=0), GLOBAL_STRIDE=D, SMEM_STRIDE=D_STRIDE
        // ==================================================================================
        WMMA_GEMM_DOT_PRODUCT<Config, GemmType::rowdot_dQ, D_STRIDE>(
          o_ptr,   sdO, lse_ptr, sLse,
          sRowDot, D, block.valid_q_rows, 0, tid);
        __syncthreads();

        // ==================================================================================
        // MAIN LOOP (iterates over K/V blocks for current Q block)
        // ==================================================================================
        for (int block_q = block.block_min; block_q < block.block_max; ++block_q) {
            const int start_kv      = block_q * BLOCK_N;
            const int valid_kv_rows = min(BLOCK_N, N - start_kv);

            // ==================================================================================
            // Load:     V tile from global to sV(reuse) shared memory
            // Layout:   V: global[row: BLOCK_N, D] -> shared[row: BLOCK_N, D_STRIDE]
            // Template: DUAL_LOAD=false, SRC_STRIDE=D, DST_STRIDE=D_STRIDE
            // ==================================================================================
            WMMA_GEMM_LOAD_TILE<Config, false, D_STRIDE>(
              v_ptr + start_kv * D, sV,
              nullptr, nullptr,
              D, valid_kv_rows, tid);
            __syncthreads();

            // ==================================================================================
            // Compute:  dOV = dO @ V^T
            // Layout:   dO[row: BLOCK_M, D], V[col: BLOCK_N, D] -> dOV[row: BLOCK_M, col: BLOCK_N]
            // Template: BLOCK_X=BLOCK_M, BLOCK_Y=BLOCK_N
            // ==================================================================================
            WMMA_GEMM_SCORES<Config, GemmType::dOV_dOVT, D, IS_CAUSAL, IS_ALIBI, IS_SOFTCAP, IS_WINDOW, BLOCK_M, BLOCK_N, D_STRIDE, N_STRIDE>(
              sdO, sV, sdOV,
              block.valid_q_rows, valid_kv_rows,
              0, 0, 0, 1.0f, 0.0f, 0.0f, -1, -1, warp_id, lane_id);
            __syncthreads();

            // ==================================================================================
            // Load:     K tile from global to sK(reuse) shared memory
            // Layout:   K: global[row: BLOCK_N, D] -> shared[row: BLOCK_N, D_STRIDE]
            // Template: DUAL_LOAD=false, SRC_STRIDE=D, DST_STRIDE=D_STRIDE
            // ==================================================================================
            WMMA_GEMM_LOAD_TILE<Config, false, D_STRIDE>(
              k_ptr + start_kv * D, sK,
              nullptr, nullptr,
              D, valid_kv_rows, tid);
            __syncthreads();

            // ==================================================================================
            // Compute:  S = Q @ K^T
            // Layout:   Q[row: BLOCK_M, D], K[col: BLOCK_N, D] -> S[row: BLOCK_M, col: BLOCK_N]
            // Template: BLOCK_X=BLOCK_M, BLOCK_Y=BLOCK_N
            // ==================================================================================
            WMMA_GEMM_SCORES<Config, GemmType::sQ_KT, D, IS_CAUSAL, IS_ALIBI, IS_SOFTCAP, IS_WINDOW, BLOCK_M, BLOCK_N, D_STRIDE, N_STRIDE>(
              sQ, sK, sS,
              block.valid_q_rows,  valid_kv_rows,
              block.start_q,       start_kv,
              block.seqlen_offset,
              softmax_scale, softcap, alibi_slope, window_left, window_right, warp_id, lane_id);
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
              block.valid_q_rows, valid_kv_rows,
              softmax_scale, softcap,
              p_dropout, dropout_seed, dropout_offset,
              block.start_q, start_kv, N, tid);
            __syncthreads();

            // ==================================================================================
            // Compute:  dQ += dS @ K
            // Layout:   dS[row: BLOCK_M, BLOCK_N], K[row: BLOCK_N, D] -> dQ[row: BLOCK_M, D]
            // Template: BLOCK_X=BLOCK_M, BLOCK_Y=BLOCK_N
            // ==================================================================================
            WMMA_GEMM_GRADIENTS<Config, GemmType::dQ_dSK, D, BLOCK_M, BLOCK_N, N_STRIDE, D_STRIDE>(
              sdS, sK, sdQ,
              block.valid_q_rows, valid_kv_rows, warp_id, lane_id);
            __syncthreads();
        } // END MAIN LOOP
        // ==================================================================================
        // Compute:  Store gradient dQ without normalization
        // Layout:   sdQ[valid_q_rows, D_STRIDE] -> dQ_ptr[valid_q_rows, D]
        // Template: D, D_STRIDE Head dimension and stride
        // ==================================================================================
        WMMA_GEMM_EPILOGUE<Config, GemmType::write_dQ, D_STRIDE>(
          sdQ,     dQ_ptr,
          nullptr, nullptr,
          nullptr,
          D, block.valid_q_rows, tid);
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

        // ==================================================================================
        // Grid Mapping: X for Q-blocks, Z for batch-head composite (batch * H_Q + head).
        // ==================================================================================
        const int bthd_idx     = blockIdx.z;
        const int block_idx    = blockIdx.x;

        if (bthd_idx >= B * H_K) return;

        // ======================================================================================
        // BlockInfo: Unified metadata resolution (Dense KV-centric)
        // ======================================================================================
        BlockInfo<IS_CAUSAL, IS_WINDOW, false> block;
        block.init_kv(
            block_idx,       // BLOCK_IDX:      Current KV-block index (grid.x)
            bthd_idx,        // BATCH_HEAD_ID:  Global KV-head index (batch * H_K + head_kv)
            H_Q,             // H_Q:            Number of query heads
            H_K,             // H_K:            Number of KV heads
            M,               // M:              Query sequence length
            N,               // N:              KV sequence length
            0,               // B:              Batch size (0 for dense, unused)
            BLOCK_M,         // BLOCK_M:        Tile size along KV dimension
            BLOCK_N,         // BLOCK_N:        Tile size along Q dimension
            window_left,     // WINDOW_LEFT:    Left sliding window bound (-1 if disabled)
            window_right,    // WINDOW_RIGHT:   Right sliding window bound (-1 if disabled)
            nullptr,         // CU_SEQLENS_Q:   Cumulative Q lengths (nullptr for dense)
            nullptr,         // CU_SEQLENS_K:   Cumulative KV lengths (nullptr for dense)
            nullptr          // SEQUSED_K:      Actual KV lengths override (nullptr for dense)
        );

        if (block.start_kv >= N) return;

        // ==================================================================================
        // Init:    thread/warp/lane IDs for WMMA coordination
        // ==================================================================================
        const int tid          = threadIdx.x;
        const int warp_id      = tid >> 5;
        const int lane_id      = tid & 31;

        // ==================================================================================
        // Layout:   [B, H_K, N, D] offset follows bthd_idx (KV-head space)
        // ==================================================================================
        const __half* __restrict__ k_ptr  = K  + block.kv_offset(D, H_K, N) + block.start_kv * D;
        const __half* __restrict__ v_ptr  = V  + block.kv_offset(D, H_K, N) + block.start_kv * D;
              __half* __restrict__ dK_ptr = dK + block.kv_offset(D, H_K, N) + block.start_kv * D;
              __half* __restrict__ dV_ptr = dV + block.kv_offset(D, H_K, N) + block.start_kv * D;

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
        WMMA_GEMM_LOAD_TILE<Config, true, D_STRIDE>(
          k_ptr,   sK,
          v_ptr,   sV,
          D, block.valid_kv_rows, tid);
        __syncthreads();

        // ==================================================================================
        // Q-HEADS LOOP (Iterate over Q-head groups sharing this KV-head)
        // ==================================================================================
        for (int group = 0; group < (H_Q / H_K); ++group) {

            // ==================================================================================
            // Layout:    [B, H_Q, M, D] -> offset computed from KV-head + group index
            // ==================================================================================
            const __half* __restrict__ q_ptr   = Q           + (size_t)((bthd_idx / H_K) * H_Q + (((bthd_idx % H_K) * (H_Q / H_K)) + group)) * M * D;
            const __half* __restrict__ o_ptr   = O           + (size_t)((bthd_idx / H_K) * H_Q + (((bthd_idx % H_K) * (H_Q / H_K)) + group)) * M * D;
            const __half* __restrict__ dO_ptr  = dO          + (size_t)((bthd_idx / H_K) * H_Q + (((bthd_idx % H_K) * (H_Q / H_K)) + group)) * M * D;
            const  float* __restrict__ lse_ptr = softmax_lse + (size_t)((bthd_idx / H_K) * H_Q + (((bthd_idx % H_K) * (H_Q / H_K)) + group)) * M;
            // Alibi slope only for valid block + batch
            const int   alibi_idx   = (alibi_batch > 0) ? ((bthd_idx / H_K) * alibi_batch + ((bthd_idx % H_K) * (H_Q / H_K) + group)) : (bthd_idx % H_K) * (H_Q / H_K) + group;
            const float alibi_slope = (alibi_slopes != nullptr) ? alibi_slopes[alibi_idx] : 0.0f;

            // ==================================================================================
            // Q-TILES LOOP (Iterate over Q-tiles for the current Q-head)
            // ==================================================================================
            for (int block_q = block.block_min; block_q < block.block_max; ++block_q) {
                const int start_q      = block_q * BLOCK_N;
                const int valid_q_rows = min(BLOCK_N, M - start_q);

                // ==================================================================================
                // Load:     Q tile from global to sQ(reuse) shared memory
                // Layout:   Q: global[row: BLOCK_N, D] -> shared[row: BLOCK_N, D_STRIDE]
                // Template: DUAL_LOAD=false, SRC_STRIDE=D, DST_STRIDE=D_STRIDE
                // ==================================================================================
                WMMA_GEMM_LOAD_TILE<Config, false, D_STRIDE>(
                  q_ptr + start_q * D, sQ,
                  nullptr, nullptr,
                  D, valid_q_rows, tid);
                __syncthreads();

                // ==================================================================================
                // Compute:  S = Q @ K^T
                // Layout:   Q[row: BLOCK_N, D], K[col: BLOCK_M, D] -> S[row: BLOCK_N, col: BLOCK_M]
                // Template: BLOCK_X=BLOCK_N, BLOCK_Y=BLOCK_M
                // ==================================================================================
                WMMA_GEMM_SCORES<Config, GemmType::sQ_KT, D, IS_CAUSAL, IS_ALIBI, IS_SOFTCAP, IS_WINDOW, BLOCK_N, BLOCK_M, D_STRIDE, M_STRIDE>(
                  sQ, sK, sS,
                  valid_q_rows,  block.valid_kv_rows,
                  start_q,       block.start_kv,
                  block.seqlen_offset,
                  softmax_scale, softcap, alibi_slope, window_left, window_right, warp_id, lane_id);
                __syncthreads();

                // ==================================================================================
                // Load:     dO tile from global to sdO(reuse) shared memory
                // Layout:   dO global[row: BLOCK_N, D] -> shared[row: BLOCK_N, D_STRIDE]
                // Template: DUAL_LOAD=false, SRC_STRIDE=D, DST_STRIDE=D_STRIDE
                // ==================================================================================
                WMMA_GEMM_LOAD_TILE<Config, false, D_STRIDE>(
                  dO_ptr + start_q * D, sdO,
                  nullptr, nullptr,
                  D, valid_q_rows, tid);
                __syncthreads();

                // ==================================================================================
                // Compute:  row_dot = sum(O ⊙ dO) [dK/dV backward pass]
                // Layout:   O[global: valid_q_rows, D] (pre-offset = start_q*D), dO[shared: valid_q_rows, D_STRIDE] -> sRowDot[shared]
                // Template: TYPE=rowdot_dKV (LSE_OFFSET=1), GLOBAL_STRIDE=D, SMEM_STRIDE=D_STRIDE, FULL_ROWS=BLOCK_Y
                // Note:     o_ptr must be pre-offset by caller (o_ptr + start_q*D), lse_ptr loaded with offset
                // ==================================================================================
                WMMA_GEMM_DOT_PRODUCT<Config, GemmType::rowdot_dKV, D_STRIDE>(
                  o_ptr + start_q * D, sdO,
                  lse_ptr, sLse, sRowDot,
                  D, valid_q_rows, start_q, tid);
                __syncthreads();

                // ==================================================================================
                // Compute:  dOV = dO @ V^T
                // Layout:   dO[row: BLOCK_N, D], V[col: BLOCK_M, D] -> dOV[row: BLOCK_N, col: BLOCK_M]
                // Template: BLOCK_X=BLOCK_N, BLOCK_Y=BLOCK_M
                // ==================================================================================
                WMMA_GEMM_SCORES<Config, GemmType::dOV_dOVT, D, IS_CAUSAL, IS_ALIBI, IS_SOFTCAP, IS_WINDOW, BLOCK_N, BLOCK_M, D_STRIDE, M_STRIDE>(
                  sdO, sV, sdOV,
                  valid_q_rows, block.valid_kv_rows,
                  0, 0, 0, 1.0f, 0.0f, 0.0f, -1, -1, warp_id, lane_id);
                __syncthreads();

                // ==================================================================================
                // Compute:  P = exp(S - lse), dS = P * (dOV - row_dot) * scale
                // Layout:   S[row: BLOCK_N, BLOCK_M], dOV[row: BLOCK_N, BLOCK_M],
                //           LSE[row: BLOCK_N], row_dot[row: BLOCK_N] -> P[row: BLOCK_N, BLOCK_M], dS[row: BLOCK_N, BLOCK_M]
                // Template: LDS_STRIDE=M_STRIDE, LDO_STRIDE=BLOCK_M, TILE_X=BLOCK_N, TILE_Y=BLOCK_M
                // ==================================================================================
                WMMA_GEMM_SOFTMAX_GRADIENT<Config, GemmType::compute_P_dS, IS_SOFTCAP, IS_DROPOUT, M_STRIDE, BLOCK_M, BLOCK_N, BLOCK_M>(
                  sS, sdOV, sLse, sRowDot, sP, sdS,
                  valid_q_rows,  block.valid_kv_rows,
                  softmax_scale, softcap,
                  p_dropout, dropout_seed, dropout_offset, start_q, block.start_kv, N, tid);
                __syncthreads();

                // ==================================================================================
                // Compute:  dV += P^T @ dO
                // Layout:   P^T[col: BLOCK_M, BLOCK_N], dO[row: BLOCK_N, D] -> dV[row: BLOCK_M, D]
                // Template: BLOCK_X=BLOCK_M, BLOCK_Y=BLOCK_N
                // ==================================================================================
                WMMA_GEMM_GRADIENTS<Config, GemmType::dV_PTdO, D, BLOCK_M, BLOCK_N, BLOCK_M, D_STRIDE>(
                  sP, sdO, sdV,
                  block.valid_kv_rows, valid_q_rows,
                  warp_id, lane_id);
                __syncthreads();

                // ==================================================================================
                // Load:     Q tile from global to sQ(reuse) shared memory
                // Layout:   Q: global[row: BLOCK_N, D] -> shared[row: BLOCK_N, D_STRIDE]
                // Template: DUAL_LOAD=false, SRC_STRIDE=D, DST_STRIDE=D_STRIDE
                // ==================================================================================
                WMMA_GEMM_LOAD_TILE<Config, false, D_STRIDE>(
                  q_ptr + start_q * D, sQ,
                  nullptr, nullptr,
                  D, valid_q_rows, tid);
                __syncthreads();

                // ==================================================================================
                // Compute:  dK += dS^T @ Q
                // Layout:   dS^T[col: BLOCK_M, BLOCK_N], Q[row: BLOCK_N, D] -> dK[row: BLOCK_M, D]
                // Template: BLOCK_X=BLOCK_M, BLOCK_Y=BLOCK_N
                // ==================================================================================
                WMMA_GEMM_GRADIENTS<Config, GemmType::dK_dSTQ, D, BLOCK_M, BLOCK_N, BLOCK_M, D_STRIDE>(
                  sdS, sQ, sdK,
                  block.valid_kv_rows, valid_q_rows,
                  warp_id, lane_id);
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
        WMMA_GEMM_EPILOGUE<Config, GemmType::write_dKV, D_STRIDE>(
          sdK,    dK_ptr,
          sdV,    dV_ptr,
          nullptr,
          D, block.valid_kv_rows, tid);
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
    const int    alibi_batch,
    int window_left,
    int window_right,
    uint64_t dropout_seed,
    uint64_t dropout_offset,
    cudaStream_t stream
) {
    using Config = KernelConfig<D>;

    const size_t smem = Config::TOTAL_SMEM;
    TORCH_CHECK(smem <= MAX_SMEM_PER_SM, "Shared memory exceeds 96KB for Backward kernel: ", smem, " bytes (", smem / 1024, " KB)");

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

    bool is_alibi       = (alibi_slopes != nullptr);
    bool is_softcap     = (softcap > 0.0f);
    bool is_window      = (window_left >= 0 || window_right >= 0);
    bool is_dropout     = (p_dropout > 0.0f);
    bool is_paged       = false;
    bool is_rope        = false;
    bool is_interleaved = false;

    const __half* q_ptr   = reinterpret_cast<const __half*>(Q.data_ptr());
    const __half* k_ptr   = reinterpret_cast<const __half*>(K.data_ptr());
    const __half* v_ptr   = reinterpret_cast<const __half*>(V.data_ptr());
    const __half* o_ptr   = reinterpret_cast<const __half*>(O.data_ptr());
    const __half* dO_ptr  = reinterpret_cast<const __half*>(dO.data_ptr());
    float*        lse_ptr = softmax_lse.data_ptr<float>();
    __half*       dQ_ptr  = reinterpret_cast<__half*>(dQ.data_ptr());
    __half*       dK_ptr  = reinterpret_cast<__half*>(dK.data_ptr());
    __half*       dV_ptr  = reinterpret_cast<__half*>(dV.data_ptr());

    dispatch_attention_features(is_causal, is_alibi, is_softcap, is_window, is_dropout, is_paged, is_rope, is_interleaved,
    [&](auto CAUSAL, auto ALIBI, auto SOFTCAP, auto WINDOW, auto DROPOUT, auto /*PAGED*/, auto /*ROPE*/, auto /*INTERLEAVED*/) {
        constexpr bool IS_CAUSAL  = decltype(CAUSAL)::value;
        constexpr bool IS_ALIBI   = decltype(ALIBI)::value;
        constexpr bool IS_SOFTCAP = decltype(SOFTCAP)::value;
        constexpr bool IS_WINDOW  = decltype(WINDOW)::value;
        constexpr bool IS_DROPOUT = decltype(DROPOUT)::value;

        auto kernel = flash_attention_backward_kernel<D, IS_CAUSAL, IS_ALIBI, IS_SOFTCAP, IS_WINDOW, IS_DROPOUT>;
        cudaFuncSetAttribute(kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, smem);

        kernel<<<grid, block, smem, stream>>>(
            q_ptr, k_ptr, v_ptr, o_ptr, dO_ptr, lse_ptr,
            dQ_ptr, dK_ptr, dV_ptr,
            B, H_Q, H_K, M, N, grid_dq, grid_dkv,
            softmax_scale, softcap, alibi_slopes, alibi_batch, window_left, window_right,
            p_dropout, dropout_seed, dropout_offset
        );
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
    // Device guard for multi-GPU / pipeline-parallelism
    at::cuda::CUDAGuard device_guard{q.device()};

    auto props = at::cuda::getCurrentDeviceProperties();
    TORCH_CHECK(props->major == 7 && props->minor == 0, "Kernel supports only Volta GPUs.");
    TORCH_CHECK(!deterministic, "Deterministic backward not supported in this Volta build");

    // dtype / device / contiguity
    auto q_dtype = q.dtype();
    TORCH_CHECK(q_dtype == torch::kFloat16, "q must be fp16");
    TORCH_CHECK(k.dtype() == q_dtype && v.dtype() == q_dtype, "k/v must have the same dtype as q");
    TORCH_CHECK(out.dtype() == q_dtype, "out must have the same dtype as q");
    TORCH_CHECK(dout.dtype() == q_dtype, "dout must have the same dtype as q");
    TORCH_CHECK(softmax_lse.dtype() == torch::kFloat32, "softmax_lse must be fp32");

    TORCH_CHECK(q.is_cuda() && k.is_cuda() && v.is_cuda(), "Tensors q, k, v must be on CUDA");
    TORCH_CHECK(out.is_cuda() && dout.is_cuda() && softmax_lse.is_cuda(), "out, dout, softmax_lse must be on CUDA");
    TORCH_CHECK(q.stride(-1) == 1 && k.stride(-1) == 1 && v.stride(-1) == 1, "Last dim of q, k, v must be contiguous");
    TORCH_CHECK(out.stride(-1) == 1 && dout.stride(-1) == 1, "Last dim of out, dout must be contiguous");

    // dimensions
    const int B   = q.size(0);
    const int H_Q = q.size(1);
    const int M   = q.size(2);
    const int D   = q.size(3);
    const int H_K = k.size(1);
    const int N   = k.size(2);

    TORCH_CHECK(B > 0, "batch size must be positive");
    TORCH_CHECK(D <= 256, "head dimension must be <= 256");
    TORCH_CHECK(D % 8 == 0, "head dimension must be multiple of 8");
    TORCH_CHECK(H_Q % H_K == 0, "H_Q must be divisible by H_K for GQA/MQA");

    // softcap restrictions
    if (softcap > 0.f) {
        TORCH_CHECK(p_dropout == 0.f, "Softcapping does not support dropout");
    }

    // window edge cases
    if (window_left  >= N) window_left  = -1;
    if (window_right >= N) window_right = -1;

    // alibi
    const float* alibi_ptr = nullptr;
    int alibi_batch = 0;
    if (alibi_slopes.has_value()) {
        const auto& slopes = alibi_slopes.value();
        TORCH_CHECK(slopes.dtype() == torch::kFloat32 && slopes.is_cuda(), "alibi_slopes must be fp32 on CUDA");
        TORCH_CHECK(slopes.stride(-1) == 1, "alibi_slopes last dim must be contiguous");
        auto sizes = slopes.sizes();
        bool valid = (sizes.size() == 1 && sizes[0] == H_Q) ||
                     (sizes.size() == 2 && sizes[0] == B && sizes[1] == H_Q);
        TORCH_CHECK(valid, "alibi_slopes must be [H_Q] or [B, H_Q]");
        alibi_batch = (sizes.size() == 2) ? slopes.stride(0) : 0;
        alibi_ptr = slopes.data_ptr<float>();
    }

    // dropout
    TORCH_CHECK(p_dropout >= 0.f && p_dropout < 1.f, "p_dropout must be in [0, 1)");

    uint64_t dropout_seed   = 0;
    uint64_t dropout_offset = 0;
    if (p_dropout > 0.0f) {
        TORCH_CHECK(rng_state.has_value() && rng_state->numel() == 2, "rng_state required when p_dropout > 0");
        auto rng_cpu = rng_state->cpu();
        auto rng_acc = rng_cpu.accessor<int64_t, 1>();
        dropout_seed   = static_cast<uint64_t>(rng_acc[0]);
        dropout_offset = static_cast<uint64_t>(rng_acc[1]);
    }

    // output tensors
    at::Tensor dq_fp16 = dq.has_value() ? dq.value() : torch::empty_like(q);
    at::Tensor dk_fp16 = dk.has_value() ? dk.value() : torch::empty_like(k);
    at::Tensor dv_fp16 = dv.has_value() ? dv.value() : torch::empty_like(k);

    TORCH_CHECK(dq_fp16.dtype() == q_dtype, "dq must have the same dtype as q");
    TORCH_CHECK(dk_fp16.dtype() == q_dtype, "dk must have the same dtype as q");
    TORCH_CHECK(dv_fp16.dtype() == q_dtype, "dv must have the same dtype as q");
    TORCH_CHECK(dq_fp16.is_cuda() && dk_fp16.is_cuda() && dv_fp16.is_cuda(), "d*/v* must be on CUDA");
    TORCH_CHECK(dq_fp16.stride(-1) == 1, "dq must have contiguous last dimension");
    TORCH_CHECK(dk_fp16.stride(-1) == 1, "dk must have contiguous last dimension");
    TORCH_CHECK(dv_fp16.stride(-1) == 1, "dv must have contiguous last dimension");

    if (dq.has_value()) {
        TORCH_CHECK(dq_fp16.sizes() == q.sizes(), "dq shape must match q shape");
    }
    if (dk.has_value()) {
        TORCH_CHECK(dk_fp16.sizes() == k.sizes(), "dk shape must match k shape");
    }
    if (dv.has_value()) {
        TORCH_CHECK(dv_fp16.sizes() == k.sizes(), "dv shape must match k shape");
    }

    // dsoftmax_sum
    auto dsoftmax_sum = torch::empty({B, H_Q, M}, torch::dtype(torch::kFloat32).device(q.device()));
    TORCH_CHECK(dsoftmax_sum.dtype() == torch::kFloat32, "dsoftmax_sum must be fp32");

    // empty case
    if (M == 0 || N == 0) {
        dq_fp16.zero_();
        dk_fp16.zero_();
        dv_fp16.zero_();
        dsoftmax_sum.zero_();
        return {dq_fp16, dk_fp16, dv_fp16, dsoftmax_sum};
    }

    // run kernel
    auto stream = at::cuda::getCurrentCUDAStream().stream();

    #define LAUNCH_KERNEL(DIM) \
        launcher_flash_attention_backward<DIM>(q, k, v, out, dout, softmax_lse, dq_fp16, dk_fp16, dv_fp16, softmax_scale, is_causal, \
                                               softcap, p_dropout, alibi_ptr, alibi_batch,  window_left, window_right, dropout_seed, dropout_offset, stream);
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
