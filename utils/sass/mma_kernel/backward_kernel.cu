// ======================================================================================
// * Copyright (c) 2026, D.Skryabin / tg @ai_bond007 SPDX-License: BSD-3-Clause
// ======================================================================================
#include <cuda.h>
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cstdio>
#include <cmath>
#include <cstring>
#include <cstdlib>
#include <vector>
#include <algorithm>
#include <tuple>
#include <omp.h>

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
            WMMA_GEMM_SOFTMAX_GRADIENT<Config, GemmType::compute_dS, IS_SOFTCAP, IS_DROPOUT, N_STRIDE, N_STRIDE * 2, BLOCK_M, BLOCK_N>(
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
            WMMA_GEMM_GRADIENTS<Config, GemmType::dQ_dSK, D, BLOCK_M, BLOCK_N, N_STRIDE * 2, D_STRIDE>(
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
                WMMA_GEMM_SOFTMAX_GRADIENT<Config, GemmType::compute_P_dS, IS_SOFTCAP, IS_DROPOUT, M_STRIDE, M_STRIDE * 2, BLOCK_N, BLOCK_M>(
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
                WMMA_GEMM_GRADIENTS<Config, GemmType::dV_PTdO, D, BLOCK_M, BLOCK_N, M_STRIDE * 2, D_STRIDE>(
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
                WMMA_GEMM_GRADIENTS<Config, GemmType::dK_dSTQ, D, BLOCK_M, BLOCK_N, M_STRIDE * 2, D_STRIDE>(
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
// CPU REFERENCE & METRICS
// ======================================================================================
void cpu_ref_fwd(const float* Q_ref, const float* K_ref, const float* V_ref,
                 float* Out_ref, float* LSE_ref, int B, int H, int M, int N, int D,
                 float scale, bool causal) {
    #pragma omp parallel for schedule(dynamic)
    for (int bh = 0; bh < B * H; ++bh) {
        for (int i = 0; i < M; ++i) {
            float mx = NEG_INF;
            std::vector<float> S(N);
            for (int j = 0; j < N; ++j) {
                float s = 0;
                for (int d = 0; d < D; ++d) s += Q_ref[(bh*M+i)*D+d] * K_ref[(bh*N+j)*D+d];
                s *= scale;
                if (causal && j > i) s = NEG_INF;
                S[j] = s; mx = std::max(mx, s);
            }
            float sm = 0;
            std::vector<float> P(N);
            for (int j = 0; j < N; ++j) {
                P[j] = (S[j] > NEG_INF + 1e10f) ? expf(S[j] - mx) : 0.0f;
                sm += P[j];
            }
            float inv = (sm > 1e-24f) ? 1.0f / sm : 0.0f;
            for (int d = 0; d < D; ++d) {
                float o = 0;
                for (int j = 0; j < N; ++j) o += P[j] * V_ref[(bh*N+j)*D+d];
                Out_ref[(bh*M+i)*D+d] = o * inv;
            }
            LSE_ref[bh*M+i] = mx + logf(fmaxf(sm, 1e-24f));
        }
    }
}

void cpu_ref_bwd(const float* Q_ref, const float* K_ref, const float* V_ref, const float* O_ref, const float* dO_ref,
                 float* dQ_ref, float* dK_ref, float* dV_ref,
                 int B, int H, int M, int N, int D, float scale, bool causal) {
    #pragma omp parallel for schedule(dynamic)
    for (int bh = 0; bh < B * H; ++bh) {
        std::vector<float> P(M * N, 0.0f);
        std::vector<float> D_arr(M, 0.0f);

        for (int i = 0; i < M; ++i) {
            float mx = NEG_INF;
            std::vector<float> S(N);
            for (int j = 0; j < N; ++j) {
                float s = 0;
                for (int d = 0; d < D; ++d) s += Q_ref[(bh*M+i)*D+d] * K_ref[(bh*N+j)*D+d];
                s *= scale;
                if (causal && j > i) s = NEG_INF;
                S[j] = s; mx = std::max(mx, s);
            }
            float sm = 0;
            for (int j = 0; j < N; ++j) {
                P[i*N+j] = (S[j] > NEG_INF + 1e10f) ? expf(S[j] - mx) : 0.0f;
                sm += P[i*N+j];
            }
            float inv = (sm > 1e-24f) ? 1.0f / sm : 0.0f;
            for (int j = 0; j < N; ++j) P[i*N+j] *= inv;

            float d_val = 0;
            for (int d = 0; d < D; ++d) d_val += O_ref[(bh*M+i)*D+d] * dO_ref[(bh*M+i)*D+d];
            D_arr[i] = d_val;
        }

        for (int j = 0; j < N; ++j) {
            for (int d = 0; d < D; ++d) {
                float dv = 0;
                for (int i = 0; i < M; ++i) dv += P[i*N+j] * dO_ref[(bh*M+i)*D+d];
                dV_ref[(bh*N+j)*D+d] = dv;
            }
        }

        std::vector<float> dS(M * N, 0.0f);
        for (int i = 0; i < M; ++i) {
            for (int j = 0; j < N; ++j) {
                float dov = 0;
                for (int d = 0; d < D; ++d) dov += dO_ref[(bh*M+i)*D+d] * V_ref[(bh*N+j)*D+d];
                float ds = P[i*N+j] * (dov - D_arr[i]) * scale;
                if (causal && j > i) ds = 0.0f; 
                dS[i*N+j] = ds;
            }
        }

        for (int j = 0; j < N; ++j) {
            for (int d = 0; d < D; ++d) {
                float dk = 0;
                for (int i = 0; i < M; ++i) dk += dS[i*N+j] * Q_ref[(bh*M+i)*D+d];
                dK_ref[(bh*N+j)*D+d] = dk;
            }
        }

        for (int i = 0; i < M; ++i) {
            for (int d = 0; d < D; ++d) {
                float dq = 0;
                for (int j = 0; j < N; ++j) dq += dS[i*N+j] * K_ref[(bh*N+j)*D+d];
                dQ_ref[(bh*M+i)*D+d] = dq;
            }
        }
    }
}

struct StabilityMetrics {
    float max_diff, avg_diff;
    int nan_count, inf_count, big_diff_count;
};

StabilityMetrics compute_stability_bwd(const std::vector<float>& dq_gpu, const std::vector<float>& dq_ref,
                                       const std::vector<float>& dk_gpu, const std::vector<float>& dk_ref,
                                       const std::vector<float>& dv_gpu, const std::vector<float>& dv_ref) {
    StabilityMetrics m = {0,0,0,0,0};
    double sum = 0;
    size_t total_elements = dq_gpu.size() + dk_gpu.size() + dv_gpu.size();

    auto check = [&](const std::vector<float>& gpu, const std::vector<float>& ref) {
        for (size_t i = 0; i < gpu.size(); ++i) {
            if (std::isnan(gpu[i])) { m.nan_count++; continue; }
            if (std::isinf(gpu[i])) { m.inf_count++; continue; }
            float d = fabsf(gpu[i] - ref[i]);
            m.max_diff = std::max(m.max_diff, d);
            sum += d;
            if (d > 0.01f) m.big_diff_count++;
        }
    };

    check(dq_gpu, dq_ref);
    check(dk_gpu, dk_ref);
    check(dv_gpu, dv_ref);

    if (total_elements > 0) m.avg_diff = (float)(sum / total_elements);
    return m;
}

// ======================================================================================
// TEST HARNESS
// ======================================================================================
template<int D, bool IS_CAUSAL>
bool run_test_bwd(int B, int H, int M, int N, float scale) {
    using Config = KernelConfig<D>;
    printf("  D=%-3d BN=%-3d BM=%-2d W=%-2d causal=%d ",
           D, Config::DQ::BLOCK_N, Config::DQ::BLOCK_M, WARPS, IS_CAUSAL);

    size_t qsz = (size_t)B*H*M*D, ksz = (size_t)B*H*N*D, osz = qsz, lsz = (size_t)B*H*M;

    std::vector<float> hQ(qsz), hK(ksz), hV(ksz), hdO(qsz);
    std::vector<float> hO_ref(osz), hLSE_ref(lsz);
    std::vector<float> hdQ_ref(qsz), hdK_ref(ksz), hdV_ref(ksz);
    std::vector<float> hdQ_gpu(qsz), hdK_gpu(ksz), hdV_gpu(qsz);

    srand(42);
    for (auto& v : hQ) v = ((float)rand() / RAND_MAX - 0.5f) * 0.1f;
    for (auto& v : hK) v = ((float)rand() / RAND_MAX - 0.5f) * 0.1f;
    for (auto& v : hV) v = ((float)rand() / RAND_MAX - 0.5f) * 0.1f;
    for (auto& v : hdO) v = ((float)rand() / RAND_MAX - 0.5f) * 0.1f;

    bool use_cpu_ref = (M <= 1024 && N <= 1024); 

    cpu_ref_fwd(hQ.data(), hK.data(), hV.data(), hO_ref.data(), hLSE_ref.data(), B, H, M, N, D, scale, IS_CAUSAL);

    if (use_cpu_ref) {
        cpu_ref_bwd(hQ.data(), hK.data(), hV.data(), hO_ref.data(), hdO.data(), 
                    hdQ_ref.data(), hdK_ref.data(), hdV_ref.data(), B, H, M, N, D, scale, IS_CAUSAL);
    }

    __half *dQ, *dK, *dV, *dO, *dOut, *ddQ, *ddK, *ddV; float *dLSE;
    cudaMalloc(&dQ, qsz*2); cudaMalloc(&dK, ksz*2); cudaMalloc(&dV, ksz*2);
    cudaMalloc(&dO, qsz*2); cudaMalloc(&dOut, osz*2); cudaMalloc(&dLSE, lsz*4);
    cudaMalloc(&ddQ, qsz*2); cudaMalloc(&ddK, ksz*2); cudaMalloc(&ddV, ksz*2);

    std::vector<__half> hQ_h(qsz), hK_h(ksz), hV_h(ksz), hdO_h(qsz), hOut_h(osz);
    for (size_t i = 0; i < qsz; ++i) { hQ_h[i] = __float2half(hQ[i]); hdO_h[i] = __float2half(hdO[i]); }
    for (size_t i = 0; i < ksz; ++i) { hK_h[i] = __float2half(hK[i]); hV_h[i] = __float2half(hV[i]); }
    for (size_t i = 0; i < osz; ++i) hOut_h[i] = __float2half(hO_ref[i]);

    cudaMemcpy(dQ, hQ_h.data(), qsz*2, cudaMemcpyHostToDevice);
    cudaMemcpy(dK, hK_h.data(), ksz*2, cudaMemcpyHostToDevice);
    cudaMemcpy(dV, hV_h.data(), ksz*2, cudaMemcpyHostToDevice);
    cudaMemcpy(dO, hdO_h.data(), qsz*2, cudaMemcpyHostToDevice);
    cudaMemcpy(dOut, hOut_h.data(), osz*2, cudaMemcpyHostToDevice);
    cudaMemcpy(dLSE, hLSE_ref.data(), lsz*4, cudaMemcpyHostToDevice);

    const size_t smem_bytes = Config::TOTAL_SMEM;
    auto kern = flash_attention_backward_kernel<D, IS_CAUSAL, false, false, false, false>;
    cudaFuncSetAttribute(kern, cudaFuncAttributeMaxDynamicSharedMemorySize, smem_bytes);

    const int grid_dq  = (M + Config::DQ::BLOCK_M - 1) /  Config::DQ::BLOCK_M;
    const int grid_dkv = (N + Config::DKV::BLOCK_M - 1) / Config::DKV::BLOCK_M;
    const int grid_max = (grid_dq > grid_dkv) ? grid_dq : grid_dkv;
    dim3 grid(grid_max, 2, B * H);
    dim3 block(Config::THREADS_PER_BLOCK);

    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    cudaEventRecord(start, 0);
    kern<<<grid, block, smem_bytes>>>(
        dQ, dK, dV, dOut, dO, dLSE, ddQ, ddK, ddV,
        B, H, H, M, N, grid_dq, grid_dkv,
        scale, 0.0f, nullptr, 0, -1, -1, 0.0f, 0, 0
    );
    cudaEventRecord(stop, 0);
    cudaEventSynchronize(stop);

    float milliseconds = 0;
    cudaEventElapsedTime(&milliseconds, start, stop);
    cudaEventDestroy(start);
    cudaEventDestroy(stop);

    std::vector<__half> hdQ_h(qsz), hdK_h(ksz), hdV_h(ksz);
    cudaMemcpy(hdQ_h.data(), ddQ, qsz*2, cudaMemcpyDeviceToHost);
    cudaMemcpy(hdK_h.data(), ddK, ksz*2, cudaMemcpyDeviceToHost);
    cudaMemcpy(hdV_h.data(), ddV, ksz*2, cudaMemcpyDeviceToHost);

    for (size_t i = 0; i < qsz; ++i) hdQ_gpu[i] = __half2float(hdQ_h[i]);
    for (size_t i = 0; i < ksz; ++i) { hdK_gpu[i] = __half2float(hdK_h[i]); hdV_gpu[i] = __half2float(hdV_h[i]); }

    StabilityMetrics m = {0,0,0,0,0};

    if (use_cpu_ref) {
        m = compute_stability_bwd(hdQ_gpu, hdQ_ref, hdK_gpu, hdK_ref, hdV_gpu, hdV_ref);
    } else {
        for (size_t i = 0; i < qsz; ++i) {
            if (std::isnan(hdQ_gpu[i])) m.nan_count++;
            if (std::isinf(hdQ_gpu[i])) m.inf_count++;
        }
        for (size_t i = 0; i < ksz; ++i) {
            if (std::isnan(hdK_gpu[i]) || std::isnan(hdV_gpu[i])) m.nan_count++;
            if (std::isinf(hdK_gpu[i]) || std::isinf(hdV_gpu[i])) m.inf_count++;
        }
    }

    bool pass = (m.nan_count == 0) && (m.inf_count == 0);
    if (use_cpu_ref) {
        int allowed_big = std::max(1, (int)((qsz + ksz * 2) * 0.001f));
        pass = pass && (m.max_diff < 0.05f) && (m.big_diff_count <= allowed_big); 
    }

    printf("%s  max(dQKV)=%.2e  avg(dQKV)=%.2e  NaN=%d Inf=%d big=%d GPU=%.4f ms\n",
           pass ? "✅" : "❌",
           m.max_diff, m.avg_diff,
           m.nan_count, m.inf_count, m.big_diff_count,
           milliseconds);

    cudaFree(dQ); cudaFree(dK); cudaFree(dV); cudaFree(dO); cudaFree(dOut); cudaFree(dLSE);
    cudaFree(ddQ); cudaFree(ddK); cudaFree(ddV);
    return pass;
}

bool dispatch_test_bwd(int B, int H, int M, int N, int D, bool is_causal) {
    float scale = 1.0f / sqrtf((float)D);
    switch (D) {
        case 16:  return is_causal ? run_test_bwd<16, true>(B, H, M, N, scale)  : run_test_bwd<16, false>(B, H, M, N, scale);
        case 32:  return is_causal ? run_test_bwd<32, true>(B, H, M, N, scale)  : run_test_bwd<32, false>(B, H, M, N, scale);
        case 64:  return is_causal ? run_test_bwd<64, true>(B, H, M, N, scale)  : run_test_bwd<64, false>(B, H, M, N, scale);
        case 128: return is_causal ? run_test_bwd<128, true>(B, H, M, N, scale) : run_test_bwd<128, false>(B, H, M, N, scale);
        case 256: return is_causal ? run_test_bwd<256, true>(B, H, M, N, scale) : run_test_bwd<256, false>(B, H, M, N, scale);
        default:
            printf("  Unsupported D=%d\n", D);
            return false;
    }
}

int main() {
    printf("[ FA2 Backward ]:\n");
    int ok = 0, total = 0;
    auto T = [&](auto r) { total++; if (r) ok++; };

    std::vector<std::tuple<int, int, int, int, int>> test_cases = {
        {1, 1, 16, 16, 16},
        {1, 1, 32, 32, 32},
        {1, 1, 64, 64, 64},
        {1, 1, 128, 128, 128},
        {1, 1, 256, 256, 256},
        {1, 1, 1024, 1024, 128},
        {1, 1, 2048, 2048, 128},
        {1, 1, 4096, 4096, 128},
    };

    for (const auto& tc : test_cases) {
        auto [B, H, M, N, D] = tc;
        T(dispatch_test_bwd(B, H, M, N, D, false));
        T(dispatch_test_bwd(B, H, M, N, D, true));
    }

    printf("\n%d/%d tests passed\n", ok, total);
    return (ok == total) ? 0 : 1;
}