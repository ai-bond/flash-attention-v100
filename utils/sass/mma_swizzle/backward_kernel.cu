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
#include "mat_mul.h"
#include "softmax.h"
#include "product.h"

// ======================================================================================
// BACKWARD KERNEL
// ======================================================================================
template<int D, bool IS_EVEN, bool IS_CAUSAL, bool IS_ALIBI, bool IS_SOFTCAP, bool IS_WINDOW, bool IS_DROPOUT>
__global__ void __launch_bounds__(KernelConfig<D>::THREADS_PER_BLOCK, 1)
flash_attention_backward_kernel(
    const __half* __restrict__ Q,
    const __half* __restrict__ K,
    const __half* __restrict__ V,
    const __half* __restrict__ dO,
    const  float* __restrict__ softmax_lse,
    const  float* __restrict__ row_dot,
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

        // ==================================================================================
        // Grid Mapping: X for Q-blocks, Z for batch-head composite (batch * H_Q + head)
        // ==================================================================================
        const int bthd_idx     = blockIdx.z;
        const int block_idx    = blockIdx.x;

        if (bthd_idx >= B * H_Q) return;

        // ======================================================================================
        // BlockInfo: Metadata resolution (Dense Q-centric)
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
        //   Q/O/dO/dQ: [B, M, H_Q, D]   (offset via bthd_idx, Q-head space)
        //   K/V:       [B, N, H_K, D]   (KV-head via bthd_idx % H_Q / (H_Q / H_K))
        //   LSE:       [B, H_Q, M]
        // ==================================================================================
        const __half* __restrict__ q_ptr    = Q           + block.q_offset  (D, H_Q, M);
        const __half* __restrict__ k_ptr    = K           + block.kv_offset (D, H_K, N);
        const __half* __restrict__ v_ptr    = V           + block.kv_offset (D, H_K, N);
        const __half* __restrict__ dO_ptr   = dO          + block.q_offset  (D, H_Q, M);
              __half* __restrict__ dQ_ptr   = dQ          + block.q_offset  (D, H_Q, M);
        const  float* __restrict__ lse_ptr  = softmax_lse + block.lse_offset(H_Q, M);
        const  float* __restrict__ rdot_ptr = row_dot     + block.lse_offset(H_Q, M);

        // ==================================================================================
        // Init:   shared memory with zero-fill union regions to avoid stale data
        // ==================================================================================
        extern __shared__ char smem_raw[];

        auto& smem = *reinterpret_cast<typename Config::SmemLayout*>(smem_raw);

        __half* __restrict__ sQ      = smem.phase.bdq.q;
        __half* __restrict__ sK      = smem.phase.bdq.reuse_kv.k;
        __half* __restrict__ sV      = smem.phase.bdq.reuse_kv.v;
        __half* __restrict__ sdO     = smem.phase.bdq.dO;
        __half* __restrict__ sdS     = smem.phase.bdq.dS;

        WMMA_GEMM_INIT_SMEM<Config>(smem_raw);
        WMMA_GEMM_INIT_REGS<Config, GemmType::Q_KT,  GemmPhase::BWQ> rS;
        WMMA_GEMM_INIT_REGS<Config, GemmType::dO_VT, GemmPhase::BWQ> rdOV;
        WMMA_GEMM_INIT_REGS<Config, GemmType::dS_K,  GemmPhase::BWQ> rdQ;
        __syncthreads();

        // ==================================================================================
        // Load:     Q(dO)  tile from global to sQ(sdO) shared memory
        // Layout:   Q[dO]: global[row: BLOCK_M, D] with H_Q*D -> shared[row: BLOCK_M, D]
        // Template: GemmPhase::BWQ, Stage::DUAL
        // ==================================================================================
        WMMA_GEMM_LOAD_TILE<Config, GemmPhase::BWQ, Stage::DUAL>(
          q_ptr,   sQ,
          dO_ptr,  sdO,
          H_Q * D, block.valid_q_rows, tid);
        __syncthreads();

        // ==================================================================================
        // MAIN LOOP (iterates over K/V blocks for current Q block)
        // ==================================================================================
        for (int block_kv = block.block_min; block_kv < block.block_max; ++block_kv) {
            const int start_kv      = block_kv * BLOCK_N;
            const int valid_kv_rows = min(BLOCK_N, N - start_kv);

            // ==================================================================================
            // Load:     V tile from global to sV(reuse) shared memory
            // Layout:   V: global[row: BLOCK_N, D] with H_K*D -> shared[row: BLOCK_N, D]
            // Template: GemmPhase::BWQ, Stage::SNGL
            // ==================================================================================
            WMMA_GEMM_LOAD_TILE<Config, GemmPhase::BWQ, Stage::SNGL>(
              v_ptr + start_kv * (H_K * D), sV,
              nullptr, nullptr,
              H_K * D, valid_kv_rows, tid);
            __syncthreads();

            // ==================================================================================
            // Compute:  dOV = dO @ V^T
            // Layout:   dO[row: BLOCK_M, D], V[col: BLOCK_N, D] -> dOV[row: BLOCK_M, col: BLOCK_N]
            // Template: GemmType::dO_VT, GemmPhase::BWQ
            // ==================================================================================
            WMMA_GEMM_SCORES<Config, GemmType::dO_VT, GemmPhase::BWQ, IS_CAUSAL, IS_ALIBI, IS_SOFTCAP, IS_WINDOW>(
              sdO, sV, rdOV.frags,
              block.valid_q_rows,  valid_kv_rows,
              0, 0, 0, 1.0f, 0.0f, 0.0f, -1, -1,
              warp_id, lane_id);
            __syncthreads();

            // ==================================================================================
            // Load:     K tile from global to sK(reuse) shared memory
            // Layout:   K: global[row: BLOCK_N, D] with H_K*D -> shared[row: BLOCK_N, D]
            // Template: GemmPhase::BWQ, Stage::SNGL
            // ==================================================================================
            WMMA_GEMM_LOAD_TILE<Config, GemmPhase::BWQ, Stage::SNGL>(
              k_ptr + start_kv * (H_K * D), sK,
              nullptr, nullptr,
              H_K * D, valid_kv_rows, tid);
            __syncthreads();

            // ==================================================================================
            // Compute:  S = Q @ K^T
            // Layout:   Q[row: BLOCK_M, D], K[col: BLOCK_N, D] -> S[row: BLOCK_M, col: BLOCK_N]
            // Template: GemmType::Q_KT, GemmPhase::BWQ
            // ==================================================================================
            WMMA_GEMM_SCORES<Config, GemmType::Q_KT, GemmPhase::BWQ, IS_CAUSAL, IS_ALIBI, IS_SOFTCAP, IS_WINDOW>(
              sQ, sK, rS.frags,
              block.valid_q_rows,  valid_kv_rows,
              block.start_q,       start_kv,
              block.seqlen_offset,
              softmax_scale, softcap, alibi_slope, window_left, window_right, warp_id, lane_id);
            __syncthreads();

            // ==================================================================================
            // Compute:  dS = exp(S - lse) * (dOV - row_dot) * scale
            // Layout:   S[row: BLOCK_M, BLOCK_N], dOV[row: BLOCK_M, BLOCK_N],
            //           LSE[row: BLOCK_M], row_dot[row: BLOCK_M] -> dS[row: BLOCK_M, BLOCK_N]
            // Template: GemmPhase::BWQ
            // ==================================================================================
            WMMA_GEMM_SOFTMAX_GRADIENT<Config, GemmPhase::BWQ, IS_SOFTCAP, IS_DROPOUT>(
              rS.frags, rdOV.frags, lse_ptr, rdot_ptr, nullptr, sdS,
              block.valid_q_rows, valid_kv_rows,
              softmax_scale, softcap,
              p_dropout, dropout_seed, dropout_offset,
              block.start_q, start_kv, N, warp_id, lane_id);
            __syncthreads();

            // ==================================================================================
            // Compute:  dQ += dS @ K
            // Layout:   dS[row: BLOCK_M, BLOCK_N], K[row: BLOCK_N, D] -> dQ[row: BLOCK_M, D]
            // Template: GemmType::dS_K, GemmPhase::BWQ
            // ==================================================================================
            WMMA_GEMM_GRADIENTS<Config, GemmType::dS_K, GemmPhase::BWQ>(
              sdS, sK, rdQ.frags,
              block.valid_q_rows, valid_kv_rows,
              warp_id, lane_id);
            __syncthreads();
        } // END MAIN LOOP
        // ==================================================================================
        // Compute:  Store gradient dQ without normalization
        // Layout:   sdQ[row: BLOCK_M, D_STRIDE] -> dQ_ptr[row: BLOCK_M, D] with H_Q*D
        // Template: Result::WRITE_DQ (NORMALIZE=false, DUAL_STORE=false)
        // ==================================================================================
        WMMA_GEMM_EPILOGUE<Config, Result::WRITE_DQ>(
          rdQ.frags, dQ_ptr,
          nullptr,   nullptr,
          nullptr,   nullptr,   nullptr,
          H_Q * D, block.valid_q_rows, warp_id, lane_id);
    }
    // ===================================================================================
    // PHASE 2: dKV
    // ===================================================================================
    else if (blockIdx.y == 1) {
        if (blockIdx.x >= grid_dkv) return;

        using Config = KernelConfig<D>;

        constexpr int BLOCK_M   = Config::DKV::BLOCK_M;
        constexpr int BLOCK_N   = Config::DKV::BLOCK_N;

        // ==================================================================================
        // Grid Mapping: X for KV-blocks, Z for batch-KV-head composite when H_Q == H_K
        //               GQA/MQA, blockIdx.z in [B*H_K, B*H_Q) early-returns (grid.z is B*H_Q)
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
        // Layout:
        //   K/V/dK/dV: [B, N, H_K, D],  (offset via bthd_idx, KV-head space)
        //   Q/O/dO/dQ: [B, M, H_Q, D],  (head_q resolved in the Q-heads loop)
        //   LSE:       [B, H_Q, M]
        // ==================================================================================
        const __half* __restrict__ k_ptr  = K  + block.kv_offset(D, H_K, N) + block.start_kv * (H_K * D);
        const __half* __restrict__ v_ptr  = V  + block.kv_offset(D, H_K, N) + block.start_kv * (H_K * D);
              __half* __restrict__ dK_ptr = dK + block.kv_offset(D, H_K, N) + block.start_kv * (H_K * D);
              __half* __restrict__ dV_ptr = dV + block.kv_offset(D, H_K, N) + block.start_kv * (H_K * D);

        // ==================================================================================
        // Init:   shared memory with zero-fill union regions to avoid stale data
        // ==================================================================================
        extern __shared__ char smem_raw[];

        auto& smem = *reinterpret_cast<typename Config::SmemLayout*>(smem_raw);

        __half* __restrict__ sQ            = smem.phase.bdkv.reuse_qdO.q;
        __half* __restrict__ sdO           = smem.phase.bdkv.reuse_qdO.dO;
        __half* __restrict__ sK            = smem.phase.bdkv.k;
        __half* __restrict__ sV            = smem.phase.bdkv.v;
        __half* __restrict__ sdS           = smem.phase.bdkv.dS;
        __half* __restrict__ sP            = smem.phase.bdkv.p;

        WMMA_GEMM_INIT_SMEM<Config>(smem_raw);
        WMMA_GEMM_INIT_REGS<Config, GemmType::Q_KT,  GemmPhase::BWKV> rS;
        WMMA_GEMM_INIT_REGS<Config, GemmType::dO_VT, GemmPhase::BWKV> rdOV;
        WMMA_GEMM_INIT_REGS<Config, GemmType::PT_dO, GemmPhase::BWKV> rdV;
        WMMA_GEMM_INIT_REGS<Config, GemmType::dST_Q, GemmPhase::BWKV> rdK;
        __syncthreads();

        // ==================================================================================
        // Load:     K(V)  tile from global to sK(sV) shared memory
        // Layout:   K[V]: global[row: BLOCK_M, D] with H_K*D -> shared[row: BLOCK_M, D]
        // Template: GemmPhase::BWKV, Stage::DUAL
        // ==================================================================================
        WMMA_GEMM_LOAD_TILE<Config, GemmPhase::BWKV, Stage::DUAL>(
          k_ptr,   sK,
          v_ptr,   sV,
          H_K * D, block.valid_kv_rows, tid);
        __syncthreads();

        // ==================================================================================
        // Q-HEADS LOOP (Iterate over Q-head groups sharing this KV-head)
        // ==================================================================================
        for (int group = 0; group < (H_Q / H_K); ++group) {

            // ==================================================================================
            // Layout:
            //   Q/O/dO [B, M, H_Q, D] -> batch * M * H_Q * D + head_q * D
            //   LSE:   [B, H_Q, M]    -> (batch * H_Q + head_q) * M
            // ==================================================================================
            const __half* __restrict__ q_ptr    = Q           + (size_t)(bthd_idx  / H_K) * M * H_Q * D + (size_t)(((bthd_idx % H_K) * (H_Q / H_K)) + group) * D;
            const __half* __restrict__ dO_ptr   = dO          + (size_t)(bthd_idx  / H_K) * M * H_Q * D + (size_t)(((bthd_idx % H_K) * (H_Q / H_K)) + group) * D;
            const  float* __restrict__ lse_ptr  = softmax_lse + (size_t)((bthd_idx / H_K) * H_Q + (((bthd_idx % H_K) * (H_Q / H_K)) + group)) * M;
            const  float* __restrict__ rdot_ptr = row_dot     + (size_t)((bthd_idx / H_K) * H_Q + (((bthd_idx % H_K) * (H_Q / H_K)) + group)) * M;
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
                // Layout:   Q: global[row: BLOCK_N, D] with H_Q*D -> shared[row: BLOCK_N, D]
                // Template: GemmPhase::BWKV, Stage::SNGL
                // ==================================================================================
                WMMA_GEMM_LOAD_TILE<Config, GemmPhase::BWKV, Stage::SNGL>(
                  q_ptr + start_q * (H_Q * D), sQ,
                  nullptr, nullptr,
                  H_Q * D, valid_q_rows, tid);
                __syncthreads();

                // ==================================================================================
                // Compute:  S = Q @ K^T
                // Layout:   Q[row: BLOCK_N, D], K[col: BLOCK_M, D] -> S[row: BLOCK_N, col: BLOCK_M]
                // Template: GemmType::Q_KT, GemmPhase::BWKV (IS_SWAPPED=true)
                // ==================================================================================
                WMMA_GEMM_SCORES<Config, GemmType::Q_KT, GemmPhase::BWKV, IS_CAUSAL, IS_ALIBI, IS_SOFTCAP, IS_WINDOW>(
                  sQ, sK, rS.frags,
                  valid_q_rows,  block.valid_kv_rows,
                  start_q,       block.start_kv,
                  block.seqlen_offset,
                  softmax_scale, softcap, alibi_slope, window_left, window_right, warp_id, lane_id);
                __syncthreads();

                // ==================================================================================
                // Load:     dO tile from global to sdO(reuse) shared memory
                // Layout:   dO: global[row: BLOCK_N, D] with H_Q*D -> shared[row: BLOCK_N, D]
                // Template: GemmPhase::BWKV, Stage::SNGL
                // ==================================================================================
                WMMA_GEMM_LOAD_TILE<Config, GemmPhase::BWKV, Stage::SNGL>(
                  dO_ptr + start_q * (H_Q * D), sdO,
                  nullptr, nullptr,
                  H_Q * D, valid_q_rows, tid);
                __syncthreads();

                // ==================================================================================
                // Compute:  dOV = dO @ V^T
                // Layout:   dO[row: BLOCK_N, D], V[col: BLOCK_M, D] -> dOV[row: BLOCK_N, col: BLOCK_M]
                // Template: GemmType::dO_VT, GemmPhase::BWKV (IS_SWAPPED=true)
                // ==================================================================================
                WMMA_GEMM_SCORES<Config, GemmType::dO_VT, GemmPhase::BWKV, IS_CAUSAL, IS_ALIBI, IS_SOFTCAP, IS_WINDOW>(
                  sdO, sV, rdOV.frags,
                  valid_q_rows,  block.valid_kv_rows,
                  0, 0, 0, 1.0f, 0.0f, 0.0f, -1, -1,
                  warp_id, lane_id);
                __syncthreads();

                // ==================================================================================
                // Compute:  P = exp(S - lse), dS = P * (dOV - row_dot) * softmax_scale
                // Layout:   S[row: BLOCK_N, BLOCK_M], dOV[row: BLOCK_N, BLOCK_M],
                //           LSE[row: BLOCK_N], row_dot[row: BLOCK_N] -> P[row: BLOCK_N, BLOCK_M], dS[row: BLOCK_N, BLOCK_M]
                // Template: GemmPhase::BWKV
                // ==================================================================================
                WMMA_GEMM_SOFTMAX_GRADIENT<Config, GemmPhase::BWKV, IS_SOFTCAP, IS_DROPOUT>(
                  rS.frags, rdOV.frags, lse_ptr + start_q, rdot_ptr + start_q, sP, sdS,
                  valid_q_rows,  block.valid_kv_rows,
                  softmax_scale, softcap,
                  p_dropout, dropout_seed, dropout_offset,
                  start_q, block.start_kv, N, warp_id, lane_id);
                __syncthreads();

                // ==================================================================================
                // Compute:  dV += P^T @ dO
                // Layout:   P^T[col: BLOCK_M, BLOCK_N], dO[row: BLOCK_N, D] -> dV[row: BLOCK_M, D]
                // Template: GemmType::PT_dO, GemmPhase::BWKV
                // ==================================================================================
                WMMA_GEMM_GRADIENTS<Config, GemmType::PT_dO, GemmPhase::BWKV>(
                  sP, sdO, rdV.frags,
                  block.valid_kv_rows, valid_q_rows,
                  warp_id, lane_id);
                __syncthreads();

                // ==================================================================================
                // Load:     Q tile from global to sQ(reuse) shared memory
                // Layout:   Q: global[row: BLOCK_N, D] with H_Q*D -> shared[row: BLOCK_N, D]
                // Template: GemmPhase::BWKV, Stage::SNGL
                // ==================================================================================
                WMMA_GEMM_LOAD_TILE<Config, GemmPhase::BWKV, Stage::SNGL>(
                  q_ptr + start_q * (H_Q * D), sQ,
                  nullptr, nullptr,
                  H_Q * D, valid_q_rows, tid);
                __syncthreads();

                // ==================================================================================
                // Compute:  dK += dS^T @ Q
                // Layout:   dS^T[col: BLOCK_M, BLOCK_N], Q[row: BLOCK_N, D] -> dK[row: BLOCK_M, D]
                // Template: GemmType::dST_Q, GemmPhase::BWKV
                // ==================================================================================
                WMMA_GEMM_GRADIENTS<Config, GemmType::dST_Q, GemmPhase::BWKV>(
                  sdS, sQ, rdK.frags,
                  block.valid_kv_rows, valid_q_rows,
                  warp_id, lane_id);
                __syncthreads();
            } // END Q-TILES LOOP
        } // END Q-HEADS LOOP
        // ==================================================================================
        // Compute:  Store gradients dK + dV without normalization
        // Layout:   sdK/sdV[row: BLOCK_M, D] -> dK/dV_ptr[row: BLOCK_M, D] with H_K*D
        // Template: Result::WRITE_DKV (NORMALIZE=false, DUAL_STORE=true)
        // ==================================================================================
        WMMA_GEMM_EPILOGUE<Config, Result::WRITE_DKV>(
          rdK.frags,   dK_ptr,
          rdV.frags,   dV_ptr,
          nullptr,     nullptr,   nullptr,
          H_K * D, block.valid_kv_rows, warp_id, lane_id);
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
// ВНИМАНИЕ: раскладка O в cpu_ref_fwd — [B*H, M, D].
// flash_attention_rowdot_kernel ожидает [B, M, H_Q, D].
// Для B=1, H=1 оба layout'а эквивалентны — тест-кейсы подобраны так.
// ======================================================================================
template<int D, bool IS_CAUSAL>
bool run_test_bwd(int B, int H, int M, int N, float scale) {
    using Config = KernelConfig<D>;
    printf("  D=%-3d BN=%-3d BM=%-2d W=%-2d causal=%d ",
           D, Config::DQ::BLOCK_N, Config::DQ::BLOCK_M, WARPS, IS_CAUSAL);

    const size_t qsz = (size_t)B*H*M*D;
    const size_t ksz = (size_t)B*H*N*D;
    const size_t osz = qsz;
    const size_t lsz = (size_t)B*H*M;

    std::vector<float> hQ(qsz), hK(ksz), hV(ksz), hdO(qsz);
    std::vector<float> hO_ref(osz), hLSE_ref(lsz);
    std::vector<float> hdQ_ref(qsz), hdK_ref(ksz), hdV_ref(ksz);
    std::vector<float> hdQ_gpu(qsz), hdK_gpu(ksz), hdV_gpu(ksz);

    srand(42);
    for (auto& v : hQ) v = ((float)rand() / RAND_MAX - 0.5f) * 0.1f;
    for (auto& v : hK) v = ((float)rand() / RAND_MAX - 0.5f) * 0.1f;
    for (auto& v : hV) v = ((float)rand() / RAND_MAX - 0.5f) * 0.1f;
    for (auto& v : hdO) v = ((float)rand() / RAND_MAX - 0.5f) * 0.1f;

    const bool use_cpu_ref = (M <= 2048 && N <= 2048);

    cpu_ref_fwd(hQ.data(), hK.data(), hV.data(), hO_ref.data(), hLSE_ref.data(),
                B, H, M, N, D, scale, IS_CAUSAL);

    if (use_cpu_ref) {
        cpu_ref_bwd(hQ.data(), hK.data(), hV.data(), hO_ref.data(), hdO.data(),
                    hdQ_ref.data(), hdK_ref.data(), hdV_ref.data(),
                    B, H, M, N, D, scale, IS_CAUSAL);
    }

    // ----------------------------------------------------------------------------------
    // GPU buffers: Q, K, V, dO — fp16;  O — fp16 (то, что записало бы forward-ядро)
    // ----------------------------------------------------------------------------------
    __half *d_Q, *d_K, *d_V, *d_O, *d_dO, *d_dQ, *d_dK, *d_dV;
    float  *d_lse, *d_rowdot;

    cudaMalloc(&d_Q,     qsz*2); cudaMalloc(&d_K,  ksz*2); cudaMalloc(&d_V,  ksz*2);
    cudaMalloc(&d_O,     osz*2); cudaMalloc(&d_dO, qsz*2);
    cudaMalloc(&d_dQ,    qsz*2); cudaMalloc(&d_dK, ksz*2); cudaMalloc(&d_dV, ksz*2);
    cudaMalloc(&d_lse,   lsz*4); cudaMalloc(&d_rowdot, lsz*4);

    std::vector<__half> hQ_h(qsz), hK_h(ksz), hV_h(ksz), hO_h(osz), hdO_h(qsz);
    for (size_t i = 0; i < qsz; ++i) { hQ_h[i] = __float2half(hQ[i]); hdO_h[i] = __float2half(hdO[i]); }
    for (size_t i = 0; i < ksz; ++i) { hK_h[i] = __float2half(hK[i]); hV_h[i] = __float2half(hV[i]); }
    for (size_t i = 0; i < osz; ++i) { hO_h[i] = __float2half(hO_ref[i]); }

    cudaMemcpy(d_Q,   hQ_h.data(),     qsz*2, cudaMemcpyHostToDevice);
    cudaMemcpy(d_K,   hK_h.data(),     ksz*2, cudaMemcpyHostToDevice);
    cudaMemcpy(d_V,   hV_h.data(),     ksz*2, cudaMemcpyHostToDevice);
    cudaMemcpy(d_O,   hO_h.data(),     osz*2, cudaMemcpyHostToDevice);
    cudaMemcpy(d_dO,  hdO_h.data(),    qsz*2, cudaMemcpyHostToDevice);
    cudaMemcpy(d_lse, hLSE_ref.data(), lsz*4, cudaMemcpyHostToDevice);
    // d_rowdot НЕ копируется с хоста — его заполняет rowdot-ядро на GPU.

    // ----------------------------------------------------------------------------------
    // STEP 1: row_dot = rowsum(O ⊙ dO)  — как в launcher_flash_attention_rowdot
    // Layout:  O/dO   [B, M, H_Q, D]  (в тесте B=H=1 — совпадает с [B*H, M, D] CPU-ref)
    //          row_dot [B, H_Q, M]
    // ----------------------------------------------------------------------------------
    {
        constexpr int ROWDOT_BLOCK_M   = 32;
        constexpr int ROWDOT_THREADS   = MAX_THREADS_PER_WARP * 8;
        const dim3    rowdot_grid((M + ROWDOT_BLOCK_M - 1) / ROWDOT_BLOCK_M, B * H);
        flash_attention_rowdot_kernel<D><<<rowdot_grid, ROWDOT_THREADS>>>(
            d_O, d_dO, d_rowdot, B, H, M);
    }

    // ----------------------------------------------------------------------------------
    // STEP 2: backward kernel  dQ, dK, dV
    // ----------------------------------------------------------------------------------
    const size_t smem_bytes = Config::TOTAL_SMEM;
    const bool   is_even    = (M % Config::DQ::BLOCK_M == 0) && (N % Config::DQ::BLOCK_N == 0);

    const int grid_dq  = (M + Config::DQ::BLOCK_M - 1) /  Config::DQ::BLOCK_M;
    const int grid_dkv = (N + Config::DKV::BLOCK_M - 1) / Config::DKV::BLOCK_M;
    const int grid_max = (grid_dq > grid_dkv) ? grid_dq : grid_dkv;
    dim3 grid(grid_max, 2, B * H);
    dim3 block(Config::THREADS_PER_BLOCK);

    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    auto launch = [&](auto kern) {
        cudaFuncSetAttribute(kern, cudaFuncAttributeMaxDynamicSharedMemorySize, smem_bytes);
        kern<<<grid, block, smem_bytes>>>(
            d_Q, d_K, d_V, d_dO, d_lse, d_rowdot, d_dQ, d_dK, d_dV,
            B, H, H, M, N, grid_dq, grid_dkv,
            scale, 0.0f, nullptr, 0, -1, -1, 0.0f, 0, 0);
    };

    cudaEventRecord(start, 0);
    if (is_even) launch(flash_attention_backward_kernel<D, true,  IS_CAUSAL, false, false, false, false>);
    else         launch(flash_attention_backward_kernel<D, false, IS_CAUSAL, false, false, false, false>);
    cudaEventRecord(stop, 0);
    cudaEventSynchronize(stop);

    float milliseconds = 0;
    cudaEventElapsedTime(&milliseconds, start, stop);
    cudaEventDestroy(start);
    cudaEventDestroy(stop);

    std::vector<__half> hdQ_h(qsz), hdK_h(ksz), hdV_h(ksz);
    cudaMemcpy(hdQ_h.data(), d_dQ, qsz*2, cudaMemcpyDeviceToHost);
    cudaMemcpy(hdK_h.data(), d_dK, ksz*2, cudaMemcpyDeviceToHost);
    cudaMemcpy(hdV_h.data(), d_dV, ksz*2, cudaMemcpyDeviceToHost);

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

    cudaFree(d_Q); cudaFree(d_K); cudaFree(d_V); cudaFree(d_O); cudaFree(d_dO);
    cudaFree(d_dQ); cudaFree(d_dK); cudaFree(d_dV); cudaFree(d_lse); cudaFree(d_rowdot);
    return pass;
}

int main() {
    printf("[ FA2 Backward ]:\n");
    int ok = 0, total = 0;

    std::vector<std::tuple<int, int, int, int, int>> test_cases = {
        {1, 1, 1024, 1024,  16},
        {1, 1, 1024, 1024,  32},
        {1, 1, 1024, 1024,  64},
        {1, 1, 1024, 1024, 128},
        {1, 1, 1024, 1024, 256},
    };

    for (const auto& tc : test_cases) {
        auto [B, H, M, N, D] = tc;
        float scale = 1.0f / sqrtf((float)D);
        for (bool causal : {false, true}) {
            bool r = false;
            switch (D) {
                case 16:  r = causal ? run_test_bwd<16, true >(B,H,M,N,scale) : run_test_bwd<16, false>(B,H,M,N,scale); break;
                case 32:  r = causal ? run_test_bwd<32, true >(B,H,M,N,scale) : run_test_bwd<32, false>(B,H,M,N,scale); break;
                case 64:  r = causal ? run_test_bwd<64, true >(B,H,M,N,scale) : run_test_bwd<64, false>(B,H,M,N,scale); break;
                case 128: r = causal ? run_test_bwd<128,true >(B,H,M,N,scale) : run_test_bwd<128,false>(B,H,M,N,scale); break;
                case 256: r = causal ? run_test_bwd<256,true >(B,H,M,N,scale) : run_test_bwd<256,false>(B,H,M,N,scale); break;
            }
            total++; if (r) ok++;
        }
    }

    printf("\n%d/%d tests passed\n", ok, total);
    return (ok == total) ? 0 : 1;
}
