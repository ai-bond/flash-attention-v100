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
// BACKWARD VARLEN KERNEL
// ======================================================================================
template<int D, bool IS_CAUSAL, bool IS_ALIBI, bool IS_SOFTCAP, bool IS_WINDOW, bool IS_DROPOUT>
__global__ void __launch_bounds__(KernelConfig<D>::THREADS_PER_BLOCK, 1)
flash_attention_backward_varlen_kernel(
    const __half* __restrict__ Q,
    const __half* __restrict__ K,
    const __half* __restrict__ V,
    const __half* __restrict__ O,
    const __half* __restrict__ dO,
    const float*  __restrict__ softmax_lse,
    __half* __restrict__ dQ,
    __half* __restrict__ dK,
    __half* __restrict__ dV,
    float*  __restrict__ softmax_d,
    const int* __restrict__ cu_seqlens_q,
    const int* __restrict__ cu_seqlens_k,
    const int      B,
    const int      H_Q,
    const int      H_K,
    const int      T_Q,
    const int      T_K,
    const int      grid_dq,
    const int      grid_dkv,
    const int      max_seqlen_q,
    const int      max_seqlen_k,
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
    // ======================================================================================
    // PHASE 1: dQ
    // ======================================================================================
    if (blockIdx.y == 0) {
        if (blockIdx.x >= grid_dq) return;

        using Config = KernelConfig<D>;
        constexpr int BLOCK_M   = Config::DQ::BLOCK_M;
        constexpr int BLOCK_N   = Config::DQ::BLOCK_N;

        // ======================================================================================
        // Grid Mapping: X for Q-blocks, Z for batch-head composite (B * H_Q + head)
        // ======================================================================================
        const int block_idx     = blockIdx.x;
        const int bthd_idx      = blockIdx.z;

        if (bthd_idx >= B * H_Q) return;

        // ======================================================================================
        // BlockInfo: Metadata resolution (Varlen Q-centric)
        // ======================================================================================
        BlockInfo<IS_CAUSAL, IS_WINDOW, true> block;
        block.init_q(
            block_idx,       // BLOCK_IDX:      Current Q-block index (grid.x)
            bthd_idx,        // BATCH_HEAD_ID:  Global Q-head index (batch * H_Q + head_q)
            H_Q,             // H_Q:            Number of query heads
            H_K,             // H_K:            Number of KV heads
            0,               // M:              (unused for varlen)
            0,               // N:              (unused for varlen)
            B,               // B:              Batch size (device-side loop)
            BLOCK_M,         // BLOCK_M:        Tile size along Q dimension
            BLOCK_N,         // BLOCK_N:        Tile size along KV dimension
            window_left,     // WINDOW_LEFT:    Left sliding window bound (-1 if disabled)
            window_right,    // WINDOW_RIGHT:   Right sliding window bound (-1 if disabled)
            cu_seqlens_q,    // CU_SEQLENS_Q:   Cumulative Q lengths
            cu_seqlens_k,    // CU_SEQLENS_K:   Cumulative KV lengths
            nullptr          // SEQUSED_K:      (not used in backward)
        );

        if (block.start_q >= block.seqlen_q || block.valid_q_rows <= 0) return;

        // ======================================================================================
        // EARLY EXIT: No valid KV blocks to attend to (Causal/Window/Empty KV)
        //             Writes deterministic zeros to dQ and softmax_d for numerical stability
        // ======================================================================================
        if (block.block_min >= block.block_max || block.seqlen_k == 0) {
            const size_t dq_offset  = block.q_offset(D, H_Q, T_Q);
            const size_t lse_offset = block.lse_offset(H_Q, T_Q);
            for (int row = threadIdx.x; row < block.valid_q_rows; row += blockDim.x) {
                #pragma unroll
                for (int idx = 0; idx < D; ++idx) {
                    dQ[dq_offset + row * H_Q * D + idx] = __float2half(0.0f);
                }
                softmax_d[lse_offset + row] = 0.0f;
            }
            return;
        }

        // ======================================================================================
        // Init:   thread/warp/lane IDs for WMMA coordination
        // ======================================================================================
        const int tid     = threadIdx.x;
        const int warp_id = tid >> 5;
        const int lane_id = tid & 31;
        // Alibi slope only for valid block + batch
        const int   alibi_idx   = (alibi_batch > 0) ? (block.batch_idx * alibi_batch + block.head_idx) : block.head_idx;
        const float alibi_slope = (alibi_slopes != nullptr) ? alibi_slopes[alibi_idx] : 0.0f;

        // ======================================================================================
        // Layout:
        //   Q/O/dO/dQ: [T_Q, H_Q, D]   (offset via q_base + start_q, Q-head space)
        //   K/V:       [T_K, H_K, D]   (offset via k_base, KV-head space)
        //   LSE/dLSE:  [H_Q, T_Q]      (offset via bthd_idx * T_Q)
        // ======================================================================================
        const __half* __restrict__ q_ptr   = Q  + block.q_offset (D, H_Q, T_Q);
        const __half* __restrict__ k_ptr   = K  + block.kv_offset(D, H_K, T_K);
        const __half* __restrict__ v_ptr   = V  + block.kv_offset(D, H_K, T_K);
        const __half* __restrict__ o_ptr   = O  + block.q_offset (D, H_Q, T_Q);
        const __half* __restrict__ dO_ptr  = dO + block.q_offset (D, H_Q, T_Q);
              __half* __restrict__ dQ_ptr  = dQ + block.q_offset (D, H_Q, T_Q);
        const float*  __restrict__ lse_ptr = softmax_lse + block.lse_offset(H_Q, T_Q);
        float*  __restrict__ dLse_ptr      = softmax_d   + block.lse_offset(H_Q, T_Q);

        // ======================================================================================
        // Init:   shared memory with zero-fill union regions to avoid stale data
        // ======================================================================================
        extern __shared__ char smem_raw[];

        auto& smem = *reinterpret_cast<typename Config::SmemLayout*>(smem_raw);

        __half* __restrict__ sQ      = smem.phase.bdq.q;
        __half* __restrict__ sK      = smem.phase.bdq.reuse_kv.k;
        __half* __restrict__ sV      = smem.phase.bdq.reuse_kv.v;
        float*  __restrict__ sS      = smem.phase.bdq.s;
        __half* __restrict__ sdO     = smem.phase.bdq.dO;
        float*  __restrict__ sdOV    = smem.phase.bdq.reuse_sdOVS.dOV;
        __half* __restrict__ sdS     = smem.phase.bdq.reuse_sdOVS.dS;
        float*  __restrict__ sRowDot = smem.row_dot;
        float*  __restrict__ sLse    = smem.lse;
        float*  __restrict__ sdQ     = smem.phase.bdq.dQ;

        WMMA_GEMM_INIT_SMEM<Config>(smem_raw);
        __syncthreads();

        // ======================================================================================
        // Load:     Q(dO)  tile from global to sQ(sdO) shared memory
        // Layout:   Q[dO]: global[row: BLOCK_M, D] with H_Q*D -> shared[row: BLOCK_M, D]
        // Template: GemmPhase::BWQ, Stage::DUAL
        // ======================================================================================
        WMMA_GEMM_LOAD_TILE<Config, GemmPhase::BWQ, Stage::DUAL>(
          q_ptr,   sQ,
          dO_ptr,  sdO,
          H_Q * D, block.valid_q_rows, tid);
        __syncthreads();

        // ======================================================================================
        // Compute:  row_dot = sum(O ⊙ dO) [dQ backward pass]
        // Layout:   O[global: tile from block.start_q, D], dO[shared: valid_q_rows, D] -> sRowDot[shared: valid_q_rows]
        // Template: GemmPhase::BWQ (LSE_OFFSET=0)
        // ======================================================================================
        WMMA_GEMM_DOT_PRODUCT<Config, GemmPhase::BWQ>(
          o_ptr, sdO, lse_ptr, sLse, sRowDot,
          H_Q * D, block.valid_q_rows, 0, tid);
        __syncthreads();

        // ======================================================================================
        // MAIN LOOP (iterates over K/V blocks for current Q block)
        // ======================================================================================
        for (int block_q = block.block_min; block_q < block.block_max; ++block_q) {
            const int start_kv      = block_q * BLOCK_N;
            const int valid_kv_rows = min(BLOCK_N, block.seqlen_k - start_kv);
            if (valid_kv_rows <= 0) break;

            // ======================================================================================
            // Load:     V tile from global to sV(reuse) shared memory
            // Layout:   V: global[row: BLOCK_N, D] with H_K*D -> shared[row: BLOCK_N, D]
            // Template: GemmPhase::BWQ, Stage::SNGL
            // ======================================================================================
            WMMA_GEMM_LOAD_TILE<Config, GemmPhase::BWQ, Stage::SNGL>(
              v_ptr + start_kv * H_K * D, sV,
              nullptr, nullptr,
              H_K * D, valid_kv_rows, tid);
            __syncthreads();

            // ======================================================================================
            // Compute:  dOV = dO @ V^T
            // Layout:   dO[row: BLOCK_M, D], V[col: BLOCK_N, D] -> dOV[row: BLOCK_M, col: BLOCK_N]
            // Template: GemmType::dO_VT, GemmPhase::BWQ
            // ======================================================================================
            WMMA_GEMM_SCORES<Config, GemmType::dO_VT, GemmPhase::BWQ, IS_CAUSAL, IS_ALIBI, IS_SOFTCAP, IS_WINDOW>(
              sdO, sV, sdOV,
              block.valid_q_rows,  valid_kv_rows,
              0, 0, 0, 1.0f, 0.0f, 0.0f, -1, -1,
              warp_id, lane_id);
            __syncthreads();

            // ======================================================================================
            // Load:     K tile from global to sK(reuse) shared memory
            // Layout:   K: global[row: BLOCK_N, D] with H_K*D -> shared[row: BLOCK_N, D]
            // Template: GemmPhase::BWQ, Stage::SNGL
            // ======================================================================================
            WMMA_GEMM_LOAD_TILE<Config, GemmPhase::BWQ, Stage::SNGL>(
              k_ptr + start_kv * H_K * D, sK,
              nullptr, nullptr,
              H_K * D, valid_kv_rows, tid);
            __syncthreads();

            // ======================================================================================
            // Compute:  S = Q @ K^T
            // Layout:   Q[row: BLOCK_M, D], K[col: BLOCK_N, D] -> S[row: BLOCK_M, col: BLOCK_N]
            // Template: GemmType::Q_KT, GemmPhase::BWQ
            // ======================================================================================
            WMMA_GEMM_SCORES<Config, GemmType::Q_KT, GemmPhase::BWQ, IS_CAUSAL, IS_ALIBI, IS_SOFTCAP, IS_WINDOW>(
              sQ, sK, sS,
              block.valid_q_rows,  valid_kv_rows,
              block.start_q,       start_kv,
              block.seqlen_offset,
              softmax_scale, softcap, alibi_slope, window_left, window_right,
              warp_id, lane_id);
            __syncthreads();

            // ======================================================================================
            // Compute:  dS = exp(S - lse) * (dOV - row_dot) * softmax_scale
            // Layout:   S[row: BLOCK_M, BLOCK_N], dOV[row: BLOCK_M, BLOCK_N] -> dS[row: BLOCK_M, BLOCK_N]
            // Template: GemmPhase::BWQ
            // ======================================================================================
            WMMA_GEMM_SOFTMAX_GRADIENT<Config, GemmPhase::BWQ, IS_SOFTCAP, IS_DROPOUT>(
              sS, sdOV, sLse, sRowDot, nullptr, sdS,
              block.valid_q_rows, valid_kv_rows,
              softmax_scale, softcap,
              p_dropout, dropout_seed, dropout_offset,
              block.q_base + block.start_q, start_kv, block.seqlen_k, tid);
            __syncthreads();

            // ======================================================================================
            // Compute:  dQ += dS @ K
            // Layout:   dS[row: BLOCK_M, BLOCK_N], K[row: BLOCK_N, D] -> dQ[row: BLOCK_M, D]
            // Template: GemmType::dS_K, GemmPhase::BWQ
            // ======================================================================================
            WMMA_GEMM_GRADIENTS<Config, GemmType::dS_K, GemmPhase::BWQ>(
              sdS, sK, sdQ,
              block.valid_q_rows, valid_kv_rows,
              warp_id, lane_id);
            __syncthreads();
        }   // END MAIN LOOP
        // ======================================================================================
        // Compute:  Store gradient dQ without normalization and dLSE
        // Layout:   sdQ[row: BLOCK_M, D_STRIDE] -> dQ_ptr[row: BLOCK_M, D] with H_Q*D
        //           sRowDot[row: BLOCK_M] -> dLse_ptr[row: BLOCK_M]
        // Template: Result::WRITE_DQ (NORMALIZE=false, DUAL_STORE=false)
        // ======================================================================================
        WMMA_GEMM_EPILOGUE<Config, Result::WRITE_DQ>(
          sdQ,     dQ_ptr,
          nullptr, nullptr,
          nullptr, H_Q * D, block.valid_q_rows, tid);

        if (tid < block.valid_q_rows) {
            dLse_ptr[tid] = sRowDot[tid];
        }
    }
    // ======================================================================================
    // PHASE 2: dKV
    // ======================================================================================
    else if (blockIdx.y == 1) {
        if (blockIdx.x >= grid_dkv) return;

        using Config = KernelConfig<D>;
        constexpr int BLOCK_M   = Config::DKV::BLOCK_M;
        constexpr int BLOCK_N   = Config::DKV::BLOCK_N;

        // ======================================================================================
        // Grid Mapping: X for KV-blocks, Z for batch-Q-head composite (B * H_Q + head_q)
        // ======================================================================================
        const int block_idx     = blockIdx.x;
        const int bthd_idx      = blockIdx.z;

        if (bthd_idx >= B * H_Q) return;

        // ======================================================================================
        // BlockInfo: Metadata resolution (Varlen Q-centric, maps to KV-head)
        // ======================================================================================
        BlockInfo<IS_CAUSAL, IS_WINDOW, true> block;
        block.init_kv(
            block_idx,       // BLOCK_IDX:      Current KV-block index (grid.x)
            bthd_idx,        // BATCH_HEAD_ID:  Global Q-head index (batch * H_Q + head_q)
            H_Q,             // H_Q:            Number of query heads
            H_K,             // H_K:            Number of KV heads
            0,               // M:              (unused for varlen)
            0,               // N:              (unused for varlen)
            B,               // B:              Batch size (device-side loop)
            BLOCK_M,         // BLOCK_M:        Tile size along KV dimension
            BLOCK_N,         // BLOCK_N:        Tile size along Q dimension
            window_left,     // WINDOW_LEFT:    Left sliding window bound (-1 if disabled)
            window_right,    // WINDOW_RIGHT:   Right sliding window bound (-1 if disabled)
            cu_seqlens_q,    // CU_SEQLENS_Q:   Cumulative Q lengths
            cu_seqlens_k,    // CU_SEQLENS_K:   Cumulative KV lengths
            nullptr          // SEQUSED_K:      (not used in backward)
        );

        if (block.start_kv >= block.seqlen_k || block.valid_kv_rows <= 0) return;

        // ======================================================================================
        // EARLY EXIT: No valid Q blocks attend to this KV block
        //             Writes deterministic zeros to dK and dV for the current KV-head
        // ======================================================================================
        if (block.block_min >= block.block_max || block.seqlen_q == 0) {
            const size_t dkdv_offset = static_cast<size_t>(block.k_base + block.start_kv) * H_Q * D + static_cast<size_t>(block.head_idx) * D;
            for (int row = threadIdx.x; row < block.valid_kv_rows; row += blockDim.x) {
                #pragma unroll
                for (int idx = 0; idx < D; ++idx) {
                    dK[dkdv_offset + row * H_Q * D + idx] = __float2half(0.0f);
                    dV[dkdv_offset + row * H_Q * D + idx] = __float2half(0.0f);
                }
            }
            return;
        }

        // ======================================================================================
        // Init:   thread/warp/lane IDs for WMMA coordination
        // ======================================================================================
        const int tid     = threadIdx.x;
        const int warp_id = tid >> 5;
        const int lane_id = tid & 31;
        // Alibi slope only for valid block + batch
        const int   alibi_idx   = (alibi_batch > 0) ? (block.batch_idx * alibi_batch + block.head_idx) : block.head_idx;
        const float alibi_slope = (alibi_slopes != nullptr) ? alibi_slopes[alibi_idx] : 0.0f;

        // ======================================================================================
        // Layout:
        //   K/V:       [T_K, H_K, D]   (offset via k_base + start_kv, KV-head space)
        //   Q/O/dO:    [T_Q, H_Q, D]   (base follows q_base, head_idx; sequence offset inside loop)
        //   LSE:       [H_Q, T_Q]      (base follows head_idx * T_Q + q_base)
        //   dK/dV:     [T_K, H_Q, D]   (offset via k_base + start_kv, head_idx)
        // ======================================================================================
        const __half* __restrict__ k_ptr   = K  + block.kv_offset(D, H_K, T_K);
        const __half* __restrict__ v_ptr   = V  + block.kv_offset(D, H_K, T_K);
        const __half* __restrict__ q_ptr   = Q  + static_cast<size_t>(block.q_base) * H_Q * D + static_cast<size_t>(block.head_idx) * D;
        const __half* __restrict__ o_ptr   = O  + static_cast<size_t>(block.q_base) * H_Q * D + static_cast<size_t>(block.head_idx) * D;
        const __half* __restrict__ dO_ptr  = dO + static_cast<size_t>(block.q_base) * H_Q * D + static_cast<size_t>(block.head_idx) * D;
        const  float* __restrict__ lse_ptr = softmax_lse + static_cast<size_t>(block.head_idx) * T_Q + block.q_base;
              __half* __restrict__ dK_ptr  = dK + static_cast<size_t>(block.k_base + block.start_kv) * H_Q * D + static_cast<size_t>(block.head_idx) * D;
              __half* __restrict__ dV_ptr  = dV + static_cast<size_t>(block.k_base + block.start_kv) * H_Q * D + static_cast<size_t>(block.head_idx) * D;

        // ======================================================================================
        // Init:   shared memory with zero-fill union regions to avoid stale data
        // ======================================================================================
        extern __shared__ char smem_raw[];

        auto& smem = *reinterpret_cast<typename Config::SmemLayout*>(smem_raw);

        __half* __restrict__ sQ      = smem.phase.bdkv.reuse_qdO.q;
        __half* __restrict__ sK      = smem.phase.bdkv.k;
        __half* __restrict__ sV      = smem.phase.bdkv.v;
        float*  __restrict__ sS      = smem.phase.bdkv.reuse_sp.s;
        __half* __restrict__ sdO     = smem.phase.bdkv.reuse_qdO.dO;
        float*  __restrict__ sdOV    = smem.phase.bdkv.reuse_dOVS.dOV;
        __half* __restrict__ sdS     = smem.phase.bdkv.reuse_dOVS.dS;
        __half* __restrict__ sP      = smem.phase.bdkv.reuse_sp.p;
        float*  __restrict__ sRowDot = smem.row_dot;
        float*  __restrict__ sLse    = smem.lse;
        float*  __restrict__ sdK     = smem.phase.bdkv.dK;
        float*  __restrict__ sdV     = smem.phase.bdkv.dV;

        WMMA_GEMM_INIT_SMEM<Config>(smem_raw);
        __syncthreads();

        // ======================================================================================
        // Load:     K & V tiles from global to sK/sV shared memory
        // Layout:   K[V]: global[row: BLOCK_M, D] with H_K*D -> shared[row: BLOCK_M, D]
        // Template: GemmPhase::BWKV, Stage::DUAL
        // ======================================================================================
        WMMA_GEMM_LOAD_TILE<Config, GemmPhase::BWKV, Stage::DUAL>(
          k_ptr + block.start_kv * H_K * D, sK,
          v_ptr + block.start_kv * H_K * D, sV,
          H_K * D, block.valid_kv_rows, tid);
        __syncthreads();

        // ==================================================================================
        // Q-TILES LOOP (iterates over Q-tiles for current KV block)
        // ==================================================================================
        for (int block_q = block.block_min; block_q < block.block_max; ++block_q) {
            const int start_q      = block_q * BLOCK_N;
            const int valid_q_rows = min(BLOCK_N, block.seqlen_q - start_q);
            if (valid_q_rows <= 0) break;

            // ======================================================================================
            // Load:     Q tile from global to sQ(reuse) shared memory
            // Layout:   Q: global[row: BLOCK_N, D] with H_Q*D -> shared[row: BLOCK_N, D]
            // Template: GemmPhase::BWKV, Stage::SNGL
            // ======================================================================================
            WMMA_GEMM_LOAD_TILE<Config, GemmPhase::BWKV, Stage::SNGL>(
              q_ptr + start_q * H_Q * D, sQ,
              nullptr, nullptr,
              H_Q * D, valid_q_rows, tid);
            __syncthreads();

            // ======================================================================================
            // Compute:  S = Q @ K^T
            // Layout:   Q[row: BLOCK_N, D], K[col: BLOCK_M, D] -> S[row: BLOCK_N, col: BLOCK_M]
            // Template: GemmType::Q_KT, GemmPhase::BWKV (IS_SWAPPED=true)
            // ======================================================================================
            WMMA_GEMM_SCORES<Config, GemmType::Q_KT, GemmPhase::BWKV, IS_CAUSAL, IS_ALIBI, IS_SOFTCAP, IS_WINDOW>(
              sQ, sK, sS,
              valid_q_rows, block.valid_kv_rows,
              start_q,      block.start_kv,
              block.seqlen_offset,
              softmax_scale, softcap, alibi_slope, window_left, window_right,
              warp_id, lane_id);
            __syncthreads();

            // ======================================================================================
            // Load:     dO tile from global to sdO(reuse) shared memory
            // Layout:   dO: global[row: BLOCK_N, D] with H_Q*D -> shared[row: BLOCK_N, D]
            // Template: GemmPhase::BWKV, Stage::SNGL
            // ======================================================================================
            WMMA_GEMM_LOAD_TILE<Config, GemmPhase::BWKV, Stage::SNGL>(
              dO_ptr + start_q * H_Q * D, sdO,
              nullptr, nullptr,
              H_Q * D, valid_q_rows, tid);
            __syncthreads();

            // ======================================================================================
            // Compute:  row_dot = sum(O ⊙ dO) [dKV backward pass]
            // Layout:   O[global: tile from start_q, D], dO[shared: valid_q_rows, D] -> sRowDot[shared: valid_q_rows]
            // Template: GemmPhase::BWKV (LSE_OFFSET=1)
            // ======================================================================================
            WMMA_GEMM_DOT_PRODUCT<Config, GemmPhase::BWKV>(
              o_ptr + start_q * H_Q * D, sdO, lse_ptr, sLse, sRowDot,
              H_Q * D, valid_q_rows, start_q, tid);
            __syncthreads();

            // ======================================================================================
            // Compute:  dOV = dO @ V^T
            // Layout:   dO[row: BLOCK_N, D], V[col: BLOCK_M, D] -> dOV[row: BLOCK_N, col: BLOCK_M]
            // Template: GemmType::dO_VT, GemmPhase::BWKV (IS_SWAPPED=true)
            // ======================================================================================
            WMMA_GEMM_SCORES<Config, GemmType::dO_VT, GemmPhase::BWKV, IS_CAUSAL, IS_ALIBI, IS_SOFTCAP, IS_WINDOW>(
              sdO, sV, sdOV,
              valid_q_rows, block.valid_kv_rows,
              0, 0, 0, 1.0f, 0.0f, 0.0f, -1, -1,
              warp_id, lane_id);
            __syncthreads();

            // ======================================================================================
            // Compute:  P = softmax(S), dS = P * (dOV - row_dot) * softmax_scale
            // Layout:   S[row: BLOCK_N, BLOCK_M] -> P[row: BLOCK_N, BLOCK_M], dS[row: BLOCK_N, BLOCK_M]
            // Template: GemmPhase::BWKV
            // ======================================================================================
            WMMA_GEMM_SOFTMAX_GRADIENT<Config, GemmPhase::BWKV, IS_SOFTCAP, IS_DROPOUT>(
              sS, sdOV, sLse, sRowDot, sP, sdS,
              valid_q_rows, block.valid_kv_rows,
              softmax_scale, softcap,
              p_dropout, dropout_seed, dropout_offset,
              block.q_base + start_q, block.start_kv, block.seqlen_k, tid);
            __syncthreads();

            // ======================================================================================
            // Compute:  dV += P^T @ dO
            // Layout:   P^T[col: BLOCK_M, BLOCK_N], dO[row: BLOCK_N, D] -> dV[row: BLOCK_M, D]
            // Template: GemmType::PT_dO, GemmPhase::BWKV
            // ======================================================================================
            WMMA_GEMM_GRADIENTS<Config, GemmType::PT_dO, GemmPhase::BWKV>(
              sP, sdO, sdV,
              block.valid_kv_rows, valid_q_rows,
              warp_id, lane_id);
            __syncthreads();

            // ==================================================================================
            // Load:     Q tile from global to sQ(reuse) shared memory
            // Layout:   Q: global[row: BLOCK_N, D] with H_Q*D -> shared[row: BLOCK_N, D]
            // Template: GemmPhase::BWKV, Stage::SNGL
            // ==================================================================================
            WMMA_GEMM_LOAD_TILE<Config, GemmPhase::BWKV, Stage::SNGL>(
              q_ptr + start_q * H_Q * D, sQ,
              nullptr, nullptr,
              H_Q * D, valid_q_rows, tid);
            __syncthreads();

            // ======================================================================================
            // Compute:  dK += dS^T @ Q
            // Layout:   dS^T[col: BLOCK_M, BLOCK_N], Q[row: BLOCK_N, D] -> dK[row: BLOCK_M, D]
            // Template: GemmType::dST_Q, GemmPhase::BWKV
            // ======================================================================================
            WMMA_GEMM_GRADIENTS<Config, GemmType::dST_Q, GemmPhase::BWKV>(
              sdS, sQ, sdK,
              block.valid_kv_rows, valid_q_rows,
              warp_id, lane_id);
            __syncthreads();
        }  // END Q-TILES LOOP
        // ======================================================================================
        // Compute:  Store gradients dK + dV without normalization
        // Layout:   sdK/sdV[row: BLOCK_M, D_STRIDE] -> dK/dV_ptr[row: BLOCK_M, D] with H_Q*D
        // Template: Result::WRITE_DKV (NORMALIZE=false, DUAL_STORE=true)
        // ======================================================================================
        WMMA_GEMM_EPILOGUE<Config, Result::WRITE_DKV>(
          sdK,     dK_ptr,
          sdV,     dV_ptr,
          nullptr, H_Q * D, block.valid_kv_rows, tid);
    }
}

// ======================================================================================
// LAUNCHER
// ======================================================================================
template<int D>
void launcher_flash_attention_backward_varlen(
    const torch::Tensor& Q,
    const torch::Tensor& K,
    const torch::Tensor& V,
    const torch::Tensor& O,
    const torch::Tensor& dO,
    const torch::Tensor& softmax_lse,
          torch::Tensor& dQ,
          torch::Tensor& dK,
          torch::Tensor& dV,
          torch::Tensor& softmax_d,
    const torch::Tensor& cu_seqlens_q,
    const torch::Tensor& cu_seqlens_k,
    const int    T_Q,
    const int    T_K,
    const int    max_seqlen_q,
    const int    max_seqlen_k,
    float        softmax_scale,
    bool         is_causal,
    float        softcap,
    float        p_dropout,
    const float* alibi_slopes,
    const int    alibi_batch,
    int          window_left,
    int          window_right,
    uint64_t     dropout_seed,
    uint64_t     dropout_offset,
    cudaStream_t stream
) {
    using Config = KernelConfig<D>;

    const size_t smem = Config::TOTAL_SMEM;
    TORCH_CHECK(smem <= MAX_SMEM_PER_SM, "Shared memory exceeds 96KB for Backward varlen kernel: ", smem, " bytes (", smem / 1024, " KB)");

    const int B   = cu_seqlens_q.size(0) - 1;
    const int H_Q = Q.size(1);
    const int H_K = K.size(1);

    const int grid_dq  = (max_seqlen_q + Config::DQ::BLOCK_M  - 1) / Config::DQ::BLOCK_M;
    const int grid_dkv = (max_seqlen_k + Config::DKV::BLOCK_M - 1) / Config::DKV::BLOCK_M;
    const int grid_x   = max(grid_dq, grid_dkv);
    const dim3 grid(grid_x, 2, B * H_Q);
    const dim3 block(Config::THREADS_PER_BLOCK);

    bool is_alibi       = (alibi_slopes != nullptr);
    bool is_softcap     = (softcap > 0.0f);
    bool is_window      = (window_left >= 0 || window_right >= 0);
    bool is_dropout     = (p_dropout > 0.0f);
    bool is_paged       = false;
    bool is_rope        = false;
    bool is_interleaved = false;

    const __half* q_ptr            = reinterpret_cast<const __half*>(Q.data_ptr());
    const __half* k_ptr            = reinterpret_cast<const __half*>(K.data_ptr());
    const __half* v_ptr            = reinterpret_cast<const __half*>(V.data_ptr());
    const __half* o_ptr            = reinterpret_cast<const __half*>(O.data_ptr());
    const __half* dO_ptr           = reinterpret_cast<const __half*>(dO.data_ptr());
    float*        lse_ptr          = softmax_lse.data_ptr<float>();
    __half*       dq_ptr           = reinterpret_cast<__half*>(dQ.data_ptr());
    __half*       dk_ptr           = reinterpret_cast<__half*>(dK.data_ptr());
    __half*       dv_ptr           = reinterpret_cast<__half*>(dV.data_ptr());
    float*        softmax_d_ptr    = softmax_d.data_ptr<float>();
    const int*    cu_seqlens_q_ptr = cu_seqlens_q.data_ptr<int>();
    const int*    cu_seqlens_k_ptr = cu_seqlens_k.data_ptr<int>();

    dispatch_attention_features(is_causal, is_alibi, is_softcap, is_window, is_dropout, is_paged, is_rope, is_interleaved,
    [&](auto CAUSAL, auto ALIBI, auto SOFTCAP, auto WINDOW, auto DROPOUT, auto /*PAGED*/, auto /*ROPE*/, auto /*INTERLEAVED*/) {
        constexpr bool IS_CAUSAL  = decltype(CAUSAL)::value;
        constexpr bool IS_ALIBI   = decltype(ALIBI)::value;
        constexpr bool IS_SOFTCAP = decltype(SOFTCAP)::value;
        constexpr bool IS_WINDOW  = decltype(WINDOW)::value;
        constexpr bool IS_DROPOUT = decltype(DROPOUT)::value;

        auto kernel = flash_attention_backward_varlen_kernel<D, IS_CAUSAL, IS_ALIBI, IS_SOFTCAP, IS_WINDOW, IS_DROPOUT>;
        cudaFuncSetAttribute(kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, smem);

        kernel<<<grid, block, smem, stream>>>(
            q_ptr, k_ptr, v_ptr, o_ptr, dO_ptr, lse_ptr,
            dq_ptr, dk_ptr, dv_ptr, softmax_d_ptr,
            cu_seqlens_q_ptr, cu_seqlens_k_ptr,
            B, H_Q, H_K, T_Q, T_K, grid_dq, grid_dkv, max_seqlen_q, max_seqlen_k,
            softmax_scale, softcap, alibi_slopes, alibi_batch, window_left, window_right,
            p_dropout, dropout_seed, dropout_offset
        );
    });
}

// ======================================================================================
// WRAPPER
// ======================================================================================
std::vector<at::Tensor> flash_attention_varlen_backward(
    const at::Tensor& dout,
    const at::Tensor& q,
    const at::Tensor& k,
    const at::Tensor& v,
    const at::Tensor& out,
    const at::Tensor& softmax_lse,
    std::optional<at::Tensor>& dq,
    std::optional<at::Tensor>& dk,
    std::optional<at::Tensor>& dv,
    const at::Tensor& cu_seqlens_q,
    const at::Tensor& cu_seqlens_k,
    std::optional<at::Tensor>& alibi_slopes,
    const int    max_seqlen_q,
    const int    max_seqlen_k,
    const float  p_dropout,
    const float  softmax_scale,
    const bool   zero_tensors,
    const bool   is_causal,
    int          window_left,
    int          window_right,
    const float  softcap,
    const bool   deterministic,
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
    TORCH_CHECK(out.is_cuda() && dout.is_cuda(), "out/dout must be on CUDA");
    TORCH_CHECK(q.stride(-1) == 1 && k.stride(-1) == 1 && v.stride(-1) == 1, "Last dim of q, k, v must be contiguous");
    TORCH_CHECK(out.stride(-1) == 1 && dout.stride(-1) == 1, "Last dim of out, dout must be contiguous");

    // cu_seqlens
    TORCH_CHECK(cu_seqlens_q.dtype() == torch::kInt32 && cu_seqlens_k.dtype() == torch::kInt32, "cu_seqlens must be int32");
    TORCH_CHECK(cu_seqlens_q.is_cuda() && cu_seqlens_k.is_cuda(), "cu_seqlens must be on CUDA");
    TORCH_CHECK(cu_seqlens_q.is_contiguous() && cu_seqlens_k.is_contiguous(), "cu_seqlens must be contiguous");
    TORCH_CHECK(cu_seqlens_q.dim() == 1 && cu_seqlens_k.dim() == 1, "cu_seqlens must be 1D");

    // dimensions
    const int T_Q   = q.size(0);
    const int H_Q   = q.size(1);
    const int D     = q.size(2);
    const int T_K   = k.size(0);
    const int H_K   = k.size(1);
    const int B     = cu_seqlens_q.size(0) - 1;

    TORCH_CHECK(B > 0, "batch size must be positive");
    TORCH_CHECK(D <= 256, "head dimension must be <= 256");
    TORCH_CHECK(D % 8 == 0, "head dimension must be multiple of 8");
    TORCH_CHECK(H_Q % H_K == 0, "H_Q must be divisible by H_K for GQA/MQA");
    TORCH_CHECK(cu_seqlens_q.size(0) == B + 1, "cu_seqlens_q size mismatch");
    TORCH_CHECK(cu_seqlens_k.size(0) == B + 1, "cu_seqlens_k size mismatch");

    // softcap restrictions
    if (softcap > 0.f) {
        TORCH_CHECK(p_dropout == 0.f, "Softcapping does not support dropout");
    }

    // window edge cases
    if (window_left  >= max_seqlen_k) window_left  = -1;
    if (window_right >= max_seqlen_k) window_right = -1;

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
    at::Tensor dv_fp16 = dv.has_value() ? dv.value() : torch::empty_like(v);

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

    // GQA/MQA expansion buffers
    at::Tensor dk_expanded = (H_K != H_Q) ? torch::empty({T_K, H_Q, D}, q.options()) : dk_fp16;
    at::Tensor dv_expanded = (H_K != H_Q) ? torch::empty({T_K, H_Q, D}, q.options()) : dv_fp16;

    // dLSE tensor
    auto softmax_d = torch::empty({H_Q, T_Q}, torch::dtype(torch::kFloat32).device(q.device()));
    TORCH_CHECK(softmax_d.dtype() == torch::kFloat32, "softmax_d must be fp32");

    // zero tensors / empty case
    if (zero_tensors) {
        dq_fp16.zero_();
        dk_expanded.zero_();
        dv_expanded.zero_();
        softmax_d.zero_();
    }

    if (T_Q == 0 || max_seqlen_k == 0) {
        return {dq_fp16, dk_fp16, dv_fp16, softmax_d};
    }

    // run kernel
    auto stream = at::cuda::getCurrentCUDAStream().stream();

    #define LAUNCH_KERNEL(DIM) \
        launcher_flash_attention_backward_varlen<DIM>(q, k, v, out, dout, softmax_lse, dq_fp16, dk_expanded, dv_expanded, softmax_d, cu_seqlens_q, cu_seqlens_k, \
                                                      T_Q, T_K, max_seqlen_q, max_seqlen_k, softmax_scale, is_causal, softcap, p_dropout, alibi_ptr, alibi_batch, \
                                                       window_left, window_right, dropout_seed, dropout_offset, stream);
    switch (D) {
        case 16:  LAUNCH_KERNEL(16);  break;
        case 32:  LAUNCH_KERNEL(32);  break;
        case 64:  LAUNCH_KERNEL(64);  break;
        case 128: LAUNCH_KERNEL(128); break;
        case 256: LAUNCH_KERNEL(256); break;
        default: TORCH_CHECK(false, "Unsupported D: ", D);
    }
    #undef LAUNCH_KERNEL

    // reduce expanded gradients for GQA/MQA
    if (H_K != H_Q) {
        at::sum_out(dk_fp16, at::reshape(dk_expanded, {T_K, H_K, H_Q / H_K, D}), {2});
        at::sum_out(dv_fp16, at::reshape(dv_expanded, {T_K, H_K, H_Q / H_K, D}), {2});
    }

    return {dq_fp16, dk_fp16, dv_fp16, softmax_d};
}
