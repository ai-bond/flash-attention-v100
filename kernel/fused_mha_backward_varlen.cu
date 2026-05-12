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
__global__ void __launch_bounds__(KernelConfig<D>::THREADS_PER_BLOCK, 2)
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
    // PHASE 1: dQ (blockIdx.y == 0)
    // ======================================================================================
    if (blockIdx.y == 0) {
        if (blockIdx.x >= grid_dq) return;

        using Config = KernelConfig<D>;
        constexpr int BLOCK_M   = Config::DQ::BLOCK_M;
        constexpr int BLOCK_N   = Config::DQ::BLOCK_N;
        constexpr int D_STRIDE  = Config::DQ::D_STRIDE;
        constexpr int N_STRIDE  = Config::DQ::N_STRIDE;

        // ======================================================================================
        // Grid Mapping: 1D X for Q-blocks, Z for heads. Batch resolved device-side.
        // ======================================================================================
        const int block_idx     = blockIdx.x;
        const int bthd_idx      = blockIdx.z;

        if (bthd_idx >= B * H_Q) return;

        // ======================================================================================
        // BlockInfo: Metadata resolution
        // ======================================================================================
        BlockInfo<IS_CAUSAL, IS_WINDOW, true> block;
        block.init_q(
            block_idx,       // BLOCK_IDX:      Current Q-block index (grid.x)
            bthd_idx,        // BATCH_HEAD_ID:  Batch head index
            H_Q,             // H_Q:            Number of query heads
            H_K,             // H_K:            Number of KV heads
            0,               // M:              (unused for varlen)
            0,               // N:              (unused for varlen)
            B,               // B:              B (batch size for device-side loop)
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
        // Writes deterministic zeros to dQ and softmax_d for numerical stability.
        // ======================================================================================
        if (block.block_min >= block.block_max || block.seqlen_k == 0) {
            const size_t dq_base   = block.q_offset(D, H_Q, T_Q);
            const size_t dlse_base = block.lse_offset(H_Q, T_Q);
            for (int row = threadIdx.x; row < block.valid_q_rows; row += blockDim.x) {
                #pragma unroll
                for (int d = 0; d < D; ++d) {
                    dQ[dq_base + row * H_Q * D + d] = __float2half(0.0f);
                }
                softmax_d[dlse_base + row] = 0.0f;
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
        // RAGGED POINTERS
        // Layout:
        //   Q/O/dO/dQ: [T_Q, H_Q, D]            offset follows q_base + start_q (Q, H_Q, D)
        //   K/V:       [T_K, H_K, D]            offset follows k_base (K, H_K, D)
        //   LSE/dLSE:  [H_Q, T_Q]               offset follows bthd_idx * T_Q (H_Q, T_Q)
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
        // INIT SHARED MEMORY
        // ======================================================================================
        extern __shared__ char smem_raw[];

        WMMA_GEMM_INIT_SMEM<Config>(smem_raw);

        __syncthreads();

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

        // ======================================================================================
        // Load:     Q & dO tiles from global to sQ/sdO shared memory
        // Layout:   Q:  global[row: BLOCK_M, H_Q * D] -> shared[row: BLOCK_M, D_STRIDE]
        //           dO: global[row: BLOCK_M, H_Q * D] -> shared[row: BLOCK_M, D_STRIDE]
        // Template: DUAL_LOAD=true, SMEM_STRIDE=D_STRIDE, GLOBAL_WIDTH=D
        // ======================================================================================
        WMMA_GEMM_LOAD_TILE<Config, true, D_STRIDE, D>(
          q_ptr,   sQ,
          dO_ptr,  sdO,
          H_Q * D, block.valid_q_rows, tid);
        __syncthreads();

        // ======================================================================================
        // Compute:  row_dot = sum(O ⊙ dO) element-wise
        // Layout:   o_ptr[valid_q_rows, H_Q * D] ⊙ sdO[valid_q_rows, D_STRIDE] -> sRowDot[valid_q_rows]
        // Template: D_STRIDE=D_STRIDE, HEAD_STRIDE=D_STRIDE
        // ======================================================================================
        WMMA_GEMM_DOT_PRODUCT<Config, GemmType::rowdot_dQ, D_STRIDE>(
          o_ptr, sdO, lse_ptr, sLse, sRowDot,
          D, block.valid_q_rows, 0, tid);
        __syncthreads();

        // ======================================================================================
        // MAIN LOOP (Iterates over logical KV blocks)
        // ======================================================================================
        for (int block_q = block.block_min; block_q < block.block_max; ++block_q) {
            const int start_kv      = block_q * BLOCK_N;
            const int valid_kv_rows = min(BLOCK_N, block.seqlen_k - start_kv);
            if (valid_kv_rows <= 0) break;

            // ======================================================================================
            // Load:     V tile from global to sV shared memory
            // Layout:   V: global[row: BLOCK_N, H_K * D] -> shared[row: BLOCK_N, D_STRIDE]
            // Template: DUAL_LOAD=false, SMEM_STRIDE=D_STRIDE, GLOBAL_WIDTH=D (varlen sub-tile)
            // ======================================================================================
            WMMA_GEMM_LOAD_TILE<Config, false, D_STRIDE, D>(
              v_ptr + start_kv * H_K * D, sV,
              nullptr,                    nullptr,
              H_K * D, valid_kv_rows, tid);
            __syncthreads();

            // ======================================================================================
            // Compute:  dOV = dO @ V^T
            // Layout:   sdO[valid_q_rows, D_STRIDE] @ sV^T -> sdOV[valid_q_rows, N_STRIDE]
            // Template: BLOCK_M/BLOCK_N static, valid_q/valid_kv dynamic (varlen ragged tiles)
            // ======================================================================================
            WMMA_GEMM_SCORES<Config, GemmType::dOV_dOVT, D, IS_CAUSAL, IS_ALIBI, IS_SOFTCAP, IS_WINDOW, BLOCK_M, BLOCK_N, D_STRIDE, N_STRIDE>(
              sdO, sV, sdOV,
              block.valid_q_rows,  valid_kv_rows,
              0,                   0,
              0,
              1.0f, 0.0f, 0.0f, -1, -1,
              warp_id, lane_id);
            __syncthreads();

            // ======================================================================================
            // Load:     K tile from global to sK shared memory
            // Layout:   K: global[row: BLOCK_N, H_K * D] -> shared[row: BLOCK_N, D_STRIDE]
            // Template: DUAL_LOAD=false, SMEM_STRIDE=D_STRIDE, GLOBAL_WIDTH=D (varlen sub-tile)
            // ======================================================================================
            WMMA_GEMM_LOAD_TILE<Config, false, D_STRIDE, D>(
              k_ptr + start_kv * H_K * D, sK,
              nullptr,                    nullptr,
              H_K * D, valid_kv_rows, tid);
            __syncthreads();

            // ======================================================================================
            // Compute:  S = Q @ K^T
            // Layout:   sQ[valid_q_rows, D_STRIDE] @ sK^T -> sS[valid_q_rows, N_STRIDE]
            // Template: BLOCK_M/BLOCK_N static, valid_q/valid_kv dynamic (varlen ragged tiles)
            // ======================================================================================
            WMMA_GEMM_SCORES<Config, GemmType::sQ_KT, D, IS_CAUSAL, IS_ALIBI, IS_SOFTCAP, IS_WINDOW, BLOCK_M, BLOCK_N, D_STRIDE, N_STRIDE>(
              sQ, sK, sS,
              block.valid_q_rows,  valid_kv_rows,
              block.start_q,       start_kv,
              block.seqlen_offset,
              softmax_scale, softcap, alibi_slope, window_left, window_right,
              warp_id, lane_id);
            __syncthreads();

            // ======================================================================================
            // Compute:  dS = exp(S - lse) * (dOV - row_dot) * softmax_scale
            // Layout:   sS[valid_q_rows, N_STRIDE], sdOV[valid_q_rows, N_STRIDE]
            //           -> sdS[valid_q_rows, N_STRIDE]
            // Varlen:   GLOBAL_N=block.seqlen_k (actual KV length) for correct dropout RNG stride
            // Template: IS_DROPOUT guards compile-time; runtime p_dropout > 0 enables execution
            // ======================================================================================
            WMMA_GEMM_SOFTMAX_GRADIENT<Config, GemmType::compute_dS, IS_SOFTCAP, IS_DROPOUT, N_STRIDE, N_STRIDE, BLOCK_M, BLOCK_N>(
              sS, sdOV, sLse, sRowDot, nullptr, sdS,
              block.valid_q_rows, valid_kv_rows,
              softmax_scale, softcap,
              p_dropout, dropout_seed, dropout_offset,
              block.q_base + block.start_q, start_kv, block.seqlen_k,
              tid);
            __syncthreads();

            // ======================================================================================
            // Compute:  dQ += dS @ K
            // Layout:   sdS[valid_q_rows, N_STRIDE] @ sK[valid_kv_rows, D_STRIDE] += sdQ
            // Template: BLOCK_M/BLOCK_N static, valid_q/valid_kv dynamic (varlen ragged tiles)
            // ======================================================================================
            WMMA_GEMM_GRADIENTS<Config, GemmType::dQ_dSK, D, BLOCK_M, BLOCK_N, N_STRIDE, D_STRIDE>(
              sdS, sK, sdQ,
              block.valid_q_rows, valid_kv_rows,
              warp_id, lane_id);
            __syncthreads();
        }   // END MAIN LOOP
        // ======================================================================================
        // Compute:  Store dQ and dLSE to global memory
        // Layout:   sdQ[row: valid_q_rows, D_STRIDE] -> dQ_ptr[row: valid_q_rows, H_Q * D]
        //           sRowDot[valid_q_rows] -> softmax_d_ptr[valid_q_rows]
        // Template: SMEM_STRIDE=D_STRIDE, GLOBAL_WIDTH=D (varlen sub-tile store)
        // ======================================================================================
        WMMA_GEMM_EPILOGUE<Config, GemmType::write_dQ, D_STRIDE, D>(
          sdQ,     dQ_ptr,
          nullptr, nullptr,
          nullptr, H_Q * D, block.valid_q_rows, tid);

        if (tid < block.valid_q_rows) {
            dLse_ptr[tid] = sRowDot[tid];
        }
    }
    // ======================================================================================
    // PHASE 2: dKV computation (blockIdx.y == 1)
    // ======================================================================================
    else if (blockIdx.y == 1) {
        if (blockIdx.x >= grid_dkv) return;

        using Config = KernelConfig<D>;
        constexpr int BLOCK_M   = Config::DKV::BLOCK_M;
        constexpr int BLOCK_N   = Config::DKV::BLOCK_N;
        constexpr int D_STRIDE  = Config::DKV::D_STRIDE;
        constexpr int M_STRIDE  = Config::DKV::M_STRIDE;

        // ======================================================================================
        // Grid Mapping: 1D X for KV-blocks, Z for bthd_idx. Batch resolved device-side.
        // ======================================================================================
        const int block_idx     = blockIdx.x;
        const int bthd_idx      = blockIdx.z;

        if (bthd_idx >= B * H_Q) return;

        // ======================================================================================
        // BlockInfo: Metadata resolution
        // ======================================================================================
        BlockInfo<IS_CAUSAL, IS_WINDOW, true> block;
        block.init_kv(
            block_idx,       // BLOCK_IDX:      Current KV-block index (grid.x)
            bthd_idx,        // BATCH_HEAD_ID:  Batch head index (KV head)
            H_Q,             // H_Q:            Number of query heads
            H_K,             // H_K:            Number of KV heads
            0,               // M:              (unused for varlen)
            0,               // N:              (unused for varlen)
            B,               // B:              B (batch size for device-side loop)
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
        // EARLY EXIT: No valid Q blocks attend to this KV block.
        // Writes deterministic zeros to dK and dV for ALL mapped Q-heads (GQA/MQA safe).
        // ======================================================================================
        if (block.block_min >= block.block_max || block.seqlen_q == 0) {
            for (int group = 0; group < (H_Q / H_K); ++group) {
                const int q_head = block.kv_head_idx * (H_Q / H_K) + group;
                const size_t dk_base = static_cast<size_t>(block.k_base + block.start_kv) * H_Q * D + static_cast<size_t>(q_head) * D;
                const size_t dv_base = static_cast<size_t>(block.k_base + block.start_kv) * H_Q * D + static_cast<size_t>(q_head) * D;
                for (int row = threadIdx.x; row < block.valid_kv_rows; row += blockDim.x) {
                    #pragma unroll
                    for (int d = 0; d < D; ++d) {
                        dK[dk_base + row * H_Q * D + d] = __float2half(0.0f);
                        dV[dv_base + row * H_Q * D + d] = __float2half(0.0f);
                    }
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

        // ======================================================================================
        // RAGGED POINTERS (Base for KV)
        // Layout:
        //   K/V:       [T_K, H_K, D]            offset follows k_base (K, H_K, D)
        //   dK/dV:     [T_K, H_Q, D] expanded   offset follows k_base + start_kv (KV, H_Q, D)
        // ======================================================================================
        const __half* __restrict__ k_ptr  = K  + block.kv_offset(D, H_K, T_K);
        const __half* __restrict__ v_ptr  = V  + block.kv_offset(D, H_K, T_K);
        __half* __restrict__ dK_ptr_base  = dK + static_cast<size_t>(block.k_base + block.start_kv) * H_Q * D;
        __half* __restrict__ dV_ptr_base  = dV + static_cast<size_t>(block.k_base + block.start_kv) * H_Q * D;

        // ======================================================================================
        // INIT SHARED MEMORY
        // ======================================================================================
        extern __shared__ char smem_raw[];

        WMMA_GEMM_INIT_SMEM<Config>(smem_raw);

        __syncthreads();

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

        // ======================================================================================
        // Load:     K & V tiles from global to sK/sV shared memory
        // Layout:   K: global[row: BLOCK_M, H_K * D] -> shared[row: BLOCK_M, D_STRIDE]
        //           V: global[row: BLOCK_M, H_K * D] -> shared[row: BLOCK_M, D_STRIDE]
        // Template: DUAL_LOAD=true, SMEM_STRIDE=D_STRIDE, GLOBAL_WIDTH=D
        // ======================================================================================
        WMMA_GEMM_LOAD_TILE<Config, true, D_STRIDE, D>(
          k_ptr + block.start_kv * H_K * D, sK,
          v_ptr + block.start_kv * H_K * D, sV,
          H_K * D, block.valid_kv_rows, tid);
        __syncthreads();

        // ======================================================================================
        // Q-HEADS LOOP (GQA/MQA Expansion)
        // Iterates over all Q-heads that map to the current KV-head
        // ======================================================================================
        for (int group = 0; group < (H_Q / H_K); ++group) {

            // ======================================================================================
            // RAGGED POINTERS for current Q-head
            // Layout:
            //   Q/O/dO:    [T_Q, H_Q, D]   offset follows q_base + bthd_idx * D
            //   LSE:       [H_Q, T_Q]       offset follows bthd_idx * T_Q + q_base
            //   dK/dV:     [T_K, H_Q, D]    offset follows k_base + start_kv + bthd_idx * D
            // ======================================================================================
            const size_t q_head_off = static_cast<size_t>(block.q_base) * H_Q * D + static_cast<size_t>(bthd_idx) * D;
            const __half* __restrict__ q_ptr   = Q  + q_head_off;
            const __half* __restrict__ o_ptr   = O  + q_head_off;
            const __half* __restrict__ dO_ptr  = dO + q_head_off;
            const float*  __restrict__ lse_ptr = softmax_lse + static_cast<size_t>(bthd_idx) * T_Q + block.q_base;
            // Alibi slope only for valid block + batch
            const int   alibi_idx   = (alibi_batch > 0) ? (block.batch_idx * alibi_batch + (block.kv_head_idx * (H_Q / H_K) + group)) : block.kv_head_idx * (H_Q / H_K) + group;
            const float alibi_slope = (alibi_slopes != nullptr) ? alibi_slopes[alibi_idx] : 0.0f;

            __half* __restrict__ dK_ptr = dK_ptr_base + static_cast<size_t>(bthd_idx) * D;
            __half* __restrict__ dV_ptr = dV_ptr_base + static_cast<size_t>(bthd_idx) * D;

            // ======================================================================================
            // Q-HEADS LOOP (Iterate over Q-head groups sharing this KV-head)
            // ======================================================================================
            for (int block_q = block.block_min; block_q < block.block_max; ++block_q) {
                const int start_q      = block_q * BLOCK_N;
                const int valid_q_rows = min(BLOCK_N, block.seqlen_q - start_q);
                if (valid_q_rows <= 0) break;

                // ======================================================================================
                // Load:     Q tile from global to sQ shared memory
                // Layout:   Q: global[row: BLOCK_N, H_Q * D] -> shared[row: BLOCK_N, D_STRIDE]
                // Template: DUAL_LOAD=false, SMEM_STRIDE=D_STRIDE, GLOBAL_WIDTH=D (varlen sub-tile)
                // ======================================================================================
                WMMA_GEMM_LOAD_TILE<Config, false, D_STRIDE, D>(
                  q_ptr + start_q * H_Q * D, sQ,
                  nullptr,                   nullptr,
                  H_Q * D, valid_q_rows, tid);
                __syncthreads();

                // ======================================================================================
                // Compute:  S = Q @ K^T
                // Layout:   sQ[valid_q_rows, D_STRIDE] @ sK^T -> sS[valid_q_rows, M_STRIDE]
                // Template: BLOCK_M/BLOCK_N static, valid_q/valid_kv dynamic (varlen ragged tiles)
                // Note:     M/N swapped for dKV phase (transposed layout)
                // ======================================================================================
                WMMA_GEMM_SCORES<Config, GemmType::sQ_KT, D, IS_CAUSAL, IS_ALIBI, IS_SOFTCAP, IS_WINDOW, BLOCK_N, BLOCK_M, D_STRIDE, M_STRIDE>(
                  sQ, sK, sS,
                  valid_q_rows,          block.valid_kv_rows,
                  start_q,               block.start_kv,
                  block.seqlen_offset,
                  softmax_scale, softcap, alibi_slope, window_left, window_right,
                  warp_id, lane_id);
                __syncthreads();

                // ======================================================================================
                // Load:     dO tile from global to sdO shared memory
                // Layout:   dO: global[row: BLOCK_N, H_Q * D] -> shared[row: BLOCK_N, D_STRIDE]
                // Template: DUAL_LOAD=false, SMEM_STRIDE=D_STRIDE, GLOBAL_WIDTH=D (varlen sub-tile)
                // ======================================================================================
                WMMA_GEMM_LOAD_TILE<Config, false, D_STRIDE, D>(
                  dO_ptr + start_q * H_Q * D, sdO,
                  nullptr,                    nullptr,
                  H_Q * D, valid_q_rows, tid);
                __syncthreads();

                // ======================================================================================
                // Compute:  row_dot = sum(O ⊙ dO) element-wise
                // Layout:   o_ptr[start_q:valid_q_rows, H_Q * D] ⊙ sdO -> sRowDot[valid_q_rows]
                // Template: D_STRIDE=D_STRIDE
                // ======================================================================================
                WMMA_GEMM_DOT_PRODUCT<Config, GemmType::rowdot_dKV, D_STRIDE>(
                  o_ptr + start_q * H_Q * D, sdO, lse_ptr, sLse, sRowDot,
                  D, valid_q_rows, start_q, tid);
                __syncthreads();

                // ======================================================================================
                // Compute:  dOV = dO @ V^T
                // Layout:   sdO[valid_q_rows, D_STRIDE] @ sV^T -> sdOV[valid_q_rows, M_STRIDE]
                // Template: BLOCK_N/BLOCK_M static, valid_q/valid_kv dynamic (varlen ragged tiles)
                // Note:     M/N swapped for dKV phase (transposed layout)
                // ======================================================================================
                WMMA_GEMM_SCORES<Config, GemmType::dOV_dOVT, D, IS_CAUSAL, IS_ALIBI, IS_SOFTCAP, IS_WINDOW, BLOCK_N, BLOCK_M, D_STRIDE, M_STRIDE>(
                  sdO, sV, sdOV,
                  valid_q_rows,          block.valid_kv_rows,
                  0,                     0,
                  0,
                  1.0f, 0.0f, 0.0f, -1, -1,
                  warp_id, lane_id);
                __syncthreads();

                // ======================================================================================
                // Compute:  P = softmax(S), dS = P * (dOV - row_dot) * softmax_scale
                // Layout:   sS[valid_q_rows, M_STRIDE] -> sP[valid_q_rows, M_STRIDE]
                //           sdS[valid_q_rows, M_STRIDE]
                // Varlen:   GLOBAL_N=block.seqlen_k for correct dropout RNG stride
                // Template: IS_DROPOUT guards compile-time; runtime p_dropout > 0 enables execution
                // ======================================================================================
                WMMA_GEMM_SOFTMAX_GRADIENT<Config, GemmType::compute_P_dS, IS_SOFTCAP, IS_DROPOUT, M_STRIDE, BLOCK_M, BLOCK_N, BLOCK_M>(
                  sS, sdOV, sLse, sRowDot, sP, sdS,
                  valid_q_rows, block.valid_kv_rows,
                  softmax_scale, softcap,
                  p_dropout, dropout_seed, dropout_offset,
                  block.q_base + start_q, block.start_kv, block.seqlen_k,
                  tid);
                __syncthreads();

                // ======================================================================================
                // Compute:  dV += P^T @ dO
                // Layout:   sP^T[valid_kv_rows, M_STRIDE] @ sdO[valid_q_rows, D_STRIDE] += sdV
                // Template: BLOCK_M/BLOCK_N static, valid_kv/valid_q dynamic (varlen ragged tiles)
                // ======================================================================================
                WMMA_GEMM_GRADIENTS<Config, GemmType::dV_PTdO, D, BLOCK_M, BLOCK_N, BLOCK_M, D_STRIDE>(
                  sP, sdO, sdV,
                  block.valid_kv_rows, valid_q_rows,
                  warp_id, lane_id);
                __syncthreads();

                // ======================================================================================
                // Compute:  dK += dS^T @ Q
                // Layout:   sdS^T[valid_kv_rows, M_STRIDE] @ sQ[valid_q_rows, D_STRIDE] += sdK
                // Template: BLOCK_M/BLOCK_N static, valid_kv/valid_q dynamic (varlen ragged tiles)
                // ======================================================================================
                WMMA_GEMM_GRADIENTS<Config, GemmType::dK_dSTQ, D, BLOCK_M, BLOCK_N, BLOCK_M, D_STRIDE>(
                  sdS, sQ, sdK,
                  block.valid_kv_rows, valid_q_rows,
                  warp_id, lane_id);
                __syncthreads();
            } // END Q-TILES LOOP
            // ======================================================================================
            // Compute:  Store dK & dV for current Q-head to global memory (Expanded Layout)
            // Layout:   sdK[row: valid_kv_rows, D_STRIDE] -> dK_ptr[row: valid_kv_rows, H_Q * D]
            //           sdV[row: valid_kv_rows, D_STRIDE] -> dV_ptr[row: valid_kv_rows, H_Q * D]
            // Template: DUAL_STORE=true, SMEM_STRIDE=D_STRIDE, GLOBAL_WIDTH=D (varlen sub-tile store)
            // ======================================================================================
            WMMA_GEMM_EPILOGUE<Config, GemmType::write_dKV, D_STRIDE, D>(
              sdK,     dK_ptr,
              sdV,     dV_ptr,
              nullptr, H_Q * D, block.valid_kv_rows, tid);
        } // END Q-HEADS LOOP
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

    bool is_alibi   = (alibi_slopes != nullptr);
    bool is_softcap = (softcap > 0.0f);
    bool is_window  = (window_left >= 0 || window_right >= 0);
    bool is_dropout = (p_dropout > 0.0f);
    bool is_paged   = false;

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
    [&](auto CAUSAL, auto ALIBI, auto SOFTCAP, auto WINDOW, auto DROPOUT, auto PAGED, auto /*ROPE*/, auto /*INTERLEAVED*/) {
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
            B, H_Q, H_K, T_Q, T_K, grid_dq, grid_dkv,
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
    const int   max_seqlen_q,
    const int   max_seqlen_k,
    const float p_dropout,
    const float softmax_scale,
    const bool  zero_tensors,
    const bool  is_causal,
    int         window_left,
    int         window_right,
    const float softcap,
    const bool  deterministic,
    std::optional<at::Generator> gen,
    std::optional<at::Tensor>& rng_state
) {
    // Device guard for multi-GPU / pipeline-parallelism
    at::cuda::CUDAGuard device_guard{q.device()};

    auto props = at::cuda::getCurrentDeviceProperties();
    TORCH_CHECK(props->major == 7 && props->minor == 0, "Kernel supports only Volta GPUs.");
    TORCH_CHECK(!deterministic, "Deterministic backward not supported in this Volta build");

    TORCH_CHECK(q.dtype() == torch::kFloat16, "q must be fp16");
    TORCH_CHECK(k.dtype() == torch::kFloat16, "k must be fp16");
    TORCH_CHECK(v.dtype() == torch::kFloat16, "v must be fp16");
    TORCH_CHECK(out.dtype() == torch::kFloat16, "out must be fp16");
    TORCH_CHECK(dout.dtype() == torch::kFloat16, "dout must be fp16");
    TORCH_CHECK(softmax_lse.dtype() == torch::kFloat32, "softmax_lse must be fp32");

    TORCH_CHECK(q.is_cuda() && k.is_cuda() && v.is_cuda(), "q, k, v must be on CUDA");
    TORCH_CHECK(q.stride(-1) == 1 && k.stride(-1) == 1 && v.stride(-1) == 1, "Last dim of q, k, v must be contiguous");
    TORCH_CHECK(out.stride(-1) == 1 && dout.stride(-1) == 1, "Last dim of out, dout must be contiguous");

    TORCH_CHECK(cu_seqlens_q.dtype() == torch::kInt32 && cu_seqlens_k.dtype() == torch::kInt32, "cu_seqlens must be int32");
    TORCH_CHECK(cu_seqlens_q.is_cuda() && cu_seqlens_k.is_cuda(), "cu_seqlens must be on CUDA device");
    TORCH_CHECK(cu_seqlens_q.is_contiguous() && cu_seqlens_k.is_contiguous(), "cu_seqlens must be contiguous");
    TORCH_CHECK(cu_seqlens_q.dim() == 1 && cu_seqlens_k.dim() == 1, "cu_seqlens must be 1D tensors");

    const int T_Q   = q.size(0);
    const int H_Q   = q.size(1);
    const int D     = q.size(2);
    const int T_K   = k.size(0);
    const int H_K   = k.size(1);
    const int B     = cu_seqlens_q.size(0) - 1;

    TORCH_CHECK(B > 0, "batch_size must be positive");
    TORCH_CHECK(D <= 256 && D % 8 == 0, "D must be even, <=256, multiple of 8");
    TORCH_CHECK(H_Q % H_K == 0, "H_Q must be divisible by H_K for GQA/MQA");

    // Window edge cases
    if (window_left  >= max_seqlen_k) window_left  = -1;
    if (window_right >= max_seqlen_k) window_right = -1;

    // Alibi
    const float* alibi = nullptr;
    int alibi_batch = 0;
    if (alibi_slopes.has_value()) {
        const auto& slopes = alibi_slopes.value();
        auto sizes = slopes.sizes();
        TORCH_CHECK(slopes.dtype() == torch::kFloat32 && slopes.is_cuda(), "alibi_slopes must be fp32 on CUDA");
        TORCH_CHECK(slopes.stride(-1) == 1, "alibi_slopes last dim must be contiguous");
        bool valid_shape = (sizes.size() == 1 && sizes[0] == H_Q) ||
                           (sizes.size() == 2 && sizes[0] == B && sizes[1] == H_Q);
        TORCH_CHECK(valid_shape, "alibi_slopes shape must be [H_Q] or [B, H_Q], got ", sizes);
        alibi_batch = (slopes.dim() == 2) ? slopes.stride(0) : 0;
        alibi = slopes.data_ptr<float>();
    }

    // Dropout
    TORCH_CHECK(p_dropout >= 0.f && p_dropout < 1.f, "p_dropout must be in [0, 1)");
    if (softcap > 0.f) { TORCH_CHECK(p_dropout == 0.f, "Softcapping does not support dropout"); }

    uint64_t dropout_seed   = 0;
    uint64_t dropout_offset = 0;
    if (p_dropout > 0.0f) {
        TORCH_CHECK(rng_state.has_value() && rng_state->numel() == 2, "rng_state required when p_dropout > 0");
        auto rng_cpu = rng_state->cpu();
        auto rng_acc = rng_cpu.accessor<int64_t, 1>();
        dropout_seed   = static_cast<uint64_t>(rng_acc[0]);
        dropout_offset = static_cast<uint64_t>(rng_acc[1]);
    }

    // Output tensors
    at::Tensor dq_fp16 = dq.has_value() ? dq.value() : torch::empty_like(q);
    at::Tensor dk_fp16 = dk.has_value() ? dk.value() : torch::empty_like(k);
    at::Tensor dv_fp16 = dv.has_value() ? dv.value() : torch::empty_like(v);

    // GQA/MQA expansion buffers [T_K, H_Q, D] for deterministic host-side reduction
    at::Tensor dk_expanded = (H_K != H_Q) ? torch::empty({T_K, H_Q, D}, q.options()) : dk_fp16;
    at::Tensor dv_expanded = (H_K != H_Q) ? torch::empty({T_K, H_Q, D}, q.options()) : dv_fp16;

    // dLSE tensor matches forward LSE layout [H_Q, T_Q]
    auto softmax_d = torch::empty({H_Q, T_Q}, torch::dtype(torch::kFloat32).device(q.device()));

    // Edge-case return
    if (zero_tensors) {
        dq_fp16.zero_();
        dk_expanded.zero_();
        dv_expanded.zero_();
        softmax_d.zero_();
    }

    if (T_Q == 0 || max_seqlen_k == 0) {
        return {dq_fp16, dk_fp16, dv_fp16, softmax_d};
    }

    // Run kernel
    auto stream = at::cuda::getCurrentCUDAStream().stream();

    #define LAUNCH_KERNEL(DIM) \
        launcher_flash_attention_backward_varlen<DIM>(q, k, v, out, dout, softmax_lse, dq_fp16, dk_expanded, dv_expanded, softmax_d, \
            cu_seqlens_q, cu_seqlens_k, T_Q, T_K, max_seqlen_q, max_seqlen_k, \
            softmax_scale, is_causal, softcap, p_dropout, alibi, alibi_batch, \
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

    // Reduce expanded gradients for GQA/MQA
    if (H_K != H_Q) {
        at::sum_out(dk_fp16, at::reshape(dk_expanded, {T_K, H_K, H_Q / H_K, D}), {2});
        at::sum_out(dv_fp16, at::reshape(dv_expanded, {T_K, H_K, H_Q / H_K, D}), {2});
    }

    return {dq_fp16, dk_fp16, dv_fp16, softmax_d};
}
