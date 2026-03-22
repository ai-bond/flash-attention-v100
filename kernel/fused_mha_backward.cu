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
        // Compute:  row_dot = sum(O ⊙ dO)
        // Layout:   O[global: total_q, D], dO[shared: valid_q_rows, D_STRIDE] -> sRowDot[shared: valid_q_rows]
        // Template: HAS_LSE_OFFSET=0, USE_FULL_MASK=0, D, D_STRIDE
        // ==================================================================================
        WMMA_GEMM_DOT_PRODUCT<GemmType::rowdot_dQ, D, D_STRIDE>(
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
            // Compute dS = exp(S - lse) * (dOV - row_dot) * scale
            // ==================================================================================
            if (tid < valid_q_rows * THREADS_PER_ROW) {
                const int row = tid / THREADS_PER_ROW;
                const int thread_in_row = tid % THREADS_PER_ROW;

                 float* sS_row   = sS   + row * N_STRIDE;
                 float* sdOV_row = sdOV + row * N_STRIDE;
                __half* sdS_row  = sdS  + row * N_STRIDE;

                const float lse_val     = sLse[row];
                const float row_dot_val = sRowDot[row];

                const int vec8_cols = valid_kv_rows / 8;
                const int vec8_per_thread = (vec8_cols + THREADS_PER_ROW - 1) / THREADS_PER_ROW;
                const int tail_start = vec8_cols * 8;

                float4* sS_vec4    = reinterpret_cast<float4*>(sS_row);
                float4* sdOV_vec4  = reinterpret_cast<float4*>(sdOV_row);
                uint4*  sdS_vec_u4 = reinterpret_cast<uint4*>(sdS_row);

                uint4 buf[4]; int cnt = 0;

                // Phase 1: compute full 8-elem chunks
                #pragma unroll
                for (int j = 0; j < vec8_per_thread; ++j) {
                    const int v8 = thread_in_row + j * THREADS_PER_ROW;
                    if (v8 >= vec8_cols) break;

                    float4 s0 = sS_vec4[v8 * 2],   s1 = sS_vec4[v8 * 2 + 1];
                    float4 d0 = sdOV_vec4[v8 * 2], d1 = sdOV_vec4[v8 * 2 + 1];

                    #define COMP(i, sf, df) \
                        float sh##i = (sf) - lse_val; \
                        float p##i = (sh##i < -80.0f) ? 0.0f : __expf(sh##i); \
                        float ds##i = p##i * softmax_scale * ((df) - row_dot_val);

                    COMP(0, s0.x, d0.x) COMP(1, s0.y, d0.y) COMP(2, s0.z, d0.z) COMP(3, s0.w, d0.w)
                    COMP(4, s1.x, d1.x) COMP(5, s1.y, d1.y) COMP(6, s1.z, d1.z) COMP(7, s1.w, d1.w)
                    #undef COMP

                    uint4 res;
                    asm volatile(
                        "{ mov.b32 %0, {%4,%5}; mov.b32 %1, {%6,%7}; mov.b32 %2, {%8,%9}; mov.b32 %3, {%10,%11}; }\n"
                        : "=r"(res.x), "=r"(res.y), "=r"(res.z), "=r"(res.w)
                        : "h"(__half_as_ushort(__float2half_rn(ds0))),
                          "h"(__half_as_ushort(__float2half_rn(ds1))),
                          "h"(__half_as_ushort(__float2half_rn(ds2))),
                          "h"(__half_as_ushort(__float2half_rn(ds3))),
                          "h"(__half_as_ushort(__float2half_rn(ds4))),
                          "h"(__half_as_ushort(__float2half_rn(ds5))),
                          "h"(__half_as_ushort(__float2half_rn(ds6))),
                          "h"(__half_as_ushort(__float2half_rn(ds7)))
                    );
                    buf[cnt++] = res;
                }
                // Phase 2: tail
                #pragma unroll
                for (int c = tail_start + thread_in_row; c < BLOCK_N; c += THREADS_PER_ROW) {
                    float s =   (c < valid_kv_rows) ? sS_row[c] : NEG_INF;
                    float dov = (c < valid_kv_rows) ? sdOV_row[c] : 0.0f;
                    float p =   (s - lse_val < -80.0f) ? 0.0f : __expf(s - lse_val);
                    float ds =  p * softmax_scale * ((c < valid_kv_rows) ? (dov - row_dot_val) : 0.0f);
                    sdS_row[c] = __float2half_rn(ds);
                }
                // Phase 3: write buffered uint4s
                #pragma unroll
                for (int i = 0; i < cnt; ++i) {
                        const int v8 = thread_in_row + i * THREADS_PER_ROW;
                        if (v8 < vec8_cols) {
                            sdS_vec_u4[v8] = buf[i];
                        }
                }
            }
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
        sdQ,    nullptr,
        dQ_ptr, nullptr,
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
            // Compute:  row_dot = sum(O ⊙ dO)
            // Layout:   O[global: valid_q_rows, D] (pre-offset = start_q*D), dO[shared: valid_q_rows, D_STRIDE] -> sRowDot[shared]
            // Template: HAS_LSE_OFFSET=1, USE_FULL_MASK=1, D, D_STRIDE
            // Note:     o_ptr must be pre-offset by caller (o_ptr + start_q*D), lse_ptr loaded with offset
            // ==================================================================================
            WMMA_GEMM_DOT_PRODUCT<GemmType::rowdot_dKV, D, D_STRIDE>(
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
            // Compute dP = exp(S - lse) & dS = P * (dOV - row_dot) * scale
            // ==================================================================================
            const int total_elements = BLOCK_N * BLOCK_M;
            const int total_pairs = (total_elements + 1) / 2;

            #pragma unroll 2
            for (int i = tid; i < total_pairs; i += THREADS_PER_BLOCK) {
                const int linear_idx0 = i * 2;
                const int linear_idx1 = linear_idx0 + 1;

                const int row0 = linear_idx0 / BLOCK_M;
                const int col0 = linear_idx0 % BLOCK_M;
                const bool has_pair = (linear_idx1 < total_elements);
                const int row1 = has_pair ? linear_idx1 / BLOCK_M : row0;
                const int col1 = has_pair ? linear_idx1 % BLOCK_M : col0 + 1;

                // Load S and dOV
                float s0 = 0.0f, s1 = 0.0f;
                float dov0 = 0.0f, dov1 = 0.0f;

                const bool in_bounds0 = (row0 < valid_q_rows) && (col0 < valid_kv_rows);
                const bool causal_ok0 = !IS_CAUSAL || ((start_kv + col0) <= (start_q + row0));
                const bool valid0 = in_bounds0 && causal_ok0;

                bool valid1 = false;
                if (has_pair) {
                    const bool in_bounds1 = (row1 < valid_q_rows) && (col1 < valid_kv_rows);
                    const bool causal_ok1 = !IS_CAUSAL || ((start_kv + col1) <= (start_q + row1));
                    valid1 = in_bounds1 && causal_ok1;
                }

                if (valid0) { s0 = sS[row0 * M_STRIDE + col0]; dov0 = sdOV[row0 * M_STRIDE + col0]; } else { s0 = NEG_INF; }
                if (valid1) { s1 = sS[row1 * M_STRIDE + col1]; dov1 = sdOV[row1 * M_STRIDE + col1]; } else { s1 = NEG_INF; }

                float lse0 = sLse[row0];
                float lse1 = (valid1 && row1 != row0) ? sLse[row1] : lse0;
                float row_dot0 = sRowDot[row0];
                float row_dot1 = (valid1 && row1 != row0) ? sRowDot[row1] : row_dot0;

                // Compute P0, P1 in float
                float shifted0 = s0 - lse0;
                float shifted1 = s1 - lse1;

                float p0 = (shifted0 < -80.0f) ? 0.0f : __expf(shifted0);
                float p1 = (shifted1 < -80.0f) ? 0.0f : __expf(shifted1);

                // Compute dS
                float diff0 = dov0 - row_dot0;
                float diff1 = dov1 - row_dot1;
                float ds0 = valid0 ? fmaf(p0, softmax_scale * diff0, 0.0f) : 0.0f;
                float ds1 = valid1 ? fmaf(p1, softmax_scale * diff1, 0.0f) : 0.0f;

                // Convert P and dS to half
                __half2 p_h2 = __float22half2_rn(make_float2(p0, p1));
                __half2 ds_h2 = __float22half2_rn(make_float2(ds0, ds1));

                // Store P and dS in half (row-major, BLOCK_M stride)
                if (col1 < BLOCK_M && row1 == row0) {
                    __half* p_dst  = sP  + row0 * BLOCK_M + col0;
                    __half* ds_dst = sdS + row0 * BLOCK_M + col0;
                    p_dst[0]  = p_h2.x;  p_dst[1]  = p_h2.y;
                    ds_dst[0] = ds_h2.x; ds_dst[1] = ds_h2.y;
                } else {
                    if (valid0) { sP[row0 * BLOCK_M + col0] = p_h2.x; sdS[row0 * BLOCK_M + col0] = ds_h2.x; }
                    if (valid1) { sP[row1 * BLOCK_M + col1] = p_h2.y; sdS[row1 * BLOCK_M + col1] = ds_h2.y; }
                }
            }
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
        sdK,    sdV,
        dK_ptr, dV_ptr,
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
