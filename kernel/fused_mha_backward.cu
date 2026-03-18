// ============================================================================
// * Copyright (c) 2025, D.Skryabin / tg @ai_bond007 SPDX-License: BSD-3-Clause
// ============================================================================
#include <cuda.h>
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <torch/extension.h>

#include <ATen/ATen.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>

#include <mma.h>
using namespace nvcuda::wmma;

#include "00_volta_const.cuh"
#include "01_backward_config.cuh"
#include "02_fused_func.cuh"

// ============================================================================
// BACKWARD KERNEL
// ============================================================================
template<int D, bool IS_CAUSAL>
__global__ void __launch_bounds__(KernelConfig<D>::MAX_THREADS, 2)
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
    using Config = KernelConfig<D>;

    // head index (batch * num_heads + head)
    const int batch_head_id = blockIdx.z;
    if (batch_head_id >= B * H) return;

    const int tid          = threadIdx.x;
    const int warp_id      = tid >> 5;
    const int lane_id      = tid & 31;

    // Shared memory init
    extern __shared__ char smem_raw[];
    init_smem<Config>(smem_raw);
    auto& smem = *reinterpret_cast<typename Config::SmemLayout*>(smem_raw);

    // =========================================================================
    // PHASE 1: dQ
    // =========================================================================
    if (blockIdx.y == 0) {
        if (blockIdx.x >= grid_dq_limit) return;

        using Config = KernelConfig<D>;

        constexpr int BLOCK_M            = Config::DQ::BLOCK_M;
        constexpr int BLOCK_N            = Config::DQ::BLOCK_N;
        constexpr int THREADS_PER_BLOCK  = Config::DQ::THREADS_PER_BLOCK;
        constexpr int THREADS_PER_ROW    = Config::DQ::THREADS_PER_ROW;
        constexpr int WARPS_PER_BLOCK    = Config::DQ::WARPS_PER_BLOCK;
        constexpr int Q_STRIDE           = Config::DQ::Q_STRIDE;
        constexpr int KV_STRIDE          = Config::DQ::KV_STRIDE;
        constexpr int S_STRIDE           = Config::DQ::S_STRIDE;
        constexpr int PER_UINT4          = Config::DQ::PER_UINT4;
        constexpr int NUM_UINT4_Q_BLOCK  = Config::DQ::NUM_UINT4_Q_BLOCK;
        constexpr int NUM_UINT4_KV_BLOCK = Config::DQ::NUM_UINT4_KV_BLOCK;

        const int block_idx = blockIdx.x;
        const int start_q   = block_idx * BLOCK_M;
        if (start_q  >= M) return;

        int num_kv_tiles = (N + BLOCK_N - 1)  / BLOCK_N;
        const int valid_q_rows  = min(BLOCK_M, M - start_q);

        // Early loop limit for causal
        if constexpr (IS_CAUSAL) {
            const int max_key_pos = start_q + valid_q_rows - 1;
            if (max_key_pos < 0) {
               num_kv_tiles = 0;
            } else {
                num_kv_tiles = min(num_kv_tiles, (max_key_pos + BLOCK_N) / BLOCK_N);
            }
        }

        // Global pointers
        const __half* __restrict__ q_ptr   = Q           + (size_t)batch_head_id * M * D + start_q * D;
        const __half* __restrict__ k_ptr   = K           + (size_t)batch_head_id * N * D;
        const __half* __restrict__ v_ptr   = V           + (size_t)batch_head_id * N * D;
        const __half* __restrict__ o_ptr   = O           + (size_t)batch_head_id * M * D + start_q * D;
        const __half* __restrict__ dO_ptr  = dO          + (size_t)batch_head_id * M * D + start_q * D;
              __half* __restrict__ dQ_ptr  = dQ          + (size_t)batch_head_id * M * D + start_q * D;
        const float*  __restrict__ lse_ptr = softmax_lse + (size_t)batch_head_id * M + start_q;

        // Shared memory layout
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

        // Vector strides
        constexpr int  d_stride_uint4  = (D + PER_UINT4 - 1) / PER_UINT4;
        constexpr int  q_stride_uint4  = (Q_STRIDE  + PER_UINT4 - 1) / PER_UINT4;
        constexpr int  kv_stride_uint4 = (KV_STRIDE + PER_UINT4 - 1) / PER_UINT4;

        // ========================================================================
        // Load Q (into sQ) and dO (into sdO)
        // ========================================================================
        const uint4* q_vec   = reinterpret_cast<const uint4*>(q_ptr);
        const uint4* do_vec  = reinterpret_cast<const uint4*>(dO_ptr);
              uint4* sQ_vec  = reinterpret_cast<uint4*>(sQ);
              uint4* sdO_vec = reinterpret_cast<uint4*>(sdO);

        #pragma unroll 2
        for (int idx = tid; idx < NUM_UINT4_Q_BLOCK; idx += THREADS_PER_BLOCK) {
            const int row = idx / d_stride_uint4;
            const int vec_col = idx % d_stride_uint4;

            uint4 q_val  = make_uint4(0, 0, 0, 0);
            uint4 do_val = make_uint4(0, 0, 0, 0);

            if (row < valid_q_rows) {
                q_val  = __ldg(&q_vec[row * d_stride_uint4 + vec_col]);
                do_val = __ldg(&do_vec[row * d_stride_uint4 + vec_col]);
            }
            sQ_vec[row * q_stride_uint4 + vec_col] = q_val;
            sdO_vec[row * q_stride_uint4 + vec_col] = do_val;
        }
        __syncthreads();

        // ========================================================================
        // Compute row_dot = sum(O ⊙ dO)
        // ========================================================================
        if (tid < valid_q_rows * THREADS_PER_ROW) {
            const int row = tid / THREADS_PER_ROW;
            const int thread_in_row = tid % THREADS_PER_ROW;
            const int fp16_x4_per_row = D / 4;
            const int work_per_thread = (fp16_x4_per_row + THREADS_PER_ROW - 1) / THREADS_PER_ROW;
            const unsigned mask = (valid_q_rows == BLOCK_M) ? 0xFFFFFFFFU : __activemask();

            float thread_dot = 0.0f;

            #pragma unroll
            for (int j = 0; j < work_per_thread; ++j) {
                const int chunk_idx = thread_in_row + j * THREADS_PER_ROW;
                if (chunk_idx >= fp16_x4_per_row) break;
                const int col = chunk_idx * 4;

                const __half* o_addr = o_ptr + row * D + col;
                ushort o_h0, o_h1, o_h2, o_h3;
                asm volatile(
                    "ld.global.v4.u16 {%0, %1, %2, %3}, [%4];"
                    : "=h"(o_h0), "=h"(o_h1), "=h"(o_h2), "=h"(o_h3)
                    : "l"(o_addr)
                    : "memory"
                );

                const __half* dO_addr = sdO + row * Q_STRIDE + col;
                ushort d_h0, d_h1, d_h2, d_h3;
                const uint32_t ptr_dO = static_cast<uint32_t>(reinterpret_cast<uintptr_t>(__cvta_generic_to_shared(dO_addr)));
                asm volatile(
                    "ld.shared.v4.u16 {%0, %1, %2, %3}, [%4];"
                    : "=h"(d_h0), "=h"(d_h1), "=h"(d_h2), "=h"(d_h3)
                    : "r"(ptr_dO)
                    : "memory"
                );

                const float fo_0 = __half2float(__ushort_as_half(o_h0));
                const float fo_1 = __half2float(__ushort_as_half(o_h1));
                const float fo_2 = __half2float(__ushort_as_half(o_h2));
                const float fo_3 = __half2float(__ushort_as_half(o_h3));

                const float fd_0 = __half2float(__ushort_as_half(d_h0));
                const float fd_1 = __half2float(__ushort_as_half(d_h1));
                const float fd_2 = __half2float(__ushort_as_half(d_h2));
                const float fd_3 = __half2float(__ushort_as_half(d_h3));

                thread_dot = __fmaf_rn(fo_0, fd_0, thread_dot);
                thread_dot = __fmaf_rn(fo_1, fd_1, thread_dot);
                thread_dot = __fmaf_rn(fo_2, fd_2, thread_dot);
                thread_dot = __fmaf_rn(fo_3, fd_3, thread_dot);
            }

            #pragma unroll
            for (int o = THREADS_PER_ROW / 2; o > 0; o >>= 1)
                thread_dot += __shfl_down_sync(mask, thread_dot, o, THREADS_PER_ROW);

            if (thread_in_row == 0) {
                sRowDot[row] = thread_dot;
            }
        }

        // Load LSE (one per row)
        if (tid < valid_q_rows) { sLse[tid] = lse_ptr[tid]; }
        __syncthreads();

        // ========================================================================
        // MAIN LOOP (iterates over K/V blocks for current Q block)
        // ========================================================================
        for (int block = 0; block < num_kv_tiles; ++block) {
            const int start_kv = block * BLOCK_N;
            if (start_kv >= N) break;
            const int valid_kv_rows = min(BLOCK_N, N - start_kv);

            // Early skip per tile
            if constexpr (IS_CAUSAL) { if (start_kv >= start_q + valid_q_rows) continue; }

            // ========================================================================
            // Load V (into sK alias)
            // ========================================================================
            const uint4* v_vec  = reinterpret_cast<const uint4*>(v_ptr + start_kv * D);
                  uint4* sV_vec = reinterpret_cast<uint4*>(sV);

            #pragma unroll 2
            for (int idx = tid; idx < NUM_UINT4_KV_BLOCK; idx += THREADS_PER_BLOCK) {
                const int row = idx / d_stride_uint4;
                const int vec_col = idx % d_stride_uint4;
                uint4 v_val = make_uint4(0, 0, 0, 0);
                if (row < valid_kv_rows) {
                    v_val = __ldg(&v_vec[row * d_stride_uint4 + vec_col]);
                }
                sV_vec[row * kv_stride_uint4 + vec_col] = v_val;
            }
            __syncthreads();

            // ========================================================================
            // Compute dOV = dO @ V^T
            // ========================================================================
            const int num_tiles_m_dov    = (BLOCK_M + WMMA_M - 1) / WMMA_M;   // dO @ V^T: along M
            const int num_tiles_n_dov    = (BLOCK_N + WMMA_N - 1) / WMMA_N;   // dO @ V^T: along N
            const int num_tiles_k_dov    = (D + WMMA_K - 1) / WMMA_K;         // dO @ V^T: inner along D
            const int total_tiles_dov    = num_tiles_m_dov * num_tiles_n_dov;
            const int tiles_per_warp_dov = (total_tiles_dov + WARPS_PER_BLOCK - 1) / WARPS_PER_BLOCK;

            for (int tile_idx = 0; tile_idx < tiles_per_warp_dov; ++tile_idx) {
                const int global_tile_idx = warp_id * tiles_per_warp_dov + tile_idx;

                if (global_tile_idx >= total_tiles_dov) break;

                const int tile_m_idx = global_tile_idx / num_tiles_n_dov;
                const int tile_n_idx = global_tile_idx % num_tiles_n_dov;

                const int tile_m = tile_m_idx * WMMA_M;
                const int tile_n = tile_n_idx * WMMA_N;

                if (tile_m >= valid_q_rows || tile_n >= valid_kv_rows) continue;

                fragment<matrix_a, WMMA_M, WMMA_N, WMMA_K, __half, row_major> a_frag;
                fragment<matrix_b, WMMA_M, WMMA_N, WMMA_K, __half, col_major> b_frag;
                fragment<accumulator, WMMA_M, WMMA_N, WMMA_K, float> acc_frag;

                fill_fragment(acc_frag, 0.0f);

                #pragma unroll
                for (int k_tile = 0; k_tile < num_tiles_k_dov; ++k_tile) {
                    const int k_offset = k_tile * WMMA_K;
                    if (k_offset >= D) break;
                    load_matrix_sync(a_frag, sdO + tile_m * Q_STRIDE + k_offset, Q_STRIDE);
                    load_matrix_sync(b_frag, sV + tile_n * KV_STRIDE + k_offset, KV_STRIDE);
                    mma_sync(acc_frag, a_frag, b_frag, acc_frag);
                }
                store_matrix_sync(sdOV + tile_m * S_STRIDE + tile_n, acc_frag, S_STRIDE, mem_row_major);
            }
            __syncthreads();

            // ========================================================================
            // Load K (into sK)
            // ========================================================================
            const uint4* k_vec  = reinterpret_cast<const uint4*>(k_ptr + start_kv * D);
                  uint4* sK_vec = reinterpret_cast<uint4*>(sK);

            #pragma unroll 2
            for (int idx = tid; idx < NUM_UINT4_KV_BLOCK; idx += THREADS_PER_BLOCK) {
                const int row = idx / d_stride_uint4;
                const int vec_col = idx % d_stride_uint4;
                uint4 k_val = make_uint4(0, 0, 0, 0);
                if (row < valid_kv_rows) {
                    k_val = __ldg(&k_vec[row * d_stride_uint4 + vec_col]);
                }
                sK_vec[row * kv_stride_uint4 + vec_col] = k_val;
            }
            __syncthreads();

            // ========================================================================
            // Compute S = Q @ K^T
            // ========================================================================
            const int num_tiles_m_qk    = (BLOCK_M + WMMA_M - 1) / WMMA_M;   // Q @ K^T: along M
            const int num_tiles_n_qk    = (BLOCK_N + WMMA_N - 1) / WMMA_N;   // Q @ K^T: along N
            const int num_tiles_k_qk    = (D + WMMA_K - 1) / WMMA_K;         // Q @ K^T: inner along D
            const int total_tiles_qk    = num_tiles_m_qk * num_tiles_n_qk;
            const int tiles_per_warp_qk = (total_tiles_qk + WARPS_PER_BLOCK - 1) / WARPS_PER_BLOCK;
            const unsigned row_causal   = (lane_id & 0b1) + ((lane_id >> 2) & 0b1) * 8 + ((lane_id >> 4) & 0b1) * 4;
            const unsigned col_causal   = ((lane_id >> 1) & 0b1) * 2 + ((lane_id >> 3) & 0b1) * 8;

            for (int tile_idx = 0; tile_idx < tiles_per_warp_qk; ++tile_idx) {
                const int global_tile_idx = warp_id * tiles_per_warp_qk + tile_idx;

                if (global_tile_idx >= total_tiles_qk) break;

                const int tile_m_idx = global_tile_idx / num_tiles_n_qk;
                const int tile_n_idx = global_tile_idx % num_tiles_n_qk;

                const int tile_m = tile_m_idx * WMMA_M;
                const int tile_n = tile_n_idx * WMMA_N;

                if (tile_m >= valid_q_rows || tile_n >= valid_kv_rows) continue;

                fragment<matrix_a, WMMA_M, WMMA_N, WMMA_K, __half, row_major> a_frag;
                fragment<matrix_b, WMMA_M, WMMA_N, WMMA_K, __half, col_major> b_frag;
                fragment<accumulator, WMMA_M, WMMA_N, WMMA_K, float> acc_frag;

                fill_fragment(acc_frag, 0.0f);

                #pragma unroll
                for (int k_tile = 0; k_tile < num_tiles_k_qk; ++k_tile) {
                    const int k_offset = k_tile * WMMA_K;
                    if (k_offset >= D) break;
                    load_matrix_sync(a_frag, sQ + tile_m * Q_STRIDE + k_offset, Q_STRIDE);
                    load_matrix_sync(b_frag, sK + tile_n * KV_STRIDE + k_offset, KV_STRIDE);
                    mma_sync(acc_frag, a_frag, b_frag, acc_frag);
                }
                // Fused scaling + causal mask
                if constexpr (IS_CAUSAL) {
                    #pragma unroll
                    for (int i = 0; i < acc_frag.num_elements; ++i) {
                        const unsigned col = col_causal + (i & 0b1) + ((i >> 2) & 0b1) * 4;
                        const unsigned row = row_causal + ((i >> 1) & 0b1) * 2;

                        const int global_m = start_q + tile_m + row;
                        const int global_n = start_kv + tile_n + col;

                        const bool is_valid = (global_m < start_q + valid_q_rows) &&
                                              (global_n < start_kv + valid_kv_rows);

                        acc_frag.x[i] = is_valid
                            ? ((global_n > global_m) ? NEG_INF : acc_frag.x[i] * softmax_scale)
                            : NEG_INF;
                    }
                } else {
                    #pragma unroll
                    for (int i = 0; i < acc_frag.num_elements; ++i) {
                        acc_frag.x[i] *= softmax_scale;
                    }
                }
                store_matrix_sync(sS + tile_m * S_STRIDE + tile_n, acc_frag, S_STRIDE, mem_row_major);
            }
            __syncthreads();

            // ========================================================================
            // Compute dS = exp(S - lse) * (dOV - row_dot) * scale
            // ========================================================================
            if (tid < valid_q_rows * THREADS_PER_ROW) {
                const int row = tid / THREADS_PER_ROW;
                const int thread_in_row = tid % THREADS_PER_ROW;

                 float* sS_row   = sS   + row * S_STRIDE;
                 float* sdOV_row = sdOV + row * S_STRIDE;
                __half* sdS_row  = sdS  + row * S_STRIDE;

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

            // ========================================================================
            // Compute dQ += dS @ K
            // ========================================================================
            const int num_tiles_m_dq    = (BLOCK_M + WMMA_M - 1) / WMMA_M;   // dS @ K: along M
            const int num_tiles_n_dq    = (D + WMMA_N - 1) / WMMA_N;         // dS @ K: along D
            const int num_tiles_k_dq    = (BLOCK_N + WMMA_K - 1) / WMMA_K;   // dS @ K: inner along N
            const int total_tiles_dq    = num_tiles_m_dq * num_tiles_n_dq;
            const int tiles_per_warp_dq = (total_tiles_dq + WARPS_PER_BLOCK - 1) / WARPS_PER_BLOCK;

            for (int tile_local = 0; tile_local < tiles_per_warp_dq; ++tile_local) {
                const int global_tile_idx = warp_id * tiles_per_warp_dq + tile_local;
                if (global_tile_idx >= total_tiles_dq) break;

                const int tile_m_idx = global_tile_idx / num_tiles_n_dq;
                const int tile_n_idx = global_tile_idx % num_tiles_n_dq;

                const int tile_m = tile_m_idx * WMMA_M;
                const int tile_n = tile_n_idx * WMMA_N;

                if (tile_m >= valid_q_rows || tile_n >= D) continue;

                fragment<matrix_a, WMMA_M, WMMA_N, WMMA_K, __half, row_major> a_frag;
                fragment<matrix_b, WMMA_M, WMMA_N, WMMA_K, __half, row_major> b_frag;
                fragment<accumulator, WMMA_M, WMMA_N, WMMA_K, float> acc_frag;
                fill_fragment(acc_frag, 0.0f);

                #pragma unroll
                for (int k_tile = 0; k_tile < num_tiles_k_dq; ++k_tile) {
                    const int k_offset = k_tile * WMMA_K;
                    if (k_offset >= valid_kv_rows) break;

                    load_matrix_sync(a_frag, sdS + tile_m * S_STRIDE + k_offset, S_STRIDE);
                    load_matrix_sync(b_frag, sK + k_offset * KV_STRIDE + tile_n, KV_STRIDE);
                    mma_sync(acc_frag, a_frag, b_frag, acc_frag);
                }

                fragment<accumulator, WMMA_M, WMMA_N, WMMA_K, float> curr_frag;
                load_matrix_sync(curr_frag, sdQ + tile_m * Q_STRIDE + tile_n, Q_STRIDE, mem_row_major);

                #pragma unroll
                for (int i = 0; i < curr_frag.num_elements; ++i) {
                    curr_frag.x[i] += acc_frag.x[i];
                }
                store_matrix_sync(sdQ + tile_m * Q_STRIDE + tile_n, curr_frag, Q_STRIDE, mem_row_major);
            }
            __syncthreads();
        } // END MAIN LOOP

        // ========================================================================
        // Store final dQ to global memory
        // ========================================================================
        const int total_fp16_x4 = (valid_q_rows * D) / 4;
        for (int i = tid; i < total_fp16_x4; i += THREADS_PER_BLOCK) {
            const int row = i / (D / 4);
            const int col = (i % (D / 4)) * 4;

            const float* s_dQ_row = sdQ + row * Q_STRIDE;

            const __half h0 = __float2half_rn(s_dQ_row[col + 0]);
            const __half h1 = __float2half_rn(s_dQ_row[col + 1]);
            const __half h2 = __float2half_rn(s_dQ_row[col + 2]);
            const __half h3 = __float2half_rn(s_dQ_row[col + 3]);

            asm volatile(
                "st.global.v4.u16 [%0], {%1, %2, %3, %4};"
                :
                : "l"(dQ_ptr + row * D + col),
                  "h"(__half_as_ushort(h0)),
                  "h"(__half_as_ushort(h1)),
                  "h"(__half_as_ushort(h2)),
                  "h"(__half_as_ushort(h3))
                : "memory"
            );
        }
    }
    // =========================================================================
    // PHASE 2: dKV
    // =========================================================================
    else if (blockIdx.y == 1) {
        if (blockIdx.x >= grid_dkv_limit) return;

        using Config = KernelConfig<D>;

        constexpr int BLOCK_M            = Config::DKV::BLOCK_M;
        constexpr int BLOCK_N            = Config::DKV::BLOCK_N;
        constexpr int THREADS_PER_BLOCK  = Config::DKV::THREADS_PER_BLOCK;
        constexpr int THREADS_PER_ROW    = Config::DKV::THREADS_PER_ROW;
        constexpr int WARPS_PER_BLOCK    = Config::DKV::WARPS_PER_BLOCK;
        constexpr int Q_STRIDE           = Config::DKV::Q_STRIDE;
        constexpr int KV_STRIDE          = Config::DKV::KV_STRIDE;
        constexpr int S_STRIDE           = Config::DKV::S_STRIDE;
        constexpr int PER_UINT4          = Config::DKV::PER_UINT4;
        constexpr int NUM_UINT4_Q_BLOCK  = Config::DKV::NUM_UINT4_Q_BLOCK;
        constexpr int NUM_UINT4_KV_BLOCK = Config::DKV::NUM_UINT4_KV_BLOCK;

        const int block_idx = blockIdx.x;
        const int start_kv  = block_idx * BLOCK_M;
        if (start_kv >= N) return;

        int num_q_tiles  = (M + BLOCK_N - 1) / BLOCK_N;
        const int valid_kv_rows = min(BLOCK_M, N - start_kv);

        // Global pointers
        const __half* __restrict__   q_ptr = Q           + (size_t)batch_head_id * M * D;
        const __half* __restrict__   k_ptr = K           + (size_t)batch_head_id * N * D + start_kv * D;
        const __half* __restrict__   v_ptr = V           + (size_t)batch_head_id * N * D + start_kv * D;
        const __half* __restrict__   o_ptr = O           + (size_t)batch_head_id * M * D;
        const __half* __restrict__  dO_ptr = dO          + (size_t)batch_head_id * M * D;
        const  float* __restrict__ lse_ptr = softmax_lse + (size_t)batch_head_id * M;
              __half* __restrict__  dK_ptr = dK          + (size_t)batch_head_id * N * D + start_kv * D;
              __half* __restrict__  dV_ptr = dV          + (size_t)batch_head_id * N * D + start_kv * D;

        // Shared memory layout
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

        // Vector strides
        constexpr int  d_stride_uint4  = (D + PER_UINT4 - 1) / PER_UINT4;
        constexpr int  q_stride_uint4  = (Q_STRIDE  + PER_UINT4 - 1) / PER_UINT4;
        constexpr int  kv_stride_uint4 = (KV_STRIDE + PER_UINT4 - 1) / PER_UINT4;

        // ========================================================================
        // Load K (into sK) and V (into sV)
        // ========================================================================
        const uint4* k_vec  = reinterpret_cast<const uint4*>(k_ptr);
        const uint4* v_vec  = reinterpret_cast<const uint4*>(v_ptr);
              uint4* sK_vec = reinterpret_cast<uint4*>(sK);
              uint4* sV_vec = reinterpret_cast<uint4*>(sV);

        #pragma unroll 2
        for (int idx = tid; idx < NUM_UINT4_KV_BLOCK; idx += THREADS_PER_BLOCK) {
            const int row = idx / d_stride_uint4;
            const int vec_col = idx % d_stride_uint4;

            uint4 k_val = make_uint4(0, 0, 0, 0);
            uint4 v_val = make_uint4(0, 0, 0, 0);

            if (row < valid_kv_rows) {
                k_val = __ldg(&k_vec[row * d_stride_uint4 + vec_col]);
                v_val = __ldg(&v_vec[row * d_stride_uint4 + vec_col]);
            }
            sK_vec[row * kv_stride_uint4 + vec_col] = k_val;
            sV_vec[row * kv_stride_uint4 + vec_col] = v_val;
        }
        __syncthreads();

        // ========================================================================
        // MAIN LOOP (iterates over Q blocks for current K/V block)
        // ========================================================================
        for (int block = 0; block < num_q_tiles; ++block) {
            const int start_q = block * BLOCK_N;
            if (start_q >= M) break;
            const int valid_q_rows = min(BLOCK_N, M - start_q);

            // Early skip per tile
            if constexpr (IS_CAUSAL) { if (start_kv >= start_q + valid_q_rows) continue; }

            // ========================================================================
            // Load Q (into sdO alias)
            // ========================================================================
            const uint4* q_vec  = reinterpret_cast<const uint4*>(q_ptr + start_q * D);
                  uint4* sQ_vec = reinterpret_cast<uint4*>(sdO);

            #pragma unroll 2
            for (int idx = tid; idx < NUM_UINT4_Q_BLOCK; idx += THREADS_PER_BLOCK) {
                const int row = idx / d_stride_uint4;
                const int vec_col = idx % d_stride_uint4;
                uint4 q_val = make_uint4(0, 0, 0, 0);
                if (row < valid_q_rows) {
                    q_val = __ldg(&q_vec[row * d_stride_uint4 + vec_col]);
                }
                sQ_vec[row * q_stride_uint4 + vec_col] = q_val;
            }
            __syncthreads();

            // ========================================================================
            // Compute S = Q @ K^T
            // ========================================================================
            const int num_tiles_m_qk    = (BLOCK_N + WMMA_M - 1) / WMMA_M;   // Q @ K^T: along M (Q)
            const int num_tiles_n_qk    = (BLOCK_M + WMMA_N - 1) / WMMA_N;   // Q @ K^T: along N (K)
            const int num_tiles_k_qk    = (D + WMMA_K - 1) / WMMA_K;         // Q @ K^T: inner along D
            const int total_tiles_qk    = num_tiles_m_qk * num_tiles_n_qk;
            const int tiles_per_warp_qk = (total_tiles_qk + WARPS_PER_BLOCK - 1) / WARPS_PER_BLOCK;
            const unsigned row_causal   = (lane_id & 0b1) + ((lane_id >> 2) & 0b1) * 8 + ((lane_id >> 4) & 0b1) * 4;
            const unsigned col_causal   = ((lane_id >> 1) & 0b1) * 2 + ((lane_id >> 3) & 0b1) * 8;

            for (int tile_local = 0; tile_local < tiles_per_warp_qk; ++tile_local) {
                const int tile_idx = warp_id * tiles_per_warp_qk + tile_local;
                if (tile_idx >= total_tiles_qk) break;

                const int tile_m_idx = tile_idx / num_tiles_n_qk;
                const int tile_n_idx = tile_idx % num_tiles_n_qk;

                const int tile_m = tile_m_idx * WMMA_M;
                const int tile_n = tile_n_idx * WMMA_N;

                if (tile_m >= valid_q_rows || tile_n >= valid_kv_rows) continue;

                fragment<matrix_a, WMMA_M, WMMA_N, WMMA_K, __half, row_major> a_frag;
                fragment<matrix_b, WMMA_M, WMMA_N, WMMA_K, __half, col_major> b_frag;
                fragment<accumulator, WMMA_M, WMMA_N, WMMA_K, float> acc_frag;
                fill_fragment(acc_frag, 0.0f);

                #pragma unroll
                for (int k_tile = 0; k_tile < num_tiles_k_qk; ++k_tile) {
                    const int k_offset = k_tile * WMMA_K;
                    if (k_offset >= D) break;
                    load_matrix_sync(a_frag, sQ + tile_m * Q_STRIDE + k_offset, Q_STRIDE);
                    load_matrix_sync(b_frag, sK + tile_n * KV_STRIDE + k_offset, KV_STRIDE);
                    mma_sync(acc_frag, a_frag, b_frag, acc_frag);
                }
                // Fused scaling + causal mask
                if constexpr (IS_CAUSAL) {
                    #pragma unroll
                    for (int i = 0; i < acc_frag.num_elements; ++i) {
                        const unsigned col = col_causal + (i & 0b1) + ((i >> 2) & 0b1) * 4;
                        const unsigned row = row_causal + ((i >> 1) & 0b1) * 2;

                        const int global_m = start_q + tile_m + row;
                        const int global_n = start_kv  + tile_n + col;

                        const bool is_valid = (global_m < start_q + valid_q_rows) &&
                                              (global_n < start_kv  + valid_kv_rows);

                        acc_frag.x[i] = is_valid
                            ? ((global_n > global_m) ? NEG_INF : acc_frag.x[i] * softmax_scale)
                            : NEG_INF;
                    }
                } else {
                    #pragma unroll
                    for (int i = 0; i < acc_frag.num_elements; ++i) {
                        acc_frag.x[i] *= softmax_scale;
                    }
                }
                store_matrix_sync(sS + tile_m * S_STRIDE + tile_n, acc_frag, S_STRIDE, mem_row_major);
            }
            __syncthreads();

            // ========================================================================
            // Load dO (into sdO alias)
            // ========================================================================
            const uint4* do_vec  = reinterpret_cast<const uint4*>(dO_ptr + start_q * D);
                  uint4* sdO_vec = reinterpret_cast<uint4*>(sdO);

            #pragma unroll 2
            for (int idx = tid; idx < NUM_UINT4_Q_BLOCK; idx += THREADS_PER_BLOCK) {
                const int row = idx / d_stride_uint4;
                const int vec_col = idx % d_stride_uint4;
                uint4 do_val = make_uint4(0, 0, 0, 0);
                if (row < valid_q_rows) {
                    do_val = __ldg(&do_vec[row * d_stride_uint4 + vec_col]);
                }
                sdO_vec[row * q_stride_uint4 + vec_col] = do_val;
            }
            __syncthreads();

            // ========================================================================
            // Compute row_dot = O ⊙ dO
            // ========================================================================
            const __half* current_o_ptr = o_ptr + start_q * D;

            if (tid < valid_q_rows * THREADS_PER_ROW) {
                const int row = tid / THREADS_PER_ROW;
                const int thread_in_row = tid % THREADS_PER_ROW;
                const int fp16_x4_per_row = D / 4;
                const int work_per_thread = (fp16_x4_per_row + THREADS_PER_ROW - 1) / THREADS_PER_ROW;
                const unsigned mask = (valid_q_rows == BLOCK_N) ? 0xFFFFFFFFU : __activemask();

                float thread_dot = 0.0f;

                #pragma unroll
                for (int j = 0; j < work_per_thread; ++j) {
                    const int chunk_idx = thread_in_row + j * THREADS_PER_ROW;
                    if (chunk_idx >= fp16_x4_per_row) break;
                    const int col = chunk_idx * 4;

                    const half* o_addr = current_o_ptr + row * D + col;
                    ushort o_h0, o_h1, o_h2, o_h3;
                    asm volatile(
                        "ld.global.v4.u16 {%0, %1, %2, %3}, [%4];"
                        : "=h"(o_h0), "=h"(o_h1), "=h"(o_h2), "=h"(o_h3)
                        : "l"(o_addr)
                        : "memory"
                    );

                    const half* dO_addr = sdO + row * Q_STRIDE + col;
                    ushort d_h0, d_h1, d_h2, d_h3;
                    const uint32_t ptr_dO = static_cast<uint32_t>(reinterpret_cast<uintptr_t>(__cvta_generic_to_shared(dO_addr)));
                    asm volatile(
                        "ld.shared.v4.u16 {%0, %1, %2, %3}, [%4];"
                        : "=h"(d_h0), "=h"(d_h1), "=h"(d_h2), "=h"(d_h3)
                        : "r"(ptr_dO)
                        : "memory"
                    );

                    const float fo_0 = __half2float(__ushort_as_half(o_h0));
                    const float fo_1 = __half2float(__ushort_as_half(o_h1));
                    const float fo_2 = __half2float(__ushort_as_half(o_h2));
                    const float fo_3 = __half2float(__ushort_as_half(o_h3));

                    const float fd_0 = __half2float(__ushort_as_half(d_h0));
                    const float fd_1 = __half2float(__ushort_as_half(d_h1));
                    const float fd_2 = __half2float(__ushort_as_half(d_h2));
                    const float fd_3 = __half2float(__ushort_as_half(d_h3));

                    thread_dot = __fmaf_rn(fo_0, fd_0, thread_dot);
                    thread_dot = __fmaf_rn(fo_1, fd_1, thread_dot);
                    thread_dot = __fmaf_rn(fo_2, fd_2, thread_dot);
                    thread_dot = __fmaf_rn(fo_3, fd_3, thread_dot);
                }

                #pragma unroll
                for (int o = THREADS_PER_ROW / 2; o > 0; o >>= 1)
                    thread_dot += __shfl_down_sync(mask, thread_dot, o, THREADS_PER_ROW);

                if (thread_in_row == 0) { sRowDot[row] = thread_dot; }
            }

            // Load LSE
            if (tid < valid_q_rows) { sLse[tid] = lse_ptr[start_q + tid]; }
            __syncthreads();

            // ========================================================================
            // Compute dOV = dO @ V^T
            // ========================================================================
            const int num_tiles_m_dov    = (BLOCK_N + WMMA_M - 1) / WMMA_M;   // dO @ V^T: along M
            const int num_tiles_n_dov    = (BLOCK_M + WMMA_N - 1) / WMMA_N;   // dO @ V^T: along N
            const int num_tiles_k_dov    = (D + WMMA_K - 1) / WMMA_K;         // dO @ V^T: inner along D
            const int total_tiles_dov    = num_tiles_m_dov * num_tiles_n_dov;
            const int tiles_per_warp_dov = (total_tiles_dov + WARPS_PER_BLOCK - 1) / WARPS_PER_BLOCK;

            for (int tile_local = 0; tile_local < tiles_per_warp_dov; ++tile_local) {
                const int tile_idx = warp_id * tiles_per_warp_dov + tile_local;
                if (tile_idx >= total_tiles_dov) break;

                const int tile_m_idx = tile_idx / num_tiles_n_dov;
                const int tile_n_idx = tile_idx % num_tiles_n_dov;

                const int tile_m = tile_m_idx * WMMA_M;
                const int tile_n = tile_n_idx * WMMA_N;

                if (tile_m >= valid_q_rows || tile_n >= valid_kv_rows) continue;

                fragment<matrix_a, WMMA_M, WMMA_N, WMMA_K, __half, row_major> a_frag;
                fragment<matrix_b, WMMA_M, WMMA_N, WMMA_K, __half, col_major> b_frag;
                fragment<accumulator, WMMA_M, WMMA_N, WMMA_K, float> acc_frag;
                fill_fragment(acc_frag, 0.0f);

                #pragma unroll
                for (int k_tile = 0; k_tile < num_tiles_k_dov; ++k_tile) {
                    const int k_offset = k_tile * WMMA_K;
                    if (k_offset >= D) break;
                    load_matrix_sync(a_frag, sdO + tile_m * Q_STRIDE + k_offset, Q_STRIDE);
                    load_matrix_sync(b_frag, sV + tile_n * KV_STRIDE + k_offset, KV_STRIDE);
                    mma_sync(acc_frag, a_frag, b_frag, acc_frag);
                }
                store_matrix_sync(sdOV + tile_m * S_STRIDE + tile_n, acc_frag, S_STRIDE, mem_row_major);
            }
            __syncthreads();

            // ========================================================================
            // Compute dP = exp(S - lse) & dS = P * (dOV - row_dot) * scale
            // ========================================================================
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

                if (valid0) { s0 = sS[row0 * S_STRIDE + col0]; dov0 = sdOV[row0 * S_STRIDE + col0]; } else { s0 = NEG_INF; }
                if (valid1) { s1 = sS[row1 * S_STRIDE + col1]; dov1 = sdOV[row1 * S_STRIDE + col1]; } else { s1 = NEG_INF; }

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

           // ========================================================================
           // Compute dV = P^T @ dO
           // ========================================================================
            const int num_tiles_m_dv    = (BLOCK_M + WMMA_M - 1) / WMMA_M;    // ← P^T @ dO: along M (P^T rows = K rows)
            const int num_tiles_n_dv    = (D + WMMA_N - 1) / WMMA_N;          // ← P^T @ dO: along N (dO cols = D)
            const int num_tiles_k_dv    = (BLOCK_N + WMMA_K - 1) / WMMA_K;    // ← P^T @ dO: inner along N (cols of P = rows of dO)
            const int total_tiles_dv    = num_tiles_m_dv * num_tiles_n_dv;
            const int tiles_per_warp_dv = (total_tiles_dv + WARPS_PER_BLOCK - 1) / WARPS_PER_BLOCK;

            for (int tile_local = 0; tile_local < tiles_per_warp_dv; ++tile_local) {
                const int tile_idx = warp_id * tiles_per_warp_dv + tile_local;
                if (tile_idx >= total_tiles_dv) break;

                const int tile_m_idx = tile_idx / num_tiles_n_dv;
                const int tile_n_idx = tile_idx % num_tiles_n_dv;

                const int tile_m = tile_m_idx * WMMA_M;
                const int tile_n = tile_n_idx * WMMA_N;

                if (tile_m >= valid_kv_rows || tile_n >= D) continue;

                fragment<matrix_a, WMMA_M, WMMA_N, WMMA_K, __half, col_major> a_frag;
                fragment<matrix_b, WMMA_M, WMMA_N, WMMA_K, __half, row_major> b_frag;
                fragment<accumulator, WMMA_M, WMMA_N, WMMA_K, float> acc_frag;
                load_matrix_sync(acc_frag, sdV + tile_m * KV_STRIDE + tile_n, KV_STRIDE, mem_row_major);

                #pragma unroll
                for (int k_tile = 0; k_tile < num_tiles_k_dv; ++k_tile) {
                    const int k_offset = k_tile * WMMA_K;
                    if (k_offset >= valid_q_rows) break;

                    load_matrix_sync(a_frag, sP + k_offset * BLOCK_M + tile_m, BLOCK_M);
                    load_matrix_sync(b_frag, sdO + k_offset * Q_STRIDE + tile_n, Q_STRIDE);
                    mma_sync(acc_frag, a_frag, b_frag, acc_frag);
                }
                store_matrix_sync(sdV + tile_m * KV_STRIDE + tile_n, acc_frag, KV_STRIDE, mem_row_major);
            }
            __syncthreads();

            // ========================================================================
            // Load Q (into sdO alias)
            // ========================================================================
            __half* sQ = sdO;
            #pragma unroll 2
            for (int idx = tid; idx < NUM_UINT4_Q_BLOCK; idx += THREADS_PER_BLOCK) {
                const int row = idx / d_stride_uint4;
                const int vec_col = idx % d_stride_uint4;
                uint4 q_val = make_uint4(0, 0, 0, 0);
                if (row < valid_q_rows) {
                    q_val = __ldg(&q_vec[row * d_stride_uint4 + vec_col]);
                }
                sQ_vec[row * q_stride_uint4 + vec_col] = q_val;
            }
            __syncthreads();

            // ========================================================================
            // Compute dK = dS^T @ Q
            // ========================================================================
            const int num_tiles_m_dk    = (BLOCK_M + WMMA_M - 1) / WMMA_M;    // dS^T @ Q: along M (dS^T rows = K rows)
            const int num_tiles_n_dk    = (D + WMMA_N - 1) / WMMA_N;          // dS^T @ Q: along N (Q cols = D)
            const int num_tiles_k_dk    = (BLOCK_N + WMMA_K - 1) / WMMA_K;    // dS^T @ Q: inner along N (cols of dS = rows of Q)
            const int total_tiles_dk    = num_tiles_m_dk * num_tiles_n_dk;
            const int tiles_per_warp_dk = (total_tiles_dk + WARPS_PER_BLOCK - 1) / WARPS_PER_BLOCK;

            for (int tile_local = 0; tile_local < tiles_per_warp_dk; ++tile_local) {
                const int tile_idx = warp_id * tiles_per_warp_dk + tile_local;
                if (tile_idx >= total_tiles_dk) break;

                const int tile_m_idx = tile_idx / num_tiles_n_dk;
                const int tile_n_idx = tile_idx % num_tiles_n_dk;

                const int tile_m = tile_m_idx * WMMA_M;
                const int tile_n = tile_n_idx * WMMA_N;

                if (tile_m >= valid_kv_rows || tile_n >= D) continue;

                fragment<matrix_a, WMMA_M, WMMA_N, WMMA_K, __half, col_major> a_frag;
                fragment<matrix_b, WMMA_M, WMMA_N, WMMA_K, __half, row_major> b_frag;
                fragment<accumulator, WMMA_M, WMMA_N, WMMA_K, float> acc_frag;
                load_matrix_sync(acc_frag, sdK + tile_m * KV_STRIDE + tile_n, KV_STRIDE, mem_row_major);

                #pragma unroll
                for (int k_tile = 0; k_tile < num_tiles_k_dk; ++k_tile) {
                    const int k_offset = k_tile * WMMA_K;
                    if (k_offset >= valid_q_rows) break;

                    load_matrix_sync(a_frag, sdS + k_offset * BLOCK_M + tile_m, BLOCK_M);
                    load_matrix_sync(b_frag, sQ + k_offset * Q_STRIDE + tile_n, Q_STRIDE);
                    mma_sync(acc_frag, a_frag, b_frag, acc_frag);
                }
                store_matrix_sync(sdK + tile_m * KV_STRIDE + tile_n, acc_frag, KV_STRIDE, mem_row_major);
            }
            __syncthreads();
        }

        // ========================================================================
        // Store final dK + dV to global memory
        // ========================================================================
        const int total_fp16_x4 = (valid_kv_rows * D) / 4;
        for (int i = tid; i < total_fp16_x4; i += THREADS_PER_BLOCK) {
            const int row = i / (D / 4);
            const int col = (i % (D / 4)) * 4;

            const float* s_dK_row = sdK + row * KV_STRIDE;
            const float* s_dV_row = sdV + row * KV_STRIDE;

            const __half hk0 = __float2half_rn(s_dK_row[col + 0]);
            const __half hk1 = __float2half_rn(s_dK_row[col + 1]);
            const __half hk2 = __float2half_rn(s_dK_row[col + 2]);
            const __half hk3 = __float2half_rn(s_dK_row[col + 3]);

            const __half hv0 = __float2half_rn(s_dV_row[col + 0]);
            const __half hv1 = __float2half_rn(s_dV_row[col + 1]);
            const __half hv2 = __float2half_rn(s_dV_row[col + 2]);
            const __half hv3 = __float2half_rn(s_dV_row[col + 3]);

            asm volatile(
                "st.global.v4.u16 [%0], {%1, %2, %3, %4};"
                :
                : "l"(dK_ptr + row * D + col),
                  "h"(__half_as_ushort(hk0)),
                  "h"(__half_as_ushort(hk1)),
                  "h"(__half_as_ushort(hk2)),
                  "h"(__half_as_ushort(hk3))
                : "memory"
            );

            asm volatile(
                "st.global.v4.u16 [%0], {%1, %2, %3, %4};"
                :
                : "l"(dV_ptr + row * D + col),
                  "h"(__half_as_ushort(hv0)),
                  "h"(__half_as_ushort(hv1)),
                  "h"(__half_as_ushort(hv2)),
                  "h"(__half_as_ushort(hv3))
                : "memory"
            );
        }
    }
}

// ============================================================================
// LAUNCHER
// ============================================================================
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
    const dim3 block(Config::MAX_THREADS);
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

// ============================================================================
// WRAPPER
// ============================================================================
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
