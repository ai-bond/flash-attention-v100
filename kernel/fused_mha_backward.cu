//============================================================================
// * Copyright (c) 2025, D.Skryabin / tg @ai_bond007 SPDX-License BSD-3-Clause
//============================================================================
#include <cuda.h>
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <torch/extension.h>

#include <ATen/ATen.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>

#include <mma.h>
using namespace nvcuda::wmma;

// ============================================================================
// VOLTA SM70 WMMA CONSTANTS
// ============================================================================
#define WMMA_M 16
#define WMMA_N 16
#define WMMA_K 16
#define WARP_SIZE 32
#define MAX_SMEM (96 * 1024)

// ============================================================================
// CONFIGURATIONS DQ
// ============================================================================
#define BLOCK_M_16  16
#define BLOCK_N_16  256
#define WARPS_16    2

#define BLOCK_M_32  32
#define BLOCK_N_32  128
#define WARPS_32    4

#define BLOCK_M_64  64
#define BLOCK_N_64  80
#define WARPS_64    8

#define BLOCK_M_128 32
#define BLOCK_N_128 112
#define WARPS_128   8

#define BLOCK_M_256 32
#define BLOCK_N_256 32
#define WARPS_256   8

// ============================================================================
// CONFIGURATIONS DKV
// ============================================================================
#define BLOCK_KV_16  64
#define BLOCK_Q_16   64
#define WARPS_DKV_16 8

#define BLOCK_KV_32  32
#define BLOCK_Q_32   64
#define WARPS_DKV_32 16

#define BLOCK_KV_64  32
#define BLOCK_Q_64   96
#define WARPS_DKV_64 12

#define BLOCK_KV_128  16
#define BLOCK_Q_128   96
#define WARPS_DKV_128 12

#define BLOCK_KV_256  16
#define BLOCK_Q_256   32
#define WARPS_DKV_256 16

// ============================================================================
// COMPILE-TIME CONFIG DQ
// ============================================================================
template<int D>
struct dQKernelConfig {
    static constexpr int BLOCK_M = (D == 16) ? BLOCK_M_16 : (D == 32) ? BLOCK_M_32 : (D == 64)  ? BLOCK_M_64 : (D == 128) ? BLOCK_M_128 : BLOCK_M_256;
    static constexpr int BLOCK_N = (D == 16) ? BLOCK_N_16 : (D == 32) ? BLOCK_N_32 : (D == 64)  ? BLOCK_N_64 : (D == 128) ? BLOCK_N_128 : BLOCK_N_256;
    static constexpr int WARPS_PER_BLOCK = (D == 16) ? WARPS_16 : (D == 32) ? WARPS_32 : (D == 64) ? WARPS_64 : (D == 128) ? WARPS_128 : WARPS_256;
    
    static constexpr int THREADS_PER_BLOCK = WARPS_PER_BLOCK * WARP_SIZE;
    static constexpr int THREADS_PER_ROW   = THREADS_PER_BLOCK / BLOCK_M;
    static constexpr int NUM_K_TILES       = (D + WMMA_K - 1) / WMMA_K;
    static constexpr int PAD               = (8 - (D % 32) + 32) % 32;
    static constexpr int Q_STRIDE          = D + PAD;
    static constexpr int KV_STRIDE         = D + PAD;
    static constexpr int S_STRIDE          = BLOCK_N + PAD;
    static constexpr int PER_UINT4         = 8;
    static constexpr int VECTOR_D          = (D + PER_UINT4 - 1) / PER_UINT4;
    static constexpr int VECTOR_Q          = BLOCK_M * VECTOR_D;
    static constexpr int VECTOR_KV         = BLOCK_N * VECTOR_D;
    
    static constexpr size_t SMEM_Q_BUFFER   = (((size_t)BLOCK_M * Q_STRIDE * sizeof(__half) + 127) & ~static_cast<size_t>(127));
    static constexpr size_t SMEM_KV_BUFFER  = (((size_t)BLOCK_N * KV_STRIDE * sizeof(__half) + 127) & ~static_cast<size_t>(127));
    static constexpr size_t SMEM_QKV_BUFFER = SMEM_Q_BUFFER + SMEM_KV_BUFFER;
    static constexpr size_t SMEM_ACC_BUFFER = (((size_t)BLOCK_M * S_STRIDE * sizeof(float) + 127) & ~static_cast<size_t>(127));
    static constexpr size_t SMEM_DOV_BUFFER = SMEM_ACC_BUFFER;
    static constexpr size_t SMEM_STATS      = (((size_t)2 * BLOCK_M * sizeof(float) + 127) & ~static_cast<size_t>(127));
    static constexpr size_t SMEM_DQ         = (((size_t)BLOCK_M * Q_STRIDE * sizeof(float) + 127) & ~static_cast<size_t>(127));
    static constexpr size_t TOTAL_SMEM      = SMEM_QKV_BUFFER + 2 * SMEM_ACC_BUFFER + SMEM_STATS + SMEM_DQ;
};

// ============================================================================
// COMPILE-TIME CONFIG DKV
// ============================================================================
template<int D>
struct dKVKernelConfig {
    static constexpr int BLOCK_M = (D == 16) ? BLOCK_KV_16 : (D == 32) ? BLOCK_KV_32 : (D == 64) ? BLOCK_KV_64 : (D == 128) ? BLOCK_KV_128 : BLOCK_KV_256;
    static constexpr int BLOCK_N = (D == 16) ? BLOCK_Q_16 : (D == 32) ? BLOCK_Q_32 : (D == 64) ? BLOCK_Q_64 : (D == 128) ? BLOCK_Q_128 : BLOCK_Q_256;
    static constexpr int WARPS_PER_BLOCK = (D == 16) ? WARPS_DKV_16 : (D == 32) ? WARPS_DKV_32 : (D == 64) ? WARPS_DKV_64 : (D == 128) ? WARPS_DKV_128 : WARPS_DKV_256;
    
    static constexpr int THREADS_PER_BLOCK = WARPS_PER_BLOCK * WARP_SIZE;
    static constexpr int THREADS_PER_ROW   = THREADS_PER_BLOCK / BLOCK_N;
    static constexpr int NUM_K_TILES       = (D + WMMA_K - 1) / WMMA_K;
    static constexpr int PAD               = 8;
    static constexpr int Q_STRIDE          = D + PAD;
    static constexpr int KV_STRIDE         = D + PAD;
    static constexpr int S_STRIDE          = BLOCK_M + PAD;
    static constexpr int PER_UINT4         = 8;
    static constexpr int VECTOR_D          = (D + PER_UINT4 - 1) / PER_UINT4;
    static constexpr int VECTOR_Q          = BLOCK_N * VECTOR_D;
    static constexpr int VECTOR_KV         = BLOCK_M * VECTOR_D;
    
    static constexpr size_t SMEM_Q_BUFFER   = (((size_t)BLOCK_N * Q_STRIDE * sizeof(__half) + 127) & ~static_cast<size_t>(127));
    static constexpr size_t SMEM_KV_BUFFER = (((size_t)BLOCK_M * KV_STRIDE * sizeof(__half) + 127) & ~static_cast<size_t>(127));
    static constexpr size_t SMEM_QKV_BUFFER = SMEM_KV_BUFFER + SMEM_Q_BUFFER;
    static constexpr size_t SMEM_ACC_BUFFER = (((size_t)BLOCK_N * S_STRIDE * sizeof(float) + 127) & ~static_cast<size_t>(127));
    static constexpr size_t SMEM_DOV_BUFFER = SMEM_ACC_BUFFER;
    static constexpr size_t SMEM_STATS      = (((size_t)2 * BLOCK_N * sizeof(float) + 127) & ~static_cast<size_t>(127));
    static constexpr size_t SMEM_DKV_SINGLE = (((size_t)BLOCK_M * KV_STRIDE * sizeof(float) + 127) & ~static_cast<size_t>(127));
    static constexpr size_t SMEM_DKV_DUAL   = 2 * SMEM_DKV_SINGLE;
    static constexpr size_t TOTAL_SMEM      = SMEM_QKV_BUFFER + 2 * SMEM_ACC_BUFFER + SMEM_STATS + SMEM_DKV_DUAL;
};

// ============================================================================
// BACKWARD KERNEL dQ
// ============================================================================
template<int D, bool IS_CAUSAL>
__global__ void __launch_bounds__(dQKernelConfig<D>::THREADS_PER_BLOCK, 2)
flash_attention_backward_dq_kernel(
    const __half*  __restrict__ Q,
    const __half*  __restrict__ K,
    const __half*  __restrict__ V,
    const __half*  __restrict__ O,
    const __half*  __restrict__ dO,
    const  float*  __restrict__ softmax_lse,
          __half*  __restrict__ dQ,
    const int B,
    const int H,
    const int M,
    const int N,
    const float softmax_scale
) {
    using Config = dQKernelConfig<D>;

    constexpr int BLOCK_M           = Config::BLOCK_M;
    constexpr int BLOCK_N           = Config::BLOCK_N;
    constexpr int THREADS           = Config::THREADS_PER_BLOCK;
    constexpr int THREADS_PER_ROW   = Config::THREADS_PER_ROW;
    constexpr int NUM_K_TILES       = Config::NUM_K_TILES;
    constexpr int WARPS_PER_BLOCK   = Config::WARPS_PER_BLOCK;
    constexpr int Q_STRIDE          = Config::Q_STRIDE;
    constexpr int KV_STRIDE         = Config::KV_STRIDE;
    constexpr int S_STRIDE          = Config::S_STRIDE;
    constexpr int PER_UINT4         = Config::PER_UINT4;
    constexpr int VECTOR_D          = Config::VECTOR_D;
    constexpr int VECTOR_Q          = Config::VECTOR_Q;
    constexpr int VECTOR_KV         = Config::VECTOR_KV;

    const float NEG_INF = -1e30f;

    // Thread and block indices
    const int batch_head_id = blockIdx.z;
    if (batch_head_id >= B * H) return;

    const int block_m   = blockIdx.x;
    const int start_row = block_m * BLOCK_M;
    if (start_row >= M) return;

    const int valid_q_rows = min(BLOCK_M, M - start_row);
    const int tid          = threadIdx.x;
    const int warp_id      = tid / WARP_SIZE;

    // Global pointers
    const __half* q_ptr   = Q           + (size_t)batch_head_id * M * D + start_row * D;
    const __half* k_ptr   = K           + (size_t)batch_head_id * N * D;
    const __half* v_ptr   = V           + (size_t)batch_head_id * N * D;
    const __half* o_ptr   = O           + (size_t)batch_head_id * M * D + start_row * D;
    const __half* dO_ptr  = dO          + (size_t)batch_head_id * M * D + start_row * D;
          __half* dQ_ptr  = dQ          + (size_t)batch_head_id * M * D + start_row * D;
    const float*  lse_ptr = softmax_lse + (size_t)batch_head_id * M + start_row;

    // Shared memory layout
    extern __shared__ char smem[];
    
    __half* qkv_buffer  = reinterpret_cast<__half*>(smem);
    __half* sK          = qkv_buffer;
    __half* sV          = qkv_buffer;
    __half* sdO         = sK + BLOCK_N * KV_STRIDE;
    __half* sQ          = sdO;    
    float* acc_buffer   = reinterpret_cast<float*>(smem + Config::SMEM_QKV_BUFFER);
    float* dov_buffer   = acc_buffer + BLOCK_M * S_STRIDE;
    float* stats_buffer = dov_buffer + BLOCK_M * S_STRIDE;
    float* sRowDot      = stats_buffer;
    float* sLse         = stats_buffer + BLOCK_M;
    float* sdQ_accum    = reinterpret_cast<float*>(smem + Config::SMEM_QKV_BUFFER + 2 * Config::SMEM_ACC_BUFFER + Config::SMEM_STATS);

    // Zero dQ accumulator
    for (int i = tid; i < BLOCK_M * Q_STRIDE; i += THREADS) {
        sdQ_accum[i] = 0.0f;
    }

    // Compute row_dot = sum(O ⊙ dO) using global memory
    if (tid < valid_q_rows * THREADS_PER_ROW) {
        const int row = tid / THREADS_PER_ROW;
        const int thread_in_row = tid % THREADS_PER_ROW;
        const int fp16_x4_per_row = D / 4;
        const int work_per_thread = (fp16_x4_per_row + THREADS_PER_ROW - 1) / THREADS_PER_ROW;

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

            const __half* dO_addr = dO_ptr + row * D + col;
            ushort d_h0, d_h1, d_h2, d_h3;
            asm volatile(
                "ld.global.v4.u16 {%0, %1, %2, %3}, [%4];"
                : "=h"(d_h0), "=h"(d_h1), "=h"(d_h2), "=h"(d_h3)
                : "l"(dO_addr)
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
        for (int offset = 1; offset < THREADS_PER_ROW; offset <<= 1) {
            thread_dot += __shfl_xor_sync(0xffffffff, thread_dot, offset);
        }

        if (thread_in_row == 0) {
            sRowDot[row] = thread_dot;
        }
    }

    // Load LSE (one per row)
    if (tid < valid_q_rows) { sLse[tid] = lse_ptr[tid]; }
    __syncthreads();

    // Prefetch first block of K and V (L2 cache line = 128 bytes = 64 fp16)
    const int cache_lines = ((BLOCK_N * D) + 63) / 64;
    const int prefetch_threads = min(THREADS, cache_lines);
    if (tid < prefetch_threads) {
        const __half* first_k = k_ptr + tid * 64;
        const __half* first_v = v_ptr + tid * 64;
        asm volatile("prefetch.global.L2 [%0];" :: "l"(first_k));
        asm volatile("prefetch.global.L2 [%0];" :: "l"(first_v));
    }
    __syncthreads();

    // ========================================================================
    // MAIN LOOP
    // ========================================================================

    const int num_n_blocks = (N + BLOCK_N - 1) / BLOCK_N;

    // Iterate over K/V blocks
    for (int block_n = 0; block_n < num_n_blocks; ++block_n) {
        const int start_col = block_n * BLOCK_N;
        if (start_col >= N) break;
        const int valid_k_rows = min(BLOCK_N, N - start_col);

        // Prefetch next block of K and V
        if (block_n + 1 < num_n_blocks) {
            const int cache_lines = ((BLOCK_N * D) + 63) / 64;
            const int prefetch_threads = min(THREADS, cache_lines);
            if (tid < prefetch_threads) {
                const __half* next_k = k_ptr + (block_n + 1) * BLOCK_N * D + tid * 64;
                const __half* next_v = v_ptr + (block_n + 1) * BLOCK_N * D + tid * 64;
                asm volatile("prefetch.global.L2 [%0];" :: "l"(next_k));
                asm volatile("prefetch.global.L2 [%0];" :: "l"(next_v));
            }
        }

        // Unified load: dO and V
        const uint4* do_vec       = reinterpret_cast<const uint4*>(dO_ptr);
        const uint4* v_vec        = reinterpret_cast<const uint4*>(v_ptr + start_col * D);
        uint4*       sdO_vec      = reinterpret_cast<uint4*>(sdO);
        uint4*       sV_vec       = reinterpret_cast<uint4*>(sV);

        const int q_stride_uint4  = (Q_STRIDE  + PER_UINT4 - 1) / PER_UINT4;
        const int kv_stride_uint4 = (KV_STRIDE + PER_UINT4 - 1) / PER_UINT4;
        const int max_work = max(VECTOR_Q, VECTOR_KV);

        #pragma unroll 2
        for (int idx = tid; idx < max_work; idx += THREADS) {
            // Load dO → sdO
            if (idx < VECTOR_Q) {
                const int row = idx / VECTOR_D;
                const int col = idx % VECTOR_D;
                uint4 do_val = make_uint4(0, 0, 0, 0);
                if (row < valid_q_rows && col < VECTOR_D) {
                    do_val = __ldg(&do_vec[row * VECTOR_D + col]);
                }
                sdO_vec[row * q_stride_uint4 + col] = do_val;
            }

            // Load V → sV
            if (idx < VECTOR_KV) {
                const int row = idx / VECTOR_D;
                const int vec_col = idx % VECTOR_D;
                uint4 v_val = make_uint4(0, 0, 0, 0);
                if (row < valid_k_rows && vec_col < VECTOR_D) {
                    v_val = __ldg(&v_vec[row * VECTOR_D + vec_col]);
                }
                sV_vec[row * kv_stride_uint4 + vec_col] = v_val;
            }
        }
        __syncthreads();

        // Compute dOV = dO @ V^T with 2D warp distribution
        constexpr int num_tiles_m = (BLOCK_M + WMMA_M - 1) / WMMA_M;
        constexpr int num_tiles_n = (BLOCK_N + WMMA_N - 1) / WMMA_N;
        constexpr int total_tiles = num_tiles_m * num_tiles_n;
        constexpr int tiles_per_warp = (total_tiles + WARPS_PER_BLOCK - 1) / WARPS_PER_BLOCK;
        
        float* sdOV = dov_buffer;
        
        for (int tile_idx = 0; tile_idx < tiles_per_warp; ++tile_idx) {
            const int global_tile_idx = warp_id * tiles_per_warp + tile_idx;
            
            if (global_tile_idx >= total_tiles) break;
            
            const int tile_m_idx = global_tile_idx / num_tiles_n;
            const int tile_n_idx = global_tile_idx % num_tiles_n;
            
            const int tile_m = tile_m_idx * WMMA_M;
            const int tile_n = tile_n_idx * WMMA_N;
            
            if (tile_m >= valid_q_rows || tile_n >= valid_k_rows) continue;
            
            fragment<matrix_a, WMMA_M, WMMA_N, WMMA_K, __half, row_major> a_frag;
            fragment<matrix_b, WMMA_M, WMMA_N, WMMA_K, __half, col_major> b_frag;
            fragment<accumulator, WMMA_M, WMMA_N, WMMA_K, float> acc_frag;

            fill_fragment(acc_frag, 0.0f);

            #pragma unroll
            for (int k_tile = 0; k_tile < NUM_K_TILES; ++k_tile) {
                const int k_offset = k_tile * WMMA_K;
                if (k_offset >= D) break;
                load_matrix_sync(a_frag, sdO + tile_m * Q_STRIDE + k_offset, Q_STRIDE);
                load_matrix_sync(b_frag, sV + tile_n * KV_STRIDE + k_offset, KV_STRIDE);
                mma_sync(acc_frag, a_frag, b_frag, acc_frag);
            }

            store_matrix_sync(sdOV + tile_m * S_STRIDE + tile_n, acc_frag, S_STRIDE, mem_row_major);
        }
        __syncthreads();

        // Unified load: K and Q
        const uint4* k_vec     = reinterpret_cast<const uint4*>(k_ptr + start_col * D);
        const uint4* q_vec     = reinterpret_cast<const uint4*>(q_ptr);
        uint4*       sK_vec    = reinterpret_cast<uint4*>(sK);
        uint4*       sQ_vec    = reinterpret_cast<uint4*>(sdO);

        #pragma unroll 2
        for (int idx = tid; idx < max_work; idx += THREADS) {
            // Load K → sK
            if (idx < VECTOR_KV) {
                const int row = idx / VECTOR_D;
                const int vec_col = idx % VECTOR_D;
                uint4 k_val = make_uint4(0, 0, 0, 0);
                if (row < valid_k_rows && vec_col < VECTOR_D) {
                    k_val = __ldg(&k_vec[row * VECTOR_D + vec_col]);
                }
                sK_vec[row * kv_stride_uint4 + vec_col] = k_val;
            }
            // Load Q → sQ (= sdO)
            if (idx < VECTOR_Q) {
                const int row = idx / VECTOR_D;
                const int vec_col = idx % VECTOR_D;
                uint4 q_val = make_uint4(0, 0, 0, 0);
                if (row < valid_q_rows && vec_col < VECTOR_D) {
                    q_val = __ldg(&q_vec[row * VECTOR_D + vec_col]);
                }
                sQ_vec[row * q_stride_uint4 + vec_col] = q_val;
            }
        }
        __syncthreads();

        // Compute S = Q @ K^T with 2D warp distribution
        float* sS = acc_buffer;
       
        for (int tile_idx = 0; tile_idx < tiles_per_warp; ++tile_idx) {
            const int global_tile_idx = warp_id * tiles_per_warp + tile_idx;
            
            if (global_tile_idx >= total_tiles) break;
            
            const int tile_m_idx = global_tile_idx / num_tiles_n;
            const int tile_n_idx = global_tile_idx % num_tiles_n;
            
            const int tile_m = tile_m_idx * WMMA_M;
            const int tile_n = tile_n_idx * WMMA_N;
            
            if (tile_m >= valid_q_rows || tile_n >= valid_k_rows) continue;
            
            fragment<matrix_a, WMMA_M, WMMA_N, WMMA_K, __half, row_major> a_frag;
            fragment<matrix_b, WMMA_M, WMMA_N, WMMA_K, __half, col_major> b_frag;
            fragment<accumulator, WMMA_M, WMMA_N, WMMA_K, float> acc_frag;

            fill_fragment(acc_frag, 0.0f);

            #pragma unroll
            for (int k_tile = 0; k_tile < NUM_K_TILES; ++k_tile) {
                const int k_offset = k_tile * WMMA_K;
                if (k_offset >= D) break;
                load_matrix_sync(a_frag, sQ + tile_m * Q_STRIDE + k_offset, Q_STRIDE);
                load_matrix_sync(b_frag, sK + tile_n * KV_STRIDE + k_offset, KV_STRIDE);
                mma_sync(acc_frag, a_frag, b_frag, acc_frag);
            }

            #pragma unroll
            for (int i = 0; i < acc_frag.num_elements; ++i) {
                acc_frag.x[i] *= softmax_scale;
            }

            store_matrix_sync(sS + tile_m * S_STRIDE + tile_n, acc_frag, S_STRIDE, mem_row_major);
        }

        __syncthreads();

        // Apply causal mask
        if (IS_CAUSAL) {
            #pragma unroll 2
            for (int idx = tid; idx < BLOCK_M * BLOCK_N; idx += THREADS) {
                const int local_m  = idx / BLOCK_N;
                const int local_n  = idx % BLOCK_N;
                const int global_m = start_row + local_m;
                const int global_n = start_col + local_n;

                if (local_m < valid_q_rows && local_n < valid_k_rows && global_n > global_m) {
                    sS[local_m * S_STRIDE + local_n] = NEG_INF;
                }
            }
            __syncthreads();
        }
        
        // Compute dS = exp(S - lse) * (dOV - row_dot) * scale → store directly as half
        __half* sdS_base = reinterpret_cast<__half*>(dov_buffer);
       
        const int total_elements = BLOCK_M * BLOCK_N;
        const int total_pairs = (total_elements + 1) / 2;

        #pragma unroll 2
        for (int i = tid; i < total_pairs; i += THREADS) {
            const int linear_idx0 = i * 2;
            const int linear_idx1 = linear_idx0 + 1;

            const int row = linear_idx0 / BLOCK_N;
            const int col0 = linear_idx0 % BLOCK_N;
            const int col1 = col0 + 1;

            bool valid0 = (row < valid_q_rows && col0 < valid_k_rows);
            bool valid1 = (linear_idx1 < total_elements && row < valid_q_rows && col1 < valid_k_rows);

            float s0 = 0.0f, s1 = 0.0f;
            float dov0 = 0.0f, dov1 = 0.0f;

            if (valid0) { s0 = sS[row * S_STRIDE + col0]; dov0 = sdOV[row * S_STRIDE + col0]; }
            if (valid1) { s1 = sS[row * S_STRIDE + col1]; dov1 = sdOV[row * S_STRIDE + col1]; }

            float lse_val = sLse[row];
            float row_dot_val = sRowDot[row];

            // Compute P0, P1
            float shifted0 = s0 - lse_val;
            float shifted1 = s1 - lse_val;
            float p0 = (shifted0 < -80.0f) ? 0.0f : __expf(shifted0);
            float p1 = (shifted1 < -80.0f) ? 0.0f : __expf(shifted1);

            // Compute dS
            float diff0 = dov0 - row_dot_val;
            float diff1 = dov1 - row_dot_val;
            float ds0 = valid0 ? fmaf(p0, softmax_scale * diff0, 0.0f) : 0.0f;
            float ds1 = valid1 ? fmaf(p1, softmax_scale * diff1, 0.0f) : 0.0f;

            // Convert to half and store in row-major layout (BLOCK_N columns, no padding)
            __half2 h2 = __float22half2_rn(make_float2(ds0, ds1));
            __half* dst = sdS_base + row * BLOCK_N + col0;
            dst[0] = h2.x;
            if (col1 < BLOCK_N) { dst[1] = h2.y; }
        }
        __syncthreads();        

        // Compute dQ += dS @ K with 2D warp distribution
        const int num_tiles_m_dq = (BLOCK_M + WMMA_M - 1) / WMMA_M;
        const int num_tiles_n_dq = (D + WMMA_N - 1) / WMMA_N;
        const int total_tiles_dq = num_tiles_m_dq * num_tiles_n_dq;
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

            const int num_k_tiles_dq = (BLOCK_N + WMMA_K - 1) / WMMA_K;
            #pragma unroll
            for (int k_tile = 0; k_tile < num_k_tiles_dq; ++k_tile) {
                const int k_offset = k_tile * WMMA_K;
                if (k_offset >= valid_k_rows) break;

                load_matrix_sync(a_frag, sdS_base + tile_m * BLOCK_N + k_offset, BLOCK_N);
                load_matrix_sync(b_frag, sK + k_offset * KV_STRIDE + tile_n, KV_STRIDE);
                mma_sync(acc_frag, a_frag, b_frag, acc_frag);
            }

            fragment<accumulator, WMMA_M, WMMA_N, WMMA_K, float> curr_frag;
            load_matrix_sync(curr_frag, sdQ_accum + tile_m * Q_STRIDE + tile_n, Q_STRIDE, mem_row_major);

            #pragma unroll
            for (int i = 0; i < curr_frag.num_elements; ++i) {
                curr_frag.x[i] += acc_frag.x[i];
            }

            store_matrix_sync(sdQ_accum + tile_m * Q_STRIDE + tile_n, curr_frag, Q_STRIDE, mem_row_major);
        }
        __syncthreads();
       
    }

    // Store final dQ to global memory
    const int total_fp16_x4 = (valid_q_rows * D) / 4;
    for (int i = tid; i < total_fp16_x4; i += THREADS) {
        const int row = i / (D / 4);
        const int col = (i % (D / 4)) * 4;

        const float* s_dQ_row = sdQ_accum + row * Q_STRIDE;

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

// ============================================================================
// BACKWARD KERNEL dK and dV
// ============================================================================
template<int D, bool IS_CAUSAL>
__global__ void __launch_bounds__(dKVKernelConfig<D>::THREADS_PER_BLOCK, 2)
flash_attention_backward_dkv_kernel(
    const __half* __restrict__ Q,
    const __half* __restrict__ K,
    const __half* __restrict__ V,
    const __half* __restrict__ O,
    const __half* __restrict__ dO,
    const  float* __restrict__ softmax_lse,
          __half* __restrict__ dK,
          __half* __restrict__ dV,
    const int B,
    const int H,
    const int M,
    const int N,
    const float softmax_scale
) {
    using Config = dKVKernelConfig<D>;
    constexpr int BLOCK_M         = Config::BLOCK_M;
    constexpr int BLOCK_N         = Config::BLOCK_N;
    constexpr int THREADS         = Config::THREADS_PER_BLOCK;
    constexpr int THREADS_PER_ROW = Config::THREADS_PER_ROW;
    constexpr int NUM_K_TILES     = Config::NUM_K_TILES;
    constexpr int WARPS_PER_BLOCK = Config::WARPS_PER_BLOCK;
    constexpr int Q_STRIDE        = Config::Q_STRIDE;
    constexpr int KV_STRIDE       = Config::KV_STRIDE;
    constexpr int S_STRIDE        = Config::S_STRIDE;
    constexpr int PER_UINT4       = Config::PER_UINT4;
    constexpr int VECTOR_D        = Config::VECTOR_D;
    constexpr int VECTOR_Q        = Config::VECTOR_Q;
    constexpr int VECTOR_KV       = Config::VECTOR_KV;

    const float NEG_INF = -1e30f;
    const int batch_head_id = blockIdx.z;
    if (batch_head_id >= B * H) return;

    const int block_kv = blockIdx.x;
    const int start_kv = block_kv * BLOCK_M;
    if (start_kv >= N) return;

    const int valid_kv_rows = min(BLOCK_M, N - start_kv);
    const int tid = threadIdx.x;
    const int warp_id = tid / WARP_SIZE;

    // Global pointers
    const __half*   q_ptr = Q           + (size_t)batch_head_id * M * D;
    const __half*   k_ptr = K           + (size_t)batch_head_id * N * D + start_kv * D;
    const __half*   v_ptr = V           + (size_t)batch_head_id * N * D + start_kv * D;
    const __half*   o_ptr = O           + (size_t)batch_head_id * M * D;
    const __half*  dO_ptr = dO          + (size_t)batch_head_id * M * D;
    const  float* lse_ptr = softmax_lse + (size_t)batch_head_id * M;
          __half*  dK_ptr = dK          + (size_t)batch_head_id * N * D + start_kv * D;
          __half*  dV_ptr = dV          + (size_t)batch_head_id * N * D + start_kv * D;

    // Shared memory layout
    extern __shared__ char smem[];
    
    __half* qkv_buffer  = reinterpret_cast<__half*>(smem);
    __half* sK          = qkv_buffer;
    __half* sV          = qkv_buffer;
    __half* sdO         = reinterpret_cast<__half*>(reinterpret_cast<char*>(qkv_buffer) + Config::SMEM_KV_BUFFER);
    __half* sQ          = sdO;
    float* acc_buffer   = reinterpret_cast<float*>(smem + Config::SMEM_QKV_BUFFER);
    float* dov_buffer   = acc_buffer + BLOCK_N * S_STRIDE;
    float* stats_buffer = dov_buffer + BLOCK_N * S_STRIDE;
    float* sRowDot      = stats_buffer;
    float* sLse         = stats_buffer + BLOCK_N;
    float* sdK_accum    = reinterpret_cast<float*>(smem + Config::SMEM_QKV_BUFFER + 2 * Config::SMEM_ACC_BUFFER + Config::SMEM_STATS);
    float* sdV_accum    = sdK_accum + BLOCK_M * KV_STRIDE;
    
    // Init accum with zero
    for (int idx = tid; idx < BLOCK_M * KV_STRIDE; idx += THREADS) {
        sdK_accum[idx] = 0.0f;
        sdV_accum[idx] = 0.0f;
    }
    __syncthreads();

    // ========================================================================
    // MAIN LOOP
    // ========================================================================

    const int num_q_blocks = (M + BLOCK_N - 1) / BLOCK_N;
    
    for (int block_n = 0; block_n < num_q_blocks; ++block_n) {
        const int start_col = block_n * BLOCK_N;
        if (start_col >= M) break;
        const int valid_q_rows = min(BLOCK_N, M - start_col);
        
        // Load K (into qkv_buffer) and Q (into sdO)
        const uint4* k_vec = reinterpret_cast<const uint4*>(k_ptr);
        const uint4* q_vec = reinterpret_cast<const uint4*>(q_ptr + start_col * D);
        uint4* sK_vec = reinterpret_cast<uint4*>(qkv_buffer);
        uint4* sQ_vec = reinterpret_cast<uint4*>(sdO);

        const int kv_stride_uint4 = (KV_STRIDE + PER_UINT4 - 1) / PER_UINT4;
        const int q_stride_uint4  = (Q_STRIDE + PER_UINT4 - 1) / PER_UINT4;

        const int max_work = max(VECTOR_KV, VECTOR_Q);

        #pragma unroll 2
        for (int idx = tid; idx < max_work; idx += THREADS) {
            // Load K → sK = qkv_buffer
            if (idx < VECTOR_KV) {
                const int row = idx / VECTOR_D;
                const int vec_col = idx % VECTOR_D;
                uint4 k_val = make_uint4(0, 0, 0, 0);
                if (row < valid_kv_rows && vec_col < VECTOR_D) {
                    k_val = __ldg(&k_vec[row * VECTOR_D + vec_col]);
                }
                sK_vec[row * kv_stride_uint4 + vec_col] = k_val;
            }

            // Load Q → sQ = sdO
            if (idx < VECTOR_Q) {
                const int row = idx / VECTOR_D;
                const int vec_col = idx % VECTOR_D;
                uint4 q_val = make_uint4(0, 0, 0, 0);
                if (row < valid_q_rows && vec_col < VECTOR_D) {
                    q_val = __ldg(&q_vec[row * VECTOR_D + vec_col]);
                }
                sQ_vec[row * q_stride_uint4 + vec_col] = q_val;
            }
        }
        __syncthreads();
        
        // Compute dQ += dS @ K with 2D warp distribution
        float* sS = acc_buffer;
        
        constexpr int num_tiles_s_m = (BLOCK_N + WMMA_M - 1) / WMMA_M;
        constexpr int num_tiles_s_n = (BLOCK_M + WMMA_N - 1) / WMMA_N;
        constexpr int total_tiles_s = num_tiles_s_m * num_tiles_s_n;
        constexpr int tiles_per_warp_s = (total_tiles_s + WARPS_PER_BLOCK - 1) / WARPS_PER_BLOCK;
        
        for (int tile_local = 0; tile_local < tiles_per_warp_s; ++tile_local) {
            const int tile_idx = warp_id * tiles_per_warp_s + tile_local;
            if (tile_idx >= total_tiles_s) break;
            
            const int tile_m_idx = tile_idx / num_tiles_s_n;
            const int tile_n_idx = tile_idx % num_tiles_s_n;
            const int tile_m = tile_m_idx * WMMA_M;
            const int tile_n = tile_n_idx * WMMA_N;
            
            if (tile_m >= valid_q_rows || tile_n >= valid_kv_rows) continue;
            
            fragment<matrix_a, WMMA_M, WMMA_N, WMMA_K, half, row_major> a_frag;
            fragment<matrix_b, WMMA_M, WMMA_N, WMMA_K, half, col_major> b_frag;
            fragment<accumulator, WMMA_M, WMMA_N, WMMA_K, float> acc_frag;
            fill_fragment(acc_frag, 0.0f);

            #pragma unroll
            for (int k_tile = 0; k_tile < NUM_K_TILES; ++k_tile) {
                const int k_offset = k_tile * WMMA_K;
                if (k_offset >= D) break;
                load_matrix_sync(a_frag, sQ + tile_m * Q_STRIDE + k_offset, Q_STRIDE);
                load_matrix_sync(b_frag, sK + tile_n * KV_STRIDE + k_offset, KV_STRIDE);
                mma_sync(acc_frag, a_frag, b_frag, acc_frag);
            }

            #pragma unroll
            for (int i = 0; i < acc_frag.num_elements; ++i) {
                acc_frag.x[i] *= softmax_scale;
            }

            store_matrix_sync(sS + tile_m * S_STRIDE + tile_n, acc_frag, S_STRIDE, mem_row_major);
        }
        __syncthreads();
        
        // Apply causal mask
        if (IS_CAUSAL) {
            #pragma unroll 2
            for (int idx = tid; idx < BLOCK_N * BLOCK_M; idx += THREADS) {
                const int local_m = idx / BLOCK_M;
                const int local_n = idx % BLOCK_M;
                const int global_m = start_col + local_m;
                const int global_n = start_kv + local_n;
                if (local_m < valid_q_rows && local_n < valid_kv_rows && global_n > global_m) {
                    sS[local_m * S_STRIDE + local_n] = NEG_INF;
                }
            }
            __syncthreads();
        }

        // Load dO into sdO (as fp16)
        const uint4* do_vec = reinterpret_cast<const uint4*>(dO_ptr + start_col * D);
        uint4* sdO_vec      = reinterpret_cast<uint4*>(sdO);
        #pragma unroll 2
        for (int idx = tid; idx < VECTOR_Q; idx += THREADS) {
            const int row = idx / VECTOR_D;
            const int vec_col = idx % VECTOR_D;
            uint4 do_val = make_uint4(0, 0, 0, 0);
            if (row < valid_q_rows && vec_col < VECTOR_D) {
                do_val = __ldg(&do_vec[row * VECTOR_D + vec_col]);
            }
            sdO_vec[row * q_stride_uint4 + vec_col] = do_val;
        }
        __syncthreads();        

        // Compute row_dot = O ⊙ dO (element-wise dot product)
        const __half* current_o_ptr = o_ptr + start_col * D;

        if (tid < valid_q_rows * THREADS_PER_ROW) {
            const int row = tid / THREADS_PER_ROW;
            const int thread_in_row = tid % THREADS_PER_ROW;
            const int fp16_x4_per_row = D / 4;
            const int work_per_thread = (fp16_x4_per_row + THREADS_PER_ROW - 1) / THREADS_PER_ROW;
    
            float thread_dot = 0.0f;
    
            #pragma unroll
            for (int j = 0; j < work_per_thread; ++j) {
                const int chunk_idx = thread_in_row + j * THREADS_PER_ROW;
                if (chunk_idx >= fp16_x4_per_row) break;
        
                const int col = chunk_idx * 4;
        
                const half* o_addr = current_o_ptr + row * D + col;
                const half* dO_addr = sdO + row * Q_STRIDE + col;
        
                ushort o_h0, o_h1, o_h2, o_h3;
                asm volatile(
                    "ld.global.v4.u16 {%0, %1, %2, %3}, [%4];"
                    : "=h"(o_h0), "=h"(o_h1), "=h"(o_h2), "=h"(o_h3)
                    : "l"(o_addr)
                    : "memory"
                );
        
                const half dO_0 = dO_addr[0], dO_1 = dO_addr[1], dO_2 = dO_addr[2], dO_3 = dO_addr[3];
                const half o_0 = __ushort_as_half(o_h0), o_1 = __ushort_as_half(o_h1), o_2 = __ushort_as_half(o_h2), o_3 = __ushort_as_half(o_h3);
                const float fo_0 = __half2float(o_0),  fo_1 = __half2float(o_1),  fo_2 = __half2float(o_2),  fo_3 = __half2float(o_3);
                const float fd_0 = __half2float(dO_0), fd_1 = __half2float(dO_1), fd_2 = __half2float(dO_2), fd_3 = __half2float(dO_3);
        
                // FMA accumulation
                thread_dot = __fmaf_rn(fo_0, fd_0, thread_dot);
                thread_dot = __fmaf_rn(fo_1, fd_1, thread_dot);
                thread_dot = __fmaf_rn(fo_2, fd_2, thread_dot);
                thread_dot = __fmaf_rn(fo_3, fd_3, thread_dot);
            }
    
            #pragma unroll
            for (int offset = 1; offset < THREADS_PER_ROW; offset <<= 1) {
                thread_dot += __shfl_xor_sync(0xffffffff, thread_dot, offset);
            }
    
            if (thread_in_row == 0) { sRowDot[row] = thread_dot; }
        }

        // Load LSE
        if (tid < valid_q_rows) { sLse[tid] = lse_ptr[start_col + tid]; }
        __syncthreads();
        
        // Load V (overwrite K in shared memory)
        sV = qkv_buffer;
        const uint4* v_vec = reinterpret_cast<const uint4*>(v_ptr);
        uint4* sV_vec = reinterpret_cast<uint4*>(sV);

        #pragma unroll 2
        for (int idx = tid; idx < VECTOR_KV; idx += THREADS) {
            const int row = idx / VECTOR_D;
            const int vec_col = idx % VECTOR_D;
            uint4 val = make_uint4(0, 0, 0, 0);
            if (row < valid_kv_rows && vec_col < VECTOR_D) {
                val = __ldg(&v_vec[row * VECTOR_D + vec_col]);
            }
            sV_vec[row * kv_stride_uint4 + vec_col] = val;
        }
        __syncthreads();

        // Compute dOV = dO @ V^T with with 2D warp distribution
        float* sdOV = dov_buffer;
        
        for (int tile_local = 0; tile_local < tiles_per_warp_s; ++tile_local) {
            const int tile_idx = warp_id * tiles_per_warp_s + tile_local;
            if (tile_idx >= total_tiles_s) break;
            
            const int tile_m_idx = tile_idx / num_tiles_s_n;
            const int tile_n_idx = tile_idx % num_tiles_s_n;
            const int tile_m = tile_m_idx * WMMA_M;
            const int tile_n = tile_n_idx * WMMA_N;
            
            if (tile_m >= valid_q_rows || tile_n >= valid_kv_rows) continue;
            
            fragment<matrix_a, WMMA_M, WMMA_N, WMMA_K, half, row_major> a_frag;
            fragment<matrix_b, WMMA_M, WMMA_N, WMMA_K, half, col_major> b_frag;
            fragment<accumulator, WMMA_M, WMMA_N, WMMA_K, float> acc_frag;
            fill_fragment(acc_frag, 0.0f);

            #pragma unroll
            for (int k_tile = 0; k_tile < NUM_K_TILES; ++k_tile) {
                const int k_offset = k_tile * WMMA_K;
                if (k_offset >= D) break;
                load_matrix_sync(a_frag, sdO + tile_m * Q_STRIDE + k_offset, Q_STRIDE);
                load_matrix_sync(b_frag, sV + tile_n * KV_STRIDE + k_offset, KV_STRIDE);
                mma_sync(acc_frag, a_frag, b_frag, acc_frag);
            }

            store_matrix_sync(sdOV + tile_m * S_STRIDE + tile_n, acc_frag, S_STRIDE, mem_row_major);
        }
        __syncthreads();

        // Compute P = exp(S - lse) AND dS = P * (dOV - row_dot) * scale → store P in half (in acc_buffer), dS in half (in dov_buffer)
        float* sP = acc_buffer;
        __half* sP_half = reinterpret_cast<__half*>(acc_buffer);
        __half* sdS_base = reinterpret_cast<__half*>(dov_buffer);

        const int total_elements = BLOCK_N * BLOCK_M;
        const int total_pairs = (total_elements + 1) / 2;

        #pragma unroll 2
        for (int i = tid; i < total_pairs; i += THREADS) {
            const int linear_idx0 = i * 2;
            const int linear_idx1 = linear_idx0 + 1;

            const int row0 = linear_idx0 / BLOCK_M;
            const int col0 = linear_idx0 % BLOCK_M;
            const int row1 = (linear_idx1 < total_elements) ? linear_idx1 / BLOCK_M : row0;
            const int col1 = (linear_idx1 < total_elements) ? linear_idx1 % BLOCK_M : col0 + 1;

            // Load S and dOV
            float s0 = 0.0f, s1 = 0.0f;
            float dov0 = 0.0f, dov1 = 0.0f;

            bool valid0 = (row0 < valid_q_rows && col0 < valid_kv_rows);
            bool valid1 = (linear_idx1 < total_elements && row1 < valid_q_rows && col1 < valid_kv_rows);

            if (valid0) { s0 = sS[row0 * S_STRIDE + col0]; dov0 = sdOV[row0 * S_STRIDE + col0]; }
            if (valid1) { s1 = sS[row1 * S_STRIDE + col1]; dov1 = sdOV[row1 * S_STRIDE + col1]; }

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
                __half* p_dst  = sP_half  + row0 * BLOCK_M + col0;
                __half* ds_dst = sdS_base + row0 * BLOCK_M + col0;
                p_dst[0]  = p_h2.x;  p_dst[1]  = p_h2.y;
                ds_dst[0] = ds_h2.x; ds_dst[1] = ds_h2.y;
            } else {
                if (valid0) { sP_half[row0 * BLOCK_M + col0] = p_h2.x; sdS_base[row0 * BLOCK_M + col0] = ds_h2.x; }
                if (valid1) { sP_half[row1 * BLOCK_M + col1] = p_h2.y; sdS_base[row1 * BLOCK_M + col1] = ds_h2.y; }
            }
        }
        __syncthreads();

        // Compute dV = P^T @ dO with deterministic tile distribution
        const int num_tiles_dv_m = (valid_kv_rows + WMMA_M - 1) / WMMA_M;
        const int num_tiles_dv_n = (D + WMMA_N - 1) / WMMA_N;
        const int total_tiles_dv = num_tiles_dv_m * num_tiles_dv_n;
        const int tiles_per_warp_dv = (total_tiles_dv + WARPS_PER_BLOCK - 1) / WARPS_PER_BLOCK;
        
        for (int tile_local = 0; tile_local < tiles_per_warp_dv; ++tile_local) {
            const int tile_idx = warp_id * tiles_per_warp_dv + tile_local;
            if (tile_idx >= total_tiles_dv) break;
            
            const int tile_m_idx = tile_idx / num_tiles_dv_n;
            const int tile_n_idx = tile_idx % num_tiles_dv_n;
            const int tile_m = tile_m_idx * WMMA_M;
            const int tile_n = tile_n_idx * WMMA_N;
            
            if (tile_m >= valid_kv_rows || tile_n >= D) continue;

            fragment<matrix_a, WMMA_M, WMMA_N, WMMA_K, half, col_major> a_frag;
            fragment<matrix_b, WMMA_M, WMMA_N, WMMA_K, half, row_major> b_frag;
            fragment<accumulator, WMMA_M, WMMA_N, WMMA_K, float> acc_frag;
            load_matrix_sync(acc_frag, sdV_accum + tile_m * KV_STRIDE + tile_n, KV_STRIDE, mem_row_major);

            const int num_k_tiles = (valid_q_rows + WMMA_K - 1) / WMMA_K;
            #pragma unroll
            for (int k_tile = 0; k_tile < num_k_tiles; ++k_tile) {
                const int k_offset = k_tile * WMMA_K;
                if (k_offset >= valid_q_rows) break;
                
                load_matrix_sync(a_frag, sP_half + k_offset * BLOCK_M + tile_m, BLOCK_M);
                load_matrix_sync(b_frag, sdO + k_offset * Q_STRIDE + tile_n, Q_STRIDE);
                mma_sync(acc_frag, a_frag, b_frag, acc_frag);
            }

            store_matrix_sync(sdV_accum + tile_m * KV_STRIDE + tile_n, acc_frag, KV_STRIDE, mem_row_major);
        }
        __syncthreads();

        // Load Q into sdO (as fp16)
        __half* sQ = sdO;
        #pragma unroll 2
        for (int idx = tid; idx < VECTOR_Q; idx += THREADS) {
            const int row = idx / VECTOR_D;
            const int vec_col = idx % VECTOR_D;
            uint4 q_val = make_uint4(0, 0, 0, 0);
            if (row < valid_q_rows && vec_col < VECTOR_D) {
                q_val = __ldg(&q_vec[row * VECTOR_D + vec_col]);
            }
            sQ_vec[row * q_stride_uint4 + vec_col] = q_val;
        }
        __syncthreads();

        // Compute dK = dS^T @ Q with deterministic tile distribution
        const int num_tiles_dk_m = (valid_kv_rows + WMMA_M - 1) / WMMA_M;
        const int num_tiles_dk_n = (D + WMMA_N - 1) / WMMA_N;
        const int total_tiles_dk = num_tiles_dk_m * num_tiles_dk_n;
        const int tiles_per_warp_dk = (total_tiles_dk + WARPS_PER_BLOCK - 1) / WARPS_PER_BLOCK;
        
        for (int tile_local = 0; tile_local < tiles_per_warp_dk; ++tile_local) {
            const int tile_idx = warp_id * tiles_per_warp_dk + tile_local;
            if (tile_idx >= total_tiles_dk) break;
            
            const int tile_m_idx = tile_idx / num_tiles_dk_n;
            const int tile_n_idx = tile_idx % num_tiles_dk_n;
            const int tile_m = tile_m_idx * WMMA_M;
            const int tile_n = tile_n_idx * WMMA_N;
            
            if (tile_m >= valid_kv_rows || tile_n >= D) continue;

            fragment<matrix_a, WMMA_M, WMMA_N, WMMA_K, half, col_major> a_frag;
            fragment<matrix_b, WMMA_M, WMMA_N, WMMA_K, half, row_major> b_frag;
            fragment<accumulator, WMMA_M, WMMA_N, WMMA_K, float> acc_frag;
            load_matrix_sync(acc_frag, sdK_accum + tile_m * KV_STRIDE + tile_n, KV_STRIDE, mem_row_major);

            const int num_k_tiles = (valid_q_rows + WMMA_K - 1) / WMMA_K;
            #pragma unroll
            for (int k_tile = 0; k_tile < num_k_tiles; ++k_tile) {
                const int k_offset = k_tile * WMMA_K;
                if (k_offset >= valid_q_rows) break;
                
                load_matrix_sync(a_frag, sdS_base + k_offset * BLOCK_M + tile_m, BLOCK_M);
                load_matrix_sync(b_frag, sQ + k_offset * Q_STRIDE + tile_n, Q_STRIDE);
                mma_sync(acc_frag, a_frag, b_frag, acc_frag);
            }
            
            store_matrix_sync(sdK_accum + tile_m * KV_STRIDE + tile_n, acc_frag, KV_STRIDE, mem_row_major);
        }
        __syncthreads();
    }

    // Write dK + dV to global memory at once
    const int total_fp16_x4 = (valid_kv_rows * D) / 4;
    for (int i = tid; i < total_fp16_x4; i += THREADS) {
        const int row = i / (D / 4);
        const int col = (i % (D / 4)) * 4;

        const float* s_dK_row = sdK_accum + row * KV_STRIDE;
        const float* s_dV_row = sdV_accum + row * KV_STRIDE;

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
// ============================================================================
// LAUNCHER FOR dQ
// ============================================================================
template<int D>
void launcher_flash_attention_backward_dq(
    const torch::Tensor& Q,
    const torch::Tensor& K,
    const torch::Tensor& V,
    const torch::Tensor& O,
    torch::Tensor& dO,
    const torch::Tensor& softmax_lse,
    torch::Tensor& dQ,
    float softmax_scale,
    bool is_causal,
    cudaStream_t stream
) {
    using Config = dQKernelConfig<D>;
    
    const int B = Q.size(0);
    const int H = Q.size(1);
    const int M = Q.size(2);
    const int N = K.size(2);
    
    const int grid_x = (M + Config::BLOCK_M - 1) / Config::BLOCK_M;
    const dim3 grid(grid_x, 1, B * H);
    const dim3 block(Config::THREADS_PER_BLOCK);
    const size_t smem = Config::TOTAL_SMEM;
    
    TORCH_CHECK(smem <= MAX_SMEM, "Shared memory exceeds 96KB for dQ: ", smem, " bytes");
    
    auto kernel = is_causal ? 
        (void*)flash_attention_backward_dq_kernel<D, true> :
        (void*)flash_attention_backward_dq_kernel<D, false>;
    
    cudaFuncSetAttribute(kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, smem);
    
    if (is_causal) {
        flash_attention_backward_dq_kernel<D, true><<<grid, block, smem, stream>>>(
            reinterpret_cast<const __half*>(Q.data_ptr()),
            reinterpret_cast<const __half*>(K.data_ptr()),
            reinterpret_cast<const __half*>(V.data_ptr()),
            reinterpret_cast<const __half*>(O.data_ptr()),
            reinterpret_cast<__half*>(dO.data_ptr()),
            softmax_lse.data_ptr<float>(),
            reinterpret_cast<__half*>(dQ.data_ptr()),
            B, H, M, N, softmax_scale
        );
    } else {
        flash_attention_backward_dq_kernel<D, false><<<grid, block, smem, stream>>>(
            reinterpret_cast<const __half*>(Q.data_ptr()),
            reinterpret_cast<const __half*>(K.data_ptr()),
            reinterpret_cast<const __half*>(V.data_ptr()),
            reinterpret_cast<const __half*>(O.data_ptr()),
            reinterpret_cast<__half*>(dO.data_ptr()),
            softmax_lse.data_ptr<float>(),
            reinterpret_cast<__half*>(dQ.data_ptr()),
            B, H, M, N, softmax_scale
        );
    }
}

// ============================================================================
// LAUNCHER FOR dK + dV
// ============================================================================
template<int D>
void launcher_flash_attention_backward_dkv(
    const torch::Tensor& Q,
    const torch::Tensor& K,
    const torch::Tensor& V,
    const torch::Tensor& O,
    torch::Tensor& dO,
    const torch::Tensor& softmax_lse,
    torch::Tensor& dK,
    torch::Tensor& dV,
    float softmax_scale,
    bool is_causal,
    cudaStream_t stream
) {
    using Config = dKVKernelConfig<D>;
    
    const int B = Q.size(0);
    const int H = Q.size(1);
    const int M = Q.size(2);
    const int N = K.size(2);
    
    const int grid_x = (N + Config::BLOCK_M - 1) / Config::BLOCK_M;
    const dim3 grid(grid_x, 1, B * H);
    const dim3 block(Config::THREADS_PER_BLOCK);
    const size_t smem = Config::TOTAL_SMEM;
    
    TORCH_CHECK(smem <= MAX_SMEM, "Shared memory exceeds 96KB for unified dKV: ", smem, " bytes (", smem / 1024, " KB)");
    
    auto kernel = is_causal ?
        (void*)flash_attention_backward_dkv_kernel<D, true> :
        (void*)flash_attention_backward_dkv_kernel<D, false>;
    
    cudaFuncSetAttribute(kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, smem);
    
    if (is_causal) {
        flash_attention_backward_dkv_kernel<D, true><<<grid, block, smem, stream>>>(
            reinterpret_cast<const __half*>(Q.data_ptr()),
            reinterpret_cast<const __half*>(K.data_ptr()),
            reinterpret_cast<const __half*>(V.data_ptr()),
            reinterpret_cast<const __half*>(O.data_ptr()),
            reinterpret_cast<__half*>(dO.data_ptr()),
            softmax_lse.data_ptr<float>(),
            reinterpret_cast<__half*>(dK.data_ptr()),
            reinterpret_cast<__half*>(dV.data_ptr()),
            B, H, M, N, softmax_scale
        );
    } else {
        flash_attention_backward_dkv_kernel<D, false><<<grid, block, smem, stream>>>(
            reinterpret_cast<const __half*>(Q.data_ptr()),
            reinterpret_cast<const __half*>(K.data_ptr()),
            reinterpret_cast<const __half*>(V.data_ptr()),
            reinterpret_cast<const __half*>(O.data_ptr()),
            reinterpret_cast<__half*>(dO.data_ptr()),
            softmax_lse.data_ptr<float>(),
            reinterpret_cast<__half*>(dK.data_ptr()),
            reinterpret_cast<__half*>(dV.data_ptr()),
            B, H, M, N, softmax_scale
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
            launcher_flash_attention_backward_dq<16>(q, k, v, out, const_cast<at::Tensor&>(dout), softmax_lse, dq_fp16, softmax_scale, is_causal, stream);
            launcher_flash_attention_backward_dkv<16>(q, k, v, out, const_cast<at::Tensor&>(dout), softmax_lse, dk_fp16, dv_fp16, softmax_scale, is_causal, stream);
            break;
        case 32:
            launcher_flash_attention_backward_dq<32>(q, k, v, out, const_cast<at::Tensor&>(dout), softmax_lse, dq_fp16, softmax_scale, is_causal, stream);
            launcher_flash_attention_backward_dkv<32>(q, k, v, out, const_cast<at::Tensor&>(dout), softmax_lse, dk_fp16, dv_fp16, softmax_scale, is_causal, stream);
            break;
        case 64:
            launcher_flash_attention_backward_dq<64>(q, k, v, out, const_cast<at::Tensor&>(dout), softmax_lse, dq_fp16, softmax_scale, is_causal, stream);
            launcher_flash_attention_backward_dkv<64>(q, k, v, out, const_cast<at::Tensor&>(dout), softmax_lse, dk_fp16, dv_fp16, softmax_scale, is_causal, stream);
            break;
        case 128:
            launcher_flash_attention_backward_dq<128>(q, k, v, out, const_cast<at::Tensor&>(dout), softmax_lse, dq_fp16, softmax_scale, is_causal, stream);
            launcher_flash_attention_backward_dkv<128>(q, k, v, out, const_cast<at::Tensor&>(dout), softmax_lse, dk_fp16, dv_fp16, softmax_scale, is_causal, stream);
            break;
        case 256:
            launcher_flash_attention_backward_dq<256>(q, k, v, out, const_cast<at::Tensor&>(dout), softmax_lse, dq_fp16, softmax_scale, is_causal, stream);
            launcher_flash_attention_backward_dkv<256>(q, k, v, out, const_cast<at::Tensor&>(dout), softmax_lse, dk_fp16, dv_fp16, softmax_scale, is_causal, stream);
            break;
        default: TORCH_CHECK(false, "Unsupported D: ", D);
    }
    return {dq_fp16, dk_fp16, dv_fp16, dsoftmax_sum};
}