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
    static constexpr size_t SMEM_QKV_BUFFER = 2 * SMEM_Q_BUFFER + SMEM_KV_BUFFER;
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
    static constexpr size_t SMEM_QKV_BUFFER = SMEM_KV_BUFFER + 2 * SMEM_Q_BUFFER;
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
    const __half* __restrict__ Q,
    const __half* __restrict__ K,
    const __half* __restrict__ V,
    const float*  __restrict__ O,
    const __half* __restrict__ dO,
    const float*  __restrict__ softmax_lse,
    float*        __restrict__ dQ,
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
    const __half* q_ptr   = Q + (size_t)batch_head_id * M * D + start_row * D;
    const __half* k_ptr   = K + (size_t)batch_head_id * N * D;
    const __half* v_ptr   = V + (size_t)batch_head_id * N * D;
    const float*  o_ptr   = O + (size_t)batch_head_id * M * D + start_row * D;
    const __half* dO_ptr  = dO + (size_t)batch_head_id * M * D + start_row * D;
    float*        dQ_ptr  = dQ + (size_t)batch_head_id * M * D + start_row * D;
    const float*  lse_ptr = softmax_lse + (size_t)batch_head_id * M + start_row;

    // Shared memory layout
    extern __shared__ char smem[];
    __half* qkv_buffer   = reinterpret_cast<__half*>(smem);
    float*  acc_buffer   = reinterpret_cast<float*>(smem + Config::SMEM_QKV_BUFFER);
    float*  dov_buffer   = acc_buffer + BLOCK_M * S_STRIDE;
    float*  stats_buffer = dov_buffer + BLOCK_M * S_STRIDE;
    float*  sRowDot      = stats_buffer;
    float*  sLse         = stats_buffer + BLOCK_M;
    float*  s_dQ         = reinterpret_cast<float*>(smem + Config::SMEM_QKV_BUFFER + 2 * Config::SMEM_ACC_BUFFER + Config::SMEM_STATS);
    
    __half* sQ       = qkv_buffer;
    __half* sKV_base = qkv_buffer + BLOCK_M * Q_STRIDE;
    __half* sdO_base = sKV_base + BLOCK_N * KV_STRIDE;
    __half* sdS_base = reinterpret_cast<__half*>(dov_buffer);

    // Load Q + dO into shared memory at once
    const uint4* q_vec     = reinterpret_cast<const uint4*>(q_ptr);
    uint4*       sQ_vec    = reinterpret_cast<uint4*>(sQ);
    const int    q_stride_uint4 = (Q_STRIDE + PER_UINT4 - 1) / PER_UINT4;
    const uint4* do_vec     = reinterpret_cast<const uint4*>(dO_ptr);
    uint4*       sdO_vec   = reinterpret_cast<uint4*>(sdO_base);

    #pragma unroll 2
    for (int idx = tid; idx < VECTOR_Q; idx += THREADS) {
        const int row     = idx / VECTOR_D;
        const int vec_col = idx % VECTOR_D;
        uint4 q_val = make_uint4(0, 0, 0, 0);
        uint4 do_val = make_uint4(0, 0, 0, 0);
        if (row < valid_q_rows && vec_col < VECTOR_D) {
            q_val = __ldg(&q_vec[row * VECTOR_D + vec_col]);
            do_val = __ldg(&do_vec[row * VECTOR_D + vec_col]);
        }
        sQ_vec[row * q_stride_uint4 + vec_col] = q_val;
        sdO_vec[row * q_stride_uint4 + vec_col] = do_val;
    }

    // Initialize dQ buffer to zero
    for (int idx = tid; idx < BLOCK_M * Q_STRIDE; idx += THREADS) {
        s_dQ[idx] = 0.0f;
    }
    __syncthreads();

    // Compute row_dot = O @ dO using sdO from shared memory
    if (tid < valid_q_rows * THREADS_PER_ROW) {
        const int row = tid / THREADS_PER_ROW;
        const int thread_in_row = tid % THREADS_PER_ROW;
        float thread_dot = 0.0f;
        const int cols_per_thread = (D + THREADS_PER_ROW - 1) / THREADS_PER_ROW;

        #pragma unroll
        for (int j = 0; j < cols_per_thread; ++j) {
            const int col = thread_in_row + j * THREADS_PER_ROW;
            if (col < D) {
                float dO_val = __half2float(sdO_base[row * Q_STRIDE + col]);
                thread_dot += o_ptr[row * D + col] * dO_val;
            }
        }

        #pragma unroll
        for (int offset = 1; offset < THREADS_PER_ROW; offset <<= 1) {
            thread_dot += __shfl_xor_sync(0xffffffff, thread_dot, offset);
        }

        if (thread_in_row == 0) { sRowDot[row] = thread_dot; }
    }

    if (tid < valid_q_rows) { sLse[tid] = lse_ptr[tid]; }
    __syncthreads();

    const int num_n_blocks = (N + BLOCK_N - 1) / BLOCK_N;

    // Iterate over K/V blocks
    for (int block_n = 0; block_n < num_n_blocks; ++block_n) {
        const int start_col = block_n * BLOCK_N;
        if (start_col >= N) break;
        const int valid_k_rows = min(BLOCK_N, N - start_col);

        // Load K into shared memory
        __half* sK = sKV_base;
        const uint4* k_vec     = reinterpret_cast<const uint4*>(k_ptr + start_col * D);
        uint4*       sK_vec    = reinterpret_cast<uint4*>(sK);
        const int    kv_stride_uint4 = (KV_STRIDE + PER_UINT4 - 1) / PER_UINT4;

        #pragma unroll 2
        for (int idx = tid; idx < VECTOR_KV; idx += THREADS) {
            const int row     = idx / VECTOR_D;
            const int vec_col = idx % VECTOR_D;
            uint4 val = make_uint4(0, 0, 0, 0);
            if (row < valid_k_rows && vec_col < VECTOR_D) {
                val = __ldg(&k_vec[row * VECTOR_D + vec_col]);
            }
            sK_vec[row * kv_stride_uint4 + vec_col] = val;
        }

        __syncthreads();

        // Compute S = Q @ K^T with 2D warp distribution
        float* sS = acc_buffer;
        
        constexpr int num_tiles_m = (BLOCK_M + WMMA_M - 1) / WMMA_M;
        constexpr int num_tiles_n = (BLOCK_N + WMMA_N - 1) / WMMA_N;
        constexpr int total_tiles = num_tiles_m * num_tiles_n;
        constexpr int tiles_per_warp = (total_tiles + WARPS_PER_BLOCK - 1) / WARPS_PER_BLOCK;
        
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

        // Compute P = exp(S - lse)
        #pragma unroll 2
        for (int idx = tid; idx < BLOCK_M * BLOCK_N; idx += THREADS) {
            const int row = idx / BLOCK_N;
            const int col = idx % BLOCK_N;
            if (row < valid_q_rows && col < valid_k_rows) {
                float shifted = sS[row * S_STRIDE + col] - sLse[row];
                shifted = fmaxf(shifted, -80.0f);
                shifted = fminf(shifted,  80.0f);
                sS[row * S_STRIDE + col] = __expf(shifted);
            }
        }
        __syncthreads();

        // Load V into shared memory
        __half* sV = sKV_base;
        const uint4* v_vec     = reinterpret_cast<const uint4*>(v_ptr + start_col * D);
        uint4*       sV_vec    = reinterpret_cast<uint4*>(sV);
        const int    v_stride_uint4 = (KV_STRIDE + PER_UINT4 - 1) / PER_UINT4;

        #pragma unroll 2
        for (int idx = tid; idx < VECTOR_KV; idx += THREADS) {
            const int row     = idx / VECTOR_D;
            const int vec_col = idx % VECTOR_D;
            uint4 val = make_uint4(0, 0, 0, 0);
            if (row < valid_k_rows && vec_col < VECTOR_D) {
                val = __ldg(&v_vec[row * VECTOR_D + vec_col]);
            }
            sV_vec[row * v_stride_uint4 + vec_col] = val;
        }

        __syncthreads();

        // Compute dOV = dO @ V^T with 2D warp distribution
        __half* sdO = sdO_base;
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

        // Compute dS = P * (dOV - row_dot) * scale
        float* sP = sS;
        float* s_dS = dov_buffer;
        #pragma unroll 2
        for (int idx = tid; idx < BLOCK_M * BLOCK_N; idx += THREADS) {
            const int row = idx / BLOCK_N;
            const int col = idx % BLOCK_N;
            if (row < valid_q_rows && col < valid_k_rows) {
                float p       = sP[row * S_STRIDE + col];
                float dov     = sdOV[row * S_STRIDE + col];
                float row_dot = sRowDot[row];
                float diff    = dov - row_dot;
                float ds      = fmaf(p, softmax_scale * diff, 0.0f);
                s_dS[row * S_STRIDE + col] = ds;
            }
        }

        __syncthreads();

        // Convert dS to half precision with reuse dov_buffer
        #pragma unroll 2
        for (int i = tid; i < (BLOCK_M * BLOCK_N + 1) / 2; i += THREADS) {
            const int row      = i / ((BLOCK_N + 1) / 2);
            const int half_col = i % ((BLOCK_N + 1) / 2);
            const int col0     = half_col * 2;
            const int col1     = col0 + 1;

            float val0 = 0.0f, val1 = 0.0f;
            if (row < valid_q_rows && col0 < valid_k_rows) { val0 = s_dS[row * S_STRIDE + col0]; }
            if (row < valid_q_rows && col1 < valid_k_rows) { val1 = s_dS[row * S_STRIDE + col1]; }

            __half2 h2 = __float22half2_rn(make_float2(val0, val1));
            __half* dst = sdS_base + row * BLOCK_N + col0;
            dst[0] = h2.x;
            if (col1 < BLOCK_N) {
                dst[1] = h2.y;
            }
        }
        __syncthreads();

        // Reload K (for dQ computation)
        sK = sKV_base;
        sK_vec = reinterpret_cast<uint4*>(sK);
        #pragma unroll 2
        for (int idx = tid; idx < VECTOR_KV; idx += THREADS) {
            const int row     = idx / VECTOR_D;
            const int vec_col = idx % VECTOR_D;
            uint4 val = make_uint4(0, 0, 0, 0);
            if (row < valid_k_rows && vec_col < VECTOR_D) {
                val = __ldg(&k_vec[row * VECTOR_D + vec_col]);
            }
            sK_vec[row * kv_stride_uint4 + vec_col] = val;
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
            load_matrix_sync(curr_frag, s_dQ + tile_m * Q_STRIDE + tile_n, Q_STRIDE, mem_row_major);

            #pragma unroll
            for (int i = 0; i < curr_frag.num_elements; ++i) {
                curr_frag.x[i] += acc_frag.x[i];
            }

            store_matrix_sync(s_dQ + tile_m * Q_STRIDE + tile_n, curr_frag, Q_STRIDE, mem_row_major);
        }
        __syncthreads();
    }

    // Store final dQ to global memory
    int vec_elem_dq = valid_q_rows * D;
    int vec_size_dq = (vec_elem_dq + 4 - 1) / 4;

    for (int vec_idx = tid; vec_idx < vec_size_dq; vec_idx += THREADS) {
        int linear_idx = vec_idx * 4;
        int row = linear_idx / D;
        int col = linear_idx % D;
    
        if (row < valid_q_rows) {
            float4 vec_val;

            vec_val.x = s_dQ[row * Q_STRIDE + col];
            vec_val.y = (col + 1 < D) ? s_dQ[row * Q_STRIDE + col + 1] : 0.0f;
            vec_val.z = (col + 2 < D) ? s_dQ[row * Q_STRIDE + col + 2] : 0.0f;
            vec_val.w = (col + 3 < D) ? s_dQ[row * Q_STRIDE + col + 3] : 0.0f;
        
            reinterpret_cast<float4*>(&dQ_ptr[row * D + col])[0] = vec_val;
        }
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
    const float* __restrict__ O,
    const __half* __restrict__ dO,
    const float* __restrict__ softmax_lse,
    float* __restrict__ dK,
    float* __restrict__ dV,
    const int B,
    const int H,
    const int M,
    const int N,
    const float softmax_scale
) {
    using Config = dKVKernelConfig<D>;
    constexpr int BLOCK_M = Config::BLOCK_M;
    constexpr int BLOCK_N = Config::BLOCK_N;
    constexpr int THREADS = Config::THREADS_PER_BLOCK;
    constexpr int THREADS_PER_ROW = Config::THREADS_PER_ROW;
    constexpr int NUM_K_TILES = Config::NUM_K_TILES;
    constexpr int WARPS_PER_BLOCK = Config::WARPS_PER_BLOCK;
    constexpr int Q_STRIDE = Config::Q_STRIDE;
    constexpr int KV_STRIDE = Config::KV_STRIDE;
    constexpr int S_STRIDE = Config::S_STRIDE;
    constexpr int PER_UINT4 = Config::PER_UINT4;
    constexpr int VECTOR_D = Config::VECTOR_D;
    constexpr int VECTOR_Q = Config::VECTOR_Q;
    constexpr int VECTOR_KV = Config::VECTOR_KV;

    const float NEG_INF = -1e30f;
    const int batch_head_id = blockIdx.z;
    if (batch_head_id >= B * H) return;

    const int block_kv = blockIdx.x;
    const int start_kv = block_kv * BLOCK_M;
    if (start_kv >= N) return;

    const int valid_kv_rows = min(BLOCK_M, N - start_kv);
    const int tid = threadIdx.x;
    const int warp_id = tid / WARP_SIZE;

    const __half* q_ptr = Q + (size_t)batch_head_id * M * D;
    const __half* k_ptr = K + (size_t)batch_head_id * N * D + start_kv * D;
    const __half* v_ptr = V + (size_t)batch_head_id * N * D + start_kv * D;
    const float* o_ptr = O + (size_t)batch_head_id * M * D;
    const __half* dO_ptr = dO + (size_t)batch_head_id * M * D;
    const float* lse_ptr = softmax_lse + (size_t)batch_head_id * M;
    float* dK_ptr = dK + (size_t)batch_head_id * N * D + start_kv * D;
    float* dV_ptr = dV + (size_t)batch_head_id * N * D + start_kv * D;

    // Shared memory layout
    extern __shared__ char smem[];
    __half* qkv_buffer = reinterpret_cast<__half*>(smem);
    __half* sKV_base = qkv_buffer;
    __half* sK = sKV_base;
    __half* sV = sKV_base;
    __half* sQ = reinterpret_cast<__half*>(reinterpret_cast<char*>(qkv_buffer) + Config::SMEM_KV_BUFFER);
    __half* sdO = reinterpret_cast<__half*>(reinterpret_cast<char*>(sQ) + Config::SMEM_Q_BUFFER);
    float* acc_buffer = reinterpret_cast<float*>(smem + Config::SMEM_QKV_BUFFER);
    float* dov_buffer = acc_buffer + BLOCK_N * S_STRIDE;
    float* stats_buffer = dov_buffer + BLOCK_N * S_STRIDE;
    float* sRowDot = stats_buffer;
    float* sLse = stats_buffer + BLOCK_N;
    float* s_dK_accum = reinterpret_cast<float*>(smem + Config::SMEM_QKV_BUFFER + 2 * Config::SMEM_ACC_BUFFER + Config::SMEM_STATS);
    float* s_dV_accum = s_dK_accum + BLOCK_M * KV_STRIDE;
    
    const int kv_stride_uint4 = (KV_STRIDE + PER_UINT4 - 1) / PER_UINT4;
    const int num_q_blocks = (M + BLOCK_N - 1) / BLOCK_N;

    // Init accum with zero
    for (int idx = tid; idx < BLOCK_M * KV_STRIDE; idx += THREADS) {
        s_dK_accum[idx] = 0.0f;
        s_dV_accum[idx] = 0.0f;
    }
    __syncthreads();

    // Load K
    const uint4* k_vec = reinterpret_cast<const uint4*>(k_ptr);
    uint4* sK_vec = reinterpret_cast<uint4*>(sK);
    #pragma unroll 2
    for (int idx = tid; idx < VECTOR_KV; idx += THREADS) {
        const int row = idx / VECTOR_D;
        const int vec_col = idx % VECTOR_D;
        uint4 val = make_uint4(0, 0, 0, 0);
        if (row < valid_kv_rows && vec_col < VECTOR_D) {
            val = __ldg(&k_vec[row * VECTOR_D + vec_col]);
        }
        sK_vec[row * kv_stride_uint4 + vec_col] = val;
    }
    __syncthreads();

    // ========================================================================
    // UNIFIED LOOP
    // ========================================================================
    for (int block_n = 0; block_n < num_q_blocks; ++block_n) {
        const int start_col = block_n * BLOCK_N;
        if (start_col >= M) break;
        const int valid_q_rows = min(BLOCK_N, M - start_col);

        // Load Q + dO together
        const uint4* q_vec  = reinterpret_cast<const uint4*>(q_ptr + start_col * D);
        const uint4* do_vec = reinterpret_cast<const uint4*>(dO_ptr + start_col * D);
        uint4* sQ_vec    = reinterpret_cast<uint4*>(sQ);
        uint4* sdO_vec  = reinterpret_cast<uint4*>(sdO);
        const int q_stride_uint4 = (Q_STRIDE + PER_UINT4 - 1) / PER_UINT4;

        #pragma unroll 2
        for (int idx = tid; idx < VECTOR_Q; idx += THREADS) {
            const int row = idx / VECTOR_D;
            const int vec_col = idx % VECTOR_D;
            uint4 q_val = make_uint4(0, 0, 0, 0);
            uint4 do_val = make_uint4(0, 0, 0, 0);
            if (row < valid_q_rows && vec_col < VECTOR_D) {
                q_val  = __ldg(&q_vec[row * VECTOR_D + vec_col]);
                do_val = __ldg(&do_vec[row * VECTOR_D + vec_col]);
            }
            sQ_vec[row * q_stride_uint4 + vec_col] = q_val;
            sdO_vec[row * q_stride_uint4 + vec_col] = do_val;
        }
        __syncthreads();

        // Compute (row_dot + lse)
        const float* current_o_ptr = o_ptr + start_col * D;
        if (tid < valid_q_rows * THREADS_PER_ROW) {
            const int row = tid / THREADS_PER_ROW;
            const int thread_in_row = tid % THREADS_PER_ROW;
            float thread_dot = 0.0f;
            const int cols_per_thread = (D + THREADS_PER_ROW - 1) / THREADS_PER_ROW;

            #pragma unroll
            for (int j = 0; j < cols_per_thread; ++j) {
                const int col = thread_in_row + j * THREADS_PER_ROW;
                if (col < D) {
                    float dO_val = __half2float(sdO[row * Q_STRIDE + col]);
                    thread_dot += current_o_ptr[row * D + col] * dO_val;
                }
            }

            #pragma unroll
            for (int offset = 1; offset < THREADS_PER_ROW; offset <<= 1) {
                thread_dot += __shfl_xor_sync(0xffffffff, thread_dot, offset);
            }

            if (thread_in_row == 0) {
                sRowDot[row] = thread_dot;
            }
        }

        if (tid < valid_q_rows) {
            sLse[tid] = lse_ptr[start_col + tid];
        }
        __syncthreads();

        // Compute dQ += dS @ K with 2D warp distribution
        float* sS = acc_buffer;
        
        constexpr int num_tiles_s_m = (BLOCK_N + WMMA_M - 1) / WMMA_M;  // по Q
        constexpr int num_tiles_s_n = (BLOCK_M + WMMA_N - 1) / WMMA_N;  // по K/V
        constexpr int total_tiles_s = num_tiles_s_m * num_tiles_s_n;
        constexpr int tiles_per_warp_s = (total_tiles_s + WARPS_PER_BLOCK - 1) / WARPS_PER_BLOCK;
        
        for (int tile_local = 0; tile_local < tiles_per_warp_s; ++tile_local) {
            const int tile_idx = warp_id * tiles_per_warp_s + tile_local;
            if (tile_idx >= total_tiles_s) break;
            
            // Row-major mapping: iterate M first, then N
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

         // Compute P = exp(S - lse)
        #pragma unroll 2
        for (int idx = tid; idx < BLOCK_N * BLOCK_M; idx += THREADS) {
            const int row = idx / BLOCK_M;
            const int col = idx % BLOCK_M;
            if (row < valid_q_rows && col < valid_kv_rows) {
                float shifted = sS[row * S_STRIDE + col] - sLse[row];
                shifted = fmaxf(shifted, -80.0f);
                shifted = fminf(shifted,  80.0f);
                sS[row * S_STRIDE + col] = __expf(shifted);
            }
        }
        __syncthreads();

        // Load V (overwrite K in shared memory)
        sV = sKV_base;
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

        // Compute dS = P * (dOV - row_dot) * scale
        float* sP = sS;
        float* s_dS = dov_buffer;
        #pragma unroll 2
        for (int idx = tid; idx < BLOCK_N * BLOCK_M; idx += THREADS) {
            const int row = idx / BLOCK_M;
            const int col = idx % BLOCK_M;
            if (row < valid_q_rows && col < valid_kv_rows) {
                float p = sP[row * S_STRIDE + col];
                float dov = sdOV[row * S_STRIDE + col];
                float row_dot = sRowDot[row];
                float diff = dov - row_dot;
                s_dS[row * S_STRIDE + col] = fmaf(p, softmax_scale * diff, 0.0f);
            }
        }
        __syncthreads();

        // Convert dS to half with dov_buffer reuse
        __half* sdS_base = reinterpret_cast<__half*>(dov_buffer);
        #pragma unroll 2
        for (int i = tid; i < (BLOCK_N * BLOCK_M + 1) / 2; i += THREADS) {
            const int row = i / ((BLOCK_M + 1) / 2);
            const int half_col = i % ((BLOCK_M + 1) / 2);
            const int col0 = half_col * 2;
            const int col1 = col0 + 1;

            float val0 = 0.0f, val1 = 0.0f;

            if (row < BLOCK_N && col0 < BLOCK_M) {
                if (row < valid_q_rows && col0 < valid_kv_rows) {
                    val0 = s_dS[row * S_STRIDE + col0];
                }
            }
            if (row < BLOCK_N && col1 < BLOCK_M) {
                if (row < valid_q_rows && col1 < valid_kv_rows) {
                    val1 = s_dS[row * S_STRIDE + col1];
                }
            }

            __half2 h2 = __float22half2_rn(make_float2(val0, val1));

           __half* dst = sdS_base + row * BLOCK_M + col0;
            dst[0] = h2.x;
            if (col1 < BLOCK_M) { dst[1] = h2.y; }
         }
         __syncthreads();

        // Load K with reuse(rewrite) V
        sK = sKV_base;
        sK_vec = reinterpret_cast<uint4*>(sK);
        #pragma unroll 2
        for (int idx = tid; idx < VECTOR_KV; idx += THREADS) {
            const int row = idx / VECTOR_D;
            const int vec_col = idx % VECTOR_D;
            uint4 val = make_uint4(0, 0, 0, 0);
            if (row < valid_kv_rows && vec_col < VECTOR_D) {
                val = __ldg(&k_vec[row * VECTOR_D + vec_col]);
            }
            sK_vec[row * kv_stride_uint4 + vec_col] = val;
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
            
            // Row-major mapping for better memory access pattern
            const int tile_m_idx = tile_idx / num_tiles_dk_n;
            const int tile_n_idx = tile_idx % num_tiles_dk_n;
            const int tile_m = tile_m_idx * WMMA_M;
            const int tile_n = tile_n_idx * WMMA_N;
            
            if (tile_m >= valid_kv_rows || tile_n >= D) continue;

            fragment<matrix_a, WMMA_M, WMMA_N, WMMA_K, half, col_major> a_frag;
            fragment<matrix_b, WMMA_M, WMMA_N, WMMA_K, half, row_major> b_frag;
            fragment<accumulator, WMMA_M, WMMA_N, WMMA_K, float> acc_frag;
            load_matrix_sync(acc_frag, s_dK_accum + tile_m * KV_STRIDE + tile_n, KV_STRIDE, mem_row_major);

            const int num_k_tiles = (valid_q_rows + WMMA_K - 1) / WMMA_K;
            #pragma unroll
            for (int k_tile = 0; k_tile < num_k_tiles; ++k_tile) {
                const int k_offset = k_tile * WMMA_K;
                if (k_offset >= valid_q_rows) break;
                
                load_matrix_sync(a_frag, sdS_base + k_offset * BLOCK_M + tile_m, BLOCK_M);
                load_matrix_sync(b_frag, sQ + k_offset * Q_STRIDE + tile_n, Q_STRIDE);
                mma_sync(acc_frag, a_frag, b_frag, acc_frag);
            }
            
            store_matrix_sync(s_dK_accum + tile_m * KV_STRIDE + tile_n, acc_frag, KV_STRIDE, mem_row_major);
        }
        __syncthreads();

        // Convert P to half with acc_buffer reuse
        __half* sP_half = reinterpret_cast<__half*>(acc_buffer);
        #pragma unroll 2
        for (int i = tid; i < (BLOCK_N * BLOCK_M + 1) / 2; i += THREADS) {
            const int row = i / ((BLOCK_M + 1) / 2);
            const int half_col = i % ((BLOCK_M + 1) / 2);
            const int col0 = half_col * 2;
            const int col1 = col0 + 1;

            float val0 = 0.0f, val1 = 0.0f;
            if (row < BLOCK_N && col0 < BLOCK_M) {
                if (row < valid_q_rows && col0 < valid_kv_rows) {
                    val0 = sP[row * S_STRIDE + col0];
                }
            }
            if (row < BLOCK_N && col1 < BLOCK_M) {
                if (row < valid_q_rows && col1 < valid_kv_rows) {
                    val1 = sP[row * S_STRIDE + col1];
                }
            }

            __half2 h2 = __float22half2_rn(make_float2(val0, val1));
            __half* dst = sP_half + row * BLOCK_M + col0;
            dst[0] = h2.x;
            if (col1 < BLOCK_M) { dst[1] = h2.y; }
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
            load_matrix_sync(acc_frag, s_dV_accum + tile_m * KV_STRIDE + tile_n, KV_STRIDE, mem_row_major);

            const int num_k_tiles = (valid_q_rows + WMMA_K - 1) / WMMA_K;
            #pragma unroll
            for (int k_tile = 0; k_tile < num_k_tiles; ++k_tile) {
                const int k_offset = k_tile * WMMA_K;
                if (k_offset >= valid_q_rows) break;
                
                load_matrix_sync(a_frag, sP_half + k_offset * BLOCK_M + tile_m, BLOCK_M);
                load_matrix_sync(b_frag, sdO + k_offset * Q_STRIDE + tile_n, Q_STRIDE);
                mma_sync(acc_frag, a_frag, b_frag, acc_frag);
            }

            store_matrix_sync(s_dV_accum + tile_m * KV_STRIDE + tile_n, acc_frag, KV_STRIDE, mem_row_major);
        }
        __syncthreads();
    }

    // Write dK + dV to global memory at once
    int vec_elem_dkv = valid_kv_rows * D;
    int vec_size_dkv = (vec_elem_dkv + 4 - 1) / 4;

    for (int vec_idx = tid; vec_idx < vec_size_dkv; vec_idx += THREADS) {
        int linear_idx = vec_idx * 4;
        int row = linear_idx / D;
        int col = linear_idx % D;
        
        if (row < valid_kv_rows) {
            float4 dK_val, dV_val;
            dK_val.x = s_dK_accum[row * KV_STRIDE + col];
            dK_val.y = (col + 1 < D) ? s_dK_accum[row * KV_STRIDE + col + 1] : 0.0f;
            dK_val.z = (col + 2 < D) ? s_dK_accum[row * KV_STRIDE + col + 2] : 0.0f;
            dK_val.w = (col + 3 < D) ? s_dK_accum[row * KV_STRIDE + col + 3] : 0.0f;
            
            dV_val.x = s_dV_accum[row * KV_STRIDE + col];
            dV_val.y = (col + 1 < D) ? s_dV_accum[row * KV_STRIDE + col + 1] : 0.0f;
            dV_val.z = (col + 2 < D) ? s_dV_accum[row * KV_STRIDE + col + 2] : 0.0f;
            dV_val.w = (col + 3 < D) ? s_dV_accum[row * KV_STRIDE + col + 3] : 0.0f;
            
            reinterpret_cast<float4*>(&dK_ptr[row * D + col])[0] = dK_val;
            reinterpret_cast<float4*>(&dV_ptr[row * D + col])[0] = dV_val;
        }
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
    torch::Tensor& O,
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
            O.data_ptr<float>(),
            reinterpret_cast<const __half*>(dO.data_ptr()),
            softmax_lse.data_ptr<float>(),
            dQ.data_ptr<float>(),
            B, H, M, N, softmax_scale
        );
    } else {
        flash_attention_backward_dq_kernel<D, false><<<grid, block, smem, stream>>>(
            reinterpret_cast<const __half*>(Q.data_ptr()),
            reinterpret_cast<const __half*>(K.data_ptr()),
            reinterpret_cast<const __half*>(V.data_ptr()),
            O.data_ptr<float>(),
            reinterpret_cast<const __half*>(dO.data_ptr()),
            softmax_lse.data_ptr<float>(),
            dQ.data_ptr<float>(),
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
    torch::Tensor& O,
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
            O.data_ptr<float>(),
            reinterpret_cast<const __half*>(dO.data_ptr()),
            softmax_lse.data_ptr<float>(),
            dK.data_ptr<float>(),
            dV.data_ptr<float>(),
            B, H, M, N, softmax_scale
        );
    } else {
        flash_attention_backward_dkv_kernel<D, false><<<grid, block, smem, stream>>>(
            reinterpret_cast<const __half*>(Q.data_ptr()),
            reinterpret_cast<const __half*>(K.data_ptr()),
            reinterpret_cast<const __half*>(V.data_ptr()),
            O.data_ptr<float>(),
            reinterpret_cast<const __half*>(dO.data_ptr()),
            softmax_lse.data_ptr<float>(),
            dK.data_ptr<float>(),
            dV.data_ptr<float>(),
            B, H, M, N, softmax_scale
        );
    }
}


// ============================================================================
// WRAPPER
// ============================================================================
void flash_attention_backward(
    const torch::Tensor& Q,
    const torch::Tensor& K,
    const torch::Tensor& V,
    torch::Tensor& O,
    torch::Tensor& dO,
    torch::Tensor& softmax_lse,
    torch::Tensor& dQ,
    torch::Tensor& dK,
    torch::Tensor& dV,
    float softmax_scale,
    bool is_causal
) {
    TORCH_CHECK(Q.is_cuda() && K.is_cuda() && V.is_cuda() && O.is_cuda() && dO.is_cuda() && softmax_lse.is_cuda(), "All tensors must be on CUDA");
    TORCH_CHECK(Q.dtype() == torch::kFloat16 && K.dtype() == torch::kFloat16 && V.dtype() == torch::kFloat16, "Q/K/V must be fp16");
    TORCH_CHECK(dO.dtype() == torch::kFloat16, "dO must be fp16");
    TORCH_CHECK(O.dtype() == torch::kFloat32 && dQ.dtype() == torch::kFloat32 && dK.dtype() == torch::kFloat32 && dV.dtype() == torch::kFloat32, 
                "O and gradients must be fp32");
    TORCH_CHECK(softmax_lse.dtype() == torch::kFloat32, "softmax_lse must be fp32");
    const int D = Q.size(3);
    TORCH_CHECK(D % 2 == 0, "Embedding dimension D must be even, but got D = ", D);

    dQ.zero_();
    dK.zero_();
    dV.zero_();

    auto stream = at::cuda::getCurrentCUDAStream().stream();
    auto dprops = at::cuda::getCurrentDeviceProperties();
    bool is_sm70 = dprops->major == 7 && dprops->minor == 0;
    TORCH_CHECK(is_sm70, "Kernel supports only Volta GPUs.");
    
    switch (D) {
        case 16:  
            launcher_flash_attention_backward_dq<16>(Q, K, V, O, dO, softmax_lse, dQ, softmax_scale, is_causal, stream); 
            launcher_flash_attention_backward_dkv<16>(Q, K, V, O, dO, softmax_lse, dK, dV, softmax_scale, is_causal, stream); 
            break;
        case 32:  
            launcher_flash_attention_backward_dq<32>(Q, K, V, O, dO, softmax_lse, dQ, softmax_scale, is_causal, stream); 
            launcher_flash_attention_backward_dkv<32>(Q, K, V, O, dO, softmax_lse, dK, dV, softmax_scale, is_causal, stream); 
            break;
        case 64:  
            launcher_flash_attention_backward_dq<64>(Q, K, V, O, dO, softmax_lse, dQ, softmax_scale, is_causal, stream); 
            launcher_flash_attention_backward_dkv<64>(Q, K, V, O, dO, softmax_lse, dK, dV, softmax_scale, is_causal, stream); 
            break;
        case 128: 
            launcher_flash_attention_backward_dq<128>(Q, K, V, O, dO, softmax_lse, dQ, softmax_scale, is_causal, stream); 
            launcher_flash_attention_backward_dkv<128>(Q, K, V, O, dO, softmax_lse, dK, dV, softmax_scale, is_causal, stream); 
            break;
        case 256: 
            launcher_flash_attention_backward_dq<256>(Q, K, V, O, dO, softmax_lse, dQ, softmax_scale, is_causal, stream); 
            launcher_flash_attention_backward_dkv<256>(Q, K, V, O, dO, softmax_lse, dK, dV, softmax_scale, is_causal, stream); 
            break;
        default: TORCH_CHECK(false, "Unsupported D: ", D);
    }
    
    cudaError_t err = cudaGetLastError();
    TORCH_CHECK(err == cudaSuccess, "Kernel failed: ", cudaGetErrorString(err));
}