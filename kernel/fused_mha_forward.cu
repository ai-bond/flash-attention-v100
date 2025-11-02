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

// ============================================================================
// VOLTA SM70 WMMA CONSTANTS
// ============================================================================
#define WMMA_M 16
#define WMMA_N 16
#define WMMA_K 16
#define WARP_SIZE 32
#define MAX_SMEM (96 * 1024)

// ============================================================================
// CONFIGURATIONS
// ============================================================================
#define BLOCK_M_16  16
#define BLOCK_N_16  512
#define WARPS_16    2

#define BLOCK_M_32  32
#define BLOCK_N_32  256
#define WARPS_32    4

#define BLOCK_M_64  64
#define BLOCK_N_64  128
#define WARPS_64    8

#define BLOCK_M_128 32
#define BLOCK_N_128 176
#define WARPS_128   8

#define BLOCK_M_256 32
#define BLOCK_N_256 64
#define WARPS_256   8

// ============================================================================
// COMPILE-TIME CONFIG
// ============================================================================
template<int D>
struct KernelConfig {
    static constexpr int BLOCK_M = (D == 16) ? BLOCK_M_16 : (D == 32) ? BLOCK_M_32 : (D == 64) ? BLOCK_M_64 : (D == 128) ? BLOCK_M_128 : BLOCK_M_256;
    static constexpr int BLOCK_N = (D == 16) ? BLOCK_N_16 : (D == 32) ? BLOCK_N_32 : (D == 64) ? BLOCK_N_64 : (D == 128) ? BLOCK_N_128 : BLOCK_N_256;
    static constexpr int WARPS_PER_BLOCK = (D == 16) ? WARPS_16 : (D == 32) ? WARPS_32 : (D == 64) ? WARPS_64 : (D == 128) ? WARPS_128 : WARPS_256;
    
    static constexpr int THREADS_PER_BLOCK = WARPS_PER_BLOCK * WARP_SIZE;
    static constexpr int NUM_K_TILES       = (D + WMMA_K - 1) / WMMA_K;
    static constexpr int THREADS_PER_ROW   = THREADS_PER_BLOCK / BLOCK_M;
    static constexpr int PAD               = (8 - (D % 32) + 32) % 32;
    static constexpr int Q_STRIDE          = D + PAD;
    static constexpr int KV_STRIDE         = D + PAD;
    static constexpr int S_STRIDE          = BLOCK_N + PAD;
    static constexpr int O_STRIDE          = D + PAD;
    static constexpr int PER_UINT4         = 8;
    static constexpr int PER_FLOAT4        = 4;
    static constexpr int VECTOR_D          = (D + PER_UINT4 - 1) / PER_UINT4;
    static constexpr int VECTOR_F          = (BLOCK_M * O_STRIDE + PER_FLOAT4 - 1) / PER_FLOAT4;
    static constexpr size_t SMEM_Q = (((size_t)BLOCK_M * Q_STRIDE * sizeof(__half) + 127) & ~static_cast<size_t>(127));
    static constexpr size_t SMEM_K = (((size_t)BLOCK_N * KV_STRIDE * sizeof(__half) + 127) & ~static_cast<size_t>(127));
    static constexpr size_t SMEM_S = (((size_t)BLOCK_M * S_STRIDE * sizeof(float) + 127) & ~static_cast<size_t>(127));
    static constexpr size_t SMEM_O = (((size_t)BLOCK_M * O_STRIDE * sizeof(float) + 127) & ~static_cast<size_t>(127));
    static constexpr size_t SMEM_STATE = (((size_t)3 * BLOCK_M * sizeof(float) + 127) & ~static_cast<size_t>(127));
    static constexpr size_t TOTAL_SMEM = SMEM_Q + SMEM_K + SMEM_S + SMEM_O + SMEM_STATE;
};

// ============================================================================
// OPTIMIZED KERNEL
// ============================================================================
template<int D, bool IS_CAUSAL>
__global__ void __launch_bounds__(KernelConfig<D>::THREADS_PER_BLOCK, 2)
flash_attention_forward_kernel(
    const __half* __restrict__ Q,
    const __half* __restrict__ K,
    const __half* __restrict__ V,
          __half* __restrict__ Out,
           float* __restrict__ softmax_lse,
    const int B,
    const int H,
    const int M,
    const int N,
    const float softmax_scale
) {
    using Config = KernelConfig<D>;
    constexpr int BLOCK_M = Config::BLOCK_M;
    constexpr int BLOCK_N = Config::BLOCK_N;
    constexpr int THREADS = Config::THREADS_PER_BLOCK;
    constexpr int NUM_K_TILES = Config::NUM_K_TILES;
    constexpr int THREADS_PER_ROW = Config::THREADS_PER_ROW;
    constexpr int WARPS_PER_BLOCK = Config::WARPS_PER_BLOCK;
    constexpr int Q_STRIDE = Config::Q_STRIDE;
    constexpr int KV_STRIDE = Config::KV_STRIDE;
    constexpr int S_STRIDE = Config::S_STRIDE;
    constexpr int O_STRIDE = Config::O_STRIDE;
    constexpr int PER_UINT4 = Config::PER_UINT4;
    constexpr int PER_FLOAT4 = Config::PER_FLOAT4;
    constexpr int VECTOR_D = Config::VECTOR_D;
    constexpr int VECTOR_F = Config::VECTOR_F;
    const float NEG_INF = -1e30f;
    
    const int batch_head_id = blockIdx.z;
    if (batch_head_id >= B * H) return;
    
    const int block_m = blockIdx.x;
    const int start_row = block_m * BLOCK_M;
    if (start_row >= M) return;
    
    const int valid_q_rows = min(BLOCK_M, M - start_row);
    const int tid = threadIdx.x;
    const int warp_id = tid / WARP_SIZE;
    const int lane_id = tid % WARP_SIZE;
    
    // Global pointers
    const __half* q_ptr    = Q +           (size_t)batch_head_id * M * D + start_row * D;
    const __half* k_ptr    = K +           (size_t)batch_head_id * N * D;
    const __half* v_ptr    = V +           (size_t)batch_head_id * N * D;
          __half* out_ptr  = Out +         (size_t)batch_head_id * M * D + start_row * D;
    float* softmax_lse_ptr = softmax_lse + (size_t)batch_head_id * M + start_row;
    
    // Shared memory layout
    extern __shared__ char smem[];
   
    __half* sQ      = reinterpret_cast<__half*>(smem);
    __half* sKV     = sQ + BLOCK_M * Q_STRIDE;
    float*  sS      = reinterpret_cast<float*>(sKV + BLOCK_N * KV_STRIDE);
    float*  sO      = sS + BLOCK_M * S_STRIDE;
    float*  sRowMax = sO + BLOCK_M * O_STRIDE;
    float*  sRowSum = sRowMax + BLOCK_M;
    float*  sOldMax = sRowSum + BLOCK_M;

    // Load Q into shared memory at once
    const uint4* q_vec = reinterpret_cast<const uint4*>(q_ptr);
    uint4* sQ_vec = reinterpret_cast<uint4*>(sQ);
    const int q_stride_uint4 = (Q_STRIDE + PER_UINT4 - 1) / PER_UINT4;
    #pragma unroll 4
    for (int idx = tid; idx < BLOCK_M * VECTOR_D; idx += THREADS) {
        const int row = idx / VECTOR_D;
        const int vec_col = idx % VECTOR_D;
        uint4 val = make_uint4(0, 0, 0, 0);
        if (row < valid_q_rows && vec_col < VECTOR_D) {
            val = __ldg(&q_vec[row * VECTOR_D + vec_col]);
        }
        sQ_vec[row * q_stride_uint4 + vec_col] = val;
    }
    __syncthreads();
    
    // Initialize buffer's
    if (tid < BLOCK_M) {
        sRowMax[tid] = NEG_INF;
        sRowSum[tid] = 0.0f;
        sOldMax[tid] = 1.0f;
    }
    
    // Initialize O to zero
    float4* sO_vec = reinterpret_cast<float4*>(sO);
    const float4 zero4 = make_float4(0.0f, 0.0f, 0.0f, 0.0f);    
    #pragma unroll 4
    for (int i = tid; i < VECTOR_F; i += THREADS) {
        sO_vec[i] = zero4;
    }
    __syncthreads();
    
    const int num_n_blocks = (N + BLOCK_N - 1) / BLOCK_N;
    
    // ========================================================================
    // MAIN LOOP
    // ========================================================================
    for (int block_n = 0; block_n < num_n_blocks; ++block_n) {
        const int start_col = block_n * BLOCK_N;
        if (start_col >= N) break;
        const int valid_k_rows = min(BLOCK_N, N - start_col);
        
        // Load K into shared memory
        __half* sK = sKV;
        const uint4* k_vec = reinterpret_cast<const uint4*>(k_ptr + start_col * D);
        uint4* sK_vec = reinterpret_cast<uint4*>(sK);
        const int kv_stride_uint4 = (KV_STRIDE + PER_UINT4 - 1) / PER_UINT4;
        #pragma unroll 4
        for (int idx = tid; idx < BLOCK_N * VECTOR_D; idx += THREADS) {
            const int row = idx / VECTOR_D;
            const int vec_col = idx % VECTOR_D;
            uint4 val = make_uint4(0, 0, 0, 0);
            if (row < valid_k_rows && vec_col < VECTOR_D) {
                val = __ldg(&k_vec[row * VECTOR_D + vec_col]);
            }
            sK_vec[row * kv_stride_uint4 + vec_col] = val;
        }
        __syncthreads();
        
        // Compute S = Q @ K^T
        const int num_tiles_m = (BLOCK_M + WMMA_M - 1) / WMMA_M;
        const int num_tiles_n_actual = (valid_k_rows + WMMA_N - 1) / WMMA_N;
        const int total_tiles = num_tiles_m * num_tiles_n_actual;
        const int tiles_per_warp = (total_tiles + WARPS_PER_BLOCK - 1) / WARPS_PER_BLOCK;
        
        for (int tile_idx = 0; tile_idx < tiles_per_warp; ++tile_idx) {
            const int global_tile_idx = warp_id * tiles_per_warp + tile_idx;
            if (global_tile_idx >= total_tiles) break;
            
            const int tile_m_idx = global_tile_idx / num_tiles_n_actual;
            const int tile_n_idx = global_tile_idx % num_tiles_n_actual;
            
            const int tile_m = tile_m_idx * WMMA_M;
            const int tile_n = tile_n_idx * WMMA_N;
            
            if (tile_m >= valid_q_rows || tile_n >= valid_k_rows) continue;
            
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
            for (int idx = tid; idx < BLOCK_M * BLOCK_N; idx += THREADS) {
                const int local_m = idx / BLOCK_N;
                const int local_n = idx % BLOCK_N;
                const int global_m = start_row + local_m;
                const int global_n = start_col + local_n;
                
                if (local_m < valid_q_rows && local_n < valid_k_rows && global_n > global_m) {
                    sS[local_m * S_STRIDE + local_n] = NEG_INF;
                }
            }
            __syncthreads();
        }
        
        // Online Softmax
        if (tid < valid_q_rows * THREADS_PER_ROW) {
            const int row = tid / THREADS_PER_ROW;
            const int thread_in_row = tid % THREADS_PER_ROW;
            float* row_scores = sS + row * S_STRIDE;
            
            float thread_max = NEG_INF;
            const int cols_per_thread = (valid_k_rows + THREADS_PER_ROW - 1) / THREADS_PER_ROW;
            
            #pragma unroll 4
            for (int j = 0; j < cols_per_thread; ++j) {
                const int col = thread_in_row + j * THREADS_PER_ROW;
                if (col < valid_k_rows) {
                    thread_max = fmaxf(thread_max, row_scores[col]);
                }
            }
            
            #pragma unroll
            for (int offset = 1; offset < THREADS_PER_ROW; offset *= 2) {
                thread_max = fmaxf(thread_max, __shfl_xor_sync(0xffffffff, thread_max, offset));
            }
            
            const float old_max = sRowMax[row];
            const float new_max = fmaxf(old_max, thread_max);
            const float exp_diff = __expf(old_max - new_max);
            
            float thread_sum = 0.0f;
            #pragma unroll 4
            for (int j = 0; j < cols_per_thread; ++j) {
                const int col = thread_in_row + j * THREADS_PER_ROW;
                if (col < valid_k_rows) {
                    float shifted = row_scores[col] - new_max;
                    float exp_val = (shifted < -80.0f) ? 0.0f : __expf(shifted);
                    row_scores[col] = exp_val;
                    thread_sum += exp_val;
                }
            }
            
            #pragma unroll
            for (int offset = 1; offset < THREADS_PER_ROW; offset *= 2) {
                thread_sum += __shfl_xor_sync(0xffffffff, thread_sum, offset);
            }
            
            if (thread_in_row == 0) {
                const float old_sum = sRowSum[row];
                const float new_sum = exp_diff * old_sum + thread_sum;
                sRowSum[row] = new_sum;
                sRowMax[row] = new_max;
                sOldMax[row] = exp_diff;
            }
        }
        __syncthreads();
        
        // Scale output
        if (block_n > 0) {
            const int total_elems = BLOCK_M * O_STRIDE;
            const int vec_size = (total_elems + PER_FLOAT4 - 1) / PER_FLOAT4;
            float4* sO_vec4 = reinterpret_cast<float4*>(sO);
            
            #pragma unroll 4
            for (int i = tid; i < vec_size; i += THREADS) {
                int row = (i * PER_FLOAT4) / O_STRIDE;
                if (row >= valid_q_rows) continue;
                
                float s = sOldMax[row];
                float4& v = sO_vec4[i];
                v.x *= s;
                v.y *= s;
                v.z *= s;
                v.w *= s;
            }
            __syncthreads();
        }
        
        // Convert dS to half precision
        __half* sP = reinterpret_cast<__half*>(sS);
        #pragma unroll 4
        for (int i = tid; i < (BLOCK_M * BLOCK_N + 1) / 2; i += THREADS) {
            const int row = i / ((BLOCK_N + 1) / 2);
            const int half_col = i % ((BLOCK_N + 1) / 2);
            const int col0 = half_col * 2;
            const int col1 = col0 + 1;

            float2 vals;
            vals.x = (col0 < BLOCK_N && row < valid_q_rows && col0 < valid_k_rows) ? 
                     sS[row * S_STRIDE + col0] : 0.0f;
            vals.y = (col1 < BLOCK_N && row < valid_q_rows && col1 < valid_k_rows) ? 
                     sS[row * S_STRIDE + col1] : 0.0f;
            
            vals.x = isfinite(vals.x) ? vals.x : 0.0f;
            vals.y = isfinite(vals.y) ? vals.y : 0.0f;

            __half2 h2 = __float22half2_rn(vals);
            __half* dst = sP + row * S_STRIDE + col0;
            dst[0] = h2.x;
            if (col1 < BLOCK_N) {
                dst[1] = h2.y;
            }
        }
        __syncthreads();
        

        // Load V
        __half* sV = sKV;
        const uint4* v_vec = reinterpret_cast<const uint4*>(v_ptr + start_col * D);
        uint4* sV_vec = reinterpret_cast<uint4*>(sV);
        
        #pragma unroll 4
        for (int idx = tid; idx < BLOCK_N * VECTOR_D; idx += THREADS) {
            const int row = idx / VECTOR_D;
            const int vec_col = idx % VECTOR_D;
            uint4 val = make_uint4(0, 0, 0, 0);
            if (row < valid_k_rows && vec_col < VECTOR_D) {
                val = __ldg(&v_vec[row * VECTOR_D + vec_col]);
            }
            sV_vec[row * kv_stride_uint4 + vec_col] = val;
        }
        __syncthreads();
        

        // Compute P @ V
        const int num_tiles_m_pv = (BLOCK_M + WMMA_M - 1) / WMMA_M;
        const int num_tiles_d = (D + WMMA_N - 1) / WMMA_N;
        const int total_tiles_pv = num_tiles_m_pv * num_tiles_d;
        const int tiles_per_warp_pv = (total_tiles_pv + WARPS_PER_BLOCK - 1) / WARPS_PER_BLOCK;
        
        for (int tile_idx = 0; tile_idx < tiles_per_warp_pv; ++tile_idx) {
            const int global_tile_idx = warp_id * tiles_per_warp_pv + tile_idx;
            if (global_tile_idx >= total_tiles_pv) break;
            
            const int tile_m_idx = global_tile_idx / num_tiles_d;
            const int tile_d_idx = global_tile_idx % num_tiles_d;
            
            const int tile_m = tile_m_idx * WMMA_M;
            const int tile_d = tile_d_idx * WMMA_N;
            
            if (tile_m >= valid_q_rows) continue;
            
            if (tile_d + WMMA_N < D && lane_id < 4) {
                const int prefetch_offset = tile_d + WMMA_N;
                const int prefetch_row = lane_id * 4;
                if (prefetch_row < valid_k_rows) {
                    const __half* prefetch_addr = sV + prefetch_row * KV_STRIDE + prefetch_offset;
                    asm volatile("prefetch.global.L1 [%0];" :: "l"(prefetch_addr));
                }
            }
            
            fragment<matrix_a, WMMA_M, WMMA_N, WMMA_K, half, row_major> p_frag;
            fragment<matrix_b, WMMA_M, WMMA_N, WMMA_K, half, row_major> v_frag;
            fragment<accumulator, WMMA_M, WMMA_N, WMMA_K, float> o_frag;
            
            load_matrix_sync(o_frag, sO + tile_m * O_STRIDE + tile_d, O_STRIDE, mem_row_major);
            
            #pragma unroll
            for (int tile_k = 0; tile_k < (valid_k_rows + WMMA_K - 1) / WMMA_K; ++tile_k) {
                const int k_offset = tile_k * WMMA_K;
                if (k_offset >= valid_k_rows) break;
                
                load_matrix_sync(p_frag, sP + tile_m * S_STRIDE + k_offset, S_STRIDE);
                load_matrix_sync(v_frag, sV + k_offset * KV_STRIDE + tile_d, KV_STRIDE);
                mma_sync(o_frag, p_frag, v_frag, o_frag);
            }
            
            store_matrix_sync(sO + tile_m * O_STRIDE + tile_d, o_frag, O_STRIDE, mem_row_major);
        }
        __syncthreads();
    }
    
    // Store final Sum to global memory
    const int total_fp16_x4 = (valid_q_rows * D) / 4;
    
    for (int i = tid; i < total_fp16_x4; i += THREADS) {
        const int row = i / (D / 4);
        const int col = (i % (D / 4)) * 4;

        const float inv_sum = (sRowSum[row] > 1e-6f) ? (1.0f / sRowSum[row]) : 1.0f;
        const float* sO_row = sO + row * O_STRIDE;

        const __half h0 = __float2half_rn(sO_row[col + 0] * inv_sum);
        const __half h1 = __float2half_rn(sO_row[col + 1] * inv_sum);
        const __half h2 = __float2half_rn(sO_row[col + 2] * inv_sum);
        const __half h3 = __float2half_rn(sO_row[col + 3] * inv_sum);

        asm volatile(
            "st.global.v4.u16 [%0], {%1, %2, %3, %4};"
            :
            : "l"(out_ptr + row * D + col),
              "h"(__half_as_ushort(h0)),
              "h"(__half_as_ushort(h1)),
              "h"(__half_as_ushort(h2)),
              "h"(__half_as_ushort(h3))
            : "memory"
        );
    }
    
    if (tid < valid_q_rows) {
        float sum = fmaxf(sRowSum[tid], 1e-24f);
        softmax_lse_ptr[tid] = sRowMax[tid] + logf(sum);
    }
}

// ============================================================================
// LAUNCHER
// ============================================================================
template<int D>
void launcher_flash_attention_forward(
    const torch::Tensor& Q,
    const torch::Tensor& K,
    const torch::Tensor& V,
    torch::Tensor& Out,
    torch::Tensor& softmax_lse,
    float softmax_scale,
    bool is_causal,
    cudaStream_t stream
) {
    using Config = KernelConfig<D>;
    
    const int B = Q.size(0);
    const int H = Q.size(1);
    const int M = Q.size(2);
    const int N = K.size(2);
    
    const int grid_x = (M + Config::BLOCK_M - 1) / Config::BLOCK_M;
    const dim3 grid(grid_x, 1, B * H);
    const dim3 block(Config::THREADS_PER_BLOCK);
    const size_t smem = Config::TOTAL_SMEM;

    TORCH_CHECK(smem <= MAX_SMEM, "Shared memory exceeds 96KB: ", smem, " bytes");
    
    auto kernel = is_causal ?
        (void*)flash_attention_forward_kernel<D, true> :
        (void*)flash_attention_forward_kernel<D, false>;
    
    cudaFuncSetAttribute(kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, smem);
    
    if (is_causal) {
        flash_attention_forward_kernel<D, true><<<grid, block, smem, stream>>>(
            reinterpret_cast<const __half*>(Q.data_ptr()),
            reinterpret_cast<const __half*>(K.data_ptr()),
            reinterpret_cast<const __half*>(V.data_ptr()),
            reinterpret_cast<__half*>(Out.data_ptr()),
            softmax_lse.data_ptr<float>(),
            B, H, M, N, softmax_scale
        );
    } else {
        flash_attention_forward_kernel<D, false><<<grid, block, smem, stream>>>(
            reinterpret_cast<const __half*>(Q.data_ptr()),
            reinterpret_cast<const __half*>(K.data_ptr()),
            reinterpret_cast<const __half*>(V.data_ptr()),
            reinterpret_cast<__half*>(Out.data_ptr()),
            softmax_lse.data_ptr<float>(),
            B, H, M, N, softmax_scale
        );
    }
}

// ============================================================================
// WRAPPER
// ============================================================================
std::vector<at::Tensor> flash_attention_forward(
    at::Tensor& q,
    const at::Tensor& k,
    const at::Tensor& v,
    std::optional<at::Tensor>& out_,
    std::optional<at::Tensor>& alibi_slopes_,
    const float p_dropout,
    const float softmax_scale,
    bool is_causal,
    int window_size_left,
    int window_size_right,
    const float softcap,
    const bool return_softmax,
    std::optional<at::Generator> gen_
) {
    // Now unsupported functions
    TORCH_CHECK(!alibi_slopes_.has_value(), "alibi_slopes not supported");
    TORCH_CHECK(p_dropout == 0.f, "dropout not supported");
    TORCH_CHECK(window_size_left == -1, "window_size_left not supported");
    TORCH_CHECK(window_size_right == -1 || (is_causal && window_size_right == 0), "window not supported");
    TORCH_CHECK(softcap == 0.f, "softcap not supported");
    TORCH_CHECK(!return_softmax, "return_softmax not supported");
    TORCH_CHECK(!gen_.has_value(), "Generator not supported");

    // Check layouts
    TORCH_CHECK(q.dtype() == torch::kFloat16, "q must be fp16");
    TORCH_CHECK(k.dtype() == torch::kFloat16, "k must be fp16");
    TORCH_CHECK(v.dtype() == torch::kFloat16, "v must be fp16");
    TORCH_CHECK(q.is_cuda() && k.is_cuda() && v.is_cuda(), "Tensors must be on CUDA");
    TORCH_CHECK(q.stride(-1) == 1 && k.stride(-1) == 1 && v.stride(-1) == 1, "Last dim must be contiguous");

    const auto sizes = q.sizes();
    const int B = sizes[0], H = sizes[1], M = sizes[2], D = sizes[3];
    const int N = k.size(2);
    TORCH_CHECK(D <= 256 && D % 8 == 0 && D % 2 == 0, "D must be even, <=256, multiple of 8");

    // Out tensors
    at::Tensor out_fp16 = out_.has_value() ? out_.value() : torch::empty_like(q);
    TORCH_CHECK(out_fp16.dtype() == torch::kFloat16, "out must be fp16");
    auto softmax_lse = torch::empty({B, H, M}, torch::dtype(torch::kFloat32).device(q.device()));
    TORCH_CHECK(softmax_lse.dtype() == torch::kFloat32, "softmax_lse must be fp32");

    auto stream = at::cuda::getCurrentCUDAStream().stream();
    auto props  = at::cuda::getCurrentDeviceProperties();
    bool sm70   = props->major == 7 && props->minor == 0;
    TORCH_CHECK(sm70, "Kernel supports only Volta GPUs.");

    switch (D) {
        case 16:  launcher_flash_attention_forward<16>(q, k, v, out_fp16, softmax_lse, softmax_scale, is_causal, stream); break;
        case 32:  launcher_flash_attention_forward<32>(q, k, v, out_fp16, softmax_lse, softmax_scale, is_causal, stream); break;
        case 64:  launcher_flash_attention_forward<64>(q, k, v, out_fp16, softmax_lse, softmax_scale, is_causal, stream); break;
        case 128: launcher_flash_attention_forward<128>(q, k, v, out_fp16, softmax_lse, softmax_scale, is_causal, stream); break;
        case 256: launcher_flash_attention_forward<256>(q, k, v, out_fp16, softmax_lse, softmax_scale, is_causal, stream); break;
        default: TORCH_CHECK(false, "Unsupported D: ", D);
    }
    
    auto p = torch::empty({0}, q.options());
    auto rng_state = torch::empty({2}, torch::dtype(torch::kInt64).device(q.device()));
    return {out_fp16, softmax_lse, p, rng_state};
}