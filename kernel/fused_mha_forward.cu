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

#include <mma.h>
using namespace nvcuda::wmma;

#include "00_volta_const.cuh"
#include "01_forward_config.cuh"
#include "02_fused_func.cuh"
#include "03_wmma.cuh"

// ======================================================================================
// FORWARD KERNEL
// ======================================================================================
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
    constexpr int BLOCK_M           = Config::BLOCK_M;
    constexpr int BLOCK_N           = Config::BLOCK_N;
    constexpr int THREADS_PER_BLOCK = Config::THREADS_PER_BLOCK;
    constexpr int THREADS_PER_ROW   = Config::THREADS_PER_ROW;
    constexpr int WARPS_PER_BLOCK   = Config::WARPS_PER_BLOCK;
    constexpr int D_STRIDE          = Config::D_STRIDE;
    constexpr int N_STRIDE          = Config::N_STRIDE;

    // head index (batch * num_heads + head)
    const int batch_head_id = blockIdx.z;
    if (batch_head_id >= B * H) return;

    const int block_idx = blockIdx.x;
    const int start_q = block_idx * BLOCK_M;
    if (start_q >= M) return;

    int num_kv_tiles = (N + BLOCK_N - 1) / BLOCK_N;
    const int valid_q_rows = min(BLOCK_M, M - start_q);

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
            num_kv_tiles = min(num_kv_tiles, (max_key_pos + BLOCK_N + 0) / BLOCK_N);
        }
    }

    // ==================================================================================
    // Init:   thread/warp/lane IDs for WMMA coordination
    // ==================================================================================
    const int tid     = threadIdx.x;
    const int warp_id = tid >> 5;
    const int lane_id = tid & 31;

    // ==================================================================================
    // Layout: [B, H, M/N, D] linear offset: batch_head_id * (M/N) * D + start_* * D
    // ==================================================================================
    const __half* __restrict__ q_ptr           = Q +           (size_t)batch_head_id * M * D + start_q * D;
    const __half* __restrict__ k_ptr           = K +           (size_t)batch_head_id * N * D;
    const __half* __restrict__ v_ptr           = V +           (size_t)batch_head_id * N * D;
          __half* __restrict__ out_ptr         = Out +         (size_t)batch_head_id * M * D + start_q * D;
           float* __restrict__ softmax_lse_ptr = softmax_lse + (size_t)batch_head_id * M + start_q;

    // ==================================================================================
    // Init:   shared memory with zero-fill union regions to avoid stale data
    // ==================================================================================
    extern __shared__ char smem_raw[];

    INIT_SMEM<Config>(smem_raw);
    __syncthreads();

    auto& smem = *reinterpret_cast<typename Config::SmemLayout*>(smem_raw);

    __half* __restrict__ sQ      = smem.q;
    __half* __restrict__ sK      = smem.reuse_kv.k;
    __half* __restrict__ sV      = smem.reuse_kv.v;
    float*  __restrict__ sS      = smem.reuse_sp.s;
    __half* __restrict__ sP      = smem.reuse_sp.p;
    float*  __restrict__ sO      = smem.o;
    float*  __restrict__ sRowMax = smem.row_max;
    float*  __restrict__ sRowSum = smem.row_sum;

    if (tid < BLOCK_M) {
        sRowMax[tid] = NEG_INF;
    }

    // ==================================================================================
    // Load:     Q tile from global to sQ shared memory
    // Layout:   Q: global[row: BLOCK_M, D] -> shared[row: BLOCK_M, D_STRIDE]
    // Template: SRC_STRIDE=D, DST_STRIDE=D_STRIDE
    // ==================================================================================
    const uint4*      q_vec = reinterpret_cast<const uint4*>(q_ptr);
          uint4*     sQ_vec = reinterpret_cast<uint4*>(sQ);

    LOAD_TILE<D, D_STRIDE>(q_vec, sQ_vec, valid_q_rows, tid, THREADS_PER_BLOCK);

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
        // Load:     K tile from global to sK(reuse) shared memory
        // Layout:   K: global[row: BLOCK_N, D] -> shared[row: BLOCK_N, D_STRIDE]
        // Template: SRC_STRIDE=D, DST_STRIDE=D_STRIDE
        // ==================================================================================
        const uint4* k_vec     = reinterpret_cast<const uint4*>(k_ptr + start_kv * D);
              uint4* sK_vec    = reinterpret_cast<uint4*>(sK);

        LOAD_TILE<D, D_STRIDE>(k_vec, sK_vec, valid_kv_rows, tid, THREADS_PER_BLOCK);

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
        // Compute Online Softmax + Scale
        // ==================================================================================
        if (tid < valid_q_rows * THREADS_PER_ROW) {
            const int row = tid / THREADS_PER_ROW;
            const int thread_in_row = tid % THREADS_PER_ROW;
            const unsigned mask = (valid_q_rows == BLOCK_M) ? 0xFFFFFFFFU : __activemask();
            const int row_leader = __ffs(mask) - 1;

            float*  sS_row_f = sS + row * N_STRIDE;
            __half* sP_row_h = sP + row * N_STRIDE;

            const int vec_cols = valid_kv_rows >> 2;
            const int vecs_per_thread = (vec_cols + THREADS_PER_ROW - 1) / THREADS_PER_ROW;
            const int tail_start = vec_cols << 2;

            // Phase 1: Max reduction with prefetch
            float thread_max = NEG_INF;
            float4* sS_vec4 = reinterpret_cast<float4*>(sS_row_f);

            #pragma unroll 4
            for (int j = 0; j < vecs_per_thread; ++j) {
                int vc = thread_in_row + j * THREADS_PER_ROW;
                if (vc < vec_cols) {
                    float4 v4 = sS_vec4[vc];
                    thread_max = fmaxf(thread_max, fmaxf(fmaxf(v4.x, v4.y), fmaxf(v4.z, v4.w)));
                }
            }

            // 1.1 warp reduction
            #pragma unroll
            for (int o = THREADS_PER_ROW / 2; o > 0; o >>= 1)
                thread_max = fmaxf(thread_max, __shfl_down_sync(mask, thread_max, o, THREADS_PER_ROW));

            const float row_max  = __shfl_sync(mask, thread_max, row_leader, THREADS_PER_ROW);
            const float old_max  = sRowMax[row];
            const float new_max  = fmaxf(old_max, row_max);
            const float exp_diff = __expf(old_max - new_max);

            // Phase 2: Unified vectorized + tail processing
            float thread_sum = 0.0f;
            __half2 half_buffer[20];
            int vc_base = thread_in_row;
            int h2_idx = 0;

            #pragma unroll 4
            for (int j = 0; j < vecs_per_thread; ++j, vc_base += THREADS_PER_ROW) {
                if (vc_base < vec_cols) {
                    float4 v4 = sS_vec4[vc_base];

                    float e0 = __expf(fmaxf(v4.x - new_max, -80.0f));
                    float e1 = __expf(fmaxf(v4.y - new_max, -80.0f));
                    float e2 = __expf(fmaxf(v4.z - new_max, -80.0f));
                    float e3 = __expf(fmaxf(v4.w - new_max, -80.0f));

                    thread_sum += (e0 + e1) + (e2 + e3);

                    half_buffer[h2_idx++] = __float22half2_rn(make_float2(e0, e1));
                    half_buffer[h2_idx++] = __float22half2_rn(make_float2(e2, e3));
                }
            }

            #pragma unroll 4
            for (int c = tail_start + thread_in_row; c < BLOCK_N; c += THREADS_PER_ROW) {
                float v = (c < valid_kv_rows) ? sS_row_f[c] : NEG_INF;
                float e = __expf(fmaxf(v - new_max, -80.0f));
                thread_sum += (c < valid_kv_rows) ? e : 0.0f;
                sP_row_h[c] = (c < valid_kv_rows) ? __float2half_rn(e) : __float2half(0.f);
            }

            #pragma unroll
            for (int o = THREADS_PER_ROW / 2; o > 0; o >>= 1)
                thread_sum += __shfl_down_sync(mask, thread_sum, o, THREADS_PER_ROW);

            float row_sum = __shfl_sync(mask, thread_sum, row_leader, THREADS_PER_ROW);

            if (thread_in_row == 0) {
                sRowSum[row] = exp_diff * sRowSum[row] + row_sum;
                sRowMax[row] = new_max;
            }

            // Phase 3: Vectorized writes
            h2_idx = 0;
            vc_base = thread_in_row;
            __half2* sP_half2 = reinterpret_cast<__half2*>(sP_row_h);

            #pragma unroll 4
            for (int j = 0; j < vecs_per_thread; ++j, vc_base += THREADS_PER_ROW) {
                if (vc_base < vec_cols) {
                    int base_offset = vc_base * 2;

                    sP_half2[base_offset]     = half_buffer[h2_idx++];
                    sP_half2[base_offset + 1] = half_buffer[h2_idx++];
                }
            }

            // Fused sO scaling
            if (block > 0) {
                float*  sO_row = sO + row * D_STRIDE;
                float4* sO_vec = reinterpret_cast<float4*>(sO_row);
                const int o_vec_count = (D_STRIDE + 3) >> 2;
                float scale = exp_diff;

                #pragma unroll 4
                for (int ov = thread_in_row; ov < o_vec_count; ov += THREADS_PER_ROW) {
                    float4 v = sO_vec[ov];
                    v.x *= scale;
                    v.y *= scale;
                    v.z *= scale;
                    v.w *= scale;

                    sO_vec[ov] = v;
                }
            }
        }
        __syncthreads();

        // ==================================================================================
        // Load:     V tile from global to sV(reuse) shared memory
        // Layout:   V: global[row: BLOCK_N, D] -> shared[row: BLOCK_N, D_STRIDE]
        // Template: SRC_STRIDE=D, DST_STRIDE=D_STRIDE
        // ==================================================================================
        const uint4* v_vec     = reinterpret_cast<const uint4*>(v_ptr + start_kv * D);
              uint4* sV_vec    = reinterpret_cast<uint4*>(sV);

        LOAD_TILE<D, D_STRIDE>(v_vec, sV_vec, valid_kv_rows, tid, THREADS_PER_BLOCK);

        __syncthreads();

        // ==================================================================================
        // Compute P @ V
        // ==================================================================================
        const int num_tiles_m_pv    = (BLOCK_M + WMMA_M - 1) / WMMA_M;   // P @ V: along M
        const int num_tiles_n_pv    = (D + WMMA_N - 1) / WMMA_N;         // P @ V: along D
        const int num_tiles_k_pv    = (BLOCK_N + WMMA_K - 1) / WMMA_K;   // P @ V: inner along N
        const int total_tiles_pv    = num_tiles_m_pv * num_tiles_n_pv;
        const int tiles_per_warp_pv = (total_tiles_pv + WARPS_PER_BLOCK - 1) / WARPS_PER_BLOCK;

        for (int tile_idx = 0; tile_idx < tiles_per_warp_pv; ++tile_idx) {
            const int global_tile_idx = warp_id * tiles_per_warp_pv + tile_idx;
            if (global_tile_idx >= total_tiles_pv) break;

            const int tile_m_idx = global_tile_idx / num_tiles_n_pv;
            const int tile_d_idx = global_tile_idx % num_tiles_n_pv;

            const int tile_m = tile_m_idx * WMMA_M;
            const int tile_d = tile_d_idx * WMMA_N;

            if (tile_m >= valid_q_rows) continue;

            fragment<matrix_a, WMMA_M, WMMA_N, WMMA_K, half, row_major> a_frag;
            fragment<matrix_b, WMMA_M, WMMA_N, WMMA_K, half, row_major> b_frag;
            fragment<accumulator, WMMA_M, WMMA_N, WMMA_K, float> acc_frag;

            load_matrix_sync(acc_frag, sO + tile_m * D_STRIDE + tile_d, D_STRIDE, mem_row_major);

            #pragma unroll
            for (int tile_k = 0; tile_k < num_tiles_k_pv; ++tile_k) {
                const int k_offset = tile_k * WMMA_K;
                if (k_offset >= valid_kv_rows) break;

                load_matrix_sync(a_frag, sP + tile_m * N_STRIDE + k_offset, N_STRIDE);
                load_matrix_sync(b_frag, sV + k_offset * D_STRIDE + tile_d, D_STRIDE);
                mma_sync(acc_frag, a_frag, b_frag, acc_frag);
            }
            store_matrix_sync(sO + tile_m * D_STRIDE + tile_d, acc_frag, D_STRIDE, mem_row_major);
        }
        __syncthreads();
    }

    // ==================================================================================
    // Store final sO to global memory
    // ==================================================================================
    const int total_fp16_x4 = (valid_q_rows * D) / 4;

    for (int i = tid; i < total_fp16_x4; i += THREADS_PER_BLOCK) {
        const int row = i / (D / 4);
        const int col = (i % (D / 4)) * 4;

        const float sum_clamped = fmaxf(sRowSum[row], 1e-24f);
        const float inv_sum = 1.0f / sum_clamped;
        const float* sO_row = sO + row * D_STRIDE;

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
        const float sum = fmaxf(sRowSum[tid], 1e-24f);
        softmax_lse_ptr[tid] = sRowMax[tid] + logf(sum);
    }
}

// ======================================================================================
// LAUNCHER
// ======================================================================================
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

    TORCH_CHECK(smem <= MAX_SMEM_PER_SM, "Shared memory exceeds 96KB for Forward kernel: ", smem, " bytes (", smem / 1024, " KB)");

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

// ======================================================================================
// WRAPPER
// ======================================================================================
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
