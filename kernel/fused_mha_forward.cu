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
#include "01_forward_config.cuh"
#include "02_wmma.cuh"

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

    WMMA_GEMM_INIT_SMEM<Config>(smem_raw);

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
    // Template: DUAL_LOAD=false, SRC_STRIDE=D, DST_STRIDE=D_STRIDE
    // ==================================================================================
    WMMA_GEMM_LOAD_TILE<false, D, D_STRIDE>(
    q_ptr,   sQ,
    nullptr, nullptr,
    valid_q_rows, tid,
    THREADS_PER_BLOCK);

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
        // Template: DUAL_LOAD=false, SRC_STRIDE=D, DST_STRIDE=D_STRIDE
        // ==================================================================================
        WMMA_GEMM_LOAD_TILE<false, D, D_STRIDE>(
        k_ptr + start_kv * D, sK,
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
        // Compute:  Online Softmax + O-scaling
        // Layout:   S[BLOCK_M, BLOCK_N] -> P[BLOCK_M, BLOCK_N], O[BLOCK_M, D] scaled
        // Template: BLOCK_M, BLOCK_N, N_STRIDE, D_STRIDE, THREADS_PER_ROW, FULL_ROWS
        // ==================================================================================
        ONLINE_SOFTMAX<BLOCK_M, BLOCK_N, N_STRIDE, D_STRIDE, THREADS_PER_ROW>(
        sS, sP, sO,
        sRowMax, sRowSum,
        valid_q_rows, valid_kv_rows,
        tid, block);

        __syncthreads();

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
        // Compute:  dO += P @ V
        // Layout:   P[row: BLOCK_M, BLOCK_N], V[row: BLOCK_N, D] -> dO[row: BLOCK_M, D]
        // Template: BLOCK_X=BLOCK_M, BLOCK_Y=BLOCK_N
        // ==================================================================================
        WMMA_GEMM_GRADIENTS<GemmType::dO_PV, D, BLOCK_M, BLOCK_N, N_STRIDE, D_STRIDE, WARPS_PER_BLOCK>(
        sP, sV, sO,
        valid_q_rows, valid_kv_rows,
        warp_id,      lane_id);

        __syncthreads();

    }   // END MAIN LOOP

    // ==================================================================================
    // Compute:  Store normalized attention output O = softmax(S) @ V
    // Layout:   sO[valid_q_rows, D_STRIDE] -> out_ptr[valid_q_rows, D]
    // Template  D, D_STRIDE  : Head dimension and shared memory stride
    // ==================================================================================
    WMMA_GEMM_EPILOGUE<GemmType::write_dO, D, D_STRIDE>(
    sO,      out_ptr,
    nullptr, nullptr,
    sRowSum,
    valid_q_rows, tid,
    THREADS_PER_BLOCK);

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
