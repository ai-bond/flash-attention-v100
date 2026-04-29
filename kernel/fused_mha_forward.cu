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

#include "debug.h"
#include "kernel.h"
#include "forward.h"
#include "gemm_smem.h"
#include "mat_mul.h"
#include "softmax.h"
#include "dropout.h"
#include "template.h"

// ======================================================================================
// FORWARD KERNEL
// ======================================================================================
template<int D, bool IS_CAUSAL, bool IS_ALIBI, bool IS_SOFTCAP, bool IS_WINDOW, bool IS_DROPOUT>
__global__ void __launch_bounds__(KernelConfig<D>::THREADS_PER_BLOCK, 2)
flash_attention_forward_kernel(
    const __half* __restrict__ Q,
    const __half* __restrict__ K,
    const __half* __restrict__ V,
          __half* __restrict__ Out,
           float* __restrict__ softmax_lse,
          __half* __restrict__ dmask,
    const int B,
    const int H_Q,
    const int H_K,
    const int M,
    const int N,
    const float    softmax_scale,
    const float    softcap,
    const float*   alibi_slopes,
    int window_left,
    int window_right,
    const float    p_dropout,
    const uint64_t dropout_seed,
    const uint64_t dropout_offset
) {
    using Config = KernelConfig<D>;
    constexpr int BLOCK_M   = Config::DO::BLOCK_M;
    constexpr int BLOCK_N   = Config::DO::BLOCK_N;
    constexpr int D_STRIDE  = Config::DO::D_STRIDE;
    constexpr int N_STRIDE  = Config::DO::N_STRIDE;

    const int batch_head_id = blockIdx.z;
    if (batch_head_id >= B * H_Q) return;

    const float alibi_slope = (alibi_slopes) ? alibi_slopes[batch_head_id % H_Q] : 0.0f;

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
    // Layout:
    //   Q/Out/LSE: [B, H_Q, M, D] offset follows batch_head_id (Q-head space)
    //   K/V:       [B, H_K, N, D] mapped via batch_head_id % H_Q / (H_Q / H_K)
    // ==================================================================================
    const __half* __restrict__ q_ptr           = Q +           (size_t)batch_head_id * M * D + start_q * D;
    const __half* __restrict__ k_ptr           = K +           (size_t)((batch_head_id / H_Q) * H_K + (batch_head_id % H_Q) / (H_Q / H_K)) * N * D;
    const __half* __restrict__ v_ptr           = V +           (size_t)((batch_head_id / H_Q) * H_K + (batch_head_id % H_Q) / (H_Q / H_K)) * N * D;
          __half* __restrict__ out_ptr         = Out +         (size_t)batch_head_id * M * D + start_q * D;
           float* __restrict__ softmax_lse_ptr = softmax_lse + (size_t)batch_head_id * M + start_q;
          __half* __restrict__ dmask_ptr       = (dmask != nullptr) ? dmask + (size_t)batch_head_id * M * N : nullptr;

    // ==================================================================================
    // Init:   shared memory with zero-fill union regions to avoid stale data
    // ==================================================================================
    extern __shared__ char smem_raw[];

    WMMA_GEMM_INIT_SMEM<Config>(smem_raw);

    __syncthreads();

    auto& smem = *reinterpret_cast<typename Config::SmemLayout*>(smem_raw);

    __half* __restrict__ sQ      = smem.phase.fdo.q;
    __half* __restrict__ sK      = smem.phase.fdo.reuse_kv.k;
    __half* __restrict__ sV      = smem.phase.fdo.reuse_kv.v;
    float*  __restrict__ sS      = smem.phase.fdo.reuse_sp.s;
    __half* __restrict__ sP      = smem.phase.fdo.reuse_sp.p;
    float*  __restrict__ sRowMax = smem.row_max;
    float*  __restrict__ sRowSum = smem.row_sum;
    float*  __restrict__ sO      = smem.phase.fdo.o;

    if (tid < BLOCK_M) {
        sRowMax[tid] = NEG_INF;
    }

    // ==================================================================================
    // Load:     Q tile from global to sQ shared memory
    // Layout:   Q: global[row: BLOCK_M, D] -> shared[row: BLOCK_M, D_STRIDE]
    // Template: DUAL_LOAD=false, SRC_STRIDE=D, DST_STRIDE=D_STRIDE
    // ==================================================================================
    WMMA_GEMM_LOAD_TILE<Config, false, D, D_STRIDE>(
    q_ptr,   sQ,
    nullptr, nullptr,
    valid_q_rows, tid);

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
        WMMA_GEMM_LOAD_TILE<Config, false, D, D_STRIDE>(
        k_ptr + start_kv * D, sK,
        nullptr, nullptr,
        valid_kv_rows, tid);

        __syncthreads();

        // ==================================================================================
        // Compute:  S = Q @ K^T
        // Layout:   Q[row: BLOCK_M, D], K[col: BLOCK_N, D] -> S[row: BLOCK_M, col: BLOCK_N]
        // Template: BLOCK_X=BLOCK_M, BLOCK_Y=BLOCK_N
        // ==================================================================================
        WMMA_GEMM_SCORES<Config, GemmType::sQ_KT, D, IS_CAUSAL, IS_ALIBI, IS_SOFTCAP, IS_WINDOW, BLOCK_M, BLOCK_N, D_STRIDE, N_STRIDE>(
        sQ, sK, sS,
        valid_q_rows,  valid_kv_rows,
        start_q,       start_kv,
        softmax_scale, softcap, alibi_slope, window_left, window_right,
        warp_id,       lane_id);

        __syncthreads();

        // ==================================================================================
        // Compute:  Online Softmax + O-scaling
        // Layout:   S[BLOCK_M, BLOCK_N] -> P[BLOCK_M, BLOCK_N], O[BLOCK_M, D] scaled
        // Template: BLOCK_M, BLOCK_N, N_STRIDE, D_STRIDE
        // ==================================================================================
        WMMA_GEMM_SOFTMAX<Config, BLOCK_M, BLOCK_N, N_STRIDE, D_STRIDE>(
        sS, sP, sO,
        sRowMax, sRowSum,
        valid_q_rows, valid_kv_rows,
        tid, block);

        __syncthreads();

        // ==================================================================================
        // Compute:  Dropout mask to P tile
        // Layout:   P[BLOCK_M, BLOCK_N] masked by dmask[start_q:start_q+BLOCK_M, start_kv:start_kv+BLOCK_N]
        // Template: BLOCK_M, BLOCK_N, N_STRIDE, IS_DROPOUT
        // ==================================================================================
        WMMA_GEMM_DROPOUT<Config, BLOCK_M, BLOCK_N, N_STRIDE, IS_DROPOUT>(
        sP, dmask_ptr ? dmask_ptr + start_q * N + start_kv : nullptr,
        valid_q_rows, valid_kv_rows,
        start_q, start_kv, N,
        p_dropout, dropout_seed, dropout_offset, tid);

        __syncthreads();

        // ==================================================================================
        // Load:     V tile from global to sV(reuse) shared memory
        // Layout:   V: global[row: BLOCK_N, D] -> shared[row: BLOCK_N, D_STRIDE]
        // Template: DUAL_LOAD=false, SRC_STRIDE=D, DST_STRIDE=D_STRIDE
        // ==================================================================================
        WMMA_GEMM_LOAD_TILE<Config, false, D, D_STRIDE>(
        v_ptr + start_kv * D, sV,
        nullptr, nullptr,
        valid_kv_rows, tid);

        __syncthreads();

        // ==================================================================================
        // Compute:  dO += P @ V
        // Layout:   P[row: BLOCK_M, BLOCK_N], V[row: BLOCK_N, D] -> dO[row: BLOCK_M, D]
        // Template: BLOCK_X=BLOCK_M, BLOCK_Y=BLOCK_N
        // ==================================================================================
        WMMA_GEMM_GRADIENTS<Config, GemmType::dO_PV, D, BLOCK_M, BLOCK_N, N_STRIDE, D_STRIDE>(
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
    WMMA_GEMM_EPILOGUE<Config, GemmType::write_dO, D, D_STRIDE>(
    sO,      out_ptr,
    nullptr, nullptr,
    sRowSum,
    valid_q_rows, tid);

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
    const torch::Tensor& dmask,
    float softmax_scale,
    bool is_causal,
    float softcap,
    float p_dropout,
    const float* alibi_slopes,
    int window_left,
    int window_right,
    uint64_t dropout_seed,
    uint64_t dropout_offset,
    cudaStream_t stream
) {
    using Config = KernelConfig<D>;

    const int B   = Q.size(0);
    const int H_Q = Q.size(1);
    const int H_K = K.size(1);
    const int M   = Q.size(2);
    const int N   = K.size(2);

    const dim3 grid((M + Config::DO::BLOCK_M - 1) / Config::DO::BLOCK_M, 1, B * H_Q);
    const dim3 block(Config::THREADS_PER_BLOCK);
    const size_t smem = Config::TOTAL_SMEM;

    TORCH_CHECK(smem <= MAX_SMEM_PER_SM, "Shared memory exceeds 96KB for Forward kernel: ", smem, " bytes (", smem / 1024, " KB)");

    bool is_alibi   = (alibi_slopes != nullptr);
    bool is_softcap = (softcap > 0.0f);
    bool is_window  = (window_left >= 0 || window_right >= 0);
    bool is_dropout = (p_dropout > 0.0f);

    __half* dmask_ptr = dmask.numel() > 0 ? reinterpret_cast<__half*>(dmask.data_ptr()) : nullptr;

    dispatch_attention_features(is_causal, is_alibi, is_softcap, is_window, is_dropout,
        [&](auto CAUSAL, auto ALIBI, auto SOFTCAP, auto WINDOW, auto DROPOUT) {
            constexpr bool IS_CAUSAL  = decltype(CAUSAL)::value;
            constexpr bool IS_ALIBI   = decltype(ALIBI)::value;
            constexpr bool IS_SOFTCAP = decltype(SOFTCAP)::value;
            constexpr bool IS_WINDOW  = decltype(WINDOW)::value;
            constexpr bool IS_DROPOUT = decltype(DROPOUT)::value;

            auto kernel = flash_attention_forward_kernel<D, IS_CAUSAL, IS_ALIBI, IS_SOFTCAP, IS_WINDOW, IS_DROPOUT>;
            cudaFuncSetAttribute(kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, smem);

            kernel<<<grid, block, smem, stream>>>(
                reinterpret_cast<const __half*>(Q.data_ptr()),
                reinterpret_cast<const __half*>(K.data_ptr()),
                reinterpret_cast<const __half*>(V.data_ptr()),
                reinterpret_cast<__half*>(Out.data_ptr()),
                softmax_lse.data_ptr<float>(), dmask_ptr,
                B, H_Q, H_K, M, N,
                softmax_scale, softcap, alibi_slopes, window_left, window_right,
                p_dropout, dropout_seed, dropout_offset);
        });
}

// ======================================================================================
// WRAPPER
// ======================================================================================
std::vector<at::Tensor> flash_attention_forward(
    at::Tensor& q,
    const at::Tensor& k,
    const at::Tensor& v,
    std::optional<at::Tensor>& out,
    std::optional<at::Tensor>& alibi_slopes,
    const float p_dropout,
    const float softmax_scale,
    bool is_causal,
    int window_left,
    int window_right,
    const float softcap,
    const bool return_softmax,
    std::optional<at::Generator> gen
) {
    // Check layouts
    TORCH_CHECK(q.dtype() == torch::kFloat16, "q must be fp16");
    TORCH_CHECK(k.dtype() == torch::kFloat16, "k must be fp16");
    TORCH_CHECK(v.dtype() == torch::kFloat16, "v must be fp16");
    TORCH_CHECK(q.is_cuda() && k.is_cuda() && v.is_cuda(), "Tensors must be on CUDA");
    TORCH_CHECK(q.stride(-1) == 1 && k.stride(-1) == 1 && v.stride(-1) == 1, "Last dim must be contiguous");

    const auto sizes = q.sizes();
    const int B      = sizes[0], H_Q = sizes[1], M = sizes[2], D = sizes[3];
    const int H_K    = k.size(1);
    const int N      = k.size(2);
    TORCH_CHECK(D <= 256 && D % 8 == 0 && D % 2 == 0, "D must be even, <=256, multiple of 8");
    TORCH_CHECK(H_Q % H_K == 0, "H_Q must be divisible by H_K for GQA/MQA");

    const float* alibi = nullptr;
    if (alibi_slopes.has_value()) {
        const auto& slopes = alibi_slopes.value();
        auto sizes = slopes.sizes();
        TORCH_CHECK(slopes.dtype() == torch::kFloat32, "alibi_slopes must be fp32");
        TORCH_CHECK(slopes.is_cuda(), "alibi_slopes must be on CUDA");
        TORCH_CHECK(slopes.stride(-1) == 1, "alibi_slopes last dim must be contiguous");
        TORCH_CHECK(slopes.sizes() == torch::IntArrayRef({H_Q}) || slopes.sizes() == torch::IntArrayRef({B, H_Q}), "alibi_slopes shape must be [H_Q] or [B, H_Q]");
        alibi = slopes.data_ptr<float>();
    }

    TORCH_CHECK(p_dropout >= 0.f && p_dropout < 1.f, "p_dropout must be in [0, 1)");
    TORCH_CHECK(!return_softmax || p_dropout > 0.f, "return_softmax requires p_dropout > 0");
    if (softcap > 0.f) { TORCH_CHECK(p_dropout == 0.f, "Softcapping does not support dropout"); }

    uint64_t dropout_seed   = 0;
    uint64_t dropout_offset = 0;

    at::Tensor rng_state;
    if (p_dropout > 0.0f) {
        auto gen_cuda = at::get_generator_or_default<at::CUDAGeneratorImpl>(gen, at::cuda::detail::getDefaultCUDAGenerator());
        std::lock_guard<std::mutex> lock(gen_cuda->mutex_);

        dropout_seed   = gen_cuda->current_seed();
        dropout_offset = gen_cuda->get_offset();

        uint64_t counter_offset = static_cast<uint64_t>(B) * static_cast<uint64_t>(H_Q) * 32ULL;
        gen_cuda->set_offset(dropout_offset + counter_offset);

        rng_state = torch::tensor(
            {static_cast<int64_t>(dropout_seed), static_cast<int64_t>(dropout_offset)},
            torch::dtype(torch::kInt64).device(q.device())
        );
    } else {
        rng_state = torch::empty({2}, torch::dtype(torch::kInt64).device(q.device()));
    }

    // Output tensors
    at::Tensor out_fp16 = out.has_value() ? out.value() : torch::empty_like(q);
    TORCH_CHECK(out_fp16.dtype() == torch::kFloat16, "out must be fp16");

    auto softmax_lse = torch::empty({B, H_Q, M}, torch::dtype(torch::kFloat32).device(q.device()));
    TORCH_CHECK(softmax_lse.dtype() == torch::kFloat32, "softmax_lse must be fp32");

    at::Tensor dmask;
    if (return_softmax && (p_dropout > 0.0f)) {
        dmask = torch::zeros({B, H_Q, M, N}, q.options());
    } else {
        dmask = torch::empty({0}, q.options());
    }

    auto stream = at::cuda::getCurrentCUDAStream().stream();
    auto props  = at::cuda::getCurrentDeviceProperties();
    bool sm70   = props->major == 7 && props->minor == 0;
    TORCH_CHECK(sm70, "Kernel supports only Volta GPUs.");

    #define LAUNCH_KERNEL(DIM) \
        launcher_flash_attention_forward<DIM>(q, k, v, out_fp16, softmax_lse, dmask, softmax_scale, is_causal, \
                                        softcap, p_dropout, alibi, window_left, window_right, dropout_seed, dropout_offset, stream);
    switch (D) {
        case 16:  LAUNCH_KERNEL(16);  break;
        case 32:  LAUNCH_KERNEL(32);  break;
        case 64:  LAUNCH_KERNEL(64);  break;
        case 128: LAUNCH_KERNEL(128); break;
        case 256: LAUNCH_KERNEL(256); break;
        default: TORCH_CHECK(false, "Unsupported D: ", D);
    }
    #undef LAUNCH_KERNEL

    return {out_fp16, softmax_lse, dmask, rng_state};
}
