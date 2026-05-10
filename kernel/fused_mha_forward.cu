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
#include "forward.h"
#include "gemm_smem.h"
#include "mat_mul.h"
#include "softmax.h"
#include "dropout.h"

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
    int            window_left,
    int            window_right,
    const float    p_dropout,
    const uint64_t dropout_seed,
    const uint64_t dropout_offset
) {
    using Config = KernelConfig<D>;

    constexpr int BLOCK_M  = Config::DO::BLOCK_M;
    constexpr int BLOCK_N  = Config::DO::BLOCK_N;
    constexpr int D_STRIDE = Config::DO::D_STRIDE;
    constexpr int N_STRIDE = Config::DO::N_STRIDE;

    // ==================================================================================
    // Grid Mapping: X for Q-blocks, Z for batch-head composite (batch * H_Q + head).
    // ==================================================================================
    const int bthd_idx     = blockIdx.z;
    const int block_idx    = blockIdx.x;

    if (bthd_idx >= B * H_Q) return;

    // ======================================================================================
    // BlockInfo: Metadata resolution (Dense Q-centric)
    // ======================================================================================
    BlockInfo<IS_CAUSAL, IS_WINDOW, false> block;
    block.init_q(
        block_idx,       // BLOCK_IDX:      Current Q-block index (grid.x)
        bthd_idx,        // BATCH_HEAD_ID:  Global Q-head index (batch * H_Q + head_q)
        H_Q,             // H_Q:            Number of query heads
        H_K,             // H_K:            Number of KV heads
        M,               // M:              Query sequence length
        N,               // N:              KV sequence length
        0,               // B:              Batch size (0 for dense, unused)
        BLOCK_M,         // BLOCK_M:        Tile size along Q dimension
        BLOCK_N,         // BLOCK_N:        Tile size along KV dimension
        window_left,     // WINDOW_LEFT:    Left sliding window bound (-1 if disabled)
        window_right,    // WINDOW_RIGHT:   Right sliding window bound (-1 if disabled)
        nullptr,         // CU_SEQLENS_Q:   Cumulative Q lengths (nullptr for dense)
        nullptr,         // CU_SEQLENS_K:   Cumulative KV lengths (nullptr for dense)
        nullptr          // SEQUSED_K:      Actual KV lengths override (nullptr for dense)
    );

    if (block.start_q >= M) return;

    // ==================================================================================
    // Init:   thread/warp/lane IDs for WMMA coordination
    // ==================================================================================
    const int tid     = threadIdx.x;
    const int warp_id = tid >> 5;
    const int lane_id = tid & 31;
    // Alibi slope only for valid block
    const float alibi_slope = (alibi_slopes) ? alibi_slopes[bthd_idx % H_Q] : 0.0f;

    // ==================================================================================
    // Layout:
    //   Q/Out/LSE: [B, H_Q, M, D] offset follows bthd_idx (Q-head space)
    //   K/V:       [B, H_K, N, D] mapped via bthd_idx % H_Q / (H_Q / H_K)
    // ==================================================================================
    const __half* __restrict__ q_ptr           = Q           + block.q_offset  (D, H_Q, M);
    const __half* __restrict__ k_ptr           = K           + block.kv_offset (D, H_K, N);
    const __half* __restrict__ v_ptr           = V           + block.kv_offset (D, H_K, N);
          __half* __restrict__ out_ptr         = Out         + block.q_offset  (D, H_Q, M);
           float* __restrict__ softmax_lse_ptr = softmax_lse + block.lse_offset(H_Q, M);
          __half* __restrict__ dmask_ptr       = (dmask != nullptr) ? dmask + block.dmask_offset(H_Q, M, N) : nullptr;

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
        sRowSum[tid] = 0.0f;
    }

    // ==================================================================================
    // Load:     Q tile from global to sQ shared memory
    // Layout:   Q: global[row: BLOCK_M, D] -> shared[row: BLOCK_M, D_STRIDE]
    // Template: DUAL_LOAD=false, SRC_STRIDE=D, DST_STRIDE=D_STRIDE
    // ==================================================================================
    WMMA_GEMM_LOAD_TILE<Config, false, D_STRIDE>(
      q_ptr,   sQ,
      nullptr, nullptr,
      D, block.valid_q_rows, tid);
    __syncthreads();

    // ==================================================================================
    // MAIN LOOP (iterates over K/V blocks for current Q block)
    // ==================================================================================
    for (int block_q = block.block_min; block_q < block.block_max; ++block_q) {
        const int start_kv      = block_q * BLOCK_N;
        const int valid_kv_rows = min(BLOCK_N, N - start_kv);

        // ==================================================================================
        // Load:     K tile from global to sK(reuse) shared memory
        // Layout:   K: global[row: BLOCK_N, D] -> shared[row: BLOCK_N, D_STRIDE]
        // Template: DUAL_LOAD=false, SRC_STRIDE=D, DST_STRIDE=D_STRIDE
        // ==================================================================================
        WMMA_GEMM_LOAD_TILE<Config, false, D_STRIDE>(
          k_ptr + start_kv * D, sK,
          nullptr, nullptr,
          D, valid_kv_rows, tid);
        __syncthreads();

        // ==================================================================================
        // Compute:  S = Q @ K^T
        // Layout:   Q[row: BLOCK_M, D], K[col: BLOCK_N, D] -> S[row: BLOCK_M, col: BLOCK_N]
        // Template: BLOCK_X=BLOCK_M, BLOCK_Y=BLOCK_N
        // ==================================================================================
        WMMA_GEMM_SCORES<Config, GemmType::sQ_KT, D, IS_CAUSAL, IS_ALIBI, IS_SOFTCAP, IS_WINDOW, BLOCK_M, BLOCK_N, D_STRIDE, N_STRIDE>(
          sQ, sK, sS,
          block.valid_q_rows,  valid_kv_rows,
          block.start_q,       start_kv,
          block.seqlen_offset,
          softmax_scale, softcap, alibi_slope, window_left, window_right, warp_id, lane_id);
        __syncthreads();

        // ==================================================================================
        // Compute:  Online Softmax + O-scaling
        // Layout:   S[BLOCK_M, BLOCK_N] -> P[BLOCK_M, BLOCK_N], O[BLOCK_M, D] scaled
        // Template: BLOCK_M, BLOCK_N, N_STRIDE, D_STRIDE
        // ==================================================================================
        WMMA_GEMM_SOFTMAX<Config, BLOCK_M, BLOCK_N, N_STRIDE, D_STRIDE>(
          sS, sP, sO,
          sRowMax, sRowSum,
          block.valid_q_rows, valid_kv_rows, tid, block_q);
        __syncthreads();

        // ==================================================================================
        // Compute:  Apply dropout to P tile and generate dropout mask
        // Layout:   P[BLOCK_M, BLOCK_N] (in SMEM) -> P_masked (in SMEM) Optionally store mask to dmask[BLOCK_M, BLOCK_N] in GMEM
        // Template: BLOCK_M, BLOCK_N, N_STRIDE, IS_DROPOUT
        // ==================================================================================
        WMMA_GEMM_DROPOUT<Config, BLOCK_M, BLOCK_N, N_STRIDE, IS_DROPOUT>(
          sP, dmask_ptr ? dmask_ptr + block.start_q * N + start_kv : nullptr,
          block.valid_q_rows, valid_kv_rows,
          block.start_q, start_kv, N,
          p_dropout, dropout_seed, dropout_offset, tid);
        __syncthreads();

        // ==================================================================================
        // Load:     V tile from global to sV(reuse) shared memory
        // Layout:   V: global[row: BLOCK_N, D] -> shared[row: BLOCK_N, D_STRIDE]
        // Template: DUAL_LOAD=false, SRC_STRIDE=D, DST_STRIDE=D_STRIDE
        // ==================================================================================
        WMMA_GEMM_LOAD_TILE<Config, false, D_STRIDE>(
          v_ptr + start_kv * D, sV,
          nullptr, nullptr,
          D, valid_kv_rows, tid);
        __syncthreads();

        // ==================================================================================
        // Compute:  dO += P @ V
        // Layout:   P[row: BLOCK_M, BLOCK_N], V[row: BLOCK_N, D] -> dO[row: BLOCK_M, D]
        // Template: BLOCK_X=BLOCK_M, BLOCK_Y=BLOCK_N
        // ==================================================================================
        WMMA_GEMM_GRADIENTS<Config, GemmType::dO_PV, D, BLOCK_M, BLOCK_N, N_STRIDE, D_STRIDE>(
          sP, sV, sO,
          block.valid_q_rows, valid_kv_rows, warp_id, lane_id);
        __syncthreads();
    }   // END MAIN LOOP
    // ==================================================================================
    // Compute:  Store normalized attention output O = softmax(S) @ V
    // Layout:   sO[valid_q_rows, D_STRIDE] -> out_ptr[valid_q_rows, D]
    // Template  D, D_STRIDE  : Head dimension and shared memory stride
    // ==================================================================================
    WMMA_GEMM_EPILOGUE<Config, GemmType::write_dO, D_STRIDE>(
      sO,      out_ptr,
      nullptr, nullptr,
      sRowSum, D, block.valid_q_rows, tid);

    if (tid < block.valid_q_rows) {
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
    float        softmax_scale,
    bool         is_causal,
    float        softcap,
    float        p_dropout,
    const float* alibi_slopes,
    int          window_left,
    int          window_right,
    uint64_t     dropout_seed,
    uint64_t     dropout_offset,
    cudaStream_t stream
) {
    using Config = KernelConfig<D>;

    const size_t smem = Config::TOTAL_SMEM;
    TORCH_CHECK(smem <= MAX_SMEM_PER_SM, "Shared memory exceeds 96KB for Forward kernel: ", smem, " bytes (", smem / 1024, " KB)");

    const int B   = Q.size(0);
    const int H_Q = Q.size(1);
    const int H_K = K.size(1);
    const int M   = Q.size(2);
    const int N   = K.size(2);

    const dim3 grid((M + Config::DO::BLOCK_M - 1) / Config::DO::BLOCK_M, 1, B * H_Q);
    const dim3 block(Config::THREADS_PER_BLOCK);

    bool is_alibi   = (alibi_slopes != nullptr);
    bool is_softcap = (softcap > 0.0f);
    bool is_window  = (window_left >= 0 || window_right >= 0);
    bool is_dropout = (p_dropout > 0.0f);

    const __half* q_ptr     = reinterpret_cast<const __half*>(Q.data_ptr());
    const __half* k_ptr     = reinterpret_cast<const __half*>(K.data_ptr());
    const __half* v_ptr     = reinterpret_cast<const __half*>(V.data_ptr());
          __half* out_ptr   = reinterpret_cast<__half*>(Out.data_ptr());
           float* lse_ptr   = softmax_lse.data_ptr<float>();
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
            q_ptr, k_ptr, v_ptr, out_ptr, lse_ptr, dmask_ptr,
            B, H_Q, H_K, M, N,
            softmax_scale, softcap, alibi_slopes, window_left, window_right,
            p_dropout, dropout_seed, dropout_offset
        );
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
    const float  p_dropout,
    const float  softmax_scale,
    bool         is_causal,
    int          window_left,
    int          window_right,
    const float  softcap,
    const bool   return_softmax,
    std::optional<at::Generator> gen
) {
    // Device guard for multi-GPU / pipeline-parallelism
    at::cuda::CUDAGuard device_guard{q.device()};

    auto props  = at::cuda::getCurrentDeviceProperties();
    TORCH_CHECK(props->major == 7 && props->minor == 0, "Kernel supports only Volta GPUs.");
    TORCH_CHECK(q.dtype() == torch::kFloat16, "q must be fp16");
    TORCH_CHECK(k.dtype() == torch::kFloat16, "k must be fp16");
    TORCH_CHECK(v.dtype() == torch::kFloat16, "v must be fp16");
    TORCH_CHECK(q.is_cuda() && k.is_cuda() && v.is_cuda(), "q, k, v must be on CUDA");
    TORCH_CHECK(q.stride(-1) == 1 && k.stride(-1) == 1 && v.stride(-1) == 1, "Last dim of q, k, v must be contiguous");

    const auto sizes = q.sizes();
    const int B      = sizes[0];
    const int H_Q    = sizes[1];
    const int M      = sizes[2];
    const int D      = sizes[3];
    const int H_K    = k.size(1);
    const int N      = k.size(2);

    TORCH_CHECK(B > 0, "batch_size must be positive");
    TORCH_CHECK(D <= 256 && D % 8 == 0 && D % 2 == 0, "D must be even, <=256, multiple of 8");
    TORCH_CHECK(H_Q % H_K == 0, "H_Q must be divisible by H_K for GQA/MQA");

    // Window edge cases
    if (window_left  >= N) { window_left  = -1; }
    if (window_right >= N) { window_right = -1; }
    if (M == 1 && !alibi_slopes.has_value()) { is_causal = false; }

    // Alibi
    const float* alibi = nullptr;
    if (alibi_slopes.has_value()) {
        const auto& slopes = alibi_slopes.value();
        auto sizes = slopes.sizes();
        TORCH_CHECK(slopes.dtype() == torch::kFloat32, "alibi_slopes must be fp32");
        TORCH_CHECK(slopes.is_cuda(), "alibi_slopes must be on CUDA");
        TORCH_CHECK(slopes.stride(-1) == 1, "alibi_slopes last dim must be contiguous");
        bool valid_shape = (sizes.size() == 1 && sizes[0] == H_Q) ||
                           (sizes.size() == 2 && sizes[0] == B && sizes[1] == H_Q);
        TORCH_CHECK(valid_shape, "alibi_slopes shape must be [H_Q] or [B, H_Q], got ", sizes);
        alibi = slopes.data_ptr<float>();
    }

    // Dropout
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
    auto softmax_lse = torch::empty({B, H_Q, M}, torch::dtype(torch::kFloat32).device(q.device()));
    TORCH_CHECK(out_fp16.dtype() == torch::kFloat16, "out must be fp16");
    TORCH_CHECK(softmax_lse.dtype() == torch::kFloat32, "softmax_lse must be fp32");

    at::Tensor dmask;
    if (return_softmax && (p_dropout > 0.0f)) {
        dmask = torch::empty({B, H_Q, M, N}, q.options());
    } else {
        dmask = torch::empty({0}, q.options());
    }

    // Edge-case return
    if (N == 0) {
        out_fp16.zero_();
        softmax_lse.fill_(-std::numeric_limits<float>::infinity());
        return {out_fp16, softmax_lse, dmask, rng_state};
    }

    // Run kernel
    auto stream = at::cuda::getCurrentCUDAStream().stream();

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
