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
// FORWARD VARLEN KERNEL
// ======================================================================================
template<int D, bool IS_CAUSAL, bool IS_ALIBI, bool IS_SOFTCAP, bool IS_WINDOW, bool IS_DROPOUT, bool IS_PAGED>
__global__ void __launch_bounds__(KernelConfig<D>::THREADS_PER_BLOCK, 2)
flash_attention_forward_varlen_kernel(
    const __half* __restrict__ Q,
    const __half* __restrict__ K,
    const __half* __restrict__ V,
          __half* __restrict__ Out,
           float* __restrict__ softmax_lse,
          __half* __restrict__ dmask,
    const    int* __restrict__ cu_seqlens_q,
    const    int* __restrict__ cu_seqlens_k,
    const    int* __restrict__ seqused_k,
    const    int* __restrict__ leftpad_k,
    const    int* __restrict__ block_table,
    const int      B,
    const int      H_Q,
    const int      H_K,
    const int      T_Q,
    const int      max_seqlen_q,
    const int      max_seqlen_k,
    const int      block_page,
    const int      block_table_stride,
    const int      kv_block_stride,
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
    using Config = KernelConfig<D>;

    constexpr int BLOCK_M  = Config::DO::BLOCK_M;
    constexpr int BLOCK_N  = Config::DO::BLOCK_N;
    constexpr int D_STRIDE = Config::DO::D_STRIDE;
    constexpr int N_STRIDE = Config::DO::N_STRIDE;

    // ======================================================================================
    // Grid Mapping: 1D X for Q-blocks, Z for heads. Batch resolved device-side.
    // ======================================================================================
    const int block_idx    = blockIdx.x;
    const int bthd_idx     = blockIdx.z;

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
        seqused_k        // SEQUSED_K:      Actual KV lengths override
    );

    if (block.start_q >= block.seqlen_q || block.valid_q_rows <= 0) return;

    // ======================================================================================
    // EARLY EXIT: No valid KV blocks to attend to (Causal/Window/Empty KV)
    // Writes deterministic zeros to Out and NEG_INF to LSE for numerical stability.
    // ======================================================================================
    if (block.block_min >= block.block_max || block.seqlen_k == 0) {
        const size_t q_base   = block.q_offset(D, H_Q, T_Q);
        const size_t lse_base = block.lse_offset(H_Q, T_Q);
        for (int row = threadIdx.x; row < block.valid_q_rows; row += blockDim.x) {
            #pragma unroll
            for (int d = 0; d < D; ++d) {
                Out[q_base + row * H_Q * D + d] = __float2half(0.0f);
            }
            softmax_lse[lse_base + row] = NEG_INF;
        }
        return;
    }

    // ======================================================================================
    // Init:   thread/warp/lane IDs for WMMA coordination
    // ======================================================================================
    const int tid       = threadIdx.x;
    const int warp_id   = tid >> 5;
    const int lane_id   = tid & 31;
    // Alibi slope only for valid block + batch
    const int   alibi_idx   = (alibi_batch > 0) ? (block.batch_idx * alibi_batch + block.head_idx) : block.head_idx;
    const float alibi_slope = (alibi_slopes != nullptr) ? alibi_slopes[alibi_idx] : 0.0f;

    // ======================================================================================
    // RAGGED POINTERS
    // Layout:
    //   Q/Out: [T_Q, H_Q, D]            offset follows q_base + start_q (Q, H_Q, D)
    //   LSE:   [H_Q, T_Q]               offset follows bthd_idx * T_Q (H_Q, T_Q)
    //   dmask: [T_Q, H_Q, max_seqlen_k] offset follows q_base + start_q (max_seqlen_k, H_Q, T_Q)
    // ======================================================================================
    const __half* __restrict__ q_ptr           = Q           + block.q_offset  (D, H_Q, T_Q);
          __half* __restrict__ out_ptr         = Out         + block.q_offset  (D, H_Q, T_Q);
           float* __restrict__ softmax_lse_ptr = softmax_lse + block.lse_offset(H_Q, T_Q);
          __half* __restrict__ dmask_ptr       = (dmask != nullptr) ? dmask + block.dmask_offset(H_Q, T_Q, max_seqlen_k) : nullptr;

    // ======================================================================================
    // INIT SHARED MEMORY
    // ======================================================================================
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

    // ======================================================================================
    // Load:     Q tile from global to sQ shared memory
    // Layout:   Q: global[row: BLOCK_M, H_Q * D] -> shared[row: BLOCK_M, D_STRIDE]
    // Template: DUAL_LOAD=false, SMEM_STRIDE=D_STRIDE, GLOBAL_WIDTH=D
    // ======================================================================================
    WMMA_GEMM_LOAD_TILE<Config, false, D_STRIDE, D>(
      q_ptr,   sQ,
      nullptr, nullptr,
      H_Q * D, block.valid_q_rows, tid);
    __syncthreads();

    // ======================================================================================
    // MAIN LOOP (Iterates over logical KV blocks)
    // ======================================================================================
    for (int block_q = block.block_min; block_q < block.block_max; ++block_q) {
        const int start_kv      = block_q * BLOCK_N;
        const int valid_kv_rows = min(BLOCK_N, block.seqlen_k - start_kv);
        if (valid_kv_rows <= 0) break;

        // ======================================================================================
        // KV-CACHE (Paged vs Contiguous)
        // ======================================================================================
        const __half* __restrict__ k_ptr_page;
        const __half* __restrict__ v_ptr_page;

        if constexpr (IS_PAGED) {
            const int page_idx    = start_kv / block_page;
            const int page_offset = start_kv % block_page;
            const int bt_idx      = block.batch_idx * block_table_stride + page_idx;
            const int phys_page   = block_table[bt_idx];
            const size_t kv_load  = static_cast<size_t>(phys_page) * kv_block_stride +
                                    static_cast<size_t>(page_offset) * H_K * D +
                                    static_cast<size_t>(block.kv_head_idx) * D;
            k_ptr_page = K + kv_load;
            v_ptr_page = V + kv_load;
        } else {
            const size_t kv_load  = static_cast<size_t>(block.k_base + start_kv) * H_K * D +
                                    static_cast<size_t>(block.kv_head_idx) * D;
            k_ptr_page = K + kv_load;
            v_ptr_page = V + kv_load;
        }

        // ======================================================================================
        // Load:     K tile from global to sK shared memory
        // Layout:   K: global[row: BLOCK_N, H_K * D] -> shared[row: BLOCK_N, D_STRIDE]
        // Template: DUAL_LOAD=false, SMEM_STRIDE=D_STRIDE, GLOBAL_WIDTH=D (varlen sub-tile)
        // ======================================================================================
        WMMA_GEMM_LOAD_TILE<Config, false, D_STRIDE, D>(
          k_ptr_page, sK,
          nullptr,    nullptr,
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
          softmax_scale, softcap, alibi_slope, window_left, window_right, warp_id, lane_id);
        __syncthreads();

        // ======================================================================================
        // Compute:  Online softmax + O rescaling (block_q > 0)
        // Layout:   sS[valid_q_rows, N_STRIDE] -> sP[valid_q_rows, N_STRIDE]
        //           sO[valid_q_rows, D_STRIDE] *= exp(old_max - new_max)
        // Template: SCORE_STRIDE=N_STRIDE, HEAD_STRIDE=D_STRIDE, TILES=true (varlen tail handling)
        // ======================================================================================
        WMMA_GEMM_SOFTMAX<Config, BLOCK_M, BLOCK_N, N_STRIDE, D_STRIDE, true>(
          sS, sP, sO,
          sRowMax, sRowSum,
          block.valid_q_rows, valid_kv_rows, tid, block_q);
        __syncthreads();

        // ======================================================================================
        // Compute:  Dropout mask application
        // Layout:   sP[row: valid_q_rows, N_STRIDE] masked, dmask stored if requested
        // Varlen:   GLOBAL_N=block.seqlen_k (actual KV length) for correct RNG stride
        // Template: IS_DROPOUT guards compile-time; runtime p_dropout > 0 enables execution
        // ======================================================================================
        WMMA_GEMM_DROPOUT<Config, BLOCK_M, BLOCK_N, N_STRIDE, IS_DROPOUT>(
          sP, dmask_ptr ? dmask_ptr + start_kv : nullptr,
          block.valid_q_rows, valid_kv_rows,
          block.q_base + block.start_q, start_kv, max_seqlen_k, H_Q * max_seqlen_k,
          p_dropout, dropout_seed, dropout_offset, tid);
        __syncthreads();

        // ======================================================================================
        // Load:     V tile from global to sV shared memory
        // Layout:   V: global[row: BLOCK_N, H_K * D] -> shared[row: BLOCK_N, D_STRIDE]
        // Template: DUAL_LOAD=false, SMEM_STRIDE=D_STRIDE, GLOBAL_WIDTH=D (varlen sub-tile)
        // ======================================================================================
        WMMA_GEMM_LOAD_TILE<Config, false, D_STRIDE, D>(
          v_ptr_page, sV,
          nullptr,    nullptr,
          H_K * D, valid_kv_rows, tid);
        __syncthreads();

        // ==============================================================================
        // Compute:  dO += P @ V
        // Layout:   sP[valid_q_rows, N_STRIDE] @ sV[valid_kv_rows, D_STRIDE] += sO
        // Template: BLOCK_M/BLOCK_N static, valid_q/valid_kv dynamic (varlen ragged tiles)
        // ==============================================================================
        WMMA_GEMM_GRADIENTS<Config, GemmType::dO_PV, D, BLOCK_M, BLOCK_N, N_STRIDE, D_STRIDE>(
          sP, sV, sO,
          block.valid_q_rows, valid_kv_rows,
          warp_id, lane_id);
        __syncthreads();
    }   // END MAIN LOOP
    // ======================================================================================
    // Compute:  Store normalized attention output O = softmax(S) @ V
    // Layout:   sO[row: valid_q_rows, D_STRIDE] -> out_ptr[row: valid_q_rows, H_Q * D]
    // Template: SMEM_STRIDE=D_STRIDE, GLOBAL_WIDTH=D (varlen sub-tile store)
    // ======================================================================================
    WMMA_GEMM_EPILOGUE<Config, GemmType::write_dO, D_STRIDE, D>(
      sO,      out_ptr,
      nullptr, nullptr,
      sRowSum, H_Q * D, block.valid_q_rows, tid);

    if (tid < block.valid_q_rows) {
        const float sum = fmaxf(sRowSum[tid], 1e-24f);
        softmax_lse_ptr[tid] = sRowMax[tid] + logf(sum);
    }
}

// ======================================================================================
// LAUNCHER
// ======================================================================================
template<int D>
void launcher_flash_attention_forward_varlen(
    const torch::Tensor& Q,
    const torch::Tensor& K,
    const torch::Tensor& V,
          torch::Tensor& Out,
          torch::Tensor& softmax_lse,
    const torch::Tensor& dmask,
    const torch::Tensor& cu_seqlens_q,
    const torch::Tensor& cu_seqlens_k,
    const torch::Tensor& seqused_k,
    const torch::Tensor& leftpad_k,
    const torch::Tensor& block_table,
    const bool   paged_KV,
    const int    T_Q,
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
    TORCH_CHECK(smem <= MAX_SMEM_PER_SM, "Shared memory exceeds 96KB for Forward varlen kernel: ", smem, " bytes (", smem / 1024, " KB)");

    const int B                  = cu_seqlens_q.size(0) - 1;
    const int H_Q                = Q.size(1);
    const int H_K                = paged_KV ? K.size(2) : K.size(1);
    const int block_page         = paged_KV ? K.size(1) : 1;
    const int block_table_stride = (paged_KV && block_table.defined()) ? block_table.stride(0) : 0;
    const int kv_block_stride    = paged_KV ? K.stride(0) : 0;

    const dim3 grid((max_seqlen_q + Config::DO::BLOCK_M - 1) / Config::DO::BLOCK_M, 1, B * H_Q);
    const dim3 block(Config::THREADS_PER_BLOCK);

    bool is_alibi       = (alibi_slopes != nullptr);
    bool is_softcap     = (softcap > 0.0f);
    bool is_window      = (window_left >= 0 || window_right >= 0);
    bool is_dropout     = (p_dropout > 0.0f);
    bool is_paged       = paged_KV;
    bool is_rope        = false;
    bool is_interleaved = false;

    const __half* q_ptr            = reinterpret_cast<const __half*>(Q.data_ptr());
    const __half* k_ptr            = reinterpret_cast<const __half*>(K.data_ptr());
    const __half* v_ptr            = reinterpret_cast<const __half*>(V.data_ptr());
          __half* out_ptr          = reinterpret_cast<__half*>(Out.data_ptr());
           float* lse_ptr          = softmax_lse.data_ptr<float>();
          __half* dmask_ptr        = dmask.numel() > 0 ? reinterpret_cast<__half*>(dmask.data_ptr()) : nullptr;
    const int*    cu_seqlens_q_ptr = cu_seqlens_q.data_ptr<int>();
    const int*    cu_seqlens_k_ptr = cu_seqlens_k.data_ptr<int>();
    const int*    seqused_k_ptr    = seqused_k.defined() ? seqused_k.data_ptr<int>() : nullptr;
    const int*    leftpad_k_ptr    = leftpad_k.defined() ? leftpad_k.data_ptr<int>() : nullptr;
    const int*    block_table_ptr  = paged_KV ? block_table.data_ptr<int>() : nullptr;

    dispatch_attention_features(is_causal, is_alibi, is_softcap, is_window, is_dropout, is_paged, is_rope, is_interleaved,
    [&](auto CAUSAL, auto ALIBI, auto SOFTCAP, auto WINDOW, auto DROPOUT, auto PAGED, auto /*ROPE*/, auto /*INTERLEAVED*/) {
        constexpr bool IS_CAUSAL  = decltype(CAUSAL)::value;
        constexpr bool IS_ALIBI   = decltype(ALIBI)::value;
        constexpr bool IS_SOFTCAP = decltype(SOFTCAP)::value;
        constexpr bool IS_WINDOW  = decltype(WINDOW)::value;
        constexpr bool IS_DROPOUT = decltype(DROPOUT)::value;
        constexpr bool IS_PAGED   = decltype(PAGED)::value;

        auto kernel = flash_attention_forward_varlen_kernel<D, IS_CAUSAL, IS_ALIBI, IS_SOFTCAP, IS_WINDOW, IS_DROPOUT, IS_PAGED>;
        cudaFuncSetAttribute(kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, smem);

        kernel<<<grid, block, smem, stream>>>(
            q_ptr, k_ptr, v_ptr, out_ptr, lse_ptr, dmask_ptr,
            cu_seqlens_q_ptr, cu_seqlens_k_ptr,
            seqused_k_ptr, leftpad_k_ptr, block_table_ptr,
            B, H_Q, H_K, T_Q, max_seqlen_q, max_seqlen_k,
            block_page, block_table_stride, kv_block_stride,
            softmax_scale, softcap, alibi_slopes, alibi_batch, window_left, window_right,
            p_dropout, dropout_seed, dropout_offset
        );
    });
}

// ======================================================================================
// WRAPPER
// ======================================================================================
std::vector<at::Tensor> flash_attention_varlen_forward(
          at::Tensor& q,
    const at::Tensor& k,
    const at::Tensor& v,
    std::optional<at::Tensor> &out,
    const at::Tensor& cu_seqlens_q,
    const at::Tensor& cu_seqlens_k,
    std::optional<at::Tensor> &seqused_k,
    std::optional<const at::Tensor> &leftpad_k,
    std::optional<at::Tensor> &block_table,
    std::optional<at::Tensor> alibi_slopes,
    int          max_seqlen_q,
    const int    max_seqlen_k,
    const float  p_dropout,
    const float  softmax_scale,
    const bool   zero_tensors,
    bool         is_causal,
    int          window_left,
    int          window_right,
    const float  softcap,
    const bool   return_softmax,
    std::optional<at::Generator> gen,
    int          num_splits = 0
) {
    // Device guard for multi-GPU / pipeline-parallelism
    at::cuda::CUDAGuard device_guard{q.device()};

    auto props = at::cuda::getCurrentDeviceProperties();
    TORCH_CHECK(props->major == 7 && props->minor == 0, "Kernel supports only Volta GPUs.");

    // dtype / device / contiguity
    auto q_dtype = q.dtype();
    TORCH_CHECK(q_dtype == torch::kFloat16, "q must be fp16");
    TORCH_CHECK(k.dtype() == q_dtype && v.dtype() == q_dtype, "k/v must have the same dtype as q");
    TORCH_CHECK(q.is_cuda() && k.is_cuda() && v.is_cuda(), "Tensors q, k, v must be on CUDA");
    TORCH_CHECK(q.stride(-1) == 1 && k.stride(-1) == 1 && v.stride(-1) == 1, "Last dim of q, k, v must be contiguous");

    // dimensions
    const int T_Q = q.size(0);
    const int H_Q = q.size(1);
    const int D   = q.size(2);
    const int B   = cu_seqlens_q.size(0) - 1;
    const bool paged_KV = block_table.has_value();

    int H_K = paged_KV ? k.size(2) : k.size(1);
    int page_block_size = paged_KV ? k.size(1) : 1;

    TORCH_CHECK(B > 0, "batch size must be positive");
    TORCH_CHECK(D <= 256, "head dimension must be <= 256");
    TORCH_CHECK(D % 8 == 0, "head dimension must be multiple of 8");
    TORCH_CHECK(H_Q % H_K == 0, "H_Q must be divisible by H_K for GQA/MQA");
    TORCH_CHECK(num_splits <= 1, "num_splits > 1 not supported");

    // causal / window optimizations
    if (max_seqlen_q == 1 && !alibi_slopes.has_value()) { is_causal = false; }

    // softcap restrictions
    if (softcap > 0.f) {
        TORCH_CHECK(p_dropout == 0.f, "Softcapping does not support dropout");
    }

    // paged KV
    at::Tensor block_table_tensor;
    if (paged_KV) {
        block_table_tensor = block_table.value();
        TORCH_CHECK(block_table_tensor.is_cuda(), "block_table must be on CUDA");
        TORCH_CHECK(block_table_tensor.dtype() == torch::kInt32, "block_table must have dtype int32");
        TORCH_CHECK(block_table_tensor.stride(-1) == 1, "block_table must have contiguous last dimension");
        TORCH_CHECK(page_block_size % 256 == 0, "Paged KV block size must be divisible by 256");

        const int num_blocks = k.size(0);
        const int max_num_blocks_per_seq = block_table_tensor.size(1);
        TORCH_CHECK(block_table_tensor.size(0) == B, "block_table batch dim must match q");
        TORCH_CHECK(k.sizes() == v.sizes(), "K and V paged shapes must match");
        TORCH_CHECK(k.size(0) == num_blocks, "k block dim mismatch");
        TORCH_CHECK(k.size(1) == page_block_size, "k page size mismatch");
        TORCH_CHECK(k.size(2) == H_K, "k heads mismatch");
        TORCH_CHECK(k.size(3) == D, "k head_dim mismatch");
    }

    // cu_seqlens
    TORCH_CHECK(cu_seqlens_q.dtype() == torch::kInt32 && cu_seqlens_k.dtype() == torch::kInt32, "cu_seqlens must be int32");
    TORCH_CHECK(cu_seqlens_q.is_cuda() && cu_seqlens_k.is_cuda(), "cu_seqlens must be on CUDA");
    TORCH_CHECK(cu_seqlens_q.is_contiguous() && cu_seqlens_k.is_contiguous(), "cu_seqlens must be contiguous");
    TORCH_CHECK(cu_seqlens_q.dim() == 1 && cu_seqlens_k.dim() == 1, "cu_seqlens must be 1D");
    TORCH_CHECK(cu_seqlens_q.size(0) == B + 1, "cu_seqlens_q size mismatch");
    TORCH_CHECK(cu_seqlens_k.size(0) == B + 1, "cu_seqlens_k size mismatch");

    // seqused_k
    at::Tensor seqused_tensor;
    if (seqused_k.has_value()) {
        seqused_tensor = seqused_k.value();
        TORCH_CHECK(seqused_tensor.dtype() == torch::kInt32, "seqused_k must have dtype int32");
        TORCH_CHECK(seqused_tensor.is_cuda(), "seqused_k must be on CUDA");
        TORCH_CHECK(seqused_tensor.is_contiguous(), "seqused_k must be contiguous");
        TORCH_CHECK(seqused_tensor.dim() == 1 && seqused_tensor.size(0) == B, "seqused_k must be 1D with size == batch_size");
    }

    // leftpad_k
    at::Tensor leftpad_tensor;
    if (leftpad_k.has_value()) {
        TORCH_CHECK(!paged_KV, "Paged KV and leftpad_k are not supported simultaneously");
        leftpad_tensor = leftpad_k.value();
        TORCH_CHECK(leftpad_tensor.dtype() == torch::kInt32, "leftpad_k must have dtype int32");
        TORCH_CHECK(leftpad_tensor.is_cuda(), "leftpad_k must be on CUDA");
        TORCH_CHECK(leftpad_tensor.is_contiguous(), "leftpad_k must be contiguous");
        TORCH_CHECK(leftpad_tensor.dim() == 1 && leftpad_tensor.size(0) == B, "leftpad_k must be 1D with size == batch_size");
    }

    // window edge cases
    if (window_left  >= max_seqlen_k) window_left  = -1;
    if (window_right >= max_seqlen_k) window_right = -1;

    // alibi
    const float* alibi_ptr = nullptr;
    int alibi_batch = 0;
    if (alibi_slopes.has_value()) {
        const auto& slopes = alibi_slopes.value();
        TORCH_CHECK(slopes.dtype() == torch::kFloat32 && slopes.is_cuda(), "alibi_slopes must be fp32 on CUDA");
        TORCH_CHECK(slopes.stride(-1) == 1, "alibi_slopes last dim must be contiguous");
        auto sizes = slopes.sizes();
        bool valid = (sizes.size() == 1 && sizes[0] == H_Q) ||
                     (sizes.size() == 2 && sizes[0] == B && sizes[1] == H_Q);
        TORCH_CHECK(valid, "alibi_slopes must be [H_Q] or [B, H_Q]");
        alibi_batch = (sizes.size() == 2) ? slopes.stride(0) : 0;
        alibi_ptr = slopes.data_ptr<float>();
    }

    // dropout
    TORCH_CHECK(p_dropout >= 0.f && p_dropout < 1.f, "p_dropout must be in [0, 1)");

    uint64_t dropout_seed   = 0;
    uint64_t dropout_offset = 0;
    at::Tensor rng_state = torch::empty({2}, torch::dtype(torch::kInt64).device(q.device()));

    if (p_dropout > 0.0f) {
        auto gen_cuda = at::get_generator_or_default<at::CUDAGeneratorImpl>(gen, at::cuda::detail::getDefaultCUDAGenerator());
        std::lock_guard<std::mutex> lock(gen_cuda->mutex_);
        dropout_seed   = gen_cuda->current_seed();
        dropout_offset = gen_cuda->get_offset();
        uint64_t counter_offset = static_cast<uint64_t>(B) * static_cast<uint64_t>(H_Q) * 32ULL;
        gen_cuda->set_offset(dropout_offset + counter_offset);
        rng_state[0] = static_cast<int64_t>(dropout_seed);
        rng_state[1] = static_cast<int64_t>(dropout_offset);
    }

    // output tensors
    at::Tensor out_fp16 = out.has_value() ? out.value() : torch::empty_like(q);
    auto softmax_lse = torch::empty({H_Q, T_Q}, torch::dtype(torch::kFloat32).device(q.device()));

    TORCH_CHECK(out_fp16.dtype() == q_dtype, "out must have the same dtype as q");
    TORCH_CHECK(out_fp16.is_cuda(), "out must be on CUDA");
    TORCH_CHECK(out_fp16.stride(-1) == 1, "out must have contiguous last dimension");
    if (out.has_value()) {
        TORCH_CHECK(out_fp16.sizes() == q.sizes(), "out shape must match q shape");
    }
    TORCH_CHECK(softmax_lse.dtype() == torch::kFloat32, "softmax_lse must be fp32");

    // dropout mask
    at::Tensor dmask;
    if (return_softmax && p_dropout > 0.0f) {
        dmask = torch::empty({T_Q, H_Q, max_seqlen_k}, q.options());
    } else {
        dmask = torch::empty({0}, q.options());
    }

    // zero tensors / empty case
    if (zero_tensors) {
        out_fp16.zero_();
        softmax_lse.fill_(-std::numeric_limits<float>::infinity());
        if (dmask.numel() > 0) dmask.zero_();
    }

    if (T_Q == 0 || max_seqlen_k == 0) {
        return {out_fp16, softmax_lse, dmask, rng_state};
    }

    // run kernel
    auto stream = at::cuda::getCurrentCUDAStream().stream();

    #define LAUNCH_KERNEL(DIM) \
        launcher_flash_attention_forward_varlen<DIM>(q, k, v, out_fp16, softmax_lse, dmask, cu_seqlens_q, cu_seqlens_k, seqused_tensor, leftpad_tensor, \
                                                     block_table_tensor, paged_KV, T_Q, max_seqlen_q, max_seqlen_k, softmax_scale, is_causal, softcap, p_dropout, \
                                                     alibi_ptr, alibi_batch, window_left, window_right, dropout_seed, dropout_offset, stream);
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
