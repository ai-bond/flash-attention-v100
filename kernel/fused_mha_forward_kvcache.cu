// ======================================================================================
// * Copyright (c) 2026, D.Skryabin / tg @ai_bond007 SPDX-License: BSD-3-Clause
// ======================================================================================
#include <cuda.h>
#include <cuda_runtime.h>
#include <cuda_fp16.h>

#include <torch/extension.h>
#include <ATen/ATen.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>

#include "template.h"
#include "kernel.h"
#include "forward.h"
#include "gemm_smem.h"
#include "mat_mul.h"
#include "softmax.h"
#include "rotary.h"

// ======================================================================================
// KV-CACHE FORWARD KERNEL
// ======================================================================================
template<int D, bool IS_CAUSAL, bool IS_ALIBI, bool IS_SOFTCAP, bool IS_WINDOW, bool IS_PAGED, bool IS_ROPE, bool IS_INTERLEAVED>
__global__ void __launch_bounds__(KernelConfig<D>::THREADS_PER_BLOCK, 2)
flash_attention_kvcache_kernel(
    const __half* __restrict__ Q,
    const __half* __restrict__ K_new,
    const __half* __restrict__ V_new,
          __half* __restrict__ K_cache,
          __half* __restrict__ V_cache,
          __half* __restrict__ Out,
           float* __restrict__ softmax_lse,
    const int*    __restrict__ cache_seqlens,
    const int*    __restrict__ cache_batch_idx,
    const int*    __restrict__ leftpad_k,
    const int*    __restrict__ block_table,
    const __half* __restrict__ rotary_cos,
    const __half* __restrict__ rotary_sin,
    const int     B,
    const int     H_Q,
    const int     H_K,
    const int     T_Q,
    const int     T_NEW,
    const int     max_seqlen_k,
    const int     block_page,
    const int     block_table_stride,
    const int     kv_block_stride,
    const int     rotary_dim,
    const float   softmax_scale,
    const float   softcap,
    const float*  alibi_slopes,
    const int     alibi_batch,
    int           window_left,
    int           window_right,
    const int     stride_q_b, const int stride_q_s, const int stride_q_h,
    const int     stride_k_b, const int stride_k_s, const int stride_k_h,
    const int     stride_n_b, const int stride_n_s, const int stride_n_h,
    const int     stride_o_b, const int stride_o_s, const int stride_o_h
) {
    using Config = KernelConfig<D>;

    constexpr int BLOCK_M  = Config::DO::BLOCK_M;
    constexpr int BLOCK_N  = Config::DO::BLOCK_N;
    constexpr int D_STRIDE = Config::DO::D_STRIDE;
    constexpr int N_STRIDE = Config::DO::N_STRIDE;

    // ==================================================================================
    // Grid Mapping: X for Q-blocks, Z for batch-head composite (batch * H_Q + head)
    // ==================================================================================
    const int block_idx    = blockIdx.x;
    const int bthd_idx     = blockIdx.z;

    if (bthd_idx >= B * H_Q) return;

    // ======================================================================================
    // BlockInfo: Metadata resolution
    // ======================================================================================
    const int batch_idx   = bthd_idx / H_Q;
    const int head_idx    = bthd_idx % H_Q;
    const int kv_head_idx = head_idx / (H_Q / H_K);

    const int cache_bidx     = (cache_batch_idx != nullptr) ? cache_batch_idx[batch_idx] : batch_idx;
    const int leftpad        = (leftpad_k != nullptr) ? leftpad_k[batch_idx] : 0;
    const int cache_seqlen   = (cache_seqlens != nullptr) ? cache_seqlens[cache_bidx] : 0;
    const int total_seqlen_k = cache_seqlen + T_NEW;

    BlockInfo<IS_CAUSAL, IS_WINDOW, false> block;
    block.init_q(
        block_idx,       // BLOCK_IDX:      Current Q-block index (grid.x)
        bthd_idx,        // BATCH_HEAD_ID:  Flattened batch-head index (grid.z)
        H_Q,             // H_Q:            Number of query heads
        H_K,             // H_K:            Number of KV heads
        T_Q,             // T_Q:            Total query length (uniform across batch)
        total_seqlen_k,  // T_K:            Total KV length = cache_seqlen + T_NEW
        B,               // B:              Batch size
        BLOCK_M,         // BLOCK_M:        Tile size along Q dimension
        BLOCK_N,         // BLOCK_N:        Tile size along KV dimension
        window_left,     // WINDOW_LEFT:    Left sliding window bound (-1 if disabled)
        window_right,    // WINDOW_RIGHT:   Right sliding window bound (-1 if disabled)
        nullptr,         // CU_SEQLENS_Q:   nullptr (even layout, no varlen)
        nullptr,         // CU_SEQLENS_K:   nullptr (even layout, no varlen)
        nullptr          // SEQUSED_K:      nullptr (no partial KV usage)
    );

    if (block.start_q >= block.seqlen_q || block.valid_q_rows <= 0) return;

    if (block.block_min >= block.block_max || total_seqlen_k == 0) {
        const size_t out_off = static_cast<size_t>(batch_idx) * stride_o_b +
                               static_cast<size_t>(head_idx)  * stride_o_h + block.start_q * stride_o_s;
        const size_t lse_off = static_cast<size_t>(batch_idx) * H_Q * T_Q +
                               static_cast<size_t>(head_idx)  * T_Q + block.start_q;
        for (int row = threadIdx.x; row < block.valid_q_rows; row += blockDim.x) {
            #pragma unroll
            for (int d = 0; d < D; ++d) Out[out_off + row * stride_o_s + d] = __float2half(0.0f);
            softmax_lse[lse_off + row] = NEG_INF;
        }
        return;
    }

    // ==================================================================================
    // Init: thread/warp/lane IDs
    // ==================================================================================
    const int tid     = threadIdx.x;
    const int warp_id = tid >> 5;
    const int lane_id = tid & 31;
    // Alibi slope only for valid block + batch
    const int   alibi_idx   = (alibi_batch > 0) ? (batch_idx * alibi_batch + head_idx) : head_idx;
    const float alibi_slope = (alibi_slopes != nullptr) ? alibi_slopes[alibi_idx] : 0.0f;

    // ==================================================================================
    // KV-CACHE UPDATE
    // ==================================================================================
    WMMA_GEMM_UPDATE_KVCACHE<Config, D, IS_PAGED, IS_ROPE, IS_INTERLEAVED>(
      K_new, V_new, K_cache, V_cache, rotary_cos, rotary_sin, block_table,
      T_NEW, H_K,
      stride_n_b, stride_n_s, stride_n_h,
      stride_k_b, stride_k_s, stride_k_h,
      batch_idx, kv_head_idx, cache_bidx, cache_seqlen, rotary_dim,
      leftpad, block_page, block_table_stride, tid);
    __syncthreads();

    // ==================================================================================
    // RAGGED POINTERS
    // Layout:
    //   Q/Out: [B, T_Q, H_Q, D]       offset = batch_idx*stride_q_b + head_idx*stride_q_h + start_q*stride_q_s
    //   LSE:   [B*H_Q, T_Q]           offset = bthd_idx*T_Q + start_q
    // ==================================================================================
    const size_t q_base = static_cast<size_t>(batch_idx) * stride_q_b + static_cast<size_t>(head_idx) * stride_q_h;
    const __half* __restrict__ q_ptr   = Q + q_base + block.start_q * stride_q_s;
          __half* __restrict__ out_ptr = Out + static_cast<size_t>(batch_idx) * stride_o_b + static_cast<size_t>(head_idx) * stride_o_h + block.start_q * stride_o_s;
           float* __restrict__ lse_ptr = softmax_lse + static_cast<size_t>(bthd_idx) * T_Q + block.start_q;

    // ==================================================================================
    // Init: shared memory
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
    // Load Q tile + Rotary
    // ==================================================================================
    WMMA_GEMM_TILE_ROTARY<Config, D_STRIDE, IS_CAUSAL, IS_WINDOW, IS_ROPE, IS_INTERLEAVED, D>(
      q_ptr, sQ, rotary_cos, rotary_sin,
      stride_q_s, block.valid_q_rows,
      cache_seqlen, rotary_dim, leftpad,
      block.start_q, tid);
    __syncthreads();

    // ==================================================================================
    // MAIN LOOP (iterates over K/V blocks for current Q block)
    // ==================================================================================
    for (int block_q = block.block_min; block_q < block.block_max; ++block_q) {
        const int start_kv      = block_q * BLOCK_N;
        const int valid_kv_rows = min(BLOCK_N, total_seqlen_k - start_kv);
        if (valid_kv_rows <= 0) break;

        // ======================================================================================
        // KV-CACHE (Paged vs Contiguous)
        // ======================================================================================
        const __half* __restrict__ k_ptr_page;
        const __half* __restrict__ v_ptr_page;

        if constexpr (IS_PAGED) {
            const int page_idx    = (start_kv + leftpad) / block_page;
            const int page_offset = (start_kv + leftpad) % block_page;
            const int bt_idx      = cache_bidx * block_table_stride + page_idx;
            const int phys_page   = block_table[bt_idx];
            const size_t kv_load  = static_cast<size_t>(phys_page) * kv_block_stride +
                                    static_cast<size_t>(page_offset) * stride_k_s +
                                    static_cast<size_t>(kv_head_idx) * stride_k_h;
            k_ptr_page = K_cache + kv_load;
            v_ptr_page = V_cache + kv_load;
        } else {
            const size_t kv_load = static_cast<size_t>(cache_bidx) * stride_k_b +
                                   static_cast<size_t>(start_kv + leftpad) * stride_k_s +
                                   static_cast<size_t>(kv_head_idx) * stride_k_h;
            k_ptr_page = K_cache + kv_load;
            v_ptr_page = V_cache + kv_load;
        }

        // ======================================================================================
        // Load:     K tile from global to sK shared memory
        // Layout:   K: global[row: BLOCK_N, H_K * D] -> shared[row: BLOCK_N, D_STRIDE]
        // Template: DUAL_LOAD=false, SMEM_STRIDE=D_STRIDE, GLOBAL_WIDTH=D (varlen sub-tile)
        // ======================================================================================
        WMMA_GEMM_LOAD_TILE<Config, false, D_STRIDE, D>(
          k_ptr_page, sK,
          nullptr,    nullptr,
          stride_k_s, valid_kv_rows, tid);
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
        // Load:     V tile from global to sV shared memory
        // Layout:   V: global[row: BLOCK_N, H_K * D] -> shared[row: BLOCK_N, D_STRIDE]
        // Template: DUAL_LOAD=false, SMEM_STRIDE=D_STRIDE, GLOBAL_WIDTH=D (varlen sub-tile)
        // ======================================================================================
        WMMA_GEMM_LOAD_TILE<Config, false, D_STRIDE, D>(
          v_ptr_page, sV,
          nullptr,    nullptr,
          stride_k_s, valid_kv_rows, tid);
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
      sRowSum, stride_o_s, block.valid_q_rows, tid);

    if (tid < block.valid_q_rows) {
        const float sum = fmaxf(sRowSum[tid], 1e-24f);
        const size_t lse_idx = static_cast<size_t>(batch_idx) * H_Q * T_Q +
                               static_cast<size_t>(head_idx) * T_Q +
                               static_cast<size_t>(block.start_q + tid);
        softmax_lse[lse_idx] = (sRowSum[tid] <= 1e-24f) ? NEG_INF : (sRowMax[tid] + logf(sum));
    }
}

// ======================================================================================
// LAUNCHER
// ======================================================================================
template<int D>
void launcher_flash_attention_kvcache(
    const torch::Tensor& Q,
    const torch::Tensor& K_new,
    const torch::Tensor& V_new,
    const torch::Tensor& K_cache,
    const torch::Tensor& V_cache,
    torch::Tensor& Out,
    torch::Tensor& softmax_lse,
    const torch::Tensor& cache_seqlens,
    const torch::Tensor& cache_batch_idx,
    const torch::Tensor& leftpad_k,
    const torch::Tensor& block_table,
    const torch::Tensor& rotary_cos,
    const torch::Tensor& rotary_sin,
    const int      T_NEW,
    const int      max_seqlen_k,
    const bool     is_rope,
    const bool     is_interleaved,
    const int      rotary_dim,
    const float    softmax_scale,
    const bool     is_causal,
    const float    softcap,
    const float*   alibi_slopes,
    const int      alibi_batch,
    int            window_left,
    int            window_right,
    const bool     is_paged,
    cudaStream_t   stream
) {
    using Config = KernelConfig<D>;

    const size_t smem = Config::TOTAL_SMEM;
    TORCH_CHECK(smem <= MAX_SMEM_PER_SM, "Shared memory exceeds 96KB for Forward KV-Cache kernel: ", smem, " bytes (", smem / 1024, " KB)");

    const int B   = Q.size(0);
    const int T_Q = Q.size(1);
    const int H_Q = Q.size(2);
    const int H_K = K_cache.size(2);

    const int block_page         = is_paged ? K_cache.size(1) : 1;
    const int block_table_stride = is_paged ? block_table.stride(0) : 0;
    const int kv_block_stride    = is_paged ? K_cache.stride(0) : 0;

    const dim3 grid((T_Q + Config::DO::BLOCK_M - 1) / Config::DO::BLOCK_M, 1, B * H_Q);
    const dim3 block(Config::THREADS_PER_BLOCK);

    // Data pointers
    const __half* q_ptr          = reinterpret_cast<const __half*>(Q.data_ptr());
    const __half* k_new_ptr      = K_new.defined() ? reinterpret_cast<const __half*>(K_new.data_ptr()) : nullptr;
    const __half* v_new_ptr      = V_new.defined() ? reinterpret_cast<const __half*>(V_new.data_ptr()) : nullptr;
    __half*       k_cache_ptr    = reinterpret_cast<__half*>(K_cache.data_ptr());
    __half*       v_cache_ptr    = reinterpret_cast<__half*>(V_cache.data_ptr());
    __half*       out_ptr        = reinterpret_cast<__half*>(Out.data_ptr());
    float*        lse_ptr        = softmax_lse.data_ptr<float>();

    const int*    cache_seqlens_ptr   = cache_seqlens.defined() ? cache_seqlens.data_ptr<int>() : nullptr;
    const int*    cache_batch_idx_ptr = cache_batch_idx.defined() ? cache_batch_idx.data_ptr<int>() : nullptr;
    const int*    leftpad_k_ptr       = leftpad_k.defined() ? leftpad_k.data_ptr<int>() : nullptr;
    const int*    block_table_ptr     = is_paged ? block_table.data_ptr<int>() : nullptr;
    const __half* rotary_cos_ptr      = rotary_cos.defined() ? reinterpret_cast<const __half*>(rotary_cos.data_ptr()) : nullptr;
    const __half* rotary_sin_ptr      = rotary_sin.defined() ? reinterpret_cast<const __half*>(rotary_sin.data_ptr()) : nullptr;

    // Strides
    const int stride_q_b = Q.stride(0);
    const int stride_q_s = Q.stride(1);
    const int stride_q_h = Q.stride(2);

    const int stride_k_b = K_cache.stride(0);
    const int stride_k_s = K_cache.stride(1);
    const int stride_k_h = K_cache.stride(2);

    const int stride_n_b = K_new.defined() ? K_new.stride(0) : 0;
    const int stride_n_s = K_new.defined() ? K_new.stride(1) : 0;
    const int stride_n_h = K_new.defined() ? K_new.stride(2) : 0;

    const int stride_o_b = Out.stride(0);
    const int stride_o_s = Out.stride(1);
    const int stride_o_h = Out.stride(2);

    bool is_alibi   = (alibi_slopes != nullptr);
    bool is_softcap = (softcap > 0.0f);
    bool is_window  = (window_left >= 0 || window_right >= 0);
    bool is_dropout = false;

    dispatch_attention_features(is_causal, is_alibi, is_softcap, is_window, is_dropout, is_paged, is_rope, is_interleaved,
    [&](auto CAUSAL, auto ALIBI, auto SOFTCAP, auto WINDOW, auto /*DROPOUT*/, auto PAGED, auto ROPE, auto INTERLEAVED) {
        constexpr bool IS_CAUSAL      = decltype(CAUSAL)::value;
        constexpr bool IS_ALIBI       = decltype(ALIBI)::value;
        constexpr bool IS_SOFTCAP     = decltype(SOFTCAP)::value;
        constexpr bool IS_WINDOW      = decltype(WINDOW)::value;
        constexpr bool IS_PAGED       = decltype(PAGED)::value;
        constexpr bool IS_ROPE        = decltype(ROPE)::value;
        constexpr bool IS_INTERLEAVED = decltype(INTERLEAVED)::value;

        auto kernel = flash_attention_kvcache_kernel<D, IS_CAUSAL, IS_ALIBI, IS_SOFTCAP, IS_WINDOW, IS_PAGED, IS_ROPE, IS_INTERLEAVED>;
        cudaFuncSetAttribute(kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, smem);
        kernel<<<grid, block, smem, stream>>>(
            q_ptr, k_new_ptr, v_new_ptr, k_cache_ptr, v_cache_ptr, out_ptr, lse_ptr,
            cache_seqlens_ptr, cache_batch_idx_ptr, leftpad_k_ptr, block_table_ptr,
            rotary_cos_ptr, rotary_sin_ptr,
            B, H_Q, H_K, T_Q, T_NEW, max_seqlen_k,
            block_page, block_table_stride, kv_block_stride, rotary_dim,
            softmax_scale, softcap, alibi_slopes, alibi_batch,
            window_left, window_right,
            stride_q_b, stride_q_s, stride_q_h,
            stride_k_b, stride_k_s, stride_k_h,
            stride_n_b, stride_n_s, stride_n_h,
            stride_o_b, stride_o_s, stride_o_h
        );
    });
}

// ======================================================================================
// WRAPPER
// ======================================================================================
std::vector<at::Tensor> flash_attention_kvcache(
    at::Tensor &q,
    const at::Tensor &kcache,
    const at::Tensor &vcache,
    std::optional<const at::Tensor> &k,
    std::optional<const at::Tensor> &v,
    std::optional<const at::Tensor> &seqlens_k,
    std::optional<const at::Tensor> &rotary_cos,
    std::optional<const at::Tensor> &rotary_sin,
    std::optional<const at::Tensor> &cache_batch_idx,
    std::optional<const at::Tensor> &leftpad_k,
    std::optional<at::Tensor> &block_table,
    std::optional<at::Tensor> alibi_slopes,
    std::optional<at::Tensor> &out,
    const float  softmax_scale,
    bool         is_causal,
    int          window_left,
    int          window_right,
    const float  softcap,
    bool         is_rotary_interleaved,
    int          num_splits = 0
) {
    // Device guard for multi-GPU / pipeline-parallelism
    at::cuda::CUDAGuard device_guard{q.device()};

    auto props = at::cuda::getCurrentDeviceProperties();
    TORCH_CHECK(props->major == 7 && props->minor == 0, "Kernel supports only Volta GPUs.");

    // dtype / device / contiguity
    TORCH_CHECK(q.dtype() == torch::kFloat16, "q must be fp16");
    TORCH_CHECK(kcache.dtype() == torch::kFloat16 && vcache.dtype() == torch::kFloat16, "kcache/vcache must be fp16");
    TORCH_CHECK(q.is_cuda() && kcache.is_cuda() && vcache.is_cuda(), "Tensors q, kcache, vcache must be on CUDA");
    TORCH_CHECK(q.stride(-1) == 1 && kcache.stride(-1) == 1 && vcache.stride(-1) == 1, "Last dim of q, kcache, vcache must be contiguous");

    // dimensions
    const int B         = q.size(0);
    const int T_Q       = q.size(1);
    const int H_Q       = q.size(2);
    const int D         = q.size(3);
    const int H_K       = kcache.size(2);
    const bool is_paged = block_table.has_value();

    TORCH_CHECK(B > 0, "batch size must be positive");
    TORCH_CHECK(D <= 256, "head dimension must be <= 256");
    TORCH_CHECK(H_Q % H_K == 0, "H_Q must be divisible by H_K for GQA/MQA");
    TORCH_CHECK(D % 8 == 0, "head dimension must be multiple of 8");
    TORCH_CHECK(num_splits <= 1, "num_splits > 1 not supported now, for future develop");

    // causal optimizations
    if (T_Q == 1 && !alibi_slopes.has_value()) { is_causal = false; }
    if (is_causal) { window_right = 0; }

    // softcap restrictions
    if (softcap > 0.f) {
        TORCH_CHECK(window_left < 0 && window_right < 0, "Softcap + window not supported");
        TORCH_CHECK(!alibi_slopes.has_value(), "Softcap + ALiBi not supported");
    }

    // paged KV
    at::Tensor block_table_tensor;
    int page_block_size = 1;
    int max_seqlen_k = 0;

    if (is_paged) {
        TORCH_CHECK(!cache_batch_idx.has_value(), "Paged KVcache does not support cache_batch_idx");
        block_table_tensor = block_table.value();
        TORCH_CHECK(block_table_tensor.is_cuda(), "block_table must be on CUDA");
        TORCH_CHECK(block_table_tensor.dtype() == torch::kInt32, "block_table must have dtype int32");
        TORCH_CHECK(block_table_tensor.stride(-1) == 1, "block_table must have contiguous last dimension");
        TORCH_CHECK(block_table_tensor.size(0) == B, "block_table batch dim must match q");

        page_block_size = kcache.size(1);
        TORCH_CHECK(page_block_size % 256 == 0, "Paged KV block size must be divisible by 256");

        const int num_blocks = kcache.size(0);
        const int max_num_blocks_per_seq = block_table_tensor.size(1);
        max_seqlen_k = max_num_blocks_per_seq * page_block_size;

        TORCH_CHECK(kcache.sizes() == vcache.sizes(), "K and V paged shapes must match");
        TORCH_CHECK(kcache.size(0) == num_blocks, "kcache block dim mismatch");
        TORCH_CHECK(kcache.size(1) == page_block_size, "kcache page size mismatch");
        TORCH_CHECK(kcache.size(2) == H_K, "kcache heads mismatch");
        TORCH_CHECK(kcache.size(3) == D, "kcache head_dim mismatch");
    } else {
        max_seqlen_k = kcache.size(1);
    }

    // new K/V tensors
    at::Tensor k_tensor, v_tensor;
    const int T_NEW = (k.has_value() && v.has_value()) ? k->size(1) : 0;
    if (k.has_value()) {
        TORCH_CHECK(v.has_value(), "If key is supplied, value must also be passed in");
        TORCH_CHECK(seqlens_k.has_value(), "If key is supplied, seqlens_k must also be passed in");
        TORCH_CHECK(T_Q <= max_seqlen_k, "If key is supplied, it must have seqlen <= the seqlen of the KV cache");

        k_tensor = k.value();
        v_tensor = v.value();
        TORCH_CHECK(k_tensor.dtype() == q.dtype(), "Key must have the same dtype as query");
        TORCH_CHECK(v_tensor.dtype() == q.dtype(), "Value must have the same dtype as query");
        TORCH_CHECK(k_tensor.is_cuda() && v_tensor.is_cuda(), "k/v must be on CUDA");
        TORCH_CHECK(k_tensor.stride(-1) == 1, "Key tensor must have contiguous last dimension");
        TORCH_CHECK(v_tensor.stride(-1) == 1, "Value tensor must have contiguous last dimension");

        int seqlen_knew = k_tensor.size(1);
        TORCH_CHECK(k_tensor.size(0) == B, "k batch dim mismatch");
        TORCH_CHECK(k_tensor.size(1) == T_NEW, "k seqlen mismatch");
        TORCH_CHECK(k_tensor.size(2) == H_K, "k heads mismatch");
        TORCH_CHECK(k_tensor.size(3) == D, "k head_dim mismatch");
        TORCH_CHECK(v_tensor.size(0) == B, "v batch dim mismatch");
        TORCH_CHECK(v_tensor.size(1) == T_NEW, "v seqlen mismatch");
        TORCH_CHECK(v_tensor.size(2) == H_K, "v heads mismatch");
        TORCH_CHECK(v_tensor.size(3) == D, "v head_dim mismatch");
    } else {
        k_tensor = torch::empty({0}, q.options());
        v_tensor = torch::empty({0}, q.options());
    }

    // seqlens_k
    at::Tensor seqlens_tensor;
    if (seqlens_k.has_value()) {
        seqlens_tensor = seqlens_k.value();
        TORCH_CHECK(seqlens_tensor.dtype() == torch::kInt32, "seqlens_k must have dtype int32");
        TORCH_CHECK(seqlens_tensor.is_cuda(), "seqlens_k must be on CUDA");
        TORCH_CHECK(seqlens_tensor.is_contiguous(), "seqlens_k must be contiguous");
        TORCH_CHECK(seqlens_tensor.dim() == 1 && seqlens_tensor.size(0) == B, "seqlens_k must be 1D with size == batch_size");
    }

    // cache_batch_idx
    at::Tensor cbi_tensor;
    if (cache_batch_idx.has_value()) {
        TORCH_CHECK(!is_paged, "cache_batch_idx not supported with paged KV");
        cbi_tensor = cache_batch_idx.value();
        TORCH_CHECK(cbi_tensor.dtype() == torch::kInt32, "cache_batch_idx must have dtype int32");
        TORCH_CHECK(cbi_tensor.is_cuda(), "cache_batch_idx must be on CUDA");
        TORCH_CHECK(cbi_tensor.is_contiguous(), "cache_batch_idx must be contiguous");
    }

    // leftpad_k
    at::Tensor leftpad_tensor;
    if (leftpad_k.has_value()) {
        TORCH_CHECK(!is_paged, "Paged KV and leftpad_k are not supported simultaneously");
        leftpad_tensor = leftpad_k.value();
        TORCH_CHECK(leftpad_tensor.dtype() == torch::kInt32, "leftpad_k must have dtype int32");
        TORCH_CHECK(leftpad_tensor.is_cuda(), "leftpad_k must be on CUDA");
        TORCH_CHECK(leftpad_tensor.is_contiguous(), "leftpad_k must be contiguous");
        TORCH_CHECK(leftpad_tensor.dim() == 1 && leftpad_tensor.size(0) == B, "leftpad_k must be 1D with size == batch_size");
    }

    // Rotary
    bool is_rope        = false;
    bool is_interleaved = false;
    int  rotary_dim     = 0;
    at::Tensor cos_tensor, sin_tensor;
    if (rotary_cos.has_value() && rotary_sin.has_value()) {
        TORCH_CHECK(k.has_value(), "If rotary cos/sin are provided, new key/value must also be provided");

        cos_tensor = rotary_cos.value();
        sin_tensor = rotary_sin.value();

        TORCH_CHECK(cos_tensor.is_cuda(), "rotary_cos must be on CUDA");
        TORCH_CHECK(sin_tensor.is_cuda(), "rotary_sin must be on CUDA");
        TORCH_CHECK(cos_tensor.dtype() == q.dtype(), "rotary_cos must have the same dtype as query (fp16)");
        TORCH_CHECK(sin_tensor.dtype() == q.dtype(), "rotary_sin must have the same dtype as query (fp16)");
        TORCH_CHECK(cos_tensor.is_contiguous(), "rotary_cos must be contiguous");
        TORCH_CHECK(sin_tensor.is_contiguous(), "rotary_sin must be contiguous");

        rotary_dim = cos_tensor.size(1) * 2;
        TORCH_CHECK(cos_tensor.dim() == 2, "rotary_cos must be 2D");
        TORCH_CHECK(rotary_dim <= D, "rotary_dim must be <= head_dim");
        TORCH_CHECK(rotary_dim % 16 == 0, "rotary_dim must be divisible by 16");

        const int seqlen_ro = cos_tensor.size(0);
        TORCH_CHECK(seqlen_ro >= max_seqlen_k + T_NEW, "rotary_cos seqlen too small");
        TORCH_CHECK(cos_tensor.size(1) == rotary_dim / 2, "rotary_cos last dim mismatch");
        TORCH_CHECK(sin_tensor.size(0) == seqlen_ro, "rotary_sin seqlen mismatch");
        TORCH_CHECK(sin_tensor.size(1) == rotary_dim / 2, "rotary_sin last dim mismatch");
        is_rope        = (rotary_dim > 0);
        is_interleaved = is_rotary_interleaved;
    }

    // Window edge cases
    if (window_left  >= max_seqlen_k) window_left  = -1;
    if (window_right >= max_seqlen_k) window_right = -1;

    // Alibi
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

    // Output tensors
    at::Tensor out_fp16 = out.has_value() ? out.value() : torch::empty_like(q);
    auto softmax_lse = torch::empty({B, H_Q, T_Q}, torch::dtype(torch::kFloat32).device(q.device()));

    TORCH_CHECK(out_fp16.dtype() == q.dtype(), "out must have the same dtype as q");
    TORCH_CHECK(out_fp16.is_cuda(), "out must be on CUDA");
    TORCH_CHECK(out_fp16.stride(-1) == 1, "out must have contiguous last dimension");
    if (out.has_value()) {
        TORCH_CHECK(out_fp16.sizes() == q.sizes(), "out shape must match q shape");
    }
    TORCH_CHECK(softmax_lse.dtype() == torch::kFloat32, "softmax_lse must be fp32");

    // Empty case
    if (T_Q == 0 || max_seqlen_k == 0) {
        out_fp16.zero_();
        softmax_lse.fill_(-std::numeric_limits<float>::infinity());
        return {out_fp16, softmax_lse};
    }

    // Run kernel
    auto stream = at::cuda::getCurrentCUDAStream().stream();

    #define LAUNCH_KERNEL(DIM) \
    launcher_flash_attention_kvcache<DIM>(q, k_tensor, v_tensor, kcache, vcache, out_fp16, softmax_lse, seqlens_tensor, cbi_tensor, \
                                    leftpad_tensor, block_table_tensor, cos_tensor, sin_tensor, T_NEW, max_seqlen_k, is_rope, is_interleaved, rotary_dim, \
                                    softmax_scale, is_causal, softcap, alibi_ptr, alibi_batch, window_left, window_right, is_paged, stream);
    switch (D) {
        case 16:  LAUNCH_KERNEL(16); break;
        case 32:  LAUNCH_KERNEL(32); break;
        case 64:  LAUNCH_KERNEL(64); break;
        case 128: LAUNCH_KERNEL(128); break;
        case 256: LAUNCH_KERNEL(256); break;
        default: TORCH_CHECK(false, "Unsupported D: ", D);
    }
    #undef LAUNCH_KERNEL

    return {out_fp16, softmax_lse};
}
