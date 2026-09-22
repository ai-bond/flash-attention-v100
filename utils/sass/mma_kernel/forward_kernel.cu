// ======================================================================================
// * Copyright (c) 2026, D.Skryabin / tg @ai_bond007 SPDX-License: BSD-3-Clause
// ======================================================================================
#include <cuda.h>
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cstdio>
#include <cmath>
#include <cstring>
#include <cstdlib>
#include <vector>
#include <algorithm>
#include <tuple>
#include <omp.h>

#include "debug.h"
#include "template.h"
#include "kernel.h"
#include "forward.h"
#include "gemm_smem.h"
#include "mat_mul.h"
#include "softmax.h"

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
    // Alibi slope only for valid block + batch
    const int   alibi_idx   = (alibi_batch > 0) ? (block.batch_idx * alibi_batch + block.head_idx) : block.head_idx;
    const float alibi_slope = (alibi_slopes != nullptr) ? alibi_slopes[alibi_idx] : 0.0f;

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

    auto& smem = *reinterpret_cast<typename Config::SmemLayout*>(smem_raw);

    __half* __restrict__ sQ      = smem.phase.fdo.q;
    __half* __restrict__ sK      = smem.phase.fdo.reuse_kv.k;
    __half* __restrict__ sV      = smem.phase.fdo.reuse_kv.v;
    float*  __restrict__ sS      = smem.phase.fdo.reuse_sp.s;
    __half* __restrict__ sP      = smem.phase.fdo.reuse_sp.p;
    float*  __restrict__ sRowMax = smem.row_max;
    float*  __restrict__ sRowSum = smem.row_sum;
    float*  __restrict__ sO      = smem.phase.fdo.o;

    WMMA_GEMM_INIT_SMEM<Config>(smem_raw);
    __syncthreads();
    WMMA_GEMM_INIT_SMEM<Config>(smem.row_max, NEG_INF);
    __syncthreads();

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
        // Compute:  Online Softmax + O-scaling + Dropout
        // Layout:   S[BLOCK_M, BLOCK_N] -> P[BLOCK_M, BLOCK_N], O[BLOCK_M, D] scaled
        // Template: BLOCK_M, BLOCK_N, N_STRIDE, D_STRIDE, TAIL=false, IS_DROPOUT
        // ==================================================================================
        WMMA_GEMM_SOFTMAX<Config, BLOCK_M, BLOCK_N, N_STRIDE, D_STRIDE, IS_DROPOUT>(
          sS, sP, sO,
          sRowMax, sRowSum, dmask_ptr ? dmask_ptr + block.start_q * N + start_kv : nullptr,
          block.valid_q_rows, valid_kv_rows, tid, block_q,
          block.start_q, start_kv, N,
          p_dropout, dropout_seed, dropout_offset, N);
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
        WMMA_GEMM_GRADIENTS<Config, GemmType::dO_PV, D, BLOCK_M, BLOCK_N, N_STRIDE * 2, D_STRIDE>(
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
// CPU REFERENCE & METRICS
// ======================================================================================
void cpu_ref_fwd(const float* Q_ref, const float* K_ref, const float* V_ref,
                 float* Out_ref, float* LSE_ref, int B, int H, int M, int N, int D,
                 float scale, bool causal) {
    #pragma omp parallel for schedule(dynamic)
    for (int bh = 0; bh < B * H; ++bh) {
        for (int i = 0; i < M; ++i) {
            float mx = NEG_INF;
            std::vector<float> S(N);
            for (int j = 0; j < N; ++j) {
                float s = 0;
                for (int d = 0; d < D; ++d) s += Q_ref[(bh*M+i)*D+d] * K_ref[(bh*N+j)*D+d];
                s *= scale;
                if (causal && j > i) s = NEG_INF;
                S[j] = s; mx = std::max(mx, s);
            }
            float sm = 0;
            std::vector<float> P(N);
            for (int j = 0; j < N; ++j) {
                P[j] = (S[j] > NEG_INF + 1e10f) ? expf(S[j] - mx) : 0.0f;
                sm += P[j];
            }
            float inv = (sm > 1e-24f) ? 1.0f / sm : 0.0f;
            for (int d = 0; d < D; ++d) {
                float o = 0;
                for (int j = 0; j < N; ++j) o += P[j] * V_ref[(bh*N+j)*D+d];
                Out_ref[(bh*M+i)*D+d] = o * inv;
            }
            LSE_ref[bh*M+i] = mx + logf(fmaxf(sm, 1e-24f));
        }
    }
}

struct StabilityMetrics {
    float max_diff, avg_diff;
    int nan_count, inf_count, big_diff_count;
};

StabilityMetrics compute_stability_fwd(const std::vector<float>& out_gpu, const std::vector<float>& out_ref,
                                       const std::vector<float>& lse_gpu, const std::vector<float>& lse_ref) {
    StabilityMetrics m = {0,0,0,0,0};
    double sum = 0;
    size_t total_elements = out_gpu.size() + lse_gpu.size();

    auto check = [&](const std::vector<float>& gpu, const std::vector<float>& ref, float big_thr) {
        for (size_t i = 0; i < gpu.size(); ++i) {
            if (std::isnan(gpu[i])) { m.nan_count++; continue; }
            if (std::isinf(gpu[i])) { m.inf_count++; continue; }
            float d = fabsf(gpu[i] - ref[i]);
            m.max_diff = std::max(m.max_diff, d);
            sum += d;
            if (d > big_thr) m.big_diff_count++;
        }
    };

    check(out_gpu, out_ref, 0.01f);
    check(lse_gpu, lse_ref, 0.05f);

    if (total_elements > 0) m.avg_diff = (float)(sum / total_elements);
    return m;
}

// ======================================================================================
// TEST HARNESS
// ======================================================================================
template<int D, bool IS_CAUSAL>
bool run_test_fwd(int B, int H, int M, int N, float scale) {
    using Config = KernelConfig<D>;
    printf("  D=%-3d BN=%-3d BM=%-2d W=%-2d causal=%d ",
           D, Config::DO::BLOCK_N, Config::DO::BLOCK_M, WARPS, IS_CAUSAL);

    size_t qsz = (size_t)B*H*M*D, ksz = (size_t)B*H*N*D, osz = qsz, lsz = (size_t)B*H*M;

    std::vector<float> hQ(qsz), hK(ksz), hV(ksz);
    std::vector<float> hO_ref(osz), hLSE_ref(lsz);
    std::vector<float> hO_gpu(osz), hLSE_gpu(lsz);

    srand(42);
    for (auto& v : hQ) v = ((float)rand() / RAND_MAX - 0.5f) * 0.1f;
    for (auto& v : hK) v = ((float)rand() / RAND_MAX - 0.5f) * 0.1f;
    for (auto& v : hV) v = ((float)rand() / RAND_MAX - 0.5f) * 0.1f;

    bool use_cpu_ref = (M <= 1024 && N <= 1024);

    cpu_ref_fwd(hQ.data(), hK.data(), hV.data(), hO_ref.data(), hLSE_ref.data(),
                B, H, M, N, D, scale, IS_CAUSAL);

    __half *dQ, *dK, *dV, *dOut; float *dLSE;
    cudaMalloc(&dQ,   qsz*2);
    cudaMalloc(&dK,   ksz*2);
    cudaMalloc(&dV,   ksz*2);
    cudaMalloc(&dOut, osz*2);
    cudaMalloc(&dLSE, lsz*4);

    std::vector<__half> hQ_h(qsz), hK_h(ksz), hV_h(ksz);
    for (size_t i = 0; i < qsz; ++i) hQ_h[i] = __float2half(hQ[i]);
    for (size_t i = 0; i < ksz; ++i) { hK_h[i] = __float2half(hK[i]); hV_h[i] = __float2half(hV[i]); }

    cudaMemcpy(dQ, hQ_h.data(), qsz*2, cudaMemcpyHostToDevice);
    cudaMemcpy(dK, hK_h.data(), ksz*2, cudaMemcpyHostToDevice);
    cudaMemcpy(dV, hV_h.data(), ksz*2, cudaMemcpyHostToDevice);

    const size_t smem_bytes = Config::TOTAL_SMEM;
    auto kern = flash_attention_forward_kernel<D, IS_CAUSAL, false, false, false, false>;
    cudaFuncSetAttribute(kern, cudaFuncAttributeMaxDynamicSharedMemorySize, smem_bytes);

    const int grid_dq  = (M + Config::DO::BLOCK_M - 1) / Config::DO::BLOCK_M;
    dim3 grid(grid_dq, 1, B * H);
    dim3 block(Config::THREADS_PER_BLOCK);

    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    cudaEventRecord(start, 0);
    kern<<<grid, block, smem_bytes>>>(
        dQ, dK, dV, dOut, dLSE, /*dmask=*/nullptr,
        B, H, H, M, N,
        scale, 0.0f, nullptr, 0,
        -1, -1,
        0.0f, 0, 0
    );
    cudaEventRecord(stop, 0);
    cudaEventSynchronize(stop);

    float milliseconds = 0;
    cudaEventElapsedTime(&milliseconds, start, stop);
    cudaEventDestroy(start);
    cudaEventDestroy(stop);

    std::vector<__half> hOut_h(osz);
    cudaMemcpy(hOut_h.data(), dOut, osz*2, cudaMemcpyDeviceToHost);
    cudaMemcpy(hLSE_gpu.data(), dLSE, lsz*4, cudaMemcpyDeviceToHost);

    for (size_t i = 0; i < osz; ++i) hO_gpu[i] = __half2float(hOut_h[i]);

    StabilityMetrics m = {0,0,0,0,0};

    if (use_cpu_ref) {
        m = compute_stability_fwd(hO_gpu, hO_ref, hLSE_gpu, hLSE_ref);
    } else {
        for (size_t i = 0; i < osz; ++i) {
            if (std::isnan(hO_gpu[i])) m.nan_count++;
            if (std::isinf(hO_gpu[i])) m.inf_count++;
        }
        for (size_t i = 0; i < lsz; ++i) {
            if (std::isnan(hLSE_gpu[i])) m.nan_count++;
            if (std::isinf(hLSE_gpu[i])) m.inf_count++;
        }
    }

    bool pass = (m.nan_count == 0) && (m.inf_count == 0);
    if (use_cpu_ref) {
        int allowed_big = std::max(1, (int)((osz + lsz) * 0.001f));
        pass = pass && (m.max_diff < 0.05f) && (m.big_diff_count <= allowed_big);
    }

    printf("%s  max(O,LSE)=%.2e  avg(O,LSE)=%.2e  NaN=%d Inf=%d big=%d GPU=%.4f ms\n",
           pass ? "✅" : "❌",
           m.max_diff, m.avg_diff,
           m.nan_count, m.inf_count, m.big_diff_count,
           milliseconds);

    cudaFree(dQ); cudaFree(dK); cudaFree(dV); cudaFree(dOut); cudaFree(dLSE);
    return pass;
}

bool dispatch_test_fwd(int B, int H, int M, int N, int D, bool is_causal) {
    float scale = 1.0f / sqrtf((float)D);
    switch (D) {
        case 16:  return is_causal ? run_test_fwd<16, true>(B, H, M, N, scale)  : run_test_fwd<16, false>(B, H, M, N, scale);
        case 32:  return is_causal ? run_test_fwd<32, true>(B, H, M, N, scale)  : run_test_fwd<32, false>(B, H, M, N, scale);
        case 64:  return is_causal ? run_test_fwd<64, true>(B, H, M, N, scale)  : run_test_fwd<64, false>(B, H, M, N, scale);
        case 128: return is_causal ? run_test_fwd<128, true>(B, H, M, N, scale) : run_test_fwd<128, false>(B, H, M, N, scale);
        case 256: return is_causal ? run_test_fwd<256, true>(B, H, M, N, scale) : run_test_fwd<256, false>(B, H, M, N, scale);
        default:
            printf("  Unsupported D=%d\n", D);
            return false;
    }
}

int main() {
    printf("[ FA2 Forward ]:\n");
    int ok = 0, total = 0;
    auto T = [&](auto r) { total++; if (r) ok++; };

    std::vector<std::tuple<int, int, int, int, int>> test_cases = {
        {1, 1, 16, 16, 16},
        {1, 1, 32, 32, 32},
        {1, 1, 64, 64, 64},
        {1, 1, 128, 128, 128},
        {1, 1, 256, 256, 256},
        {1, 1, 1024, 1024, 128},
        {1, 1, 2048, 2048, 128},
        {1, 1, 4096, 4096, 128},
    };

    for (const auto& tc : test_cases) {
        auto [B, H, M, N, D] = tc;
        T(dispatch_test_fwd(B, H, M, N, D, false));
        T(dispatch_test_fwd(B, H, M, N, D, true));
    }

    printf("\n%d/%d tests passed\n", ok, total);
    return (ok == total) ? 0 : 1;
}