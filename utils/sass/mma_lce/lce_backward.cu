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

#include "kernel.h"
#include "entropy.h"
#include "swizzle.h"
#include "gemm_smem.h"
#ifdef MMA_NATIVE
    #include <mma.h>
    using namespace nvcuda::wmma;
#else
    #include "mma_m16n16k16.h"
    using namespace volta::wmma;
#endif

// ======================================================================================
// KERNEL 1: dLogits[M, V_pad] = SoftmaxDerivative(Input @ Weightᵀ)
//           Register-resident epilogue (no sLogits SMEM round-trip)
// ======================================================================================
template<typename Config>
__global__ void __launch_bounds__(Config::THREADS_PER_BLOCK, 1)
backward_lce_dlogits(
    const __half* __restrict__ Input,
    const __half* __restrict__ Weight,
    const   int*  __restrict__ Target,
    const float*  __restrict__ LSE,
    const float*  __restrict__ GradOut,
          __half* __restrict__ dLogits,
    int M,
    int V,
    int V_pad,
    int D,
    int reduction,
    float logit_scale,
    float label_smooth,
    float lse_sscale,
    int ignore_index
) {
    constexpr int BLOCK_M         = Config::BLOCK_M;
    constexpr int BLOCK_N         = Config::BLOCK_N;
    constexpr int DIM             = Config::DIM;
    constexpr int WARPS_PER_BLOCK = Config::WARPS_PER_BLOCK;

    // ==================================================================================
    // Grid Mapping: X for M-blocks, Y for V-blocks
    // ==================================================================================
    const int start_m = blockIdx.x * BLOCK_M;
    const int start_v = blockIdx.y * BLOCK_N;

    if (start_m >= M || start_v >= V) return;

    const int valid_m = min(BLOCK_M, M - start_m);
    const int valid_v = min(BLOCK_N, V - start_v);

    // ==================================================================================
    // Init:   thread/warp/lane IDs for WMMA coordination
    // ==================================================================================
    const int tid     = threadIdx.x;
    const int warp_id = tid >> 5;
    const int lane_id = tid & 31;

    // ==================================================================================
    // Layout:
    //   Input:      [M, D]              (row-major, offset via start_m)
    //   Weight:     [V, D]              (row-major, offset via start_v)
    //   dLogits:    [M, V_pad]          (row-major)
    // ==================================================================================
    const __half* __restrict__ input_ptr   = Input   + (size_t)start_m * D;
    const __half* __restrict__ weight_ptr  = Weight  + (size_t)start_v * D;

    // ==================================================================================
    // Init:   shared memory (only for Input and Weight tiles, sLogits is eliminated)
    // ==================================================================================
    extern __shared__ char smem_raw[];
    auto& smem = *reinterpret_cast<typename Config::SmemLayout*>(smem_raw);

    __half* __restrict__ sInput  = smem.input;
    __half* __restrict__ sWeight = smem.weight;

    constexpr int num_tiles_m    = (BLOCK_M + WMMA_M - 1) / WMMA_M;
    constexpr int num_tiles_n    = (BLOCK_N + WMMA_N - 1) / WMMA_N;
    constexpr int num_tiles_k    = DIM / WMMA_K;
    constexpr int total_tiles    = num_tiles_m * num_tiles_n;
    constexpr int tiles_per_warp = (total_tiles + WARPS_PER_BLOCK - 1) / WARPS_PER_BLOCK;

    fragment<matrix_a,    WMMA_M, WMMA_N, WMMA_K, half, row_major> a_frag[tiles_per_warp];
    fragment<matrix_b,    WMMA_M, WMMA_N, WMMA_K, half, col_major> b_frag[tiles_per_warp];
    fragment<accumulator, WMMA_M, WMMA_N, WMMA_K, float>         acc_frag[tiles_per_warp];

    #pragma unroll
    for (int idx = 0; idx < tiles_per_warp; ++idx)
        fill_fragment(acc_frag[idx], 0.0f);

    // ==================================================================================
    // MAIN LOOP (iterates over D-tiles)
    // ==================================================================================
    for (int k_tile = 0; k_tile < D; k_tile += DIM) {
        WMMA_GEMM_LOAD_TILE<Config, GemmPhase::LCDX, Stage::SNGL>(
          input_ptr  + k_tile, sInput,
          nullptr, nullptr,
          D, valid_m, tid);
        WMMA_GEMM_LOAD_TILE<Config, GemmPhase::LCDX, Stage::SNGL>(
          weight_ptr + k_tile, sWeight,
          nullptr, nullptr,
          D, valid_v, tid);
        __syncthreads();

        #pragma unroll
        for (int idx = 0; idx < tiles_per_warp; ++idx) {
            const int tile_index = warp_id + idx * WARPS_PER_BLOCK;
            if (tile_index >= total_tiles) break;

            const int tile_m = (tile_index / num_tiles_n) * WMMA_M;
            const int tile_n = (tile_index % num_tiles_n) * WMMA_N;
            if (tile_m >= valid_m || tile_n >= valid_v) continue;

            #pragma unroll
            for (int k = 0; k < num_tiles_k; ++k) {
                load_matrix_sync(a_frag[idx], sInput  + tile_m * DIM + (k * WMMA_K), DIM);
                load_matrix_sync(b_frag[idx], sWeight + tile_n * DIM + (k * WMMA_K), DIM);
                mma_sync(acc_frag[idx], a_frag[idx], b_frag[idx], acc_frag[idx]);
            }
        }
        __syncthreads();
    }

    // ==================================================================================
    // Epilogue: Compute Softmax Derivative and store to global memory directly from registers
    // ==================================================================================
    #pragma unroll
    for (int idx = 0; idx < tiles_per_warp; ++idx) {
        const int tile_index = warp_id + idx * WARPS_PER_BLOCK;
        if (tile_index >= total_tiles) break;

        const int wm = tile_index / num_tiles_n;
        const int wn = tile_index % num_tiles_n;
        const int tile_m = wm * WMMA_M;
        const int tile_n = wn * WMMA_N;

        const int r_base = ((lane_id >> 2) & 1) * 8 + ((lane_id >> 4) & 1) * 4 + (lane_id & 1);
        const int c_base = ((lane_id >> 3) & 1) * 8 + ((lane_id >> 1) & 1) * 2;

        const int r1 = tile_m + r_base;
        const int r2 = r1 + 2;
        const int c0 = tile_n + c_base;

        const float* __restrict__ acc = acc_frag[idx].x;

        auto process_row = [&](int r_local, int r_global, float acc0, float acc1, float acc4, float acc5) {
            if (r_local < valid_m && r_global < M) {
                const float lse           = LSE[r_global];
                const int   target        = Target[r_global];
                const bool  ignore_row    = (target == ignore_index);
                const float grad_scale    = (reduction == 0) ? GradOut[r_global] : GradOut[0];
                const float scale_factor  = ignore_row ? 0.0f : grad_scale * logit_scale;
                const float uniform_prob  = label_smooth / static_cast<float>(V);
                const float target_adjust = -(1.0f - label_smooth);
                const float z_loss_factor = 1.0f + 2.0f * lse_sscale * lse;

                float p0 = __expf(fmaxf(acc0 * logit_scale - lse, -80.0f));
                float p1 = __expf(fmaxf(acc1 * logit_scale - lse, -80.0f));
                float p4 = __expf(fmaxf(acc4 * logit_scale - lse, -80.0f));
                float p5 = __expf(fmaxf(acc5 * logit_scale - lse, -80.0f));

                int gc0 = start_v + c0;
                int gc1 = gc0 + 1;
                int gc4 = gc0 + 4;
                int gc5 = gc0 + 5;

                p0 = (p0 * z_loss_factor - uniform_prob) + ((gc0 == target && !ignore_row) ? target_adjust : 0.0f);
                p1 = (p1 * z_loss_factor - uniform_prob) + ((gc1 == target && !ignore_row) ? target_adjust : 0.0f);
                p4 = (p4 * z_loss_factor - uniform_prob) + ((gc4 == target && !ignore_row) ? target_adjust : 0.0f);
                p5 = (p5 * z_loss_factor - uniform_prob) + ((gc5 == target && !ignore_row) ? target_adjust : 0.0f);

                p0 *= scale_factor;
                p1 *= scale_factor;
                p4 *= scale_factor;
                p5 *= scale_factor;

                __half* dst_row = dLogits + (size_t)r_global * V_pad;

                if (gc1 < V) {
                    *reinterpret_cast<__half2*>(dst_row + gc0) = __float22half2_rn(make_float2(p0, p1));
                } else if (gc0 < V) {
                    dst_row[gc0] = __float2half_rn(p0);
                }

                if (gc5 < V) {
                    *reinterpret_cast<__half2*>(dst_row + gc4) = __float22half2_rn(make_float2(p4, p5));
                } else if (gc4 < V) {
                    dst_row[gc4] = __float2half_rn(p4);
                }
            }
        };

        process_row(r1, start_m + r1, acc[0], acc[1], acc[4], acc[5]);
        process_row(r2, start_m + r2, acc[2], acc[3], acc[6], acc[7]);
    }
}

// ======================================================================================
// KERNEL 2/3: dX = dLogits @ W      (A_LAYOUT = row_major)
//             dW = dLogitsᵀ @ Input (A_LAYOUT = col_major)
// ======================================================================================
template<GemmType TYPE, typename Config>
__global__ void __launch_bounds__(Config::THREADS_PER_BLOCK, 2)
backward_lce_dxw(
    const __half* __restrict__ SMEM_A,
    const __half* __restrict__ SMEM_B,
          __half* __restrict__ SMEM_C,
    int M,
    int N,
    int K,
    int A_STRIDE,
    int B_STRIDE,
    int C_STRIDE
) {
    constexpr int BLOCK_M           = Config::BLOCK_M;
    constexpr int BLOCK_N           = Config::BLOCK_N;
    constexpr int DIM               = Config::DIM;
    constexpr int GRID_X            = Config::GRID_X;
    constexpr int GRID_Y            = Config::GRID_Y;
    constexpr int THREADS_PER_BLOCK = Config::THREADS_PER_BLOCK;

    // ==================================================================================
    // Grid Mapping: X for M-blocks, Y for N-blocks
    // ==================================================================================
    const int block_m = blockIdx.x * BLOCK_M;
    const int block_n = blockIdx.y * BLOCK_N;

    if (block_m >= M || block_n >= N) return;

    const int valid_m = min(BLOCK_M, M - block_m);
    const int valid_n = min(BLOCK_N, N - block_n);

    if (valid_m <= 0 || valid_n <= 0) return;

    // ==================================================================================
    // Init:   thread/warp/lane IDs for WMMA coordination
    // ==================================================================================
    const int tid     = threadIdx.x;
    const int warp_id = tid >> 5;
    const int warp_m  = warp_id / GRID_Y;
    const int warp_n  = warp_id % GRID_Y;

    // ==================================================================================
    // Init:   shared memory with zero-fill union regions to avoid stale data
    // ==================================================================================
    extern __shared__ char smem_raw[];

    auto& smem = *reinterpret_cast<typename Config::SmemLayout*>(smem_raw);

    __half* __restrict__ sdL = smem.TdLogits;
    __half* __restrict__ swI = smem.TweightInput;

    // ==================================================================================
    // Compute:  WMMA accumulation (sdL @ swI)
    // ==================================================================================
    constexpr int num_tiles_m    = (BLOCK_M / GRID_X) / WMMA_M;
    constexpr int num_tiles_n    = (BLOCK_N / GRID_Y) / WMMA_N;
    constexpr int tiles_per_warp = (DIM + WMMA_K - 1) / WMMA_K;

    using GTraits  = GemmTraits<TYPE, GemmPhase::LCDX>;
    using A_LAYOUT = std::conditional_t<GTraits::A_IS_COL, col_major, row_major>;

    fragment<matrix_a,    WMMA_M, WMMA_N, WMMA_K, half,  A_LAYOUT>  a_frag;
    fragment<matrix_b,    WMMA_M, WMMA_N, WMMA_K, half,  row_major> b_frag;
    fragment<accumulator, WMMA_M, WMMA_N, WMMA_K, float> acc[num_tiles_m][num_tiles_n];

    #pragma unroll
    for (int idx = 0; idx < num_tiles_m; ++idx)
        #pragma unroll
        for (int jdx = 0; jdx < num_tiles_n; ++jdx)
            fill_fragment(acc[idx][jdx], 0.0f);

    // ==================================================================================
    // MAIN LOOP (iterates over K-dimension tiles)
    // ==================================================================================
    for (int k_block = 0; k_block < ((K + DIM - 1) / DIM); ++k_block) {
        const int k_offset = k_block * DIM;
        const int k_valid  = min(DIM, K - k_offset);

        if constexpr (GTraits::A_IS_COL) {
            WMMA_GEMM_LOAD_TILE<Config, GemmPhase::LCDX, Stage::SNGL>(
              SMEM_A + k_offset * A_STRIDE + block_m, sdL,
              nullptr, nullptr,
              A_STRIDE, k_valid, tid);
        } else {
            WMMA_GEMM_LOAD_TILE<Config, GemmPhase::LCDX, Stage::SNGL>(
              SMEM_A + block_m * A_STRIDE + k_offset, sdL,
              nullptr, nullptr,
              A_STRIDE, valid_m, tid);
        }
        WMMA_GEMM_LOAD_TILE<Config, GemmPhase::LCDX, Stage::TRNSP>(
          SMEM_B + k_offset * B_STRIDE + block_n, swI,
          nullptr, nullptr,
          B_STRIDE, k_valid, tid);

        // Swizzled zero-fill for K-tails
        if (k_valid < DIM) {
            if constexpr (GTraits::A_IS_COL) {
                const int total_a = (DIM - k_valid) * BLOCK_M;
                for (int idx = tid; idx < total_a; idx += THREADS_PER_BLOCK) {
                    const int k = k_valid + (idx / BLOCK_M);
                    const int m = idx % BLOCK_M;
                    const uint32_t addr = __cvta_generic_to_shared(&sdL[k * BLOCK_M + m]);
                    st_half(addr, __float2half(0.0f), k);
                }
            } else {
                const int total_a = BLOCK_M * (DIM - k_valid);
                for (int idx = tid; idx < total_a; idx += THREADS_PER_BLOCK) {
                    const int m = idx / (DIM - k_valid);
                    const int k = k_valid + (idx % (DIM - k_valid));
                    const uint32_t addr = __cvta_generic_to_shared(&sdL[m * DIM + k]);
                    st_half(addr, __float2half(0.0f), m);
                }
            }
            const int total_b = (DIM - k_valid) * BLOCK_N;
            for (int idx = tid; idx < total_b; idx += THREADS_PER_BLOCK) {
                const int k = k_valid + (idx / BLOCK_N);
                const int n = idx % BLOCK_N;
                const uint32_t addr = __cvta_generic_to_shared(&swI[k * BLOCK_N + n]);
                st_half(addr, __float2half(0.0f), k);
            }
        }
        __syncthreads();

        #pragma unroll
        for (int k_step = 0; k_step < tiles_per_warp; ++k_step) {
            #pragma unroll
            for (int idx = 0; idx < num_tiles_m; ++idx) {

                const int a_offset = GTraits::A_IS_COL
                    ? (k_step * WMMA_K) * BLOCK_M + ((warp_m * num_tiles_m + idx) * WMMA_M)
                    : ((warp_m * num_tiles_m + idx) * WMMA_M) * DIM + (k_step * WMMA_K);

                load_matrix_sync(a_frag, sdL + a_offset, GTraits::A_IS_COL ? BLOCK_M : DIM);

                #pragma unroll
                for (int jdx = 0; jdx < num_tiles_n; ++jdx) {
                    load_matrix_sync(b_frag, swI + (k_step * WMMA_K) * BLOCK_N + ((warp_n * num_tiles_n + jdx) * WMMA_N), BLOCK_N);
                    mma_sync(acc[idx][jdx], a_frag, b_frag, acc[idx][jdx]);
                }
            }
        }
        __syncthreads();
    }

    // ==================================================================================
    // Epilogue: Store accumulated fragments -> global C
    // ==================================================================================
    const int accum_row = (tid & 1) | (((tid >> 4) & 1) << 2) | (((tid >> 2) & 1) << 3);
    const int accum_col = (((tid >> 1) & 1) << 1) | (((tid >> 3) & 1) << 3);

    #pragma unroll
    for (int idx = 0; idx < num_tiles_m; ++idx) {
        #pragma unroll
        for (int jdx = 0; jdx < num_tiles_n; ++jdx) {
            const int global_m_base = block_m + (warp_m * num_tiles_m + idx) * WMMA_M;
            const int global_n_base = block_n + (warp_n * num_tiles_n + jdx) * WMMA_N;

            if (global_m_base >= M || global_n_base >= N) continue;

            #pragma unroll
            for (int e = 0; e < 8; e += 2) {
                const int global_m = global_m_base + (accum_row | (((e >> 1) & 1) << 1));
                const int global_n = global_n_base + (accum_col | (((e >> 2) & 1) << 2));

                if (global_m >= M) continue;

                const float value_lo = acc[idx][jdx].x[e];
                const float value_hi = acc[idx][jdx].x[e + 1];

                __half* dst = &SMEM_C[(size_t)global_m * C_STRIDE + global_n];

                if (global_n + 1 < N) {
                    *reinterpret_cast<__half2*>(dst) = __float22half2_rn(make_float2(value_lo, value_hi));
                } else {
                    if (global_n     < N) dst[0] = __float2half_rn(value_lo);
                    if (global_n + 1 < N) dst[1] = __float2half_rn(value_hi);
                }
            }
        }
    }
}

// ======================================================================================
// CPU REFERENCE & METRICS
// ======================================================================================
void cpu_ref_lce_backward(const float* Input, const float* Weight, const int* Target,
                          const float* LSE, const float* GradOut,
                          float* GradInput, float* GradWeight,
                          int M, int V, int D, int reduction,
                          float logit_scale, float label_smooth, float lse_sscale, int ignore_index) {
    std::vector<float> dLogits((size_t)M * V, 0.0f);

    for (int i = 0; i < M; ++i) {
        float lse = LSE[i];
        float grad_scale = (reduction == 0) ? GradOut[i] : GradOut[0];
        int target = Target[i];
        bool ignore_row = (target == ignore_index);
        if (ignore_row) grad_scale = 0.0f;

        float uniform_prob = label_smooth / (float)V;
        float target_adjust = -(1.0f - label_smooth);
        float z_loss_factor = 1.0f + 2.0f * lse_sscale * lse;

        for (int v = 0; v < V; ++v) {
            float logit = 0.0f;
            for (int d = 0; d < D; ++d) logit += Input[i * D + d] * Weight[v * D + d];
            logit *= logit_scale;

            float p = expf(logit - lse);
            float dl = (p * z_loss_factor - uniform_prob) + ((v == target && !ignore_row) ? target_adjust : 0.0f);
            dl *= grad_scale * logit_scale;
            dLogits[i * V + v] = dl;
        }

        for (int d = 0; d < D; ++d) {
            float gi = 0.0f;
            for (int v = 0; v < V; ++v) {
                gi += dLogits[i * V + v] * Weight[v * D + d];
            }
            GradInput[i * D + d] = gi;
        }
    }

    for (int v = 0; v < V; ++v) {
        for (int d = 0; d < D; ++d) {
            float gw = 0.0f;
            for (int i = 0; i < M; ++i) {
                gw += dLogits[i * V + v] * Input[i * D + d];
            }
            GradWeight[v * D + d] = gw;
        }
    }
}

struct StabilityMetrics {
    float max_diff, avg_diff;
    int nan_count, inf_count, big_diff_count;
};

// ======================================================================================
// TEST HARNESS
// ======================================================================================
template<typename LogitsConf, typename dXWConf>
bool run_test_lce_backward(int M, int V, int D, float logit_scale, float label_smooth, float lse_sscale, int ignore_index) {
    constexpr int DL_BLOCK_M = LogitsConf::BLOCK_M;
    constexpr int DL_BLOCK_N = LogitsConf::BLOCK_N;
    constexpr size_t DL_SMEM = LogitsConf::TOTAL_SMEM;

    constexpr int GXW_BLOCK_M = dXWConf::BLOCK_M;
    constexpr int GXW_BLOCK_N = dXWConf::BLOCK_N;
    constexpr size_t GXW_SMEM = dXWConf::TOTAL_SMEM;

    printf("  M=%-5d V=%-5d D=%-4d BM1=%-3d BN1=%-3d BM2=%-3d BN2=%-3d ", M, V, D, DL_BLOCK_M, DL_BLOCK_N, GXW_BLOCK_M, GXW_BLOCK_N);

    size_t isz = (size_t)M * D, wsz = (size_t)V * D, tsz = (size_t)M;
    size_t lsz = (size_t)M, gosz = (size_t)M;
    int V_pad = ((V + GXW_BLOCK_M - 1) / GXW_BLOCK_M) * GXW_BLOCK_M;
    size_t dlsz = (size_t)M * V_pad;

    std::vector<float> hInput(isz), hWeight(wsz);
    std::vector<int>   hTarget(tsz);
    std::vector<float> hLSE(lsz);
    std::vector<float> hGradOut(gosz);

    std::vector<float> hGradInput_ref(isz), hGradWeight_ref(wsz);
    std::vector<float> hGradInput_gpu(isz), hGradWeight_gpu(wsz);

    srand(42);
    for (auto& v : hInput) v = ((float)rand() / RAND_MAX - 0.5f) * 0.1f;
    for (auto& v : hWeight) v = ((float)rand() / RAND_MAX - 0.5f) * 0.1f;
    for (auto& v : hTarget) v = rand() % (V + 10);
    for (auto& v : hGradOut) v = ((float)rand() / RAND_MAX - 0.5f) * 0.1f;

    // Compute LSE on CPU using simple loop
    for (int i = 0; i < M; ++i) {
        float max_logit = NEG_INF;
        std::vector<float> logits(V);
        for (int v = 0; v < V; ++v) {
            float sum = 0;
            for (int d = 0; d < D; ++d) sum += hInput[i * D + d] * hWeight[v * D + d];
            logits[v] = sum * logit_scale;
            max_logit = std::max(max_logit, logits[v]);
        }
        float sum_exp = 0;
        for (int v = 0; v < V; ++v) sum_exp += expf(logits[v] - max_logit);
        hLSE[i] = max_logit + logf(fmaxf(sum_exp, 1e-24f));
    }

    bool use_cpu_ref = (M <= 256 && V <= 2048 && D <= 256);

    if (use_cpu_ref) {
        cpu_ref_lce_backward(hInput.data(), hWeight.data(), hTarget.data(), hLSE.data(), hGradOut.data(),
                             hGradInput_ref.data(), hGradWeight_ref.data(), M, V, D, 0, logit_scale, label_smooth, lse_sscale, ignore_index);
    }

    __half *dInput, *dWeight, *dLogits, *dGradInput, *dGradWeight; 
    int    *dTarget;
    float  *dLSE, *dGradOut;

    cudaMalloc(&dInput, isz * 2);
    cudaMalloc(&dWeight, wsz * 2);
    cudaMalloc(&dTarget, tsz * 4);
    cudaMalloc(&dLSE, lsz * 4);
    cudaMalloc(&dGradOut, gosz * 4);
    cudaMalloc(&dLogits, dlsz * 2);
    cudaMalloc(&dGradInput, isz * 2);
    cudaMalloc(&dGradWeight, wsz * 2);

    std::vector<__half> hInput_h(isz), hWeight_h(wsz), hLogits_h(dlsz, __float2half(0.0f));
    for (size_t i = 0; i < isz; ++i) hInput_h[i] = __float2half(hInput[i]);
    for (size_t i = 0; i < wsz; ++i) hWeight_h[i] = __float2half(hWeight[i]);

    cudaMemcpy(dInput, hInput_h.data(), isz * 2, cudaMemcpyHostToDevice);
    cudaMemcpy(dWeight, hWeight_h.data(), wsz * 2, cudaMemcpyHostToDevice);
    cudaMemcpy(dTarget, hTarget.data(), tsz * 4, cudaMemcpyHostToDevice);
    cudaMemcpy(dLSE, hLSE.data(), lsz * 4, cudaMemcpyHostToDevice);
    cudaMemcpy(dGradOut, hGradOut.data(), gosz * 4, cudaMemcpyHostToDevice);
    cudaMemset(dLogits, 0, dlsz * 2);
    cudaMemset(dGradInput, 0, isz * 2);
    cudaMemset(dGradWeight, 0, wsz * 2);

    cudaFuncSetAttribute(backward_lce_dlogits<LogitsConf>, cudaFuncAttributeMaxDynamicSharedMemorySize, DL_SMEM);
    cudaFuncSetAttribute(backward_lce_dxw<GemmType::dL_W, dXWConf>, cudaFuncAttributeMaxDynamicSharedMemorySize, GXW_SMEM);
    cudaFuncSetAttribute(backward_lce_dxw<GemmType::dLT_I, dXWConf>, cudaFuncAttributeMaxDynamicSharedMemorySize, GXW_SMEM);

    dim3 grid_logits((M + DL_BLOCK_M - 1) / DL_BLOCK_M, (V + DL_BLOCK_N - 1) / DL_BLOCK_N);
    dim3 threads_logits(LogitsConf::THREADS_PER_BLOCK);

    dim3 grid_dX((M + GXW_BLOCK_M - 1) / GXW_BLOCK_M, (D + GXW_BLOCK_N - 1) / GXW_BLOCK_N);
    dim3 threads_dX(dXWConf::THREADS_PER_BLOCK);

    dim3 grid_dW((V + GXW_BLOCK_M - 1) / GXW_BLOCK_M, (D + GXW_BLOCK_N - 1) / GXW_BLOCK_N);
    dim3 threads_dW(dXWConf::THREADS_PER_BLOCK);

    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    cudaEventRecord(start, 0);

    backward_lce_dlogits<LogitsConf><<<grid_logits, threads_logits, DL_SMEM>>>(
        dInput, dWeight, dTarget, dLSE, dGradOut, dLogits,
        M, V, V_pad, D, 0, logit_scale, label_smooth, lse_sscale, ignore_index
    );

    backward_lce_dxw<GemmType::dL_W, dXWConf><<<grid_dX, threads_dX, GXW_SMEM>>>(
        dLogits, dWeight, dGradInput, M, D, V, V_pad, D, D
    );

    backward_lce_dxw<GemmType::dLT_I, dXWConf><<<grid_dW, threads_dW, GXW_SMEM>>>(
        dLogits, dInput, dGradWeight, V, D, M, V_pad, D, D
    );

    cudaEventRecord(stop, 0);
    cudaEventSynchronize(stop);
    float milliseconds = 0;
    cudaEventElapsedTime(&milliseconds, start, stop);
    cudaEventDestroy(start);
    cudaEventDestroy(stop);

    std::vector<__half> hGI_h(isz), hGW_h(wsz);
    cudaMemcpy(hGI_h.data(), dGradInput, isz * 2, cudaMemcpyDeviceToHost);
    cudaMemcpy(hGW_h.data(), dGradWeight, wsz * 2, cudaMemcpyDeviceToHost);

    for (size_t i = 0; i < isz; ++i) hGradInput_gpu[i] = __half2float(hGI_h[i]);
    for (size_t i = 0; i < wsz; ++i) hGradWeight_gpu[i] = __half2float(hGW_h[i]);

    StabilityMetrics m = {0,0,0,0,0};
    if (use_cpu_ref) {
        auto check = [&](const std::vector<float>& gpu, const std::vector<float>& ref, float big_thr) {
            for (size_t i = 0; i < gpu.size(); ++i) {
                if (std::isnan(gpu[i])) { m.nan_count++; continue; }
                if (std::isinf(gpu[i])) { m.inf_count++; continue; }
                float d = fabsf(gpu[i] - ref[i]);
                m.max_diff = std::max(m.max_diff, d);
                m.avg_diff += d;
                if (d > big_thr) m.big_diff_count++;
            }
        };
        check(hGradInput_gpu, hGradInput_ref, 0.01f);
        check(hGradWeight_gpu, hGradWeight_ref, 0.01f);
        size_t total_elements = isz + wsz;
        if (total_elements > 0) m.avg_diff = m.avg_diff / total_elements;
    } else {
        for (size_t i = 0; i < isz; ++i) {
            if (std::isnan(hGradInput_gpu[i])) m.nan_count++;
            if (std::isinf(hGradInput_gpu[i])) m.inf_count++;
        }
        for (size_t i = 0; i < wsz; ++i) {
            if (std::isnan(hGradWeight_gpu[i])) m.nan_count++;
            if (std::isinf(hGradWeight_gpu[i])) m.inf_count++;
        }
    }

    bool pass = (m.nan_count == 0) && (m.inf_count == 0);
    if (use_cpu_ref) {
        int allowed_big = std::max(1, (int)((isz + wsz) * 0.001f));
        pass = pass && (m.max_diff < 0.05f) && (m.big_diff_count <= allowed_big);
    }

    printf("%s  max(dI,dW)=%.2e  avg=%.2e  NaN=%d Inf=%d big=%d GPU=%.4f ms\n", pass ? "PASS" : "FAIL",
           m.max_diff, m.avg_diff, m.nan_count, m.inf_count, m.big_diff_count, milliseconds);

    cudaFree(dInput); cudaFree(dWeight); cudaFree(dTarget);
    cudaFree(dLSE); cudaFree(dGradOut); cudaFree(dLogits);
    cudaFree(dGradInput); cudaFree(dGradWeight);

    return pass;
}

bool dispatch_test_lce_backward(int M, int V, int D, float logit_scale, float label_smooth, float lse_sscale, int ignore_index) {
    using LogitsConf = LogitsConfig;
    using dXWConf = dXWConfig;
    return run_test_lce_backward<LogitsConf, dXWConf>(M, V, D, logit_scale, label_smooth, lse_sscale, ignore_index);
}

int main() {
    printf("[ LCE Backward ]:\n");
    int ok = 0, total = 0;
    auto T = [&](auto r) { total++; if (r) ok++; };

    std::vector<std::tuple<int, int, int, float, float, float, int>> test_cases = {
        {16,   128,  128, 1.0f, 0.0f, 0.0f, -1},
        {64,   512,  128, 1.0f, 0.0f, 0.0f, -1},
        {128,  1024, 128, 1.0f, 0.1f, 0.0f, -1},
        {256,  2048, 128, 1.0f, 0.1f, 1e-4f, -1},
        {512,  4096, 128, 1.0f, 0.0f, 0.0f, -1},
        {1024, 8192, 128, 1.0f, 0.0f, 0.0f, -1},
        {2048, 16384,128, 1.0f, 0.0f, 0.0f, -1},
        {128,  1024, 128, 1.0f, 0.1f, 1e-4f, 42},
    };

    for (const auto& tc : test_cases) {
        auto [M, V, D, ls, lsm, lses, ign] = tc;
        T(dispatch_test_lce_backward(M, V, D, ls, lsm, lses, ign));
    }

    printf("\n%d/%d tests passed\n", ok, total);
    return (ok == total) ? 0 : 1;
}
