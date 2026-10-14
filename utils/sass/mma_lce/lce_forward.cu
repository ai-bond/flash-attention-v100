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
// KERNEL 1: PartialMax/PartialSum/PartialSumLogits/TargetLogit = Reduce(Input @ Weightᵀ)
//           Register-resident reduction (no sLogits SMEM round-trip)
// ======================================================================================
template<typename Config>
__global__ void __launch_bounds__(LogitsConfig::THREADS_PER_BLOCK, 1)
forward_lce_partial(
    const __half* __restrict__ Input,
    const __half* __restrict__ Weight,
    const int*    __restrict__ Target,
          float*  __restrict__ PartialMax,
          float*  __restrict__ PartialSum,
          float*  __restrict__ PartialSumLogits,
          float*  __restrict__ TargetLogit,
    int   M,
    int   V,
    int   D,
    int   stride_v,
    float logit_scale,
    int   ignore_index
) {
    constexpr int BLOCK_M         = Config::BLOCK_M;
    constexpr int BLOCK_N         = Config::BLOCK_N;
    constexpr int DIM             = Config::DIM;
    constexpr int WARPS_PER_BLOCK = Config::WARPS_PER_BLOCK;
    constexpr int N_WN            = Config::N_WN;

    static_assert(WMMA_M == 16 && WMMA_N == 16, "Register reduction assumes m16n16k16 fragment layout");
    static_assert(WARPS_PER_BLOCK == (BLOCK_M / WMMA_M) * N_WN,
                  "Each warp must own exactly one 16x16 tile");

    // ==================================================================================
    // Grid Mapping
    // ==================================================================================
    const int start_m = blockIdx.x * BLOCK_M;
    const int v_tile  = blockIdx.y * BLOCK_N;
    if (start_m >= M || v_tile >= V) return;

    const int valid_m = min(BLOCK_M, M - start_m);
    const int valid_v = min(BLOCK_N, V - v_tile);

    const int tid     = threadIdx.x;
    const int warp_id = tid >> 5;
    const int lane_id = tid & 31;

    // ==================================================================================
    // Layout
    // ==================================================================================
    const __half* __restrict__ input_ptr  = Input  + (size_t)start_m * D;
    const __half* __restrict__ weight_ptr = Weight + (size_t)v_tile  * D;

    // ==================================================================================
    // Shared memory
    // ==================================================================================
    extern __shared__ char smem_raw[];
    auto& smem = *reinterpret_cast<typename Config::SmemLayout*>(smem_raw);

    __half* __restrict__ sInput   = smem.input;
    __half* __restrict__ sWeight  = smem.weight;
    float*  __restrict__ sPartMax = smem.part_max;
    float*  __restrict__ sPartSum = smem.part_sum;
    float*  __restrict__ sPartSL  = smem.part_sl;
    float*  __restrict__ sRowMax  = smem.row_max;
    float*  __restrict__ sRowSum  = smem.row_sum;
    float*  __restrict__ sRowSL   = smem.row_sl;

    const bool has_sl = (PartialSumLogits != nullptr);

    // ==================================================================================
    // FIX: Explicitly initialize all reduction buffers to prevent garbage reads
    // ==================================================================================
    constexpr int part_size = BLOCK_M * N_WN;
    constexpr int row_size  = BLOCK_M;
    
    if (tid < part_size) {
        sPartMax[tid] = NEG_INF;
        sPartSum[tid] = 0.0f;
        if (has_sl) sPartSL[tid] = 0.0f;
    }
    if (tid < row_size) {
        sRowMax[tid] = NEG_INF;
        sRowSum[tid] = 0.0f;
        if (has_sl) sRowSL[tid] = 0.0f;
    }
    __syncthreads();

    // ==================================================================================
    // GEMM: Logits = Input @ Weight^T  (same as before)
    // ==================================================================================
    constexpr int num_tiles_m    = (BLOCK_M + WMMA_M - 1) / WMMA_M;
    constexpr int num_tiles_n    = (BLOCK_N + WMMA_N - 1) / WMMA_N;
    constexpr int total_tiles    = num_tiles_m * num_tiles_n;
    constexpr int tiles_per_warp = (total_tiles + WARPS_PER_BLOCK - 1) / WARPS_PER_BLOCK;

    fragment<matrix_a,    WMMA_M, WMMA_N, WMMA_K, half, row_major> a_frag[tiles_per_warp];
    fragment<matrix_b,    WMMA_M, WMMA_N, WMMA_K, half, col_major> b_frag[tiles_per_warp];
    fragment<accumulator, WMMA_M, WMMA_N, WMMA_K, float>         acc_frag[tiles_per_warp];

    #pragma unroll
    for (int idx = 0; idx < tiles_per_warp; ++idx)
        fill_fragment(acc_frag[idx], 0.0f);

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
            const int tile_idx = warp_id + idx * WARPS_PER_BLOCK;
            if (tile_idx >= total_tiles) break;

            const int tile_m = (tile_idx / num_tiles_n) * WMMA_M;
            const int tile_n = (tile_idx % num_tiles_n) * WMMA_N;
            if (tile_m >= valid_m || tile_n >= valid_v) continue;

            #pragma unroll
            for (int k = 0; k < DIM / WMMA_K; ++k) {
                load_matrix_sync(a_frag[idx], sInput  + tile_m * DIM + (k * WMMA_K), DIM);
                load_matrix_sync(b_frag[idx], sWeight + tile_n * DIM + (k * WMMA_K), DIM);
                mma_sync(acc_frag[idx], a_frag[idx], b_frag[idx], acc_frag[idx]);
            }
        }
        __syncthreads();
    }

    // ==================================================================================
    // Register-resident reduction over acc_frag[0]
    // ----------------------------------------------------------------------------------
    // Warp-tile mapping: wm = warp_id / N_WN, wn = warp_id % N_WN
    // Lane fragment layout (m16n16k16):
    //   lr = lane_id >> 2  (0..7)   -> row offset within tile
    //   lc = lane_id &  3  (0..3)   -> column pair index within tile
    //   acc.x[0] -> (lr,   lc*2 + 0)
    //   acc.x[1] -> (lr,   lc*2 + 1)
    //   acc.x[2] -> (lr+8, lc*2 + 0)
    //   acc.x[3] -> (lr+8, lc*2 + 1)
    //   acc.x[4] -> (lr,   lc*2 + 8)
    //   acc.x[5] -> (lr,   lc*2 + 9)
    //   acc.x[6] -> (lr+8, lc*2 + 8)
    //   acc.x[7] -> (lr+8, lc*2 + 9)
    // ==================================================================================
    const int wm = warp_id / N_WN;
    const int wn = warp_id % N_WN;
    const int tile_m = wm * WMMA_M;
    const int tile_n = wn * WMMA_N;

    const int lr = lane_id >> 2;
    const int lc = lane_id & 3;

    const int row_lo = tile_m + lr;
    const int row_hi = tile_m + lr + 8;
    const int col_a  = tile_n + lc * 2;   // covers 0..7 inside tile
    const int col_b  = col_a  + 8;        // covers 8..15 inside tile

    const bool r_lo_ok = row_lo < valid_m;
    const bool r_hi_ok = row_hi < valid_m;
    const bool cA0 = (col_a + 0) < valid_v;
    const bool cA1 = (col_a + 1) < valid_v;
    const bool cB0 = (col_b + 0) < valid_v;
    const bool cB1 = (col_b + 1) < valid_v;

    const float* __restrict__ acc = acc_frag[0].x;

    // ---- Phase 1: intra-warp partial max + slogits over 16 columns of the tile ----
    float m_lo = NEG_INF, m_hi = NEG_INF;
    float sl_lo = 0.f,   sl_hi = 0.f;

    if (r_lo_ok) {
        if (cA0) { m_lo = fmaxf(m_lo, acc[0]); if (has_sl) sl_lo += acc[0]; }
        if (cA1) { m_lo = fmaxf(m_lo, acc[1]); if (has_sl) sl_lo += acc[1]; }
        if (cB0) { m_lo = fmaxf(m_lo, acc[4]); if (has_sl) sl_lo += acc[4]; }
        if (cB1) { m_lo = fmaxf(m_lo, acc[5]); if (has_sl) sl_lo += acc[5]; }
    }
    if (r_hi_ok) {
        if (cA0) { m_hi = fmaxf(m_hi, acc[2]); if (has_sl) sl_hi += acc[2]; }
        if (cA1) { m_hi = fmaxf(m_hi, acc[3]); if (has_sl) sl_hi += acc[3]; }
        if (cB0) { m_hi = fmaxf(m_hi, acc[6]); if (has_sl) sl_hi += acc[6]; }
        if (cB1) { m_hi = fmaxf(m_hi, acc[7]); if (has_sl) sl_hi += acc[7]; }
    }

    // 4-lane group reduction (width=4 → lanes 4g..4g+3, same lr)
    #pragma unroll
    for (int off = 1; off <= 2; off <<= 1) {
        m_lo = fmaxf(m_lo, __shfl_xor_sync(0xFFFFFFFFu, m_lo, off, 4));
        m_hi = fmaxf(m_hi, __shfl_xor_sync(0xFFFFFFFFu, m_hi, off, 4));
        if (has_sl) {
            sl_lo += __shfl_xor_sync(0xFFFFFFFFu, sl_lo, off, 4);
            sl_hi += __shfl_xor_sync(0xFFFFFFFFu, sl_hi, off, 4);
        }
    }

    // logit_scale (positive) — deferred, safe on max & sums
    m_lo *= logit_scale;
    m_hi *= logit_scale;
    if (has_sl) { sl_lo *= logit_scale; sl_hi *= logit_scale; }

    if (lc == 0) {
        if (r_lo_ok) { sPartMax[row_lo * N_WN + wn] = m_lo; if (has_sl) sPartSL[row_lo * N_WN + wn] = sl_lo; }
        if (r_hi_ok) { sPartMax[row_hi * N_WN + wn] = m_hi; if (has_sl) sPartSL[row_hi * N_WN + wn] = sl_hi; }
    }
    __syncthreads();

    // ---- Phase 2: cross-wn reduce → row_max, row_sl ----
    // 8 threads per row (one per wn), shuffle-reduce within 8-lane segment.
    if (tid < BLOCK_M * N_WN) {
        const int r = tid / N_WN;
        const int w = tid % N_WN;
        float m  = sPartMax[r * N_WN + w];
        float sl = has_sl ? sPartSL[r * N_WN + w] : 0.f;

        #pragma unroll
        for (int off = 1; off < N_WN; off <<= 1) {
            m = fmaxf(m, __shfl_xor_sync(0xFFFFFFFFu, m, off, N_WN));
            if (has_sl) sl += __shfl_xor_sync(0xFFFFFFFFu, sl, off, N_WN);
        }

        if (w == 0) {
            sRowMax[r] = m;
            if (has_sl) sRowSL[r] = sl;
        }
    }
    __syncthreads();

    // ---- Phase 3: sum of exp over fragment (per-lane registers) ----
    const float rmx_lo = r_lo_ok ? sRowMax[row_lo] : 0.f;
    const float rmx_hi = r_hi_ok ? sRowMax[row_hi] : 0.f;

    float se_lo = 0.f, se_hi = 0.f;
    if (r_lo_ok) {
        if (cA0) se_lo += __expf(fmaxf(__fmaf_rn(acc[0], logit_scale, -rmx_lo), -80.0f));
        if (cA1) se_lo += __expf(fmaxf(__fmaf_rn(acc[1], logit_scale, -rmx_lo), -80.0f));
        if (cB0) se_lo += __expf(fmaxf(__fmaf_rn(acc[4], logit_scale, -rmx_lo), -80.0f));
        if (cB1) se_lo += __expf(fmaxf(__fmaf_rn(acc[5], logit_scale, -rmx_lo), -80.0f));
    }
    if (r_hi_ok) {
        if (cA0) se_hi += __expf(fmaxf(__fmaf_rn(acc[2], logit_scale, -rmx_hi), -80.0f));
        if (cA1) se_hi += __expf(fmaxf(__fmaf_rn(acc[3], logit_scale, -rmx_hi), -80.0f));
        if (cB0) se_hi += __expf(fmaxf(__fmaf_rn(acc[6], logit_scale, -rmx_hi), -80.0f));
        if (cB1) se_hi += __expf(fmaxf(__fmaf_rn(acc[7], logit_scale, -rmx_hi), -80.0f));
    }

    #pragma unroll
    for (int off = 1; off <= 2; off <<= 1) {
        se_lo += __shfl_xor_sync(0xFFFFFFFFu, se_lo, off, 4);
        se_hi += __shfl_xor_sync(0xFFFFFFFFu, se_hi, off, 4);
    }

    if (lc == 0) {
        if (r_lo_ok) sPartSum[row_lo * N_WN + wn] = se_lo;
        if (r_hi_ok) sPartSum[row_hi * N_WN + wn] = se_hi;
    }
    __syncthreads();

    // ---- Phase 4: cross-wn reduce → row_sum ----
    if (tid < BLOCK_M * N_WN) {
        const int r = tid / N_WN;
        const int w = tid % N_WN;
        float s = sPartSum[r * N_WN + w];

        #pragma unroll
        for (int off = 1; off < N_WN; off <<= 1) {
            s += __shfl_xor_sync(0xFFFFFFFFu, s, off, N_WN);
        }
        if (w == 0) sRowSum[r] = s;
    }
    __syncthreads();

    // ---- Phase 5: write partials to global ----
    if (tid < BLOCK_M && tid < valid_m) {
        const size_t idx = (size_t)(start_m + tid) * stride_v + blockIdx.y;
        PartialMax[idx] = sRowMax[tid];
        PartialSum[idx] = sRowSum[tid];
        if (has_sl) PartialSumLogits[idx] = sRowSL[tid];
    }

    // ==================================================================================
    // Target logit extraction (register-based, single lane per row writes)
    // ==================================================================================
    if (r_lo_ok) {
        const int tgt = Target[start_m + row_lo];
        if (tgt != ignore_index) {
            const int tcol = tgt - v_tile;
            float tl = 0.f; bool found = false;
            if      (tcol == col_a + 0) { tl = acc[0]; found = true; }
            else if (tcol == col_a + 1) { tl = acc[1]; found = true; }
            else if (tcol == col_b + 0) { tl = acc[4]; found = true; }
            else if (tcol == col_b + 1) { tl = acc[5]; found = true; }
            if (found && tcol >= 0 && tcol < valid_v) {
                TargetLogit[start_m + row_lo] = tl * logit_scale;
            }
        }
    }
    if (r_hi_ok) {
        const int tgt = Target[start_m + row_hi];
        if (tgt != ignore_index) {
            const int tcol = tgt - v_tile;
            float tl = 0.f; bool found = false;
            if      (tcol == col_a + 0) { tl = acc[2]; found = true; }
            else if (tcol == col_a + 1) { tl = acc[3]; found = true; }
            else if (tcol == col_b + 0) { tl = acc[6]; found = true; }
            else if (tcol == col_b + 1) { tl = acc[7]; found = true; }
            if (found && tcol >= 0 && tcol < valid_v) {
                TargetLogit[start_m + row_hi] = tl * logit_scale;
            }
        }
    }
}

// ======================================================================================
// KERNEL 2: Loss / LSE / Z_Loss = FinalReduce(PartialMax, PartialSum, PartialSumLogits, TargetLogit)
// ======================================================================================
template<typename Config>
__global__ void __launch_bounds__(LogitsConfig::MAX_THREADS * MAX_THREADS_PER_WARP, 4)
forward_lce_reduce(
    const float* __restrict__ PartialMax,
    const float* __restrict__ PartialSum,
    const float* __restrict__ PartialSumLogits,
    const float* __restrict__ TargetLogit,
    const int*   __restrict__ Target,
          float* __restrict__ Loss,
          float* __restrict__ Logsumexp,
          float* __restrict__ Z_Loss,
    int   M, int V, int n_vblocks, int stride_v,
    float label_smooth, float lse_sscale, int ignore_index
) {
    constexpr int MAX_THREADS = Config::MAX_THREADS;

    const int warp_id = threadIdx.x >> 5;
    const int lane_id = threadIdx.x & 31;
    const int row     = blockIdx.x * MAX_THREADS + warp_id;
    if (row >= M) return;

    const float* __restrict__ pmax = PartialMax + (size_t)row * stride_v;
    const float* __restrict__ psum = PartialSum + (size_t)row * stride_v;
    const float* __restrict__ psl  = (PartialSumLogits != nullptr) ? PartialSumLogits + (size_t)row * stride_v : nullptr;

    float4 rPmax[MAX_THREADS];
    float thread_max     = NEG_INF;
    float thread_sum     = 0.0f;
    float thread_slogits = 0.0f;

    #pragma unroll
    for (int idx = 0; idx < MAX_THREADS; ++idx) {
        const int addr = lane_id + idx * 32;
        if (addr < (n_vblocks >> 2)) {
            rPmax[idx] = reinterpret_cast<const float4*>(pmax)[addr];
            thread_max = fmaxf(thread_max, fmaxf(fmaxf(rPmax[idx].x, rPmax[idx].y), fmaxf(rPmax[idx].z, rPmax[idx].w)));
        }
    }
    for (int idx = ((n_vblocks >> 2) << 2) + lane_id; idx < n_vblocks; idx += 32)
        thread_max = fmaxf(thread_max, pmax[idx]);

    #pragma unroll
    for (int off = 16; off > 0; off >>= 1)
        thread_max = fmaxf(thread_max, __shfl_xor_sync(0xFFFFFFFFu, thread_max, off));

    #pragma unroll
    for (int idx = 0; idx < MAX_THREADS; ++idx) {
        const int addr = lane_id + idx * 32;
        if (addr < (n_vblocks >> 2)) {
            const float4 ps = reinterpret_cast<const float4*>(psum)[addr];
            thread_sum += ps.x * __expf(rPmax[idx].x - thread_max)
                        + ps.y * __expf(rPmax[idx].y - thread_max)
                        + ps.z * __expf(rPmax[idx].z - thread_max)
                        + ps.w * __expf(rPmax[idx].w - thread_max);
            if (PartialSumLogits != nullptr) {
                const float4 pl = reinterpret_cast<const float4*>(psl)[addr];
                thread_slogits += (pl.x + pl.y) + (pl.z + pl.w);
            }
        }
    }
    for (int idx = ((n_vblocks >> 2) << 2) + lane_id; idx < n_vblocks; idx += 32) {
        thread_sum += psum[idx] * __expf(pmax[idx] - thread_max);
        if (PartialSumLogits != nullptr) thread_slogits += psl[idx];
    }

    #pragma unroll
    for (int off = 16; off > 0; off >>= 1) {
        thread_sum += __shfl_xor_sync(0xFFFFFFFFu, thread_sum, off);
        if (PartialSumLogits != nullptr)
            thread_slogits += __shfl_xor_sync(0xFFFFFFFFu, thread_slogits, off);
    }

    if (lane_id != 0) return;

    const float LSE       = thread_max + __logf(fmaxf(thread_sum, 1e-24f));
    const int   target    = Target[row];
    const float tgt_logit = TargetLogit[row];
    const bool  valid     = (target != ignore_index) && (target >= 0) && (target < V);

    float loss_val = 0.0f, z_loss_val = 0.0f;
    if (valid) {
        const float smooth = (PartialSumLogits != nullptr)
                           ? (label_smooth * thread_slogits / (float)V) : 0.0f;
        loss_val   = LSE - (1.0f - label_smooth) * tgt_logit - smooth;
        z_loss_val = (lse_sscale > 0.0f) ? (lse_sscale * LSE * LSE) : 0.0f;
    }

    Loss[row]      = loss_val + z_loss_val;
    Logsumexp[row] = LSE;
    Z_Loss[row]    = z_loss_val;
}

// ======================================================================================
// CPU REFERENCE & METRICS
// ======================================================================================
void cpu_ref_lce(const float* Input, const float* Weight, const int* Target,
                 float* Loss_ref, float* LSE_ref, float* ZLoss_ref,
                 int M, int V, int D, float logit_scale, float label_smooth, float lse_sscale, int ignore_index) {
    #pragma omp parallel for schedule(dynamic)
    for (int i = 0; i < M; ++i) {
        float max_logit = NEG_INF;
        std::vector<float> logits(V);
        for (int v = 0; v < V; ++v) {
            float sum = 0;
            for (int d = 0; d < D; ++d) {
                sum += Input[i * D + d] * Weight[v * D + d];
            }
            logits[v] = sum * logit_scale;
            max_logit = std::max(max_logit, logits[v]);
        }

        float sum_exp = 0;
        float sum_logits = 0;
        for (int v = 0; v < V; ++v) {
            float e = expf(logits[v] - max_logit);
            sum_exp += e;
            sum_logits += logits[v];
        }

        float LSE = max_logit + logf(fmaxf(sum_exp, 1e-24f));
        LSE_ref[i] = LSE;

        float loss_val = 0.0f, z_loss_val = 0.0f;
        int target = Target[i];
        bool valid = (target != ignore_index) && (target >= 0) && (target < V);
        if (valid) {
            float smooth = (label_smooth != 0.0f) ? (label_smooth * sum_logits / (float)V) : 0.0f;
            loss_val = LSE - (1.0f - label_smooth) * logits[target] - smooth;
            z_loss_val = (lse_sscale > 0.0f) ? (lse_sscale * LSE * LSE) : 0.0f;
        }
        Loss_ref[i] = loss_val + z_loss_val;
        ZLoss_ref[i] = z_loss_val;
    }
}

struct StabilityMetrics {
    float max_diff, avg_diff;
    int nan_count, inf_count, big_diff_count;
};

StabilityMetrics compute_stability_lce(const std::vector<float>& loss_gpu, const std::vector<float>& loss_ref,
                                       const std::vector<float>& lse_gpu, const std::vector<float>& lse_ref) {
    StabilityMetrics m = {0,0,0,0,0};
    double sum = 0;
    size_t total_elements = loss_gpu.size() + lse_gpu.size();
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
    check(loss_gpu, loss_ref, 0.01f);
    check(lse_gpu, lse_ref, 0.05f);
    if (total_elements > 0) m.avg_diff = (float)(sum / total_elements);
    return m;
}

// ======================================================================================
// TEST HARNESS
// ======================================================================================
template<typename Config>
bool run_test_lce(int M, int V, int D, float logit_scale, float label_smooth, float lse_sscale, int ignore_index) {
    constexpr int BLOCK_M = Config::BLOCK_M;
    constexpr int BLOCK_N = Config::BLOCK_N;
    constexpr size_t SMEM = Config::TOTAL_SMEM;
    constexpr int THREADS_PER_BLOCK  = Config::THREADS_PER_BLOCK;

    printf("  M=%-5d V=%-5d D=%-4d BM=%-3d BN=%-3d ", M, V, D, BLOCK_M, BLOCK_N);

    const int n_vblocks = (V + BLOCK_N - 1) / BLOCK_N;
    const int stride_v  = (n_vblocks + 3) & ~3;

    size_t isz = (size_t)M * D, wsz = (size_t)V * D, tsz = (size_t)M;
    size_t osz = (size_t)M, lsz = (size_t)M, zsz = (size_t)M;
    size_t psz = (size_t)M * stride_v;

    std::vector<float> hInput(isz), hWeight(wsz);
    std::vector<int>   hTarget(tsz);
    std::vector<float> hLoss_ref(osz), hLSE_ref(lsz), hZLoss_ref(zsz);
    std::vector<float> hLoss_gpu(osz), hLSE_gpu(lsz), hZLoss_gpu(zsz);

    srand(42);
    for (auto& v : hInput) v = ((float)rand() / RAND_MAX - 0.5f) * 0.1f;
    for (auto& v : hWeight) v = ((float)rand() / RAND_MAX - 0.5f) * 0.1f;
    for (auto& v : hTarget) v = rand() % (V + 10);

    bool use_cpu_ref = (M <= 1024 && V <= 4096);
    if (use_cpu_ref) {
        cpu_ref_lce(hInput.data(), hWeight.data(), hTarget.data(), 
                    hLoss_ref.data(), hLSE_ref.data(), hZLoss_ref.data(),
                    M, V, D, logit_scale, label_smooth, lse_sscale, ignore_index);
    }

    __half *dInput, *dWeight; int *dTarget;
    float *dPartialMax, *dPartialSum, *dPartialSumLogits, *dTargetLogit;
    float *dLoss, *dLogsumexp, *dZLoss;

    cudaMalloc(&dInput,  isz * 2);
    cudaMalloc(&dWeight, wsz * 2);
    cudaMalloc(&dTarget, tsz * 4);
    cudaMalloc(&dPartialMax, psz * 4);
    cudaMalloc(&dPartialSum, psz * 4);
    cudaMalloc(&dPartialSumLogits, psz * 4);
    cudaMalloc(&dTargetLogit, osz * 4);
    cudaMalloc(&dLoss, osz * 4);
    cudaMalloc(&dLogsumexp, lsz * 4);
    cudaMalloc(&dZLoss, zsz * 4);

    std::vector<__half> hInput_h(isz), hWeight_h(wsz);
    for (size_t i = 0; i < isz; ++i) hInput_h[i] = __float2half(hInput[i]);
    for (size_t i = 0; i < wsz; ++i) hWeight_h[i] = __float2half(hWeight[i]);

    cudaMemcpy(dInput, hInput_h.data(), isz * 2, cudaMemcpyHostToDevice);
    cudaMemcpy(dWeight, hWeight_h.data(), wsz * 2, cudaMemcpyHostToDevice);
    cudaMemcpy(dTarget, hTarget.data(), tsz * 4, cudaMemcpyHostToDevice);

    std::vector<float> hTargetLogit(osz, NEG_INF);
    cudaMemcpy(dTargetLogit, hTargetLogit.data(), osz * 4, cudaMemcpyHostToDevice);

    cudaFuncSetAttribute(forward_lce_partial<Config>, cudaFuncAttributeMaxDynamicSharedMemorySize, SMEM);

    dim3 grid_partial((M + BLOCK_M - 1) / BLOCK_M, n_vblocks);
    dim3 threads_partial(THREADS_PER_BLOCK);

    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    cudaEventRecord(start, 0);

    forward_lce_partial<Config><<<grid_partial, threads_partial, SMEM>>>(
        dInput, dWeight, dTarget,
        dPartialMax, dPartialSum, dPartialSumLogits, dTargetLogit,
        M, V, D, stride_v, logit_scale, ignore_index
    );

    dim3 grid_reduce((M + (Config::MAX_THREADS * MAX_THREADS_PER_WARP) - 1) / Config::MAX_THREADS);
    dim3 threads_reduce(Config::MAX_THREADS * MAX_THREADS_PER_WARP);

    forward_lce_reduce<Config><<<grid_reduce, threads_reduce, 0>>>(
        dPartialMax, dPartialSum, (label_smooth != 0.0f ? dPartialSumLogits : nullptr), dTargetLogit, dTarget,
        dLoss, dLogsumexp, dZLoss,
        M, V, n_vblocks, stride_v, label_smooth, lse_sscale, ignore_index
    );

    cudaEventRecord(stop, 0);
    cudaEventSynchronize(stop);
    float milliseconds = 0;
    cudaEventElapsedTime(&milliseconds, start, stop);
    cudaEventDestroy(start);
    cudaEventDestroy(stop);

    cudaMemcpy(hLoss_gpu.data(), dLoss, osz * 4, cudaMemcpyDeviceToHost);
    cudaMemcpy(hLSE_gpu.data(), dLogsumexp, lsz * 4, cudaMemcpyDeviceToHost);

    StabilityMetrics m = {0,0,0,0,0};
    if (use_cpu_ref) {
        m = compute_stability_lce(hLoss_gpu, hLoss_ref, hLSE_gpu, hLSE_ref);
    } else {
        for (size_t i = 0; i < osz; ++i) {
            if (std::isnan(hLoss_gpu[i])) m.nan_count++;
            if (std::isinf(hLoss_gpu[i])) m.inf_count++;
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

    printf("%s  max(Loss,LSE)=%.2e  avg=%.2e  NaN=%d Inf=%d big=%d GPU=%.4f ms\n", pass ? "PASS" : "FAIL",
           m.max_diff, m.avg_diff, m.nan_count, m.inf_count, m.big_diff_count, milliseconds);

    cudaFree(dInput); cudaFree(dWeight); cudaFree(dTarget);
    cudaFree(dPartialMax); cudaFree(dPartialSum); cudaFree(dPartialSumLogits); cudaFree(dTargetLogit);
    cudaFree(dLoss); cudaFree(dLogsumexp); cudaFree(dZLoss);

    return pass;
}

bool dispatch_test_lce(int M, int V, int D, float logit_scale, float label_smooth, float lse_sscale, int ignore_index) {
    using Config = LogitsConfig;
    return run_test_lce<Config>(M, V, D, logit_scale, label_smooth, lse_sscale, ignore_index);
}

int main() {
    printf("[ LCE Forward ]:\n");
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
        T(dispatch_test_lce(M, V, D, ls, lsm, lses, ign));
    }

    printf("\n%d/%d tests passed\n", ok, total);
    return (ok == total) ? 0 : 1;
}
