#include <cuda.h>
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <algorithm>
#include <vector>
#include "mma_m16n16k16.h"

using namespace volta::wmma;

#define NEG_INF (-1e30f)

template<int D>
struct KernelConfig {
    struct DQ {
        static constexpr int BLOCK_M           = 32;
        static constexpr int BLOCK_N           = 112;
        static constexpr int WARPS_PER_BLOCK   = 16;
        static constexpr int THREADS_PER_BLOCK = WARPS_PER_BLOCK * 32;
        static constexpr int THREADS_PER_ROW   = THREADS_PER_BLOCK / BLOCK_M;
        static constexpr int D_STRIDE          = D;
        static constexpr int N_STRIDE          = BLOCK_N;
    };
    
    struct DKV {
        static constexpr int BLOCK_M           = 16;
        static constexpr int BLOCK_N           = 144;
        static constexpr int WARPS_PER_BLOCK   = 12;
        static constexpr int THREADS_PER_BLOCK = WARPS_PER_BLOCK * 32;
        static constexpr int THREADS_PER_ROW   = THREADS_PER_BLOCK / BLOCK_N;
        static constexpr int D_STRIDE          = D;
        static constexpr int M_STRIDE          = BLOCK_M;
    };

    static constexpr int MAX_THREADS = (DQ::THREADS_PER_BLOCK > DKV::THREADS_PER_BLOCK) ? DQ::THREADS_PER_BLOCK : DKV::THREADS_PER_BLOCK;
    static constexpr int MAX_LSE     = (DQ::BLOCK_M > DKV::BLOCK_N) ? DQ::BLOCK_M : DKV::BLOCK_N;

    struct alignas(128) SmemLayout {
        union PhaseMem {
            struct DQ_Phase {
                alignas(16) __half q  [DQ::BLOCK_M * DQ::D_STRIDE];
                alignas(16) __half dO [DQ::BLOCK_M * DQ::D_STRIDE];
                union {
                    alignas(16) __half k [DQ::BLOCK_N * DQ::D_STRIDE];
                    alignas(16) __half v [DQ::BLOCK_N * DQ::D_STRIDE];
                } reuse_kv;
                alignas(16) float  s  [DQ::BLOCK_M * DQ::N_STRIDE];
                union {
                    alignas(16) float  dOV[DQ::BLOCK_M * DQ::N_STRIDE];
                    alignas(16) __half dS [DQ::BLOCK_M * DQ::N_STRIDE];
                } reuse_dov_ds;
                alignas(16) float  dQ [DQ::BLOCK_M * DQ::D_STRIDE];
            } dq;

            struct DKV_Phase {
                alignas(16) __half k [DKV::BLOCK_M * DKV::D_STRIDE];
                alignas(16) __half v [DKV::BLOCK_M * DKV::D_STRIDE];
                union {
                    alignas(16) __half q [DKV::BLOCK_N * DKV::D_STRIDE];
                    alignas(16) __half dO[DKV::BLOCK_N * DKV::D_STRIDE];
                } reuse_q_do;
                union {
                    alignas(16) float  s [DKV::BLOCK_N * DKV::M_STRIDE];
                    alignas(16) __half p [DKV::BLOCK_N * DKV::M_STRIDE];
                } reuse_s_p;
                union {
                    alignas(16) float  dOV[DKV::BLOCK_N * DKV::M_STRIDE];
                    alignas(16) __half dS [DKV::BLOCK_N * DKV::M_STRIDE];
                } reuse_dov_ds;
                alignas(16) float dK[DKV::BLOCK_M * DKV::D_STRIDE];
                alignas(16) float dV[DKV::BLOCK_M * DKV::D_STRIDE];
            } dkv;
        } phase;
        alignas(16) float lse     [MAX_LSE];
        alignas(16) float row_dot [MAX_LSE];
    };
    
    static constexpr size_t TOTAL_SMEM = ((sizeof(SmemLayout) + 127) & ~size_t(127));
};

template<typename Config>
__device__ __forceinline__ void init_smem(char* smem_raw) {
    constexpr int N_U4 = Config::TOTAL_SMEM / 16;
    const int tid = threadIdx.x;
    uint32_t addr = static_cast<uint32_t>(__cvta_generic_to_shared(smem_raw));
    
    #pragma unroll 4
    for (int i = tid; i < N_U4; i += blockDim.x) {
        asm volatile("st.shared.v4.u32 [%0], {0x0, 0x0, 0x0, 0x0};" :: "r"(addr + (i << 4)) : "memory");
    }
    __syncthreads();
}

template<int D, bool IS_CAUSAL>
__global__ void __launch_bounds__(KernelConfig<D>::MAX_THREADS, 1)
flash_attention_backward_kernel(
    const __half* __restrict__ Q,
    const __half* __restrict__ K,
    const __half* __restrict__ V,
    const __half* __restrict__ O,
    const __half* __restrict__ dO,
    const float*  __restrict__ softmax_lse,
          __half* __restrict__ dQ,
          __half* __restrict__ dK,
          __half* __restrict__ dV,
    const int B, const int H, const int M, const int N, 
    const int grid_dq_limit, const int grid_dkv_limit,
    const float softmax_scale) 
{
    using Config = KernelConfig<D>;
    const int batch_head_id = blockIdx.z;
    if (batch_head_id >= B * H) return;

    const int tid     = threadIdx.x;
    const int warp_id = tid >> 5;
    const int lane_id = tid & 31;

    extern __shared__ char smem_raw[];
    init_smem<Config>(smem_raw);
    auto& smem = *reinterpret_cast<typename Config::SmemLayout*>(smem_raw);

    if (blockIdx.y == 0) {
        if (blockIdx.x >= grid_dq_limit) return;

        constexpr int BLOCK_M         = Config::DQ::BLOCK_M;
        constexpr int BLOCK_N         = Config::DQ::BLOCK_N;
        constexpr int THREADS_PER_BLOCK = Config::DQ::THREADS_PER_BLOCK;
        constexpr int THREADS_PER_ROW = Config::DQ::THREADS_PER_ROW;
        constexpr int WARPS_PER_BLOCK = Config::DQ::WARPS_PER_BLOCK;
        constexpr int D_STRIDE        = Config::DQ::D_STRIDE;
        constexpr int N_STRIDE        = Config::DQ::N_STRIDE;
        constexpr int D_UINT4         = D / 8;

        const int block_idx = blockIdx.x;
        const int start_q   = block_idx * BLOCK_M;
        if (start_q >= M) return;

        int num_kv_tiles = (N + BLOCK_N - 1) / BLOCK_N;
        const int valid_q_rows = min(BLOCK_M, M - start_q);

        if constexpr (IS_CAUSAL) {
            const int max_key_pos = start_q + valid_q_rows - 1;
            if (max_key_pos < 0) num_kv_tiles = 0;
            else num_kv_tiles = min(num_kv_tiles, (max_key_pos + BLOCK_N) / BLOCK_N);
        }

        const __half* q_ptr   = Q   + (size_t)batch_head_id * M * D + start_q * D;
        const __half* k_ptr   = K   + (size_t)batch_head_id * N * D;
        const __half* v_ptr   = V   + (size_t)batch_head_id * N * D;
        const __half* o_ptr   = O   + (size_t)batch_head_id * M * D + start_q * D;
        const __half* dO_ptr  = dO  + (size_t)batch_head_id * M * D + start_q * D;
              __half* dQ_ptr  = dQ  + (size_t)batch_head_id * M * D + start_q * D;
        const float*  lse_ptr = softmax_lse + (size_t)batch_head_id * M + start_q;

        __half* sQ   = smem.phase.dq.q;
        __half* sdO  = smem.phase.dq.dO;
        __half* sK   = smem.phase.dq.reuse_kv.k;
        __half* sV   = smem.phase.dq.reuse_kv.v;
        float*  sS   = smem.phase.dq.s;
        float*  sdOV = smem.phase.dq.reuse_dov_ds.dOV;
        __half* sdS  = smem.phase.dq.reuse_dov_ds.dS;
        float*  sRowDot = smem.row_dot;
        float*  sLse    = smem.lse;
        float*  sdQ     = smem.phase.dq.dQ;

        uint4* sQ_vec  = reinterpret_cast<uint4*>(sQ);
        uint4* sdO_vec = reinterpret_cast<uint4*>(sdO);
        const uint4* q_vec_g  = reinterpret_cast<const uint4*>(q_ptr);
        const uint4* do_vec_g = reinterpret_cast<const uint4*>(dO_ptr);

        for (int idx = tid; idx < valid_q_rows * D_UINT4; idx += THREADS_PER_BLOCK) {
            int row = idx / D_UINT4;
            int col = idx % D_UINT4;
            sQ_vec[row * D_UINT4 + col]  = __ldg(&q_vec_g[row * D_UINT4 + col]);
            sdO_vec[row * D_UINT4 + col] = __ldg(&do_vec_g[row * D_UINT4 + col]);
        }
        __syncthreads();

        if (tid < valid_q_rows * THREADS_PER_ROW) {
            int row = tid / THREADS_PER_ROW;
            int t_in_row = tid % THREADS_PER_ROW;
            float thread_dot = 0.0f;
            for (int j = t_in_row; j < D; j += THREADS_PER_ROW) {
                float ov = __half2float(o_ptr[row * D + j]);
                float dv = __half2float(sdO[row * D_STRIDE + j]);
                thread_dot += ov * dv;
            }
            for (int o = THREADS_PER_ROW / 2; o > 0; o >>= 1)
                thread_dot += __shfl_down_sync(0xFFFFFFFFU, thread_dot, o, THREADS_PER_ROW);
            if (t_in_row == 0) sRowDot[row] = thread_dot;
        }

        if (tid < valid_q_rows) sLse[tid] = lse_ptr[tid];
        __syncthreads();

        for (int block = 0; block < num_kv_tiles; ++block) {
            const int start_kv = block * BLOCK_N;
            if (start_kv >= N) break;
            const int valid_kv_rows = min(BLOCK_N, N - start_kv);

            if constexpr (IS_CAUSAL) {
                if (start_kv >= start_q + valid_q_rows) continue;
            }

            uint4* sV_vec = reinterpret_cast<uint4*>(sV);
            const uint4* v_vec_g = reinterpret_cast<const uint4*>(v_ptr + start_kv * D);
            for (int idx = tid; idx < valid_kv_rows * D_UINT4; idx += THREADS_PER_BLOCK) {
                int row = idx / D_UINT4;
                int col = idx % D_UINT4;
                sV_vec[row * D_UINT4 + col] = __ldg(&v_vec_g[row * D_UINT4 + col]);
            }
            __syncthreads();

            {
                const int num_tm = (BLOCK_M + 15) / 16;
                const int num_tn = (BLOCK_N + 15) / 16;
                const int num_tk = (D + 15) / 16;
                const int total = num_tm * num_tn;
                const int per_warp = (total + WARPS_PER_BLOCK - 1) / WARPS_PER_BLOCK;

                for (int t = 0; t < per_warp; ++t) {
                    int g = warp_id * per_warp + t;
                    if (g >= total) break;
                    int tm = (g / num_tn) * 16;
                    int tn = (g % num_tn) * 16;
                    if (tm >= valid_q_rows || tn >= valid_kv_rows) continue;

                    fragment<matrix_a, 16, 16, 16, half, row_major> a;
                    fragment<matrix_b, 16, 16, 16, half, col_major> b;
                    fragment<accumulator, 16, 16, 16, float> acc;
                    fill_fragment(acc, 0.0f);

                    for (int k = 0; k < num_tk; ++k) {
                        load_matrix_sync(a, sdO + tm * D_STRIDE + k * 16, D_STRIDE);
                        load_matrix_sync(b, sV  + tn * D_STRIDE + k * 16, D_STRIDE);
                        mma_sync(acc, a, b, acc);
                    }
                    store_matrix_sync(sdOV + tm * N_STRIDE + tn, acc, N_STRIDE, mem_row_major);
                }
            }
            __syncthreads();

            uint4* sK_vec = reinterpret_cast<uint4*>(sK);
            const uint4* k_vec_g = reinterpret_cast<const uint4*>(k_ptr + start_kv * D);
            for (int idx = tid; idx < valid_kv_rows * D_UINT4; idx += THREADS_PER_BLOCK) {
                int row = idx / D_UINT4;
                int col = idx % D_UINT4;
                sK_vec[row * D_UINT4 + col] = __ldg(&k_vec_g[row * D_UINT4 + col]);
            }
            __syncthreads();

            {
                const int num_tm = (BLOCK_M + 15) / 16;
                const int num_tn = (BLOCK_N + 15) / 16;
                const int num_tk = (D + 15) / 16;
                const int total = num_tm * num_tn;
                const int per_warp = (total + WARPS_PER_BLOCK - 1) / WARPS_PER_BLOCK;

                for (int t = 0; t < per_warp; ++t) {
                    int g = warp_id * per_warp + t;
                    if (g >= total) break;
                    int tm = (g / num_tn) * 16;
                    int tn = (g % num_tn) * 16;
                    if (tm >= valid_q_rows || tn >= valid_kv_rows) continue;

                    fragment<matrix_a, 16, 16, 16, half, row_major> a;
                    fragment<matrix_b, 16, 16, 16, half, col_major> b;
                    fragment<accumulator, 16, 16, 16, float> acc;
                    fill_fragment(acc, 0.0f);

                    for (int k = 0; k < num_tk; ++k) {
                        load_matrix_sync(a, sQ + tm * D_STRIDE + k * 16, D_STRIDE);
                        load_matrix_sync(b, sK + tn * D_STRIDE + k * 16, D_STRIDE);
                        mma_sync(acc, a, b, acc);
                    }

                    if constexpr (IS_CAUSAL) {
                        #pragma unroll
                        for (int i = 0; i < 8; ++i) {
                            int r_local = ((lane_id >> 2) & 1) * 8 + ((lane_id >> 4) & 1) * 4 + (lane_id & 1) + ((i >> 1) & 1) * 2;
                            int c_local = ((lane_id >> 3) & 1) * 8 + ((lane_id >> 1) & 1) * 2 + (i & 1) + ((i >> 2) & 1) * 4;
                            int gm = start_q + tm + r_local;
                            int gn = start_kv + tn + c_local;
                            bool valid = (gm < start_q + valid_q_rows) && (gn < start_kv + valid_kv_rows);
                            acc.x[i] = valid ? ((gn > gm) ? NEG_INF : acc.x[i] * softmax_scale) : NEG_INF;
                        }
                    } else {
                        #pragma unroll
                        for (int i = 0; i < 8; ++i) acc.x[i] *= softmax_scale;
                    }
                    store_matrix_sync(sS + tm * N_STRIDE + tn, acc, N_STRIDE, mem_row_major);
                }
            }
            __syncthreads();

            if (tid < valid_q_rows * THREADS_PER_ROW) {
                int row = tid / THREADS_PER_ROW;
                int t_in_row = tid % THREADS_PER_ROW;
                float lse_val = sLse[row];
                float row_dot_val = sRowDot[row];
                for (int c = t_in_row; c < BLOCK_N; c += THREADS_PER_ROW) {
                    float s_val = (c < valid_kv_rows) ? sS[row * N_STRIDE + c] : NEG_INF;
                    float dov = (c < valid_kv_rows) ? sdOV[row * N_STRIDE + c] : 0.0f;
                    float p = (s_val - lse_val < -80.0f) ? 0.0f : __expf(s_val - lse_val);
                    float ds = p * softmax_scale * (dov - row_dot_val);
                    sdS[row * N_STRIDE + c] = __float2half_rn(ds);
                }
            }
            __syncthreads();

            {
                const int num_tm = (BLOCK_M + 15) / 16;
                const int num_tn = (D + 15) / 16;
                const int num_tk = (BLOCK_N + 15) / 16;
                const int total = num_tm * num_tn;
                const int per_warp = (total + WARPS_PER_BLOCK - 1) / WARPS_PER_BLOCK;

                for (int t = 0; t < per_warp; ++t) {
                    int g = warp_id * per_warp + t;
                    if (g >= total) break;
                    int tm = (g / num_tn) * 16;
                    int tn = (g % num_tn) * 16;
                    if (tm >= valid_q_rows || tn >= D) continue;

                    fragment<matrix_a, 16, 16, 16, half, row_major> a;
                    fragment<matrix_b, 16, 16, 16, half, row_major> b;
                    fragment<accumulator, 16, 16, 16, float> acc;
                    fill_fragment(acc, 0.0f);

                    for (int k = 0; k < num_tk; ++k) {
                        if (k * 16 >= valid_kv_rows) break;
                        load_matrix_sync(a, sdS + tm * N_STRIDE + k * 16, N_STRIDE);
                        load_matrix_sync(b, sK + k * 16 * D_STRIDE + tn, D_STRIDE);
                        mma_sync(acc, a, b, acc);
                    }

                    fragment<accumulator, 16, 16, 16, float> cur;
                    load_matrix_sync(cur, sdQ + tm * D_STRIDE + tn, D_STRIDE, mem_row_major);
                    for (int i = 0; i < 8; ++i) cur.x[i] += acc.x[i];
                    store_matrix_sync(sdQ + tm * D_STRIDE + tn, cur, D_STRIDE, mem_row_major);
                }
            }
            __syncthreads();
        }

        for (int i = tid; i < valid_q_rows * D_UINT4; i += THREADS_PER_BLOCK) {
            int row = i / D_UINT4;
            int col = i % D_UINT4;
            const float* src_row = sdQ + row * D_STRIDE + col * 8;
            __half out[8];
            #pragma unroll
            for (int j = 0; j < 8; ++j) out[j] = __float2half_rn(src_row[j]);
            reinterpret_cast<uint4*>(dQ_ptr + row * D)[col] = *reinterpret_cast<uint4*>(out);
        }
    }
    else if (blockIdx.y == 1) {
        if (blockIdx.x >= grid_dkv_limit) return;

        constexpr int BLOCK_M         = Config::DKV::BLOCK_M;
        constexpr int BLOCK_N         = Config::DKV::BLOCK_N;
        constexpr int THREADS_PER_BLOCK = Config::DKV::THREADS_PER_BLOCK;
        constexpr int THREADS_PER_ROW = Config::DKV::THREADS_PER_ROW;
        constexpr int WARPS_PER_BLOCK = Config::DKV::WARPS_PER_BLOCK;
        constexpr int D_STRIDE        = Config::DKV::D_STRIDE;
        constexpr int M_STRIDE        = Config::DKV::M_STRIDE;
        constexpr int D_UINT4         = D / 8;

        const int block_idx = blockIdx.x;
        const int start_kv  = block_idx * BLOCK_M;
        if (start_kv >= N) return;

        const int num_q_tiles   = (M + BLOCK_N - 1) / BLOCK_N;
        const int valid_kv_rows = min(BLOCK_M, N - start_kv);

        const __half* q_ptr   = Q   + (size_t)batch_head_id * M * D;
        const __half* k_ptr   = K   + (size_t)batch_head_id * N * D + start_kv * D;
        const __half* v_ptr   = V   + (size_t)batch_head_id * N * D + start_kv * D;
        const __half* o_ptr   = O   + (size_t)batch_head_id * M * D;
        const __half* dO_ptr  = dO  + (size_t)batch_head_id * M * D;
        const float*  lse_ptr = softmax_lse + (size_t)batch_head_id * M;
              __half* dK_ptr  = dK  + (size_t)batch_head_id * N * D + start_kv * D;
              __half* dV_ptr  = dV  + (size_t)batch_head_id * N * D + start_kv * D;

        __half* sK   = smem.phase.dkv.k;
        __half* sV   = smem.phase.dkv.v;
        __half* sQ   = smem.phase.dkv.reuse_q_do.q;
        __half* sdO  = smem.phase.dkv.reuse_q_do.dO;
        float*  sS   = smem.phase.dkv.reuse_s_p.s;
        __half* sP   = smem.phase.dkv.reuse_s_p.p;
        float*  sdOV = smem.phase.dkv.reuse_dov_ds.dOV;
        __half* sdS  = smem.phase.dkv.reuse_dov_ds.dS;
        float*  sRowDot = smem.row_dot;
        float*  sLse    = smem.lse;
        float*  sdK     = smem.phase.dkv.dK;
        float*  sdV     = smem.phase.dkv.dV;

        uint4* sK_vec = reinterpret_cast<uint4*>(sK);
        uint4* sV_vec = reinterpret_cast<uint4*>(sV);
        const uint4* k_vec_g = reinterpret_cast<const uint4*>(k_ptr);
        const uint4* v_vec_g = reinterpret_cast<const uint4*>(v_ptr);

        for (int idx = tid; idx < valid_kv_rows * D_UINT4; idx += THREADS_PER_BLOCK) {
            int row = idx / D_UINT4;
            int col = idx % D_UINT4;
            sK_vec[row * D_UINT4 + col] = __ldg(&k_vec_g[row * D_UINT4 + col]);
            sV_vec[row * D_UINT4 + col] = __ldg(&v_vec_g[row * D_UINT4 + col]);
        }
        __syncthreads();

        for (int block = 0; block < num_q_tiles; ++block) {
            const int start_q = block * BLOCK_N;
            if (start_q >= M) break;
            const int valid_q_rows = min(BLOCK_N, M - start_q);

            if constexpr (IS_CAUSAL) {
                if (start_kv >= start_q + valid_q_rows) continue;
            }

            uint4* sQ_vec = reinterpret_cast<uint4*>(sQ);
            const uint4* q_vec_g = reinterpret_cast<const uint4*>(q_ptr + start_q * D);
            for (int idx = tid; idx < valid_q_rows * D_UINT4; idx += THREADS_PER_BLOCK) {
                int row = idx / D_UINT4;
                int col = idx % D_UINT4;
                sQ_vec[row * D_UINT4 + col] = __ldg(&q_vec_g[row * D_UINT4 + col]);
            }
            __syncthreads();

            {
                const int num_tm = (BLOCK_N + 15) / 16;
                const int num_tn = (BLOCK_M + 15) / 16;
                const int num_tk = (D + 15) / 16;
                const int total = num_tm * num_tn;
                const int per_warp = (total + WARPS_PER_BLOCK - 1) / WARPS_PER_BLOCK;

                for (int t = 0; t < per_warp; ++t) {
                    int g = warp_id * per_warp + t;
                    if (g >= total) break;
                    int tm = (g / num_tn) * 16;
                    int tn = (g % num_tn) * 16;
                    if (tm >= valid_q_rows || tn >= valid_kv_rows) continue;

                    fragment<matrix_a, 16, 16, 16, half, row_major> a;
                    fragment<matrix_b, 16, 16, 16, half, col_major> b;
                    fragment<accumulator, 16, 16, 16, float> acc;
                    fill_fragment(acc, 0.0f);

                    for (int k = 0; k < num_tk; ++k) {
                        load_matrix_sync(a, sQ + tm * D_STRIDE + k * 16, D_STRIDE);
                        load_matrix_sync(b, sK + tn * D_STRIDE + k * 16, D_STRIDE);
                        mma_sync(acc, a, b, acc);
                    }

                    if constexpr (IS_CAUSAL) {
                        #pragma unroll
                        for (int i = 0; i < 8; ++i) {
                            int r_local = ((lane_id >> 2) & 1) * 8 + ((lane_id >> 4) & 1) * 4 + (lane_id & 1) + ((i >> 1) & 1) * 2;
                            int c_local = ((lane_id >> 3) & 1) * 8 + ((lane_id >> 1) & 1) * 2 + (i & 1) + ((i >> 2) & 1) * 4;
                            int gm = start_q + tm + r_local;
                            int gn = start_kv + tn + c_local;
                            bool valid = (gm < start_q + valid_q_rows) && (gn < start_kv + valid_kv_rows);
                            acc.x[i] = valid ? ((gn > gm) ? NEG_INF : acc.x[i] * softmax_scale) : NEG_INF;
                        }
                    } else {
                        #pragma unroll
                        for (int i = 0; i < 8; ++i) acc.x[i] *= softmax_scale;
                    }
                    store_matrix_sync(sS + tm * M_STRIDE + tn, acc, M_STRIDE, mem_row_major);
                }
            }
            __syncthreads();

            uint4* sdO_vec = reinterpret_cast<uint4*>(sdO);
            const uint4* do_vec_g = reinterpret_cast<const uint4*>(dO_ptr + start_q * D);
            for (int idx = tid; idx < valid_q_rows * D_UINT4; idx += THREADS_PER_BLOCK) {
                int row = idx / D_UINT4;
                int col = idx % D_UINT4;
                sdO_vec[row * D_UINT4 + col] = __ldg(&do_vec_g[row * D_UINT4 + col]);
            }
            __syncthreads();

            const __half* cur_o_ptr = o_ptr + start_q * D;
            if (tid < valid_q_rows * THREADS_PER_ROW) {
                int row = tid / THREADS_PER_ROW;
                int t_in_row = tid % THREADS_PER_ROW;
                float thread_dot = 0.0f;
                for (int j = t_in_row; j < D; j += THREADS_PER_ROW) {
                    thread_dot += __half2float(cur_o_ptr[row * D + j]) * __half2float(sdO[row * D_STRIDE + j]);
                }
                for (int o = THREADS_PER_ROW / 2; o > 0; o >>= 1)
                    thread_dot += __shfl_down_sync(0xFFFFFFFFU, thread_dot, o, THREADS_PER_ROW);
                if (t_in_row == 0) sRowDot[row] = thread_dot;
            }
            if (tid < valid_q_rows) sLse[tid] = lse_ptr[start_q + tid];
            __syncthreads();

            {
                const int num_tm = (BLOCK_N + 15) / 16;
                const int num_tn = (BLOCK_M + 15) / 16;
                const int num_tk = (D + 15) / 16;
                const int total = num_tm * num_tn;
                const int per_warp = (total + WARPS_PER_BLOCK - 1) / WARPS_PER_BLOCK;

                for (int t = 0; t < per_warp; ++t) {
                    int g = warp_id * per_warp + t;
                    if (g >= total) break;
                    int tm = (g / num_tn) * 16;
                    int tn = (g % num_tn) * 16;
                    if (tm >= valid_q_rows || tn >= valid_kv_rows) continue;

                    fragment<matrix_a, 16, 16, 16, half, row_major> a;
                    fragment<matrix_b, 16, 16, 16, half, col_major> b;
                    fragment<accumulator, 16, 16, 16, float> acc;
                    fill_fragment(acc, 0.0f);

                    for (int k = 0; k < num_tk; ++k) {
                        load_matrix_sync(a, sdO + tm * D_STRIDE + k * 16, D_STRIDE);
                        load_matrix_sync(b, sV  + tn * D_STRIDE + k * 16, D_STRIDE);
                        mma_sync(acc, a, b, acc);
                    }
                    store_matrix_sync(sdOV + tm * M_STRIDE + tn, acc, M_STRIDE, mem_row_major);
                }
            }
            __syncthreads();

            {
                int total_elems = valid_q_rows * valid_kv_rows;
                for (int i = tid; i < total_elems; i += THREADS_PER_BLOCK) {
                    int r = i / valid_kv_rows;
                    int c = i % valid_kv_rows;

                    int global_m = start_q + r;
                    int global_n = start_kv + c;

                    if constexpr (IS_CAUSAL) {
                        if (global_n > global_m) {
                            sP[r * M_STRIDE + c]  = __float2half_rn(0.0f);
                            sdS[r * M_STRIDE + c] = __float2half_rn(0.0f);
                            continue;
                        }
                    }

                    float s = sS[r * M_STRIDE + c];
                    float dov = sdOV[r * M_STRIDE + c];
                    float lse_val = sLse[r];
                    float row_dot_val = sRowDot[r];
                    float p = (s - lse_val < -80.0f) ? 0.0f : __expf(s - lse_val);
                    float ds = p * softmax_scale * (dov - row_dot_val);
                    sP[r * M_STRIDE + c]  = __float2half_rn(p);
                    sdS[r * M_STRIDE + c] = __float2half_rn(ds);
                }
            }
            __syncthreads();

            {
                const int num_tm = (BLOCK_M + 15) / 16;
                const int num_tn = (D + 15) / 16;
                const int num_tk = (BLOCK_N + 15) / 16;
                const int total = num_tm * num_tn;
                const int per_warp = (total + WARPS_PER_BLOCK - 1) / WARPS_PER_BLOCK;

                for (int t = 0; t < per_warp; ++t) {
                    int g = warp_id * per_warp + t;
                    if (g >= total) break;
                    int tm = (g / num_tn) * 16;
                    int tn = (g % num_tn) * 16;
                    if (tm >= valid_kv_rows || tn >= D) continue;

                    fragment<matrix_a, 16, 16, 16, half, col_major> a;
                    fragment<matrix_b, 16, 16, 16, half, row_major> b;
                    fragment<accumulator, 16, 16, 16, float> acc;
                    load_matrix_sync(acc, sdV + tm * D_STRIDE + tn, D_STRIDE, mem_row_major);

                    for (int k = 0; k < num_tk; ++k) {
                        if (k * 16 >= valid_q_rows) break;
                        load_matrix_sync(a, sP + k * 16 * M_STRIDE + tm, M_STRIDE);
                        load_matrix_sync(b, sdO + k * 16 * D_STRIDE + tn, D_STRIDE);
                        mma_sync(acc, a, b, acc);
                    }
                    store_matrix_sync(sdV + tm * D_STRIDE + tn, acc, D_STRIDE, mem_row_major);
                }
            }
            __syncthreads();

            {
                uint4* sQ_vec2 = reinterpret_cast<uint4*>(sQ);
                for (int idx = tid; idx < valid_q_rows * D_UINT4; idx += THREADS_PER_BLOCK) {
                    int row = idx / D_UINT4;
                    int col = idx % D_UINT4;
                    sQ_vec2[row * D_UINT4 + col] = __ldg(&q_vec_g[row * D_UINT4 + col]);
                }
            }
            __syncthreads();

            {
                const int num_tm = (BLOCK_M + 15) / 16;
                const int num_tn = (D + 15) / 16;
                const int num_tk = (BLOCK_N + 15) / 16;
                const int total = num_tm * num_tn;
                const int per_warp = (total + WARPS_PER_BLOCK - 1) / WARPS_PER_BLOCK;

                for (int t = 0; t < per_warp; ++t) {
                    int g = warp_id * per_warp + t;
                    if (g >= total) break;
                    int tm = (g / num_tn) * 16;
                    int tn = (g % num_tn) * 16;
                    if (tm >= valid_kv_rows || tn >= D) continue;

                    fragment<matrix_a, 16, 16, 16, half, col_major> a;
                    fragment<matrix_b, 16, 16, 16, half, row_major> b;
                    fragment<accumulator, 16, 16, 16, float> acc;
                    load_matrix_sync(acc, sdK + tm * D_STRIDE + tn, D_STRIDE, mem_row_major);

                    for (int k = 0; k < num_tk; ++k) {
                        if (k * 16 >= valid_q_rows) break;
                        load_matrix_sync(a, sdS + k * 16 * M_STRIDE + tm, M_STRIDE);
                        load_matrix_sync(b, sQ + k * 16 * D_STRIDE + tn, D_STRIDE);
                        mma_sync(acc, a, b, acc);
                    }
                    store_matrix_sync(sdK + tm * D_STRIDE + tn, acc, D_STRIDE, mem_row_major);
                }
            }
            __syncthreads();
        }

        for (int i = tid; i < valid_kv_rows * D_UINT4; i += THREADS_PER_BLOCK) {
            int row = i / D_UINT4;
            int col = i % D_UINT4;
            const float* dk_row = sdK + row * D_STRIDE + col * 8;
            const float* dv_row = sdV + row * D_STRIDE + col * 8;
            __half ok[8], ov[8];
            #pragma unroll
            for (int j = 0; j < 8; ++j) {
                ok[j] = __float2half_rn(dk_row[j]);
                ov[j] = __float2half_rn(dv_row[j]);
            }
            reinterpret_cast<uint4*>(dK_ptr + row * D)[col] = *reinterpret_cast<uint4*>(ok);
            reinterpret_cast<uint4*>(dV_ptr + row * D)[col] = *reinterpret_cast<uint4*>(ov);
        }
    }
}

void cpu_attention_backward(
    const std::vector<float>& Q, const std::vector<float>& K, const std::vector<float>& V,
    const std::vector<float>& O, const std::vector<float>& dO,
    std::vector<float>& dQ, std::vector<float>& dK, std::vector<float>& dV,
    int M, int N, int D, float scale, bool causal) 
{
    std::vector<float> S(M * N), P(M * N), dP(M * N), dS(M * N), D_row(M);

    for (int i = 0; i < M; ++i) {
        float max_val = -1e30f;
        for (int j = 0; j < N; ++j) {
            if (causal && j > i) { S[i * N + j] = -1e30f; continue; }
            float sum = 0;
            for (int k = 0; k < D; ++k) sum += Q[i * D + k] * K[j * D + k];
            S[i * N + j] = sum * scale;
            max_val = std::max(max_val, S[i * N + j]);
        }
        float sum_exp = 0;
        for (int j = 0; j < N; ++j) { P[i * N + j] = expf(S[i * N + j] - max_val); sum_exp += P[i * N + j]; }
        for (int j = 0; j < N; ++j) P[i * N + j] /= sum_exp;
    }

    for (int i = 0; i < M; ++i) {
        float sum = 0;
        for (int k = 0; k < D; ++k) sum += O[i * D + k] * dO[i * D + k];
        D_row[i] = sum;
    }

    for (int i = 0; i < M; ++i) {
        for (int j = 0; j < N; ++j) {
            float sum = 0;
            for (int k = 0; k < D; ++k) sum += dO[i * D + k] * V[j * D + k];
            dP[i * N + j] = sum;
        }
    }

    for (int i = 0; i < M; ++i) {
        for (int j = 0; j < N; ++j) {
            dS[i * N + j] = P[i * N + j] * (dP[i * N + j] - D_row[i]) * scale;
        }
    }

    std::fill(dQ.begin(), dQ.end(), 0.0f);
    for (int i = 0; i < M; ++i)
        for (int k = 0; k < D; ++k) {
            float sum = 0;
            for (int j = 0; j < N; ++j) sum += dS[i * N + j] * K[j * D + k];
            dQ[i * D + k] = sum;
        }

    std::fill(dK.begin(), dK.end(), 0.0f);
    for (int j = 0; j < N; ++j)
        for (int k = 0; k < D; ++k) {
            float sum = 0;
            for (int i = 0; i < M; ++i) sum += dS[i * N + j] * Q[i * D + k];
            dK[j * D + k] = sum;
        }

    std::fill(dV.begin(), dV.end(), 0.0f);
    for (int j = 0; j < N; ++j)
        for (int k = 0; k < D; ++k) {
            float sum = 0;
            for (int i = 0; i < M; ++i) sum += P[i * N + j] * dO[i * D + k];
            dV[j * D + k] = sum;
        }
}

void cpu_attention_forward(
    const std::vector<float>& Q, const std::vector<float>& K, const std::vector<float>& V,
    std::vector<float>& Out, std::vector<float>& LSE,
    int M, int N, int D, float scale, bool causal) 
{
    std::vector<float> S(N);
    for (int i = 0; i < M; ++i) {
        float max_val = -1e30f;
        for (int j = 0; j < N; ++j) {
            if (causal && j > i) { S[j] = -1e30f; continue; }
            float sum = 0;
            for (int k = 0; k < D; ++k) sum += Q[i * D + k] * K[j * D + k];
            S[j] = sum * scale;
            max_val = std::max(max_val, S[j]);
        }
        float sum_exp = 0;
        for (int j = 0; j < N; ++j) { S[j] = expf(S[j] - max_val); sum_exp += S[j]; }
        for (int j = 0; j < N; ++j) S[j] /= sum_exp;
        for (int k = 0; k < D; ++k) {
            float sum = 0;
            for (int j = 0; j < N; ++j) sum += S[j] * V[j * D + k];
            Out[i * D + k] = sum;
        }
        LSE[i] = max_val + logf(sum_exp);
    }
}

#define CHECK_CUDA(call) do { cudaError_t e = call; if (e != cudaSuccess) { \
    printf("CUDA Err: %s\n", cudaGetErrorString(e)); exit(1); } } while(0)

template<bool IS_CAUSAL>
void run_test(int B, int H, int M, int N, float scale) {
    constexpr int D = 128;
    using Config = KernelConfig<D>;

    printf("\n=== FA2 Backward ===\n");
    printf("D=%d, SMEM=%.2f KB, CAUSAL=%d\n", D, Config::TOTAL_SMEM / 1024.0f, IS_CAUSAL);

    size_t q_size = (size_t)B * H * M * D * sizeof(__half);
    size_t k_size = (size_t)B * H * N * D * sizeof(__half);
    size_t v_size = k_size;
    size_t o_size = q_size;
    size_t lse_size = (size_t)B * H * M * sizeof(float);

    __half *d_Q, *d_K, *d_V, *d_O, *d_dO, *d_dQ, *d_dK, *d_dV;
    float *d_LSE;
    CHECK_CUDA(cudaMalloc(&d_Q, q_size)); CHECK_CUDA(cudaMalloc(&d_K, k_size));
    CHECK_CUDA(cudaMalloc(&d_V, v_size)); CHECK_CUDA(cudaMalloc(&d_O, o_size));
    CHECK_CUDA(cudaMalloc(&d_dO, o_size)); CHECK_CUDA(cudaMalloc(&d_LSE, lse_size));
    CHECK_CUDA(cudaMalloc(&d_dQ, q_size)); CHECK_CUDA(cudaMalloc(&d_dK, k_size));
    CHECK_CUDA(cudaMalloc(&d_dV, v_size));

    std::vector<float> h_Q_f(M * D), h_K_f(N * D), h_V_f(N * D), h_dO_f(M * D);
    std::vector<__half> h_Q(M * D), h_K(N * D), h_V(N * D), h_dO(M * D);
    std::vector<float> h_O_f(M * D), h_LSE_f(M);

    srand(42);
    for (int i = 0; i < M * D; ++i) { h_Q_f[i] = ((float)rand() / RAND_MAX - 0.5f) * 0.2f; h_Q[i] = __float2half(h_Q_f[i]); }
    for (int i = 0; i < N * D; ++i) { h_K_f[i] = ((float)rand() / RAND_MAX - 0.5f) * 0.2f; h_K[i] = __float2half(h_K_f[i]); }
    for (int i = 0; i < N * D; ++i) { h_V_f[i] = ((float)rand() / RAND_MAX - 0.5f) * 0.2f; h_V[i] = __float2half(h_V_f[i]); }
    for (int i = 0; i < M * D; ++i) { h_dO_f[i] = ((float)rand() / RAND_MAX - 0.5f) * 0.2f; h_dO[i] = __float2half(h_dO_f[i]); }

    cpu_attention_forward(h_Q_f, h_K_f, h_V_f, h_O_f, h_LSE_f, M, N, D, scale, IS_CAUSAL);

    std::vector<__half> h_O(M * D);
    for (int i = 0; i < M * D; ++i) h_O[i] = __float2half(h_O_f[i]);

    CHECK_CUDA(cudaMemcpy(d_Q, h_Q.data(), q_size, cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(d_K, h_K.data(), k_size, cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(d_V, h_V.data(), v_size, cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(d_O, h_O.data(), o_size, cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(d_dO, h_dO.data(), o_size, cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(d_LSE, h_LSE_f.data(), lse_size, cudaMemcpyHostToDevice));

    CHECK_CUDA(cudaMemset(d_dQ, 0, q_size));
    CHECK_CUDA(cudaMemset(d_dK, 0, k_size));
    CHECK_CUDA(cudaMemset(d_dV, 0, v_size));

    int grid_dq  = (M + Config::DQ::BLOCK_M - 1) / Config::DQ::BLOCK_M;
    int grid_dkv = (N + Config::DKV::BLOCK_M - 1) / Config::DKV::BLOCK_M;
    dim3 grid(std::max(grid_dq, grid_dkv), 2, B * H);
    dim3 block(Config::MAX_THREADS);

    auto kernel = flash_attention_backward_kernel<D, IS_CAUSAL>;
    CHECK_CUDA(cudaFuncSetAttribute(kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, Config::TOTAL_SMEM));
    kernel<<<grid, block, Config::TOTAL_SMEM>>>(
        d_Q, d_K, d_V, d_O, d_dO, d_LSE, d_dQ, d_dK, d_dV,
        B, H, M, N, grid_dq, grid_dkv, scale);
    CHECK_CUDA(cudaDeviceSynchronize());

    std::vector<__half> h_dQ_out(M * D), h_dK_out(N * D), h_dV_out(N * D);
    CHECK_CUDA(cudaMemcpy(h_dQ_out.data(), d_dQ, q_size, cudaMemcpyDeviceToHost));
    CHECK_CUDA(cudaMemcpy(h_dK_out.data(), d_dK, k_size, cudaMemcpyDeviceToHost));
    CHECK_CUDA(cudaMemcpy(h_dV_out.data(), d_dV, v_size, cudaMemcpyDeviceToHost));

    std::vector<float> h_dQ_ref(M * D), h_dK_ref(N * D), h_dV_ref(N * D);
    cpu_attention_backward(h_Q_f, h_K_f, h_V_f, h_O_f, h_dO_f, h_dQ_ref, h_dK_ref, h_dV_ref, M, N, D, scale, IS_CAUSAL);

    float max_err_q = 0, max_err_k = 0, max_err_v = 0;
    for (int i = 0; i < M * D; ++i) max_err_q = std::max(max_err_q, fabsf(__half2float(h_dQ_out[i]) - h_dQ_ref[i]));
    for (int i = 0; i < N * D; ++i) {
        max_err_k = std::max(max_err_k, fabsf(__half2float(h_dK_out[i]) - h_dK_ref[i]));
        max_err_v = std::max(max_err_v, fabsf(__half2float(h_dV_out[i]) - h_dV_ref[i]));
    }

    bool pass = (max_err_q < 5e-2f) && (max_err_k < 5e-2f) && (max_err_v < 5e-2f);
    printf("Max Err -> dQ: %.5f | dK: %.5f | dV: %.5f  %s\n",
           max_err_q, max_err_k, max_err_v, pass ? "PASS" : "FAIL");

    cudaFree(d_Q); cudaFree(d_K); cudaFree(d_V); cudaFree(d_O); cudaFree(d_dO);
    cudaFree(d_dQ); cudaFree(d_dK); cudaFree(d_dV); cudaFree(d_LSE);
}

int main() {
    run_test<false>(1, 1, 128, 128, 0.125f);
    return 0;
}