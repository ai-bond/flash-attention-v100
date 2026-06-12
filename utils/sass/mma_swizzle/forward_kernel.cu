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

#define NEG_INF     (-1e30f)
#define BLOCK_M_128 32
#define BLOCK_N_128 160
#define WARPS_128   16

template<int D>
struct KernelConfig {
    static constexpr int BLOCK_M           = BLOCK_M_128;
    static constexpr int BLOCK_N           = BLOCK_N_128;
    static constexpr int WARPS_PER_BLOCK   = WARPS_128;
    static constexpr int THREADS_PER_BLOCK = WARPS_PER_BLOCK * 32;
    static constexpr int THREADS_PER_ROW   = THREADS_PER_BLOCK / BLOCK_M;
    static constexpr int D_STRIDE          = D;
    static constexpr int N_STRIDE          = BLOCK_N;
    static constexpr int PER_UINT4         = 8;

    struct alignas(128) SmemLayout {
        alignas(16) __half q[BLOCK_M * D_STRIDE];
        union {
            alignas(16) __half k[BLOCK_N * D_STRIDE];
            alignas(16) __half v[BLOCK_N * D_STRIDE];
        } reuse_kv;
        union {
            alignas(16) float  s[BLOCK_M * N_STRIDE];
            alignas(16) __half p[BLOCK_M * N_STRIDE];
        } reuse_sp;
        alignas(16) float  o[BLOCK_M * D_STRIDE];
        alignas(16) float  row_max[BLOCK_M];
        alignas(16) float  row_sum[BLOCK_M];
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
__global__ void __launch_bounds__(KernelConfig<D>::THREADS_PER_BLOCK, 2)
flash_attention_forward_kernel(
    const __half* __restrict__ Q,
    const __half* __restrict__ K,
    const __half* __restrict__ V,
    __half* __restrict__ Out,
    float* __restrict__ softmax_lse,
    const int B, const int H, const int M, const int N,
    const float softmax_scale)
{
    using Config = KernelConfig<D>;
    constexpr int BLOCK_M         = Config::BLOCK_M;
    constexpr int BLOCK_N         = Config::BLOCK_N;
    constexpr int THREADS_PER_BLOCK = Config::THREADS_PER_BLOCK;
    constexpr int THREADS_PER_ROW = Config::THREADS_PER_ROW;
    constexpr int WARPS_PER_BLOCK = Config::WARPS_PER_BLOCK;
    constexpr int D_STRIDE        = Config::D_STRIDE;
    constexpr int N_STRIDE        = Config::N_STRIDE;
    constexpr int PER_UINT4       = Config::PER_UINT4;

    const int batch_head_id = blockIdx.z;
    if (batch_head_id >= B * H) return;

    const int start_q = blockIdx.x * BLOCK_M;
    if (start_q >= M) return;

    int num_n_tiles = (N + BLOCK_N - 1) / BLOCK_N;
    const int valid_q_rows = min(BLOCK_M, M - start_q);

    if constexpr (IS_CAUSAL) {
        const int max_key_pos = start_q + valid_q_rows - 1;
        if (max_key_pos < 0) num_n_tiles = 0;
        else num_n_tiles = min(num_n_tiles, (max_key_pos + BLOCK_N) / BLOCK_N);
    }

    const int tid = threadIdx.x;
    const int warp_id = tid / 32;
    const int lane_id = tid % 32;

    const __half* q_ptr = Q + (size_t)batch_head_id * M * D + start_q * D;
    const __half* k_ptr = K + (size_t)batch_head_id * N * D;
    const __half* v_ptr = V + (size_t)batch_head_id * N * D;
    __half* out_ptr = Out + (size_t)batch_head_id * M * D + start_q * D;
    float* softmax_lse_ptr = softmax_lse + (size_t)batch_head_id * M + start_q;

    extern __shared__ char smem_raw[];
    init_smem<Config>(smem_raw);
    auto& smem = *reinterpret_cast<typename Config::SmemLayout*>(smem_raw);

    __half* sQ = smem.q; 
    __half* sK = smem.reuse_kv.k; 
    __half* sV = smem.reuse_kv.v;
    float*  sS = smem.reuse_sp.s; 
    __half* sP = smem.reuse_sp.p; 
    float*  sO = smem.o;

    if (tid < BLOCK_M) { 
        smem.row_max[tid] = NEG_INF; 
        smem.row_sum[tid] = 0.0f; 
    }

    const int d_stride_uint4 = (D_STRIDE + PER_UINT4 - 1) / PER_UINT4;

    uint4* sQ_u4 = reinterpret_cast<uint4*>(sQ);
    const uint4* q_vec = reinterpret_cast<const uint4*>(q_ptr);
    for (int idx = tid; idx < (valid_q_rows * d_stride_uint4); idx += THREADS_PER_BLOCK) {
        int row = idx / d_stride_uint4;
        int vec_col = idx % d_stride_uint4;
        uint4 q_val = (row < valid_q_rows) ? __ldg(&q_vec[row * d_stride_uint4 + vec_col]) : make_uint4(0, 0, 0, 0);
        sQ_u4[row * d_stride_uint4 + vec_col] = q_val;
    }
    __syncthreads();

    for (int block = 0; block < num_n_tiles; ++block) {
        const int start_kv = block * BLOCK_N;
        if (start_kv >= N) break;

        const int valid_kv_rows = min(BLOCK_N, N - start_kv);
        if constexpr (IS_CAUSAL) {
            if (start_kv >= start_q + valid_q_rows) continue;
        }

        uint4* sK_u4 = reinterpret_cast<uint4*>(sK);
        const uint4* k_vec = reinterpret_cast<const uint4*>(k_ptr + start_kv * D);
        for (int idx = tid; idx < (valid_kv_rows * d_stride_uint4); idx += THREADS_PER_BLOCK) {
            int row = idx / d_stride_uint4;
            int vec_col = idx % d_stride_uint4;
            uint4 k_val = (row < valid_kv_rows) ? __ldg(&k_vec[row * d_stride_uint4 + vec_col]) : make_uint4(0, 0, 0, 0);
            sK_u4[row * d_stride_uint4 + vec_col] = k_val;
        }
        __syncthreads();

        const int num_tiles_m = (BLOCK_M + 15) / 16;
        const int num_tiles_n = (BLOCK_N + 15) / 16;
        const int num_tiles_k = (D + 15) / 16;
        const int total_tiles = num_tiles_m * num_tiles_n;
        const int tiles_per_warp = (total_tiles + WARPS_PER_BLOCK - 1) / WARPS_PER_BLOCK;

        const unsigned row_causal = (lane_id & 1) + ((lane_id >> 2) & 1) * 8 + ((lane_id >> 4) & 1) * 4;
        const unsigned col_causal = ((lane_id >> 1) & 1) * 2 + ((lane_id >> 3) & 1) * 8;

        for (int tile_idx = 0; tile_idx < tiles_per_warp; ++tile_idx) {
            int g_idx = warp_id * tiles_per_warp + tile_idx;
            if (g_idx >= total_tiles) break;

            int tm = (g_idx / num_tiles_n) * 16;
            int tn = (g_idx % num_tiles_n) * 16;
            if (tm >= valid_q_rows || tn >= valid_kv_rows) continue;

            fragment<matrix_a, 16, 16, 16, half, row_major> a;
            fragment<matrix_b, 16, 16, 16, half, col_major> b;
            fragment<accumulator, 16, 16, 16, float> acc;
            fill_fragment(acc, 0.0f);

            for (int k = 0; k < num_tiles_k; ++k) {
                int k_off = k * 16;
                if (k_off >= D) break;
                load_matrix_sync(a, sQ + tm * D_STRIDE + k_off, D_STRIDE);
                load_matrix_sync(b, sK + tn * D_STRIDE + k_off, D_STRIDE);
                mma_sync(acc, a, b, acc);
            }

            if constexpr (IS_CAUSAL) {
                for (int i = 0; i < 8; ++i) {
                    int r = row_causal + ((i >> 1) & 1) * 2;
                    int c = col_causal + (i & 1) + ((i >> 2) & 1) * 4;
                    int gm = start_q + tm + r;
                    int gn = start_kv + tn + c;
                    bool valid = (gm < start_q + valid_q_rows) && (gn < start_kv + valid_kv_rows);
                    acc.x[i] = valid ? ((gn > gm) ? NEG_INF : acc.x[i] * softmax_scale) : NEG_INF;
                }
            } else {
                for (int i = 0; i < 8; ++i) acc.x[i] *= softmax_scale;
            }

            store_matrix_sync(sS + tm * N_STRIDE + tn, acc, N_STRIDE, mem_row_major);
        }
        __syncthreads();

        if (tid < valid_q_rows * THREADS_PER_ROW) {
            int row = tid / THREADS_PER_ROW;
            int t_in_row = tid % THREADS_PER_ROW;
            unsigned mask = (valid_q_rows == BLOCK_M) ? 0xFFFFFFFFU : __activemask();
            int leader = __ffs(mask) - 1;

            float4* sS_row = reinterpret_cast<float4*>(sS + row * N_STRIDE);
            __half2* sP_row = reinterpret_cast<__half2*>(sP + row * N_STRIDE);
            float4* sO_row = reinterpret_cast<float4*>(sO + row * D_STRIDE);

            int vec_cols = valid_kv_rows >> 2;
            int vecs_per_thread = (vec_cols + THREADS_PER_ROW - 1) / THREADS_PER_ROW;

            float thread_max = NEG_INF;
            for (int j = 0; j < vecs_per_thread; ++j) {
                int vc = t_in_row + j * THREADS_PER_ROW;
                if (vc < vec_cols) {
                    float4 v4 = sS_row[vc];
                    thread_max = fmaxf(thread_max, fmaxf(fmaxf(v4.x, v4.y), fmaxf(v4.z, v4.w)));
                }
            }
            for (int o = THREADS_PER_ROW / 2; o > 0; o >>= 1)
                thread_max = fmaxf(thread_max, __shfl_down_sync(mask, thread_max, o, THREADS_PER_ROW));
            float row_max = __shfl_sync(mask, thread_max, leader, THREADS_PER_ROW);

            float old_max = smem.row_max[row];
            float new_max = fmaxf(old_max, row_max);
            float exp_diff = __expf(old_max - new_max);

            float thread_sum = 0.0f;
            __half2 half_buffer[20]; 
            int h2_idx = 0;
            
            for (int j = 0; j < vecs_per_thread; ++j) {
                int vc = t_in_row + j * THREADS_PER_ROW;
                if (vc < vec_cols) {
                    float4 v4 = sS_row[vc];
                    float e0 = __expf(fmaxf(v4.x - new_max, -80.0f));
                    float e1 = __expf(fmaxf(v4.y - new_max, -80.0f));
                    float e2 = __expf(fmaxf(v4.z - new_max, -80.0f));
                    float e3 = __expf(fmaxf(v4.w - new_max, -80.0f));
                    thread_sum += (e0 + e1) + (e2 + e3);
                    half_buffer[h2_idx++] = __float22half2_rn(make_float2(e0, e1));
                    half_buffer[h2_idx++] = __float22half2_rn(make_float2(e2, e3));
                }
            }
            for (int o = THREADS_PER_ROW / 2; o > 0; o >>= 1)
                thread_sum += __shfl_down_sync(mask, thread_sum, o, THREADS_PER_ROW);
            float row_sum = __shfl_sync(mask, thread_sum, leader, THREADS_PER_ROW);

            if (t_in_row == 0) {
                smem.row_sum[row] = exp_diff * smem.row_sum[row] + row_sum;
                smem.row_max[row] = new_max;
            }

            h2_idx = 0;
            for (int j = 0; j < vecs_per_thread; ++j) {
                int vc = t_in_row + j * THREADS_PER_ROW;
                if (vc < vec_cols) {
                    sP_row[vc * 2]     = half_buffer[h2_idx++];
                    sP_row[vc * 2 + 1] = half_buffer[h2_idx++];
                }
            }

            if (block > 0) {
                int o_vec_count = (D_STRIDE + 3) >> 2;
                for (int ov = t_in_row; ov < o_vec_count; ov += THREADS_PER_ROW) {
                    float4 v = sO_row[ov];
                    v.x *= exp_diff; v.y *= exp_diff; v.z *= exp_diff; v.w *= exp_diff;
                    sO_row[ov] = v;
                }
            }
        }
        __syncthreads();

        uint4* sV_u4 = reinterpret_cast<uint4*>(sV);
        const uint4* v_vec = reinterpret_cast<const uint4*>(v_ptr + start_kv * D);
        for (int idx = tid; idx < (valid_kv_rows * d_stride_uint4); idx += THREADS_PER_BLOCK) {
            int row = idx / d_stride_uint4;
            int vec_col = idx % d_stride_uint4;
            uint4 v_val = (row < valid_kv_rows) ? __ldg(&v_vec[row * d_stride_uint4 + vec_col]) : make_uint4(0, 0, 0, 0);
            sV_u4[row * d_stride_uint4 + vec_col] = v_val;
        }
        __syncthreads();

        const int num_tiles_m_pv = (BLOCK_M + 15) / 16;
        const int num_tiles_n_pv = (D + 15) / 16;
        const int num_tiles_k_pv = (BLOCK_N + 15) / 16;
        const int total_tiles_pv = num_tiles_m_pv * num_tiles_n_pv;
        const int tiles_per_warp_pv = (total_tiles_pv + WARPS_PER_BLOCK - 1) / WARPS_PER_BLOCK;

        for (int tile_idx = 0; tile_idx < tiles_per_warp_pv; ++tile_idx) {
            int g_idx = warp_id * tiles_per_warp_pv + tile_idx;
            if (g_idx >= total_tiles_pv) break;

            int tm = (g_idx / num_tiles_n_pv) * 16;
            int td = (g_idx % num_tiles_n_pv) * 16;
            if (tm >= valid_q_rows) continue;

            fragment<matrix_a, 16, 16, 16, half, row_major> a;
            fragment<matrix_b, 16, 16, 16, half, row_major> b;
            fragment<accumulator, 16, 16, 16, float> acc;
            load_matrix_sync(acc, sO + tm * D_STRIDE + td, D_STRIDE, mem_row_major);

            for (int k = 0; k < num_tiles_k_pv; ++k) {
                int k_off = k * 16;
                if (k_off >= valid_kv_rows) break;
                load_matrix_sync(a, sP + tm * N_STRIDE + k_off, N_STRIDE);
                load_matrix_sync(b, sV + k_off * D_STRIDE + td, D_STRIDE);
                mma_sync(acc, a, b, acc);
            }
            store_matrix_sync(sO + tm * D_STRIDE + td, acc, D_STRIDE, mem_row_major);
        }
        __syncthreads();
    }

    float4* sO_u4 = reinterpret_cast<float4*>(sO);
    for (int i = tid; i < (valid_q_rows * D) / 4; i += THREADS_PER_BLOCK) {
        int row = i / (D / 4);
        int col = (i % (D / 4)) * 4;
        float inv_sum = 1.0f / fmaxf(smem.row_sum[row], 1e-24f);

        float4 v4 = sO_u4[row * (D_STRIDE / 4) + col / 4];

        __half h0 = __float2half_rn(v4.x * inv_sum);
        __half h1 = __float2half_rn(v4.y * inv_sum);
        __half h2 = __float2half_rn(v4.z * inv_sum);
        __half h3 = __float2half_rn(v4.w * inv_sum);

        asm volatile("st.global.v4.u16 [%0], {%1, %2, %3, %4};" :: "l"(out_ptr + row * D + col),
                     "h"(__half_as_ushort(h0)), "h"(__half_as_ushort(h1)), 
                     "h"(__half_as_ushort(h2)), "h"(__half_as_ushort(h3)) : "memory");
    }

    if (tid < valid_q_rows) {
        softmax_lse_ptr[tid] = smem.row_max[tid] + logf(fmaxf(smem.row_sum[tid], 1e-24f));
    }
}

#define CHECK_CUDA(call) do { \
    cudaError_t e = call; \
    if(e != cudaSuccess) { \
        printf("CUDA Err: %s\n", cudaGetErrorString(e)); \
        exit(1); \
    } \
} while(0)

void cpu_attention(const std::vector<float>& Q, const std::vector<float>& K, const std::vector<float>& V,
                   std::vector<float>& Out, int M, int N, int D, float scale, bool causal) {
    for (int i = 0; i < M; ++i) {
        std::vector<float> S(N); 
        float max_val = -1e30f;
        for (int j = 0; j < N; ++j) {
            if (causal && j > i) { S[j] = -1e30f; continue; }
            float sum = 0; 
            for (int k = 0; k < D; ++k) sum += Q[i * D + k] * K[j * D + k];
            S[j] = sum * scale; 
            max_val = std::max(max_val, S[j]);
        }
        float sum_exp = 0;
        for (int j = 0; j < N; ++j) { 
            S[j] = expf(S[j] - max_val); 
            sum_exp += S[j]; 
        }
        for (int j = 0; j < N; ++j) S[j] /= sum_exp;
        for (int k = 0; k < D; ++k) {
            float sum = 0; 
            for (int j = 0; j < N; ++j) sum += S[j] * V[j * D + k];
            Out[i * D + k] = sum;
        }
    }
}

template<int D, bool IS_CAUSAL>
void run_test(int B, int H, int M, int N, float scale) {
    using Config = KernelConfig<D>;
    printf("\n=== FA2 Forward (D=%d, N_STRIDE=%d, SMEM=%.2f KB) ===\n", D, Config::N_STRIDE, Config::TOTAL_SMEM / 1024.0f);

    size_t q_size = B * H * M * D * sizeof(__half);
    size_t k_size = B * H * N * D * sizeof(__half);
    size_t v_size = k_size;
    size_t o_size = q_size;
    size_t lse_size = B * H * M * sizeof(float);

    __half *d_Q, *d_K, *d_V, *d_Out; 
    float *d_LSE;

    CHECK_CUDA(cudaMalloc(&d_Q, q_size)); 
    CHECK_CUDA(cudaMalloc(&d_K, k_size));
    CHECK_CUDA(cudaMalloc(&d_V, v_size)); 
    CHECK_CUDA(cudaMalloc(&d_Out, o_size)); 
    CHECK_CUDA(cudaMalloc(&d_LSE, lse_size));

    std::vector<__half> h_Q(M * D), h_K(N * D), h_V(N * D);
    std::vector<float> h_Q_f(M * D), h_K_f(N * D), h_V_f(N * D);
    srand(42);

    for(int i = 0; i < M * D; ++i) { 
        h_Q_f[i] = ((float)rand() / RAND_MAX - 0.5f) * 2.0f; 
        h_Q[i] = __float2half(h_Q_f[i]); 
    }
    for(int i = 0; i < N * D; ++i) { 
        h_K_f[i] = ((float)rand() / RAND_MAX - 0.5f) * 2.0f; 
        h_K[i] = __float2half(h_K_f[i]); 
    }
    for(int i = 0; i < N * D; ++i) { 
        h_V_f[i] = ((float)rand() / RAND_MAX - 0.5f) * 2.0f; 
        h_V[i] = __float2half(h_V_f[i]); 
    }

    CHECK_CUDA(cudaMemcpy(d_Q, h_Q.data(), q_size, cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(d_K, h_K.data(), k_size, cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(d_V, h_V.data(), v_size, cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemset(d_Out, 0, o_size)); 
    CHECK_CUDA(cudaMemset(d_LSE, 0, lse_size));

    dim3 grid((M + Config::BLOCK_M - 1) / Config::BLOCK_M, 1, B * H);
    dim3 block(Config::THREADS_PER_BLOCK);
    auto kernel = flash_attention_forward_kernel<D, IS_CAUSAL>;

    CHECK_CUDA(cudaFuncSetAttribute(kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, Config::TOTAL_SMEM));
    kernel<<<grid, block, Config::TOTAL_SMEM>>>(d_Q, d_K, d_V, d_Out, d_LSE, B, H, M, N, scale);
    CHECK_CUDA(cudaDeviceSynchronize());

    std::vector<__half> h_Out(M * D);
    CHECK_CUDA(cudaMemcpy(h_Out.data(), d_Out, o_size, cudaMemcpyDeviceToHost));

    std::vector<float> h_Out_ref(M * D);
    cpu_attention(h_Q_f, h_K_f, h_V_f, h_Out_ref, M, N, D, scale, IS_CAUSAL);

    float max_err = 0.0f;
    for(int i = 0; i < M * D; ++i) {
        max_err = std::max(max_err, fabsf(__half2float(h_Out[i]) - h_Out_ref[i]));
    }
    printf("Validation Max Error: %.5f %s\n", max_err, max_err < 5e-2f ? "PASS" : "FAIL");

    cudaFree(d_Q); cudaFree(d_K); cudaFree(d_V); cudaFree(d_Out); cudaFree(d_LSE);
}

int main() {
    run_test<128, true>(1, 1, 128, 128, 0.125f);
    return 0;
}