// ======================================================================================
// * Copyright (c) 2026, D.Skryabin / tg @ai_bond007 SPDX-License: BSD-3-Clause
// ======================================================================================
#include <cuda_fp16.h>
#include <cstdint>
#include <cstdio>
#include <cmath>
#include <cuda_runtime.h>

#define TILE_M          8
#define TILE_N          8
#define TILE_K          4
#define TILE_A_HALVES   (TILE_M * TILE_K)   // 32
#define TILE_B_HALVES   (TILE_K * TILE_N)   // 32
#define TILE_C_FLOATS   (TILE_M * TILE_N)   // 64
#define FRAG_HALVES     4
#define FRAG_U32        2
#define FRAG_FLOATS     8

#ifdef USE_VOLTA_MMA
    #include "mma_m8n8k4.h"     // swizzle-aware
    #include "swizzle.h"
#else
    #include "mma_native.h"     // plain reference
#endif

using namespace volta::wmma;
using FragmentA = fragment<matrix_a,    TILE_M, TILE_N, TILE_K, half, row_major>;
using FragmentB = fragment<matrix_b,    TILE_M, TILE_N, TILE_K, half, col_major>;
using FragmentC = fragment<accumulator, TILE_M, TILE_N, TILE_K, float>;

__device__ __forceinline__ void
fill_shared_a(const half* __restrict__ global_a, half* smem_a)
{
#ifdef USE_VOLTA_MMA
    const uint32_t smem_base = __cvta_generic_to_shared(smem_a);
    for (int idx = threadIdx.x; idx < TILE_A_HALVES; idx += 32) {
        const int      outer    = idx / TILE_K;
        const uint32_t byte_off = (uint32_t)idx * (uint32_t)sizeof(half);
        st_half(smem_base + byte_off, global_a[idx], outer);
    }
#else
    for (int idx = threadIdx.x; idx < TILE_A_HALVES; idx += 32) {
        smem_a[idx] = global_a[idx];
    }
#endif
}

__device__ __forceinline__ void
fill_shared_b(const half* __restrict__ global_b, half* smem_b)
{
    for (int idx = threadIdx.x; idx < TILE_B_HALVES; idx += 32) {
        smem_b[idx] = global_b[idx];
    }
}

__global__ void gemm_8x8x4_kernel(
    const half* __restrict__ global_a,
    const half* __restrict__ global_b,
    uint32_t*   __restrict__ frag_d_dump,
    uint32_t*   __restrict__ smem_d_dump)
{
    if (threadIdx.x >= 32) return;

    __shared__ half     smem_a[TILE_A_HALVES];
    __shared__ half     smem_b[TILE_B_HALVES];
    __shared__ float    smem_d[TILE_C_FLOATS];
    __shared__ uint32_t staging[32 * FRAG_FLOATS];

    fill_shared_a(global_a, smem_a);
    fill_shared_b(global_b, smem_b);
    __syncthreads();

    FragmentA frag_a;
    FragmentB frag_b;
    FragmentC frag_c, frag_d;

    load_matrix_sync(frag_a, smem_a, TILE_K);     // ldm = K = 4 (A row_major)
    load_matrix_sync(frag_b, smem_b, TILE_K);     // ldm = K = 4 (B col_major)

    fill_fragment(frag_c, 0.0f);
    mma_sync(frag_d, frag_a, frag_b, frag_c);     // D = A @ B + C

    const uint32_t* src = reinterpret_cast<const uint32_t*>(frag_d.x);
    uint32_t* dst = staging + threadIdx.x * FRAG_FLOATS;
    #pragma unroll
    for (int i = 0; i < FRAG_FLOATS; ++i) dst[i] = src[i];
    __syncthreads();

    #pragma unroll
    for (int i = 0; i < FRAG_FLOATS; ++i)
        frag_d_dump[threadIdx.x * FRAG_FLOATS + i] = staging[threadIdx.x * FRAG_FLOATS + i];
    __syncthreads();

    store_matrix_sync(smem_d, frag_d, TILE_N, mem_row_major);
    __syncthreads();

    const uint32_t* sh = reinterpret_cast<const uint32_t*>(smem_d);
    for (int idx = threadIdx.x; idx < TILE_C_FLOATS; idx += 32) {
        smem_d_dump[idx] = sh[idx];
    }
}

static void build_global_a(half* global_a)
{
    // A[i][k] = i*K + k ; row-major storage: global_a[i*K + k]
    for (int i = 0; i < TILE_M; ++i)
        for (int k = 0; k < TILE_K; ++k)
            global_a[i * TILE_K + k] = __float2half((float)(i * TILE_K + k));
}

static void build_global_b(half* global_b)
{
    // B[k][j] = k*N + j ; col-major storage: global_b[j*K + k]
    for (int k = 0; k < TILE_K; ++k)
        for (int j = 0; j < TILE_N; ++j)
            global_b[j * TILE_K + k] = __float2half((float)(k * TILE_N + j));
}

static void compute_ref(const half* host_a, const half* host_b, float* ref_c)
{
    for (int i = 0; i < TILE_M; ++i) {
        for (int j = 0; j < TILE_N; ++j) {
            float sum = 0.0f;
            for (int k = 0; k < TILE_K; ++k) {
                sum += __half2float(host_a[i * TILE_K + k])
                     * __half2float(host_b[j * TILE_K + k]);
            }
            ref_c[i * TILE_N + j] = sum;
        }
    }
}

static void extract_frag_to_c(const uint32_t* frag_dump, float* out_c)
{
    for (int lane = 0; lane < 32; ++lane) {
        const int   r0 = (lane & 1) + ((lane >> 4) & 1) * 4;
        const int   c0 = lane & 2;
        const float* f = reinterpret_cast<const float*>(&frag_dump[lane * FRAG_FLOATS]);
        out_c[(r0    ) * TILE_N + (c0    )] = f[0];
        out_c[(r0    ) * TILE_N + (c0 + 1)] = f[1];
        out_c[(r0 + 2) * TILE_N + (c0    )] = f[2];
        out_c[(r0 + 2) * TILE_N + (c0 + 1)] = f[3];
        out_c[(r0    ) * TILE_N + (c0 + 4)] = f[4];
        out_c[(r0    ) * TILE_N + (c0 + 5)] = f[5];
        out_c[(r0 + 2) * TILE_N + (c0 + 4)] = f[6];
        out_c[(r0 + 2) * TILE_N + (c0 + 5)] = f[7];
    }
}

static bool compare_c(const float* got, const float* ref, float tol = 1e-2f)
{
    for (int i = 0; i < TILE_C_FLOATS; ++i) {
        if (std::fabs(got[i] - ref[i]) > tol) return false;
    }
    return true;
}

static void print_mat_f(const char* title, const float* m)
{
    printf("\n%s\n", title);
    for (int i = 0; i < TILE_M; ++i) {
        printf("  row %d: ", i);
        for (int j = 0; j < TILE_N; ++j) printf("%7.1f ", m[i * TILE_N + j]);
        printf("\n");
    }
}

static void print_shared_d(const uint32_t* smem_d_dump)
{
    printf("\nShared-memory layout of smem_D (PHYSICAL, %d lines x %d floats):\n",
           TILE_M, TILE_N);
    const float* sh = reinterpret_cast<const float*>(smem_d_dump);
    for (int line = 0; line < TILE_M; ++line) {
        printf("  s%2d: ", line);
        for (int item = 0; item < TILE_N; ++item) printf("%7.1f ", sh[line * TILE_N + item]);
        printf("\n");
    }
}

int main(void)
{
#ifdef USE_VOLTA_MMA
    printf("=== (mma_m8n8k4.h + swizzle.h)  ===  GEMM 8x8x4  (A.row x B.col)\n");
#else
    printf("=== (mma_native.h, linear smem) ===  GEMM 8x8x4  (A.row x B.col)\n");
#endif

    half  host_a[TILE_A_HALVES];
    half  host_b[TILE_B_HALVES];
    float ref_c[TILE_C_FLOATS];
    build_global_a(host_a);
    build_global_b(host_b);
    compute_ref(host_a, host_b, ref_c);

    half*      device_a;
    half*      device_b;
    uint32_t*  device_frag_d;
    uint32_t*  device_smem_d;
    cudaMalloc(&device_a,     TILE_A_HALVES  * sizeof(half));
    cudaMalloc(&device_b,     TILE_B_HALVES  * sizeof(half));
    cudaMalloc(&device_frag_d, 32 * FRAG_FLOATS * sizeof(uint32_t));
    cudaMalloc(&device_smem_d, TILE_C_FLOATS * sizeof(uint32_t));

    cudaMemcpy(device_a, host_a, TILE_A_HALVES * sizeof(half), cudaMemcpyHostToDevice);
    cudaMemcpy(device_b, host_b, TILE_B_HALVES * sizeof(half), cudaMemcpyHostToDevice);

    gemm_8x8x4_kernel<<<1, 32>>>(device_a, device_b, device_frag_d, device_smem_d);
    cudaError_t status = cudaDeviceSynchronize();
    if (status != cudaSuccess) {
        printf("CUDA error: %s\n", cudaGetErrorString(status));
        return 1;
    }

    uint32_t host_frag_d[32 * FRAG_FLOATS];
    uint32_t host_smem_d[TILE_C_FLOATS];
    cudaMemcpy(host_frag_d, device_frag_d, sizeof(host_frag_d), cudaMemcpyDeviceToHost);
    cudaMemcpy(host_smem_d, device_smem_d, sizeof(host_smem_d), cudaMemcpyDeviceToHost);

    float got_c[TILE_C_FLOATS];
    extract_frag_to_c(host_frag_d, got_c);

    print_mat_f("CPU reference: C = A @ B", ref_c);
    print_mat_f("GPU result   : C = A @ B (from D.fragment)", got_c);
    print_shared_d(host_smem_d);

    printf("\n");
    if (compare_c(got_c, ref_c)) {
        printf("GEMM OK: fragment D matches CPU reference (tol=1e-2)\n");
    } else {
        printf("GEMM FAILED: mismatch between fragment D and CPU reference\n");
    }

    cudaFree(device_a);
    cudaFree(device_b);
    cudaFree(device_frag_d);
    cudaFree(device_smem_d);
    return 0;
}
