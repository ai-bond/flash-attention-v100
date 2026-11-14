// ======================================================================================
// * Copyright (c) 2026, D.Skryabin / tg @ai_bond007 SPDX-License: BSD-3-Clause
// ======================================================================================
#include <cuda_fp16.h>
#include <cstdint>
#include <cstdio>
#include <cmath>
#include <cuda_runtime.h>

#define TILE_M 8
#define TILE_N 8
#define TILE_K 4
#define TILE_A_HALVES (TILE_M * TILE_K)   // 32
#define TILE_B_HALVES (TILE_K * TILE_N)   // 32
#define TILE_C_FLOATS (TILE_M * TILE_N)   // 64
#define FRAG_FLOATS   8

#ifdef USE_VOLTA_MMA
    #include "mma_m8n8k4.h"
    #include "swizzle.h"
#else
    #include "mma_native.h"
#endif
using namespace volta::wmma;

using FragmentA = fragment<matrix_a,    TILE_M, TILE_N, TILE_K, half, col_major>;
using FragmentB = fragment<matrix_b,    TILE_M, TILE_N, TILE_K, half, col_major>;
using FragmentC = fragment<accumulator, TILE_M, TILE_N, TILE_K, float>;

// ======================================================================================
// fill A col_major:
//   idx = k*M + i  (i in [0,M), k in [0,K)) ; logical A[i][k]
//   linear_bytes = k*M*2 + i*2 = 2*(k*M+i)
//   swz          = ((k&1)<<5) | ((i/4)<<4)
//   phys         = linear_bytes ^ swz
// ======================================================================================
__device__ __forceinline__ void
fill_shared_a(const half* __restrict__ g, half* smem)
{
#ifdef USE_VOLTA_MMA
    const uint32_t smem_base = __cvta_generic_to_shared(smem);
    for (int idx = threadIdx.x; idx < TILE_A_HALVES; idx += 32) {
        const int i            = idx % TILE_M;
        const int k            = idx / TILE_M;
        const int linear_bytes = k * TILE_M * 2 + i * 2;
        const int swz          = ((k & 1) << 5) | ((i / 4) << 4);
        const int phys         = linear_bytes ^ swz;
        asm volatile("st.shared.u16 [%0], %1;" :: "r"(smem_base + phys), "h"(__half_as_ushort(g[idx])) : "memory");
    }
#else
    for (int idx = threadIdx.x; idx < TILE_A_HALVES; idx += 32) smem[idx] = g[idx];
#endif
}

__device__ __forceinline__ void
fill_shared_b(const half* __restrict__ g, half* smem)
{
    for (int idx = threadIdx.x; idx < TILE_B_HALVES; idx += 32) smem[idx] = g[idx];
}

__global__ void kernel(const half* ga, const half* gb,
                       uint32_t* frag_d, uint32_t* smem_d_dump)
{
    if (threadIdx.x >= 32) return;

    __shared__ half     sa[TILE_A_HALVES], sb[TILE_B_HALVES];
    __shared__ float    sd[TILE_C_FLOATS];
    __shared__ uint32_t stage[32 * FRAG_FLOATS];

    fill_shared_a(ga, sa);
    fill_shared_b(gb, sb);
    __syncthreads();

    FragmentA fa; FragmentB fb; FragmentC fc, fd;
    load_matrix_sync(fa, sa, TILE_M);   // A col_major ldm = M = 8
    load_matrix_sync(fb, sb, TILE_K);   // B col_major ldm = K = 4
    fill_fragment(fc, 0.0f);
    mma_sync(fd, fa, fb, fc);           // col × col

    const uint32_t* src = reinterpret_cast<const uint32_t*>(fd.x);
    uint32_t* dst = stage + threadIdx.x * FRAG_FLOATS;
    #pragma unroll
    for (int i = 0; i < FRAG_FLOATS; ++i) dst[i] = src[i];
    __syncthreads();
    #pragma unroll
    for (int i = 0; i < FRAG_FLOATS; ++i)
        frag_d[threadIdx.x * FRAG_FLOATS + i] = stage[threadIdx.x * FRAG_FLOATS + i];
    __syncthreads();

    store_matrix_sync(sd, fd, TILE_N, mem_row_major);
    __syncthreads();
    const uint32_t* sh = reinterpret_cast<const uint32_t*>(sd);
    for (int idx = threadIdx.x; idx < TILE_C_FLOATS; idx += 32) smem_d_dump[idx] = sh[idx];
}

static void build_a(half* a) {
    for (int i = 0; i < TILE_M; ++i)
        for (int k = 0; k < TILE_K; ++k)
            a[k * TILE_M + i] = __float2half((float)(i * TILE_K + k));
}
static void build_b(half* b) {
    for (int k = 0; k < TILE_K; ++k)
        for (int j = 0; j < TILE_N; ++j)
            b[j * TILE_K + k] = __float2half((float)(k * TILE_N + j));
}
static void cpu_ref(const half* a, const half* b, float* c) {
    for (int i = 0; i < TILE_M; ++i)
        for (int j = 0; j < TILE_N; ++j) {
            float s = 0;
            for (int k = 0; k < TILE_K; ++k)
                s += __half2float(a[k*TILE_M+i]) * __half2float(b[j*TILE_K+k]);
            c[i*TILE_N+j] = s;
        }
}
static void extract_c(const uint32_t* fd, float* c) {
    for (int lane = 0; lane < 32; ++lane) {
        const int r0 = (lane & 1) + ((lane >> 4) & 1) * 4;
        const int c0 = lane & 2;
        const float* f = reinterpret_cast<const float*>(&fd[lane * FRAG_FLOATS]);
        c[(r0    )*TILE_N + c0    ] = f[0]; c[(r0    )*TILE_N + c0 + 1] = f[1];
        c[(r0 + 2)*TILE_N + c0    ] = f[2]; c[(r0 + 2)*TILE_N + c0 + 1] = f[3];
        c[(r0    )*TILE_N + c0 + 4] = f[4]; c[(r0    )*TILE_N + c0 + 5] = f[5];
        c[(r0 + 2)*TILE_N + c0 + 4] = f[6]; c[(r0 + 2)*TILE_N + c0 + 5] = f[7];
    }
}
static void print_m(const char* t, const float* m) {
    printf("\n%s\n", t);
    for (int i = 0; i < TILE_M; ++i) {
        printf("  row %d: ", i);
        for (int j = 0; j < TILE_N; ++j) printf("%7.1f ", m[i*TILE_N+j]);
        printf("\n");
    }
}
static void print_smem(const uint32_t* d) {
    printf("\nShared-memory layout of smem_D (PHYSICAL):\n");
    const float* s = reinterpret_cast<const float*>(d);
    for (int i = 0; i < TILE_M; ++i) {
        printf("  s%2d: ", i);
        for (int j = 0; j < TILE_N; ++j) printf("%7.1f ", s[i*TILE_N+j]);
        printf("\n");
    }
}

int main(void)
{
#ifdef USE_VOLTA_MMA
    printf("=== (mma_m8n8k4.h) === GEMM 8x8x4  A.col x B.col\n");
#else
    printf("=== (mma_native.h) === GEMM 8x8x4  A.col x B.col\n");
#endif
    half ha[TILE_A_HALVES], hb[TILE_B_HALVES];
    float ref[TILE_C_FLOATS];
    build_a(ha); build_b(hb); cpu_ref(ha, hb, ref);

    half *da, *db; uint32_t *dfd, *dsd;
    cudaMalloc(&da,  TILE_A_HALVES  * sizeof(half));
    cudaMalloc(&db,  TILE_B_HALVES  * sizeof(half));
    cudaMalloc(&dfd, 32 * FRAG_FLOATS * sizeof(uint32_t));
    cudaMalloc(&dsd, TILE_C_FLOATS * sizeof(uint32_t));
    cudaMemcpy(da, ha, TILE_A_HALVES * sizeof(half), cudaMemcpyHostToDevice);
    cudaMemcpy(db, hb, TILE_B_HALVES * sizeof(half), cudaMemcpyHostToDevice);

    kernel<<<1, 32>>>(da, db, dfd, dsd);
    cudaError_t st = cudaDeviceSynchronize();
    if (st != cudaSuccess) { printf("CUDA error: %s\n", cudaGetErrorString(st)); return 1; }

    uint32_t hfd[32 * FRAG_FLOATS], hsd[TILE_C_FLOATS];
    cudaMemcpy(hfd, dfd, sizeof(hfd), cudaMemcpyDeviceToHost);
    cudaMemcpy(hsd, dsd, sizeof(hsd), cudaMemcpyDeviceToHost);
    float got[TILE_C_FLOATS];
    extract_c(hfd, got);

    print_m("CPU ref:", ref);
    print_m("GPU C (from fragment):", got);
    print_smem(hsd);

    bool ok = true;
    for (int i = 0; i < TILE_C_FLOATS; ++i)
        if (std::fabs(got[i] - ref[i]) > 1e-2f) { ok = false; break; }
    printf("\n%s: A.col x B.col GEMM\n", ok ? "PASS" : "FAIL");

    cudaFree(da); cudaFree(db); cudaFree(dfd); cudaFree(dsd);
    return ok ? 0 : 1;
}
