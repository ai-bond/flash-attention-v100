// ======================================================================================
// * Copyright (c) 2026, D.Skryabin / tg @ai_bond007 SPDX-License: BSD-3-Clause
// ======================================================================================
#include <cuda_fp16.h>
#include <cstdint>
#include <cstdio>
#include <cuda_runtime.h>

#define TILE_M          8
#define TILE_N          8
#define TILE_K          4
#define TILE_B_HALVES   (TILE_K * TILE_N)
#define TILE_B_U32      (TILE_B_HALVES / 2)
#define FRAG_HALVES     4
#define FRAG_U32        2
#define BENCH_ITER      10000
#define B_IS_COL_MAJOR  1

#ifdef USE_VOLTA_MMA
    #include "mma_m8n8k4.h"
    #include "swizzle.h"
#else
    #include "mma_native.h"
#endif
    using namespace volta::wmma;

#if B_IS_COL_MAJOR
    using FragmentB = fragment<matrix_b, TILE_M, TILE_N, TILE_K, half, col_major>;
#else
    using FragmentB = fragment<matrix_b, TILE_M, TILE_N, TILE_K, half, row_major>;
#endif

__device__ __forceinline__ void
fill_shared_memory(const half* __restrict__ global_b, half* shared_b)
{
    for (int idx = threadIdx.x; idx < TILE_B_HALVES; idx += 32) {
        shared_b[idx] = global_b[idx];
    }
}

__global__ void dump_b_regs(
    const half* __restrict__ global_b,
    uint32_t*   __restrict__ register_dump,
    uint32_t*   __restrict__ shared_dump,
    long long*  __restrict__ cycles_out)
{
    if (threadIdx.x >= 32) return;

    __shared__ half     shared_b[TILE_B_HALVES];
    __shared__ uint32_t staging[32 * FRAG_U32];

    fill_shared_memory(global_b, shared_b);
    __syncthreads();

    const uint32_t* shared_as_u32 = reinterpret_cast<const uint32_t*>(shared_b);
    for (int idx = threadIdx.x; idx < TILE_B_U32; idx += 32) shared_dump[idx] = shared_as_u32[idx];
    __syncthreads();

    FragmentB frag;
    load_matrix_sync(frag, shared_b, B_IS_COL_MAJOR ? TILE_K : TILE_N);

    const uint32_t* fragment_regs = reinterpret_cast<const uint32_t*>(frag.x);
    uint32_t* lane_dst = staging + threadIdx.x * FRAG_U32;
    #pragma unroll
    for (int reg = 0; reg < FRAG_U32; reg++) lane_dst[reg] = fragment_regs[reg];
    __syncthreads();

    #pragma unroll
    for (int reg = 0; reg < FRAG_U32; reg++)
        register_dump[threadIdx.x * FRAG_U32 + reg] = staging[threadIdx.x * FRAG_U32 + reg];
    __syncthreads();

    long long cycles_start = clock64();
    unsigned  checksum     = 0;
    #pragma unroll 1
    for (int iteration = 0; iteration < BENCH_ITER; ++iteration) {
        fill_shared_memory(global_b, shared_b);
        __syncwarp();
        FragmentB bench_frag;
        load_matrix_sync(bench_frag, shared_b, B_IS_COL_MAJOR ? TILE_K : TILE_N);
        checksum += (unsigned)__half2float(bench_frag.x[0])
                  + (unsigned)__half2float(bench_frag.x[3]);
    }
    long long cycles_end = clock64();

    if (threadIdx.x == 0) {
        cycles_out[0] = cycles_end - cycles_start;
        cycles_out[1] = (long long)checksum;
    }
}

static void build_global_b(half* global_b)
{
    // Logical B[k][n] = k*TILE_N + n  (k in [0,K), n in [0,N))
#if B_IS_COL_MAJOR
    for (int n = 0; n < TILE_N; ++n)
        for (int k = 0; k < TILE_K; ++k)
            global_b[n * TILE_K + k] = __float2half((float)(k * TILE_N + n));
#else
    for (int k = 0; k < TILE_K; ++k)
        for (int n = 0; n < TILE_N; ++n)
            global_b[k * TILE_N + n] = __float2half((float)(k * TILE_N + n));
#endif
}

static void print_expected(void)
{
    printf("\nExpected logical B[row][col] = row*%d + col  (%d x %d):\n", TILE_N, TILE_K, TILE_N);
    for (int row = 0; row < TILE_K; ++row) {
        printf("  row %2d: ", row);
        for (int col = 0; col < TILE_N; ++col) printf("%3d ", row * TILE_N + col);
        printf("\n");
    }
}

static void print_shared_layout(const uint32_t* shared_dump)
{
#if B_IS_COL_MAJOR
    printf("\nShared-memory layout of smem_B (PHYSICAL, %d lines x %d halves):\n", TILE_N, TILE_K);
#else
    printf("\nShared-memory layout of smem_B (PHYSICAL, %d lines x %d halves):\n", TILE_K, TILE_N);
#endif
    const half* sh = reinterpret_cast<const half*>(shared_dump);
    for (int line = 0; line < LINES; ++line) {
        printf("  s%2d: ", line);
        for (int item = 0; item < ITEMS; ++item) printf("%4.0f ", __half2float(sh[line * ITEMS + item]));
        printf("\n");
    }
}

static void print_fragment_registers(const uint32_t* register_dump)
{
    printf("\nFragment register dump (LOGICAL, 32 lanes x %d halves):\n", FRAG_HALVES);
    for (int lane = 0; lane < 32; ++lane) {
        printf("L%2d: ", lane);
        const half* lane_halves = reinterpret_cast<const half*>(&register_dump[lane * FRAG_U32]);
        for (int item = 0; item < FRAG_HALVES; ++item) printf("%4.0f ", __half2float(lane_halves[item]));
        printf("\n");
    }
}

int main(void)
{
#ifdef USE_VOLTA_MMA
    printf("=== (mma_m8n8k4.h + swizzle.h)  ===  B layout: %s\n",  B_IS_COL_MAJOR ? "col_major" : "row_major");
#else
    printf("=== (mma_native.h, linear smem) ===  B layout: %s\n", B_IS_COL_MAJOR ? "col_major" : "row_major");
#endif

    half host_b[TILE_B_HALVES];
    build_global_b(host_b);

    half*      device_b;
    uint32_t*  device_registers;
    uint32_t*  device_shared;
    long long* device_cycles;
    cudaMalloc(&device_b,         TILE_B_HALVES    * sizeof(half));
    cudaMalloc(&device_registers, 32 * FRAG_U32    * sizeof(uint32_t));
    cudaMalloc(&device_shared,    TILE_B_U32       * sizeof(uint32_t));
    cudaMalloc(&device_cycles,    2                * sizeof(long long));

    cudaMemcpy(device_b, host_b, TILE_B_HALVES * sizeof(half), cudaMemcpyHostToDevice);
    cudaMemset(device_cycles, 0, 2 * sizeof(long long));

    dump_b_regs<<<1, 32>>>(device_b, device_registers, device_shared, device_cycles);
    cudaError_t status = cudaDeviceSynchronize();
    if (status != cudaSuccess) {
        printf("CUDA error: %s\n", cudaGetErrorString(status));
        return 1;
    }

    uint32_t  host_registers[32 * FRAG_U32];
    uint32_t  host_shared[TILE_B_U32];
    long long host_cycles[2];
    cudaMemcpy(host_registers, device_registers, sizeof(host_registers), cudaMemcpyDeviceToHost);
    cudaMemcpy(host_shared,    device_shared,    sizeof(host_shared),    cudaMemcpyDeviceToHost);
    cudaMemcpy(host_cycles,    device_cycles,    sizeof(host_cycles),    cudaMemcpyDeviceToHost);

    print_expected();
    print_shared_layout(host_shared);
    print_fragment_registers(host_registers);

    printf("\nBench: iterations=%d  cycles=%lld  per-iter=%.2f cyc  checksum=0x%llx\n",
           BENCH_ITER, host_cycles[0],
           (double)host_cycles[0] / (double)BENCH_ITER,
           (unsigned long long)host_cycles[1]);

    cudaFree(device_b);
    cudaFree(device_registers);
    cudaFree(device_shared);
    cudaFree(device_cycles);
    return 0;
}
