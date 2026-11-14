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
#define TILE_A_HALVES   (TILE_M * TILE_K)   // 32
#define TILE_A_U32      (TILE_A_HALVES / 2) // 16
#define FRAG_U32_REGS   2                   // A: half[4] = 2 uint32
#define FRAG_HALVES     (FRAG_U32_REGS * 2) // 4

#define BENCH_ITER      10000
#define A_IS_COL_MAJOR  1

#ifdef USE_VOLTA_MMA
    #include "mma_m8n8k4.h"
    #include "swizzle.h"
#else
    #include "mma_native.h"
#endif

using namespace volta::wmma;

#if A_IS_COL_MAJOR
    using FragmentA = fragment<matrix_a, TILE_M, TILE_N, TILE_K, half, col_major>;
#else
    using FragmentA = fragment<matrix_a, TILE_M, TILE_N, TILE_K, half, row_major>;
#endif

__device__ __forceinline__ void
fill_shared_memory(const half* __restrict__ global_a, half* shared_a)
{
#ifdef USE_VOLTA_MMA
    const uint32_t smem_base = __cvta_generic_to_shared(shared_a);
    for (int idx = threadIdx.x; idx < TILE_A_HALVES; idx += 32) {
        const int      row      = idx / TILE_K;
        const uint32_t byte_off = (uint32_t)idx * (uint32_t)sizeof(half);
        st_half(smem_base + byte_off, global_a[idx], row);
    }
#else
    for (int idx = threadIdx.x; idx < TILE_A_HALVES; idx += 32)
        shared_a[idx] = global_a[idx];
#endif
}

__global__ void dump_a_regs(
    const half* __restrict__ global_a,
    uint32_t*   __restrict__ register_dump,
    uint32_t*   __restrict__ shared_dump,
    long long*  __restrict__ cycles_out)
{
    if (threadIdx.x >= 32) return;

    __shared__ half     shared_a[TILE_A_HALVES];
    __shared__ uint32_t staging[32 * FRAG_U32_REGS];

    fill_shared_memory(global_a, shared_a);
    __syncthreads();

    const uint32_t* shared_as_u32 = reinterpret_cast<const uint32_t*>(shared_a);
    for (int idx = threadIdx.x; idx < TILE_A_U32; idx += 32) shared_dump[idx] = shared_as_u32[idx];
    __syncthreads();

    FragmentA frag;
    load_matrix_sync(frag, shared_a, A_IS_COL_MAJOR ? TILE_M : TILE_K);

    const uint32_t* fragment_regs = reinterpret_cast<const uint32_t*>(frag.x);
    uint32_t* lane_dst = staging + threadIdx.x * FRAG_U32_REGS;
    #pragma unroll
    for (int reg = 0; reg < FRAG_U32_REGS; reg++) lane_dst[reg] = fragment_regs[reg];
    __syncthreads();

    #pragma unroll
    for (int reg = 0; reg < FRAG_U32_REGS; reg++)
        register_dump[threadIdx.x * FRAG_U32_REGS + reg] = staging[threadIdx.x * FRAG_U32_REGS + reg];
    __syncthreads();

    long long cycles_start = clock64();
    unsigned  checksum     = 0;
    #pragma unroll 1
    for (int iteration = 0; iteration < BENCH_ITER; ++iteration) {
        fill_shared_memory(global_a, shared_a);
        __syncwarp();
        FragmentA bench_frag;
        load_matrix_sync(bench_frag, shared_a, A_IS_COL_MAJOR ? TILE_M : TILE_K);
        checksum += (unsigned)__half2float(bench_frag.x[0])
                  + (unsigned)__half2float(bench_frag.x[3]);
    }
    long long cycles_end = clock64();

    if (threadIdx.x == 0) {
        cycles_out[0] = cycles_end - cycles_start;
        cycles_out[1] = (long long)checksum;
    }
}

static void build_global_a(half* global_a)
{
    // logical A[r][c] = r*TILE_K + c
#if A_IS_COL_MAJOR
    for (int r = 0; r < TILE_M; ++r)
        for (int c = 0; c < TILE_K; ++c)
            global_a[r + c*TILE_M] = __float2half((float)(r*TILE_K + c));
#else
    for (int r = 0; r < TILE_M; ++r)
        for (int c = 0; c < TILE_K; ++c)
            global_a[r*TILE_K + c] = __float2half((float)(r*TILE_K + c));
#endif
}

static void print_expected(void)
{
    printf("\nExpected logical A[row][col] = row*%d + col  (%d x %d):\n",  TILE_K, TILE_M, TILE_K);
    for (int row = 0; row < TILE_M; ++row) {
        printf("  row %2d: ", row);
        for (int col = 0; col < TILE_K; ++col)
            printf("%3d ", row * TILE_K + col);
        printf("\n");
    }
}

static void print_shared_layout(const uint32_t* shared_dump)
{
    printf("\nShared-memory layout of smem_A (PHYSICAL, %d lines x %d halves):\n",  TILE_M, TILE_K);
    const half* shared_halves = reinterpret_cast<const half*>(shared_dump);
    for (int line = 0; line < TILE_M; ++line) {
        printf("  s%2d: ", line);
        for (int item = 0; item < TILE_K; ++item)
            printf("%4.0f ", __half2float(shared_halves[line * TILE_K + item]));
        printf("\n");
    }
}

static void print_fragment_registers(const uint32_t* register_dump)
{
    printf("\nFragment register dump (LOGICAL, 32 lanes x %d halves):\n", FRAG_HALVES);
    for (int lane = 0; lane < 32; ++lane) {
        printf("L%2d: ", lane);
        const half* lane_halves =
            reinterpret_cast<const half*>(&register_dump[lane * FRAG_U32_REGS]);
        for (int item = 0; item < FRAG_HALVES; ++item)
            printf("%4.0f ", __half2float(lane_halves[item]));
        printf("\n");
    }
}

int main(void)
{
#ifdef USE_VOLTA_MMA
    printf("=== (mma_m8n8k4.h + swizzle.h)  ===  A layout: %s\n", A_IS_COL_MAJOR ? "col_major" : "row_major");
#else
    printf("=== (mma_native.h, linear smem) ===  A layout: %s\n", A_IS_COL_MAJOR ? "col_major" : "row_major");
#endif

    half host_a[TILE_A_HALVES];
    build_global_a(host_a);

    half*      device_a;
    uint32_t*  device_registers;
    uint32_t*  device_shared;
    long long* device_cycles;
    cudaMalloc(&device_a,         TILE_A_HALVES                  * sizeof(half));
    cudaMalloc(&device_registers, 32 * FRAG_U32_REGS             * sizeof(uint32_t));
    cudaMalloc(&device_shared,    TILE_A_U32                     * sizeof(uint32_t));
    cudaMalloc(&device_cycles,    2                              * sizeof(long long));

    cudaMemcpy(device_a, host_a, TILE_A_HALVES * sizeof(half), cudaMemcpyHostToDevice);
    cudaMemset(device_cycles, 0, 2 * sizeof(long long));

    dump_a_regs<<<1, 32>>>(device_a, device_registers, device_shared, device_cycles);
    cudaError_t status = cudaDeviceSynchronize();
    if (status != cudaSuccess) {
        printf("CUDA error: %s\n", cudaGetErrorString(status));
        return 1;
    }

    uint32_t  host_registers[32 * FRAG_U32_REGS];
    uint32_t  host_shared[TILE_A_U32];
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

    cudaFree(device_a);
    cudaFree(device_registers);
    cudaFree(device_shared);
    cudaFree(device_cycles);
    return 0;
}
