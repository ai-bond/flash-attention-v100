// ======================================================================================
// * Copyright (c) 2026, D.Skryabin / tg @ai_bond007 SPDX-License: BSD-3-Clause
// ======================================================================================
#include <cuda_fp16.h>
#include <cstdint>
#include <cstdio>
#include <cuda_runtime.h>

#define TILE_M           8
#define TILE_N           8
#define TILE_K           4
#define TILE_FLOATS      (TILE_M * TILE_N)
#define FRAG_FLOATS      8
#define BENCH_ITER       10000
#define ACC_IS_COL_MAJOR 0

#ifdef USE_VOLTA_MMA
    #include "mma_m8n8k4.h"
    #include "swizzle.h"
#else
    #include "mma_native.h"
#endif
    using namespace volta::wmma;

using FragmentAcc = fragment<accumulator, TILE_M, TILE_N, TILE_K, float>;

#if ACC_IS_COL_MAJOR
    #define ACC_LAYOUT mem_col_major
#else
    #define ACC_LAYOUT mem_row_major
#endif

__device__ __forceinline__ void
fill_shared_acc(const float* __restrict__ global_acc, float* shared_acc)
{
#ifdef USE_VOLTA_MMA
    constexpr int CHUNK_FLOATS  = 4;                             // 16 B
    constexpr int CHUNKS_PER_LD = TILE_N / CHUNK_FLOATS;         // 2
    constexpr int TOTAL_CHUNKS  = TILE_M * CHUNKS_PER_LD;        // 16

    const uint32_t smem_base = __cvta_generic_to_shared(shared_acc);
    for (int i = threadIdx.x; i < TOTAL_CHUNKS; i += 32) {
        const int      outer  = i / CHUNKS_PER_LD;
        const int      inner  = (i % CHUNKS_PER_LD) * CHUNK_FLOATS;
        const int      offset = outer * TILE_N + inner;
        const uint32_t addr   = smem_base + (uint32_t)offset * (uint32_t)sizeof(float);
        const float4   v      = *reinterpret_cast<const float4*>(&global_acc[offset]);
        st_float4(addr, v, outer);
    }
#else
    for (int idx = threadIdx.x; idx < TILE_FLOATS; idx += 32) {
        shared_acc[idx] = global_acc[idx];
    }
#endif
}

__global__ void dump_acc_regs(
    const float* __restrict__ global_acc,
    uint32_t*    __restrict__ register_dump,
    uint32_t*    __restrict__ shared_dump,
    long long*   __restrict__ cycles_out)
{
    if (threadIdx.x >= 32) return;

    __shared__ float    shared_acc[TILE_FLOATS];
    __shared__ uint32_t staging[32 * FRAG_FLOATS];

    fill_shared_acc(global_acc, shared_acc);
    __syncthreads();

    const uint32_t* shared_as_u32 = reinterpret_cast<const uint32_t*>(shared_acc);
    for (int idx = threadIdx.x; idx < TILE_FLOATS; idx += 32) {
        shared_dump[idx] = shared_as_u32[idx];
    }
    __syncthreads();

    FragmentAcc frag;
    load_matrix_sync(frag, shared_acc, TILE_M, ACC_LAYOUT);

    const uint32_t* fragment_regs = reinterpret_cast<const uint32_t*>(frag.x);
    uint32_t* lane_dst = staging + threadIdx.x * FRAG_FLOATS;
    #pragma unroll
    for (int reg = 0; reg < FRAG_FLOATS; reg++) {
        lane_dst[reg] = fragment_regs[reg];
    }
    __syncthreads();

    #pragma unroll
    for (int reg = 0; reg < FRAG_FLOATS; reg++) {
        register_dump[threadIdx.x * FRAG_FLOATS + reg] = staging[threadIdx.x * FRAG_FLOATS + reg];
    }
    __syncthreads();

    long long cycles_start = clock64();
    unsigned  checksum     = 0;
    #pragma unroll 1
    for (int iteration = 0; iteration < BENCH_ITER; ++iteration) {
        fill_shared_acc(global_acc, shared_acc);
        __syncwarp();
        FragmentAcc bench_frag;
        load_matrix_sync(bench_frag, shared_acc, TILE_M, ACC_LAYOUT);
        checksum += __float_as_uint(bench_frag.x[0])
                  + __float_as_uint(bench_frag.x[7]);
    }
    long long cycles_end = clock64();

    if (threadIdx.x == 0) {
        cycles_out[0] = cycles_end - cycles_start;
        cycles_out[1] = (long long)checksum;
    }
}

static void build_global_acc(float* global_acc)
{
    // Logical Acc[r][c] = r*8 + c ; storage follows ACC_IS_COL_MAJOR
    for (int r = 0; r < TILE_M; ++r)
        for (int c = 0; c < TILE_N; ++c)
#if ACC_IS_COL_MAJOR
            global_acc[c * TILE_M + r] = (float)(r * TILE_N + c);
#else
            global_acc[r * TILE_N + c] = (float)(r * TILE_N + c);
#endif
}

static void print_expected(void)
{
    printf("\nExpected logical Acc[row][col] = row*%d + col  (%d x %d):\n",  TILE_N, TILE_M, TILE_N);
    for (int row = 0; row < TILE_M; ++row) {
        printf("  row %2d: ", row);
        for (int col = 0; col < TILE_N; ++col) printf("%3d ", row * TILE_N + col);
        printf("\n");
    }
}

static void print_shared_layout(const uint32_t* shared_dump)
{
    printf("\nShared-memory layout of smem_Acc (PHYSICAL, %d lines x %d floats):\n", TILE_M, TILE_N);
    const float* sh = reinterpret_cast<const float*>(shared_dump);
    for (int line = 0; line < TILE_M; ++line) {
        printf("  s%2d: ", line);
        for (int item = 0; item < TILE_N; ++item) printf("%5.0f ", sh[line * TILE_N + item]);
        printf("\n");
    }
}

static void print_fragment_registers(const uint32_t* register_dump)
{
    printf("\nFragment register dump (LOGICAL, 32 lanes x %d floats):\n", FRAG_FLOATS);
    for (int lane = 0; lane < 32; ++lane) {
        printf("L%2d: ", lane);
        const float* lane_floats = reinterpret_cast<const float*>(&register_dump[lane * FRAG_FLOATS]);
        for (int item = 0; item < FRAG_FLOATS; ++item) printf("%5.0f ", lane_floats[item]);
        printf("\n");
    }
}

int main(void)
{
#ifdef USE_VOLTA_MMA
    printf("=== (mma_m8n8k4.h + swizzle.h)  ===  Acc layout: %s\n",  ACC_IS_COL_MAJOR ? "col_major" : "row_major");
#else
    printf("=== (mma_native.h, linear smem) ===  Acc layout: %s\n", ACC_IS_COL_MAJOR ? "col_major" : "row_major");
#endif

    float host_acc[TILE_FLOATS];
    build_global_acc(host_acc);

    float*     device_acc;
    uint32_t*  device_registers;
    uint32_t*  device_shared;
    long long* device_cycles;
    cudaMalloc(&device_acc,       TILE_FLOATS        * sizeof(float));
    cudaMalloc(&device_registers, 32 * FRAG_FLOATS   * sizeof(uint32_t));
    cudaMalloc(&device_shared,    TILE_FLOATS        * sizeof(uint32_t));
    cudaMalloc(&device_cycles,    2                  * sizeof(long long));

    cudaMemcpy(device_acc, host_acc, TILE_FLOATS * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemset(device_cycles, 0, 2 * sizeof(long long));

    dump_acc_regs<<<1, 32>>>(device_acc, device_registers, device_shared, device_cycles);
    cudaError_t status = cudaDeviceSynchronize();
    if (status != cudaSuccess) { printf("CUDA error: %s\n", cudaGetErrorString(status)); return 1; }

    uint32_t  host_registers[32 * FRAG_FLOATS];
    uint32_t  host_shared[TILE_FLOATS];
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

    cudaFree(device_acc);
    cudaFree(device_registers);
    cudaFree(device_shared);
    cudaFree(device_cycles);
    return 0;
}
