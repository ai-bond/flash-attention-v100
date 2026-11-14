// ======================================================================================
// * Copyright (c) 2026, D.Skryabin / tg @ai_bond007 SPDX-License: BSD-3-Clause
// ======================================================================================
#include <cuda_fp16.h>
#include <cstdint>
#include <cstdio>
#include <cuda_runtime.h>

#define TILE_SIZE       16
#define TILE_HALVES     (TILE_SIZE * TILE_SIZE)
#define BENCH_ITER      10000
#define A_IS_COL_MAJOR  0 // 1 -> col_major, 0 -> row_major

#ifdef USE_VOLTA_MMA
    #include "mma_m16n16k16.h"
    #include "swizzle.h"
    #include "kernel.h"
    #include "gemm_smem.h"
    using namespace volta;

    struct TestConfig {
        static constexpr int DIM               = TILE_SIZE;
        static constexpr int WARPS_PER_BLOCK   = 1;
        static constexpr int THREADS_PER_BLOCK = 32;
        struct DO  { static constexpr int BLOCK_M=TILE_SIZE, BLOCK_N=TILE_SIZE, THREADS_PER_ROW=32; };
        struct DQ  { static constexpr int BLOCK_M=TILE_SIZE, BLOCK_N=TILE_SIZE, THREADS_PER_ROW=32; };
        struct DKV { static constexpr int BLOCK_M=TILE_SIZE, BLOCK_N=TILE_SIZE, THREADS_PER_ROW=32; };
    };
#else
    #include <mma.h>
    using namespace nvcuda;
#endif

#if A_IS_COL_MAJOR
    using FragmentA = wmma::fragment<wmma::matrix_a, TILE_SIZE, TILE_SIZE, TILE_SIZE, half, wmma::col_major>;
#else
    using FragmentA = wmma::fragment<wmma::matrix_a, TILE_SIZE, TILE_SIZE, TILE_SIZE, half, wmma::row_major>;
#endif

// ======================================================================================
// fill_shared_memory:
//   volta  -> WMMA_GEMM_LOAD_TILE<...>  (swizzle store)
//   native -> plain linear copy
// ======================================================================================
__device__ __forceinline__ void
fill_shared_memory(const half* __restrict__ global_a, half* shared_a)
{
#ifdef USE_VOLTA_MMA
    WMMA_GEMM_LOAD_TILE<TestConfig, GemmPhase::FWD, Stage::SNGL>(global_a, shared_a, nullptr, nullptr, TILE_SIZE, TILE_SIZE, threadIdx.x);
#else
    for (int idx = threadIdx.x; idx < TILE_HALVES; idx += 32) shared_a[idx] = global_a[idx];
#endif
}

// ======================================================================================
// kernel: dump_a_regs
// ======================================================================================
__global__ void dump_a_regs(
    const half* __restrict__ global_a,
    uint32_t*   __restrict__ register_dump,
    uint32_t*   __restrict__ shared_dump,
    long long*  __restrict__ cycles_out)
{
    if (threadIdx.x >= 32) return;

    __shared__ half     shared_a[TILE_HALVES];
    __shared__ uint32_t staging[32 * 8];

    fill_shared_memory(global_a, shared_a);
    __syncthreads();

    const uint32_t* shared_as_u32 = reinterpret_cast<const uint32_t*>(shared_a);
    for (int idx = threadIdx.x; idx < TILE_HALVES / 2; idx += 32) shared_dump[idx] = shared_as_u32[idx];
    __syncthreads();

    FragmentA frag;
    wmma::load_matrix_sync(frag, shared_a, TILE_SIZE);

    const uint32_t* fragment_regs = reinterpret_cast<const uint32_t*>(frag.x);
    uint32_t* lane_dst = staging + threadIdx.x * 8;
    #pragma unroll
    for (int reg = 0; reg < 8; reg++) lane_dst[reg] = fragment_regs[reg];
    __syncthreads();

    #pragma unroll
    for (int reg = 0; reg < 8; reg++) register_dump[threadIdx.x * 8 + reg] = staging[threadIdx.x * 8 + reg];
    __syncthreads();

    long long cycles_start = clock64();
    unsigned  checksum     = 0;
    #pragma unroll 1
    for (int iteration = 0; iteration < BENCH_ITER; ++iteration) {
        fill_shared_memory(global_a, shared_a);
        __syncwarp();
        FragmentA bench_frag;
        wmma::load_matrix_sync(bench_frag, shared_a, TILE_SIZE);
        checksum += (unsigned)(float)bench_frag.x[0] + (unsigned)(float)bench_frag.x[7];
    }
    long long cycles_end = clock64();

    if (threadIdx.x == 0) {
        cycles_out[0] = cycles_end - cycles_start;
        cycles_out[1] = (long long)checksum;
    }
}

// ======================================================================================
// host
// ======================================================================================
static void build_global_a(half* global_a)
{
#ifdef USE_VOLTA_MMA
    #if A_IS_COL_MAJOR
        // col_major: transposed values, SNGL yields col-major smem
        for (int n = 0; n < 16; ++n)
            for (int d = 0; d < 16; ++d)
                global_a[n*16 + d] = __float2half((float)(d*16 + n));
    #else
        // row_major: linear values, SNGL yields row-major smem
        for (int r = 0; r < 16; ++r)
            for (int c = 0; c < 16; ++c)
                global_a[r*16 + c] = __float2half((float)(r*16 + c));
    #endif
#else
    for (int r = 0; r < 16; ++r)
        for (int c = 0; c < 16; ++c)
#if A_IS_COL_MAJOR
            global_a[r + c*16] = __float2half((float)(r*16 + c));
#else
            global_a[r*16 + c] = __float2half((float)(r*16 + c));
#endif
#endif
}

static void print_expected(void)
{
    printf("\nExpected logical A[row][col] = row*16 + col:\n");
    for (int row = 0; row < TILE_SIZE; ++row) {
        printf("  row %2d: ", row);
        for (int col = 0; col < TILE_SIZE; ++col) {
            printf("%3d ", row * TILE_SIZE + col);
        }
        printf("\n");
    }
}

static void print_shared_layout(const uint32_t* shared_dump)
{
    printf("\nShared-memory layout of smem_A (16 lines x 16 halves):\n");
    const half* shared_halves = reinterpret_cast<const half*>(shared_dump);
    for (int line = 0; line < TILE_SIZE; ++line) {
        printf("  s%2d: ", line);
        for (int item = 0; item < TILE_SIZE; ++item) {
            printf("%4.0f ", __half2float(shared_halves[line * TILE_SIZE + item]));
        }
        printf("\n");
    }
}

static void print_fragment_registers(const uint32_t* register_dump)
{
    printf("\nFragment register dump (32 lanes x 16 halves):\n");
    for (int lane = 0; lane < 32; ++lane) {
        printf("L%2d: ", lane);
        const half* lane_halves = reinterpret_cast<const half*>(&register_dump[lane * 8]);
        for (int item = 0; item < 16; ++item) {
            printf("%4.0f ", __half2float(lane_halves[item]));
        }
        printf("\n");
    }
}

int main(void)
{
#ifdef USE_VOLTA_MMA
    printf("=== VOLTA MMA (volta::wmma) ===  A layout: %s\n",   A_IS_COL_MAJOR ? "col_major" : "row_major");
#else
    printf("=== NVCUDA MMA (nvcuda::wmma) ===  A layout: %s\n", A_IS_COL_MAJOR ? "col_major" : "row_major");
#endif

    half host_a[TILE_HALVES];
    build_global_a(host_a);

    half*      device_a;
    uint32_t*  device_registers;
    uint32_t*  device_shared;
    long long* device_cycles;
    cudaMalloc(&device_a,         TILE_HALVES     * sizeof(half));
    cudaMalloc(&device_registers, 32 * 8          * sizeof(uint32_t));
    cudaMalloc(&device_shared,    TILE_HALVES / 2 * sizeof(uint32_t));
    cudaMalloc(&device_cycles,    2               * sizeof(long long));

    cudaMemcpy(device_a, host_a, TILE_HALVES * sizeof(half), cudaMemcpyHostToDevice);
    cudaMemset(device_cycles, 0, 2 * sizeof(long long));

    dump_a_regs<<<1, 32>>>(device_a, device_registers, device_shared, device_cycles);
    cudaError_t status = cudaDeviceSynchronize();
    if (status != cudaSuccess) {
        printf("CUDA error: %s\n", cudaGetErrorString(status));
        return 1;
    }

    uint32_t  host_registers[32 * 8];
    uint32_t  host_shared[TILE_HALVES / 2];
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
