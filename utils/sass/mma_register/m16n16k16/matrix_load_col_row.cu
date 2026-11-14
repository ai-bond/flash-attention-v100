// ======================================================================================
// * Copyright (c) 2026, D.Skryabin / tg @ai_bond007 SPDX-License: BSD-3-Clause
// ======================================================================================
#include <cuda_fp16.h>
#include <cstdint>
#include <cstdio>
#include <cuda_runtime.h>

#define TILE_SIZE        16
#define TILE_FLOATS      (TILE_SIZE * TILE_SIZE)
#define BENCH_ITER       10000
#define ACC_IS_COL_MAJOR 0 // 1 -> col_major, 0 -> row_major

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

using FragmentAcc = wmma::fragment<wmma::accumulator, TILE_SIZE, TILE_SIZE, TILE_SIZE, float>;

#if ACC_IS_COL_MAJOR
    #define ACC_LAYOUT_VOLTA  volta::wmma::mem_col_major
    #define ACC_LAYOUT_NATIVE nvcuda::wmma::mem_col_major
#else
    #define ACC_LAYOUT_VOLTA  volta::wmma::mem_row_major
    #define ACC_LAYOUT_NATIVE nvcuda::wmma::mem_row_major
#endif

// ======================================================================================
// fill_shared_acc:
//   volta  -> store with swizzle (row for row_major, col for col_major)
//   native -> plain linear copy
// ======================================================================================
__device__ __forceinline__ void
fill_shared_acc(const float* __restrict__ global_acc, float* shared_acc)
{
#ifdef USE_VOLTA_MMA
    #if ACC_IS_COL_MAJOR
        // smem layout: shared_acc[c*16 + r] holds logical A[r][c]
        // swizzle applied by column index c
        for (int c = threadIdx.x; c < TILE_SIZE; c += 32) {
            for (int r = 0; r < TILE_SIZE; r += 4) {
                uint32_t addr = __cvta_generic_to_shared(shared_acc + c * TILE_SIZE + r);
                float4 v = *reinterpret_cast<const float4*>(&global_acc[c * TILE_SIZE + r]);
                st_float4(addr, v, c);
            }
        }
    #else
        // smem layout: shared_acc[r*16 + c] holds logical A[r][c]
        // swizzle applied by row index r
        for (int r = threadIdx.x; r < TILE_SIZE; r += 32) {
            for (int c = 0; c < TILE_SIZE; c += 4) {
                uint32_t addr = __cvta_generic_to_shared(shared_acc + r * TILE_SIZE + c);
                float4 v = *reinterpret_cast<const float4*>(&global_acc[r * TILE_SIZE + c]);
                st_float4(addr, v, r);
            }
        }
    #endif
#else
    for (int idx = threadIdx.x; idx < TILE_FLOATS; idx += 32) {
        shared_acc[idx] = global_acc[idx];
    }
#endif
}

// ======================================================================================
// kernel: dump_acc_regs
// ======================================================================================
__global__ void dump_acc_regs(
    const float* __restrict__ global_acc,
    uint32_t*    __restrict__ register_dump,
    uint32_t*    __restrict__ shared_dump,
    long long*   __restrict__ cycles_out)
{
    if (threadIdx.x >= 32) return;

    __shared__ float    shared_acc[TILE_FLOATS];
    __shared__ uint32_t staging[32 * 8];

    fill_shared_acc(global_acc, shared_acc);
    __syncthreads();

    const uint32_t* shared_as_u32 = reinterpret_cast<const uint32_t*>(shared_acc);
    for (int idx = threadIdx.x; idx < TILE_FLOATS; idx += 32) {
        shared_dump[idx] = shared_as_u32[idx];
    }
    __syncthreads();

    FragmentAcc frag;
#ifdef USE_VOLTA_MMA
    wmma::load_matrix_sync(frag, shared_acc, TILE_SIZE, ACC_LAYOUT_VOLTA);
#else
    wmma::load_matrix_sync(frag, shared_acc, TILE_SIZE, ACC_LAYOUT_NATIVE);
#endif

    const uint32_t* fragment_regs = reinterpret_cast<const uint32_t*>(frag.x);
    uint32_t* lane_dst = staging + threadIdx.x * 8;
    #pragma unroll
    for (int reg = 0; reg < 8; reg++) {
        lane_dst[reg] = fragment_regs[reg];
    }
    __syncthreads();

    #pragma unroll
    for (int reg = 0; reg < 8; reg++) {
        register_dump[threadIdx.x * 8 + reg] = staging[threadIdx.x * 8 + reg];
    }
    __syncthreads();

    long long cycles_start = clock64();
    unsigned  checksum     = 0;
    #pragma unroll 1
    for (int iteration = 0; iteration < BENCH_ITER; ++iteration) {
        fill_shared_acc(global_acc, shared_acc);
        __syncwarp();
        FragmentAcc bench_frag;
#ifdef USE_VOLTA_MMA
        wmma::load_matrix_sync(bench_frag, shared_acc, TILE_SIZE, ACC_LAYOUT_VOLTA);
#else
        wmma::load_matrix_sync(bench_frag, shared_acc, TILE_SIZE, ACC_LAYOUT_NATIVE);
#endif
        checksum += __float_as_uint(bench_frag.x[0])
                  + __float_as_uint(bench_frag.x[7]);
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
static void build_global_acc(float* global_acc)
{
    // Logical A[r][c] = r*16 + c
#ifdef USE_VOLTA_MMA
    #if ACC_IS_COL_MAJOR
        for (int c = 0; c < TILE_SIZE; ++c)
            for (int r = 0; r < TILE_SIZE; ++r)
                global_acc[c * TILE_SIZE + r] = (float)(r * TILE_SIZE + c);
    #else
        for (int r = 0; r < TILE_SIZE; ++r)
            for (int c = 0; c < TILE_SIZE; ++c)
                global_acc[r * TILE_SIZE + c] = (float)(r * TILE_SIZE + c);
    #endif
#else
    for (int r = 0; r < TILE_SIZE; ++r)
        for (int c = 0; c < TILE_SIZE; ++c)
#if ACC_IS_COL_MAJOR
            global_acc[c * TILE_SIZE + r] = (float)(r * TILE_SIZE + c);
#else
            global_acc[r * TILE_SIZE + c] = (float)(r * TILE_SIZE + c);
#endif
#endif
}

static void print_expected(void)
{
    printf("\nExpected logical Acc[row][col] = row*16 + col:\n");
    for (int row = 0; row < TILE_SIZE; ++row) {
        printf("  row %2d: ", row);
        for (int col = 0; col < TILE_SIZE; ++col) printf("%3d ", row * TILE_SIZE + col);
        printf("\n");
    }
}

static void print_shared_layout(const uint32_t* shared_dump)
{
    printf("\nShared-memory layout of smem_Acc (16 lines x 16 floats):\n");
    const float* sh = reinterpret_cast<const float*>(shared_dump);
    for (int line = 0; line < TILE_SIZE; ++line) {
        printf("  s%2d: ", line);
        for (int item = 0; item < TILE_SIZE; ++item) printf("%5.0f ", sh[line * TILE_SIZE + item]);
        printf("\n");
    }
}

static void print_fragment_registers(const uint32_t* register_dump)
{
    printf("\nFragment register dump (32 lanes x 8 floats):\n");
    for (int lane = 0; lane < 32; ++lane) {
        printf("L%2d: ", lane);
        const float* lane_floats = reinterpret_cast<const float*>(&register_dump[lane * 8]);
        for (int item = 0; item < 8; ++item) printf("%5.0f ", lane_floats[item]);
        printf("\n");
    }
}

int main(void)
{
#ifdef USE_VOLTA_MMA
    printf("=== VOLTA MMA (volta::wmma) ===  Acc layout: %s\n",   ACC_IS_COL_MAJOR ? "col_major" : "row_major");
#else
    printf("=== NVCUDA MMA (nvcuda::wmma) ===  Acc layout: %s\n", ACC_IS_COL_MAJOR ? "col_major" : "row_major");
#endif

    float host_acc[TILE_FLOATS];
    build_global_acc(host_acc);

    float*     device_acc;
    uint32_t*  device_registers;
    uint32_t*  device_shared;
    long long* device_cycles;
    cudaMalloc(&device_acc,       TILE_FLOATS    * sizeof(float));
    cudaMalloc(&device_registers, 32 * 8         * sizeof(uint32_t));
    cudaMalloc(&device_shared,    TILE_FLOATS    * sizeof(uint32_t));
    cudaMalloc(&device_cycles,    2              * sizeof(long long));

    cudaMemcpy(device_acc, host_acc, TILE_FLOATS * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemset(device_cycles, 0, 2 * sizeof(long long));

    dump_acc_regs<<<1, 32>>>(device_acc, device_registers, device_shared, device_cycles);
    cudaError_t status = cudaDeviceSynchronize();
    if (status != cudaSuccess) { printf("CUDA error: %s\n", cudaGetErrorString(status)); return 1; }

    uint32_t  host_registers[32 * 8];
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