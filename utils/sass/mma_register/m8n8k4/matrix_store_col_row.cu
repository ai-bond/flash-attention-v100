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
#define BENCH_ITER       1000
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
fill_fragment_pattern(FragmentAcc& frag, int lane)
{
    const int r0 = (lane & 1) + ((lane >> 4) & 1) * 4;
    const int c0 = lane & 2;
    frag.x[0] = (float)( r0      * TILE_N + c0    );
    frag.x[1] = (float)( r0      * TILE_N + c0 + 1);
    frag.x[2] = (float)((r0 + 2) * TILE_N + c0    );
    frag.x[3] = (float)((r0 + 2) * TILE_N + c0 + 1);
    frag.x[4] = (float)( r0      * TILE_N + c0 + 4);
    frag.x[5] = (float)( r0      * TILE_N + c0 + 5);
    frag.x[6] = (float)((r0 + 2) * TILE_N + c0 + 4);
    frag.x[7] = (float)((r0 + 2) * TILE_N + c0 + 5);
}

__global__ void store_roundtrip_kernel(
    uint32_t*  __restrict__ reg_dump_orig,
    uint32_t*  __restrict__ reg_dump_after,
    uint32_t*  __restrict__ shared_dump,
    long long* __restrict__ cycles_out)
{
    if (threadIdx.x >= 32) return;

    __shared__ float    shared_acc[TILE_FLOATS];
    __shared__ uint32_t staging[32 * FRAG_FLOATS];

    FragmentAcc frag;
    fill_fragment_pattern(frag, threadIdx.x);

    const uint32_t* src = reinterpret_cast<const uint32_t*>(frag.x);
    uint32_t* dst = staging + threadIdx.x * FRAG_FLOATS;
    #pragma unroll
    for (int i = 0; i < FRAG_FLOATS; ++i) dst[i] = src[i];
    __syncthreads();
    #pragma unroll
    for (int i = 0; i < FRAG_FLOATS; ++i)
        reg_dump_orig[threadIdx.x * FRAG_FLOATS + i] = staging[threadIdx.x * FRAG_FLOATS + i];
    __syncthreads();

    store_matrix_sync(shared_acc, frag, TILE_M, ACC_LAYOUT);
    __syncthreads();

    const uint32_t* sh = reinterpret_cast<const uint32_t*>(shared_acc);
    for (int idx = threadIdx.x; idx < TILE_FLOATS; idx += 32) {
        shared_dump[idx] = sh[idx];
    }
    __syncthreads();

    FragmentAcc frag_back;
    load_matrix_sync(frag_back, shared_acc, TILE_M, ACC_LAYOUT);

    const uint32_t* src_back = reinterpret_cast<const uint32_t*>(frag_back.x);
    uint32_t* dst_back = staging + threadIdx.x * FRAG_FLOATS;
    #pragma unroll
    for (int i = 0; i < FRAG_FLOATS; ++i) dst_back[i] = src_back[i];
    __syncthreads();
    #pragma unroll
    for (int i = 0; i < FRAG_FLOATS; ++i)
        reg_dump_after[threadIdx.x * FRAG_FLOATS + i] = staging[threadIdx.x * FRAG_FLOATS + i];
    __syncthreads();

    long long cycles_start = clock64();
    unsigned  checksum     = 0;
    #pragma unroll 1
    for (int iteration = 0; iteration < BENCH_ITER; ++iteration) {
        FragmentAcc f;
        fill_fragment_pattern(f, threadIdx.x);
        store_matrix_sync(shared_acc, f, TILE_M, ACC_LAYOUT);
        __syncwarp();
        FragmentAcc fb;
        load_matrix_sync(fb, shared_acc, TILE_M, ACC_LAYOUT);
        checksum += __float_as_uint(fb.x[0]) + __float_as_uint(fb.x[7]);
    }
    long long cycles_end = clock64();

    if (threadIdx.x == 0) {
        cycles_out[0] = cycles_end - cycles_start;
        cycles_out[1] = (long long)checksum;
    }
}

static void print_shared_layout(const uint32_t* shared_dump)
{
    printf("\nShared-memory layout of smem_Acc (PHYSICAL, %d lines x %d floats):\n", TILE_M, TILE_N);
    const float* sh = reinterpret_cast<const float*>(shared_dump);
    for (int line = 0; line < TILE_M; ++line) {
        printf("  s%2d: ", line);
        for (int item = 0; item < TILE_N; ++item) printf("%7.1f ", sh[line * TILE_N + item]);
        printf("\n");
    }
}

static void print_fragment(const char* title, const uint32_t* register_dump)
{
    printf("\n%s\n", title);
    for (int lane = 0; lane < 32; ++lane) {
        printf("L%2d: ", lane);
        const float* lane_floats = reinterpret_cast<const float*>(&register_dump[lane * FRAG_FLOATS]);
        for (int item = 0; item < FRAG_FLOATS; ++item) printf("%7.1f ", lane_floats[item]);
        printf("\n");
    }
}

static bool fragments_equal(const uint32_t* a, const uint32_t* b, int n = 32 * FRAG_FLOATS)
{
    for (int i = 0; i < n; ++i) if (a[i] != b[i]) return false;
    return true;
}

int main(void)
{
#ifdef USE_VOLTA_MMA
    printf("=== (mma_m8n8k4.h + swizzle.h)  ===  Acc layout: %s (STORE round-trip)\n",  ACC_IS_COL_MAJOR ? "col_major" : "row_major");
#else
    printf("=== (mma_native.h, linear smem) ===  Acc layout: %s (STORE round-trip)\n", ACC_IS_COL_MAJOR ? "col_major" : "row_major");
#endif

    uint32_t*  device_reg_orig;
    uint32_t*  device_reg_after;
    uint32_t*  device_shared;
    long long* device_cycles;
    cudaMalloc(&device_reg_orig,  32 * FRAG_FLOATS * sizeof(uint32_t));
    cudaMalloc(&device_reg_after, 32 * FRAG_FLOATS * sizeof(uint32_t));
    cudaMalloc(&device_shared,    TILE_FLOATS     * sizeof(uint32_t));
    cudaMalloc(&device_cycles,    2               * sizeof(long long));

    cudaMemset(device_cycles, 0, 2 * sizeof(long long));

    store_roundtrip_kernel<<<1, 32>>>(device_reg_orig, device_reg_after,
                                      device_shared, device_cycles);
    cudaError_t status = cudaDeviceSynchronize();
    if (status != cudaSuccess) { printf("CUDA error: %s\n", cudaGetErrorString(status)); return 1; }

    uint32_t  host_reg_orig[32 * FRAG_FLOATS];
    uint32_t  host_reg_after[32 * FRAG_FLOATS];
    uint32_t  host_shared[TILE_FLOATS];
    long long host_cycles[2];
    cudaMemcpy(host_reg_orig,  device_reg_orig,  sizeof(host_reg_orig),  cudaMemcpyDeviceToHost);
    cudaMemcpy(host_reg_after, device_reg_after, sizeof(host_reg_after), cudaMemcpyDeviceToHost);
    cudaMemcpy(host_shared,    device_shared,    sizeof(host_shared),    cudaMemcpyDeviceToHost);
    cudaMemcpy(host_cycles,    device_cycles,    sizeof(host_cycles),    cudaMemcpyDeviceToHost);

    print_shared_layout(host_shared);
    print_fragment("Fragment BEFORE store (original):", host_reg_orig);
    print_fragment("Fragment AFTER load-back (round-trip):", host_reg_after);

    printf("\n");
    if (fragments_equal(host_reg_orig, host_reg_after)) {
        printf("ROUND-TRIP OK: fragment survives store+load intact\n");
    } else {
        printf("ROUND-TRIP FAILED: store+load mismatch\n");
    }

    printf("\nBench: iterations=%d  cycles=%lld  per-iter=%.2f cyc  checksum=0x%llx\n",
           BENCH_ITER, host_cycles[0],
           (double)host_cycles[0] / (double)BENCH_ITER,
           (unsigned long long)host_cycles[1]);

    cudaFree(device_reg_orig);
    cudaFree(device_reg_after);
    cudaFree(device_shared);
    cudaFree(device_cycles);
    return 0;
}
