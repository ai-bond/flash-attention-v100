// Accumulator store row_major dump
#include <cuda_fp16.h>
#include <cstdint>
#include <cstdio>

#ifdef USE_VOLTA_MMA
    #include "fused_mma_m16n16k16.h"
    using namespace volta;
#else
    #include <mma.h>
    using namespace nvcuda;
#endif

#define WMMA_M 16
#define WMMA_N 16
#define WMMA_K 16

__global__ void dump_acc_store_regs(
    const float* __restrict__ C_in,
    float* __restrict__ C_out,
    uint32_t* __restrict__ reg_dump_before,
    uint32_t* __restrict__ reg_dump_after) {

    if (threadIdx.x >= 32) return;

    __shared__ float smem_C[256];

    for (int i = threadIdx.x; i < 256; i += 32) {
        smem_C[i] = C_in[i];
    }
    __syncthreads();

    wmma::fragment<wmma::accumulator, 16, 16, 16, float> acc_frag;
    wmma::load_matrix_sync(acc_frag, smem_C, 16, wmma::mem_row_major);

    __shared__ uint32_t smem_dump[32 * 8];
    uint32_t* dst = smem_dump + threadIdx.x * 8;

    const uint32_t* src = reinterpret_cast<const uint32_t*>(acc_frag.x);
    #pragma unroll
    for (int i = 0; i < 8; i++) {
        dst[i] = src[i];
    }
    __syncthreads();

    #pragma unroll
    for (int i = 0; i < 8; i++) {
        reg_dump_before[threadIdx.x * 8 + i] = smem_dump[threadIdx.x * 8 + i];
    }

    wmma::store_matrix_sync(smem_C + 0, acc_frag, 16, wmma::mem_row_major);
    __syncthreads();

    for (int i = threadIdx.x; i < 256; i += 32) {
        C_out[i] = smem_C[i];
    }
}

int main() {
    printf("Accumulator store row_major dump\n");

    // C in ROW-MAJOR: C[i][j] = i*16 + j (as float)
    float h_C_in[256];
    for (int i = 0; i < 16; i++) {
        for (int j = 0; j < 16; j++) {
            h_C_in[i * 16 + j] = (float)(i * 16 + j);
        }
    }

    float* d_C_in;
    float* d_C_out;
    uint32_t* d_regs_before;
    uint32_t* d_regs_after;

    cudaMalloc(&d_C_in, 256 * sizeof(float));
    cudaMalloc(&d_C_out, 256 * sizeof(float));
    cudaMalloc(&d_regs_before, 32 * 8 * sizeof(uint32_t));
    cudaMalloc(&d_regs_after, 32 * 8 * sizeof(uint32_t));

    cudaMemcpy(d_C_in, h_C_in, 256 * sizeof(float), cudaMemcpyHostToDevice);

    dump_acc_store_regs<<<1, 32>>>(d_C_in, d_C_out, d_regs_before, d_regs_after);
    cudaDeviceSynchronize();

    uint32_t h_regs[32 * 8];
    cudaMemcpy(h_regs, d_regs_before, 32 * 8 * sizeof(uint32_t), cudaMemcpyDeviceToHost);

    printf("=== Registers BEFORE store (row_major) ===\n");
    for (int lane = 0; lane < 32; lane++) {
        printf("L%2d: ", lane);
        const float* v = reinterpret_cast<const float*>(&h_regs[lane * 8]);
        for (int h = 0; h < 8; h++) {
            printf("%.0f ", v[h]);
        }
        printf("\n");
    }

    float h_C_out[256];
    cudaMemcpy(h_C_out, d_C_out, 256 * sizeof(float), cudaMemcpyDeviceToHost);

    printf("\n=== Stored matrix C (16x16 row_major) ===\n");
    for (int i = 0; i < 16; i++) {
        printf("Row %2d: ", i);
        for (int j = 0; j < 16; j++) {
            printf("%.0f ", h_C_out[i * 16 + j]);
        }
        printf("\n");
    }

    cudaFree(d_C_in);
    cudaFree(d_C_out);
    cudaFree(d_regs_before);
    cudaFree(d_regs_after);
    return 0;
}