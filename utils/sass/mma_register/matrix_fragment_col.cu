// Accumulator load col_major register dump
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

__global__ void dump_acc_load_col_regs(
    const float* __restrict__ C,
    uint32_t* __restrict__ reg_dump) {

    if (threadIdx.x >= 32) return;

    __shared__ float smem_C[256];
    for (int i = threadIdx.x; i < 256; i += 32) {
        smem_C[i] = C[i];
    }
    __syncthreads();

    wmma::fragment<wmma::accumulator, 16, 16, 16, float> acc_frag;
    wmma::load_matrix_sync(acc_frag, smem_C, 16, wmma::mem_col_major);

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
        reg_dump[threadIdx.x * 8 + i] = smem_dump[threadIdx.x * 8 + i];
    }
}

int main() {
    printf("Accumulator load col_major dump\n");

    // C in COL-MAJOR: C[i][j] = i*16 + j, stored at offset i + j*16
    float h_C[256];
    for (int i = 0; i < 16; i++) {
        for (int j = 0; j < 16; j++) {
            h_C[i + j * 16] = (float)(i * 16 + j);
        }
    }

    float* d_C;
    uint32_t* d_regs;
    cudaMalloc(&d_C, 256 * sizeof(float));
    cudaMalloc(&d_regs, 32 * 8 * sizeof(uint32_t));
    cudaMemcpy(d_C, h_C, 256 * sizeof(float), cudaMemcpyHostToDevice);

    dump_acc_load_col_regs<<<1, 32>>>(d_C, d_regs);
    cudaDeviceSynchronize();

    uint32_t h_regs[32 * 8];
    cudaMemcpy(h_regs, d_regs, 32 * 8 * sizeof(uint32_t), cudaMemcpyDeviceToHost);

    for (int lane = 0; lane < 32; lane++) {
        printf("L%2d: ", lane);
        const float* v = reinterpret_cast<const float*>(&h_regs[lane * 8]);
        for (int h = 0; h < 8; h++) {
            printf("%.0f ", v[h]);
        }
        printf("\n");
    }

    cudaFree(d_C);
    cudaFree(d_regs);
    return 0;
}