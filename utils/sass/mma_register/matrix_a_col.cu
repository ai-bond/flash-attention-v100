// Matrix A col_major register dump
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

__global__ void dump_a_col_regs(
    const half* __restrict__ A,
    uint32_t* __restrict__ reg_dump) {

    if (threadIdx.x >= 32) return;

    __shared__ half smem_A[256];
    for (int i = threadIdx.x; i < 256; i += 32) {
        smem_A[i] = A[i];
    }
    __syncthreads();

    wmma::fragment<wmma::matrix_a, 16, 16, 16, half, wmma::col_major> a_frag;
    wmma::load_matrix_sync(a_frag, smem_A, 16);

    __shared__ uint32_t smem_dump[32 * 8];
    uint32_t* dst = smem_dump + threadIdx.x * 8;
    const uint32_t* src = reinterpret_cast<const uint32_t*>(a_frag.x);
    #pragma unroll
    for (int i = 0; i < 8; i++) dst[i] = src[i];
    __syncthreads();

    #pragma unroll
    for (int i = 0; i < 8; i++) {
        reg_dump[threadIdx.x * 8 + i] = smem_dump[threadIdx.x * 8 + i];
    }
}

int main() {
    printf("Matrix A col_major dump\n");

    // A in COL-MAJOR: A[i][j] = i*16 + j, stored at offset i + j*16
    half h_A[256];
    for (int i = 0; i < 16; i++) {
        for (int j = 0; j < 16; j++) {
            h_A[i + j * 16] = __float2half((float)(i * 16 + j));
        }
    }

    half* d_A;
    uint32_t* d_regs;
    cudaMalloc(&d_A, 256 * sizeof(half));
    cudaMalloc(&d_regs, 32 * 8 * sizeof(uint32_t));
    cudaMemcpy(d_A, h_A, 256 * sizeof(half), cudaMemcpyHostToDevice);

    dump_a_col_regs<<<1, 32>>>(d_A, d_regs);
    cudaDeviceSynchronize();

    uint32_t h_regs[32 * 8];
    cudaMemcpy(h_regs, d_regs, 32 * 8 * sizeof(uint32_t), cudaMemcpyDeviceToHost);

    for (int lane = 0; lane < 32; lane++) {
        printf("L%2d: ", lane);
        const half* v = reinterpret_cast<const half*>(&h_regs[lane * 8]);
        for (int h = 0; h < 16; h++) printf("%.0f ", __half2float(v[h]));
        printf("\n");
    }

    cudaFree(d_A);
    cudaFree(d_regs);
    return 0;
}