// ======================================================================================
// * Copyright (c) 2026, D.Skryabin / tg @ai_bond007 SPDX-License: BSD-3-Clause
// ======================================================================================
// Accumulator store col_major dump
// ======================================================================================
#include <cuda_fp16.h>
#include <cstdint>
#include <cstdio>

#ifdef USE_VOLTA_MMA
    #include "mma_m16n16k16.h"
    #include "swizzle.h"
    using namespace volta;
#else
    #include <mma.h>
    using namespace nvcuda;
#endif

#define WMMA_M 16
#define WMMA_N 16
#define WMMA_K 16

__global__ void dump_acc_store_col_regs(
    const float* __restrict__ C_in,
    float* __restrict__ C_out,
    uint32_t* __restrict__ reg_dump_before) {

    if (threadIdx.x >= 32) return;

    __shared__ float smem_C[512];

    // ---- load input matrix into smem_C[0..255] (col_major) ----
    for (int i = threadIdx.x; i < 256; i += 32) {
#ifdef USE_VOLTA_MMA
        int row = i / 16;
        unsigned b = __cvta_generic_to_shared(smem_C) + i * 4;
        st_float(b, C_in[i], row);
#else
        smem_C[i] = C_in[i];
#endif
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
        reg_dump_before[threadIdx.x * 8 + i] = smem_dump[threadIdx.x * 8 + i];
    }

    // ---- prefill store target smem_C[256..511] with -1.0f via swizzle ----
    for (int i = threadIdx.x; i < 256; i += 32) {
#ifdef USE_VOLTA_MMA
        int row = (256 + i) / 16;   // = 16 + i/16 -> same swizzle mask as row i/16
        unsigned b = __cvta_generic_to_shared(smem_C) + (256 + i) * 4;
        st_float(b, -1.0f, row);
#else
        smem_C[256 + i] = -1.0f;
#endif
    }
    __syncthreads();

    // ---- store acc_frag into swizzled smem_C[256..511] ----
    wmma::store_matrix_sync(smem_C + 256, acc_frag, 16, wmma::mem_col_major);
    __syncthreads();

    // ---- readout smem_C[256..511] via ld_float (inverse of st_float swizzle) ----
    for (int i = threadIdx.x; i < 256; i += 32) {
#ifdef USE_VOLTA_MMA
        int row = (256 + i) / 16;
        unsigned b = __cvta_generic_to_shared(smem_C) + (256 + i) * 4;
        C_out[i] = ld_float(b, row);
#else
        C_out[i] = smem_C[256 + i];
#endif
    }
}

int main() {
    printf("Accumulator store col_major dump\n");

    // C in COL-MAJOR: C[i][j] = i*16 + j, stored at offset i + j*16
    float h_C_in[256];
    for (int i = 0; i < 16; i++) {
        for (int j = 0; j < 16; j++) {
            h_C_in[i + j * 16] = (float)(i * 16 + j);
        }
    }

    float* d_C_in;
    float* d_C_out;
    uint32_t* d_regs_before;

    cudaMalloc(&d_C_in, 256 * sizeof(float));
    cudaMalloc(&d_C_out, 256 * sizeof(float));
    cudaMalloc(&d_regs_before, 32 * 8 * sizeof(uint32_t));

    cudaMemcpy(d_C_in, h_C_in, 256 * sizeof(float), cudaMemcpyHostToDevice);

    dump_acc_store_col_regs<<<1, 32>>>(d_C_in, d_C_out, d_regs_before);
    cudaDeviceSynchronize();

    uint32_t h_regs[32 * 8];
    cudaMemcpy(h_regs, d_regs_before, 32 * 8 * sizeof(uint32_t), cudaMemcpyDeviceToHost);

    printf("=== Registers BEFORE store (col_major) ===\n");
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

    printf("\n=== Stored matrix C (16x16 col_major) ===\n");
    for (int i = 0; i < 16; i++) {
        printf("Row %2d: ", i);
        for (int j = 0; j < 16; j++) {
            printf("%.0f ", h_C_out[i + j * 16]);
        }
        printf("\n");
    }

    cudaFree(d_C_in);
    cudaFree(d_C_out);
    cudaFree(d_regs_before);
    return 0;
}