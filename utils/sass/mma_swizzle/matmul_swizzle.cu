#include <cuda_fp16.h>
#include <cstdint>
#include <cstdio>
#include "mma_m16n16k16.h"

using namespace volta::wmma;

__global__ void matmul_swizzle(
    float* __restrict__ max_diff_out,
    float* __restrict__ dump_out) 
{
    __shared__ __align__(16) float smem_C[256];
    unsigned sC_base = __cvta_generic_to_shared(smem_C);

    // 1. Prepare accumulator with known logical values: C[r][c] = r*16 + c
    fragment<accumulator, 16, 16, 16, float> acc;
    int lid = threadIdx.x;

    int r0 = (lid & 1) + ((lid >> 2) & 1) * 8 + ((lid >> 4) & 1) * 4;
    int c0 = ((lid >> 1) & 1) * 2 + ((lid >> 3) & 1) * 8;

    acc.x[0] = (float)(r0 * 16 + c0);
    acc.x[1] = (float)(r0 * 16 + c0 + 1);
    acc.x[2] = (float)((r0+2) * 16 + c0);
    acc.x[3] = (float)((r0+2) * 16 + c0 + 1);
    acc.x[4] = (float)(r0 * 16 + c0 + 4);
    acc.x[5] = (float)(r0 * 16 + c0 + 5);
    acc.x[6] = (float)((r0+2) * 16 + c0 + 4);
    acc.x[7] = (float)((r0+2) * 16 + c0 + 5);

    // 2. Store to SMEM_C using store_matrix_sync (linear gather + swizzle)
    store_matrix_sync(smem_C, acc, 16, mem_row_major);
    __syncthreads();

    // 3. Read back using softmax loading pattern (float4 + absolute swizzle)
    float thread_max_diff = 0.0f;
    int row = threadIdx.x % 16;
    if (threadIdx.x < 16) {
        uint32_t row_byte_off = row * 16 * sizeof(float);
        
        #pragma unroll
        for (int idx = 0; idx < 4; ++idx) {
            uint32_t addr = sC_base + row_byte_off + (idx << 4);
            addr ^= ((addr >> 7) & 0x3) << 4; // Absolute XOR swizzle
            
            float4 buffer;
            asm volatile("ld.shared.v4.f32 {%0, %1, %2, %3}, [%4];"
                         : "=f"(buffer.x), "=f"(buffer.y), "=f"(buffer.z), "=f"(buffer.w)
                         : "r"(addr));

            float exp0 = (float)(row * 16 + idx*4 + 0);
            float exp1 = (float)(row * 16 + idx*4 + 1);
            float exp2 = (float)(row * 16 + idx*4 + 2);
            float exp3 = (float)(row * 16 + idx*4 + 3);

            thread_max_diff = fmaxf(thread_max_diff, fabsf(buffer.x - exp0));
            thread_max_diff = fmaxf(thread_max_diff, fabsf(buffer.y - exp1));
            thread_max_diff = fmaxf(thread_max_diff, fabsf(buffer.z - exp2));
            thread_max_diff = fmaxf(thread_max_diff, fabsf(buffer.w - exp3));

            if (row == 0) {
                dump_out[idx*4 + 0] = buffer.x;
                dump_out[idx*4 + 1] = buffer.y;
                dump_out[idx*4 + 2] = buffer.z;
                dump_out[idx*4 + 3] = buffer.w;
            }
        }
    }

    #pragma unroll
    for (int offset = 16; offset > 0; offset >>= 1) {
        thread_max_diff = fmaxf(thread_max_diff, __shfl_xor_sync(0xFFFFFFFFU, thread_max_diff, offset));
    }

    if (threadIdx.x == 0) max_diff_out[0] = thread_max_diff;
}

int main() {
    printf("=== MatMul Store Softmax Load Symmetry ===\n");
    float *d_max_diff, *d_dump;
    cudaMalloc(&d_max_diff, sizeof(float));
    cudaMalloc(&d_dump, 16 * sizeof(float));

    matmul_swizzle<<<1, 32>>>(d_max_diff, d_dump);
    cudaDeviceSynchronize();

    float h_max_diff, h_dump[16];
    cudaMemcpy(&h_max_diff, d_max_diff, sizeof(float), cudaMemcpyDeviceToHost);
    cudaMemcpy(h_dump, d_dump, 16 * sizeof(float), cudaMemcpyDeviceToHost);

    bool pass = (h_max_diff < 1e-4f);
    printf("[Result] max_diff = %.6f %s\n", h_max_diff, pass ? "✅ PASS" : "❌ FAIL");
    printf("\n[Row 0 Readback]\nExpected: ");
    for(int i=0;i<16;++i) printf("%4.0f ", (float)i);
    printf("\nActual  : ");
    for(int i=0;i<16;++i) printf("%4.0f ", h_dump[i]);
    printf("\n=================================================================\n");
    if(pass) printf("✅ MATMUL STORE VERIFIED. Linear layout + swizzle symmetry.\n");
    else     printf("❌ ASYMMETRY REMAINS. Wrong store_matrix_sync lane mapping.\n");

    cudaFree(d_max_diff); cudaFree(d_dump);
    return pass ? 0 : 1;
}