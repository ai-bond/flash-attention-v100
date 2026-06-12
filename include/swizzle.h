// ======================================================================================
// * Copyright (c) 2025, D.Skryabin / tg @ai_bond007 SPDX-License: BSD-3-Clause
// ======================================================================================
#pragma once

#include <cuda.h>
#include <cuda_runtime.h>
#include <cuda_fp16.h>

// ======================================================================================
// SWIZZLE HELPERS
// ======================================================================================
__device__ __forceinline__ uint32_t swizzle(uint32_t addr, int row) {
    return addr ^ ((row & 3) | ((row >> 1) & 4)) << 4;
}

__device__ __forceinline__ float4 ld_float4(uint32_t addr, int row) {
    addr = swizzle(addr, row);
    float4 v;
    asm volatile("ld.shared.v4.f32 {%0, %1, %2, %3}, [%4];" : "=f"(v.x), "=f"(v.y), "=f"(v.z), "=f"(v.w) : "r"(addr));
    return v;
}

__device__ __forceinline__ void st_float4(uint32_t addr, float4 v, int row) {
    addr = swizzle(addr, row);
    asm volatile("st.shared.v4.f32 [%0], {%1, %2, %3, %4};" :: "r"(addr), "f"(v.x), "f"(v.y), "f"(v.z), "f"(v.w) : "memory");
}

__device__ __forceinline__ float ld_float(uint32_t addr, int row) {
    addr = swizzle(addr, row);
    float v;
    asm volatile("ld.shared.f32 %0, [%1];" : "=f"(v) : "r"(addr));
    return v;
}

__device__ __forceinline__ void st_half2(uint32_t addr, __half2 v, int row) {
    addr = swizzle(addr, row);
    unsigned int val = *reinterpret_cast<unsigned int*>(&v);
    asm volatile("st.shared.u32 [%0], %1;" :: "r"(addr), "r"(val) : "memory");
}

__device__ __forceinline__ void st_half(uint32_t addr, __half v, int row) {
    addr = swizzle(addr, row);
    unsigned short val = __half_as_ushort(v);
    asm volatile("st.shared.u16 [%0], %1;" :: "r"(addr), "h"(val) : "memory");
}
