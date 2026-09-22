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

// ======================================================================================
// Helper: 32-bit load for 1 float element (1 x float)
// ======================================================================================
__device__ __forceinline__ float ld_float(uint32_t addr, int row) {
    addr = swizzle(addr, row);
    float v;
    asm volatile("ld.shared.f32 %0, [%1];" : "=f"(v) : "r"(addr));
    return v;
}

// ======================================================================================
// Helper: 64-bit load for 2 float elements (1 x float2)
// ======================================================================================
__device__ __forceinline__ float2 ld_float2(uint32_t addr, int row) {
    addr = swizzle(addr, row);
    float2 v;
    asm volatile("ld.shared.v2.f32 {%0, %1}, [%2];" : "=f"(v.x), "=f"(v.y) : "r"(addr));
    return v;
}

// ======================================================================================
// Helper: 128-bit load for 4 float elements (1 x float4)
// ======================================================================================
__device__ __forceinline__ float4 ld_float4(uint32_t addr, int row) {
    addr = swizzle(addr, row);
    float4 v;
    asm volatile("ld.shared.v4.f32 {%0, %1, %2, %3}, [%4];"
                 : "=f"(v.x), "=f"(v.y), "=f"(v.z), "=f"(v.w)
                 : "r"(addr));
    return v;
}

// ======================================================================================
// Helper: 32-bit store for 1 float element (1 x float)
// ======================================================================================
__device__ __forceinline__ void st_float(uint32_t addr, float v, int row) {
    addr = swizzle(addr, row);
    asm volatile("st.shared.f32 [%0], %1;" :: "r"(addr), "f"(v) : "memory");
}

// ======================================================================================
// Helper: 64-bit store for 2 float elements (1 x float2)
// ======================================================================================
__device__ __forceinline__ void st_float2(uint32_t addr, float2 v, int row) {
    addr = swizzle(addr, row);
    asm volatile("st.shared.v2.f32 [%0], {%1, %2};"
                 :: "r"(addr), "f"(v.x), "f"(v.y)
                 : "memory");
}

// ======================================================================================
// Helper: 128-bit store for 4 float elements (1 x float4)
// ======================================================================================
__device__ __forceinline__ void st_float4(uint32_t addr, float4 v, int row) {
    addr = swizzle(addr, row);
    asm volatile("st.shared.v4.f32 [%0], {%1, %2, %3, %4};"
                 :: "r"(addr), "f"(v.x), "f"(v.y), "f"(v.z), "f"(v.w)
                 : "memory");
}

// ======================================================================================
// Helper: 16-bit load for 1 half element (1 x __half)
// ======================================================================================
__device__ __forceinline__ __half ld_half(uint32_t addr, int row) {
    addr = swizzle(addr, row);
    unsigned short val;
    asm volatile("ld.shared.u16 %0, [%1];" : "=h"(val) : "r"(addr));
    return __ushort_as_half(val);
}

// ======================================================================================
// Helper: 32-bit load for 2 half elements (1 x __half2)
// ======================================================================================
__device__ __forceinline__ __half2 ld_half2(uint32_t addr, int row) {
    addr = swizzle(addr, row);
    unsigned int val;
    asm volatile("ld.shared.u32 %0, [%1];" : "=r"(val) : "r"(addr));
    return *reinterpret_cast<__half2*>(&val);
}

// ======================================================================================
// Helper: 64-bit load for 4 half elements (2 x __half2)
// ======================================================================================
__device__ __forceinline__ void ld_half4(uint32_t addr, __half2& h0, __half2& h1, int row) {
    addr = swizzle(addr, row);
    asm volatile("ld.shared.v2.u32 {%0, %1}, [%2];"
                 : "=r"(*reinterpret_cast<unsigned int*>(&h0)),
                   "=r"(*reinterpret_cast<unsigned int*>(&h1))
                 : "r"(addr));
}

// ======================================================================================
// Helper: 128-bit load for 8 half elements (4 x __half2)
// ======================================================================================
__device__ __forceinline__ void ld_half8(uint32_t addr, __half2& h0, __half2& h1, __half2& h2, __half2& h3, int row) {
    addr = swizzle(addr, row);
    asm volatile("ld.shared.v4.u32 {%0, %1, %2, %3}, [%4];"
                 : "=r"(*reinterpret_cast<unsigned int*>(&h0)),
                   "=r"(*reinterpret_cast<unsigned int*>(&h1)),
                   "=r"(*reinterpret_cast<unsigned int*>(&h2)),
                   "=r"(*reinterpret_cast<unsigned int*>(&h3))
                 : "r"(addr));
}

// ======================================================================================
// Helper: 16-bit store for 1 half element (1 x __half)
// ======================================================================================
__device__ __forceinline__ void st_half(uint32_t addr, __half v, int row) {
    addr = swizzle(addr, row);
    asm volatile("st.shared.u16 [%0], %1;"
                 :: "r"(addr),
                    "h"(*reinterpret_cast<const unsigned short*>(&v))
                 : "memory");
}

// ======================================================================================
// Helper: 32-bit store for 2 half elements (1 x __half2)
// ======================================================================================
__device__ __forceinline__ void st_half2(uint32_t addr, __half2 h0, int row) {
    addr = swizzle(addr, row);
    asm volatile("st.shared.u32 [%0], %1;"
                 :: "r"(addr),
                    "r"(*reinterpret_cast<const unsigned int*>(&h0))
                 : "memory");
}

// ======================================================================================
// Helper: 64-bit store for 4 half elements (2 x __half2)
// ======================================================================================
__device__ __forceinline__ void st_half4(uint32_t addr, __half2 h0, __half2 h1, int row) {
    addr = swizzle(addr, row);
    asm volatile("st.shared.v2.u32 [%0], {%1, %2};"
                 :: "r"(addr),
                    "r"(*reinterpret_cast<const unsigned int*>(&h0)),
                    "r"(*reinterpret_cast<const unsigned int*>(&h1))
                 : "memory");
}

// ======================================================================================
// Helper: 128-bit store for 8 half elements (4 x __half2)
// ======================================================================================
__device__ __forceinline__ void st_half8(uint32_t addr, __half2 h0, __half2 h1, __half2 h2, __half2 h3, int row) {
    addr = swizzle(addr, row);
    asm volatile("st.shared.v4.u32 [%0], {%1, %2, %3, %4};" 
                 :: "r"(addr), 
                    "r"(reinterpret_cast<const unsigned int&>(h0)), 
                    "r"(reinterpret_cast<const unsigned int&>(h1)), 
                    "r"(reinterpret_cast<const unsigned int&>(h2)), 
                    "r"(reinterpret_cast<const unsigned int&>(h3)) 
                 : "memory");
}