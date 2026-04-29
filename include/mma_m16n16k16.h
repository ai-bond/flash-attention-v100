// ======================================================================================
// * Copyright (c) 2026, D.Skryabin / tg @ai_bond007 SPDX-License: BSD-3-Clause
// ======================================================================================
// * Volta WMMA (sm_70) production wrapper for FlashAttention-2.
// * Replaces nvcuda::wmma with explicit PTX for full control over
// * register packing, lane mapping, and LSU vectorization.
// *
// * 1. SUPPORTED SHAPE: m16n16k16 (HMMA.884.F32.F32)
// * 2. FRAGMENT LAYOUT: A/B = 8x uint32_t (16x half), C/D = 8x float
// * 3. LOAD/STORE:      Explicit lane-to-memory mapping. Vectorized where contiguous.
// * 4. DATA PACKING:    f16x2 passed as uint32_t to prevent compiler repacking.
// * 5. ALIGNMENT:       Base pointers require 16B alignment. ldm in elements.
// * 6. THREAD MAPPING:  Hardware-fixed distribution. Do not shuffle fragments.
// ======================================================================================
#pragma once

#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ != 700)
#error "Volta WMMA: This header is for sm_70 ONLY! Compile with -arch=sm_70"
#endif

#include <cuda_fp16.h>
#include <cstdint>

namespace volta {
namespace wmma {

// ======================================================================================
// TYPE TAGS
// ======================================================================================
struct row_major {};
struct col_major {};
struct matrix_a {};
struct matrix_b {};
struct accumulator {};

enum layout_t {
    mem_row_major,
    mem_col_major
};

// ======================================================================================
// FRAGMENT DECLARATIONS
// ======================================================================================
template <typename Use, int M, int N, int K, typename T, typename Layout = void>
struct fragment;

template <> struct fragment<matrix_a, 16, 16, 16, half, row_major> { uint32_t x[8]; static constexpr int num_elements = 16; };
template <> struct fragment<matrix_a, 16, 16, 16, half, col_major> { uint32_t x[8]; static constexpr int num_elements = 16; };
template <> struct fragment<matrix_b, 16, 16, 16, half, row_major> { uint32_t x[8]; static constexpr int num_elements = 16; };
template <> struct fragment<matrix_b, 16, 16, 16, half, col_major> { uint32_t x[8]; static constexpr int num_elements = 16; };
template <> struct fragment<accumulator, 16, 16, 16, float>        { float x[8];    static constexpr int num_elements = 8;  };

// ======================================================================================
// FILL: accumulator m16n16k16
// ======================================================================================
// Data per lane:   8 floats (full accumulator fragment)
// Lane mapping:    Uniform broadcast across all 32 lanes
// Replication:     N/A (scalar initialization)
// Memory access:   N/A (register-only operation)
// ======================================================================================
template <int M, int N, int K>
__device__ __forceinline__ void fill_fragment(fragment<accumulator, M, N, K, float>& frag, float value) {
    asm volatile(
        "{\n\t"
        "mov.f32 %0, %8;\n\t"
        "mov.f32 %1, %8;\n\t"
        "mov.f32 %2, %8;\n\t"
        "mov.f32 %3, %8;\n\t"
        "mov.f32 %4, %8;\n\t"
        "mov.f32 %5, %8;\n\t"
        "mov.f32 %6, %8;\n\t"
        "mov.f32 %7, %8;\n\t"
        "}"
        : "=f"(frag.x[0]), "=f"(frag.x[1]), "=f"(frag.x[2]), "=f"(frag.x[3]),
          "=f"(frag.x[4]), "=f"(frag.x[5]), "=f"(frag.x[6]), "=f"(frag.x[7])
        : "f"(value)
    );
}

// ======================================================================================
// LOAD: matrix_a m16n16k16 (ROW MAJOR)
// ======================================================================================
// Data per lane:   1 complete row (16 contiguous halves = 8 uint32_t)
// Lane mapping:    r_base = (lid & 3) + ((lid >> 4) & 1) * 4 + ((lid >> 2) & 1) * 8
// Replication:     Bit 3 ignored L0-7 & L16-23 duplicate rows 0-7.
//                  L8-15 & L24-31 duplicate rows 8-15. (Hardware 2x crossbar routing)
// Memory access:   Contiguous row storage 2x ld.shared.v4.u32 (128-bit loads).
// ======================================================================================
__device__ __forceinline__ void load_matrix_sync(
    fragment<matrix_a, 16, 16, 16, half, row_major>& frag,
    const half* __restrict__ smem_ptr, unsigned ldm) {

    unsigned smem_addr = __cvta_generic_to_shared(smem_ptr);

    asm volatile(
        "{\n\t"
        ".reg .u32 lid, r_base, base, t;\n\t"
        "mov.u32 lid, %%laneid;\n\t"

        // row = (lid & 3) + ((lid >> 4) & 1) * 4 + ((lid >> 2) & 1) * 8
        "and.b32 r_base, lid, 3;\n\t"
        "shr.b32 t, lid, 4; and.b32 t, t, 1; mad.lo.u32 r_base, t, 4, r_base;\n\t"
        "shr.b32 t, lid, 2; and.b32 t, t, 1; mad.lo.u32 r_base, t, 8, r_base;\n\t"

        // base = smem_addr + row * ldm * sizeof(half)
        "mul.lo.u32 base, r_base, %9;\n\t"
        "shl.b32 base, base, 1;\n\t"
        "add.u32 base, base, %8;\n\t"

        "ld.shared.v4.u32 {%0, %1, %2, %3}, [base];\n\t"
        "ld.shared.v4.u32 {%4, %5, %6, %7}, [base+16];\n\t"
        "}"
        : "=r"(frag.x[0]), "=r"(frag.x[1]), "=r"(frag.x[2]), "=r"(frag.x[3]),
          "=r"(frag.x[4]), "=r"(frag.x[5]), "=r"(frag.x[6]), "=r"(frag.x[7])
        : "r"(smem_addr), "r"(ldm)
        : "memory"
    );
}

// ======================================================================================
// LOAD: matrix_a m16n16k16 (COL MAJOR)
// ======================================================================================
// Data per lane:   4 vertical chunks of 4 halves (16 halves total)
// Lane mapping:    c_base = lid & 3 r_base = ((lid >> 2) & 1) * 8 + ((lid >> 4) & 1) * 4
// Replication:     Bit 3 ignored L0==L8, L4==L12, etc. (2x crossbar replication)
// Memory access:   Strided columns base + k*(4*ldm). Vectorized via ld.shared.v2.u32.
// ======================================================================================
__device__ __forceinline__ void load_matrix_sync(
    fragment<matrix_a, 16, 16, 16, half, col_major>& frag,
    const half* __restrict__ smem_ptr, unsigned ldm) {

    unsigned smem_addr = __cvta_generic_to_shared(smem_ptr);

    asm volatile(
        "{\n\t"
        ".reg .u32 lid, r_base, c_base, base, stride, a0, a1, a2, a3, t;\n\t"
        "mov.u32 lid, %%laneid;\n\t"

        // c_base = lid & 3
        "and.b32 c_base, lid, 3;\n\t"
        // r_base = ((lid >> 2) & 1) * 8 + ((lid >> 4) & 1) * 4
        "shr.b32 t, lid, 2; and.b32 t, t, 1; shl.b32 r_base, t, 3;\n\t"
        "shr.b32 t, lid, 4; and.b32 t, t, 1; mad.lo.u32 r_base, t, 4, r_base;\n\t"

        // base = smem_addr + (r_base + c_base * ldm) * sizeof(half)
        "mad.lo.u32 base, c_base, %9, r_base;\n\t"
        "shl.b32 base, base, 1;\n\t"
        "add.u32 base, base, %8;\n\t"

        // stride = 4 columns * ldm * 2 bytes = 8 * ldm
        "shl.b32 stride, %9, 3;\n\t"
        "mov.u32 a0, base;\n\t"
        "add.u32 a1, base, stride;\n\t"
        "add.u32 a2, a1, stride;\n\t"
        "add.u32 a3, a2, stride;\n\t"

        "ld.shared.v2.u32 {%0, %1}, [a0];\n\t"
        "ld.shared.v2.u32 {%2, %3}, [a1];\n\t"
        "ld.shared.v2.u32 {%4, %5}, [a2];\n\t"
        "ld.shared.v2.u32 {%6, %7}, [a3];\n\t"
        "}"
        : "=r"(frag.x[0]), "=r"(frag.x[1]), "=r"(frag.x[2]), "=r"(frag.x[3]),
          "=r"(frag.x[4]), "=r"(frag.x[5]), "=r"(frag.x[6]), "=r"(frag.x[7])
        : "r"(smem_addr), "r"(ldm)
        : "memory"
    );
}

// ======================================================================================
// LOAD: matrix_b m16n16k16 (ROW MAJOR)
// ======================================================================================
// Data per lane:   4 contiguous chunks of 4 halves (16 halves total)
// Lane mapping:    r_base = lid & 3 c_base = ((lid >> 3) & 1) * 8 + ((lid >> 4) & 1) * 4 (bits 3 & 4 swapped)
// Replication:     Bit 2 ignored L0==L4, L8==L12, etc. (2x crossbar replication)
// Memory access:   Strided rows base + k*(4*ldm). Vectorized via ld.shared.v2.u32.
// ======================================================================================
__device__ __forceinline__ void load_matrix_sync(
    fragment<matrix_b, 16, 16, 16, half, row_major>& frag,
    const half* __restrict__ smem_ptr, unsigned ldm) {

    unsigned smem_addr = __cvta_generic_to_shared(smem_ptr);

    asm volatile(
        "{\n\t"
        ".reg .u32 lid, r_base, c_base, base, stride, a0, a1, a2, a3, t1, t2;\n\t"
        "mov.u32 lid, %%laneid;\n\t"

        // r_base = lid & 3
        "and.b32 r_base, lid, 3;\n\t"
        // c_base = ((lid >> 3) & 1) * 8 + ((lid >> 4) & 1) * 4
        "shr.b32 t1, lid, 3; and.b32 t1, t1, 1; shl.b32 t1, t1, 3;\n\t"
        "shr.b32 t2, lid, 4; and.b32 t2, t2, 1; shl.b32 t2, t2, 2;\n\t"
        "add.u32 c_base, t1, t2;\n\t"

        // base = smem_addr + (r_base * ldm + c_base) * sizeof(half)
        "mad.lo.u32 base, r_base, %9, c_base;\n\t"
        "shl.b32 base, base, 1;\n\t"
        "add.u32 base, base, %8;\n\t"

        // stride = 4 rows * ldm * 2 bytes = 8 * ldm
        "shl.b32 stride, %9, 3;\n\t"
        "mov.u32 a0, base;\n\t"
        "add.u32 a1, base, stride;\n\t"
        "add.u32 a2, a1, stride;\n\t"
        "add.u32 a3, a2, stride;\n\t"

        "ld.shared.v2.u32 {%0, %1}, [a0];\n\t"
        "ld.shared.v2.u32 {%2, %3}, [a1];\n\t"
        "ld.shared.v2.u32 {%4, %5}, [a2];\n\t"
        "ld.shared.v2.u32 {%6, %7}, [a3];\n\t"
        "}"
        : "=r"(frag.x[0]), "=r"(frag.x[1]), "=r"(frag.x[2]), "=r"(frag.x[3]),
          "=r"(frag.x[4]), "=r"(frag.x[5]), "=r"(frag.x[6]), "=r"(frag.x[7])
        : "r"(smem_addr), "r"(ldm)
        : "memory"
    );
}

// ======================================================================================
// LOAD: matrix_b m16n16k16 (COL MAJOR)
// ======================================================================================
// Data per lane:   1 complete column (16 contiguous halves = 8 uint32_t)
// Lane mapping:    g = lid >> 2 idx = ((g >> 2) & 1) | (g & 2) c_base = (lid & 3) + (idx << 2)
// Replication:     L0-3 & L4-7 duplicate cols 0-3. Pattern repeats per column group.
//                  (Hardware 2x crossbar replication via lane grouping)
// Memory access:   Contiguous column storage 2x ld.shared.v4.u32 (128-bit loads).
// ======================================================================================
__device__ __forceinline__ void load_matrix_sync(
    fragment<matrix_b, 16, 16, 16, half, col_major>& frag,
    const half* __restrict__ smem_ptr, unsigned ldm) {

    unsigned smem_addr = __cvta_generic_to_shared(smem_ptr);

    asm volatile(
        "{\n\t"
        ".reg .u32 lid, g, idx, c_base, base, t;\n\t"
        "mov.u32 lid, %%laneid;\n\t"

        // g = lid >> 2
        "shr.b32 g, lid, 2;\n\t"
        // idx = ((g >> 2) & 1) | (g & 2)
        "shr.b32 t, g, 2; and.b32 t, t, 1;\n\t"
        "and.b32 idx, g, 2;\n\t"
        "or.b32 idx, idx, t;\n\t"

        // col = (lid & 3) + (idx << 2)
        "and.b32 c_base, lid, 3;\n\t"
        "shl.b32 t, idx, 2;\n\t"
        "add.u32 c_base, c_base, t;\n\t"

        // base = smem_addr + col * ldm * sizeof(half)
        "mul.lo.u32 base, c_base, %9;\n\t"
        "shl.b32 base, base, 1;\n\t"
        "add.u32 base, base, %8;\n\t"

        "ld.shared.v4.u32 {%0, %1, %2, %3}, [base];\n\t"
        "ld.shared.v4.u32 {%4, %5, %6, %7}, [base+16];\n\t"
        "}"
        : "=r"(frag.x[0]), "=r"(frag.x[1]), "=r"(frag.x[2]), "=r"(frag.x[3]),
          "=r"(frag.x[4]), "=r"(frag.x[5]), "=r"(frag.x[6]), "=r"(frag.x[7])
        : "r"(smem_addr), "r"(ldm)
        : "memory"
    );
}

// ======================================================================================
// LOAD: accumulator m16n16k16 (ROW & COL MAJOR)
// ======================================================================================
// Data per lane:   8 floats total, distributed as 4 contiguous pairs.
// Lane mapping:    r_base = ((lid>>2)&1)*8 + ((lid>>4)&1)*4 + (lid&1) c_base = ((lid>>3)&1)*8 + ((lid>>1)&1)*2
// Replication:     Hardware 2x crossbar duplication via ignored lane bits.
// Memory access:   ROW: Pairs row-contiguous. Stride=2*ldm*4B. Optimal: 4x v2.
//                  COL: Pairs col-contiguous. Elements strided by ldm. Optimal: 8x scalar.
// ======================================================================================
__device__ __forceinline__ void load_matrix_sync(
    fragment<accumulator, 16, 16, 16, float>& frag,
    const float* __restrict__ smem_ptr, unsigned ldm, layout_t layout) {

    unsigned smem_addr = __cvta_generic_to_shared(smem_ptr);

    if (layout == mem_row_major) {
        asm volatile(
            "{\n\t"
            ".reg .u32 lid, r_base, c_base, base, stride, a0, a1, a2, a3, t;\n\t"
            "mov.u32 lid, %%laneid;\n\t"

            // r0 = ((lid>>2)&1)*8 + ((lid>>4)&1)*4 + (lid&1)
            "and.b32 r_base, lid, 1;\n\t"
            "shr.b32 t, lid, 2; and.b32 t, t, 1; shl.b32 t, t, 3; add.u32 r_base, r_base, t;\n\t"
            "shr.b32 t, lid, 4; and.b32 t, t, 1; shl.b32 t, t, 2; add.u32 r_base, r_base, t;\n\t"

            // c0 = ((lid>>3)&1)*8 + ((lid>>1)&1)*2
            "shr.b32 c_base, lid, 3; and.b32 c_base, c_base, 1; shl.b32 c_base, c_base, 3;\n\t"
            "shr.b32 t, lid, 1; and.b32 t, t, 1; shl.b32 t, t, 1; add.u32 c_base, c_base, t;\n\t"

            // base = smem_addr + (r0 * ldm + c0) * sizeof(float)
            "mad.lo.u32 base, r_base, %9, c_base;\n\t"
            "shl.b32 base, base, 2;\n\t"
            "add.u32 base, base, %8;\n\t"

            // stride = 2 rows * ldm * 4 bytes = ldm << 3
            "shl.b32 stride, %9, 3;\n\t"
            "mov.u32 a0, base;\n\t"
            "add.u32 a1, base, stride;\n\t"
            "add.u32 a2, base, 16;\n\t"
            "add.u32 a3, a2, stride;\n\t"

            "ld.shared.v2.f32 {%0, %1}, [a0];\n\t"
            "ld.shared.v2.f32 {%2, %3}, [a1];\n\t"
            "ld.shared.v2.f32 {%4, %5}, [a2];\n\t"
            "ld.shared.v2.f32 {%6, %7}, [a3];\n\t"
            "}"
            : "=f"(frag.x[0]), "=f"(frag.x[1]), "=f"(frag.x[2]), "=f"(frag.x[3]),
              "=f"(frag.x[4]), "=f"(frag.x[5]), "=f"(frag.x[6]), "=f"(frag.x[7])
            : "r"(smem_addr), "r"(ldm)
            : "memory"
        );
    } else {
        asm volatile(
            "{\n\t"
            ".reg .u32 lid, r_base, c_base, base, sc, sc4, a0, a1, a2, a3, a4, a5, a6, a7, t;\n\t"
            "mov.u32 lid, %%laneid;\n\t"

            // r0 = ((lid>>2)&1)*8 + ((lid>>4)&1)*4 + (lid&1)
            "and.b32 r_base, lid, 1;\n\t"
            "shr.b32 t, lid, 2; and.b32 t, t, 1; shl.b32 t, t, 3; add.u32 r_base, r_base, t;\n\t"
            "shr.b32 t, lid, 4; and.b32 t, t, 1; shl.b32 t, t, 2; add.u32 r_base, r_base, t;\n\t"

            // c0 = ((lid>>3)&1)*8 + ((lid>>1)&1)*2
            "shr.b32 c_base, lid, 3; and.b32 c_base, c_base, 1; shl.b32 c_base, c_base, 3;\n\t"
            "shr.b32 t, lid, 1; and.b32 t, t, 1; shl.b32 t, t, 1; add.u32 c_base, c_base, t;\n\t"

            // base = smem_addr + (c0 * ldm + r0) * sizeof(float)
            "mad.lo.u32 base, c_base, %9, r_base;\n\t"
            "shl.b32 base, base, 2;\n\t"
            "add.u32 base, base, %8;\n\t"

            // strides: sc = ldm*4 (col step), sc4 = ldm*16 (col+4 step)
            "shl.b32 sc, %9, 2;\n\t"
            "shl.b32 sc4, %9, 4;\n\t"

            "mov.u32 a0, base;\n\t"
            "add.u32 a1, base, sc;\n\t"
            "add.u32 a2, base, 8;\n\t"
            "add.u32 a3, a2, sc;\n\t"
            "add.u32 a4, base, sc4;\n\t"
            "add.u32 a5, a4, sc;\n\t"
            "add.u32 a6, a4, 8;\n\t"
            "add.u32 a7, a6, sc;\n\t"

            "ld.shared.f32 %0, [a0];\n\t"
            "ld.shared.f32 %1, [a1];\n\t"
            "ld.shared.f32 %2, [a2];\n\t"
            "ld.shared.f32 %3, [a3];\n\t"
            "ld.shared.f32 %4, [a4];\n\t"
            "ld.shared.f32 %5, [a5];\n\t"
            "ld.shared.f32 %6, [a6];\n\t"
            "ld.shared.f32 %7, [a7];\n\t"
            "}"
            : "=f"(frag.x[0]), "=f"(frag.x[1]), "=f"(frag.x[2]), "=f"(frag.x[3]),
              "=f"(frag.x[4]), "=f"(frag.x[5]), "=f"(frag.x[6]), "=f"(frag.x[7])
            : "r"(smem_addr), "r"(ldm)
            : "memory"
        );
    }
}

// ======================================================================================
// STORE: accumulator m16n16k16 (ROW & COL MAJOR)
// ======================================================================================
// Data per lane:   8 floats total, distributed as 4 contiguous pairs.
// Lane mapping:    r_base = ((lid>>2)&1)*8 + ((lid>>4)&1)*4 + (lid&1) c_base = ((lid>>3)&1)*8 + ((lid>>1)&1)*2
// Replication:     Hardware 2x crossbar duplication via ignored lane bits.
// Memory access:   ROW: Pairs row-contiguous. Stride=2*ldm*4B. Optimal: 4x v2.
//                  COL: Pairs col-contiguous. Elements strided by ldm. Optimal: 8x scalar.
// ======================================================================================
__device__ __forceinline__ void store_matrix_sync(
    float* __restrict__ smem_ptr,
    const fragment<accumulator, 16, 16, 16, float>& frag,
    unsigned ldm, layout_t layout) {

    unsigned smem_addr = __cvta_generic_to_shared(smem_ptr);

    if (layout == mem_row_major) {
        asm volatile(
            "{\n\t"
            ".reg .u32 lid, r_base, c_base, base, stride, a0, a1, a2, a3, t;\n\t"
            "mov.u32 lid, %%laneid;\n\t"

            // r0 = ((lid>>2)&1)*8 + ((lid>>4)&1)*4 + (lid&1)
            "and.b32 r_base, lid, 1;\n\t"
            "shr.b32 t, lid, 2; and.b32 t, t, 1; shl.b32 t, t, 3; add.u32 r_base, r_base, t;\n\t"
            "shr.b32 t, lid, 4; and.b32 t, t, 1; shl.b32 t, t, 2; add.u32 r_base, r_base, t;\n\t"

            // c0 = ((lid>>3)&1)*8 + ((lid>>1)&1)*2
            "shr.b32 c_base, lid, 3; and.b32 c_base, c_base, 1; shl.b32 c_base, c_base, 3;\n\t"
            "shr.b32 t, lid, 1; and.b32 t, t, 1; shl.b32 t, t, 1; add.u32 c_base, c_base, t;\n\t"

            // base = smem_addr + (r0 * ldm + c0) * sizeof(float)
            "mad.lo.u32 base, r_base, %9, c_base;\n\t"
            "shl.b32 base, base, 2;\n\t"
            "add.u32 base, base, %8;\n\t"

            // stride = 2 rows * ldm * 4 bytes = ldm << 3
            "shl.b32 stride, %9, 3;\n\t"

            "mov.u32 a0, base;\n\t"
            "add.u32 a1, base, stride;\n\t"
            "add.u32 a2, base, 16;\n\t"
            "add.u32 a3, a2, stride;\n\t"

            "st.shared.v2.f32 [a0], {%0, %1};\n\t"
            "st.shared.v2.f32 [a1], {%2, %3};\n\t"
            "st.shared.v2.f32 [a2], {%4, %5};\n\t"
            "st.shared.v2.f32 [a3], {%6, %7};\n\t"
            "}"
            :
            : "f"(frag.x[0]), "f"(frag.x[1]), "f"(frag.x[2]), "f"(frag.x[3]),
              "f"(frag.x[4]), "f"(frag.x[5]), "f"(frag.x[6]), "f"(frag.x[7]),
              "r"(smem_addr), "r"(ldm)
            : "memory"
        );
    } else {
        asm volatile(
            "{\n\t"
            ".reg .u32 lid, r_base, c_base, base, sc, sc4, a0, a1, a2, a3, a4, a5, a6, a7, t;\n\t"
            "mov.u32 lid, %%laneid;\n\t"

            // r0 = ((lid>>2)&1)*8 + ((lid>>4)&1)*4 + (lid&1)
            "and.b32 r_base, lid, 1;\n\t"
            "shr.b32 t, lid, 2; and.b32 t, t, 1; shl.b32 t, t, 3; add.u32 r_base, r_base, t;\n\t"
            "shr.b32 t, lid, 4; and.b32 t, t, 1; shl.b32 t, t, 2; add.u32 r_base, r_base, t;\n\t"

            // c0 = ((lid>>3)&1)*8 + ((lid>>1)&1)*2
            "shr.b32 c_base, lid, 3; and.b32 c_base, c_base, 1; shl.b32 c_base, c_base, 3;\n\t"
            "shr.b32 t, lid, 1; and.b32 t, t, 1; shl.b32 t, t, 1; add.u32 c_base, c_base, t;\n\t"

            // base = smem_addr + (c0 * ldm + r0) * sizeof(float)
            "mad.lo.u32 base, c_base, %9, r_base;\n\t"
            "shl.b32 base, base, 2;\n\t"
            "add.u32 base, base, %8;\n\t"

             // strides: sc = ldm*4 (col step), sc4 = ldm*16 (col+4 step)
            "shl.b32 sc, %9, 2;\n\t"
            "shl.b32 sc4, %9, 4;\n\t"

            "mov.u32 a0, base;\n\t"
            "add.u32 a1, base, sc;\n\t"
            "add.u32 a2, base, 8;\n\t"
            "add.u32 a3, a2, sc;\n\t"
            "add.u32 a4, base, sc4;\n\t"
            "add.u32 a5, a4, sc;\n\t"
            "add.u32 a6, a4, 8;\n\t"
            "add.u32 a7, a6, sc;\n\t"

            "st.shared.f32 [a0], %0;\n\t"
            "st.shared.f32 [a1], %1;\n\t"
            "st.shared.f32 [a2], %2;\n\t"
            "st.shared.f32 [a3], %3;\n\t"
            "st.shared.f32 [a4], %4;\n\t"
            "st.shared.f32 [a5], %5;\n\t"
            "st.shared.f32 [a6], %6;\n\t"
            "st.shared.f32 [a7], %7;\n\t"
            "}"
            :
            : "f"(frag.x[0]), "f"(frag.x[1]), "f"(frag.x[2]), "f"(frag.x[3]),
              "f"(frag.x[4]), "f"(frag.x[5]), "f"(frag.x[6]), "f"(frag.x[7]),
              "r"(smem_addr), "r"(ldm)
            : "memory"
        );
    }
}

// ======================================================================================
// MMA_SYNC: m16n16k16 (ALL LAYOUTS)
// ======================================================================================
// Data per lane:   D = C + A @ B (8 floats accumulated in-place)
// Lane mapping:    Hardware-fixed crossbar routing. Fragments must match load layout.
// Replication:     Handled implicitly by TC crossbar. No manual shuffling allowed.
// Memory access:   N/A (register-only operation)
// Optimization:    Uses f32 accumulation for attention numerical stability.
//                  Constraint indices strictly match HMMA.884 operand order.
// ======================================================================================
#define WMMA_MMA_F32(M, N, K, ALAY, BLAY) \
__device__ __forceinline__ void mma_sync( \
    fragment<accumulator, M, N, K, float>& d, \
    const fragment<matrix_a, M, N, K, half, ALAY##_major>& a, \
    const fragment<matrix_b, M, N, K, half, BLAY##_major>& b, \
    const fragment<accumulator, M, N, K, float>& c) { \
    asm volatile( \
        "wmma.mma.sync.aligned." #ALAY "." #BLAY ".m" #M "n" #N "k" #K ".f32.f32 " \
        "{%0,%1,%2,%3,%4,%5,%6,%7}, " \
        "{%8,%9,%10,%11,%12,%13,%14,%15}, " \
        "{%16,%17,%18,%19,%20,%21,%22,%23}, " \
        "{%24,%25,%26,%27,%28,%29,%30,%31};" \
        : "=f"(d.x[0]), "=f"(d.x[1]), "=f"(d.x[2]), "=f"(d.x[3]), \
          "=f"(d.x[4]), "=f"(d.x[5]), "=f"(d.x[6]), "=f"(d.x[7]) \
        : "r"(a.x[0]), "r"(a.x[1]), "r"(a.x[2]), "r"(a.x[3]), \
          "r"(a.x[4]), "r"(a.x[5]), "r"(a.x[6]), "r"(a.x[7]), \
          "r"(b.x[0]), "r"(b.x[1]), "r"(b.x[2]), "r"(b.x[3]), \
          "r"(b.x[4]), "r"(b.x[5]), "r"(b.x[6]), "r"(b.x[7]), \
          "f"(c.x[0]), "f"(c.x[1]), "f"(c.x[2]), "f"(c.x[3]), \
          "f"(c.x[4]), "f"(c.x[5]), "f"(c.x[6]), "f"(c.x[7]) \
    ); \
}

WMMA_MMA_F32(16, 16, 16, row, col)
WMMA_MMA_F32(16, 16, 16, row, row)
WMMA_MMA_F32(16, 16, 16, col, col)
WMMA_MMA_F32(16, 16, 16, col, row)

#undef WMMA_MMA_F32

} // namespace wmma
} // namespace volta
