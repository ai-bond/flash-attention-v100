// ======================================================================================
// * Copyright (c) 2026, D.Skryabin / tg @ai_bond007 SPDX-License: BSD-3-Clause
// ======================================================================================
// * Volta WMMA (sm_70) production wrapper for FlashAttention-2.
// * Replaces nvcuda::wmma with explicit PTX for full control over
// * register packing, lane mapping, and LSU vectorization.
// *
// * 1. SUPPORTED SHAPE: m8n8k4 (HMMA.884.F32.F32)
// * 2. FRAGMENT LAYOUT: A/B = 4x half (2x uint32_t packed), C/D = 8x float
// * 3. LOAD/STORE:      Explicit lane-to-memory mapping
// * 4. DATA PACKING:    f16x2 passed as uint32_t to prevent repacking.
// * 5. ALIGNMENT:       Base pointers require 8B alignment. ldm in elements.
// * 6. THREAD MAPPING:  Hardware-fixed quad-pair distribution. Do not shuffle fragments.
// * 7. SWIZZLE:         Tile-Relative Logical Swizzle (16B chunk index).
// *                     Eliminates 16B vectorized bank conflicts on Volta (sm_70)
// *                     by XORing logical row/col bits into the 16B chunk index.
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

template <> struct fragment<matrix_a,    8, 8, 4, half,  row_major> { half  x[4]; static constexpr int num_elements = 4; };
template <> struct fragment<matrix_a,    8, 8, 4, half,  col_major> { half  x[4]; static constexpr int num_elements = 4; };
template <> struct fragment<matrix_b,    8, 8, 4, half,  row_major> { half  x[4]; static constexpr int num_elements = 4; };
template <> struct fragment<matrix_b,    8, 8, 4, half,  col_major> { half  x[4]; static constexpr int num_elements = 4; };
template <> struct fragment<accumulator, 8, 8, 4, float>            { float x[8]; static constexpr int num_elements = 8; };

// ======================================================================================
// FILL: matrix_a / matrix_b m8n8k4 (HALF)
// ======================================================================================
// Data per lane:   2x uint32_t registers (each holds 2 halves = 4 halves total)
// Lane mapping:    Broadcast single half into both slots of each uint32_t register
// Replication:     N/A (scalar initialization)
// Memory access:   N/A (register-only operation)
// ======================================================================================
template <typename Use, int M, int N, int K, typename Layout>
__device__ __forceinline__ void fill_fragment(fragment<Use, M, N, K, half, Layout>& frag, half value) {
    uint32_t packed = (uint32_t(__half_as_ushort(value)) << 16) | uint32_t(__half_as_ushort(value));
    asm volatile(
        "{\n\t"
        "mov.b32 %0, %2;\n\t"
        "mov.b32 %1, %2;\n\t"
        "}"
        : "=r"(*reinterpret_cast<uint32_t*>(&frag.x[0])),
          "=r"(*reinterpret_cast<uint32_t*>(&frag.x[2]))
        : "r"(packed)
    );
}

// ======================================================================================
// FILL: accumulator m8n8k4 (FLOAT)
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
// LOAD: matrix_a m8n8k4 (ROW MAJOR)
// ======================================================================================
// Data per lane:   4 contiguous halves (2 uint32_t)
// Lane mapping:    row = (lid & 3) | ((lid >> 4) & 1) * 4   [ 0..7 ]
// Replication:     Hardware 2x crossbar duplication via ignored lane bits
// Memory access:   Contiguous row storage 1x ld.shared.v2.u32 (64-bit load)
// ======================================================================================
__device__ __forceinline__ void load_matrix_sync(fragment<matrix_a, 8, 8, 4, half, row_major>& frag, const half* __restrict__ smem_ptr, unsigned ldm) {

    unsigned smem_addr = __cvta_generic_to_shared(smem_ptr);

    asm volatile(
        "{\n\t"
        ".reg .u32 lane_id, row_lo, row_hi, row, ldm2, offset, swz;\n\t"
        "mov.u32    lane_id, %%laneid;\n\t"

        // row = (lane_id & 3) | ((lane_id & 16) >> 2)
        "and.b32    row_lo, lane_id, 3;\n\t"
        "bfe.u32    row_hi, lane_id, 4, 1;\n\t"
        "mad.lo.u32 row,    row_hi, 4, row_lo;\n\t"

        // offset = row * (ldm*2) + smem_addr
        "shl.b32    ldm2, %3, 1;\n\t"
        "mad.lo.u32 offset, row, ldm2, %2;\n\t"

        // Tile-Relative Logical Swizzle: swz = (lid & 3) << 4
        "shl.b32    swz, row_lo, 4;\n\t"
        "xor.b32    offset, offset, swz;\n\t"

        "ld.shared.v2.u32 {%0, %1}, [offset];\n\t"
        "}"
        : "=r"(*reinterpret_cast<uint32_t*>(&frag.x[0])),
          "=r"(*reinterpret_cast<uint32_t*>(&frag.x[2]))
        : "r"(smem_addr), "r"(ldm)
        : "memory"
    );
}

// ======================================================================================
// LOAD: matrix_a m8n8k4 (COL MAJOR)
// ======================================================================================
// Data per lane:   4 contiguous halves (2 uint32_t)
// Lane mapping:    col = (lid & 3),  row_offset = (lid >> 4) << 2
// Replication:     Hardware 2x crossbar duplication via ignored lane bits
// Memory access:   Contiguous column storage
// Swizzle:         Tile-Relative Logical Swizzle on 16B chunk index
// ======================================================================================
__device__ __forceinline__ void load_matrix_sync(fragment<matrix_a, 8, 8, 4, half, col_major>& frag, const half* __restrict__ smem_ptr, unsigned ldm) {

    unsigned smem_addr = __cvta_generic_to_shared(smem_ptr);

    asm volatile(
        "{\n\t"
        ".reg .u32 lane_id, col, col_lo, row_off, ldm2, offset, swz, tmp_h;\n\t"
        "mov.u32    lane_id, %%laneid;\n\t"

        // col    = lane_id & 3
        // tmp_h  = lane_id & 16   (bit 4 = row_hi)
        "and.b32    col,   lane_id, 3;\n\t"
        "and.b32    tmp_h, lane_id, 16;\n\t"

        // row_off = (lid & 16) >> 2
        "shr.b32    row_off, tmp_h, 2;\n\t"

        // offset = col * (ldm*2) + row_off*2 + smem_addr
        "shl.b32    ldm2, %3, 1;\n\t"
        "mad.lo.u32 offset, row_off, 2, %2;\n\t"
        "mad.lo.u32 offset, col, ldm2, offset;\n\t"

        // Tile-Relative Logical Swizzle: swz = ((col & 1) << 5) | (lid & 16)
        "and.b32    col_lo, lane_id, 1;\n\t"
        "shl.b32    swz,    col_lo, 5;\n\t"
        "or.b32     swz,    swz, tmp_h;\n\t"

        "xor.b32    offset, offset, swz;\n\t"
        "ld.shared.v2.u32 {%0, %1}, [offset];\n\t"
        "}"
        : "=r"(*reinterpret_cast<uint32_t*>(&frag.x[0])),
          "=r"(*reinterpret_cast<uint32_t*>(&frag.x[2]))
        : "r"(smem_addr), "r"(ldm)
        : "memory"
    );
}

// ======================================================================================
// LOAD: matrix_b m8n8k4 (ROW MAJOR)
// ======================================================================================
// Data per lane:   4 contiguous halves (2 uint32_t)
// Lane mapping:    row = (lid & 3), col_offset = (lid >> 4) << 2
// Replication:     Hardware 2x crossbar duplication via ignored lane bits.
// Memory access:   Contiguous row storage 1x ld.shared.v2.u32 (64-bit load).
// Optimization:    Minimal register usage, direct address math, 7 PTX instructions.
// Swizzle:         Tile-Relative Logical Swizzle on 16B chunk index
// ======================================================================================
__device__ __forceinline__ void load_matrix_sync(fragment<matrix_b, 8, 8, 4, half, row_major>& frag, const half* __restrict__ smem_ptr, unsigned ldm) {

    unsigned smem_addr = __cvta_generic_to_shared(smem_ptr);

    asm volatile(
        "{\n\t"
        ".reg .u32 lane_id, r_base, c_bytes, ldm2, offset;\n\t"
        "mov.u32    lane_id, %%laneid;\n\t"

        // r_base  = lane_id & 3
        // c_bytes = (lane_id >> 4) & 1  (0 or 1)
        "and.b32    r_base,  lane_id, 3;\n\t"
        "bfe.u32    c_bytes, lane_id, 4, 1;\n\t"

        // offset = r_base * (ldm*2) + smem_addr
        "shl.b32    ldm2,    %3, 1;\n\t"
        "mad.lo.u32 offset,  r_base, ldm2, %2;\n\t"

        // offset += 8 * c_bytes (second half of row)
        "mad.lo.u32 offset,  c_bytes, 8, offset;\n\t"

        "ld.shared.v2.u32 {%0, %1}, [offset];\n\t"
        "}"
        : "=r"(*reinterpret_cast<uint32_t*>(&frag.x[0])),
          "=r"(*reinterpret_cast<uint32_t*>(&frag.x[2]))
        : "r"(smem_addr), "r"(ldm)
        : "memory"
    );
}

// ======================================================================================
// LOAD: matrix_b m8n8k4 (COL MAJOR)
// ======================================================================================
// Data per lane:   4 contiguous halves (2 uint32_t) = one full B column
// Lane mapping:    col = (lid & 3) + ((lid >> 4) << 2)   [ 0..7 ]
// Replication:     Hardware 2x crossbar duplication via ignored lane bits
// Memory access:   Contiguous column storage (64-bit load)
// Swizzle:         Tile-Relative Logical Swizzle on 16B chunk index
// ======================================================================================
__device__ __forceinline__ void load_matrix_sync(fragment<matrix_b, 8, 8, 4, half, col_major>& frag, const half* __restrict__ smem_ptr, unsigned ldm) {

    unsigned smem_addr = __cvta_generic_to_shared(smem_ptr);

    asm volatile(
        "{\n\t"
        ".reg .u32 lane_id, col_lo, row_hi, col, ldm2, offset;\n\t"
        "mov.u32    lane_id, %%laneid;\n\t"

        // col = (lane_id & 3) + ((lane_id & 16) >> 2)
        "and.b32    col_lo, lane_id, 3;\n\t"
        "bfe.u32    row_hi, lane_id, 4, 1;\n\t"
        "mad.lo.u32 col,    row_hi, 4, col_lo;\n\t"

        // offset = col * (ldm*2) + smem_addr
        "shl.b32    ldm2,   %3, 1;\n\t"
        "mad.lo.u32 offset, col, ldm2, %2;\n\t"

        "ld.shared.v2.u32 {%0, %1}, [offset];\n\t"
        "}"
        : "=r"(*reinterpret_cast<uint32_t*>(&frag.x[0])),
          "=r"(*reinterpret_cast<uint32_t*>(&frag.x[2]))
        : "r"(smem_addr), "r"(ldm)
        : "memory"
    );
}

// ======================================================================================
// LOAD: accumulator m8n8k4 (ROW & COL MAJOR)
// ======================================================================================
// Data per lane:   8 floats total, distributed as 4 v2 pairs.
// Logical pairs:   (r0,c0) -> x0,x1   (r0+2,c0) -> x2,x3
//                  (r0,c0+4) -> x4,x5 (r0+2,c0+4) -> x6,x7
// Lane mapping:    r0 = (lid&1) + ((lid>>4)&1)*4 ; c0 = lid&2
// Memory access:   ROW: 4x ld.shared.v2.f32 (contiguous pairs)
//                  COL: 8x scalar ld.shared.f32 (elements strided by ldm)
// Swizzle:         Tile-Relative Logical Swizzle per row / per column
// ======================================================================================
__device__ __forceinline__ void load_matrix_sync(fragment<accumulator, 8, 8, 4, float>& frag, const float* __restrict__ smem_ptr, unsigned ldm, layout_t layout) {

    unsigned smem_addr = __cvta_generic_to_shared(smem_ptr);

    if (layout == mem_row_major) {
        asm volatile(
            "{\n\t"
            ".reg .u32 lid, r0, ldm_b, a0, a1, a2, a3, swz0, swz1, tmp;\n\t"
            "mov.u32    lid, %%laneid;\n\t"

            // r0 = (lid & 1) + ((lid >> 4) & 1) * 4
            "and.b32    r0, lid, 1;\n\t"
            "bfe.u32    tmp, lid, 4, 1;\n\t"
            "mad.lo.u32 r0, tmp, 4, r0;\n\t"

            // ldm_b = ldm * 4 (bytes per row)
            "shl.b32    ldm_b, %9, 2;\n\t"

            // a0 = r0*ldm_b + smem_addr
            "mad.lo.u32 a0, r0, ldm_b, %8;\n\t"

            // a0 += (lid & 2) * 4 = (lid<<2)&8
            "shl.b32    tmp, lid, 2;\n\t"
            "and.b32    tmp, tmp, 8;\n\t"
            "add.u32    a0, a0, tmp;\n\t"

            // swz0 = (lid & 1) << 4 ; swz1 = swz0 | 0x20
            "shl.b32    swz0, lid, 4;\n\t"
            "and.b32    swz0, swz0, 0x10;\n\t"
            "or.b32     swz1, swz0, 0x20;\n\t"

            // stride2 = 2*ldm_b
            "shl.b32    tmp, ldm_b, 1;\n\t"

            // a1 = a0 + 2*ldm_b  (r0+2, c0)
            // a2 = a0 + 16       (r0, c0+4)
            // a3 = a1 + 16       (r0+2, c0+4)
            "add.u32    a1, a0, tmp;\n\t"
            "add.u32    a2, a0, 16;\n\t"
            "add.u32    a3, a1, 16;\n\t"

            // Swizzle (row-major -> swz by row)
            "xor.b32    a0, a0, swz0;\n\t"
            "xor.b32    a2, a2, swz0;\n\t"
            "xor.b32    a1, a1, swz1;\n\t"
            "xor.b32    a3, a3, swz1;\n\t"

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
            ".reg .u32 lid, r0, c0, ldm_b, a0, a1, a2, a3, a4, a5, a6, a7, swz0, swz1, tmp;\n\t"
            "mov.u32    lid, %%laneid;\n\t"

            // r0 = (lid & 1) + ((lid >> 4) & 1) * 4
            "and.b32    r0, lid, 1;\n\t"
            "bfe.u32    tmp, lid, 4, 1;\n\t"
            "mad.lo.u32 r0, tmp, 4, r0;\n\t"

            // ldm_b = ldm * 4 (bytes per column)
            "shl.b32    ldm_b, %9, 2;\n\t"

            // a0 = r0*4 + smem_addr
            "mad.lo.u32 a0, r0, 4, %8;\n\t"

            // c0 = lid & 2 ; a0 += c0 * ldm_b
            "and.b32    c0, lid, 2;\n\t"
            "mad.lo.u32 a0, c0, ldm_b, a0;\n\t"

            // swz0 = (lid<<4)&0x20 ; swz1 = swz0 | 0x10
            "shl.b32    swz0, lid, 4;\n\t"
            "and.b32    swz0, swz0, 0x20;\n\t"
            "or.b32     swz1, swz0, 0x10;\n\t"

            // col c0:            a0  (r0,c0)     a2  (r0+2,c0)
            // col c0+1:          a1  (r0,c0+1)   a3  (r0+2,c0+1)
            // col c0+4:          a4  (r0,c0+4)   a6  (r0+2,c0+4)
            // col c0+5:          a5  (r0,c0+5)   a7  (r0+2,c0+5)
            "add.u32    a1, a0, ldm_b;\n\t"
            "add.u32    a2, a0, 8;\n\t"
            "add.u32    a3, a1, 8;\n\t"

            // stride4 = 4*ldm_b (reuse c0 as temp)
            "shl.b32    c0, ldm_b, 2;\n\t"
            "add.u32    a4, a0, c0;\n\t"
            "add.u32    a5, a4, ldm_b;\n\t"
            "add.u32    a6, a4, 8;\n\t"
            "add.u32    a7, a5, 8;\n\t"

            // Swizzle (col-major -> swz by column)
            "xor.b32    a0, a0, swz0;\n\t"
            "xor.b32    a2, a2, swz0;\n\t"
            "xor.b32    a4, a4, swz0;\n\t"
            "xor.b32    a6, a6, swz0;\n\t"
            "xor.b32    a1, a1, swz1;\n\t"
            "xor.b32    a3, a3, swz1;\n\t"
            "xor.b32    a5, a5, swz1;\n\t"
            "xor.b32    a7, a7, swz1;\n\t"

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
// STORE: accumulator m8n8k4 (ROW & COL MAJOR)
// ======================================================================================
// Data per lane:   8 floats = 4 v2 pairs (exact inverse scatter of load).
// Logical pairs:   (r0,c0) -> x0,x1   (r0+2,c0) -> x2,x3
//                  (r0,c0+4) -> x4,x5 (r0+2,c0+4) -> x6,x7
// Lane mapping:    r0 = (lid&1) + ((lid>>4)&1)*4 ; c0 = lid&2
// Memory access:   ROW: 4x st.shared.v2.f32 (contiguous pairs).
//                  COL: 8x scalar st.shared.f32 (elements strided by ldm)
// Swizzle:         Tile-Relative Logical Swizzle, symmetric to load_matrix_sync
// ======================================================================================
__device__ __forceinline__ void store_matrix_sync(float* __restrict__ smem_ptr, const fragment<accumulator, 8, 8, 4, float>& frag, unsigned ldm, layout_t layout) {

    unsigned smem_addr = __cvta_generic_to_shared(smem_ptr);

    if (layout == mem_row_major) {
        asm volatile(
            "{\n\t"
            ".reg .u32 lid, r0, ldm_b, a0, a1, a2, a3, swz0, swz1, tmp;\n\t"
            "mov.u32    lid, %%laneid;\n\t"

            // r0 = (lid & 1) + ((lid >> 4) & 1) * 4
            "and.b32    r0, lid, 1;\n\t"
            "bfe.u32    tmp, lid, 4, 1;\n\t"
            "mad.lo.u32 r0, tmp, 4, r0;\n\t"

            // ldm_b = ldm * 4 (bytes per row)
            "shl.b32    ldm_b, %9, 2;\n\t"

            // a0 = r0*ldm_b + smem_addr
            "mad.lo.u32 a0, r0, ldm_b, %8;\n\t"

            // a0 += (lid & 2) * 4 = (lid<<2)&8
            "shl.b32    tmp, lid, 2;\n\t"
            "and.b32    tmp, tmp, 8;\n\t"
            "add.u32    a0, a0, tmp;\n\t"

            // swz0 = (lid & 1) << 4 ; swz1 = swz0 | 0x20
            "shl.b32    swz0, lid, 4;\n\t"
            "and.b32    swz0, swz0, 0x10;\n\t"
            "or.b32     swz1, swz0, 0x20;\n\t"

            // stride2 = 2*ldm_b
            "shl.b32    tmp, ldm_b, 1;\n\t"

            // a1 = a0 + 2*ldm_b  (r0+2, c0)
            // a2 = a0 + 16       (r0, c0+4)
            // a3 = a1 + 16       (r0+2, c0+4)
            "add.u32    a1, a0, tmp;\n\t"
            "add.u32    a2, a0, 16;\n\t"
            "add.u32    a3, a1, 16;\n\t"

            // Swizzle (row-major -> swz by row)
            "xor.b32    a0, a0, swz0;\n\t"
            "xor.b32    a2, a2, swz0;\n\t"
            "xor.b32    a1, a1, swz1;\n\t"
            "xor.b32    a3, a3, swz1;\n\t"

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
            ".reg .u32 lid, r0, c0, ldm_b, a0, a1, a2, a3, a4, a5, a6, a7, swz0, swz1, tmp;\n\t"
            "mov.u32    lid, %%laneid;\n\t"

            // r0 = (lid & 1) + ((lid >> 4) & 1) * 4
            "and.b32    r0, lid, 1;\n\t"
            "bfe.u32    tmp, lid, 4, 1;\n\t"
            "mad.lo.u32 r0, tmp, 4, r0;\n\t"

            // ldm_b = ldm * 4 (bytes per column)
            "shl.b32    ldm_b, %9, 2;\n\t"

            // a0 = r0*4 + smem_addr
            "mad.lo.u32 a0, r0, 4, %8;\n\t"

            // c0 = lid & 2 ; a0 += c0 * ldm_b
            "and.b32    c0, lid, 2;\n\t"
            "mad.lo.u32 a0, c0, ldm_b, a0;\n\t"

            // swz0 = (lid<<4)&0x20 ; swz1 = swz0 | 0x10
            "shl.b32    swz0, lid, 4;\n\t"
            "and.b32    swz0, swz0, 0x20;\n\t"
            "or.b32     swz1, swz0, 0x10;\n\t"

            // col c0:            a0  (r0,c0)     a2  (r0+2,c0)
            // col c0+1:          a1  (r0,c0+1)   a3  (r0+2,c0+1)
            // col c0+4:          a4  (r0,c0+4)   a6  (r0+2,c0+4)
            // col c0+5:          a5  (r0,c0+5)   a7  (r0+2,c0+5)
            "add.u32    a1, a0, ldm_b;\n\t"
            "add.u32    a2, a0, 8;\n\t"
            "add.u32    a3, a1, 8;\n\t"

            // stride4 = 4*ldm_b (reuse c0 as temp)
            "shl.b32    c0, ldm_b, 2;\n\t"
            "add.u32    a4, a0, c0;\n\t"
            "add.u32    a5, a4, ldm_b;\n\t"
            "add.u32    a6, a4, 8;\n\t"
            "add.u32    a7, a5, 8;\n\t"

            // Swizzle (col-major -> swz by column)
            "xor.b32    a0, a0, swz0;\n\t"
            "xor.b32    a2, a2, swz0;\n\t"
            "xor.b32    a4, a4, swz0;\n\t"
            "xor.b32    a6, a6, swz0;\n\t"
            "xor.b32    a1, a1, swz1;\n\t"
            "xor.b32    a3, a3, swz1;\n\t"
            "xor.b32    a5, a5, swz1;\n\t"
            "xor.b32    a7, a7, swz1;\n\t"

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
// MMA_SYNC: m8n8k4 (ALL LAYOUTS)
// ======================================================================================
// Data per lane:   D = C + A @ B (8 floats accumulated in-place)
// Lane mapping:    Hardware-fixed crossbar routing. Fragments must match load layout.
// Replication:     Handled implicitly by TC crossbar. No manual shuffling allowed.
// Memory access:   N/A (register-only operation)
// Optimization:    Uses f32 accumulation for attention numerical stability.
//                  A/B operands packed as 2x uint32_t per thread (indices 0 & 2).
// Instruction:     mma.sync.aligned.m8n8k4.{row|col}.{row|col}.f32.f16.f16.f32
// ======================================================================================
#define WMMA_MMA_F32(A_LAYOUT, B_LAYOUT) \
__device__ __forceinline__ void mma_sync( \
    fragment<accumulator, 8, 8, 4, float>& d, \
    const fragment<matrix_a, 8, 8, 4, half, A_LAYOUT##_major>& a, \
    const fragment<matrix_b, 8, 8, 4, half, B_LAYOUT##_major>& b, \
    const fragment<accumulator, 8, 8, 4, float>& c) { \
    asm volatile( \
        "mma.sync.aligned.m8n8k4." #A_LAYOUT "." #B_LAYOUT ".f32.f16.f16.f32 " \
        "{%0,%1,%2,%3,%4,%5,%6,%7}, " \
        "{%8,%9}, " \
        "{%10,%11}, " \
        "{%12,%13,%14,%15,%16,%17,%18,%19};" \
        : "=f"(d.x[0]), "=f"(d.x[1]), "=f"(d.x[2]), "=f"(d.x[3]), \
          "=f"(d.x[4]), "=f"(d.x[5]), "=f"(d.x[6]), "=f"(d.x[7]) \
        : "r"(*reinterpret_cast<const uint32_t*>(&a.x[0])), \
          "r"(*reinterpret_cast<const uint32_t*>(&a.x[2])), \
          "r"(*reinterpret_cast<const uint32_t*>(&b.x[0])), \
          "r"(*reinterpret_cast<const uint32_t*>(&b.x[2])), \
          "f"(c.x[0]), "f"(c.x[1]), "f"(c.x[2]), "f"(c.x[3]), \
          "f"(c.x[4]), "f"(c.x[5]), "f"(c.x[6]), "f"(c.x[7]) \
    ); \
}

WMMA_MMA_F32(row, col)
WMMA_MMA_F32(row, row)
WMMA_MMA_F32(col, col)
WMMA_MMA_F32(col, row)

#undef WMMA_MMA_F32

} // namespace wmma
} // namespace volta
