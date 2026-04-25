// ======================================================================================
// * Copyright (c) 2025, D.Skryabin / tg @ai_bond007 SPDX-License: BSD-3-Clause
// ======================================================================================
// * Volta HMMA m8n8k4 wrapper for FlashAttention-2 forward and backward passes.
// *
// * 1. SUPPORTED SHAPE (PTX ISA 6.4, mma.sync.aligned.m8n8k4)
// *    m8n8k4 : compact 8x8x4 configuration optimized for small tile reductions
// *
// * 2. FRAGMENT REGISTER LAYOUT
// *    A/B fragments store 4 half values per thread. C/D accumulators store 8 floats.
// *    Registers allocated per thread:
// *      A/B : 4 half (packed as 2 uint32_t for PTX operands)
// *      C/D : 8 float
// *
// * 3. LOAD/STORE CONSTRAINTS
// *    The wmma.load.c and wmma.store.d instructions do not support m8n8k4 on sm_70.
// *    Accumulator load/store operations use explicit lane-to-memory mapping.
// *    Matrix A/B loads use direct shared memory indexing aligned with warp distribution.
// *
// * 4. DATA PACKING AND PTX OPERANDS
// *    PTX mma.sync requires A/B operands as 32-bit packed registers.
// *    The wrapper reinterprets half arrays as uint32_t at indices 0 and 2.
// *    Never pass unpacked half values directly to mma_sync.
// *
// * 5. THREAD AND LANE MAPPING
// *    Data distribution follows Volta quad-pair topology.
// *    Lane indexing uses bitwise decomposition to match hardware expectations
// *    for 8x8 tile coverage across 32 threads. Manual mapping guarantees
// *    correct alignment without compiler interference.
// *
// * 6. ACCUMULATION PRECISION
// *    Uses f32 accumulation (.f32.f16.f16.f32) for numerical stability in attention.
// *    The f16 accumulation variant is unsupported for this shape and precision profile.
// ======================================================================================

#ifndef FUSED_MMA_M8N8K4_H
#define FUSED_MMA_M8N8K4_H

#pragma once

#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ != 700)
#error "Volta HMMA m8n8k4: This header is for sm_70 ONLY! Compile with -arch=sm_70"
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

enum layout_t { mem_row_major, mem_col_major };

// ======================================================================================
// FRAGMENT DECLARATIONS
// ======================================================================================
template <typename T, int size>
struct alignas(4) __frag_base {
    T x[size];
    static constexpr int num_elements = size;
};

template <typename Use, int M, int N, int K, typename T, typename Layout = void>
struct fragment;

template <> struct fragment<matrix_a, 8, 8, 4, half, row_major> : __frag_base<half, 4> {};
template <> struct fragment<matrix_a, 8, 8, 4, half, col_major> : __frag_base<half, 4> {};
template <> struct fragment<matrix_b, 8, 8, 4, half, row_major> : __frag_base<half, 4> {};
template <> struct fragment<matrix_b, 8, 8, 4, half, col_major> : __frag_base<half, 4> {};
template <> struct fragment<accumulator, 8, 8, 4, float, row_major> : __frag_base<float, 8> {};
template <> struct fragment<accumulator, 8, 8, 4, float, col_major> : __frag_base<float, 8> {};
template <typename Use, int M, int N, int K, typename T>
struct fragment<Use, M, N, K, T, void> : fragment<Use, M, N, K, T, row_major> {};

// ======================================================================================
// FILL: accumulator m8n8k4
// ======================================================================================
// Data per lane:   8 floats (full accumulator fragment)
// Lane mapping:    Uniform broadcast across all 32 lanes
// Replication:     N/A (scalar initialization)
// Memory access:   N/A (register-only operation)
// ======================================================================================
template <int M, int N, int K, typename Layout>
__device__ __forceinline__ void fill_fragment(
    fragment<accumulator, M, N, K, float, Layout>& frag, float value) {
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
// Data per lane: 4 contiguous halves (2 uint32_t)
// Lane mapping:  row = (lid & 3) + ((lid >> 4) & 1) * 4
// Replication:   Hardware 2x crossbar duplication via ignored lane bits.
// Memory access: Contiguous row storage 1x ld.shared.v2.u32.
// Optimization:  Minimal register usage, direct address math, 64-bit load.
// Note:          Constraints: %0-%1 outputs, %2 smem_addr, %3 ldm.
// ======================================================================================
__device__ __forceinline__ void load_matrix_sync(
    fragment<matrix_a, 8, 8, 4, half, row_major>& frag,
    const half* __restrict__ smem_ptr, unsigned ldm) {

    unsigned smem_addr = __cvta_generic_to_shared(smem_ptr);

    asm volatile(
        "{\n\t"
        ".reg .u32 lane_id, row, offset;\n\t"
        "mov.u32 lane_id, %%laneid;\n\t"

        // row = (lane_id & 0x3) + ((lane_id >> 4) << 2)
        "and.b32 row, lane_id, 3;\n\t"
        "shr.b32 offset, lane_id, 4;\n\t"
        "shl.b32 offset, offset, 2;\n\t"
        "add.u32 row, row, offset;\n\t"

        // byte_offset = row * ldm * 2 + smem_addr
        "mul.lo.u32 offset, row, %3;\n\t"
        "shl.b32 offset, offset, 1;\n\t"
        "add.u32 offset, %2, offset;\n\t"

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
// Data per lane: 4 contiguous halves (2 uint32_t)
// Lane mapping:  col = (lid & 3) + ((lid >> 4) & 1) * 4
// Replication:   Hardware 2x crossbar duplication via ignored lane bits.
// Memory access: Contiguous column storage 1x ld.shared.v2.u32.
// Optimization:  Symmetric to row_major, minimal regs, 64-bit load.
// Note:          Constraints: %0-%1 outputs, %2 smem_addr, %3 ldm.
// ======================================================================================
__device__ __forceinline__ void load_matrix_sync(
    fragment<matrix_a, 8, 8, 4, half, col_major>& frag,
    const half* __restrict__ smem_ptr, unsigned ldm) {

    unsigned smem_addr = __cvta_generic_to_shared(smem_ptr);

    asm volatile(
        "{\n\t"
        ".reg .u32 lane_id, col, row_off, offset;\n\t"
        "mov.u32 lane_id, %%laneid;\n\t"

        // col = lane_id & 0x3
        "and.b32 col, lane_id, 3;\n\t"

        // row_offset = (lane_id >> 4) << 2
        "shr.b32 row_off, lane_id, 4;\n\t"
        "shl.b32 row_off, row_off, 2;\n\t"

        // byte_offset = (col * ldm + row_off) * 2 + smem_addr
        "mul.lo.u32 offset, col, %3;\n\t"
        "add.u32 offset, offset, row_off;\n\t"
        "shl.b32 offset, offset, 1;\n\t"
        "add.u32 offset, %2, offset;\n\t"

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
// Data per lane: 4 contiguous halves (2 uint32_t)
// Lane mapping:  row = (lid & 3) + ((lid >> 4) & 1) * 4
// Replication:   Hardware 2x crossbar duplication via ignored lane bits.
// Memory access: Contiguous row storage → 1x ld.shared.v2.u32.
// Optimization:  Identical to matrix_a row_major due to square 8x8 shape.
// Note:          Constraints: %0-%1 outputs, %2 smem_addr, %3 ldm.
// ======================================================================================
__device__ __forceinline__ void load_matrix_sync(
    fragment<matrix_b, 8, 8, 4, half, row_major>& frag,
    const half* __restrict__ smem_ptr, unsigned ldm) {

    unsigned smem_addr = __cvta_generic_to_shared(smem_ptr);

    asm volatile(
        "{\n\t"
        ".reg .u32 lane_id, row, col_off, off;\n\t"
        "mov.u32 lane_id, %%laneid;\n\t"

        "and.b32 row, lane_id, 3;\n\t"
        "shr.b32 col_off, lane_id, 4;\n\t"
        "shl.b32 col_off, col_off, 2;\n\t"

        "mul.lo.u32 off, row, %3;\n\t"
        "add.u32 off, off, col_off;\n\t"
        "shl.b32 off, off, 1;\n\t"
        "add.u32 off, %2, off;\n\t"

        "ld.shared.v2.u32 {%0, %1}, [off];\n\t"
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
// Data per lane: 4 contiguous halves (2 uint32_t)
// Lane mapping:  col = (lid & 3) + ((lid >> 4) & 1) * 4
// Replication:   Hardware 2x crossbar duplication via ignored lane bits.
// Memory access: Contiguous column storage → 1x ld.shared.v2.u32.
// Optimization:  Symmetric to row_major, minimal regs, 64-bit load.
// Note:          Constraints: %0-%1 outputs, %2 smem_addr, %3 ldm.
// ======================================================================================
__device__ __forceinline__ void load_matrix_sync(
    fragment<matrix_b, 8, 8, 4, half, col_major>& frag,
    const half* __restrict__ smem_ptr, unsigned ldm) {

    unsigned smem_addr = __cvta_generic_to_shared(smem_ptr);

    asm volatile(
        "{\n\t"
        ".reg .u32 lane_id, col, row_off, off;\n\t"
        "mov.u32 lane_id, %%laneid;\n\t"

        "and.b32 col, lane_id, 3;\n\t"
        "shr.b32 row_off, lane_id, 4;\n\t"
        "shl.b32 row_off, row_off, 2;\n\t"
        "add.u32 col, col, row_off;\n\t"

        "mul.lo.u32 off, col, %3;\n\t"
        "shl.b32 off, off, 1;\n\t"
        "add.u32 off, %2, off;\n\t"

        "ld.shared.v2.u32 {%0, %1}, [off];\n\t"
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
// Data lane:    8 floats total, distributed as 4 contiguous pairs.
// Logical pair: (r0,c0), (r0+2,c0), (r0,c0+4), (r0+2,c0+4)
// Lane mapping: r0 = (lid & 1) + ((lid >> 4) & 1) * 4  c0 = lid & 2
// Replication:   Hardware 2x crossbar duplication via ignored lane bits.
// Memory Access & Vectorization:
//       ROW MAJOR: Pairs are row-contiguous. Row stride = 2*ldm*4 bytes.
//                  Col jump = 16 bytes. Optimal: 4x ld.shared.v2.f32.
//       COL MAJOR: Pairs are col-contiguous. Elements within a pair are
//                  strided by ldm. Vector loads impossible. Optimal: 8x scalar ld.shared.f32.
// Note:         Constraints: %0-%7 outputs, %8 smem_addr, %9 ldm.
// ======================================================================================
__device__ __forceinline__ void load_matrix_sync(
    fragment<accumulator, 8, 8, 4, float, row_major>& frag,
    const float* __restrict__ smem_ptr, unsigned ldm, layout_t layout) {

    unsigned smem_addr = __cvta_generic_to_shared(smem_ptr);

    if (layout == mem_row_major) {
        asm volatile(
            "{\n\t"
            ".reg .u32 lid, r0, c0, addr;\n\t"
            "mov.u32 lid, %%laneid;\n\t"
            "and.b32 r0, lid, 1;\n\t"
            ".reg .u32 tmp; and.b32 tmp, lid, 16; shr.b32 tmp, tmp, 2; add.u32 r0, r0, tmp;\n\t"
            "and.b32 c0, lid, 2;\n\t"

            // Pair 0: i=0,1 (row=r0, col=c0)
            "mad.lo.u32 addr, r0, %9, c0;\n\t"
            "shl.b32 addr, addr, 2;\n\t"
            "add.u32 addr, addr, %8;\n\t"
            "ld.shared.v2.f32 {%0, %1}, [addr];\n\t"

            // Pair 1: i=2,3 (row=r0+2, col=c0)
            "add.u32 r0, r0, 2;\n\t"
            "mad.lo.u32 addr, r0, %9, c0;\n\t"
            "shl.b32 addr, addr, 2;\n\t"
            "add.u32 addr, addr, %8;\n\t"
            "ld.shared.v2.f32 {%2, %3}, [addr];\n\t"

            // Pair 2: i=4,5 (row=r0-2, col=c0+4)
            "sub.u32 r0, r0, 2;\n\t"
            "add.u32 c0, c0, 4;\n\t"
            "mad.lo.u32 addr, r0, %9, c0;\n\t"
            "shl.b32 addr, addr, 2;\n\t"
            "add.u32 addr, addr, %8;\n\t"
            "ld.shared.v2.f32 {%4, %5}, [addr];\n\t"

            // Pair 3: i=6,7 (row=r0+2, col=c0)
            "add.u32 r0, r0, 2;\n\t"
            "mad.lo.u32 addr, r0, %9, c0;\n\t"
            "shl.b32 addr, addr, 2;\n\t"
            "add.u32 addr, addr, %8;\n\t"
            "ld.shared.v2.f32 {%6, %7}, [addr];\n\t"
            "}"
            : "=f"(frag.x[0]), "=f"(frag.x[1]), "=f"(frag.x[2]), "=f"(frag.x[3]),
              "=f"(frag.x[4]), "=f"(frag.x[5]), "=f"(frag.x[6]), "=f"(frag.x[7])
            : "r"(smem_addr), "r"(ldm)
            : "memory"
        );
    } else {
        asm volatile(
            "{\n\t"
            ".reg .u32 lid, r0, c0, r, c, addr;\n\t"
            "mov.u32 lid, %%laneid;\n\t"
            "and.b32 r0, lid, 1;\n\t"
            ".reg .u32 tmp; and.b32 tmp, lid, 16; shr.b32 tmp, tmp, 2; add.u32 r0, r0, tmp;\n\t"
            "and.b32 c0, lid, 2;\n\t"
            "mov.u32 r, r0; mov.u32 c, c0;\n\t"

            "mad.lo.u32 addr, c, %9, r; shl.b32 addr, addr, 2; add.u32 addr, addr, %8; ld.shared.f32 %0, [addr];\n\t"
            "mov.u32 r, r0; add.u32 c, c0, 1;\n\t"
            "mad.lo.u32 addr, c, %9, r; shl.b32 addr, addr, 2; add.u32 addr, addr, %8; ld.shared.f32 %1, [addr];\n\t"
            "add.u32 r, r0, 2; mov.u32 c, c0;\n\t"
            "mad.lo.u32 addr, c, %9, r; shl.b32 addr, addr, 2; add.u32 addr, addr, %8; ld.shared.f32 %2, [addr];\n\t"
            "add.u32 r, r0, 2; add.u32 c, c0, 1;\n\t"
            "mad.lo.u32 addr, c, %9, r; shl.b32 addr, addr, 2; add.u32 addr, addr, %8; ld.shared.f32 %3, [addr];\n\t"
            "mov.u32 r, r0; add.u32 c, c0, 4;\n\t"
            "mad.lo.u32 addr, c, %9, r; shl.b32 addr, addr, 2; add.u32 addr, addr, %8; ld.shared.f32 %4, [addr];\n\t"
            "mov.u32 r, r0; add.u32 c, c0, 5;\n\t"
            "mad.lo.u32 addr, c, %9, r; shl.b32 addr, addr, 2; add.u32 addr, addr, %8; ld.shared.f32 %5, [addr];\n\t"
            "add.u32 r, r0, 2; add.u32 c, c0, 4;\n\t"
            "mad.lo.u32 addr, c, %9, r; shl.b32 addr, addr, 2; add.u32 addr, addr, %8; ld.shared.f32 %6, [addr];\n\t"
            "add.u32 r, r0, 2; add.u32 c, c0, 5;\n\t"
            "mad.lo.u32 addr, c, %9, r; shl.b32 addr, addr, 2; add.u32 addr, addr, %8; ld.shared.f32 %7, [addr];\n\t"
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
// Data lane: 8 floats total, distributed as 4 contiguous pairs.
// Logical pair: (r0,c0), (r0+2,c0), (r0,c0+4), (r0+2,c0+4)
// Lane mapping (identical to load, store is exact inverse scatter):
//       r0 = (lid & 1) + ((lid >> 4) & 1) * 4
//       c0 = lid & 2
// Replication:   Hardware 2x crossbar duplication via ignored lane bits.
// Memory Access & Vectorization:
//       ROW MAJOR: Pairs are row-contiguous. Row stride = 2*ldm*4 bytes.
//                  Col jump = 16 bytes. Optimal: 4x st.shared.v2.f32.
//       COL MAJOR: Pairs are col-contiguous. Elements within a pair are
//                  strided by ldm. Vector stores impossible. Optimal: 8x scalar st.shared.f32.
// Note:            Constraints: %0-%7 inputs, %8 smem_addr, %9 ldm.
// ======================================================================================
__device__ __forceinline__ void store_matrix_sync(
    float* __restrict__ smem_ptr,
    const fragment<accumulator, 8, 8, 4, float, row_major>& frag,
    unsigned ldm, layout_t layout) {

    unsigned smem_addr = __cvta_generic_to_shared(smem_ptr);

    if (layout == mem_row_major) {
        asm volatile(
            "{\n\t"
            ".reg .u32 lid, r0, c0, addr;\n\t"
            "mov.u32 lid, %%laneid;\n\t"
            "and.b32 r0, lid, 1;\n\t"
            ".reg .u32 tmp; and.b32 tmp, lid, 16; shr.b32 tmp, tmp, 2; add.u32 r0, r0, tmp;\n\t"
            "and.b32 c0, lid, 2;\n\t"

            // Pair 0: i=0,1
            "mad.lo.u32 addr, r0, %9, c0;\n\t"
            "shl.b32 addr, addr, 2;\n\t"
            "add.u32 addr, addr, %8;\n\t"
            "st.shared.v2.f32 [addr], {%0, %1};\n\t"

            // Pair 1: i=2,3
            "add.u32 r0, r0, 2;\n\t"
            "mad.lo.u32 addr, r0, %9, c0;\n\t"
            "shl.b32 addr, addr, 2;\n\t"
            "add.u32 addr, addr, %8;\n\t"
            "st.shared.v2.f32 [addr], {%2, %3};\n\t"

            // Pair 2: i=4,5
            "sub.u32 r0, r0, 2;\n\t"
            "add.u32 c0, c0, 4;\n\t"
            "mad.lo.u32 addr, r0, %9, c0;\n\t"
            "shl.b32 addr, addr, 2;\n\t"
            "add.u32 addr, addr, %8;\n\t"
            "st.shared.v2.f32 [addr], {%4, %5};\n\t"

            // Pair 3: i=6,7
            "add.u32 r0, r0, 2;\n\t"
            "mad.lo.u32 addr, r0, %9, c0;\n\t"
            "shl.b32 addr, addr, 2;\n\t"
            "add.u32 addr, addr, %8;\n\t"
            "st.shared.v2.f32 [addr], {%6, %7};\n\t"
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
            ".reg .u32 lid, r0, c0, r, c, addr;\n\t"
            "mov.u32 lid, %%laneid;\n\t"
            "and.b32 r0, lid, 1;\n\t"
            ".reg .u32 tmp; and.b32 tmp, lid, 16; shr.b32 tmp, tmp, 2; add.u32 r0, r0, tmp;\n\t"
            "and.b32 c0, lid, 2;\n\t"

            "mov.u32 r, r0; mov.u32 c, c0;\n\t"
            "mad.lo.u32 addr, c, %9, r; shl.b32 addr, addr, 2; add.u32 addr, addr, %8; st.shared.f32 [addr], %0;\n\t"
            "mov.u32 r, r0; add.u32 c, c0, 1;\n\t"
            "mad.lo.u32 addr, c, %9, r; shl.b32 addr, addr, 2; add.u32 addr, addr, %8; st.shared.f32 [addr], %1;\n\t"
            "add.u32 r, r0, 2; mov.u32 c, c0;\n\t"
            "mad.lo.u32 addr, c, %9, r; shl.b32 addr, addr, 2; add.u32 addr, addr, %8; st.shared.f32 [addr], %2;\n\t"
            "add.u32 r, r0, 2; add.u32 c, c0, 1;\n\t"
            "mad.lo.u32 addr, c, %9, r; shl.b32 addr, addr, 2; add.u32 addr, addr, %8; st.shared.f32 [addr], %3;\n\t"
            "mov.u32 r, r0; add.u32 c, c0, 4;\n\t"
            "mad.lo.u32 addr, c, %9, r; shl.b32 addr, addr, 2; add.u32 addr, addr, %8; st.shared.f32 [addr], %4;\n\t"
            "mov.u32 r, r0; add.u32 c, c0, 5;\n\t"
            "mad.lo.u32 addr, c, %9, r; shl.b32 addr, addr, 2; add.u32 addr, addr, %8; st.shared.f32 [addr], %5;\n\t"
            "add.u32 r, r0, 2; add.u32 c, c0, 4;\n\t"
            "mad.lo.u32 addr, c, %9, r; shl.b32 addr, addr, 2; add.u32 addr, addr, %8; st.shared.f32 [addr], %6;\n\t"
            "add.u32 r, r0, 2; add.u32 c, c0, 5;\n\t"
            "mad.lo.u32 addr, c, %9, r; shl.b32 addr, addr, 2; add.u32 addr, addr, %8; st.shared.f32 [addr], %7;\n\t"
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
// MMA_SYNC: m8n8k4 (ALL LAYOUT COMBINATIONS)
// ======================================================================================
// Shape: 8x8x4, FP16 inputs, FP32 accumulation.
// Operands: A/B packed as 2x uint32_t per thread (indices 0 & 2).
// Accumulator: 8x float registers.
// Instruction: mma.sync.aligned.m8n8k4.{row/col}.{row/col}.f32.f16.f16.f32
// Note:        Constraints: %0-%7 D outputs, %8-%9 A, %10-%11 B, %12-%19 C.
// ======================================================================================
#define WMMA_MMA_F32_F32(A_LAYOUT, B_LAYOUT) \
__device__ __forceinline__ void mma_sync( \
    fragment<accumulator, 8, 8, 4, float, row_major>& d, \
    const fragment<matrix_a, 8, 8, 4, half, A_LAYOUT##_major>& a, \
    const fragment<matrix_b, 8, 8, 4, half, B_LAYOUT##_major>& b, \
    const fragment<accumulator, 8, 8, 4, float, row_major>& c) { \
    asm volatile( \
        "mma.sync.aligned.m8n8k4." #A_LAYOUT "." #B_LAYOUT ".f32.f16.f16.f32 " \
        "{%0,%1,%2,%3,%4,%5,%6,%7}, " \
        "{%8,%9}, " \
        "{%10,%11}, " \
        "{%12,%13,%14,%15,%16,%17,%18,%19}; " \
        : "=f"(d.x[0]), "=f"(d.x[1]), "=f"(d.x[2]), "=f"(d.x[3]), \
          "=f"(d.x[4]), "=f"(d.x[5]), "=f"(d.x[6]), "=f"(d.x[7]) \
        : "r"(*reinterpret_cast<const uint32_t*>(&a.x[0])), \
          "r"(*reinterpret_cast<const uint32_t*>(&a.x[2])), \
          "r"(*reinterpret_cast<const uint32_t*>(&b.x[0])), \
          "r"(*reinterpret_cast<const uint32_t*>(&b.x[2])), \
          "f"(c.x[0]), "f"(c.x[1]), "f"(c.x[2]), "f"(c.x[3]), \
          "f"(c.x[4]), "f"(c.x[5]), "f"(c.x[6]), "f"(c.x[7])); \
}

WMMA_MMA_F32_F32(col, col)
WMMA_MMA_F32_F32(row, col)
WMMA_MMA_F32_F32(col, row)
WMMA_MMA_F32_F32(row, row)

#undef WMMA_MMA_F32_F32

} // namespace wmma
} // namespace volta

#endif // FUSED_MMA_M8N8K4_H
