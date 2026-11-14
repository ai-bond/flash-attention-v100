// ======================================================================================
// * Copyright (c) 2025, D.Skryabin / tg @ai_bond007 SPDX-License: BSD-3-Clause
// ======================================================================================
// * Volta HMMA m8n8k4 NATIVE reference wrapper (NO swizzle, plain indexing).
// *
// * 1. PURPOSE
// *    Ground-truth, plain-indexing implementation of m8n8k4 used to validate the
// *    swizzle-aware wrapper (mma_m8n8k4.h). Same public API, same fragment
// *    register layout, same mma_sync PTX — only smem indexing differs:
// *      mma_native.h  -> linear smem (reference)
// *      mma_m8n8k4.h  -> swizzled smem (optimized)
// *    Both headers must be MUTUALLY EXCLUSIVE in one translation unit.
// *
// * 2. SUPPORTED SHAPE (PTX ISA 6.4, mma.sync.aligned.m8n8k4)
// *    m8n8k4 : compact 8x8x4 configuration optimized for small tile reductions.
// *
// * 3. FRAGMENT REGISTER LAYOUT
// *    A/B fragments store 4 half values per thread (packed as 2x uint32_t).
// *    C/D accumulators store 8 float values per thread.
// *
// * 4. DATA DISTRIBUTION (identical to wmma_extension/m8n8k4.hpp)
// *    A row_major : lane -> row       = (lane & 3) + ((lane >> 4) << 2)
// *    A col_major : lane -> col       = (lane & 3), row_offset = (lane >> 4) << 2
// *    B row_major : lane -> row       = (lane & 3), col_offset = (lane >> 4) << 2
// *    B col_major : lane -> col       = (lane & 3) + ((lane >> 4) << 2)
// *    C/D         : lane -> (r0, c0)  = ((lane & 1) + ((lane >> 4) << 2), lane & 2)
// *
// * 5. LOAD / STORE
// *    wmma.load.c / wmma.store.d are not available for m8n8k4 on sm_70.
// *    A/B loads are plain shared-memory reads at logical indices.
// *    Accumulator load/store use explicit lane-to-memory mapping.
// *
// * 6. MMA PTX
// *    mma.sync.aligned.m8n8k4.{row,col}.{row,col}.f32.f16.f16.f32
// *    A/B operands passed as 32-bit packed registers (a.x[0..1], b.x[0..1]).
// *
// * 7. THIS IS THE "NATIVE" REFERENCE — do NOT add swizzle here.
// ======================================================================================

#ifndef FUSED_MMA_NATIVE_M8N8K4_H
#define FUSED_MMA_NATIVE_M8N8K4_H

#pragma once

#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ != 700)
#error "Volta HMMA m8n8k4 native: This header is for sm_70 ONLY! Compile with -arch=sm_70"
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

template <> struct fragment<matrix_a,    8, 8, 4, half, row_major> : __frag_base<half, 4> {};
template <> struct fragment<matrix_a,    8, 8, 4, half, col_major> : __frag_base<half, 4> {};
template <> struct fragment<matrix_b,    8, 8, 4, half, row_major> : __frag_base<half, 4> {};
template <> struct fragment<matrix_b,    8, 8, 4, half, col_major> : __frag_base<half, 4> {};
template <> struct fragment<accumulator, 8, 8, 4, float, row_major> : __frag_base<float, 8> {};
template <> struct fragment<accumulator, 8, 8, 4, float, col_major> : __frag_base<float, 8> {};

// Catch-all: void layout defaults to row_major (canonical fragment layout).
template <typename Use, int M, int N, int K, typename T>
struct fragment<Use, M, N, K, T, void> : fragment<Use, M, N, K, T, row_major> {};

// ======================================================================================
// LANE ID
// ======================================================================================
__device__ __forceinline__ unsigned get_lane_id() {
    unsigned lane_id;
    asm volatile("mov.u32 %0, %%laneid;" : "=r"(lane_id));
    return lane_id;
}

// ======================================================================================
// FILL: accumulator m8n8k4
// ======================================================================================
// Data per lane:  8 floats (full accumulator fragment).
// Lane mapping:   uniform broadcast across all 32 lanes.
// Replication:    N/A (scalar initialization).
// ======================================================================================
template <int M, int N, int K, typename Layout>
__device__ __forceinline__ void fill_fragment(
    fragment<accumulator, M, N, K, float, Layout>& frag, float value) {
#pragma unroll
    for (int i = 0; i < 8; ++i) frag.x[i] = value;
}

// ======================================================================================
// LOAD: matrix_a m8n8k4 (ROW MAJOR)
// ======================================================================================
// Data per lane:  4 contiguous halves.
// Lane mapping:   row = (lane & 3) + ((lane >> 4) << 2)
// Memory access:  smem[row * ldm + 0..3] — 4 halves starting at row's ldm offset.
// Note:           ldm = K (tile_k) for A row-major.
// ======================================================================================
__device__ __forceinline__ void load_matrix_sync(
    fragment<matrix_a, 8, 8, 4, half, row_major>& frag,
    const half* __restrict__ smem_ptr, unsigned ldm) {
    const unsigned lane_id    = get_lane_id();
    const unsigned row        = (lane_id & 0x3) + ((lane_id >> 4) << 2);
    const unsigned mem_offset = row * ldm;
#pragma unroll
    for (int i = 0; i < 4; ++i) frag.x[i] = smem_ptr[mem_offset + i];
}

// ======================================================================================
// LOAD: matrix_a m8n8k4 (COL MAJOR)
// ======================================================================================
// Data per lane:  4 contiguous halves.
// Lane mapping:   col = lane & 3,  row_offset = (lane >> 4) << 2
// Memory access:  smem[col * ldm + row_offset + 0..3].
// Note:           ldm = M (tile_m) for A col-major.
// ======================================================================================
__device__ __forceinline__ void load_matrix_sync(
    fragment<matrix_a, 8, 8, 4, half, col_major>& frag,
    const half* __restrict__ smem_ptr, unsigned ldm) {
    const unsigned lane_id    = get_lane_id();
    const unsigned col        = lane_id & 0x3;
    const unsigned row_offset = (lane_id >> 4) << 2;
    const unsigned mem_offset = col * ldm + row_offset;
#pragma unroll
    for (int i = 0; i < 4; ++i) frag.x[i] = smem_ptr[mem_offset + i];
}

// ======================================================================================
// LOAD: matrix_b m8n8k4 (ROW MAJOR)
// ======================================================================================
// Data per lane:  4 contiguous halves.
// Lane mapping:   row = lane & 3,  col_offset = (lane >> 4) << 2
// Memory access:  smem[row * ldm + col_offset + 0..3].
// Note:           ldm = N (tile_n) for B row-major.
// ======================================================================================
__device__ __forceinline__ void load_matrix_sync(
    fragment<matrix_b, 8, 8, 4, half, row_major>& frag,
    const half* __restrict__ smem_ptr, unsigned ldm) {
    const unsigned lane_id    = get_lane_id();
    const unsigned row        = lane_id & 0x3;
    const unsigned col_offset = (lane_id >> 4) << 2;
    const unsigned mem_offset = row * ldm + col_offset;
#pragma unroll
    for (int i = 0; i < 4; ++i) frag.x[i] = smem_ptr[mem_offset + i];
}

// ======================================================================================
// LOAD: matrix_b m8n8k4 (COL MAJOR)
// ======================================================================================
// Data per lane:  4 contiguous halves.
// Lane mapping:   col = (lane & 3) + ((lane >> 4) << 2)
// Memory access:  smem[col * ldm + 0..3].
// Note:           ldm = K (tile_k) for B col-major.
// ======================================================================================
__device__ __forceinline__ void load_matrix_sync(
    fragment<matrix_b, 8, 8, 4, half, col_major>& frag,
    const half* __restrict__ smem_ptr, unsigned ldm) {
    const unsigned lane_id    = get_lane_id();
    const unsigned col        = (lane_id & 0x3) + ((lane_id >> 4) << 2);
    const unsigned mem_offset = col * ldm;
#pragma unroll
    for (int i = 0; i < 4; ++i) frag.x[i] = smem_ptr[mem_offset + i];
}

// ======================================================================================
// LOAD: accumulator m8n8k4 (ROW & COL MAJOR)
// ======================================================================================
// Data per lane:  8 floats = 4 logical pairs.
// Pair order:     (r0, c0), (r0+2, c0), (r0, c0+4), (r0+2, c0+4)
// Lane mapping:   r0 = (lane & 1) + ((lane >> 4) << 2),  c0 = lane & 2
// Memory access:  ROW -> smem[row * ldm + col]
//                 COL -> smem[row + col * ldm]
// Note:           ldm = N (tile_n) for C row-major, ldm = M (tile_m) for C col-major.
// ======================================================================================
__device__ __forceinline__ void load_matrix_sync(
    fragment<accumulator, 8, 8, 4, float, row_major>& frag,
    const float* __restrict__ smem_ptr, unsigned ldm, layout_t layout) {
    const unsigned lane_id    = get_lane_id();
    const unsigned row_offset = (lane_id & 0x1) + ((lane_id & 0x10) >> 2);
    const unsigned col_offset = (lane_id & 0x2);
    if (layout == mem_col_major) {
#pragma unroll
        for (int i = 0; i < 8; ++i) {
            const unsigned row = row_offset + (i & 0x2);
            const unsigned col = col_offset + ((i & 0x1) + (i & 0x4));
            frag.x[i] = smem_ptr[row + col * ldm];
        }
    } else {
#pragma unroll
        for (int i = 0; i < 8; ++i) {
            const unsigned row = row_offset + (i & 0x2);
            const unsigned col = col_offset + ((i & 0x1) + (i & 0x4));
            frag.x[i] = smem_ptr[row * ldm + col];
        }
    }
}

// ======================================================================================
// STORE: accumulator m8n8k4 (ROW & COL MAJOR)
// ======================================================================================
// Data per lane:  8 floats = 4 logical pairs (exact inverse of load mapping).
// Lane mapping:   r0 = (lane & 1) + ((lane >> 4) << 2),  c0 = lane & 2
// Note:           ldm = N for C row-major, ldm = M for C col-major.
// ======================================================================================
__device__ __forceinline__ void store_matrix_sync(
    float* __restrict__ smem_ptr,
    const fragment<accumulator, 8, 8, 4, float, row_major>& frag,
    unsigned ldm, layout_t layout) {
    const unsigned lane_id    = get_lane_id();
    const unsigned row_offset = (lane_id & 0x1) + ((lane_id & 0x10) >> 2);
    const unsigned col_offset = (lane_id & 0x2);
    if (layout == mem_col_major) {
#pragma unroll
        for (int i = 0; i < 8; ++i) {
            const unsigned row = row_offset + (i & 0x2);
            const unsigned col = col_offset + ((i & 0x1) + (i & 0x4));
            smem_ptr[row + col * ldm] = frag.x[i];
        }
    } else {
#pragma unroll
        for (int i = 0; i < 8; ++i) {
            const unsigned row = row_offset + (i & 0x2);
            const unsigned col = col_offset + ((i & 0x1) + (i & 0x4));
            smem_ptr[row * ldm + col] = frag.x[i];
        }
    }
}

// ======================================================================================
// MMA_SYNC: m8n8k4 (ALL LAYOUT COMBINATIONS)
// ======================================================================================
// Shape:       8x8x4, FP16 inputs, FP32 accumulation.
// Operands:    A/B packed as 2x uint32_t per thread (indices 0 & 2).
// Accumulator: 8x float registers.
// Instruction: mma.sync.aligned.m8n8k4.{row|col}.{row|col}.f32.f16.f16.f32
// Note:        1:1 with WMMAE_MMA884_F32_F32 from wmma_extension/m8n8k4.hpp.
// ======================================================================================
#define VOLTA_MMA_M8N8K4_F32_F32(ALAY, BLAY)                                          \
__device__ __forceinline__ void mma_sync(                                             \
    fragment<accumulator, 8, 8, 4, float, row_major>& d,                              \
    const fragment<matrix_a, 8, 8, 4, half, ALAY##_major>& a,                         \
    const fragment<matrix_b, 8, 8, 4, half, BLAY##_major>& b,                         \
    const fragment<accumulator, 8, 8, 4, float, row_major>& c) {                      \
    asm volatile(                                                                     \
        "mma.sync.aligned.m8n8k4." #ALAY "." #BLAY ".f32.f16.f16.f32 "                \
        "{%0,%1,%2,%3,%4,%5,%6,%7}, "                                                 \
        "{%8,%9}, "                                                                   \
        "{%10,%11}, "                                                                 \
        "{%12,%13,%14,%15,%16,%17,%18,%19}; "                                         \
        : "=f"(d.x[0]), "=f"(d.x[1]), "=f"(d.x[2]), "=f"(d.x[3]),                     \
          "=f"(d.x[4]), "=f"(d.x[5]), "=f"(d.x[6]), "=f"(d.x[7])                      \
        : "r"(*reinterpret_cast<const uint32_t*>(&a.x[0])),                           \
          "r"(*reinterpret_cast<const uint32_t*>(&a.x[2])),                           \
          "r"(*reinterpret_cast<const uint32_t*>(&b.x[0])),                           \
          "r"(*reinterpret_cast<const uint32_t*>(&b.x[2])),                           \
          "f"(c.x[0]), "f"(c.x[1]), "f"(c.x[2]), "f"(c.x[3]),                         \
          "f"(c.x[4]), "f"(c.x[5]), "f"(c.x[6]), "f"(c.x[7]));                        \
}

VOLTA_MMA_M8N8K4_F32_F32(col, col)
VOLTA_MMA_M8N8K4_F32_F32(col, row)
VOLTA_MMA_M8N8K4_F32_F32(row, col)
VOLTA_MMA_M8N8K4_F32_F32(row, row)

#undef VOLTA_MMA_M8N8K4_F32_F32

}  // namespace wmma
}  // namespace volta

#endif  // FUSED_MMA_NATIVE_M8N8K4_H
