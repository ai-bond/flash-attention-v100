/**
 * Copyright (c) 2025, D.Skryabin / tg @ai_bond007 SPDX-License: BSD-3-Clause
 *
 * Volta-native WMMA (sm_70): facts for fused_mma.h correctness & performance
 *
 * 1. SUPPORTED SHAPES (PTX ISA 6.4 §9.7.13.1, Volta HMMA):
 *    • m16n16k16 — reference shape
 *    • m32n8k16  — tall A (32×16), short B (16×8), accumulator 32×8
 *    • m8n32k16  — short A (8×16), wide B (16×32), accumulator 8×32
 *
 * 2. FRAGMENT LAYOUTS (non-negotiable, Volta QP-based distribution):
 *    — A-fragments (f16):
 *        • m16n16k16: 8 × uint32_t (→ 16 f16)
 *        • m32n8k16:  4 × uint32_t (→  8 f16)  // A = 32×16 → 8 f16/warp
 *        • m8n32k16:  8 × uint32_t (→ 16 f16)  // A = 8×16
 *    — B-fragments (f16):
 *        • m16n16k16: 8 × uint32_t (→ 16 f16)
 *        • m32n8k16:  8 × uint32_t (→ 16 f16)  // B = 16×8 → 16 f16/warp
 *        • m8n32k16:  4 × uint32_t (→  8 f16)  // B = 16×32 → 8 f16/warp
 *    — Accumulator C/D (f32, ALL shapes):
 *        • 8 × float (→ 8 f32/warp, 32×8 → 256 elems / 32 warps)
 *
 * 3. WMMA ≠ mma.sync — it is a *macro-layer* over HMMA.884.F32.F32.
 *    - Every `wmma.mma.sync.*.m{16,32,8}n{16,8,32}k16.f32.f32` compiles to 4 SETs × 4 STEP = 16 HMMA ops/warp.
 *    - F32 accumulation (`.f32.f32`) is native, precise, and mandatory for attention.
 *      — Avoid `.f16.f16`: HMMA.884.F16.F16 has only 2 STEPs/SET and loses precision.
 *
 * 4. wmma.load/store are *mandatory* — no direct memory→TC paths on Volta.
 *    - Use ONLY:
 *        wmma.load.a.sync.aligned.{row/col}.m{16,32,8}n{16,8,32}k16.f16
 *        wmma.load.b.sync.aligned.{row/col}.m{16,32,8}n{16,8,32}k16.f16
 *        wmma.load.c.sync.aligned.{row/col}.m{16,32,8}n{16,8,32}k16.f32
 *        wmma.store.d.sync.aligned.{row/col}.m{16,32,8}n{16,8,32}k16.f32
 *    - *Never* replace with generic ld/st: HMMA expects data pre-packed in registers.
 *
 * 5. Data format: f16x2 MUST be passed as `uint32_t`, *not* `half2`.
 *    - PTX HMMA instructions require physical 32-bit packed lanes.
 *    - `uint32_t[N]` prevents compiler mispacking and guarantees correct reuse-cache layout.
 *
 * 6. Memory alignment & stride (PTX ISA 6.4 §9.7.13.4):
 *    - Shared memory base pointers must be 256-byte aligned (for 128-bit LDS.U.128).
 *    - `ldm` (leading-dimension stride) is in *elements*, not bytes.
 *      — Required minimum ldm (≥):
 *          m16n16k16: A.row=16, A.col=16; B.row=16, B.col=16; C/D.row=16, C/D.col=16
 *          m32n8k16:  A.row=16, A.col=32; B.row=16, B.col=16; C/D.row=8,  C/D.col=32
 *          m8n32k16:  A.row=16, A.col=8;  B.row=32, B.col=16; C/D.row=32, C/D.col=8
 *
 * 7. Thread mapping is fixed by hardware (4 QPs: QP0={0,1,2,3,16,17,18,19}, etc.):
 *    - Each thread in a QP holds exactly:
 *        • m16n16k16: 2×f16x2 (A/B) or 2×f32 (C/D)
 *        • m32n8k16/m8n32k16: scaled accordingly (but accumulator always 2×f32/thread)
 *    - Do *not* manually shuffle data — WMMA ops assume this layout.
 */

#ifndef FUSED_MMA_H
#define FUSED_MMA_H

#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ != 700)
#error "Volta WMMA: This header is for sm_70 ONLY! Compile with -arch=sm_70"
#endif

#include <cuda_fp16.h>

namespace volta {

// ============================================================================
// TYPE TAGS
// ============================================================================

struct row_major {};
struct col_major {};
struct matrix_a {};
struct matrix_b {};
struct accumulator {};

enum layout_t {
    mem_row_major,
    mem_col_major
};

// ============================================================================
// FRAGMENT DECLARATIONS
// ============================================================================

template <typename Use, int M, int N, int K, typename T, typename Layout = void>
struct fragment;

// matrix_a: 8×f16x2 → 16 half
template <> struct fragment<matrix_a, 16, 16, 16, half, row_major> { uint32_t x[8]; static constexpr int num_elements = 16; };
template <> struct fragment<matrix_a, 16, 16, 16, half, col_major> { uint32_t x[8]; static constexpr int num_elements = 16; };
// matrix_a: 8×f16x2 → 16 half (32 rows × 16 K)
template <> struct fragment<matrix_a, 32, 8, 16, half, row_major>  { uint32_t x[8]; static constexpr int num_elements = 16; };
template <> struct fragment<matrix_a, 32, 8, 16, half, col_major>  { uint32_t x[8]; static constexpr int num_elements = 16; };
// matrix_a: 8×f16x2 → 16 half (8 rows × 16 K) - PTX spec: always 8 regs for f16!
template <> struct fragment<matrix_a, 8, 32, 16, half, row_major>  { uint32_t x[8]; static constexpr int num_elements = 16; };
template <> struct fragment<matrix_a, 8, 32, 16, half, col_major>  { uint32_t x[8]; static constexpr int num_elements = 16; };

// matrix_b: 8×f16x2 → 16 half
template <> struct fragment<matrix_b, 16, 16, 16, half, row_major> { uint32_t x[8]; static constexpr int num_elements = 16; };
template <> struct fragment<matrix_b, 16, 16, 16, half, col_major> { uint32_t x[8]; static constexpr int num_elements = 16; };
// matrix_b: 8×f16x2 → 16 half (16 K × 8 cols) - PTX spec: always 8 regs for f16!
template <> struct fragment<matrix_b, 32, 8, 16, half, row_major>  { uint32_t x[8]; static constexpr int num_elements = 16; };
template <> struct fragment<matrix_b, 32, 8, 16, half, col_major>  { uint32_t x[8]; static constexpr int num_elements = 16; };
// matrix_b: 8×f16x2 → 16 half (16 K × 32 cols)
template <> struct fragment<matrix_b, 8, 32, 16, half, row_major>  { uint32_t x[8]; static constexpr int num_elements = 16; };
template <> struct fragment<matrix_b, 8, 32, 16, half, col_major>  { uint32_t x[8]; static constexpr int num_elements = 16; };

// accumulator: 8×f32
template <> struct fragment<accumulator, 16, 16, 16, float> { float x[8]; static constexpr int num_elements = 8; };
template <> struct fragment<accumulator, 32, 8, 16, float>  { float x[8]; static constexpr int num_elements = 8; };
template <> struct fragment<accumulator, 8, 32, 16, float>  { float x[8]; static constexpr int num_elements = 8; };

// ============================================================================
// UTILITY FUNCTIONS
// ============================================================================

__device__ __forceinline__ unsigned get_lane_id() {
    unsigned lane_id;
    asm volatile ("mov.u32 %0, %%laneid;" : "=r"(lane_id));
    return lane_id;
}

// ============================================================================
// fill_fragment — float accumulators only
// ============================================================================

template <int M, int N, int K>
__device__ __forceinline__ void fill_fragment(fragment<accumulator, M, N, K, float>& frag, float value) {
    #pragma unroll
    for (int i = 0; i < 8; ++i) frag.x[i] = value;
}

// ============================================================================
// LOAD FUNCTIONS — wmma.load.* for all geometries
// ============================================================================
__device__ __forceinline__ void load_matrix_sync(
    fragment<matrix_a, 16, 16, 16, half, row_major>& frag,
    const half* smem_ptr, unsigned ldm) {
    asm volatile(
        "wmma.load.a.sync.aligned.row.m16n16k16.f16 "
        "{%0,%1,%2,%3,%4,%5,%6,%7}, [%8], %9;"
        : "=r"(frag.x[0]), "=r"(frag.x[1]), "=r"(frag.x[2]), "=r"(frag.x[3]),
          "=r"(frag.x[4]), "=r"(frag.x[5]), "=r"(frag.x[6]), "=r"(frag.x[7])
        : "l"(smem_ptr), "r"(ldm)
        : "memory"
    );
}

__device__ __forceinline__ void load_matrix_sync(
    fragment<matrix_a, 16, 16, 16, half, col_major>& frag,
    const half* smem_ptr, unsigned ldm) {
    asm volatile(
        "wmma.load.a.sync.aligned.col.m16n16k16.f16 "
        "{%0,%1,%2,%3,%4,%5,%6,%7}, [%8], %9;"
        : "=r"(frag.x[0]), "=r"(frag.x[1]), "=r"(frag.x[2]), "=r"(frag.x[3]),
          "=r"(frag.x[4]), "=r"(frag.x[5]), "=r"(frag.x[6]), "=r"(frag.x[7])
        : "l"(smem_ptr), "r"(ldm)
        : "memory"
    );
}

__device__ __forceinline__ void load_matrix_sync(
    fragment<matrix_b, 16, 16, 16, half, row_major>& frag,
    const half* smem_ptr, unsigned ldm) {
    asm volatile(
        "wmma.load.b.sync.aligned.row.m16n16k16.f16 "
        "{%0,%1,%2,%3,%4,%5,%6,%7}, [%8], %9;"
        : "=r"(frag.x[0]), "=r"(frag.x[1]), "=r"(frag.x[2]), "=r"(frag.x[3]),
          "=r"(frag.x[4]), "=r"(frag.x[5]), "=r"(frag.x[6]), "=r"(frag.x[7])
        : "l"(smem_ptr), "r"(ldm)
        : "memory"
    );
}

__device__ __forceinline__ void load_matrix_sync(
    fragment<matrix_b, 16, 16, 16, half, col_major>& frag,
    const half* smem_ptr, unsigned ldm) {
    asm volatile(
        "wmma.load.b.sync.aligned.col.m16n16k16.f16 "
        "{%0,%1,%2,%3,%4,%5,%6,%7}, [%8], %9;"
        : "=r"(frag.x[0]), "=r"(frag.x[1]), "=r"(frag.x[2]), "=r"(frag.x[3]),
          "=r"(frag.x[4]), "=r"(frag.x[5]), "=r"(frag.x[6]), "=r"(frag.x[7])
        : "l"(smem_ptr), "r"(ldm)
        : "memory"
    );
}

__device__ __forceinline__ void load_matrix_sync(
    fragment<accumulator, 16, 16, 16, float>& frag,
    const float* smem_ptr, unsigned ldm, layout_t layout) {
    if (layout == mem_row_major) {
        asm volatile(
            "wmma.load.c.sync.aligned.row.m16n16k16.f32 "
            "{%0,%1,%2,%3,%4,%5,%6,%7}, [%8], %9;"
            : "=f"(frag.x[0]), "=f"(frag.x[1]), "=f"(frag.x[2]), "=f"(frag.x[3]),
              "=f"(frag.x[4]), "=f"(frag.x[5]), "=f"(frag.x[6]), "=f"(frag.x[7])
            : "l"(smem_ptr), "r"(ldm)
            : "memory"
        );
    } else {
        asm volatile(
            "wmma.load.c.sync.aligned.col.m16n16k16.f32 "
            "{%0,%1,%2,%3,%4,%5,%6,%7}, [%8], %9;"
            : "=f"(frag.x[0]), "=f"(frag.x[1]), "=f"(frag.x[2]), "=f"(frag.x[3]),
              "=f"(frag.x[4]), "=f"(frag.x[5]), "=f"(frag.x[6]), "=f"(frag.x[7])
            : "l"(smem_ptr), "r"(ldm)
            : "memory"
        );
    }
}

__device__ __forceinline__ void load_matrix_sync(
    fragment<matrix_a, 32, 8, 16, half, row_major>& frag,
    const half* smem_ptr, unsigned ldm) {
    asm volatile(
        "wmma.load.a.sync.aligned.row.m32n8k16.f16 "
        "{%0,%1,%2,%3,%4,%5,%6,%7}, [%8], %9;"
        : "=r"(frag.x[0]), "=r"(frag.x[1]), "=r"(frag.x[2]), "=r"(frag.x[3]),
          "=r"(frag.x[4]), "=r"(frag.x[5]), "=r"(frag.x[6]), "=r"(frag.x[7])
        : "l"(smem_ptr), "r"(ldm)
        : "memory"
    );
}

__device__ __forceinline__ void load_matrix_sync(
    fragment<matrix_a, 32, 8, 16, half, col_major>& frag,
    const half* smem_ptr, unsigned ldm) {
    asm volatile(
        "wmma.load.a.sync.aligned.col.m32n8k16.f16 "
        "{%0,%1,%2,%3,%4,%5,%6,%7}, [%8], %9;"
        : "=r"(frag.x[0]), "=r"(frag.x[1]), "=r"(frag.x[2]), "=r"(frag.x[3]),
          "=r"(frag.x[4]), "=r"(frag.x[5]), "=r"(frag.x[6]), "=r"(frag.x[7])
        : "l"(smem_ptr), "r"(ldm)
        : "memory"
    );
}

__device__ __forceinline__ void load_matrix_sync(
    fragment<matrix_b, 32, 8, 16, half, row_major>& frag,
    const half* smem_ptr, unsigned ldm) {
    asm volatile(
        "wmma.load.b.sync.aligned.row.m32n8k16.f16 "
        "{%0,%1,%2,%3,%4,%5,%6,%7}, [%8], %9;"
        : "=r"(frag.x[0]), "=r"(frag.x[1]), "=r"(frag.x[2]), "=r"(frag.x[3]),
          "=r"(frag.x[4]), "=r"(frag.x[5]), "=r"(frag.x[6]), "=r"(frag.x[7])
        : "l"(smem_ptr), "r"(ldm)
        : "memory"
    );
}

__device__ __forceinline__ void load_matrix_sync(
    fragment<matrix_b, 32, 8, 16, half, col_major>& frag,
    const half* smem_ptr, unsigned ldm) {
    asm volatile(
        "wmma.load.b.sync.aligned.col.m32n8k16.f16 "
        "{%0,%1,%2,%3,%4,%5,%6,%7}, [%8], %9;"
        : "=r"(frag.x[0]), "=r"(frag.x[1]), "=r"(frag.x[2]), "=r"(frag.x[3]),
          "=r"(frag.x[4]), "=r"(frag.x[5]), "=r"(frag.x[6]), "=r"(frag.x[7])
        : "l"(smem_ptr), "r"(ldm)
        : "memory"
    );
}

__device__ __forceinline__ void load_matrix_sync(
    fragment<accumulator, 32, 8, 16, float>& frag,
    const float* smem_ptr, unsigned ldm, layout_t layout) {
    if (layout == mem_row_major) {
        asm volatile(
            "wmma.load.c.sync.aligned.row.m32n8k16.f32 "
            "{%0,%1,%2,%3,%4,%5,%6,%7}, [%8], %9;"
            : "=f"(frag.x[0]), "=f"(frag.x[1]), "=f"(frag.x[2]), "=f"(frag.x[3]),
              "=f"(frag.x[4]), "=f"(frag.x[5]), "=f"(frag.x[6]), "=f"(frag.x[7])
            : "l"(smem_ptr), "r"(ldm)
            : "memory"
        );
    } else {
        asm volatile(
            "wmma.load.c.sync.aligned.col.m32n8k16.f32 "
            "{%0,%1,%2,%3,%4,%5,%6,%7}, [%8], %9;"
            : "=f"(frag.x[0]), "=f"(frag.x[1]), "=f"(frag.x[2]), "=f"(frag.x[3]),
              "=f"(frag.x[4]), "=f"(frag.x[5]), "=f"(frag.x[6]), "=f"(frag.x[7])
            : "l"(smem_ptr), "r"(ldm)
            : "memory"
        );
    }
}

__device__ __forceinline__ void load_matrix_sync(
    fragment<matrix_a, 8, 32, 16, half, row_major>& frag,
    const half* smem_ptr, unsigned ldm) {
    asm volatile(
        "wmma.load.a.sync.aligned.row.m8n32k16.f16 "
        "{%0,%1,%2,%3,%4,%5,%6,%7}, [%8], %9;"
        : "=r"(frag.x[0]), "=r"(frag.x[1]), "=r"(frag.x[2]), "=r"(frag.x[3]),
          "=r"(frag.x[4]), "=r"(frag.x[5]), "=r"(frag.x[6]), "=r"(frag.x[7])
        : "l"(smem_ptr), "r"(ldm)
        : "memory"
    );
}

__device__ __forceinline__ void load_matrix_sync(
    fragment<matrix_a, 8, 32, 16, half, col_major>& frag,
    const half* smem_ptr, unsigned ldm) {
    asm volatile(
        "wmma.load.a.sync.aligned.col.m8n32k16.f16 "
        "{%0,%1,%2,%3,%4,%5,%6,%7}, [%8], %9;"
        : "=r"(frag.x[0]), "=r"(frag.x[1]), "=r"(frag.x[2]), "=r"(frag.x[3]),
          "=r"(frag.x[4]), "=r"(frag.x[5]), "=r"(frag.x[6]), "=r"(frag.x[7])
        : "l"(smem_ptr), "r"(ldm)
        : "memory"
    );
}

__device__ __forceinline__ void load_matrix_sync(
    fragment<matrix_b, 8, 32, 16, half, row_major>& frag,
    const half* smem_ptr, unsigned ldm) {
    asm volatile(
        "wmma.load.b.sync.aligned.row.m8n32k16.f16 "
        "{%0,%1,%2,%3,%4,%5,%6,%7}, [%8], %9;"
        : "=r"(frag.x[0]), "=r"(frag.x[1]), "=r"(frag.x[2]), "=r"(frag.x[3]),
          "=r"(frag.x[4]), "=r"(frag.x[5]), "=r"(frag.x[6]), "=r"(frag.x[7])
        : "l"(smem_ptr), "r"(ldm)
        : "memory"
    );
}

__device__ __forceinline__ void load_matrix_sync(
    fragment<matrix_b, 8, 32, 16, half, col_major>& frag,
    const half* smem_ptr, unsigned ldm) {
    asm volatile(
        "wmma.load.b.sync.aligned.col.m8n32k16.f16 "
        "{%0,%1,%2,%3,%4,%5,%6,%7}, [%8], %9;"
        : "=r"(frag.x[0]), "=r"(frag.x[1]), "=r"(frag.x[2]), "=r"(frag.x[3]),
          "=r"(frag.x[4]), "=r"(frag.x[5]), "=r"(frag.x[6]), "=r"(frag.x[7])
        : "l"(smem_ptr), "r"(ldm)
        : "memory"
    );
}

__device__ __forceinline__ void load_matrix_sync(
    fragment<accumulator, 8, 32, 16, float>& frag,
    const float* smem_ptr, unsigned ldm, layout_t layout) {
    if (layout == mem_row_major) {
        asm volatile(
            "wmma.load.c.sync.aligned.row.m8n32k16.f32 "
            "{%0,%1,%2,%3,%4,%5,%6,%7}, [%8], %9;"
            : "=f"(frag.x[0]), "=f"(frag.x[1]), "=f"(frag.x[2]), "=f"(frag.x[3]),
              "=f"(frag.x[4]), "=f"(frag.x[5]), "=f"(frag.x[6]), "=f"(frag.x[7])
            : "l"(smem_ptr), "r"(ldm)
            : "memory"
        );
    } else {
        asm volatile(
            "wmma.load.c.sync.aligned.col.m8n32k16.f32 "
            "{%0,%1,%2,%3,%4,%5,%6,%7}, [%8], %9;"
            : "=f"(frag.x[0]), "=f"(frag.x[1]), "=f"(frag.x[2]), "=f"(frag.x[3]),
              "=f"(frag.x[4]), "=f"(frag.x[5]), "=f"(frag.x[6]), "=f"(frag.x[7])
            : "l"(smem_ptr), "r"(ldm)
            : "memory"
        );
    }
}

// ============================================================================
// STORE FUNCTIONS — wmma.store.d.* for all geometries
// ============================================================================

__device__ __forceinline__ void store_matrix_sync(
    float* smem_ptr,
    const fragment<accumulator, 16, 16, 16, float>& frag,
    unsigned ldm, layout_t layout) {
    if (layout == mem_row_major) {
        asm volatile(
            "wmma.store.d.sync.aligned.row.m16n16k16.f32 "
            "[%0], {%1,%2,%3,%4,%5,%6,%7,%8}, %9;"
            :
            : "l"(smem_ptr),
              "f"(frag.x[0]), "f"(frag.x[1]), "f"(frag.x[2]), "f"(frag.x[3]),
              "f"(frag.x[4]), "f"(frag.x[5]), "f"(frag.x[6]), "f"(frag.x[7]),
              "r"(ldm)
            : "memory"
        );
    } else {
        asm volatile(
            "wmma.store.d.sync.aligned.col.m16n16k16.f32 "
            "[%0], {%1,%2,%3,%4,%5,%6,%7,%8}, %9;"
            :
            : "l"(smem_ptr),
              "f"(frag.x[0]), "f"(frag.x[1]), "f"(frag.x[2]), "f"(frag.x[3]),
              "f"(frag.x[4]), "f"(frag.x[5]), "f"(frag.x[6]), "f"(frag.x[7]),
              "r"(ldm)
            : "memory"
        );
    }
}

__device__ __forceinline__ void store_matrix_sync(
    float* smem_ptr,
    const fragment<accumulator, 32, 8, 16, float>& frag,
    unsigned ldm, layout_t layout) {
    if (layout == mem_row_major) {
        asm volatile(
            "wmma.store.d.sync.aligned.row.m32n8k16.f32 "
            "[%0], {%1,%2,%3,%4,%5,%6,%7,%8}, %9;"
            :
            : "l"(smem_ptr),
              "f"(frag.x[0]), "f"(frag.x[1]), "f"(frag.x[2]), "f"(frag.x[3]),
              "f"(frag.x[4]), "f"(frag.x[5]), "f"(frag.x[6]), "f"(frag.x[7]),
              "r"(ldm)
            : "memory"
        );
    } else {
        asm volatile(
            "wmma.store.d.sync.aligned.col.m32n8k16.f32 "
            "[%0], {%1,%2,%3,%4,%5,%6,%7,%8}, %9;"
            :
            : "l"(smem_ptr),
              "f"(frag.x[0]), "f"(frag.x[1]), "f"(frag.x[2]), "f"(frag.x[3]),
              "f"(frag.x[4]), "f"(frag.x[5]), "f"(frag.x[6]), "f"(frag.x[7]),
              "r"(ldm)
            : "memory"
        );
    }
}

__device__ __forceinline__ void store_matrix_sync(
    float* smem_ptr,
    const fragment<accumulator, 8, 32, 16, float>& frag,
    unsigned ldm, layout_t layout) {
    if (layout == mem_row_major) {
        asm volatile(
            "wmma.store.d.sync.aligned.row.m8n32k16.f32 "
            "[%0], {%1,%2,%3,%4,%5,%6,%7,%8}, %9;"
            :
            : "l"(smem_ptr),
              "f"(frag.x[0]), "f"(frag.x[1]), "f"(frag.x[2]), "f"(frag.x[3]),
              "f"(frag.x[4]), "f"(frag.x[5]), "f"(frag.x[6]), "f"(frag.x[7]),
              "r"(ldm)
            : "memory"
        );
    } else {
        asm volatile(
            "wmma.store.d.sync.aligned.col.m8n32k16.f32 "
            "[%0], {%1,%2,%3,%4,%5,%6,%7,%8}, %9;"
            :
            : "l"(smem_ptr),
              "f"(frag.x[0]), "f"(frag.x[1]), "f"(frag.x[2]), "f"(frag.x[3]),
              "f"(frag.x[4]), "f"(frag.x[5]), "f"(frag.x[6]), "f"(frag.x[7]),
              "r"(ldm)
            : "memory"
        );
    }
}

// ============================================================================
// MMA_SYNC — wmma.mma.sync for all geometries
// ============================================================================

// Universal macro for ALL f16 geometries (A=8 regs, B=8 regs per PTX spec)
#define VOLTA_WMMA_MMA_F32(M, N, K, ALAY, BLAY) \
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

VOLTA_WMMA_MMA_F32(16, 16, 16, row, col)
VOLTA_WMMA_MMA_F32(16, 16, 16, row, row)
VOLTA_WMMA_MMA_F32(16, 16, 16, col, col)
VOLTA_WMMA_MMA_F32(16, 16, 16, col, row)

VOLTA_WMMA_MMA_F32(32, 8, 16, row, col)
VOLTA_WMMA_MMA_F32(32, 8, 16, row, row)
VOLTA_WMMA_MMA_F32(32, 8, 16, col, col)
VOLTA_WMMA_MMA_F32(32, 8, 16, col, row)

VOLTA_WMMA_MMA_F32(8, 32, 16, row, col)
VOLTA_WMMA_MMA_F32(8, 32, 16, row, row)
VOLTA_WMMA_MMA_F32(8, 32, 16, col, col)
VOLTA_WMMA_MMA_F32(8, 32, 16, col, row)

#undef VOLTA_WMMA_MMA_F32

} // namespace volta

#endif // FUSED_MMA_H