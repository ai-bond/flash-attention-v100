// ======================================================================================
// * Copyright (c) 2026, D.Skryabin / tg @ai_bond007 SPDX-License: BSD-3-Clause
// ======================================================================================
#pragma once

#include <cuda.h>
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include "swizzle.h"
#include "template.h"

#ifdef MMA_NATIVE
    #include <mma.h>
#elif defined(MMA_884)
    #include "mma_m8n8k4.h"
#else
    #include "mma_m16n16k16.h"
#endif

// ======================================================================================
// SMEM CORE HELPER
// ======================================================================================
template<typename T>
__device__ __forceinline__ uint32_t GET_PATTERN(T value) {
    if constexpr (std::is_same_v<T, float>) {
        return (value == 0.0f) ? 0u : __float_as_uint(value);
    } else if constexpr (std::is_same_v<T, __half>) {
        uint32_t h = __half_as_ushort(value);
        return (h == 0u) ? 0u : ((h << 16) | h);
    } else if constexpr (std::is_same_v<T, uint32_t> || std::is_same_v<T, int>) {
        return static_cast<uint32_t>(value);
    } else {
        static_assert(sizeof(T) == 0, "SMEM: Unsupported type. Use float, __half, or uint32_t.");
        return 0;
    }
}

// ======================================================================================
// INIT SMEM LAYOUT
// ======================================================================================
template<typename Config, typename T, size_t TOTAL_BYTES>
__device__ __forceinline__ void INIT_SMEM_CORE(char* smem_ptr, T value) {
    constexpr int CHUNKS = (TOTAL_BYTES + 15) / 16;
    constexpr int STRIDE = Config::THREADS_PER_BLOCK;
    constexpr int ITERS  = (CHUNKS + STRIDE - 1) / STRIDE;

    uint32_t pattern = GET_PATTERN(value);
    uint32_t base    = static_cast<uint32_t>(__cvta_generic_to_shared(smem_ptr));

    uint32_t str_ptr = base + ((threadIdx.x ^ (threadIdx.x >> 3)) << 4);
    uint32_t end_ptr = base + (CHUNKS << 4);
    constexpr uint32_t step = STRIDE << 4;

    #pragma unroll
    for (int i = 0; i < ITERS; ++i) {
        asm volatile(
            "setp.lt.u32 %%p0, %0, %1;\n\t"
            "@%%p0 st.shared.v4.u32 [%0], {%2, %2, %2, %2};\n\t"
            :: "r"(str_ptr), "r"(end_ptr), "r"(pattern) : "memory"
        );
        str_ptr += step;
    }
}

// ======================================================================================
// WMMA_GEMM_INIT_SMEM OVERLOADS
// ======================================================================================
template<typename Config>
__device__ __forceinline__ void WMMA_GEMM_INIT_SMEM(char* smem_raw) {
    INIT_SMEM_CORE<Config, uint32_t, Config::TOTAL_SMEM>(smem_raw, 0u);
}

template<typename Config, typename T, int N>
__device__ __forceinline__ void WMMA_GEMM_INIT_SMEM(T (&arr)[N], T VALUE) {
    constexpr size_t TOTAL_BYTES = N * sizeof(T);
    INIT_SMEM_CORE<Config, T, TOTAL_BYTES>(reinterpret_cast<char*>(arr), VALUE);
}

// ======================================================================================
// WMMA_GEMM_INIT_REGS: Register fragment array
// ======================================================================================
template<typename Config, GemmType TYPE, GemmPhase PHASE, bool SCALARS = false>
struct WMMA_GEMM_INIT_REGS {

    using GTraits = GemmTraits<TYPE, PHASE>;
    using PTraits = PhaseTraits<PHASE>;

    static constexpr bool IS_SCORES  = GTraits::B_IS_COL; 
    static constexpr int TILE_M      = IS_SCORES ? PTraits::template BLOCK_X<Config>() : (PTraits::IS_SWAPPED ? PTraits::template BLOCK_Y<Config>() : PTraits::template BLOCK_X<Config>());
    static constexpr int TILE_N      = IS_SCORES ? PTraits::template BLOCK_Y<Config>() : Config::DIM;
    static constexpr int num_tiles_m = (TILE_M + WMMA_M - 1) / WMMA_M;
    static constexpr int num_tiles_n = (TILE_N + WMMA_N - 1) / WMMA_N;
    static constexpr int total_mn    = num_tiles_m * num_tiles_n;
    static constexpr int tiles       = SCALARS ? (num_tiles_m / Config::WARPS_PER_BLOCK) * 2 : (total_mn + Config::WARPS_PER_BLOCK - 1) / Config::WARPS_PER_BLOCK;

#ifdef MMA_NATIVE
    using frag_t = nvcuda::wmma::fragment<nvcuda::wmma::accumulator, WMMA_M, WMMA_N, WMMA_K, float>;
#else
    using frag_t = volta::wmma::fragment<volta::wmma::accumulator, WMMA_M, WMMA_N, WMMA_K, float>;
#endif

    std::conditional_t<SCALARS, float, frag_t> frags[tiles];

    __device__ __forceinline__ WMMA_GEMM_INIT_REGS(float fill = 0.0f) {
        if constexpr (SCALARS) {
            #pragma unroll
            for (int i = 0; i < tiles; ++i) frags[i] = fill;
        } else if constexpr (!IS_SCORES) {
#ifdef MMA_NATIVE
            using namespace nvcuda::wmma;
#else
            using namespace volta::wmma;
#endif
            #pragma unroll
            for (int i = 0; i < tiles; ++i) fill_fragment(frags[i], fill);
        }
    }
};

// ======================================================================================
// TILE LOADER: Global-shared load for unified dense/varlen/kvcache layouts
// ======================================================================================
template<typename Config, GemmPhase Phase, Stage Option>
__device__ __forceinline__ void WMMA_GEMM_LOAD_TILE(
    const __half* __restrict__ GMEM,
          __half* __restrict__ SMEM,
    int GLOBAL_STRIDE,
    int VALID_ROWS,
    int THREAD_ID
) {
    using LTraits = LoadTraits<Phase, Option>;

    constexpr int THREADS_PER_BLOCK = Config::THREADS_PER_BLOCK;
    constexpr int dst_stride_uint4  = (LTraits::template TRANSPOSE<Config>() + 7) >> 3;
    const     int src_stride_uint4  = (GLOBAL_STRIDE + 7) >> 3;
    const     int total_iters       = VALID_ROWS * dst_stride_uint4;

    if (total_iters == 0) return;

    uint64_t src_base = static_cast<uint64_t>(__cvta_generic_to_global(GMEM));
    uint32_t dst_base = static_cast<uint32_t>(__cvta_generic_to_shared(SMEM));

    #pragma unroll 2
    for (int idx = THREAD_ID; idx < total_iters; idx += THREADS_PER_BLOCK) {
        const int row = idx / dst_stride_uint4;
        const int col = idx % dst_stride_uint4;

        const int src_offset = row * src_stride_uint4 + col;
        const int dst_offset = row * dst_stride_uint4 + col;
        const int pred       = (col < src_stride_uint4) ? 1 : 0;

        uint64_t src_addr = src_base + (static_cast<uint64_t>(src_offset) << 4);
        uint32_t dst_addr = swizzle(dst_base + (static_cast<uint32_t>(dst_offset) << 4), row);

        uint32_t r0, r1, r2, r3;

        asm volatile(
            "{\n\t"
            ".reg .pred p;\n\t"
            "setp.ne.b32 p, %6, 0;\n\t"
            "mov.u32 %0, 0;\n\t"
            "mov.u32 %1, 0;\n\t"
            "mov.u32 %2, 0;\n\t"
            "mov.u32 %3, 0;\n\t"
            "@p ld.global.v4.u32 {%0, %1, %2, %3}, [%4];\n\t"
            "st.shared.v4.u32 [%5], {%0, %1, %2, %3};\n\t"
            "}\n\t"
            : "=r"(r0), "=r"(r1), "=r"(r2), "=r"(r3)
            :  "l"(src_addr), "r"(dst_addr), "r"(pred)
            : "memory"
        );
    }
}

// ======================================================================================
// WMMA_GEMM_EPILOGUE: Register-based flush to gmem
// ======================================================================================
//   WRITE_O:   single store, normalize by ROW_SUM + lse
//   WRITE_DQ:  single store, no normalization
//   WRITE_DKV: dual   store, no normalization
// ======================================================================================
template<typename Config, Result TYPE, typename Fragment>
__device__ __forceinline__ void WMMA_GEMM_EPILOGUE(
       Fragment* __restrict__ REGS0,
         __half* __restrict__ GMEM0,
     const void* __restrict__ REGS1,
         __half* __restrict__ GMEM1,
          float* __restrict__ GMEM_LSE,
    const float*              ROW_SUM,
    const float*              ROW_MAX,
    int STRIDE,  int VALID_M,
    int WARP_ID, int LANE_ID
) {
    using ETraits = EpilogueTraits<TYPE>;
    using PTraits = PhaseTraits<TYPE == Result::WRITE_O ? GemmPhase::FWD : TYPE == Result::WRITE_DQ ? GemmPhase::BWQ : GemmPhase::BWKV>;

    constexpr bool NORMALIZE  = ETraits::NORMALIZE;
    constexpr bool DUAL_STORE = ETraits::DUAL_STORE;
    constexpr bool WRITE_LSE  = ETraits::WRITE_LSE;

    constexpr int D            = Config::DIM;
    constexpr int TILE_M       = PTraits::IS_SWAPPED ? PTraits::template BLOCK_Y<Config>() : PTraits::template BLOCK_X<Config>();
    constexpr int num_tiles_m  = (TILE_M + WMMA_M - 1) / WMMA_M;
    constexpr int num_tiles_n  = (D      + WMMA_N - 1) / WMMA_N;
    constexpr int total_tiles  = num_tiles_m * num_tiles_n;
    constexpr int TILES        = (total_tiles + Config::WARPS_PER_BLOCK - 1) / Config::WARPS_PER_BLOCK;

    const uint2 lines     = LINE_CHORDS(static_cast<unsigned>(LANE_ID));

    #pragma unroll
    for (int tile_idx = 0; tile_idx < TILES; ++tile_idx) {

        if ((WARP_ID * TILES + tile_idx) >= total_tiles) break;

        const int tile_m = ((WARP_ID * TILES + tile_idx) / num_tiles_n) * WMMA_M;
        const int tile_n = ((WARP_ID * TILES + tile_idx) % num_tiles_n) * WMMA_N;

        if (tile_m >= VALID_M) continue;

        auto& fragment0 = REGS0[tile_idx];

        #pragma unroll
        for (int idx = 0; idx < 8; idx += 2) {
            const int row_offset  = (idx >> 1) & 0b1;
            const int global_m    = tile_m + static_cast<int>(lines.x + row_offset * 2);
            const int global_n    = tile_n + static_cast<int>(lines.y + (idx & 0b1) + ((idx >> 2) & 0b1) * 4);

            if (global_m >= VALID_M) continue;

            float scale = 1.0f;
            if constexpr (NORMALIZE) {
                scale = __frcp_rn(fmaxf(ROW_SUM[(tile_idx / num_tiles_n) * 2 + row_offset], 1e-24f));
            }

            const __half2 h0 = __float22half2_rn(make_float2(fragment0.x[idx] * scale, fragment0.x[idx + 1] * scale));
            const ushort  u0 = __half_as_ushort(h0.x);
            const ushort  u1 = __half_as_ushort(h0.y);
            const int pred   = (global_n + 1 < D) ? 1 : 0;

            __half* dst0 = GMEM0 + (size_t)global_m * STRIDE + global_n;

            if constexpr (DUAL_STORE) {
                const Fragment* regs1 = static_cast<const Fragment*>(REGS1);
                auto& fragment1  = regs1[tile_idx];
                const __half2 h1 = __float22half2_rn(make_float2(fragment1.x[idx], fragment1.x[idx + 1]));
                const ushort  u2 = __half_as_ushort(h1.x);
                const ushort  u3 = __half_as_ushort(h1.y);

                __half* dst1 = GMEM1 + (size_t)global_m * STRIDE + global_n;

                asm volatile(
                    "{\n\t"
                    ".reg .pred p;\n\t"
                    "setp.ne.b32 p, %4, 0;\n\t"
                    "@p st.global.v2.u16 [%2], {%0, %1};\n\t"
                    "@p st.global.v2.u16 [%3], {%5, %6};\n\t"
                    "}\n\t"
                    :
                    : "h"(u0), "h"(u1), "l"(dst0), "l"(dst1), "r"(pred), "h"(u2), "h"(u3)
                    : "memory"
                );
            } else {
                asm volatile(
                    "{\n\t"
                    ".reg .pred p;\n\t"
                    "setp.ne.b32 p, %3, 0;\n\t"
                    "@p st.global.v2.u16 [%2], {%0, %1};\n\t"
                    "}\n\t"
                    :
                    : "h"(u0), "h"(u1), "l"(dst0), "r"(pred)
                    : "memory"
                );

                if constexpr (WRITE_LSE) {
                    if (GMEM_LSE != nullptr
                        && (((LANE_ID >> 3) & 1) == 0)
                        && tile_n                == 0
                        && row_offset            == ((LANE_ID >> 1) & 1)
                        && idx                   == ((LANE_ID >> 1) & 1) * 2)
                    {
                        const int   jdx = (tile_idx / num_tiles_n) * 2 + ((LANE_ID >> 1) & 1);
                        const float sum = fmaxf(ROW_SUM[jdx], 1e-24f);
                        GMEM_LSE[global_m] = ROW_MAX[jdx] + __logf(sum);
                    }
                }
            }
        }
    }
}
