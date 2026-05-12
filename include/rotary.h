// ======================================================================================
// * Copyright (c) 2025, D.Skryabin / tg @ai_bond007 SPDX-License: BSD-3-Clause
// ======================================================================================
#pragma once

#include <cuda.h>
#include <cuda_runtime.h>
#include <cuda_fp16.h>

// ======================================================================================
// KV-CACHE UPDATER: Global(K_NEW/V_NEW) -> Registers -> (RoPE on K) -> Global(K_CACHE/V_CACHE)
// ======================================================================================
template<typename Config, int GLOBAL_STRIDE, bool IS_ROPE, bool IS_INTERLEAVED>
__device__ __forceinline__ void WMMA_GEMM_UPDATE_KVCACHE(
    const __half* __restrict__ K_NEW,
    const __half* __restrict__ V_NEW,
          __half* __restrict__ K_CACHE,
          __half* __restrict__ V_CACHE,
    const __half* __restrict__ ROTARY_COS,
    const __half* __restrict__ ROTARY_SIN,
    int T_NEW,
    int H_K,
    int STRIDE_NEW_B,   int STRIDE_NEW_S,   int STRIDE_NEW_H,
    int STRIDE_CACHE_B, int STRIDE_CACHE_S, int STRIDE_CACHE_H,
    int BATCH_IDX,
    int KV_HEAD_IDX,
    int CACHE_BIDX,
    int CACHE_SEQLEN,
    int ROTARY_DIM,
    int LEFTPAD,
    int THREAD_ID
) {
    if (K_NEW == nullptr || V_NEW == nullptr || T_NEW <= 0 || KV_HEAD_IDX >= H_K) return;

    constexpr int THREADS      = Config::THREADS_PER_BLOCK;
    constexpr int CHUNKS       = GLOBAL_STRIDE / 8;
    const     int total_iters  = T_NEW * CHUNKS;

    if (total_iters == 0) return;

    const int half_rot      = IS_ROPE ? (ROTARY_DIM / 2) : 0;

    uint64_t base_k_new   = reinterpret_cast<uint64_t>(K_NEW);
    uint64_t base_v_new   = reinterpret_cast<uint64_t>(V_NEW);
    uint64_t base_k_cache = reinterpret_cast<uint64_t>(K_CACHE);
    uint64_t base_v_cache = reinterpret_cast<uint64_t>(V_CACHE);

    #pragma unroll 2
    for (int idx = THREAD_ID; idx < total_iters; idx += THREADS) {
        const int row    = idx / CHUNKS;
        const int chunk  = idx % CHUNKS;
        const int d_base = chunk * 8;

        const size_t off_new   = static_cast<size_t>(BATCH_IDX)    * STRIDE_NEW_B +
                                 static_cast<size_t>(KV_HEAD_IDX)  * STRIDE_NEW_H +
                                 static_cast<size_t>(row)          * STRIDE_NEW_S + d_base;
        const size_t off_cache = static_cast<size_t>(CACHE_BIDX)   * STRIDE_CACHE_B +
                                 static_cast<size_t>(KV_HEAD_IDX)  * STRIDE_CACHE_H +
                                 static_cast<size_t>(CACHE_SEQLEN  + LEFTPAD + row) * STRIDE_CACHE_S + d_base;

        uint64_t src_k = base_k_new   + off_new * 2;
        uint64_t src_v = base_v_new   + off_new * 2;
        uint64_t dst_k = base_k_cache + off_cache * 2;
        uint64_t dst_v = base_v_cache + off_cache * 2;

        uint32_t k[4], v[4];

        asm volatile("ld.global.v4.u32 {%0, %1, %2, %3}, [%4];" : "=r"(k[0]), "=r"(k[1]), "=r"(k[2]), "=r"(k[3]) : "l"(src_k));
        asm volatile("ld.global.v4.u32 {%0, %1, %2, %3}, [%4];" : "=r"(v[0]), "=r"(v[1]), "=r"(v[2]), "=r"(v[3]) : "l"(src_v));

        if constexpr(IS_ROPE) {
            const int pos = CACHE_SEQLEN + LEFTPAD + row;
            if (IS_INTERLEAVED) {
                #pragma unroll
                for (int i = 0; i < 4; ++i) {
                    int d = d_base + i * 2;

                    if (d >= ROTARY_DIM) break;

                    half2 x   = *reinterpret_cast<half2*>(&k[i]);
                    float2 xf = __half22float2(x);
                    float c   = __half2float(ROTARY_COS[pos * half_rot + d / 2]);
                    float s   = __half2float(ROTARY_SIN[pos * half_rot + d / 2]);
                    float y0  = __fmaf_rn(xf.x, c, -xf.y * s);
                    float y1  = __fmaf_rn(xf.x, s,  xf.y * c);
                    half2 y   = __float22half2_rn(make_float2(y0, y1));
                    k[i]      = *reinterpret_cast<uint32_t*>(&y);
                }
                asm volatile("st.global.v4.u32 [%0], {%1, %2, %3, %4};" : : "l"(dst_k), "r"(k[0]), "r"(k[1]), "r"(k[2]), "r"(k[3]));
            } else {
                if (d_base < half_rot) {
                    uint64_t src_k2 = src_k + half_rot * 2;
                    uint32_t k2[4];
                    asm volatile("ld.global.v4.u32 {%0, %1, %2, %3}, [%4];" : "=r"(k2[0]), "=r"(k2[1]), "=r"(k2[2]), "=r"(k2[3]) : "l"(src_k2));

                    #pragma unroll
                    for (int i = 0; i < 4; ++i) {
                        int d = d_base + i * 2;

                        if (d >= half_rot) continue;

                        float x0_0 = __half2float(__ushort_as_half((k[i] & 0xFFFF)));
                        float x1_0 = __half2float(__ushort_as_half((k2[i] & 0xFFFF)));
                        float c0   = __half2float(ROTARY_COS[pos * half_rot + d]);
                        float s0   = __half2float(ROTARY_SIN[pos * half_rot + d]);
                        float y0_0 = __fmaf_rn(x0_0, c0, -x1_0 * s0);
                        float y1_0 = __fmaf_rn(x0_0, s0,  x1_0 * c0);

                        float x0_1 = __half2float(__ushort_as_half((k[i] >> 16)));
                        float x1_1 = __half2float(__ushort_as_half((k2[i] >> 16)));
                        float c1   = __half2float(ROTARY_COS[pos * half_rot + d + 1]);
                        float s1   = __half2float(ROTARY_SIN[pos * half_rot + d + 1]);
                        float y0_1 = __fmaf_rn(x0_1, c1, -x1_1 * s1);
                        float y1_1 = __fmaf_rn(x0_1, s1,  x1_1 * c1);

                        k[i]  = (__half_as_ushort(__float2half_rn(y0_1)) << 16) | __half_as_ushort(__float2half_rn(y0_0));
                        k2[i] = (__half_as_ushort(__float2half_rn(y1_1)) << 16) | __half_as_ushort(__float2half_rn(y1_0));
                    }
                    uint64_t dst_k2 = dst_k + half_rot * 2;
                    asm volatile("st.global.v4.u32 [%0], {%1, %2, %3, %4};" : : "l"(dst_k), "r"(k[0]), "r"(k[1]), "r"(k[2]), "r"(k[3]));
                    asm volatile("st.global.v4.u32 [%0], {%1, %2, %3, %4};" : : "l"(dst_k2), "r"(k2[0]), "r"(k2[1]), "r"(k2[2]), "r"(k2[3]));
                } else if (d_base >= ROTARY_DIM) {
                    asm volatile("st.global.v4.u32 [%0], {%1, %2, %3, %4};" : : "l"(dst_k), "r"(k[0]), "r"(k[1]), "r"(k[2]), "r"(k[3]));
                }
            }
        } else {
            asm volatile("st.global.v4.u32 [%0], {%1, %2, %3, %4};" : : "l"(dst_k), "r"(k[0]), "r"(k[1]), "r"(k[2]), "r"(k[3]));
        }
        asm volatile("st.global.v4.u32 [%0], {%1, %2, %3, %4};" : : "l"(dst_v), "r"(v[0]), "r"(v[1]), "r"(v[2]), "r"(v[3]));
    }
}

// ======================================================================================
// TILE LOADER + ROTARY
// ======================================================================================
template<typename Config, int SMEM_STRIDE, bool IS_CAUSAL, bool IS_WINDOW, bool IS_ROPE, bool IS_INTERLEAVED, int GLOBAL_WIDTH = -1>
__device__ __forceinline__ void WMMA_GEMM_TILE_ROTARY(
    const __half* __restrict__ GMEM,
          __half* __restrict__ SMEM,
    const __half* __restrict__ ROTARY_COS,
    const __half* __restrict__ ROTARY_SIN,
    int GLOBAL_STRIDE,
    int VALID_ROWS,
    int CACHE_SEQLEN,
    int ROTARY_DIM,
    int LEFTPAD,
    int THREAD_ID
) {
    constexpr int THREADS_PER_BLOCK = Config::THREADS_PER_BLOCK;
    constexpr int dst_stride_uint4  = (SMEM_STRIDE + 7) >> 3;
    const     int wth_stride_uint4  = (((GLOBAL_WIDTH > 0) ? GLOBAL_WIDTH : GLOBAL_STRIDE) + 7) >> 3;
    const     int src_stride_uint4  = (GLOBAL_STRIDE + 7) >> 3;
    const     int total_iters       = VALID_ROWS * wth_stride_uint4;

    if (total_iters == 0) return;

    const int q_rope_stride = (IS_CAUSAL || IS_WINDOW) ? 1 : 0;
    const int half_rot      = IS_ROPE ? (ROTARY_DIM / 2) : 0;

    uint64_t src_base = static_cast<uint64_t>(__cvta_generic_to_global(GMEM));
    uint32_t dst_base = static_cast<uint32_t>(__cvta_generic_to_shared(SMEM));

    #pragma unroll 2
    for (int idx = THREAD_ID; idx < total_iters; idx += THREADS_PER_BLOCK) {
        const int row = idx / wth_stride_uint4;
        const int col = idx % wth_stride_uint4;

        const int src_off = row * src_stride_uint4 + col;
        const int dst_off = row * dst_stride_uint4 + col;

        uint64_t src_addr = src_base + (static_cast<uint64_t>(src_off) << 4);
        uint32_t dst_addr = dst_base + (static_cast<uint32_t>(dst_off) << 4);

        uint32_t r[4];
        asm volatile("ld.global.v4.u32 {%0, %1, %2, %3}, [%4];"
                     : "=r"(r[0]), "=r"(r[1]), "=r"(r[2]), "=r"(r[3])
                     : "l"(src_addr) : "memory");

        if constexpr(IS_ROPE) {
            const int pos    = CACHE_SEQLEN + LEFTPAD + row * q_rope_stride;
            const int d_base = col * 8;

            if constexpr (IS_INTERLEAVED) {
                #pragma unroll
                for (int i = 0; i < 4; ++i) {
                    int d = d_base + i * 2;

                    if (d >= ROTARY_DIM) break;

                    half2 x   = *reinterpret_cast<half2*>(&r[i]);
                    float2 xf = __half22float2(x);
                    float c   = __half2float(ROTARY_COS[pos * half_rot + d / 2]);
                    float s   = __half2float(ROTARY_SIN[pos * half_rot + d / 2]);
                    float y0  = __fmaf_rn(xf.x, c, -xf.y * s);
                    float y1  = __fmaf_rn(xf.x, s,  xf.y * c);
                    half2 y   = __float22half2_rn(make_float2(y0, y1));
                    r[i]      = *reinterpret_cast<uint32_t*>(&y);
                }
                asm volatile("st.shared.v4.u32 [%0], {%1, %2, %3, %4};"
                             : : "r"(dst_addr), "r"(r[0]), "r"(r[1]), "r"(r[2]), "r"(r[3]) : "memory");
            } else {
                if (d_base < half_rot) {
                    uint64_t src_r2 = src_addr + half_rot * 2;
                    uint32_t r2[4];
                    asm volatile("ld.global.v4.u32 {%0, %1, %2, %3}, [%4];"
                                 : "=r"(r2[0]), "=r"(r2[1]), "=r"(r2[2]), "=r"(r2[3])
                                 : "l"(src_r2) : "memory");

                    #pragma unroll
                    for (int i = 0; i < 4; ++i) {
                        int d = d_base + i * 2;

                        if (d >= half_rot) continue;

                        float x0_0 = __half2float(__ushort_as_half((r[i] & 0xFFFF)));
                        float x1_0 = __half2float(__ushort_as_half((r2[i] & 0xFFFF)));
                        float c0   = __half2float(ROTARY_COS[pos * half_rot + d]);
                        float s0   = __half2float(ROTARY_SIN[pos * half_rot + d]);
                        float y0_0 = __fmaf_rn(x0_0, c0, -x1_0 * s0);
                        float y1_0 = __fmaf_rn(x0_0, s0,  x1_0 * c0);

                        float x0_1 = __half2float(__ushort_as_half((r[i] >> 16)));
                        float x1_1 = __half2float(__ushort_as_half((r2[i] >> 16)));
                        float c1   = __half2float(ROTARY_COS[pos * half_rot + d + 1]);
                        float s1   = __half2float(ROTARY_SIN[pos * half_rot + d + 1]);
                        float y0_1 = __fmaf_rn(x0_1, c1, -x1_1 * s1);
                        float y1_1 = __fmaf_rn(x0_1, s1,  x1_1 * c1);

                        r[i]  = (__half_as_ushort(__float2half_rn(y0_1)) << 16) | __half_as_ushort(__float2half_rn(y0_0));
                        r2[i] = (__half_as_ushort(__float2half_rn(y1_1)) << 16) | __half_as_ushort(__float2half_rn(y1_0));
                    }
                    uint32_t dst_r2 = dst_addr + half_rot * 2;
                    asm volatile("st.shared.v4.u32 [%0], {%1, %2, %3, %4};"
                                 : : "r"(dst_addr), "r"(r[0]), "r"(r[1]), "r"(r[2]), "r"(r[3]) : "memory");
                    asm volatile("st.shared.v4.u32 [%0], {%1, %2, %3, %4};"
                                 : : "r"(dst_r2), "r"(r2[0]), "r"(r2[1]), "r"(r2[2]), "r"(r2[3]) : "memory");
                } else if (d_base >= ROTARY_DIM) {
                    asm volatile("st.shared.v4.u32 [%0], {%1, %2, %3, %4};"
                                 : : "r"(dst_addr), "r"(r[0]), "r"(r[1]), "r"(r[2]), "r"(r[3]) : "memory");
                }
            }
        } else {
            asm volatile("st.shared.v4.u32 [%0], {%1, %2, %3, %4};"
                         : : "r"(dst_addr), "r"(r[0]), "r"(r[1]), "r"(r[2]), "r"(r[3]) : "memory");
        }
    }
}
