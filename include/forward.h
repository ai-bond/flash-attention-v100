// ======================================================================================
// * Copyright (c) 2026, D.Skryabin / tg @ai_bond007 SPDX-License: BSD-3-Clause
// ======================================================================================
#pragma once

#define BLOCK_M_16  128
#define BLOCK_N_16  128

#define BLOCK_M_32  128
#define BLOCK_N_32  128

#define BLOCK_M_64  128
#define BLOCK_N_64  64

#define BLOCK_M_128 128
#define BLOCK_N_128 128

#define BLOCK_M_256 128
#define BLOCK_N_256 32

#define WARPS 8

// ======================================================================================
// CONFIGURATIONS
// ======================================================================================
template<int D>
struct KernelConfig {
    struct DO {
        static constexpr int BLOCK_M = (D == 16) ? BLOCK_M_16 : (D == 32) ? BLOCK_M_32 : (D == 64) ? BLOCK_M_64 : (D == 128) ? BLOCK_M_128 : BLOCK_M_256;
        static constexpr int BLOCK_N = (D == 16) ? BLOCK_N_16 : (D == 32) ? BLOCK_N_32 : (D == 64) ? BLOCK_N_64 : (D == 128) ? BLOCK_N_128 : BLOCK_N_256;
        static constexpr int THREADS_PER_ROW = (WARPS * MAX_THREADS_PER_WARP) / BLOCK_M;
    };

    static constexpr int DIM = D;
    static constexpr int WARPS_PER_BLOCK     = WARPS;
    static constexpr int THREADS_PER_BLOCK   = WARPS * MAX_THREADS_PER_WARP;

    struct alignas(128) SmemLayout {
        union PhaseMem {
            struct DO_Phase {
                    alignas(16) __half q      [DO::BLOCK_M * D];
                union {
                    alignas(16) __half k      [DO::BLOCK_N * D];
                    alignas(16) __half v      [DO::BLOCK_N * D];
                } reuse_kv;
                    alignas(16) __half p      [DO::BLOCK_M * DO::BLOCK_N];
            } fdo;
        } phase;
    };

    static constexpr int TOTAL_SMEM = static_cast<int>(((sizeof(SmemLayout) + 127) & ~size_t(127)));
};
