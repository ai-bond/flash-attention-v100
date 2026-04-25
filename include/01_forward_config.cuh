// ======================================================================================
// * Copyright (c) 2025, D.Skryabin / tg @ai_bond007 SPDX-License: BSD-3-Clause
// ======================================================================================

#pragma once

// ======================================================================================
// CONFIGURATIONS
// ======================================================================================
#define BLOCK_M_16  16
#define BLOCK_N_16  512

#define BLOCK_M_32  32
#define BLOCK_N_32  256

#define BLOCK_M_64  64
#define BLOCK_N_64  128

#define BLOCK_M_128 32
#define BLOCK_N_128 176

#define BLOCK_M_256 32
#define BLOCK_N_256 64

#define WARPS 16

// ======================================================================================
// CONFIGURATIONS
// ======================================================================================
template<int D>
struct KernelConfig {
    struct DO {
        static constexpr int BLOCK_M = (D == 16) ? BLOCK_M_16 : (D == 32) ? BLOCK_M_32 : (D == 64) ? BLOCK_M_64 : (D == 128) ? BLOCK_M_128 : BLOCK_M_256;
        static constexpr int BLOCK_N = (D == 16) ? BLOCK_N_16 : (D == 32) ? BLOCK_N_32 : (D == 64) ? BLOCK_N_64 : (D == 128) ? BLOCK_N_128 : BLOCK_N_256;
        static constexpr int THREADS_PER_ROW   = (WARPS * MAX_THREADS_PER_WARP) / BLOCK_M;
        static constexpr int PAD               = (8 - (D % 32) + 32) % 32;
        static constexpr int D_STRIDE          = D + PAD + (((D + PAD) % 64 == 0) ? 1 : 0);
        static constexpr int N_STRIDE          = BLOCK_N + PAD + (((BLOCK_N + PAD) % 32 == 0) ? 1 : 0);
    };

    static constexpr int WARPS_PER_BLOCK       = WARPS;
    static constexpr int THREADS_PER_BLOCK     = WARPS * MAX_THREADS_PER_WARP;

    struct alignas(128) SmemLayout {
        union PhaseMem {
            struct DO_Phase {
                    alignas(16) __half q      [DO::BLOCK_M * DO::D_STRIDE];
                union {
                    alignas(16) __half k      [DO::BLOCK_N * DO::D_STRIDE];
                    alignas(16) __half v      [DO::BLOCK_N * DO::D_STRIDE];
                } reuse_kv;
                union {
                    alignas(16) float  s      [DO::BLOCK_M * DO::N_STRIDE];
                    alignas(16) __half p      [DO::BLOCK_M * DO::N_STRIDE];
                } reuse_sp;
                    alignas(16) float  o      [DO::BLOCK_M * DO::D_STRIDE];
            } fdo;
        } phase;
                    alignas(16) float  row_max[DO::BLOCK_M];
                    alignas(16) float  row_sum[DO::BLOCK_M];
    };

    static constexpr int TOTAL_SMEM = static_cast<int>(((sizeof(SmemLayout) + 127) & ~size_t(127)));
};