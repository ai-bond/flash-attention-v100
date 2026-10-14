// ======================================================================================
// * Copyright (c) 2026, D.Skryabin / tg @ai_bond007 SPDX-License: BSD-3-Clause
// ======================================================================================
#pragma once

struct LogitsConfig {
    static constexpr int BLOCK_M = 64;
    static constexpr int BLOCK_N = 128;
    static constexpr int DIM     = 128;

    static constexpr int WARPS_PER_BLOCK    = 32;
    static constexpr int THREADS_PER_BLOCK  = WARPS_PER_BLOCK * MAX_THREADS_PER_WARP;
    static constexpr int THREADS_PER_ROW    = THREADS_PER_BLOCK / BLOCK_M;
    static constexpr int MAX_ITERS          = ((BLOCK_N >> 2) + THREADS_PER_ROW - 1) / THREADS_PER_ROW;
    static constexpr int MAX_TAILS          = (4 + THREADS_PER_ROW - 1) / THREADS_PER_ROW;
    static constexpr int MAX_THREADS        = 8;

    struct alignas(128) SmemLayout {
        alignas(16) __half input [BLOCK_M * DIM];
        alignas(16) __half weight[BLOCK_N * DIM];
        alignas(16) float  logits[BLOCK_M * BLOCK_N];
    };
    static constexpr int TOTAL_SMEM = (sizeof(SmemLayout) + 127) & ~size_t(127);
};

struct dXWConfig {
    static constexpr int BLOCK_M           = 128;
    static constexpr int BLOCK_N           = 64;
    static constexpr int DIM               = BLOCK_M;
    static constexpr int GRID_X            = 4;
    static constexpr int GRID_Y            = 2;
    static constexpr int THREADS_PER_BLOCK = (GRID_X * GRID_Y) * MAX_THREADS_PER_WARP;

    struct alignas(128) SmemLayout {
        alignas(16) __half TdLogits    [BLOCK_M * DIM];
        alignas(16) __half TweightInput[DIM * BLOCK_N];
    };
    static constexpr int TOTAL_SMEM = (sizeof(SmemLayout) + 127) & ~size_t(127);
};
