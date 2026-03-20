#pragma once

// ============================================================================
// CONFIGURATIONS
// ============================================================================
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

// ============================================================================
// CONFIGURATIONS
// ============================================================================
template<int D>
struct KernelConfig {
    static constexpr int BLOCK_M = (D == 16) ? BLOCK_M_16 : (D == 32) ? BLOCK_M_32 : (D == 64) ? BLOCK_M_64 : (D == 128) ? BLOCK_M_128 : BLOCK_M_256;
    static constexpr int BLOCK_N = (D == 16) ? BLOCK_N_16 : (D == 32) ? BLOCK_N_32 : (D == 64) ? BLOCK_N_64 : (D == 128) ? BLOCK_N_128 : BLOCK_N_256;
    static constexpr int WARPS_PER_BLOCK   = WARPS;
    static constexpr int THREADS_PER_BLOCK = WARPS_PER_BLOCK * MAX_THREADS_PER_WARP;
    static constexpr int THREADS_PER_ROW   = THREADS_PER_BLOCK / BLOCK_M;
    static constexpr int PAD               = (8 - (D % 32) + 32) % 32;
    static constexpr int D_STRIDE          = D + PAD + (((D + PAD) % 64 == 0) ? 1 : 0);
    static constexpr int N_STRIDE          = BLOCK_N + PAD + (((BLOCK_N + PAD) % 32 == 0) ? 1 : 0);

    struct alignas(128) SmemLayout {
        alignas(16) __half q      [BLOCK_M * D_STRIDE];
    union {
        alignas(16) __half k      [BLOCK_N * D_STRIDE];
        alignas(16) __half v      [BLOCK_N * D_STRIDE];
    } reuse_kv;
    union {
        alignas(16) float  s      [BLOCK_M * N_STRIDE];
        alignas(16) __half p      [BLOCK_M * N_STRIDE];
    } reuse_sp;
        alignas(16) float  o      [BLOCK_M * D_STRIDE];
        alignas(16) float  row_max[BLOCK_M];
        alignas(16) float  row_sum[BLOCK_M];
    };

    static constexpr size_t TOTAL_SMEM = ((sizeof(SmemLayout) + 127) & ~size_t(127));
};
