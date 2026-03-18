#pragma once

// ============================================================================
// CONFIGURATIONS
// ============================================================================
#define BLOCK_M_DQ_16   16
#define BLOCK_N_DQ_16   256
#define WARPS_DQ_16     16

#define BLOCK_M_DQ_32   32
#define BLOCK_N_DQ_32   128
#define WARPS_DQ_32     16

#define BLOCK_M_DQ_64   64
#define BLOCK_N_DQ_64   80
#define WARPS_DQ_64     16

#define BLOCK_M_DQ_128  32
#define BLOCK_N_DQ_128  112
#define WARPS_DQ_128    16

#define BLOCK_M_DQ_256  32
#define BLOCK_N_DQ_256  32
#define WARPS_DQ_256    16

#define BLOCK_M_DKV_16  32
#define BLOCK_N_DKV_16  224
#define WARPS_DKV_16    14

#define BLOCK_M_DKV_32  32
#define BLOCK_N_DKV_32  192
#define WARPS_DKV_32    12

#define BLOCK_M_DKV_64  32
#define BLOCK_N_DKV_64  128
#define WARPS_DKV_64    8

#define BLOCK_M_DKV_128 16
#define BLOCK_N_DKV_128 144
#define WARPS_DKV_128   12

#define BLOCK_M_DKV_256 16
#define BLOCK_N_DKV_256 64
#define WARPS_DKV_256   16

// ============================================================================
// COMPILE-TIME CONFIGURATION & SHARED MEMORY LAYOUT
// ============================================================================
template<int D>
struct KernelConfig {
    struct DQ {
        static constexpr int BLOCK_M            = (D == 16) ? BLOCK_M_DQ_16 : (D == 32) ? BLOCK_M_DQ_32 : (D == 64) ? BLOCK_M_DQ_64 : (D == 128) ? BLOCK_M_DQ_128 : BLOCK_M_DQ_256;
        static constexpr int BLOCK_N            = (D == 16) ? BLOCK_N_DQ_16 : (D == 32) ? BLOCK_N_DQ_32 : (D == 64) ? BLOCK_N_DQ_64 : (D == 128) ? BLOCK_N_DQ_128 : BLOCK_N_DQ_256;
        static constexpr int WARPS_PER_BLOCK    = (D == 16) ? WARPS_DQ_16 : (D == 32) ? WARPS_DQ_32 : (D == 64) ? WARPS_DQ_64 : (D == 128) ? WARPS_DQ_128 : WARPS_DQ_256;
        static constexpr int THREADS_PER_BLOCK  = WARPS_PER_BLOCK * MAX_THREADS_PER_WARP;
        static constexpr int THREADS_PER_ROW    = THREADS_PER_BLOCK / BLOCK_M;
        static constexpr int PAD                = (8 - (D % 32) + 32) % 32;
        static constexpr int Q_STRIDE           = D + PAD;
        static constexpr int KV_STRIDE          = D + PAD;
        static constexpr int S_STRIDE           = BLOCK_N + PAD;
        static constexpr int PER_UINT4          = 8;
        static constexpr int NUM_UINT4_Q_BLOCK  = BLOCK_M * ((D + PER_UINT4 - 1) / PER_UINT4);
        static constexpr int NUM_UINT4_KV_BLOCK = BLOCK_N * ((D + PER_UINT4 - 1) / PER_UINT4);
    };
    struct DKV {
        static constexpr int BLOCK_M            = (D == 16) ? BLOCK_M_DKV_16 : (D == 32) ? BLOCK_M_DKV_32 : (D == 64) ? BLOCK_M_DKV_64 : (D == 128) ? BLOCK_M_DKV_128 : BLOCK_M_DKV_256;
        static constexpr int BLOCK_N            = (D == 16) ? BLOCK_N_DKV_16 : (D == 32) ? BLOCK_N_DKV_32 : (D == 64) ? BLOCK_N_DKV_64 : (D == 128) ? BLOCK_N_DKV_128 : BLOCK_N_DKV_256;
        static constexpr int WARPS_PER_BLOCK    = (D == 16) ? WARPS_DKV_16 : (D == 32) ? WARPS_DKV_32 : (D == 64) ? WARPS_DKV_64 : (D == 128) ? WARPS_DKV_128 : WARPS_DKV_256;
        static constexpr int THREADS_PER_BLOCK  = WARPS_PER_BLOCK * MAX_THREADS_PER_WARP;
        static constexpr int THREADS_PER_ROW    = THREADS_PER_BLOCK / BLOCK_N;
        static constexpr int PAD                = 8;
        static constexpr int Q_STRIDE           = D + PAD;
        static constexpr int KV_STRIDE          = D + PAD;
        static constexpr int S_STRIDE           = BLOCK_M + PAD;
        static constexpr int PER_UINT4          = 8;
        static constexpr int NUM_UINT4_Q_BLOCK  = BLOCK_N * ((D + PER_UINT4 - 1) / PER_UINT4);
        static constexpr int NUM_UINT4_KV_BLOCK = BLOCK_M * ((D + PER_UINT4 - 1) / PER_UINT4);
    };

    static constexpr int MAX_THREADS = (DQ::THREADS_PER_BLOCK > DKV::THREADS_PER_BLOCK) ? DQ::THREADS_PER_BLOCK : DKV::THREADS_PER_BLOCK;
    static constexpr int MAX_LSE = (DQ::BLOCK_M > DKV::BLOCK_N) ? DQ::BLOCK_M : DKV::BLOCK_N;

    struct alignas(128) SmemLayout {
        union PhaseMem {
            struct DQ_Phase {
                union {
                    alignas(16) __half k  [ DQ::BLOCK_N * DQ::KV_STRIDE ];
                    alignas(16) __half v  [ DQ::BLOCK_N * DQ::KV_STRIDE ];
                } reuse_kv;
                    alignas(16) __half dO [ DQ::BLOCK_M * DQ::Q_STRIDE ];
                    alignas(16) __half q  [ DQ::BLOCK_M * DQ::Q_STRIDE ];
                    alignas(16) float  s  [ DQ::BLOCK_M * DQ::S_STRIDE ];
                union {
                    alignas(16) float  dOV[ DQ::BLOCK_M * DQ::S_STRIDE ];
                    alignas(16) __half dS [ DQ::BLOCK_M * DQ::S_STRIDE ];
                } reuse_sdOVS;
                    alignas(16) float  dQ [ DQ::BLOCK_M * DQ::Q_STRIDE ];
            } dq;

            struct DKV_Phase {
                    alignas(16) __half k [ DKV::BLOCK_M * DKV::KV_STRIDE ];
                    alignas(16) __half v [ DKV::BLOCK_M * DKV::KV_STRIDE ];
                union {
                    alignas(16) __half dO[ DKV::BLOCK_N * DKV::Q_STRIDE ];
                    alignas(16) __half q [ DKV::BLOCK_N * DKV::Q_STRIDE ];
                } reuse_qdO;
                union {
                    alignas(16) float  s [ DKV::BLOCK_N * DKV::S_STRIDE ];
                    alignas(16) __half p [ DKV::BLOCK_N * DKV::BLOCK_M ];
                } reuse_sp;
                union {
                    alignas(16) float  dOV[ DKV::BLOCK_N * DKV::S_STRIDE ];
                    alignas(16) __half dS [ DKV::BLOCK_N * DKV::BLOCK_M ];
                } reuse_dOVS;
                    alignas(16) float dK[ DKV::BLOCK_M * DKV::KV_STRIDE ];
                    alignas(16) float dV[ DKV::BLOCK_M * DKV::KV_STRIDE ];
                } dkv;
        } phase;
                    alignas(16) float lse     [MAX_LSE];
                    alignas(16) float row_dot [MAX_LSE];
    };

    static constexpr size_t TOTAL_SMEM = ((sizeof(SmemLayout) + 127) & ~size_t(127));
};
