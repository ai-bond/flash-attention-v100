#pragma once

// ============================================================================
// CONFIGURATIONS
// ============================================================================
#define BLOCK_M_DQ_16   16
#define BLOCK_N_DQ_16   256

#define BLOCK_M_DQ_32   32
#define BLOCK_N_DQ_32   128

#define BLOCK_M_DQ_64   64
#define BLOCK_N_DQ_64   80

#define BLOCK_M_DQ_128  32
#define BLOCK_N_DQ_128  112

#define BLOCK_M_DQ_256  32
#define BLOCK_N_DQ_256  32

#define BLOCK_M_DKV_16  32
#define BLOCK_N_DKV_16  224

#define BLOCK_M_DKV_32  32
#define BLOCK_N_DKV_32  192

#define BLOCK_M_DKV_64  32
#define BLOCK_N_DKV_64  128

#define BLOCK_M_DKV_128 16
#define BLOCK_N_DKV_128 128

#define BLOCK_M_DKV_256 16
#define BLOCK_N_DKV_256 64

#define WARPS 16

// ============================================================================
// CONFIGURATION DQ/KV
// ============================================================================
template<int D>
struct KernelConfig {
    struct DQ {
        static constexpr int BLOCK_M            = (D == 16) ? BLOCK_M_DQ_16 : (D == 32) ? BLOCK_M_DQ_32 : (D == 64) ? BLOCK_M_DQ_64 : (D == 128) ? BLOCK_M_DQ_128 : BLOCK_M_DQ_256;
        static constexpr int BLOCK_N            = (D == 16) ? BLOCK_N_DQ_16 : (D == 32) ? BLOCK_N_DQ_32 : (D == 64) ? BLOCK_N_DQ_64 : (D == 128) ? BLOCK_N_DQ_128 : BLOCK_N_DQ_256;
        static constexpr int WARPS_PER_BLOCK    = WARPS;
        static constexpr int THREADS_PER_ROW    = (WARPS_PER_BLOCK * MAX_THREADS_PER_WARP) / BLOCK_M;
        static constexpr int PAD                = (8 - (D % 32) + 32) % 32;
        static constexpr int D_STRIDE           = D + PAD + (((D + PAD) % 64 == 0) ? 1 : 0);
        static constexpr int N_STRIDE           = BLOCK_N + PAD + (((BLOCK_N + PAD) % 32 == 0) ? 1 : 0);
    };
    struct DKV {
        static constexpr int BLOCK_M            = (D == 16) ? BLOCK_M_DKV_16 : (D == 32) ? BLOCK_M_DKV_32 : (D == 64) ? BLOCK_M_DKV_64 : (D == 128) ? BLOCK_M_DKV_128 : BLOCK_M_DKV_256;
        static constexpr int BLOCK_N            = (D == 16) ? BLOCK_N_DKV_16 : (D == 32) ? BLOCK_N_DKV_32 : (D == 64) ? BLOCK_N_DKV_64 : (D == 128) ? BLOCK_N_DKV_128 : BLOCK_N_DKV_256;
        static constexpr int WARPS_PER_BLOCK    = WARPS;
        static constexpr int THREADS_PER_ROW    = (WARPS_PER_BLOCK * MAX_THREADS_PER_WARP) / BLOCK_N;
        static constexpr int PAD                = 8;
        static constexpr int D_STRIDE           = D + PAD + (((D + PAD) % 64 == 0) ? 1 : 0);
        static constexpr int M_STRIDE           = BLOCK_M + PAD + (((BLOCK_M + PAD) % 32 == 0) ? 1 : 0);
    };

    static constexpr int THREADS_PER_BLOCK  = WARPS * MAX_THREADS_PER_WARP;

    struct alignas(128) SmemLayout {
        union PhaseMem {
            struct DQ_Phase {
                union {
                    alignas(16) __half k  [ DQ::BLOCK_N * DQ::D_STRIDE ];
                    alignas(16) __half v  [ DQ::BLOCK_N * DQ::D_STRIDE ];
                } reuse_kv;
                    alignas(16) __half dO [ DQ::BLOCK_M * DQ::D_STRIDE ];
                    alignas(16) __half q  [ DQ::BLOCK_M * DQ::D_STRIDE ];
                    alignas(16) float  s  [ DQ::BLOCK_M * DQ::N_STRIDE ];
                union {
                    alignas(16) float  dOV[ DQ::BLOCK_M * DQ::N_STRIDE ];
                    alignas(16) __half dS [ DQ::BLOCK_M * DQ::N_STRIDE ];
                } reuse_sdOVS;
                    alignas(16) float  dQ [ DQ::BLOCK_M * DQ::D_STRIDE ];
            } dq;

            struct DKV_Phase {
                    alignas(16) __half k [ DKV::BLOCK_M * DKV::D_STRIDE ];
                    alignas(16) __half v [ DKV::BLOCK_M * DKV::D_STRIDE ];
                union {
                    alignas(16) __half dO[ DKV::BLOCK_N * DKV::D_STRIDE ];
                    alignas(16) __half q [ DKV::BLOCK_N * DKV::D_STRIDE ];
                } reuse_qdO;
                union {
                    alignas(16) float  s [ DKV::BLOCK_N * DKV::M_STRIDE ];
                    alignas(16) __half p [ DKV::BLOCK_N * DKV::BLOCK_M ];
                } reuse_sp;
                union {
                    alignas(16) float  dOV[ DKV::BLOCK_N * DKV::M_STRIDE ];
                    alignas(16) __half dS [ DKV::BLOCK_N * DKV::BLOCK_M ];
                } reuse_dOVS;
                    alignas(16) float dK[ DKV::BLOCK_M * DKV::D_STRIDE ];
                    alignas(16) float dV[ DKV::BLOCK_M * DKV::D_STRIDE ];
                } dkv;
        } phase;
                    alignas(16) float lse     [ (DQ::BLOCK_M > DKV::BLOCK_N) ? DQ::BLOCK_M : DKV::BLOCK_N ];
                    alignas(16) float row_dot [ (DQ::BLOCK_M > DKV::BLOCK_N) ? DQ::BLOCK_M : DKV::BLOCK_N ];
    };

    static constexpr size_t TOTAL_SMEM = ((sizeof(SmemLayout) + 127) & ~size_t(127));
};
