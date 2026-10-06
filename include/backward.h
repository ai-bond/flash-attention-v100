// ======================================================================================
// * Copyright (c) 2026, D.Skryabin / tg @ai_bond007 SPDX-License: BSD-3-Clause
// ======================================================================================
#pragma once

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

// ======================================================================================
// CONFIGURATION DQ/KV
// ======================================================================================
template<int D>
struct KernelConfig {
    struct DQ {
        static constexpr int BLOCK_M         = (D == 16) ? BLOCK_M_DQ_16 : (D == 32) ? BLOCK_M_DQ_32 : (D == 64) ? BLOCK_M_DQ_64 : (D == 128) ? BLOCK_M_DQ_128 : BLOCK_M_DQ_256;
        static constexpr int BLOCK_N         = (D == 16) ? BLOCK_N_DQ_16 : (D == 32) ? BLOCK_N_DQ_32 : (D == 64) ? BLOCK_N_DQ_64 : (D == 128) ? BLOCK_N_DQ_128 : BLOCK_N_DQ_256;
        static constexpr int THREADS_PER_ROW = (WARPS * MAX_THREADS_PER_WARP) / BLOCK_M;
    };
    struct DKV {
        static constexpr int BLOCK_M         = (D == 16) ? BLOCK_M_DKV_16 : (D == 32) ? BLOCK_M_DKV_32 : (D == 64) ? BLOCK_M_DKV_64 : (D == 128) ? BLOCK_M_DKV_128 : BLOCK_M_DKV_256;
        static constexpr int BLOCK_N         = (D == 16) ? BLOCK_N_DKV_16 : (D == 32) ? BLOCK_N_DKV_32 : (D == 64) ? BLOCK_N_DKV_64 : (D == 128) ? BLOCK_N_DKV_128 : BLOCK_N_DKV_256;
        static constexpr int THREADS_PER_ROW = (WARPS * MAX_THREADS_PER_WARP) / BLOCK_N;
    };

    static constexpr int DIM                 = D;
    static constexpr int WARPS_PER_BLOCK     = WARPS;
    static constexpr int THREADS_PER_BLOCK   = WARPS * MAX_THREADS_PER_WARP;

    struct alignas(128) SmemLayout {
        union PhaseMem {
            struct DQ_Phase {
                union {
                    alignas(16) __half k  [ DQ::BLOCK_N * D ];
                    alignas(16) __half v  [ DQ::BLOCK_N * D ];
                } reuse_kv;
                    alignas(16) __half dO [ DQ::BLOCK_M * D ];
                    alignas(16) __half q  [ DQ::BLOCK_M * D ];
                    alignas(16) float  s  [ DQ::BLOCK_M * DQ::BLOCK_N ];
                union {
                    alignas(16) float  dOV[ DQ::BLOCK_M * DQ::BLOCK_N ];
                    alignas(16) __half dS [ DQ::BLOCK_M * DQ::BLOCK_N * 2];
                } reuse_sdOVS;
                    alignas(16) float  dQ [ DQ::BLOCK_M * D ];
            } bdq;

            struct DKV_Phase {
                    alignas(16) __half k [ DKV::BLOCK_M * D ];
                    alignas(16) __half v [ DKV::BLOCK_M * D ];
                union {
                    alignas(16) __half dO[ DKV::BLOCK_N * D ];
                    alignas(16) __half q [ DKV::BLOCK_N * D ];
                } reuse_qdO;
                union {
                    alignas(16) float  s [ DKV::BLOCK_N * DKV::BLOCK_M ];
                    alignas(16) __half p [ DKV::BLOCK_N * DKV::BLOCK_M * 2];
                } reuse_sp;
                union {
                    alignas(16) float  dOV[ DKV::BLOCK_N * DKV::BLOCK_M ];
                    alignas(16) __half dS [ DKV::BLOCK_N * DKV::BLOCK_M * 2];
                } reuse_dOVS;
                    alignas(16) float dK[ DKV::BLOCK_M * D ];
                    alignas(16) float dV[ DKV::BLOCK_M * D ];
            } bdkv;
        } phase;
                    alignas(16) float lse     [ (DQ::BLOCK_M > DKV::BLOCK_N) ? DQ::BLOCK_M : DKV::BLOCK_N ];
                    alignas(16) float row_dot [ (DQ::BLOCK_M > DKV::BLOCK_N) ? DQ::BLOCK_M : DKV::BLOCK_N ];
    };

    static constexpr int TOTAL_SMEM = static_cast<int>(((sizeof(SmemLayout) + 127) & ~size_t(127)));
};
