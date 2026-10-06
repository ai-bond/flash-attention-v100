// ======================================================================================
// * Copyright (c) 2026, D.Skryabin / tg @ai_bond007 SPDX-License: BSD-3-Clause
// ======================================================================================
#pragma once

// ======================================================================================
// VOLTA SM70 WMMA CONSTANTS
// ======================================================================================

#ifdef MMA_884
    #define WMMA_M 8
    #define WMMA_N 8
    #define WMMA_K 4
#else
    #define WMMA_M 16
    #define WMMA_N 16
    #define WMMA_K 16
#endif

#define NEG_INF                 (-1e30f)

#define MAX_THREADS_PER_WARP    32
#define MAX_THREADS_PER_SM      2048
#define MAX_THREAD_BLOCK_SIZE   1024
#define MAX_THREAD_BLOCK_PER_SM 32
#define MAX_WARPS_PER_SM        64
#define MAX_SM_PER_GPU          80
#define MAX_SMEM_PER_SM         98304

#define WARP_ALLOC_GROUP        4

#define MAX_REG_PER_UNIT        256
#define MAX_REG_PER_THREAD      255
#define MAX_REG_PER_BLOCK       65536
#define MAX_REG_BUFFER          65536

// ======================================================================================
// GEMM OPERATION TYPE
// ======================================================================================
enum class GemmType : uint8_t {
    Q_KT,        // Q    @ K^T  (Query @ Key transpose)
    dO_VT,       // dO   @ V^T  (dOutput @ Value transpose)
    P_V,         // P    @ V    (Probability @ Value)
    dS_K,        // dS   @ K    (dScore @ Key)
    PT_dO,       // P^T  @ dO   (Probability transpose @ dOutput)
    dST_Q,       // dS^T @ Q    (dScore transpose @ Query)
};

// ======================================================================================
// GEMM PHASE
// ======================================================================================
enum class GemmPhase : uint8_t {
    FWD,         // Forward pass       (Q-centric: BLOCK_M × BLOCK_N)
    BWQ,         // Backward dQ phase  (Q-centric: BLOCK_M × BLOCK_N)
    BWKV         // Backward dKV phase (KV-centric: BLOCK_N × BLOCK_M)
};

// ======================================================================================
// EPILOGUE RESULT TYPE
// ======================================================================================
enum class Result : uint8_t {
    WRITE_O,     // Forward:  normalize + store O (divide by row_sum)
    WRITE_DQ,    // Backward: store dQ (no normalization)
    WRITE_DKV    // Backward: dual store dK+dV (no normalization)
};

// ======================================================================================
// TILE LOAD STAGE
// ======================================================================================
enum class Stage : uint8_t {
    DUAL,        // Dual   tile load (Q+dO or K+V initial load)
    SNGL         // Single tile load (K/V or Q/dO loop loads)
};

// ======================================================================================
// GEMM OPERATION TRAITS
// ======================================================================================
template<GemmType Option>
struct GemmTraits {
    static constexpr bool A_IS_COL   = (Option == GemmType::PT_dO || Option == GemmType::dST_Q);
    static constexpr bool B_IS_COL   = (Option == GemmType::Q_KT  || Option == GemmType::dO_VT);
    static constexpr bool APPLY_MASK = (Option == GemmType::Q_KT);
};

// ======================================================================================
// PHASE TRAITS
// ======================================================================================
template<GemmPhase Phase>
struct PhaseTraits {
    static constexpr bool IS_SWAPPED = (Phase == GemmPhase::BWKV);
    static constexpr int  LSE_OFFSET = IS_SWAPPED ? 1 : 0;
    static constexpr bool IS_SDS_SP  = IS_SWAPPED;

    template<typename Config>
    static constexpr int BLOCK_X() {
        if constexpr (Phase == GemmPhase::FWD) {
            return IS_SWAPPED ? Config::DO::BLOCK_N :  Config::DO::BLOCK_M;
        } else if constexpr (Phase == GemmPhase::BWQ) {
            return IS_SWAPPED ? Config::DQ::BLOCK_N :  Config::DQ::BLOCK_M;
        } else {
            return IS_SWAPPED ? Config::DKV::BLOCK_N : Config::DKV::BLOCK_M;
        }
    }

    template<typename Config>
    static constexpr int BLOCK_Y() {
        if constexpr (Phase == GemmPhase::FWD) {
            return IS_SWAPPED ? Config::DO::BLOCK_M :  Config::DO::BLOCK_N;
        } else if constexpr (Phase == GemmPhase::BWQ) {
            return IS_SWAPPED ? Config::DQ::BLOCK_M :  Config::DQ::BLOCK_N;
        } else {
            return IS_SWAPPED ? Config::DKV::BLOCK_M : Config::DKV::BLOCK_N;
        }
    }

    template<typename Config>
    static constexpr int LDS_STRIDE() { return BLOCK_Y<Config>(); }

    template<typename Config>
    static constexpr int LDO_STRIDE() { return BLOCK_Y<Config>() * 2; }
};

// ======================================================================================
// LOAD TRAITS
// ======================================================================================
template<GemmPhase Phase, Stage Option>
struct LoadTraits {
    static constexpr bool DUAL_LOAD = ((Phase == GemmPhase::BWQ || Phase == GemmPhase::BWKV) && Option == Stage::DUAL);
};

// ======================================================================================
// EPILOGUE TRAITS
// ======================================================================================
template<Result Type>
struct EpilogueTraits {
    static constexpr bool NORMALIZE  = (Type == Result::WRITE_O);
    static constexpr bool DUAL_STORE = (Type == Result::WRITE_DKV);
};
