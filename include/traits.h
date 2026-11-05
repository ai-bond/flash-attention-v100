// ======================================================================================
// * Copyright (c) 2026, D.Skryabin / tg @ai_bond007 SPDX-License: BSD-3-Clause
// ======================================================================================
#pragma once

#include <cuda.h>
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <array>
#include <type_traits>

#ifdef MMA_NATIVE
    #include <mma.h>
    using namespace nvcuda::wmma;
#elif defined(MMA_884)
    #include "mma_m8n8k4.h"
    using namespace volta::wmma;
#else
    #include "mma_m16n16k16.h"
    using namespace volta::wmma;
#endif

// ======================================================================================
// GEMM OPERATION TYPE
// ======================================================================================
enum class GemmType : uint8_t {
    Q_KT,        // Q         @ K^T    (Query @ Key transpose)
    dO_VT,       // dO        @ V^T    (dOutput @ Value transpose)
    P_V,         // P         @ V      (Probability @ Value)
    dS_K,        // dS        @ K      (dScore @ Key)
    PT_dO,       // P^T       @ dO     (Probability transpose @ dOutput)
    dST_Q,       // dS^T      @ Q      (dScore transpose @ Query)
    dL_W,        // dLogits   @ Weight
    dLT_I,       // dLogits^T @ Input
};

// ======================================================================================
// GEMM PHASE
// ======================================================================================
enum class GemmPhase : uint8_t {
    FWD,         // Forward pass       (Q-centric:  BLOCK_M × BLOCK_N)
    BWQ,         // Backward dQ phase  (Q-centric:  BLOCK_M × BLOCK_N)
    BWKV,        // Backward dKV phase (KV-centric: BLOCK_N × BLOCK_M)
    LCDX         // LCE dX / dW        (A @ B, B = [K, BLOCK_N], smem row stride = BLOCK_N)
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
    SNGL,        // Single tile load (K/V or Q/dO loop loads)
    TRNSP        // Narrow single load
};

// ======================================================================================
// GEMM OPERATION TRAITS
// ======================================================================================
template<GemmType Option, GemmPhase Phase>
struct GemmTraits {
    static constexpr bool A_IS_COL   = (Option == GemmType::PT_dO || Option == GemmType::dST_Q || Option == GemmType::dLT_I);
    static constexpr bool B_IS_COL   = (Option == GemmType::Q_KT  || Option == GemmType::dO_VT);
    static constexpr bool APPLY_MASK = (Option == GemmType::Q_KT);
};

// ======================================================================================
// PHASE TRAITS
// ======================================================================================
template<GemmPhase Phase>
struct PhaseTraits {
    static constexpr bool IS_SWAPPED = (Phase == GemmPhase::BWKV);
    static constexpr bool WRITE_P    = IS_SWAPPED;

    template<typename Config>
    static constexpr int BLOCK_X() {
        if constexpr (Phase == GemmPhase::FWD) {
            return IS_SWAPPED ? Config::DO::BLOCK_N  : Config::DO::BLOCK_M;
        } else if constexpr (Phase == GemmPhase::BWQ) {
            return IS_SWAPPED ? Config::DQ::BLOCK_N  : Config::DQ::BLOCK_M;
        } else {
            return IS_SWAPPED ? Config::DKV::BLOCK_N : Config::DKV::BLOCK_M;
        }
    }

    template<typename Config>
    static constexpr int BLOCK_Y() {
        if constexpr (Phase == GemmPhase::FWD) {
            return IS_SWAPPED ? Config::DO::BLOCK_M  : Config::DO::BLOCK_N;
        } else if constexpr (Phase == GemmPhase::BWQ) {
            return IS_SWAPPED ? Config::DQ::BLOCK_M  : Config::DQ::BLOCK_N;
        } else {
            return IS_SWAPPED ? Config::DKV::BLOCK_M : Config::DKV::BLOCK_N;
        }
    }

    template<typename Config>
    static constexpr int STRIDE() { return BLOCK_Y<Config>(); }
};

// ======================================================================================
// LOAD TRAITS
// ======================================================================================
template<GemmPhase Phase, Stage Option>
struct LoadTraits {
    static constexpr bool DUAL_LOAD = ((Phase == GemmPhase::BWQ || Phase == GemmPhase::BWKV) && Option == Stage::DUAL);

    template<typename Config>
    static constexpr int TRANSPOSE() { if constexpr (Option == Stage::TRNSP) return Config::BLOCK_N; else return Config::DIM; }
};

// ======================================================================================
// EPILOGUE TRAITS
// ======================================================================================
template<Result Type>
struct EpilogueTraits {
    static constexpr bool WRITE_LSE  = (Type == Result::WRITE_O);
    static constexpr bool NORMALIZE  = (Type == Result::WRITE_O);
    static constexpr bool DUAL_STORE = (Type == Result::WRITE_DKV);
};

// ======================================================================================
// TILE TRAITS
// ======================================================================================
template<typename Config, GemmType TYPE, GemmPhase PHASE>
struct TileTraits {
    using GTraits = GemmTraits<TYPE, PHASE>;
    using PTraits = PhaseTraits<PHASE>;

    static constexpr bool IS_SCORES   = GTraits::B_IS_COL;
    static constexpr int  TILE_M      = IS_SCORES ? PTraits::template BLOCK_X<Config>()
                                                  : (PTraits::IS_SWAPPED ? PTraits::template BLOCK_Y<Config>()
                                                                         : PTraits::template BLOCK_X<Config>());
    static constexpr int  TILE_N      = IS_SCORES ? PTraits::template BLOCK_Y<Config>() : Config::DIM;
    static constexpr int  TILE_K      = IS_SCORES ? Config::DIM
                                                  : (PTraits::IS_SWAPPED ? PTraits::template BLOCK_X<Config>()
                                                                         : PTraits::template BLOCK_Y<Config>());
    static constexpr int  num_tiles_m = (TILE_M + WMMA_M - 1) / WMMA_M;
    static constexpr int  num_tiles_n = (TILE_N + WMMA_N - 1) / WMMA_N;
    static constexpr int  num_tiles_k = (TILE_K + WMMA_K - 1) / WMMA_K;
    static constexpr int  total_mn    = num_tiles_m * num_tiles_n;
    static constexpr int  tiles       = (total_mn + Config::WARPS_PER_BLOCK - 1) / Config::WARPS_PER_BLOCK;
    static constexpr int  rows        = (num_tiles_m / Config::WARPS_PER_BLOCK) * 2;
#ifdef MMA_NATIVE
    using accum_t       = nvcuda::wmma::fragment<nvcuda::wmma::accumulator, WMMA_M, WMMA_N, WMMA_K, float>;
    using matrixa_row_t = nvcuda::wmma::fragment<nvcuda::wmma::matrix_a,    WMMA_M, WMMA_N, WMMA_K, __half, nvcuda::wmma::row_major>;
    using matrixa_col_t = nvcuda::wmma::fragment<nvcuda::wmma::matrix_a,    WMMA_M, WMMA_N, WMMA_K, __half, nvcuda::wmma::col_major>;
    using matrixb_row_t = nvcuda::wmma::fragment<nvcuda::wmma::matrix_b,    WMMA_M, WMMA_N, WMMA_K, __half, nvcuda::wmma::row_major>;
    using matrixb_col_t = nvcuda::wmma::fragment<nvcuda::wmma::matrix_b,    WMMA_M, WMMA_N, WMMA_K, __half, nvcuda::wmma::col_major>;
#else
    using accum_t       = volta::wmma::fragment<volta::wmma::accumulator, WMMA_M, WMMA_N, WMMA_K, float>;
    using matrixa_row_t = volta::wmma::fragment<volta::wmma::matrix_a,    WMMA_M, WMMA_N, WMMA_K, __half, volta::wmma::row_major>;
    using matrixa_col_t = volta::wmma::fragment<volta::wmma::matrix_a,    WMMA_M, WMMA_N, WMMA_K, __half, volta::wmma::col_major>;
    using matrixb_row_t = volta::wmma::fragment<volta::wmma::matrix_b,    WMMA_M, WMMA_N, WMMA_K, __half, volta::wmma::row_major>;
    using matrixb_col_t = volta::wmma::fragment<volta::wmma::matrix_b,    WMMA_M, WMMA_N, WMMA_K, __half, volta::wmma::col_major>;
#endif
};

// ======================================================================================
// WMMA_GEMM_INIT_BUFFER
// ======================================================================================
template<typename Config, GemmType TYPE, GemmPhase PHASE>
struct WMMA_GEMM_INIT_BUFFER {
    using TTraits = TileTraits<Config, TYPE, PHASE>;
    static constexpr int tiles = TTraits::rows;

    float frags[tiles];

    __device__ __forceinline__ WMMA_GEMM_INIT_BUFFER(float fill) {
        #pragma unroll
        for (int i = 0; i < tiles; ++i) frags[i] = fill;
    }
};

// ======================================================================================
// WMMA_GEMM_INIT_ACCUM
// ======================================================================================
template<typename Config, GemmType TYPE, GemmPhase PHASE>
struct WMMA_GEMM_INIT_ACCUM {
    using TTraits = TileTraits<Config, TYPE, PHASE>;
    static constexpr int tiles = TTraits::tiles;

    typename TTraits::accum_t frags[tiles];

    __device__ __forceinline__ WMMA_GEMM_INIT_ACCUM(float fill = 0.0f) {
        if constexpr (!TTraits::IS_SCORES) {
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
// WMMA_GEMM_INIT_MATRIX
// ======================================================================================
template<typename Config, GemmType TYPE, GemmPhase PHASE>
struct WMMA_GEMM_INIT_MATRIX {
    using TTraits = TileTraits<Config, GemmType::Q_KT, PHASE>;
    using GTraits = GemmTraits<TYPE, PHASE>;
    using frag_t  = std::conditional_t<GTraits::A_IS_COL, typename TTraits::matrixa_col_t, typename TTraits::matrixa_row_t>;

    static constexpr int tiles = TTraits::tiles;

    frag_t  storage [tiles];
    frag_t (&frags) [tiles];

    template<typename Owner>
    __device__ __forceinline__ explicit WMMA_GEMM_INIT_MATRIX(Owner& owner) : frags(*reinterpret_cast<frag_t(*)[tiles]>(&owner.frags)) {}
    __device__ __forceinline__ WMMA_GEMM_INIT_MATRIX() : frags(storage) {}
};
