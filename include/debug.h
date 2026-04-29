// ======================================================================================
// * Copyright (c) 2026, D.Skryabin / tg @ai_bond007 SPDX-License: BSD-3-Clause
// ======================================================================================
#pragma once

#include <cstdint>
#include <cstdio>
#include <cuda_fp16.h>

#ifndef KERNEL_DEBUG
    #define KERNEL_DEBUG 0
#endif

#if KERNEL_DEBUG
    #define __ASM_MARK(STG, CTX, TYPE, MAGIC) \
        do { \
            volatile unsigned int point = 0; \
            asm volatile("mov.u32 %0, " #MAGIC "; // DBG_PTX_" #STG "_" #CTX "_" #TYPE "\n\t" \
                         : "+r"(point) :: "memory"); \
            (void)point; \
        } while(0)

    #define __ASM_DEBUG_BEGIN(STG, CTX) __ASM_MARK(STG, CTX, BEGIN, 0xBEEF0001)
    #define __ASM_DEBUG_END(STG, CTX)   __ASM_MARK(STG, CTX, END,   0xCAFE0002)

enum DebugType : uint8_t {
    SMEM     = 0, TILE     = 1, SQKT     = 2, DOVT     = 3, DOPV     = 4,
    DQDSK    = 5, DVPTDO   = 6, DKDSTQ   = 7, SOFTMAX  = 8, ROWDQ    = 9,
    ROWDKV   = 10, WRITEO  = 11, WRITEQ  = 12, WRITEKV = 13, NONE    = 14
};

__device__ __forceinline__ const char* stage_name(DebugType stage) {
    switch(stage) {
        case SMEM:    return "SMEM";    case TILE:    return "TILE";
        case SQKT:    return "SQKT";    case DOVT:    return "DOVT";
        case DOPV:    return "DOPV";    case DQDSK:   return "DQDSK";
        case DVPTDO:  return "DVPTDO";  case DKDSTQ:  return "DKDSTQ";
        case SOFTMAX: return "SOFTMAX"; case ROWDQ:   return "ROWDQ";
        case ROWDKV:  return "ROWDKV";  case WRITEO:  return "WRITEO";
        case WRITEQ:  return "WRITEQ";  case WRITEKV: return "WRITEKV";
        case NONE:    return "NONE";    default:      return "????";
    }
}

template<typename T> struct element_type { using type = T; };
template<typename T> struct element_type<T[]> { using type = T; };
template<typename T, size_t N> struct element_type<T[N]> { using type = T; };
template<typename T> using element_type_t = typename element_type<T>::type;

template<typename Layout, typename FieldTag, typename = void>
struct field_info {};

#define DEFINE_FIELD(FIELD_NAME, FIELD_PATH) \
    template<typename T, typename = void> \
    struct has_##FIELD_NAME : std::false_type {}; \
    template<typename T> \
    struct has_##FIELD_NAME<T, std::void_t<decltype(std::declval<T>().FIELD_PATH)>> : std::true_type {}; \
    struct TAG_##FIELD_NAME {}; \
    template<typename Layout> \
    struct field_info<Layout, TAG_##FIELD_NAME> { \
        using type = element_type_t<decltype(std::declval<Layout>().FIELD_PATH)>; \
        __device__ static constexpr size_t get_offset() { \
            return offsetof(Layout, FIELD_PATH); \
        } \
    };

// Forward fields
DEFINE_FIELD(q_fwd,       phase.fdo.q)
DEFINE_FIELD(k_fwd,       phase.fdo.reuse_kv.k)
DEFINE_FIELD(v_fwd,       phase.fdo.reuse_kv.v)
DEFINE_FIELD(s_fwd,       phase.fdo.reuse_sp.s)
DEFINE_FIELD(p_fwd,       phase.fdo.reuse_sp.p)
DEFINE_FIELD(row_max,     row_max)
DEFINE_FIELD(row_sum,     row_sum)
DEFINE_FIELD(o_fwd,       smem.phase.fdo.o)

// Backward dQ fields
DEFINE_FIELD(q_dq,        phase.bdq.q)
DEFINE_FIELD(k_dq,        phase.bdq.reuse_kv.k)
DEFINE_FIELD(v_dq,        phase.bdq.reuse_kv.v)
DEFINE_FIELD(s_dq,        phase.bdq.s)
DEFINE_FIELD(dO_dq,       phase.bdq.dO)
DEFINE_FIELD(dOV_dq,      phase.bdq.reuse_sdOVS.dOV)
DEFINE_FIELD(dS_dq,       phase.bdq.reuse_sdOVS.dS)
DEFINE_FIELD(dQ_dq,       phase.bdq.dQ)

// Backward dKV fields
DEFINE_FIELD(k_dkv,       phase.bdkv.k)
DEFINE_FIELD(v_dkv,       phase.bdkv.v)
DEFINE_FIELD(q_dkv,       phase.bdkv.reuse_qdO.q)
DEFINE_FIELD(dO_dkv,      phase.bdkv.reuse_qdO.dO)
DEFINE_FIELD(s_dkv,       phase.bdkv.reuse_sp.s)
DEFINE_FIELD(p_dkv,       phase.bdkv.reuse_sp.p)
DEFINE_FIELD(dS_dkv,      phase.bdkv.reuse_dOVS.dS)
DEFINE_FIELD(dOV_dkv,     phase.bdkv.reuse_dOVS.dOV)
DEFINE_FIELD(dK_dkv,      phase.bdkv.dK)
DEFINE_FIELD(dV_dkv,      phase.bdkv.dV)

// Common vectors
DEFINE_FIELD(lse,         lse)
DEFINE_FIELD(row_dot,     row_dot)
#undef DEFINE_FIELD

__device__ __forceinline__ void _print_val(float v) {
    if (isnan(v))             printf("    nan ");
    else if (isinf(v) || v <= -1e20f) printf("   -inf ");
    else                      printf("%7.3f ", v);
}

// ======================================================================================
// __CHECK_INIT: Validate SMEM zero-fill
// ======================================================================================
#define __CHECK_INIT(FIELD_TAG, EXPECTED_VAL, VALID_ROWS) \
do { \
    extern __shared__ char smem_raw[]; \
    uint32_t base = static_cast<uint32_t>(__cvta_generic_to_shared(smem_raw)); \
    using Layout = typename Config::SmemLayout; \
    constexpr bool has_field = has_##FIELD_TAG<Layout>::value; \
    if constexpr (has_field) { \
        using Info = field_info<Layout, TAG_##FIELD_TAG>; \
        using FieldType = typename Info::type; \
        bool mismatch = false; \
        if (tid < (VALID_ROWS)) { \
            size_t offset = Info::get_offset(); \
            uint32_t elem_addr = base + static_cast<uint32_t>(offset) + (tid * sizeof(FieldType)); \
            if constexpr (std::is_same<FieldType, __half>::value) { \
                uint16_t val_bits, exp_bits; \
                asm volatile("ld.shared.u16 %0, [%1];" : "=h"(val_bits) : "r"(elem_addr)); \
                union { __half h; uint16_t u; } conv; conv.h = __float2half_rn(EXPECTED_VAL); exp_bits = conv.u; \
                mismatch = (val_bits != exp_bits); \
            } else if constexpr (std::is_same<FieldType, float>::value) { \
                uint32_t val_bits, exp_bits; \
                asm volatile("ld.shared.u32 %0, [%1];" : "=r"(val_bits) : "r"(elem_addr)); \
                union { float f; uint32_t u; } conv; conv.f = EXPECTED_VAL; exp_bits = conv.u; \
                mismatch = (val_bits != exp_bits); \
            } \
            if (mismatch && (tid & 31) == 0) { \
                printf("[DBG_ERR][B%d][INIT]: " #FIELD_TAG "[%d] MISMATCH @ 0x%x\n", blockIdx.x, tid, elem_addr); \
            } \
        } \
    } \
} while(0)

// ======================================================================================
// __CHECK_ERRORS: Stage-aware inf/nan scan
// ======================================================================================
#define __CHECK_ERRORS(STG, VM, VN, SC, WID, LID, TID) \
do { \
    if ((TID) == 0 && (blockIdx.x == 0 && blockIdx.z == 0)) { \
        extern __shared__ char smem_raw[]; \
        auto& smem = *reinterpret_cast<typename Config::SmemLayout*>(smem_raw); \
        using Layout = typename Config::SmemLayout; \
        bool found_err = false; \
        int rows = (VM) < WMMA_M ? (VM) : WMMA_M; \
        int cols = (VN) < WMMA_N ? (VN) : WMMA_N; \
        for (int r = 0; r < rows; ++r) { \
            for (int c = 0; c < cols; ++c) { \
                float val = 0.f; \
                switch(STG) { \
                    case SQKT: case DOVT: \
                        if constexpr (has_s_fwd<Layout>::value) val = smem.phase.fdo.reuse_sp.s[r * (SC) + c]; \
                        else if constexpr (has_s_dq<Layout>::value) val = smem.phase.bdq.s[r * (SC) + c]; \
                        else if constexpr (has_s_dkv<Layout>::value) val = smem.phase.bdkv.reuse_sp.s[r * (SC) + c]; \
                        break; \
                    case SOFTMAX: \
                        if constexpr (has_p_fwd<Layout>::value) val = __half2float(smem.phase.fdo.reuse_sp.p[r * (SC) + c]); \
                        else if constexpr (has_p_dkv<Layout>::value) val = __half2float(smem.phase.bdkv.reuse_sp.p[r * (SC) + c]); \
                        break; \
                    case DQDSK: case DVPTDO: case DKDSTQ: \
                        if constexpr (has_dS_dq<Layout>::value) val = __half2float(smem.phase.bdq.reuse_sdOVS.dS[r * (SC) + c]); \
                        else if constexpr (has_dS_dkv<Layout>::value) val = __half2float(smem.phase.bdkv.reuse_dOVS.dS[r * (SC) + c]); \
                        break; \
                    case DOPV: case WRITEO: \
                        if constexpr (has_o_fwd<Layout>::value) val = smem.phase.fdo.o[r * (SC) + c]; \
                        break; \
                    case WRITEQ: \
                        if constexpr (has_dQ_dq<Layout>::value) val = smem.phase.bdq.dQ[r * (SC) + c]; \
                        break; \
                    case WRITEKV: \
                        if constexpr (has_dK_dkv<Layout>::value) val = smem.phase.bdkv.dK[r * (SC) + c]; \
                        else if constexpr (has_dV_dkv<Layout>::value) val = smem.phase.bdkv.dV[r * (SC) + c]; \
                        break; \
                    default: break; \
                } \
                if (isnan(val) || isinf(val)) { \
                    printf("[DBG_ERR][B%d][%s][%d,%d]: inf/nan\n", blockIdx.x, stage_name(STG), r, c); \
                    found_err = true; \
                } \
            } \
        } \
        if (!found_err) printf("[DBG_OK ][B%d][%s] tile checked\n", blockIdx.x, stage_name(STG)); \
    } \
} while(0)

// ======================================================================================
// __PRINT_MATRIX: 2D tile dump
// ======================================================================================
#define __PRINT_MATRIX(STG, VM, VN, SC, WID, LID, TID, TILE_IDX) \
do { \
    if ((TID) == 0 && (WID) == 0 && (blockIdx.x == 0 && blockIdx.z == 0)) { \
        extern __shared__ char smem_raw[]; \
        auto& smem = *reinterpret_cast<typename Config::SmemLayout*>(smem_raw); \
        using Layout = typename Config::SmemLayout; \
        printf("[DBG_MAT][B%d][T%d][%s] tile[%dx%d]:\n", blockIdx.x, TILE_IDX, stage_name(STG), VM, VN); \
        int print_rows = (VM) < WMMA_M ? (VM) : WMMA_M; \
        int print_cols = (VN) < WMMA_N ? (VN) : WMMA_N; \
        for (int r = 0; r < print_rows; ++r) { \
            printf("  row %2d: ", r); \
            for (int c = 0; c < print_cols; ++c) { \
                float v = 0.f; \
                switch(STG) { \
                    case SQKT: case DOVT: \
                        if constexpr (has_s_fwd<Layout>::value) v = smem.phase.fdo.reuse_sp.s[r * (SC) + c]; \
                        else if constexpr (has_s_dq<Layout>::value) v = smem.phase.bdq.s[r * (SC) + c]; \
                        break; \
                    case DQDSK: \
                        if constexpr (has_s_dq<Layout>::value) v = smem.phase.bdq.s[r * (SC) + c]; \
                        break; \
                    case DVPTDO: \
                        if constexpr (has_dOV_dq<Layout>::value) v = __half2float(smem.phase.bdq.reuse_sdOVS.dOV[r * (SC) + c]); \
                        else if constexpr (has_dOV_dkv<Layout>::value) v = __half2float(smem.phase.bdkv.reuse_dOVS.dOV[r * (SC) + c]); \
                        break; \
                    case DKDSTQ: case DOPV: \
                        if constexpr (has_dQ_dq<Layout>::value) v = smem.phase.bdq.dQ[r * (SC) + c]; \
                        else if constexpr (has_o_fwd<Layout>::value) v = smem.phase.fdo.o[r * (SC) + c]; \
                        break; \
                    default: break; \
                } \
                _print_val(v); \
            } \
            printf("\n"); \
        } \
    } \
} while(0)

// ======================================================================================
// __PRINT_RESULT: 1D vector/scalar dump
// ======================================================================================
#define __PRINT_RESULT(FIELD_TAG, VLEN, TILE_IDX) \
do { \
    if (tid == 0 && _DEBUG_GLOBAL_GUARD()) { \
        extern __shared__ char smem_raw[]; \
        using Layout = typename Config::SmemLayout; \
        constexpr bool has_field = has_##FIELD_TAG<Layout>::value; \
        if constexpr (has_field) { \
            using Info = field_info<Layout, TAG_##FIELD_TAG>; \
            using FieldType = typename Info::type; \
            uint32_t base = static_cast<uint32_t>(__cvta_generic_to_shared(smem_raw)); \
            uint32_t vec_addr = base + static_cast<uint32_t>(Info::get_offset()); \
            printf("[DBG_VEC][B%d][T%d][" #FIELD_TAG "] len=%d: ", blockIdx.x, TILE_IDX, VLEN); \
            int print_len = (VLEN) < WMMA_M ? (VLEN) : WMMA_M; \
            for (int i = 0; i < print_len; ++i) { \
                float v = 0.f; \
                if constexpr (std::is_same<FieldType, float>::value) { \
                    uint32_t bits; \
                    uint32_t elem_addr = vec_addr + (uint32_t)(i * sizeof(float)); \
                    asm volatile("ld.shared.u32 %0, [%1];" : "=r"(bits) : "r"(elem_addr)); \
                    v = *reinterpret_cast<float*>(&bits); \
                } else if constexpr (std::is_same<FieldType, __half>::value) { \
                    uint16_t bits; \
                    uint32_t elem_addr = vec_addr + (uint32_t)(i * sizeof(__half)); \
                    asm volatile("ld.shared.u16 %0, [%1];" : "=h"(bits) : "r"(elem_addr)); \
                    v = __half2float(*reinterpret_cast<__half*>(&bits)); \
                } \
                _print_val(v); \
            } \
            printf("\n"); \
        } \
    } \
} while(0)

#else
    #define __ASM_DEBUG_BEGIN(STG, CTX) ((void)0)
    #define __ASM_DEBUG_END(STG, CTX)   ((void)0)
    #define __CHECK_INIT(FIELD_TAG, EXPECTED_VAL, VALID_ROWS)
    #define __CHECK_ERRORS(STG, VM, VN, SC, WID, LID, TID)
    #define __PRINT_MATRIX(STG, VM, VN, SC, WID, LID, TID, TILE_IDX)
    #define __PRINT_RESULT(FIELD_TAG, VLEN, TILE_IDX)
#endif
