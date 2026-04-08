#pragma once

// ============================================================================
// VOLTA SM70 WMMA CONSTANTS
// ============================================================================
#define WMMA_M 16
#define WMMA_N 16
#define WMMA_K 16

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

// ============================================================================
// GEMM OPERATION
// Bit 0 (0x1): CONTEXT-DEPENDENT FLAG
//  * In WMMA_GEMM_SCORES:    APPLY_MASK (1=apply causal mask to output, 0=no mask)
//  * In WMMA_GEMM_GRADIENTS: ACCUMULATE (1=C += A@B, 0=C = A@B / overwrite)
// Bit 1 (0x2): A_IS_COL   (1=load A as col_major / interpret as A^T, 0=row_major)
// Bit 2 (0x4): B_IS_COL   (1=load B as col_major / interpret as B^T, 0=row_major)
// ============================================================================
enum class GemmType : uint8_t {
    sQ_KT        = (1<<0) | (0<<1) | (1<<2),  // 0b101 = 5:    Q(row) @  K(col)^T, APPLY_MASK=1, A=row, B=col
    dOV_dOVT     = (0<<0) | (0<<1) | (1<<2),  // 0b100 = 4:   dO(row) @  V(col)^T, APPLY_MASK=0, A=row, B=col
    dO_PV        = (1<<0) | (0<<1) | (0<<2),  // 0b001 = 1:    P(row) @  V(col)^T, ACCUMULATE=1, A=row, B=col
    dQ_dSK       = (1<<0) | (0<<1) | (0<<2),  // 0b001 = 1:   dS(row) @  K(row),   ACCUMULATE=1, A=row, B=row
    dV_PTdO      = (1<<0) | (1<<1) | (0<<2),  // 0b011 = 3:  P^T(col) @ dO(row),   ACCUMULATE=1, A=col, B=row
    dK_dSTQ      = (1<<0) | (1<<1) | (0<<2),  // 0b011 = 3: dS^T(col) @  Q(row),   ACCUMULATE=1, A=col, B=row
    rowdot_dQ    = (0<<0) | (0<<1) | (0<<2),  // 0b000 = 0: LSE_OFFSET=0
    rowdot_dKV   = (1<<0) | (0<<1) | (0<<2),  // 0b011 = 3: LSE_OFFSET=1
    compute_dS   = (0<<0) | (0<<1) | (0<<2),  // 0b000 = 0: IS_SDS_SP=0  dQ
    compute_P_dS = (1<<0) | (0<<1) | (0<<2),  // 0b001 = 1: IS_SDS_SP=1  dKV
    write_dO     = (1<<0) | (0<<1) | (0<<2),  // 0b001 = 1: NORMALIZE=1, DUAL_OUTPUT=0 (forward  O)
    write_dQ     = (0<<0) | (0<<1) | (0<<2),  // 0b000 = 0: NORMALIZE=0, DUAL_OUTPUT=0 (backward dQ)
    write_dKV    = (0<<0) | (1<<1) | (0<<2),  // 0b010 = 2: NORMALIZE=0, DUAL_OUTPUT=1 (backward dK+dV)
};
