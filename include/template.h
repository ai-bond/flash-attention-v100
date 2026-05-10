// ======================================================================================
// * Copyright (c) 2026, D.Skryabin / tg @ai_bond007 SPDX-License: BSD-3-Clause
// ======================================================================================
#pragma once

#include <type_traits>

// ======================================================================================
// BlockInfo: Computes iteration bounds, offsets, and block for forward/backward kernels.
//   - Forward/dQ: Grid Y = B * H_Q  -> BATCH_HEAD_ID decodes via H_Q
//   - Backward dKV:
//       * Varlen: Grid Y = B * H_Q  -> H_Q, maps to KV-head internally
//       * Dense:  Grid Y = B * H_K  -> H_K directly
// ======================================================================================
template<bool IS_CAUSAL, bool IS_WINDOW, bool IS_VARLEN>
struct BlockInfo {
    int block_min;
    int block_max;
    int start_q;           // Q-block start position (forward/dQ phase)
    int valid_q_rows;      // Valid rows in Q-block (forward/dQ phase)
    int start_kv;          // KV-block start position (dKV phase)
    int valid_kv_rows;     // Valid rows in KV-block (dKV phase)
    int seqlen_q;
    int seqlen_k;
    int seqlen_offset;     // seqlen_k - seqlen_q
    int batch_idx;
    int head_idx;          // Q-head index
    int kv_head_idx;       // KV-head index
    int q_base;            // cu_seqlens_q[batch_idx] (varlen) or 0 (dense)
    int k_base;            // cu_seqlens_k[batch_idx] (varlen) or 0 (dense)

    // ======================================================================================
    // Initialize for Q-centric phases (Forward, dQ)
    // Fixes Q-block, iterates over K/V blocks.
    // ======================================================================================
    __device__ __forceinline__ void init_q(
        int BLOCK_IDX,
        int BATCH_HEAD_ID,
        int H_Q,
        int H_K,
        int M,
        int N,
        int B,
        int BLOCK_M,
        int BLOCK_N,
        int WINDOW_LEFT,
        int WINDOW_RIGHT,
        const int* CU_SEQLENS_Q,
        const int* CU_SEQLENS_K,
        const int* SEQUSED_K
    ) {
        start_kv      = 0;
        valid_kv_rows = 0;

        if constexpr (IS_VARLEN) {
            batch_idx   = BATCH_HEAD_ID / H_Q;
            head_idx    = BATCH_HEAD_ID % H_Q;
            kv_head_idx = head_idx / (H_Q / H_K);

            q_base      = CU_SEQLENS_Q[batch_idx];
            k_base      = CU_SEQLENS_K[batch_idx];
            seqlen_q    = CU_SEQLENS_Q[batch_idx + 1] - q_base;
            seqlen_k    = CU_SEQLENS_K[batch_idx + 1] - k_base;

            if (SEQUSED_K != nullptr) {
                int used_k = SEQUSED_K[batch_idx];
                seqlen_k = (used_k > 0) ? min(seqlen_k, used_k) : 0;
            }
            start_q = BLOCK_IDX * BLOCK_M;
        } else {
            batch_idx   = BATCH_HEAD_ID / H_Q;
            head_idx    = BATCH_HEAD_ID % H_Q;
            kv_head_idx = head_idx / (H_Q / H_K);
            seqlen_q    = M;
            seqlen_k    = N;
            q_base      = 0;
            k_base      = 0;
            start_q     = BLOCK_IDX * BLOCK_M;
        }

        if (start_q >= seqlen_q) {
            valid_q_rows = 0;
            return;
        }

        valid_q_rows  = min(BLOCK_M, seqlen_q - start_q);
        seqlen_offset = seqlen_k - seqlen_q;
        block_min     = 0;
        block_max     = (seqlen_k + BLOCK_N - 1) / BLOCK_N;

        // ==================================================================================
        // Trim K/V iteration range for causal and sliding window attention (forward/dQ phase)
        // Logic:    causal restricts K/V blocks beyond Q position (right bound)
        //           window_left restricts blocks before Q - window_left (left bound)
        //           window_right restricts blocks beyond Q + window_right (right bound)
        //           block_min/block_max define valid [start, end) K/V tile index range
        // ==================================================================================
        if constexpr (IS_CAUSAL) {
            int max_key_pos = start_q + valid_q_rows - 1 + seqlen_offset;
            block_max = (max_key_pos < 0) ? 0 : min(block_max, (max_key_pos / BLOCK_N) + 1);
        }
        if constexpr (IS_WINDOW) {
            if (WINDOW_LEFT >= 0) {
                int min_key_pos = start_q + seqlen_offset - WINDOW_LEFT;
                block_min = max(block_min, (min_key_pos > 0) ? (min_key_pos / BLOCK_N) : 0);
            }
            if (WINDOW_RIGHT >= 0) {
                int max_key_pos_win = start_q + valid_q_rows - 1 + seqlen_offset + WINDOW_RIGHT;
                block_max = min(block_max, (max_key_pos_win >= 0) ? (max_key_pos_win / BLOCK_N) + 1 : 0);
            }
        }
    }

    // ======================================================================================
    // Initialize for KV-centric phase (dKV backward)
    // Fixes K/V-block, iterates over Q blocks.
    // ======================================================================================
    __device__ __forceinline__ void init_kv(
        int BLOCK_IDX,
        int BATCH_HEAD_ID,
        int H_Q,
        int H_K,
        int M,
        int N,
        int B,
        int BLOCK_M,
        int BLOCK_N,
        int WINDOW_LEFT,
        int WINDOW_RIGHT,
        const int* CU_SEQLENS_Q,
        const int* CU_SEQLENS_K,
        const int* SEQUSED_K
    ) {
        start_q         = 0;
        valid_q_rows    = 0;

        if constexpr (IS_VARLEN) {
            batch_idx   = BATCH_HEAD_ID / H_Q;
            head_idx    = BATCH_HEAD_ID % H_Q;
            kv_head_idx = head_idx / (H_Q / H_K);
            q_base      = CU_SEQLENS_Q[batch_idx];
            k_base      = CU_SEQLENS_K[batch_idx];
            seqlen_q    = CU_SEQLENS_Q[batch_idx + 1] - q_base;
            seqlen_k    = CU_SEQLENS_K[batch_idx + 1] - k_base;

            if (SEQUSED_K != nullptr) {
                int used_k = SEQUSED_K[batch_idx];
                seqlen_k = (used_k > 0) ? min(seqlen_k, used_k) : 0;
            }
            start_kv = BLOCK_IDX * BLOCK_M;
        } else {
            batch_idx   = BATCH_HEAD_ID / H_K;
            kv_head_idx = BATCH_HEAD_ID % H_K;
            head_idx    = -1;
            seqlen_q    = M;
            seqlen_k    = N;
            q_base      = 0;
            k_base      = 0;
            start_kv    = BLOCK_IDX * BLOCK_M;
        }

        if (start_kv >= seqlen_k) {
            valid_kv_rows = 0;
            return;
        }

        valid_kv_rows = min(BLOCK_M, seqlen_k - start_kv);
        seqlen_offset = seqlen_k - seqlen_q;
        block_min     = 0;
        block_max     = (seqlen_q + BLOCK_N - 1) / BLOCK_N;

        // ==================================================================================
        // Trim Q-tile iteration range for causal and sliding window attention (dKV phase)
        // Logic:    in dKV phase, K/V positions are fixed (start_kv), Q tiles iterate
        //           causal restricts Q blocks before K - seqlen_offset (left bound)
        //           window_left restricts Q blocks beyond K + window_left (right bound)
        //           window_right restricts Q blocks before K - window_right (left bound)
        //           block_min/block_max define valid [start, end) Q-tile index range
        // ==================================================================================
        if constexpr (IS_CAUSAL) {
            int min_q_pos = start_kv - seqlen_offset;
            block_min = max(block_min, (min_q_pos > 0) ? (min_q_pos / BLOCK_N) : 0);
        }
        if constexpr (IS_WINDOW) {
            if (WINDOW_RIGHT >= 0) {
                int min_q_pos_win = start_kv - seqlen_offset - WINDOW_RIGHT;
                block_min = max(block_min, (min_q_pos_win > 0) ? (min_q_pos_win / BLOCK_N) : 0);
            }
            if (WINDOW_LEFT >= 0) {
                int max_q_pos_win = start_kv + valid_kv_rows - 1 - seqlen_offset + WINDOW_LEFT;
                block_max = min(block_max, (max_q_pos_win >= 0) ? (max_q_pos_win / BLOCK_N) + 1 : 0);
            }
        }
    }

    // ======================================================================================
    // Offset computation for Q tensor
    // ======================================================================================
    __device__ __forceinline__ size_t q_offset(int D, int H_Q, int M_or_TQ) const {
        if constexpr (IS_VARLEN) {
            return static_cast<size_t>(q_base + start_q) * H_Q * D + static_cast<size_t>(head_idx) * D;
        } else {
            int batch_head_id = batch_idx * H_Q + head_idx;
            return static_cast<size_t>(batch_head_id) * M_or_TQ * D + static_cast<size_t>(start_q) * D;
        }
    }

    // ======================================================================================
    // Offset computation for K/V tensors
    // ======================================================================================
    __device__ __forceinline__ size_t kv_offset(int D, int H_K, int N_or_TK) const {
        if constexpr (IS_VARLEN) {
            return static_cast<size_t>(k_base) * H_K * D + static_cast<size_t>(kv_head_idx) * D;
        } else {
            int batch_head_id = batch_idx * H_K + kv_head_idx;
            return static_cast<size_t>(batch_head_id) * N_or_TK * D;
        }
    }

    // ======================================================================================
    // Offset computation for LSE
    // ======================================================================================
    __device__ __forceinline__ size_t lse_offset(int H_Q, int M_or_TQ) const {
        if constexpr (IS_VARLEN) {
            return static_cast<size_t>(head_idx) * M_or_TQ + q_base + start_q;
        } else {
            int batch_head_id = batch_idx * H_Q + head_idx;
            return static_cast<size_t>(batch_head_id) * M_or_TQ + start_q;
        }
    }

    // ======================================================================================
    // Offset computation for dropout mask
    // ======================================================================================
    __device__ __forceinline__ size_t dmask_offset(int H_Q, int M_or_TQ, int N_or_MSEQ_K) const {
        if constexpr (IS_VARLEN) {
            return static_cast<size_t>(q_base + start_q) * H_Q * N_or_MSEQ_K + static_cast<size_t>(head_idx) * N_or_MSEQ_K;
        } else {
            int batch_head_id = batch_idx * H_Q + head_idx;
            return static_cast<size_t>(batch_head_id) * M_or_TQ * N_or_MSEQ_K;
        }
    }
};

// ======================================================================================
// Internal recursive dispatcher: accumulates runtime flags into a compile-time pack
// ======================================================================================
template <typename LaunchFn, bool... Resolved>
__host__ inline void dispatch_flags(LaunchFn& launch) {
    launch(std::integral_constant<bool, Resolved>{}...);
}

template <typename LaunchFn, bool... Resolved, typename... Rest>
__host__ inline void dispatch_flags(LaunchFn& launch, bool next, Rest... rest) {
    if (next) {
        dispatch_flags<LaunchFn, Resolved..., true>(launch, rest...);
    } else {
        dispatch_flags<LaunchFn, Resolved..., false>(launch, rest...);
    }
}

// ======================================================================================
// Runtime / Compile-Time Dispatcher
// ======================================================================================
template <typename LaunchFn>
__host__ inline void dispatch_attention_features(
    bool is_causal,
    bool is_alibi,
    bool is_softcap,
    bool is_window,
    bool is_dropout,
    bool is_paged,
    LaunchFn&& launch)
{
    const bool chk_causal  = is_causal;
    const bool chk_alibi   = is_causal && is_alibi;
    const bool chk_softcap = is_causal && is_softcap;
    const bool chk_window  = is_causal && is_window;
    const bool chk_dropout = is_dropout;
    const bool chk_paged   = is_paged;

    dispatch_flags<LaunchFn>(
        launch,
        chk_causal, chk_alibi, chk_softcap, chk_window, chk_dropout, chk_paged
    );
}