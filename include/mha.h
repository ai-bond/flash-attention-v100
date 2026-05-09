// ======================================================================================
// * Copyright (c) 2026, D.Skryabin / tg @ai_bond007 SPDX-License: BSD-3-Clause
// ======================================================================================
#include <cuda_runtime.h>
#include <stdexcept>
#include <torch/extension.h>
#include <ATen/ATen.h>

/**
 * Flash Attention Forward Pass
 *
 * @param q                   Query tensor                     [B, H, M, D] (fp16, input)
 * @param k                   Key tensor                       [B, H, N, D] (fp16, input)
 * @param v                   Value tensor                     [B, H, N, D] (fp16, input)
 * @param out                 Optional output tensor           [B, H, M, D] (fp16, output)
 * @param alibi_slopes        Optional ALiBi slopes tensor     [H] or [B,H] (fp32, input)
 * @param p_dropout           Dropout probability              (float, input)
 * @param softmax_scale       Softmax scale factor             (float, input)
 * @param is_causal           Enable causal masking            (bool, input)
 * @param window_left         Left window size for sliding attn(int, input)
 * @param window_right        Right window size for sliding attn(int, input)
 * @param softcap             Soft capping value for logits    (float, input)
 * @param return_softmax      Whether to return softmax matrix (bool, input)
 * @param gen                 Optional RNG for dropout         (input)
 * @return                    Vector of output tensors {out, softmax_lse, dmask, rng_state}
 */
std::vector<at::Tensor> flash_attention_forward(
          at::Tensor& q,
    const at::Tensor& k,
    const at::Tensor& v,
    std::optional<at::Tensor>& out,
    std::optional<at::Tensor>& alibi_slopes,
    const float  p_dropout,
    const float  softmax_scale,
    bool         is_causal,
    int          window_left,
    int          window_right,
    const float  softcap,
    const bool   return_softmax,
    std::optional<at::Generator> gen
);

/**
 * Flash Attention Backward Pass
 *
 * @param dout                Gradient output                  [B, H, M, D] (fp16, input)
 * @param q                   Query tensor from forward        [B, H, M, D] (fp16, input)
 * @param k                   Key tensor from forward          [B, H, N, D] (fp16, input)
 * @param v                   Value tensor from forward        [B, H, N, D] (fp16, input)
 * @param out                 Output from forward pass         [B, H, M, D] (fp16, input)
 * @param softmax_lse         Forward's Softmax logsumexp      [B, H, M]    (fp32, input)
 * @param dq                  Optional gradient query          [B, H, M, D] (fp16, output)
 * @param dk                  Optional gradient key            [B, H, N, D] (fp16, output)
 * @param dv                  Optional gradient value          [B, H, N, D] (fp16, output)
 * @param alibi_slopes        Optional ALiBi slopes tensor     (must match forward)
 * @param p_dropout           Dropout probability              (must match forward)
 * @param softmax_scale       Softmax scale factor             (must match forward)
 * @param is_causal           Causal masking flag              (must match forward)
 * @param window_left         Left window size                 (must match forward)
 * @param window_right        Right window size                (must match forward)
 * @param softcap             Soft capping value               (must match forward)
 * @param deterministic       Use deterministic backward pass  (bool, input)
 * @param gen                 Optional RNG for dropout         (input)
 * @param rng_state           Optional RNG state for repro     (input)
 * @return                    Vector of gradient tensors {dq, dk, dv, softmax_d}
 */
std::vector<at::Tensor> flash_attention_backward(
    const at::Tensor& dout,
    const at::Tensor& q,
    const at::Tensor& k,
    const at::Tensor& v,
    const at::Tensor& out,
    const at::Tensor& softmax_lse,
    std::optional<at::Tensor>& dq,
    std::optional<at::Tensor>& dk,
    std::optional<at::Tensor>& dv,
    std::optional<at::Tensor>& alibi_slopes,
    const float  p_dropout,
    const float  softmax_scale,
    const bool   is_causal,
    int          window_left,
    int          window_right,
    const float  softcap,
    const bool   deterministic,
    std::optional<at::Generator> gen,
    std::optional<at::Tensor>& rng_state
);

/**
 * FlashAttention Forward Pass (Varlen / Packed Layout)
 *
 * @param q                   Query tensor                     [total_q, H, D] (fp16, input)
 * @param k                   Key tensor                       [total_k, H, D] (fp16, input)
 * @param v                   Value tensor                     [total_k, H, D] (fp16, input)
 * @param out                 Optional output tensor           [total_q, H, D] (fp16, output)
 * @param cu_seqlens_q        Cumulative seq lengths for Q     [B+1] (int32, input)
 * @param cu_seqlens_k        Cumulative seq lengths for K     [B+1] (int32, input)
 * @param seqused_k           Optional actual seq lengths for K[B]   (int32, input)
 * @param leftpad_k           Optional left padding for K      [B]   (int32, input)
 * @param block_table         Optional block table for paged attn[B, max_blocks] (int32, input)
 * @param alibi_slopes        Optional ALiBi slopes tensor     [H] or [B,H] (fp32, input)
 * @param max_seqlen_q        Maximum sequence length for Q    (int, input)
 * @param max_seqlen_k        Maximum sequence length for K    (int, input)
 * @param p_dropout           Dropout probability              (float, input)
 * @param softmax_scale       Softmax scale factor             (float, input)
 * @param zero_tensors        Zero output tensors before compute(bool, input)
 * @param is_causal           Enable causal masking            (bool, input)
 * @param window_left         Left window size for sliding attn(int, input)
 * @param window_right        Right window size for sliding attn(int, input)
 * @param softcap             Soft capping value for logits    (float, input)
 * @param return_softmax      Whether to return softmax matrix (bool, input)
 * @param gen                 Optional RNG for dropout         (input)
 * @param num_splits          KV-cache split heuristic         (int, input)
 * @return                    Vector of output tensors {out, softmax_lse, dmask, rng_state}
 */
std::vector<at::Tensor> flash_attention_varlen_forward(
          at::Tensor& q,
    const at::Tensor& k,
    const at::Tensor& v,
    std::optional<at::Tensor> &out,
    const at::Tensor& cu_seqlens_q,
    const at::Tensor& cu_seqlens_k,
    std::optional<at::Tensor> &seqused_k,
    std::optional<const at::Tensor> &leftpad_k,
    std::optional<at::Tensor> &block_table,
    std::optional<at::Tensor> alibi_slopes,
    int          max_seqlen_q,
    const int    max_seqlen_k,
    const float  p_dropout,
    const float  softmax_scale,
    const bool   zero_tensors,
    bool         is_causal,
    int          window_left,
    int          window_right,
    const float  softcap,
    const bool   return_softmax,
    std::optional<at::Generator> gen,
    int          num_splits = 0
);
