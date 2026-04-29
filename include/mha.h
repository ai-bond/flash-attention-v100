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
 * @param q                   Query tensor           [B, H, M, D] (fp16, input)
 * @param k                   Key tensor             [B, H, N, D] (fp16, input)
 * @param v                   Value tensor           [B, H, N, D] (fp16, input)
 * @param out                 Optional output tensor [B, H, M, D] (fp16, output)
 * @param alibi_slopes        Optional ALiBi slopes tensor for relative positioning
 * @param p_dropout           Dropout probability (0.0 for no dropout)
 * @param softmax_scale       Softmax scale (typically 1/sqrt(D))
 * @param is_causal           Enable causal masking (autoregressive)
 * @param window_left         Left window size for sliding window attention
 * @param window_right        Right window size for sliding window attention
 * @param softcap             Soft capping value for attention scores
 * @param return_softmax      Whether to return softmax matrix (memory intensive)
 * @param gen                 Optional random generator for dropout
 * @return                    Vector of output tensors [output, softmax_lse, ...]
 */
std::vector<at::Tensor> flash_attention_forward(
    at::Tensor& q,
    const at::Tensor& k,
    const at::Tensor& v,
    std::optional<at::Tensor>& out,
    std::optional<at::Tensor>& alibi_slopes,
    const float p_dropout,
    const float softmax_scale,
    bool is_causal,
    int window_left,
    int window_right,
    const float softcap,
    const bool return_softmax,
    std::optional<at::Generator> gen
);

/**
 * Flash Attention Backward Pass
 *
 * @param dout          Gradient output             [B, H, M, D] (fp16, input)
 * @param q             Query tensor from forward   [B, H, M, D] (fp16, input)
 * @param k             Key tensor from forward     [B, H, N, D] (fp16, input)
 * @param v             Value tensor from forward   [B, H, N, D] (fp16, input)
 * @param out           Output from forward pass    [B, H, M, D] (fp16, input)
 * @param softmax_lse   Forward's Softmax logsumexp [B, H, M]    (fp32, input)
 * @param dq            Optional gradient query     [B, H, M, D] (fp16, output)
 * @param dk            Optional gradient key       [B, H, N, D] (fp16, output)
 * @param dv            Optional gradient value     [B, H, N, D] (fp16, output)
 * @param alibi_slopes  Optional ALiBi slopes tensor (must match forward)
 * @param p_dropout     Dropout probability (must match forward)
 * @param softmax_scale Softmax scale (must match forward)
 * @param is_causal     Causal masking flag (must match forward)
 * @param window_left   Left window size (must match forward)
 * @param window_right  Right window size (must match forward)
 * @param softcap       Soft capping value (must match forward)
 * @param deterministic Whether to use deterministic backward pass
 * @param gen           Optional random generator for dropout
 * @param rng_state     Optional RNG state for reproducibility
 * @return              Vector of gradient tensors [dq, dk, dv, ...]
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
    const float p_dropout,
    const float softmax_scale,
    const bool is_causal,
    int window_left,
    int window_right,
    const float softcap,
    const bool deterministic,
    std::optional<at::Generator> gen,
    std::optional<at::Tensor>& rng_state
);
