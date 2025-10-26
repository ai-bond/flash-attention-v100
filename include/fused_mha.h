// ============================================================================
// * Copyright (c) 2025, D.Skryabin / tg @ai_bond007 SPDX-License: BSD-3-Clause
// ============================================================================
#ifndef FUSED_MHA_H
#define FUSED_MHA_H

#include <cuda_runtime.h>
#include <stdexcept>
#include <torch/extension.h>
#include <ATen/ATen.h>

/**
 * Flash Attention Forward Pass
 * 
 * @param Q                   Query tensor [B, H, M, D] (fp16)
 * @param K                   Key tensor [B, H, N, D] (fp16)
 * @param V                   Value tensor [B, H, N, D] (fp16)
 * @param Out                 Output tensor [B, H, M, D] (fp32)
 * @param softmax_lse Softmax lse for backward [B, H, M] (fp32)
 * @param scale               Softmax scale (typically 1/sqrt(D))
 * @param causal              Enable causal masking
 */
void flash_attention_forward(
    const torch::Tensor& Q,      // [B, H, M, D], fp16
    const torch::Tensor& K,      // [B, H, N, D], fp16
    const torch::Tensor& V,      // [B, H, N, D], fp16
    torch::Tensor& Out,          // [B, H, M, D], fp32
    torch::Tensor& softmax_lse,  // [B, H, M], fp32
    float scale,
    bool causal
);

/**
 * Flash Attention Backward Pass (FlashAttention-2)
 * 
 * Computes gradients dQ, dK, dV given inputs and output gradients.
 * 
 * @param Q           Query tensor [B, H, M, D] (fp16)
 * @param K           Key tensor [B, H, N, D] (fp16)
 * @param V           Value tensor [B, H, N, D] (fp16)
 * @param O           Output from forward pass [B, H, M, D] (fp32)
 * @param dO          Gradient w.r.t. output [B, H, M, D] (fp16)
 * @param softmax_lse Softmax lse from forward [B, H, M] (fp32)
 * @param dQ          Gradient w.r.t. query [B, H, M, D] (fp32, output)
 * @param dK          Gradient w.r.t. key [B, H, N, D] (fp16, output)
 * @param dV          Gradient w.r.t. value [B, H, N, D] (fp16, output)
 * @param scale       Softmax scale (typically 1/sqrt(D))
 * @param causal      Enable causal masking (must match forward pass)
 */
void flash_attention_backward(
    const torch::Tensor& Q,      // [B, H, M, D], fp16
    const torch::Tensor& K,      // [B, H, N, D], fp16
    const torch::Tensor& V,      // [B, H, N, D], fp16
    torch::Tensor& O,            // [B, H, M, D], fp32
    torch::Tensor& dO,           // [B, H, M, D], fp16
    torch::Tensor& softmax_lse,  // [B, H, M],    fp32
    torch::Tensor& dQ,           // [B, H, M, D], fp32
    torch::Tensor& dK,           // [B, H, N, D], fp16
    torch::Tensor& dV,           // [B, H, N, D], fp16
    float scale,
    bool causal
);

#endif // FUSED_MHA_H