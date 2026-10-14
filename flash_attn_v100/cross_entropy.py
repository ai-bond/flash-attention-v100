# ======================================================================================
# Copyright (c) 2026, D.Skryabin / tg @ai_bond007 SPDX-License: BSD-3-Clause
# ======================================================================================
import torch
import torch.nn as nn
from typing import Tuple

from flash_attn_v100_cuda import lce_fwd
from flash_attn_v100_cuda import lce_bwd

# ======================================================================================
# Autograd Function: Forward / Backward via LCE kernel
# ======================================================================================
class LinearCrossEntropyFunction(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx,
        input:            torch.Tensor,
        weight:           torch.Tensor,
        target:           torch.Tensor,
        reduction:        int,
        label_smoothing:  float,
        logit_scale:      float,
        lse_square_scale: float,
        ignore_index:     int,
    ) -> Tuple[torch.Tensor, torch.Tensor]:

        if not input.is_cuda or not weight.is_cuda or not target.is_cuda:
            raise RuntimeError("LCE kernels require CUDA tensors")

        orig_dtype = input.dtype

        if input.dtype != torch.float16 or weight.dtype != torch.float16:
            input  = input.to(torch.float16)
            weight = weight.to(torch.float16)

        loss, lse, z_loss = lce_fwd(
            input, 
            weight, 
            target,
            reduction,
            float(label_smoothing),
            float(logit_scale),
            float(lse_square_scale),
            int(ignore_index),
        )

        ctx.save_for_backward(lse, input, weight, target)
        ctx.mark_non_differentiable(z_loss)

        ctx.label_smoothing  = float(label_smoothing)
        ctx.logit_scale      = float(logit_scale)
        ctx.lse_square_scale = float(lse_square_scale)
        ctx.ignore_index     = int(ignore_index)
        ctx.reduction        = int(reduction)
        ctx.orig_dtype       = orig_dtype

        return loss, z_loss

    @staticmethod
    def backward(ctx, grad_loss: torch.Tensor, grad_z_loss: torch.Tensor):
        del grad_z_loss

        lse, input, weight, target = ctx.saved_tensors

        grad_loss = grad_loss.contiguous().to(torch.float32)

        grad_input, grad_weight = lce_bwd(
            grad_loss,
            lse,
            input,
            weight,
            target,
            ctx.reduction,
            ctx.label_smoothing,
            ctx.logit_scale,
            ctx.lse_square_scale,
            ctx.ignore_index,
        )

        if ctx.orig_dtype != torch.float16:
            grad_input  = grad_input.to(ctx.orig_dtype)
            grad_weight = grad_weight.to(ctx.orig_dtype)

        return grad_input, grad_weight, None, None, None, None, None, None

# ======================================================================================
# API call function
# ======================================================================================
def cross_entropy_loss(
    input:            torch.Tensor,
    weight:           torch.Tensor,
    labels:           torch.Tensor,
    reduction:        str   = "none",
    label_smoothing:  float = 0.0,
    logit_scale:      float = 1.0,
    lse_square_scale: float = 0.0,
    ignore_index:     int   = -100,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Linear + Cross-Entropy Loss
    Computes logits and loss in a single fused kernel without materializing the full logits tensor

    Arguments:
        input:            [.., D] Hidden states, CUDA. Native FP16
        weight:           [V,  D] Classification head weight, CUDA. Native FP16
        labels:           [.....] Target indices, int32/int64, CUDA
        reduction:        "none" | "mean" | "sum"
        label_smoothing:  Float in [0, 1)
        logit_scale:      Float > 0. Scale factor for logits before softmax
        lse_square_scale: Float. If > 0, adds z-loss = lse_square_scale * LSE^2
        ignore_index:     Int. Rows with this target value contribute 0 to the loss

    Returns:
        losses: float32 tensor
        z_loss: float32 tensor
    """
    if reduction not in ("none", "mean", "sum"):
        raise ValueError(f"reduction must be 'none' | 'mean' | 'sum', got {reduction!r}")

    reduction_map = {"none": 0, "mean": 1, "sum": 2}

    loss, z_loss = LinearCrossEntropyFunction.apply(
        input,
        weight,
        labels,
        reduction_map[reduction],
        float(label_smoothing),
        float(logit_scale),
        float(lse_square_scale),
        int(ignore_index),
    )

    return loss, z_loss
