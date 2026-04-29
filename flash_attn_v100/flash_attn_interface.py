# *
# * Copyright (c) 2026, D.Skryabin / tg @ai_bond007
# * SPDX-License-Identifier: BSD-3-Clause
# *
import torch
import warnings
import traceback
import flash_attn_v100_cuda
from typing import Optional, Tuple

def maybe_contiguous(x: torch.Tensor) -> torch.Tensor:
    return x.contiguous() if x is not None and not x.is_contiguous() else x

class FlashAttnFunc(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        dropout_p: float,
        softmax_scale: float,
        causal: bool,
        window_size: Tuple[int, int],
        softcap: float,
        alibi_slopes: Optional[torch.Tensor],
        deterministic: bool,
        return_attn_probs: bool = False
    ) -> torch.Tensor:
        q_ = q.permute(0, 2, 1, 3)
        k_ = k.permute(0, 2, 1, 3)
        v_ = v.permute(0, 2, 1, 3)

        B, H_Q, M, head_size_og = q_.shape
        H_K = k_.shape[1]
        N = k_.shape[2]

        if head_size_og % 8 != 0:
            pad_size = 8 - head_size_og % 8
            q_ = torch.nn.functional.pad(q_, [0, pad_size])
            k_ = torch.nn.functional.pad(k_, [0, pad_size])
            v_ = torch.nn.functional.pad(v_, [0, pad_size])
            head_size = head_size_og + pad_size
        else:
            head_size = head_size_og
            pad_size = 0

        q_ = q_.contiguous()
        k_ = k_.contiguous()
        v_ = v_.contiguous()

        if softmax_scale is None:
            softmax_scale = head_size_og ** -0.5

        window_left, window_size_right = window_size

        out_, lse_, dmask_, rng_state = flash_attn_v100_cuda.fwd(
            q_, k_, v_,
            None, alibi_slopes,
            dropout_p, softmax_scale, causal,
            window_left, window_size_right,
            softcap, return_attn_probs, None
        )

        out = out_[..., :head_size_og].permute(0, 2, 1, 3).contiguous()

        if q.requires_grad or k.requires_grad or v.requires_grad:
            ctx.save_for_backward(q_, k_, v_, out_, lse_, rng_state)
            ctx.dropout_p = dropout_p
            ctx.softmax_scale = softmax_scale
            ctx.causal = causal
            ctx.window_size = window_size
            ctx.softcap = softcap
            ctx.alibi_slopes = alibi_slopes
            ctx.deterministic = deterministic
            ctx.head_size_og = head_size_og
            ctx.pad_size = pad_size

        return (out, lse_, dmask_) if return_attn_probs else out

    @staticmethod
    def backward(ctx, dout, *args):
        q_, k_, v_, out_, lse_, rng_state = ctx.saved_tensors
        head_size_og = ctx.head_size_og
        pad_size = ctx.pad_size

        if pad_size > 0:
            dout = torch.nn.functional.pad(dout, [0, pad_size])
        dout_ = dout.permute(0, 2, 1, 3).contiguous()

        dq_ = torch.empty_like(q_)
        dk_ = torch.empty_like(k_)
        dv_ = torch.empty_like(v_)

        window_left, window_right = ctx.window_size

        grads = flash_attn_v100_cuda.bwd(
            dout_, q_, k_, v_, out_, lse_,
            dq_, dk_, dv_,
            ctx.alibi_slopes,
            ctx.dropout_p, ctx.softmax_scale, ctx.causal,
            window_left, window_right,
            ctx.softcap, ctx.deterministic, None, rng_state
        )

        dq = grads[0][..., :head_size_og].permute(0, 2, 1, 3)
        dk = grads[1][..., :head_size_og].permute(0, 2, 1, 3)
        dv = grads[2][..., :head_size_og].permute(0, 2, 1, 3)

        return dq, dk, dv, None, None, None, None, None, None, None, None

def flash_attn_func(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    dropout_p: float = 0.0,
    softmax_scale: float = None,
    causal: bool = False,
    window_size: Tuple[int, int] = (-1, -1),
    softcap: float = 0.0,
    alibi_slopes: Optional[torch.Tensor] = None,
    deterministic: bool = False,
    return_attn_probs: bool = False
):
    """
    dropout_p should be set to 0.0 during evaluation
    Supports multi-query and grouped-query attention (MQA/GQA) by passing in KV with fewer heads
    than Q. Note that the number of heads in Q must be divisible by the number of heads in KV.
    For example, if Q has 6 heads and K, V have 2 heads, head 0, 1, 2 of Q will attention to head
    0 of K, V, and head 3, 4, 5 of Q will attention to head 1 of K, V.
    If window_size != (-1, -1), implements sliding window local attention. Query at position i
    will only attend to keys between
    [i + seqlen_k - seqlen_q - window_size[0], i + seqlen_k - seqlen_q + window_size[1]] inclusive.

    Arguments:
        q: (batch_size, seqlen, nheads, headdim)
        k: (batch_size, seqlen, nheads_k, headdim)
        v: (batch_size, seqlen, nheads_k, headdim)
        dropout_p: float. Dropout probability.
        softmax_scale: float. The scaling of QK^T before applying softmax.
            Default to 1 / sqrt(headdim).
        causal: bool. Whether to apply causal attention mask (e.g., for auto-regressive modeling).
        window_size: (left, right). If not (-1, -1), implements sliding window local attention.
        alibi_slopes: (nheads,) or (batch_size, nheads), fp32. A bias of
            (-alibi_slope * |i + seqlen_k - seqlen_q - j|)
            is added to the attention score of query i and key j.
        deterministic: bool. Whether to use the deterministic implementation of the backward pass,
            which is slightly slower and uses more memory. The forward pass is always deterministic.
    Return:
        out: (batch_size, seqlen, nheads, headdim).
    """

    if deterministic:
        warnings.warn("The forward pass is always deterministic. Deterministic backward is not supported.", RuntimeWarning)
        deterministic = False

    try:
        return FlashAttnFunc.apply(
            q,
            k,
            v,
            dropout_p,
            softmax_scale,
            causal,
            window_size,
            softcap,
            alibi_slopes,
            deterministic,
            return_attn_probs
        )
    except Exception as e:
        print(f"[VOLTA FA2 FAILED] {type(e).__name__}: {e}")
        traceback.print_exc()
        raise

flash_attn_gpu = flash_attn_func
__all__ = ["flash_attn_func", "flash_attn_gpu"]