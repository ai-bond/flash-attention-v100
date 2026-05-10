# ======================================================================================
# * Copyright (c) 2026, D.Skryabin / tg @ai_bond007
# * SPDX-License-Identifier: BSD-3-Clause
# ======================================================================================
import torch
import warnings
import traceback
import flash_attn_v100_cuda
from typing import Optional, Tuple

def maybe_contiguous(x: torch.Tensor) -> torch.Tensor:
    return x.contiguous() if x is not None and not x.is_contiguous() else x

# ======================================================================================
# DENSE ATTENTION (B, M, H, D) -> (B, H, M, D)
# ======================================================================================
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

        pad_size = 0
        if head_size_og % 8 != 0:
            pad_size = 8 - head_size_og % 8
            q_ = torch.nn.functional.pad(q_, [0, pad_size])
            k_ = torch.nn.functional.pad(k_, [0, pad_size])
            v_ = torch.nn.functional.pad(v_, [0, pad_size])

        q_ = q_.contiguous()
        k_ = k_.contiguous()
        v_ = v_.contiguous()

        if softmax_scale is None:
            softmax_scale = head_size_og ** -0.5

        window_left, window_right = window_size

        out_, lse_, dmask_, rng_state = flash_attn_v100_cuda.fwd(
            q_, k_, v_, None, alibi_slopes,
            dropout_p, softmax_scale, causal,
            window_left, window_right, softcap,
            return_attn_probs, None
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

        dq = grads[0][..., :head_size_og].permute(0, 2, 1, 3).contiguous()
        dk = grads[1][..., :head_size_og].permute(0, 2, 1, 3).contiguous()
        dv = grads[2][..., :head_size_og].permute(0, 2, 1, 3).contiguous()

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
    """Dense Flash Attention (B, M, H, D)"""
    if deterministic:
        warnings.warn("Forward is always deterministic. Deterministic backward is not supported.", RuntimeWarning)
        deterministic = False

    try:
        return FlashAttnFunc.apply(
            q, k, v, dropout_p, softmax_scale, causal,
            window_size, softcap, alibi_slopes, deterministic, return_attn_probs
        )
    except Exception as e:
        print(f"[VOLTA FA2 DENSE FAILED] {type(e).__name__}: {e}")
        traceback.print_exc()
        raise


# ======================================================================================
# VARLEN ATTENTION (T, H, D) -> (T, H, D)
# ======================================================================================
class FlashAttnVarlenFunc(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        cu_seqlens_q: torch.Tensor,
        cu_seqlens_k: torch.Tensor,
        max_seqlen_q: int,
        max_seqlen_k: int,
        dropout_p: float,
        softmax_scale: float,
        causal: bool,
        window_size: Tuple[int, int],
        softcap: float,
        alibi_slopes: Optional[torch.Tensor],
        deterministic: bool,
        return_attn_probs: bool,
        block_table: Optional[torch.Tensor],
        seqused_k: Optional[torch.Tensor],
        leftpad_k: Optional[torch.Tensor],
        is_grad_enabled: bool
    ) -> torch.Tensor:
        cu_seqlens_q = cu_seqlens_q.to(torch.int32)
        cu_seqlens_k = cu_seqlens_k.to(torch.int32)

        q = maybe_contiguous(q)
        k = maybe_contiguous(k)
        v = maybe_contiguous(v)

        head_size_og = q.size(2)
        pad_size = 0
        if head_size_og % 8 != 0:
            pad_size = 8 - head_size_og % 8
            q = torch.nn.functional.pad(q, [0, pad_size])
            k = torch.nn.functional.pad(k, [0, pad_size])
            v = torch.nn.functional.pad(v, [0, pad_size])

        q = q.contiguous()
        k = k.contiguous()
        v = v.contiguous()

        if softmax_scale is None:
            softmax_scale = head_size_og ** -0.5

        window_left, window_right = window_size

        out, lse, dmask, rng_state = flash_attn_v100_cuda.varlen_fwd(
            q, k, v, None, cu_seqlens_q, cu_seqlens_k,
            seqused_k, leftpad_k, block_table, alibi_slopes,
            max_seqlen_q, max_seqlen_k, dropout_p, softmax_scale,
            False, causal, window_left, window_right, softcap,
            return_attn_probs and dropout_p > 0.0, None, 0
        )

        out = out[..., :head_size_og].contiguous()

        if is_grad_enabled and (q.requires_grad or k.requires_grad or v.requires_grad):
            ctx.save_for_backward(q, k, v, out, lse, cu_seqlens_q, cu_seqlens_k, rng_state)
            ctx.dropout_p = dropout_p
            ctx.softmax_scale = softmax_scale
            ctx.causal = causal
            ctx.window_size = window_size
            ctx.softcap = softcap
            ctx.alibi_slopes = alibi_slopes
            ctx.deterministic = deterministic
            ctx.head_size_og = head_size_og
            ctx.pad_size = pad_size
            ctx.max_seqlen_q = max_seqlen_q
            ctx.max_seqlen_k = max_seqlen_k

        return (out, lse, dmask) if return_attn_probs else out

    @staticmethod
    def backward(ctx, dout, *args):
        q, k, v, out, lse, cu_seqlens_q, cu_seqlens_k, rng_state = ctx.saved_tensors
        head_size_og = ctx.head_size_og
        pad_size = ctx.pad_size

        cu_seqlens_q = cu_seqlens_q.to(torch.int32)
        cu_seqlens_k = cu_seqlens_k.to(torch.int32)

        dout = maybe_contiguous(dout)
        if pad_size > 0:
            dout = torch.nn.functional.pad(dout, [0, pad_size])
        dout = dout.contiguous()

        dq = torch.empty_like(q)
        dk = torch.empty_like(k)
        dv = torch.empty_like(v)

        window_left, window_right = ctx.window_size

        grads = flash_attn_v100_cuda.varlen_bwd(
            dout, q, k, v, out, lse,
            dq, dk, dv, cu_seqlens_q, cu_seqlens_k,
            ctx.alibi_slopes, ctx.max_seqlen_q, ctx.max_seqlen_k,
            ctx.dropout_p, ctx.softmax_scale, False,
            ctx.causal, window_left, window_right, ctx.softcap,
            ctx.deterministic, None, rng_state
        )

        dq = grads[0][..., :head_size_og].contiguous()
        dk = grads[1][..., :head_size_og].contiguous()
        dv = grads[2][..., :head_size_og].contiguous()

        return dq, dk, dv, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None


def flash_attn_varlen_func(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    cu_seqlens_q: torch.Tensor,
    cu_seqlens_k: torch.Tensor,
    max_seqlen_q: int,
    max_seqlen_k: int,
    dropout_p: float = 0.0,
    softmax_scale: float = None,
    causal: bool = False,
    window_size: Tuple[int, int] = (-1, -1),
    softcap: float = 0.0,
    alibi_slopes: Optional[torch.Tensor] = None,
    deterministic: bool = False,
    return_attn_probs: bool = False,
    block_table: Optional[torch.Tensor] = None,
    seqused_k: Optional[torch.Tensor] = None,
    leftpad_k: Optional[torch.Tensor] = None
):
    """Varlen Flash Attention (T, H, D)"""
    if deterministic:
        warnings.warn("Forward is always deterministic. Deterministic backward is not supported.", RuntimeWarning)
        deterministic = False

    try:
        return FlashAttnVarlenFunc.apply(
            q, k, v, cu_seqlens_q, cu_seqlens_k,
            max_seqlen_q, max_seqlen_k, dropout_p, softmax_scale, causal,
            window_size, softcap, alibi_slopes, deterministic,
            return_attn_probs, block_table, seqused_k, leftpad_k,
            torch.is_grad_enabled()
        )
    except Exception as e:
        print(f"[VOLTA FA2 VARLEN FAILED] {type(e).__name__}: {e}")
        traceback.print_exc()
        raise


flash_attn_gpu = flash_attn_func
flash_attn_varlen_gpu = flash_attn_varlen_func

__all__ = [
    "flash_attn_func", "flash_attn_gpu",
    "flash_attn_varlen_func", "flash_attn_varlen_gpu"
]
