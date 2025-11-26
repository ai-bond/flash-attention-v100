# *
# * Copyright (c) 2025, D.Skryabin / tg @ai_bond007
# * SPDX-License-Identifier: BSD-3-Clause
# *
import torch
import traceback
import flash_attn_v100_cuda
from typing import Optional, Sequence, Tuple, Union

def maybe_contiguous(x):
    return x.contiguous() if x is not None and not x.is_contiguous() else x

def _flash_attn_forward(
    q: torch.Tensor,  # (B, H, M, D)
    k: torch.Tensor,
    v: torch.Tensor,
    dropout_p: float,
    softmax_scale: float,
    causal: bool,
    window_size_left: int,
    window_size_right: int,
    softcap: float,
    alibi_slopes: torch.Tensor,
    return_softmax: bool,
) -> tuple:
    q, k, v = map(maybe_contiguous, (q, k, v))
    out = torch.empty_like(q)
    lse = torch.empty(q.shape[0] * q.shape[1] * q.shape[2], dtype=torch.float32, device=q.device)
    outputs = flash_attn_v100_cuda.fwd(
        q, k, v,
        out, alibi_slopes,
        dropout_p, softmax_scale, causal,
        window_size_left, window_size_right,
        softcap, return_softmax, None
    )
    return outputs[0], outputs[1], None, None

def _flash_attn_backward(
    dout: torch.Tensor,
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    out: torch.Tensor,
    softmax_lse: torch.Tensor,
    dq: torch.Tensor,
    dk: torch.Tensor,
    dv: torch.Tensor,
    dropout_p: float,
    softmax_scale: float,
    causal: bool,
    window_size_left: int,
    window_size_right: int,
    softcap: float,
    alibi_slopes: torch.Tensor,
    deterministic: bool,
    rng_state: torch.Tensor = None,
) -> torch.Tensor:
    dout, q, k, v, out = map(maybe_contiguous, (dout, q, k, v, out))
    grads = flash_attn_v100.bwd(
        dout, q, k, v, out, softmax_lse,
        dq, dk, dv,
        alibi_slopes,
        dropout_p, softmax_scale, causal,
        window_size_left, window_size_right,
        softcap, deterministic, None, rng_state
    )
    return grads[0], grads[1], grads[2]

class FlashAttnFunc(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx,
        q: torch.Tensor,  # (B, M, H, D)
        k: torch.Tensor,
        v: torch.Tensor,
        dropout_p: float,
        softmax_scale: float,
        causal: bool,
        window_size: tuple,
        softcap: float,
        alibi_slopes: torch.Tensor,
        deterministic: bool,
        return_softmax: bool,
        is_grad_enabled: bool,
    ):

        # Layout: FA2 — (B, M, H, D) → kernel — (B, H, M, D)
        q_ = q.permute(0, 2, 1, 3).contiguous()
        k_ = k.permute(0, 2, 1, 3).contiguous()
        v_ = v.permute(0, 2, 1, 3).contiguous()

        B, M, H, D = q.shape
        _, N, _, _ = k.shape

        if D % 8 != 0:
            raise ValueError(f"head_dim={D} must be divisible by 8 for Volta kernel")

        if dropout_p != 0.0:
            raise NotImplementedError("dropout_p != 0.0 not supported")

        if alibi_slopes is not None:
            raise NotImplementedError("alibi_slopes not supported")

        if softcap != 0.0:
            raise NotImplementedError("softcap != 0.0 not supported")

        if q_.shape[1] != k_.shape[1]:
            raise ValueError(f"n_heads mismatch: q has {q_.shape[1]}, k has {k_.shape[1]} (MQA/GQA not supported)")

        window_size_left, window_size_right = window_size
        if causal and (window_size_left != -1 or window_size_right != -1):
            if window_size_left > 0 and window_size_right > 0:
                window_size_left, window_size_right = -1, -1
            else:
                raise NotImplementedError(f"Unsupported window_size={window_size} with causal=True")

        # Forward pass
        out_, lse_, _, rng_state = _flash_attn_forward(
            q_, k_, v_,
            dropout_p, softmax_scale, causal,
            window_size_left, window_size_right,
            softcap, alibi_slopes, return_softmax
        )

        # Convert back: (B, H, M, D) → (B, M, H, D)
        out = out_.permute(0, 2, 1, 3).contiguous()

        if is_grad_enabled and q.requires_grad:
            ctx.save_for_backward(q_, k_, v_, out_, lse_, rng_state)
            ctx.dropout_p = dropout_p
            ctx.softmax_scale = softmax_scale
            ctx.causal = causal
            ctx.window_size = window_size
            ctx.softcap = softcap
            ctx.alibi_slopes = alibi_slopes
            ctx.deterministic = deterministic

        return out if not return_softmax else (out, lse_, None)

    @staticmethod
    def backward(ctx, dout, *args):
        q_, k_, v_, out_, lse_, rng_state = ctx.saved_tensors

        # dout: (B, M, H, D) → (B, H, M, D)
        dout_ = dout.permute(0, 2, 1, 3).contiguous()

        dq_ = torch.empty_like(q_)
        dk_ = torch.empty_like(k_)
        dv_ = torch.empty_like(v_)

        # Backward pass
        _flash_attn_backward(
            dout_, q_, k_, v_, out_, lse_,
            dq_, dk_, dv_,
            ctx.dropout_p,
            ctx.softmax_scale,
            ctx.causal,
            ctx.window_size[0],
            ctx.window_size[1],
            ctx.softcap,
            ctx.alibi_slopes,
            ctx.deterministic,
            rng_state,
        )

        # Convert back to FA2 layout
        dq = dq_.permute(0, 2, 1, 3)
        dk = dk_.permute(0, 2, 1, 3)
        dv = dv_.permute(0, 2, 1, 3)

        return dq, dk, dv, None, None, None, None, None, None, None, None, None

def flash_attn_func(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    dropout_p: float = 0.0,
    softmax_scale: float = None,
    causal: bool = False,
    window_size: tuple = (-1, -1),
    softcap: float = 0.0,
    alibi_slopes: torch.Tensor = None,
    deterministic: bool = False,
    return_attn_probs: bool = False,
):
    """
    FA2-compatible flash_attn_func for Volta.
    Layout: (batch_size, seqlen, nheads, headdim)
    """
    if softmax_scale is None:
        softmax_scale = q.shape[-1] ** -0.5

    try:
        return FlashAttnFunc.apply(
            q, k, v,
            dropout_p,
            softmax_scale,
            causal,
            window_size,
            softcap,
            alibi_slopes,
            deterministic,
            return_attn_probs,
            torch.is_grad_enabled(),
        )
    except Exception as e:
        print("VOLTA FA2 FAILED in flash_attn_func")
        print(f"  q.shape = {list(q.shape)}, dtype = {q.dtype}, device = {q.device}, contiguous = {q.is_contiguous()}")
        print(f"  k.shape = {list(k.shape)}, dtype = {k.dtype}, device = {k.device}, contiguous = {k.is_contiguous()}")
        print(f"  v.shape = {list(v.shape)}, dtype = {v.dtype}, device = {v.device}, contiguous = {v.is_contiguous()}")
        print(f"  causal = {causal}, window_size = {window_size}, softmax_scale = {softmax_scale}")
        print(f"  Exception type: {type(e).__name__}")
        print(f"  Exception message: {e}")
        traceback.print_exc()
        raise

__all__ = ["flash_attn_func"]