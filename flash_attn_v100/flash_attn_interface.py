# *
# * Copyright (c) 2025, D.Skryabin / tg @ai_bond007
# * SPDX-License-Identifier: BSD-3-Clause
# *
import torch
import flash_attn_v100_cuda

"""
Python interface for FlashAttention-2 Volta implementation.
Expects tensors in (batch, nheads, seqlen, headdim) layout — matching kernel expectations.
"""

def flash_attn_func(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    dropout_p: float = 0.0,
    softmax_scale: float = None,
    causal: bool = False,
    alibi_slopes: torch.Tensor = None,
    deterministic: bool = False,
    return_lse: bool = False,
):
    """
    FlashAttention forward pass for Volta (SM70).

    Arguments:
        q: (batch_size, nheads, seqlen_q, headdim)
        k: (batch_size, nheads_k, seqlen_k, headdim)
        v: (batch_size, nheads_k, seqlen_k, headdim)
        dropout_p: Must be 0.0 (not supported).
        softmax_scale: Scaling factor for QK. Defaults to 1/sqrt(headdim).
        causal: Apply causal mask if True.
        alibi_slopes: Not supported (must be None).
        deterministic: Ignored.
        return_lse: If True, return log-sum-exp of softmax.

    Returns:
        out: (batch_size, nheads, seqlen_q, headdim)
        softmax_lse [optional]: (batch_size, nheads, seqlen_q)
    """
    if dropout_p != 0.0:
        raise NotImplementedError("Dropout is not implemented in this Volta kernel.")
    if alibi_slopes is not None:
        raise NotImplementedError("ALiBi is not supported.")
    window_size_left = kwargs.get('window_size_left', -1)
    window_size_right = kwargs.get('window_size_right', -1)
    softcap = kwargs.get('softcap', 0.0)
    return_softmax = kwargs.get('return_softmax', False)

    if window_size_left != -1:
        raise NotImplementedError("window_size_left not supported")
    if window_size_right != -1 and not (causal and window_size_right == 0):
        raise NotImplementedError("window not supported")
    if softcap != 0.0:
        raise NotImplementedError("softcap not supported")
    if return_softmax:
        raise NotImplementedError("return_softmax not supported")
    if deterministic:
        pass  # ignored

    # Shape validation
    B, H, M, D = q.shape
    B_k, H_k, N, D_k = k.shape

    if not (B == B_k and D == D_k):
        raise ValueError("Batch and head_dim must match between Q, K, V")
    if k.shape != v.shape:
        raise ValueError("K and V must have identical shapes")
    if H != H_k:
        raise ValueError("MQA/GQA not supported: nheads must be equal for Q, K, V")
    if D not in {16, 32, 64, 128, 256}:
        raise ValueError(f"Unsupported head_dim: {D}. Supported: 16, 32, 64, 128, 256")
    if D % 2 != 0:
        raise ValueError("head_dim must be even")
    if D % 8 != 0:
        raise ValueError("head_dim must be multiple of 8")
    if not all(t.is_contiguous() for t in [q, k, v]):
        raise ValueError("All input tensors must be contiguous")
    if q.dtype != torch.float16 or k.dtype != torch.float16 or v.dtype != torch.float16:
        raise ValueError("All input tensors must be fp16")

    if softmax_scale is None:
        softmax_scale = 1.0 / (D ** 0.5)

    out = torch.empty_like(q, dtype=torch.float16)
    softmax_lse_flat = torch.empty(B * H * M, dtype=torch.float32, device=q.device)

    flash_attn_v100_cuda.fwd(
        q, k, v, out, softmax_lse_flat, softmax_scale, causal
    )

    if return_lse:
        softmax_lse = softmax_lse_flat.view(B, H, M)
        return out, softmax_lse
    return out


def flash_attn_backward(
    dout: torch.Tensor,
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    out: torch.Tensor,
    softmax_lse: torch.Tensor,
    dq: torch.Tensor = None,
    dk: torch.Tensor = None,
    dv: torch.Tensor = None,
    dropout_p: float = 0.0,
    softmax_scale: float = None,
    causal: bool = False,
    alibi_slopes: torch.Tensor = None,
    deterministic: bool = False,
):
    """
    FlashAttention backward pass for Volta.

    Arguments:
        dout: (batch_size, nheads, seqlen_q, headdim) — grad w.r.t. output
        q, k, v: same as in forward
        out: output from forward (B, H, M, D), float32
        softmax_lse: (B, H, M) or flat (B*H*M)
        dq, dk, dv: optional pre-allocated output grads (must be float32)
        ... other args as in forward

    Returns:
        dq: (B, H, seqlen_q, D)
        dk: (B, H, seqlen_k, D)
        dv: (B, H, seqlen_k, D)
    """
    if dropout_p != 0.0:
        raise NotImplementedError("Dropout not supported.")
    if alibi_slopes is not None:
        raise NotImplementedError("ALiBi not supported.")

    B, H, M, D = q.shape
    B_k, H_k, N, D_k = k.shape

    if not (B == B_k and D == D_k and H == H_k):
        raise ValueError("Shape mismatch between Q, K, V")
    if k.shape != v.shape:
        raise ValueError("K and V must have same shape")
    if dout.shape != q.shape:
        raise ValueError("dout must have same shape as q")
    if out.shape != q.shape:
        raise ValueError("out must have same shape as q")
    if D not in {16, 32, 64, 128, 256}:
        raise ValueError(f"Unsupported head_dim: {D}")
    if D % 2 != 0:
        raise ValueError("head_dim must be even")
    if D % 8 != 0:
        raise ValueError("head_dim must be multiple of 8")

    if softmax_scale is None:
        softmax_scale = 1.0 / (D ** 0.5)

    # Validate softmax_lse
    expected_size = B * H * M
    if softmax_lse.dim() == 3:
        if softmax_lse.shape != (B, H, M):
            raise ValueError(f"Expected softmax_lse shape: ({B}, {H}, {M})")
        lse_flat = softmax_lse.reshape(-1)
    elif softmax_lse.dim() == 1:
        if softmax_lse.numel() != expected_size:
            raise ValueError(f"softmax_lse must have {expected_size} elements")
        lse_flat = softmax_lse
    else:
        raise ValueError("softmax_lse must be 1D or 3D")

    # Prepare outputs
    if dq is None:
        dq = torch.empty_like(q, dtype=torch.float16)
    else:
        if dq.shape != q.shape or dq.dtype != torch.float16:
            raise ValueError("dq must match q shape and be fp16")
    if dk is None:
        dk = torch.empty_like(k, dtype=torch.float16)
    else:
        if dk.shape != k.shape or dk.dtype != torch.float16:
            raise ValueError("dk must match k shape and be fp16")
    if dv is None:
        dv = torch.empty_like(v, dtype=torch.float16)
    else:
        if dv.shape != v.shape or dv.dtype != torch.float16:
            raise ValueError("dv must match v shape and be fp16")

    if not all(t.is_contiguous() for t in [q, k, v, out, dout, dq, dk, dv, lse_flat]):
        raise ValueError("All tensors must be contiguous")
    if q.dtype != torch.float16 or k.dtype != torch.float16 or v.dtype != torch.float16:
        raise ValueError("All input tensors must be fp16")

    flash_attn_v100.bwd(
        q, k, v, out, dout, lse_flat, dq, dk, dv, softmax_scale, causal
    )

    return dq, dk, dv

__all__ = ["flash_attn_func", "flash_attn_backward"]