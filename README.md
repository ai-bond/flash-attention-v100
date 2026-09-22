# FlashAttention for unsupported Tesla v100

This repository implementation of [FlashAttention-2](https://github.com/ai-bond/flash-attention-v100/blob/main/utils/docs/attention.md) under unsupported in TriDao repo [Nvidia Tesla V100](https://github.com/ai-bond/flash-attention-v100/blob/main/utils/docs/volta.md)

> It attempt to build flash attention from scratch without "Vibe Code" for self education.

If this code saved you time, treat me to a coffee ☕: 
- 🪙 BTC:          `bc1q5dydywjqmlewmenkyymzrmy482f9jdh7mr42aw`
- 💵 USDT (TRC20): `TAuNnyYigkcUvA2aPoEZnN48JaJ21HoWkb`

Installation Guide for Volta
-------------

The installation instructions for Volta GPUs are available here at [`install.md`](https://github.com/ai-bond/flash-attention-v100/blob/main/utils/docs/install.md).

How to use FlashAttention
-------------

The main functions implement scaled dot product attention (softmax(Q @ K^T * softmax_scale) @ V):
```
from flash_attn      import flash_attn_func, flash_attn_varlen_func, flash_attn_with_kvcache
from flash_attn_v100 import flash_attn_func, flash_attn_varlen_func, flash_attn_with_kvcache
```
```
flash_attn_func(q, k, v, dropout_p=0.0, softmax_scale=None, causal=False,
                window_size=(-1, -1), softcap=0.0, alibi_slopes=None, deterministic=False):
"""
FlashAttention for dense tensors.

Arguments:
    q, k, v      : (batch_size, seqlen, nheads, headdim)
    dropout_p    : float. Dropout probability (set to 0.0 for evaluation).
    softmax_scale: float. Scaling of QK^T. Default: 1/sqrt(headdim).
    causal       : bool. Apply causal attention mask.
    window_size  : (left, right). Sliding window attention. (-1, -1) means full attention.
    softcap      : float. Softcap for attention scores (0.0 disables).
    alibi_slopes : (nheads,) or (batch_size, nheads). ALiBi bias.
    deterministic: bool. Deterministic backward (not fully supported).
Return:
    out: (batch_size, seqlen, nheads, headdim) or (out, lse, dmask)
"""
```
```
flash_attn_varlen_func(q, k, v, cu_seqlens_q, cu_seqlens_k, max_seqlen_q, max_seqlen_k,
                       dropout_p=0.0, softmax_scale=None, causal=False,
                       window_size=(-1, -1), softcap=0.0, alibi_slopes=None,
                       deterministic=False, return_attn_probs=False, block_table=None):
"""
FlashAttention for variable-length sequences.

Arguments:
    q, k, v                   : (total_seqlen, nheads, headdim) - packed sequences
    cu_seqlens_q, cu_seqlens_k: (batch_size+1,) Cumulative sequence lengths
    max_seqlen_q, max_seqlen_k: int. Maximum sequence lengths
    dropout_p                 : float. Dropout probability (set to 0.0 for evaluation)
    softmax_scale             : float. Scaling of QK^T. Default: 1/sqrt(headdim)
    causal                    : bool. Apply causal attention mask
    window_size               : (left, right). Sliding window attention. (-1, -1) means full attention
    softcap                   : float. Softcap for attention scores (0.0 disables)
    alibi_slopes              : (nheads,) or (batch_size, nheads). ALiBi bias
    deterministic             : bool. Deterministic backward (not fully supported)
    return_attn_probs         : bool. Return attention probabilities and log-sum-exp
    block_table               : Optional. PagedAttention block table

Return:
    out: (total_seqlen, nheads, headdim) or (out, lse, dmask)
"""
```
```
flash_attn_with_kvcache(q, k_cache, v_cache, k=None, v=None, rotary_cos=None, 
                        rotary_sin=None, cache_seqlens=None, cache_batch_idx=None,
                        cache_leftpad=None, block_table=None, softmax_scale=None,
                        causal=False, window_size=(-1, -1), softcap=0.0,
                        rotary_interleaved=True, alibi_slopes=None, num_splits=0,
                        return_softmax_lse=False):
"""
FlashAttention for incremental decoding with KV cache.

Arguments:
    q                     : (batch_size, seqlen_q, nheads, headdim) - New queries
    k_cache, v_cache      : (batch_size, max_seqlen_k, nheads_k, headdim) - KV cache (updated inplace)
    k, v                  : Optional. New KV to append to cache (batch_size, seqlen_k, nheads_k, headdim)
    rotary_cos, rotary_sin: Optional. Rotary embeddings for positional encoding
    cache_seqlens         : Optional. Current sequence lengths per batch (batch_size,) 
    cache_batch_idx       : Optional. Batch index remapping for cache
    cache_leftpad         : Optional. Left padding information for cache
    block_table           : Optional. PagedAttention block table
    softmax_scale         : float. Scaling of QK^T. Default: 1/sqrt(headdim)
    causal                : bool. Apply causal attention mask (typically True for autoregressive decoding)
    window_size           : (left, right). Sliding window attention. (-1, -1) means full attention
    softcap               : float. Softcap for attention scores (0.0 disables)
    rotary_interleaved    : bool. Whether to use interleaved rotary embeddings
    alibi_slopes          : (nheads,) or (batch_size, nheads). ALiBi bias
    num_splits            : int. Number of splits for parallel computation (0 = auto)
    return_softmax_lse    : bool. Return log-sum-exp with output

Return:
    out: (batch_size, seqlen_q, nheads, headdim) or (out, softmax_lse)
"""
```