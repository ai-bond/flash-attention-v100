# FlashAttention for unsupported Tesla v100

This repository implementation of [FlashAttention-2](https://github.com/ai-bond/flash-attention-v100/blob/main/utils/docs/attention.md) under unsupported in TriDao repo [Nvidia Tesla V100](https://github.com/ai-bond/flash-attention-v100/blob/main/utils/docs/volta.md)

> It attempt to build flash attention from scratch without "Vibe Code" for self education.

Last one available CUDA for Volta:
-------------

According to [Nvidia Deprecated Architectures](https://docs.nvidia.com/cuda/cuda-toolkit-release-notes/index.html#deprecated-architectures): Architecture support for Volta is considered feature-complete. Offline compilation and library support for these architectures have been removed in CUDA Toolkit 13.0 major version release.

```
# Download package
wget https://developer.download.nvidia.com/compute/cuda/12.9.1/local_installers/cuda_12.9.1_575.57.08_linux.run

# Install, this cuda package with NVIDIA driver version 575.57.08 that can be installed together
sudo sh cuda_12.9.1_575.57.08_linux.run

# Export and apply
cat >> ~/.bashrc << 'EOF'
export LD_LIBRARY_PATH=/usr/local/cuda/lib64:$LD_LIBRARY_PATH
export PATH=/usr/local/cuda/bin:$PATH
EOF
source ~/.bashrc
```

Deployment and compilation
-------------

```bash
# Create new python virtual env or use own existed:
python -m venv env
source env/bin/activate

# Update pip
pip install --upgrade pip

# Clone code and install packages:
git clone https://github.com/ai-bond/flash-attention-v100/
cd ./flash-attention-v100

# Install req packages
pip install -r requirements.txt
```
As NVIDIA deprecated Volta support in CUDA since viersion 13 then PyTorch also restrict and deprecated support in new versions:  [PyTorch is dropping Volta support from CUDA-12.8 binaries for release 2.11](https://dev-discuss.pytorch.org/t/dropping-volta-support-from-cuda-12-8-binaries-for-release-2-11/) and check [PyTorch \[release 2.8-2.9\] delete support for Maxwell, Pascal, and Volta architectures for CUDA 12.8 and 12.9 builds](https://github.com/pytorch/pytorch/issues/157517)

```bash
# Install last one PyTorch that's support with 12.9 CUDA
pip install torch==2.10.0+cu129 --index-url https://download.pytorch.org/whl/cu129

# Check is package supports Volta
python -c "import torch; p=torch.cuda.get_device_properties(0); print(f'{p.name} SM {p.major}.{p.minor} supported')"

# If you will see Tesla V100-XXX-XXGB SM 7.0 supported all is done.
# We can compile and install project with just:

./run.sh 

or 

pip install . --no-build-isolation -v
```
Also after you can final check ready venv

```
Successfully built flash_attn_v100
Installing collected packages: flash_attn_v100
Successfully installed flash_attn_v100-XX.XX

# just check exactly flash_attn import thru

python -c 'import flash_attn; print(f"Version: {flash_attn.__doc__}")'
Should: Flash Attention for Tesla V100 v2.8.3

and

pip show flash_attn
Name: flash-attn
Version: 2.8.3
Summary: Flash Attention for Tesla V100

```

And gl and hf :)

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
