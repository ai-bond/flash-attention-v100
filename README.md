# FlashAttention for unsupported Tesla v100
This repository want to implement the official implementation of FlashAttention and [FlashAttention-2](https://github.com/ai-bond/flash-attention-v100/blob/main/docs/attention.md) under unsupported in TriDao repo [Nvidia Tesla V100](https://github.com/ai-bond/flash-attention-v100/blob/main/docs/volta.md)

**FlashAttention: Fast and Memory-Efficient Exact Attention with IO-Awareness** by Tri Dao, Daniel Y. Fu, Stefano Ermon, Atri Rudra, Christopher Ré

![FlashAttention](docs/fa2.jpeg)

### TD; TR

- **What is suboptimal but necessary due to Volta:**
  - Softmax: Improving numerical stability above 1e-3 to 5e-4
  - Error correction: Volta unsupport bfloat or tf32 and WMMA instructions can't use float32 as marix_a(b) only half. Code need error correction implementation of conversion.

### **Tests Speed Comparison with PyTorch (1000 iterations, warmps=10)**  
**All tests pass** (Forward & Backward correct at `atol=1e-3`).  

| Test Case (B, H, M, N, D, Causal)       | Forward Correct | Backward Correct | Memory Savings | Forward Speedup | Backward Speedup | Total Speedup | Notes |
|----------------------------------------|:---------------:|:----------------:|:--------------:|:---------------:|:----------------:|:-------------:|-------|
| **Small scale**                        |                 |                  |                |                 |                  |               |       |
| (1, 1, 16, 16, 16, False)              | ✅              | ✅               | ~0%            | 5.55×           | 26.65×           | 19.36×        | Extremely fast; kernel fusion shines |
| (1, 1, 16, 16, 16, True)               | ✅              | ✅               | ~0%            | 7.25×           | 27.93×           | 21.09×        | Causal adds negligible overhead |
| (1, 1, 32, 32, 32, False)              | ✅              | ✅               | ~0%            | 6.41×           | 24.25×           | 18.35×        | Consistent with D=16 trend |
| (1, 1, 32, 32, 32, True)               | ✅              | ✅               | ~0%            | 7.14×           | 29.22×           | 21.63×        | Backward peak at 29× |
| (1, 1, 64, 64, 64, False)              | ✅              | ✅               | ~1%            | 6.61×           | 25.95×           | 19.34×        | Optimal for WMMA (D=64) |
| (1, 1, 64, 64, 64, True)               | ✅              | ✅               | ~1%            | 7.23×           | 29.83×           | 21.81×        | Best backward perf in small scale |
| (1, 1, 128, 128, 128, False)           | ✅              | ✅               | ~3%            | 6.30×           | 26.63×           | 19.96×        | Slight forward dip, backward stable |
| (1, 1, 128, 128, 128, True)            | ✅              | ✅               | ~4%            | 7.48×           | 28.98×           | 21.81×        | Causal boosts forward by ~18% |
| (1, 1, 256, 256, 256, False)           | ✅              | ✅               | ~10%           | 6.46×           | 21.33×           | 16.87×        | Shared memory pressure visible |
| (1, 1, 256, 256, 256, True)            | ✅              | ✅               | ~12%           | 7.48×           | 23.96×           | 19.13×        | Causal mitigates D=256 overhead |
| **Medium scale**                       |                 |                  |                |                 |                  |               |       |
| (1, 16, 1024, 1024, 64, False)         | ✅              | ✅               | 82%            | 2.18×           | 4.77×            | 4.10×         | Memory savings dominate |
| (1, 16, 1024, 1024, 64, True)          | ✅              | ✅               | 82%            | 2.70×           | 5.53×            | 4.78×         | Causal improves forward by 24% |
| (1, 32, 1024, 1024, 64, False)         | ✅              | ✅               | 84%            | 2.16×           | 2.88×            | 2.70×         | Higher H reduces occupancy |
| (1, 32, 1024, 1024, 64, True)          | ✅              | ✅               | 84%            | 2.78×           | 3.13×            | 3.04×         | Causal still beneficial |
| (1, 16, 1024, 1024, 128, False)        | ✅              | ✅               | 73%            | 1.91×           | 2.75×            | 2.56×         | Larger D increases register pressure |
| (1, 16, 1024, 1024, 128, True)         | ✅              | ✅               | 73%            | 2.29×           | 3.12×            | 2.93×         | Consistent causal advantage |
| **Large scale**                        |                 |                  |                |                 |                  |               |       |
| (1, 32, 2048, 2048, 128, False)        | ✅              | ✅               | 85%            | 1.79×           | 1.42×            | 1.50×         | Memory-bound regime |
| (1, 32, 2048, 2048, 128, True)         | ✅              | ✅               | 85%            | 2.24×           | 1.66×            | 1.78×         | Causal forward gain: +25% |
| (1, 32, 4096, 4096, 128, False)        | ✅              | ✅               | 92%            | 1.80×           | 1.33×            | 1.42×         | Backward latency dominates |
| (1, 32, 4096, 4096, 128, True)         | ✅              | ✅               | 92%            | 2.27×           | 1.57×            | 1.71×         | Causal maintains ~2× forward gain |
| **Very large D**                       |                 |                  |                |                 |                  |               |       |
| (1, 16, 1024, 1024, 256, False)        | ✅              | ✅               | 61%            | 1.44×           | 1.20×            | 1.25×         | D=256 stresses shared memory |
| (1, 16, 1024, 1024, 256, True)         | ✅              | ✅               | 61%            | 1.63×           | 1.36×            | 1.41×         | Modest but consistent gains |
| (1, 32, 2048, 2048, 256, False)        | ✅              | ✅               | 75%            | 1.38×           | **1.03×**        | **1.05×**     | **No backward slowdown!** |
| (1, 32, 2048, 2048, 256, True)         | ✅              | ✅               | 75%            | 1.57×           | 1.05×            | 1.15×         | Causal recovers forward perf |
| (1, 32, 4096, 4096, 256, False)        | ✅              | ✅               | 86%            | 1.34×           | **1.06×**        | **1.02×**     | **Parity with PyTorch (no regression!)** |
| (1, 32, 4096, 4096, 256, True)         | ✅              | ✅               | 86%            | 1.55×           | 1.04×            | 1.14×         | Causal ensures net gain |