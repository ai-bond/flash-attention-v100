# FlashAttention for unsupported Tesla v100
This repository want to implement the official implementation of FlashAttention and [FlashAttention-2](https://github.com/ai-bond/flash-attention-v100/blob/main/docs/attention.md) under unsupported in TriDao repo [Nvidia Tesla V100](https://github.com/ai-bond/flash-attention-v100/blob/main/docs/volta.md)

**FlashAttention: Fast and Memory-Efficient Exact Attention with IO-Awareness** by Tri Dao, Daniel Y. Fu, Stefano Ermon, Atri Rudra, Christopher Ré

![FlashAttention](docs/fa2.jpeg)

**Already implemented**
| Aspect / Feature                             | FlashAttention-2 (Reference)                                                                 | Provided Volta Implementation                                                                 | Match? | Notes |
|---------------------------------------------|---------------------------------------------------------------------------------------------|----------------------------------------------------------------------------------------------|--------|-------|
| **Target Architecture**                     | Ampere (A100) and newer (uses `mma.sync` with Tensor Cores, async copies, etc.)             | Volta (SM70) only, uses `nvcuda::wmma` (synchronous, limited WMMA shapes)                    | ❌     | Volta lacks async copy and newer Tensor Core features; this is a necessary adaptation. |
| **Kernel Fusion**                           | Fully fused forward and backward passes (no materialization of attention matrix)            | Fully fused (no explicit S matrix stored globally)                                           | ✅     | Both avoid materializing full attention matrix. |
| **Tiling Strategy**                         | 2D tiling over both Q and KV sequences with careful shared memory reuse                     | 1D tiling over sequence length (Q in tiles of `BLOCK_M`, KV in `BLOCK_N`)                    | ⚠️     | Less optimal tiling due to Volta’s limited shared memory and lack of async copy. |
| **Shared Memory Reuse**                     | Aggressive reuse: same buffer for K/V/P/dS across phases                                    | Reuse present (e.g., `sKV_base` reused for K, then V, then dS), but less aggressive          | ⚠️     | Shared memory layout is fragmented; reuse is limited by Volta’s 96 KB limit and WMMA alignment needs. |
| **Online Softmax (Forward)**                | Yes — numerically stable online softmax across KV blocks                                    | Yes — uses `sRowMax`, `sRowSum`, and rescaling per block                                     | ✅     | Correctly implemented. |
| **Causal Masking**                          | Supported                                                                                   | Supported (explicit `if (IS_CAUSAL)` with `global_n > global_m`)                             | ✅     | Matches logic. |
| **Backward: dQ, dK, dV Separation**         | Separate kernels for dQ and (dK + dV)                                                       | Separate kernels (`flash_attention_backward_dq_kernel`, `flash_attention_backward_dkv_kernel`) | ✅     | Same strategy. |
| **Backward: Recomputation of S**            | Yes — S is recomputed from Q and K in backward to save memory                               | Yes — S recomputed in both dQ and dKV kernels                                                | ✅     | Memory-optimal. |
| **Backward: dS Computation**                | `dS = P * (dOV - row_dot)`                                                                  | Same formula used                                                                            | ✅     | Correct. |
| **Precision**                               | Typically fp16 inputs, fp32 accumulators                                                    | fp16 inputs, fp32 accumulators                                                               | ✅     | Matches. |
| **Use of WMMA / Tensor Cores**              | Uses `mma.sync` with flexible layouts (Ampere+)                                             | Uses `nvcuda::wmma` (Volta-only, fixed 16x16x16, row/col major)                              | ⚠️     | Functional but less flexible; Volta limitation. |
| **Memory Padding & Alignment**              | Careful alignment to 128B for shared memory banks                                            | Uses `PAD`, aligns buffers to 128B (`& ~127`)                                                | ✅     | Good practice for Volta. |
| **Vectorized Loads (uint4 / float4)**       | Yes — for coalesced memory access                                                           | Yes — uses `uint4` for fp16, `float4` for fp32                                               | ✅     | Efficient on Volta. |
| **dQ Kernel: Parallel Reduction for row_dot** | Yes — warp-level reduction                                                                  | Yes — uses `__shfl_xor_sync` for reduction                                                   | ✅     | Correct. |
| **dKV Kernel: Unified dK/dV Accumulation**  | Yes — accumulates both in shared memory before write                                        | Yes — `s_dK_accum` and `s_dV_accum`                                                          | ✅     | Matches. |
| **Support for D = 16, 32, 64, 128, 256**    | Typically supports powers of 2 up to 128 or 256                                             | Explicitly supports exactly these values                                                     | ✅     | Complete coverage. |
| **Shared Memory Budget ≤ 96 KB**            | Not applicable (Ampere has 164 KB+), but FA2 aims for ≤ 100 KB for compatibility             | Enforced via `TORCH_CHECK(smem <= MAX_SMEM)`                                                 | ✅     | Necessary for Volta. |
| **Prefetching / Async Copies**              | Uses async copies (`cp.async`) on Ampere+                                                   | ❌ Not possible on Volta                                                                     | ❌     | Volta limitation; impacts performance but not correctness. |
| **Register Pressure Optimization**          | High — minimizes spills via careful scheduling                                              | Moderate — uses shared memory heavily due to WMMA constraints                                | ⚠️     | Expected on Volta. |
| **Numerical Stability (exp clamping)**      | Clamps `S - lse` to [-80, 80] before `exp`                                                  | Yes — `fmaxf(-80.0f)`, `fminf(80.0f)`                                                        | ✅     | Correct. |
| **Handling of Edge Cases (small M/N)**      | Robust via masking and bounds checks                                                        | Uses `valid_q_rows`, `valid_k_rows`, and zero-padding                                        | ✅     | Handles boundaries correctly. |
| **Backward: dQ/dK Layout Consistency**      | Ensures correct matrix layout (row/col major) in GEMMs                                      | Uses consistent `row_major`/`col_major` as required by WMMA                                  | ✅     | Correct WMMA usage. |
| **Support for Non-Causal & Causal**         | Yes                                                                                         | Yes — templated on `IS_CAUSAL`                                                               | ✅     | Full support. |

### Summary

- ✅ **What is implemented correctly (FA2-aligned):**  
  - Fully fused attention without materializing S.  
  - Online softmax with numerical stability.  
  - Correct backward formulas and recomputation strategy.  
  - Proper causal masking.  
  - Memory alignment, vectorized loads, and boundary handling.  
  - Separate dQ and dKV kernels with shared memory accumulation.

- ⚠️ **What is suboptimal but necessary due to Volta constraints:**  
  - No async memory copies → higher latency.  
  - Limited tiling (1D over sequence) due to shared memory and WMMA restrictions.  
  - Less aggressive buffer reuse (e.g., separate buffers for Q, K, V, dO, etc.).  
  - Higher shared memory footprint per kernel due to padding and alignment needs.

- ❌ **What is missing (by hardware limitation, not design flaw):**  
  - Tensor Core async operations (`cp.async`).  
  - Advanced scheduling and register optimization seen in FA2 on Ampere+. 

**Tests speed comparsion with PyTorch on warmps 10 with 1000 iterations:**
| Test Case (B, H, M, N, D, Causal)       | Forward Correct (atol 1e-3) | Backward Correct (atol 1e-3) | Memory Savings | Forward Speedup | Backward Speedup | Total Speedup | Notes |
|----------------------------------------|-----------------|------------------|----------------|------------------|-------------------|----------------|-------|
| (1, 1, 16, 16, 16, False)              | ✅              | ✅               | ~0%            | 5.36×            | 21.81×            | 16.66×         | Small sizes: high speedup due to kernel fusion & no materialization |
| (1, 1, 16, 16, 16, True)               | ✅              | ✅               | ~0%            | 7.17×            | 21.43×            | 16.94×         | Causal adds mask but still very fast |
| (1, 1, 32, 32, 32, False)              | ✅              | ✅               | ~0%            | 10.09×           | 22.18×            | 18.25×         | Optimal tiling for D=32 |
| (1, 1, 32, 32, 32, True)               | ✅              | ✅               | ~0%            | 15.21×           | 24.54×            | 21.67×         | Causal improves relative forward gain |
| (1, 1, 64, 64, 64, False)              | ✅              | ✅               | ~1%            | 11.80×           | 21.78×            | 18.72×         | Peak efficiency for Volta WMMA |
| (1, 1, 64, 64, 64, True)               | ✅              | ✅               | ~1%            | 15.12×           | 25.68×            | 22.30×         | Strong causal benefit |
| (1, 1, 128, 128, 128, False)           | ✅              | ✅               | ~3%            | 9.41×            | 22.41×            | 18.22×         | Slight overhead from larger D |
| (1, 1, 128, 128, 128, True)            | ✅              | ✅               | ~4%            | 15.01×           | 26.60×            | 22.81×         | Causal continues to help |
| (1, 1, 256, 256, 256, False)           | ✅              | ✅               | ~10%           | 11.50×           | 18.49×            | 16.55×         | Shared memory pressure starts to appear |
| (1, 1, 256, 256, 256, True)            | ✅              | ✅               | ~12%           | 18.18×           | 24.81×            | 23.04×         | Causal mask reduces effective N, boosting perf |
| **Medium scale**                       |                 |                  |                |                  |                   |                |       |
| (1, 16, 1024, 1024, 64, False)         | ✅              | ✅               | 82%            | 3.00×            | 3.36×             | 3.28×          | Large M/N: memory savings dominate |
| (1, 16, 1024, 1024, 64, True)          | ✅              | ✅               | 82%            | 3.80×            | 3.73×             | 3.75×          | Causal improves forward significantly |
| (1, 32, 1024, 1024, 64, False)         | ✅              | ✅               | 84%            | 2.57×            | 2.10×             | 2.19×          | Higher H reduces per-kernel occupancy |
| (1, 32, 1024, 1024, 64, True)          | ✅              | ✅               | 84%            | 3.20×            | 2.44×             | 2.59×          | Causal still beneficial |
| (1, 16, 1024, 1024, 128, False)        | ✅              | ✅               | 73%            | 2.57×            | 2.28×             | 2.33×          | Larger D increases register pressure |
| (1, 16, 1024, 1024, 128, True)         | ✅              | ✅               | 73%            | 3.18×            | 2.52×             | 2.64×          | Consistent causal advantage |
| **Large scale**                        |                 |                  |                |                  |                   |                |       |
| (1, 32, 2048, 2048, 128, False)        | ✅              | ✅               | 85%            | 1.80×            | 1.10×             | 1.22×          | Near memory-bound regime; backward barely faster |
| (1, 32, 2048, 2048, 128, True)         | ✅              | ✅               | 85%            | 2.28×            | 1.28×             | 1.45×          | Causal helps forward more than backward |
| (1, 32, 4096, 4096, 128, False)        | ✅              | ✅               | 92%            | 1.79×            | 1.04×             | 1.17×          | Backward almost same as PyTorch; kernel latency dominates |
| (1, 32, 4096, 4096, 128, True)         | ✅              | ✅               | 92%            | 2.27×            | 1.23×             | 1.40×          | Causal still yields ~2× forward gain |
| **Very large D**                       |                 |                  |                |                  |                   |                |       |
| (1, 16, 1024, 1024, 256, False)        | ✅              | ✅               | 61%            | 1.65×            | 1.12×             | 1.22×          | D=256 stresses shared memory (96 KB limit) |
| (1, 16, 1024, 1024, 256, True)         | ✅              | ✅               | 61%            | 1.85×            | 1.25×             | 1.35×          | Modest gains due to memory constraints |
| (1, 32, 2048, 2048, 256, False)        | ✅              | ✅               | 75%            | 1.33×            | **1.17× slowdown**| **1.07× slowdown**| Shared memory pressure causes backward regression |
| (1, 32, 2048, 2048, 256, True)         | ✅              | ✅               | 75%            | 1.54×            | **1.08× slowdown**| ~1.0×          | Causal mitigates but not enough |
| (1, 32, 4096, 4096, 256, False)        | ✅              | ✅               | 86%            | 1.33×            | **1.20× slowdown**| **1.09× slowdown**| Kernel launch + memory latency dominates |
| (1, 32, 4096, 4096, 256, True)         | ✅              | ✅               | 86%            | 1.54×            | **1.10× slowdown**| ~1.0×          | Performance parity with PyTorch |