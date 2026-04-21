# *
# * Copyright (c) 2025, D.Skryabin / tg @ai_bond007
# * SPDX-License-Identifier: BSD-3-Clause
# *

import gc
import torch
import time
import statistics
import math

try:
    import flash_attn_v100_cuda
except ImportError:
    print("flash_attn_v100_cuda not found. Skipping test.")
    exit(0)

def ref_mha_forward(q, k, v, scale=1.0, causal=False, upcast=True):
    """
    Queries [B,H,M,D] → Scores [B,H,M,N] → Weights [B,H,M,N] → Output [B,H,M,D]
      Keys [B,H,N,D] ↗              ↓              ↓
                               Causal Mask    Values [B,H,N,D] ↗
    """
    dtype_og = q.dtype
    if upcast:
        q, k, v = q.float(), k.float(), v.float()

    s = torch.einsum('bhmd,bhnd->bhmn', q, k) * scale
    if causal:
        mask = torch.triu(torch.ones(s.shape[-2], s.shape[-1], device=s.device, dtype=torch.bool), diagonal=1)
        s = s.masked_fill(mask, float('-inf'))
    p = torch.softmax(s, dim=-1)
    o = torch.einsum('bhmn,bhnd->bhmd', p, v)
    return o.to(dtype=dtype_og)

def ref_mha_backward(q, k, v, do, scale=1.0, causal=False, upcast=True):
    """
    Q [B,H,M,D] ────┐
    K [B,H,N,D] ────┼─→ QK^T * scale → Scores [B,H,M,N] → Softmax → Weights [B,H,M,N]
    V [B,H,N,D] ────┘                                                           │
    ↓                                                                           │
    dO [B,H,M,D] ←─ Gradient Flow ── Output [B,H,M,D] ←─── Weights @ V ←────────┘
    ↓              ↓              ↓
    dQ            dK             dV
    """
    q = q.detach().clone().requires_grad_(True)
    k = k.detach().clone().requires_grad_(True)
    v = v.detach().clone().requires_grad_(True)
    
    dtype_og = q.dtype
    if upcast:
        q, k, v = q.float(), k.float(), v.float()

    s = torch.einsum('bhmd,bhnd->bhmn', q, k) * scale
    if causal:
        mask = torch.triu(torch.ones(s.shape[-2], s.shape[-1], device=s.device, dtype=torch.bool), diagonal=1)
        s = s.masked_fill(mask, float('-inf'))
    p = torch.softmax(s, dim=-1)
    o_ref = torch.einsum('bhmn,bhnd->bhmd', p, v)

    grads = torch.autograd.grad(o_ref.to(dtype=dtype_og), (q, k, v), do)
    return tuple(g.to(dtype=dtype_og) for g in grads)

def ensure_contiguous(tensor):
    return tensor if tensor.is_contiguous() else tensor.contiguous()

def report_tensor_stats(name, tensor):
    finite_mask = torch.isfinite(tensor)
    if finite_mask.all():
        max_val = tensor.abs().max().item()
        mean_val = tensor.abs().mean().item()
        print(f"  {name}: max={max_val:.6e}, mean={mean_val:.6e}")
    else:
        num_nan = torch.isnan(tensor).sum().item()
        num_inf = torch.isinf(tensor).sum().item()
        print(f"  {name}: ❌ NaN={num_nan}, Inf={num_inf}")

def format_performance_comparison(custom_time, ref_time):
    if custom_time <= 0 or ref_time <= 0:
        return "N/A", "N/A", "N/A"

    speedup = ref_time / custom_time
    slowdown = custom_time / ref_time
    time_diff_percent = (custom_time - ref_time) / ref_time * 100
    return speedup, slowdown, time_diff_percent

def benchmark_kernel(kernel_func, num_warmup=3, num_runs=10):
    for _ in range(num_warmup):
        kernel_func()
        torch.cuda.synchronize()
    times = []
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    for _ in range(num_runs):
        start.record()
        kernel_func()
        end.record()
        end.synchronize()
        times.append(start.elapsed_time(end) / 1000.0)
    return statistics.median(times)

def measure_gpu_memory(kernel_func):
    torch.cuda.reset_peak_memory_stats()
    torch.cuda.empty_cache()
    gc.collect()
    kernel_func()
    torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats()
    kernel_func()
    torch.cuda.synchronize()
    peak_mem_bytes = torch.cuda.max_memory_allocated()
    return peak_mem_bytes / (1024 * 1024)

def test_combined():
    test_cases = [
        (1, 1, 16, 16, 16),
        (1, 1, 32, 32, 32),
        (1, 1, 64, 64, 64),
        (1, 1, 128, 128, 128),
        (1, 1, 256, 256, 256),

        (1, 16, 1024, 1024, 16),
        (1, 32, 1024, 1024, 16),

        (1, 16, 1024, 1024, 32),
        (1, 32, 1024, 1024, 32),

        (1, 16, 1024, 1024, 64),
        (1, 32, 1024, 1024, 64),

        (1, 16, 1024, 1024, 128),
        (1, 32, 2048, 2048, 128),
        (1, 32, 4096, 4096, 128),

        (1, 16, 1024, 1024, 256),
        (1, 32, 2048, 2048, 256),
        (1, 32, 4096, 4096, 256),
        (1, 32, 8192, 8192, 256),
    ]

    all_passed = True
    for B, H, M, N, D in test_cases:
        for causal in [False, True]:
            if causal and M > N:
                continue

            print(f"\n{'='*70}")
            print(f"Test: B={B}, H={H}, M={M}, N={N}, D={D}, causal={causal}")

            # For represent
            torch.manual_seed(421)

            # Generate reference input data forward+backward
            q = torch.randn(B, H, M, D, device='cuda', dtype=torch.float16)
            k = torch.randn(B, H, N, D, device='cuda', dtype=torch.float16)
            v = torch.randn(B, H, N, D, device='cuda', dtype=torch.float16)
            dO = torch.randn(B, H, M, D, device='cuda', dtype=torch.float16)

            softmax_scale = 1.0 / math.sqrt(D)

            # Check for contiguous
            q, k, v, dO = map(ensure_contiguous, (q, k, v, dO))

            # Unique buffers for ref/cuda forward+backward
            o_ref = torch.empty(B, H, M, D, device='cuda', dtype=torch.float16)
            o_custom = torch.empty(B, H, M, D, device='cuda', dtype=torch.float16)

            softmax_lse_ref = torch.empty(B, H, M, device='cuda', dtype=torch.float32)
            softmax_lse_custom = torch.empty(B, H, M, device='cuda', dtype=torch.float32)

            dQ_ref = torch.empty(B, H, M, D, device='cuda', dtype=torch.float16)
            dK_ref = torch.empty(B, H, N, D, device='cuda', dtype=torch.float16)
            dV_ref = torch.empty(B, H, N, D, device='cuda', dtype=torch.float16)

            dQ_custom = torch.empty(B, H, M, D, device='cuda', dtype=torch.float16)
            dK_custom = torch.empty(B, H, N, D, device='cuda', dtype=torch.float16)
            dV_custom = torch.empty(B, H, N, D, device='cuda', dtype=torch.float16)

            # Check for contiguous
            buffers = [o_ref, o_custom, softmax_lse_ref, softmax_lse_custom, dQ_ref, dK_ref, dV_ref, dQ_custom, dK_custom, dV_custom]
            o_ref, o_custom, softmax_lse_ref, softmax_lse_custom, dQ_ref, dK_ref, dV_ref, dQ_custom, dK_custom, dV_custom = map(ensure_contiguous, buffers)

            # Clean cache before PyTorch
            torch.cuda.empty_cache()
            torch.cuda.synchronize()
            gc.collect()
            time.sleep(0.5)
            torch.cuda.reset_peak_memory_stats()

            # Run PyTorch forward+backward
            ref_oom = False
            try:
                def run_ref_fwd():
                    nonlocal o_ref
                    o_ref = ref_mha_forward(q, k, v, softmax_scale, causal, upcast=True)

                def run_ref_bwd():
                    nonlocal dQ_ref, dK_ref, dV_ref
                    dQ_ref, dK_ref, dV_ref = ref_mha_backward(q, k, v, dO, softmax_scale, causal, upcast=True)

                ref_fwd_time = benchmark_kernel(run_ref_fwd)
                ref_bwd_time = benchmark_kernel(run_ref_bwd)
                ref_total_time = ref_fwd_time + ref_bwd_time

                def run_ref_total():
                    run_ref_fwd()
                    run_ref_bwd()
                ref_total_mem = measure_gpu_memory(run_ref_total)
            except torch.OutOfMemoryError:
                print("  ⚠️  PyTorch  OOM")
                torch.cuda.empty_cache()
                gc.collect()
                ref_oom = True
                ref_fwd_time = ref_bwd_time = ref_total_time = 0.0
                ref_total_mem = 0.0

            # Clean cache before Cuda
            torch.cuda.empty_cache()
            torch.cuda.synchronize()
            gc.collect()
            time.sleep(0.5)
            torch.cuda.reset_peak_memory_stats()

            # Run Cuda forward+backward
            def run_custom_fwd():
                nonlocal o_custom, softmax_lse_custom
                result = flash_attn_v100_cuda.fwd(q, k, v, None, None, 0.0, softmax_scale, causal, -1, -1, 0.0, False, None)
                o_custom = result[0]
                softmax_lse_custom = result[1]

            def run_custom_bwd():
                nonlocal dQ_custom, dK_custom, dV_custom
                result = flash_attn_v100_cuda.bwd(dO, q, k, v, o_custom, softmax_lse_custom, None, None, None, None, 0.0, softmax_scale, causal, -1, -1, 0.0, False, None, None)
                dQ_custom = result[0]
                dK_custom = result[1]
                dV_custom = result[2]

            custom_fwd_time = benchmark_kernel(run_custom_fwd)
            custom_bwd_time = benchmark_kernel(run_custom_bwd)
            custom_total_time = custom_fwd_time + custom_bwd_time

            custom_fwd_mem = measure_gpu_memory(run_custom_fwd)
            custom_bwd_mem = measure_gpu_memory(run_custom_bwd)
            def run_custom_total():
                run_custom_fwd()
                run_custom_bwd()
            custom_total_mem = measure_gpu_memory(run_custom_total)

            # Forward pass
            if ref_oom:
                has_nan_custom_fwd = torch.isnan(o_custom).any()
                has_inf_custom_fwd = torch.isinf(o_custom).any()
                if has_nan_custom_fwd or has_inf_custom_fwd:
                    print("  ❌ Forward: Custom kernel produced NaN/Inf")
                    all_passed = False
                    continue
                print("  ✅ Forward  OK")
                err_custom_fwd = err_pt_fwd = 0.0
            else:
                has_nan_custom_fwd = torch.isnan(o_custom).any()
                has_inf_custom_fwd = torch.isinf(o_custom).any()
                has_nan_ref_fwd = torch.isnan(o_ref).any()
                has_inf_ref_fwd = torch.isinf(o_ref).any()

                if has_nan_custom_fwd or has_inf_custom_fwd or has_nan_ref_fwd or has_inf_ref_fwd:
                    print("⚠️  NaN/Inf detected in forward!")
                    report_tensor_stats("Out (cust)", o_custom)
                    report_tensor_stats("Out (refr)", o_ref)
                    all_passed = False
                    continue

                # Native FP16 baseline for relative tolerance validation
                out_pt = ref_mha_forward(q, k, v, softmax_scale, causal, upcast=False)

                err_custom_fwd = (o_custom - o_ref).abs().max().item()
                err_pt_fwd = (out_pt - o_ref).abs().max().item()
                ok_fwd = err_custom_fwd <= 2.0 * err_pt_fwd + 1e-5 and torch.isfinite(o_custom).all()

                if not ok_fwd:
                    diff = torch.abs(o_custom - o_ref)
                    idx = diff.argmax()
                    max_diff = diff.max().item()
                    mean_diff = diff.mean().item()
                    rel_err = (diff / (o_ref.abs() + 1e-12)).max().item()
                    print(f"  ❌ Forward mismatch: max_diff={max_diff:.6e}, mean_diff={mean_diff:.6e}, max_rel_err={rel_err:.6e}")
                    print(f"     [PyTorch] sample: {o_ref[0,0,0,:7].cpu().numpy()}")
                    print(f"     [ CUDA  ] sample: {o_custom[0,0,0,:7].cpu().numpy()}")
                    print(f"     [ Error ] Max diff at idx={idx}: PyTorch={o_ref.flatten()[idx].item():.6e}, CUDA={o_custom.flatten()[idx].item():.6e}")
                    all_passed = False
                    continue
                else:
                    print("  ✅ Forward  match OK")

            # Backward pass
            if ref_oom:
                has_nan_custom_bwd = torch.isnan(dQ_custom).any() or torch.isnan(dK_custom).any() or torch.isnan(dV_custom).any()
                has_inf_custom_bwd = torch.isinf(dQ_custom).any() or torch.isinf(dK_custom).any() or torch.isinf(dV_custom).any()
                if has_nan_custom_bwd or has_inf_custom_bwd:
                    print("  ❌ Backward: Custom kernel produced NaN/Inf")
                    all_passed = False
                    continue
                print("  ✅ Backward OK")
                err_dQ = err_dK = err_dV = err_pt_dQ = err_pt_dK = err_pt_dV = 0.0
            else:
                has_nan_custom_bwd = torch.isnan(dQ_custom).any() or torch.isnan(dK_custom).any() or torch.isnan(dV_custom).any()
                has_inf_custom_bwd = torch.isinf(dQ_custom).any() or torch.isinf(dK_custom).any() or torch.isinf(dV_custom).any()
                has_nan_ref_bwd = torch.isnan(dQ_ref).any() or torch.isnan(dK_ref).any() or torch.isnan(dV_ref).any()
                has_inf_ref_bwd = torch.isinf(dQ_ref).any() or torch.isinf(dK_ref).any() or torch.isinf(dV_ref).any()

                if has_nan_custom_bwd or has_inf_custom_bwd or has_nan_ref_bwd or has_inf_ref_bwd:
                    print("⚠️  NaN/Inf detected in backward!")
                    report_tensor_stats("dQ (custom)", dQ_custom)
                    report_tensor_stats("dK (custom)", dK_custom)
                    report_tensor_stats("dV (custom)", dV_custom)
                    report_tensor_stats("dQ (ref)", dQ_ref)
                    report_tensor_stats("dK (ref)", dK_ref)
                    report_tensor_stats("dV (ref)", dV_ref)
                    all_passed = False
                    continue

                # Native FP16 baseline for relative tolerance validation
                dQ_pt, dK_pt, dV_pt = ref_mha_backward(q, k, v, dO, softmax_scale, causal, upcast=False)

                err_dQ = (dQ_custom - dQ_ref).abs().max().item()
                err_dK = (dK_custom - dK_ref).abs().max().item()
                err_dV = (dV_custom - dV_ref).abs().max().item()
                err_pt_dQ = (dQ_pt - dQ_ref).abs().max().item()
                err_pt_dK = (dK_pt - dK_ref).abs().max().item()
                err_pt_dV = (dV_pt - dV_ref).abs().max().item()

                ok_dQ = err_dQ <= 3.0 * err_pt_dQ + 1e-4
                ok_dK = err_dK <= 3.0 * err_pt_dK + 1e-4
                ok_dV = err_dV <= 3.0 * err_pt_dV + 1e-4
                ok_bwd = ok_dQ and ok_dK and ok_dV and all(torch.isfinite(g).all() for g in (dQ_custom, dK_custom, dV_custom))

                if not ok_bwd:
                    print("\n--- Gradient Comparison ---")
                    for name, custom, ref, err_c, err_p, ok_flag in [
                        ("dQ", dQ_custom, dQ_ref, err_dQ, err_pt_dQ, ok_dQ),
                        ("dK", dK_custom, dK_ref, err_dK, err_pt_dK, ok_dK),
                        ("dV", dV_custom, dV_ref, err_dV, err_pt_dV, ok_dV),
                    ]:
                        status = "✅ OK" if ok_flag else f"❌ err={err_c:.2e} > 3×{err_p:.2e}"
                        print(f"  {name}: {status}")
                    all_passed = False
                    continue
                else:
                    print("  ✅ Backward match OK")

            print("Performance:")

            if not ref_oom:
                custom_tot = custom_total_mem
                torch_tot = ref_total_mem
                delta_mem = custom_tot - torch_tot
                pct_diff = (delta_mem / torch_tot) * 100 if torch_tot > 0 else 0.0

                print(f" (Mem):   Custom: {custom_total_mem:.1f} MB, PyTorch: {ref_total_mem:.1f} MB (Δ: {delta_mem:+.1f} MB, {pct_diff:+.1f}%)")
                for label, c_time, r_time in [
                    ("(fwd)", custom_fwd_time, ref_fwd_time),
                    ("(bwd)", custom_bwd_time, ref_bwd_time),
                    ("(tot)", custom_total_time, ref_total_time),
                ]:
                    speedup, slowdown, time_diff_percent = format_performance_comparison(c_time, r_time)
                    if speedup != "N/A":
                        if speedup > 1:
                            perf_info = f"Custom: {c_time*1000:.2f}ms, PyTorch: {r_time*1000:.2f}ms ({speedup:.2f}x speedup)"
                        else:
                            perf_info = f"Custom: {c_time*1000:.2f}ms, PyTorch: {r_time*1000:.2f}ms ({slowdown:.2f}x slowdown, +{time_diff_percent:+.1f}%)"
                    else:
                        perf_info = f"Custom: {c_time*1000:.2f}ms, PyTorch: {r_time*1000:.2f}ms"
                    print(f" {label}:   {perf_info}")
            else:
                print(f" (Mem):   Custom: {custom_total_mem:.1f} MB, PyTorch: OOM")
                for label, c_time in [
                    ("(fwd)", custom_fwd_time),
                    ("(bwd)", custom_bwd_time),
                    ("(tot)", custom_total_time),
                ]:
                    print(f" {label}:   Custom: {c_time*1000:.2f}ms, PyTorch: skipped")

            print("Validation:")
            if not ref_oom:
                print(f" (Fwd):   dO err={err_custom_fwd:.2e} ≤ 2×{err_pt_fwd:.2e}")
                print(f" (Bwd):   dQ err={err_dQ:.2e} ≤ 3×{err_pt_dQ:.2e}")
                print(f"          dK err={err_dK:.2e} ≤ 3×{err_pt_dK:.2e}")
                print(f"          dV err={err_dV:.2e} ≤ 3×{err_pt_dV:.2e}")
            else:
                print(" (Fwd):   Skipped")
                print(" (Bwd):   Skipped")

            ## Cleanup tensors
            del q, k, v, dO, o_custom, o_ref, dQ_custom, dK_custom, dV_custom, dQ_ref, dK_ref, dV_ref

    return all_passed

if __name__ == "__main__":
    if not torch.cuda.is_available():
        print("CUDA not available")
        exit(1)

    cap = torch.cuda.get_device_capability()
    if cap < (7, 0):
        print(f"Warning: device capability {cap} < (7,0). Volta (e.g., V100) required.")

    print(f"Running on {torch.cuda.get_device_name()} (capability {cap})")

    success = test_combined()
    if success:
        print("\nAll combined tests passed!")
    else:
        print("\nSome combined tests failed! Check mismatches above.")
        exit(1)
