import gc
import json
import time
import math
import argparse
from pathlib import Path
from contextlib import contextmanager

import torch
import torch.nn.functional as F
import numpy as np

from flash_attn import flash_attn_func

SEED         = 42
WARMUP_ITERS = 20
BENCH_ITERS  = 100
PEAK_TFLOPS  = 125.0
BANDWIDTH    = 900.0

torch.backends.cudnn.benchmark = False
torch.backends.cuda.matmul.allow_tf32 = False
torch.backends.cudnn.allow_tf32 = False

TEST_SHAPES = [
    # (B, M, H, D, causal, label)

    # Short sequences (decode-heavy)
    (1,   64,  16, 128, True,  "decode-tiny"),
    (1,  128,  16, 128, True,  "decode-small"),
    (2,  256,  16, 128, True,  "decode-medium"),
    (4,  512,  16, 128, True,  "short-context"),
    # Medium sequences
    (4, 1024,  16, 128, True,  "medium-short"),
    (4, 2048,  16, 128, True,  "medium-context"),
    (2, 4096,  16, 128, True,  "long-context"),
    (1, 8192,  16, 128, True,  "very-long-context"),

    # Very long sequences (16K, 32K)
    (1, 16384, 16, 128, True,  "16K-context"),
    (1, 32768, 16, 128, True,  "32K-context"),

    # Non-causal
    (4, 2048,  16, 128, False, "non-causal-medium"),
    (2, 4096,  16, 128, False, "non-causal-long"),

    # Different head configurations
    (4, 2048,  32,  64, True,  "more-heads-smaller-d"),
    (4, 2048,   8, 256, True,  "fewer-heads-larger-d"),
    (8, 1024,  16, 128, True,  "larger-batch"),
]

def set_seed(seed: int):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)

def reset_cuda_stats():
    gc.collect()
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    torch.cuda.reset_accumulated_memory_stats()
    torch.cuda.synchronize()

def cuda_timer_wrap(fn, n_iters: int):
    times = []
    for _ in range(n_iters):
        torch.cuda.synchronize()
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        fn()
        end.record()
        end.synchronize()
        times.append(start.elapsed_time(end))
    return times

def stats(times_ms):
    arr = np.array(times_ms)
    return {
        "median_ms": float(np.median(arr)),
        "mean_ms":   float(np.mean(arr)),
        "std_ms":    float(np.std(arr)),
        "min_ms":    float(np.min(arr)),
        "max_ms":    float(np.max(arr)),
        "p95_ms":    float(np.percentile(arr, 95)),
        "p99_ms":    float(np.percentile(arr, 99)),
    }

@torch.no_grad()
def reference_forward(q, k, v, causal: bool, softmax_scale: float):
    q_cpu = q.float().cpu().permute(0, 2, 1, 3).contiguous()
    k_cpu = k.float().cpu().permute(0, 2, 1, 3).contiguous()
    v_cpu = v.float().cpu().permute(0, 2, 1, 3).contiguous()

    out = F.scaled_dot_product_attention(
        q_cpu, k_cpu, v_cpu,
        attn_mask=None, dropout_p=0.0,
        is_causal=causal, scale=softmax_scale,
    )
    return out.permute(0, 2, 1, 3).contiguous()


def reference_backward(q, k, v, dout, causal: bool, softmax_scale: float):
    q_cpu = q.float().cpu().permute(0, 2, 1, 3).contiguous().detach()
    k_cpu = k.float().cpu().permute(0, 2, 1, 3).contiguous().detach()
    v_cpu = v.float().cpu().permute(0, 2, 1, 3).contiguous().detach()

    q_cpu.requires_grad_(True)
    k_cpu.requires_grad_(True)
    v_cpu.requires_grad_(True)

    out = F.scaled_dot_product_attention(
        q_cpu, k_cpu, v_cpu,
        attn_mask=None, dropout_p=0.0,
        is_causal=causal, scale=softmax_scale,
    )
    dout_cpu = dout.float().cpu().permute(0, 2, 1, 3).contiguous()
    out.backward(dout_cpu)

    assert q_cpu.grad is not None
    assert k_cpu.grad is not None
    assert v_cpu.grad is not None

    return (
        q_cpu.grad.permute(0, 2, 1, 3).contiguous(),
        k_cpu.grad.permute(0, 2, 1, 3).contiguous(),
        v_cpu.grad.permute(0, 2, 1, 3).contiguous(),
    )

def check_correctness(flash_attn_func, B, M, H, D, causal, softmax_scale):
    set_seed(SEED)
    device = torch.device("cuda")

    q = torch.randn(B, M, H, D, dtype=torch.float16, device=device)
    k = torch.randn(B, M, H, D, dtype=torch.float16, device=device)
    v = torch.randn(B, M, H, D, dtype=torch.float16, device=device)

    q.requires_grad_(True)
    k.requires_grad_(True)
    v.requires_grad_(True)

    out_flash = flash_attn_func(q, k, v, causal=causal, softmax_scale=softmax_scale)
    out_ref   = reference_forward(q, k, v, causal, softmax_scale)

    out_flash_f32 = out_flash.float().cpu().detach()
    fwd_abs_err = (out_flash_f32 - out_ref).abs()
    fwd_rel_err = fwd_abs_err / (out_ref.abs() + 1e-6)

    dout = torch.randn_like(out_flash)
    out_flash.backward(dout)
    dq_ref, dk_ref, dv_ref = reference_backward(q, k, v, dout, causal, softmax_scale)

    def grad_err(g_flash, g_ref):
        gf = g_flash.float().cpu().detach() if g_flash is not None else torch.zeros_like(g_ref)
        ae = (gf - g_ref).abs()
        re = ae / (g_ref.abs() + 1e-6)
        return {
            "max_abs": float(ae.max()),
            "mean_abs": float(ae.mean()),
            "max_rel": float(re.max()),
            "mean_rel": float(re.mean()),
            "finite": bool(torch.isfinite(gf).all()),
        }

    return {
        "forward": {
            "max_abs": float(fwd_abs_err.max()),
            "mean_abs": float(fwd_abs_err.mean()),
            "max_rel": float(fwd_rel_err.max()),
            "mean_rel": float(fwd_rel_err.mean()),
            "finite": bool(torch.isfinite(out_flash_f32).all()),
        },
        "dq": grad_err(q.grad, dq_ref),
        "dk": grad_err(k.grad, dk_ref),
        "dv": grad_err(v.grad, dv_ref),
    }

def benchmark(flash_attn_func, B, M, H, D, causal, softmax_scale):
    set_seed(SEED)
    device = torch.device("cuda")

    def make_inputs():
        q = torch.randn(B, M, H, D, dtype=torch.float16, device=device)
        k = torch.randn(B, M, H, D, dtype=torch.float16, device=device)
        v = torch.randn(B, M, H, D, dtype=torch.float16, device=device)
        q.requires_grad_(True)
        k.requires_grad_(True)
        v.requires_grad_(True)
        return q, k, v

    def fwd_only():
        q, k, v = make_inputs()
        out = flash_attn_func(q, k, v, causal=causal, softmax_scale=softmax_scale)
        return out

    def fwd_bwd():
        q, k, v = make_inputs()
        out  = flash_attn_func(q, k, v, causal=causal, softmax_scale=softmax_scale)
        dout = torch.randn_like(out)
        out.backward(dout)

    for _ in range(WARMUP_ITERS):
        fwd_only()
        fwd_bwd()
    torch.cuda.synchronize()

    reset_cuda_stats()
    fwd_times = cuda_timer_wrap(fwd_only, BENCH_ITERS)
    peak_fwd  = torch.cuda.max_memory_allocated()
    torch.cuda.synchronize()

    reset_cuda_stats()
    fb_times = cuda_timer_wrap(fwd_bwd, BENCH_ITERS)
    peak_fb  = torch.cuda.max_memory_allocated()
    torch.cuda.synchronize()

    # Forward: QK^T (2*B*H*M*M*D) + softmax*V (2*B*H*M*M*D) = 4*B*H*M^2*D
    flops_fwd = 4.0 * B * H * M * M * D
    flops_bwd = 10.0 * B * H * M * M * D

    # Input:  q, k, v = 3 * B * M * H * D * 2 (fp16)
    # Output: out = B * M * H * D * 2
    # LSE:    B * H * M * 4 (fp32)
    bytes_fwd = (3 + 1) * B * M * H * D * 2 + B * H * M * 4
    bytes_bwd = bytes_fwd + (3 * B * M * H * D * 2)

    fwd_stats = stats(fwd_times)
    bwd_stats = stats(fb_times)
    bwd_only_median = max(0, bwd_stats["median_ms"] - fwd_stats["median_ms"])

    tflops_fwd = flops_fwd / (fwd_stats["median_ms"] * 1e-3) / 1e12
    tflops_bwd = flops_bwd / (bwd_stats["median_ms"] * 1e-3) / 1e12
    gbps_fwd   = bytes_fwd / (fwd_stats["median_ms"] * 1e-3) / 1e9

    roofline_eff_fwd  = (tflops_fwd / PEAK_TFLOPS) * 100
    bandwidth_eff_fwd = (gbps_fwd   / BANDWIDTH)   * 100

    return {
        "forward": {
        **fwd_stats, "tflops": tflops_fwd, "gbps": gbps_fwd, "roofline_efficiency_pct": roofline_eff_fwd, "bandwidth_efficiency_pct": bandwidth_eff_fwd},
        "forward_backward":   {  **bwd_stats, "tflops": tflops_bwd, },
        "backward_only_ms":   bwd_only_median,
        "peak_memory_fwd_gb": peak_fwd / 1024**3,
        "peak_memory_fb_gb":  peak_fb / 1024**3,
        "flops_fwd":          flops_fwd,
        "bytes_moved_fwd":    bytes_fwd,
    }

def run(env_name: str, attn_fn, version: str, output_dir: Path):
    device = torch.device("cuda")
    print(f"\n{'='*78}")
    print(f"BENCHMARK: {env_name}")
    print(f"Device:    {torch.cuda.get_device_name(0)}")
    print(f"PyTorch:   {torch.__version__}")
    print(f"CUDA:      {torch.version.cuda}")
    print(f"API:       {version}")
    print(f"{'='*78}\n")

    results = {"env": env_name, "version": version, "tests": []}

    for B, M, H, D, causal, label in TEST_SHAPES:
        print(f"\n* [{label}] B={B} M={M} H={H} D={D} causal={causal}")
        softmax_scale = D ** -0.5

        reset_cuda_stats()

        try:
            corr = check_correctness(attn_fn, B, M, H, D, causal, softmax_scale)
            print(f"   fwd err:  max_abs={corr['forward']['max_abs']:.2e} mean_abs={corr['forward']['mean_abs']:.2e}  finite={corr['forward']['finite']}")
            print(f"   dq err:   max_abs={corr['dq']['max_abs']:.2e}  finite={corr['dq']['finite']}")
            print(f"   dk err:   max_abs={corr['dk']['max_abs']:.2e}  finite={corr['dk']['finite']}")
            print(f"   dv err:   max_abs={corr['dv']['max_abs']:.2e}  finite={corr['dv']['finite']}")
        except Exception as e:
            corr = {"error": str(e)}
            print(f"   CORRECTNESS FAILED: {e}")

        reset_cuda_stats()

        try:
            perf = benchmark(attn_fn, B, M, H, D, causal, softmax_scale)
            f = perf["forward"]
            fb = perf["forward_backward"]
            print(f"   fwd:      median={f['median_ms']:.3f} ms p95={f['p95_ms']:.3f}  TFLOPS={f['tflops']:.1f}  GB/s={f['gbps']:.1f}")
            print(f"   fwd+bwd:  median={fb['median_ms']:.3f} ms p95={fb['p95_ms']:.3f}  TFLOPS={fb['tflops']:.1f}")
            print(f"   bwd only: {perf['backward_only_ms']:.3f} ms (approx)")
            print(f"   peak mem: fwd={perf['peak_memory_fwd_gb']:.3f} GB fwd+bwd={perf['peak_memory_fb_gb']:.3f} GB")
            print(f"   roofline: {f['roofline_efficiency_pct']:.1f}% TFLOPS, {f['bandwidth_efficiency_pct']:.1f}% BW")
        except Exception as e:
            perf = {"error": str(e)}
            print(f"   BENCHMARK FAILED: {e}")

        results["tests"].append({
            "shape": {"B": B, "M": M, "H": H, "D": D, "causal": causal, "label": label},
            "correctness": corr,
            "performance": perf,
        })

    output_dir.mkdir(parents=True, exist_ok=True)
    out_path = output_dir / f"bench_{env_name}.json"
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\n* Results saved to: {out_path}")

    print(f"\n{'='*78}")
    print(f"SUMMARY [{env_name}]")
    print(f"{'='*78}")
    print(f"{'Label':<22} {'Fwd ms':>10} {'Bwd ms':>10} {'PeakGB':>10} {'TFLOPS':>10} {'BW%':>8}")
    print("=" * 78)
    for t in results["tests"]:
        lbl = t["shape"]["label"]
        fwd_ms = t["performance"].get("forward", {}).get("median_ms", "ERR")
        bwd_ms = t["performance"].get("backward_only_ms", "ERR")
        peak   = t["performance"].get("peak_memory_fb_gb", "ERR")
        tflops = t["performance"].get("forward", {}).get("tflops", "ERR")
        bw_eff = t["performance"].get("forward", {}).get("bandwidth_efficiency_pct", "ERR")

        fwd_s = f"{fwd_ms:.3f}" if isinstance(fwd_ms, float) else fwd_ms
        bwd_s = f"{bwd_ms:.3f}" if isinstance(bwd_ms, float) else bwd_ms
        pk_s  = f"{peak:.3f}"   if isinstance(peak,  float) else peak
        tf_s  = f"{tflops:.1f}" if isinstance(tflops, float) else tflops
        bw_s  = f"{bw_eff:.1f}" if isinstance(bw_eff, float) else bw_eff

        print(f"{lbl:<22} {fwd_s:>10} {bwd_s:>10} {pk_s:>10} {tf_s:>10} {bw_s:>8}")
    print(f"{'='*78}")

    return results

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Flash Attention Benchmark")
    parser.add_argument("--env", type=str, default=None, metavar="ENV_NAME", help="Environment name ('native_api' or 'v100_api')")
    parser.add_argument("--output", type=str, default="./results", help="Output directory for JSON results")

    args = parser.parse_args()

    if args.env == "native_api":
        import flash_attn as mod
        func = mod.flash_attn_func
        ver  = (f"flash_attn v{getattr(mod, '__version__', '?')} " f"(backend: flash_attn_v100 v{getattr(mod, '_v100_version', '?')})")
    else:
        import flash_attn_v100 as mod
        func = mod.flash_attn_func
        ver  = f"flash_attn_v100 v{getattr(mod, '__version__', '?')}"

    if args.env is None:
        parser.error("script must run as -> python benchmark_flash.py --env native_api or v100_api")

    run(env_name=args.env, attn_fn=func, version=ver, output_dir=Path(args.output))
