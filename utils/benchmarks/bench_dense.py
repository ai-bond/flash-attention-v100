# ======================================================================================
# * Copyright (c) 2026, D.Skryabin / tg @ai_bond007 SPDX-License: BSD-3-Clause
# ======================================================================================
import torch
import sys

try:
    from flash_attn import flash_attn_func
except ImportError:
    flash_attn_func = None

torch.backends.cuda.matmul.allow_tf32 = False
torch.backends.cudnn.allow_tf32 = False
torch.set_float32_matmul_precision("highest")

D2, DEV = torch.float16, "cuda"

# ======================================================================================
# Helpers
# ======================================================================================
def alibi_slopes(n):
    return -torch.tensor([2.0 ** (-8.0 * (i + 1) / n) for i in range(n)], dtype=torch.float32, device=DEV)

def softmax_sdpa(s):
    m = s.max(dim=-1, keepdim=True).values
    all_inf = torch.isinf(m) & (m < 0)
    m_safe = torch.where(all_inf, torch.zeros_like(m), m)
    p = torch.exp(s - m_safe)
    p_sum = p.sum(dim=-1, keepdim=True)
    p = torch.where(p_sum > 0, p / p_sum, torch.zeros_like(p))
    return torch.where(all_inf.expand_as(p), torch.zeros_like(p), p)

# ======================================================================================
# Reference (B, H, M, D). upcast=True fp32 truth; False fp16 baseline
# Right-align semantics: row i corresponds to absolute position i + (N-M)
# ======================================================================================
def ref_attn(q, k, v, scale, causal=False, window=(-1, -1), softcap=0.0, slopes=None, dropout_p=0.0, dropout_mask=None, upcast=True):
    orig = q.dtype
    if upcast:
        q, k, v = q.float(), k.float(), v.float()

    B, H_Q, M, D = q.shape
    N, H_K = k.shape[2], k.shape[1]
    if H_Q != H_K:
        k = k.repeat_interleave(H_Q // H_K, dim=1)
        v = v.repeat_interleave(H_Q // H_K, dim=1)

    s = torch.matmul(q, k.transpose(-2, -1)) * scale

    mi = torch.arange(M, device=q.device).view(-1, 1)
    ni = torch.arange(N, device=q.device).view(1, -1)
    offset = N - M

    if slopes is not None:
        dist = (mi + offset - ni).abs()
        s = s - slopes.view(1, -1, 1, 1).to(s.dtype) * dist.to(s.dtype)

    if softcap > 0:
        s = softcap * torch.tanh(s / softcap)

    if causal:
        s.masked_fill_(ni > mi + offset, float("-inf"))
    wl, wr = window
    if wl >= 0:
        s.masked_fill_(ni < mi + offset - wl, float("-inf"))
    if wr >= 0:
        s.masked_fill_(ni > mi + offset + wr, float("-inf"))

    p = softmax_sdpa(s)
    if dropout_p > 0 and dropout_mask is not None:
        p = p * dropout_mask.to(p.dtype) / (1.0 - dropout_p)

    return torch.matmul(p, v).to(orig)

# ======================================================================================
# Comparison
# ======================================================================================
def _cmp(xc, xr, xp, factor, name):
    xc = torch.nan_to_num(xc.float())
    xr = torch.nan_to_num(xr.float())
    xp = torch.nan_to_num(xp.float())
    err_c = (xc - xr).abs().max().item()
    err_p = (xp - xr).abs().max().item()
    thr   = factor * err_p + 1e-5
    ok    = err_c <= thr
    return ok, f"{name}={('OK  ' if ok else 'FAIL')}  err_c={err_c:.2e}  err_p={err_p:.2e}  thr={thr:.2e}"

# ======================================================================================
# Single case: custom vs fp32-truth vs fp16-baseline
# ======================================================================================
def run_case(label, B, M, N, H_Q, H_K, D, causal=False, window=(-1, -1), softcap=0.0, use_alibi=False, dropout_p=0.0, seed=0):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    scale = D ** -0.5

    q_d  = torch.randn(B, M, H_Q, D, device=DEV, dtype=D2)
    k_d  = torch.randn(B, N, H_K, D, device=DEV, dtype=D2)
    v_d  = torch.randn(B, N, H_K, D, device=DEV, dtype=D2)
    dout = torch.randn_like(q_d)
    slopes = alibi_slopes(H_Q) if use_alibi else None

    q_c = q_d.detach().clone().requires_grad_()
    k_c = k_d.detach().clone().requires_grad_()
    v_c = v_d.detach().clone().requires_grad_()
    res = flash_attn_func(q_c, k_c, v_c, dropout_p=dropout_p, softmax_scale=scale, causal=causal, window_size=window, softcap=softcap, alibi_slopes=slopes, return_attn_probs=(dropout_p > 0))
    if isinstance(res, tuple):
        out_c, _, s_dmask = res
    else:
        out_c, s_dmask = res, None
    dq_c, dk_c, dv_c = torch.autograd.grad(out_c, (q_c, k_c, v_c), dout)

    dmask = None
    if dropout_p > 0 and s_dmask is not None and s_dmask.numel() > 0:
        dmask = (~torch.signbit(s_dmask)).to(torch.bool)

    q_r = q_d.detach().float().transpose(1, 2).requires_grad_()
    k_r = k_d.detach().float().transpose(1, 2).requires_grad_()
    v_r = v_d.detach().float().transpose(1, 2).requires_grad_()
    dor = dout.transpose(1, 2).float()
    out_r = ref_attn(q_r, k_r, v_r, scale, causal, window, softcap, slopes, dropout_p, dmask, upcast=True)
    dq_r, dk_r, dv_r = torch.autograd.grad(out_r, (q_r, k_r, v_r), dor)

    q_p = q_d.detach().transpose(1, 2).requires_grad_()
    k_p = k_d.detach().transpose(1, 2).requires_grad_()
    v_p = v_d.detach().transpose(1, 2).requires_grad_()
    dop = dout.transpose(1, 2)
    out_p = ref_attn(q_p, k_p, v_p, scale, causal, window, softcap, slopes, dropout_p, dmask, upcast=False)
    dq_p, dk_p, dv_p = torch.autograd.grad(out_p, (q_p, k_p, v_p), dop)

    checks = [
        (out_c.transpose(1, 2), out_r, out_p, 2.0, "out"),
        (dq_c.transpose(1, 2),  dq_r,  dq_p,  3.0, "dq"),
        (dk_c.transpose(1, 2),  dk_r,  dk_p,  3.0, "dk"),
        (dv_c.transpose(1, 2),  dv_r,  dv_p,  3.0, "dv"),
    ]

    all_ok, lines = True, []

    for xc, xr, xp, f, name in checks:
        ok, line = _cmp(xc, xr, xp, f, name)
        all_ok &= ok
        lines.append(line)
    return all_ok, lines

# ======================================================================================
# Suite runner
# ======================================================================================
def run_suite(title, cases):
    print(f"\n{'=' * 80}\nSUITE: {title}\n{'=' * 80}")
    n_pass = 0
    for i, (label, cfg) in enumerate(cases, 1):
        try:
            ok, lines = run_case(label, seed=1000 + i, **cfg)
        except torch.cuda.OutOfMemoryError:
            print(f"  [{i}] ⚠️  {label}  OOM (skip)")
            continue
        except Exception as e:
            print(f"  [{i}] ❌ {label}  ERROR: {type(e).__name__}: {e}")
            continue
        if ok:
            print(f"  [{i}] ✅ {label}")
            n_pass += 1
        else:
            print(f"  [{i}] ❌ {label}")
            for line in lines:
                print(f"         {line}")
    print(f"  -> {n_pass}/{len(cases)}")
    return n_pass, len(cases)

# ======================================================================================
# Case builders
# ======================================================================================
def _c(**kw):
    d = dict(B=1, M=256, N=256, H_Q=8, H_K=8, D=64, causal=False, window=(-1, -1), softcap=0.0, use_alibi=False, dropout_p=0.0)
    d.update(kw)
    return d

def suite_gqa():
    return [
        ("MHA 8:8",        _c(H_Q=8,  H_K=8)),
        ("GQA 8:2",        _c(H_Q=8,  H_K=2)),
        ("MQA 8:1",        _c(H_Q=8,  H_K=1)),
        ("MHA 16:16",      _c(H_Q=16, H_K=16)),
        ("GQA 16:4",       _c(H_Q=16, H_K=4)),
        ("MQA 16:1",       _c(H_Q=16, H_K=1)),
        ("GQA 32:8 D=128", _c(H_Q=32, H_K=8, D=128, M=128, N=128)),
        ("B=2 4:2",        _c(B=2, H_Q=4, H_K=2, M=128, N=128)),
    ]

def suite_causal():
    return [
        ("causal MHA",         _c(H_Q=8,  H_K=8, causal=True)),
        ("causal GQA 8:2",     _c(H_Q=8,  H_K=2, causal=True)),
        ("causal MQA 8:1",     _c(H_Q=8,  H_K=1, causal=True)),
        ("M=1 decode",         _c(H_Q=8,  H_K=8, M=1,   N=256, causal=True)),
        ("M=1 GQA decode",     _c(H_Q=8,  H_K=2, M=1,   N=256, causal=True)),
        ("M<N causal 256/512", _c(H_Q=8,  H_K=8, M=256, N=512, causal=True)),
        ("M>N causal 512/256", _c(H_Q=8,  H_K=8, M=512, N=256, causal=True)),
        ("M=N=1 causal",       _c(H_Q=8,  H_K=8, M=1,   N=1,   causal=True)),
    ]

def suite_window():
    return [
        ("causal win(32,32)",      _c(H_Q=8, causal=True,  window=(32, 32))),
        ("causal win(0,0) self",   _c(H_Q=8, causal=True,  window=(0, 0))),
        ("causal win(-1,32)",      _c(H_Q=8, causal=True,  window=(-1, 32))),
        ("causal win(32,-1)",      _c(H_Q=8, causal=True,  window=(32, -1))),
        ("noncausal win(32,32)",   _c(H_Q=8, causal=False, window=(32, 32))),
        ("noncausal win(64,64)",   _c(H_Q=8, causal=False, window=(64, 64))),
        ("noncausal win(200,200)", _c(H_Q=8, causal=False, window=(200, 200))),
    ]

def suite_softcap():
    return [
        ("softcap=5 causal",     _c(H_Q=8, causal=True,  softcap=5.0)),
        ("softcap=5 noncausal",  _c(H_Q=8, causal=False, softcap=5.0)),
        ("softcap=20 causal",    _c(H_Q=8, causal=True,  softcap=20.0)),
        ("softcap=20 noncausal", _c(H_Q=8, causal=False, softcap=20.0)),
        ("softcap=0.5 causal",   _c(H_Q=8, causal=True,  softcap=0.5, M=128, N=128)),
    ]

def suite_alibi():
    return [
        ("alibi causal GQA 8:2",    _c(H_Q=8,  H_K=2, causal=True,  use_alibi=True)),
        ("alibi causal MQA 8:1",    _c(H_Q=8,  H_K=1, causal=True,  use_alibi=True)),
        ("alibi noncausal GQA 8:2", _c(H_Q=8,  H_K=2, causal=False, use_alibi=True)),
        ("alibi causal H=16",       _c(H_Q=16, H_K=4, causal=True,  use_alibi=True, M=128, N=128)),
    ]

def suite_dropout():
    return [
        ("dropout=0.1 causal",    _c(H_Q=8, H_K=8, causal=True,  dropout_p=0.1)),
        ("dropout=0.5 causal",    _c(H_Q=8, H_K=8, causal=True,  dropout_p=0.5)),
        ("dropout=0.1 noncausal", _c(H_Q=8, H_K=8, causal=False, dropout_p=0.1)),
        ("dropout=0.3 GQA 8:2",   _c(H_Q=8, H_K=2, causal=True,  dropout_p=0.3)),
        ("dropout=0.5 MQA 8:1",   _c(H_Q=8, H_K=1, causal=False, dropout_p=0.5)),
    ]

def suite_combined():
    return [
        ("win+softcap+alibi causal", _c(H_Q=8, H_K=8, causal=True,  window=(32, 32), softcap=10.0, use_alibi=True)),
        ("win+softcap noncausal",    _c(H_Q=8, H_K=8, causal=False, window=(32, 32), softcap=10.0)),
        ("win+alibi causal GQA",     _c(H_Q=8, H_K=2, causal=True,  window=(32, 32), use_alibi=True)),
    ]

def suite_d_sweep():
    out = []
    for D in (16, 32, 64, 128, 256):
        out.append((f"D={D} causal",    _c(H_Q=4, H_K=4, D=D, M=128, N=128, causal=True)))
        out.append((f"D={D} noncausal", _c(H_Q=4, H_K=4, D=D, M=128, N=128, causal=False)))
    return out

def suite_edge():
    return [
        ("tiny M=1 N=1",     _c(H_Q=2, H_K=2, M=1,   N=1,   causal=True)),
        ("M=2 N=128 causal", _c(H_Q=4, H_K=4, M=2,   N=128, causal=True)),
        ("M=128 N=2 causal", _c(H_Q=4, H_K=4, M=128, N=2,   causal=True)),
        ("B=4 GQA 4:1",      _c(B=4, H_Q=4, H_K=1, M=64,  N=64)),
        ("H_Q=32 H_K=32",    _c(H_Q=32, H_K=32, M=64,  N=64)),
        ("H_Q=64 H_K=8",     _c(H_Q=64, H_K=8,  M=64,  N=64)),
    ]

# ======================================================================================
# Determinism
# ======================================================================================
def test_determinism():
    print(f"\n{'=' * 80}\nSUITE: Determinism (3 runs, bitwise)\n{'=' * 80}")
    torch.manual_seed(123)
    q = torch.randn(1, 256, 8, 128, device=DEV, dtype=D2)
    k = torch.randn(1, 256, 8, 128, device=DEV, dtype=D2)
    v = torch.randn(1, 256, 8, 128, device=DEV, dtype=D2)
    s = alibi_slopes(8)
    outs = []
    for _ in range(3):
        torch.manual_seed(42)
        outs.append(flash_attn_func(q, k, v, softmax_scale=128 ** -0.5, causal=True, alibi_slopes=s))
    if all(torch.equal(outs[0], o) for o in outs[1:]):
        print("  ✅ PASS: deterministic")
        return 1, 1
    print("  ❌ FAIL: non-deterministic")
    return 0, 1

# ======================================================================================
# Main
# ======================================================================================
def main():
    if not torch.cuda.is_available():
        print("❌ CUDA required"); sys.exit(1)
    if flash_attn_func is None:
        print("⚠️  flash_attn not available - nothing to test"); sys.exit(0)

    cap = torch.cuda.get_device_capability()
    print(f"Device: {torch.cuda.get_device_name()} (cap {cap})")
    if cap < (7, 0):
        print("⚠️  Volta+ required")

    total_p = total_t = 0

    for title, builder in [
        ("GQA/MQA/MHA (non-causal)", suite_gqa),
        ("Causal",                   suite_causal),
        ("Sliding Window",           suite_window),
        ("Softcap",                  suite_softcap),
        ("ALiBi",                    suite_alibi),
        ("Dropout",                  suite_dropout),
        ("Combined features",        suite_combined),
        ("D sweep",                  suite_d_sweep),
        ("Edge cases",               suite_edge),
    ]:
        p, t = run_suite(title, builder())
        total_p += p; total_t += t

    p, t = test_determinism()
    total_p += p; total_t += t

    print(f"\n{'=' * 80}\n DENSE: {total_p}/{total_t} passed\n{'=' * 80}")
    sys.exit(0 if total_p == total_t else 1)

if __name__ == "__main__":
    main()
