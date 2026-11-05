# ======================================================================================
# * Copyright (c) 2026, D.Skryabin / tg @ai_bond007 SPDX-License: BSD-3-Clause
# ======================================================================================
import torch
import sys
from itertools import accumulate

try:
    from flash_attn import flash_attn_varlen_func
except ImportError:
    flash_attn_varlen_func = None

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
# Reference (B, H, M, D) upcast=True fp32 truth; False fp16 baseline
# Right-align: row i ~ abs position i + (N - M)
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
        if slopes.dim() == 1:
            sl = slopes.view(1, -1, 1, 1)
        else:
            sl = slopes.view(slopes.shape[0], -1, 1, 1)
        s = s - sl.to(s.dtype) * dist.to(s.dtype)

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

def _cmp(xc, xr, xp, factor, name):
    xc = torch.nan_to_num(xc.float())
    xr = torch.nan_to_num(xr.float())
    xp = torch.nan_to_num(xp.float())
    err_c = (xc - xr).abs().max().item()
    err_p = (xp - xr).abs().max().item()
    thr = factor * err_p + 1e-5
    ok = err_c <= thr
    return ok, f"{name}={('OK  ' if ok else 'FAIL')}  err_c={err_c:.2e}  err_p={err_p:.2e}  thr={thr:.2e}"

# ======================================================================================
# Single varlen case
# ======================================================================================
def run_case_varlen(label, cfg, seed=0):
    seqs_q = cfg["seqs_q"]; seqs_k = cfg["seqs_k"]
    B = len(seqs_q)
    H_Q, H_K, D = cfg["H_Q"], cfg["H_K"], cfg["D"]
    causal, window, softcap = cfg["causal"], cfg["window"], cfg["softcap"]
    use_alibi, dropout_p = cfg["use_alibi"], cfg["dropout_p"]

    torch.manual_seed(seed); torch.cuda.manual_seed(seed)
    scale = D ** -0.5

    T_Q, T_K = sum(seqs_q), sum(seqs_k)
    max_sq, max_sk = max(seqs_q), max(seqs_k)

    cu_q = torch.tensor([0] + list(accumulate(seqs_q)), dtype=torch.int32, device=DEV)
    cu_k = torch.tensor([0] + list(accumulate(seqs_k)), dtype=torch.int32, device=DEV)

    q_d  = torch.randn(T_Q, H_Q, D, device=DEV, dtype=D2)
    k_d  = torch.randn(T_K, H_K, D, device=DEV, dtype=D2)
    v_d  = torch.randn(T_K, H_K, D, device=DEV, dtype=D2)
    dout = torch.randn_like(q_d)
    slopes = alibi_slopes(H_Q) if use_alibi else None

    q_c = q_d.detach().clone().requires_grad_()
    k_c = k_d.detach().clone().requires_grad_()
    v_c = v_d.detach().clone().requires_grad_()
    res = flash_attn_varlen_func(q_c, k_c, v_c, cu_q, cu_k, max_sq, max_sk, dropout_p=dropout_p, softmax_scale=scale, causal=causal, window_size=window, softcap=softcap, alibi_slopes=slopes, return_attn_probs=(dropout_p > 0))
    if isinstance(res, tuple):
        out_c, _, s_dmask = res
    else:
        out_c, s_dmask = res, None
    dq_c, dk_c, dv_c = torch.autograd.grad(out_c, (q_c, k_c, v_c), dout)

    dmask_list = []
    if dropout_p > 0 and s_dmask is not None and s_dmask.numel() > 0:
        for b in range(B):
            mb = ~torch.signbit(s_dmask[cu_q[b]:cu_q[b+1], :, :seqs_k[b]]).to(torch.bool)
            mb = mb.transpose(0, 1).unsqueeze(0)
            dmask_list.append(mb)

    err = {k: [0.0, 0.0] for k in ("out", "dq", "dk", "dv")}
    for b in range(B):
        sq_b, sk_b = seqs_q[b], seqs_k[b]
        qb  = q_d[cu_q[b]:cu_q[b+1]]
        kb  = k_d[cu_k[b]:cu_k[b+1]]
        vb  = v_d[cu_k[b]:cu_k[b+1]]
        dob = dout[cu_q[b]:cu_q[b+1]]
        dmb = dmask_list[b] if dmask_list else None

        qr = qb.float().unsqueeze(0).transpose(1, 2).requires_grad_()
        kr = kb.float().unsqueeze(0).transpose(1, 2).requires_grad_()
        vr = vb.float().unsqueeze(0).transpose(1, 2).requires_grad_()
        dr = dob.float().unsqueeze(0).transpose(1, 2)
        out_r = ref_attn(qr, kr, vr, scale, causal, window, softcap, slopes, dropout_p, dmb, upcast=True)
        dqr, dkr, dvr = torch.autograd.grad(out_r, (qr, kr, vr), dr)

        qp = qb.detach().unsqueeze(0).transpose(1, 2).requires_grad_()
        kp = kb.detach().unsqueeze(0).transpose(1, 2).requires_grad_()
        vp = vb.detach().unsqueeze(0).transpose(1, 2).requires_grad_()
        dp = dob.unsqueeze(0).transpose(1, 2)
        out_p = ref_attn(qp, kp, vp, scale, causal, window, softcap, slopes,dropout_p, dmb, upcast=False)
        dqp, dkp, dvp = torch.autograd.grad(out_p, (qp, kp, vp), dp)

        out_cb = out_c[cu_q[b]:cu_q[b+1]].transpose(0, 1).unsqueeze(0)
        dq_cb  = dq_c [cu_q[b]:cu_q[b+1]].transpose(0, 1).unsqueeze(0)
        dk_cb  = dk_c [cu_k[b]:cu_k[b+1]].transpose(0, 1).unsqueeze(0)
        dv_cb  = dv_c [cu_k[b]:cu_k[b+1]].transpose(0, 1).unsqueeze(0)

        for name, xc, xr, xp, f in [
            ("out", out_cb, out_r, out_p, 2.0),
            ("dq",  dq_cb,  dqr,   dqp,  3.0),
            ("dk",  dk_cb,  dkr,   dkp,  3.0),
            ("dv",  dv_cb,  dvr,   dvp,  3.0),
        ]:
            ok, _ = _cmp(xc, xr, xp, f, name)
            ec = (torch.nan_to_num(xc.float()) - torch.nan_to_num(xr.float())).abs().max().item()
            ep = (torch.nan_to_num(xp.float()) - torch.nan_to_num(xr.float())).abs().max().item()
            err[name][0] = max(err[name][0], ec)
            err[name][1] = max(err[name][1], ep)

    ok, lines = True, []
    for name, f in [("out", 2.0), ("dq", 3.0), ("dk", 3.0), ("dv", 3.0)]:
        ec, ep = err[name]
        thr = f * ep + 1e-5
        good = ec <= thr
        ok &= good
        lines.append(f"{name}={('OK  ' if good else 'FAIL')}  err_c={ec:.2e}  err_p={ep:.2e}  thr={thr:.2e}")

    finite_ok = all(torch.isfinite(t).all() for t in (out_c, dq_c, dk_c, dv_c))
    if not finite_ok:
        ok = False
        lines.append("FAIL: NaN/Inf in custom output")
    return ok, lines

# ======================================================================================
# Suite runner
# ======================================================================================
def run_suite(title, cases):
    print(f"\n{'=' * 80}\nSUITE: {title}\n{'=' * 80}")
    n_pass = 0
    for i, (label, cfg) in enumerate(cases, 1):
        try:
            ok, lines = run_case_varlen(label, cfg, seed=2000 + i)
        except torch.cuda.OutOfMemoryError:
            print(f"  [{i}] ⚠️  {label}  OOM (skip)"); continue
        except Exception as e:
            print(f"  [{i}] ❌ {label}  ERROR: {type(e).__name__}: {e}"); continue
        if ok:
            print(f"  [{i}] ✅ {label}"); n_pass += 1
        else:
            print(f"  [{i}] ❌ {label}")
            for line in lines: print(f"         {line}")
    print(f"  -> {n_pass}/{len(cases)}")
    return n_pass, len(cases)

# ======================================================================================
# Case builders (seqs_q / seqs_k drive B and total lengths)
# ======================================================================================
def _c(seqs_q, seqs_k, **kw):
    d = dict(H_Q=8, H_K=8, D=64, causal=False, window=(-1, -1), softcap=0.0, use_alibi=False, dropout_p=0.0)
    d.update(kw)
    d["seqs_q"] = list(seqs_q); d["seqs_k"] = list(seqs_k)
    return d

def suite_basic():
    return [
        ("B=1 sq=sk=256",            _c([256], [256])),
        ("B=1 sq=sk=256 causal",     _c([256], [256], causal=True)),
        ("B=2 same 128",             _c([128, 128], [128, 128])),
        ("B=2 same 128 causal",      _c([128, 128], [128, 128], causal=True)),
        ("B=3 varying sq=sk",        _c([64, 128, 96], [64, 128, 96])),
        ("B=3 varying causal",       _c([64, 128, 96], [64, 128, 96], causal=True)),
        ("B=4 seqs [16,32,64,128]",  _c([16, 32, 64, 128], [16, 32, 64, 128], causal=True)),
    ]

def suite_gqa():
    return [
        ("MHA 8:8",                  _c([128, 128],    [128, 128],    H_Q=8,  H_K=8)),
        ("GQA 8:2",                  _c([128, 128],    [128, 128],    H_Q=8,  H_K=2)),
        ("MQA 8:1",                  _c([128, 128],    [128, 128],    H_Q=8,  H_K=1)),
        ("GQA 16:4",                 _c([128, 128],    [128, 128],    H_Q=16, H_K=4)),
        ("MQA 16:1",                 _c([64, 128, 96], [64, 128, 96], H_Q=16, H_K=1)),
        ("GQA 32:8",                 _c([64, 128],     [64, 128],     H_Q=32, H_K=8, D=128)),
    ]

def suite_mixed_len():
    return [
        ("sq<sk 64/128 causal",                       _c([64],      [128], causal=True)),
        ("sq>sk 128/64 causal",                       _c([128],     [64],  causal=True)),
        ("sq<sk B=2 (64,128)/(128,256) causal",       _c([64, 128], [128, 256], causal=True)),
        ("sq>sk B=2 (128,64)/(64,128) causal",        _c([128, 64], [64, 128], causal=True)),
        ("B=2 long+short [16, 200]/[16, 200] causal", _c([16, 200], [16, 200], causal=True)),
        ("sq<sk 32/256 non-causal",                   _c([32],      [256], causal=False)),
    ]

def suite_window():
    return [
        ("win(32,32) causal",          _c([128, 128], [128, 128], causal=True,  window=(32, 32))),
        ("win(0,0) causal",            _c([128],      [128],      causal=True,  window=(0, 0))),
        ("win(-1,32) causal",          _c([128],      [128],      causal=True,  window=(-1, 32))),
        ("win(32,-1) causal",          _c([128],      [128],      causal=True,  window=(32, -1))),
        ("win(32,32) noncausal",       _c([128],      [128],      causal=False, window=(32, 32))),
        ("win(64,64) noncausal",       _c([64, 96],   [64, 96],   causal=False, window=(64, 64))),
    ]

def suite_softcap():
    return [
        ("softcap=5 causal",           _c([128], [128], causal=True,  softcap=5.0)),
        ("softcap=5 noncausal",        _c([128], [128], causal=False, softcap=5.0)),
        ("softcap=20 causal",          _c([128], [128], causal=True,  softcap=20.0)),
        ("softcap=0.5 noncausal",      _c([128], [128], causal=False, softcap=0.5)),
    ]

def suite_alibi():
    return [
        ("alibi causal GQA 8:2",       _c([128], [128], H_Q=8,  H_K=2, causal=True,  use_alibi=True)),
        ("alibi causal MQA 8:1",       _c([128], [128], H_Q=8,  H_K=1, causal=True,  use_alibi=True)),
        ("alibi noncausal GQA 8:2",    _c([128], [128], H_Q=8,  H_K=2, causal=False, use_alibi=True)),
        ("alibi causal 16:4 B=2",      _c([128, 96], [128, 96], H_Q=16, H_K=4, causal=True, use_alibi=True)),
    ]

def suite_dropout():
    return [
        ("dropout=0.1 causal",         _c([128, 128], [128, 128], causal=True,  dropout_p=0.1)),
        ("dropout=0.5 causal",         _c([128],      [128],      causal=True,  dropout_p=0.5)),
        ("dropout=0.1 noncausal",      _c([128],      [128],      causal=False, dropout_p=0.1)),
        ("dropout=0.3 GQA 8:2 causal", _c([64, 96],   [64, 96],   H_Q=8, H_K=2, causal=True, dropout_p=0.3)),
        ("dropout=0.5 MQA 8:1",        _c([128],      [128],      H_Q=8, H_K=1, causal=False, dropout_p=0.5)),
        ("dropout=0.2 sq<sk causal",   _c([64],       [128],      causal=True, dropout_p=0.2)),
    ]

def suite_combined():
    return [
        ("win+softcap+alibi causal",   _c([128], [128], causal=True,  window=(32, 32), softcap=10.0, use_alibi=True)),
        ("win+softcap noncausal",      _c([128], [128], causal=False, window=(32, 32), softcap=10.0)),
        ("win+alibi causal GQA",       _c([128], [128], H_Q=8, H_K=2, causal=True, window=(32, 32), use_alibi=True)),
    ]

def suite_d_sweep():
    out = []
    for D in (16, 32, 64, 128, 256):
        out.append((f"D={D} causal",    _c([128], [128], H_Q=4, H_K=4, D=D, causal=True)))
        out.append((f"D={D} noncausal", _c([128], [128], H_Q=4, H_K=4, D=D, causal=False)))
    return out

def suite_edge():
    return [
        ("B=1 single-token sq=sk=1 causal",  _c([1],   [1],   causal=True)),
        ("B=1 sq=1 sk=64 causal decode",     _c([1],   [64],  causal=True)),
        ("B=8 tiny varying [1..8] causal",   _c([1,2,3,4,5,6,7,8], [1,2,3,4,5,6,7,8], causal=True)),
        ("B=2 sq>sk causal 64/32 + 32/64",   _c([64, 32], [32, 64], causal=True)),
        ("B=1 H_Q=64 H_K=8",                 _c([64], [64], H_Q=64, H_K=8)),
        ("B=2 heavy mixed [4, 256]/[4, 256]",_c([4, 256], [4, 256], causal=True)),
    ]

# ======================================================================================
# Determinism
# ======================================================================================
def test_determinism():
    print(f"\n{'=' * 80}\nSUITE: Determinism (3 runs, bitwise)\n{'=' * 80}")
    torch.manual_seed(321)
    sq, sk = [128, 96, 64], [128, 96, 64]
    T_Q, T_K = sum(sq), sum(sk)
    cu_q = torch.tensor([0] + list(accumulate(sq)), dtype=torch.int32, device=DEV)
    cu_k = torch.tensor([0] + list(accumulate(sk)), dtype=torch.int32, device=DEV)
    q = torch.randn(T_Q, 8, 128, device=DEV, dtype=D2)
    k = torch.randn(T_K, 8, 128, device=DEV, dtype=D2)
    v = torch.randn(T_K, 8, 128, device=DEV, dtype=D2)
    s = alibi_slopes(8)
    outs = []
    for _ in range(3):
        torch.manual_seed(42)
        outs.append(flash_attn_varlen_func(
            q, k, v, cu_q, cu_k, max(sq), max(sk),
            softmax_scale=128 ** -0.5, causal=True, alibi_slopes=s))
    if all(torch.equal(outs[0], o) for o in outs[1:]):
        print("  ✅ PASS: deterministic"); return 1, 1
    print("  ❌ FAIL: non-deterministic"); return 0, 1

# ======================================================================================
# Main
# ======================================================================================
def main():
    if not torch.cuda.is_available():
        print("❌ CUDA required"); sys.exit(1)
    if flash_attn_varlen_func is None:
        print("⚠️  flash_attn_varlen_func not available - nothing to test")
        sys.exit(0)

    cap = torch.cuda.get_device_capability()
    print(f"Device: {torch.cuda.get_device_name()} (cap {cap})")
    if cap < (7, 0):
        print("⚠️  Volta+ required")

    total_p = total_t = 0
    for title, builder in [
        ("Basic varlen",              suite_basic),
        ("GQA/MQA/MHA",               suite_gqa),
        ("Mixed seq lengths",         suite_mixed_len),
        ("Sliding window",            suite_window),
        ("Softcap",                   suite_softcap),
        ("ALiBi",                     suite_alibi),
        ("Dropout",                   suite_dropout),
        ("Combined features",         suite_combined),
        ("D sweep",                   suite_d_sweep),
        ("Edge cases",                suite_edge),
    ]:
        p, t = run_suite(title, builder())
        total_p += p; total_t += t

    p, t = test_determinism()
    total_p += p; total_t += t

    print(f"\n{'=' * 80}\n VARLEN: {total_p}/{total_t} passed\n{'=' * 80}")
    sys.exit(0 if total_p == total_t else 1)

if __name__ == "__main__":
    main()
