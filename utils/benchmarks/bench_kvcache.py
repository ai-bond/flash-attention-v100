# ======================================================================================
# * Copyright (c) 2026, D.Skryabin / tg @ai_bond007 SPDX-License: BSD-3-Clause
# ======================================================================================
import torch
import sys

try:
    from flash_attn import flash_attn_with_kvcache
except ImportError:
    flash_attn_with_kvcache = None

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

def apply_rope(x, cos, sin, rotary_dim, interleaved, positions):
    T, H, D = x.shape
    half = rotary_dim // 2
    c = cos[positions].view(T, 1, half)
    s = sin[positions].view(T, 1, half)
    out = x.clone()
    if interleaved:
        xe = x[:, :, 0:rotary_dim:2]
        xo = x[:, :, 1:rotary_dim:2]
        out[:, :, 0:rotary_dim:2] = xe * c - xo * s
        out[:, :, 1:rotary_dim:2] = xe * s + xo * c
    else:
        x1 = x[:, :, :half]
        x2 = x[:, :, half:rotary_dim]
        out[:, :, :half] = x1 * c - x2 * s
        out[:, :, half:rotary_dim] = x1 * s + x2 * c
    return out

def ref_attn_fwd(q, k, v, scale, causal, window, softcap, slopes, upcast=True):
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
    return torch.matmul(p, v).to(orig)

def write_cache(cache, bidx, start, data, paged, block_table, block_page):
    for i in range(data.shape[0]):
        pos = start + i
        if paged:
            phys = block_table[bidx, pos // block_page].item()
            cache[phys, pos % block_page] = data[i]
        else:
            cache[bidx, pos] = data[i]

def read_cache(cache, bidx, start, length, paged, block_table, block_page):
    out = torch.empty(length, *cache.shape[-2:], device=cache.device, dtype=cache.dtype)
    for i in range(length):
        pos = start + i
        if paged:
            phys = block_table[bidx, pos // block_page].item()
            out[i] = cache[phys, pos % block_page]
        else:
            out[i] = cache[bidx, pos]
    return out

def _cmp(xc, xr, xp, factor, name, eps=1e-5):
    xc = torch.nan_to_num(xc.float())
    xr = torch.nan_to_num(xr.float())
    xp = torch.nan_to_num(xp.float())
    err_c = (xc - xr).abs().max().item()
    err_p = (xp - xr).abs().max().item()
    thr = factor * err_p + eps
    ok = err_c <= thr
    return ok, f"{name}={('OK  ' if ok else 'FAIL')}  err_c={err_c:.2e}  err_p={err_p:.2e}  thr={thr:.2e}"

# ======================================================================================
# Single case
# ======================================================================================
def run_case_kvcache(cfg, seed=0):
    B, T_Q, T_NEW = cfg["B"], cfg["T_Q"], cfg["T_NEW"]
    H_Q, H_K, D = cfg["H_Q"], cfg["H_K"], cfg["D"]
    max_k = cfg["max_k"]
    cs_vals = cfg["cache_seqlens"]
    paged = cfg.get("paged", False)
    use_rope = cfg.get("use_rope", False)
    interleaved = cfg.get("interleaved", True)
    rotary_dim = cfg.get("rotary_dim", D if use_rope else 0)
    leftpad_vals = cfg.get("leftpad", None)
    cbi_vals = cfg.get("cache_batch_idx", None)
    causal = cfg.get("causal", False)
    window = cfg.get("window", (-1, -1))
    softcap = cfg.get("softcap", 0.0)
    use_alibi = cfg.get("use_alibi", False)

    torch.manual_seed(seed); torch.cuda.manual_seed(seed)
    scale = D ** -0.5

    cache_seqlens = torch.tensor(cs_vals, dtype=torch.int32, device=DEV)
    leftpad = torch.tensor(leftpad_vals, dtype=torch.int32, device=DEV) if leftpad_vals else None
    cache_bidx = torch.tensor(cbi_vals, dtype=torch.int32, device=DEV) if cbi_vals else None

    q = torch.randn(B, T_Q, H_Q, D, device=DEV, dtype=D2)
    k_new = torch.randn(B, T_NEW, H_K, D, device=DEV, dtype=D2) if T_NEW > 0 else None
    v_new = torch.randn(B, T_NEW, H_K, D, device=DEV, dtype=D2) if T_NEW > 0 else None

    if paged:
        block_page = 256
        max_pages = (max_k + block_page - 1) // block_page
        num_blocks = B * max_pages
        perm = torch.randperm(num_blocks, device=DEV).to(torch.int32)
        block_table = perm.view(B, max_pages).contiguous()
        k_cache_init = torch.zeros(num_blocks, block_page, H_K, D, device=DEV, dtype=D2)
        v_cache_init = torch.zeros(num_blocks, block_page, H_K, D, device=DEV, dtype=D2)
    else:
        block_page, block_table = 0, None
        k_cache_init = torch.zeros(B, max_k, H_K, D, device=DEV, dtype=D2)
        v_cache_init = torch.zeros(B, max_k, H_K, D, device=DEV, dtype=D2)

    if use_rope:
        max_pos = max_k + T_NEW + (max(leftpad_vals) if leftpad_vals else 0) + 10
        half_rot = rotary_dim // 2
        inv_freq = 1.0 / (10000.0 ** (torch.arange(0, half_rot, device=DEV, dtype=torch.float32) / half_rot))
        angles = torch.arange(max_pos, device=DEV, dtype=torch.float32).unsqueeze(1) * inv_freq.unsqueeze(0)
        rotary_cos = torch.cos(angles).to(D2)
        rotary_sin = torch.sin(angles).to(D2)
    else:
        rotary_cos = rotary_sin = None

    alibi = alibi_slopes(H_Q) if use_alibi else None

    for b in range(B):
        cbidx = cbi_vals[b] if cbi_vals else b
        cs, lp = cs_vals[cbidx], (leftpad_vals[cbidx] if leftpad_vals else 0)
        if cs > 0:
            rk = torch.randn(cs, H_K, D, device=DEV, dtype=D2)
            rv = torch.randn(cs, H_K, D, device=DEV, dtype=D2)
            write_cache(k_cache_init, cbidx, lp, rk, paged, block_table, block_page)
            write_cache(v_cache_init, cbidx, lp, rv, paged, block_table, block_page)

    k_cache_c, v_cache_c = k_cache_init.clone(), v_cache_init.clone()
    k_cache_r, v_cache_r = k_cache_init.clone(), v_cache_init.clone()

    out_c = flash_attn_with_kvcache(
        q, k_cache_c, v_cache_c,
        k=k_new, v=v_new,
        rotary_cos=rotary_cos, rotary_sin=rotary_sin,
        cache_seqlens=cache_seqlens,
        cache_batch_idx=cache_bidx,
        cache_leftpad=leftpad,
        block_table=block_table,
        softmax_scale=scale, causal=causal, window_size=window, softcap=softcap,
        rotary_interleaved=interleaved, alibi_slopes=alibi,
    )

    for b in range(B):
        cbidx = cbi_vals[b] if cbi_vals else b
        cs, lp = cs_vals[cbidx], (leftpad_vals[cbidx] if leftpad_vals else 0)
        if T_NEW > 0:
            kn = k_new[b].float()
            if use_rope:
                pos_new = torch.arange(cs + lp, cs + lp + T_NEW, device=DEV)
                kn = apply_rope(kn, rotary_cos.float(), rotary_sin.float(), rotary_dim, interleaved, pos_new)
            write_cache(k_cache_r, cbidx, lp + cs, kn.to(D2), paged, block_table, block_page)
            write_cache(v_cache_r, cbidx, lp + cs, v_new[b],  paged, block_table, block_page)

    out_r_fp32 = torch.zeros(B, T_Q, H_Q, D, device=DEV, dtype=torch.float32)
    out_r_fp16 = torch.zeros(B, T_Q, H_Q, D, device=DEV, dtype=D2)

    for b in range(B):
        cbidx = cbi_vals[b] if cbi_vals else b
        cs, lp = cs_vals[cbidx], (leftpad_vals[cbidx] if leftpad_vals else 0)
        total_N_b = cs + T_NEW
        if total_N_b == 0:
            continue
        kb = read_cache(k_cache_r, cbidx, lp, total_N_b, paged, block_table, block_page).float()
        vb = read_cache(v_cache_r, cbidx, lp, total_N_b, paged, block_table, block_page).float()

        qb = q[b].float()
        if use_rope:
            pos_q = torch.arange(cs + lp, cs + lp + T_Q, device=DEV)
            qb = apply_rope(qb, rotary_cos.float(), rotary_sin.float(), rotary_dim, interleaved, pos_q)

        q4 = qb.unsqueeze(0).transpose(1, 2)
        k4 = kb.unsqueeze(0).transpose(1, 2)
        v4 = vb.unsqueeze(0).transpose(1, 2)

        o32 = ref_attn_fwd(q4, k4, v4, scale, causal, window, softcap, alibi, upcast=True)
        o16 = ref_attn_fwd(q4.to(D2), k4.to(D2), v4.to(D2), scale, causal, window, softcap, alibi, upcast=False)
        out_r_fp32[b] = o32.squeeze(0).transpose(0, 1)
        out_r_fp16[b] = o16.squeeze(0).transpose(0, 1).to(D2)

    ok, msg = _cmp(out_c, out_r_fp32, out_r_fp16, 2.0, "out")

    kerr = (torch.nan_to_num(k_cache_c.float()) - torch.nan_to_num(k_cache_r.float())).abs().max().item()
    verr = (torch.nan_to_num(v_cache_c.float()) - torch.nan_to_num(v_cache_r.float())).abs().max().item()
    cache_ok = kerr < 1e-2 and verr < 1e-2

    finite_ok = torch.isfinite(out_c).all().item()
    all_ok = ok and cache_ok and finite_ok
    lines = [msg, f"k_cache_err={kerr:.2e}  v_cache_err={verr:.2e}"]
    if not finite_ok:
        lines.append("FAIL: NaN/Inf in custom output")
    return all_ok, lines

# ======================================================================================
# Suite runner + case builders
# ======================================================================================
def run_suite(title, cases):
    print(f"\n{'=' * 80}\nSUITE: {title}\n{'=' * 80}")
    n_pass = 0
    for i, (label, cfg) in enumerate(cases, 1):
        try:
            ok, lines = run_case_kvcache(cfg, seed=3000 + i)
        except torch.cuda.OutOfMemoryError:
            print(f"  [{i}] ⚠️  {label}  OOM (skip)"); continue
        except Exception as e:
            print(f"  [{i}] ❌ {label}  ERROR: {type(e).__name__}: {e}")
            import traceback; traceback.print_exc()
            continue
        if ok:
            print(f"  [{i}] ✅ {label}"); n_pass += 1
        else:
            print(f"  [{i}] ❌ {label}")
            for line in lines: print(f"         {line}")
    print(f"  -> {n_pass}/{len(cases)}")
    return n_pass, len(cases)

def _c(**kw):
    d = dict(B=1, T_Q=1, T_NEW=1, H_Q=8, H_K=8, D=64, max_k=512, cache_seqlens=[0], causal=False, window=(-1, -1), softcap=0.0, use_alibi=False, use_rope=False, interleaved=True, paged=False, leftpad=None, cache_batch_idx=None)
    d.update(kw)
    return d

def suite_basic():
    return [
        ("prefill cache=0 T=128",   _c(T_Q=128, T_NEW=128, cache_seqlens=[0])),
        ("append cache=128 T=32",   _c(T_Q=32, T_NEW=32, cache_seqlens=[128])),
        ("decode T_Q=1 T_NEW=1",    _c(T_Q=1, T_NEW=1, cache_seqlens=[100])),
        ("continuation T_NEW=0",    _c(T_Q=1, T_NEW=0, cache_seqlens=[100])),
        ("B=2 varying cache 0/128", _c(B=2, T_Q=32, T_NEW=32, cache_seqlens=[0, 128])),
    ]

def suite_gqa():
    return [
        ("MHA 8:8",   _c(H_Q=8,  H_K=8,  cache_seqlens=[64], T_Q=32, T_NEW=32)),
        ("GQA 8:2",   _c(H_Q=8,  H_K=2,  cache_seqlens=[64], T_Q=32, T_NEW=32)),
        ("MQA 8:1",   _c(H_Q=8,  H_K=1,  cache_seqlens=[64], T_Q=32, T_NEW=32)),
        ("GQA 16:4",  _c(H_Q=16, H_K=4,  cache_seqlens=[64], T_Q=32, T_NEW=32)),
        ("MQA 16:1",  _c(H_Q=16, H_K=1,  cache_seqlens=[64], T_Q=32, T_NEW=32)),
        ("GQA 32:8",  _c(H_Q=32, H_K=8,  D=128, cache_seqlens=[64], T_Q=32, T_NEW=32)),
    ]

def suite_causal():
    return [
        ("causal decode",    _c(causal=True, cache_seqlens=[128])),
        ("causal prefill",   _c(causal=True, cache_seqlens=[0], T_Q=128, T_NEW=128)),
        ("causal append",    _c(causal=True, cache_seqlens=[64], T_Q=32, T_NEW=32)),
        ("causal GQA 8:2",   _c(causal=True, H_Q=8, H_K=2, cache_seqlens=[64], T_Q=32, T_NEW=32)),
        ("causal B=2",       _c(causal=True, B=2, cache_seqlens=[32, 96], T_Q=16, T_NEW=16)),
    ]

def suite_window():
    return [
        ("win(32,32) noncausal", _c(window=(32, 32), cache_seqlens=[64], T_Q=32, T_NEW=32)),
        ("win(32,·) causal",     _c(causal=True, window=(32, 32), cache_seqlens=[64], T_Q=32, T_NEW=32)),
        ("win(0,0) causal",      _c(causal=True, window=(0, 0), cache_seqlens=[64])),
        ("win(-1,32) noncausal", _c(window=(-1, 32), cache_seqlens=[64], T_Q=32, T_NEW=32)),
    ]

def suite_softcap():
    return [
        ("softcap=5 decode",     _c(softcap=5.0, cache_seqlens=[64])),
        ("softcap=5 noncausal",  _c(softcap=5.0, cache_seqlens=[64], T_Q=32, T_NEW=32)),
        ("softcap=20 causal",    _c(softcap=20.0, causal=True, cache_seqlens=[64], T_Q=32, T_NEW=32)),
    ]

def suite_alibi():
    return [
        ("alibi decode causal",    _c(use_alibi=True, causal=True, cache_seqlens=[64])),
        ("alibi noncausal",        _c(use_alibi=True, cache_seqlens=[64], T_Q=32, T_NEW=32)),
        ("alibi GQA 8:2 causal",   _c(use_alibi=True, causal=True, H_Q=8, H_K=2, cache_seqlens=[64], T_Q=32, T_NEW=32)),
    ]

def suite_rope():
    return [
        ("RoPE split decode causal",        _c(use_rope=True, interleaved=False, causal=True, cache_seqlens=[64])),
        ("RoPE interleaved decode causal",  _c(use_rope=True, interleaved=True,  causal=True, cache_seqlens=[64])),
        ("RoPE split prefill causal",       _c(use_rope=True, interleaved=False, causal=True, cache_seqlens=[0], T_Q=128, T_NEW=128)),
        ("RoPE interleaved prefill causal", _c(use_rope=True, interleaved=True,  causal=True, cache_seqlens=[0], T_Q=128, T_NEW=128)),
        ("RoPE split rotary_dim=32",        _c(use_rope=True, interleaved=False, rotary_dim=32, causal=True, cache_seqlens=[64], T_Q=32, T_NEW=32)),
    ]

def suite_paged():
    return [
        ("paged decode",                    _c(paged=True, cache_seqlens=[64])),
        ("paged causal decode",             _c(paged=True, causal=True, cache_seqlens=[64])),
        ("paged prefill causal",            _c(paged=True, causal=True, cache_seqlens=[0], T_Q=128, T_NEW=128)),
        ("paged append B=2",                _c(paged=True, B=2, cache_seqlens=[32, 96], T_Q=32, T_NEW=32)),
        ("paged + RoPE interleaved causal", _c(paged=True, use_rope=True, interleaved=True,  causal=True, cache_seqlens=[64], T_Q=32, T_NEW=32)),
        ("paged + RoPE split causal",       _c(paged=True, use_rope=True, interleaved=False, causal=True, cache_seqlens=[64], T_Q=32, T_NEW=32)),
    ]

def suite_leftpad():
    return [
        ("leftpad decode causal",       _c(leftpad=[32], cache_seqlens=[64], causal=True)),
        ("leftpad B=2 varying",         _c(B=2, leftpad=[16, 48], cache_seqlens=[64, 96], T_Q=32, T_NEW=32, causal=True)),
    ]

def suite_cbi():
    return [
        ("cache_batch_idx identity", _c(B=2, cache_batch_idx=[0, 1], cache_seqlens=[64, 96], T_Q=32, T_NEW=32)),
        ("cache_batch_idx swap",     _c(B=2, cache_batch_idx=[1, 0], cache_seqlens=[64, 96], T_Q=32, T_NEW=32, causal=True)),
    ]

def suite_d_sweep():
    out = []
    for D in (16, 32, 64, 128, 256):
        out.append((f"D={D} decode",         _c(D=D, cache_seqlens=[64])))
        out.append((f"D={D} prefill causal", _c(D=D, causal=True, cache_seqlens=[0], T_Q=64, T_NEW=64)))
    return out

def suite_edge():
    return [
        ("empty cache + no new",      _c(cache_seqlens=[0], T_NEW=0, T_Q=1)),
        ("cache full 512 + T_NEW=0",  _c(cache_seqlens=[512], T_NEW=0, T_Q=1)),
        ("cache 511 + T_NEW=1",       _c(cache_seqlens=[511], T_NEW=1)),
        ("B=4 mixed lengths causal",  _c(B=4, cache_seqlens=[0, 32, 64, 96], T_Q=16, T_NEW=16, causal=True)),
        ("H_Q=64 H_K=8",              _c(H_Q=64, H_K=8, cache_seqlens=[64], T_Q=32, T_NEW=32)),
    ]

# ======================================================================================
# Determinism
# ======================================================================================
def test_determinism():
    print(f"\n{'=' * 80}\nSUITE: Determinism\n{'=' * 80}")
    torch.manual_seed(777)
    B, T_Q, T_NEW, H_Q, H_K, D, max_k = 1, 16, 16, 8, 8, 128, 512
    cs_vals = [64]
    q = torch.randn(B, T_Q, H_Q, D, device=DEV, dtype=D2)
    k_new = torch.randn(B, T_NEW, H_K, D, device=DEV, dtype=D2)
    v_new = torch.randn(B, T_NEW, H_K, D, device=DEV, dtype=D2)
    cache_seqlens = torch.tensor(cs_vals, dtype=torch.int32, device=DEV)
    s = alibi_slopes(H_Q)
    outs = []
    for _ in range(3):
        torch.manual_seed(42)
        kc = torch.zeros(B, max_k, H_K, D, device=DEV, dtype=D2)
        vc = torch.zeros(B, max_k, H_K, D, device=DEV, dtype=D2)
        for b in range(B):
            kc[b, :cs_vals[b]] = torch.randn_like(kc[b, :cs_vals[b]])
            vc[b, :cs_vals[b]] = torch.randn_like(vc[b, :cs_vals[b]])
        outs.append(flash_attn_with_kvcache(q, kc, vc, k=k_new, v=v_new, cache_seqlens=cache_seqlens, softmax_scale=D ** -0.5, causal=True, alibi_slopes=s))
    if all(torch.equal(outs[0], o) for o in outs[1:]):
        print("  ✅ PASS: deterministic"); return 1, 1
    print("  ❌ FAIL: non-deterministic"); return 0, 1

# ======================================================================================
# Main
# ======================================================================================
def main():
    if not torch.cuda.is_available():
        print("❌ CUDA required"); sys.exit(1)
    if flash_attn_with_kvcache is None:
        print("⚠️  flash_attn_with_kvcache not available"); sys.exit(0)

    cap = torch.cuda.get_device_capability()
    print(f"Device: {torch.cuda.get_device_name()} (cap {cap})")
    if cap < (7, 0):
        print("⚠️  Volta+ required")

    total_p = total_t = 0
    for title, builder in [
        ("Basic",         suite_basic),
        ("GQA/MQA/MHA",   suite_gqa),
        ("Causal",        suite_causal),
        ("Window",        suite_window),
        ("Softcap",       suite_softcap),
        ("ALiBi",         suite_alibi),
        ("RoPE",          suite_rope),
        ("Paged",         suite_paged),
        ("Leftpad",       suite_leftpad),
        ("CacheBatchIdx", suite_cbi),
        ("D sweep",       suite_d_sweep),
        ("Edge cases",    suite_edge),
    ]:
        p, t = run_suite(title, builder())
        total_p += p; total_t += t

    p, t = test_determinism()
    total_p += p; total_t += t

    print(f"\n{'=' * 80}\n KVCACHE: {total_p}/{total_t} passed\n{'=' * 80}")
    sys.exit(0 if total_p == total_t else 1)

if __name__ == "__main__":
    main()
