# ======================================================================================
# * Copyright (c) 2026, D.Skryabin / tg @ai_bond007 SPDX-License: BSD-3-Clause
# ======================================================================================
import torch
import torch.nn.functional as F

try:
    from flash_attn import flash_attn_func, flash_attn_varlen_func, flash_attn_with_kvcache
except ImportError:
    flash_attn_func = flash_attn_varlen_func = flash_attn_with_kvcache = None

try:
    from flash_attn.losses.cross_entropy import cross_entropy_loss
except ImportError:
    cross_entropy_loss = None


# ======================================================================================
# REFERENCE single function, upcast selects fp32 (truth) or fp16 (baseline)
# ======================================================================================
def ref_attn(q, k, v, causal=False, window_size=(-1, -1), softcap=0.0, scale=None, upcast=True):
    B, M, H, D = q.shape
    N, H_K = k.shape[1], k.shape[2]
    if scale is None:
        scale = D ** -0.5
    if H != H_K:
        k = k.repeat_interleave(H // H_K, dim=2)
        v = v.repeat_interleave(H // H_K, dim=2)
    if upcast:
        q, k, v = q.float(), k.float(), v.float()

    s = torch.einsum('bmhd,bnhd->bhmn', q, k) * scale
    if softcap > 0:
        s = softcap * torch.tanh(s / softcap)

    mi = torch.arange(M, device=q.device).view(-1, 1)
    ni = torch.arange(N, device=q.device).view(1, -1)
    wl, wr = window_size
    if causal:
        s = s.masked_fill(ni > mi + (N - M), float('-inf'))
        if wl >= 0:
            s = s.masked_fill(ni < mi + (N - M) - wl, float('-inf'))
    else:
        if wl >= 0:
            s = s.masked_fill(ni < mi - wl, float('-inf'))
        if wr >= 0:
            s = s.masked_fill(ni > mi + wr, float('-inf'))

    p = F.softmax(s, dim=-1)
    return torch.einsum('bhmn,bnhd->bmhd', p, v)


def check(name, ref, custom, pt, eps=1e-5):
    err_c  = (custom.float() - ref.float()).abs().max().item()
    err_pt = (pt.float()     - ref.float()).abs().max().item()
    ok = err_c <= 2.0 * err_pt + eps and torch.isfinite(custom).all().item()
    ratio = err_c / max(err_pt, 1e-12)
    print(f"    {'OK' if ok else 'FAIL':4s} {name:20s} err_c={err_c:.3e} err_pt={err_pt:.3e} ratio={ratio:.2f}")
    return ok


# ======================================================================================
# DENSE
# ======================================================================================
def test_dense():
    if flash_attn_func is None:
        print("[SKIP] dense"); return True
    print("[DENSE]")
    torch.manual_seed(0)
    B, M, N, H_Q, H_K, D = 2, 64, 64, 4, 2, 64
    dev, dt = "cuda", torch.float16
    q = torch.randn(B, M, H_Q, D, device=dev, dtype=dt, requires_grad=True)
    k = torch.randn(B, N, H_K, D, device=dev, dtype=dt, requires_grad=True)
    v = torch.randn(B, N, H_K, D, device=dev, dtype=dt, requires_grad=True)

    qr = q.detach().float().clone().requires_grad_()
    kr = k.detach().float().clone().requires_grad_()
    vr = v.detach().float().clone().requires_grad_()
    qp = q.detach().clone().requires_grad_()
    kp = k.detach().clone().requires_grad_()
    vp = v.detach().clone().requires_grad_()

    ok = True
    for cfg in [dict(causal=False),
                dict(causal=True),
                dict(causal=False, window_size=(8, 8)),
                dict(causal=False, softcap=5.0)]:
        print(f"  cfg={cfg}")
        out     = flash_attn_func(q, k, v, **cfg)
        out_ref = ref_attn(qr, kr, vr, upcast=True,  **cfg)
        out_pt  = ref_attn(qp, kp, vp, upcast=False, **cfg)
        ok &= check("fwd", out_ref, out, out_pt)

        dout = torch.randn_like(out)
        out.backward(dout); out_ref.backward(dout); out_pt.backward(dout)
        for n, xc, xr, xp in [("q", q, qr, qp), ("k", k, kr, kp), ("v", v, vr, vp)]:
            ok &= check(f"{n}.grad", xr.grad, xc.grad, xp.grad)
            xc.grad = xr.grad = xp.grad = None
    return ok


# ======================================================================================
# VARLEN  (mix sq==sk and sq<sk; no fully-masked rows for causal)
# ======================================================================================
def test_varlen():
    if flash_attn_varlen_func is None:
        print("[SKIP] varlen"); return True
    print("[VARLEN]")
    torch.manual_seed(0)
    B = 4
    sq, sk = [10, 20, 15, 16], [10, 25, 20, 16]
    H_Q, H_K, D = 4, 2, 32
    dev, dt = "cuda", torch.float16

    cu_q = torch.tensor([0] + torch.cumsum(torch.tensor(sq), 0).tolist(), device=dev, dtype=torch.int32)
    cu_k = torch.tensor([0] + torch.cumsum(torch.tensor(sk), 0).tolist(), device=dev, dtype=torch.int32)
    T_Q, T_K = int(cu_q[-1]), int(cu_k[-1])

    q = torch.randn(T_Q, H_Q, D, device=dev, dtype=dt, requires_grad=True)
    k = torch.randn(T_K, H_K, D, device=dev, dtype=dt, requires_grad=True)
    v = torch.randn(T_K, H_K, D, device=dev, dtype=dt, requires_grad=True)

    qr = q.detach().float().clone().requires_grad_()
    kr = k.detach().float().clone().requires_grad_()
    vr = v.detach().float().clone().requires_grad_()
    qp = q.detach().clone().requires_grad_()
    kp = k.detach().clone().requires_grad_()
    vp = v.detach().clone().requires_grad_()

    ok = True
    for causal in [False, True]:
        print(f"  causal={causal}")
        out = flash_attn_varlen_func(q, k, v, cu_q, cu_k, max(sq), max(sk), causal=causal)

        refs, pts = [], []
        for b in range(B):
            qb_r = qr[cu_q[b]:cu_q[b+1]].unsqueeze(0)
            kb_r = kr[cu_k[b]:cu_k[b+1]].unsqueeze(0)
            vb_r = vr[cu_k[b]:cu_k[b+1]].unsqueeze(0)
            refs.append(ref_attn(qb_r, kb_r, vb_r, causal=causal, upcast=True).squeeze(0))

            qb_p = qp[cu_q[b]:cu_q[b+1]].unsqueeze(0)
            kb_p = kp[cu_k[b]:cu_k[b+1]].unsqueeze(0)
            vb_p = vp[cu_k[b]:cu_k[b+1]].unsqueeze(0)
            pts.append(ref_attn(qb_p, kb_p, vb_p, causal=causal, upcast=False).squeeze(0))

        out_ref = torch.cat(refs, dim=0)
        out_pt  = torch.cat(pts,  dim=0)
        ok &= check("fwd", out_ref, out, out_pt)

        dout = torch.randn_like(out)
        out.backward(dout); out_ref.backward(dout); out_pt.backward(dout)
        for n, xc, xr, xp in [("q", q, qr, qp), ("k", k, kr, kp), ("v", v, vr, vp)]:
            ok &= check(f"{n}.grad", xr.grad, xc.grad, xp.grad)
            xc.grad = xr.grad = xp.grad = None
    return ok


# ======================================================================================
# KVCACHE (forward-only - no autograd in wrapper)
# ======================================================================================
def test_kvcache():
    if flash_attn_with_kvcache is None:
        print("[SKIP] kvcache"); return True
    print("[KVCACHE]  (forward-only)")
    torch.manual_seed(0)
    B, T_Q, T_NEW, H_Q, H_K, D, MAX_K = 2, 1, 4, 4, 2, 64, 64
    dev, dt = "cuda", torch.float16

    q       = torch.randn(B, T_Q, H_Q, D, device=dev, dtype=dt)
    k_new   = torch.randn(B, T_NEW, H_K, D, device=dev, dtype=dt)
    v_new   = torch.randn(B, T_NEW, H_K, D, device=dev, dtype=dt)
    k_cache = torch.zeros(B, MAX_K, H_K, D, device=dev, dtype=dt)
    v_cache = torch.zeros(B, MAX_K, H_K, D, device=dev, dtype=dt)
    cache_seqlens = torch.zeros(B, dtype=torch.int32, device=dev)

    out     = flash_attn_with_kvcache(q, k_cache, v_cache, k=k_new, v=v_new, cache_seqlens=cache_seqlens, causal=False)
    out_ref = ref_attn(q, k_new, v_new, causal=False, upcast=True)
    out_pt  = ref_attn(q, k_new, v_new, causal=False, upcast=False)
    return check("fwd", out_ref, out, out_pt)


# ======================================================================================
# CROSS-ENTROPY
# ======================================================================================
def test_cross_entropy():
    if cross_entropy_loss is None:
        print("[SKIP] cross_entropy"); return True
    print("[CROSS_ENTROPY]")
    torch.manual_seed(0)
    M, V, D = 64, 128, 32
    dev, dt = "cuda", torch.float16

    inp = torch.randn(M, D, device=dev, dtype=dt, requires_grad=True)
    w   = torch.randn(V, D, device=dev, dtype=dt, requires_grad=True)
    tgt = torch.randint(0, V, (M,), device=dev, dtype=torch.int64)

    ir = inp.detach().float().clone().requires_grad_()
    wr = w.detach().float().clone().requires_grad_()
    ip = inp.detach().clone().requires_grad_()
    wp = w.detach().clone().requires_grad_()

    ok = True
    for red in ["mean", "sum"]:
        print(f"  reduction={red}")

        res = cross_entropy_loss(inp, w, tgt, reduction=red)
        loss = res[0] if isinstance(res, tuple) else res

        loss_ref = F.cross_entropy(ir @ wr.T, tgt, reduction=red)
        loss_pt  = F.cross_entropy(ip @ wp.T, tgt, reduction=red)

        ok &= check("loss", loss_ref.detach().reshape(1), loss.detach().reshape(1), loss_pt.detach().reshape(1))

        loss.backward(); loss_ref.backward(); loss_pt.backward()
        ok &= check("input.grad",  ir.grad, inp.grad, ip.grad)
        ok &= check("weight.grad", wr.grad, w.grad,   wp.grad)
        inp.grad = w.grad = ir.grad = wr.grad = ip.grad = wp.grad = None
    return ok


# ======================================================================================
# MAIN
# ======================================================================================
if __name__ == "__main__":
    assert torch.cuda.is_available(), "CUDA required"
    torch.manual_seed(0); torch.cuda.manual_seed(0)

    results = {}
    for name, fn in [("dense",         test_dense),
                     ("varlen",        test_varlen),
                     ("kvcache",       test_kvcache),
                     ("cross_entropy", test_cross_entropy)]:
        try:
            results[name] = fn()
        except Exception as e:
            print(f"  [ERROR] {name}: {type(e).__name__}: {e}")
            results[name] = False

    print("\n=== SUMMARY ===")
    for n, r in results.items():
        print(f"  {n:15s}: {'PASS' if r else 'FAIL'}")
    print("DONE" if all(results.values()) else "FAILED")