# ======================================================================================
# * Copyright (c) 2026, D.Skryabin / tg @ai_bond007 SPDX-License: BSD-3-Clause
# ======================================================================================
import torch
import torch.nn.functional as F
import sys

try:
    from flash_attn.losses.cross_entropy import cross_entropy_loss
except ImportError:
    cross_entropy_loss = None

torch.backends.cuda.matmul.allow_tf32 = False
torch.backends.cudnn.allow_tf32 = False
torch.set_float32_matmul_precision("highest")

D2, DEV = torch.float16, "cuda"

# ======================================================================================
# Reference
# ======================================================================================
def _ref_lce(inp, w, tgt, logit_scale, ls, lse_sq, ignore_index, reduction, upcast=True):
    if upcast:
        logits = (inp.float() @ w.float().T) * logit_scale
    else:
        logits = (inp @ w.T).float() * logit_scale

    ce = F.cross_entropy(logits, tgt, reduction="none", label_smoothing=ls, ignore_index=ignore_index)

    if lse_sq > 0:
        lse = torch.logsumexp(logits, dim=-1)
        z = lse_sq * lse**2
        mask = (tgt != ignore_index) & (tgt >= 0) & (tgt < logits.size(-1))
        z = z * mask.float()
    else:
        z = torch.zeros_like(ce)

    loss_per_row = ce + z

    if reduction == "none":
        return loss_per_row, z

    mask = (tgt != ignore_index) & (tgt >= 0) & (tgt < logits.size(-1))
    if reduction == "mean":
        n = mask.sum().float().clamp_min(1.0)
        return loss_per_row.sum() / n, z.sum() / n
    return loss_per_row.sum(), z.sum()     # sum

# ======================================================================================
# Comparison
# ======================================================================================
def _cmp(xc, xr, xp, factor, name, eps=1e-5):
    xc = torch.nan_to_num(xc.float().flatten())
    xr = torch.nan_to_num(xr.float().flatten())
    xp = torch.nan_to_num(xp.float().flatten())
    err_c = (xc - xr).abs().max().item()
    err_p = (xp - xr).abs().max().item()
    thr = factor * err_p + eps
    ok = err_c <= thr
    return ok, f"{name}={('OK  ' if ok else 'FAIL')}  err_c={err_c:.2e}  err_p={err_p:.2e}  thr={thr:.2e}"

# ======================================================================================
# Single case
# ======================================================================================
def run_case(label, M, V, D, reduction="mean", ls=0.0, logit_scale=1.0, lse_sq=0.0, ignore_index=-100, n_ignore=0, seed=0):
    torch.manual_seed(seed); torch.cuda.manual_seed(seed)

    inp_d = torch.randn(M, D, device=DEV, dtype=D2)
    w_d   = torch.randn(V, D, device=DEV, dtype=D2)
    tgt   = torch.randint(0, V, (M,), device=DEV, dtype=torch.int64)
    if n_ignore > 0:
        n_ignore = min(n_ignore, M)
        idx = torch.randperm(M, device=DEV)[:n_ignore]
        tgt[idx] = ignore_index

    inp_c = inp_d.detach().clone().requires_grad_()
    w_c   = w_d.detach().clone().requires_grad_()
    loss_c, z_c = cross_entropy_loss(inp_c, w_c, tgt, reduction=reduction, label_smoothing=ls, logit_scale=logit_scale, lse_square_scale=lse_sq, ignore_index=ignore_index)

    inp_r = inp_d.detach().float().clone().requires_grad_()
    w_r   = w_d.detach().float().clone().requires_grad_()
    loss_r, z_r = _ref_lce(inp_r, w_r, tgt, logit_scale, ls, lse_sq, ignore_index, reduction, upcast=True)

    inp_p = inp_d.detach().clone().requires_grad_()
    w_p   = w_d.detach().clone().requires_grad_()
    loss_p, z_p = _ref_lce(inp_p, w_p, tgt, logit_scale, ls, lse_sq, ignore_index, reduction, upcast=False)

    grad_out = torch.randn_like(loss_c) if reduction == "none" else torch.tensor(1.0, device=DEV)
    loss_c.backward(grad_out)
    loss_r.backward(grad_out.to(loss_r.dtype))
    loss_p.backward(grad_out.to(loss_p.dtype))

    checks = [
        (loss_c,     loss_r,     loss_p,     2.0, "loss"),
        (z_c,        z_r,        z_p,        2.0, "z_loss"),
        (inp_c.grad, inp_r.grad, inp_p.grad, 3.0, "dInput"),
        (w_c.grad,   w_r.grad,   w_p.grad,   3.0, "dWeight"),
    ]

    all_ok, lines = True, []

    for xc, xr, xp, f, name in checks:
        ok, line = _cmp(xc, xr, xp, f, name)
        all_ok &= ok
        lines.append(line)

    finite_ok = all(torch.isfinite(t).all().item()
                    for t in (loss_c, z_c, inp_c.grad, w_c.grad))
    if not finite_ok:
        all_ok = False
        lines.append("FAIL: NaN/Inf in custom output")
    return all_ok, lines

# ============================================================================
# Suite runner
# ============================================================================
def run_suite(title, cases):
    print(f"\n{'=' * 80}\nSUITE: {title}\n{'=' * 80}")
    n_pass = 0
    for i, (label, cfg) in enumerate(cases, 1):
        try:
            ok, lines = run_case(label, seed=4000 + i, **cfg)
        except torch.cuda.OutOfMemoryError:
            print(f"  [{i}] ⚠️  {label}  OOM (skip)"); continue
        except Exception as e:
            print(f"  [{i}] ❌ {label}  ERROR: {type(e).__name__}: {e}")
            import traceback; traceback.print_exc(); continue
        if ok:
            print(f"  [{i}] ✅ {label}"); n_pass += 1
        else:
            print(f"  [{i}] ❌ {label}")
            for line in lines: print(f"         {line}")
    print(f"  -> {n_pass}/{len(cases)}")
    return n_pass, len(cases)

# ======================================================================================
# Case builders
# ======================================================================================
def _c(**kw):
    d = dict(M=64, V=128, D=64, reduction="mean", ls=0.0, logit_scale=1.0, lse_sq=0.0, ignore_index=-100, n_ignore=0)
    d.update(kw)
    return d

def suite_basic():
    return [
        ("none",  _c(reduction="none")),
        ("mean",  _c(reduction="mean")),
        ("sum",   _c(reduction="sum")),
        ("M=1",   _c(M=1)),
        ("M=2",   _c(M=2)),
        ("M=512", _c(M=512)),
    ]

def suite_reductions():
    return [
        ("none",        _c(reduction="none")),
        ("mean",        _c(reduction="mean")),
        ("sum",         _c(reduction="sum")),
        ("none ls=0.1", _c(reduction="none", ls=0.1)),
        ("mean ls=0.1", _c(reduction="mean", ls=0.1)),
    ]

def suite_label_smoothing():
    return [
        ("ls=0.0",     _c(ls=0.0)),
        ("ls=0.1",     _c(ls=0.1)),
        ("ls=0.5",     _c(ls=0.5)),
        ("ls=0.1 none", _c(ls=0.1, reduction="none")),
    ]

def suite_logit_scale():
    return [
        ("scale=0.5",  _c(logit_scale=0.5)),
        ("scale=1.0",  _c(logit_scale=1.0)),
        ("scale=2.0",  _c(logit_scale=2.0)),
        ("scale=10.0", _c(logit_scale=10.0)),
    ]

def suite_z_loss():
    return [
        ("lse_sq=0.0",     _c(lse_sq=0.0)),
        ("lse_sq=0.01",    _c(lse_sq=0.01)),
        ("lse_sq=0.1",     _c(lse_sq=0.1)),
        ("lse_sq=0.1 sum", _c(lse_sq=0.1, reduction="sum")),
    ]

def suite_ignore():
    return [
        ("all valid",      _c()),
        ("1/4 ignore",     _c(n_ignore=16)),
        ("half ignore",    _c(n_ignore=32)),
        ("all ignore",     _c(n_ignore=64)),
        ("ignore_index=0", _c(ignore_index=0, n_ignore=16)),
    ]

def suite_d_sweep():
    return [(f"D={D}", _c(D=D)) for D in (16, 32, 64, 128, 256)]

def suite_v_sweep():
    return [
        ("V=1",    _c(V=1)),
        ("V=15",   _c(V=15)),
        ("V=16",   _c(V=16)),
        ("V=17",   _c(V=17)),
        ("V=100",  _c(V=100)),
        ("V=127",  _c(V=127)),
        ("V=129",  _c(V=129)),
        ("V=1000", _c(V=1000)),
    ]

def suite_combined():
    return [
        ("ls+z",         _c(ls=0.1, lse_sq=0.1)),
        ("ls+scale",     _c(ls=0.1, logit_scale=2.0)),
        ("z+scale",      _c(lse_sq=0.1, logit_scale=2.0)),
        ("ls+ignore",    _c(ls=0.1, n_ignore=16)),
        ("z+ignore",     _c(lse_sq=0.1, n_ignore=16)),
        ("all features", _c(ls=0.1, lse_sq=0.1, logit_scale=0.5, n_ignore=16)),
    ]

def suite_edge():
    return [
        ("M=1 V=1 D=16",       _c(M=1, V=1, D=16)),
        ("M=1 all ignore",     _c(M=1, n_ignore=1)),
        ("M=1 V=1 all ignore", _c(M=1, V=1, D=16, n_ignore=1)),
        ("V=1 (all tgt=0)",    _c(V=1)),
        ("V=15 non-mult-16",   _c(V=15)),
        ("V=17 non-mult-16",   _c(V=17)),
        ("M=256 V=1000 D=128", _c(M=256, V=1000, D=128)),
    ]

# ======================================================================================
# Determinism
# ======================================================================================
def test_determinism():
    print(f"\n{'=' * 80}\nSUITE: Determinism (3 runs, bitwise)\n{'=' * 80}")
    torch.manual_seed(123)
    M, V, D = 64, 128, 64
    inp = torch.randn(M, D, device=DEV, dtype=D2)
    w   = torch.randn(V, D, device=DEV, dtype=D2)
    tgt = torch.randint(0, V, (M,), device=DEV, dtype=torch.int64)

    outs = []
    for _ in range(3):
        torch.manual_seed(42)
        loss, _ = cross_entropy_loss(inp, w, tgt, reduction="mean", label_smoothing=0.1, logit_scale=1.0, lse_square_scale=0.1, ignore_index=-100 )
        outs.append(loss.clone())

    if all(torch.equal(outs[0], o) for o in outs[1:]):
        print("  ✅ PASS: deterministic"); return 1, 1
    print("  ❌ FAIL: non-deterministic"); return 0, 1

# ======================================================================================
# Main
# ======================================================================================
def main():
    if not torch.cuda.is_available():
        print("❌ CUDA required"); sys.exit(1)
    if cross_entropy_loss is None:
        print("⚠️  cross_entropy_loss not available - nothing to test")
        sys.exit(0)

    cap = torch.cuda.get_device_capability()
    print(f"Device: {torch.cuda.get_device_name()} (cap {cap})")
    if cap < (7, 0):
        print("⚠️  Volta+ required")

    total_p = total_t = 0
    for title, builder in [
        ("Basic",           suite_basic),
        ("Reductions",      suite_reductions),
        ("Label smoothing", suite_label_smoothing),
        ("Logit scale",     suite_logit_scale),
        ("Z-loss",          suite_z_loss),
        ("Ignore index",    suite_ignore),
        ("D sweep",         suite_d_sweep),
        ("V sweep",         suite_v_sweep),
        ("Combined",        suite_combined),
        ("Edge cases",      suite_edge),
    ]:
        p, t = run_suite(title, builder())
        total_p += p; total_t += t

    p, t = test_determinism()
    total_p += p; total_t += t

    print(f"\n{'=' * 80}\n🏁 LCE: {total_p}/{total_t} passed\n{'=' * 80}")
    sys.exit(0 if total_p == total_t else 1)

if __name__ == "__main__":
    main()
