import math
import torch

from flash_attn.losses.cross_entropy import CrossEntropyLoss

def reference_lce(X, W, T, *, ignore_index=-100, reduction="mean", label_smoothing=0.0, logit_scale=1.0, lse_square_scale=0.0):
    """Loss = LSE - (1-α)·tgt - α·mean + lse_ss·LSE²."""
    Xf = X.float()
    Wf = W.float()
    logits = (Xf @ Wf.t()) * logit_scale              # [N, V]
    lse    = torch.logsumexp(logits, dim=-1)          # [N]

    valid = (T != ignore_index) & (T >= 0) & (T < W.size(0))

    T_safe      = T.clamp(min=0)
    tgt_logit   = logits.gather(-1, T_safe.unsqueeze(-1)).squeeze(-1)
    mean_logits = logits.mean(dim=-1)

    ce = lse - (1.0 - label_smoothing) * tgt_logit - label_smoothing * mean_logits
    z  = lse_square_scale * lse * lse

    total = torch.where(valid, ce + z, torch.zeros_like(lse))
    z_out = torch.where(valid, z,      torch.zeros_like(lse))

    if reduction == "none":
        return total, z_out
    if reduction == "sum":
        return total.sum(), z_out.sum()
    if reduction == "mean":
        cnt = valid.sum().clamp_min(1).float()
        return total.sum() / cnt, z_out.sum() / cnt
    raise ValueError(f"Unknown reduction: {reduction}")


def report(name, a, b, rtol, atol):
    a = a.detach().float()
    b = b.detach().float()
    err   = (a - b).abs().max().item()
    scale = max(b.abs().max().item(), 1e-12)
    tol   = atol + rtol * scale
    ok    = err <= tol
    print(f"  {name:<22} max|Δ| = {err:.3e}   tol = {tol:.3e}   {'OK' if ok else 'FAIL'}")
    return ok


def diag(tag, gf, gr):
    mf  = gf.abs().max().item()
    mr  = gr.abs().max().item()
    ef  = (gf - gr).abs().max().item()
    rel = ef / max(mr, 1e-12)
    print(f"  {tag:<22} max|lce|={mf:.3e}  max|ref|={mr:.3e} max|Δ|={ef:.3e}  rel={rel:.2%}")

def run_case(name, *, N, D, V, ignore_index=-100, reduction="mean", label_smoothing=0.0, logit_scale=1.0, lse_square_scale=0.0, seed=0):
    torch.manual_seed(seed)
    dev = "cuda"
    X = (torch.randn(N, D, device=dev) / math.sqrt(D)).half().requires_grad_(True)
    W = (torch.randn(V, D, device=dev) / math.sqrt(D)).half().requires_grad_(True)
    T = torch.randint(0, V, (N,), dtype=torch.int64, device=dev)
    if ignore_index >= 0:
        T[0] = ignore_index

    fn = CrossEntropyLoss(
        ignore_index     = ignore_index,
        reduction        = reduction,
        label_smoothing  = label_smoothing,
        logit_scale      = logit_scale,
        lse_square_scale = lse_square_scale,
        return_z_loss    = True,
    )

    X_fa = X.detach().clone().requires_grad_(True)
    W_fa = W.detach().clone().requires_grad_(True)
    loss_fa, z_fa = fn(X_fa, W_fa, T)

    X_rf = X.detach().clone().requires_grad_(True)
    W_rf = W.detach().clone().requires_grad_(True)
    loss_rf, z_rf = reference_lce(
        X_rf, W_rf, T,
        ignore_index     = ignore_index,
        reduction        = reduction,
        label_smoothing  = label_smoothing,
        logit_scale      = logit_scale,
        lse_square_scale = lse_square_scale,
    )

    print(f"\n=== {name} ===")
    print(f"  reduction={reduction}  α={label_smoothing}  scale={logit_scale} lse_ss={lse_square_scale}  ignore={ignore_index}")

    ok = True
    ok &= report("forward loss",   loss_fa, loss_rf, rtol=2e-3, atol=2e-3)
    ok &= report("forward z-loss", z_fa,    z_rf,    rtol=2e-3, atol=2e-3)

    if loss_fa.dim() == 0:
        loss_fa.backward()
        loss_rf.backward()
    else:
        g = torch.ones_like(loss_fa)
        loss_fa.backward(g)
        loss_rf.backward(g)

    ok &= report("backward dX", X_fa.grad, X_rf.grad, rtol=2e-2, atol=1e-4)
    ok &= report("backward dW", W_fa.grad, W_rf.grad, rtol=2e-2, atol=1e-4)

    diag("diag loss", loss_fa,   loss_rf)
    diag("diag dX",   X_fa.grad, X_rf.grad)
    diag("diag dW",   W_fa.grad, W_rf.grad)

    if ignore_index >= 0:
        gx0 = X_fa.grad[0].abs().max().item()
        print(f"  max|grad X[ignored]|  = {gx0:.3e} {'OK' if gx0 < 1e-3 else 'FAIL'}")
        ok &= (gx0 < 1e-3)

    return ok

def main():
    N, D, V = 32, 64, 256

    print("=== LCE vs PyTorch reference ===")
    print("    Loss = LSE - (1-α)·tgt - α·mean + lse_ss·LSE²")

    ok = True
    ok &= run_case("CE mean",        N=N, D=D, V=V)
    ok &= run_case("CE none",        N=N, D=D, V=V, reduction="none")
    ok &= run_case("CE sum",         N=N, D=D, V=V, reduction="sum")
    ok &= run_case("LS α=0.1",       N=N, D=D, V=V, label_smoothing=0.1)
    ok &= run_case("LS α=0.3",       N=N, D=D, V=V, label_smoothing=0.3)
    ok &= run_case("z-loss only",    N=N, D=D, V=V, lse_square_scale=0.01)
    ok &= run_case("LS+z, scale=2",  N=N, D=D, V=V, label_smoothing=0.1, lse_square_scale=0.01, logit_scale=2.0)
    ok &= run_case("ignore_index=0", N=N, D=D, V=V, ignore_index=0, label_smoothing=0.1, lse_square_scale=0.01)

    print("\n" + ("✅ ALL TESTS PASSED" if ok else "❌ SOME TESTS FAILED"))

if __name__ == "__main__":
    main()