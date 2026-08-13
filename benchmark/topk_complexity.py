"""Cost of differentiable top-k mechanisms at DiT-realistic sequence lengths.

Backs the Table 4 complexity claims with measured numbers, and answers the request
to compare against SOFT top-k (Xie et al., NeurIPS 2020).

The point is not raw FLOPs. A DiT calls the selection operator once per layer per
denoising step -- 24 x 28 = 672 times for SD3.5 at 28 steps -- so an operator whose
cost is dominated by *serial* iterations (SOFT top-k's Sinkhorn loop) pays that
launch latency 672 times over, even though each iteration is tiny.

Methods
  shiva            hard argsort+gather forward; O(N^2) pairwise soft rank in the
                   backward only, and only over the SCALAR score vector.
  neuralsort_ste   the repo's adaptation: build the N x N relaxed permutation,
                   take soft gates from its first k rows, hard-select in forward.
  neuralsort_full  the original formulation, sorted features = P_hat @ x -> O(N^2 D).
  soft_topk        entropic OT to two anchors, solved by serial Sinkhorn iterations.

Run:
    python -m benchmark.topk_complexity                      # default sweep
    python -m benchmark.topk_complexity --tokens 4096 --repeats 20
"""

import argparse
import time

import torch

from shiva_dit.utils.ste import ShivaTopkFunc
from baselines.neuralsort import deterministic_neuralsort

# SD3.5-Medium: 24 joint blocks, 28 denoising steps, CFG doubles the batch.
DIT_LAYERS = 24
DIT_STEPS = 28


# --------------------------------------------------------------------------- #
# device helpers -- MPS is asynchronous; without an explicit sync the timings
# come back monotonically decreasing and are meaningless.
# --------------------------------------------------------------------------- #
def sync(device):
    if device.startswith("cuda"):
        torch.cuda.synchronize()
    elif device.startswith("mps"):
        torch.mps.synchronize()


def allocated_mb(device):
    if device.startswith("cuda"):
        return torch.cuda.memory_allocated() / 1e6
    if device.startswith("mps"):
        return torch.mps.current_allocated_memory() / 1e6
    return float("nan")


def graph_memory_mb(fn, x, scores, k, ctx, device):
    """Activation memory still held by the autograd graph after the forward.

    This -- not the transient peak -- is what decides whether training fits, and it
    is where an O(N^2) relaxation shows up. MPS exposes no max_memory_allocated in
    torch 2.6, so measuring the retained delta is both portable and the more
    meaningful number; on CUDA the transient peak is reported alongside it.
    """
    if device.startswith("cuda"):
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
    elif device.startswith("mps"):
        torch.mps.empty_cache()
    sync(device)

    before = allocated_mb(device)
    x_ = x.detach().clone().requires_grad_(True)
    s_ = scores.detach().clone().requires_grad_(True)
    out = fn(x_, s_, k, ctx)  # graph deliberately kept alive
    sync(device)
    retained = allocated_mb(device) - before

    peak = (
        torch.cuda.max_memory_allocated() / 1e6 - before
        if device.startswith("cuda")
        else float("nan")
    )
    del out, x_, s_
    return retained, peak


# --------------------------------------------------------------------------- #
# methods
# --------------------------------------------------------------------------- #
def m_shiva(x, scores, k, ctx):
    x_sel, _, _, _ = ShivaTopkFunc.apply(
        x, scores, ctx["k_float"], ctx["Ts"], ctx["Tk"]
    )
    return x_sel


def m_neuralsort_ste(x, scores, k, ctx):
    B, N, D = x.shape
    s = scores.unsqueeze(-1)
    s_norm = (s - s.mean(1, keepdim=True)) / s.std(1, keepdim=True).clamp(min=1e-6)
    P_hat = deterministic_neuralsort(s_norm.clamp(-5, 5), ctx["tau"])
    soft_gate = P_hat[:, :k, :].sum(dim=1)

    _, topk_idx = torch.topk(scores, k, dim=1, sorted=False)
    sorted_idx, _ = torch.sort(topk_idx, dim=1)
    mask_hard = torch.zeros_like(soft_gate)
    mask_hard.scatter_(1, sorted_idx, 1.0)
    gate = soft_gate + (mask_hard - soft_gate).detach()

    gather_index = sorted_idx.unsqueeze(-1).expand(-1, -1, D)
    x_sel = torch.gather(x, 1, gather_index)
    return x_sel * torch.gather(gate, 1, sorted_idx).unsqueeze(-1)


def m_neuralsort_full(x, scores, k, ctx):
    """Original NeuralSort: the sorted sequence is P_hat @ x, hence O(N^2 D)."""
    s = scores.unsqueeze(-1)
    s_norm = (s - s.mean(1, keepdim=True)) / s.std(1, keepdim=True).clamp(min=1e-6)
    P_hat = deterministic_neuralsort(s_norm.clamp(-5, 5), ctx["tau"])
    x_sorted = torch.bmm(P_hat, x)
    return x_sorted[:, :k]


def m_soft_topk(x, scores, k, ctx):
    """SOFT top-k (Xie et al., NeurIPS 2020).

    Entropic OT between the N scores and two anchors {min, max}, with target
    marginals (k, N-k). Solved by Sinkhorn. Each iteration is cheap; the cost is
    that there are `sinkhorn_iters` of them and they are strictly sequential.
    """
    B, N, D = x.shape
    eps, iters = ctx["eps"], ctx["sinkhorn_iters"]

    s = scores.unsqueeze(-1)  # (B, N, 1)
    anchors = torch.stack(
        [scores.min(dim=1).values, scores.max(dim=1).values], dim=-1
    ).unsqueeze(1)  # (B, 1, 2)
    C = (s - anchors) ** 2  # (B, N, 2)

    mu = torch.full((B, N), 1.0 / N, device=x.device, dtype=x.dtype)
    nu = torch.stack(
        [
            torch.full((B,), (N - k) / N, device=x.device, dtype=x.dtype),
            torch.full((B,), k / N, device=x.device, dtype=x.dtype),
        ],
        dim=-1,
    )  # (B, 2)

    K = torch.exp(-C / eps)
    u = torch.ones_like(mu)
    for _ in range(iters):  # <-- strictly serial
        v = nu / (torch.bmm(K.transpose(1, 2), u.unsqueeze(-1)).squeeze(-1) + 1e-9)
        u = mu / (torch.bmm(K, v.unsqueeze(-1)).squeeze(-1) + 1e-9)

    plan = u.unsqueeze(-1) * K * v.unsqueeze(1)  # (B, N, 2)
    soft_ind = plan[..., 1] * N / max(k, 1)  # soft top-k indicator

    _, topk_idx = torch.topk(scores, k, dim=1, sorted=False)
    sorted_idx, _ = torch.sort(topk_idx, dim=1)
    hard = torch.zeros_like(soft_ind)
    hard.scatter_(1, sorted_idx, 1.0)
    gate = soft_ind + (hard - soft_ind).detach()

    gather_index = sorted_idx.unsqueeze(-1).expand(-1, -1, D)
    x_sel = torch.gather(x, 1, gather_index)
    return x_sel * torch.gather(gate, 1, sorted_idx).unsqueeze(-1)


METHODS = {
    "Shiva (ours)": m_shiva,
    "NeuralSort (STE adapt.)": m_neuralsort_ste,
    "NeuralSort (orig. P@x)": m_neuralsort_full,
    "SOFT top-k (Sinkhorn)": m_soft_topk,
}


# --------------------------------------------------------------------------- #
def bench(fn, x, scores, k, ctx, device, repeats, warmup=5):
    def step(backward):
        x_ = x.detach().clone().requires_grad_(True)
        s_ = scores.detach().clone().requires_grad_(True)
        out = fn(x_, s_, k, ctx)
        if backward:
            out.sum().backward()
        return out

    for _ in range(warmup):
        step(True)
    sync(device)

    t0 = time.perf_counter()
    for _ in range(repeats):
        step(False)
    sync(device)
    fwd = (time.perf_counter() - t0) / repeats * 1e3

    t0 = time.perf_counter()
    for _ in range(repeats):
        step(True)
    sync(device)
    total = (time.perf_counter() - t0) / repeats * 1e3

    retained, peak = graph_memory_mb(fn, x, scores, k, ctx, device)
    return fwd, max(total - fwd, 0.0), retained, peak


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--tokens", type=int, nargs="+", default=[1024, 2048, 4096])
    p.add_argument("--dim", type=int, default=1536)
    p.add_argument("--batch", type=int, default=2, help="2 = CFG cond+uncond")
    p.add_argument("--ratio", type=float, default=0.6)
    p.add_argument("--repeats", type=int, default=10)
    p.add_argument("--sinkhorn-iters", type=int, default=50)
    p.add_argument(
        "--device",
        default="cuda"
        if torch.cuda.is_available()
        else ("mps" if torch.backends.mps.is_available() else "cpu"),
    )
    args = p.parse_args()

    dev = args.device
    print(
        f"device={dev}  dim={args.dim}  batch={args.batch}  r={args.ratio}  "
        f"repeats={args.repeats}  sinkhorn_iters={args.sinkhorn_iters}"
    )
    print(
        f"amortisation: {DIT_LAYERS} layers x {DIT_STEPS} steps "
        f"= {DIT_LAYERS * DIT_STEPS} operator calls per image\n"
    )

    for N in args.tokens:
        k = int(N * args.ratio)
        x = torch.randn(args.batch, N, args.dim, device=dev)
        scores = torch.randn(args.batch, N, device=dev)
        ctx = {
            "k_float": torch.tensor(float(k), device=dev),
            "Ts": torch.tensor(10.0, device=dev),
            "Tk": torch.tensor(10.0, device=dev),
            "tau": 8.0,
            "eps": 0.05,
            "sinkhorn_iters": args.sinkhorn_iters,
        }

        print(f"--- N={N}  k={k} ---")
        print(
            f"{'method':<26} {'fwd ms':>8} {'bwd ms':>8} {'act MB':>8} "
            f"{'fwd x672':>10} {'vs ours':>8}"
        )
        base = None
        for name, fn in METHODS.items():
            try:
                fwd, bwd, retained, peak = bench(
                    fn, x, scores, k, ctx, dev, args.repeats
                )
            except RuntimeError as e:
                print(f"{name:<26} FAILED: {str(e)[:60]}")
                continue
            if base is None:
                base = fwd
            total_s = fwd * DIT_LAYERS * DIT_STEPS / 1e3
            peak_str = "" if peak != peak else f" (peak {peak:.0f})"
            print(
                f"{name:<26} {fwd:>8.3f} {bwd:>8.3f} {retained:>8.1f} "
                f"{total_s:>9.2f}s {fwd / base:>7.1f}x{peak_str}"
            )
        print()


if __name__ == "__main__":
    main()
