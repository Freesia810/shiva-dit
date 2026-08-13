"""Gradient-estimator validation on a REAL SD3 transformer block.

Appendix A.3 validates the estimator on a linear objective with N=50 and no network.
Its yardstick is sound -- for L = sum(x_sel) - sum(x_rej) the vector x.sum(-1) is the
exact discrete marginal, not a self-referential surrogate -- but the task is trivial,
so it says nothing about behaviour inside an actual DiT block.

This script keeps the same idea and swaps the task for the real thing:

  * architecture : diffusers JointTransformerBlock at SD3-medium dims
                   (dim=1536, 24 heads, head_dim=64), randomly initialised.
                   No checkpoint required -- we study the estimator, not the weights.
  * selection    : physical gather of the top-k image tokens, identity skip for the
                   rejected ones -- exactly what ShivaDitBlockHook does.
  * objective    : L(S) = || y(S) - y_full ||^2, the Stage-1 distillation loss the
                   router is actually trained with.
  * ground truth : exact discrete marginal at fixed budget k,
                   g_true[i] = L(S_i^+) - L(S_i^-),
                   obtained by really running the block on both configurations.

Reported:
  cos(g_hat, g_true)   -- does the single-path estimator (Eq. 23) point the right way
  cos(g_22,  g_true)   -- does the two-term form (Eq. 22) point the right way
  cos(g_gum, g_true)   -- Gumbel-Sigmoid straight-through, same yardstick
  cos(g_hat, g_22)     -- the quantity Fig. 7(b) reports, now on a real block

Run:  python -m test.toy.estimator_real_block --trials 30 --tokens 256
"""

import argparse
import time

import torch
import torch.nn.functional as F

from diffusers.models.attention import JointTransformerBlock

DIM, HEADS, HEAD_DIM = 1536, 24, 64
CTX_LEN = 77


class RealBlockTask:
    def __init__(
        self,
        n_tokens,
        device,
        dtype,
        generator,
        block=None,
        x=None,
        ctx=None,
        temb=None,
    ):
        self.device, self.dtype = device, dtype

        if block is None:
            block = JointTransformerBlock(
                dim=DIM, num_attention_heads=HEADS, attention_head_dim=HEAD_DIM
            )
        self.block = block.to(device=device, dtype=dtype).eval()
        for p in self.block.parameters():
            p.requires_grad_(False)

        def rnd(*shape):
            return torch.randn(*shape, generator=generator).to(
                device=device, dtype=dtype
            )

        self.x = rnd(1, n_tokens, DIM) if x is None else x.to(device, dtype)
        self.ctx = rnd(1, CTX_LEN, DIM) if ctx is None else ctx.to(device, dtype)
        self.temb = rnd(1, DIM) if temb is None else temb.to(device, dtype)
        self.N = self.x.shape[1]

        with torch.no_grad():
            _, y_full = self.block(self.x, self.ctx, self.temb)
        self.y_full = y_full

    @classmethod
    def from_capture(cls, path, device, dtype, generator, max_tokens=None):
        """Rebuild the exact situation dumped by test.toy.capture_block_inputs:
        the block's trained weights plus the tensors that actually entered it."""
        blob = torch.load(path, map_location="cpu", weights_only=False)
        cfg = blob["block_config"]

        block = JointTransformerBlock(
            dim=cfg["hidden_dim"],
            num_attention_heads=cfg["num_attention_heads"],
            attention_head_dim=cfg["attention_head_dim"],
            context_pre_only=cfg.get("context_pre_only", False),
        )
        block.load_state_dict(blob["block_state_dict"], strict=True)

        x = blob["hidden_states"][:1]
        if max_tokens is not None and x.shape[1] > max_tokens:
            # contiguous crop keeps neighbouring tokens together, so the spatial
            # redundancy the experiment is about survives the truncation
            start = (x.shape[1] - max_tokens) // 2
            x = x[:, start : start + max_tokens]

        task = cls(
            x.shape[1],
            device,
            dtype,
            generator,
            block=block,
            x=x,
            ctx=blob["encoder_hidden_states"][:1],
            temb=blob["temb"][:1],
        )
        task.meta = {k: blob[k] for k in ("layer", "step") if k in blob}
        return task

    def _loss(self, y):
        return ((y - self.y_full) ** 2).mean()

    def forward_pruned(self, sel_idx, rej_idx, need_grad=True):
        """Gather -> block -> scatter, with the identity skip on rejected tokens.

        Returns (loss, grad_sel, grad_rej) where grad_sel is d(L)/d(x_sel) with one
        row per SELECTED token, grad_rej is d(L)/d(x_rej) per REJECTED token.
        """
        x_sel = self.x[:, sel_idx].detach().clone().requires_grad_(need_grad)
        x_rej = self.x[:, rej_idx].detach().clone().requires_grad_(need_grad)

        _, h_out = self.block(x_sel, self.ctx, self.temb)

        y = torch.empty_like(self.x)
        y[:, sel_idx] = h_out
        y[:, rej_idx] = x_rej

        loss = self._loss(y)
        if not need_grad:
            return loss.detach(), None, None

        gs, gr = torch.autograd.grad(loss, [x_sel, x_rej], allow_unused=True)
        return loss.detach(), gs[0], gr[0]


def cos(a, b):
    return F.cosine_similarity(a.unsqueeze(0), b.unsqueeze(0)).item()


def one_trial(task, k, generator, gt_subset=None, probe=None, return_vectors=False):
    """gt_subset: evaluate the exact marginal for this many randomly chosen tokens
    instead of all N. The forward still runs on the full sequence, so attention is
    unchanged; cosine over a random coordinate subset is an unbiased view of the
    full-vector cosine, and it makes N=4096 affordable."""
    N = task.N
    dev = task.device

    logits = torch.randn(N, generator=generator).to(dev)
    order = torch.argsort(logits, descending=True)
    sel_idx, rej_idx = order[:k], order[k:]
    in_set = torch.zeros(N, dtype=torch.bool, device=dev)
    in_set[sel_idx] = True

    lowest_in = order[k - 1].item()  # weakest member, evicted to admit an outsider
    highest_out = order[k].item()  # best outsider, admitted when a member leaves

    # --- base configuration: gives g_hat (Eq. 23) ---
    base_loss, g_sel, g_rej = task.forward_pruned(sel_idx, rej_idx)
    x = task.x[0].float()

    g_hat = torch.zeros(N, device=dev)
    g_hat[sel_idx] = (g_sel.float() * x[sel_idx]).sum(-1)
    g_hat[rej_idx] = -(g_rej.float() * x[rej_idx]).sum(-1)

    # executed-path term per token, reused when assembling Eq. 22
    term_sel = torch.zeros(N, device=dev)
    term_rej = torch.zeros(N, device=dev)
    term_sel[sel_idx] = (g_sel.float() * x[sel_idx]).sum(-1)
    term_rej[rej_idx] = (g_rej.float() * x[rej_idx]).sum(-1)

    # --- flip sweep: exact discrete marginal + the counterfactual path term ---
    g_true = torch.zeros(N, device=dev)
    sel_list = sel_idx.tolist()

    if probe is not None:
        # A fixed probe set lets the caller accumulate estimates across draws on
        # consistent coordinates, which is the quantity that actually drives training.
        probe = probe.to(dev)
    elif gt_subset is not None and gt_subset < N:
        probe = torch.randperm(N, generator=generator)[:gt_subset].to(dev)
    else:
        probe = torch.arange(N, device=dev)

    for i in probe.tolist():
        if in_set[i]:
            # evict i, admit the best outsider -> i lands on the REJECTED path
            new_sel = [t for t in sel_list if t != i] + [highest_out]
        else:
            # admit i by evicting the weakest member -> i lands on the SELECTED path
            new_sel = [t for t in sel_list if t != lowest_in] + [i]

        new_sel_t = torch.tensor(sorted(new_sel), device=dev)
        mask = torch.ones(N, dtype=torch.bool, device=dev)
        mask[new_sel_t] = False
        new_rej_t = torch.arange(N, device=dev)[mask]

        f_loss, f_gs, f_gr = task.forward_pruned(new_sel_t, new_rej_t)

        if in_set[i]:
            g_true[i] = base_loss - f_loss
            pos = (new_rej_t == i).nonzero()[0, 0]
            term_rej[i] = (f_gr[pos].float() * x[i]).sum()
        else:
            g_true[i] = f_loss - base_loss
            pos = (new_sel_t == i).nonzero()[0, 0]
            term_sel[i] = (f_gs[pos].float() * x[i]).sum()

    # Eq. 22 needs both paths per token. Under hard gather-based selection only one is
    # observable, so the missing half is taken from the flipped configuration above.
    g_22 = term_sel - term_rej

    # --- Gumbel-Sigmoid STE: y = x + m*(z - x), z the full (unpruned) block output ---
    xg = task.x.detach().clone().requires_grad_(True)
    _, z = task.block(xg, task.ctx, task.temb)
    m = torch.zeros(1, N, 1, device=dev, dtype=task.dtype)
    m[:, sel_idx] = 1.0
    y = xg + m * (z - xg)
    (grad_y,) = torch.autograd.grad(task._loss(y), y)
    g_gum = (grad_y[0].float() * (z - xg).detach()[0].float()).sum(-1)

    # --- controls, so a small cosine can be told apart from chance ---
    # a random direction has E[cos]=0 with std ~ 1/sqrt(len(probe)); without this row
    # a cosine of ~0.1 is indistinguishable from noise.
    g_rand = torch.randn(N, generator=generator).to(dev)
    # naive saliency heuristic: is the estimator any better than "keep big tokens"?
    g_norm = x.norm(dim=-1)

    # every comparison is restricted to the tokens whose true marginal was measured
    def c(v):
        return cos(v[probe], g_true[probe])

    out = {
        "Residual STE (Eq. 23, ours) vs true": c(g_hat),
        "Two-term (Eq. 22)          vs true": c(g_22),
        "Gumbel-Sigmoid STE         vs true": c(g_gum),
        "[control] random direction vs true": c(g_rand),
        "[control] feature norm     vs true": c(g_norm),
        "Eq. 23 vs Eq. 22  (Fig. 7b quantity)": cos(g_hat[probe], g_22[probe]),
    }
    if return_vectors:
        # raw per-token values on the probe set, for the accumulation analysis
        return out, {
            "g_hat": g_hat[probe].clone(),
            "g_true": g_true[probe].clone(),
            "g_gum": g_gum[probe].clone(),
            "g_rand": g_rand[probe].clone(),
        }
    return out


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--trials", type=int, default=30)
    p.add_argument("--tokens", type=int, default=256)
    p.add_argument("--ratio", type=float, default=0.6)
    p.add_argument(
        "--device",
        default="cuda"
        if torch.cuda.is_available()
        else ("mps" if torch.backends.mps.is_available() else "cpu"),
    )
    p.add_argument("--seed", type=int, default=0)
    p.add_argument(
        "--capture",
        default=None,
        help=".pt from test.toy.capture_block_inputs -- trained weights "
        "and real latents instead of random init",
    )
    p.add_argument(
        "--ratio-jitter",
        type=float,
        default=0.0,
        help="vary the budget per draw by +-this much. Sec. 3.5 accumulates "
        "over 'randomised budgets and perturbed scores', so matching "
        "the protocol means jittering k, not only the scores",
    )
    p.add_argument(
        "--accumulate",
        action="store_true",
        help="also report how alignment grows when estimates are averaged "
        "across draws -- the quantity that actually drives training, "
        "and the real-block analogue of the toy accumulation table",
    )
    p.add_argument(
        "--gt-subset",
        type=int,
        default=None,
        help="measure the exact marginal for this many random tokens "
        "instead of all N (required to afford N=4096)",
    )
    args = p.parse_args()

    dtype = torch.float32
    acc = {}
    t0 = time.perf_counter()

    fixed_probe, accum = None, {}
    for t in range(args.trials):
        gen = torch.Generator().manual_seed(args.seed + t)
        if args.capture:
            task = RealBlockTask.from_capture(
                args.capture, args.device, dtype, gen, max_tokens=args.tokens
            )
        else:
            task = RealBlockTask(args.tokens, args.device, dtype, gen)
        r = args.ratio
        if args.ratio_jitter:
            r += (torch.rand(1, generator=gen).item() * 2 - 1) * args.ratio_jitter
            r = min(max(r, 0.05), 0.95)
        k = max(1, int(task.N * r))
        if args.accumulate:
            if fixed_probe is None:
                # one probe set for every draw, so the running sums stay aligned
                g0 = torch.Generator().manual_seed(args.seed)
                fixed_probe = (
                    torch.randperm(task.N, generator=g0)[: args.gt_subset]
                    if args.gt_subset
                    else torch.arange(task.N)
                )
            res, vecs = one_trial(
                task,
                k,
                gen,
                gt_subset=args.gt_subset,
                probe=fixed_probe,
                return_vectors=True,
            )
            for key, v in vecs.items():
                accum[key] = v if key not in accum else accum[key] + v
            accum.setdefault("_n", 0)
            accum["_n"] += 1
            if accum["_n"] in (1, 2, 5, 10, 20, 30, 50):
                a = cos(accum["g_hat"], accum["g_true"])
                r = cos(accum["g_rand"], accum["g_true"])
                print(
                    f"    [accum {accum['_n']:>3} draws]  ours {a:+.4f}   "
                    f"random-control {r:+.4f}",
                    flush=True,
                )
        else:
            res = one_trial(task, k, gen, gt_subset=args.gt_subset)
        for key, val in res.items():
            acc.setdefault(key, []).append(val)
        print(
            f"  trial {t + 1}/{args.trials}  "
            + "  ".join(f"{v:+.3f}" for v in res.values()),
            flush=True,
        )

    n_probe = min(args.gt_subset or task.N, task.N)

    print(f"\n{'=' * 84}")
    if args.capture:
        meta = getattr(task, "meta", {})
        print(
            f"Real SD3 block, TRAINED weights + REAL latents "
            f"(layer {meta.get('layer', '?')}, denoise step {meta.get('step', '?')})"
        )
        print(f"capture: {args.capture}")
    else:
        print(
            f"Real SD3 JointTransformerBlock (dim={DIM}, {HEADS} heads, "
            f"RANDOM init, synthetic latents)"
        )
    print(
        f"N={task.N} image tokens, ctx={task.ctx.shape[1]}, k={k} (r={args.ratio}), "
        f"trials={args.trials}, {time.perf_counter() - t0:.0f}s"
    )
    print("objective: Stage-1 distillation  L(S) = ||y(S) - y_full||^2")
    # chance is set by how many coordinates are actually compared, not by N
    chance = 1.0 / n_probe**0.5
    print(
        f"exact marginal measured on {n_probe}/{task.N} tokens; "
        f"chance cosine: mean 0, std ~{chance:.3f}"
    )
    print(f"{'=' * 84}")
    print(f"{'':<40} {'mean':>8} {'std':>8} {'>0 rate':>9} {'mean/chance':>12}")
    for key, vals in acc.items():
        v = torch.tensor(vals)
        print(
            f"  {key:<38} {v.mean():>8.3f} {v.std():>8.3f} "
            f"{(v > 0).float().mean() * 100:>8.1f}% {v.mean() / chance:>11.1f}x"
        )
    print(f"{'=' * 84}\n")


if __name__ == "__main__":
    main()
