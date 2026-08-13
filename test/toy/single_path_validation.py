"""Does dropping the second term of the two-path gradient actually hurt?

The synthetic study in App. A.3 uses the objective L = sum(x_sel) - sum(x_rej),
for which dL/dx_sel = +1 and dL/dx_rej = -1 are *constants*. Substituting these
into Eq. 23 gives g_hat_i = <1, x_i> for selected AND rejected tokens alike, which
coincides exactly with the two-path expression of Eq. 22. The single-path drop is
therefore identity on that task -- it cannot be probed there, whatever the measured
cosine turns out to be.

This task removes that degeneracy. Rejected tokens keep influencing the loss through
the identity skip, and every token carries a distinct gradient:

    y_i(S) = x_i + Block(x_S)_i   if i in S        (selected: block + skip)
    y_i(S) = x_i                  otherwise        (rejected: skip only)
    L(S)   = || y(S) - y_full ||^2                 (the Stage-1 distillation loss)

Ground truth is the exact discrete marginal at fixed budget k, which references
neither Eq. 21 nor Eq. 22:

    Delta_i = L(S_i^+) - L(S_i^-)

Three questions, three experiments:

  E1  single draw       -- how far is g_hat (Eq. 23) from Delta, versus g_star (Eq. 22)?
  E2  accumulation      -- App. A.2 argues stochastic k and score noise let g_hat
                           approach the expected direction over training. Does the
                           running mean of g_hat actually converge towards Delta?
  E3  optimisation      -- train the router with g_hat / g_star / Delta and compare.
                           If the three converge alike, the drop is benign in practice.

Run:  python -m test.toy.single_path_validation
"""

import argparse

import torch
import torch.nn.functional as F

N, D = 100, 16
N_SIGNAL = 20
SIGNAL_GAIN = 3.0


class MaskedBlock(torch.nn.Module):
    """Single-head attention + FFN with residual, evaluated on the selected subset.

    Masking is applied to the attention logits so rejected tokens neither attend
    nor are attended to, and their block output is zeroed -- functionally the same
    as gather -> block -> scatter, but written with full-length tensors so that
    grad_sel and grad_rej are both defined for every token at one operating point,
    which is what reading off Eq. 22 requires.
    """

    def __init__(self, generator):
        super().__init__()
        g = generator
        for name in ("wq", "wk", "wv", "w1", "w2"):
            w = torch.randn(D, D, generator=g) / D**0.5
            self.register_buffer(name, w)

    def forward(self, x, mask):
        # x: (N, D), mask: (N, 1) in {0, 1}
        q, k, v = x @ self.wq, x @ self.wk, x @ self.wv
        att = (q @ k.T) / D**0.5
        keep = mask.squeeze(-1).bool()
        att = att.masked_fill(~keep.unsqueeze(0), float("-inf"))
        att = att.masked_fill(~keep.unsqueeze(1), float("-inf"))
        att = torch.softmax(att, dim=-1)
        att = torch.nan_to_num(att, 0.0)  # rows of rejected tokens are all -inf
        h = att @ v
        h = torch.tanh(h @ self.w1) @ self.w2
        return h * mask


class Task:
    def __init__(self, generator, device="cpu"):
        self.x = torch.randn(N, D, generator=generator)
        self.x[:N_SIGNAL] *= SIGNAL_GAIN  # signal / noise split as in App. A.3
        self.x = self.x.to(device)
        self.block = MaskedBlock(generator).to(device)
        self.device = device
        with torch.no_grad():
            full = torch.ones(N, 1, device=device)
            self.y_full = self.x + self.block(self.x, full)

    def _loss_from_parts(self, x_sel, x_rej, mask):
        y = x_sel + x_rej + self.block(x_sel, mask)
        return ((y - self.y_full) ** 2).mean()

    def loss(self, mask):
        with torch.no_grad():
            return self._loss_from_parts(mask * self.x, (1 - mask) * self.x, mask)

    def gradients(self, mask):
        """Both paths at one operating point -> g_hat (Eq. 23) and g_star (Eq. 22)."""
        x_sel = (mask * self.x).detach().requires_grad_(True)
        x_rej = ((1 - mask) * self.x).detach().requires_grad_(True)
        loss = self._loss_from_parts(x_sel, x_rej, mask)
        g_sel, g_rej = torch.autograd.grad(loss, [x_sel, x_rej])

        sel = mask.squeeze(-1).bool()
        term_sel = (g_sel * self.x).sum(-1)
        term_rej = (g_rej * self.x).sum(-1)

        g_hat = torch.where(sel, term_sel, -term_rej)  # Eq. 23, executed path only
        return g_hat, term_sel, term_rej, loss.detach()

    def g_star_counterfactual(self, mask):
        """Eq. 22 needs BOTH paths per token, but under hard selection only one is
        executed. Reading the unexecuted term off the same forward pass silently
        yields zero -- the block masks that token out, so its Jacobian vanishes.
        The missing half must come from a forward pass with the token flipped."""
        sel = mask.squeeze(-1).bool()
        _, term_sel, term_rej, _ = self.gradients(mask)

        for i in range(N):
            m = mask.clone()
            m[i] = 1.0 - m[i]
            _, t_sel_f, t_rej_f, _ = self.gradients(m)
            if sel[i]:
                term_rej[i] = t_rej_f[i]  # i is rejected in the flipped config
            else:
                term_sel[i] = t_sel_f[i]  # i is selected in the flipped config
        return term_sel - term_rej

    def true_marginal(self, order, k):
        """Exact discrete marginal with the budget pinned at k."""
        sel = order[:k]
        mask = torch.zeros(N, 1, device=self.device)
        mask[sel] = 1.0
        base = self.loss(mask)

        in_set = mask.squeeze(-1).bool()
        weakest_in = order[k - 1].item()
        best_out = order[k].item()

        delta = torch.zeros(N, device=self.device)
        for i in range(N):
            m = mask.clone()
            if in_set[i]:
                m[i], m[best_out] = 0.0, 1.0  # evict i, admit the best outsider
                delta[i] = base - self.loss(m)
            else:
                m[i], m[weakest_in] = 1.0, 0.0  # admit i, evict the weakest member
                delta[i] = self.loss(m) - base
        return delta


def cos(a, b):
    return F.cosine_similarity(a.unsqueeze(0), b.unsqueeze(0)).item()


def mask_from(order, k, device):
    m = torch.zeros(N, 1, device=device)
    m[order[:k]] = 1.0
    return m


# --------------------------------------------------------------------------- #
def e1_single_draw(trials, ratio, seed):
    print("\n[E1] single draw: how far is each estimator from the true marginal?")
    k = int(N * ratio)
    rows = {
        "g_hat (Eq.23) vs g_star (Eq.22)": [],
        "g_hat (Eq.23) vs true marginal": [],
        "g_star (Eq.22) vs true marginal": [],
        "[control] random vs true marginal": [],
    }

    for t in range(trials):
        g = torch.Generator().manual_seed(seed + t)
        task = Task(g)
        order = torch.argsort(torch.randn(N, generator=g), descending=True)
        mask = mask_from(order, k, task.device)

        g_hat, _, _, _ = task.gradients(mask)
        g_star = task.g_star_counterfactual(mask)
        delta = task.true_marginal(order, k)
        rand = torch.randn(N, generator=g)

        rows["g_hat (Eq.23) vs g_star (Eq.22)"].append(cos(g_hat, g_star))
        rows["g_hat (Eq.23) vs true marginal"].append(cos(g_hat, delta))
        rows["g_star (Eq.22) vs true marginal"].append(cos(g_star, delta))
        rows["[control] random vs true marginal"].append(cos(rand, delta))

    chance = 1.0 / N**0.5
    print(f"     N={N} D={D} k={k} trials={trials}   chance |cos| ~ {chance:.3f}")
    for name, vals in rows.items():
        v = torch.tensor(vals)
        print(
            f"     {name:<28} {v.mean():+.3f} +/- {v.std():.3f}   "
            f">0 in {(v > 0).float().mean() * 100:.0f}% of trials"
        )
    return rows


def e2_accumulation(draws, ratio, seed, trials=20):
    print("\n[E2] accumulation: does averaging g_hat over stochastic draws")
    print("     converge towards the true marginal?  (App. A.2's stated mechanism)")
    k_ref = int(N * ratio)
    checkpoints = [1, 2, 5, 10, 25, 50, 100, 200]
    checkpoints = [c for c in checkpoints if c <= draws]
    acc = {c: [] for c in checkpoints}

    for t in range(trials):
        g = torch.Generator().manual_seed(seed + 1000 + t)
        task = Task(g)
        logits = torch.randn(N, generator=g)
        order_ref = torch.argsort(logits, descending=True)
        delta = task.true_marginal(order_ref, k_ref)

        running = torch.zeros(N)
        for d in range(1, draws + 1):
            # stratified ratio sampling + score perturbation, as in Sec. 3.5
            r = 0.2 + 0.7 * torch.rand(1, generator=g).item()
            k_d = max(1, min(int(N * r), N - 1))
            noisy = logits + 0.5 * torch.randn(N, generator=g)
            order_d = torch.argsort(noisy, descending=True)
            mask = mask_from(order_d, k_d, task.device)

            g_hat, _, _, _ = task.gradients(mask)
            running += g_hat / g_hat.norm().clamp(min=1e-12)

            if d in acc:
                acc[d].append(cos(running, delta))

    print(f"     {'draws':>7}  {'cos(mean g_hat, true)':>24}")
    for c in checkpoints:
        v = torch.tensor(acc[c])
        print(f"     {c:>7}  {v.mean():>+14.3f} +/- {v.std():.3f}")
    return acc


def e3_optimisation(steps, ratio, seed, runs=10):
    print("\n[E3] optimisation: train the router with each estimator and compare.")
    print(f"     metric = fraction of the {N_SIGNAL} signal tokens inside the top-k")
    k = int(N * ratio)
    variants = ["g_hat (Eq.23)", "g_star (Eq.22)", "true marginal"]
    final = {v: [] for v in variants}

    for r in range(runs):
        for variant in variants:
            g = torch.Generator().manual_seed(seed + 2000 + r)
            task = Task(g)
            logits = (torch.randn(N, generator=g) * 0.01).requires_grad_(True)
            opt = torch.optim.Adam([logits], lr=0.1)

            for _ in range(steps):
                order = torch.argsort(logits.detach(), descending=True)
                mask = mask_from(order, k, task.device)

                if variant == "true marginal":
                    signal = task.true_marginal(order, k)
                else:
                    g_hat, _, _, _ = task.gradients(mask)
                    signal = (
                        g_hat
                        if variant.startswith("g_hat")
                        else task.g_star_counterfactual(mask)
                    )

                # descend the loss: a positive marginal means including the token
                # raises L, so its score should go down
                opt.zero_grad()
                logits.grad = signal.detach()
                opt.step()

            order = torch.argsort(logits.detach(), descending=True)
            hits = (order[:k] < N_SIGNAL).sum().item()
            final[variant].append(hits / N_SIGNAL)

    for variant in variants:
        v = torch.tensor(final[variant])
        print(f"     {variant:<18} signal recall {v.mean():.3f} +/- {v.std():.3f}")
    return final


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--trials", type=int, default=100)
    p.add_argument("--draws", type=int, default=200)
    p.add_argument("--steps", type=int, default=60)
    p.add_argument("--ratio", type=float, default=0.6)
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args()

    print("=" * 72)
    print("Single-path approximation: does dropping Eq. 22's second term hurt?")
    print("residual task, rejected tokens keep gradient through the identity skip")
    print("=" * 72)

    e1_single_draw(args.trials, args.ratio, args.seed)
    e2_accumulation(args.draws, args.ratio, args.seed)
    e3_optimisation(args.steps, args.ratio, args.seed)
    print()
