import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.utils.parametrizations as parametrizations
import numpy as np

from shiva_dit.utils.ste import ShivaTopkFunc


def hard_topk_select(x, logits, k, need_rejected=True, scores_shared=False):
    """Forward-only hard top-k: rank the scores, gather the selected tokens.

    `k` may be a plain int -- which is the point. When the ratio comes from the
    deployed LUT the budget is known without touching the device, so no .item()
    sync is needed to obtain a sliceable integer.

    `need_rejected` is False whenever the discarded tokens are never read, which at
    inference is always: they pass through unchanged and the caller can restore them
    from its own input. That removes their gather, and with it the need for a full
    ordering, so the selection becomes a top-k rather than a sort of all N scores.

    `scores_shared` says every row of `logits` is a copy of the same vector, which is
    what classifier-free guidance produces once the conditional and unconditional
    scores have been reconciled. Ranking it once and reusing the indices halves the
    selection at no cost to the result.
    """
    B, N, D = x.shape

    if isinstance(k, int):
        k_int = k
    else:
        k_int = int(torch.ceil(k).item())
    k_int = max(1, min(k_int, N))

    if not need_rejected:
        src = logits[:1] if scores_shared else logits
        topk_idx = torch.topk(src, k_int, dim=-1, sorted=False).indices
        if scores_shared and B > 1:
            topk_idx = topk_idx.expand(B, k_int)
        x_selected = torch.gather(x, 1, topk_idx.unsqueeze(-1).expand(B, k_int, D))
        return x_selected, None, topk_idx, None

    idx_sorted = torch.argsort(logits, dim=-1, descending=True)

    topk_idx = idx_sorted[:, :k_int]
    rest_idx = idx_sorted[:, k_int:]
    x_selected = torch.gather(x, 1, topk_idx.unsqueeze(-1).expand(B, k_int, D))
    x_rejected = torch.gather(x, 1, rest_idx.unsqueeze(-1).expand(B, N - k_int, D))
    return x_selected, x_rejected, topk_idx, rest_idx


class ShivaPrunerExpert(torch.nn.Module):
    def __init__(self, init_Ts, init_Tk, pi_mode: str = "residual", gumbel_tau: float = 1.0):
        super().__init__()
        self.register_buffer("Ts", torch.tensor(float(init_Ts)), persistent=False)
        self.register_buffer("Tk", torch.tensor(float(init_Tk)), persistent=False)
        # "residual" (ours) or "gumbel" (independent Gumbel-Sigmoid, DyDiT-style).
        # The forward pass -- hard top-k under the same budget -- is identical either way;
        # only the backward surrogate for pi_i changes.
        self.pi_mode = pi_mode
        self.gumbel_tau = gumbel_tau

    def forward(self, x, logits, k, need_rejected=True, scores_shared=False):
        # Inference needs neither the custom backward nor a differentiable k, and
        # going through autograd.Function costs real time: the node machinery plus
        # the .item() that turns k into a sliceable int forces a device sync on
        # every call (24 layers x 50 steps). Under no_grad we do the hard select
        # directly; the forward result is bit-identical.
        if not torch.is_grad_enabled():
            return hard_topk_select(x, logits, k, need_rejected, scores_shared)
        return ShivaTopkFunc.apply(
            x, logits, k, self.Ts, self.Tk, self.pi_mode, self.gumbel_tau
        )

class ShivaContextPruner(nn.Module):
    def __init__(self, img_dim, txt_dim, proj_dim, init_Ts, init_Tk):
        super().__init__()
        self.q_proj = nn.Linear(img_dim, proj_dim, bias=False)
        self.k_proj = nn.Linear(txt_dim, proj_dim, bias=False)

        parametrizations.orthogonal(self.q_proj, "weight")
        parametrizations.orthogonal(self.k_proj, "weight")

        self.logit_scale = nn.Parameter(torch.tensor(np.log(10.0)))

        self.register_buffer("Ts", torch.tensor(float(init_Ts)), persistent=False)
        self.register_buffer("Tk", torch.tensor(float(init_Tk)), persistent=False)

    def forward(self, hidden_states, encoder_hidden_states, k):
        original_dtype = hidden_states.dtype

        h_in = hidden_states.to(torch.float32)
        e_in = encoder_hidden_states.to(torch.float32)

        img_query = h_in.mean(dim=1) 
        q = self.q_proj(img_query).unsqueeze(1) # (B, 1, R)
        k_proj = self.k_proj(e_in)              # (B, L, R)

        q = F.normalize(q, p=2, dim=-1)
        k_proj = F.normalize(k_proj, p=2, dim=-1)

        logit_scale = self.logit_scale.exp()
        scores = (q * k_proj).sum(dim=-1) * logit_scale

        self.last_scores = scores.detach()

        if not isinstance(k, torch.Tensor):
            k = torch.tensor(k, device=hidden_states.device, dtype=torch.float32)
        else:
            k = k.to(dtype=torch.float32)

        x_sel, x_rej, sel_idx, rej_idx = ShivaTopkFunc.apply(e_in, scores, k, self.Ts, self.Tk)

        return x_sel.to(original_dtype), x_rej.to(original_dtype), sel_idx, rej_idx

    def get_aux_loss(self):
        if not hasattr(self, "last_scores"):
            return torch.tensor(0.0)

        return -torch.var(self.last_scores)
