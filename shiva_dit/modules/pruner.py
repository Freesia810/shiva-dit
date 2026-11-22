import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.utils.parametrizations as parametrizations
import numpy as np

from shiva_dit.utils.ste import ShivaTopkFunc


def hard_topk_select(x, logits, k, need_rejected=True):
    """Select tokens without invoking the custom backward function."""
    B, N, D = x.shape

    if isinstance(k, int):
        k_int = k
    else:
        k_int = int(torch.ceil(k).item())
    k_int = max(1, min(k_int, N))

    # Match the training path's order and choices at tied scores.
    idx_sorted = torch.argsort(logits, dim=-1, descending=True)
    topk_idx = idx_sorted[:, :k_int]
    x_selected = torch.gather(x, 1, topk_idx.unsqueeze(-1).expand(B, k_int, D))
    if not need_rejected:
        return x_selected, None, topk_idx, None

    rest_idx = idx_sorted[:, k_int:]
    x_rejected = torch.gather(x, 1, rest_idx.unsqueeze(-1).expand(B, N - k_int, D))
    return x_selected, x_rejected, topk_idx, rest_idx


class ShivaPrunerExpert(torch.nn.Module):
    def __init__(self, init_Ts, init_Tk):
        super().__init__()
        self.register_buffer("Ts", torch.tensor(float(init_Ts)), persistent=False)
        self.register_buffer("Tk", torch.tensor(float(init_Tk)), persistent=False)

    def forward(self, x, logits, k, need_rejected=True):
        if not torch.is_grad_enabled():
            return hard_topk_select(x, logits, k, need_rejected)
        return ShivaTopkFunc.apply(x, logits, k, self.Ts, self.Tk)

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
