import inspect
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Literal

from shiva_dit.hook import BaseDiTBlockHook, BaseDelegator, HookManager
from shiva_dit.utils import (
    global_context,
    get_pipeline,
    process_quick_start,
    parse_image_rotary_emb,
    join_image_rotary_emb,
)


def deterministic_neuralsort(s, tau):
    """
    s: (B, N, 1) scores
    tau: temperature scalar
    returns: (B, N, N) relaxed permutation matrix
    """
    n = s.size(1)
    one = torch.ones(n, 1, dtype=s.dtype, device=s.device)

    A_s = torch.abs(s - s.permute(0, 2, 1))                  # (B, N, N)
    B_mat = torch.matmul(A_s, torch.matmul(one, one.t()))     # (B, N, N)

    scaling = (n + 1 - 2 * (torch.arange(n, dtype=s.dtype, device=s.device) + 1))  # (N,)
    C = torch.matmul(s, scaling.unsqueeze(0))                  # (B, N, N)

    P_max = (C - B_mat).permute(0, 2, 1)                      # (B, N, N)
    P_max_std = P_max.std(dim=-1, keepdim=True).clamp(min=1e-6)
    P_max_norm = P_max / P_max_std
    
    P_hat = torch.softmax(P_max_norm / tau, dim=-1)            # (B, N, N)

    return P_hat

class NeuralSortDitBlockHook(BaseDelegator, BaseDiTBlockHook, nn.Module):
    """
    NeuralSort: Stochastic Optimization of Sorting Networks via Continuous Relaxations
    Reference:
        - Paper: https://arxiv.org/abs/1903.08850

    """

    def __init__(self, target_ratio=0.6, tau=8.0, **kwargs):
        super().__init__(**kwargs)
        self.tau = tau
        self.target_ratio = target_ratio

    def on_register_hook(self, block: nn.Module):
        hidden_dim = global_context.hidden_dim
        if hidden_dim is None:
            raise ValueError("hidden_dim must be set in global_context")

        self.hidden_router = nn.Linear(hidden_dim, 1, bias=True)
        nn.init.xavier_uniform_(self.hidden_router.weight)
        nn.init.constant_(self.hidden_router.bias, 0.0)

        self.dtype = torch.float32
        self.to(device=next(block.parameters()).device, dtype=self.dtype)

    def _soft_topk(self, x, scores, k):
        B, N, D = x.shape
        original_dtype = x.dtype

        s = scores.unsqueeze(-1)                           
        s_norm = (s - s.mean(dim=1, keepdim=True)) / s.std(dim=1, keepdim=True).clamp(min=1e-6)
        s_norm = s_norm.clamp(-5.0, 5.0)

        P_hat = deterministic_neuralsort(s_norm, self.tau)  
        soft_gate = P_hat[:, :k, :].sum(dim=1)             

        _, topk_idx = torch.topk(scores, k, dim=1, sorted=False)
        sorted_topk_idx, _ = torch.sort(topk_idx, dim=1)   

        mask_hard = torch.zeros_like(soft_gate)
        mask_hard.scatter_(1, sorted_topk_idx, 1.0)
        gate = soft_gate + (mask_hard - soft_gate).detach()

        gather_index = sorted_topk_idx.unsqueeze(-1).expand(-1, -1, D)
        x_sel_raw = torch.gather(x.float(), 1, gather_index)
        gate_sel = torch.gather(gate, 1, sorted_topk_idx).unsqueeze(-1)
        x_sel = (x_sel_raw * gate_sel).to(original_dtype)

        remaining_x = x.clone()
        remaining_x.scatter_(1, gather_index, 0.0)

        return x_sel, remaining_x, sorted_topk_idx

    def _hard_topk(self, x, scores, k):
        B, N, D = x.shape
        _, topk_idx = torch.topk(scores, k, dim=1, sorted=False)
        sorted_topk_idx, _ = torch.sort(topk_idx, dim=1)

        gather_index = sorted_topk_idx.unsqueeze(-1).expand(-1, -1, D)
        x_sel = torch.gather(x, 1, gather_index)

        remaining_x = x.clone()
        remaining_x.scatter_(1, gather_index, 0.0)

        return x_sel, remaining_x, sorted_topk_idx

    def delegated_call(self, func, module_self, *args, **kwargs):
        sig = inspect.signature(func)
        bound = sig.bind_partial(module_self, *args, **kwargs)
        bound.apply_defaults()

        hidden_states = bound.arguments.get("hidden_states", None)
        encoder_hidden_states = bound.arguments.get("encoder_hidden_states", None)
        image_rotary_emb = bound.arguments.get("image_rotary_emb", None)

        B, N, D = hidden_states.shape
        k = max(1, min(int(N * self.target_ratio), N))

        # --- Score (clamped to prevent gradient explosion) ---
        token_scores = self.hidden_router(hidden_states.to(self.dtype)).squeeze(-1)  # (B, N)
        token_scores = token_scores.clamp(-10.0, 10.0)

        if self.training:
            x_sel, remaining_x, sel_idx = self._soft_topk(hidden_states, token_scores, k)
        else:
            x_sel, remaining_x, sel_idx = self._hard_topk(hidden_states, token_scores, k)

        if image_rotary_emb is not None:
            assert hidden_states.shape[0] == 1, "image_rotary_emb only supports batch size 1"
            hidden_emb, encoder_emb, _ = parse_image_rotary_emb(
                self.model_type,
                image_rotary_emb,
                global_context.hidden_seq_len,
                global_context.encoder_hidden_seq_len_gen(),
                split_extra=False,
            )
            hidden_emb = tuple(
                torch.gather(t, dim=0, index=sel_idx[0].unsqueeze(-1).expand(-1, t.shape[-1]))
                for t in hidden_emb
            )
            image_rotary_emb = join_image_rotary_emb(self.model_type, hidden_emb, encoder_emb)
            bound.arguments["image_rotary_emb"] = image_rotary_emb

        # --- Forward on selected tokens ---
        bound.arguments["hidden_states"] = x_sel
        if encoder_hidden_states is not None:
            bound.arguments["encoder_hidden_states"] = encoder_hidden_states

        result = func(*bound.args, **bound.kwargs)

        if isinstance(result, tuple) and len(result) == 2:
            enc_out, hid_out = result
            ret_tuple = True
        else:
            hid_out = result
            enc_out = None
            ret_tuple = False

        # --- Scatter back ---
        full_hidden = remaining_x.scatter_add(
            dim=1,
            index=sel_idx.unsqueeze(-1).expand(-1, -1, hid_out.shape[-1]),
            src=hid_out,
        )

        if ret_tuple:
            return enc_out, full_hidden
        else:
            return full_hidden


if __name__ == "__main__":
    model_type = "sd3"
    pipe, model_attrs = get_pipeline(model_type)

    neuralsort_config = {
        "target_ratio": 0.999,
        "tau": 8.0,
    }

    HookManager.configure_hooks(
        model_type=model_type,
        default=(NeuralSortDitBlockHook, neuralsort_config),
    )

    with HookManager.enabled():
        HookManager.inject_hooks(
            model_type=model_type, pipe=pipe, model_attrs=model_attrs
        )
        res = process_quick_start(pipe)
        res.save(f"{model_type}_neuralsort.png")

    print(global_context)