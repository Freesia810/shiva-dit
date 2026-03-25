import inspect
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Literal

from shiva_dit.hook import BaseDiTBlockHook, BaseDelegator, HookManager
from shiva_dit.utils import (
    ceil_ste,
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
    P_hat = torch.softmax(P_max / tau, dim=-1)                 # (B, N, N)

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
        """
        Differentiable top-k via NeuralSort (Identity-Preserving version).
        
        Fix: 
            Instead of reordering tokens via bmm(P, x), we calculate the 
            probability of each token being in the top-k and use STE to 
            perform physical truncation while maintaining original order.
        """
        B, N, D = x.shape
        original_dtype = x.dtype

        # 1. --- Normalize scores to prevent O(N)-scale logits ---
        s = scores.unsqueeze(-1)                           # (B, N, 1), fp32
        s_mean = s.mean(dim=1, keepdim=True)
        s_std = s.std(dim=1, keepdim=True).clamp(min=1e-6)
        s_norm = (s - s_mean) / s_std                      # zero-mean, unit-var
        s_norm = s_norm.clamp(-5.0, 5.0)

        # 2. --- Generate Relaxed Permutation Matrix P_hat ---
        P_hat = deterministic_neuralsort(s_norm, self.tau * N)  # (B, N, N), fp32

        # 3. --- Calculate Soft Gate (Probability of being in Top-K) ---
        soft_gate = P_hat[:, :k, :].sum(dim=1)             # (B, N)

        # 4. --- Hard Selection (Identity-Preserving) ---
        _, topk_idx = torch.topk(scores, k, dim=1, sorted=False)
        sorted_topk_idx, _ = torch.sort(topk_idx, dim=1)   # (B, k)

        # 5. --- Straight-Through Estimator (STE) ---
        mask_hard = torch.zeros_like(soft_gate)
        mask_hard.scatter_(1, sorted_topk_idx, 1.0)
        gate = soft_gate + (mask_hard - soft_gate).detach()

        # 6. --- Physical Truncation ---
        x_f32 = x.float()
        x_sel_raw = torch.gather(x_f32, 1, sorted_topk_idx.unsqueeze(-1).expand(-1, -1, D))
        gate_sel = torch.gather(gate, 1, sorted_topk_idx).unsqueeze(-1)
        x_sel = (x_sel_raw * gate_sel).to(original_dtype)

        # 7. --- Handle Remaining Tokens (x_rej) ---
        if k < N:
            all_idx = torch.arange(N, device=x.device).unsqueeze(0).expand(B, N)
            mask_rej = torch.ones(B, N, dtype=torch.bool, device=x.device)
            mask_rej.scatter_(1, sorted_topk_idx, False)
            rej_idx = all_idx[mask_rej].view(B, N - k)
            
            x_rej_raw = torch.gather(x_f32, 1, rej_idx.unsqueeze(-1).expand(-1, -1, D))
            gate_rej = torch.gather(gate, 1, rej_idx).unsqueeze(-1)
            x_rej = (x_rej_raw * gate_rej).to(original_dtype)
            
            self._cached_rej_idx = rej_idx
        else:
            x_rej = x.new_empty(B, 0, D)
            self._cached_rej_idx = None

        # 8. --- Cache for delegated_call and scatter_back ---
        self._cached_P_hat = P_hat
        self._cached_sel_idx = sorted_topk_idx
        self._cached_k = k

        return x_sel, x_rej

    def _hard_topk(self, x, scores, k):
        """Standard hard top-k (inference)."""
        B, N, D = x.shape

        _, topk_idx = torch.topk(scores, k, dim=1, sorted=False)
        topk_idx, _ = torch.sort(topk_idx, dim=1)

        x_sel = torch.gather(x, 1, topk_idx.unsqueeze(-1).expand(B, k, D))

        all_idx = torch.arange(N, device=x.device).unsqueeze(0).expand(B, N)
        mask = torch.ones(B, N, dtype=torch.bool, device=x.device)
        mask.scatter_(1, topk_idx, False)
        rej_idx = all_idx[mask].view(B, N - k)
        x_rej = torch.gather(x, 1, rej_idx.unsqueeze(-1).expand(B, N - k, D))

        return x_sel, x_rej, topk_idx

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
            x_sel, x_rej = self._soft_topk(hidden_states, token_scores, k)
            sel_idx = self._cached_sel_idx
        else:
            x_sel, x_rej, sel_idx = self._hard_topk(hidden_states, token_scores, k)

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
        if self.training:
            B, k_, D = hid_out.shape
            N = hidden_states.shape[1]
            full_hidden = torch.zeros(B, N, D, device=hid_out.device, dtype=hid_out.dtype)
            sel_idx_train = self._cached_sel_idx
            full_hidden.scatter_(1, sel_idx_train.unsqueeze(-1).expand(-1, -1, D), hid_out)
            if x_rej.shape[1] > 0:
                rej_idx_train = self._cached_rej_idx
                full_hidden.scatter_(1, rej_idx_train.unsqueeze(-1).expand(-1, -1, D), x_rej.to(hid_out.dtype))

            self._cached_P_hat = None
            self._cached_sel_idx = None
            self._cached_rej_idx = None
            self._cached_k = None
        else:
            # Hard mode: standard scatter
            full_hidden = torch.zeros(B, N, D, device=hid_out.device, dtype=hid_out.dtype)
            full_hidden.scatter_(1, sel_idx.unsqueeze(-1).expand(-1, -1, D), hid_out)

            if x_rej.shape[1] > 0:
                all_idx = torch.arange(N, device=hid_out.device).unsqueeze(0).expand(B, N)
                mask = torch.ones(B, N, dtype=torch.bool, device=hid_out.device)
                mask.scatter_(1, sel_idx, False)
                rej_positions = all_idx[mask].view(B, N - k)
                full_hidden.scatter_(1, rej_positions.unsqueeze(-1).expand(-1, -1, D), x_rej)

        if ret_tuple:
            return enc_out, full_hidden
        else:
            return full_hidden


if __name__ == "__main__":
    model_type = "sd3"
    pipe, model_attrs = get_pipeline(model_type)

    neuralsort_config = {
        "target_ratio": 1.0,
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