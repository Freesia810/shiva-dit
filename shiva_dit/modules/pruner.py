import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.utils.parametrizations as parametrizations
import numpy as np

from shiva_dit.utils.ste import ShivaTopkFunc

class ShivaPrunerExpert(torch.nn.Module):
    def __init__(self, init_Ts, init_Tk):
        super().__init__()
        self.register_buffer("Ts", torch.tensor(float(init_Ts)), persistent=False)
        self.register_buffer("Tk", torch.tensor(float(init_Tk)), persistent=False)

    def forward(self, x, logits, k):
        return ShivaTopkFunc.apply(x, logits, k, self.Ts, self.Tk)


class ShivaContextPruner(nn.Module):
    def __init__(
        self,
        init_Ts,
        init_Tk,
        mode="learned",  # "learned" or "free"
        pool_size=None,  # e.g., 2 (means 2x2=4 regions) or None (global mean)
        img_dim=None,
        txt_dim=None,
        proj_dim=None,
    ):
        super().__init__()
        self.mode = mode
        self.pool_size = pool_size

        self.free_ops = None

        # ================== Learned Mode Parameters ==================
        if self.mode == "learned":
            self.q_proj = nn.Linear(img_dim, proj_dim, bias=False)
            self.k_proj = nn.Linear(txt_dim, proj_dim, bias=False)
            self.logit_scale = nn.Parameter(torch.tensor(np.log(10.0)))
        else:
            self.q_proj = None
            self.k_proj = None
            self.logit_scale = None

        # ================== Thresholds (Buffer) ==================
        self.register_buffer("Ts", torch.tensor(float(init_Ts)), persistent=False)
        self.register_buffer("Tk", torch.tensor(float(init_Tk)), persistent=False)

    def on_register(self, block: nn.Module):
        if self.mode != "free":
            return

        attn = block.attn if hasattr(block, "attn") else block

        ops = {}

        if hasattr(attn, "add_k_proj"):
            # Text Key Projection (Context)
            ops["to_k"] = attn.add_k_proj
            if hasattr(attn, "norm_added_k"):
                ops["norm_k"] = attn.norm_added_k

            # Image Query Projection
            if hasattr(attn, "to_q"):
                ops["to_q"] = attn.to_q
            if hasattr(attn, "norm_q"):
                ops["norm_q"] = attn.norm_q

            ops["num_heads"] = attn.heads
        else:
            raise RuntimeError(
                "Pruner in 'free' mode expects the module to have 'add_k_proj' method."
            )

        self.free_ops = ops
        print(
            f"[ShivaContextPruner] Registered free ops from {type(block).__name__}. Keys: {list(ops.keys())}"
        )

    def _apply_pooling(self, q, H=None, W=None):
        if self.pool_size is None or self.pool_size <= 1:
            return q.mean(dim=1, keepdim=True)

        is_4d = q.dim() == 4
        if is_4d:
            # (B, N, Heads, D)
            B, N, Heads, D = q.shape
            # (B, N, Heads, D) -> (B, Heads, N, D) -> (B*Heads, N, D)
            q_working = q.transpose(1, 2).reshape(B * Heads, N, D)
        else:
            # (B, N, D)
            B, N, D = q.shape
            q_working = q

        q_in = q_working.transpose(1, 2)  # (Batch_Eff, D, N)

        pool_sz = (
            (self.pool_size, self.pool_size)
            if isinstance(self.pool_size, int)
            else self.pool_size
        )

        q_pooled = None

        if H is not None and W is not None and H * W == N:
            q_2d = q_in.view(-1, D, H, W)
            q_out_2d = F.adaptive_avg_pool2d(
                q_2d, pool_sz
            )
            q_pooled = q_out_2d.flatten(2)  # (Batch_Eff, D, M)

        if q_pooled is None:
            # 1D Pooling Fallback
            target_size = (
                pool_sz[0] * pool_sz[1]
                if isinstance(pool_sz, tuple)
                else pool_sz * pool_sz
            )
            q_pooled = F.adaptive_avg_pool1d(q_in, target_size)  # (Batch_Eff, D, M)

        if is_4d:
            # (B*Heads, D, M) -> (B, Heads, D, M) -> (B, Heads, M, D) -> (B, M, Heads, D)
            M = q_pooled.shape[-1]
            q_pooled = q_pooled.view(B, Heads, D, M).permute(0, 3, 1, 2)
        else:
            # (B, D, M) -> (B, M, D)
            q_pooled = q_pooled.transpose(1, 2)

        return q_pooled

    def forward(self, hidden_states, encoder_hidden_states, k, H=None, W=None):
        original_dtype = hidden_states.dtype

        h_in = hidden_states.to(torch.float32)
        e_in = encoder_hidden_states.to(torch.float32)

        if self.mode == "learned":
            q_pooled_raw = self._apply_pooling(h_in, H, W)  # (B, M, Dim_in)
            q = self.q_proj(q_pooled_raw)  # (B, M, Proj_Dim)
            k_proj = self.k_proj(e_in)  # (B, L, Proj_Dim)

            q = F.normalize(q, p=2, dim=-1)
            k_proj = F.normalize(k_proj, p=2, dim=-1)
            scale = self.logit_scale.exp()

            # MatMul: (B, M, D) @ (B, L, D)^T -> (B, M, L)
            attn_map = torch.matmul(q, k_proj.transpose(-2, -1)) * scale
        else:
            if self.free_ops is None:
                raise RuntimeError(
                    "Pruner is in 'free' mode but 'on_register' was never called to bind model layers."
                )

            ops = self.free_ops
            with torch.no_grad():
                if "to_q" in ops:
                    q_full = ops["to_q"](h_in)
                    if "norm_q" in ops:
                        q_full = ops["norm_q"](q_full)

                    num_heads = ops.get("num_heads", 1)
                    if num_heads > 1:
                        B, N, C = q_full.shape
                        head_dim = C // num_heads
                        q_full = q_full.view(B, N, num_heads, head_dim)
                    else:
                        q_full = q_full.unsqueeze(2)  # (B, N, 1, D)
                else:
                    q_full = h_in.unsqueeze(2)  # Fallback: (B, N, 1, D)

                # Apply Pooling: (B, N, H, D) -> (B, M, H, D)
                q = self._apply_pooling(q_full, H, W)

                # --- Extract Text K (Retain Heads) ---
                if "to_k" in ops:
                    k_full = ops["to_k"](e_in)
                    if "norm_k" in ops:
                        k_full = ops["norm_k"](k_full)

                    # Reshape: (B, L, Heads, HeadDim)
                    num_heads = ops.get("num_heads", 1)
                    if num_heads > 1:
                        B, L, C = k_full.shape
                        head_dim = C // num_heads
                        k_full = k_full.view(B, L, num_heads, head_dim)
                    else:
                        k_full = k_full.unsqueeze(2)
                    k_proj = k_full
                else:
                    k_proj = e_in.unsqueeze(2)

                # Normalize per head
                q = F.normalize(q, p=2, dim=-1)
                k_proj = F.normalize(k_proj, p=2, dim=-1)

                # Einsum: (B, M, H, D) @ (B, L, H, D) -> (B, M, H, L)
                attn_map = torch.einsum("bmhd,blhd->bmhl", q, k_proj)
                attn_map = attn_map.mean(dim=2)

        scores = attn_map.max(dim=1).values

        self.last_scores = scores.detach()

        if not isinstance(k, torch.Tensor):
            k = torch.tensor(k, device=hidden_states.device, dtype=torch.float32)
        else:
            k = k.to(dtype=torch.float32)

        x_sel, x_rej, sel_idx, rej_idx = ShivaTopkFunc.apply(
            e_in, scores, k, self.Ts, self.Tk
        )

        return x_sel.to(original_dtype), x_rej.to(original_dtype), sel_idx, rej_idx

    def get_aux_loss(self):
        if not hasattr(self, "last_scores"):
            return torch.tensor(0.0)
        return -torch.var(self.last_scores)
