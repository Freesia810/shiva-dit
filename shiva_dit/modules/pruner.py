import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

from shiva_dit.utils.ste import ShivaTopkFunc
from shiva_dit.utils import global_context
from shiva_dit.hook.hook_manager import HookManager

SAVE_DIR = 'evolution_data'

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

            # scheduled shared context configs
            self.shared_spatial_config = HookManager.get_global("shiva_context_shared_spatial", None)
            self.shared_temporal_config = HookManager.get_global("shiva_context_shared_temporal", None)

        # ================== Thresholds (Buffer) ==================
        self.register_buffer("Ts", torch.tensor(float(init_Ts)), persistent=False)
        self.register_buffer("Tk", torch.tensor(float(init_Tk)), persistent=False)

    def on_register_layer_idx(self, layer_id):
        self.layer_id = layer_id

    def on_register(self, block: nn.Module):
        if self.mode != "free":
            return

        # free mode
        self.eval()
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

        elif hasattr(attn, "to_k") and hasattr(attn, "to_q"):            
            ops["to_k"] = attn.to_k
            ops["to_q"] = attn.to_q

            if hasattr(attn, "norm_k"):
                ops["norm_k"] = attn.norm_k
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

    def _get_spatial_importance(self, batch_size, H, W, device, dtype):
        if global_context.current_iteration == 0:
            return torch.ones(batch_size, H * W, 1, device=device, dtype=dtype)

        last_cfg = getattr(global_context, "last_cfg_hidden_state", None)
        if last_cfg is None:
            raise RuntimeError(
                f"No last_cfg_hidden_state found at iter {global_context.current_iteration}."
            )
        
        with torch.no_grad():
            if last_cfg.dim() == 3:
                # Case 1: Sequence format (B, S, D)
                importance = last_cfg.mean(dim=-1, keepdim=True) # (B, S, 1)

                B, S, _ = importance.shape
                if H * W == S:
                    importance_flat = importance
                else:
                    side = int(math.sqrt(S))
                    importance_2d = importance.permute(0, 2, 1).view(B, 1, side, side)
                    importance_2d = F.interpolate(
                        importance_2d, size=(H, W), mode="bilinear", align_corners=False
                    )
                    importance_flat = importance_2d.flatten(2).transpose(1, 2)

            elif last_cfg.dim() == 4:
                # Case 2: Image format (B, C, H, W)
                importance = last_cfg.mean(dim=1, keepdim=True) # (B, 1, H_src, W_src)

                if importance.shape[-1] != W or importance.shape[-2] != H:
                    importance = F.interpolate(
                        importance, size=(H, W), mode="bilinear", align_corners=False
                    )
                importance_flat = importance.flatten(2).transpose(1, 2)
            
            else:
                raise RuntimeError(
                    f"Unknown last_cfg_hidden_state shape: {last_cfg.shape}"
                )

            importance_flat = importance_flat / (importance_flat.mean() + 1e-6)
            
            return importance_flat.to(device=device, dtype=dtype)

    def _apply_pooling(self, q, H=None, W=None, importance_map=None):
        if importance_map is not None:
            # q: (B, N, Heads, D) or (B, N, D)
            # importance_map: (1, N, 1)
            if q.dim() == 4:
                weight = importance_map.unsqueeze(-1)  # (1, N, 1, 1)
            else:
                weight = importance_map  # (1, N, 1)

            q = q * weight

        if self.pool_size is None or self.pool_size <= 1:
            return q.max(dim=1, keepdim=True).values

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
            q_out_2d = F.adaptive_max_pool2d(q_2d, pool_sz)
            q_pooled = q_out_2d.flatten(2)  # (Batch_Eff, D, M)

        if q_pooled is None:
            # 1D Pooling Fallback
            target_size = (
                pool_sz[0] * pool_sz[1]
                if isinstance(pool_sz, tuple)
                else pool_sz * pool_sz
            )
            q_pooled = F.adaptive_max_pool1d(q_in, target_size)  # (Batch_Eff, D, M)

        if is_4d:
            # (B*Heads, D, M) -> (B, Heads, D, M) -> (B, Heads, M, D) -> (B, M, Heads, D)
            M = q_pooled.shape[-1]
            q_pooled = q_pooled.view(B, Heads, D, M).permute(0, 3, 1, 2)
        else:
            # (B, D, M) -> (B, M, D)
            q_pooled = q_pooled.transpose(1, 2)

        return q_pooled

    def _save_full_score_trajectory(self, scores):
        import os
        step_idx = global_context.current_iteration
        # evolution_data/step_001/
        step_dir = os.path.join(SAVE_DIR, f"step_{step_idx:03d}")
        os.makedirs(step_dir, exist_ok=True)
        save_path = os.path.join(step_dir, f"layer_{self.layer_id:03d}.npy")
        np.save(save_path, scores.detach().cpu().float().numpy())

    def _get_feature_image(self, h_in, H, W):
        if H is None or W is None or H * W != h_in.shape[0]:
            return None

        # 1. Normalize features
        x = h_in.float()
        x = (x - x.mean(dim=0)) / (x.std(dim=0) + 1e-5)

        # 2. PCA Projection (Project to top 3 components)
        if x.shape[0] > 3 and x.shape[1] > 3:
            _, _, V = torch.pca_lowrank(x, q=3, center=False, niter=2)
            rgb = torch.matmul(x, V) # (N, 3)
        else:
            # Fallback: simple slice
            rgb = x[:, :3]

        # 3. Normalize RGB to [0, 1] for visualization
        rgb_min = rgb.min(dim=0, keepdim=True).values
        rgb_max = rgb.max(dim=0, keepdim=True).values
        rgb = (rgb - rgb_min) / (rgb_max - rgb_min + 1e-5)

        # 4. Reshape to Image
        img = rgb.view(H, W, 3).cpu().numpy()
        return img

    def _get_active_policy(self, current_iter, num_inference_steps):
        active_spatial_group = 1
        active_temporal_interval = 1

        if self.shared_spatial_config:
            if current_iter < self.shared_spatial_config["spatial_step_thresh"] * num_inference_steps:
                active_spatial_group = self.shared_spatial_config["spatial_group_size"]

        if self.shared_temporal_config:
            if self.layer_id < self.shared_temporal_config["temporal_layer_thresh"]:
                active_temporal_interval = self.shared_temporal_config["temporal_interval"]

        return active_spatial_group, active_temporal_interval

    def _process_attn(self, h_in, e_in, H, W):
        if self.free_ops is None:
            raise RuntimeError(
                "Pruner is in 'free' mode but 'on_register' was never called to bind model layers."
            )

        ops = self.free_ops
        with torch.no_grad():
            num_heads = ops.get("num_heads", 1)
            importance_map = self._get_spatial_importance(h_in.shape[0], H, W, h_in.device, h_in.dtype)
            if "to_q" in ops:
                q_full = ops["to_q"](h_in)
                if "norm_q" in ops:
                    q_full = ops["norm_q"](q_full.unflatten(-1, (num_heads, -1)))
            else:
                q_full = h_in.unsqueeze(2)  # Fallback: (B, N, 1, D)

            # Apply Pooling: (B, N, H, D) -> (B, M, H, D)
            q = self._apply_pooling(q_full, H, W, importance_map=importance_map)

            # --- Extract Text K (Retain Heads) ---
            if "to_k" in ops:
                k_full = ops["to_k"](e_in)
                if "norm_k" in ops:
                    k_full = ops["norm_k"](k_full.unflatten(-1, (num_heads, -1)))
                k_proj = k_full
            else:
                k_proj = e_in.unsqueeze(2)

            scale = 1.0 / math.sqrt(h_in.shape[-1] // num_heads)
            attn_map_heads = torch.einsum("bmhd,blhd->bmhl", q, k_proj) * scale # Einsum: (B, M, H, D) @ (B, L, H, D) -> (B, M, H, L)
            attn_map = attn_map_heads.mean(dim=2) # (B, M, L)

        return attn_map

    def _forward_learned(self, h_in, e_in, H, W):
        q_pooled_raw = self._apply_pooling(h_in, H, W)  # (B, M, Dim_in)
        q = self.q_proj(q_pooled_raw)  # (B, M, Proj_Dim)
        k_proj = self.k_proj(e_in)  # (B, L, Proj_Dim)

        q = F.normalize(q, p=2, dim=-1)
        k_proj = F.normalize(k_proj, p=2, dim=-1)
        scale = self.logit_scale.exp()

        # MatMul: (B, M, D) @ (B, L, D)^T -> (B, M, L)
        attn_map = torch.matmul(q, k_proj.transpose(-2, -1)) * scale
        scores = attn_map.max(dim=1).values  # (B, L)

        return scores

    def _forward_free(self, h_in, e_in, H, W, current_iter, num_inference_steps):
        active_spatial_group, active_temporal_interval = self._get_active_policy(current_iter, num_inference_steps)

        if active_spatial_group > 1:
            leader_id = (self.layer_id // active_spatial_group) * active_spatial_group
        else:
            leader_id = self.layer_id

        cache = HookManager.get_global(f"SHIVA_SCORE_CACHE_{global_context.encoder_flag}", {})
        if self.layer_id != leader_id:
            leader_data = cache.get(leader_id)
            if leader_data is not None:
                return leader_data["score"], leader_data.get("idx_sorted")

        elif active_temporal_interval > 1 and current_iter % active_temporal_interval != 0:
            my_data = cache.get(self.layer_id)
            if my_data is not None:
                return my_data["score"], my_data.get("idx_sorted")

        # print(current_iter, self.layer_id, leader_id, active_spatial_group, active_temporal_interval, global_context.encoder_flag)
        attn_map = self._process_attn(h_in, e_in, H, W)
        scores = attn_map.max(dim=1).values  # (B, L)
        idx_sorted = torch.argsort(scores, dim=-1, descending=True)
        if self.layer_id == leader_id:
            cache[self.layer_id] = {
                "score": scores.detach(), 
                "idx_sorted": idx_sorted.detach(),
                "step": current_iter
            }
            HookManager.set_global(f"SHIVA_SCORE_CACHE_{global_context.encoder_flag}", cache)

        return scores, idx_sorted

    def forward(self, hidden_states, encoder_hidden_states, k):
        original_dtype = hidden_states.dtype

        h_in = hidden_states
        e_in = encoder_hidden_states
        H, W = global_context.hidden_seq_height, global_context.hidden_seq_width

        if self.mode == "learned":
            scores = self._forward_learned(h_in, e_in, H, W)
            idx_sorted = None
        else:
            scores, idx_sorted = self._forward_free(h_in, e_in, H, W, global_context.current_iteration, global_context.num_inference_steps)

        # self._save_full_score_trajectory(scores[0])

        self.last_scores = scores.detach()

        if isinstance(k, torch.Tensor):
            k_int = int(k.item())
        else:
            k_int = int(k)

        x_sel, x_rej, sel_idx, rej_idx = ShivaTopkFunc.fast_forward(
            e_in, scores, k_int, idx_sorted=idx_sorted
        )

        return x_sel.to(original_dtype), x_rej.to(original_dtype), sel_idx, rej_idx

    def get_aux_loss(self):
        if not hasattr(self, "last_scores"):
            return torch.tensor(0.0)
        return -torch.var(self.last_scores)
