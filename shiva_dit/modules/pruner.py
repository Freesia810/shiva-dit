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

    @staticmethod
    def _get_free_ops(block: nn.Module):
        is_omnigen = block.__class__.__name__ == "OmniGenBlock"
        if is_omnigen:
            attn = block.self_attn
        else:
            attn = block.attn if hasattr(block, "attn") else block

        ops = {}

        if is_omnigen:
            # in OmniGen, we use to_k and to_q instead of add_k_proj, and no norm
            if hasattr(attn, "to_k"):
                ops["to_k"] = attn.to_k

            # Image Query Projection
            if hasattr(attn, "to_q"):
                ops["to_q"] = attn.to_q
            ops["num_heads"] = attn.heads

            return ops

        if hasattr(attn, "add_k_proj") and attn.add_k_proj is not None:
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
        
        return ops

    def on_register(self, block: nn.Module):
        if self.mode != "free":
            return

        # free mode
        self.eval()
        self.free_ops = ShivaContextPruner._get_free_ops(block)
        print(
            f"[ShivaContextPruner] Registered free ops from {type(block).__name__}. Keys: {list(self.free_ops.keys())}"
        )

    @staticmethod
    def _get_spatial_importance(batch_size, H, W, device, dtype):
        if global_context.current_iteration == 0:
            return None

        last_cfg = getattr(global_context, "last_cfg_hidden_state", None)
        if last_cfg is None:
            return None
        
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

    def _apply_pooling(q, H=None, W=None, pool_size=None, importance_map=None):
        if importance_map is not None:
            dims_to_add = q.dim() - importance_map.dim()
            weight = importance_map.view(*importance_map.shape, *(1,) * dims_to_add)

            # (Sequence Length Alignment)
            q_len = q.size(1)
            w_len = weight.size(1)

            if q_len != w_len:
                if q_len > w_len:
                    q[:, :w_len, ...] *= weight
                else:
                    q *= weight[:, :q_len, ...]
            else:
                q = q * weight

        if pool_size is None or pool_size <= 1:
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
            (pool_size, pool_size)
            if isinstance(pool_size, int)
            else pool_size
        )

        q_pooled = None
        
        if H is not None and W is not None and q_in.shape[1] >= H * W:
            num_spatial = H * W
            q_spatial = q_in[:, :num_spatial, :] # (B_eff, H*W, D)
            q_extra = q_in[:, num_spatial:, :]   # (B_eff, N-H*W, D)

            q_2d = q_spatial.transpose(1, 2).view(-1, D, H, W)
            q_out_2d = F.adaptive_max_pool2d(q_2d, pool_sz) 
            q_pooled_spatial = q_out_2d.flatten(2) # (B_eff, D, M)

            if q_extra.size(1) > 0:
                q_extra_mean = q_extra.mean(dim=1, keepdim=True).transpose(1, 2)
                q_pooled = torch.cat([q_pooled_spatial, q_extra_mean], dim=2)
            else:
                q_pooled = q_pooled_spatial

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

    @staticmethod
    def _process_attn(h_in, e_in, H, W, free_ops, pool_size, attention_mask=None, enable_importance=True):
        if free_ops is None:
            raise RuntimeError(
                "Pruner is in 'free' mode but 'on_register' was never called to bind model layers."
            )

        ops = free_ops
        with torch.no_grad():
            num_heads = ops.get("num_heads", 1)
            # TODO: is importance really important?
            enable_importance = False
            if enable_importance:
                importance_map = ShivaContextPruner._get_spatial_importance(h_in.shape[0], H, W, h_in.device, h_in.dtype)
            else:
                importance_map = None

            if "to_q" in ops:
                q_proj = ops["to_q"](h_in) # (B, N, dim_q)
            else:
                q_proj = h_in.unsqueeze(2) # Fallback
            
            query_dim = q_proj.shape[-1]
            head_dim = query_dim // num_heads

            q_full = q_proj.unflatten(-1, (num_heads, head_dim)) # (B, N, num_heads, head_dim)

            if "norm_q" in ops:
                q_full = ops["norm_q"](q_full)

            # Apply Pooling: (B, N, H, D) -> (B, M, H, D)
            q = ShivaContextPruner._apply_pooling(q_full, H, W, pool_size=pool_size, importance_map=importance_map).transpose(1, 2) # (B, H, M, D)

            if "to_k" in ops:
                k_proj = ops["to_k"](e_in) # (B, L, dim_k)
                
                inner_dim = k_proj.shape[-1]
                kv_heads = inner_dim // head_dim
                
                # Reshape K: (B, L, kv_heads, head_dim)
                k_full = k_proj.unflatten(-1, (kv_heads, head_dim))

                if "norm_k" in ops:
                    k_full = ops["norm_k"](k_full)
            else:
                kv_heads = num_heads
                k_full = e_in.unsqueeze(2)
            
            k = k_full.transpose(1, 2) # (B, kv_heads, L, head_dim)

            if kv_heads < num_heads:
                n_rep = num_heads // kv_heads
                k = k.repeat_interleave(n_rep, dim=1)
            
            scale = 1.0 / math.sqrt(head_dim)

            attn_map_heads = q @ k.transpose(-2, -1) * scale # (B, H, M, L)

            if attention_mask is not None:
                sliced_mask = attention_mask[:, :, -q.shape[-2]:, :k_proj.shape[-2]]
                attn_map_heads = attn_map_heads + sliced_mask

        return attn_map_heads

    def _forward_learned(self, h_in, e_in, H, W):
        q_full = self.q_proj(h_in)
        q_pooled = ShivaContextPruner._apply_pooling(q_full, H, W, pool_size=self.pool_size)  # (B, M, Dim_in)
        k_proj = self.k_proj(e_in)  # (B, L, Proj_Dim)

        q = F.normalize(q_pooled, p=2, dim=-1)
        k_proj = F.normalize(k_proj, p=2, dim=-1)
        scale = self.logit_scale.exp()

        # MatMul: (B, M, D) @ (B, L, D)^T -> (B, M, L)
        attn_map = torch.matmul(q, k_proj.transpose(-2, -1)) * scale
        scores = attn_map.max(dim=1).values  # (B, L)

        return scores

    def _forward_free(self, h_in, e_in, H, W, current_iter, num_inference_steps, attention_mask=None):
        active_spatial_group, active_temporal_interval = self._get_active_policy(current_iter, num_inference_steps)

        if active_spatial_group > 1:
            leader_id = (self.layer_id // active_spatial_group) * active_spatial_group
        else:
            leader_id = self.layer_id

        if active_temporal_interval > 1:
            anchor_step = current_iter - (current_iter % active_temporal_interval)
        else:
            anchor_step = current_iter

        cache = HookManager.get_global(f"SHIVA_SCORE_CACHE_{global_context.encoder_forward_key()}", {})
        if self.layer_id != leader_id:
            leader_data = cache.get(leader_id)
            if leader_data is not None and leader_data["step"] == anchor_step:
                return leader_data["score"], leader_data.get("idx_sorted")
        elif active_temporal_interval > 1 and current_iter != anchor_step:
            my_data = cache.get(self.layer_id)
            if my_data is not None and my_data["step"] == anchor_step:
                return my_data["score"], my_data.get("idx_sorted")

        # print(current_iter, self.layer_id, leader_id, active_spatial_group, active_temporal_interval, global_context.encoder_forward_key())
        attn_map_heads = ShivaContextPruner._process_attn(h_in, e_in, H, W, self.free_ops, self.pool_size, attention_mask=attention_mask)

        # cfg guard
        attn_map = attn_map_heads.mean(dim=1) # (B, H, M, L) -> (B, M, L)
        B = attn_map.shape[0]
        avg_map = attn_map.mean(dim=0, keepdim=True) # (B, M, L) -> (1, M, L)
        scores = avg_map.max(dim=1).values.expand(B, -1) # (B, L)
        idx_sorted = torch.argsort(scores, dim=-1, descending=True)
        if self.layer_id == leader_id:
            cache[self.layer_id] = {
                "score": scores.detach(), 
                "idx_sorted": idx_sorted.detach(),
                "step": current_iter
            }
            HookManager.set_global(f"SHIVA_SCORE_CACHE_{global_context.encoder_forward_key()}", cache)

        return scores, idx_sorted

    def forward(self, hidden_states, encoder_hidden_states, k, attention_mask=None):
        original_dtype = hidden_states.dtype

        h_in = hidden_states
        e_in = encoder_hidden_states
        H, W = global_context.hidden_seq_height, global_context.hidden_seq_width

        if self.mode == "learned":
            scores = self._forward_learned(h_in, e_in, H, W)
            idx_sorted = None
        else:
            scores, idx_sorted = self._forward_free(h_in, e_in, H, W, global_context.current_iteration, global_context.num_inference_steps, attention_mask)

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
