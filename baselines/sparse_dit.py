import inspect
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Literal, Tuple, Union
from shiva_dit.hook import BaseDiTBlockHook, BaseDelegator, HookManager
from shiva_dit.utils import (
    global_context,
    get_pipeline,
    process_quick_start,
    parse_image_rotary_emb,
    join_image_rotary_emb,
)


class SparseDiTBlockHook(BaseDelegator, BaseDiTBlockHook, nn.Module):
    """
    SparseDiT: Token Sparsification for Efficient Diffusion Transformer
    Reference:
        - Paper: https://arxiv.org/abs/2412.06028
    """

    def __init__(
        self,
        poolingformer_layers: list = [0, 1],
        sdtm_ranges: list = [(2, 10), (14, 22)],
        r_min: float = 0.5,
        r_max: float = 0.0,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.sdtm_ranges = sdtm_ranges
        self.poolingformer_layers = poolingformer_layers
        self.r_min = r_min
        self.r_max = r_max

        # Key: sdtm_group_index, Value: Tensor
        self.dense_buffer = {}

        self.recovery_projs = nn.ModuleList()
        for _ in sdtm_ranges:
            self.recovery_projs.append(
                nn.ModuleDict({"w1": nn.Identity(), "w2": nn.Identity()})
            )

    def on_register_hook(self, block: nn.Module):
        hidden_dim = global_context.hidden_dim
        if hidden_dim is None:
            raise ValueError("Cannot get hidden_dim")

        device = next(block.parameters()).device
        dtype = next(block.parameters()).dtype

        # X_merged = Upsample(Xs)W1 + XW2
        for i in range(len(self.sdtm_ranges)):
            self.recovery_projs[i]["w1"] = nn.Linear(hidden_dim, hidden_dim).to(
                device, dtype
            )
            self.recovery_projs[i]["w2"] = nn.Linear(hidden_dim, hidden_dim).to(
                device, dtype
            )
            nn.init.zeros_(self.recovery_projs[i]["w1"].weight)
            nn.init.zeros_(self.recovery_projs[i]["w1"].bias)
            nn.init.eye_(self.recovery_projs[i]["w2"].weight)
            nn.init.zeros_(self.recovery_projs[i]["w2"].bias)

        self.to(device, dtype)

    def get_current_pruning_rate(self):
        if (
            not hasattr(global_context, "current_timestep")
            or global_context.current_timestep is None
        ):
            raise ValueError("global_context.current_timestep is not set")

        i = global_context.current_iteration
        T = global_context.num_inference_steps
        t = T - i

        threshold = T / 4.0

        if t < threshold:
            r = self.r_min
        else:
            ratio = (t - threshold) / (T - threshold)
            r = self.r_min + (self.r_max - self.r_min) * ratio

        return r

    def _manual_attention(self, attn_module, x_q, x_kv, encoder_hidden_states=None):
        if hasattr(attn_module, "to_q"):
            q = attn_module.to_q(x_q)
            k = attn_module.to_k(x_kv)
            v = attn_module.to_v(x_kv)
        else:
            raise NotImplementedError(
                "Unknown attention module structure (packed QKV not supported)"
            )

        if encoder_hidden_states is not None:
            if hasattr(attn_module, "add_k_proj") and hasattr(
                attn_module, "add_v_proj"
            ):
                k_context = attn_module.add_k_proj(encoder_hidden_states)
                v_context = attn_module.add_v_proj(encoder_hidden_states)

                k = torch.cat([k, k_context], dim=1)
                v = torch.cat([v, v_context], dim=1)

        # 3. Split Heads
        B, N_q, C = q.shape
        B, N_kv, _ = k.shape

        num_heads = attn_module.heads
        head_dim = C // num_heads

        q = q.view(B, N_q, num_heads, head_dim).transpose(1, 2)
        k = k.view(B, N_kv, num_heads, head_dim).transpose(1, 2)
        v = v.view(B, N_kv, num_heads, head_dim).transpose(1, 2)

        # 4. Scaled Dot Product Attention
        out = F.scaled_dot_product_attention(q, k, v, dropout_p=0.0)

        out = out.transpose(1, 2).reshape(B, N_q, C)

        # 5. Output Projection
        if hasattr(attn_module, "to_out"):
            out = attn_module.to_out[0](out)
            out = attn_module.to_out[1](out)

        return out

    def delegated_call(self, func, module_self, *args, **kwargs):
        sig = inspect.signature(func)
        bound = sig.bind_partial(module_self, *args, **kwargs)
        bound.apply_defaults()

        hidden_states = bound.arguments.get("hidden_states", None)
        encoder_hidden_states = bound.arguments.get("encoder_hidden_states", None)
        image_rotary_emb = bound.arguments.get("image_rotary_emb", None)

        if image_rotary_emb is not None:
            raise ValueError(
                "SparseDiT baseline does not support explicit `image_rotary_emb` input. "
                "Because pooling disrupts the spatial grid required for pre-computed RoPE. "
                "Please set image_rotary_emb=None or modify the model to re-compute RoPE dynamically."
            )

        layer_idx = self.layer_idx
        H, W = global_context.hidden_seq_height, global_context.hidden_seq_width

        # === 1. Poolingformer (Bottom Layers) ===
        if layer_idx in self.poolingformer_layers:
            residual = hidden_states
            x = module_self.norm1(hidden_states)
            v = module_self.attn.to_v(x)
            v_mean = v.mean(dim=1, keepdim=True)
            x = residual + v_mean

            residual = x
            x = module_self.norm2(x)
            x = module_self.ffn(x)
            x = x + residual

            return (
                (encoder_hidden_states, x) if encoder_hidden_states is not None else x
            )

        # Determine SDTM Role
        sdtm_role = None
        group_id = -1
        for i, (start, end) in enumerate(self.sdtm_ranges):
            if layer_idx == start:
                sdtm_role = "gen"
                group_id = i
                break
            elif start < layer_idx < end:
                sdtm_role = "sparse"
                group_id = i
                break
            elif layer_idx == end:
                sdtm_role = "rec"
                group_id = i
                break

        if sdtm_role is None:
            return func(module_self, *args, **kwargs)

        # === 2. SDTM Logic ===

        # A. Sparse Generation (Gen)
        if sdtm_role == "gen":
            self.dense_buffer[group_id] = hidden_states
            r = self.get_current_pruning_rate()

            ratio = math.sqrt((1 - r))
            H_new, W_new = int(H * ratio), int(W * ratio)

            # Pooling (Dense -> Sparse)
            x_dense = hidden_states.transpose(1, 2).view(
                hidden_states.shape[0], -1, H, W
            )
            x_sparse = F.adaptive_avg_pool2d(x_dense, (H_new, W_new))
            x_sparse = x_sparse.flatten(2).transpose(1, 2)

            # Manual Attention (Sparse Q, Dense K/V + Text)
            x_norm_dense = module_self.norm1(hidden_states)
            x_norm_sparse = module_self.norm1(x_sparse)

            attn_out = self._manual_attention(
                module_self.attn,
                x_q=x_norm_sparse,
                x_kv=x_norm_dense,
                encoder_hidden_states=encoder_hidden_states,
            )

            x_sparse = x_sparse + attn_out

            # FFN
            residual = x_sparse
            x_sparse = module_self.norm2(x_sparse)
            x_sparse = module_self.ffn(x_sparse)
            x_sparse = x_sparse + residual

            final_hidden = x_sparse
            return (
                (encoder_hidden_states, final_hidden)
                if encoder_hidden_states is not None
                else final_hidden
            )

        # B. Sparse Transformers
        elif sdtm_role == "sparse":
            return func(module_self, *args, **kwargs)

        # C. Dense Recovery (Rec)
        elif sdtm_role == "rec":
            x_sparse = hidden_states
            x_dense_orig = self.dense_buffer[group_id]

            M = x_sparse.shape[1]
            ratio = math.sqrt(M / (H * W))
            H_new, W_new = int(H * ratio), int(W * ratio)

            # Upsample (Sparse -> Dense)
            x_sparse_2d = x_sparse.transpose(1, 2).view(
                x_sparse.shape[0], -1, H_new, W_new
            )
            x_upsampled = F.interpolate(x_sparse_2d, size=(H, W), mode="nearest")
            x_upsampled = x_upsampled.flatten(2).transpose(1, 2)

            # Linear Fusion
            w1 = self.recovery_projs[group_id]["w1"]
            w2 = self.recovery_projs[group_id]["w2"]
            x_merged = w1(x_upsampled) + w2(x_dense_orig)

            # Manual Attention (Dense Q, Sparse K/V + Text)
            x_norm_merged = module_self.norm1(x_merged)
            x_norm_sparse = module_self.norm1(x_sparse)

            attn_out = self._manual_attention(
                module_self.attn,
                x_q=x_norm_merged,
                x_kv=x_norm_sparse,
                encoder_hidden_states=encoder_hidden_states,
            )

            x_dense = x_merged + attn_out

            # FFN
            residual = x_dense
            x_dense = module_self.norm2(x_dense)
            x_dense = module_self.ffn(x_dense)
            x_dense = x_dense + residual

            return (
                (encoder_hidden_states, x_dense)
                if encoder_hidden_states is not None
                else x_dense
            )

        return func(module_self, *args, **kwargs)


if __name__ == "__main__":
    model_type = "sd3"
    pipe, model_attrs = get_pipeline(model_type)

    sparsedit_config = {
        "sdtm_ranges": [(2, 6), (10, 14)],
        "poolingformer_layers": [0, 1],
        "r_min": 0.0,
        "r_max": 0.5,
    }

    HookManager.configure_hooks(
        model_type=model_type,
        default=(SparseDiTBlockHook, sparsedit_config),
    )

    with HookManager.enabled():
        HookManager.inject_hooks(
            model_type=model_type, pipe=pipe, model_attrs=model_attrs
        )
        res = process_quick_start(pipe)
        res.save(f"{model_type}_sparsedit.png")
