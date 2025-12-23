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
        batch_size = x_q.shape[0]

        # 1. Image Stream Projections (`sample` projections)
        query = attn_module.to_q(x_q)
        key = attn_module.to_k(x_kv)
        value = attn_module.to_v(x_kv)

        inner_dim = key.shape[-1]
        head_dim = inner_dim // attn_module.heads

        def split_heads(x):
            return x.view(batch_size, -1, attn_module.heads, head_dim).transpose(1, 2)

        query = split_heads(query)
        key = split_heads(key)
        value = split_heads(value)

        if attn_module.norm_q is not None:
            query = attn_module.norm_q(query)
        if attn_module.norm_k is not None:
            key = attn_module.norm_k(key)

        # 2. Context Stream Projections & Concatenation (`context` projections)
        if encoder_hidden_states is not None:
            # Project
            enc_query = attn_module.add_q_proj(encoder_hidden_states)
            enc_key = attn_module.add_k_proj(encoder_hidden_states)
            enc_value = attn_module.add_v_proj(encoder_hidden_states)

            # Reshape
            enc_query = split_heads(enc_query)
            enc_key = split_heads(enc_key)
            enc_value = split_heads(enc_value)

            # Norm
            if attn_module.norm_added_q is not None:
                enc_query = attn_module.norm_added_q(enc_query)
            if attn_module.norm_added_k is not None:
                enc_key = attn_module.norm_added_k(enc_key)

            # Joint Attention: Concatenate along sequence dim (dim=2)
            query = torch.cat([query, enc_query], dim=2)
            key = torch.cat([key, enc_key], dim=2)
            value = torch.cat([value, enc_value], dim=2)

        # 3. Attention Calculation
        hidden_states = F.scaled_dot_product_attention(query, key, value, dropout_p=0.0, is_causal=False)

        # 4. Reshape Output
        hidden_states = hidden_states.transpose(1, 2).reshape(batch_size, -1, attn_module.heads * head_dim)
        hidden_states = hidden_states.to(query.dtype)

        # 5. Split and Output Projections
        encoder_out = None

        if encoder_hidden_states is not None:
            # Split the attention outputs back into Image and Context
            # x_q.shape[1] is the sequence length of the image part
            img_seq_len = x_q.shape[1]
            
            img_hidden_states = hidden_states[:, :img_seq_len]
            enc_hidden_states = hidden_states[:, img_seq_len:]

            # Context output projection
            if not getattr(attn_module, "context_pre_only", False):
                encoder_out = attn_module.to_add_out(enc_hidden_states)
            
            hidden_states = img_hidden_states

        # Image output projection (linear)
        hidden_states = attn_module.to_out[0](hidden_states)
        
        # Image output dropout (if present in to_out list)
        if len(attn_module.to_out) > 1:
            hidden_states = attn_module.to_out[1](hidden_states)

        return hidden_states, encoder_out

    def _apply_sd3_block_logic(
        self,
        module_self,
        hidden_states,
        encoder_hidden_states,
        x_q_norm,
        x_kv_norm,
        norm_encoder_hidden_states,
        modulations,  # (gate_msa, shift_mlp, scale_mlp, gate_mlp)
        context_modulations,  # (c_gate_msa, c_shift_mlp, c_scale_mlp, c_gate_mlp) or None
    ):
        gate_msa, shift_mlp, scale_mlp, gate_mlp = modulations

        # Unpack context modulations safely
        if context_modulations is not None:
            c_gate_msa, c_shift_mlp, c_scale_mlp, c_gate_mlp = context_modulations
        else:
            c_gate_msa = c_shift_mlp = c_scale_mlp = c_gate_mlp = None

        attn_output, context_attn_output = self._manual_attention(
            module_self.attn,
            x_q=x_q_norm,
            x_kv=x_kv_norm,
            encoder_hidden_states=norm_encoder_hidden_states,
        )

        # Residual 1 (Attention)
        hidden_states = hidden_states + gate_msa.unsqueeze(1) * attn_output
        if encoder_hidden_states is not None and context_attn_output is not None:
            encoder_hidden_states = (
                encoder_hidden_states + c_gate_msa.unsqueeze(1) * context_attn_output
            )

        # === 2. FFN ===
        # Image Stream FFN
        norm_hidden_states = module_self.norm2(hidden_states)
        norm_hidden_states = norm_hidden_states * (
            1 + scale_mlp.unsqueeze(1)
        ) + shift_mlp.unsqueeze(1)
        ff_output = module_self.ff(norm_hidden_states)
        hidden_states = hidden_states + gate_mlp.unsqueeze(1) * ff_output

        # Context Stream FFN
        if encoder_hidden_states is not None:
            norm_encoder_hidden_states = module_self.norm2_context(
                encoder_hidden_states
            )
            norm_encoder_hidden_states = norm_encoder_hidden_states * (
                1 + c_scale_mlp.unsqueeze(1)
            ) + c_shift_mlp.unsqueeze(1)
            context_ff_output = module_self.ff_context(norm_encoder_hidden_states)

            encoder_hidden_states = (
                encoder_hidden_states + c_gate_mlp.unsqueeze(1) * context_ff_output
            )

        return encoder_hidden_states, hidden_states

    def delegated_call(self, func, module_self, *args, **kwargs):
        sig = inspect.signature(func)
        bound = sig.bind_partial(module_self, *args, **kwargs)
        bound.apply_defaults()

        hidden_states = bound.arguments.get("hidden_states", None)
        encoder_hidden_states = bound.arguments.get("encoder_hidden_states", None)
        temb = bound.arguments.get("temb", None)
        image_rotary_emb = bound.arguments.get("image_rotary_emb", None)

        if image_rotary_emb is not None:
            raise ValueError(
                "SparseDiT does not support explicit image_rotary_emb. Set it to None."
            )

        layer_idx = self.layer_idx
        H, W = global_context.hidden_seq_height, global_context.hidden_seq_width

        # === 1. Poolingformer (Bottom Layers) ===
        if layer_idx in self.poolingformer_layers:
            norm_hidden_states, gate_msa, shift_mlp, scale_mlp, gate_mlp = (
                module_self.norm1(hidden_states, emb=temb)
            )
            v = module_self.attn.to_v(norm_hidden_states)
            v_mean = v.mean(dim=1, keepdim=True)
            hidden_states = hidden_states + gate_msa.unsqueeze(1) * v_mean

            norm_hidden_states = module_self.norm2(hidden_states)
            norm_hidden_states = norm_hidden_states * (
                1 + scale_mlp.unsqueeze(1)
            ) + shift_mlp.unsqueeze(1)

            ff_output = module_self.ff(norm_hidden_states)
            hidden_states = hidden_states + gate_mlp.unsqueeze(1) * ff_output

            return (encoder_hidden_states, hidden_states)

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
        norm_encoder_hidden_states = None
        context_modulations = None
        if encoder_hidden_states is not None:
            res_ctx = module_self.norm1_context(encoder_hidden_states, emb=temb)
            norm_encoder_hidden_states = res_ctx[0]

            # (c_gate_msa, c_shift_mlp, c_scale_mlp, c_gate_mlp)
            context_modulations = res_ctx[1:]  

        x_residual = None
        x_q_norm = None
        x_kv_norm = None
        modulations = None  # (gate_msa, shift_mlp, scale_mlp, gate_mlp)

        # A. Sparse Generation (Gen)
        if sdtm_role == "gen":
            HookManager.set_global(f"dense_buffer_{group_id}", hidden_states)
            r = self.get_current_pruning_rate()

            norm_hidden_states, *modulations = module_self.norm1(
                hidden_states, emb=temb
            )

            ratio = math.sqrt((1 - r))
            H_new, W_new = int(H * ratio), int(W * ratio)

            # Downsample Normed Dense -> Sparse Query
            x_norm_dense_2d = norm_hidden_states.transpose(1, 2).view(
                hidden_states.shape[0], -1, H, W
            )
            x_q_norm = F.adaptive_avg_pool2d(x_norm_dense_2d, (H_new, W_new))
            x_q_norm = x_q_norm.flatten(2).transpose(1, 2)

            x_kv_norm = norm_hidden_states
            x_dense_2d = hidden_states.transpose(1, 2).view(
                hidden_states.shape[0], -1, H, W
            )
            x_residual = F.adaptive_avg_pool2d(x_dense_2d, (H_new, W_new))
            x_residual = x_residual.flatten(2).transpose(1, 2)

        # B. Sparse Transformers (Sparse)
        elif sdtm_role == "sparse":
            return func(module_self, *args, **kwargs)

        # C. Dense Recovery (Rec)
        elif sdtm_role == "rec":
            x_sparse = hidden_states
            x_dense_orig = HookManager.get_global(f"dense_buffer_{group_id}")

            M = x_sparse.shape[1]
            ratio = math.sqrt(M / (H * W))
            H_new, W_new = int(H * ratio), int(W * ratio)

            # Upsample Sparse -> Dense
            x_sparse_2d = x_sparse.transpose(1, 2).view(
                x_sparse.shape[0], -1, H_new, W_new
            )
            x_upsampled = F.interpolate(x_sparse_2d, size=(H, W), mode="nearest")
            x_upsampled = x_upsampled.flatten(2).transpose(1, 2)

            # Fuse with Original Dense -> Merged Residual Base
            w1 = self.recovery_projs[group_id]["w1"]
            w2 = self.recovery_projs[group_id]["w2"]
            x_residual = w1(x_upsampled) + w2(x_dense_orig)

            # Norm Merged Dense -> Dense Query
            norm_x_merged, *modulations = module_self.norm1(x_residual, emb=temb)
            x_q_norm = norm_x_merged

            # Downsample Normed Merged -> Sparse KV
            norm_x_merged_2d = norm_x_merged.transpose(1, 2).view(
                x_residual.shape[0], -1, H, W
            )
            x_kv_norm = F.adaptive_avg_pool2d(norm_x_merged_2d, (H_new, W_new))
            x_kv_norm = x_kv_norm.flatten(2).transpose(1, 2)

        # === Execute Common Logic ===
        return self._apply_sd3_block_logic(
            module_self,
            hidden_states=x_residual,
            encoder_hidden_states=encoder_hidden_states,
            x_q_norm=x_q_norm,
            x_kv_norm=x_kv_norm,
            norm_encoder_hidden_states=norm_encoder_hidden_states,
            modulations=modulations,
            context_modulations=context_modulations,
        )


if __name__ == "__main__":
    model_type = "sd3"
    pipe, model_attrs = get_pipeline(model_type)

    sparsedit_config = {
        "sdtm_ranges": [(2, 6)],
        "poolingformer_layers": [-1],
        "r_min": 0.0,
        "r_max": 0.0,
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
