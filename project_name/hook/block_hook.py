import torch
import torch.nn as nn
from typing import Literal, Optional, Tuple

# ===============================================================
#                    DiT Block Forward Overview
# ===============================================================
#
#   Input:
#       hidden_states: Tensor [B, L_img, D]              ← image tokens
#       encoder_hidden_states: Tensor [B, L_cond, D] or None  ← condition tokens (e.g. text)
#       temb: Tensor [B, D_t]                            ← time embedding (optional)
#
# ---------------------------------------------------------------
#  ① Pre-Norm Stage
# ---------------------------------------------------------------
#   hook_pre_norm("self-attn" / "cross-attn" / "ffn" / "parallel")
#
#   hidden_states -----> Norm1 ------------------------------+
#                                                            |
#                                                            v
# ---------------------------------------------------------------
#  ② Attention Stage (Self-Attn)
# ---------------------------------------------------------------
#   hook_pre_attn("self-attn", hidden_states_qkv, encoder_hidden_states_qkv)
#
#   (1) Compute q, k, v:
#           q, k, v = proj(hidden_states or concat(hidden_states, encoder_hidden_states))
#
#   (2) Attention Map:
#           attn = softmax(q @ k^T / sqrt(d)) @ v
#
#   hook_after_attn("self-attn", attn_weights) means attention map
#
#   (3) AfterModule:
#   hook_after_module("self-attn", attn_output)
#
#   (4) Output:
#           hidden_states = hidden_states + attn_output
#
# ---------------------------------------------------------------
#  ③ Cross-Attention (if exists)
# ---------------------------------------------------------------
#   hook_pre_norm("cross-attn", ...)
#   hook_pre_attn("cross-attn", hidden_states_qkv, encoder_hidden_states_qkv)
#
#   (1) q from hidden_states, k/v from encoder_hidden_states
#   (2) attn = softmax(q @ k^T / sqrt(d)) @ v
#
#   hook_after_attn("cross-attn", attn_weights)
#
#   (3) AfterModule:
#   hook_after_module("cross-attn", attn_output)
#
#   (4) hidden_states = hidden_states + attn_output
#
# ---------------------------------------------------------------
#  ④ FFN Stage
# ---------------------------------------------------------------
#   hook_pre_norm("ffn", ...) -> after_norm -> after_module
#
#   hidden_states -----> Norm3 --> MLP(hidden_states)
#   hidden_states = hidden_states + MLP_out
#
# ---------------------------------------------------------------
#  ⑤ Parallel Path (for models like Flux-Single-DiT)
# ---------------------------------------------------------------
#   hook_pre_norm("parallel", ...)
#
#   Attention(hidden_states)   │
#                              ├── parallel paths
#   MLP(hidden_states)         │
#
#   hidden_states = hidden_states + attn_out + mlp_out
#
# ===============================================================
#
# prenorm -> (norm) -> afternorm ->     (Attention/FFN)  -> aftermodule
#                                     preattn -> afterattn
#                                   (QKV hook) (Attn Map hook)

class BaseDiTBlockHook:
    def __init__(self, **kwargs):
        pass

    def register_name(self, model_type:str, layer_name: str):
        self.model_type = model_type
        self.layer_name = layer_name

    def close(self):
        pass

    def on_register_hook(self, block: nn.Module):
        pass

    def hook_pre_norm(
        self, 
        module_name: Literal["self-attn", "cross-attn", "ffn", "parallel"],
        hidden_states: torch.Tensor,
        encoder_hidden_states: torch.Tensor = None,
        temb: torch.Tensor = None,
    ):
        return hidden_states, encoder_hidden_states
    
    def hook_after_norm(
        self, 
        module_name: Literal["self-attn", "cross-attn", "ffn"],
        norm_hidden_states: torch.Tensor,
        norm_encoder_hidden_states: torch.Tensor = None,
        image_rotary_emb: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
    ):
        if image_rotary_emb is not None:
            return norm_hidden_states, norm_encoder_hidden_states, image_rotary_emb
        else:
            return norm_hidden_states, norm_encoder_hidden_states
        
    
    def hook_after_module(
        self, 
        module_name: Literal["self-attn", "cross-attn", "ffn"],
        hidden_output: torch.Tensor,
        encoder_hidden_output: torch.Tensor = None,
    ):
        return hidden_output, encoder_hidden_output

    def hook_pre_attn(
        self,
        module_name: Literal["self-attn", "cross-attn"],
        hidden_states_qkv: tuple[
            torch.Tensor | None, torch.Tensor | None, torch.Tensor | None
        ],
        encoder_hidden_states_qkv: tuple[
            torch.Tensor | None, torch.Tensor | None, torch.Tensor | None
        ] = None,
    ):
        return hidden_states_qkv, encoder_hidden_states_qkv

    def hook_after_attn(
        self,
        module_name: Literal["self-attn", "cross-attn"],
        attn_weights: torch.Tensor,
        encoder_attn_weights: torch.Tensor = None,
    ):
        return attn_weights, encoder_attn_weights
