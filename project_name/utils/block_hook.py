import torch
from typing import Literal

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
#           attn = softmax(q @ k^T / sqrt(d))
#
#   hook_after_attn("self-attn", attn_weights)
#
#   (3) Output:
#           hidden_states = hidden_states + attn @ v
#
# ---------------------------------------------------------------
#  ③ Cross-Attention (if exists)
# ---------------------------------------------------------------
#   hook_pre_norm("cross-attn", ...)
#   hook_pre_attn("cross-attn", hidden_states_qkv, encoder_hidden_states_qkv)
#
#   (1) q from hidden_states, k/v from encoder_hidden_states
#   (2) attn = softmax(q @ k^T)
#
#   hook_after_attn("cross-attn", attn_weights)
#
#   (3) hidden_states = hidden_states + attn @ v
#
# ---------------------------------------------------------------
#  ④ FFN Stage
# ---------------------------------------------------------------
#   hook_pre_norm("ffn", ...)
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


class BaseDiTBlockHook:
    def close(self):
        pass

    def hook_pre_norm(
        self, 
        layer_name: Literal["self-attn", "cross-attn", "ffn", "parallel"],
        hidden_states: torch.Tensor,
        encoder_hidden_states: torch.Tensor = None,
        temb: torch.Tensor = None,
    ):
        return hidden_states, encoder_hidden_states

    def hook_pre_attn(
        self,
        layer_name: Literal["self-attn", "cross-attn"],
        hidden_states_qkv: tuple[torch.Tensor | None, torch.Tensor | None, torch.Tensor | None],
        encoder_hidden_states_qkv: tuple[torch.Tensor | None, torch.Tensor | None, torch.Tensor | None] = None,
    ):
        pass

    def hook_after_attn(
        self,
        layer_name: Literal["self-attn", "cross-attn"],
        attn_weights: torch.Tensor,
        encoder_attn_weights: torch.Tensor = None,
    ):
        pass
    