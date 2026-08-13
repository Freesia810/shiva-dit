from typing import Iterable, Optional

import torch.nn as nn
import torch.nn.functional as F

CONTEXT_SIGNALS = ("time", "prompt", "layer")


class ShivaRouter(nn.Module):
    def __init__(
        self,
        hidden_dim: int,
        prompt_embed_dim: int,
        layer_embed_dim: int,
        bottleneck_dim: int,
        time_embed_dim: int = None,
        context_signals: Optional[Iterable[str]] = CONTEXT_SIGNALS,
    ):
        super().__init__()

        if time_embed_dim is None:
            time_embed_dim = hidden_dim

        # Which terms of h_ctx (Eq. 8) are kept. Dropping all of them degenerates the
        # router to a purely local scorer s_i = w^T LN(sigma(Proj_x(x_i))). Individual
        # signals can be dropped to separate "denoising stage" (time), "semantic
        # alignment" (prompt) and "abstraction level" (layer) in the ablation.
        signals = tuple(context_signals or ())
        unknown = set(signals) - set(CONTEXT_SIGNALS)
        if unknown:
            raise ValueError(
                f"Unknown context signals {sorted(unknown)}; expected subset of {CONTEXT_SIGNALS}"
            )
        self.context_signals = signals
        self.use_context = len(signals) > 0

        self.token_proj = nn.Linear(hidden_dim, bottleneck_dim)

        if "time" in signals:
            self.temb_proj = nn.Linear(time_embed_dim, bottleneck_dim, bias=False)
        if "prompt" in signals:
            self.prompt_proj = nn.Linear(prompt_embed_dim, bottleneck_dim, bias=False)
        if "layer" in signals:
            self.layer_proj = nn.Linear(layer_embed_dim, bottleneck_dim, bias=False)
        if self.use_context:
            self.context_norm = nn.LayerNorm(bottleneck_dim)

        self.fused_norm = nn.LayerNorm(bottleneck_dim)

        self.router_head = nn.Linear(bottleneck_dim, 1, bias=True)

        nn.init.normal_(self.router_head.weight, mean=0.0, std=0.001)
        nn.init.constant_(self.router_head.bias, 0.0)

    def forward(
        self,
        hidden_states,
        pooled_encoder_states,
        temb,
        layer_emb,
    ):
        """
        hidden_states: (B, seq_len, D_h)
        temb: (B, D_t)
        pooled_encoder_states: (B, D_p)
        layer_emb: (B, D_l)

        Context arguments are accepted but ignored when the corresponding signal is
        disabled, so the call site in ShivaDitBlockHook is identical across all arms.
        """

        token_feat = self.token_proj(hidden_states)  # (B, N, c)

        if self.use_context:
            context_feat_unnorm = None
            if "time" in self.context_signals:
                time_feat = self.temb_proj(temb).unsqueeze(1)
                context_feat_unnorm = time_feat
            if "prompt" in self.context_signals:
                prompt_feat = self.prompt_proj(pooled_encoder_states).unsqueeze(1)
                context_feat_unnorm = (
                    prompt_feat if context_feat_unnorm is None
                    else context_feat_unnorm + prompt_feat
                )
            if "layer" in self.context_signals:
                layer_feat = self.layer_proj(layer_emb).unsqueeze(1)
                context_feat_unnorm = (
                    layer_feat if context_feat_unnorm is None
                    else context_feat_unnorm + layer_feat
                )

            context_feat = self.context_norm(context_feat_unnorm)

            fused_hidden_pre_norm = token_feat + context_feat
        else:
            fused_hidden_pre_norm = token_feat

        fused_hidden = F.silu(fused_hidden_pre_norm)

        fused_hidden = self.fused_norm(fused_hidden)

        token_scores = self.router_head(fused_hidden)

        return token_scores.squeeze(-1)
