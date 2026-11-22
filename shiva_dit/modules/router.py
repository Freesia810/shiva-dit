import torch.nn as nn
import torch.nn.functional as F

class ShivaRouter(nn.Module):
    def __init__(
        self,
        hidden_dim: int,
        prompt_embed_dim: int,
        layer_embed_dim: int,
        bottleneck_dim: int,
        time_embed_dim: int = None,
    ):
        super().__init__()

        if time_embed_dim is None:
            time_embed_dim = hidden_dim

        self.token_proj = nn.Linear(hidden_dim, bottleneck_dim)
        self.prompt_proj = nn.Linear(prompt_embed_dim, bottleneck_dim, bias=False)
        self.temb_proj = nn.Linear(time_embed_dim, bottleneck_dim, bias=False)
        self.layer_proj = nn.Linear(layer_embed_dim, bottleneck_dim, bias=False)

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
        """

        time_feat = self.temb_proj(temb).unsqueeze(1)
        prompt_feat = self.prompt_proj(pooled_encoder_states).unsqueeze(1)
        layer_feat = self.layer_proj(layer_emb).unsqueeze(1)
        context_feat_unnorm = time_feat + prompt_feat + layer_feat
        context_feat = self.context_norm(context_feat_unnorm)

        token_feat = self.token_proj(hidden_states)  # (B, N, c)

        fused_hidden_pre_norm = token_feat + context_feat

        fused_hidden = F.silu(fused_hidden_pre_norm)

        fused_hidden = self.fused_norm(fused_hidden)

        token_scores = self.router_head(fused_hidden)

        return token_scores.squeeze(-1)