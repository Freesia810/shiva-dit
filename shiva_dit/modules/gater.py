import torch
import torch.nn as nn

class ShivaGaterPolicy(nn.Module):
    def __init__(self, time_embed_dim: int, layer_embed_dim: int, bottleneck_dim: int = 64):
        super().__init__()
        fused_dim = time_embed_dim + layer_embed_dim
        self.net = nn.Sequential(
            nn.Linear(fused_dim, bottleneck_dim),
            nn.SiLU(),
            nn.Linear(bottleneck_dim, 2),  # 2-way Logits: [Prune, Fuse]
        )

    def forward(self, temb: torch.Tensor, layer_emb: torch.Tensor) -> torch.Tensor:
        fused_input = torch.cat([temb, layer_emb], dim=-1)
        return self.net(fused_input)

# class ShivaRatioPolicy(nn.Module):
#     def __init__(self, time_embed_dim: int, layer_embed_dim: int, bottleneck_dim: int = 64):
#         super().__init__()
#         self.time_norm = nn.LayerNorm(time_embed_dim)
#         self.layer_norm = nn.LayerNorm(layer_embed_dim)
#         fused_dim = time_embed_dim + layer_embed_dim
#         self.net = nn.Sequential(
            
#             nn.Linear(fused_dim, bottleneck_dim),
#             nn.SiLU(),
#             nn.Linear(bottleneck_dim, 1), # 1-way Logit
#         )
    
#     def forward(self, temb: torch.Tensor, layer_emb: torch.Tensor) -> torch.Tensor:
#         t_feat = self.time_norm(temb)
#         l_feat = self.layer_norm(layer_emb)

#         fused_input = torch.cat([t_feat, l_feat], dim=-1)
#         return self.net(fused_input)


class ShivaRatioPolicy(nn.Module):
    def __init__(
        self, 
        time_embed_dim: int, 
        layer_embed_dim: int, 
        bottleneck_dim: int = 64
    ):
        super().__init__()
        self.time_norm = nn.LayerNorm(time_embed_dim)
        self.layer_norm = nn.LayerNorm(layer_embed_dim)
        fused_dim = time_embed_dim + layer_embed_dim + 2 
        self.net = nn.Sequential(
            nn.Linear(fused_dim, bottleneck_dim),
            nn.SiLU(),
            nn.Linear(bottleneck_dim, 1),
        )

    def forward(self, temb, layer_emb, t_scalar, l_scalar):
        t_feat = self.time_norm(temb)
        l_feat = self.layer_norm(layer_emb)
        fused_input = torch.cat([t_feat, l_feat, t_scalar, l_scalar], dim=-1)
        return self.net(fused_input)