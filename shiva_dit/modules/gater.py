import torch
import torch.nn as nn

class ShivaGaterPolicy(nn.Module):
    def __init__(self, time_embed_dim: int, layer_embed_dim: int, bottleneck_dim: int):
        super().__init__()
        fused_dim = time_embed_dim + layer_embed_dim
        self.net = nn.Sequential(
            nn.Linear(fused_dim, bottleneck_dim),
            nn.SiLU(),
            nn.Linear(bottleneck_dim, 2),  # 2-way Logits: [Prune, Fuse]
        )

        nn.init.constant_(self.net[-1].bias, 0.0)
        with torch.no_grad():
            self.net[-1].bias[0] = 3.0   # Prune Logit
            self.net[-1].bias[1] = 0.0  # Fuse Logit

    def forward(self, temb: torch.Tensor, layer_emb: torch.Tensor) -> torch.Tensor:
        fused_input = torch.cat([temb, layer_emb], dim=-1)
        return self.net(fused_input)


class ShivaRatioPolicy(nn.Module):
    def __init__(
        self, 
        total_layers: int,
        ratio_bottleneck_dim: int = 32,
        **kwargs
    ):
        super().__init__()
        
        self.layer_emb = nn.Embedding(total_layers, ratio_bottleneck_dim)

        self.time_mlp = nn.Sequential(
            nn.Linear(1, ratio_bottleneck_dim),
            nn.SiLU(),
            nn.Linear(ratio_bottleneck_dim, ratio_bottleneck_dim),
            nn.SiLU(),
        )

        self.head = nn.Linear(ratio_bottleneck_dim, 1)

        nn.init.normal_(self.layer_emb.weight, std=0.01)
        nn.init.constant_(self.head.bias, 1.386) 
        nn.init.zeros_(self.head.weight)

    def forward(self, t_scalar, layer_idx_tensor):
        """
        t_scalar: (B, 1)  0.0 ~ 1.0
        layer_idx_tensor: (B,)
        """
        t_feat = self.time_mlp(t_scalar)
        l_feat = self.layer_emb(layer_idx_tensor)
        
        features = l_feat * t_feat + l_feat
        logits = self.head(features)
        
        return logits