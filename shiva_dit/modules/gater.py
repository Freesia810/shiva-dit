import torch
import torch.nn as nn
import math

class ShivaGaterPolicy(nn.Module):
    def __init__(self, time_embed_dim, layer_embed_dim, bottleneck_dim, num_layers):
        super().__init__()
        self.layer_embedder = nn.Embedding(num_layers, layer_embed_dim)
        
        self.in_layer = nn.Linear(time_embed_dim + layer_embed_dim, bottleneck_dim)
        self.out_layer = nn.Linear(bottleneck_dim, 2)  # [Keep, Fuse]
        self.act = nn.SiLU()

        nn.init.constant_(self.out_layer.bias, 0.0)
        with torch.no_grad():
            self.out_layer.bias[0] = 3.0   # Prune Logit
            self.out_layer.bias[1] = 0.0  # Fuse Logit

    def forward(self, temb, layer_idx_tensor):
        # temb: (B, D)
        # layer_idx: (B,)
        
        layer_emb = self.layer_embedder(layer_idx_tensor) # (B, layer_dim)

        x = torch.cat([temb, layer_emb], dim=-1)
        x = self.in_layer(x)
        x = self.act(x)
        logits = self.out_layer(x)
        return logits


class ShivaRatioPolicy(nn.Module):
    def __init__(
        self, 
        total_layers: int,
        ratio_bottleneck_dim: int = 32,
        init_ratio: float = 0.8,
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
        eps = 1e-6
        safe_ratio = max(eps, min(init_ratio, 1.0 - eps))
        bias_val = math.log(safe_ratio / (1.0 - safe_ratio))
        nn.init.constant_(self.head.bias, bias_val)
        print(f"[ShivaRatioPolicy] Initialized bias to {bias_val:.4f} (Sigmoid -> {safe_ratio:.4f})")
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


# class ShivaRatioPolicy(nn.Module):
#     def __init__(
#         self, 
#         total_layers: int,
#         ratio_bottleneck_dim: int = 32,
#         init_ratio: float = 0.8,
#         **kwargs
#     ):
#         super().__init__()
        
#         self.layer_emb = nn.Embedding(total_layers, ratio_bottleneck_dim)

#         self.time_mlp = nn.Sequential(
#             nn.Linear(1, ratio_bottleneck_dim),
#             nn.SiLU(),
#             nn.Linear(ratio_bottleneck_dim, ratio_bottleneck_dim * 2),
#         )

#         self.head = nn.Linear(ratio_bottleneck_dim, 1)

#         nn.init.normal_(self.layer_emb.weight, std=0.01)
#         eps = 1e-6
#         safe_ratio = max(eps, min(init_ratio, 1.0 - eps))
#         bias_val = math.log(safe_ratio / (1.0 - safe_ratio))
#         nn.init.constant_(self.head.bias, bias_val)
#         print(f"[ShivaRatioPolicy] Initialized bias to {bias_val:.4f} (Sigmoid -> {safe_ratio:.4f})")
#         nn.init.zeros_(self.head.weight)

#     def forward(self, t_scalar, layer_idx_tensor):
#         """
#         t_scalar: (B, 1)  0.0 ~ 1.0
#         layer_idx_tensor: (B,)
#         """
#         time_out = self.time_mlp(t_scalar)
#         scale, shift = time_out.chunk(2, dim=-1)
#         l_feat = self.layer_emb(layer_idx_tensor)
#         features = l_feat * (1 + scale) + shift
#         logits = self.head(features)
        
#         return logits