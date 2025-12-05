import torch
import torch.nn as nn
import math

class ShivaGaterPolicy(nn.Module):
    def __init__(self, time_embed_dim, layer_embed_dim, bottleneck_dim, num_layers):
        super().__init__()
        self.layer_embedder = nn.Embedding(num_layers, layer_embed_dim)
        
        self.in_layer = nn.Linear(time_embed_dim + layer_embed_dim, bottleneck_dim)
        self.out_layer = nn.Linear(bottleneck_dim, 2)  # [Choice A, Choice B]
        self.act = nn.SiLU()
        
        self.reset_parameters()
    
    def reset_parameters(self):
        nn.init.normal_(self.layer_embedder.weight, std=0.02)
        
        nn.init.xavier_uniform_(self.in_layer.weight)
        nn.init.zeros_(self.in_layer.bias)

        nn.init.normal_(self.out_layer.weight, mean=0.0, std=0.01)
        
        nn.init.constant_(self.out_layer.bias, 0.0)
        with torch.no_grad():
            self.out_layer.bias[0] = 1.0   
            self.out_layer.bias[1] = 0.0   
        
        print(f"[ShivaGaterPolicy] Initialized. Gradient Flow: ACTIVE. Init Prob[0]: ~73%")

    def forward(self, temb, layer_idx_tensor):
        # layer_idx_tensor: (B,) -> layer_emb: (B, layer_dim)
        layer_emb = self.layer_embedder(layer_idx_tensor) 

        if temb.dim() > 2: 
            temb = temb.squeeze(1) 
            
        x = torch.cat([temb, layer_emb], dim=-1)
        x = self.in_layer(x)
        x = self.act(x)
        logits = self.out_layer(x)
        return logits


# class ShivaRatioPolicy(nn.Module):
#     def __init__(
#         self, 
#         total_layers: int,
#         time_embed_dim: int,
#         ratio_bottleneck_dim: int,
#         init_ratio: float,
#         **kwargs
#     ):
#         super().__init__()
#         self.init_ratio_val = init_ratio
        
#         # 1. Layer Identity Embedding
#         self.layer_emb = nn.Embedding(total_layers, ratio_bottleneck_dim)

#         # 2. Time Projection (Dimension Reduction)
#         self.time_proj = nn.Sequential(
#             nn.Linear(time_embed_dim, ratio_bottleneck_dim),
#             nn.SiLU(),
#         )

#         # 3. Mixing MLP (The "Brain")
#         in_dim = ratio_bottleneck_dim * 2
#         self.mix_mlp = nn.Sequential(
#             nn.Linear(in_dim, ratio_bottleneck_dim),
#             nn.SiLU(),
#             nn.Linear(ratio_bottleneck_dim, 1)  # Output scalar logit
#         )
        
#         # self.has_reset_parameters = False
#         self.reset_parameters()

#     def reset_parameters(self):
#         nn.init.xavier_uniform_(self.layer_emb.weight)


#         for m in self.modules():
#             if isinstance(m, nn.Linear):
#                 nn.init.xavier_uniform_(m.weight)
#                 if m.bias is not None:
#                     nn.init.zeros_(m.bias)

#         head = self.mix_mlp[-1]
#         nn.init.xavier_uniform_(head.weight)
        
#         eps = 1e-6
#         safe_ratio = max(eps, min(self.init_ratio_val, 1.0 - eps))
#         bias_val = math.log(safe_ratio / (1.0 - safe_ratio))
#         nn.init.constant_(head.bias, bias_val)
        
#         print(f"[ShivaRatioPolicy] Reset: InitRatio={self.init_ratio_val}, Bias={bias_val:.4f}, Head=Xavier, Layer=Xavier(Boosted)")

#     def forward(self, temb, layer_idx_tensor):
#         """
#         Args:
#             temb: [B, time_embed_dim] - Time embeddings from DiT
#             layer_idx_tensor: [B] or [1] - Layer indices
#         Returns:
#             logits: [B, 1] - Raw logits (apply sigmoid outside to get ratio)
#         """
#         # if self.has_reset_parameters is False:
#         #     self.has_reset_parameters = True
#         #     self.reset_parameters()

#         t_feat = self.time_proj(temb)      # [B, 256]
        
#         l_feat = self.layer_emb(layer_idx_tensor) # [B, 256] (Broadcasting works if layer_idx is scalar)
#         if l_feat.shape[0] != t_feat.shape[0]:
#              l_feat = l_feat.expand(t_feat.shape[0], -1)

#         combined = torch.cat([t_feat, l_feat], dim=-1) # [B, 512]
        
#         logits = self.mix_mlp(combined) # [B, 1]
#         return logits


class ShivaRatioPolicy(nn.Module):
    def __init__(
        self, 
        total_layers: int,
        time_embed_dim: int,
        ratio_bottleneck_dim: int,
        init_ratio: float,
        **kwargs
    ):
        super().__init__()
        self.init_ratio_val = init_ratio
        
        self.layer_emb = nn.Embedding(total_layers, ratio_bottleneck_dim)
        self.time_proj = nn.Sequential(
            nn.Linear(time_embed_dim, ratio_bottleneck_dim),
            nn.SiLU(),
            nn.Linear(ratio_bottleneck_dim, ratio_bottleneck_dim * 2)
        )

        in_dim = ratio_bottleneck_dim 
        self.layer_norm = nn.LayerNorm(ratio_bottleneck_dim)
        
        self.mix_mlp = nn.Sequential(
            nn.Linear(in_dim, ratio_bottleneck_dim),
            nn.SiLU(),
            nn.Linear(ratio_bottleneck_dim, 1)
        )
        
        # self.has_reset_parameters = False
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.xavier_uniform_(self.layer_emb.weight)

        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

        with torch.no_grad():
            last_linear = self.time_proj[-1] 
            last_linear.weight.data.normal_(0, 0.05)
            last_linear.bias.data.fill_(0)

        head = self.mix_mlp[-1]
        nn.init.normal_(head.weight, std=1e-5)
        
        eps = 1e-6
        safe_ratio = max(eps, min(self.init_ratio_val, 1.0 - eps))
        bias_val = math.log(safe_ratio / (1.0 - safe_ratio))
        nn.init.constant_(head.bias, bias_val)
        
        print(f"[ShivaRatioPolicy] Reset (FiLM): InitRatio={self.init_ratio_val}, Bias={bias_val:.4f}")

    def forward(self, temb, layer_idx_tensor):
        # if self.has_reset_parameters is False:
        #     self.has_reset_parameters = True
        #     self.reset_parameters()

        l_feat = self.layer_emb(layer_idx_tensor) # [B, dim]
        l_feat = self.layer_norm(l_feat)
        
        scale, shift = self.time_proj(temb).chunk(2, dim=-1)

        if l_feat.shape[0] != scale.shape[0]:
             l_feat = l_feat.expand(scale.shape[0], -1)

        combined = l_feat * (1 + scale) + shift 
        
        logits = self.mix_mlp(combined) # [B, 1]
        return logits