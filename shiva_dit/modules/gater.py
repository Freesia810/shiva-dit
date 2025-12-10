import torch
import torch.nn as nn
import math
from shiva_dit.utils import global_context

RATIO_STAGE = 1

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

        self.layer_emb.weight.register_hook(lambda grad: grad * 0.1)
        
        # self.has_reset_parameters = False
        self.reset_parameters()

    def reset_parameters(self):
        for m in self.time_proj.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

        nn.init.normal_(self.layer_emb.weight, mean=0.0, std=0.01)

        for m in self.mix_mlp.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

        head = self.mix_mlp[-1]
        nn.init.normal_(head.weight, mean=0.0, std=0.01)
        
        # Sigmoid^-1(x) = ln(x / (1-x))
        eps = 1e-6
        safe_ratio = max(eps, min(self.init_ratio_val, 1.0 - eps))
        bias_val = math.log(safe_ratio / (1.0 - safe_ratio))
        
        nn.init.constant_(head.bias, bias_val)

        print(f"[ShivaRatioPolicy] Reset: InitRatio={self.init_ratio_val}, HeadBias={bias_val:.4f}")

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

        if self.training and torch.rand(1) < 1e-3:
            is_main_process = not torch.distributed.is_initialized() or torch.distributed.get_rank() == 0
            
            if is_main_process and global_context.current_stage_idx >= RATIO_STAGE:
                with torch.no_grad():
                    current_t = global_context.current_timestep
                    probs = torch.sigmoid(logits)
                    
                    l_idx_flat = layer_idx_tensor.detach().flatten()
                    l_idx_str = str(l_idx_flat.cpu().tolist()[0])

                    mag_layer = l_feat.abs().mean().item()
                    mag_time_scale = scale.abs().mean().item()
                    mag_time_shift = shift.abs().mean().item()

                    print(f"\n{'='*30} [Policy Forward Monitor] {'='*30}")
                    print(f"  > Current Timestep: {current_t}")
                    print(f"  > Layer Indices:    {l_idx_str}")
                    print(f"  > Outputs Stats:")
                    print(f"      Logits:  Mean={logits.mean().item():.4f} | Std={logits.std().item():.4f} | Range=[{logits.min().item():.4f}, {logits.max().item():.4f}]")
                    print(f"      Probs:   Mean={probs.mean().item():.4f}  | Std={probs.std().item():.4f}  | Range=[{probs.min().item():.4f}, {probs.max().item():.4f}]")
                    print(f"  > Weight Analysis:")
                    print(f"      Layer Emb Mag:  {mag_layer:.4f}")
                    print(f"      Time Scale Mag: {mag_time_scale:.4f} | Shift Mag: {mag_time_shift:.4f}")
                    print(f"{'='*78}\n")

        return logits