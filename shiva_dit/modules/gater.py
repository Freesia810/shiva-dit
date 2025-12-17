import torch
import torch.nn as nn
import torch.distributed as dist
import math
from shiva_dit.utils import global_context

RATIO_TIME_STAGE = 1
RATIO_LAYER_STAGE = 2

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
        self.layer_mlp = nn.Sequential(
            nn.Linear(ratio_bottleneck_dim, ratio_bottleneck_dim),
            nn.SiLU(),
            nn.Linear(ratio_bottleneck_dim, 1)
        )

        self.time_mlp = nn.Sequential(
            nn.Linear(time_embed_dim, ratio_bottleneck_dim),
            nn.SiLU(),
            nn.Linear(ratio_bottleneck_dim, 1)
        )

        self.register_buffer("global_bias", torch.zeros(1))

        # self.has_reset = False
        self.reset_parameters()

        self.grad_scale = 1.0 / float(total_layers)
        
        def scale_grad_hook(grad):
            return grad * self.grad_scale

        print(f"[ShivaRatioPolicy] Registering gradient scaler (1/{total_layers}) for Time MLP")
        for param in self.time_mlp.parameters():
            if param.requires_grad:
                param.register_hook(scale_grad_hook)

    def reset_parameters(self):
        nn.init.normal_(self.layer_emb.weight, std=0.02)
        for m in self.layer_mlp.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, a=0.2)
                nn.init.zeros_(m.bias)
        
        for m in self.time_mlp.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, a=0.2)
                nn.init.zeros_(m.bias)
        
        nn.init.zeros_(self.time_mlp[-1].weight)
        nn.init.zeros_(self.time_mlp[-1].bias)

        eps = 1e-6
        safe_ratio = max(eps, min(self.init_ratio_val, 1.0 - eps))
        bias_val = math.log(safe_ratio / (1.0 - safe_ratio))
        nn.init.constant_(self.global_bias, bias_val)
        
        if not dist.is_initialized() or dist.get_rank() == 0:
            print(f"[ShivaRatioPolicy] Initialized (Additive). Bias={bias_val:.4f} (Target={self.init_ratio_val})")

    def forward(self, temb, layer_idx_tensor):
        # if not self.has_reset:
        #     self.reset_parameters()
        #     self.has_reset = True

        l_feat = self.layer_emb(layer_idx_tensor)
        logits_l = self.layer_mlp(l_feat)

        if temb.dim() == 3:
            temb = temb.squeeze(1)
        logits_t = self.time_mlp(temb)

        is_time_stage = (
            self.training
            and getattr(global_context, "current_stage_idx", -1)
            == RATIO_TIME_STAGE
        )

        is_layer_stage = (
            self.training
            and getattr(global_context, "current_stage_idx", -1)
            == RATIO_LAYER_STAGE
        )

        if is_time_stage:
            final_logits = logits_l + logits_t + self.global_bias
            noise_scale = 1.0 
            noise = torch.randn_like(final_logits) * noise_scale
            final_logits = final_logits + noise
        elif is_layer_stage:
            logits_t = logits_t.detach() + logits_t * 0.0
            final_logits = logits_l + logits_t + self.global_bias
        else:
            final_logits = logits_l + logits_t + self.global_bias
        
        if self.training:
            clamp_val = 6.0
            final_logits = torch.clamp(final_logits, min=-clamp_val, max=clamp_val)
            
            # final_logits = clamp_val * torch.tanh(final_logits / clamp_val)
        
        if self.training and torch.rand(1).item() < 1e-3 and (is_time_stage or is_layer_stage):
            with torch.no_grad():
                is_main_process = not dist.is_initialized() or dist.get_rank() == 0
                if is_main_process:
                    probs = torch.sigmoid(final_logits)
                    current_t = global_context.current_timestep
                    
                    l_mean, l_std = logits_l.mean().item(), logits_l.std().item()
                    t_mean, t_std = logits_t.mean().item(), logits_t.std().item()
                    bias_val = self.global_bias.item()
                    
                    final_mean = final_logits.mean().item()
                    prob_mean = probs.mean().item()
                    prob_std = probs.std().item()
                    prob_min, prob_max = probs.min().item(), probs.max().item()
                    
                    unique_layers = layer_idx_tensor.unique().tolist()
                    unique_layers.sort()
                    if len(unique_layers) > 5:
                        layers_str = f"[{min(unique_layers)} ... {max(unique_layers)}] (Count: {len(unique_layers)})"
                    else:
                        layers_str = str(unique_layers)

                    print(f"\n{'='*35} [Policy Monitor] {'='*35}")
                    print(f"  > Context:")
                    print(f"      Timestep (Global): {current_t}")
                    print(f"      Layers in Batch:   {layers_str}")
                    print(f"      [Logit Clamped]:   Yes (+/- 6.0)")
                    
                    print(f"  > Components (Logits):")
                    print(f"      [L] Layer Branch:  Mean={l_mean:+.4f} | Std={l_std:.4f}")
                    print(f"      [T] Time Branch:   Mean={t_mean:+.4f} | Std={t_std:.4f}")
                    print(f"      [B] Global Bias:   Val ={bias_val:+.4f} (Base Ratio ≈ {torch.sigmoid(self.global_bias).item():.3f})")
                    
                    print(f"  > Output Analysis:")
                    print(f"      Final Logits:      Mean={final_mean:+.4f}")
                    print(f"      Final Probs:       Mean={prob_mean:.4f} | Std={prob_std:.4f}")
                    print(f"      Probs Range:       [{prob_min:.4f}, {prob_max:.4f}]")
                    
                    if prob_mean < 0.05 or prob_mean > 0.95:
                        print(f"      ⚠️  WARNING: Ratio Saturation! (Too low/high)")
                    
                    print(f"{'='*96}\n")

        return final_logits