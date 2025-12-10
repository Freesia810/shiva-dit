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

        self.layer_emb.weight.register_hook(lambda grad: grad * 0.1)
        
        # self.has_reset_parameters = False
        self.reset_parameters()

    # def reset_parameters(self):
    #     nn.init.normal_(self.layer_emb.weight, mean=0.0, std=0.01)
    #     # nn.init.xavier_uniform_(self.layer_emb.weight)

    #     for m in self.modules():
    #         if isinstance(m, nn.Linear):
    #             nn.init.xavier_uniform_(m.weight)
    #             if m.bias is not None:
    #                 nn.init.zeros_(m.bias)

    #     with torch.no_grad():
    #         last_linear = self.time_proj[-1] 
    #         last_linear.weight.data.normal_(0, 0.001)
    #         last_linear.bias.data.fill_(0)

    #     head = self.mix_mlp[-1]
    #     nn.init.normal_(head.weight, std=1e-5)
        
    #     eps = 1e-6
    #     safe_ratio = max(eps, min(self.init_ratio_val, 1.0 - eps))
    #     bias_val = math.log(safe_ratio / (1.0 - safe_ratio))
    #     nn.init.constant_(head.bias, bias_val)
        
    #     print(f"[ShivaRatioPolicy] Reset (FiLM): InitRatio={self.init_ratio_val}, Bias={bias_val:.4f}")

    def reset_parameters(self):
        # =========================================================
        # 1. 让时间信号“炸”出来 (Time Projection)
        # =========================================================
        # 之前的错误：用了 normal_(0, 0.001)。这导致 Time Scale/Shift 接近 0。
        # 你的公式是：feat * (1 + scale) + shift。
        # 如果 scale/shift 是 0，时间就被忽略了。
        # 修正：直接用 Xavier Uniform。让初始的 scale/shift 在 +/- 0.x 波动。
        for m in self.time_proj.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight) # 让权重随机且活跃
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

        # =========================================================
        # 2. Layer Embedding (保持温和)
        # =========================================================
        # 这一部分没问题，保持小一点，防止 Layer Bias 一开始就占主导
        nn.init.normal_(self.layer_emb.weight, mean=0.0, std=0.01)

        # =========================================================
        # 3. 中间层 (Mix MLP)
        # =========================================================
        # 标准初始化，保证梯度传导通畅
        for m in self.mix_mlp.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

        # =========================================================
        # 4. 输出头 (The Head) - 决定初始 Ratio
        # =========================================================
        head = self.mix_mlp[-1]
        
        # 这里的 Weight 可以小一点，为了让初始输出主要由 Bias 决定
        # 但绝不能是 0，否则梯度回不去
        nn.init.normal_(head.weight, mean=0.0, std=0.01)
        
        # 计算 Bias：确保初始输出均值准确落在 target (0.6) 附近
        # Sigmoid^-1(x) = ln(x / (1-x))
        # 0.6 -> ln(0.6/0.4) ≈ 0.4055
        eps = 1e-6
        # 强制限制在安全区间
        safe_ratio = max(eps, min(self.init_ratio_val, 1.0 - eps))
        bias_val = math.log(safe_ratio / (1.0 - safe_ratio))
        
        nn.init.constant_(head.bias, bias_val)

        print(f"[ShivaRatioPolicy] Reset (Active-Time Mode): InitRatio={self.init_ratio_val}, HeadBias={bias_val:.4f}")
        
        # ⚠️ 绝对不要再加这段代码了！！！就是它杀死了时间分化
        # with torch.no_grad():
        #    self.time_proj[-1].weight.data.normal_(0, 0.001) <--- 删掉！

    def forward(self, temb, layer_idx_tensor):
        # if self.has_reset_parameters is False:
        #     self.has_reset_parameters = True
        #     self.reset_parameters()

        l_feat = self.layer_emb(layer_idx_tensor) # [B, dim]
        l_feat = self.layer_norm(l_feat)
        
        scale, shift = self.time_proj(temb).chunk(2, dim=-1)

        if self.training and torch.rand(1) < 1e-3:
            with torch.no_grad():
                mag_layer = l_feat.abs().mean()
                mag_time_scale = scale.abs().mean()
                mag_time_shift = shift.abs().mean()
                ratio_impact = (mag_time_scale + mag_time_shift) / (mag_layer + 1e-6)
                
                print(f"\n[Policy Monitor] LayerMag: {mag_layer:.4f} | "
                      f"TimeScale: {mag_time_scale:.4f} | TimeShift: {mag_time_shift:.4f} | "
                      f"Time/Layer Ratio: {ratio_impact:.4f}")
            
                print(f"\n[DEBUG TEMB] Shape: {temb.shape}")
                print(f"  > Temb Mean: {temb.mean().item():.4f}, Std: {temb.std().item():.4f}")
                print(f"  > Temb Min: {temb.min().item():.4f}, Max: {temb.max().item():.4f}")
                
                scale, shift = self.time_proj(temb).chunk(2, dim=-1)
                print(f"  > Proj Scale Mean: {scale.abs().mean().item():.6f}")
                print(f"  > Proj Shift Mean: {shift.abs().mean().item():.6f}")

        if l_feat.shape[0] != scale.shape[0]:
             l_feat = l_feat.expand(scale.shape[0], -1)

        combined = l_feat * (1 + scale) + shift 
        
        logits = self.mix_mlp(combined) # [B, 1]
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
        
#         time_hidden_dim = max(256, ratio_bottleneck_dim * 4)
        
#         self.time_base_mlp = nn.Sequential(
#             nn.Linear(time_embed_dim, time_hidden_dim),
#             nn.SiLU(),
#             nn.Linear(time_hidden_dim, ratio_bottleneck_dim),
#         )
        
#         self.layer_emb = nn.Embedding(total_layers, ratio_bottleneck_dim)
#         self.layer_modulator = nn.Sequential(
#             nn.Linear(ratio_bottleneck_dim, ratio_bottleneck_dim * 2) 
#         )


#         self.base_norm = nn.LayerNorm(ratio_bottleneck_dim)
        
#         in_dim = ratio_bottleneck_dim 
#         self.mix_mlp = nn.Sequential(
#             nn.Linear(in_dim, ratio_bottleneck_dim),
#             nn.SiLU(),
#             nn.Linear(ratio_bottleneck_dim, 1)
#         )
        
#         self.has_reset_parameters = False
#         self.reset_parameters()

#     def reset_parameters(self):
#         for m in self.time_base_mlp.modules():
#             if isinstance(m, nn.Linear):
#                 nn.init.xavier_uniform_(m.weight)
#                 if m.bias is not None: nn.init.zeros_(m.bias)

#         nn.init.xavier_uniform_(self.layer_emb.weight)

#         with torch.no_grad():
#             last_mod_linear = self.layer_modulator[-1]
#             last_mod_linear.weight.data.normal_(0, 0.05)
#             last_mod_linear.bias.data.fill_(0)

#         head = self.mix_mlp[-1]
#         nn.init.normal_(head.weight, std=1e-3)
        
#         eps = 1e-6
#         safe_ratio = max(eps, min(self.init_ratio_val, 1.0 - eps))
#         bias_val = math.log(safe_ratio / (1.0 - safe_ratio))
#         nn.init.constant_(head.bias, bias_val)
        
#         print(f"[ShivaRatioPolicy] Reset (Inverted FiLM/Time-Base): InitRatio={self.init_ratio_val}, Bias={bias_val:.4f}")

#     def forward(self, temb, layer_idx_tensor):
#         if self.has_reset_parameters is False:
#             self.has_reset_parameters = True
#             self.reset_parameters()

#         t_feat = self.time_base_mlp(temb)
#         t_feat = self.base_norm(t_feat)
        
#         l_emb = self.layer_emb(layer_idx_tensor)
        
#         scale, shift = self.layer_modulator(l_emb).chunk(2, dim=-1)
        
#         if t_feat.shape[0] != scale.shape[0]:
#             if scale.shape[0] == 1:
#                  scale = scale.expand(t_feat.shape[0], -1)
#                  shift = shift.expand(t_feat.shape[0], -1)
#             elif t_feat.shape[0] == 1:
#                  t_feat = t_feat.expand(scale.shape[0], -1)
        
#         combined = t_feat * (1 + scale) + shift 
        
#         logits = self.mix_mlp(combined) # [B, 1]
#         return logits