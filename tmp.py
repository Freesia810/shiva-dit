import torch
import torch.nn as nn
import torch.nn.functional as F

class ShivaRouter(nn.Module):
    def __init__(self, hidden_dim: int, encoder_hidden_dim: int, time_embed_dim: int, layer_embed_dim: int, bottleneck_dim: int = 64):
        super().__init__()
        self.token_proj = nn.Linear(hidden_dim, bottleneck_dim)
        self.prompt_proj = nn.Linear(encoder_hidden_dim, bottleneck_dim, bias=False)

        self.temb_proj = nn.Linear(time_embed_dim, bottleneck_dim, bias=False)
        self.layer_proj = nn.Linear(layer_embed_dim, bottleneck_dim, bias=False)
        
        self.router_head = nn.Linear(bottleneck_dim, 1)

    def forward(self, hidden_states, temb, encoder_hidden_states):
        """        
        hidden_states: (B, seq_len, D)
        temb: TODO
        encoder_hidden_states: (B, encoder_seq_len, D)
        """
        context_feat = self.temb_proj(temb).unsqueeze(1) + self.prompt_proj(encoder_hidden_states.mean(dim=1)).unsqueeze(1)
        token_feat = self.token_proj(hidden_states) # [B, N, c]
        
        fused_hidden = F.silu(token_feat + context_feat)
        token_scores = self.router_head(fused_hidden) # [B, N, 1]
        return token_scores.squeeze(-1) # [B, N]

class StrategyGater(nn.Module):
    """
    During training, we use this module to gating the experts.
    During inference, we can obtain the layer id and the timesteps offline, so we can directly use the gating logits.
    """
    def __init__(self, layer_aware_temb_dim: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(layer_aware_temb_dim, 64),
            nn.SiLU(),
            nn.Linear(64, 2)
        )
    
    def forward(self, layer_aware_temb):
        return self.net(layer_aware_temb) # [B, 2]

class FuseExpert(nn.Module):
    def __init__(self, hidden_dim: int, ffn_mult: int = 2):
        super().__init__()
        self.fusion_ffn = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim * ffn_mult),
            nn.SiLU(),
            nn.Linear(hidden_dim * ffn_mult, hidden_dim)
        )
    
    def forward(self, survivors, discarded):
        if discarded.shape[1] > 0:
            global_info = torch.mean(discarded, dim=1, keepdim=True)
            projected_info = self.fusion_ffn(global_info)
            fused_survivors = survivors + projected_info
        else:
            fused_survivors = survivors

        return fused_survivors

class ShivaDitBlockHook(BaseDelegator, BaseDiTBlockHook, nn.Module):
    def __init__(self, hidden_dim: int, encoder_hidden_dim: int, time_embed_dim: int, layer_embed_dim: int, **kwargs):
        super().__init__(**kwargs) # 继承 self.diff_ratio
        
        self.router = ShivaRouter(model_dim, time_embed_dim, encoder_hidden_dim, layer_embed_dim)
        
        # 2. MoE 门控
        self.gater = StrategyGater(time_embed_dim) # 假设 layer_aware_temb 维度与 temb 一致
        
        # 3. 专家
        self.prune_expert = nn.Identity() # Expert 0
        self.fuse_expert = FuseExpert(model_dim) # Expert 1

        # 4. 存储用于自蒸馏损失的中间值
        self.output_prune_for_loss = None
        self.output_fuse_for_loss = None

    def delegated_call(self, func, module_self, *args, **kwargs):
        """
        这是 LCaM-MoE 的核心工作流。
        'func' 是原始 DiT 块 (例如 FluxTransformerBlock) 的 forward 方法。
        """
        # --- 1. 获取所有输入上下文 ---
        sig = inspect.signature(func)
        bound = sig.bind_partial(module_self, *args, **kwargs)
        bound.apply_defaults()
        
        hidden_states = bound.arguments.get('hidden_states')
        encoder_hidden_states = bound.arguments.get('encoder_hidden_states')
        image_rotary_emb = bound.arguments.get('image_rotary_emb')
        temb = bound.arguments.get('temb') # Flux/SD3 会传入 temb
        
        # (对于 LCaM-MoE-S 共享版): 假设 temb 已经被上游修改为 "layer_aware_temb"
        
        B, N, D = hidden_states.shape
        
        # --- 2. 共享主干 (打分) ---
        token_scores = self.router(hidden_states.to(self.dtype), temb.to(self.dtype), encoder_hidden_states.to(self.dtype))
        
        # --- 3. 策略门控 (MoE 路由) ---
        gater_logits = self.gater(temb.to(self.dtype)) # [B, 2]
        gating_weights = F.softmax(gater_logits, dim=-1)
        g_prune = gating_weights[:, 0].unsqueeze(-1).unsqueeze(-1)
        g_fuse  = gating_weights[:, 1].unsqueeze(-1).unsqueeze(-1)
        
        # --- 4. 幸存者选择 (Pre-Experts) ---
        # 继承 DiffCR 的可学习比率
        kept_ratio = torch.clamp(self.diff_ratio, 0.1, 1.0) 
        k = ceil_ste(kept_ratio * N).to(torch.int32)
        k = torch.min(k, torch.tensor(N, device=hidden_states.device))
        
        # 继承 DiffCR 的 topk 和 RoPE gather 逻辑
        # self.token_scores, survivors, remaining_x, sorted_indices, self.index
        (
            self.token_scores, # [B, k] (用于梯度)
            survivors,      # [B, k, D]
            remaining_x,    # [B, N, D] (稀疏)
            sorted_indices, # [B, k] (用于 scatter_add)
            self.index       # [B, k] (用于梯度)
        ) = self._select_topk_and_remaining_tokens(hidden_states, token_scores, k, D)
        
        # 提取 discarded tokens (N-k)
        discarded_mask = torch.ones_like(hidden_states, dtype=torch.bool)
        discarded_mask.scatter_(1, sorted_indices.unsqueeze(-1).expand(-1, -1, D), False)
        discarded = hidden_states[discarded_mask].view(B, N - k, D)
        
        # [RoPE 解决方案]: 继承 DiffCR 的 RoPE gather 逻辑
        survivor_rope = self._gather_rope(image_rotary_emb, sorted_indices) # 伪代码，表示 RoPE gather

        # --- 5. 专家执行 ---
        output_prune = self.prune_expert(survivors)
        output_fuse  = self.fuse_expert(survivors, discarded)
        
        # 存储中间值用于计算 L_distill
        if self.training:
            self.output_prune_for_loss = output_prune
            self.output_fuse_for_loss = output_fuse

        # --- 6. 混合输出 ---
        # 注意: 推理时，gating_weights 应该是 [1, 0] 或 [0, 1] (通过 L_gate 损失实现)
        final_topk_states = (g_prune * output_prune) + (g_fuse * output_fuse)
        
        # --- 7. 调用原始 DiT 块 ---
        bound.arguments["hidden_states"] = final_topk_states
        bound.arguments["image_rotary_emb"] = survivor_rope
        
        # 调用 func (例如 FluxTransformerBlock.forward)
        processed_output = func(*bound.args, **bound.kwargs) 
        
        # Flux/SD3 返回元组 (encoder_hidden_states, hidden_states)
        if isinstance(processed_output, tuple):
            processed_encoder_states, processed_hidden_states = processed_output
        else:
            processed_hidden_states = processed_output

        # --- 8. Unmerge / Scatter ---
        # 继承 DiffCR 的 scatter 逻辑
        final_hidden_states = self._scatter_add(
            processed_hidden_states, remaining_x, sorted_indices
        ) # 伪代码
        
        if isinstance(processed_output, tuple):
             # 我们没有修改 encoder_hidden_states，原样返回
            return processed_encoder_states, final_hidden_states
        else:
            return final_hidden_states

    # def hook_after_module(...)
    # 此处省略，我们完全继承 DiffCR 的 FFN 梯度门控技巧
    #
    # 梯度将通过 self.token_scores 流回 self.router