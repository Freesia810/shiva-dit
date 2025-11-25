import torch
import torch.nn as nn
import torch.nn.utils.parametrizations as parametrizations

from shiva_dit.utils.ste import ShivaTopkFunc


class ShivaPrunerExpert(torch.nn.Module):
    def __init__(self, init_Ts, init_Tk):
        super().__init__()

        self.register_buffer("Ts", torch.tensor(float(init_Ts)), persistent=False)
        self.register_buffer("Tk", torch.tensor(float(init_Tk)), persistent=False)

    def forward(self, x, logits, k):
        """
        support both training and eval
        """
        return ShivaTopkFunc.apply(x, logits, k, self.Ts, self.Tk)


class ShivaContextPruner(nn.Module):
    def __init__(self, img_dim, txt_dim, proj_dim, init_Ts, init_Tk):
        super().__init__()
        self.q_proj = nn.Linear(img_dim, proj_dim, bias=False)
        self.k_proj = nn.Linear(txt_dim, proj_dim, bias=False)

        parametrizations.orthogonal(self.q_proj, "weight")
        parametrizations.orthogonal(self.k_proj, "weight")

        self.ste = ShivaPrunerExpert(init_Ts, init_Tk)
        
        self.scale = proj_dim ** -0.5

    def forward(self, hidden_states, encoder_hidden_states, k):
        # 1. 备份原始精度
        original_dtype = hidden_states.dtype
        
        # 2. [关键] 强制转换为 float32 进行计算
        # 这一步能解决 90% 的 NaN 问题
        h_in = hidden_states.to(torch.float32)
        e_in = encoder_hidden_states.to(torch.float32)
        
        # 输入 NaN 检查 (可选，调试用)
        # if torch.isnan(h_in).any(): h_in = torch.nan_to_num(h_in)

        # 3. 计算 Query & Projection
        img_query = h_in.mean(dim=1) 
        q = self.q_proj(img_query).unsqueeze(1) # (B, 1, R)
        k_proj = self.k_proj(e_in)              # (B, L, R)

        # 4. 计算 Scores (带 Scaling)
        # 类似于 Attention: (Q @ K.T) / sqrt(d)
        scores = (q * k_proj).sum(dim=-1) * self.scale

        # 暂存分数 (detach 并转回 CPU 或保留在 GPU 但不需梯度)
        self.last_scores = scores.detach()

        # 6. 处理 k (类型转换)
        if not isinstance(k, torch.Tensor):
            k = torch.tensor(k, device=hidden_states.device, dtype=torch.float32)
        else:
            k = k.to(dtype=torch.float32)

        # 7. STE 选择
        x_sel, x_rej, sel_idx, rej_idx = self.ste(e_in, scores, k)

        return x_sel.to(original_dtype), x_rej.to(original_dtype), sel_idx, rej_idx
