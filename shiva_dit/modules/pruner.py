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

    def forward(self, hidden_states, encoder_hidden_states, k):
        original_dtype = hidden_states.dtype
        
        hidden_states_f32 = hidden_states.to(torch.float32)
        encoder_hidden_states_f32 = encoder_hidden_states.to(torch.float32)

        if torch.isnan(hidden_states_f32).any():
            print(f"[Warning] ContextPruner: Input hidden_states contains NaN!")
            hidden_states_f32 = torch.nan_to_num(hidden_states_f32)
            
        # 2. 计算 Query (Mean Pooling)
        img_query = hidden_states_f32.mean(dim=1) 

        # 3. 投影
        q_vec = self.q_proj(img_query).unsqueeze(1) 
        k_vec = self.k_proj(encoder_hidden_states_f32)
        
        # 4. 计算分数 (增加数值稳定性保护)
        # 点积可能会很大，除以 sqrt(dim) 是标准的 Attention 操作，有助于稳定数值
        scale_factor = (self.q_proj.weight.shape[0]) ** -0.5 # 1 / sqrt(proj_dim)
        scores = (q_vec * k_vec).sum(dim=-1) * scale_factor
        
        if torch.isnan(scores).any() or torch.isinf(scores).any():
            print(f"[Warning] ContextPruner: Scores contain NaN/Inf! Handling...")
            scores = torch.nan_to_num(scores, nan=0.0, posinf=1e4, neginf=-1e4)

        self.last_scores = scores 

        if not isinstance(k, torch.Tensor):
            k = torch.tensor(k, device=hidden_states.device, dtype=torch.float32)
        else:
            k = k.to(dtype=torch.float32)

        # 6. STE 选择
        x_sel, _, sel_idx, _ = self.ste(encoder_hidden_states_f32, scores, k)
        
        return x_sel.to(original_dtype), sel_idx
