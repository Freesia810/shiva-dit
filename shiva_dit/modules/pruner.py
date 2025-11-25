import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.utils.parametrizations as parametrizations

from shiva_dit.utils.ste import ShivaTopkFunc

class ShivaPrunerExpert(torch.nn.Module):
    def __init__(self, init_Ts, init_Tk):
        super().__init__()
        self.register_buffer("Ts", torch.tensor(float(init_Ts)), persistent=False)
        self.register_buffer("Tk", torch.tensor(float(init_Tk)), persistent=False)

    def forward(self, x, logits, k):
        return ShivaTopkFunc.apply(x, logits, k, self.Ts, self.Tk)

class ShivaContextPruner(nn.Module):
    def __init__(self, img_dim, txt_dim, proj_dim, init_Ts, init_Tk):
        super().__init__()
        self.q_proj = nn.Linear(img_dim, proj_dim, bias=False)
        self.k_proj = nn.Linear(txt_dim, proj_dim, bias=False)

        parametrizations.orthogonal(self.q_proj, "weight")
        parametrizations.orthogonal(self.k_proj, "weight")

        self.register_buffer("Ts", torch.tensor(float(init_Ts)), persistent=False)
        self.register_buffer("Tk", torch.tensor(float(init_Tk)), persistent=False)

        self.scale = proj_dim ** -0.5

    def forward(self, hidden_states, encoder_hidden_states, k):
        original_dtype = hidden_states.dtype
        
        h_in = hidden_states.to(torch.float32)
        e_in = encoder_hidden_states.to(torch.float32)

        img_query = h_in.mean(dim=1) 
        q = self.q_proj(img_query).unsqueeze(1) # (B, 1, R)
        k_proj = self.k_proj(e_in)              # (B, L, R)

        q = F.normalize(q, p=2, dim=-1)
        k_proj = F.normalize(k_proj, p=2, dim=-1)
        scores = (q * k_proj).sum(dim=-1) * 10.0 

        self.last_scores = scores.detach()

        if not isinstance(k, torch.Tensor):
            k = torch.tensor(k, device=hidden_states.device, dtype=torch.float32)
        else:
            k = k.to(dtype=torch.float32)

        x_sel, x_rej, sel_idx, rej_idx = ShivaTopkFunc.apply(e_in, scores, k, self.Ts, self.Tk)

        return x_sel.to(original_dtype), x_rej.to(original_dtype), sel_idx, rej_idx