import torch
import torch.nn as nn

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

        parametrize.register_parametrization(self.q_proj, "weight", nn.utils.parametrizations.Orthogonal(self.q_proj.weight))
        parametrize.register_parametrization(self.k_proj, "weight", nn.utils.parametrizations.Orthogonal(self.k_proj.weight))

        self.ste = ShivaPrunerExpert(init_Ts, init_Tk)

    def forward(self, hidden_states, encoder_hidden_states, k):
        img_query = hidden_states.mean(dim=1) 

        scores = (self.q_proj(img_query).unsqueeze(1) * self.k_proj(encoder_hidden_states)).sum(dim=-1)

        return self.ste(encoder_hidden_states, scores, k)
