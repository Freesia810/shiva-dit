import torch
import torch.nn as nn

class ShivaFuserExpert(nn.Module):
    def __init__(self, hidden_dim: int, ffn_mult: int = 2, dropout: float = 0.1):
        super().__init__()
        self.fusion_ffn = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim * ffn_mult),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim * ffn_mult, hidden_dim),
        )

        self.gate_mlp = nn.Sequential(
           nn.Linear(hidden_dim, hidden_dim),
           nn.SiLU(),
           nn.Linear(hidden_dim, hidden_dim),
           nn.Sigmoid()
        )

        nn.init.zeros_(self.fusion_ffn[-1].weight)
        nn.init.zeros_(self.fusion_ffn[-1].bias)
        nn.init.zeros_(self.gate_mlp[-2].weight)
        nn.init.constant_(self.gate_mlp[-2].bias, -5.0)

    def forward(self, survivors, discarded):
        """
        survivors: (B, k, D)
        discarded: (B, N-k, D)
        """
        if discarded.shape[1] > 0:
            global_info = torch.mean(discarded, dim=1, keepdim=True)  # (B, 1, D)
            projected_info = self.fusion_ffn(global_info)  # (B, 1, D)

            gate = self.gate_mlp(survivors) # (B, k, D)
            fused_survivors = survivors + (gate * projected_info)
        else:
            fused_survivors = survivors
            if self.training:
                dummy_output = self.fusion_ffn(
                    torch.zeros(
                        survivors.shape[0],
                        1,
                        survivors.shape[2],
                        device=survivors.device,
                        dtype=survivors.dtype,
                    )
                )
                fused_survivors = fused_survivors + (dummy_output * 0.0)

        return fused_survivors
