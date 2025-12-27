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

        reduction_dim = max(16, hidden_dim // 16) 
        
        self.gate_mlp = nn.Sequential(
           nn.Linear(hidden_dim, reduction_dim), # D -> D/16
           nn.SiLU(),
           nn.Linear(reduction_dim, hidden_dim), # D/16 -> D
           nn.Sigmoid()
        )

        self.reset_parameters()

    def reset_parameters(self):
        nn.init.zeros_(self.fusion_ffn[-1].weight)
        nn.init.zeros_(self.fusion_ffn[-1].bias)
        
        nn.init.normal_(self.gate_mlp[-2].weight, mean=0.0, std=0.01)
        
        nn.init.constant_(self.gate_mlp[-2].bias, -5.0)
        
        nn.init.xavier_uniform_(self.gate_mlp[0].weight)
        nn.init.zeros_(self.gate_mlp[0].bias)

        print("[ShivaFuserExpert] Initialized. Gate Weight: Normal(0.01), Gate Bias: -5.0")

    def forward(self, survivors, discarded):
        """
        survivors: (B, k, D)
        discarded: (B, N-k, D)
        """
        if discarded.shape[1] > 0:
            global_info = torch.mean(discarded, dim=1, keepdim=True)  # (B, 1, D)
            projected_info = self.fusion_ffn(global_info)  # (B, 1, D)

            # gate shape: (B, k, D)
            gate = self.gate_mlp(survivors) 
            
            fused_survivors = survivors + (gate * projected_info)
        else:
            fused_survivors = survivors
            
            if self.training:
                dummy_input = torch.zeros(
                    survivors.shape[0], 1, survivors.shape[2],
                    device=survivors.device, dtype=survivors.dtype
                )
                dummy_output = self.fusion_ffn(dummy_input)
                dummy_gate = self.gate_mlp(survivors)
                fused_survivors = fused_survivors + (dummy_output * 0.0) + (dummy_gate * 0.0)

        return fused_survivors