import torch
import torch.nn as nn

class ShivaGater(nn.Module):
    """
    During training, we use this module to gating the experts.
    During inference, we can obtain the layer id and the timesteps offline, so we can directly use the gating logits.
    """

    def __init__(
        self,
        time_embed_dim: int,
        layer_embed_dim: int,
    ):
        super().__init__()
        fused_dim = time_embed_dim + layer_embed_dim
        self.net = nn.Sequential(
            nn.Linear(fused_dim, 64),
            nn.SiLU(),
            nn.Linear(64, 2),  # Prune, Fuse
        )

    def forward(self, temb: torch.Tensor, layer_emb: torch.Tensor):
        """
        :param temb: (B, time_embed_dim)
        :param layer_emb: (B, layer_embed_dim)
        :return: (B, 2)
        """
        fused_input = torch.cat([temb, layer_emb], dim=-1)
        return self.net(fused_input)  # [B, 2]
