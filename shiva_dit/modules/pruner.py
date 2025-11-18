import torch
import torch.nn as nn

from shiva_dit.utils.ste import ShivaTopkFunc


class ShivaContextPruner(nn.Module):
    """
    TODO: if the model has larger context length, we should propose a better pruning strategy
    Now we only use this module to pool the encoder states for the hidden's router
    """

    def __init__(self, threshold: int = None, ratio: float = None):
        super().__init__()
        if ratio is not None and not 0.0 < ratio <= 1.0:
            raise ValueError(f"Pruning ratio must be between (0, 1], but got {ratio}")

        self.threshold = threshold
        self.ratio = ratio

    @torch.no_grad()
    def forward(self, encoder_hidden_states: torch.Tensor):
        """
        :param encoder_hidden_states: (B, C, D)
        :return: (pruned_states, pooled_vector)
                 pruned_states: (B, c_pruned, D) 或 (B, C, D)
                 pooled_vector: (B, D)
        """
        B, C, D = encoder_hidden_states.shape

        if self.threshold is None:
            # default: mean pooling
            pooled_vector = torch.mean(encoder_hidden_states, dim=1)
            return encoder_hidden_states, pooled_vector

        if C <= self.threshold or self.ratio == 1.0:
            # default: mean pooling
            pooled_vector = torch.mean(encoder_hidden_states, dim=1)
            return encoder_hidden_states, pooled_vector

        c_pruned = int(C * self.ratio)
        c_pruned = max(c_pruned, 1)
        if c_pruned == C:
            pooled_vector = torch.mean(encoder_hidden_states, dim=1)
            return encoder_hidden_states, pooled_vector

        token_scores = torch.norm(
            encoder_hidden_states.to(torch.float32), p=2, dim=-1
        ).detach()
        topk_indices = torch.topk(token_scores, k=c_pruned, dim=-1).indices
        topk_indices_expanded = topk_indices.unsqueeze(-1).expand(-1, -1, D)
        pruned_states = torch.gather(encoder_hidden_states, 1, topk_indices_expanded)
        pooled_vector = torch.mean(pruned_states, dim=1)

        return pruned_states, pooled_vector


class ShivaPrunerExpert(torch.nn.Module):
    def __init__(self, init_Ts, init_Tk):
        super().__init__()

        self.register_buffer("Ts", torch.tensor(float(init_Ts)))
        self.register_buffer("Tk", torch.tensor(float(init_Tk)))

    def forward(self, x, logits, k):
        """
        support both training and eval
        """
        return ShivaTopkFunc.apply(x, logits, k, self.Ts, self.Tk)
