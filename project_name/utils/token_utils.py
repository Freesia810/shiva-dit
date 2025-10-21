import torch


def select_topk_tokens(x: torch.Tensor, token_scores: torch.Tensor, k: int):
    """
    Select top-k tokens based on token scores using torch.topk + gather (highly efficient).

    Args:
        x (torch.Tensor): Input tensor of shape (B, S, D)
        token_scores (torch.Tensor): Token scores of shape (B, S)
        k (int): Number of tokens to select.

    Returns:
        selected_x (torch.Tensor): Selected tokens of shape (B, K, D)
        topk_indices (torch.Tensor): Indices of selected tokens of shape (B, K)
    """
    _, topk_indices = torch.topk(token_scores, k, dim=1)  # (B, K)
    idx_expanded = topk_indices.unsqueeze(-1).expand(-1, -1, x.size(-1))  # (B, K, D)
    selected_x = torch.gather(x, dim=1, index=idx_expanded) # (B, K, D)

    return selected_x, topk_indices


def reconstruct_tokens_by_topk(
    x: torch.Tensor,
    selected_tokens: torch.Tensor,
    topk_indices: torch.Tensor,
    zero_out_remain: bool = False,
):
    """
    Reconstruct full sequence by placing selected tokens back to their original positions.

    Args:
        x (torch.Tensor): Original input tensor (B, N, D)
        selected_tokens (torch.Tensor): Selected tokens (B, k, D)
        topk_indices (torch.Tensor): Indices of selected tokens (B, k)
        zero_out_remain (bool): Whether to zero out unselected tokens. Default False.

    Returns:
        reconstructed (torch.Tensor): (B, N, D)
    """
    reconstructed = torch.zeros_like(x) if zero_out_remain else x.clone()
    reconstructed.scatter_(
        1, topk_indices.unsqueeze(-1).expand(-1, -1, x.size(-1)), selected_tokens
    )
    return reconstructed
