from typing import Tuple, Union
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
    topk_scores, topk_indices = torch.topk(token_scores, k, dim=1)  # (B, K)
    idx_expanded = topk_indices.unsqueeze(-1).expand(-1, -1, x.size(-1))  # (B, K, D)
    selected_x = torch.gather(x, dim=1, index=idx_expanded)  # (B, K, D)

    return selected_x, topk_indices, topk_scores

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


def sample_dst_src_with_2D_stride(
    size: Union[int, Tuple[int, int]],
    stride: Union[int, Tuple[int, int]],
    generator: torch.Generator = None,
    rand: bool = True,
    device: torch.device = None,
    partition_idx: torch.Tensor = None,
    batch_constraint: bool = True,
    batch_size: int = 1,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Sample destination (dst) and source (src) token indices from a 2D token grid,
    by dividing the input feature map into non-overlapping blocks of given stride.

    Each stride-sized block will have exactly one dst token (randomly chosen),
    and all remaining tokens in the block are marked as src.

    size (h, w) -> h * w == N

    partition_idx: must be (B, h_patch, w_patch, 1)

    batch_constraint: whether to apply the same sampling value across the batch, default True, when partition_idx is given, this will be ignored and always be False.
    """
    with torch.no_grad():
        if isinstance(size, int):
            h = w = size
        else:
            h, w = size
        if isinstance(stride, int):
            stride_h = stride_w = stride
        else:
            stride_h, stride_w = stride

        h_stride_num, w_stride_num = h // stride_h, w // stride_w

        if partition_idx is None:
            if batch_constraint:
                if rand:
                    partition_idx = torch.randint(
                        stride_h * stride_w,
                        size=(h_stride_num, w_stride_num, 1),
                        device=device,
                        generator=generator,
                    ).expand(
                        batch_size, -1, -1, -1
                    )  # (B, h_stride_num, w_stride_num, 1)
                else:
                    partition_idx = torch.zeros(
                        h_stride_num, w_stride_num, 1, device=device, dtype=torch.int64
                    ).expand(
                        batch_size, -1, -1, -1
                    )  # (B, h_stride_num, w_stride_num, 1)
            else:
                if rand:
                    partition_idx = torch.randint(
                        stride_h * stride_w,
                        (batch_size, h_stride_num, w_stride_num, 1),
                        device=device,
                        generator=generator,
                    )  # (B, h_stride_num, w_stride_num, 1)
                else:
                    partition_idx = torch.zeros(
                        batch_size,
                        h_stride_num,
                        w_stride_num,
                        1,
                        device=device,
                        dtype=torch.int64,
                    )  # (B, h_stride_num, w_stride_num, 1)

        idx_buffer_view = torch.zeros(
            batch_size,
            h_stride_num,
            w_stride_num,
            stride_h * stride_w,
            device=device,
            dtype=torch.int64,
        )  # (B, h_stride_num, w_stride_num, stride_h * stride_w)
        idx_buffer_view.scatter_(
            dim=3,
            index=partition_idx,
            src=-torch.ones_like(partition_idx, dtype=torch.int64),
        )  # (B, h_stride_num, w_stride_num, stride_h * stride_w) dst_index = -1

        idx_buffer_view = (
            idx_buffer_view.view(batch_size, h_stride_num, w_stride_num, stride_h, stride_w)
            .transpose(2, 3)
            .reshape(batch_size, h_stride_num * stride_h, w_stride_num * stride_w)
        )  # (B, h_stride_num * stride_h, w_stride_num * stride_w)

        if (h_stride_num * stride_h) < h or (w_stride_num * stride_w) < w:
            idx_buffer = torch.zeros(batch_size, h, w, device=device, dtype=torch.int64)
            idx_buffer[:, : (h_stride_num * stride_h), : (w_stride_num * stride_w)] = (
                idx_buffer_view
            )
        else:
            idx_buffer = idx_buffer_view

        # [dst_1, dst_2, ..., src_1, src_2, ...] because -1 < 1
        partition_idx = idx_buffer.reshape(batch_size, -1, 1).argsort(dim=1)

        # Split indices into dst (front) and src (remaining)
        num_dst = h_stride_num * w_stride_num
        dst_idx = partition_idx[:, :num_dst, :]  # (B, num_dst, 1)
        src_idx = partition_idx[:, num_dst:, :]  # (B, N - num_dst, 1)

        return dst_idx, src_idx


def get_dst_src_tokens(x: torch.Tensor, dst_idx: torch.Tensor, src_idx: torch.Tensor):
    # x: (B, N, D)
    # dst_idx: (B, num_dst, 1)
    # src_idx: (B, N - num_dst, 1)
    B, N, C = x.shape
    num_dst = dst_idx.shape[1]
    dst_tokens = torch.gather(x, dim=1, index=dst_idx.expand(B, num_dst, C))
    src_tokens = torch.gather(x, dim=1, index=src_idx.expand(B, N - num_dst, C))
    return dst_tokens, src_tokens


def select_dst_src_tokens(scores: torch.Tensor, k: int):
    # scores: (B, num_src, num_dst)
    k = min(k, scores.shape[1])
    node_max, node_idx = scores.max(dim=-1, keepdim=True)  # (B, N - num_dst, 1)
    edge_idx = node_max.argsort(dim=-2, descending=True)  # (B, N - num_dst, 1)

    merge_src_idx = edge_idx[..., :k, :]  # (B, k, 1)
    unmerge_src_idx = edge_idx[..., k:, :]  # (B, N - num_dst - k, 1)
    merge_dst_idx = torch.gather(node_idx, dim=-2, index=merge_src_idx)  # (B, k, 1)

    return merge_dst_idx, merge_src_idx, unmerge_src_idx


def gather_dst_unmerge_tokens(
    x: torch.Tensor,
    dst_idx: torch.Tensor,
    src_idx: torch.Tensor,
    merge_dst_idx: torch.Tensor,
    unmerge_src_idx: torch.Tensor,
):
    # x: (B, N, D)
    # dst_idx: (B, num_dst, 1)
    # src_idx: (B, N - num_dst, 1)
    # merge_dst_idx: (B, k, 1)
    # unmerge_src_idx: (B, N - num_dst - k, 1)
    if x.dim() == 2:
        x = x.unsqueeze(0)  # (1, N, D)
    dst_tokens, src_tokens = get_dst_src_tokens(
        x, dst_idx, src_idx
    )  # (B, num_dst, D), (B, N - num_dst, D)

    B, N, C = x.shape
    num_src = src_idx.shape[1]
    k = merge_dst_idx.shape[1]

    unmerge_src_tokens = torch.gather(
        src_tokens, dim=-2, index=unmerge_src_idx.expand(B, num_src - k, C)
    )  # (B, N - num_dst - k, D)

    res = torch.cat([unmerge_src_tokens, dst_tokens], dim=1)  # (B, N - k, D)
    if B == 1:
        return res.squeeze(0)  # (N - num_dst - k, D)
    return res


def merge_dst_src_tokens(
    x: torch.Tensor,
    dst_idx: torch.Tensor,
    src_idx: torch.Tensor,
    merge_dst_idx: torch.Tensor,
    merge_src_idx: torch.Tensor,
    unmerge_src_idx: torch.Tensor,
    reduce_mode: str = "mean",
):
    # x: (B, N, D)
    # dst_idx: (B, num_dst, 1)
    # src_idx: (B, N - num_dst, 1)
    # merge_dst_idx: (B, k, 1)
    # merge_src_idx: (B, k, 1)
    # unmerge_src_idx: (B, N - num_dst - k, 1)
    dst_tokens, src_tokens = get_dst_src_tokens(
        x, dst_idx, src_idx
    )  # (B, num_dst, D), (B, N - num_dst, D)

    B, N, C = x.shape
    num_src = src_idx.shape[1]
    k = merge_dst_idx.shape[1]

    merge_src_tokens = torch.gather(
        src_tokens, dim=-2, index=merge_src_idx.expand(B, k, C)
    )  # (B, k, D)
    unmerge_src_tokens = torch.gather(
        src_tokens, dim=-2, index=unmerge_src_idx.expand(B, num_src - k, C)
    )  # (B, N - num_dst - k, D)

    if reduce_mode == 'prune':
        merge_dst_tokens = dst_tokens
    else:
        merge_dst_tokens = dst_tokens.scatter_reduce(
            dim=-2,
            index=merge_dst_idx.expand(B, k, C),
            src=merge_src_tokens,
            reduce=reduce_mode,
        )  # (B, num_dst, D)

    return torch.cat([unmerge_src_tokens, merge_dst_tokens], dim=1)  # (B, N - k, D)


def unmerge_dst_src_tokens(
    merged_x: torch.Tensor,
    dst_idx: torch.Tensor,
    src_idx: torch.Tensor,
    merge_dst_idx: torch.Tensor,
    merge_src_idx: torch.Tensor,
    unmerge_src_idx: torch.Tensor,
):
    # merged_x: (B, N - k, D)
    # dst_idx: (B, num_dst, 1)
    # src_idx: (B, N - num_dst, 1)
    # merge_dst_idx: (B, k, 1)
    # merge_src_idx: (B, k, 1)
    # unmerge_src_idx: (B, N - num_dst - k, 1)
    B, N_k, C = merged_x.shape
    num_dst = dst_idx.shape[1]
    k = merge_dst_idx.shape[1]
    N = N_k + k
    unmerge_len = N - num_dst - k

    unmerge_src_tokens, merge_dst_tokens = (
        merged_x[..., :unmerge_len, :],
        merged_x[..., unmerge_len:, :],
    )  # (B, N - num_dst - k, D), (B, num_dst, D)
    simulated_merged_src_tokens = torch.gather(
        merge_dst_tokens, dim=-2, index=merge_dst_idx.expand(B, k, C)
    )  # use dst tokens to simulate src tokens

    out = torch.zeros(
        B, N, C, device=merged_x.device, dtype=merged_x.dtype
    )  # (B, N, C)
    out.scatter_(dim=-2, index=dst_idx.expand(B, num_dst, C), src=merge_dst_tokens)
    out.scatter_(
        dim=-2,
        index=torch.gather(
            src_idx.expand(B, N - num_dst, 1), dim=1, index=unmerge_src_idx
        ).expand(B, unmerge_len, C),
        src=unmerge_src_tokens,
    )
    out.scatter_(
        dim=-2,
        index=torch.gather(
            src_idx.expand(B, N - num_dst, 1), dim=1, index=merge_src_idx
        ).expand(B, k, C),
        src=simulated_merged_src_tokens,
    )

    return out
