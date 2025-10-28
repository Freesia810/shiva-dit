import torch
import torch.nn.functional as F

from .utils import qr_based_pseudoinverse, conjugate_transpose_inverse


# def compute_attn_wts(x: torch.Tensor, dst: torch.Tensor) -> torch.Tensor:
#     """
#     Computes the attention weights for the input tensor x given the destination tensor dst.
#     """
#     # Normalize query and key tensors
#     # Q = F.normalize(x, p=2, dim=-1).float()  # Q: [B, N, C]
#     # K = F.normalize(dst, p=2, dim=-1).float()  # K: [B, num_dst, C]

#     Q = x
#     K = dst

#     # Compute attention scores and normalize
#     A = K @ Q.transpose(-1, -2)  # attn_weights: [B, num_dst, N]

#     _, max_indices = torch.max(A, dim=1)
#     A.zero_()
#     A.scatter_(dim=1, index=max_indices.unsqueeze(1), value=1)
#     count_per_dst = A.sum(dim=-1, keepdim=True)  # count_per_dst: [B, num_dst, 1]
#     avg_A = A / count_per_dst  # Normalize attention scores: [B, num_dst, N]

#     # return avg_A.half(), A.half().transpose(-1, -2)
#     # return avg_A, A.transpose(-1, -2)
#     return avg_A, torch.linalg.pinv(avg_A.float()).half()


def compute_attn_wts(x: torch.Tensor, dst: torch.Tensor) -> torch.Tensor:
    """
    Computes the attention weights for the input tensor x given the destination tensor dst.
    """
    # Normalize query and key tensors
    # Q = F.normalize(x, p=2, dim=-1).float()  # Q: [B, N, C]
    # K = F.normalize(dst, p=2, dim=-1).float()  # K: [B, num_dst, C]

    Q = x
    K = dst

    # Compute attention scores and normalize
    A = K @ Q.transpose(-1, -2)  # attn_weights: [B, num_dst, N]

    _, max_indices = torch.max(A, dim=1)
    A.zero_()
    A.scatter_(dim=1, index=max_indices.unsqueeze(1), value=1)
    count_per_dst = A.sum(dim=-1, keepdim=True)  # count_per_dst: [B, num_dst, 1]
    avg_A = A / count_per_dst  # Normalize attention scores: [B, num_dst, N]

    # return avg_A.half(), A.half().transpose(-1, -2)
    return avg_A, A.transpose(-1, -2)


def locally_compute_attn_wts(x: torch.Tensor, dst_idx: torch.Tensor) -> torch.Tensor:
    """
    Computes the attention weights for the input tensor x given the destination tensor dst.
    """
    tile_size = 4
    num_tiles = x.shape[1] // tile_size**2

    B, N, C = x.shape
    
    def fold_with_indices(x, num_tiles):
        H = W = int(N**0.5)

        num_tiles_per_side = int(num_tiles**0.5)
        tile_side_len = H // num_tiles_per_side

        idx = torch.arange(N, device=x.device, dtype=torch.long)
        idx = idx.reshape(1, H, W, 1)

        tile_idx = torch.as_strided(
            idx,
            (1, num_tiles_per_side, num_tiles_per_side, tile_side_len, tile_side_len, 1),
            (N * 1, tile_side_len * W * 1, tile_side_len * 1, W * 1, 1, 1),
        )

        flatten_tile_idx = tile_idx.reshape(1, N, 1).expand(B, -1, C)

        flatten_tile_x = torch.gather(x, 1, flatten_tile_idx)
        tile_x = flatten_tile_x.reshape(B, num_tiles, -1, C)

        return tile_x, flatten_tile_idx


    def unfold_with_indices(x_prime, flatten_tile_idx):
        B, N, C = flatten_tile_idx.shape

        x_prime = x_prime.reshape(B, -1, C)

        restored_x = torch.zeros_like(flatten_tile_idx, dtype=x_prime.dtype)

        restored_x = restored_x.scatter(1, flatten_tile_idx, x_prime)

        return restored_x
    
    def unmerge(x: torch.Tensor) -> torch.Tensor:
        num_tiles = A_inv.shape[1]
        x = x.reshape(B, num_tiles, -1, C)
        res = A_inv @ x

        unfold_x = unfold_with_indices(res, flatten_idx)
        return unfold_x
    
    def merge(x: torch.Tensor) -> torch.Tensor:
        x_reshaped, _ = fold_with_indices(x, num_tiles)
        x_merged = avg_A @ x_reshaped
        return x_merged.reshape(B, -1, C)


    x_reshaped, flatten_idx = fold_with_indices(x, num_tiles)
    dst = torch.gather(x_reshaped, 2, dst_idx.unsqueeze(-1).repeat(1, 1, 1, C))

    Q = x_reshaped
    K = dst

    # Compute attention scores and normalize
    A = K @ Q.transpose(-1, -2)  # attn_weights: [B, num_dst, N]

    _, max_indices = torch.max(A, dim=2)
    A.zero_()
    A.scatter_(dim=-2, index=max_indices.unsqueeze(2), value=1)
    count_per_dst = A.sum(dim=-1, keepdim=True)  # count_per_dst: [B, num_dst, 1]
    avg_A = A / count_per_dst  # Normalize attention scores: [B, num_dst, N]]
    A_inv= A.transpose(-1, -2)

    return merge, unmerge
