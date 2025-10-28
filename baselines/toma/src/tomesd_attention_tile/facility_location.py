import torch
import os


def original(
    metric: torch.Tensor, w: int, h: int, sx: int, sy: int, no_rand: bool = False, generator: torch.Generator = None
) -> tuple[torch.Tensor, torch.Tensor]:
    return None
#     """
#     Randomly assigns tokens to be source (src) or destination (dst) based on the given parameters.

#     Args:
#         metric (torch.Tensor): The input tensor.
#         w (int): The width of the image.
#         h (int): The height of the image.
#         sx (int): The kernel width.
#         sy (int): The kernel height.
#         no_rand (bool, optional): If True, disables random assignment of tokens. Defaults to False.
#         generator (torch.Generator, optional): The random number generator. Defaults to None.

#     Returns:
#         Tuple[torch.Tensor, torch.Tensor]: A tuple containing the source indices (a_idx), destination indices (b_idx), and the number of destination tokens (num_dst).
#     """
#     hsy, wsx = h // sy, w // sx

#     # For each sy by sx kernel, randomly assign one token to be dst and the rest src
#     if no_rand:
#         rand_idx = torch.zeros(hsy, wsx, 1, device=metric.device, dtype=torch.int64)
#     else:
#         rand_idx = torch.randint(sy * sx, size=(hsy, wsx, 1), device=generator.device, generator=generator).to(
#             metric.device
#         )

#     # The image might not divide sx and sy, so we need to work on a view of the top left if the idx buffer instead
#     idx_buffer_view = torch.zeros(hsy, wsx, sy * sx, device=metric.device, dtype=torch.int64)
#     idx_buffer_view.scatter_(dim=2, index=rand_idx, src=-torch.ones_like(rand_idx, dtype=rand_idx.dtype))
#     idx_buffer_view = idx_buffer_view.view(hsy, wsx, sy, sx).transpose(1, 2).reshape(hsy * sy, wsx * sx)

#     # Image is not divisible by sx or sy so we need to move it into a new buffer
#     if (hsy * sy) < h or (wsx * sx) < w:
#         idx_buffer = torch.zeros(h, w, device=metric.device, dtype=torch.int64)
#         idx_buffer[: (hsy * sy), : (wsx * sx)] = idx_buffer_view
#     else:
#         idx_buffer = idx_buffer_view

#     # We set dst tokens to be -1 and src to be 0, so an argsort gives us dst|src indices
#     rand_idx = idx_buffer.reshape(1, -1, 1).argsort(dim=1)

#     # We're finished with these
#     del idx_buffer, idx_buffer_view

#     # rand_idx is currently dst|src, so split them
#     num_dst = hsy * wsx
#     a_idx = rand_idx[:, num_dst:, :]  # src
#     b_idx = rand_idx[:, :num_dst, :]  # dst

#     return a_idx, b_idx


def batched_facility_location(x, r, method, sigma):
    """
    Find the top r representatives (for dst) for the batch using the facility location algorithm
    and save the gain at each step to a file.

    Args:
        x (torch.Tensor): Input tensor of shape (B, N, C), where B is the batch size,
            N is the number of tokens, and C is the latent dimension.
        r (int): The number of representatives to select.

    Returns:
        torch.Tensor: Tensor of shape (B, r) containing the indices of the selected representatives.

    Raises:
        AssertionError: If r is not within the range (0, N].
    """
    B, N, C = x.shape
    assert 0 < r <= N, "r should be within the range (0, N]"
    device = x.device
    # Initialize similarity matrix based on method
    if method == "Cosin":
        similarity_matrix = x @ x.transpose(-1, -2)
    elif method == "RBF":
        X_norm = (x**2).sum(dim=-1, keepdim=True)
        distance_matrix = X_norm + X_norm.transpose(-1, -2) - 2 * (x @ x.transpose(-1, -2))
        sigma = sigma
        similarity_matrix = torch.exp(-distance_matrix / (2 * sigma**2))

    # Sum similarity matrix rows and initialize first representative
    row_sums = torch.sum(similarity_matrix, dim=2)
    init_v = torch.argmax(row_sums, dim=1)

    # [IMPROVEMENTS@2024.9.19]
    # max_sim = similarity_matrix[torch.arange(B), init_v]
    max_sim = torch.gather(similarity_matrix, 1, init_v.unsqueeze(-1).unsqueeze(-1).expand(-1, 1, N)).squeeze(1)

    # Initialize representatives tensor
    representatives = torch.zeros(B, r, dtype=torch.long, device=similarity_matrix.device)
    representatives[:, 0] = init_v  # Set initial representatives

    # Loop to find the remaining 1 representative using facility location
    for i in range(1, r):  # Here we loop only once because we are selecting 1 more representative
        # Expand max_sim to match dimensions for broadcasting
        expanded_max_sim = max_sim.unsqueeze(1).expand(B, N, N)

        # Compute differences, ensuring non-negative values with ReLU
        differences = torch.relu(similarity_matrix - expanded_max_sim)

        row_sums = torch.sum(differences, dim=2)

        # Find the index of the maximum sum in each batch
        next_v = torch.argmax(row_sums, dim=1)

        # Update representatives and maximum similarity vector
        representatives[:, i] = next_v

        max_sim = torch.max(max_sim, similarity_matrix[torch.arange(B, device=device), next_v])

    # i = i + 1
    # remaining_r = r - i

    # # Create a mask for selected indices
    # if remaining_r > 0:
    #     # Generate random values for all tokens
    #     rand_vals = torch.rand(B, N, device=device)

    #     # Set random values of already selected indices to +inf
    #     rand_vals.scatter_(1, representatives[:, :i], float("inf"))

    #     # Select tokens with the smallest random values
    #     _, random_indices = torch.topk(rand_vals, k=remaining_r, largest=False, dim=1)

    #     # Update representatives with the randomly selected indices
    #     representatives[:, i:r] = random_indices

    return representatives



def tile_wise_batched_facility(x, r, num_tiles):
    x = x[0].unsqueeze(0)  # modify for batch 1
    B, HW, C = x.shape
    H = W = int(HW**0.5)

    num_tiles_per_side = int(num_tiles**0.5)
    tile_side_len = H // num_tiles_per_side
    # CHANGES made here
    r_per_tile = r // num_tiles

    indices = torch.arange(HW, device=x.device).reshape(1, H, W, 1)
    patch_indices = torch.as_strided(
        indices,
        (1, num_tiles_per_side, num_tiles_per_side, tile_side_len, tile_side_len, 1),
        (HW * 1, tile_side_len * W * 1, tile_side_len * 1, W * 1, 1, 1),
    )

    x_reshaped = torch.as_strided(
        x,
        (1, num_tiles_per_side, num_tiles_per_side, tile_side_len, tile_side_len, C),
        (HW * C, tile_side_len * H * C, tile_side_len * C, H * C, C, 1),
    )
    x_reshaped = x_reshaped.reshape(-1, tile_side_len**2, C)

    stacked_dst = batched_facility_location(x_reshaped, r_per_tile, "Cosin", 0)

    stacked_dst_indices = torch.gather(patch_indices.reshape(-1, tile_side_len**2), -1, stacked_dst)

    return stacked_dst_indices.reshape(B, -1).expand(2, -1)


def patch_fast_batched_facility_location(x, r, num_patches, method = 'Cosin', sigma = 1, return_local = False):
    """
    Divide the original tokens into regional patches and perform batched facility location algorithm on each chunk.

    Args:
        x (torch.Tensor): Input tensor of shape (B, H*W, C)
        r (int): The number of representatives to select
        num_patches (int): The number of patches to split the tensor into。

    Returns:
        None
    """
    
    B, HW, C = x.shape
    assert 0 < r <= HW, "r should be within the range (0, N]"
    x = x[0].unsqueeze(0) # modify for batch 1

    H = W = int(HW**0.5)  # Calculate H and W assuming a square image 
    r_per_patch = r // num_patches # Number of representatives per patch ####WENBO DID IT BECAUSE HE IS STUPID

    # Reshape the tensor to B, H, W, C
    x = x.view(1, H, W, C)
    # Calculate size of each patch
    patch_side_len = H // int(num_patches**0.5) 

    # Create indices offset lookup table
    indices = torch.arange(H * W, device=x.device).view(H, W)
    patch_indices = indices.unfold(0, patch_side_len, patch_side_len).unfold(1, patch_side_len, patch_side_len)
    patch_indices = patch_indices.contiguous().view(-1, patch_side_len**2)

    # create actual patches for input tensor
    # unfold along height and width
    x = x.permute(0, 3, 1, 2)
    patches = x.unfold(2, patch_side_len, patch_side_len).unfold(3, patch_side_len, patch_side_len)
    # Reshape to combine patches into the batch dimension
    stacked_patches = patches.contiguous().view(1, C, -1, patch_side_len * patch_side_len).permute(0, 2, 3, 1)

    # batched facility location algorithm
    stacked_representatives = batched_facility_location(stacked_patches.reshape(num_patches, patch_side_len**2, C), r_per_patch, method, sigma).reshape(1, num_patches, r_per_patch)
    representatives = torch.gather(patch_indices.unsqueeze(0).repeat(B, 1, 1), -1, stacked_representatives)
    stacked_representatives = stacked_representatives.repeat(B, 1, 1)
    return stacked_representatives 
