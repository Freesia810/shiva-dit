import torch
from typing import Tuple, Callable
from .merge_methods import compute_attn_wts
from tomesd_attention_tile.facility_location import (
    original,
    batched_facility_location,
    tile_wise_batched_facility,
)
from tomesd_attention_tile.utils import do_nothing, mps_gather_workaround


def bipartite_soft_matching_random2d(
    x: torch.Tensor,
    w: int,
    h: int,
    sx: int,
    sy: int,
    r: int,
    no_rand: bool = False,
    generator: torch.Generator = None,
    dst_selection: str = "original",
    k: int = 16,
    merge_method: str = "original",
    unet_scheduler=None,
) -> Tuple[Callable, Callable]:
    """
    Partitions the tokens into src and dst and merges r tokens from src to dst.
    Dst tokens are partitioned by choosing one randomly in each (sx, sy) region.

    Args:
     - x [B, N, C]: x to use for similarity
     - w: image width in tokens
     - h: image height in tokens
     - sx: stride in the x dimension for dst, must divide w
     - sy: stride in the y dimension for dst, must divide h
     - r: number of tokens to remove (by merging)
     - no_rand: if true, disable randomness (use top left corner only)
     - rand_seed: if no_rand is false, and if not None, sets random seed.
    """
    B, N, C = x.shape

    if r <= 0:
        return do_nothing, do_nothing

    gather = mps_gather_workaround if x.device.type == "mps" else torch.gather

    with torch.no_grad():

        num_dst = (w // sx) * (h // sy) if dst_selection == "original" else N - r

        def select_destination():
            if dst_selection == "facility":
                dst_idx = batched_facility_location(x, num_dst)
            elif dst_selection == "tile_wise_facility":
                dst_idx = tile_wise_batched_facility(x, num_dst, k)
            elif dst_selection == "original":
                return original(x, w, h, sx, sy, no_rand, generator)
            elif dst_selection == "random":
                dst_idx = torch.randperm(N).unsqueeze(0).repeat(B, 1).to(x.device)
            else:
                raise ValueError(f"Unknown dst_selection: {dst_selection}")

            return dst_idx[:, :num_dst].unsqueeze(2)

        if_recompute_attn = unet_scheduler.step()

        if if_recompute_attn:
            dst_idx = unet_scheduler.get_dst_idx(select_destination)
            dst = gather(x, dim=1, index=dst_idx.expand(B, num_dst, C))
            A, A_inv = unet_scheduler.get_A(compute_attn_wts, x, dst)
        else:
            A, A_inv = unet_scheduler.get_A(do_nothing, x, None)

    def merge(x: torch.Tensor) -> torch.Tensor:
        return torch.bmm(A, x)

    def unmerge(x: torch.Tensor) -> torch.Tensor:
        return torch.bmm(A_inv, x)

    return merge, unmerge
