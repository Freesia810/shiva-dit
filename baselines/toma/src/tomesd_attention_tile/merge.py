import torch
from typing import Tuple, Callable
from tomesd_attention_tile.merge_methods import locally_compute_attn_wts
from tomesd_attention_tile.facility_location import patch_fast_batched_facility_location
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
    k = x.shape[1] // 16
    with torch.no_grad():

        num_dst = (w // sx) * (h // sy) if dst_selection == "original" else N - r

        def select_destination():
            dst_idx = patch_fast_batched_facility_location(x, num_dst, k)
            return dst_idx

        if_recompute_attn = unet_scheduler.step()

        if if_recompute_attn:
            dst_idx = unet_scheduler.get_dst_idx(select_destination)
            merge_func, unmerge_func = unet_scheduler.get_A(locally_compute_attn_wts, x, dst_idx)
        else:
            merge_func, unmerge_func = unet_scheduler.get_A(do_nothing, x, None)

    def merge(x: torch.Tensor) -> torch.Tensor:
        a = merge_func(x)
        # print(a.shape)
        return a

    def unmerge(x: torch.Tensor) -> torch.Tensor:
        return unmerge_func(x)

    return merge, unmerge
