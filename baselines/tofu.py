import torch
import torch.nn as nn
from typing import Literal, Optional, Tuple, Union
from baselines.tome_sd import ToMeSDDitBlockHook
from shiva_dit.hook import HookManager
from shiva_dit.utils import (
    global_context,
    merge_dst_src_tokens,
    get_pipeline,
    process_quick_start,
)

class ToFuDitBlockHook(ToMeSDDitBlockHook):
    def __init__(self, 
        stride: Union[int, Tuple[int, int]]=(2, 2), 
        merged_ratio: float = 0.2,
        merge_config=None, 
        prune_layer_nums: int = 10,
        prune_single_layer_nums: Optional[int] = None,
        generator: torch.Generator = None, 
        **kwargs
    ):
        super().__init__(stride=stride, merged_ratio=merged_ratio, merge_config=merge_config, generator=generator, **kwargs)

        self.prune_layer_nums = prune_layer_nums
        self.prune_single_layer_nums = prune_single_layer_nums
        self.reduce_mode = 'mlerp'
    
    def on_register_hook(self, block: nn.Module):
        super().on_register_hook(block)
        if self.layer_idx < self.prune_layer_nums:
            self.reduce_mode = 'prune'
        if self.single_layer_idx is not None and self.prune_single_layer_nums is not None and self.single_layer_idx < self.prune_single_layer_nums:
            self.reduce_mode = 'prune'
    
    def merge(self, x: torch.Tensor, indices_dict: dict):
        return merge_dst_src_tokens(
            x,
            indices_dict['dst_idx'],
            indices_dict['src_idx'],
            indices_dict['merge_dst_idx'],
            indices_dict['merge_src_idx'],
            indices_dict['unmerge_src_idx'],
            self.reduce_mode,
        )

if __name__ == "__main__":
    model_type = "flux-dev"
    pipe, model_attrs = get_pipeline(model_type)

    tofu_config = {
        "stride": (2, 2),
        "merged_ratio": 0.5,
        "merge_config": {
            "self-attn": True,
            "cross-attn": False,
            "ffn": False,
            "parallel": False,
        },
        "generator": torch.Generator("cuda").manual_seed(0),
        "prune_layer_nums": 10,
        "prune_single_layer_nums": None,
    }

    HookManager.configure_hooks(
        model_type=model_type,
        default=(ToFuDitBlockHook, tofu_config),
    )

    with HookManager.enabled():
        HookManager.inject_hooks(model_type=model_type, pipe=pipe, model_attrs=model_attrs)
        res = process_quick_start(pipe)
        res.save(f"{model_type}_tofu.png")
    
    print(global_context)