import torch
import torch.nn as nn
from typing import Literal, Optional, Tuple, Union
from diffusers.models.transformers.transformer_flux import FluxSingleTransformerBlock
from project_name.hook import BaseDiTBlockHook, HookManager
from project_name.utils import (
    global_context,
    get_dst_src_tokens,
    sample_dst_src_with_2D_stride,
    select_dst_src_tokens,
    gather_dst_unmerge_tokens,
    merge_dst_src_tokens,
    unmerge_dst_src_tokens,
    get_pipeline,
    process_quick_start
)

class ToMeSDDitBlockHook(BaseDiTBlockHook):
    def __init__(self, stride: Union[int, Tuple[int, int]]=(2, 2), merged_ratio: float = 0.2, merge_config=None, generator: torch.Generator = None, **kwargs):
        super().__init__(**kwargs)
        self.stride = stride
        self.ratio = merged_ratio
        self.generator = generator
        if merge_config is None:
            self.merge_config = {
                'self-attn': True,
                'cross-attn': True,
                'ffn': True,
                'parallel': True,
            }
        else:
            self.merge_config = merge_config

        self.size = None # dynamic attr
        self.k = None
        self.indices_dict = {
            'dst_idx': None,
            'src_idx': None,
            'merge_dst_idx': None,
            'merge_src_idx': None,
            'unmerge_src_idx': None,
        }

        # only work for edit task with dual hidden states
        self.extra_size = None
        self.extra_k = None
        self.extra_indices_dict = {
            'dst_idx': None,
            'src_idx': None,
            'merge_dst_idx': None,
            'merge_src_idx': None,
            'unmerge_src_idx': None,
        }

    def _clear(self):
        self.indices_dict = {
            'dst_idx': None,
            'src_idx': None,
            'merge_dst_idx': None,
            'merge_src_idx': None,
            'unmerge_src_idx': None,
        }
        self.extra_indices_dict = {
            'dst_idx': None,
            'src_idx': None,
            'merge_dst_idx': None,
            'merge_src_idx': None,
            'unmerge_src_idx': None,
        }

    def on_register_hook(self, block: nn.Module):
        if isinstance(block, FluxSingleTransformerBlock):
            for key in ('self-attn', 'cross-attn', 'ffn'):
                if self.merge_config.get(key):
                    self.merge_config[key] = False
        else:
            if self.merge_config.get('parallel'):
                self.merge_config['parallel'] = False

        self.enable = any(self.merge_config.values())

    def calculate_scores(self, x: torch.Tensor, dst_idx: torch.Tensor, src_idx: torch.Tensor) -> torch.Tensor:
        # x: (B, N, D)
        # dst_idx: (1, num_dst, 1)
        # src_idx: (1, N - num_dst, 1)
        # return: (B, num_src, num_dst)
        x = x / x.norm(dim=-1, keepdim=True)
        dst_tokens, src_tokens = get_dst_src_tokens(x, dst_idx, src_idx) # (B, num_dst, D), (B, N - num_dst, D)
        scores = src_tokens @ dst_tokens.transpose(-1, -2) # (B, N - num_dst, num_dst)

        return scores

    def hook_pre_norm(
        self, 
        module_name: Literal["self-attn", "cross-attn", "ffn", "parallel"],
        hidden_states: torch.Tensor,
        encoder_hidden_states: torch.Tensor = None,
        temb: torch.Tensor = None,
    ):
        if module_name in ("self-attn", "parallel") and self.enable:
            # only sample and select tokens when self-attention's start
            def _select(size, indices_dict, x, k):
                indices_dict['dst_idx'], indices_dict['src_idx'] = sample_dst_src_with_2D_stride(size, self.stride, self.generator, rand=True, device=hidden_states.device)
                scores = self.calculate_scores(
                    x, indices_dict['dst_idx'], indices_dict['src_idx']
                )  # (B, num_src, num_dst)
                indices_dict['merge_dst_idx'], indices_dict['merge_src_idx'], indices_dict['unmerge_src_idx'] = select_dst_src_tokens(
                    scores, indices_dict['dst_idx'], indices_dict['src_idx'], k
                )

            if global_context.hidden_extra_seq_len is not None:
                # dual hidden states for edit task
                if (
                    global_context.hidden_seq_height is None
                    or global_context.hidden_seq_width is None
                ):
                    raise ValueError(
                        "hidden_seq_height and hidden_seq_width must be set in global_context"
                    )
                if (
                    global_context.hidden_extra_seq_height is None
                    or global_context.hidden_extra_seq_width is None
                ):
                    raise ValueError(
                        "hidden_extra_seq_height and hidden_extra_seq_width must be set in global_context"
                    )
                if self.size is None:
                    self.size = (
                        global_context.hidden_seq_height,
                        global_context.hidden_seq_width,
                    )
                    self.k = int(self.ratio * self.size[0] * self.size[1])
                if self.extra_size is None:
                    self.extra_size = (
                        global_context.hidden_extra_seq_height,
                        global_context.hidden_extra_seq_width,
                    )
                    self.extra_k = int(self.ratio * self.extra_size[0] * self.extra_size[1])
                _select(self.size, self.indices_dict, hidden_states[:,:global_context.hidden_seq_len,:], self.k)
                _select(self.extra_size, self.extra_indices_dict, hidden_states[:,global_context.hidden_seq_len:,:], self.extra_k)
            else:
                # normal case
                if global_context.hidden_seq_height is None or global_context.hidden_seq_width is None:
                    raise ValueError("hidden_seq_height and hidden_seq_width must be set in global_context")
                if self.size is None:
                    self.size = (global_context.hidden_seq_height, global_context.hidden_seq_width)
                    self.k = int(self.ratio * self.size[0] * self.size[1])
                _select(self.size, self.indices_dict, hidden_states, self.k)

        return hidden_states, encoder_hidden_states

    def hook_after_norm(
        self, 
        module_name: Literal["self-attn", "cross-attn", "ffn", "parallel"],
        norm_hidden_states: torch.Tensor,
        norm_encoder_hidden_states: torch.Tensor = None,
        image_rotary_emb: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
    ):
        if self.merge_config.get(module_name, False):
            def _merge(x, indices_dict):
                return merge_dst_src_tokens(
                    x,
                    indices_dict['dst_idx'],
                    indices_dict['src_idx'],
                    indices_dict['merge_dst_idx'],
                    indices_dict['merge_src_idx'],
                    indices_dict['unmerge_src_idx'],
                )

            def _select_image_rotary_emb(image_rotary_emb, encoder_hidden_seq_len, hidden_seq_len, hidden_extra_seq_len=None):
                ## normal [encoder, hidden]
                ## flux-kontext, qwen-edit [encoder, hidden, extra_hidden]
                cos, sin = image_rotary_emb
                cos_encoder = cos[:encoder_hidden_seq_len, ...]
                sin_encoder = sin[:encoder_hidden_seq_len, ...]
                cos_hidden = cos[encoder_hidden_seq_len:encoder_hidden_seq_len + hidden_seq_len, ...]
                sin_hidden = sin[encoder_hidden_seq_len:encoder_hidden_seq_len + hidden_seq_len, ...]
                cos_hidden = gather_dst_unmerge_tokens(cos_hidden, self.indices_dict['dst_idx'], self.indices_dict['src_idx'], self.indices_dict['merge_dst_idx'], self.indices_dict['unmerge_src_idx'])
                sin_hidden = gather_dst_unmerge_tokens(sin_hidden, self.indices_dict['dst_idx'], self.indices_dict['src_idx'], self.indices_dict['merge_dst_idx'], self.indices_dict['unmerge_src_idx'])
                if hidden_extra_seq_len is not None:
                    cos_extra_hidden = cos[encoder_hidden_seq_len + hidden_seq_len:, ...]
                    sin_extra_hidden = sin[encoder_hidden_seq_len + hidden_seq_len:, ...]
                    cos_extra_hidden = gather_dst_unmerge_tokens(cos_extra_hidden, self.extra_indices_dict['dst_idx'], self.extra_indices_dict['src_idx'], self.extra_indices_dict['merge_dst_idx'], self.extra_indices_dict['unmerge_src_idx'])
                    sin_extra_hidden = gather_dst_unmerge_tokens(sin_extra_hidden, self.extra_indices_dict['dst_idx'], self.extra_indices_dict['src_idx'], self.extra_indices_dict['merge_dst_idx'], self.extra_indices_dict['unmerge_src_idx'])
                    cos = torch.cat([cos_encoder, cos_hidden, cos_extra_hidden], dim=0)
                    sin = torch.cat([sin_encoder, sin_hidden, sin_extra_hidden], dim=0)
                else:
                    cos = torch.cat([cos_encoder, cos_hidden], dim=0)
                    sin = torch.cat([sin_encoder, sin_hidden], dim=0)
                return (cos, sin)

            if global_context.hidden_extra_seq_len is not None:
                # dual hidden states for edit task
                h1, h2 = norm_hidden_states.split([global_context.hidden_seq_len, norm_hidden_states.shape[1] - global_context.hidden_seq_len], dim=1)
                norm_hidden_states = torch.cat([_merge(h1, self.indices_dict), _merge(h2, self.extra_indices_dict)], dim=1)
            else:
                # normal case
                norm_hidden_states = _merge(norm_hidden_states, self.indices_dict)

            if image_rotary_emb is not None:
                image_rotary_emb = _select_image_rotary_emb(
                    image_rotary_emb,
                    encoder_hidden_seq_len=global_context.encoder_hidden_seq_len,
                    hidden_seq_len=global_context.hidden_seq_len,
                    hidden_extra_seq_len=global_context.hidden_extra_seq_len,
                )

        if image_rotary_emb is not None:
            return norm_hidden_states, norm_encoder_hidden_states, image_rotary_emb
        return norm_hidden_states, norm_encoder_hidden_states

    def hook_after_module(
        self, 
        module_name: Literal["self-attn", "cross-attn", "ffn", "parallel"],
        hidden_output: torch.Tensor,
        encoder_hidden_output: torch.Tensor = None,
    ):
        if self.merge_config.get(module_name, False):
            def _unmerge(x, indices_dict):
                return unmerge_dst_src_tokens(
                    x,
                    indices_dict['dst_idx'],
                    indices_dict['src_idx'],
                    indices_dict['merge_dst_idx'],
                    indices_dict['merge_src_idx'],
                    indices_dict['unmerge_src_idx'],
                )

            if global_context.hidden_extra_seq_len is not None:
                # dual hidden states for edit task
                k = self.indices_dict['merge_dst_idx'].shape[1]
                hidden_seq_len = global_context.hidden_seq_len - k
                h1, h2 = hidden_output.split([hidden_seq_len, hidden_output.shape[1] - hidden_seq_len], dim=1)
                hidden_output = torch.cat([_unmerge(h1, self.indices_dict), _unmerge(h2, self.extra_indices_dict)], dim=1)
            else:
                # normal case
                hidden_output = _unmerge(hidden_output, self.indices_dict)

        if module_name in ("ffn", "parallel"):
            # clear indices after ffn
            self._clear()

        return hidden_output, encoder_hidden_output

if __name__ == "__main__":
    model_type = "flux-dev"
    pipe, model_attrs = get_pipeline(model_type)

    tomesd_config = {
        "stride": (2, 2),
        "merged_ratio": 0.2,
        "merge_config": {
            "self-attn": True,
            "cross-attn": False,
            "ffn": False,
            "parallel": True,
        },
        "generator": torch.Generator("cuda").manual_seed(0),
    }

    HookManager.configure_hooks(
        model_type=model_type,
        default=(ToMeSDDitBlockHook, tomesd_config),
    )

    with HookManager.enabled():
        HookManager.inject_hooks(model_type=model_type, pipe=pipe, model_attrs=model_attrs)
        res = process_quick_start(pipe)
        res.save(f"{model_type}_tomesd.png")
    
    print(global_context)
