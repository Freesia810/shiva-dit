import torch
from typing import Literal, Tuple, Union

from project_name.hook import BaseDiTBlockHook
from project_name.utils.context import global_context
from project_name.utils.token_utils import (
    get_dst_src_tokens,
    sample_dst_src_with_2D_stride,
    select_dst_src_tokens,
    merge_dst_src_tokens,
    unmerge_dst_src_tokens,
)

class ToMeSDDitBlockHook(BaseDiTBlockHook):
    def __init__(self, stride: Union[int, Tuple[int, int]]=(2, 2), merged_ratio: float = 0.2, merge_config=None, generator: torch.Generator = None, **kwargs):
        super().__init__(**kwargs)
        self.stride = stride
        self.ratio = merged_ratio
        self.generator = generator
        self.k = int(self.ratio * self.size[0] * self.size[1])
        if merge_config is None:
            self.merge_config = {
                'self-attn': True,
                'cross-attn': True,
                'ffn': True
            }
        else:
            self.merge_config = merge_config
        self.enable = any(self.merge_config.values())

        self.size = None # dynamic attr
        self.indices_dict = {
            'dst_idx': None,
            'src_idx': None,
            'merge_dst_idx': None,
            'merge_src_idx': None,
            'unmerge_src_idx': None,
        }

        # only work for edit task with dual hidden states
        self.extra_size = None
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
        if module_name == "self-attn" and self.enable:
            # only sample and select tokens when self-attention's start
            def _select(size, indices_dict, x):
                indices_dict['dst_idx'], indices_dict['src_idx'] = sample_dst_src_with_2D_stride(size, self.stride, self.generator, rand=True, device=hidden_states.device)
                scores = self.calculate_scores(
                    x, indices_dict['dst_idx'], indices_dict['src_idx']
                )  # (B, num_src, num_dst)
                indices_dict['merge_dst_idx'], indices_dict['merge_src_idx'], indices_dict['unmerge_src_idx'] = select_dst_src_tokens(
                    scores, indices_dict['dst_idx'], indices_dict['src_idx'], self.k
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
                if self.extra_size is None:
                    self.extra_size = (
                        global_context.hidden_extra_seq_height,
                        global_context.hidden_extra_seq_width,
                    )
                _select(self.size, self.indices_dict, hidden_states[:,:global_context.hidden_seq_len,:])
                _select(self.extra_size, self.extra_indices_dict, hidden_states[:,global_context.hidden_seq_len:,:])
            else:
                # normal case
                if global_context.hidden_seq_height is None or global_context.hidden_seq_width is None:
                    raise ValueError("hidden_seq_height and hidden_seq_width must be set in global_context")
                if self.size is None:
                    self.size = (global_context.hidden_seq_height, global_context.hidden_seq_width)
                _select(self.size, self.indices_dict, hidden_states)

        return hidden_states, encoder_hidden_states

    def hook_after_norm(
        self, 
        module_name: Literal["self-attn", "cross-attn", "ffn"],
        norm_hidden_states: torch.Tensor,
        norm_encoder_hidden_states: torch.Tensor = None,
        temb: torch.Tensor = None,
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

            if global_context.hidden_extra_seq_len is not None:
                # dual hidden states for edit task
                h1, h2 = norm_hidden_states.split([global_context.hidden_seq_len, norm_hidden_states.shape[1] - global_context.hidden_seq_len], dim=1)
                norm_hidden_states = torch.cat([_merge(h1, self.indices_dict), _merge(h2, self.extra_indices_dict)], dim=1)
            else:
                # normal case
                norm_hidden_states = _merge(norm_hidden_states, self.indices_dict)

        return norm_hidden_states, norm_encoder_hidden_states

    def hook_after_module(
        self, 
        module_name: Literal["self-attn", "cross-attn", "ffn"],
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
            
            if module_name == "ffn":
                # clear indices after ffn
                self._clear()

        return hidden_output, encoder_hidden_output
