from project_name.hook import BaseDiTBlockHook
from project_name.utils.context import global_context

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
        self.dual_hidden = False # only work for flux-kontext because flux-kontext has dual image hidden
    
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
            if self.size is None:
                if global_context.hidden_seq_height is None or global_context.hidden_seq_width is None:
                    raise ValueError("hidden_seq_height and hidden_seq_width must be set in global_context")
                if global_context.hidden_extra_seq_len is not None:
                    # only work for flux-kontext
                    # we only calculate scores for the first image
                    # two seqs share the same merge/unmerge indices
                    self.dual_hidden = True
                    self.single_seg_len = global_context.hidden_seq_len
                
                self.size = (global_context.hidden_seq_height, global_context.hidden_seq_width)

            self.dst_idx, self.src_idx = sample_dst_src_with_2D_stride(self.size, self.stride, self.generator, rand=True, device=hidden_states.device)
            scores = self.calculate_scores(hidden_states, self.dst_idx, self.src_idx) # (B, num_src, num_dst)
            self.merge_dst_idx, self.merge_src_idx, self.unmerge_src_idx = select_dst_src_tokens(scores, self.dst_idx, self.src_idx, self.k)

        return hidden_states, encoder_hidden_states
    
    def hook_after_norm(
        self, 
        module_name: Literal["self-attn", "cross-attn", "ffn"],
        norm_hidden_states: torch.Tensor,
        norm_encoder_hidden_states: torch.Tensor = None,
        temb: torch.Tensor = None,
    ):
        if self.merge_config.get(module_name, False):
            def _merge(x):
                return merge_dst_src_tokens(
                    x,
                    self.dst_idx,
                    self.src_idx,
                    self.merge_dst_idx,
                    self.merge_src_idx,
                    self.unmerge_src_idx
                )

            if self.dual_hidden:
                # only for flux-kontext
                h1, h2 = norm_hidden_states.split([self.single_seg_len, norm_hidden_states.shape[1] - self.single_seg_len], dim=1)
                norm_hidden_states = torch.cat([_merge(h1), _merge(h2)], dim=1)
            else:
                norm_hidden_states = _merge(norm_hidden_states)

        return norm_hidden_states, norm_encoder_hidden_states

    
    def hook_after_module(
        self, 
        module_name: Literal["self-attn", "cross-attn", "ffn"],
        hidden_output: torch.Tensor,
        encoder_hidden_output: torch.Tensor = None,
    ):
        if self.merge_config.get(module_name, False):
            def _unmerge(x):
                return unmerge_dst_src_tokens(
                    x,
                    self.dst_idx,
                    self.src_idx,
                    self.merge_dst_idx,
                    self.merge_src_idx,
                    self.unmerge_src_idx
                )

            if self.dual_hidden:
                # only for flux-kontext
                N = hidden_output.shape[1]
                h1, h2 = hidden_output.split([N // 2, N - N // 2], dim=1)
                hidden_output = torch.cat([_unmerge(h1), _unmerge(h2)], dim=1)
            else:
                hidden_output = _unmerge(hidden_output)

        return hidden_output, encoder_hidden_output