import torch
import torch.nn.functional as F
from typing import Union, Tuple
from project_name.hook import BaseDiTBlockHook, HookManager
from project_name.utils import global_context

class SDTMDiTBlockHook(BaseDiTBlockHook):
    """
    Reference:
        - Paper: https://arxiv.org/pdf/2505.11707

    Implemented from https://github.com/ICTMCG/SDTM
    """
    def __init__(self,
        stride: Union[int, Tuple[int, int]]=(2, 2), 
        switch_step: int = 20,
        alpha_s = 0.05,
        **kwargs    
    ):
        super().__init__(**kwargs)
        self.stride = stride
        self.switch_step = switch_step
        self.alpha_s = alpha_s

        # dynamic attr
        self.size = None
        self.indices_dict = {
            'dst_idx': None,
            'src_idx': None,
            'merge_idx': None,
            'unmerge_idx': None,
        }

        # only work for edit task with dual hidden states
        self.extra_size = None
        self.extra_indices_dict = {
            'dst_idx': None,
            'src_idx': None,
            'merge_idx': None,
            'unmerge_idx': None,
        }

        # cache for unmerge
        self.prev_attn_output = None
        self.prev_mlp_output = None
    
    def _clean(self):
        self.indices_dict = {
            'dst_idx': None,
            'src_idx': None,
            'merge_idx': None,
            'unmerge_idx': None,
        }
        self.extra_indices_dict = {
            'dst_idx': None,
            'src_idx': None,
            'merge_idx': None,
            'unmerge_idx': None,
        }
    
    @torch.no_grad()
    @staticmethod
    def calculate_sims_patched(x: torch.Tensor, size: Union[int, Tuple[int, int]], stride: Union[int, Tuple[int, int]]=(2, 2)):
        """
        Calculate similarity map for SDTM.
        x: (B, S, C)
        size: (h, w)
        stride: (h, w)
        Return: similarity_map_flat: (B, h_patch * w_patch)
        """
        if isinstance(size, int):
            h = w = size
        else:
            h, w = size
        
        if isinstance(stride, int):
            stride_h = stride_w = stride
        else:
            stride_h, stride_w = stride
        
        B, S, C = x.shape

        h_patch = h // stride_h
        w_patch = w // stride_w
        num_tokens_in_patch = stride_h * stride_w

        x = x.view(B, h, w, C) # (B, h, w, C)
        x_patched = x.view(B, h_patch, stride_h, w_patch, stride_w, C) # (B, h_patch, stride_h, w_patch, stride_w, C)
        x_patched = x_patched.permute(0, 1, 3, 5, 2, 4) # (B, h_patch, w_patch, C, stride_h, stride_w)
        x_flat = x_patched.reshape(B, h_patch, w_patch, C, num_tokens_in_patch) # (B, h_patch, w_patch, C, K)

        x_flat_1 = x_flat.unsqueeze(-1) # (B, h_patch, w_patch, C, K, 1)
        x_flat_2 = x_flat.unsqueeze(-2) # (B, h_patch, w_patch, C, 1, K)
        sims = F.cosine_similarity(x_flat_1, x_flat_2, dim=3) # (B, h_patch, w_patch, C, K, K)

        # for paper's formula, we need to divide by K * (K - 1), but we found that the origin code makes a division by K * K
        similarity_map = sims.sum(dim=(-2, -1)) / (num_tokens_in_patch * (num_tokens_in_patch - 1)) # (B, h_patch, w_patch)
        similarity_map_flat = similarity_map.reshape(B, -1) # (B, h_patch * w_patch)

        return similarity_map_flat
    
    @torch.no_grad()
    @staticmethod
    def calculate_freq_scores(size: Union[int, Tuple[int, int]], stride: Union[int, Tuple[int, int]]=(2, 2), dtype: torch.dtype = torch.float32):
        last_independent = HookManager.get_global('last_independent', None) # (B, N)
        B, _ = last_independent.shape
        
        if last_independent is None:
            return None
        
        if isinstance(size, int):
            h = w = size
        else:
            h, w = size

        if isinstance(stride, int):
            stride_h = stride_w = stride
        else:
            stride_h, stride_w = stride

        h_patch = H // stride_h
        w_patch = W // stride_w

        eps = 1e-6
        last_independent_f = last_independent.to(dtype)
        mean_last_independent = last_independent_f.mean(dim=1, keepdim=True) + eps
        indiv_priority = last_independent_f / mean_last_independent # (B, N)
        indiv_priority_grid = indiv_priority.view(B, H, W) # (B, H, W)
        indiv_priority_windows = indiv_priority_grid.view(B, h_patch, stride_h, w_patch, stride_w) # (B, h_patch, stride_h, w_patch, stride_w)
        indiv_priority_windows = indiv_priority_windows.permute(0, 1, 3, 2, 4) # (B, h_patch, w_patch, stride_h, stride_w)
        indiv_priority_windows = indiv_priority_windows.reshape(B, h_patch, w_patch, stride_h * stride_w) # (B, h_patch, w_patch, stride_h * stride_w)
        indiv_priority_mean = indiv_priority_windows.mean(dim=-1) # (B, h_patch, w_patch)
        indiv_priority_flat = indiv_priority_mean.reshape(B, -1) # (B, h_patch * w_patch)

        return indiv_priority_flat

    @torch.no_grad()
    @staticmethod
    def update_freq(independent_indices: torch.Tensor):
        last_independent = HookManager.get_global('last_independent', None)
        if last_independent is None:
            HookManager.set_global('last_independent', torch.zeros(independent_indices.shape[0], global_context.hidden_seq_len, device=independent_indices.device, dtype=torch.int32))
            last_independent = HookManager.get_global('last_independent')
        
        last_independent.add_(1)
        zeros_src = torch.zeros_like(independent_indices, dtype=last_independent.dtype)
        last_independent.scatter_(1, independent_indices, zeros_src)
    
    @torch.no_grad()
    @staticmethod
    def select_window_tokens(
        scores: torch.Tensor, 
        reduce_num: int, 
        size: Union[int, Tuple[int, int]], 
        stride: Union[int, Tuple[int, int]]=(2, 2), 
        rand: bool = True,
        generator: torch.Generator = None
    ):
        """
        scores: (B, h_patch * w_patch)
        """
        B, _ = scores.shape

        if isinstance(size, int):
            h = w = size
        else:
            h, w = size

        if isinstance(stride, int):
            stride_h = stride_w = stride
        else:
            stride_h, stride_w = stride

        h_patch = H // stride_h
        w_patch = W // stride_w
        
        _, selected_window_flat_indices = scores.topk(reduce_num, dim=-1) # (B, reduce_num)

        tensor = torch.arange(H * W, device=scores.device).reshape(H, W) # (H, W)
        tensor = tensor.unsqueeze(0).repeat(B, 1, 1) # (B, H, W)
        windowed_tensor = tensor.unfold(1, stride_h, stride_h).unfold(2, stride_w, stride_w) # (B, h_patch, w_patch, stride_h, stride_w)
        windowed_tensor = windowed_tensor.reshape(B, -1, stride_h * stride_w) # (B, h_patch * w_patch, stride_h * stride_w)

        mask = torch.ones((B, windowed_tensor.shape[1]), dtype=bool, device=scores.device) # (B, h_patch * w_patch)
        mask_values = torch.zeros_like(selected_window_flat_indices, dtype=torch.bool, device=scores.device) # (B, reduce_num)
        mask.scatter_(1, selected_window_flat_indices, mask_values) # (B, h_patch * w_patch) 0 for selected windows, 1 for unselected windows
        remaining_tensor = windowed_tensor[mask.unsqueeze(-1).expand(-1, -1, stride_h * stride_w)].reshape(B, -1, stride_h * stride_w) # (B, h_patch * w_patch - reduce_num, stride_h * stride_w)
        unmerge_idx = remaining_tensor.reshape(B, -1).unsqueeze(-1) # (B, (h_patch * w_patch - reduce_num) * stride_h * stride_w, 1)

        selected_window_indices = torch.gather(windowed_tensor, 1, selected_window_flat_indices.unsqueeze(-1).expand(-1, -1, stride_h * stride_w)) # (B, reduce_num, stride_h * stride_w)
        full = torch.arange(stride_h * stride_w, device=scores.device)
        matrix = full.unsqueeze(0).expand(stride_h * stride_w, stride_h * stride_w) # (stride_h * stride_w, stride_h * stride_w)
        mask = torch.eye(stride_h * stride_w, dtype=torch.bool, device=scores.device) # (stride_h * stride_w, stride_h * stride_w)
        src_table = matrix[~mask].view(stride_h * stride_w, stride_h * stride_w - 1) # (stride_h * stride_w, stride_h * stride_w - 1)

        if rand:
            rand_idx = torch.randint(stride_h * stride_w, (B, reduce_num, 1), device=metric.device, generator=generator) # (B, reduce_num, 1)
        else:
            rand_idx = torch.zeros(B, reduce_num, 1, dtype=torch.long, device=scores.device) # (B, reduce_num, 1)
        
        dst_idx = torch.gather(selected_window_indices, 2, rand_idx) # (B, reduce_num, 1)
        src_pos = src_table[rand_idx.squeeze(-1)] # (B, reduce_num, stride_h * stride_w - 1)
        src_vals = torch.gather(selected_window_indices, 2, src_pos) # (B, reduce_num, stride_h * stride_w - 1)
        src_idx = src_vals.reshape(B, reduce_num * (stride_h * stride_w - 1)).unsqueeze(-1) # (B, reduce_num * (stride_h * stride_w - 1), 1)
        merge_idx = torch.arange(reduce_num, device=metric.device).repeat_interleave(stride_h * stride_w - 1).repeat(B, 1).unsqueeze(-1) # (B, reduce_num * (stride_h * stride_w - 1), 1)

        return dst_idx, src_idx, merge_idx, unmerge_idx

    @torch.no_grad()
    def process_SSM(self, x: torch.Tensor, indices_dict: dict, size: Union[int, Tuple[int, int]], , reduce_num: int):
        similarity_map_flat = self.calculate_sims_patched(self.x, size, self.stride) # (B, h_patch * w_patch)
        indiv_priority_flat = self.calculate_freq_scores(size, self.stride, similarity_map_flat.dtype) # (B, h_patch * w_patch)
        if indiv_priority_flat is not None:
            similarity_map_flat = similarity_map_flat + self.alpha_s * indiv_priority_flat # (B, h_patch * w_patch)
        
        # dst_idx       (B, reduce_num, 1)
        # src_idx       (B, reduce_num * (stride_h * stride_w - 1), 1)
        # merge_idx     (B, reduce_num * (stride_h * stride_w - 1), 1)
        # unmerge_idx   (B, (h_patch * w_patch - reduce_num) * stride_h * stride_w, 1)
        indices_dict['dst_idx'], indices_dict['src_idx'], indices_dict['merge_idx'], indices_dict['unmerge_idx'] = self.select_window_tokens(similarity_map_flat, reduce_num, size, self.stride, rand=True, generator=self.generator)

        self.update_freq(unmerge_idx.squeeze(-1))
    
    def process_IDM(self, x: torch.Tensor, attn_output: torch.Tensor, indices_dict: dict):
        """
        x: (B, S, C)
        attn_output: (B, S, C)
        """
        
            