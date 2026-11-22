import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from typing import Literal, Optional, Tuple, Union
from diffusers.models.transformers.transformer_flux import FluxSingleTransformerBlock
from shiva_dit.hook import BaseDiTBlockHook, HookManager
from shiva_dit.utils import (
    global_context,
    get_dst_src_tokens,
    sample_dst_src_with_2D_stride,
    select_dst_src_tokens,
    gather_dst_unmerge_tokens,
    merge_dst_src_tokens,
    unmerge_dst_src_tokens,
    get_pipeline,
    process_quick_start,
    parse_image_rotary_emb,
    join_image_rotary_emb,
)

class IBTMDiTBlockHook(BaseDiTBlockHook):
    def __init__(self, merged_ratio: float = 0.2, merge_config=None, pool_param: float = 0.4, dst_ratio_k: float = 0.25, prune_timestep_ratio = 0.1, generator: torch.Generator = None, **kwargs):
        super().__init__(**kwargs)
        self.merged_ratio = merged_ratio
        self.pool_param = pool_param
        self.dst_ratio_k = dst_ratio_k
        self.generator = generator
        self.prune_timestep_ratio = prune_timestep_ratio

        if merge_config is None:
            self.merge_config = {
                'self-attn': True,
                'cross-attn': True,
                'ffn': True,
                'parallel': True,
            }
        else:
            self.merge_config = merge_config

        self.k = None
        self.indices_dict = {
            'dst_idx': None,
            'ind_idx': None,
            'merge_idx': None,
            'merge_assignments': None,
        }

        self.extra_k = None
        self.extra_indices_dict = {
            'dst_idx': None,
            'ind_idx': None,
            'merge_idx': None,
            'merge_assignments': None,
        }
    
    def _clean(self):
        self.indices_dict = {
            'dst_idx': None,
            'ind_idx': None,
            'merge_idx': None,
            'merge_assignments': None,
        }
        self.extra_indices_dict = {
            'dst_idx': None,
            'ind_idx': None,
            'merge_idx': None,
            'merge_assignments': None,
        }
    
    def on_register_hook(self, block: nn.Module):
        if self.model_type == 'wan':
            raise NotImplementedError("wan model is not supported")

        if isinstance(block, FluxSingleTransformerBlock):
            for key in ('self-attn', 'cross-attn', 'ffn'):
                if self.merge_config.get(key):
                    self.merge_config[key] = False
        else:
            if self.merge_config.get('parallel'):
                self.merge_config['parallel'] = False

        self.enable = any(self.merge_config.values())

        if self.model_type == 'sd3':
            total_layers = 24
            if self.layer_idx < int(total_layers * 0.25) or self.layer_idx > int(total_layers * 0.75):
                self.enable = False
                print(f"  - Layer {self.layer_idx} is not enabled")

    @torch.no_grad()
    def select_dst_src_ind_tokens(self, x, r, generator: torch.Generator = None, **kwargs):
        if r == 0:
            global_context.importance_unmerge = None
            return x

        B, N, C = x.shape

        cfg_hidden_state = global_context.last_cfg_hidden_state # (1, C, H, W)
        H, W = cfg_hidden_state.shape[-2:]
        c = cfg_hidden_state.shape[1]
        patch_H = H // global_context.hidden_seq_height
        patch_W = W // global_context.hidden_seq_width
        cfg_hidden_state = cfg_hidden_state.view(1, -1, patch_H, global_context.hidden_seq_height, patch_W, global_context.hidden_seq_width).permute(0, 3, 5, 2, 4, 1)
        importance = cfg_hidden_state.reshape(1, N, c * patch_H * patch_W) # (1, N, C * patch_H * patch_W)
        importance = importance.sum(dim=-1).squeeze(0) # (N)

        N_final = N - r
        N_dst = int(N * self.dst_ratio_k)
        N_ind = N_final - N_dst
        if N_ind < 0:
            N_ind = 0
            N_dst = N_final

        pool_size = min(int(N_final * (1 + self.pool_param)), N)
        if pool_size < N_dst:
            pool_size = min(N_dst, N)

        N_src_pool_max = pool_size - N_dst
        if N_ind > N_src_pool_max:
            N_ind = max(0, N_src_pool_max)
            N_dst = N_final - N_ind

        pool_indices = torch.topk(
            importance, k=pool_size, dim=-1
        ).indices # (pool_size,)

        rand_indices = torch.randperm(pool_size, device=x.device, generator=generator)
        dst_indices_in_pool = rand_indices[:N_dst] # (N_dst,)
        src_indices_in_pool = rand_indices[N_dst:]  # (pool_size - N_dst)

        dst_indices_orig = pool_indices[dst_indices_in_pool]  # (N_dst,)
        src_pool_indices_orig = pool_indices[src_indices_in_pool]  # (pool_size - N_dst,)

        pool_mask = torch.zeros(N, device=x.device, dtype=torch.bool)
        pool_mask.scatter_(dim=0, index=pool_indices, value=1)
        unimportant_indices_orig = (~pool_mask).nonzero(as_tuple=True)[0]  # (N - pool_size,)

        src_indices_orig = torch.cat(
            [src_pool_indices_orig, unimportant_indices_orig]
        )  # (N - N_dst,)

        H = global_context.hidden_seq_height
        W = global_context.hidden_seq_width
        grid_y, grid_x = torch.meshgrid(torch.arange(H, device=x.device), torch.arange(W, device=x.device), indexing='ij')
        coords = torch.stack([grid_x.flatten(), grid_y.flatten()], dim=-1).float() # (N, 2)

        coords_dst = coords[dst_indices_orig].unsqueeze(0).expand(B, -1, -1)
        coords_src = coords[src_indices_orig].unsqueeze(0).expand(B, -1, -1)
        dists = torch.cdist(coords_src, coords_dst, p=2) 
        sim = -dists
        
        sim_max, sim_idx = sim.max(dim=-1)  # (B, N - N_dst)

        N_src_pool = src_pool_indices_orig.shape[0]
        sim_max_pool = sim_max[:, :N_src_pool]  # (B, pool_size - N_dst)

        ind_indices_in_src_pool = torch.topk(
            sim_max_pool[0], k=N_ind, dim=-1, largest=False
        ).indices  # (N_ind)

        ind_indices_orig = src_pool_indices_orig[ind_indices_in_src_pool]  # (N_ind)

        ind_mask_in_src_pool = torch.zeros(
            N_src_pool, device=x.device, dtype=torch.bool
        ) # (pool_size - N_dst,)
        if N_ind > 0:
            ind_mask_in_src_pool.scatter_(
                dim=0, index=ind_indices_in_src_pool, value=1
            )

        merge_indices_in_src_pool = (~ind_mask_in_src_pool).nonzero(as_tuple=True)[0]
        src_pool_merge_indices_orig = src_pool_indices_orig[merge_indices_in_src_pool] # (pool_size - N_dst - N_ind,)

        all_merge_indices_orig = torch.cat(
            [src_pool_merge_indices_orig, unimportant_indices_orig]
        ) # (N - N_dst - N_ind)

        sim_idx_pool = sim_idx[0, :N_src_pool] # (pool_size - N_dst,)
        sim_idx_unimportant = sim_idx[0, N_src_pool:] # (N - pool_size)
        assignments_pool_merge = sim_idx_pool.gather(
            dim=0, index=merge_indices_in_src_pool
        ) # (pool_size - N_dst - N_ind)

        all_merge_assignments = torch.cat(
            [assignments_pool_merge, sim_idx_unimportant]
        )  # (N - N_dst - N_ind)

        return dst_indices_orig, ind_indices_orig, all_merge_indices_orig, all_merge_assignments

    @torch.no_grad()
    @staticmethod
    def merge(x: torch.Tensor, indices_dict: dict, reduce_mode="mean"):
        """
        x: (B, N, C)
        dst_indices_orig: (N_dst,)
        ind_indices_orig: (N_ind,)
        all_merge_indices_orig: (N - N_dst - N_ind,)
        all_merge_assignments: (N - N_dst - N_ind,)
        """
        dst_indices_orig, ind_indices_orig, all_merge_indices_orig, all_merge_assignments = indices_dict["dst_idx"], indices_dict["ind_idx"], indices_dict["merge_idx"], indices_dict["merge_assignments"]

        B, _, C = x.shape

        x_ind = x.gather(
            dim=1,
            index=ind_indices_orig.unsqueeze(0).unsqueeze(-1).expand(B, -1, C),
        ) # (B, N_ind, C)

        x_src = x.gather(
            dim=1,
            index=all_merge_indices_orig.unsqueeze(0).unsqueeze(-1).expand(B, -1, C),
        ) # (B, N - N_dst - N_ind, C)

        x_dst = x.gather(
            dim=1,
            index=dst_indices_orig.unsqueeze(0).unsqueeze(-1).expand(B, -1, C),
        ) # (B, N_dst, C)

        if reduce_mode == 'prune':
            merge_dst_tokens = x_dst
        else:
            merge_dst_tokens = x_dst.scatter_reduce(
                dim=1,
                index=all_merge_assignments.unsqueeze(0).unsqueeze(-1).expand(B, -1, C),
                src=x_src,
                reduce=reduce_mode,
            ) # (B, N_dst, C)

        return torch.cat([x_ind, merge_dst_tokens], dim=1) # (B, N_ind + N_dst, C)

    @torch.no_grad()
    @staticmethod
    def unmerge(merged_x: torch.Tensor, indices_dict: dict):
        """
        merged_x: (B, N_ind + N_dst, C)
        dst_indices_orig: (N_dst,)
        ind_indices_orig: (N_ind,)
        all_merge_indices_orig: (N - N_dst - N_ind,)
        all_merge_assignments: (N - N_dst - N_ind,)
        """
        dst_indices_orig, ind_indices_orig, all_merge_indices_orig, all_merge_assignments = indices_dict["dst_idx"], indices_dict["ind_idx"], indices_dict["merge_idx"], indices_dict["merge_assignments"]

        B, _, C = merged_x.shape

        N_dst = dst_indices_orig.shape[0]
        N_ind = ind_indices_orig.shape[0]
        N = N_dst + N_ind + all_merge_indices_orig.shape[0]

        x_ind = merged_x[:, :N_ind, :] # (B, N_ind, C)
        x_dst = merged_x[:, N_ind:, :] # (B, N_dst, C)

        x_merged_out = x_dst.gather(
            dim=1,
            index=all_merge_assignments.unsqueeze(0).unsqueeze(-1).expand(B, -1, C),
        ) # (B, N - N_dst - N_ind, C)

        x_full = torch.empty(B, N, C, device=merged_x.device, dtype=merged_x.dtype)
        x_full.scatter_(
            dim=1,
            index=dst_indices_orig.unsqueeze(0).unsqueeze(-1).expand(B, -1, C),
            src=x_dst,
        )
        x_full.scatter_(
            dim=1,
            index=ind_indices_orig.unsqueeze(0).unsqueeze(-1).expand(B, -1, C),
            src=x_ind,
        )
        x_full.scatter_(
            dim=1,
            index=all_merge_indices_orig.unsqueeze(0).unsqueeze(-1).expand(B, -1, C),
            src=x_merged_out,
        )

        return x_full

    @torch.no_grad()
    @staticmethod
    def select_image_rotary_emb(
        img_rotary_emb: torch.Tensor, indices_dict: dict
    ) -> torch.Tensor:
        """
        img_rotary_emb: (N, D)
        dst_indices_orig: (N_dst,)
        ind_indices_orig: (N_ind,)
        """
        N_dst = indices_dict["dst_idx"].shape[0]
        N_ind = indices_dict["ind_idx"].shape[0]
        C = img_rotary_emb.shape[-1]

        ind = torch.gather(
            img_rotary_emb,
            dim=0,
            index=indices_dict["ind_idx"].expand(N_ind, C),
        ) # (N_ind, C)

        dst = torch.gather(
            img_rotary_emb,
            dim=0,
            index=indices_dict["dst_idx"].expand(N_ind, C),
        ) # (N_dst, C)

        img_rotary_emb = torch.cat([ind, dst], dim=0)
        return img_rotary_emb

    def hook_pre_norm(
        self, 
        module_name: Literal["self-attn", "cross-attn", "ffn", "parallel"],
        hidden_states: torch.Tensor,
        encoder_hidden_states: torch.Tensor = None,
        image_rotary_emb: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
        temb: torch.Tensor = None,
    ):
        if module_name in ("self-attn", "parallel") and self.enable and global_context.current_iteration > 0:
            # only sample and select tokens when self-attention's start
            def _select(indices_dict, x, k):
                indices_dict['dst_idx'], indices_dict['ind_idx'], indices_dict['merge_idx'], indices_dict['merge_assignments'] = self.select_dst_src_ind_tokens(x, k, generator=self.generator)

            def _select_image_rotary_emb(image_rotary_emb, encoder_hidden_seq_len, hidden_seq_len):
                hidden_emb, encoder_emb, extra_emb = parse_image_rotary_emb(
                    self.model_type, image_rotary_emb,
                    hidden_seq_len, encoder_hidden_seq_len
                )

                def _gather_pair(rotary_pair, indices_dict):
                    if rotary_pair is None:
                        return None

                    return tuple(
                        IBTMDiTBlockHook.select_image_rotary_emb(x, indices_dict)
                        for x in rotary_pair
                    )

                hidden_emb = _gather_pair(hidden_emb, self.indices_dict)
                if extra_emb is not None:
                    extra_emb = _gather_pair(extra_emb, self.extra_indices_dict)
                return join_image_rotary_emb(self.model_type, hidden_emb, encoder_emb, extra_emb)

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
                if self.k is None:
                    self.k = int(self.merged_ratio * global_context.hidden_seq_len)
                if self.extra_k is None:
                    self.extra_k = int(self.merged_ratio * global_context.hidden_extra_seq_len)
                _select(self.indices_dict, hidden_states[:,:global_context.hidden_seq_len,:], self.k)
                _select(self.extra_indices_dict, hidden_states[:,global_context.hidden_seq_len:,:], self.extra_k)
            else:
                # normal case
                if global_context.hidden_seq_height is None or global_context.hidden_seq_width is None:
                    raise ValueError("hidden_seq_height and hidden_seq_width must be set in global_context")
                if self.k is None:
                    self.k = int(self.merged_ratio * global_context.hidden_seq_len)
                _select(self.indices_dict, hidden_states, self.k)

            if image_rotary_emb is not None and (
                self.merge_config.get("self-attn") or self.merge_config.get("parallel")
            ):
                image_rotary_emb = _select_image_rotary_emb(
                    image_rotary_emb,
                    encoder_hidden_seq_len=global_context.encoder_hidden_seq_len_gen(),
                    hidden_seq_len=global_context.hidden_seq_len,
                )

        return super().hook_pre_norm(module_name, hidden_states, encoder_hidden_states, image_rotary_emb, temb)

    def hook_after_norm(
        self, 
        module_name: Literal["self-attn", "cross-attn", "ffn", "parallel"],
        norm_hidden_states: torch.Tensor,
        norm_encoder_hidden_states: torch.Tensor = None,
        image_rotary_emb: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
    ):
        if self.merge_config.get(module_name, False) and self.enable and global_context.current_iteration > 0:
            reduce_mode = 'prune' if global_context.current_iteration <= self.prune_timestep_ratio * global_context.num_inference_steps else 'mean'
            
            if global_context.hidden_extra_seq_len is not None:
                # dual hidden states for edit task
                h1, h2 = norm_hidden_states.split([global_context.hidden_seq_len, norm_hidden_states.shape[1] - global_context.hidden_seq_len], dim=1)
                norm_hidden_states = torch.cat([IBTMDiTBlockHook.merge(h1, self.indices_dict, reduce_mode), IBTMDiTBlockHook.merge(h2, self.extra_indices_dict, reduce_mode)], dim=1)
            else:
                # normal case
                norm_hidden_states = IBTMDiTBlockHook.merge(norm_hidden_states, self.indices_dict, reduce_mode)

        return super().hook_after_norm(module_name, norm_hidden_states, norm_encoder_hidden_states, image_rotary_emb)
    
    def hook_after_module(
        self, 
        module_name: Literal["self-attn", "cross-attn", "ffn", "parallel"],
        hidden_output: torch.Tensor,
        encoder_hidden_output: torch.Tensor = None,
    ):
        if self.merge_config.get(module_name, False) and self.enable and global_context.current_iteration > 0:
            if global_context.hidden_extra_seq_len is not None:
                # dual hidden states for edit task
                k = self.indices_dict['merge_idx'].shape[0]
                hidden_seq_len = global_context.hidden_seq_len - k
                h1, h2 = hidden_output.split([hidden_seq_len, hidden_output.shape[1] - hidden_seq_len], dim=1)
                hidden_output = torch.cat([IBTMDiTBlockHook.unmerge(h1, self.indices_dict), IBTMDiTBlockHook.unmerge(h2, self.extra_indices_dict)], dim=1)
            else:
                # normal case
                hidden_output = IBTMDiTBlockHook.unmerge(hidden_output, self.indices_dict)

        if module_name in ("ffn", "parallel"):
            # clear indices after ffn
            self._clean()

        return super().hook_after_module(module_name, hidden_output, encoder_hidden_output)

if __name__ == "__main__":
    model_type = "sd3"
    pipe, model_attrs = get_pipeline(model_type)

    ibtm_config = {
        "merged_ratio": 0.2,
        "merge_config": {
            "self-attn": True,
            "cross-attn": False,
            "ffn": True,
            "parallel": False,
        },
        "pool_param": 0.4,
        "dst_ratio_k": 0.25,
        "prune_timestep_ratio": 0.12,
        "generator": torch.Generator("cuda").manual_seed(0),
    }

    HookManager.configure_hooks(
        model_type=model_type,
        default=(IBTMDiTBlockHook, ibtm_config),
    )

    with HookManager.enabled():
        HookManager.inject_hooks(model_type=model_type, pipe=pipe, model_attrs=model_attrs)
        res = process_quick_start(pipe)
        res.save(f"{model_type}_ibtm.png", num_inference_steps=50, negative_prompt=" ")
    
    print(global_context)