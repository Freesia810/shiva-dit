import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Literal, Optional, Tuple, Union
from diffusers.models.transformers.transformer_flux import FluxSingleTransformerBlock
from shiva_dit.hook import BaseDiTBlockHook, HookManager
from shiva_dit.utils import (
    global_context,
    get_pipeline,
    process_quick_start,
    parse_image_rotary_emb,
    join_image_rotary_emb,
)

class ToMADitBlockHook(BaseDiTBlockHook):
    def __init__(
        self,
        tile_nums: Union[int, Tuple[int, int]],
        merged_ratio: float = 0.2,
        merge_config=None,
        skip_layer_nums: int = 10,
        skip_single_layer_nums: Optional[int] = None,
        merge_step_interval: int = 1,
        **kwargs,
    ):
        super().__init__()
        if isinstance(tile_nums, int):
            self.h_tile_num = self.w_tile_num = tile_nums
        else:
            self.h_tile_num, self.w_tile_num = tile_nums
        self.num_tiles = self.h_tile_num * self.w_tile_num
        self.ratio = merged_ratio
        
        if merge_config is None:
            self.merge_config = {"self-attn": True, "cross-attn": True, "ffn": True, "parallel": True}
        else:
            self.merge_config = merge_config

        self.size = None 
        self.context_dict = {"dst_idx": None, "A": None, "A_inv": None}
        self.extra_context_dict = {"dst_idx": None, "A": None, "A_inv": None}

        self.skip_layer_nums = skip_layer_nums
        self.skip_single_layer_nums = skip_single_layer_nums
        if not isinstance(merge_step_interval, int) or merge_step_interval < 1:
            raise ValueError("merge_step_interval must be a positive integer")
        self.merge_step_interval = merge_step_interval

    def _merge_this_step(self):
        # Same iteration-modulo mechanism as SDTM's protected steps. ToMA
        # enables merging on 0, interval, 2*interval, ... instead of protecting
        # those steps. Default 1 preserves the previous every-step behavior.
        return self.enable and (
            self.merge_step_interval == 1
            or global_context.current_iteration % self.merge_step_interval == 0
        )
    
    @staticmethod
    def fold_with_indices(x: torch.Tensor, h_tile_num: int, w_tile_num: int):
        B, N, C = x.shape
        H = int(N**0.5) 
        tile_h = H // h_tile_num
        tile_w = H // w_tile_num 
        x_reshaped = x.reshape(B, h_tile_num, tile_h, w_tile_num, tile_w, C)
        x_reshaped = x_reshaped.permute(0, 1, 3, 2, 4, 5)
        x_folded = x_reshaped.reshape(B, h_tile_num * w_tile_num, tile_h * tile_w, C)
        return x_folded

    @staticmethod
    def unfold_with_indices(x_folded: torch.Tensor, h_tile_num: int, w_tile_num: int, original_N: int):
        B, num_tiles, tile_size, C = x_folded.shape
        H = int(original_N**0.5)
        tile_h = H // h_tile_num
        tile_w = H // w_tile_num
        x = x_folded.reshape(B, h_tile_num, w_tile_num, tile_h, tile_w, C)
        x = x.permute(0, 1, 3, 2, 4, 5)
        x_restored = x.reshape(B, original_N, C)
        return x_restored

    @torch.no_grad()
    def compute_partition(self, x: torch.Tensor, ratio: float):
        B, N, C = x.shape
        x_folded = ToMADitBlockHook.fold_with_indices(x, self.h_tile_num, self.w_tile_num) # (B, tiles, tile_size, C)
        tile_size = x_folded.shape[2]
        
        r = int(tile_size * (1 - ratio))
        if r <= 0: r = 1
        
        x_norm = F.normalize(x_folded, p=2, dim=-1)
        sim_matrix = x_norm @ x_norm.transpose(-1, -2)
        
        row_sums = torch.sum(sim_matrix, dim=-1)
        init_v = torch.argmax(row_sums, dim=-1)
        
        dst_idx = torch.zeros(B, self.num_tiles, r, dtype=torch.long, device=x.device)
        dst_idx[:, :, 0] = init_v
        
        max_sim = torch.gather(sim_matrix, 2, init_v.unsqueeze(-1).unsqueeze(-1).expand(-1, -1, -1, tile_size)).squeeze(2)
        
        for i in range(1, r):
            differences = torch.relu(sim_matrix - max_sim.unsqueeze(2))
            gain = torch.sum(differences, dim=-1)
            mask_idx = dst_idx[:, :, :i]
            gain.scatter_(2, mask_idx, -float('inf'))
            next_v = torch.argmax(gain, dim=-1)
            dst_idx[:, :, i] = next_v
            
            new_sim = torch.gather(sim_matrix, 2, next_v.unsqueeze(-1).unsqueeze(-1).expand(-1, -1, -1, tile_size)).squeeze(2)
            max_sim = torch.max(max_sim, new_sim)
            
        dst_idx, _ = torch.sort(dst_idx, dim=-1)
        
        dst_feats = torch.gather(x_norm, 2, dst_idx.unsqueeze(-1).expand(-1, -1, -1, C)) # (B, tiles, r, C)
        
        scale = 1000.0
        # shape: (B, tiles, r, tile_size)
        attn_logits = (dst_feats @ x_norm.transpose(-1, -2)) * scale
        A_probs = torch.softmax(attn_logits, dim=-2) 
        count_per_dst = A_probs.sum(dim=-1, keepdim=True) # (B, tiles, r, 1)
        A_merge = A_probs / (count_per_dst + 1e-6) 
        A_unmerge = A_probs.transpose(-1, -2) 
        
        return dst_idx, A_merge, A_unmerge

    def _clean(self):
        self.context_dict = {"A": None, "A_inv": None, "dst_idx": None}
        self.extra_context_dict = {"A": None, "A_inv": None, "dst_idx": None}

    @torch.no_grad()
    def merge(self, x: torch.Tensor, context: dict):
        if context["A"] is None: return x
        B, N, C = x.shape
        context["original_N"] = N
        
        x_folded = ToMADitBlockHook.fold_with_indices(x, self.h_tile_num, self.w_tile_num)
        x_merged = context["A"] @ x_folded 
        return x_merged.reshape(B, -1, C)

    @torch.no_grad()
    def unmerge(self, x: torch.Tensor, context: dict):
        if context["A_inv"] is None: return x
        B, _, C = x.shape
        r = context["A_inv"].shape[-1]
        
        x_reshaped = x.reshape(B, self.num_tiles, r, C)
        x_unmerged = context["A_inv"] @ x_reshaped
        return ToMADitBlockHook.unfold_with_indices(x_unmerged, self.h_tile_num, self.w_tile_num, context["original_N"])

    def on_register_hook(self, block: nn.Module):
        if self.model_type == "wan":
            raise NotImplementedError("wan model is not supported")
        
        if isinstance(block, FluxSingleTransformerBlock):
            for key in ("self-attn", "cross-attn", "ffn"):
                if self.merge_config.get(key):
                    self.merge_config[key] = False
        else:
            if self.merge_config.get("parallel"):
                self.merge_config["parallel"] = False

        self.enable = any(self.merge_config.values())

        if self.layer_idx < self.skip_layer_nums:
            self.enable = False
        if self.single_layer_idx is not None and self.single_layer_idx < (self.skip_single_layer_nums or 0):
            self.enable = False

    def hook_pre_norm(
        self,
        module_name,
        hidden_states,
        encoder_hidden_states=None,
        image_rotary_emb=None,
        temb=None,
    ):
        if module_name in ("self-attn", "parallel"):
            self._clean()
        if module_name in ("self-attn", "parallel") and self._merge_this_step():
            def _gather_rope_with_tile(rope_pair, dst_idx):
                if rope_pair is None: return None
                res = []
                for x in rope_pair:
                    if x.ndim == 2: x = x.unsqueeze(0).expand(dst_idx.shape[0], -1, -1)
                    x_folded = ToMADitBlockHook.fold_with_indices(x, self.h_tile_num, self.w_tile_num)
                    idx_expanded = dst_idx.unsqueeze(-1).expand(-1, -1, -1, x.shape[-1])
                    x_gathered = torch.gather(x_folded, 2, idx_expanded)
                    gathered = x_gathered.reshape(x_gathered.shape[0], -1, x_gathered.shape[-1])
                    if self.model_type == "flux-dev":
                        if gathered.shape[0] != 1:
                            raise ValueError("Flux ToMA currently supports batch=1")
                        gathered = gathered[0]
                    res.append(gathered)
                return tuple(res)

            if global_context.hidden_extra_seq_len is not None:
                h_img = hidden_states[:, : global_context.hidden_seq_len, :]
                h_extra = hidden_states[:, global_context.hidden_seq_len :, :]

                self.context_dict["dst_idx"], self.context_dict["A"], self.context_dict["A_inv"] = \
                    self.compute_partition(h_img, self.ratio)
                
                self.extra_context_dict["dst_idx"], self.extra_context_dict["A"], self.extra_context_dict["A_inv"] = \
                    self.compute_partition(h_extra, self.ratio)
            else:
                self.context_dict["dst_idx"], self.context_dict["A"], self.context_dict["A_inv"] = \
                    self.compute_partition(hidden_states, self.ratio)

            if image_rotary_emb is not None and (
                self.merge_config.get("self-attn") or self.merge_config.get("parallel")
            ):
                hidden_emb, encoder_emb, extra_emb = parse_image_rotary_emb(
                    self.model_type,
                    image_rotary_emb,
                    global_context.hidden_seq_len,
                    global_context.encoder_hidden_seq_len_gen(),
                )
                hidden_emb = _gather_rope_with_tile(hidden_emb, self.context_dict["dst_idx"])
                if extra_emb is not None:
                    extra_emb = _gather_rope_with_tile(extra_emb, self.extra_context_dict["dst_idx"])
                image_rotary_emb = join_image_rotary_emb(
                    self.model_type, hidden_emb, encoder_emb, extra_emb
                )

        return super().hook_pre_norm(module_name, hidden_states, encoder_hidden_states, image_rotary_emb, temb)

    def hook_after_norm(
        self,
        module_name,
        norm_hidden_states,
        norm_encoder_hidden_states=None,
        image_rotary_emb=None,
    ):
        if self.merge_config.get(module_name, False) and self._merge_this_step():

            if global_context.hidden_extra_seq_len is not None:
                h1, h2 = norm_hidden_states.split(
                    [global_context.hidden_seq_len, norm_hidden_states.shape[1] - global_context.hidden_seq_len], dim=1
                )
                norm_hidden_states = torch.cat(
                    [self.merge(h1, self.context_dict), self.merge(h2, self.extra_context_dict)], dim=1
                )
            else:
                norm_hidden_states = self.merge(norm_hidden_states, self.context_dict)

        return super().hook_after_norm(module_name, norm_hidden_states, norm_encoder_hidden_states, image_rotary_emb)

    def hook_after_module(
        self,
        module_name,
        hidden_output,
        encoder_hidden_output=None,
    ):
        if self.merge_config.get(module_name, False) and self._merge_this_step():
            if global_context.hidden_extra_seq_len is not None:
                r = self.context_dict["A"].shape[2]
                split_len = self.num_tiles * r
                
                h1, h2 = hidden_output.split([split_len, hidden_output.shape[1] - split_len], dim=1)
                hidden_output = torch.cat(
                    [self.unmerge(h1, self.context_dict), self.unmerge(h2, self.extra_context_dict)], dim=1
                )
            else:
                hidden_output = self.unmerge(hidden_output, self.context_dict)

        if module_name in ("ffn", "parallel"):
            self._clean()

        return super().hook_after_module(module_name, hidden_output, encoder_hidden_output)

if __name__ == "__main__":
    model_type = "sd3"
    pipe, model_attrs = get_pipeline(model_type)

    toma_config = {
        "tile_nums": (8, 8),
        "merged_ratio": 0.25,
        "merge_config": {
            "self-attn": True,
            "cross-attn": False,
            "ffn": True,
            "parallel": False,
        },
        "skip_layer_nums": 10,
        "skip_single_layer_nums": None,
    }

    HookManager.configure_hooks(
        model_type=model_type,
        default=(ToMADitBlockHook, toma_config),
    )

    with HookManager.enabled():
        HookManager.inject_hooks(
            model_type=model_type, pipe=pipe, model_attrs=model_attrs
        )
        res = process_quick_start(pipe)
        res.save(f"{model_type}_toma.png")

    print(global_context)
