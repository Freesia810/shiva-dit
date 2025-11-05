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
        **kwargs,
    ):
        super().__init__()
        self.tile_nums = tile_nums
        self.ratio = merged_ratio
        if merge_config is None:
            self.merge_config = {
                "self-attn": True,
                "cross-attn": True,
                "ffn": True,
                "parallel": True,
            }
        else:
            self.merge_config = merge_config

        self.size = None  # dynamic attr
        self.k = None
        self.context_dict = {
            "dst_idx": None,
            "A": None,
            "A_inv": None,
        }

        # only work for edit task with dual hidden states
        self.extra_size = None
        self.extra_k = None
        self.extra_context_dict = {
            "dst_idx": None,
            "A": None,
            "A_inv": None,
        }

        self.skip_layer_nums = skip_layer_nums
        self.skip_single_layer_nums = skip_single_layer_nums

    @torch.no_grad()
    @staticmethod
    def facility_location(x: torch.Tensor, k: int) -> torch.Tensor:
        """
        x: (B, N, C)
        """
        B, N, C = x.shape
        device = x.device

        x_normalized = F.normalize(x, p=2, dim=-1, eps=1e-8)
        scores = x_normalized @ x_normalized.transpose(-1, -2)  # (B, N, N)

        row_sums = torch.sum(scores, dim=2)  # (B, N)
        init_v = torch.argmax(row_sums, dim=1)  # (B,)

        max_sim = scores[torch.arange(B, device=device), init_v]  # (B, N)

        res = torch.zeros(B, k, dtype=torch.long, device=device)  # (B, k)
        res[:, 0] = init_v

        for i in range(1, k):
            differences = torch.relu(scores - max_sim.unsqueeze(1))
            row_sums = torch.sum(differences, dim=2)  # (B, N)
            row_sums.scatter_(dim=1, index=res[:, :i], value=-torch.inf)
            next_v = torch.argmax(row_sums, dim=1)  # (B,)
            res[:, i] = next_v
            new_sims = scores[torch.arange(B, device=device), next_v]  # (B, N)
            max_sim = torch.max(max_sim, new_sims)

        return res  # (B, k)

    @torch.no_grad()
    @staticmethod
    def tile_facility_location(
        x: torch.Tensor,
        k: int,
        size: Union[int, Tuple[int, int]],
        tile_nums: Union[int, Tuple[int, int]],
    ) -> torch.Tensor:
        """
        x: (B, N, C)
        size: (h, w)
        tile_nums: (h_tile_num, w_tile_num)

        we add a sort to guard the original order of selected indices
        """
        if isinstance(size, int):
            h = w = size
        else:
            h, w = size

        if isinstance(tile_nums, int):
            h_tile_num, w_tile_num = tile_nums, tile_nums
        else:
            h_tile_num, w_tile_num = tile_nums

        h_tile_stride = h // h_tile_num
        w_tile_stride = w // w_tile_num
        k_tile = k // (h_tile_num * w_tile_num)

        B, N, C = x.shape

        indices = torch.arange(N, device=x.device).unsqueeze(0).expand(B, -1).reshape(B, h, w)  # (B, h, w)
        patch_indices = indices.unfold(1, h_tile_stride, h_tile_stride).unfold(
            2, w_tile_stride, w_tile_stride
        )  # (B, h_tile_num, w_tile_num, h_tile_stride, w_tile_stride)
        x_patch = (
            x.reshape(B, h, w, C)
            .unfold(1, h_tile_stride, h_tile_stride)
            .unfold(2, w_tile_stride, w_tile_stride)
            .permute(0, 1, 2, 4, 5, 3)
        )  # (B, h_tile_num, w_tile_num, h_tile_stride, w_tile_stride, C)
        x_batched_patch = x_patch.reshape(
            B * h_tile_num * w_tile_num, h_tile_stride * w_tile_stride, C
        )  # (B * h_tile_num * w_tile_num, h_tile_stride * w_tile_stride, C)

        selected_x_batched_patch = ToMADitBlockHook.facility_location(
            x_batched_patch, k_tile
        )  # (B * h_tile_num * w_tile_num, k_tile)
        selected_x_batched_patch_indices = torch.gather(
            patch_indices.reshape(-1, h_tile_stride * w_tile_stride),
            -1,
            selected_x_batched_patch,
        )  # (B * h_tile_num * w_tile_num, k_tile)

        dst_indices = selected_x_batched_patch_indices.reshape(
            B, -1
        )  # (B, h_tile_num * w_tile_num * k_tile)
        dst_indices = dst_indices[:, :k].unsqueeze(-1)  # (B, k, 1)
        dst_indices = dst_indices.sort(dim=-2).values  # (B, k, 1)
        return dst_indices

    @torch.no_grad()
    @staticmethod
    def calculate_attn_tensor(x: torch.Tensor, dst_idx: torch.Tensor):
        """
        x: (B, N, C)
        dst_idx: (B, k, 1)
        """

        B, N, C = x.shape
        eps = 1e-6

        dst = torch.gather(x, 1, dst_idx.expand(B, -1, C))  # (B, k, C)
        q = dst  # (B, k, C)
        k = x # (B, N, C)
        A = q @ k.transpose(-1, -2) # (B, k, N)

        res = torch.softmax(A, dim=-2)
        sum = torch.sum(res, dim=-1, keepdim=True) # (B, k, 1)
        res = res / (sum + eps)

        return res, res.transpose(-1, -2)  # (B, k, N), (B, N, k)

    def _clean(self):
        self.context_dict = {
            "A": None,
            "A_inv": None,
            "dst_idx": None,
        }
        self.extra_context_dict = {
            "A": None,
            "A_inv": None,
            "dst_idx": None,
        }

    @torch.no_grad()
    def merge(self, x: torch.Tensor, context: dict):
        """
        x: (B, N, C)
        A: (B, k, N)
        """
        return torch.bmm(context["A"], x)

    @torch.no_grad()
    def unmerge(self, merged_x: torch.Tensor, context: dict):
        """
        x: (B, k, C)
        A_inv: (B, N, k)
        """
        return torch.bmm(context["A_inv"], merged_x)

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
        if self.single_layer_idx is not None and self.single_layer_idx < self.skip_single_layer_nums:
            self.enable = False

    def hook_pre_norm(
        self,
        module_name: Literal["self-attn", "cross-attn", "ffn", "parallel"],
        hidden_states: torch.Tensor,
        encoder_hidden_states: torch.Tensor = None,
        image_rotary_emb: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
        temb: torch.Tensor = None,
    ):
        if module_name in ("self-attn", "parallel") and self.enable:
            # only sample and select tokens when self-attention's start
            def _select(x, k, size, context_dict):
                context_dict["dst_idx"] = ToMADitBlockHook.tile_facility_location(
                    x, k, size, self.tile_nums
                )
                context_dict["A"], context_dict["A_inv"] = (
                    ToMADitBlockHook.calculate_attn_tensor(x, context_dict["dst_idx"])
                )

            def _select_image_rotary_emb(
                image_rotary_emb, encoder_hidden_seq_len, hidden_seq_len
            ):
                hidden_emb, encoder_emb, extra_emb = parse_image_rotary_emb(
                    self.model_type,
                    image_rotary_emb,
                    hidden_seq_len,
                    encoder_hidden_seq_len,
                )

                def _gather_pair(rotary_pair, context_dict):
                    if rotary_pair is None:
                        return None
                    return tuple(
                        torch.gather(
                            x,
                            0,
                            context_dict["dst_idx"].squeeze(0).expand(-1, x.shape[-1]),
                        )
                        for x in rotary_pair
                    )

                hidden_emb = _gather_pair(hidden_emb, self.context_dict)
                if extra_emb is not None:
                    extra_emb = _gather_pair(extra_emb, self.extra_context_dict)
                return join_image_rotary_emb(
                    self.model_type, hidden_emb, encoder_emb, extra_emb
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
                    self.k = int(global_context.hidden_seq_len * (1 - self.ratio))
                if self.extra_size is None:
                    self.extra_size = (
                        global_context.hidden_extra_seq_height,
                        global_context.hidden_extra_seq_width,
                    )
                    self.extra_k = int(
                        global_context.hidden_extra_seq_len * (1 - self.ratio)
                    )
                _select(
                    hidden_states[:, : global_context.hidden_seq_len, :],
                    self.k,
                    self.size,
                    self.context_dict,
                )
                _select(
                    hidden_states[:, global_context.hidden_seq_len :, :],
                    self.extra_k,
                    self.extra_size,
                    self.extra_context_dict,
                )
            else:
                # normal case
                if (
                    global_context.hidden_seq_height is None
                    or global_context.hidden_seq_width is None
                ):
                    raise ValueError(
                        "hidden_seq_height and hidden_seq_width must be set in global_context"
                    )
                if self.size is None:
                    self.size = (
                        global_context.hidden_seq_height,
                        global_context.hidden_seq_width,
                    )
                    self.k = int(global_context.hidden_seq_len * (1 - self.ratio))
                _select(hidden_states, self.k, self.size, self.context_dict)

            if image_rotary_emb is not None and (
                self.merge_config.get("self-attn") or self.merge_config.get("parallel")
            ):
                image_rotary_emb = _select_image_rotary_emb(
                    image_rotary_emb,
                    global_context.encoder_hidden_seq_len_gen(),
                    global_context.hidden_seq_len,
                )

        return super().hook_pre_norm(
            module_name, hidden_states, encoder_hidden_states, image_rotary_emb, temb
        )

    def hook_after_norm(
        self,
        module_name: Literal["self-attn", "cross-attn", "ffn", "parallel"],
        norm_hidden_states: torch.Tensor,
        norm_encoder_hidden_states: torch.Tensor = None,
        image_rotary_emb: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
    ):
        if self.merge_config.get(module_name, False) and self.enable:
            if global_context.hidden_extra_seq_len is not None:
                # dual hidden states for edit task
                h1, h2 = norm_hidden_states.split(
                    [
                        global_context.hidden_seq_len,
                        norm_hidden_states.shape[1] - global_context.hidden_seq_len,
                    ],
                    dim=1,
                )
                norm_hidden_states = torch.cat(
                    [
                        self.merge(h1, self.context_dict),
                        self.merge(h2, self.extra_context_dict),
                    ],
                    dim=1,
                )
            else:
                # normal case
                norm_hidden_states = self.merge(norm_hidden_states, self.context_dict)

        return super().hook_after_norm(
            module_name,
            norm_hidden_states,
            norm_encoder_hidden_states,
            image_rotary_emb,
        )

    def hook_after_module(
        self,
        module_name: Literal["self-attn", "cross-attn", "ffn", "parallel"],
        hidden_output: torch.Tensor,
        encoder_hidden_output: torch.Tensor = None,
    ):
        if self.merge_config.get(module_name, False) and self.enable:
            if global_context.hidden_extra_seq_len is not None:
                # dual hidden states for edit task
                hidden_seq_len = self.context_dict["dst_idx"].shape[1]
                h1, h2 = hidden_output.split(
                    [hidden_seq_len, hidden_output.shape[1] - hidden_seq_len], dim=1
                )
                hidden_output = torch.cat(
                    [
                        self.unmerge(h1, self.context_dict),
                        self.unmerge(h2, self.extra_context_dict),
                    ],
                    dim=1,
                )
            else:
                # normal case
                hidden_output = self.unmerge(hidden_output, self.context_dict)

        if module_name in ("ffn", "parallel"):
            # clear indices after ffn
            self._clean()

        return super().hook_after_module(
            module_name, hidden_output, encoder_hidden_output
        )


if __name__ == "__main__":
    model_type = "flux-dev"
    pipe, model_attrs = get_pipeline(model_type)

    toma_config = {
        "tile_nums": (8, 8),
        "merged_ratio": 0.8,
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
