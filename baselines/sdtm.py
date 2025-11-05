import math
import torch
import torch.nn.functional as F
from typing import Union, Tuple, Literal, Optional
from shiva_dit.hook import BaseDiTBlockHook, HookManager, BaseDelegator
from shiva_dit.utils import (
    global_context,
    merge_dst_src_tokens,
    get_dst_src_tokens,
    sample_dst_src_with_2D_stride,
    select_dst_src_tokens,
    parse_image_rotary_emb,
    join_image_rotary_emb,
    get_pipeline,
    process_quick_start,
)


class SDTMDiTBlockHook(BaseDelegator, BaseDiTBlockHook):
    """
    Reference:
        - Paper: https://arxiv.org/pdf/2505.11707

    Implemented from https://github.com/ICTMCG/SDTM
    """

    def __init__(
        self,
        stride: Union[int, Tuple[int, int]] = (2, 2),
        switch_step_fraction: float = 0.4,
        init_ratio: float = 0.5,
        deviation: float = 0.2,
        alpha_s=0.05,
        mcw: float = 0.2,
        protected_step_stride=3,
        generator: torch.Generator = None,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.stride = stride

        # calculrate reduce num params
        self.switch_step_fraction = switch_step_fraction
        self.init_ratio = init_ratio  # ρ in paper
        self.deviation = deviation  # d in paper

        # calculrate SSM scores
        self.alpha_s = alpha_s

        self.protected_step_stride = protected_step_stride

        self.generator = generator

        # dynamic attr
        self.size = None
        self.indices_dict = {}

        # only work for edit task with dual hidden states
        self.extra_start_idx = None
        self.extra_size = None
        self.extra_indices_dict = {}

        # cache for unmerge
        self.mcw = mcw
        self.prev_output = {
            "self-attn": None,
            "cross-attn": None,
            "ffn": None,
            "parallel": None,
        }
        self.prev_output_extra = {
            "self-attn": None,
            "cross-attn": None,
            "ffn": None,
            "parallel": None,
        }

        self.mode_flag = None  # 'SSM' or 'IDM'

    def _clean(self):
        self.indices_dict = {}
        self.extra_indices_dict = {}
        self.mode_flag = None

    def calculate_reduce_num(self, seq_len: int) -> int:
        if global_context.num_inference_steps > 1:
            progress = global_context.current_iteration / (
                global_context.num_inference_steps - 1
            )
        else:
            progress = 0.0

        alpha = math.cos(progress * math.pi / 2)
        ratio_current = (self.init_ratio - self.deviation) + (
            2.0 * self.deviation
        ) * alpha

        reduce_num = int(seq_len * float(ratio_current))
        return reduce_num

    @torch.no_grad()
    @staticmethod
    def calculate_sims_patched(
        x: torch.Tensor,
        size: Union[int, Tuple[int, int]],
        stride: Union[int, Tuple[int, int]] = (2, 2),
    ):
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

        x = x.view(B, h, w, C)  # (B, h, w, C)
        x_patched = x.view(
            B, h_patch, stride_h, w_patch, stride_w, C
        )  # (B, h_patch, stride_h, w_patch, stride_w, C)
        x_patched = x_patched.permute(
            0, 1, 3, 5, 2, 4
        )  # (B, h_patch, w_patch, C, stride_h, stride_w)
        x_flat = x_patched.reshape(
            B, h_patch, w_patch, C, num_tokens_in_patch
        )  # (B, h_patch, w_patch, C, K)

        x_flat_1 = x_flat.unsqueeze(-1)  # (B, h_patch, w_patch, C, K, 1)
        x_flat_2 = x_flat.unsqueeze(-2)  # (B, h_patch, w_patch, C, 1, K)
        sims = F.cosine_similarity(
            x_flat_1, x_flat_2, dim=3
        )  # (B, h_patch, w_patch, K, K)

        # for paper's formula, we need to divide by K * (K - 1), but we found that the origin code makes a division by K * K
        similarity_map = sims.sum(dim=(-2, -1)) / (
            num_tokens_in_patch * (num_tokens_in_patch - 1)
        )  # (B, h_patch, w_patch)
        similarity_map_flat = similarity_map.reshape(B, -1)  # (B, h_patch * w_patch)

        return similarity_map_flat

    @torch.no_grad()
    @staticmethod
    def calculate_freq_scores_patched(
        size: Union[int, Tuple[int, int]],
        stride: Union[int, Tuple[int, int]] = (2, 2),
        dtype: torch.dtype = torch.float32,
    ):
        last_independent = HookManager.get_global("last_independent", None)  # (B, N)
        if last_independent is None:
            return None

        B, _ = last_independent.shape

        if isinstance(size, int):
            h = w = size
        else:
            h, w = size

        if isinstance(stride, int):
            stride_h = stride_w = stride
        else:
            stride_h, stride_w = stride

        h_patch = h // stride_h
        w_patch = w // stride_w

        eps = 1e-6
        last_independent_f = last_independent.to(dtype)
        mean_last_independent = last_independent_f.mean(dim=1, keepdim=True) + eps
        indiv_priority = last_independent_f / mean_last_independent  # (B, N)
        indiv_priority_grid = indiv_priority.view(B, h, w)  # (B, H, W)
        indiv_priority_windows = indiv_priority_grid.view(
            B, h_patch, stride_h, w_patch, stride_w
        )  # (B, h_patch, stride_h, w_patch, stride_w)
        indiv_priority_windows = indiv_priority_windows.permute(
            0, 1, 3, 2, 4
        )  # (B, h_patch, w_patch, stride_h, stride_w)
        indiv_priority_windows = indiv_priority_windows.reshape(
            B, h_patch, w_patch, stride_h * stride_w
        )  # (B, h_patch, w_patch, stride_h * stride_w)
        indiv_priority_mean = indiv_priority_windows.mean(
            dim=-1
        )  # (B, h_patch, w_patch)
        indiv_priority_flat = indiv_priority_mean.reshape(
            B, -1
        )  # (B, h_patch * w_patch)

        return indiv_priority_flat

    @torch.no_grad()
    @staticmethod
    def update_freq(independent_indices: torch.Tensor, is_extra: bool = False):
        """
        freq is shared by all DiT Blocks
        """
        if is_extra:
            last_independent = HookManager.get_global("extra_last_independent", None)
            if last_independent is None:
                HookManager.set_global(
                    "extra_last_independent",
                    torch.zeros(
                        independent_indices.shape[0],
                        global_context.hidden_extra_seq_len,
                        device=independent_indices.device,
                        dtype=torch.int32,
                    ),
                )
                last_independent = HookManager.get_global("extra_last_independent")
        else:
            last_independent = HookManager.get_global("last_independent", None)
            if last_independent is None:
                HookManager.set_global(
                    "last_independent",
                    torch.zeros(
                        independent_indices.shape[0],
                        global_context.hidden_seq_len,
                        device=independent_indices.device,
                        dtype=torch.int32,
                    ),
                )
                last_independent = HookManager.get_global("last_independent")

        last_independent.add_(1)
        zeros_src = torch.zeros_like(independent_indices, dtype=last_independent.dtype)
        last_independent.scatter_(1, independent_indices, zeros_src)

    @torch.no_grad()
    @staticmethod
    def select_window_tokens(
        scores: torch.Tensor,
        reduce_num: int,
        size: Union[int, Tuple[int, int]],
        stride: Union[int, Tuple[int, int]] = (2, 2),
        rand: bool = True,
        generator: torch.Generator = None,
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

        # from original code
        reduce_num = reduce_num // 48 * 16

        _, selected_window_flat_indices = scores.topk(
            reduce_num, dim=-1
        )  # (B, reduce_num)

        tensor = torch.arange(h * w, device=scores.device).reshape(h, w)  # (H, W)
        tensor = tensor.unsqueeze(0).repeat(B, 1, 1)  # (B, H, W)
        windowed_tensor = tensor.unfold(1, stride_h, stride_h).unfold(
            2, stride_w, stride_w
        )  # (B, h_patch, w_patch, stride_h, stride_w)
        windowed_tensor = windowed_tensor.reshape(
            B, -1, stride_h * stride_w
        )  # (B, h_patch * w_patch, stride_h * stride_w)

        mask = torch.ones(
            (B, windowed_tensor.shape[1]), dtype=bool, device=scores.device
        )  # (B, h_patch * w_patch)
        mask_values = torch.zeros_like(
            selected_window_flat_indices, dtype=torch.bool, device=scores.device
        )  # (B, reduce_num)
        mask.scatter_(
            1, selected_window_flat_indices, mask_values
        )  # (B, h_patch * w_patch) 0 for selected windows, 1 for unselected windows
        remaining_tensor = windowed_tensor[
            mask.unsqueeze(-1).expand(-1, -1, stride_h * stride_w)
        ].reshape(
            B, -1, stride_h * stride_w
        )  # (B, h_patch * w_patch - reduce_num, stride_h * stride_w)
        unmerge_idx = remaining_tensor.reshape(B, -1).unsqueeze(
            -1
        )  # (B, (h_patch * w_patch - reduce_num) * stride_h * stride_w, 1)

        selected_window_indices = torch.gather(
            windowed_tensor,
            1,
            selected_window_flat_indices.unsqueeze(-1).expand(
                -1, -1, stride_h * stride_w
            ),
        )  # (B, reduce_num, stride_h * stride_w)
        full = torch.arange(stride_h * stride_w, device=scores.device)
        matrix = full.unsqueeze(0).expand(
            stride_h * stride_w, stride_h * stride_w
        )  # (stride_h * stride_w, stride_h * stride_w)
        mask = torch.eye(
            stride_h * stride_w, dtype=torch.bool, device=scores.device
        )  # (stride_h * stride_w, stride_h * stride_w)
        src_table = matrix[~mask].view(
            stride_h * stride_w, stride_h * stride_w - 1
        )  # (stride_h * stride_w, stride_h * stride_w - 1)

        if rand:
            rand_idx = torch.randint(
                stride_h * stride_w,
                (B, reduce_num, 1),
                device=scores.device,
                generator=generator,
            )  # (B, reduce_num, 1)
        else:
            rand_idx = torch.zeros(
                B, reduce_num, 1, dtype=torch.long, device=scores.device
            )  # (B, reduce_num, 1)

        dst_idx = torch.gather(
            selected_window_indices, 2, rand_idx
        )  # (B, reduce_num, 1)
        src_pos = src_table[
            rand_idx.squeeze(-1)
        ]  # (B, reduce_num, stride_h * stride_w - 1)
        src_vals = torch.gather(
            selected_window_indices, 2, src_pos
        )  # (B, reduce_num, stride_h * stride_w - 1)
        src_idx = src_vals.reshape(B, reduce_num * (stride_h * stride_w - 1)).unsqueeze(
            -1
        )  # (B, reduce_num * (stride_h * stride_w - 1), 1)
        merge_idx = (
            torch.arange(reduce_num, device=scores.device)
            .repeat_interleave(stride_h * stride_w - 1)
            .repeat(B, 1)
            .unsqueeze(-1)
        )  # (B, reduce_num * (stride_h * stride_w - 1), 1)

        return dst_idx, src_idx, merge_idx, unmerge_idx

    @torch.no_grad()
    def process_SSM(
        self,
        x: torch.Tensor,
        indices_dict: dict,
        size: Union[int, Tuple[int, int]],
        reduce_num: int,
    ):
        similarity_map_flat = SDTMDiTBlockHook.calculate_sims_patched(
            x, size, self.stride
        )  # (B, h_patch * w_patch)
        indiv_priority_flat = SDTMDiTBlockHook.calculate_freq_scores_patched(
            size, self.stride, similarity_map_flat.dtype
        )  # (B, h_patch * w_patch)
        if indiv_priority_flat is not None:
            similarity_map_flat = (
                similarity_map_flat + self.alpha_s * indiv_priority_flat
            )  # (B, h_patch * w_patch)

        # dst_idx       (B, reduce_num, 1)
        # src_idx       (B, reduce_num * (stride_h * stride_w - 1), 1)
        # merge_idx     (B, reduce_num * (stride_h * stride_w - 1), 1)
        # unmerge_idx   (B, (h_patch * w_patch - reduce_num) * stride_h * stride_w, 1)
        (
            indices_dict["dst_idx"],
            indices_dict["src_idx"],
            indices_dict["merge_idx"],
            indices_dict["unmerge_idx"],
        ) = SDTMDiTBlockHook.select_window_tokens(
            similarity_map_flat,
            reduce_num,
            size,
            self.stride,
            rand=True,
            generator=self.generator,
        )

    @torch.no_grad()
    @staticmethod
    def select_image_rotary_emb(
        img_rotary_emb: torch.Tensor, indices_dict: dict
    ) -> torch.Tensor:
        # img_rotary_emb: (N, D)
        # dst_idx       (B, reduce_num, 1)
        # src_idx       (B, reduce_num * (stride_h * stride_w - 1), 1)
        # merge_idx     (B, reduce_num * (stride_h * stride_w - 1), 1)
        # unmerge_idx   (B, (h_patch * w_patch - reduce_num) * stride_h * stride_w, 1)
        # B should be 1 because rotary emb is passed without batch dimension
        if img_rotary_emb.dim() == 2:
            img_rotary_emb = img_rotary_emb.unsqueeze(0)  # (1, N, D)

        B, _, C = img_rotary_emb.shape
        reduce_num = indices_dict["dst_idx"].shape[1]
        unmerge_dim = indices_dict["unmerge_idx"].shape[1]

        unm = torch.gather(
            img_rotary_emb,
            dim=-2,
            index=indices_dict["unmerge_idx"].expand(B, unmerge_dim, C),
        )  # (B, (h_patch * w_patch - reduce_num) * stride_h * stride_w, C)

        dst = torch.gather(
            img_rotary_emb,
            dim=-2,
            index=indices_dict["dst_idx"].expand(B, reduce_num, C),
        )  # (B, reduce_num, C)

        img_rotary_emb = torch.cat([unm, dst], dim=1)
        if B == 1:
            return img_rotary_emb.squeeze(0)  # (N - num_dst - k, D)
        return img_rotary_emb  # (B, N - num_dst - k, D)

    @torch.no_grad()
    def merge_SSM(
        self,
        x: torch.Tensor,
        indices_dict: dict,
        reduce_mode: str = "mean",
    ) -> torch.Tensor:
        # dst_idx       (B, reduce_num, 1)
        # src_idx       (B, reduce_num * (stride_h * stride_w - 1), 1)
        # merge_idx     (B, reduce_num * (stride_h * stride_w - 1), 1)
        # unmerge_idx   (B, (h_patch * w_patch - reduce_num) * stride_h * stride_w, 1)
        B, N, C = x.shape
        reduce_num = indices_dict["dst_idx"].shape[1]
        src_dim = indices_dict["src_idx"].shape[1]
        unmerge_dim = indices_dict["unmerge_idx"].shape[1]

        src = torch.gather(
            x,
            dim=-2,
            index=indices_dict["src_idx"].expand(B, src_dim, C),
        )
        dst = torch.gather(
            x, dim=-2, index=indices_dict["dst_idx"].expand(B, reduce_num, C)
        )
        unm = torch.gather(
            x,
            dim=-2,
            index=indices_dict["unmerge_idx"].expand(B, unmerge_dim, C),
        )  # (B, (h_patch * w_patch - reduce_num) * stride_h * stride_w, C)
        dst = dst.scatter_reduce(
            -2,
            indices_dict["merge_idx"].expand(B, src_dim, C),
            src,
            reduce=reduce_mode,
        )  # (B, reduce_num, C)
        x = torch.cat([unm, dst], dim=1)
        return x

    def unmerge_SSM(
        self,
        x: torch.Tensor,
        indices_dict: dict,
        prev_output: torch.Tensor = None,
    ) -> torch.Tensor:
        # dst_idx       (B, reduce_num, 1)
        # src_idx       (B, reduce_num * (stride_h * stride_w - 1), 1)
        # merge_idx     (B, reduce_num * (stride_h * stride_w - 1), 1)
        # unmerge_idx   (B, (h_patch * w_patch - reduce_num) * stride_h * stride_w, 1)
        B, _, C = x.shape
        unm_len = indices_dict["unmerge_idx"].shape[1]
        unm, dst = x[..., :unm_len, :], x[..., unm_len:, :]
        reduce_num = indices_dict["dst_idx"].shape[1]
        src_dim = indices_dict["src_idx"].shape[1]
        unmerge_dim = indices_dict["unmerge_idx"].shape[1]
        total_seq_len = reduce_num + unmerge_dim + src_dim

        src = torch.gather(
            dst,
            dim=-2,
            index=indices_dict["merge_idx"].expand(B, src_dim, C),
        )

        if prev_output is not None:
            prev_output = prev_output.to(device=x.device, dtype=x.dtype)
            cached_dst = torch.gather(
                prev_output,
                dim=-2,
                index=indices_dict["dst_idx"].expand(B, reduce_num, C),
            )

            cached_src = torch.gather(
                prev_output,
                dim=-2,
                index=indices_dict["src_idx"].expand(B, src_dim, C),
            )

            dst = self.mcw * dst + (1.0 - self.mcw) * cached_dst
            src = self.mcw * src + (1.0 - self.mcw) * cached_src

        out = torch.zeros(B, total_seq_len, C, device=x.device, dtype=x.dtype)
        out.scatter_(
            dim=-2, index=indices_dict["dst_idx"].expand(B, reduce_num, C), src=dst
        )
        out.scatter_(
            dim=-2, index=indices_dict["unmerge_idx"].expand(B, unmerge_dim, C), src=unm
        )
        out.scatter_(
            dim=-2, index=indices_dict["src_idx"].expand(B, src_dim, C), src=src
        )
        return out

    @torch.no_grad()
    @staticmethod
    def calculate_scores(
        x: torch.Tensor, dst_idx: torch.Tensor, src_idx: torch.Tensor
    ) -> torch.Tensor:
        # the same as ToMeSD
        # x: (B, N, D)
        # dst_idx: (1, num_dst, 1)
        # src_idx: (1, N - num_dst, 1)
        # return: (B, num_src, num_dst)
        x = x / x.norm(dim=-1, keepdim=True)
        dst_tokens, src_tokens = get_dst_src_tokens(
            x, dst_idx, src_idx
        )  # (B, num_dst, D), (B, N - num_dst, D)
        scores = src_tokens @ dst_tokens.transpose(-1, -2)  # (B, N - num_dst, num_dst)

        return scores

    @torch.no_grad()
    def process_IDM(
        self,
        x: torch.Tensor,
        size: Union[int, Tuple[int, int]],
        stride: Union[int, Tuple[int, int]],
        reduce_num: int,
        indices_dict: dict,
        device: torch.device,
        rand: bool = True,
        generator: torch.Generator = None,
    ):
        """
        x: (B, S, C)
        attn_output: (B, S, C)

        In original code, IDM is only applied to MLP, and we only find it works by calculating similarity on frequency scores.
        The values of attention map (N * N) is hard to hook, because they are calculated inside the attention kernel in a parrllel way.
        """
        last_independent = HookManager.get_global("last_independent", None)  # (B, N)
        B, _, C = x.shape

        if isinstance(size, int):
            h = w = size
        else:
            h, w = size

        if isinstance(stride, int):
            stride_h = stride_w = stride
        else:
            stride_h, stride_w = stride

        h_patch = h // stride_h
        w_patch = w // stride_w

        if last_independent is not None:
            last_independent_grid = last_independent.view(B, h, w)  # (B, H, W)
            last_independent_windows = last_independent_grid.view(
                B, h_patch, stride_h, w_patch, stride_w
            ).permute(
                0, 1, 3, 2, 4
            )  # (B, h_patch, w_patch, stride_h, stride_w)
            last_independent_windows_flat = last_independent_windows.reshape(
                B, h_patch, w_patch, stride_h * stride_w
            )  # (B, h_patch, w_patch, stride_h * stride_w)
            dst_pos = last_independent_windows_flat.argmax(
                dim=-1, keepdim=True
            )  # (B, h_patch, w_patch, 1)
        else:
            if rand:
                dst_pos = torch.randint(
                    stride_h * stride_w,
                    (B, h_patch, w_patch, 1),
                    device=device,
                    generator=generator,
                )  # (B, h_patch, w_patch, 1)
            else:
                dst_pos = torch.zeros(
                    B, h_patch, w_patch, 1, device=device, dtype=torch.int64
                )  # (B, h_patch, w_patch, 1)

        indices_dict["dst_idx"], indices_dict["src_idx"] = (
            sample_dst_src_with_2D_stride(
                size,
                self.stride,
                self.generator,
                device=x.device,
                partition_idx=dst_pos,
                batch_size=B,
            )
        )

        reduce_num = reduce_num // 16 * 16

        scores = SDTMDiTBlockHook.calculate_scores(
            x, indices_dict["dst_idx"], indices_dict["src_idx"]
        )  # (B, num_src, num_dst)

        (
            indices_dict["merge_dst_idx"],
            indices_dict["merge_src_idx"],
            indices_dict["unmerge_src_idx"],
        ) = select_dst_src_tokens(scores, reduce_num)

        indices_dict["unmerge_src_final_idx"] = torch.gather(
            indices_dict["src_idx"].expand(
                indices_dict["src_idx"].shape[0], indices_dict["src_idx"].shape[1], 1
            ),
            dim=1,
            index=indices_dict["unmerge_src_idx"],
        )
        indices_dict["merge_src_final_idx"] = torch.gather(
            indices_dict["src_idx"].expand(
                indices_dict["src_idx"].shape[0], indices_dict["src_idx"].shape[1], 1
            ),
            dim=1,
            index=indices_dict["merge_src_idx"],
        )

    @torch.no_grad()
    def merge_IDM(self, x: torch.Tensor, indices_dict: dict) -> torch.Tensor:
        # the same as ToMeSD
        return merge_dst_src_tokens(
            x,
            indices_dict["dst_idx"],
            indices_dict["src_idx"],
            indices_dict["merge_dst_idx"],
            indices_dict["merge_src_idx"],
            indices_dict["unmerge_src_idx"],
        )

    @torch.no_grad()
    def unmerge_IDM(
        self,
        merged_x: torch.Tensor,
        indices_dict: dict,
        prev_output: torch.Tensor = None,
    ) -> torch.Tensor:
        # merged_x: (B, N - k, D)
        # dst_idx: (B, num_dst, 1)
        # src_idx: (B, N - num_dst, 1)
        # merge_dst_idx: (B, k, 1)
        # merge_src_idx: (B, k, 1)
        # unmerge_src_idx: (B, N - num_dst - k, 1)
        B, N_k, C = merged_x.shape
        num_dst = indices_dict["dst_idx"].shape[1]
        k = indices_dict["merge_dst_idx"].shape[1]
        N = N_k + k
        unmerge_len = N - num_dst - k

        unmerge_src_tokens, merge_dst_tokens = (
            merged_x[..., :unmerge_len, :],
            merged_x[..., unmerge_len:, :],
        )  # (B, N - num_dst - k, D), (B, num_dst, D)
        simulated_merged_src_tokens = torch.gather(
            merge_dst_tokens,
            dim=-2,
            index=indices_dict["merge_dst_idx"].expand(B, k, C),
        )  # use dst tokens to simulate src tokens

        if prev_output is not None:
            prev_output = prev_output.to(device=merged_x.device, dtype=merged_x.dtype)
            cached_dst = torch.gather(
                prev_output,
                dim=-2,
                index=indices_dict["dst_idx"].expand(B, num_dst, C),
            )

            cached_src = torch.gather(
                prev_output,
                dim=-2,
                index=indices_dict["merge_src_final_idx"].expand(B, k, C),
            )
            merge_dst_tokens = (
                self.mcw * merge_dst_tokens + (1.0 - self.mcw) * cached_dst
            )
            simulated_merged_src_tokens = (
                self.mcw * simulated_merged_src_tokens + (1.0 - self.mcw) * cached_src
            )

        out = torch.zeros(
            B, N, C, device=merged_x.device, dtype=merged_x.dtype
        )  # (B, N, C)
        out.scatter_(
            dim=-2,
            index=indices_dict["dst_idx"].expand(B, num_dst, C),
            src=merge_dst_tokens,
        )
        out.scatter_(
            dim=-2,
            index=indices_dict["unmerge_src_final_idx"].expand(B, unmerge_len, C),
            src=unmerge_src_tokens,
        )
        out.scatter_(
            dim=-2,
            index=indices_dict["merge_src_final_idx"].expand(B, k, C),
            src=simulated_merged_src_tokens,
        )

        return out

    def delegated_call(self, func, module_self, *args, **kwargs):
        curr_iter = global_context.current_iteration
        switch_step = int(
            global_context.num_inference_steps * self.switch_step_fraction
        )
        if curr_iter <= switch_step:
            self.mode_flag = "SSM"
        else:
            self.mode_flag = "IDM"

        if (
            self.protected_step_stride > 0
            and curr_iter % self.protected_step_stride == 0
        ):
            self.mode_flag = None

        return func(module_self, *args, **kwargs)

    def hook_after_norm(
        self,
        module_name: Literal["self-attn", "cross-attn", "ffn", "parallel"],
        norm_hidden_states: torch.Tensor,
        norm_encoder_hidden_states: torch.Tensor = None,
        image_rotary_emb: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
    ):
        """
        Hook function called after normalization. Handles token processing (state update)
        and token merging (transformation) based on module name and mode flag.
        """
        final_hidden_states = norm_hidden_states  # Start with the input state

        # --- Part 1: Token Processing (Updates internal state like indices_dict) ---
        if self.mode_flag and module_name in ("self-attn", "parallel"):
            # Determine if we have dual hidden states (main + extra)
            has_extra = global_context.hidden_extra_seq_len is not None

            # --- Validation ---
            required_main = ["hidden_seq_height", "hidden_seq_width", "hidden_seq_len"]
            required_extra = [
                "hidden_extra_seq_height",
                "hidden_extra_seq_width",
                "hidden_extra_seq_len",
            ]
            required_attrs = required_main + (required_extra if has_extra else [])
            missing_attrs = [
                attr
                for attr in required_attrs
                if getattr(global_context, attr, None) is None
            ]
            if missing_attrs:
                raise ValueError(
                    f"{', '.join(missing_attrs)} must be set in global_context"
                )

            # --- Initialization & Calculations ---
            if self.size is None:
                self.size = (
                    global_context.hidden_seq_height,
                    global_context.hidden_seq_width,
                )
            seq_len = global_context.hidden_seq_len
            reduce_num = self.calculate_reduce_num(seq_len)

            # Prepare parameters for processing parts
            parts_params = []
            parts_params.append(
                {
                    "slice": norm_hidden_states[:, :seq_len, :],
                    "indices": self.indices_dict,
                    "size": self.size,
                    "reduce_num": reduce_num,
                }
            )

            if has_extra:
                if self.extra_size is None:
                    self.extra_size = (
                        global_context.hidden_extra_seq_height,
                        global_context.hidden_extra_seq_width,
                    )
                extra_reduce_num = self.calculate_reduce_num(
                    global_context.hidden_extra_seq_len
                )
                parts_params.append(
                    {
                        "slice": norm_hidden_states[:, seq_len:, :],
                        "indices": self.extra_indices_dict,
                        "size": self.extra_size,
                        "reduce_num": extra_reduce_num,
                    }
                )

            # --- Execute Processing (Updates self.indices_dict, self.extra_indices_dict) ---
            if self.mode_flag == "SSM":
                for params in parts_params:
                    self.process_SSM(
                        params["slice"],
                        params["indices"],  # Assumed to update this dict
                        params["size"],
                        params["reduce_num"],
                    )
            elif self.mode_flag == "IDM":
                common_idm_kwargs = {
                    "stride": self.stride,
                    "device": norm_hidden_states.device,
                    "rand": True,
                    "generator": self.generator,
                }
                for params in parts_params:
                    self.process_IDM(
                        params["slice"],
                        params["size"],
                        common_idm_kwargs["stride"],
                        params["reduce_num"],
                        params["indices"],  # Assumed to update this dict
                        common_idm_kwargs["device"],
                        rand=common_idm_kwargs["rand"],
                        generator=common_idm_kwargs["generator"],
                    )
            else:
                raise ValueError(f"Invalid mode_flag for processing: {self.mode_flag}")

            if has_extra:
                if self.mode_flag == "SSM":
                    SDTMDiTBlockHook.update_freq(
                        self.extra_indices_dict["unmerge_idx"].squeeze(-1),
                        is_extra=True,
                    )
                elif self.mode_flag == "IDM":
                    SDTMDiTBlockHook.update_freq(
                        self.extra_indices_dict["unmerge_src_final_idx"].squeeze(-1),
                        is_extra=True,
                    )
            else:
                if self.mode_flag == "SSM":
                    SDTMDiTBlockHook.update_freq(
                        self.indices_dict["unmerge_idx"].squeeze(-1)
                    )
                elif self.mode_flag == "IDM":
                    SDTMDiTBlockHook.update_freq(
                        self.indices_dict["unmerge_src_final_idx"].squeeze(-1)
                    )

            # Handle image rotary emb
            if image_rotary_emb is not None and self.mode_flag == "SSM":

                def _select_image_rotary_emb(
                    image_rotary_emb, encoder_hidden_seq_len, hidden_seq_len
                ):
                    hidden_emb, encoder_emb, extra_emb = parse_image_rotary_emb(
                        self.model_type,
                        image_rotary_emb,
                        hidden_seq_len,
                        encoder_hidden_seq_len,
                    )

                    def _gather_pair(rotary_pair, indices_dict):
                        if rotary_pair is None:
                            return None
                        return tuple(
                            SDTMDiTBlockHook.select_image_rotary_emb(x, indices_dict)
                            for x in rotary_pair
                        )

                    hidden_emb = _gather_pair(hidden_emb, self.indices_dict)
                    if extra_emb is not None:
                        extra_emb = _gather_pair(extra_emb, self.extra_indices_dict)
                    return join_image_rotary_emb(
                        self.model_type, hidden_emb, encoder_emb, extra_emb
                    )

                image_rotary_emb = _select_image_rotary_emb(
                    image_rotary_emb,
                    encoder_hidden_seq_len=global_context.encoder_hidden_seq_len_gen(),
                    hidden_seq_len=global_context.hidden_seq_len,
                )

        # --- Part 2: Token Merging (Transforms the hidden states) ---
        merge_func = None
        if self.mode_flag == "SSM" and module_name in (
            "ffn",
            "self-attn",
            "cross-attn",
            "parallel",
        ):
            merge_func = self.merge_SSM
        elif self.mode_flag == "IDM" and module_name == "ffn":
            merge_func = self.merge_IDM

        if merge_func is not None:
            has_extra = global_context.hidden_extra_seq_len is not None
            target_states_for_merge = norm_hidden_states  # Merge the current state

            if has_extra:
                # Re-validate seq_len existence for split
                if global_context.hidden_seq_len is None:
                    raise ValueError("hidden_seq_len must be set for merging split")

                seq_len = global_context.hidden_seq_len
                # Check if splitting is actually possible/needed
                if 0 < seq_len < target_states_for_merge.shape[1]:
                    h1, h2 = target_states_for_merge.split(
                        [seq_len, target_states_for_merge.shape[1] - seq_len], dim=1
                    )
                    # Apply merge using potentially updated indices from Part 1
                    merged_h1 = merge_func(h1, self.indices_dict)
                    merged_h2 = merge_func(h2, self.extra_indices_dict)
                    self.extra_start_idx = merged_h1.shape[1]
                    final_hidden_states = torch.cat([merged_h1, merged_h2], dim=1)
                else:
                    # If seq_len covers the whole tensor or is zero, merge as one block
                    final_hidden_states = merge_func(
                        target_states_for_merge, self.indices_dict
                    )
            else:
                # Apply merge to the whole tensor
                final_hidden_states = merge_func(
                    target_states_for_merge, self.indices_dict
                )

        # Pass the final state (potentially processed in Part 1 and merged in Part 2)
        return super().hook_after_norm(
            module_name,
            final_hidden_states,
            norm_encoder_hidden_states,
            image_rotary_emb,
        )

    def hook_after_module(
        self,
        module_name: Literal["self-attn", "cross-attn", "ffn", "parallel"],
        hidden_output: torch.Tensor,
        encoder_hidden_output: torch.Tensor = None,
    ):
        unmerge_func = None
        if self.mode_flag == "SSM" and module_name in (
            "ffn",
            "self-attn",
            "cross-attn",
            "parallel",
        ):
            unmerge_func = self.unmerge_SSM
        elif self.mode_flag == "IDM" and module_name == "ffn":
            unmerge_func = self.unmerge_IDM

        has_extra = global_context.hidden_extra_seq_len is not None
        if unmerge_func is not None:
            if has_extra:
                h1, h2 = hidden_output.split(
                    [
                        self.extra_start_idx,
                        hidden_output.shape[1] - self.extra_start_idx,
                    ],
                    dim=1,
                )
                unmerged_h1 = unmerge_func(
                    h1, self.indices_dict, self.prev_output[module_name]
                )
                unmerged_h2 = unmerge_func(
                    h2, self.extra_indices_dict, self.prev_output_extra[module_name]
                )
                hidden_output = torch.cat([unmerged_h1, unmerged_h2], dim=1)
                self.prev_output[module_name] = unmerged_h1
                self.prev_output_extra[module_name] = unmerged_h2
            else:
                hidden_output = unmerge_func(
                    hidden_output, self.indices_dict, self.prev_output[module_name]
                )
                self.prev_output[module_name] = hidden_output
        else:
            if has_extra:
                h1, h2 = hidden_output.split(
                    [
                        global_context.hidden_seq_len,
                        hidden_output.shape[1] - global_context.hidden_seq_len,
                    ],
                    dim=1,
                )
                self.prev_output[module_name] = h1
                self.prev_output_extra[module_name] = h2
            else:
                self.prev_output[module_name] = hidden_output

        if module_name in ("ffn", "parallel"):
            # clear indices after ffn
            self._clean()

        return super().hook_after_module(
            module_name, hidden_output, encoder_hidden_output
        )


if __name__ == "__main__":
    model_type = "flux-dev"
    pipe, model_attrs = get_pipeline(model_type)

    generator = torch.Generator("cuda").manual_seed(0)

    sdtm_config = {
        "stride": (2, 2),
        "switch_step_fraction": 0.4,
        "init_ratio": 0.3,
        "deviation": 0.2,
        "alpha_s": 0.05,
        "mcw": 0.1,
        "protected_step_stride": 1,
        "generator": generator,
    }

    HookManager.configure_hooks(
        model_type=model_type,
        default=(SDTMDiTBlockHook, sdtm_config),
    )

    with HookManager.enabled():
        HookManager.inject_hooks(
            model_type=model_type, pipe=pipe, model_attrs=model_attrs
        )
        res = process_quick_start(pipe)
        res.save(f"{model_type}_sdtm.png")

    print(global_context)
