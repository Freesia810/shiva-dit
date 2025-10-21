import torch
import torch.nn as nn
from typing import Literal
from project_name.hook import BaseDiTBlockHook
from project_name.utils.metadata import get_block_metadata
from project_name.utils.ste import ceil_ste
from project_name.utils.token_utils import (
    reconstruct_tokens_by_topk,
    select_topk_tokens,
)

class DiffCRDitBlockHook(BaseDiTBlockHook, nn.Module):
    def __init__(self, ratio_min=0.2, ratio_max=1.0, ratio_step=-0.1, **kwargs):
        '''
            ratio in (ratio_min, ratio_max]
        '''
        super().__init__(**kwargs)
        self.register_buffer("kept_ratio_candidate", torch.arange(ratio_max, ratio_min, ratio_step).float())
        self.diff_ratio = nn.Parameter(torch.tensor(1.0))

    def find_neighbour_bins(self, kept_ratio):
        differences = torch.abs(self.kept_ratio_candidate - kept_ratio)
        _, indices = torch.topk(differences, 2, largest=False)
        neighbour_bins = self.kept_ratio_candidate[indices]
        return neighbour_bins, indices

    def find_closest_bin(self, kept_ratio):
        differences = torch.abs(self.kept_ratio_candidate - kept_ratio)
        closest_index = torch.argmin(differences)
        return self.kept_ratio_candidate[closest_index]

    def on_register_hook(self, block: nn.Module):
        metadata = get_block_metadata(block)
        self.hidden_router = nn.Linear(metadata['hidden_states_dim'], 1, bias=False)

    def hook_pre_norm(
        self, 
        module_name: Literal["self-attn", "cross-attn", "ffn", "parallel"],
        hidden_states: torch.Tensor,
        encoder_hidden_states: torch.Tensor = None,
        temb: torch.Tensor = None,
    ):
        if module_name == "self-attn":
            # dit block's start
            seq_len = hidden_states.shape[1]
            kept_ratio = torch.clamp(self.diff_ratio, 0.1, 1.0)
            if self.training:
                (lower_bin, upper_bin), _ = self.find_neighbour_bins(kept_ratio)
                lower_weight = (upper_bin - kept_ratio) / (upper_bin - lower_bin)
                upper_weight = 1.0 - lower_weight

                lower_capacity = ceil_ste(lower_bin * seq_len).to(torch.int32)
                lower_k = torch.min(
                    lower_capacity, torch.tensor(seq_len, device=hidden_states.device)
                )
                upper_capacity = ceil_ste(upper_bin * seq_len).to(torch.int32)
                upper_k = torch.min(
                    upper_capacity, torch.tensor(seq_len, device=hidden_states.device)
                )
                token_scores = torch.sigmoid(
                    self.hidden_router(hidden_states).squeeze(-1)
                )  # (batch, seq_len)

                lower_selected, lower_idx = select_topk_tokens(hidden_states, token_scores, lower_k)  # (B, lower_k, D)
                upper_selected, upper_idx = select_topk_tokens(hidden_states, token_scores, upper_k)  # (B, upper_k, D)

                # padding to max k
                k_max = max(lower_k, upper_k)
                if lower_k < k_max:
                    pad_sz = k_max - lower_k
                    lower_selected = torch.nn.functional.pad(lower_selected, (0, 0, 0, pad_sz))  # pad seq dim on right
                if upper_k < k_max:
                    pad_sz = k_max - upper_k
                    upper_selected = torch.nn.functional.pad(upper_selected, (0, 0, 0, pad_sz))

                combined_selected = torch.cat(
                    [upper_selected, lower_selected], dim=0
                )  # (2B, K_max, D)

                self._train_meta = {
                    "B": hidden_states.shape[0],
                    "lower_k": lower_k,
                    "upper_k": upper_k,
                    "lower_idx": lower_idx,  # shape (B, lower_k)
                    "upper_idx": upper_idx,  # shape (B, upper_k)
                    "orig_hidden": hidden_states,  # (B, S, D)
                    "lw": lower_weight,
                    "uw": upper_weight,
                }

                if encoder_hidden_states is not None:
                    # for concat-attn like sd3 and flux, we must duplicate encoder_hidden_states to calculate self-attn and FFN
                    # for cross-attn, encoder_hidden_states is only used for KV, so no need to duplicate
                    encoder_hidden_states = encoder_hidden_states.repeat(2, 1, 1)

                return combined_selected, encoder_hidden_states
            else:
                with torch.no_grad():
                    kept_ratio = self.find_closest_bin(kept_ratio)
                    capacity = ceil_ste(kept_ratio * seq_len).to(torch.int32)
                    k = torch.min(
                        capacity, torch.tensor(seq_len, device=hidden_states.device)
                    )
                    token_scores = torch.sigmoid(
                        self.hidden_router(hidden_states).squeeze(-1)
                    )  # (batch, seq_len)
                    selected_hidden_states, topk_indices = select_topk_tokens(
                        hidden_states, token_scores, k
                    )
                    self.topk_indices = topk_indices
                    self.original_hidden_states = hidden_states
                    return selected_hidden_states, encoder_hidden_states
        else:
            return super().hook_pre_norm(module_name, hidden_states, encoder_hidden_states, temb)

    def hook_after_module(
        self, 
        module_name: Literal["self-attn", "cross-attn", "ffn"],
        hidden_output: torch.Tensor,
        encoder_hidden_output: torch.Tensor = None,
    ):
        if module_name == "ffn":
            # dit block's end
            if self.training:
                meta = self.__dict__.pop("_train_meta", None)
                B = meta["B"]
                lower_k = meta["lower_k"]
                upper_k = meta["upper_k"]
                lower_idx = meta["lower_idx"] # (B, lower_k)
                upper_idx = meta["upper_idx"] # (B, upper_k)
                orig_hidden = meta["orig_hidden"]
                lw = meta["lw"]
                uw = meta["uw"]
                out_upper_combined = hidden_output[:B, :upper_k, :]   # (B, upper_k, D)
                out_lower_combined = hidden_output[B:, :lower_k, :]   # (B, lower_k, D)

                out_upper_restored = reconstruct_tokens_by_topk(
                    orig_hidden,  # (B, S, D)
                    out_upper_combined,  # (B, upper_k, D)
                    upper_idx, # (B, upper_k)
                    zero_out_remain=True,
                )
                out_lower_restored = reconstruct_tokens_by_topk(
                    orig_hidden,
                    out_lower_combined,
                    lower_idx,
                    zero_out_remain=True,
                )

                # blend
                blended = lw * out_lower_restored + uw * out_upper_restored

                if encoder_hidden_output is not None:
                    # for concat-attn like sd3 and flux
                    # if cross-attn, encoder_hidden_output is unchanged, the value will be restored
                    encoder_hidden_output = (
                        uw * encoder_hidden_output[:B, :, :]
                        + lw * encoder_hidden_output[B:, :, :]
                    )

                return blended, encoder_hidden_output
            else:
                with torch.no_grad():
                    restored_hidden_states = reconstruct_tokens_by_topk(
                        self.original_hidden_states,
                        hidden_output,
                        self.topk_indices,
                        zero_out_remain=True,
                    )
                    return restored_hidden_states, encoder_hidden_output
        return super().hook_after_module(module_name, hidden_output, encoder_hidden_output)


def calculate_diffcr_loss(self, ratios, target_ratio):

    avg_mod_ratio = torch.mean(torch.stack(ratios))
    target_ratio_tensor = torch.tensor([target_ratio], device=avg_mod_ratio.device)
    loss = F.mse_loss(avg_mod_ratio, target_ratio_tensor)

    return loss
