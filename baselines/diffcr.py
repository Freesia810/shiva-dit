import torch
import torch.nn as nn
from typing import Literal
from project_name.hook import BaseDiTBlockHook
from project_name.utils.context import global_context
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
    
    def _select_topk_and_remaining_tokens(x, token_scores, k, C):
        B, N, _ = x.shape
        token_scores, topk_indices = torch.topk(torch.sigmoid(token_scores), k=k, sorted=False)
        sorted_indices, index = torch.sort(topk_indices, dim=1)

        # Get top-k tokens
        topk_x = x.gather(
            dim=1,
            index=repeat(sorted_indices, 'b t -> b t d', d=C)
        )

        # Get remaining tokens
        remaining_x = x.clone()
        remaining_x.scatter_(1, repeat(sorted_indices, 'b t -> b t d', d=C), torch.zeros_like(topk_x))

        return token_scores, topk_x, remaining_x, sorted_indices, index

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
        hidden_dim = global_context.hidden_dim
        if hidden_dim is None:
            raise ValueError("hidden_dim must be set in global_context")
        self.hidden_router = nn.Linear(hidden_dim, 1, bias=False)

    def hook_pre_norm(
        self, 
        module_name: Literal["self-attn", "cross-attn", "ffn", "parallel"],
        hidden_states: torch.Tensor,
        encoder_hidden_states: torch.Tensor = None,
        image_rotary_emb: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
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
                token_scores = self.hidden_router(hidden_states).squeeze(-1)  # (batch, seq_len)

                self.metadata = {
                    'lower': {},
                    'upper': {},
                }
                self.metadata['lower']['token_scores'], lower_selected, self.metadata['lower']['remaining_x'], self.metadata['lower']['sorted_indices'], self.metadata['lower']['index'] = self._select_topk_and_remaining_tokens(hidden_states, token_scores, k, hidden_states.shape[-1]) # (B, lower_k, D)
                self.metadata['upper']['token_scores'], upper_selected, self.metadata['upper']['remaining_x'], self.metadata['upper']['sorted_indices'], self.metadata['upper']['index'] = self._select_topk_and_remaining_tokens(hidden_states, token_scores, k, hidden_states.shape[-1]) # (B, upper_k, D)

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
                    token_scores = self.hidden_router(hidden_states).squeeze(-1)  # (batch, seq_len)
                    token_scores, topk_x, remaining_x, sorted_indices, index = self._select_topk_and_remaining_tokens(hidden_states, token_scores, k, hidden_states.shape[-1])
                    self.metadata = {
                        'token_scores': token_scores,
                        'remaining_x': remaining_x,
                        'sorted_indices': sorted_indices,
                        'index': index
                    }
                    return topk_x, encoder_hidden_states
        return super().hook_pre_norm(module_name, hidden_states, encoder_hidden_states, image_rotary_emb, temb)

    def hook_after_module(
        self, 
        module_name: Literal["self-attn", "cross-attn", "ffn", "dit"],
        hidden_output: torch.Tensor,
        encoder_hidden_output: torch.Tensor = None,
    ):
        if module_name == "ffn":
            # gate with topk scores
            if self.training:
                B = self._train_meta["B"]
                lower_k = self._train_meta["lower_k"]
                upper_k = self._train_meta["upper_k"]
                k_max = max(lower_k, upper_k)

                out_upper_combined = hidden_output[:B, :upper_k, :]   # (B, upper_k, D)
                out_lower_combined = hidden_output[B:, :lower_k, :]   # (B, lower_k, D)

                upper_scores = self.metadata['upper']['token_scores'].gather(dim=1, index=self.metadata['upper']['index']).unsqueeze(2)  # (B, upper_k, 1)
                lower_scores = self.metadata['lower']['token_scores'].gather(dim=1, index=self.metadata['lower']['index']).unsqueeze(2)  # (B, lower_k, 1)

                pad_upper = k_max - upper_k
                pad_lower = k_max - lower_k

                if pad_upper > 0:
                    upper_scores = torch.nn.functional.pad(upper_scores, (0, 0, 0, pad_upper))
                    out_upper_combined = torch.nn.functional.pad(out_upper_combined, (0, 0, 0, pad_upper))
                if pad_lower > 0:
                    lower_scores = torch.nn.functional.pad(lower_scores, (0, 0, 0, pad_lower))
                    out_lower_combined = torch.nn.functional.pad(out_lower_combined, (0, 0, 0, pad_lower))

                out_upper_combined = out_upper_combined * upper_scores
                out_lower_combined = out_lower_combined * lower_scores

                hidden_output = torch.cat([out_upper_combined, out_lower_combined], dim=0)  # (2B, k_max, D)
            else:
                hidden_output = hidden_output * self.metadata['token_scores'].gather(dim=1, index=self.metadata['index']).unsqueeze(2)
        elif module_name == "dit":
            # dit block's end
            if self.training:
                meta = self.__dict__.pop("_train_meta", None)
                B = meta["B"]
                lower_k = meta["lower_k"]
                upper_k = meta["upper_k"]
                lw = meta["lw"]
                uw = meta["uw"]
                out_upper_combined = hidden_output[:B, :upper_k, :]   # (B, upper_k, D)
                out_lower_combined = hidden_output[B:, :lower_k, :]   # (B, lower_k, D)

                out_upper_restored = self.metadata['upper']['remaining_x'].scatter_add(
                    dim=1,
                    index=repeat(self.metadata['upper']['sorted_indices'], 'b t -> b t d', d=out_upper_combined.shape[-1]),
                    src=out_upper_combined
                )

                out_lower_restored = self.metadata['lower']['remaining_x'].scatter_add(
                    dim=1,
                    index=repeat(self.metadata['lower']['sorted_indices'], 'b t -> b t d', d=out_lower_combined.shape[-1]),
                    src=out_lower_combined
                )

                # blend
                hidden_output = lw * out_lower_restored + uw * out_upper_restored

                if encoder_hidden_output is not None:
                    # for concat-attn like sd3 and flux
                    # if cross-attn, encoder_hidden_output is unchanged, the value will be restored
                    encoder_hidden_output = (
                        uw * encoder_hidden_output[:B, :, :]
                        + lw * encoder_hidden_output[B:, :, :]
                    )
            else:
                with torch.no_grad():
                    # Combine bypassed tokens and processed topk tokens
                    hidden_output = self.metadata['remaining_x'].scatter_add(
                        dim=1,
                        index=repeat(self.metadata['sorted_indices'], 'b t -> b t d', d=hidden_states.shape[-1]),
                        src=hidden_output
                    )
            self.metadata = None
            return hidden_output, encoder_hidden_output
        return super().hook_after_module(module_name, hidden_output, encoder_hidden_output)


def calculate_diffcr_loss(self, ratios, target_ratio):

    avg_mod_ratio = torch.mean(torch.stack(ratios))
    target_ratio_tensor = torch.tensor([target_ratio], device=avg_mod_ratio.device)
    loss = F.mse_loss(avg_mod_ratio, target_ratio_tensor)

    return loss
