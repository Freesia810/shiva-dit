import inspect
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Literal
from project_name.hook import BaseDiTBlockHook, BaseDelegator, HookManager
from project_name.utils import (
    ceil_ste,
    global_context,
    get_pipeline,
    process_quick_start,
    parse_image_rotary_emb,
    join_image_rotary_emb,
)

class DiffCRDitBlockHook(BaseDelegator, BaseDiTBlockHook, nn.Module):
    """
    Layer- and Timestep-Adaptive Differentiable Token Compression Ratios for Efficient Diffusion Transformers
    Reference:
        - Paper: https://arxiv.org/abs/2412.16822

    Implemented from https://github.com/GATECH-EIC/DiffCR
    """
    def __init__(self, ratio_min=0.2, ratio_max=1.0, ratio_step=-0.1, **kwargs):
        '''
            ratio in (ratio_min, ratio_max]
        '''
        super().__init__(**kwargs)
        self.register_buffer("kept_ratio_candidate", torch.arange(ratio_max, ratio_min, ratio_step).float())
        self.diff_ratio = nn.Parameter(torch.tensor(1.0))

        self.token_scores = None
        self.index = None

    def _select_topk_and_remaining_tokens(self, x, token_scores, k, C):
        B, N, _ = x.shape
        token_scores, topk_indices = torch.topk(torch.sigmoid(token_scores), k=k, sorted=False)
        sorted_indices, index = torch.sort(topk_indices, dim=1)
        # ===== Top-k tokens =====
        gather_index = sorted_indices.unsqueeze(-1).expand(-1, -1, C)  # (B, k, C)
        topk_x = torch.gather(x, dim=1, index=gather_index)             # (B, k, C)
        # ===== Remaining tokens =====
        remaining_x = x.clone()
        remaining_x.scatter_(1, gather_index, 0.0)

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
        self.dtype = torch.float32
        self.to(device=next(block.parameters()).device, dtype=self.dtype)

    def delegated_call(self, func, module_self, *args, **kwargs):
        sig = inspect.signature(func)
        bound = sig.bind_partial(module_self, *args, **kwargs)
        bound.apply_defaults()

        hidden_states = bound.arguments.get('hidden_states', None)
        encoder_hidden_states = bound.arguments.get('encoder_hidden_states', None)
        image_rotary_emb = bound.arguments.get('image_rotary_emb', None)

        seq_len = hidden_states.shape[1]
        kept_ratio = torch.clamp(self.diff_ratio, 0.1, 1.0)

        def _process(kept_ratio, hidden_states, encoder_hidden_states, image_rotary_emb):
            capacity = ceil_ste(kept_ratio * seq_len).to(torch.int32)
            k = torch.min(capacity, torch.tensor(seq_len, device=hidden_states.device))
            token_scores = self.hidden_router(hidden_states.to(self.dtype)).squeeze(-1)  # (batch, seq_len)
            self.token_scores, topk_x, remaining_x, sorted_indices, self.index = (
                self._select_topk_and_remaining_tokens(
                    hidden_states, token_scores, k, hidden_states.shape[-1]
                )
            )

            bound.arguments["hidden_states"] = topk_x
            if encoder_hidden_states is not None:
                bound.arguments["encoder_hidden_states"] = encoder_hidden_states

            if image_rotary_emb is not None:
                assert hidden_states.shape[0] == 1, "image_rotary_emb only supports batch size 1"
                hidden_emb, encoder_emb, _ = parse_image_rotary_emb(
                    self.model_type,
                    image_rotary_emb,
                    global_context.hidden_seq_len,
                    global_context.encoder_hidden_seq_len_gen(),
                    split_extra=False
                )

                hidden_emb = tuple(
                    torch.gather(
                        x,
                        dim=0,
                        index=sorted_indices[0].unsqueeze(-1).expand(-1, x.shape[-1]),
                    )
                    for x in hidden_emb
                )

                image_rotary_emb = join_image_rotary_emb(
                    self.model_type, hidden_emb, encoder_emb
                )

                bound.arguments["image_rotary_emb"] = image_rotary_emb

            # call original function
            result = func(*bound.args, **bound.kwargs)

            if isinstance(result, tuple) and len(result) == 2:
                encoder_hidden_states, hidden_states = result
                ret_tuple = True
            else:
                hidden_states = result
                ret_tuple = False

            # Combine bypassed tokens and processed topk tokens
            hidden_states = remaining_x.scatter_add(
                dim=1,
                index=sorted_indices.unsqueeze(-1).expand(-1, -1, hidden_states.shape[-1]),
                src=hidden_states,
            )

            if ret_tuple:
                return encoder_hidden_states, hidden_states
            else:
                return hidden_states

        if self.training:
            (lower_bin, upper_bin), _ = self.find_neighbour_bins(kept_ratio)
            lower_weight = (upper_bin - kept_ratio) / (upper_bin - lower_bin)
            upper_weight = 1.0 - lower_weight

            def _blend(lower_output, upper_output, lw, uw):
                if isinstance(lower_output, tuple):
                    return tuple(
                        lw * lo + uw * up for lo, up in zip(lower_output, upper_output)
                    )
                else:
                    return lw * lower_output + uw * upper_output

            lower_output = _process(lower_bin, hidden_states, encoder_hidden_states, image_rotary_emb)
            upper_output = _process(upper_bin, hidden_states, encoder_hidden_states, image_rotary_emb)
            return _blend(lower_output, upper_output, lower_weight, upper_weight)
        else:
            with torch.no_grad():
                kept_ratio = self.find_closest_bin(kept_ratio)
                return _process(kept_ratio, hidden_states, encoder_hidden_states, image_rotary_emb)

    def hook_after_module(
        self, 
        module_name: Literal["self-attn", "cross-attn", "ffn"],
        hidden_output: torch.Tensor,
        encoder_hidden_output: torch.Tensor = None,
    ):
        if module_name == "ffn":
            # gate with topk scores
            # TODO: the difference about this line
            hidden_output = hidden_output * (self.token_scores.gather(dim=1, index=self.index).unsqueeze(2).to(hidden_output.dtype))
        return super().hook_after_module(module_name, hidden_output, encoder_hidden_output)


if __name__ == "__main__":
    model_type = "pixart-sigma"
    pipe, model_attrs = get_pipeline(model_type)

    diffcr_config = {
        "ratio_min": 0.1,
        "ratio_max": 1.0,
        "ratio_step": -0.1,
    }

    HookManager.configure_hooks(
        model_type=model_type,
        default=(DiffCRDitBlockHook, diffcr_config),
    )

    with HookManager.enabled():
        HookManager.inject_hooks(model_type=model_type, pipe=pipe, model_attrs=model_attrs)
        res = process_quick_start(pipe)
        res.save(f"{model_type}_diffcr.png")
    
    print(global_context)


# def calculate_diffcr_loss(self, ratios, target_ratio):

#     avg_mod_ratio = torch.mean(torch.stack(ratios))
#     target_ratio_tensor = torch.tensor([target_ratio], device=avg_mod_ratio.device)
#     loss = F.mse_loss(avg_mod_ratio, target_ratio_tensor)

#     return loss
