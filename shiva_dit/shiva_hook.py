import inspect
import torch
import torch.nn as nn
import torch.nn.functional as F

from shiva_dit.hook.delegate import BaseDelegator
from shiva_dit.hook.block_hook import BaseDiTBlockHook
from shiva_dit.utils import global_context
from shiva_dit.utils.ste import ceil_ste
from shiva_dit.utils.token_utils import parse_image_rotary_emb, join_image_rotary_emb

from shiva_dit.modules.router import ShivaRouter
from shiva_dit.modules.gater import ShivaGater
from shiva_dit.modules.pruner import ShivaContextPruner, ShivaPrunerExpert
from shiva_dit.modules.fuser import ShivaFuserExpert


class ShivaDitBlockHook(BaseDelegator, BaseDiTBlockHook, nn.Module):
    def __init__(
        self,
        shared_layer_embedder: nn.Embedding,
        shared_router: ShivaRouter,
        shared_gater: ShivaGater,
        shared_pruner: ShivaContextPruner,
        shared_fuse_expert: ShivaFuserExpert,
        initial_kept_ratio: float = 0.5,
        use_gater_lut: bool = False,
        shared_gater_cache: torch.Tensor = None,
        init_Ts: float = 30.0,
        init_Tk: float = 30.0,
        **kwargs,
    ):
        super().__init__(**kwargs)

        initial_kept_ratio = float(initial_kept_ratio)
        self.shiva_ratio = nn.Parameter(torch.tensor(initial_kept_ratio))

        self.layer_embedder = shared_layer_embedder
        self.router = shared_router
        self.gater = shared_gater
        self.context_pruner = shared_pruner
        self.prune_expert = ShivaPrunerExpert(init_Ts=init_Ts, init_Tk=init_Tk)
        self.fuse_expert = shared_fuse_expert

        self.use_gater_lut = use_gater_lut
        self.gater_cache = shared_gater_cache

        if self.use_gater_lut and self.gater_cache is None:
            raise ValueError(
                f"ShivaDitBlockHook(layer_idx={self.layer_idx}): 'use_gater_lut' is True, "
                "but 'shared_gater_cache' was not provided."
            )

        self.gate_loss_accumulator = []
        self.prune_output_accumulator = []
        self.fuse_output_accumulator = []
        self.gater_fuse_prob_accumulator = []
        self.router_stats_accumulator = []

        self.discarded_info_norm_accumulator = []
        self.fuser_diff_norm_accumulator = []
        self.fuser_cos_sim_accumulator = []

        self.k_accumulator = []
        self.n_minus_k_accumulator = []

    def delegated_call(self, func, module_self, *args, **kwargs):
        (
            bound,
            x_sel,
            x_rej,
            remaining_x,
            sel_idx,
            sel_rope,
            pruned_encoder_states,
            B,
            N,
            D,
            k_float,
            token_scores,
            pure_temb,
            layer_emb_vector,
        ) = self._prepare_inputs(func, module_self, *args, **kwargs)

        if self.training:
            final_topk_states = self._forward_train(
                x_sel, x_rej, pure_temb, layer_emb_vector, token_scores, k_float, N
            )
        else:
            final_topk_states = self._forward_eval(
                x_sel, x_rej, pure_temb, layer_emb_vector
            )

        processed_output = self._process_outputs(
            func,
            bound,
            final_topk_states,
            sel_rope,
            pruned_encoder_states,
            remaining_x,
            sel_idx,
        )

        return processed_output

    def _prepare_inputs(self, func, module_self, *args, **kwargs):
        sig = inspect.signature(func)
        bound = sig.bind_partial(module_self, *args, **kwargs)
        bound.apply_defaults()

        hidden_states = bound.arguments.get("hidden_states")
        encoder_hidden_states = bound.arguments.get("encoder_hidden_states")
        image_rotary_emb = bound.arguments.get("image_rotary_emb")

        dtype = hidden_states.dtype
        B, N, D = hidden_states.shape

        pure_temb = global_context.pure_temb

        pruned_encoder_states, pooled_encoder_vector = self.context_pruner(
            encoder_hidden_states.to(dtype)
        )

        layer_id_tensor = torch.tensor(
            [self.layer_idx], device=pure_temb.device, dtype=torch.long
        ).expand(B)
        layer_emb_vector = self.layer_embedder(layer_id_tensor).to(dtype)

        token_scores = self.router(
            hidden_states.to(dtype), pure_temb, pooled_encoder_vector, layer_emb_vector
        )
        token_scores = torch.sigmoid(token_scores)

        unclamped_ratio = self.shiva_ratio
        clamped_ratio = torch.clamp(unclamped_ratio, min=1e-3, max=1.0)

        if self.training:
            k_float_unclamped = ceil_ste(clamped_ratio * N)
        else:
            with torch.no_grad():
                k_float_unclamped = torch.ceil(clamped_ratio * N)

        N_tensor_float = torch.tensor(
            N, device=k_float_unclamped.device, dtype=k_float_unclamped.dtype
        )
        k_float = torch.clamp(k_float_unclamped, min=1.0, max=N_tensor_float)

        x_sel, x_rej, remaining_x, sel_idx = self._select_topk(
            hidden_states, token_scores, k_float, D
        )
        sel_rope = self._gather_rope(image_rotary_emb, sel_idx)

        return (
            bound,
            x_sel,
            x_rej,
            remaining_x,
            sel_idx,
            sel_rope,
            pruned_encoder_states,
            B,
            N,
            D,
            k_float,
            token_scores,
            pure_temb,
            layer_emb_vector,
        )

    def _forward_train(
        self, x_sel, x_rej, pure_temb, layer_emb_vector, token_scores, k_float, N
    ):
        dtype = x_sel.dtype
        gater_logits = self.gater(pure_temb, layer_emb_vector)
        gating_weights = F.softmax(gater_logits, dim=-1)

        avg_gating_probs = gating_weights.mean(dim=0)
        num_experts = 2
        load_balancing_loss = num_experts * torch.sum(avg_gating_probs.pow(2))
        avg_fuse_prob = gating_weights[:, 1].mean()

        g_prune = gating_weights[:, 0].unsqueeze(-1).unsqueeze(-1)
        g_fuse = gating_weights[:, 1].unsqueeze(-1).unsqueeze(-1)

        output_prune = x_sel
        output_fuse = self.fuse_expert(x_sel.to(dtype), x_rej.to(dtype))

        final_topk_states = (g_prune * output_prune) + (g_fuse * output_fuse)

        self._collect_training_metrics(
            load_balancing_loss=load_balancing_loss,
            avg_fuse_prob=avg_fuse_prob,
            token_scores=token_scores,
            k_float=k_float,
            N=N,
            x_rej=x_rej,
            output_prune=output_prune,
            output_fuse=output_fuse,
        )
        return final_topk_states

    def _forward_eval(self, x_sel, x_rej, pure_temb, layer_emb_vector):
        self.clear_accumulators()
        dtype = x_sel.dtype

        if self.use_gater_lut:
            current_t_index = global_context.current_iteration
            decision = self.gater_cache[self.layer_idx, current_t_index].item()
        else:
            gater_logits = self.gater(pure_temb, layer_emb_vector)
            decision = torch.argmax(gater_logits[0], dim=-1).item()

        if decision == 0:
            final_topk_states = x_sel
        else:
            final_topk_states = self.fuse_expert(x_sel.to(dtype), x_rej.to(dtype))
        return final_topk_states

    def _process_outputs(
        self,
        func,
        bound,
        final_topk_states,
        sel_rope,
        pruned_encoder_states,
        remaining_x,
        sel_idx,
    ):
        bound.arguments["hidden_states"] = final_topk_states
        bound.arguments["image_rotary_emb"] = sel_rope
        bound.arguments["encoder_hidden_states"] = pruned_encoder_states

        processed_output = func(*bound.args, **bound.kwargs)

        if isinstance(processed_output, tuple):
            processed_encoder_states, processed_hidden_states = processed_output
        else:
            processed_hidden_states = processed_output

        final_hidden_states = self._scatter_add(
            processed_hidden_states, remaining_x, sel_idx
        )

        if isinstance(processed_output, tuple):
            return processed_encoder_states, final_hidden_states
        else:
            return final_hidden_states

    def _select_topk(self, x, token_scores, k, C):
        x_sel, x_rej, sel_idx, rej_idx = self.prune_expert(x, token_scores, k)
        remaining_x = torch.zeros_like(x)
        remaining_x.scatter_(1, rej_idx.unsqueeze(-1).expand(-1, -1, C), x_rej)
        return x_sel, x_rej, remaining_x, sel_idx

    def _gather_rope(self, image_rotary_emb, survivors_indices):
        if image_rotary_emb is not None:
            assert (
                survivors_indices.shape[0] == 1
            ), "image_rotary_emb only supports batch size 1"
            hidden_emb, encoder_emb, _ = parse_image_rotary_emb(
                self.model_type,
                image_rotary_emb,
                global_context.hidden_seq_len,
                global_context.encoder_hidden_seq_len_gen(),
                split_extra=False,
            )

            hidden_emb = tuple(
                torch.gather(
                    x,
                    dim=0,
                    index=survivors_indices[0].unsqueeze(-1).expand(-1, x.shape[-1]),
                )
                for x in hidden_emb
            )

            image_rotary_emb = join_image_rotary_emb(
                self.model_type, hidden_emb, encoder_emb
            )

        return image_rotary_emb

    def _scatter_add(self, processed_hidden_states, remaining_x, survivors_indices):
        B, k, D = processed_hidden_states.shape
        final_states = remaining_x.scatter_add(
            1,
            survivors_indices.unsqueeze(-1).expand(-1, -1, D),
            processed_hidden_states,
        )
        return final_states

    def clear_accumulators(self):
        self.gate_loss_accumulator.clear()
        self.prune_output_accumulator.clear()
        self.fuse_output_accumulator.clear()
        self.gater_fuse_prob_accumulator.clear()
        self.router_stats_accumulator.clear()
        self.discarded_info_norm_accumulator.clear()
        self.fuser_diff_norm_accumulator.clear()
        self.fuser_cos_sim_accumulator.clear()
        self.k_accumulator.clear()
        self.n_minus_k_accumulator.clear()

    @torch.no_grad()
    def _collect_training_metrics(
        self,
        load_balancing_loss,
        avg_fuse_prob,
        token_scores,
        k_float,
        N,
        x_rej,
        output_prune,
        output_fuse,
    ):
        self.gate_loss_accumulator.append(load_balancing_loss)
        self.gater_fuse_prob_accumulator.append(avg_fuse_prob)

        self.prune_output_accumulator.append(output_prune)
        self.fuse_output_accumulator.append(output_fuse)

        self.router_stats_accumulator.append(
            {
                "mean": token_scores.mean(),
                "std": token_scores.std(),
                "min": token_scores.min(),
                "max": token_scores.max(),
            }
        )

        k_val = int(k_float.item())
        n_minus_k_val = N - k_val
        self.k_accumulator.append(k_val)
        self.n_minus_k_accumulator.append(n_minus_k_val)

        if x_rej.shape[1] > 0:
            global_info = torch.mean(x_rej, dim=1, keepdim=True)
            discarded_info_norm = torch.norm(global_info, p=2, dim=-1).mean()
        else:
            discarded_info_norm = torch.tensor(
                0.0, device=x_rej.device, dtype=x_rej.dtype
            )

        output_diff = output_fuse - output_prune
        diff_norm = torch.norm(output_diff, p=2, dim=-1).mean()
        cos_sim = F.cosine_similarity(output_prune, output_fuse, dim=-1).mean()

        self.discarded_info_norm_accumulator.append(discarded_info_norm)
        self.fuser_diff_norm_accumulator.append(diff_norm)
        self.fuser_cos_sim_accumulator.append(cos_sim)
