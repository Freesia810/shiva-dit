import inspect
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Literal

from shiva_dit.hook.delegate import BaseDelegator
from shiva_dit.hook.block_hook import BaseDiTBlockHook
from shiva_dit.utils import global_context
from shiva_dit.utils.ste import ceil_ste
from shiva_dit.utils import parse_image_rotary_emb, join_image_rotary_emb

from shiva_dit.modules.router import ShivaRouter
from shiva_dit.modules.gater import ShivaGaterPolicy, ShivaRatioPolicy
from shiva_dit.modules.pruner import ShivaContextPruner, ShivaPrunerExpert
from shiva_dit.modules.fuser import ShivaFuserExpert


class ShivaDitBlockHook(BaseDelegator, BaseDiTBlockHook, nn.Module):

    def __init__(
        self,
        shared_layer_embedder: nn.Embedding,
        shared_router: ShivaRouter,
        shared_gater_policy: ShivaGaterPolicy,
        shared_ratio_policy: ShivaRatioPolicy,
        shared_pruner: ShivaPrunerExpert,
        shared_context_pruner: ShivaContextPruner,
        shared_fuse_expert: ShivaFuserExpert,
        use_gater_lut: bool = False,
        shared_gater_cache: torch.Tensor = None,
        total_layers: int = 1,
        **kwargs,
    ):
        super().__init__(**kwargs)

        self.layer_embedder = shared_layer_embedder
        self.router = shared_router
        self.gater_policy = shared_gater_policy
        self.ratio_policy = shared_ratio_policy
        self.context_pruner = shared_context_pruner
        self.prune_expert = shared_pruner
        self.fuse_expert = shared_fuse_expert

        self.use_gater_lut = use_gater_lut
        self.gater_cache = shared_gater_cache

        if self.use_gater_lut and self.gater_cache is None:
            raise ValueError(
                f"ShivaDitBlockHook(layer_idx={self.layer_idx}): 'use_gater_lut' is True, "
                "but 'shared_gater_cache' was not provided."
            )

        self.total_layers = total_layers

        self.gate_loss_accumulator = []
        self.gater_fuse_prob_accumulator = []
        self.router_stats_accumulator = []
        self.k_accumulator = []
        self.n_minus_k_accumulator = []
        self.ratio_accumulator = []

        self.capture_mask_for_viz = True

    def delegated_call(self, func, module_self, *args, **kwargs):
        (
            bound,
            x_sel,
            x_rej,
            remaining_x,
            sel_idx,
            sel_rope,
            pure_temb,
            layer_emb,
            pruned_encoder_states,
            N,
        ) = self._prepare_inputs(func, module_self, *args, **kwargs)

        if self.training:
            final_topk_states = self._forward_train(x_sel, x_rej, N)
        else:
            final_topk_states = self._forward_eval(x_sel, x_rej, pure_temb, layer_emb)

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

        l_val = self.layer_idx / max(1, self.total_layers - 1)
        l_scalar = torch.tensor([l_val], device=pure_temb.device, dtype=dtype).expand(B, 1)
        current_t = global_context.current_timestep
        if current_t is None:
            raise ValueError("global_context.current_timestep is None")

        t_val = current_t.to(dtype)
        t_val = t_val / 1000.0 # normalize

        if t_val.numel() == 1:
            t_scalar = t_val.view(1, 1).expand(B, 1)
        elif t_val.shape[0] == B:
            t_scalar = t_val.view(B, 1)
        else:
            raise ValueError(f"t_val.shape: {t_val.shape}")
        l_scalar = l_scalar.view(B, 1)

        gate_logits = self.gater_policy(pure_temb, layer_emb_vector)

        predicted_ratio_logits = self.ratio_policy(t_scalar, layer_id_tensor)
        policy_ratio = torch.sigmoid(predicted_ratio_logits).squeeze(-1).mean()

        is_random_training = (
            self.training 
            and getattr(global_context, "current_stage_idx", -1) == 1
        )

        if is_random_training:
            random_ratio_val = torch.rand(1, device=pure_temb.device).item() * 0.8 + 0.1
            random_ratio = torch.tensor(random_ratio_val, device=pure_temb.device, dtype=dtype)
            predicted_ratio = random_ratio + (0.0 * policy_ratio)
        else:
            predicted_ratio = policy_ratio

        if self.layer_idx == 0:
            predicted_ratio = torch.tensor(1.0, device=predicted_ratio.device, dtype=predicted_ratio.dtype)

        clamped_ratio = predicted_ratio.clamp(min=0.01)
        if self.training:
            k_float = ceil_ste(clamped_ratio * N)
        else:
            with torch.no_grad():
                k_float = torch.ceil(clamped_ratio * N)

        token_scores_logits = self.router(
            hidden_states=hidden_states.to(dtype), 
            pooled_encoder_states=pooled_encoder_vector,
            temb=pure_temb, 
            layer_emb=layer_emb_vector
        )

        if token_scores_logits.shape[0] % 2 == 0 and token_scores_logits.shape[0] > 1:
            B_half = token_scores_logits.shape[0] // 2

            scores_uncond, scores_cond = token_scores_logits.chunk(2, dim=0)
            unified_scores = torch.maximum(scores_uncond, scores_cond)
            token_scores_logits = torch.cat([unified_scores, unified_scores], dim=0)

        self.current_gate_logits_for_loss = gate_logits
        self.current_ratio_for_loss = clamped_ratio
        self.current_raw_predicted_ratio = predicted_ratio
        self.token_scores_logits = token_scores_logits
        self.k_float = k_float
        # ------------------------------------------------

        x_sel, x_rej, remaining_x, sel_idx = self._select_topk(
            hidden_states, token_scores_logits, k_float, D
        )
        sel_rope = self._gather_rope(image_rotary_emb, sel_idx)

        return (
            bound,
            x_sel,
            x_rej,
            remaining_x,
            sel_idx,
            sel_rope,
            pure_temb,
            layer_emb_vector,
            pruned_encoder_states,
            N,
        )

    def _forward_train(self, x_sel, x_rej, N):
        dtype = x_sel.dtype

        gating_weights = F.softmax(self.current_gate_logits_for_loss, dim=-1)

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
            token_scores_logits=self.token_scores_logits,
            k_float=self.k_float,
            N=N,
        )
        return final_topk_states

    def _forward_eval(self, x_sel, x_rej, pure_temb, layer_emb):
        dtype = x_sel.dtype

        if getattr(self, "is_validating", False):
            try:
                self._collect_training_metrics(
                    load_balancing_loss=torch.tensor(0.0, device=x_sel.device),
                    avg_fuse_prob=torch.tensor(0.0, device=x_sel.device),
                    token_scores_logits=self.token_scores_logits,
                    k_float=self.k_float,
                    N=x_sel.shape[1] + x_rej.shape[1],
                )
            except Exception as e:
                pass

        if self.use_gater_lut:
            current_t_index = global_context.current_iteration
            decision = self.gater_cache[self.layer_idx, current_t_index].item()
        else:
            gater_logits = self.gater_policy(pure_temb, layer_emb)
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

    def _select_topk(self, x, token_scores_logits, k, C):
        x_sel, x_rej, sel_idx, rej_idx = self.prune_expert(x, token_scores_logits, k)

        if getattr(self, "capture_mask_for_viz", False):
            self.last_sel_idx = sel_idx.detach().cpu() # (B, k)
            self.last_token_scores = torch.sigmoid(token_scores_logits).detach().cpu() 
            self.last_total_tokens = x.shape[1]
            current_t = global_context.current_timestep
            if current_t is not None:
                t_val = current_t.flatten()[0].item()
                if t_val > 1.0:
                    t_val /= 1000.0
                self.last_t_val = t_val
            else:
                self.last_t_val = -1.0

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
        self.gater_fuse_prob_accumulator.clear()
        self.router_stats_accumulator.clear()
        self.k_accumulator.clear()
        self.n_minus_k_accumulator.clear()
        self.ratio_accumulator.clear()

    def _collect_training_metrics(
        self,
        load_balancing_loss,
        avg_fuse_prob,
        token_scores_logits,
        k_float,
        N,
    ):
        self.gate_loss_accumulator.append(load_balancing_loss)
        self.gater_fuse_prob_accumulator.append(avg_fuse_prob)
        self.ratio_accumulator.append(self.current_ratio_for_loss)

        token_scores = torch.sigmoid(token_scores_logits)
        # s_scale = (1.0 + torch.tanh(token_scores_logits))
        # s_scale_mean = s_scale.mean()
        # s_scale_std = s_scale.std()
        self.router_stats_accumulator.append(
            {
                "scores_tensor": token_scores,
                "mean": token_scores.mean(),
                "std": token_scores.std(),
                "min": token_scores.min(),
                "max": token_scores.max(),
                # "s_scale_mean": s_scale_mean,
                # "s_scale_std": s_scale_std,
            }
        )

        k_val = int(k_float.item())
        n_minus_k_val = N - k_val
        self.k_accumulator.append(k_val)
        self.n_minus_k_accumulator.append(n_minus_k_val)
