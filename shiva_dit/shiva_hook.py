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
        shiva_mode: str,  # "image", "context", "both"
        total_layers: int,
        
        # --- Image Pruning Components ---
        router_layer_embedder: nn.Embedding = None,
        shared_router: ShivaRouter = None,
        shared_gater_policy: ShivaGaterPolicy = None,
        shared_image_ratio_policy: ShivaRatioPolicy = None,
        shared_pruner: ShivaPrunerExpert = None,
        shared_fuse_expert: ShivaFuserExpert = None,
        fixed_image_ratio: float = 0.5,
        
        # --- Context Pruning Components ---
        shared_context_pruner: ShivaContextPruner = None,
        shared_context_ratio_policy: ShivaRatioPolicy = None,
        fixed_context_ratio: float = 1.0,
        
        # --- Gater Settings ---
        use_gater_lut: bool = False,
        shared_gater_cache: torch.Tensor = None,
        
        **kwargs,
    ):
        super().__init__(**kwargs)

        self.shiva_mode = shiva_mode
        self.total_layers = total_layers

        # Image Modules
        self.router_layer_embedder = router_layer_embedder
        self.router = shared_router
        self.gater_policy = shared_gater_policy
        self.image_ratio_policy = shared_image_ratio_policy
        self.prune_expert = shared_pruner
        self.fuse_expert = shared_fuse_expert
        self.fixed_image_ratio = fixed_image_ratio

        # Context Modules
        self.context_pruner = shared_context_pruner
        self.context_ratio_policy = shared_context_ratio_policy
        self.fixed_context_ratio = fixed_context_ratio

        # Gater LUT
        self.use_gater_lut = use_gater_lut
        self.gater_cache = shared_gater_cache

        if self.use_gater_lut and self.gater_cache is None:
            if self.shiva_mode in ["image", "both"]:
                raise ValueError(
                    f"ShivaDitBlockHook(layer_idx={self.layer_idx}): 'use_gater_lut' is True, "
                    "but 'shared_gater_cache' was not provided."
                )

        # Metrics Accumulators
        # Image
        self.gate_loss_accumulator = []
        self.gater_fuse_prob_accumulator = []
        self.router_stats_accumulator = []
        self.image_ratio_accumulator = [] 

        # Context
        self.context_score_accumulator = []
        self.context_ratio_accumulator = []
        self.context_aux_loss_accumulator = []

        self.capture_mask_for_viz = True

        # Placeholders for loss calculation
        self.current_gate_logits_for_loss = None
        self.current_ratio_for_loss = None
        self.token_scores_logits = None
        self.k_float = None

    def delegated_call(self, func, module_self, *args, **kwargs):
        (
            bound,
            # Image Selection
            x_sel, x_rej, remaining_x, sel_idx,
            # Context Selection
            ctx_sel, remaining_ctx, sel_idx_ctx,
            # Meta
            sel_rope, pure_temb, layer_id_tensor, pooled_encoder_states, N,
        ) = self._prepare_inputs(func, module_self, *args, **kwargs)

        # 1. Spatial Pruning Forward (Image)
        # Default to x_sel (which is full hidden_states if image pruning is off)
        final_topk_states = x_sel 

        if self.shiva_mode in ["image", "both"] and self.fuse_expert is not None:
            if self.training:
                final_topk_states = self._forward_train_image(x_sel, x_rej, N)
            else:
                final_topk_states = self._forward_eval_image(x_sel, x_rej, pure_temb, layer_id_tensor)

        # 2. Context Pruning Forward (Text)
        final_topk_context = ctx_sel

        processed_output = self._process_outputs(
            func,
            bound,
            final_topk_states,
            final_topk_context,
            sel_rope,
            pooled_encoder_states,
            remaining_x,
            sel_idx,
            remaining_ctx,
            sel_idx_ctx,
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

        # --- Metadata Preparation ---
        layer_id_tensor = torch.tensor(
            [self.layer_idx], device=pure_temb.device, dtype=torch.long
        ).expand(B)

        router_layer_emb_vector = None
        if self.router_layer_embedder is not None:
            router_layer_emb_vector = self.router_layer_embedder(layer_id_tensor).to(dtype)

        current_t = global_context.current_timestep
        if current_t is None:
            raise ValueError("global_context.current_timestep is None")

        t_val = current_t.to(dtype)
        t_val = t_val / 1000.0  # normalize

        if t_val.numel() == 1:
            t_scalar = t_val.view(1, 1).expand(B, 1)
        elif t_val.shape[0] == B:
            t_scalar = t_val.view(B, 1)
        else:
            raise ValueError(f"t_val.shape: {t_val.shape}")

        # --- Context Pruning Logic ---
        ctx_sel = encoder_hidden_states
        remaining_ctx = None
        sel_idx_ctx = None

        if self.shiva_mode in ["context", "both"] and self.context_pruner is not None:
            L = encoder_hidden_states.shape[1]

            # Determine Context Ratio
            if self.context_ratio_policy is not None:
                # Adaptive Ratio
                ctx_ratio_logits = self.context_ratio_policy(t_scalar, layer_id_tensor)
                ctx_ratio_pred = torch.sigmoid(ctx_ratio_logits).squeeze(-1).mean()

                is_random_training = (
                    self.training 
                    and getattr(global_context, "current_stage_idx", -1) == 1
                )
                if is_random_training:
                    noise = torch.rand(1, device=pure_temb.device, dtype=dtype) * 0.8 + 0.1
                    ctx_ratio = noise + (0.0 * ctx_ratio_pred)
                else:
                    ctx_ratio = ctx_ratio_pred
            else:
                # Fixed Ratio
                ctx_ratio = torch.tensor(self.fixed_context_ratio, device=pure_temb.device, dtype=dtype)

            # Safety clamp
            ctx_ratio = ctx_ratio.clamp(min=0.01, max=1.0)
            self.context_ratio_accumulator.append(ctx_ratio)

            k_ctx = int(max(1, L * ctx_ratio))

            # Execute Context Pruning
            if k_ctx < L:
                ctx_sel, remaining_ctx, sel_idx_ctx = self._select_topk_context(
                    hidden_states.detach().to(dtype), 
                    encoder_hidden_states.to(dtype), 
                    k_ctx, D
                )

        pooled_encoder_vector = ctx_sel.mean(dim=1)

        # --- Image Pruning Logic ---
        x_sel = hidden_states
        x_rej = None
        remaining_x = None
        sel_idx = None

        # Reset metrics placeholders
        self.current_gate_logits_for_loss = None
        self.current_ratio_for_loss = None
        self.token_scores_logits = None
        self.k_float = None

        if self.shiva_mode in ["image", "both"] and self.router is not None:
            # Determine Image Ratio
            if self.image_ratio_policy is not None:
                # Adaptive Ratio
                img_ratio_logits = self.image_ratio_policy(t_scalar, layer_id_tensor)
                policy_ratio = torch.sigmoid(img_ratio_logits).squeeze(-1).mean()

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
            else:
                # Fixed Ratio
                predicted_ratio = torch.tensor(self.fixed_image_ratio, device=pure_temb.device, dtype=dtype)

            if self.layer_idx == 0:
                predicted_ratio = torch.tensor(1.0, device=predicted_ratio.device, dtype=predicted_ratio.dtype)

            clamped_ratio = predicted_ratio.clamp(min=0.01, max=1.0)
            self.current_ratio_for_loss = clamped_ratio

            if self.training:
                k_float = ceil_ste(clamped_ratio * N)
            else:
                with torch.no_grad():
                    k_float = torch.ceil(clamped_ratio * N)
            self.k_float = k_float

            # Gater Call
            gate_logits = self.gater_policy(pure_temb, layer_id_tensor)
            self.current_gate_logits_for_loss = gate_logits

            # Router Call
            token_scores_logits = self.router(
                hidden_states=hidden_states.to(dtype), 
                pooled_encoder_states=pooled_encoder_vector,
                temb=pure_temb, 
                layer_emb=router_layer_emb_vector
            )

            if token_scores_logits.shape[0] % 2 == 0 and token_scores_logits.shape[0] > 1:
                scores_uncond, scores_cond = token_scores_logits.chunk(2, dim=0)
                unified_scores = torch.maximum(scores_uncond, scores_cond)
                token_scores_logits = torch.cat([unified_scores, unified_scores], dim=0)

            self.token_scores_logits = token_scores_logits

            # Execute Image Pruning
            x_sel, x_rej, remaining_x, sel_idx = self._select_topk(
                hidden_states, token_scores_logits, k_float, D
            )

        # --- RoPE Gathering ---
        sel_rope = self._gather_rope(image_rotary_emb, sel_idx, sel_idx_ctx)

        return (
            bound,
            # Image
            x_sel, x_rej, remaining_x, sel_idx,
            # Context
            ctx_sel, remaining_ctx, sel_idx_ctx,
            # Meta
            sel_rope, pure_temb, layer_id_tensor, pooled_encoder_vector, N,
        )

    def _forward_train_image(self, x_sel, x_rej, N):
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

        self._collect_training_metrics_image(
            load_balancing_loss=load_balancing_loss,
            avg_fuse_prob=avg_fuse_prob,
            token_scores_logits=self.token_scores_logits,
            k_float=self.k_float,
            N=N,
        )
        return final_topk_states

    def _forward_eval_image(self, x_sel, x_rej, pure_temb, layer_id_tensor):
        dtype = x_sel.dtype

        if getattr(self, "is_validating", False):
            try:
                self._collect_training_metrics_image(
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
            gater_logits = self.gater_policy(pure_temb, layer_id_tensor)
            decision = torch.argmax(gater_logits[0], dim=-1).item()

        if decision == 0:
            final_topk_states = x_sel
        else:
            final_topk_states = self.fuse_expert(x_sel.to(dtype), x_rej.to(dtype))
        return final_topk_states

    def _process_outputs(
        self, func, bound, 
        final_topk_states, final_topk_context,
        sel_rope, pooled_encoder_states, 
        remaining_x, sel_idx, 
        remaining_ctx, sel_idx_ctx
    ):
        bound.arguments["hidden_states"] = final_topk_states
        bound.arguments["image_rotary_emb"] = sel_rope
        bound.arguments["encoder_hidden_states"] = final_topk_context

        processed_output = func(*bound.args, **bound.kwargs)

        if isinstance(processed_output, tuple):
            processed_encoder_states, processed_hidden_states = processed_output
        else:
            processed_hidden_states = processed_output
            processed_encoder_states = None

        # Scatter Image
        if remaining_x is not None and sel_idx is not None:
            final_hidden_states = self._scatter_add(
                processed_hidden_states, remaining_x, sel_idx
            )
        else:
            final_hidden_states = processed_hidden_states

        # Scatter Context
        if processed_encoder_states is not None and sel_idx_ctx is not None:
            B, k, D = processed_encoder_states.shape
            final_encoder_states = remaining_ctx.scatter_add(
                1,
                sel_idx_ctx.unsqueeze(-1).expand(-1, -1, D),
                processed_encoder_states,
            )
        else:
            final_encoder_states = processed_encoder_states

        if isinstance(processed_output, tuple):
            return final_encoder_states, final_hidden_states
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

    def _select_topk_context(self, hidden_states, encoder_hidden_states, k, C):
        ctx_sel, ctx_rej, sel_idx, rej_idx = self.context_pruner(
            hidden_states=hidden_states, 
            encoder_hidden_states=encoder_hidden_states, 
            k=k
        )

        if self.training and hasattr(self.context_pruner, "get_aux_loss"):
            aux_loss = self.context_pruner.get_aux_loss()
            self.context_aux_loss_accumulator.append(aux_loss)

        if getattr(self, "capture_mask_for_viz", False) and self.training:
            B, L, _ = encoder_hidden_states.shape
            mask = torch.zeros((B, L), device=encoder_hidden_states.device)
            mask.scatter_(1, sel_idx.long(), 1.0)
            self.last_context_mask = mask.detach().cpu()

            if hasattr(self.context_pruner, "last_scores"):
                self.context_score_accumulator.append(self.context_pruner.last_scores.detach())

        B, L, D = encoder_hidden_states.shape
        remaining_ctx = torch.zeros_like(encoder_hidden_states)
        remaining_ctx.scatter_(1, rej_idx.unsqueeze(-1).expand(-1, -1, C), ctx_rej)

        return ctx_sel, remaining_ctx, sel_idx

    def _gather_rope(self, image_rotary_emb, survivors_indices, context_survivors_indices=None):
        if image_rotary_emb is not None:
            if survivors_indices is not None:
                assert survivors_indices.shape[0] == 1, "image_rotary_emb hook currently optimized for BS=1"
            if context_survivors_indices is not None:
                assert context_survivors_indices.shape[0] == 1

            hidden_emb, encoder_emb, _ = parse_image_rotary_emb(
                self.model_type,
                image_rotary_emb,
                global_context.hidden_seq_len,
                global_context.encoder_hidden_seq_len_gen(),
                split_extra=False,
            )

            if survivors_indices is not None:
                hidden_emb = tuple(
                    torch.gather(
                        x,
                        dim=0,
                        index=survivors_indices[0].unsqueeze(-1).expand(-1, x.shape[-1]),
                    )
                    for x in hidden_emb
                )

            if context_survivors_indices is not None and encoder_emb is not None:
                encoder_emb = tuple(
                    torch.gather(
                        x,
                        dim=0,
                        index=context_survivors_indices[0].unsqueeze(-1).expand(-1, x.shape[-1]),
                    )
                    for x in encoder_emb
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
        self.image_ratio_accumulator.clear()
        self.context_score_accumulator.clear()
        self.context_ratio_accumulator.clear()
        self.context_aux_loss_accumulator.clear()

    def _collect_training_metrics_image(
        self,
        load_balancing_loss,
        avg_fuse_prob,
        token_scores_logits,
        k_float,
        N,
    ):
        self.gate_loss_accumulator.append(load_balancing_loss)
        self.gater_fuse_prob_accumulator.append(avg_fuse_prob)
        self.image_ratio_accumulator.append(self.current_ratio_for_loss)

        token_scores = torch.sigmoid(token_scores_logits)
        self.router_stats_accumulator.append(
            {
                "scores_tensor": token_scores,
                "mean": token_scores.mean(),
                "std": token_scores.std(),
                "min": token_scores.min(),
                "max": token_scores.max(),
            }
        )
