import inspect
import torch
import torch.nn as nn
import torch.nn.functional as F

from shiva_dit.hook.delegate import BaseDelegator
from shiva_dit.hook.block_hook import BaseDiTBlockHook
from shiva_dit.utils import global_context
from shiva_dit.utils.ste import ceil_ste
from shiva_dit.utils import parse_image_rotary_emb, join_image_rotary_emb

from shiva_dit.modules.router import ShivaRouter
from shiva_dit.modules.gater import ShivaGaterPolicy, ShivaRatioPolicy
from shiva_dit.modules.pruner import ShivaContextPruner, ShivaPrunerExpert
from shiva_dit.modules.fuser import ShivaFuserExpert

RANDOM_STAGE = 0
RATIO_STAGE = 1
ACCUMULATION_STEPS = 16
GATER_STAGE = 99

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
        # --- LuT Settings ---
        use_gater_lut: bool = False,
        shared_gater_cache: torch.Tensor = None,
        shared_ratio_cache: torch.Tensor = None,
        shared_context_ratio_cache: torch.Tensor = None,
        capture_mask_for_viz: bool = False,
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
        self.ratio_cache = shared_ratio_cache
        self.context_ratio_cache = shared_context_ratio_cache

        self._internal_forward_counter = 0

        if self.use_gater_lut:
            if self.shiva_mode in ["image", "both"]:
                if self.gater_cache is None:
                    raise ValueError(
                        f"use_gater_lut=True but shared_gater_cache is None."
                    )
                if self.ratio_cache is None:
                    raise ValueError(
                        f"use_gater_lut=True but shared_ratio_cache is None."
                    )
            if self.shiva_mode in ["context", "both"]:
                if self.context_ratio_cache is None:
                    raise ValueError(
                        f"use_gater_lut=True but shared_context_ratio_cache is None."
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

        self.capture_mask_for_viz = capture_mask_for_viz

    def on_register_hook(self, block: nn.Module):
        self.register_buffer(
            "layer_id_tensor", torch.tensor([self.layer_idx], dtype=torch.long)
        )

        # Placeholders for loss calculation
        self.current_gate_logits_for_loss = None
        self.current_ratio_for_loss = None
        self.current_context_ratio_for_loss = None
        self.token_scores_logits = None

    def delegated_call(self, func, module_self, *args, **kwargs):
        (
            bound,
            # Image Selection
            x_sel,
            x_rej,
            sel_idx,
            rej_idx,
            # Context Selection
            ctx_sel,
            ctx_rej,
            sel_idx_ctx,
            rej_idx_ctx,
            # Meta
            sel_rope,
            pure_temb,
            layer_id_tensor,
            pooled_encoder_states,
            N,
            L,
        ) = self._prepare_inputs(func, module_self, *args, **kwargs)

        # 1. Spatial Pruning Forward (Image)
        # Default to x_sel (which is full hidden_states if image pruning is off)
        final_topk_states = x_sel

        if self.shiva_mode in ["image", "both"] and self.fuse_expert is not None:
            if self.training:
                final_topk_states = self._forward_train_image(x_sel, x_rej, N)
            else:
                final_topk_states = self._forward_eval_image(
                    x_sel, x_rej, pure_temb, layer_id_tensor
                )

        # 2. Context Pruning Forward (Text)
        final_topk_context = ctx_sel

        processed_output = self._process_outputs(
            func,
            bound,
            final_topk_states,
            final_topk_context,
            sel_rope,
            x_rej,
            sel_idx,
            rej_idx,
            N,
            ctx_rej,
            sel_idx_ctx,
            rej_idx_ctx,
            L,
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
        device = hidden_states.device
        B, N, D = hidden_states.shape

        L = 0
        if encoder_hidden_states is not None:
            L = encoder_hidden_states.shape[1]

        pure_temb = global_context.pure_temb.detach()

        # --- Metadata Preparation ---
        layer_id_tensor = self.layer_id_tensor.expand(B)

        def get_t_scalar():
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
            
            return t_scalar

        # --- Context Pruning Logic ---
        ctx_sel = encoder_hidden_states
        ctx_rej = None
        sel_idx_ctx = None
        rej_idx_ctx = None
        pooled_encoder_vector = None

        if (
            self.shiva_mode in ["context", "both"]
            and self.context_pruner is not None
            and encoder_hidden_states is not None
        ):
            # Initialize ctx_rej with empty for safety if pruning happens
            ctx_rej = torch.empty(
                (B, 0, encoder_hidden_states.shape[2]),
                device=encoder_hidden_states.device,
                dtype=dtype,
            )

            # Determine Context Ratio
            ctx_ratio = None
            if (
                self.use_gater_lut
                and self.context_ratio_cache is not None
                and not self.training
            ):
                current_t_index = global_context.current_iteration
                ctx_ratio = self.context_ratio_cache[self.layer_idx, current_t_index]

            if ctx_ratio is None:
                if self.context_ratio_policy is not None:
                    # Adaptive Ratio
                    ctx_ratio_logits = self.context_ratio_policy(
                        pure_temb, layer_id_tensor
                    )
                    ctx_ratio_pred = torch.sigmoid(ctx_ratio_logits).squeeze(-1).mean()

                    is_random_training = (
                        self.training
                        and getattr(global_context, "current_stage_idx", -1)
                        == RANDOM_STAGE
                    )
                    if is_random_training:
                        noise = torch.rand(1, device=device, dtype=dtype) * 0.8 + 0.1
                        ctx_ratio = noise + (0.0 * ctx_ratio_pred)
                    else:
                        ctx_ratio = ctx_ratio_pred
                else:
                    # Fixed Ratio
                    ctx_ratio = torch.tensor(
                        self.fixed_context_ratio, device=device, dtype=dtype
                    )

            # Safety clamp
            ctx_ratio = ctx_ratio.clamp(min=0.01, max=1.0)
            self.current_context_ratio_for_loss = ctx_ratio

            if self.training:
                k_ctx = ceil_ste(ctx_ratio * L)
            else:
                with torch.no_grad():
                    k_ctx = torch.ceil(ctx_ratio * L)

            # Execute Context Pruning
            if k_ctx < L:
                ctx_sel, ctx_rej, sel_idx_ctx, rej_idx_ctx = self._select_topk_context(
                    hidden_states.detach().to(dtype),
                    encoder_hidden_states.to(dtype),
                    k_ctx,
                    D,
                )

        if ctx_sel is not None:
            pooled_encoder_vector = ctx_sel.mean(dim=1)
        elif encoder_hidden_states is not None:
            pooled_encoder_vector = encoder_hidden_states.mean(dim=1)

        # --- Image Pruning Logic ---
        x_sel = hidden_states
        x_rej = torch.empty((B, 0, D), device=hidden_states.device, dtype=dtype)
        sel_idx = None
        rej_idx = None

        # Reset metrics placeholders
        self.current_gate_logits_for_loss = None
        self.current_ratio_for_loss = None
        self.token_scores_logits = None

        if self.shiva_mode in ["image", "both"] and self.router is not None:
            # Determine Image Ratio
            predicted_ratio = None
            if (
                self.use_gater_lut
                and self.ratio_cache is not None
                and not self.training
            ):
                current_t_index = global_context.current_iteration
                predicted_ratio = self.ratio_cache[self.layer_idx, current_t_index]

            if predicted_ratio is None:
                if self.image_ratio_policy is not None:
                    # Adaptive Ratio
                    img_ratio_logits = self.image_ratio_policy(
                        pure_temb, layer_id_tensor
                    )
                    policy_ratio = torch.sigmoid(img_ratio_logits).squeeze(-1).mean()

                    is_random_training = (
                        self.training
                        and getattr(global_context, "current_stage_idx", -1)
                        == RANDOM_STAGE
                    )

                    if is_random_training:
                        self._internal_forward_counter += 1
                        min_r = 0.2
                        max_r = 0.9
                        cycle_idx = self._internal_forward_counter % ACCUMULATION_STEPS
                        ratio_val = min_r + (max_r - min_r) * (cycle_idx / (ACCUMULATION_STEPS - 1 + 1e-6))

                        random_ratio = torch.tensor(
                            ratio_val, device=pure_temb.device, dtype=dtype
                        )
                        predicted_ratio = random_ratio + (0.0 * policy_ratio)

                        # if self.layer_idx == 0 and cycle_idx == 0:
                        #     print(f"DEBUG: New Accumulation Cycle Started. Reset Ratio to {ratio_val:.2f}")
                    else:
                        predicted_ratio = policy_ratio
                else:
                    # Fixed Ratio
                    predicted_ratio = torch.tensor(
                        self.fixed_image_ratio, device=pure_temb.device, dtype=dtype
                    )

            clamped_ratio = predicted_ratio.clamp(min=0.01, max=1.0)
            self.current_ratio_for_loss = clamped_ratio

            if self.training:
                k_float = ceil_ste(clamped_ratio * N)
            else:
                with torch.no_grad():
                    k_float = torch.ceil(clamped_ratio * N)

            # Gater Call
            gate_logits = None
            if self.training or not self.use_gater_lut:
                gate_logits = self.gater_policy(pure_temb, layer_id_tensor)
            self.current_gate_logits_for_loss = gate_logits

            # Router Call
            router_layer_emb_vector = self.router_layer_embedder(layer_id_tensor).to(dtype)
            token_scores_logits = self.router(
                hidden_states=hidden_states,
                pooled_encoder_states=pooled_encoder_vector,
                temb=pure_temb,
                layer_emb=router_layer_emb_vector,
            )

            if (
                token_scores_logits.shape[0] % 2 == 0
                and token_scores_logits.shape[0] > 1
            ):
                scores_uncond, scores_cond = token_scores_logits.chunk(2, dim=0)
                unified_scores = torch.maximum(scores_uncond, scores_cond)
                token_scores_logits = torch.cat([unified_scores, unified_scores], dim=0)

            self.token_scores_logits = token_scores_logits

            # Execute Image Pruning
            x_sel, x_rej, sel_idx, rej_idx = self._select_topk(
                hidden_states, token_scores_logits, k_float, D
            )

        # --- RoPE Gathering ---
        sel_rope = self._gather_rope(image_rotary_emb, sel_idx, sel_idx_ctx)

        return (
            bound,
            # Image
            x_sel,
            x_rej,
            sel_idx,
            rej_idx,
            # Context
            ctx_sel,
            ctx_rej,
            sel_idx_ctx,
            rej_idx_ctx,
            # Meta
            sel_rope,
            pure_temb,
            layer_id_tensor,
            pooled_encoder_vector,
            N,
            L,
        )

    def _forward_train_image(self, x_sel, x_rej, N):
        dtype = x_sel.dtype
        device = x_sel.device

        current_stage = getattr(global_context, "current_stage_idx", -1)
        is_gater_active = (self.training and current_stage == GATER_STAGE)
        logits = self.current_gate_logits_for_loss
        
        if is_gater_active and self.training:
            noise = torch.randn_like(logits) * 1.0 
            logits = logits + noise
        
        gating_weights = F.softmax(logits, dim=-1).to(dtype)
        
        raw_g_prune = gating_weights[:, 0].unsqueeze(-1).unsqueeze(-1)
        raw_g_fuse = gating_weights[:, 1].unsqueeze(-1).unsqueeze(-1)

        if not is_gater_active:
            g_fuse = raw_g_fuse * 0.0
            g_prune = torch.ones_like(raw_g_prune)
            empty_discarded = x_rej[:, :0, :] 
            output_fuse = self.fuse_expert(x_sel.to(dtype), empty_discarded.to(dtype))
            
            load_balancing_loss = torch.tensor(0.0, device=device, dtype=dtype)
            
            with torch.no_grad():
                avg_fuse_prob = torch.zeros_like(gating_weights[:, 1].mean())
        else:
            g_prune = raw_g_prune
            g_fuse = raw_g_fuse
            
            output_fuse = self.fuse_expert(x_sel.to(dtype), x_rej.to(dtype))
            
            avg_fuse_prob = gating_weights[:, 1].mean()
            load_balancing_loss = torch.tensor(0.0, device=device, dtype=dtype)

        final_topk_states = (g_prune * x_sel) + (g_fuse * output_fuse)

        self._collect_training_metrics_image(
            load_balancing_loss=load_balancing_loss,
            avg_fuse_prob=avg_fuse_prob,
            token_scores_logits=self.token_scores_logits,
            N=N,
        )
        return final_topk_states

    def _forward_eval_image(self, x_sel, x_rej, pure_temb, layer_id_tensor):
        dtype = x_sel.dtype

        if getattr(self, "is_validating", False):
            self._collect_training_metrics_image(
                load_balancing_loss=torch.tensor(0.0, device=x_sel.device),
                avg_fuse_prob=torch.tensor(0.0, device=x_sel.device),
                token_scores_logits=self.token_scores_logits,
                N=x_sel.shape[1] + x_rej.shape[1],
            )

        decision = 0
        if self.use_gater_lut:
            current_t_index = global_context.current_iteration
            decision = self.gater_cache[self.layer_idx, current_t_index].item()
        else:
            gater_logits = self.current_gate_logits_for_loss
            decision = torch.argmax(gater_logits[0], dim=-1).item()

        if decision == 0:
            final_topk_states = x_sel
        else:
            final_topk_states = self.fuse_expert(x_sel, x_rej)
        return final_topk_states

    def _process_outputs(
        self,
        func,
        bound,
        final_topk_states,
        final_topk_context,
        sel_rope,
        x_rej,
        sel_idx,
        rej_idx,
        N,
        ctx_rej,
        sel_idx_ctx,
        rej_idx_ctx,
        L,
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
        if sel_idx is not None:
            B, _, D = processed_hidden_states.shape
            final_hidden_states = torch.empty(
                (B, N, D),
                device=processed_hidden_states.device,
                dtype=processed_hidden_states.dtype,
            )
            final_hidden_states.scatter_(
                1, sel_idx.unsqueeze(-1).expand(-1, -1, D), processed_hidden_states
            )
            if x_rej is not None and x_rej.shape[1] > 0:
                final_hidden_states.scatter_(
                    1, rej_idx.unsqueeze(-1).expand(-1, -1, D), x_rej
                )
        else:
            final_hidden_states = processed_hidden_states

        # Scatter Context
        if processed_encoder_states is not None and sel_idx_ctx is not None:
            B, _, D = processed_encoder_states.shape
            final_encoder_states = torch.empty(
                (B, L, D),
                device=processed_encoder_states.device,
                dtype=processed_encoder_states.dtype,
            )
            final_encoder_states.scatter_(
                1, sel_idx_ctx.unsqueeze(-1).expand(-1, -1, D), processed_encoder_states
            )
            if ctx_rej is not None and ctx_rej.shape[1] > 0:
                final_encoder_states.scatter_(
                    1, rej_idx_ctx.unsqueeze(-1).expand(-1, -1, D), ctx_rej
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
            self.last_sel_idx = sel_idx.detach().cpu()  # (B, k)
            self.last_token_scores = torch.sigmoid(token_scores_logits).detach().cpu()
            self.last_total_tokens = x.shape[1]
            current_t = global_context.current_timestep
            if current_t is not None:
                t_val = current_t.flatten()[0].item()
                t_val /= 1000.0
                self.last_t_val = t_val
            else:
                self.last_t_val = -1.0

        return x_sel, x_rej, sel_idx, rej_idx

    def _select_topk_context(self, hidden_states, encoder_hidden_states, k, C):
        ctx_sel, ctx_rej, sel_idx, rej_idx = self.context_pruner(
            hidden_states=hidden_states,
            encoder_hidden_states=encoder_hidden_states,
            k=k,
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
                self.context_score_accumulator.append(
                    self.context_pruner.last_scores.detach()
                )

        # Logic Update: Do NOT scatter here. Return pieces to be scattered in process_outputs.
        return ctx_sel, ctx_rej, sel_idx, rej_idx

    def _gather_rope(
        self, image_rotary_emb, survivors_indices, context_survivors_indices=None
    ):
        if image_rotary_emb is None:
            return None

        hidden_emb, encoder_emb, _ = parse_image_rotary_emb(
            self.model_type,
            image_rotary_emb,
            global_context.hidden_seq_len,
            global_context.encoder_hidden_seq_len_gen(),
            split_extra=False,
        )

        if survivors_indices is not None:
            idx = survivors_indices[0]
            hidden_emb = tuple(x[idx] for x in hidden_emb)

        if context_survivors_indices is not None and encoder_emb is not None:
            idx_ctx = context_survivors_indices[0]
            encoder_emb = tuple(x[idx_ctx] for x in encoder_emb)

        image_rotary_emb = join_image_rotary_emb(
            self.model_type, hidden_emb, encoder_emb
        )
        return image_rotary_emb

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
        N,
    ):
        if self.shiva_mode in ["image", "both"]:
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
        if self.shiva_mode in ["context", "both"]:
            self.context_ratio_accumulator.append(self.current_context_ratio_for_loss)

        
