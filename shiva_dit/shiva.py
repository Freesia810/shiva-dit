import inspect
from typing import Literal, Type
import torch
import torch.nn as nn
import torch.nn.functional as F

from shiva_dit.hook.block_hook import BaseDiTBlockHook
from shiva_dit.hook.delegate import BaseDelegator
from shiva_dit.hook.hook_manager import HookManager
from shiva_dit.utils import ceil_ste, global_context
from shiva_dit.utils.utils import join_image_rotary_emb, parse_image_rotary_emb


class ShivaRouter(nn.Module):
    def __init__(
        self,
        hidden_dim: int,
        prompt_embed_dim: int,
        layer_embed_dim: int,
        bottleneck_dim: int = 64,
        time_embed_dim: int = None,
    ):
        super().__init__()

        if time_embed_dim is None:
            time_embed_dim = hidden_dim

        self.token_proj = nn.Linear(hidden_dim, bottleneck_dim)
        self.prompt_proj = nn.Linear(prompt_embed_dim, bottleneck_dim, bias=False)
        self.temb_proj = nn.Linear(time_embed_dim, bottleneck_dim, bias=False)
        self.layer_proj = nn.Linear(layer_embed_dim, bottleneck_dim, bias=False)
        self.router_head = nn.Linear(bottleneck_dim, 1, bias=True)

    def forward(
        self,
        hidden_states,
        pooled_encoder_states,
        temb,
        layer_emb,
    ):
        """
        hidden_states: (B, seq_len, D_h)
        temb: (B, D_t)
        pooled_encoder_states: (B, D_p)
        layer_emb: (B, D_l)
        """

        time_feat = self.temb_proj(temb).unsqueeze(1)
        prompt_feat = self.prompt_proj(pooled_encoder_states).unsqueeze(1)
        layer_feat = self.layer_proj(layer_emb).unsqueeze(1)
        context_feat = time_feat + prompt_feat + layer_feat

        token_feat = self.token_proj(hidden_states)  # (B, N, c)
        fused_hidden = F.silu(token_feat + context_feat)
        token_scores = self.router_head(fused_hidden)  # (B, N, 1)
        return token_scores.squeeze(-1)  # (B, N)


class ShivaContextPruner(nn.Module):
    """
    TODO: if the model has larger context length, we should propose a better pruning strategy
    Now we only use this module to pool the encoder states for the hidden's router
    """
    def __init__(self, threshold: int = None, ratio: float = None):
        super().__init__()
        if ratio is not None and not 0.0 < ratio <= 1.0:
            raise ValueError(f"Pruning ratio must be between (0, 1], but got {ratio}")

        self.threshold = threshold
        self.ratio = ratio

    @torch.no_grad()
    def forward(self, encoder_hidden_states: torch.Tensor):
        """
        :param encoder_hidden_states: (B, C, D)
        :return: (pruned_states, pooled_vector)
                 pruned_states: (B, c_pruned, D) 或 (B, C, D)
                 pooled_vector: (B, D)
        """
        B, C, D = encoder_hidden_states.shape

        if self.threshold is None:
            # default: mean pooling
            pooled_vector = torch.mean(encoder_hidden_states, dim=1)
            return encoder_hidden_states, pooled_vector

        if C <= self.threshold or self.ratio == 1.0:
            # default: mean pooling
            pooled_vector = torch.mean(encoder_hidden_states, dim=1)
            return encoder_hidden_states, pooled_vector

        c_pruned = int(C * self.ratio)
        c_pruned = max(c_pruned, 1)
        if c_pruned == C:
            pooled_vector = torch.mean(encoder_hidden_states, dim=1)
            return encoder_hidden_states, pooled_vector

        token_scores = torch.norm(
            encoder_hidden_states.to(torch.float32), p=2, dim=-1
        ).detach()
        topk_indices = torch.topk(token_scores, k=c_pruned, dim=-1).indices
        topk_indices_expanded = topk_indices.unsqueeze(-1).expand(-1, -1, D)
        pruned_states = torch.gather(encoder_hidden_states, 1, topk_indices_expanded)
        pooled_vector = torch.mean(pruned_states, dim=1)

        return pruned_states, pooled_vector


class ShivaGater(nn.Module):
    """
    During training, we use this module to gating the experts.
    During inference, we can obtain the layer id and the timesteps offline, so we can directly use the gating logits.
    """
    def __init__(
        self,
        time_embed_dim: int,
        layer_embed_dim: int,
    ):
        super().__init__()
        fused_dim = time_embed_dim + layer_embed_dim
        self.net = nn.Sequential(
            nn.Linear(fused_dim, 64),
            nn.SiLU(),
            nn.Linear(64, 2),  # Prune, Fuse
        )

    def forward(self, temb: torch.Tensor, layer_emb: torch.Tensor):
        """
        :param temb: (B, time_embed_dim)
        :param layer_emb: (B, layer_embed_dim)
        :return: (B, 2)
        """
        fused_input = torch.cat([temb, layer_emb], dim=-1)
        return self.net(fused_input)  # [B, 2]


ShivaPrunerExpert = nn.Identity


class ShivaFuserExpert(nn.Module):
    def __init__(self, hidden_dim: int, ffn_mult: int = 2):
        super().__init__()
        self.fusion_ffn = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim * ffn_mult),
            nn.SiLU(),
            nn.Linear(hidden_dim * ffn_mult, hidden_dim)
        )

    def forward(self, survivors, discarded):
        """
        survivors: (B, k, D)
        discarded: (B, N-k, D)
        """
        if discarded.shape[1] > 0:
            global_info = torch.mean(discarded, dim=1, keepdim=True) # (B, 1, D)
            projected_info = self.fusion_ffn(global_info) # (B, 1, D)
            fused_survivors = survivors + projected_info # (B, k, D)
        else:
            fused_survivors = survivors
            if self.training:
                dummy_output = self.fusion_ffn(
                    torch.zeros(
                        survivors.shape[0], 1, survivors.shape[2],
                        device=survivors.device, dtype=survivors.dtype
                    )
                )
                fused_survivors = fused_survivors + (dummy_output * 0.0)

        return fused_survivors


class ShivaDitBlockHook(BaseDelegator, BaseDiTBlockHook, nn.Module):
    """
    multiple shivahooks will share the same experts and modules, so we use dependency injection to pass them in.
    """
    @classmethod
    def setup_shared_hooks(
        cls,
        hook_manager_cls: Type["HookManager"],
        model_type: str,
        pipe,
        model_attrs: list[str],
        **kwargs,
    ):
        bottleneck_dim = kwargs.pop("bottleneck_dim", 64)
        num_router_groups = kwargs.pop("num_router_groups", 4)
        layer_embed_dim = kwargs.pop("layer_embed_dim", 256)
        fuse_expert_ffn_mult = kwargs.pop("fuse_expert_ffn_mult", 2)
        context_pruning_threshold = kwargs.pop("context_pruning_threshold", None)
        context_pruning_ratio = kwargs.pop("context_pruning_ratio", None)
        if 'initial_kept_ratio' not in kwargs:
            kwargs['initial_kept_ratio'] = 1.0

        model_dim = global_context.hidden_dim
        prompt_embed_dim = global_context.encoder_hidden_dim
        if model_dim is None:
            raise ValueError(
                "[ShivaDitBlockHook] 'hidden_dim' not found in global_context."
            )
        if prompt_embed_dim is None:
            raise ValueError(
                "[ShivaDitBlockHook] 'encoder_hidden_dim' not found in global_context."
            )

        time_embed_dim = model_dim

        all_blocks = []
        for model_attr in model_attrs:
            m = getattr(pipe, model_attr)
            for name, module in m.named_modules():
                if isinstance(module, hook_manager_cls.get_block_type(model_type)):
                    full_name = f"{model_attr}.{name}"
                    all_blocks.append((full_name, module))

        num_layers = len(all_blocks)
        if num_layers == 0:
            print("[ShivaDitBlockHook] Warning: Found 0 matching blocks.")
            return

        try:
            first_block_param = next(all_blocks[0][1].parameters())
            device = first_block_param.device
            dtype = first_block_param.dtype
        except StopIteration:
            raise RuntimeError(
                "Cannot infer device/dtype: First block has no parameters."
            )

        print(
            f"[ShivaDitBlockHook] Found {num_layers} blocks. Instantiating shared modules..."
        )
        print(
            f"  > model_dim: {model_dim}, prompt_embed_dim: {prompt_embed_dim}, layer_embed_dim: {layer_embed_dim}"
        )

        shared_layer_embedder = nn.Embedding(num_layers, layer_embed_dim).to(
            device, dtype
        )
        shared_gater = ShivaGater(
            time_embed_dim=time_embed_dim, layer_embed_dim=layer_embed_dim
        ).to(device, dtype)
        shared_pruner = ShivaContextPruner(
            threshold=context_pruning_threshold, ratio=context_pruning_ratio
        )
        shared_fuse_expert = ShivaFuserExpert(
            hidden_dim=model_dim, ffn_mult=fuse_expert_ffn_mult
        ).to(device, dtype)
        grouped_routers = nn.ModuleList(
            [
                ShivaRouter(
                    hidden_dim=model_dim,
                    time_embed_dim=time_embed_dim,
                    prompt_embed_dim=prompt_embed_dim,
                    layer_embed_dim=layer_embed_dim,
                    bottleneck_dim=bottleneck_dim,
                )
                for _ in range(num_router_groups)
            ]
        ).to(device, dtype)

        ## initialization: keep parameters to reasonable values
        # nn.init.constant_(shared_gater.net[-1].bias, 0.0)
        # shared_gater.net[-1].bias.data[0] = 5.0
        # shared_gater.net[-1].bias.data[1] = -5.0

        # fuser_last_layer = shared_fuse_expert.fusion_ffn[-1]
        # nn.init.zeros_(fuser_last_layer.weight)
        # if fuser_last_layer.bias is not None:
        #     nn.init.zeros_(fuser_last_layer.bias)

        for router in grouped_routers:
            nn.init.zeros_(router.router_head.weight)
            nn.init.constant_(router.router_head.bias, 5.0)

        layers_per_group = num_layers // num_router_groups

        injected = 0
        for i, (hook_name, module) in enumerate(all_blocks):
            router_group_id = min(i // layers_per_group, num_router_groups - 1)
            injected_router = grouped_routers[router_group_id]

            hook_kwargs = {
                "shared_layer_embedder": shared_layer_embedder,
                "shared_router": injected_router,
                "shared_gater": shared_gater,
                "shared_pruner": shared_pruner,
                "shared_fuse_expert": shared_fuse_expert,
                **kwargs,
            }

            hook = cls(**hook_kwargs)
            hook.register_name(model_type, hook_name)
            if hasattr(module, "register_hook"):
                module.register_hook(hook_name, hook)
            else:
                raise RuntimeError(
                    f"Module '{hook_name}' does not support hook registration."
                )

            hook_manager_cls._hook_map[hook_name] = hook
            hook_manager_cls._weak_blocks[hook_name] = module
            injected += 1

        print(
            f"[ShivaDitBlockHook] Injected {injected} shared hooks (using {num_router_groups} router groups)."
        )

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
        **kwargs,
    ):

        super().__init__(**kwargs)

        self.shiva_ratio = nn.Parameter(torch.tensor(float(initial_kept_ratio)))

        self.layer_embedder = shared_layer_embedder
        self.router = shared_router
        self.gater = shared_gater
        self.context_pruner = shared_pruner
        self.prune_expert = ShivaPrunerExpert()
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

        self.token_scores = None

        self.eval()

    def delegated_call(self, func, module_self, *args, **kwargs):
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

        self._last_forward_scores = token_scores.detach()
        with torch.no_grad():
            sigmoid_scores = torch.sigmoid(token_scores)
            if self.training:
                self.router_stats_accumulator.append({
                    "mean": sigmoid_scores.mean(),
                    "std": sigmoid_scores.std(),
                    "min": sigmoid_scores.min(),
                })
            else:
                self.router_stats_accumulator.clear()

        kept_ratio = torch.clamp(self.shiva_ratio, min=1e-9, max=1.0)
        k = ceil_ste(kept_ratio * N).to(torch.int32)
        k = torch.min(k, torch.tensor(N, device=hidden_states.device))

        (
            self.token_scores,  # (B, k)
            survivors,  # (B, k, D)
            remaining_x,  # (B, N, D)
            survivors_indices,  # (B, k)
        ) = self._select_topk_and_remaining_tokens(hidden_states, token_scores, k, D)

        discarded_mask = torch.ones_like(hidden_states, dtype=torch.bool)
        discarded_mask.scatter_(
            1, survivors_indices.unsqueeze(-1).expand(-1, -1, D), False
        )
        discarded = hidden_states[discarded_mask].view(B, N - k, D)

        survivor_rope = self._gather_rope(image_rotary_emb, survivors_indices)

        if self.training:
            gater_logits = self.gater(pure_temb, layer_emb_vector)  # (B, 2)
            gating_weights = F.softmax(gater_logits, dim=-1)

            entropy = -(gating_weights * torch.log(gating_weights + 1e-10)).sum(dim=-1)
            self.gate_loss_accumulator.append(entropy.mean())

            avg_fuse_prob = gating_weights[:, 1].mean()
            self.gater_fuse_prob_accumulator.append(avg_fuse_prob)

            g_prune = gating_weights[:, 0].unsqueeze(-1).unsqueeze(-1)
            g_fuse = gating_weights[:, 1].unsqueeze(-1).unsqueeze(-1)

            output_prune = self.prune_expert(survivors)
            output_fuse = self.fuse_expert(survivors.to(dtype), discarded.to(dtype))

            self.prune_output_accumulator.append(output_prune)
            self.fuse_output_accumulator.append(output_fuse)

            final_topk_states = (g_prune * output_prune) + (g_fuse * output_fuse)
        else:
            self.gate_loss_accumulator.clear()
            self.prune_output_accumulator.clear()
            self.fuse_output_accumulator.clear()
            self.gater_fuse_prob_accumulator.clear()

            if self.use_gater_lut:
                current_t_index = global_context.current_iteration # TODO: LUT Logic
                decision = self.gater_cache[self.layer_idx, current_t_index].item()
            else:
                # we can use the first batch's decision safely because all samples share the same layer_idx and timestep
                gater_logits = self.gater(pure_temb, layer_emb_vector)
                decision = torch.argmax(gater_logits[0], dim=-1).item()

            if decision == 0:
                final_topk_states = self.prune_expert(survivors)
            else:
                final_topk_states = self.fuse_expert(
                    survivors.to(dtype), discarded.to(dtype)
                )

        bound.arguments["hidden_states"] = final_topk_states
        bound.arguments["image_rotary_emb"] = survivor_rope
        bound.arguments["encoder_hidden_states"] = pruned_encoder_states

        processed_output = func(*bound.args, **bound.kwargs)

        if isinstance(processed_output, tuple):
            processed_encoder_states, processed_hidden_states = processed_output
        else:
            processed_hidden_states = processed_output

        if self.training:
            proxy_scale_ste = (kept_ratio - kept_ratio.detach()) + 1.0
            processed_hidden_states = processed_hidden_states * proxy_scale_ste

        final_hidden_states = self._scatter_add(
            processed_hidden_states, remaining_x, survivors_indices
        )

        if isinstance(processed_output, tuple):
            return processed_encoder_states, final_hidden_states
        else:
            return final_hidden_states

    def _select_topk_and_remaining_tokens(self, x, token_scores, k, C):
        B, N, _ = x.shape
        token_scores, topk_indices = torch.topk(
            torch.sigmoid(token_scores), k=k, sorted=False
        )

        # ===== Top-k tokens =====
        gather_index = topk_indices.unsqueeze(-1).expand(-1, -1, C)  # (B, k, C)
        topk_x = torch.gather(x, dim=1, index=gather_index)  # (B, k, C)
        # ===== Remaining tokens =====
        remaining_x = x.clone()
        remaining_x.scatter_(1, gather_index, 0.0)

        return token_scores, topk_x, remaining_x, topk_indices

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
                split_extra=False, # we do not need extra here
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

    def hook_after_module(
        self,
        module_name: Literal["self-attn", "cross-attn", "ffn"],
        hidden_output: torch.Tensor,
        encoder_hidden_output: torch.Tensor = None,
    ):
        if module_name == "ffn":
            # gate with topk scores, to maintain the grad
            hidden_output = hidden_output * (
                self.token_scores.unsqueeze(2).to(hidden_output.dtype)
            )
            self.token_scores = None
        return super().hook_after_module(
            module_name, hidden_output, encoder_hidden_output
        )

    @staticmethod
    def calculate_consistency_loss(config: dict, device: torch.device) -> torch.Tensor:
        consistency_loss = []

        for hook in HookManager.get_all_hooks().values():
            if isinstance(hook, ShivaDitBlockHook):
                prune_list = hook.prune_output_accumulator
                fuse_list = hook.fuse_output_accumulator

                if prune_list and fuse_list:
                    assert len(prune_list) == len(
                        fuse_list
                    ), "Prune/Fuse accumulator length mismatch"

                    micro_batch_losses = []
                    for prune_out, fuse_out in zip(prune_list, fuse_list):
                        loss = F.mse_loss(prune_out.detach(), fuse_out)
                        micro_batch_losses.append(loss)

                    if micro_batch_losses:
                        avg_loss_for_hook = torch.stack(micro_batch_losses).mean()
                        consistency_loss.append(avg_loss_for_hook)
                        device = avg_loss_for_hook.device

                hook.prune_output_accumulator.clear()
                hook.fuse_output_accumulator.clear()

        metrics = {
            "shiva/gater/consistency_loss_raw": 0.0,
            "shiva/gater/consistency_loss": 0.0,
        }
        loss_weight = config["consistency_loss_weight"]
        if consistency_loss:
            raw_loss = torch.stack(consistency_loss).mean()
            final_loss = raw_loss * loss_weight
            metrics["shiva/gater/consistency_loss_raw"] = raw_loss.item()
            metrics["shiva/gater/consistency_loss"] = final_loss.item()
            return final_loss, metrics
        else:
            return torch.tensor(0.0, device=device), metrics

    @staticmethod
    def calculate_gate_loss(config: dict, device: torch.device) -> torch.Tensor:
        gate_losses = []

        for hook in HookManager.get_all_hooks().values():
            if isinstance(hook, ShivaDitBlockHook):
                accumulator = hook.gate_loss_accumulator

                if accumulator:
                    avg_loss_for_hook = torch.stack(accumulator).mean()
                    gate_losses.append(avg_loss_for_hook)
                    device = avg_loss_for_hook.device

                hook.gate_loss_accumulator.clear()

        metrics = {
            "shiva/gater/gate_loss_raw": 0.0,
            "shiva/gater/gate_loss": 0.0,
        }
        loss_weight = config["gate_loss_weight"]
        if gate_losses:
            raw_loss = torch.stack(gate_losses).mean()
            final_loss = raw_loss * loss_weight
            metrics["shiva/gater/gate_loss_raw"] = raw_loss.item()
            metrics["shiva/gater/gate_loss"] = final_loss.item()
            return final_loss, metrics
        else:
            return torch.tensor(0.0, device=device), metrics

    @staticmethod
    def calculate_ratio_loss(config: dict, device: torch.device) -> tuple[torch.Tensor, dict]:
        ratios_per_layer = {}
        ratios = []
        for hook in HookManager.get_all_hooks().values():
            if isinstance(hook, ShivaDitBlockHook):
                layer_name = f"layer_{hook.layer_idx:02d}"
                ratios_per_layer[layer_name] = hook.shiva_ratio.item()
                ratios.append(hook.shiva_ratio)

        metrics = {
            "shiva/ratio/avg_ratio": 0.0,
            "shiva/ratio/std_ratio": 0.0,
            "shiva/ratio/ratio_loss_raw": 0.0,
            "shiva/ratio/ratio_loss": 0.0,
        }

        loss_weight = config["ratio_loss_weight"]
        current_target_ratio = config["target_ratio"]
        if ratios:
            ratios_tensor = torch.stack(ratios)
            avg_ratio = torch.mean(ratios_tensor)
            std_ratio = torch.std(ratios_tensor) 
            target_ratio_tensor = torch.tensor(current_target_ratio, device=device, dtype=avg_ratio.dtype)
            mse_loss = F.mse_loss(avg_ratio, target_ratio_tensor)
            final_loss = mse_loss * loss_weight
            metrics = {
                "shiva/ratio/avg_ratio": avg_ratio.item(),
                "shiva/ratio/std_ratio": std_ratio.item(),
                "shiva/ratio/ratio_loss_raw": mse_loss.item(),
                "shiva/ratio/ratio_loss": final_loss.item(),
            }

            return final_loss, metrics, ratios_per_layer
        else:
            return torch.tensor(0.0, device=device), metrics, ratios_per_layer

    @torch.no_grad()
    @staticmethod
    def calculate_shiva_gate_prob(device: torch.device) -> tuple[torch.Tensor, dict]:
        gate_probs_per_hook = []
        gate_probs_per_layer = {}

        for name, hook in HookManager.get_all_hooks().items():
            if isinstance(hook, ShivaDitBlockHook):
                accumulator = hook.gater_fuse_prob_accumulator

                if accumulator:
                    avg_prob_for_hook = torch.stack(accumulator).mean()
                    gate_probs_per_hook.append(avg_prob_for_hook)

                    layer_name = f"layer_{hook.layer_idx:02d}"
                    gate_probs_per_layer[layer_name] = avg_prob_for_hook.item()

                hook.gater_fuse_prob_accumulator.clear()

        if gate_probs_per_hook:
            global_avg_prob = torch.stack(gate_probs_per_hook).mean()
            return global_avg_prob, gate_probs_per_layer
        else:
            return torch.tensor(0.0, device=device), {}

    @staticmethod
    @torch.no_grad()
    def collect_router_stats(device: torch.device) -> dict:
        total_mean = []
        total_std = []
        total_min = []

        for hook in HookManager.get_all_hooks().values():
            if isinstance(hook, ShivaDitBlockHook):
                acc = hook.router_stats_accumulator
                if acc:
                    total_mean.append(torch.stack([x["mean"] for x in acc]).mean())
                    total_std.append(torch.stack([x["std"] for x in acc]).mean())
                    total_min.append(torch.stack([x["min"] for x in acc]).min())
                acc.clear()

        if total_mean:
            return {
                "shiva/router/mean_score": torch.stack(total_mean).mean(),
                "shiva/router/std_score": torch.stack(total_std).mean(),
                "shiva/router/min_score": torch.stack(total_min).min(),
            }
        else:
            return {}

if __name__ == "__main__":
    from shiva_dit.utils import get_pipeline, process_quick_start
    model_type = "qwen-image-edit"
    pipe, model_attrs = get_pipeline(model_type)

    shiva_config = {
        "bottleneck_dim": 64,
        "layer_embed_dim": 64,
        "num_router_groups": 4,
        "fuse_expert_ffn_mult": 2,
        "context_pruning_threshold": None,
        "context_pruning_ratio": None,
        "initial_kept_ratio": 1.0,
        "use_gater_lut": False,
    }

    HookManager.configure_hooks(
        model_type=model_type,
        default=(ShivaDitBlockHook, shiva_config),
    )

    with HookManager.enabled():
        HookManager.inject_hooks(
            model_type=model_type, pipe=pipe, model_attrs=model_attrs
        )
        res = process_quick_start(pipe)
        res.save(f"{model_type}_shiva.png")

    print(global_context)
