import inspect
import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm
from shiva_dit.hook.hook_manager import HookManager
from shiva_dit.training.trainer import DiffusionModelTrainer
from shiva_dit.training.config import TrainerConfig
from shiva_dit.utils import global_context
from shiva_dit.hook.delegate import BaseDelegator
from shiva_dit.hook.block_hook import BaseDiTBlockHook
from shiva_dit.modules.pruner import ShivaContextPruner

class ContextPrunerDistillHook(BaseDelegator, BaseDiTBlockHook, nn.Module):
    def __init__(self, model_dim, prompt_embed_dim, context_pool_size, context_proj_dim, **kwargs):
        super().__init__(**kwargs)
        self.context_pruner  = ShivaContextPruner(
            10.0,
            10.0,
            mode="learned",
            pool_size=context_pool_size,
            img_dim=model_dim,
            txt_dim=prompt_embed_dim,
            proj_dim=context_proj_dim,
        )

        self.learned_score = self.target_score = None
        self.record_spearman_score = False
        self.spearman_score = []
    
    def on_register_hook(self, block: nn.Module):
        self.ops = ShivaContextPruner._get_free_ops(block)

    def delegated_call(self, func, module_self, *args, **kwargs):
        sig = inspect.signature(func)
        bound = sig.bind_partial(module_self, *args, **kwargs)
        bound.apply_defaults()

        hidden_states = bound.arguments.get("hidden_states")
        encoder_hidden_states = bound.arguments.get("encoder_hidden_states")

        h_in = hidden_states
        e_in = encoder_hidden_states
        H, W = global_context.hidden_seq_height, global_context.hidden_seq_width

        self.learned_score = self.context_pruner._forward_learned(h_in, e_in, H, W) # (B, L)
        self.target_score = ShivaContextPruner._process_attn(h_in, e_in, H, W, self.ops, self.context_pruner.pool_size, enable_importance=False).max(dim=1).values  # (B, L)
        if self.record_spearman_score:
            self.spearman_score.append(self.compute_spearman_rank())

        return func(module_self, *args, **kwargs)
    
    def kl_loss(self, temperature=1.0):
        p_s = F.log_softmax(self.learned_score / temperature, dim=-1)
        with torch.no_grad():
            p_t = F.softmax(self.target_score / temperature, dim=-1)

        kl_loss = F.kl_div(p_s, p_t, reduction='batchmean') * (temperature ** 2)

        return kl_loss
    
    @torch.no_grad()
    def compute_spearman_rank(self):
        def get_rank(batch_scores):
            return torch.argsort(torch.argsort(batch_scores, dim=-1), dim=-1).float()

        rank_s = get_rank(self.learned_score)
        rank_t = get_rank(self.target_score)
        
        n = rank_s.size(-1)
        d_squared = (rank_s - rank_t).pow(2).sum(dim=-1)
        
        rho = 1 - (6 * d_squared) / (n * (n**2 - 1))
        return rho.mean()


class ContextPrunerDistiller(DiffusionModelTrainer):
    def __init__(self, config: TrainerConfig, load_evaluators: bool = True, static_graph: bool = True, find_unused_parameters: bool = False):
        super().__init__(config, load_evaluators, static_graph, find_unused_parameters)
    
    def _setup_hooks(self):
        config = self.config
        self.hook_type = None
        if config.trainable_hook_config:
            hook_class, hook_kwargs = config.trainable_hook_config

            if hook_class.__name__ == "ShivaDitBlockHook":
                self.hook_type = "context_distill"
                assert hook_kwargs['shiva_mode'] != 'image', "image mode is not supported for context pruner distiller"
                assert hook_kwargs['context_pruner_mode'] == 'learned', "context_pruner_mode must be 'learned' for context pruner distiller"


                hook_kwargs["capture_mask_for_viz"] = False
                hook_kwargs["skip_layer_indices"] = []
                if self.accelerator.is_main_process:
                    print(
                        "Hook type: Shiva. Will use shared injection via setup_shared_hooks."
                    )
            else:
                raise ValueError(f"Unknown hook class: {hook_class.__name__}")
            
            model_dim = global_context.hidden_dim
            prompt_embed_dim = global_context.encoder_hidden_dim
            if model_dim is None:
                raise ValueError("[ShivaInstaller] 'hidden_dim' not found in global_context.")
            if prompt_embed_dim is None:
                raise ValueError("[ShivaInstaller] 'encoder_hidden_dim' not found in global_context.")
            
            print(f"model_dim: {model_dim}, prompt_embed_dim: {prompt_embed_dim}, pool_size: {hook_kwargs['context_pool_size']}, proj_dim: {hook_kwargs['context_proj_dim']}")
            HookManager.configure_hooks(
                model_type=config.model_type,
                default=(ContextPrunerDistillHook, {
                    "model_dim": model_dim,
                    "prompt_embed_dim": prompt_embed_dim,
                    "context_pool_size": hook_kwargs['context_pool_size'],
                    "context_proj_dim": hook_kwargs['context_proj_dim'],
                }),
            )
            HookManager.enable()
            HookManager.inject_hooks(
                model_type=config.model_type,
                pipe=self.pipe,
                model_attrs=self.model_attrs,
            )
    

    def _compute_loss(self, batch, global_step: int, is_validation: bool = False):
        hook_loss = torch.tensor(0.0, device=self.device, dtype=self.dtype)
        hook_metrics = {}

        img_latents, noise, noisy_latents, timesteps, model_kwargs = (
            self.prepare_model_inputs(batch)
        )

        noisy_latents = noisy_latents.to(self.dtype).detach()

        noise_pred, distill_loss_final, distill_loss_mid, feat_sim = (
            self._forward_pass_with_distillation(
                noisy_latents,
                timesteps,
                model_kwargs,
                dynamic_distill_config=None,
            )
        )

        main_loss = 0
        for hook in HookManager.get_all_hooks().values():
            if isinstance(hook, ContextPrunerDistillHook):
                kl_loss = hook.kl_loss()
                main_loss += kl_loss

        total_loss = main_loss

        return (
            total_loss,
            main_loss,
            hook_loss,
            distill_loss_final,
            distill_loss_mid,
            hook_metrics,
            timesteps,
            feat_sim,
        )
    
    @torch.no_grad()
    def validate(self, global_step):
        if self.val_dataloader is None or not self.config.run_validation:
            return

        if self.accelerator.is_main_process:
            print(f"\nRunning validation at step {global_step}...")

        if self.hook_type == "context_distill":
            for hook in HookManager.get_all_hooks().values():
                if hasattr(hook, "record_spearman_score"):
                    hook.record_spearman_score = True

        self.model.eval()

        all_total_losses = []
        all_main_losses = []
        all_spearman_scores = []

        val_progress_bar = tqdm(
            self.val_dataloader,
            desc=f"Validation Step {global_step}",
            leave=False,
            total=len(self.val_dataloader),
            disable=not self.accelerator.is_local_main_process,
        )

        for batch in val_progress_bar:
            try:
                (
                    total_loss,
                    main_loss,
                    hook_loss,
                    distill_loss_final,
                    distill_loss_mid,
                    hook_metrics,
                    timesteps,
                    feat_sim,
                ) = self._compute_loss(batch, global_step, is_validation=True)

                all_total_losses.append(total_loss)
                all_main_losses.append(main_loss)

                for hook in HookManager.get_all_hooks().values():
                    if isinstance(hook, ContextPrunerDistillHook):
                        all_spearman_scores.extend(hook.spearman_score)

            except Exception as e:
                if self.accelerator.is_main_process:
                    print(f"Error during validation batch: {e}")
                continue

        avg_val_loss = None
        if len(self.val_dataloader) > 0:
            avg_val_loss = (
                self.accelerator.gather(torch.stack(all_total_losses)).mean().item()
            )
            avg_main_loss = (
                self.accelerator.gather(torch.stack(all_main_losses)).mean().item()
            )
            avg_spearman_score = (
                self.accelerator.gather(torch.stack(all_spearman_scores)).mean().item()
            )

            if self.accelerator.is_main_process:
                print(f"--- Validation Results (Step {global_step}) ---")
                print(f"  Avg Total Loss: {avg_val_loss:.4f}")
                print(f"  Avg Main Loss:  {avg_main_loss:.4f}")
                print(f"  Avg Spearman Score:  {avg_spearman_score:.4f}")
                print(f"--------------------------------------")

                self.logger.log_scalars(
                    {
                        "validation/total_loss": avg_val_loss,
                        "validation/main_loss": avg_main_loss,
                        "validation/spearman_score": avg_spearman_score,
                    },
                    step=global_step,
                )

        elif self.accelerator.is_main_process:
            print(
                f"Validation dataloader is empty or validation failed, skipping logging."
            )

        if self.accelerator.is_main_process:
            self._log_policy_visualization()

        self.model.train()
        if self.hook_type == "context_distill":
            for hook in HookManager.get_all_hooks().values():
                if hasattr(hook, "record_spearman_score"):
                    hook.record_spearman_score = False
                    hook.spearman_score = []

        return avg_main_loss if "avg_main_loss" in locals() else None
