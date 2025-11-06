import os
import json
import copy
import re
import glob
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from transformers import get_scheduler
from peft import LoraConfig as PeftLoraConfig, get_peft_model
from tqdm import tqdm
from accelerate import Accelerator
from accelerate.utils import set_seed, DistributedDataParallelKwargs, broadcast_object_list
from dataclasses import asdict

from baselines.diffcr import DiffCRDitBlockHook
from baselines.dy_dit import DynamicDiTBlockHook
from diffusers.schedulers import FlowMatchEulerDiscreteScheduler
from shiva_dit.hook import HookManager
from shiva_dit.shiva import ShivaDitBlockHook
from shiva_dit.utils import get_pipeline
from shiva_dit.training.input_preparers import get_input_preparer
from shiva_dit.training.config import TrainerConfig
from shiva_dit.training.dataset import ImageTextDataset, collate_fn
from shiva_dit.training.logging import NoOpLogger
from shiva_dit.utils import global_context


class DiffusionModelTrainer:
    def __init__(self, config: TrainerConfig):
        self.config = config
        self.model_type = config.model_type
        
        # shiva's extra schedule
        self.consistency_schedule = self.config.extra_config.get("consistency_schedule", [1.0, 0.1, 0.1])
        self.gate_schedule = self.config.extra_config.get("gate_schedule", [0.0, 0.0, 0.01])
        self.schedule_steps = self.config.extra_config.get("schedule_steps", [5000, 10000])

        self.input_preparer_fn = get_input_preparer(self.model_type)

        ## Initialize accelerator, device, seed and generator
        ddp_kwargs = DistributedDataParallelKwargs(static_graph=True)
        self.accelerator = Accelerator(
            mixed_precision=config.mixed_precision,
            gradient_accumulation_steps=config.gradient_accumulation_steps,
            log_with="tensorboard" if config.log_with_tensorboard else None,
            project_dir=os.path.join(config.output_dir, config.tensorboard_log_dir),
            kwargs_handlers=[ddp_kwargs],
        )
        self.device = self.accelerator.device
        set_seed(config.seed)
        self.generator = torch.Generator(device=self.accelerator.device).manual_seed(
            config.seed
        )

        self._setup_models_and_components()

        self._setup_hooks()

        self.optimizer = self._setup_model_and_optimizer()

        self._setup_dataloaders()

        num_update_steps_per_epoch = max(
            1, len(self.train_dataloader) // config.gradient_accumulation_steps
        )
        self.num_training_steps = num_update_steps_per_epoch * config.num_epochs
        self.lr_scheduler = self._setup_scheduler()

        self._prepare_with_accelerator()

        self.logger = self._setup_logging()

        self._setup_caption_dropout_embeds()

        self._print_setup_summary()

    @staticmethod
    def _get_dtype(self, mixed_precision: str):
        if mixed_precision == "bf16":
            dtype = torch.bfloat16
        elif mixed_precision == "fp16":
            dtype = torch.float16
        else:
            dtype = torch.float32
        return dtype

    def _setup_models_and_components(self):
        config = self.config
        # get pipeline and components
        self.pipe, self.model_attrs = get_pipeline(
            self.model_type,
            model_path=config.model_path,
            dtype=DiffusionModelTrainer._get_dtype(self, config.mixed_precision),
        )

        self.image_processor = self.pipe.image_processor
        self.vae = self.pipe.vae
        self.scheduler = self.pipe.scheduler
        self.model = tuple(
            getattr(self.pipe, model) for model in self.model_attrs
        )  # DiT
        assert len(self.model) == 1, "Currently only single DiT model is supported."
        self.model = self.model[0]

        # distillation setup
        self.teacher_model = None
        if self.config.distillation_final_weight > 0 or self.config.distillation_middle_weight > 0:
            if self.accelerator.is_main_process:
                print("Distillation loss enabled. Loading static teacher model...")

            teacher_pipe, teacher_attrs = get_pipeline(
                self.model_type,
                model_path=config.model_path,
                dtype=DiffusionModelTrainer._get_dtype(self, config.mixed_precision),
            )
            self.teacher_model = getattr(teacher_pipe, teacher_attrs[0])
            assert len(teacher_attrs) == 1, "Currently only single DiT model is supported."
            self.teacher_model.requires_grad_(False)
            self.teacher_model.eval()
            self.teacher_model = self.teacher_model.to(self.device)
            del teacher_pipe # only need DiT model
            if self.accelerator.is_main_process:
                print("Teacher model loaded.")

        # get tokenizers and text encoders
        if self.model_type in ["flux-dev", "flux-kontext"]:
            self.tokenizers = (self.pipe.tokenizer, self.pipe.tokenizer_2)
            self.text_encoders = (self.pipe.text_encoder, self.pipe.text_encoder_2)
            self.text_encoder_names = ("text_encoder", "text_encoder_2")
        elif self.model_type in ["pixart-sigma", "qwen-image", "qwen-image-edit", "wan"]:
            self.tokenizers = (self.pipe.tokenizer,)
            self.text_encoders = (self.pipe.text_encoder,)
            self.text_encoder_names = ("text_encoder",)
        elif self.model_type == "sd3":
            self.tokenizers = (
                self.pipe.tokenizer,
                self.pipe.tokenizer_2,
                self.pipe.tokenizer_3,
            )
            self.text_encoders = (
                self.pipe.text_encoder,
                self.pipe.text_encoder_2,
                self.pipe.text_encoder_3,
            )
            self.text_encoder_names = (
                "text_encoder",
                "text_encoder_2",
                "text_encoder_3",
            )
        else:
            raise ValueError(f"Undefined model_type: {self.model_type}")

        if self.image_processor is None or self.vae is None or self.model is None:
            raise ValueError("NoneType found in image_processor, vae or model.")

        # freeze vae and text encoder(s)
        self.vae.requires_grad_(False)
        self.vae.eval()
        for text_encoder in self.text_encoders:
            text_encoder.requires_grad_(False)
            text_encoder.to(dtype=torch.float32) # keep text encoder in fp32 for stability
            text_encoder.eval()

    def _setup_hooks(self):
        config = self.config
        self.hook_type = None
        if config.trainable_hook_config:
            hook_class, hook_kwargs = config.trainable_hook_config

            if hook_class.__name__ == "DynamicDiTBlockHook":
                self.hook_type = "dydit"
            elif hook_class.__name__ == "DiffCRDitBlockHook":
                self.hook_type = "diffcr"
            elif hook_class.__name__ == "ShivaDitBlockHook":
                self.hook_type = "shiva"
                if self.accelerator.is_main_process:
                    print("Hook type: Shiva. Will use shared injection via setup_shared_hooks.")
            else:
                raise ValueError(f"Unknown hook class: {hook_class.__name__}")

            HookManager.configure_hooks(
                model_type=config.model_type,
                default=(hook_class, hook_kwargs),
            )
            HookManager.enable()
            HookManager.inject_hooks(
                model_type=config.model_type,
                pipe=self.pipe,
                model_attrs=self.model_attrs,
            )

    def _setup_dataloaders(self):
        config = self.config
        self.train_dataset = ImageTextDataset(
            model_type=self.model_type,
            metadata_file=config.metadata_file,
            image_dir=config.image_dir,
            image_processor=self.pipe.image_processor,
            tokenizers=self.tokenizers,
            pipe=self.pipe,
        )
        self.train_dataloader = DataLoader(
            self.train_dataset,
            batch_size=config.train_batch_size,
            shuffle=True,
            collate_fn=collate_fn,
        )

        self.val_dataloader = None
        if config.val_metadata_file and config.val_image_dir:
            try:
                val_dataset = ImageTextDataset(
                    model_type=self.model_type,
                    metadata_file=config.val_metadata_file,
                    image_dir=config.val_image_dir,
                    image_processor=self.pipe.image_processor,
                    tokenizers=self.tokenizers,
                    pipe=self.pipe,
                )
                val_batch_size = (
                    config.val_batch_size
                    if config.val_batch_size
                    else config.train_batch_size
                )
                self.val_dataloader = DataLoader(
                    val_dataset,
                    batch_size=val_batch_size,
                    shuffle=False,
                    collate_fn=collate_fn,
                )
            except Exception as e:
                print(
                    f"Warning: Failed to load validation dataset. Validation will be skipped. Error: {e}"
                )
                self.val_dataloader = None

    def _prepare_with_accelerator(self):
        config = self.config
        prepare_list = [
            self.model,
            self.optimizer,
            self.lr_scheduler,
            self.train_dataloader,
        ]

        if self.teacher_model is not None:
            prepare_list.insert(1, self.teacher_model)

        prepared = self.accelerator.prepare(*prepare_list)

        idx = 0
        self.model = prepared[idx]; idx += 1
        if self.teacher_model is not None:
            self.teacher_model = prepared[idx]; idx += 1
        self.optimizer = prepared[idx]; idx += 1
        self.lr_scheduler = prepared[idx]; idx += 1
        self.train_dataloader = prepared[idx]; idx += 1
        if self.val_dataloader is not None:
            self.val_dataloader = self.accelerator.prepare(self.val_dataloader)

        unwrapped_model = self.accelerator.unwrap_model(self.model)
        self.dtype = unwrapped_model.dtype # final model dtype after accelerator prepare

        if self.accelerator.is_main_process:
            print("model_type:", config.model_type)
            print("image_processor:", self.image_processor.__class__.__name__)
            print("scheduler:", self.scheduler.__class__.__name__)
            print("vae:", self.vae.__class__.__name__, "dtype:", self.vae.dtype)
            print("text_encoder:", self.text_encoders[0].__class__.__name__, "dtype:", self.text_encoders[0].dtype)
            print("model (DiT):", self.model.__class__.__name__, "dtype:", self.dtype)

    def _setup_logging(self):
        if not self.accelerator.is_main_process:
            return NoOpLogger()
        config = self.config
        try:
            from shiva_dit.training.logging import (
                TensorBoardLogger, WandbLogger, MLflowLogger, LoggerCollection
            )
        except ImportError as e:
            print(f"Not able to import logging modules: {e}. Logging will be disabled.")
            return NoOpLogger()

        loggers = []

        if config.log_with_tensorboard:
            try:
                tb_log_path = os.path.join(config.output_dir, config.tensorboard_log_dir)
                loggers.append(TensorBoardLogger(log_dir=tb_log_path))
            except ImportError:
                print("Warning: Unable to import TensorBoard, skipping TensorBoard logging.")

        if getattr(config, "log_with_wandb", False):
            try:
                loggers.append(
                    WandbLogger(
                        project=getattr(config, "wandb_project", "diffusion_trainer"),
                        name=getattr(
                            config,
                            "wandb_run_name",
                            os.path.basename(config.output_dir),
                        ),
                        config_dict=asdict(config),
                        entity=getattr(config, "wandb_entity", None),
                    )
                )
            except ImportError:
                print("Warning: Unable to import wandb, skipping Wandb logging.")

        if getattr(config, "log_with_mlflow", False):
            try:
                run_name = config.mlflow_run_name or config.wandb_run_name or os.path.basename(config.output_dir)

                loggers.append(MLflowLogger(
                    experiment_name=config.mlflow_experiment_name,
                    run_name=run_name,
                    config_dict=asdict(config),
                    tracking_uri=config.mlflow_tracking_uri
                ))
            except ImportError:
                print("Warning: Unable to import mlflow, skipping MLflow logging.")

        if not loggers:
            print("No logging backends available, logging will be disabled.")
            return NoOpLogger()

        return LoggerCollection(loggers)

    def _setup_model_and_optimizer(self):
        # freeze VAE and Text Encoder(s)
        self.vae.requires_grad_(False)
        for text_encoder in self.text_encoders:
            text_encoder.requires_grad_(False)

        params_to_train = []

        if self.config.training_mode == "hook_only":
            if self.accelerator.is_main_process:
                print("Training mode: hook_only (only training Hook parameters)")
            self.model.requires_grad_(False)
            for hook in HookManager.get_all_hooks().values():
                if isinstance(hook, nn.Module):
                    hook.requires_grad_(True)
                    params_to_train.extend(hook.parameters())

            if not params_to_train:
                raise ValueError(
                    "No trainable Hook parameters found in 'hook_only' mode."
                )
        elif self.config.training_mode == "finetune":
            if self.accelerator.is_main_process:
                print("Training mode: finetune (training all DiT parameters)")
            self.model.requires_grad_(True)
            params_to_train = list(self.model.parameters())
            for hook in HookManager.get_all_hooks().values():
                if isinstance(hook, nn.Module):
                    hook.requires_grad_(True)
                    params_to_train.extend(hook.parameters())
        elif self.config.training_mode == "lora":
            if self.accelerator.is_main_process:
                print("Training mode: lora (applying LoRA to DiT model)")
            if not self.config.lora_config.enable:
                raise ValueError(
                    "LoRA configuration is disabled in 'lora' training mode."
                )

            self.model.requires_grad_(False)
            for hook in HookManager.get_all_hooks().values():
                if isinstance(hook, nn.Module):
                    hook.requires_grad_(False)

            lora_conf = self.config.lora_config
            peft_config = PeftLoraConfig(
                r=lora_conf.rank,
                lora_alpha=lora_conf.lora_alpha,
                target_modules=lora_conf.target_modules,
                lora_dropout=lora_conf.lora_dropout,
                bias="none",
            )
            self.model = get_peft_model(self.model, peft_config)
            self.model.print_trainable_parameters()
            for hook in HookManager.get_all_hooks().values():
                if isinstance(hook, nn.Module):
                    hook.requires_grad_(True)
            params_to_train = [p for p in self.model.parameters() if p.requires_grad]
        else:
            raise ValueError(f"Unknown training mode: {self.config.training_mode}")

        if self.config.optimizer == "AdamW":
            optimizer = torch.optim.AdamW(
                params_to_train,
                lr=self.config.learning_rate,
                weight_decay=self.config.weight_decay,
            )
        else:
            # TODO: add more optimizers if needed
            raise ValueError(f"Unknown optimizer: {self.config.optimizer}")

        return optimizer

    def _setup_scheduler(self):
        scheduler = get_scheduler(
            self.config.lr_scheduler,
            optimizer=self.optimizer,
            num_warmup_steps=self.config.num_warmup_steps,
            num_training_steps=self.num_training_steps,
        )
        return scheduler

    def _setup_caption_dropout_embeds(self):
        if self.config.caption_dropout_rate > 0:
            if self.accelerator.is_main_process:
                print(
                    f"Caption dropout enabled with rate: {self.config.caption_dropout_rate}"
                )
            if self.model_type == "pixart-sigma":
                self.empty_prompt_embeds, self.empty_prompt_attention_mask, _, _ = (
                    self.pipe.encode_prompt(
                        "",
                        do_classifier_free_guidance=False,
                        device=self.device,
                        clean_caption=True,
                    )
                )
                self.empty_prompt_embeds = self.empty_prompt_embeds.to(self.dtype)
                self.empty_prompt_attention_mask = self.empty_prompt_attention_mask.to(
                    self.dtype
                )
            elif self.model_type in ["flux-dev", "flux-kontext"]:
                (
                    self.empty_prompt_embeds,
                    self.empty_pooled_prompt_embeds,
                    self.empty_text_ids,
                ) = self.pipe.encode_prompt(
                    prompt="",
                    device=self.device,
                    num_images_per_prompt=1,
                )
                self.empty_prompt_embeds = self.empty_prompt_embeds.to(self.dtype)
                self.empty_pooled_prompt_embeds = self.empty_pooled_prompt_embeds.to(
                    self.dtype
                )
                self.empty_text_ids = self.empty_text_ids.to(self.dtype)
            elif self.model_type == "sd3":
                (
                    self.empty_prompt_embeds,
                    _,
                    self.empty_pooled_prompt_embeds,
                    _,
                ) = self.pipe.encode_prompt(
                    prompt="",
                    prompt_2=None,
                    prompt_3=None,
                    device=self.device,
                    do_classifier_free_guidance=False,
                    num_images_per_prompt=1,
                )
                self.empty_prompt_embeds = self.empty_prompt_embeds.to(self.dtype)
                self.empty_pooled_prompt_embeds = self.empty_pooled_prompt_embeds.to(
                    self.dtype
                )
            else:
                raise NotImplementedError(
                    "Caption dropout is not implemented for model_type: " + self.model_type
                )

    def _print_setup_summary(self):
        if self.accelerator.is_main_process:
            config = self.config
            print(f"--- Training Config ---")
            print(f"  Mode: {config.training_mode}")
            print(f"  Micro Batch Size: {config.train_batch_size}")
            print(f"  Grad Acc Step: {config.gradient_accumulation_steps}")
            print(
                f"  Actual Batch Size: {config.train_batch_size * config.gradient_accumulation_steps}"
            )
            print(f"  Total training steps: {self.num_training_steps}")
            print(f"------------------")

    def _prepare_model_inputs(self, batch):
        img_latents, noise, noisy_latents, timesteps, model_kwargs = (
            self.input_preparer_fn(self, batch)
        )

        # update global context
        # TODO: support all attr in global_context
        patch_size = 2
        hidden_seq_height = noise.shape[2] // patch_size
        hidden_seq_width = noise.shape[3] // patch_size
        hidden_seq_len = noise.shape[2] // patch_size * noise.shape[3] // patch_size
        global_context.update(hidden_seq_len=hidden_seq_len)
        global_context.update(hidden_seq_width=hidden_seq_width)
        global_context.update(hidden_seq_height=hidden_seq_height)
        if "encoder_hidden_states" in model_kwargs:
            global_context.update(
                encoder_hidden_seq_len=model_kwargs["encoder_hidden_states"].size(1)
            )

        if self.model_type == "flux-dev" or self.model_type == "flux-kontext":
            # pack for hidden state
            batch_size = batch["pixel_values"].shape[0]
            height, width = noisy_latents.shape[-2:]
            num_channels_latents = self.pipe.transformer.config.in_channels // 4
            img_latents = self.pipe._pack_latents(
                img_latents, batch_size, num_channels_latents, height, width
            )
            noisy_latents = self.pipe._pack_latents(
                noisy_latents, batch_size, num_channels_latents, height, width
            )
            noise = self.pipe._pack_latents(
                noise, batch_size, num_channels_latents, height, width
            )
            timesteps = timesteps / 1000

        return img_latents, noise, noisy_latents, timesteps, model_kwargs

    def _forward_pass_with_distillation(self, noisy_latents, timesteps, model_kwargs):
        distill_loss = torch.tensor(0.0, device=self.device, dtype=self.dtype)

        with torch.amp.autocast(
            device_type="cuda",
            enabled=(self.config.mixed_precision != "no"),
            dtype=(
                torch.bfloat16
                if self.config.mixed_precision == "bf16"
                else (torch.float16 if self.config.mixed_precision == "fp16" else None)
            ),
        ):
            distillation_enabled = self.teacher_model is not None and (
                self.config.distillation_final_weight > 0
                or self.config.distillation_middle_weight > 0
            )

            if distillation_enabled:
                if self.config.distillation_middle_weight > 0:
                    model_kwargs["output_hidden_states"] = True
                    model_kwargs["output_hidden_states_step"] = (
                        self.config.distillation_layer_skip
                    )
                else:
                    model_kwargs["output_hidden_states"] = False

                teacher_kwargs = copy.deepcopy(model_kwargs)

                with torch.no_grad():
                    teacher_output = self.teacher_model(
                        hidden_states=noisy_latents,
                        timestep=timesteps,
                        **teacher_kwargs,
                    )
                    Y_s = teacher_output[0]

                student_output = self.model(
                    hidden_states=noisy_latents,
                    timestep=timesteps,
                    **model_kwargs,
                )
                Y_d = student_output[0]
                noise_pred = Y_d

                loss_v = F.mse_loss(Y_d.float(), Y_s.float(), reduction="mean")
                distill_loss += loss_v * self.config.distillation_final_weight

                if self.config.distillation_middle_weight > 0:
                    X_s_all = teacher_output[1]
                    X_d_all = student_output[1]
                    loss_u_total = torch.tensor(
                        0.0, device=self.device, dtype=self.dtype
                    )
                    num_layers_compared = 0

                    layer_indices = range(0, len(X_s_all))

                    for l in layer_indices:
                        loss_u_total += F.mse_loss(
                            X_d_all[l].float(), X_s_all[l].float(), reduction="mean"
                        )
                        num_layers_compared += 1

                    if num_layers_compared > 0:
                        loss_u_avg = loss_u_total / num_layers_compared
                        distill_loss += (
                            loss_u_avg * self.config.distillation_middle_weight
                        )
            else:
                noise_pred = self.model(
                    hidden_states=noisy_latents,
                    timestep=timesteps,
                    **model_kwargs,
                )[0]

        return noise_pred, distill_loss

    def _calculate_main_loss(self, noise_pred, noise, img_latents):
        if self.model_type == "pixart-sigma":
            noise_pred = noise_pred.chunk(2, dim=1)[0]

        if isinstance(self.scheduler, FlowMatchEulerDiscreteScheduler):
            # in flow-matching, we need to compute the noise target as vector field
            noise_target = noise - img_latents
        else:
            noise_target = noise

        main_loss = F.mse_loss(
            noise_pred.float(), noise_target.float(), reduction="mean"
        ) * self.config.main_loss_weight

        return main_loss

    def _calculate_hook_loss(self, global_step: int):
        if self.hook_type == "dydit":
            hook_loss = DynamicDiTBlockHook.calculate_extra_loss(
                self.config.extra_config, device=self.device
            )
        elif self.hook_type == "diffcr":
            hook_loss = DiffCRDitBlockHook.calculate_extra_loss(
                self.config.extra_config, device=self.device
            )
        elif self.hook_type == "shiva":
            current_consistency_weight = self.consistency_schedule[0]
            current_gate_weight = self.gate_schedule[0]

            scaled_schedule_steps_0 = max(
                1, self.schedule_steps[0] // self.accelerator.num_processes
            )
            scaled_schedule_steps_1 = max(
                1, self.schedule_steps[1] // self.accelerator.num_processes
            )
            if global_step > scaled_schedule_steps_0:
                current_consistency_weight = self.consistency_schedule[1]
                current_gate_weight = self.gate_schedule[1]
            if global_step > scaled_schedule_steps_1:
                current_consistency_weight = self.consistency_schedule[2]
                current_gate_weight = self.gate_schedule[2]
            
            def get_annealing_weight(global_step, start_step, end_step):
                if global_step < start_step:
                    return 0.0
                progress = (global_step - start_step) / max(1.0, end_step - start_step)
                return min(1.0, progress)
            
            start_step = self.config.hook_warmup_steps
            scaled_start_step = max(
                1, start_step // self.accelerator.num_processes
            )
            anneal_steps = self.config.extra_config["anneal_steps"]
            scaled_anneal_steps = max(
                1, anneal_steps // self.accelerator.num_processes
            )
            anneal_factor = get_annealing_weight(global_step, scaled_start_step, scaled_anneal_steps)
            base_lambda = self.config.extra_config["ratio_loss_weight"]
            current_ratio_weight = base_lambda * anneal_factor

            loss_config = {
                "ratio_loss_weight": current_ratio_weight,
                "consistency_loss_weight": current_consistency_weight,
                "gate_loss_weight": current_gate_weight,
            }

            hook_loss, consistency_loss, gate_loss, ratio_loss = (
                ShivaDitBlockHook.calculate_extra_loss(
                    loss_config, device=self.device
                )
            )

            self.last_ratio_loss = ratio_loss.detach()
            self.last_gate_loss = gate_loss.detach()
            self.last_consistency_loss = consistency_loss.detach()
        else:
            hook_loss = torch.tensor(0.0, device=self.device, dtype=self.dtype)

        current_hook_loss_weight = self.config.hook_loss_weight

        if self.config.hook_warmup_steps > 0:
            scaled_hook_warmup_steps = max(
                1, self.config.hook_warmup_steps // self.accelerator.num_processes
            )
            if global_step < scaled_hook_warmup_steps:
                current_hook_loss_weight = 1e-9

        return hook_loss * current_hook_loss_weight

    def _compute_loss(self, batch, global_step: int):
        img_latents, noise, noisy_latents, timesteps, model_kwargs = (
            self._prepare_model_inputs(batch)
        )

        noisy_latents = noisy_latents.to(self.dtype).detach()

        noise_pred, distill_loss = self._forward_pass_with_distillation(
            noisy_latents, timesteps, model_kwargs
        )

        main_loss = self._calculate_main_loss(noise_pred, noise, img_latents)
        hook_loss = self._calculate_hook_loss(global_step)
        total_loss = main_loss + hook_loss + distill_loss

        return total_loss, main_loss, hook_loss, distill_loss

    def _train_step(self, batch):
        total_loss, main_loss, hook_loss, distill_loss = self._compute_loss(
            batch, self.global_step
        )
        self.accelerator.backward(total_loss)
        return total_loss.detach(), main_loss.detach(), hook_loss.detach(), distill_loss.detach()

    def _set_hook_trainable_state(self, requires_grad: bool):
        if not self.config.trainable_hook_config:
            return

        if self.accelerator.is_main_process:
            print(f"\n[Step {self.global_step}]: Setting hook parameters' requires_grad = {requires_grad}")

        for hook in HookManager.get_all_hooks().values():
            if hook.__class__.__name__ == "DiffCRDitBlockHook":
                if hasattr(hook, "diff_ratio"):
                    if hook.diff_ratio.requires_grad != requires_grad:
                        hook.diff_ratio.requires_grad = requires_grad
                        if self.accelerator.is_main_process:
                            print(f"  - Set DiffCR.diff_ratio.requires_grad = {requires_grad}")
                else:
                    if self.accelerator.is_main_process:
                        print(f"  - WARNING: Found DiffCRDitBlockHook without 'diff_ratio' param.")

            elif hook.__class__.__name__ == "DynamicDiTBlockHook":
                if hasattr(hook, "dynamic_token_selector") and hook.dynamic_token_selector is not None:
                    if any(p.requires_grad != requires_grad for p in hook.dynamic_token_selector.parameters()):
                        hook.dynamic_token_selector.requires_grad_(requires_grad)
                        if self.accelerator.is_main_process:
                            print(f"  - Set DyDiT.dynamic_token_selector.requires_grad_ = {requires_grad}")
                else:
                    if self.accelerator.is_main_process:
                        print(f"  - WARNING: Found DynamicDiTBlockHook without 'dynamic_token_selector'.")

            elif hook.__class__.__name__ == "ShivaDitBlockHook":
                if hasattr(hook, "shiva_ratio"):
                    if hook.shiva_ratio.requires_grad != requires_grad:
                        hook.shiva_ratio.requires_grad = requires_grad
                        if self.accelerator.is_main_process:
                            print(f"  - Set Shiva.shiva_ratio.requires_grad = {requires_grad}")
                else:
                    if self.accelerator.is_main_process:
                        print(f"  - WARNING: Found ShivaDitBlockHook without 'shiva_ratio' param.")


    def _set_finetuning_trainable_state(self, requires_grad: bool):        
        if self.accelerator.is_main_process:
            print(f"\n[Step {self.global_step}]: Setting finetuning parameters' requires_grad = {requires_grad}")

        unwrapped_model = self.accelerator.unwrap_model(self.model)
        for name, param in unwrapped_model.named_parameters():
            if self.config.training_mode == "lora":                
                if "lora_" in name:
                    if param.requires_grad != requires_grad:
                        param.requires_grad = requires_grad
            elif self.config.training_mode == "finetune":
                if "_hook" not in name:
                    if param.requires_grad != requires_grad:
                        param.requires_grad = requires_grad

    def _on_update_step_end(self, gathered_losses, progress_bar):
        log_trigger = max(
            1, self.config.logging_steps // self.accelerator.num_processes
        )

        validate_trigger = (
            max(
                1,
                self.config.validation_steps
                // self.accelerator.num_processes,
            )
            if hasattr(self.config, "validation_steps")
            and self.config.validation_steps > 0
            else None
        )

        inference_trigger = (
            max(
                1,
                self.config.inference_steps
                // self.accelerator.num_processes,
            )
            if hasattr(self.config, "inference_steps")
            and self.config.inference_steps > 0
            else None
        )

        save_trigger = (
            max(1, self.config.save_steps // self.accelerator.num_processes)
            if hasattr(self.config, "save_steps")
            and self.config.save_steps > 0
            else None
        )

        val_metric = None
        if validate_trigger and self.global_step % validate_trigger == 0:
            val_metric = self.validate(self.global_step)

        if self.config.save_best and val_metric is not None:
            self.auto_save_best(
                val_metric,
                higher_is_better=self.config.best_metric_higher_is_better,
            )

        if save_trigger and self.global_step % save_trigger == 0:
            self.maybe_save_periodically()

        if self.accelerator.is_main_process:
            metrics_to_log = {}
            if self.config.training_mode == 'lora':
                lora_params = [
                    p for n, p in self.model.named_parameters() if "lora_" in n
                ]

                lora_norm = torch.norm(
                    torch.stack([torch.norm(p.detach()) for p in lora_params])
                ).item()

                lora_grads = [
                    torch.norm(p.grad.detach())
                    for p in lora_params
                    if p.grad is not None
                ]

                if len(lora_grads) > 0:
                    lora_grad_norm = torch.norm(torch.stack(lora_grads)).item()
                else:
                    lora_grad_norm = 0.0

                if self.prev_lora_state is not None:
                    deltas = [
                        torch.norm(p.detach().cpu() - self.prev_lora_state[i])
                        for i, p in enumerate(lora_params)
                    ]
                    lora_delta = torch.mean(torch.stack(deltas)).item()
                else:
                    lora_delta = 0.0
                self.prev_lora_state = [p.detach().cpu().clone() for p in lora_params]
                metrics_to_log.update({
                    "lora/param_norm": lora_norm, 
                    "lora/grad_norm": lora_grad_norm, 
                    "lora/param_delta": lora_delta
                })
            elif self.config.training_mode == "finetune":
                all_trainable_params = [
                    p for p in self.model.parameters() if p.requires_grad
                ]
                param_norm = torch.norm(
                    torch.stack([torch.norm(p.detach()) for p in all_trainable_params])
                ).item()
                param_grads = [
                    torch.norm(p.grad.detach())
                    for p in all_trainable_params
                    if p.grad is not None
                ]
                grad_norm = (
                    torch.norm(torch.stack(param_grads)).item() if param_grads else 0.0
                )
                if self.prev_finetune_state is not None:
                    deltas = [
                        torch.norm(p.detach().cpu() - self.prev_finetune_state[i])
                        for i, p in enumerate(all_trainable_params)
                    ]
                    param_delta = torch.mean(torch.stack(deltas)).item()
                else:
                    param_delta = 0.0

                self.prev_finetune_state = [
                    p.detach().cpu().clone() for p in all_trainable_params
                ]
                metrics_to_log.update({
                    "finetune/param_norm": param_norm, 
                    "finetune/grad_norm": grad_norm, 
                    "finetune/param_delta": param_delta
                })

            metrics_to_log.update({
                "train/total_loss": gathered_losses["total"].item(),
                "train/main_loss": gathered_losses["main"].item(),
                "train/hook_loss": gathered_losses["hook"].item(),
                "train/distill_loss": gathered_losses["distill"].item(),
                "train/lr": self.lr_scheduler.get_last_lr()[0],
            })
            if self.hook_type == "shiva":
                shiva_metrics = {
                    "shiva/ratio_loss": self.last_ratio_loss.item(),
                    "shiva/gate_loss": self.last_gate_loss.item(),
                    "shiva/consistency_loss": self.last_consistency_loss.item(),
                }

                avg_fuse_prob, per_layer_probs = ShivaDitBlockHook.calculate_shiva_gate_prob(self.device)
                shiva_metrics["shiva/avg_fuse_prob"] = avg_fuse_prob.item()
                if per_layer_probs:
                     self.logger.add_scalars(
                         "shiva/per_layer_fuse_prob", per_layer_probs, self.global_step
                     )

                metrics_to_log.update(shiva_metrics)
            self.logger.log_scalars(metrics_to_log, step=self.global_step)

            if self.global_step % log_trigger == 0:
                postfix_dict = {
                    "loss": f"{gathered_losses['total'].item():.4f}",
                    "main_loss": f"{gathered_losses['main'].item():.4f}",
                    "hook_loss": f"{gathered_losses['hook'].item():.4f}",
                    "distill_loss": f"{gathered_losses['distill'].item():.4f}",
                    "lr": f"{self.lr_scheduler.get_last_lr()[0]:.2e}",
                    "step": self.global_step,
                }

                if self.hook_type == "shiva":
                    postfix_dict["ratio_loss"] = f"{self.last_ratio_loss.item():.4f}"
                    postfix_dict["gate_loss"] = f"{self.last_gate_loss.item():.4f}"
                    postfix_dict["consistency_loss"] = f"{self.last_consistency_loss.item():.4f}"
                    postfix_dict["fuse_prob"] = f"{avg_fuse_prob.item():.3f}"

                if self.config.training_mode == "lora":
                    postfix_dict["lora_norm"] = f"{lora_norm:.4f}"
                    postfix_dict["lora_grad_norm"] = f"{lora_grad_norm:.4e}"
                    postfix_dict["lora_delta"] = f"{lora_delta:.4e}"
                elif self.config.training_mode == "finetune":
                    postfix_dict["param_norm"] = f"{param_norm:.4f}"
                    postfix_dict["grad_norm"] = f"{grad_norm:.4e}"
                    postfix_dict["param_delta"] = f"{param_delta:.4e}"

                progress_bar.set_postfix(postfix_dict)

            if inference_trigger and self.global_step % inference_trigger == 0:
                self.run_inference_validation(self.global_step)

    def _log_trainable_parameters(self, description: str = ""):
        if not self.accelerator.is_main_process:
            return

        trainable_params = 0
        hook_params = 0
        lora_params = 0
        all_params = 0
        trainable_param_names = []

        for name, param in self.model.named_parameters():
            all_params += param.numel()
            if param.requires_grad:
                trainable_params += param.numel()
                trainable_param_names.append(name)
                if "_hook" in name:
                    hook_params += param.numel()
                if "lora_" in name:
                    lora_params += param.numel()

        print(f"\n--- Trainable Parameters ({description}) ---")
        print(f"  Currently trainable parameters: {trainable_params:,}")
        print(f"  Total parameters:   {all_params:,}")
        print(f"  Hook parameters:   {hook_params:,}")
        print(f"  LoRA parameters:   {lora_params:,}")
        print(f"  Trainable ratio:   {100 * trainable_params / all_params:.4f}%")

        if len(trainable_param_names) > 0 and len(trainable_param_names) < 15:
            print(f"  Trainable parameters: {trainable_param_names}")
        elif len(trainable_param_names) > 0:
            print(f"  Trainable parameters: ['{trainable_param_names[0]}', ..., '{trainable_param_names[-1]}']")
        else:
            print(f"  Trainable parameters: None")
        print(f"--------------------------------------------------\n")

    @torch.no_grad()
    def validate(self, global_step):
        if self.val_dataloader is None or not self.config.run_validation:
            return

        if self.accelerator.is_main_process:
            print(f"\nRunning validation at step {global_step}...")

        self.model.eval()

        all_total_losses = []
        all_main_losses = []
        all_hook_losses = []
        all_distill_losses = []

        val_progress_bar = tqdm(
            self.val_dataloader,
            desc=f"Validation Step {global_step}",
            leave=False,
            total=len(self.val_dataloader),
            disable=not self.accelerator.is_local_main_process,
        )

        for batch in val_progress_bar:
            try:
                total_loss, main_loss, hook_loss, distill_loss = self._compute_loss(
                    batch, global_step
                )

                all_total_losses.append(total_loss)
                all_main_losses.append(main_loss)
                all_hook_losses.append(hook_loss)
                all_distill_losses.append(distill_loss)
            except Exception as e:
                if self.accelerator.is_main_process:
                    print(f"Error during validation batch: {e}")
                continue

        if len(self.val_dataloader) > 0:
            avg_val_loss = (
                self.accelerator.gather(torch.stack(all_total_losses)).mean().item()
            )
            avg_main_loss = (
                self.accelerator.gather(torch.stack(all_main_losses)).mean().item()
            )
            avg_hook_loss = (
                self.accelerator.gather(torch.stack(all_hook_losses)).mean().item()
            )
            avg_distill_loss = (
                self.accelerator.gather(torch.stack(all_distill_losses)).mean().item()
            )

            if self.accelerator.is_main_process:
                print(f"--- Validation Results (Step {global_step}) ---")
                print(f"  Avg Total Loss: {avg_val_loss:.4f}")
                print(f"  Avg Main Loss:  {avg_main_loss:.4f}")
                print(f"  Avg Hook Loss:  {avg_hook_loss:.4f}")
                print(f"  Avg Distill Loss: {avg_distill_loss:.4f}")
                print(f"--------------------------------------")

                block_ratios_for_tb = {}
                hook_ratios = []
                for name, hook in HookManager.get_all_hooks().items():
                    ratio_val = None
                    if hook.__class__.__name__ == "DiffCRDitBlockHook":
                        ratio_val = hook.diff_ratio.item()
                    elif hook.__class__.__name__ == "ShivaDitBlockHook":
                        if hasattr(hook, "shiva_ratio"):
                            ratio_val = hook.shiva_ratio.item()
                    
                    if ratio_val is not None:
                        hook_ratios.append(ratio_val)
                        print(f"    {name}: {ratio_val:.4f}")
                        clean_name = name.split(".")[-1]
                        block_ratios_for_tb[clean_name] = ratio_val

                print(f"--------------------------------------")

                self.logger.log_scalars(
                    {
                        "validation/total_loss": avg_val_loss,
                        "validation/main_loss": avg_main_loss,
                        "validation/hook_loss": avg_hook_loss,
                        "validation/distill_loss": avg_distill_loss,
                    },
                    step=global_step
                )
                if block_ratios_for_tb:
                    self.logger.add_scalars(
                        "validation/block_ratios", block_ratios_for_tb, global_step
                    )
                    avg_ratio = sum(hook_ratios) / len(hook_ratios)
                    self.logger.log_scalars({"validation/avg_ratio": avg_ratio}, step=global_step)
        elif self.accelerator.is_main_process:
            print(
                f"Validation dataloader is empty or validation failed, skipping logging."
            )

        self.model.train()

        return avg_val_loss if "avg_val_loss" in locals() else None

    @torch.no_grad()
    def run_inference_validation(self, global_step):
        if (
            not self.accelerator.is_main_process
            or not self.config.run_inference_validation
            or not self.config.inference_metadata_file
        ):
            return

        print(
            f"\nRunning inference validation from file: {self.config.inference_metadata_file}..."
        )

        self.model.eval()
        prompts = []
        try:
            with open(self.config.inference_metadata_file, "r", encoding="utf-8") as f:
                for line in f:
                    item = json.loads(line)
                    if "text" in item:
                        prompts.append(item["text"])
            if not prompts:
                print(
                    "  Warning: Inference metadata file is empty or contains no captions. Skipping."
                )
                return
        except Exception as e:
            print(
                f"  Failed to read or parse inference metadata file: {e}. Skipping inference validation."
            )
            return

        unwrapped_model = self.accelerator.unwrap_model(self.model)

        assert len(self.model_attrs) == 1, "Currently only single model_attr supported"
        model_attr_name = self.model_attrs[0]
        original_model_in_pipe = getattr(self.pipe, model_attr_name)
        setattr(self.pipe, model_attr_name, unwrapped_model)
        print(
            f"  Temporarily patching pipe.{model_attr_name} with unwrapped trained model"
        )

        inference_generator = torch.Generator(device=self.device).manual_seed(
            self.config.inference_seed
        )

        original_scheduler = copy.deepcopy(self.scheduler)

        sample_dir = os.path.join(
            self.config.output_dir,
            self.config.inference_output_dir,
            f"step_{global_step}",
        )
        os.makedirs(sample_dir, exist_ok=True)

        for i, prompt in enumerate(prompts):
            try:
                if self.device.type == 'cuda':
                    torch.cuda.empty_cache()
                with self.accelerator.autocast():
                    image = self.pipe(
                        prompt=prompt,
                        generator=inference_generator,
                        num_inference_steps=self.config.inference_num_inference_steps,
                    ).images[0]

                safe_prompt = "".join(c if c.isalnum() else "_" for c in prompt)[:50]
                image.save(
                    os.path.join(sample_dir, f"sample_{i+1:03d}_{safe_prompt}.png")
                )
                tag = f"inference_sample/{i+1:03d}_{safe_prompt}"
                self.logger.log_image(tag, image, global_step)
            except Exception as e:
                print(f"  Failed to generate image for prompt: '{prompt}'. Error: {e}")

        setattr(self.pipe, model_attr_name, original_model_in_pipe)
        print(f"  Restored original model to pipe.{model_attr_name}")

        self.scheduler = original_scheduler

        self.model.train()

    def _try_resume(self):
        self.global_step = 0
        self.epoch = 0
        self.best_val_metric = None

        ckpt_dir = self.config.output_dir
        latest_ckpt_dir = None
        resumed = False

        if self.accelerator.is_main_process:
            if os.path.isdir(ckpt_dir):
                ckpt_dirs = [d for d in glob.glob(os.path.join(ckpt_dir, "checkpoint_step_*")) if os.path.isdir(d)]
                if ckpt_dirs:
                    def extract_step_dir(path):
                        match_step = re.search(r"step_(\d+)$", os.path.basename(path))
                        if match_step:
                            if os.path.exists(os.path.join(path, "trainer_state.json")):
                                return int(match_step.group(1))
                        return -1

                    valid_ckpt_dirs = [d for d in ckpt_dirs if extract_step_dir(d) != -1]
                    if valid_ckpt_dirs:
                        latest_ckpt_dir = max(valid_ckpt_dirs, key=extract_step_dir)
                        print(f"Found latest valid accelerator checkpoint: {latest_ckpt_dir}")
                    else:
                        print("No valid accelerator checkpoints found (missing trainer_state.json). Starting fresh.")
                else:
                    print("No accelerator checkpoint directories found. Starting fresh.")
            else:
                print(f"No checkpoint directory found at {ckpt_dir}. Starting fresh.")

        path_broadcast_list = [latest_ckpt_dir]
        broadcast_object_list(path_broadcast_list, from_process=0)
        latest_ckpt_dir = path_broadcast_list[0]
        self.accelerator.wait_for_everyone()

        if latest_ckpt_dir is not None:
            try:
                if self.accelerator.is_main_process:
                    print(f"[{self.accelerator.process_index}] Attempting to resume from {latest_ckpt_dir}...")

                self.accelerator.load_state(latest_ckpt_dir)
                custom_state_path = os.path.join(latest_ckpt_dir, "trainer_state.json")
                with open(custom_state_path, "r") as f:
                    custom_state = json.load(f)

                self.global_step = custom_state.get("global_step", 0)
                self.epoch = custom_state.get("epoch", 0)
                self.best_val_metric = custom_state.get("best_val_metric", None)

                resumed = True
                if self.accelerator.is_main_process:
                    print(f"Resumed successfully from step={self.global_step}, epoch={self.epoch}")

            except Exception as e:
                resumed = False
                if self.accelerator.is_main_process:
                    print(f"Failed to load checkpoint {latest_ckpt_dir}. Error: {e}. Starting fresh.")

        self.accelerator.wait_for_everyone()
        if not resumed and self.accelerator.is_main_process:
            print("No checkpoint found or resume failed. Starting fresh.")

    def _initial_phase(self):
        ## resume hook-only state
        if self.config.hook_only_steps > 0:
            scaled_hook_only_steps = max(
                1, self.config.hook_only_steps // self.accelerator.num_processes
            )

            is_in_hookonly = self.global_step < scaled_hook_only_steps
            self._set_finetuning_trainable_state(requires_grad=not is_in_hookonly)
        else:
            self._set_finetuning_trainable_state(requires_grad=True)

        ## print parameter count
        self._log_trainable_parameters('Initial')

    def _trigger_phase(self):
        # trigger hook-only completion
        if self.config.hook_only_steps > 0:
            scaled_hook_only_steps = max(1, self.config.hook_only_steps // self.accelerator.num_processes)
            if self.global_step == scaled_hook_only_steps:
                self._set_finetuning_trainable_state(requires_grad=True)
                self._log_trainable_parameters('After hook-only')

    def train(self):
        if self.accelerator.is_main_process:
            print("Starting training...")
        global_context.update(encoder_flag=True)

        self._try_resume()
        self._initial_phase()

        self.prev_lora_state = None
        self.prev_finetune_state = None

        unwrapped_model = self.accelerator.unwrap_model(self.model)
        if hasattr(unwrapped_model, "enable_adapter_layers"):
            unwrapped_model.enable_adapter_layers()

        for epoch in range(self.epoch, self.config.num_epochs):
            self.epoch = epoch
            self.model.train()
            progress_bar = tqdm(
                self.train_dataloader,
                desc=f"Epoch {epoch+1}/{self.config.num_epochs}",
                total=len(self.train_dataloader),
                dynamic_ncols=True,
                disable=not self.accelerator.is_local_main_process,
            )

            for step, batch in enumerate(progress_bar):
                if step % self.config.gradient_accumulation_steps == 0:
                    self.optimizer.zero_grad(set_to_none=True)

                total_loss, main_loss, hook_loss, distill_loss = self._train_step(batch)
                is_update_step = (
                    step + 1
                ) % self.config.gradient_accumulation_steps == 0
                is_last_step = (step + 1) == len(self.train_dataloader)

                if is_update_step or is_last_step:
                    if self.accelerator.sync_gradients:
                        self.accelerator.clip_grad_norm_(self.model.parameters(), 1.0)

                    self.optimizer.step()
                    self.lr_scheduler.step()
                    self.global_step += 1

                    self._trigger_phase()

                    gathered_losses = {
                        "total": self.accelerator.gather(total_loss).mean(),
                        "main": self.accelerator.gather(main_loss).mean(),
                        "hook": self.accelerator.gather(hook_loss).mean(),
                        "distill": self.accelerator.gather(distill_loss).mean()
                    }
                    self._on_update_step_end(gathered_losses, progress_bar)

            if self.accelerator.is_main_process:
                print(f"Epoch {epoch+1} completed.")

        if self.accelerator.is_main_process:
            print("Training completed. Saving final model...")
            final_path = os.path.join(self.config.output_dir, "final.pt")
            self.save_checkpoint(final_path)
            print(f"Final model saved to {final_path}")

    def save_checkpoint(self, save_path: str):
        if self.accelerator.is_main_process:
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            model_state = {}
            unwrapped_model = self.accelerator.unwrap_model(self.model)
            state_dict = unwrapped_model.state_dict()
            mode = self.config.training_mode
            if mode == "lora":
                for k, v in state_dict.items():
                    if "lora" in k or "_hook" in k:
                        model_state[k] = v

            elif mode == "hook_only":
                model_state = {k: v for k, v in state_dict.items() if "_hook" in k}

            elif mode == "finetune":
                model_state = state_dict

            else:
                raise ValueError(f"Unknown training_mode: {mode}")

            hook_info = None
            if self.config.trainable_hook_config:
                hook_class, hook_kwargs = self.config.trainable_hook_config
                hook_info = {
                    "class_name": hook_class.__name__,
                    "config": hook_kwargs
                }

            lora_info = None
            if self.config.lora_config and self.config.lora_config.enable:
                lora_info = asdict(self.config.lora_config)

            global_step = getattr(self, "global_step", 0)

            checkpoint = {
                "model": model_state,
                "training_mode": mode,
                "model_type": self.model_type,
                "hook_info": hook_info,
                "lora_info": lora_info,
                "global_step": global_step,
            }

            torch.save(checkpoint, save_path)
            print(
                f"Saved inference checkpoint to {save_path} "
                f"(mode={mode}, step={global_step})"
            )

        self.accelerator.wait_for_everyone()

    def auto_save_best(self, val_metric: float, higher_is_better: bool = False):
        if val_metric is None:
            if self.accelerator.is_main_process:
                print(f"Validation metric is None, skipping best model save.")
            return

        is_best = False
        if not hasattr(self, "best_val_metric") or self.best_val_metric is None:
            is_best = True
        else:
            if higher_is_better:
                is_best = val_metric > self.best_val_metric
            else:
                is_best = val_metric < self.best_val_metric

        if is_best:
            self.best_val_metric = val_metric
            save_path = os.path.join(self.config.output_dir, "best.pt")

            self.save_checkpoint(save_path)

            if self.accelerator.is_main_process:
                print(f"New best model saved at {save_path} (metric={val_metric:.4f})")
        else:
            if self.accelerator.is_main_process:
                print(
                    f"Validation metric {val_metric:.4f} not improved (best={self.best_val_metric:.4f})"
                )

    def maybe_save_periodically(self):
        save_path_dir = os.path.join(
            self.config.output_dir, f"checkpoint_step_{self.global_step:06d}"
        )

        self.accelerator.save_state(save_path_dir)
        if self.accelerator.is_main_process:
            custom_state = {
                "global_step": self.global_step,
                "epoch": self.epoch,
                "best_val_metric": self.best_val_metric
            }
            try:
                with open(os.path.join(save_path_dir, "trainer_state.json"), "w") as f:
                    json.dump(custom_state, f)
            except Exception as e:
                print(f"Warning: Failed to save trainer_state.json: {e}")

        self.accelerator.wait_for_everyone()
        if self.accelerator.is_main_process:
            print(f"Saved accelerator state to {save_path_dir} (step {self.global_step})")
