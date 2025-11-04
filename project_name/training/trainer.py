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
from accelerate.utils import set_seed

from baselines.diffcr import DiffCRDitBlockHook, calculate_ratio_loss
from baselines.dy_dit import DynamicDiTBlockHook, calculate_flops_loss
from diffusers.pipelines.flux.pipeline_flux_kontext import retrieve_latents
from diffusers.schedulers import FlowMatchEulerDiscreteScheduler
from project_name.hook import HookManager
from project_name.utils import get_pipeline
from project_name.training.config import TrainerConfig
from project_name.training.dataset import ImageTextDataset, collate_fn
from project_name.utils import global_context


class DiffusionModelTrainer:
    def __init__(self, config: TrainerConfig):
        self.config = config
        self.model_type = config.model_type

        self.accelerator = Accelerator(
            mixed_precision=config.mixed_precision,
            gradient_accumulation_steps=config.gradient_accumulation_steps,
            log_with="tensorboard" if config.log_with_tensorboard else None,
            project_dir=os.path.join(config.output_dir, config.tensorboard_log_dir),
        )
        self.device = self.accelerator.device
        set_seed(config.seed)
        self.generator = torch.Generator(device=self.accelerator.device).manual_seed(
            config.seed
        )

        if self.config.mixed_precision == "bf16":
            dtype = torch.bfloat16
        elif self.config.mixed_precision == "fp16":
            dtype = torch.float16
        else:
            dtype = torch.float32

        self.pipe, self.model_attrs = get_pipeline(
            self.model_type,
            model_path=config.model_path,
            dtype=dtype,
        )

        # we only to get the ref of this modules, no need to move them to device again
        self.image_processor = self.pipe.image_processor
        self.vae = self.pipe.vae
        self.scheduler = self.pipe.scheduler
        self.model = tuple(
            getattr(self.pipe, model) for model in self.model_attrs
        )  # DiT
        # TODO: support dual DiT like Wan
        self.model = self.model[0]

        # distillation setup
        self.teacher_model = None
        if self.config.distillation_final_weight > 0 or self.config.distillation_middle_weight > 0:
            if self.accelerator.is_main_process:
                print("Distillation loss enabled. Loading static teacher model...")

            teacher_pipe, teacher_attrs = get_pipeline(
                self.model_type,
                model_path=config.model_path,
                dtype=dtype,
            )
            self.teacher_model = getattr(teacher_pipe, teacher_attrs[0]) # TODO: support dual DiT like Wan
            self.teacher_model.requires_grad_(False)
            self.teacher_model.eval()
            self.teacher_model = self.teacher_model.to(self.device)
            del teacher_pipe
            if self.accelerator.is_main_process:
                print("Teacher model loaded.")

        if self.model_type == "flux-dev" or self.model_type == "flux-kontext":
            self.tokenizers = (self.pipe.tokenizer, self.pipe.tokenizer_2)
            self.text_encoders = (self.pipe.text_encoder, self.pipe.text_encoder_2)
            self.text_encoder_names = ("text_encoder", "text_encoder_2")
        elif (
            self.model_type == "pixart-sigma"
            or self.model_type == "qwen-image"
            or self.model_type == "qwen-image-edit"
            or self.model_type == "wan"
        ):
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

        self.vae.requires_grad_(False)
        self.vae.eval()
        for text_encoder in self.text_encoders:
            text_encoder.requires_grad_(False)
            text_encoder.eval()

        if self.accelerator.is_main_process:
            print("model_type:", config.model_type)
            print("image_processor:", self.image_processor.__class__.__name__)
            print("vae:", self.vae.__class__.__name__)
            print("scheduler:", self.scheduler.__class__.__name__)
            print("model (DiT):", self.model.__class__.__name__)

        # inject
        if config.trainable_hook_config:
            hook_class, hook_kwargs = config.trainable_hook_config

            if hook_class == DynamicDiTBlockHook:
                self.hook_type = 'dydit'
            elif hook_class == DiffCRDitBlockHook:
                self.hook_type = 'diffcr'

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

        self.optimizer = self._setup_model_and_optimizer()

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

        num_update_steps_per_epoch = max(
            1, len(self.train_dataloader) // config.gradient_accumulation_steps
        )
        self.num_training_steps = num_update_steps_per_epoch * config.num_epochs
        self.lr_scheduler = self._setup_scheduler()

        self.val_dataloader = None
        if config.val_metadata_file and config.val_image_dir:
            print("Setting up validation dataloader...")
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
                print(f"  Validation dataset size: {len(val_dataset)}")
            except Exception as e:
                print(
                    f"Warning: Failed to load validation dataset. Validation will be skipped. Error: {e}"
                )
                self.val_dataloader = None
        else:
            print(
                "Validation data not provided (val_metadata_file or val_image_dir is None), skipping validation setup."
            )

        # ===== prepare accelerator =====
        prepare_list = [
            self.model,
            self.optimizer,
            self.lr_scheduler,
            self.train_dataloader,
            self.vae,
            *self.text_encoders,
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
        self.vae = prepared[idx]; idx += 1
        self.text_encoders = tuple(prepared[idx:])
        # ==========================================================

        if self.val_dataloader is not None:
            self.val_dataloader = self.accelerator.prepare(self.val_dataloader)

        unwrapped_model = self.accelerator.unwrap_model(self.model)
        self.dtype = unwrapped_model.dtype

        if self.config.caption_dropout_rate > 0:
            print(
                f"Caption dropout enabled with rate: {self.config.caption_dropout_rate}"
            )
            if self.model_type == "pixart-sigma":
                self.empty_prompt_embeds, self.empty_prompt_attention_mask, _, _ = (
                    self.pipe.encode_prompt(
                        "",
                        do_classifier_free_guidance=False,
                        device=self.device,
                        dtype=self.dtype,
                        clean_caption=True,
                    )
                )
            elif self.model_type == "flux-dev" or self.model_type == "flux-kontext":
                (
                    self.empty_prompt_embeds,
                    self.empty_pooled_prompt_embeds,
                    self.empty_text_ids,
                ) = self.pipe.encode_prompt(
                    prompt="",
                    device=self.device,
                    num_images_per_prompt=1,
                )
            else:
                raise NotImplementedError(
                    "Caption dropout is only implemented for 'pixart-sigma' model_type currently."
                )

        self.writer = None
        self.pil_to_tensor_transform = None
        if self.accelerator.is_main_process and config.log_with_tensorboard:
            from torch.utils.tensorboard import SummaryWriter
            import torchvision.transforms as T

            tb_log_path = os.path.join(config.output_dir, config.tensorboard_log_dir)
            os.makedirs(tb_log_path, exist_ok=True)
            self.writer = SummaryWriter(log_dir=tb_log_path)
            self.pil_to_tensor_transform = T.ToTensor()
            print(f"TensorBoard logging enabled. Logs will be saved to: {tb_log_path}")

        if self.accelerator.is_main_process:
            print(f"--- Training Config ---")
            print(f"  Mode: {config.training_mode}")
            print(f"  Micro Batch Size: {config.train_batch_size}")
            print(f"  Grad Acc Step: {config.gradient_accumulation_steps}")
            print(
                f"  Actual Batch Size: {config.train_batch_size * config.gradient_accumulation_steps}"
            )
            print(f"  Total training steps: {self.num_training_steps}")
            print(f"------------------")

    def _setup_model_and_optimizer(self):
        # freeze VAE and Text Encoder(s)
        self.vae.requires_grad_(False)
        for text_encoder in self.text_encoders:
            text_encoder.requires_grad_(False)

        params_to_train = []

        if self.config.training_mode == "hook_only":
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
            print("Training mode: finetune (training all DiT parameters)")
            self.model.requires_grad_(True)
            params_to_train = list(self.model.parameters())
            for hook in HookManager.get_all_hooks().values():
                if isinstance(hook, nn.Module):
                    hook.requires_grad_(True)
                    params_to_train.extend(hook.parameters())
        elif (
            self.config.training_mode == "lora"
            or self.config.training_mode == "auto-lora"
        ):
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

        print(f"Total trainable parameters: {sum(p.numel() for p in params_to_train)}")

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

    def _prepare_model_inputs(self, batch):
        batch_size = batch["pixel_values"].shape[0]
        model_kwargs = {}

        # vae encode
        pixel_values = batch["pixel_values"].to(dtype=self.vae.dtype)
        if self.model_type == "flux-dev" or self.model_type == "flux-kontext":
            img_latents = retrieve_latents(
                self.vae.encode(pixel_values),
                generator=self.generator,
                sample_mode="argmax",
            )
            img_latents = (
                img_latents - self.vae.config.shift_factor
            ) * self.vae.config.scaling_factor

            height, width = img_latents.shape[2], img_latents.shape[3]

            img_ids = self.pipe._prepare_latent_image_ids(
                batch_size, height // 2, width // 2, self.device, self.dtype
            )
            model_kwargs["img_ids"] = img_ids
        elif self.model_type == "sd3":
            img_latents = retrieve_latents(
                self.vae.encode(pixel_values),
                generator=self.generator,
                sample_mode="argmax",
            )
            img_latents = (
                img_latents - self.vae.config.shift_factor
            ) * self.vae.config.scaling_factor

        elif self.model_type == "pixart-sigma":
            img_latents = retrieve_latents(
                self.vae.encode(pixel_values),
                generator=self.generator,
                sample_mode="argmax",
            )
            img_latents = img_latents * self.vae.config.scaling_factor
        elif self.model_type == "qwen-image":
            img_latents = retrieve_latents(
                self.vae.encode(pixel_values),
                generator=self.generator,
                sample_mode="argmax",
            )
            latents_mean = (
                torch.tensor(self.vae.config.latents_mean)
                .view(1, self.pipe.latent_channels, 1, 1, 1)
                .to(img_latents.device, img_latents.dtype)
            )
            latents_std = (
                torch.tensor(self.vae.config.latents_std)
                .view(1, self.pipe.latent_channels, 1, 1, 1)
                .to(img_latents.device, img_latents.dtype)
            )
            img_latents = (img_latents - latents_mean) / latents_std
            height = self.pipe.default_sample_size * self.pipe.vae_scale_factor
            width = self.pipe.default_sample_size * self.pipe.vae_scale_factor

            img_shapes = [
                [
                    (
                        1,
                        height // self.pipe.vae_scale_factor // 2,
                        width // self.pipe.vae_scale_factor // 2,
                    )
                ]
            ] * batch_size

            model_kwargs["img_shapes"] = img_shapes
        else:
            raise NotImplementedError(
                f"_prepare_model_inputs not implemented for model_type: {self.model_type}"
            )

        # text encode
        input_ids_tuple = batch["input_ids"]
        if self.model_type == "flux-dev" or self.model_type == "flux-kontext":
            model_kwargs["return_dict"] = False
            model_kwargs["joint_attention_kwargs"] = {}

            num_images_per_prompt = 1
            guidance_scale = 3.5
            # clip -> pooled_prompt_embeds
            input_ids_clip = input_ids_tuple[0]
            pooled_prompt_embeds = self.text_encoders[0](
                input_ids_clip, output_hidden_states=False
            )
            pooled_prompt_embeds = pooled_prompt_embeds.pooler_output
            pooled_prompt_embeds = pooled_prompt_embeds.to(
                dtype=self.dtype, device=self.device
            )
            pooled_prompt_embeds = pooled_prompt_embeds.repeat(1, num_images_per_prompt)
            pooled_prompt_embeds = pooled_prompt_embeds.view(
                batch_size * num_images_per_prompt, -1
            )

            # t5 -> prompt_embeds
            input_ids_t5 = input_ids_tuple[1]
            prompt_embeds = self.text_encoders[1](
                input_ids_t5, output_hidden_states=False
            )[0]
            prompt_embeds = prompt_embeds.to(dtype=self.dtype, device=self.device)
            _, seq_len, _ = prompt_embeds.shape
            prompt_embeds = prompt_embeds.repeat(1, num_images_per_prompt, 1)
            prompt_embeds = prompt_embeds.view(
                batch_size * num_images_per_prompt, seq_len, -1
            )

            text_ids = torch.zeros(prompt_embeds.shape[1], 3).to(
                dtype=self.dtype, device=self.device
            )

            model_kwargs["encoder_hidden_states"] = prompt_embeds
            model_kwargs["pooled_projections"] = pooled_prompt_embeds
            model_kwargs["txt_ids"] = text_ids

            if torch.rand(1).item() < self.config.caption_dropout_rate:
                prompt_embeds = self.empty_prompt_embeds
                pooled_prompt_embeds = self.empty_pooled_prompt_embeds
                text_ids = self.empty_text_ids

                model_kwargs["encoder_hidden_states"] = prompt_embeds
                model_kwargs["pooled_projections"] = pooled_prompt_embeds
                model_kwargs["txt_ids"] = text_ids

            # add noise
            noise = torch.randn_like(img_latents)
            indices = torch.randint(
                0,
                self.scheduler.config.num_train_timesteps,
                (batch_size,),
                device=self.device,
            )
            schedule_timesteps_gpu = self.scheduler.timesteps.to(self.device)
            timesteps = schedule_timesteps_gpu[indices]
            noisy_latents = self.scheduler.scale_noise(
                sample=img_latents,
                timestep=timesteps,
                noise=noise,
            )

            guidance = torch.full(
                [1], guidance_scale, device=self.device, dtype=torch.float32
            )
            guidance = guidance.expand(batch_size)
            model_kwargs["guidance"] = guidance
        elif self.model_type == "pixart-sigma":
            model_kwargs["return_dict"] = False
            added_cond_kwargs = {"resolution": None, "aspect_ratio": None}
            model_kwargs["added_cond_kwargs"] = added_cond_kwargs
            num_images_per_prompt = 1

            input_ids = input_ids_tuple[0]
            prompt_attention_mask = batch["attention_mask"][0]
            prompt_embeds = self.text_encoders[0](
                input_ids, attention_mask=prompt_attention_mask
            )[0].to(dtype=self.dtype, device=self.device)

            bs_embed, seq_len, _ = prompt_embeds.shape
            prompt_embeds = prompt_embeds.repeat(1, num_images_per_prompt, 1)
            prompt_embeds = prompt_embeds.view(
                bs_embed * num_images_per_prompt, seq_len, -1
            )
            prompt_attention_mask = prompt_attention_mask.repeat(
                1, num_images_per_prompt
            )
            prompt_attention_mask = prompt_attention_mask.view(
                bs_embed * num_images_per_prompt, -1
            )

            if torch.rand(1).item() < self.config.caption_dropout_rate:
                prompt_embeds = self.empty_prompt_embeds
                prompt_attention_mask = self.empty_prompt_attention_mask

            model_kwargs["encoder_hidden_states"] = prompt_embeds
            model_kwargs["encoder_attention_mask"] = prompt_attention_mask

            # add noise
            noise = torch.randn_like(img_latents)
            indices = torch.randint(
                0,
                self.scheduler.config.num_train_timesteps,
                (batch_size,),
                device=self.device,
            )
            schedule_timesteps_gpu = self.scheduler.timesteps.to(self.device)
            timesteps = schedule_timesteps_gpu[indices]
            noisy_latents = self.scheduler.add_noise(
                original_samples=img_latents,
                timesteps=timesteps,
                noise=noise,
            )

        elif self.model_type == "sd3":
            model_kwargs["return_dict"] = False
            model_kwargs["joint_attention_kwargs"] = {}
            num_images_per_prompt = 1

            def _clip_encode(text_encoder, text_input_ids, clip_skip=None):
                prompt_embeds = text_encoder(text_input_ids, output_hidden_states=True)
                pooled_prompt_embeds = prompt_embeds[0]

                if clip_skip is None:
                    prompt_embeds = prompt_embeds.hidden_states[-2]
                else:
                    prompt_embeds = prompt_embeds.hidden_states[-(clip_skip + 2)]

                prompt_embeds = prompt_embeds.to(dtype=self.dtype, device=self.device)

                _, seq_len, _ = prompt_embeds.shape
                # duplicate text embeddings for each generation per prompt, using mps friendly method
                prompt_embeds = prompt_embeds.repeat(1, num_images_per_prompt, 1)
                prompt_embeds = prompt_embeds.view(
                    batch_size * num_images_per_prompt, seq_len, -1
                )

                pooled_prompt_embeds = pooled_prompt_embeds.repeat(
                    1, num_images_per_prompt, 1
                )
                pooled_prompt_embeds = pooled_prompt_embeds.view(
                    batch_size * num_images_per_prompt, -1
                )

                return prompt_embeds, pooled_prompt_embeds

            def _t5_encode(text_encoder, text_input_ids):
                prompt_embeds = text_encoder(text_input_ids)[0]
                prompt_embeds = prompt_embeds.to(dtype=self.dtype, device=self.device)

                _, seq_len, _ = prompt_embeds.shape

                # duplicate text embeddings and attention mask for each generation per prompt, using mps friendly method
                prompt_embeds = prompt_embeds.repeat(1, num_images_per_prompt, 1)
                prompt_embeds = prompt_embeds.view(
                    batch_size * num_images_per_prompt, seq_len, -1
                )

                return prompt_embeds

            prompt_embed, pooled_prompt_embed = _clip_encode(
                self.text_encoders[0], input_ids_tuple[0]
            )
            prompt_embed2, pooled_prompt_embed2 = _clip_encode(
                self.text_encoders[1],
                input_ids_tuple[1],
                clip_skip=self.config.clip_skip,
            )
            clip_prompt_embeds = torch.cat([prompt_embed, prompt_embed2], dim=-1)

            t5_prompt_embed = _t5_encode(self.text_encoders[2], input_ids_tuple[2])

            clip_prompt_embeds = torch.nn.functional.pad(
                clip_prompt_embeds,
                (0, t5_prompt_embed.shape[-1] - clip_prompt_embeds.shape[-1]),
            )

            prompt_embeds = torch.cat([clip_prompt_embeds, t5_prompt_embed], dim=-2)
            pooled_prompt_embeds = torch.cat(
                [pooled_prompt_embed, pooled_prompt_embed2], dim=-1
            )

            model_kwargs["encoder_hidden_states"] = prompt_embeds
            model_kwargs["pooled_projections"] = pooled_prompt_embeds

        elif self.model_type == "qwen-image":
            model_kwargs["return_dict"] = False
            model_kwargs["attention_kwargs"] = self.pipe.attention_kwargs

            def _extract_masked_hidden(hidden_states: torch.Tensor, mask: torch.Tensor):
                bool_mask = mask.bool()
                valid_lengths = bool_mask.sum(dim=1)
                selected = hidden_states[bool_mask]
                split_result = torch.split(selected, valid_lengths.tolist(), dim=0)

                return split_result

            attention_masks_tuple = batch["attention_mask"]
            drop_idx = self.pipe.prompt_template_encode_start_idx

            encoder_hidden_states = self.text_encoders[0](
                input_ids=input_ids_tuple[0],
                attention_mask=attention_masks_tuple[0],
                output_hidden_states=True,
            )
            hidden_states = encoder_hidden_states.hidden_states[-1]
            split_hidden_states = _extract_masked_hidden(
                hidden_states, attention_masks_tuple[0]
            )
            split_hidden_states = [e[drop_idx:] for e in split_hidden_states]
            attn_mask_list = [
                torch.ones(e.size(0), dtype=torch.long, device=e.device)
                for e in split_hidden_states
            ]
            max_seq_len = max([e.size(0) for e in split_hidden_states])
            prompt_embeds = torch.stack(
                [
                    torch.cat([u, u.new_zeros(max_seq_len - u.size(0), u.size(1))])
                    for u in split_hidden_states
                ]
            )
            prompt_embeds_mask = torch.stack(
                [
                    torch.cat([u, u.new_zeros(max_seq_len - u.size(0))])
                    for u in attn_mask_list
                ]
            )

            prompt_embeds = prompt_embeds.to(dtype=self.dtype, device=self.device)

            model_kwargs["encoder_hidden_states"] = prompt_embeds
            model_kwargs["encoder_hidden_states_mask"] = prompt_embeds_mask

            # add noise
            noise = torch.randn_like(img_latents)
            indices = torch.randint(
                0,
                self.scheduler.config.num_train_timesteps,
                (batch_size,),
                device=self.device,
            )
            schedule_timesteps_gpu = self.scheduler.timesteps.to(self.device)
            timesteps = schedule_timesteps_gpu[indices]
            noisy_latents = self.scheduler.scale_noise(
                sample=img_latents,
                timestep=timesteps,
                noise=noise,
            )
        else:
            raise NotImplementedError(
                f"_prepare_model_inputs not implemented for model_type: {self.model_type}"
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

    def _compute_loss(self, batch, global_step: int):
        img_latents, noise, noisy_latents, timesteps, model_kwargs = (
            self._prepare_model_inputs(batch)
        )

        noisy_latents = (
            noisy_latents.clone().to(self.dtype).detach().requires_grad_(True)
        )

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
            if self.teacher_model is not None and (
                self.config.distillation_final_weight > 0 or self.config.distillation_middle_weight > 0
            ):
                if self.config.distillation_middle_weight > 0:
                    model_kwargs["output_hidden_states"] = True
                    model_kwargs["output_hidden_states_step"] = self.config.distillation_layer_skip
                else:
                    model_kwargs["output_hidden_states"] = False
                with torch.no_grad():
                    teacher_output = self.teacher_model(
                        hidden_states=noisy_latents,
                        timestep=timesteps,
                        **model_kwargs,
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
                    loss_u_total = torch.tensor(0.0, device=self.device, dtype=self.dtype)
                    num_layers_compared = 0

                    layer_indices = range(0, len(X_s_all))

                    for l in layer_indices:
                        loss_u_total += F.mse_loss(X_d_all[l].float(), X_s_all[l].float(), reduction="mean")
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

        if self.hook_type == 'dydit':
            hook_loss = calculate_flops_loss(
                self.config.extra_config, device=self.device
            )
        elif self.hook_type == 'diffcr':
            hook_loss = calculate_ratio_loss(self.config.extra_config, device=self.device)
        else:
            raise NotImplementedError(f"hook_type {self.hook_type} not implemented")
        
        current_hook_loss_weight = self.config.hook_loss_weight
        if self.config.hook_warmup_steps > 0:
            scaled_hook_warmup_steps = max(1, self.config.hook_warmup_steps // self.accelerator.num_processes)
            if global_step < scaled_hook_warmup_steps:
                current_hook_loss_weight = 0.0
        hook_loss = hook_loss * current_hook_loss_weight

        total_loss = main_loss + hook_loss + distill_loss

        return total_loss, main_loss, hook_loss, distill_loss

    @torch.no_grad()
    def validate(self, global_step):
        if self.val_dataloader is None or not self.config.run_validation:
            return

        if self.accelerator.is_main_process:
            print(f"\nRunning validation at step {global_step}...")

        self.model.eval()
        for hook in HookManager.get_all_hooks().values():
            if isinstance(hook, nn.Module):
                hook.eval()

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
                    if isinstance(hook, DiffCRDitBlockHook):
                        ratio_val = hook.diff_ratio.item()
                        hook_ratios.append(ratio_val)
                        print(f"    {name}: {ratio_val:.4f}")
                        clean_name = name.split(".")[-1]
                        block_ratios_for_tb[clean_name] = ratio_val
                print(f"--------------------------------------")

                if self.writer:
                    self.writer.add_scalar(
                        "validation/total_loss", avg_val_loss, global_step
                    )
                    self.writer.add_scalar(
                        "validation/main_loss", avg_main_loss, global_step
                    )
                    self.writer.add_scalar(
                        "validation/hook_loss", avg_hook_loss, global_step
                    )
                    self.writer.add_scalar(
                        "validation/distill_loss", avg_distill_loss, global_step
                    )
                    if block_ratios_for_tb:
                        self.writer.add_scalars(
                            "validation/block_ratios", block_ratios_for_tb, global_step
                        )
                        avg_ratio = sum(hook_ratios) / len(hook_ratios)
                        self.writer.add_scalar(
                            "validation/avg_ratio", avg_ratio, global_step
                        )
        elif self.accelerator.is_main_process:
            print(
                f"Validation dataloader is empty or validation failed, skipping logging."
            )

        self.model.train()
        for hook in HookManager.get_all_hooks().values():
            if isinstance(hook, nn.Module):
                hook.train()

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
        unwrapped_vae = self.accelerator.unwrap_model(self.vae)
        unwrapped_text_encoders = [
            self.accelerator.unwrap_model(te) for te in self.text_encoders
        ]

        # TODO: dual model attr
        model_attr_name = self.model_attrs[0]
        original_model_in_pipe = getattr(self.pipe, model_attr_name)
        original_vae_in_pipe = self.pipe.vae
        original_text_encoders_in_pipe = [
            getattr(self.pipe, te_name) for te_name in self.text_encoder_names
        ]

        setattr(self.pipe, model_attr_name, unwrapped_model)
        self.pipe.vae = unwrapped_vae
        for te_name, te_unwrapped in zip(
            self.text_encoder_names, unwrapped_text_encoders
        ):
            setattr(self.pipe, te_name, te_unwrapped)
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
                image = self.pipe(
                    prompt=prompt,
                    generator=inference_generator,
                    num_inference_steps=self.config.inference_num_inference_steps,
                ).images[0]

                safe_prompt = "".join(c if c.isalnum() else "_" for c in prompt)[:50]
                image.save(
                    os.path.join(sample_dir, f"sample_{i+1:03d}_{safe_prompt}.png")
                )
                if self.writer and self.pil_to_tensor_transform:
                    tag = f"inference_sample/{i+1:03d}_{safe_prompt}"
                    image_tensor = self.pil_to_tensor_transform(image)
                    self.writer.add_image(tag, image_tensor, global_step)
            except Exception as e:
                print(f"  Failed to generate image for prompt: '{prompt}'. Error: {e}")

        setattr(self.pipe, model_attr_name, original_model_in_pipe)
        self.pipe.vae = original_vae_in_pipe
        for te_name, te_original in zip(
            self.text_encoder_names, original_text_encoders_in_pipe
        ):
            setattr(self.pipe, te_name, te_original)

        print(f"  Restored original model to pipe.{model_attr_name}")

        self.scheduler = original_scheduler

        self.model.train()

    def train(self):
        print("Starting training...")
        global_context.update(encoder_flag=True)

        self.global_step = 0
        self.epoch = 0
        self.best_val_metric = None
        resumed = False
        if hasattr(self, "resume_if_available"):
            resumed = self.resume_if_available()
        if resumed:
            state_data = [
                float(self.global_step),
                float(self.epoch),
                float(self.best_val_metric if self.best_val_metric is not None else -1.0)
            ]
            state_tensor = torch.tensor(state_data, dtype=torch.float64, device=self.device)
            torch.distributed.broadcast(state_tensor, src=0)
            state_list = state_tensor.tolist()
            self.global_step = int(state_list[0])
            self.epoch = int(state_list[1])
            self.best_val_metric = float(state_list[2]) if state_list[2] != -1.0 else None
        
        if resumed and self.accelerator.is_main_process:
            print(f"Resumed training from step={self.global_step}, epoch={self.epoch}")

        if self.config.training_mode == "auto-lora" and self.config.hook_only_steps > 0:
            scaled_hook_only_steps = max(1, self.config.hook_only_steps // self.accelerator.num_processes)

            if self.accelerator.is_main_process:
                print("Training in 'auto-lora' mode with hook_only_steps > 0.")
                print(f"  Config hook_only_steps: {self.config.hook_only_steps} (equivalent single-GPU steps)")
                print(f"  Scaled hook_only_steps: {scaled_hook_only_steps} (actual optimizer steps with {self.accelerator.num_processes} GPUs)")

            if self.global_step < scaled_hook_only_steps:
                if self.accelerator.is_main_process:
                    print(f"  Resumed step ({self.global_step}) is less than scaled_hook_only_steps ({scaled_hook_only_steps}). Freezing LoRA parameters.")
                frozen_params = 0
                unwrapped_model = self.accelerator.unwrap_model(self.model)
                for name, param in unwrapped_model.named_parameters():
                    if 'lora_' in name:
                        param.requires_grad_(False)
                        frozen_params += param.numel()
                if self.accelerator.is_main_process:
                    print(f"  Froze {frozen_params} LoRA parameters.")
            else:
                 if self.accelerator.is_main_process:
                    print(f"  Resumed step ({self.global_step}) has passed scaled_hook_only_steps ({scaled_hook_only_steps}). LoRA parameters are trainable.")
        elif (
            self.config.training_mode == "auto-finetune"
            and self.config.hook_only_steps == 0
        ):
            scaled_hook_only_steps = max(1, self.config.hook_only_steps // self.accelerator.num_processes)

            if self.accelerator.is_main_process:
                print("Training in 'auto-finetune' mode with hook_only_steps > 0.")
                print(f"  Config hook_only_steps: {self.config.hook_only_steps} (equivalent single-GPU steps)")
                print(f"  Scaled hook_only_steps: {scaled_hook_only_steps} (actual optimizer steps with {self.accelerator.num_processes} GPUs)")

            if self.global_step < scaled_hook_only_steps:
                if self.accelerator.is_main_process:
                    print(f"  Resumed step ({self.global_step}) is less than scaled_hook_only_steps ({scaled_hook_only_steps}). Freezing LoRA parameters.")
                frozen_params = 0
                unwrapped_model = self.accelerator.unwrap_model(self.model)
                for name, param in unwrapped_model.named_parameters():
                    if "hook" not in name:
                        param.requires_grad_(False)
                        frozen_params += param.numel()
                if self.accelerator.is_main_process:
                    print(f"  Froze {frozen_params} parameters.")
            else:
                 if self.accelerator.is_main_process:
                    print(f"  Resumed step ({self.global_step}) has passed scaled_hook_only_steps ({scaled_hook_only_steps}). Finetune parameters are trainable.")

        prev_lora_state = None
        prev_finetune_state = None
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

                total_loss, main_loss, hook_loss, distill_loss = self._compute_loss(
                    batch, self.global_step
                )
                self.accelerator.backward(total_loss)

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

                    if self.config.hook_only_steps > 0:
                        scaled_hook_only_steps = max(1, self.config.hook_only_steps // self.accelerator.num_processes)
                        if self.global_step == scaled_hook_only_steps:
                            if self.config.training_mode == "auto-lora":
                                print(
                                    f"\n[Step {self.global_step}]: Hook-only phase complete. Unfreezing LoRA parameters..."
                                )
                                unfrozen = 0
                                for name, param in self.model.named_parameters():
                                    if "lora_" in name:
                                        param.requires_grad_(True)
                                        unfrozen += param.numel()
                                print(f"  Unfroze {unfrozen} LoRA parameters.")
                            elif self.config.training_mode == "auto-finetune":
                                print(
                                    f"\n[Step {self.global_step}]: Hook-only phase complete. Unfreezing finetune parameters..."
                                )
                                unfrozen = 0
                                for name, param in self.model.named_parameters():
                                    if "hook" not in name:
                                        param.requires_grad_(True)
                                        unfrozen += param.numel()
                                print(f"  Unfroze {unfrozen} parameters.")

                    gathered_total_loss = self.accelerator.gather(total_loss).mean()
                    gathered_main_loss = self.accelerator.gather(main_loss).mean()
                    gathered_hook_loss = self.accelerator.gather(hook_loss).mean()
                    gathered_distill_loss = self.accelerator.gather(distill_loss).mean()

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
                        # lora metrics
                        if self.config.training_mode in ["lora", "auto-lora"]:
                            lora_params = [
                                p
                                for n, p in self.model.named_parameters()
                                if "lora_" in n
                            ]

                            lora_norm = torch.norm(
                                torch.stack(
                                    [torch.norm(p.detach()) for p in lora_params]
                                )
                            ).item()

                            lora_grads = [
                                torch.norm(p.grad.detach())
                                for p in lora_params
                                if p.grad is not None
                            ]

                            if len(lora_grads) > 0:
                                lora_grad_norm = torch.norm(
                                    torch.stack(lora_grads)
                                ).item()
                            else:
                                lora_grad_norm = 0.0

                            if prev_lora_state is not None:
                                deltas = [
                                    torch.norm(p.detach().cpu() - prev_lora_state[i])
                                    for i, p in enumerate(lora_params)
                                ]
                                lora_delta = torch.mean(torch.stack(deltas)).item()
                            else:
                                lora_delta = 0.0
                            prev_lora_state = [
                                p.detach().cpu().clone() for p in lora_params
                            ]

                            if self.writer:
                                self.writer.add_scalar(
                                    "lora/param_norm", lora_norm, self.global_step
                                )
                                self.writer.add_scalar(
                                    "lora/grad_norm", lora_grad_norm, self.global_step
                                )
                                self.writer.add_scalar(
                                    "lora/param_delta", lora_delta, self.global_step
                                )
                        elif self.config.training_mode == "finetune":
                            all_trainable_params = [
                                p for p in self.model.parameters() if p.requires_grad
                            ]
                            param_norm = torch.norm(
                                torch.stack(
                                    [
                                        torch.norm(p.detach())
                                        for p in all_trainable_params
                                    ]
                                )
                            ).item()
                            param_grads = [
                                torch.norm(p.grad.detach())
                                for p in all_trainable_params
                                if p.grad is not None
                            ]
                            grad_norm = (
                                torch.norm(torch.stack(param_grads)).item()
                                if param_grads
                                else 0.0
                            )
                            if prev_finetune_state is not None:
                                deltas = [
                                    torch.norm(
                                        p.detach().cpu() - prev_finetune_state[i]
                                    )
                                    for i, p in enumerate(all_trainable_params)
                                ]
                                param_delta = torch.mean(torch.stack(deltas)).item()
                            else:
                                param_delta = 0.0

                            prev_finetune_state = [
                                p.detach().cpu().clone() for p in all_trainable_params
                            ]

                            if self.writer:
                                self.writer.add_scalar(
                                    "finetune/param_norm", param_norm, self.global_step
                                )
                                self.writer.add_scalar(
                                    "finetune/grad_norm", grad_norm, self.global_step
                                )
                                self.writer.add_scalar(
                                    "finetune/param_delta",
                                    param_delta,
                                    self.global_step,
                                )

                        if self.writer:
                            self.writer.add_scalar(
                                "train/total_loss",
                                gathered_total_loss.item(),
                                self.global_step,
                            )
                            self.writer.add_scalar(
                                "train/main_loss",
                                gathered_main_loss.item(),
                                self.global_step,
                            )
                            self.writer.add_scalar(
                                "train/hook_loss",
                                gathered_hook_loss.item(),
                                self.global_step,
                            )
                            self.writer.add_scalar(
                                "train/distill_loss",
                                gathered_distill_loss.item(),
                                self.global_step,
                            )
                            self.writer.add_scalar(
                                "train/lr",
                                self.lr_scheduler.get_last_lr()[0],
                                self.global_step,
                            )

                        if self.global_step % log_trigger == 0:
                            postfix_dict = {
                                "loss": f"{gathered_total_loss.item():.4f}",
                                "main_loss": f"{gathered_main_loss.item():.4f}",
                                "hook_loss": f"{gathered_hook_loss.item():.4f}",
                                "distill_loss": f"{gathered_distill_loss.item():.4f}",
                                "lr": f"{self.lr_scheduler.get_last_lr()[0]:.2e}",
                                "step": self.global_step,
                            }
                            if self.config.training_mode in ["lora", "auto-lora"]:
                                postfix_dict["lora_norm"] = f"{lora_norm:.4f}"
                                postfix_dict["lora_grad_norm"] = f"{lora_grad_norm:.4e}"
                                postfix_dict["lora_delta"] = f"{lora_delta:.4e}"
                            elif self.config.training_mode == "finetune":
                                postfix_dict["param_norm"] = f"{param_norm:.4f}"
                                postfix_dict["grad_norm"] = f"{grad_norm:.4e}"
                                postfix_dict["param_delta"] = f"{param_delta:.4e}"

                            progress_bar.set_postfix(postfix_dict)

                        if (
                            inference_trigger
                            and self.global_step % inference_trigger == 0
                        ):
                            self.run_inference_validation(self.global_step)

            if self.accelerator.is_main_process:
                print(f"Epoch {epoch+1} completed.")

        if self.accelerator.is_main_process:
            print("Training completed. Saving final model...")
            final_path = os.path.join(self.config.output_dir, "final.pt")
            self.save_checkpoint(final_path)  # save_checkpoint 内部需要 unwrap
            print(f"Final model saved to {final_path}")

    def save_checkpoint(self, save_path: str, save_optimizer: bool = True):
        if self.accelerator.is_main_process:
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            model_state = {}
            unwrapped_model = self.accelerator.unwrap_model(self.model)
            state_dict = unwrapped_model.state_dict()
            mode = self.config.training_mode
            if mode == "lora":
                for k, v in state_dict.items():
                    if "lora" in k or "hook" in k:
                        model_state[k] = v

            elif mode == "hook_only":
                model_state = {k: v for k, v in state_dict.items() if "hook" in k}

            elif mode == "finetune":
                model_state = state_dict

            else:
                raise ValueError(f"Unknown training_mode: {mode}")

            optimizer_state = None
            if save_optimizer and self.optimizer is not None:
                optimizer_state = self.optimizer.state_dict()

            lr_scheduler_state = None
            if save_optimizer and self.lr_scheduler is not None:
                lr_scheduler_state = self.lr_scheduler.state_dict()

            checkpoint = {
                "model": model_state,
                "optimizer": optimizer_state,
                "lr_scheduler": lr_scheduler_state,
                "epoch": getattr(self, "epoch", 0),
                "global_step": getattr(self, "global_step", 0),
                "training_mode": mode,
                "mixed_precision": self.config.mixed_precision,
                "extra_config": getattr(self.config, "extra_config", {}),
                "best_val_metric": getattr(self, "best_val_metric", None),
            }

            torch.save(checkpoint, save_path)
            print(
                f"Saved checkpoint to {save_path} "
                f"(mode={mode}, step={checkpoint['global_step']})"
            )

        self.accelerator.wait_for_everyone()

    def load_checkpoint(
        self,
        ckpt_path: str,
        strict: bool = True,
        load_optimizer: bool = True,
        for_inference: bool = False,
    ):
        if not os.path.exists(ckpt_path):
            raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}")

        checkpoint = torch.load(ckpt_path, map_location="cpu")
        ckpt_mode = checkpoint.get("training_mode", "unknown")

        if self.accelerator.is_main_process:
            print(
                f"Loading checkpoint from {ckpt_path} "
                f"(saved_mode={ckpt_mode}, current_mode={self.config.training_mode})"
            )

        model_state = checkpoint["model"]

        unwrapped_model = self.accelerator.unwrap_model(self.model)

        mode = self.config.training_mode
        if mode == "lora":
            load_dict = {
                k: v for k, v in model_state.items() if "lora" in k or "hook" in k
            }
            missing, unexpected = unwrapped_model.load_state_dict(
                load_dict, strict=False
            )
        elif mode == "hook_only":
            load_dict = {k: v for k, v in model_state.items() if "hook" in k}
            missing, unexpected = unwrapped_model.load_state_dict(
                load_dict, strict=False
            )
        elif mode == "finetune":
            missing, unexpected = unwrapped_model.load_state_dict(
                model_state, strict=strict
            )
        else:
            raise ValueError(f"Unknown training_mode: {mode}")

        print(
            f"Loaded model params (missing={len(missing)}, unexpected={len(unexpected)})"
        )

        if (not for_inference) and load_optimizer:
            if checkpoint.get("optimizer") and self.optimizer is not None:
                self.optimizer.load_state_dict(checkpoint["optimizer"])
            if checkpoint.get("lr_scheduler") and self.lr_scheduler is not None:
                self.lr_scheduler.load_state_dict(checkpoint["lr_scheduler"])
            self.global_step = checkpoint.get("global_step", 0)
            self.epoch = checkpoint.get("epoch", 0)
            self.best_val_metric = checkpoint.get("best_val_metric", None)
            print(f"Resumed training from step {self.global_step}, epoch {self.epoch}")

        if self.accelerator.is_main_process:
            print(f"Loaded checkpoint successfully (mode={self.config.training_mode})")

    def resume_if_available(self, resume_dir: str = None):
        ckpt_dir = resume_dir or self.config.output_dir

        if not self.accelerator.is_main_process:
            self.accelerator.wait_for_everyone()
            return (
                os.path.isdir(ckpt_dir)
                and len(glob.glob(os.path.join(ckpt_dir, "*.pt"))) > 0
            )

        if not os.path.isdir(ckpt_dir):
            print(f"No checkpoint directory found at {ckpt_dir}, starting fresh.")
            self.accelerator.wait_for_everyone()
            return False

        ckpts = glob.glob(os.path.join(ckpt_dir, "*.pt"))
        if not ckpts:
            print(f"No checkpoints found in {ckpt_dir}, starting fresh.")
            self.accelerator.wait_for_everyone()
            return False

        def extract_step(path):
            match_step = re.search(r"step_(\d+)\.pt$", os.path.basename(path))
            if match_step:
                return int(match_step.group(1))
            if "best.pt" in path:
                return -1
            if "final.pt" in path:
                return -2
            return -3

        latest_ckpt = max(ckpts, key=extract_step)

        step_ckpts = [p for p in ckpts if "step_" in os.path.basename(p)]
        if step_ckpts:
            latest_step_ckpt = max(step_ckpts, key=extract_step)
            if extract_step(latest_step_ckpt) > extract_step(latest_ckpt):
                latest_ckpt = latest_step_ckpt

        print(f"Found latest checkpoint: {latest_ckpt}")
        try:
            self.load_checkpoint(latest_ckpt)
            resumed = True
        except Exception as e:
            print(
                f"Failed to load checkpoint {latest_ckpt}. Error: {e}. Starting fresh."
            )
            resumed = False

        self.accelerator.wait_for_everyone()
        return resumed

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

            self.save_checkpoint(save_path, save_optimizer=False)
            
            if self.accelerator.is_main_process:
                print(f"New best model saved at {save_path} (metric={val_metric:.4f})")
        else:
            if self.accelerator.is_main_process:
                print(
                    f"Validation metric {val_metric:.4f} not improved (best={self.best_val_metric:.4f})"
                )

    def maybe_save_periodically(self):
        save_path = os.path.join(
            self.config.output_dir, f"checkpoint_step_{self.global_step:06d}.pt"
        )
        self.save_checkpoint(save_path)
