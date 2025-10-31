import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from transformers import get_scheduler
from peft import LoraConfig as PeftLoraConfig, get_peft_model, PeftModel
from tqdm import tqdm
import os

from diffusers.pipelines.flux.pipeline_flux_kontext import retrieve_latents
from project_name.hook import HookManager
from project_name.utils import get_pipeline
from project_name.training.config import TrainerConfig
from project_name.training.dataset import ImageTextDataset, collate_fn


class DiffusionModelTrainer:
    def __init__(self, config: TrainerConfig):
        self.config = config
        self.model_type = config.model_type
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.generator = (
            torch.Generator("cuda").manual_seed(config.seed)
            if torch.cuda.is_available()
            else torch.Generator().manual_seed(config.seed)
        )

        self.dtype = torch.float32
        if config.mixed_precision == "fp16":
            self.dtype = torch.float16
        elif config.mixed_precision == "bf16":
            self.dtype = torch.bfloat16

        self.pipe, self.model_attrs = get_pipeline(
            self.model_type,
            model_path=config.model_path,
            dtype=self.dtype,
            device=self.device,
        )

        # we only to get the ref of this modules, no need to move them to device again
        self.vae = self.pipe.vae
        self.scheduler = self.pipe.scheduler
        self.model = tuple(
            getattr(self.pipe, model) for model in self.model_attrs
        )  # DiT

        if self.model_type == "flux-dev" or self.model_type == "flux-kontext":
            self.tokenizers = (self.pipe.tokenizer, self.pipe.tokenizer_2)
            self.text_encoders = (self.pipe.text_encoder, self.pipe.text_encoder_2)
        elif (
            self.model_type == "pixart-sigma"
            or self.model_type == "qwen-image"
            or self.model_type == "qwen-image-edit"
            or self.model_type == "wan"
        ):
            self.tokenizers = (self.pipe.tokenizer,)
            self.text_encoders = (self.pipe.text_encoder,)
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
        else:
            raise ValueError(f"Undefined model_type: {self.model_type}")

        if self.image_processor is None or self.vae is None or self.model is None:
            raise ValueError("NoneType found in image_processor, vae or model.")

        print("model_type:", config.model_type)
        print("image_processor:", self.image_processor.__class__.__name__)
        print("vae:", self.vae.__class__.__name__)
        print("scheduler:", self.scheduler.__class__.__name__)
        print("model (DiT):", self.model.__class__.__name__)

        # inject
        if config.trainable_hook_config:
            hook_class, hook_kwargs = config.trainable_hook_config
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

        print(f"--- Training Config ---")
        print(f"  Mode: {config.training_mode}")
        print(f"  Micro Batch Size: {config.train_batch_size}")
        print(f"  Grad Acc: {config.gradient_accumulation_steps}")
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
        elif self.config.training_mode == "lora":
            print("Training mode: lora (applying LoRA to DiT model)")
            if not self.config.lora_config.enable:
                raise ValueError(
                    "LoRA configuration is disabled in 'lora' training mode."
                )

            self.model.requires_grad_(False)
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
            params_to_train.extend(self.model.parameters())

            for hook in HookManager.get_all_hooks().values():
                if isinstance(hook, nn.Module):
                    hook.requires_grad_(True)
                    params_to_train.extend(hook.parameters())
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

    @torch.no_grad()
    def _prepare_model_inputs(self, batch):
        batch_size = batch["pixel_values"].shape[0]

        # vae encode
        pixel_values = batch["pixel_values"].to(self.device, dtype=self.vae.dtype)
        if self.model_type == "flux-dev" or self.model_type == "flux-kontext":
            img_latents = retrieve_latents(
                self.vae.encode(pixel_values),
                generator=self.generator,
                sample_mode="argmax",
            )
            img_latents = (
                img_latents - self.vae.config.shift_factor
            ) * self.vae.config.scaling_factor
        elif self.model_type == "sd3":
            pass
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

            model_kwargs['img_shapes'] = img_shapes
        else:
            raise NotImplementedError(
                f"_prepare_model_inputs not implemented for model_type: {self.model_type}"
            )

        model_kwargs = {}

        # text encode
        input_ids_tuple = batch["input_ids"]
        if self.model_type == "flux-dev" or self.model_type == "flux-kontext":
            model_kwargs["return_dict"] = False
            model_kwargs["joint_attention_kwargs"] = self.pipe.joint_attention_kwargs

            num_images_per_prompt = 1
            # clip -> pooled_prompt_embeds
            input_ids_clip = input_ids_tuple[0].to(self.device)
            pooled_prompt_embeds = self.text_encoders[0](
                input_ids_clip, output_hidden_states=False
            )
            pooled_prompt_embeds = pooled_prompt_embeds.pooler_output
            pooled_prompt_embeds = pooled_prompt_embeds.to(
                dtype=self.text_encoders[0].dtype, device=self.text_encoders[0].device
            )
            pooled_prompt_embeds = pooled_prompt_embeds.repeat(1, num_images_per_prompt)
            pooled_prompt_embeds = pooled_prompt_embeds.view(
                batch_size * num_images_per_prompt, -1
            )

            # t5 -> prompt_embeds
            input_ids_t5 = input_ids_tuple[1].to(self.device)
            prompt_embeds = self.text_encoders[1](
                input_ids_t5, output_hidden_states=False
            )[0]
            prompt_embeds = prompt_embeds.to(
                dtype=self.text_encoders[1].dtype, device=self.text_encoders[1].device
            )
            _, seq_len, _ = prompt_embeds.shape
            prompt_embeds = prompt_embeds.repeat(1, num_images_per_prompt, 1)
            prompt_embeds = prompt_embeds.view(
                batch_size * num_images_per_prompt, seq_len, -1
            )

            text_ids = torch.zeros(prompt_embeds.shape[1], 3).to(
                device=self.device, dtype=self.text_encoders[0].dtype
            )

            model_kwargs["encoder_hidden_states"] = prompt_embeds
            model_kwargs["pooled_projections"] = pooled_prompt_embeds
            model_kwargs["txt_ids"] = text_ids

            # add noise
            noise = torch.randn_like(img_latents)
            timesteps = torch.randint(
                1,
                self.scheduler.config.num_train_timesteps,
                (batch_size,),
                device=img_latents.device,
            ).long()

            noisy_latents = self.scheduler.scale_noise(
                sample=img_latents,
                timestep=timesteps,
                noise=noise,
            )
        elif self.model_type == "sd3":
            pass
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
            split_hidden_states = _extract_masked_hidden(hidden_states, attention_masks_tuple[0])
            split_hidden_states = [e[drop_idx:] for e in split_hidden_states]
            attn_mask_list = [torch.ones(e.size(0), dtype=torch.long, device=e.device) for e in split_hidden_states]
            max_seq_len = max([e.size(0) for e in split_hidden_states])
            prompt_embeds = torch.stack(
                [torch.cat([u, u.new_zeros(max_seq_len - u.size(0), u.size(1))]) for u in split_hidden_states]
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
        else:
            raise NotImplementedError(
                f"_prepare_model_inputs not implemented for model_type: {self.model_type}"
            )

        return noisy_latents, timesteps, noise, model_kwargs

    def _compute_loss(self, batch):
        noisy_latents, timesteps, noise_target, model_kwargs = (
            self._prepare_model_inputs(batch)
        )
        with torch.amp.autocast(
            device_type="cuda", enabled=self.config.mixed_precision != "no"
        ):
            noise_pred = self.model(
                hidden_states=noisy_latents.to(self.dtype),
                timestep=timesteps,
                **model_kwargs,
            )[0]

        main_loss = F.mse_loss(
            noise_pred.float(), noise_target.float(), reduction="mean"
        )
        hook_loss = torch.tensor(0.0, device=self.device)
        custom_losses = []
        for hook in HookManager.get_all_hooks().values():
            if isinstance(hook, nn.Module) and hasattr(hook, "loss"):
                custom_losses.append(hook.loss())

        if custom_losses:
            hook_loss = torch.stack(custom_losses).mean()

        total_loss = (
            self.config.main_loss_weight * main_loss
            + self.config.hook_loss_weight * hook_loss
        )

        return total_loss, main_loss, hook_loss

    def train(self):
        print("Starting training...")
        global_step = 0

        for hook in HookManager.get_all_hooks().values():
            if isinstance(hook, nn.Module):
                hook.train()
        self.model.train()
        self.vae.eval()
        for text_encoder in self.text_encoders:
            text_encoder.eval()

        for epoch in range(self.config.num_epochs):
            progress_bar = tqdm(
                self.train_dataloader,
                desc=f"Epoch {epoch+1}/{self.config.num_epochs}",
                total=len(self.train_dataloader),
            )

            self.optimizer.zero_grad()

            for step, batch in enumerate(progress_bar):
                total_loss, main_loss, hook_loss = self._compute_loss(batch)
                loss_to_backward = total_loss / self.config.gradient_accumulation_steps
                loss_to_backward.backward()
                is_update_step = (
                    step + 1
                ) % self.config.gradient_accumulation_steps == 0
                is_last_step = (step + 1) == len(self.train_dataloader)

                if is_update_step or is_last_step:
                    # torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)

                    self.optimizer.step()
                    self.lr_scheduler.step()
                    self.optimizer.zero_grad()
                    global_step += 1
                    if global_step % self.config.logging_steps == 0:
                        progress_bar.set_postfix(
                            loss=total_loss.item(),  # (显示 micro-batch 的 loss)
                            main_loss=main_loss.item(),
                            hook_loss=hook_loss.item(),
                            lr=self.lr_scheduler.get_last_lr()[0],
                            step=global_step,
                        )
                    if global_step % self.config.save_steps == 0:
                        self.save_model(global_step)

        print("Training completed.")
        self.save_model("final")

    def save_model(self, step_name):
        output_dir = os.path.join(self.config.output_dir, f"checkpoint-{step_name}")
        os.makedirs(output_dir, exist_ok=True)

        save_model = self.model
        if isinstance(save_model, PeftModel):
            save_model.save_pretrained(os.path.join(output_dir, "model_lora"))
            print(f"LoRA adapter {output_dir}/model_lora")
        elif self.config.training_mode == "finetune":
            torch.save(
                save_model.state_dict(), os.path.join(output_dir, "model_dit.pt")
            )
            print(f"Finetune {output_dir}/model_dit.pt")

        hook_params = {}
        for name, hook in HookManager.get_all_hooks():
            if isinstance(hook, nn.Module):
                hook_params[name] = hook.state_dict()

        if hook_params:
            torch.save(hook_params, os.path.join(output_dir, "hook_params.pt"))
            print(f"Hook {output_dir}/hook_params.pt")
