import os
import json
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from transformers import get_scheduler
from peft import LoraConfig as PeftLoraConfig, get_peft_model, PeftModel
from tqdm import tqdm

from baselines.diffcr import DiffCRDitBlockHook
from diffusers.pipelines.flux.pipeline_flux_kontext import retrieve_latents
from project_name.hook import HookManager
from project_name.utils import get_pipeline
from project_name.training.config import TrainerConfig
from project_name.training.dataset import ImageTextDataset, collate_fn
from project_name.utils import global_context


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
        self.image_processor = self.pipe.image_processor
        self.vae = self.pipe.vae
        self.scheduler = self.pipe.scheduler
        self.model = tuple(
            getattr(self.pipe, model) for model in self.model_attrs
        )  # DiT
        # TODO: support dual DiT like Wan
        self.model = self.model[0]

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
                )
                val_batch_size = config.val_batch_size if config.val_batch_size else config.train_batch_size
                self.val_dataloader = DataLoader(
                    val_dataset,
                    batch_size=val_batch_size,
                    shuffle=False,
                    collate_fn=collate_fn,
                )
                print(f"  Validation dataset size: {len(val_dataset)}")
            except Exception as e:
                print(f"Warning: Failed to load validation dataset. Validation will be skipped. Error: {e}")
                self.val_dataloader = None
        else:
            print("Validation data not provided (val_metadata_file or val_image_dir is None), skipping validation setup.")

        num_update_steps_per_epoch = max(
            1, len(self.train_dataloader) // config.gradient_accumulation_steps
        )
        self.num_training_steps = num_update_steps_per_epoch * config.num_epochs
        self.lr_scheduler = self._setup_scheduler()

        self.writer = None
        self.pil_to_tensor_transform = None
        if config.log_with_tensorboard:
            from torch.utils.tensorboard import SummaryWriter
            import torchvision.transforms as T
            tb_log_path = os.path.join(config.output_dir, config.tensorboard_log_dir)
            os.makedirs(tb_log_path, exist_ok=True)
            self.writer = SummaryWriter(log_dir=tb_log_path)
            self.pil_to_tensor_transform = T.ToTensor()
            print(f"TensorBoard logging enabled. Logs will be saved to: {tb_log_path}")

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
        elif self.config.training_mode == "lora":
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

    @torch.no_grad()
    def _prepare_model_inputs(self, batch):
        batch_size = batch["pixel_values"].shape[0]
        model_kwargs = {}

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

            height, width = img_latents.shape[2], img_latents.shape[3]

            img_ids = self.pipe._prepare_latent_image_ids(batch_size, height // 2, width // 2, self.device, self.dtype)
            model_kwargs["img_ids"] = img_ids
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

        # text encode
        input_ids_tuple = batch["input_ids"]
        if self.model_type == "flux-dev" or self.model_type == "flux-kontext":
            model_kwargs["return_dict"] = False
            model_kwargs["joint_attention_kwargs"] = {}

            num_images_per_prompt = 1
            guidance_scale = 3.5
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
            indices = torch.randint(0, self.scheduler.config.num_train_timesteps, (batch_size,)).to(self.device)
            schedule_timesteps_gpu = self.scheduler.timesteps.to(self.device)
            timesteps = schedule_timesteps_gpu[indices]
            noisy_latents = self.scheduler.scale_noise(
                sample=img_latents,
                timestep=timesteps,
                noise=noise,
            )

            # handle guidance
            guidance = torch.full([1], guidance_scale, device=self.device, dtype=torch.float32)
            guidance = guidance.expand(batch_size)
            model_kwargs["guidance"] = guidance
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
            global_context.update(encoder_hidden_seq_len=model_kwargs["encoder_hidden_states"].size(1))

        if self.model_type == "flux-dev" or self.model_type == "flux-kontext":
            # pack for hidden state
            height, width = noisy_latents.shape[-2:]
            num_channels_latents = self.pipe.transformer.config.in_channels // 4
            noisy_latents = self.pipe._pack_latents(noisy_latents, batch_size, num_channels_latents, height, width)
            noise = self.pipe._pack_latents(noise, batch_size, num_channels_latents, height, width)
            timesteps = timesteps / 1000

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

        ### DiffCR Loss ###
        ratios = []
        for hook in HookManager.get_all_hooks().values():
            if isinstance(hook, DiffCRDitBlockHook):
                ratios.append(hook.diff_ratio)
        if ratios:
            avg_ratio = torch.mean(torch.stack(ratios))
            target_ratio = self.config.extra_config.get("target_ratio", 0.5)
            target_ratio_tensor = torch.tensor(target_ratio, device=avg_ratio.device)
            diffcr_loss = F.mse_loss(avg_ratio, target_ratio_tensor)
            custom_losses.append(diffcr_loss)

        if custom_losses:
            hook_loss = torch.stack(custom_losses).mean()

        total_loss = (
            self.config.main_loss_weight * main_loss
            + self.config.hook_loss_weight * hook_loss
        )

        return total_loss, main_loss, hook_loss
    
    @torch.no_grad()
    def validate(self, global_step):
        if self.val_dataloader is None or not self.config.run_validation:
            return

        print(f"\nRunning validation at step {global_step}...")
        
        self.model.eval()
        for hook in HookManager.get_all_hooks().values():
            if isinstance(hook, nn.Module):
                hook.eval()

        total_val_loss = 0.0
        total_main_loss = 0.0
        total_hook_loss = 0.0
        
        val_progress_bar = tqdm(
            self.val_dataloader, 
            desc=f"Validation Step {global_step}", 
            leave=False,
            total=len(self.val_dataloader)
        )

        for batch in val_progress_bar:
            try:
                total_loss, main_loss, hook_loss = self._compute_loss(batch)
                
                total_val_loss += total_loss.item()
                total_main_loss += main_loss.item()
                total_hook_loss += hook_loss.item()
            except Exception as e:
                print(f"Error during validation batch: {e}")
                continue

        if len(self.val_dataloader) > 0:
            avg_val_loss = total_val_loss / len(self.val_dataloader)
            avg_main_loss = total_main_loss / len(self.val_dataloader)
            avg_hook_loss = total_hook_loss / len(self.val_dataloader)

            print(f"--- Validation Results (Step {global_step}) ---")
            print(f"  Avg Total Loss: {avg_val_loss:.4f}")
            print(f"  Avg Main Loss:  {avg_main_loss:.4f}")
            print(f"  Avg Hook Loss:  {avg_hook_loss:.4f}")
            print(f"--------------------------------------")

            block_ratios_for_tb = {}
            hook_ratios = []
            for name, hook in HookManager.get_all_hooks().items():
                if isinstance(hook, DiffCRDitBlockHook):
                    ratio_val = hook.diff_ratio.item() 
                    hook_ratios.append(ratio_val)
                    print(f"    {name}: {ratio_val:.4f}")
                    clean_name = name.split('.')[-1]
                    block_ratios_for_tb[clean_name] = ratio_val
            print(f"--------------------------------------")

            if self.writer:
                self.writer.add_scalar("validation/total_loss", avg_val_loss, global_step)
                self.writer.add_scalar("validation/main_loss", avg_main_loss, global_step)
                self.writer.add_scalar("validation/hook_loss", avg_hook_loss, global_step)
                if block_ratios_for_tb:
                    self.writer.add_scalars("validation/block_ratios", block_ratios_for_tb, global_step)
                    avg_ratio = sum(hook_ratios) / len(hook_ratios)
                    self.writer.add_scalar("validation/avg_ratio", avg_ratio, global_step)
        else:
            print(f"Validation dataloader is empty, skipping logging.")

        self.model.train()
        for hook in HookManager.get_all_hooks().values():
            if isinstance(hook, nn.Module):
                hook.train()

    @torch.no_grad()
    def run_inference_validation(self, global_step):
        if (
            not self.config.run_inference_validation
            or not self.config.inference_metadata_file
        ):
            return

        print(f"\nRunning inference validation from file: {self.config.inference_metadata_file}...")

        prompts = []
        try:
            with open(self.config.inference_metadata_file, 'r', encoding='utf-8') as f:
                for line in f:
                    item = json.loads(line)
                    if 'text' in item:
                        prompts.append(item['text'])
            if not prompts:
                print("  Warning: Inference metadata file is empty or contains no captions. Skipping.")
                return
        except Exception as e:
            print(f"  Failed to read or parse inference metadata file: {e}. Skipping inference validation.")
            return

        self.model.eval()
        for hook in HookManager.get_all_hooks().values():
            if isinstance(hook, nn.Module):
                hook.eval()

        inference_generator = torch.Generator(device=self.device).manual_seed(self.config.inference_seed)

        # TODO: dual model attr
        model_attr_name = self.model_attrs[0]
        original_model_in_pipe = getattr(self.pipe, model_attr_name)
        setattr(self.pipe, model_attr_name, self.model)
        
        print(f"  Temporarily patching pipe.{model_attr_name} with trained model ({self.model.__class__.__name__})")

        original_timesteps = self.scheduler.timesteps.clone()
        print(f"  Saved original training scheduler timesteps (length {len(original_timesteps)})")

        sample_dir = os.path.join(self.config.output_dir, self.config.inference_output_dir, f"step_{global_step}")
        os.makedirs(sample_dir, exist_ok=True)
        for i, prompt in enumerate(prompts):
            try:
                image = self.pipe(
                    prompt=prompt,
                    generator=inference_generator,
                    num_inference_steps=self.config.inference_num_inference_steps,
                ).images[0]
                
                safe_prompt = "".join(c if c.isalnum() else '_' for c in prompt)[:50]
                image.save(os.path.join(sample_dir, f"sample_{i+1:03d}_{safe_prompt}.png"))
                if self.writer and self.pil_to_tensor_transform:
                    tag = f"inference_sample/{i+1:03d}_{safe_prompt}"
                    image_tensor = self.pil_to_tensor_transform(image)
                    self.writer.add_image(tag, image_tensor, global_step)
            except Exception as e:
                print(f"  Failed to generate image for prompt: '{prompt}'. Error: {e}")
        
        print(f"  Saved {len(prompts)} inference samples to {sample_dir}")

        setattr(self.pipe, model_attr_name, original_model_in_pipe)
        print(f"  Restored original model to pipe.{model_attr_name}")

        self.scheduler.timesteps = original_timesteps
        print(f"  Restored original training scheduler timesteps (length {len(self.scheduler.timesteps)})")

        torch.clear_autocast_cache()
        self.model.train()
        for hook in HookManager.get_all_hooks().values():
            if isinstance(hook, nn.Module):
                hook.train()

    def train(self):
        print("Starting training...")
        global_step = 0

        global_context.update(encoder_flag=True)

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

                    if self.writer:
                        self.writer.add_scalar("train/total_loss", total_loss.item(), global_step)
                        self.writer.add_scalar("train/main_loss", main_loss.item(), global_step)
                        self.writer.add_scalar("train/hook_loss", hook_loss.item(), global_step)
                        self.writer.add_scalar("train/lr", self.lr_scheduler.get_last_lr()[0], global_step)

                    if global_step % self.config.logging_steps == 0:
                        progress_bar.set_postfix(
                            loss=total_loss.item(),  # (显示 micro-batch 的 loss)
                            main_loss=main_loss.item(),
                            hook_loss=hook_loss.item(),
                            lr=self.lr_scheduler.get_last_lr()[0],
                            step=global_step,
                        )
                    
                    if global_step % self.config.validation_steps == 0:
                        self.validate(global_step)
                    
                    if global_step % self.config.inference_steps == 0:
                        self.run_inference_validation(global_step)

                    if global_step % self.config.save_steps == 0:
                        self.save_model(global_step)

        print("Training completed.")
        self.save_model("final")
    
    def save_model(self, step_name):
        output_dir = os.path.join(self.config.output_dir, f"checkpoint-{step_name}")
        os.makedirs(output_dir, exist_ok=True)

        save_model = self.model
        
        if self.config.training_mode == "lora":
            if not isinstance(save_model, PeftModel):
                 print(f"Warning: training_mode is 'lora' but model is not a PeftModel. Skipping model save.")
            else:
                save_model.save_pretrained(os.path.join(output_dir, "model_lora"))
                print(f"Saved LoRA adapter to {output_dir}/model_lora")
            
            hook_params = {}
            for name, hook in HookManager.get_all_hooks().items():
                if isinstance(hook, nn.Module):
                    hook_params[name] = hook.state_dict()
            if hook_params:
                torch.save(hook_params, os.path.join(output_dir, "hook_params.pt"))
                print(f"Saved Hook parameters to {output_dir}/hook_params.pt")

        elif self.config.training_mode == "finetune":
            torch.save(
                save_model.state_dict(), os.path.join(output_dir, "model_dit.pt")
            )
            print(f"Saved Finetuned model (DiT + Hooks) to {output_dir}/model_dit.pt")

        elif self.config.training_mode == "hook_only":
            hook_params = {}
            for name, hook in HookManager.get_all_hooks().items():
                if isinstance(hook, nn.Module):
                    hook_params[name] = hook.state_dict()
            
            if hook_params:
                torch.save(hook_params, os.path.join(output_dir, "hook_params.pt"))
                print(f"Saved Hook parameters to {output_dir}/hook_params.pt")
            else:
                print(f"Warning: 'hook_only' mode but no hook parameters found to save.")
