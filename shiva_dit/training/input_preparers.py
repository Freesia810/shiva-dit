import torch
from diffusers.pipelines.flux.pipeline_flux_kontext import retrieve_latents


def prepare_flux_inputs(trainer, batch):
    model_kwargs = {}

    # default params
    model_kwargs["return_dict"] = False
    model_kwargs["joint_attention_kwargs"] = {}

    batch_size = batch["pixel_values"].shape[0]
    pixel_values = batch["pixel_values"].to(dtype=trainer.vae.dtype)
    input_ids_tuple = batch["input_ids"]

    ## image encode
    img_latents = retrieve_latents(
        trainer.vae.encode(pixel_values),
        generator=trainer.generator,
        sample_mode="argmax",
    )
    img_latents = (
        img_latents - trainer.vae.config.shift_factor
    ) * trainer.vae.config.scaling_factor
    height, width = img_latents.shape[2], img_latents.shape[3]
    img_ids = trainer.pipe._prepare_latent_image_ids(
        batch_size, height // 2, width // 2, trainer.device, trainer.dtype
    )
    model_kwargs["img_ids"] = img_ids

    ## text encode
    num_images_per_prompt = 1
    # clip -> pooled_prompt_embeds
    input_ids_clip = input_ids_tuple[0]
    pooled_prompt_embeds = trainer.text_encoders[0](
        input_ids_clip, output_hidden_states=False
    )
    pooled_prompt_embeds = pooled_prompt_embeds.pooler_output
    pooled_prompt_embeds = pooled_prompt_embeds.to(
        dtype=trainer.dtype, device=trainer.device
    )
    pooled_prompt_embeds = pooled_prompt_embeds.repeat(1, num_images_per_prompt)
    pooled_prompt_embeds = pooled_prompt_embeds.view(
        batch_size * num_images_per_prompt, -1
    )

    # t5 -> prompt_embeds
    input_ids_t5 = input_ids_tuple[1]
    prompt_embeds = trainer.text_encoders[1](input_ids_t5, output_hidden_states=False)[0]
    prompt_embeds = prompt_embeds.to(dtype=trainer.dtype, device=trainer.device)
    _, seq_len, _ = prompt_embeds.shape
    prompt_embeds = prompt_embeds.repeat(1, num_images_per_prompt, 1)
    prompt_embeds = prompt_embeds.view(
        batch_size * num_images_per_prompt, seq_len, -1
    )
    text_ids = torch.zeros(prompt_embeds.shape[1], 3).to(
        dtype=trainer.dtype, device=trainer.device
    )

    model_kwargs["encoder_hidden_states"] = prompt_embeds
    model_kwargs["pooled_projections"] = pooled_prompt_embeds
    model_kwargs["txt_ids"] = text_ids
    if torch.rand(1).item() < trainer.config.caption_dropout_rate:
        prompt_embeds = trainer.empty_prompt_embeds
        pooled_prompt_embeds = trainer.empty_pooled_prompt_embeds
        text_ids = trainer.empty_text_ids

        model_kwargs["encoder_hidden_states"] = prompt_embeds
        model_kwargs["pooled_projections"] = pooled_prompt_embeds
        model_kwargs["txt_ids"] = text_ids

    # add noise
    noise = torch.randn_like(img_latents)
    indices = torch.randint(
        0,
        trainer.scheduler.config.num_train_timesteps,
        (batch_size,),
        device=trainer.device,
    )
    schedule_timesteps_gpu = trainer.scheduler.timesteps.to(trainer.device)
    timesteps = schedule_timesteps_gpu[indices]
    noisy_latents = trainer.scheduler.scale_noise(
        sample=img_latents,
        timestep=timesteps,
        noise=noise,
    )

    # handle guidance
    # in flux, we must use a fixed guidance scale for training
    guidance_scale = 3.5
    guidance = torch.full(
        [1], guidance_scale, device=trainer.device, dtype=torch.float32
    )
    guidance = guidance.expand(batch_size)
    model_kwargs["guidance"] = guidance

    return img_latents, noise, noisy_latents, timesteps, model_kwargs


def prepare_pixart_sigma_inputs(trainer, batch):
    model_kwargs = {}
    model_kwargs["return_dict"] = False
    model_kwargs["added_cond_kwargs"] = {"resolution": None, "aspect_ratio": None}

    batch_size = batch["pixel_values"].shape[0]
    pixel_values = batch["pixel_values"].to(dtype=trainer.vae.dtype)
    input_ids_tuple = batch["input_ids"]

    ## image encode
    img_latents = retrieve_latents(
        trainer.vae.encode(pixel_values),
        generator=trainer.generator,
        sample_mode="argmax",
    )
    img_latents = img_latents * trainer.vae.config.scaling_factor

    # text encode
    num_images_per_prompt = 1
    input_ids = input_ids_tuple[0]
    prompt_attention_mask = batch["attention_mask"][0]
    prompt_embeds = trainer.text_encoders[0](
        input_ids, attention_mask=prompt_attention_mask
    )[0].to(dtype=trainer.dtype, device=trainer.device)

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

    if torch.rand(1).item() < trainer.config.caption_dropout_rate:
        prompt_embeds = trainer.empty_prompt_embeds
        prompt_attention_mask = trainer.empty_prompt_attention_mask

    model_kwargs["encoder_hidden_states"] = prompt_embeds
    model_kwargs["encoder_attention_mask"] = prompt_attention_mask

    # add noise
    noise = torch.randn_like(img_latents)
    indices = torch.randint(
        0,
        trainer.scheduler.config.num_train_timesteps,
        (batch_size,),
        device=trainer.device,
    )
    schedule_timesteps_gpu = trainer.scheduler.timesteps.to(trainer.device)
    timesteps = schedule_timesteps_gpu[indices]
    noisy_latents = trainer.scheduler.add_noise(
        original_samples=img_latents,
        timesteps=timesteps,
        noise=noise,
    )

    return img_latents, noise, noisy_latents, timesteps, model_kwargs


def prepare_sd3_inputs(trainer, batch):
    model_kwargs = {}
    model_kwargs["return_dict"] = False
    model_kwargs["joint_attention_kwargs"] = {}

    batch_size = batch["pixel_values"].shape[0]
    pixel_values = batch["pixel_values"].to(dtype=trainer.vae.dtype)
    input_ids_tuple = batch["input_ids"]

    # image encode
    img_latents = retrieve_latents(
        trainer.vae.encode(pixel_values),
        generator=trainer.generator,
        sample_mode="argmax",
    )
    img_latents = (
        img_latents - trainer.vae.config.shift_factor
    ) * trainer.vae.config.scaling_factor

    # text encode
    num_images_per_prompt = 1
    def _clip_encode(text_encoder, text_input_ids, clip_skip=None):
        prompt_embeds = text_encoder(text_input_ids, output_hidden_states=True)
        pooled_prompt_embeds = prompt_embeds[0]

        if clip_skip is None:
            prompt_embeds = prompt_embeds.hidden_states[-2]
        else:
            prompt_embeds = prompt_embeds.hidden_states[-(clip_skip + 2)]

        prompt_embeds = prompt_embeds.to(dtype=trainer.dtype, device=trainer.device)

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
        prompt_embeds = prompt_embeds.to(dtype=trainer.dtype, device=trainer.device)

        _, seq_len, _ = prompt_embeds.shape

        # duplicate text embeddings and attention mask for each generation per prompt, using mps friendly method
        prompt_embeds = prompt_embeds.repeat(1, num_images_per_prompt, 1)
        prompt_embeds = prompt_embeds.view(
            batch_size * num_images_per_prompt, seq_len, -1
        )

        return prompt_embeds

    prompt_embed, pooled_prompt_embed = _clip_encode(
        trainer.text_encoders[0], input_ids_tuple[0]
    )
    prompt_embed2, pooled_prompt_embed2 = _clip_encode(
        trainer.text_encoders[1],
        input_ids_tuple[1],
        clip_skip=None,
    )
    clip_prompt_embeds = torch.cat([prompt_embed, prompt_embed2], dim=-1)

    t5_prompt_embed = _t5_encode(trainer.text_encoders[2], input_ids_tuple[2])

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

    if torch.rand(1).item() < trainer.config.caption_dropout_rate:
        prompt_embeds = trainer.empty_prompt_embeds
        pooled_prompt_embeds = trainer.empty_pooled_prompt_embeds

        model_kwargs["encoder_hidden_states"] = prompt_embeds
        model_kwargs["pooled_projections"] = pooled_prompt_embeds

    # add noise
    noise = torch.randn_like(img_latents)
    indices = torch.randint(
        0,
        trainer.scheduler.config.num_train_timesteps,
        (batch_size,),
        device=trainer.device,
    )
    schedule_timesteps_gpu = trainer.scheduler.timesteps.to(trainer.device)
    timesteps = schedule_timesteps_gpu[indices]
    noisy_latents = trainer.scheduler.scale_noise(
        sample=img_latents,
        timestep=timesteps,
        noise=noise,
    )

    return img_latents, noise, noisy_latents, timesteps, model_kwargs


def prepare_qwen_image_inputs(trainer, batch):
    model_kwargs = {}
    model_kwargs["return_dict"] = False
    model_kwargs["attention_kwargs"] = trainer.pipe.attention_kwargs

    batch_size = batch["pixel_values"].shape[0]
    pixel_values = batch["pixel_values"].to(dtype=trainer.vae.dtype)
    input_ids_tuple = batch["input_ids"]

    # image encode
    img_latents = retrieve_latents(
        trainer.vae.encode(pixel_values),
        generator=trainer.generator,
        sample_mode="argmax",
    )
    latents_mean = (
        torch.tensor(trainer.vae.config.latents_mean)
        .view(1, trainer.pipe.latent_channels, 1, 1, 1)
        .to(img_latents.device, img_latents.dtype)
    )
    latents_std = (
        torch.tensor(trainer.vae.config.latents_std)
        .view(1, trainer.pipe.latent_channels, 1, 1, 1)
        .to(img_latents.device, img_latents.dtype)
    )
    img_latents = (img_latents - latents_mean) / latents_std
    height = trainer.pipe.default_sample_size * trainer.pipe.vae_scale_factor
    width = trainer.pipe.default_sample_size * trainer.pipe.vae_scale_factor
    img_shapes = [
        [
            (
                1,
                height // trainer.pipe.vae_scale_factor // 2,
                width // trainer.pipe.vae_scale_factor // 2,
            )
        ]
    ] * batch_size
    model_kwargs["img_shapes"] = img_shapes

    # text encode
    def _extract_masked_hidden(hidden_states: torch.Tensor, mask: torch.Tensor):
        bool_mask = mask.bool()
        valid_lengths = bool_mask.sum(dim=1)
        selected = hidden_states[bool_mask]
        split_result = torch.split(selected, valid_lengths.tolist(), dim=0)

        return split_result

    attention_masks_tuple = batch["attention_mask"]
    drop_idx = trainer.pipe.prompt_template_encode_start_idx
    encoder_hidden_states = trainer.text_encoders[0](
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
    prompt_embeds = prompt_embeds.to(dtype=trainer.dtype, device=trainer.device)
    model_kwargs["encoder_hidden_states"] = prompt_embeds
    model_kwargs["encoder_hidden_states_mask"] = prompt_embeds_mask

    # add noise
    noise = torch.randn_like(img_latents)
    indices = torch.randint(
        0,
        trainer.scheduler.config.num_train_timesteps,
        (batch_size,),
        device=trainer.device,
    )
    schedule_timesteps_gpu = trainer.scheduler.timesteps.to(trainer.device)
    timesteps = schedule_timesteps_gpu[indices]
    noisy_latents = trainer.scheduler.scale_noise(
        sample=img_latents,
        timestep=timesteps,
        noise=noise,
    )

    return img_latents, noise, noisy_latents, timesteps, model_kwargs


def prepare_qwen_image_edit_inputs(trainer, batch):
    model_kwargs = {}
    model_kwargs["return_dict"] = False
    model_kwargs["attention_kwargs"] = trainer.pipe.attention_kwargs

    batch_size = batch["pixel_values"].shape[0]
    pixel_values = batch["pixel_values"].to(dtype=trainer.vae.dtype)

    # 1. Encode Target Image (Latents for Diffusion Loss)
    img_latents = retrieve_latents(
        trainer.vae.encode(pixel_values),
        generator=trainer.generator,
        sample_mode="argmax",
    )
    # Normalization as per Qwen-Image logic
    latents_mean = (
        torch.tensor(trainer.vae.config.latents_mean)
        .view(1, trainer.pipe.latent_channels, 1, 1, 1)
        .to(img_latents.device, img_latents.dtype)
    )
    latents_std = (
        torch.tensor(trainer.vae.config.latents_std)
        .view(1, trainer.pipe.latent_channels, 1, 1, 1)
        .to(img_latents.device, img_latents.dtype)
    )
    img_latents = (img_latents - latents_mean) / latents_std

    # Image Shapes metadata for RoPE
    height = trainer.pipe.default_sample_size * trainer.pipe.vae_scale_factor
    width = trainer.pipe.default_sample_size * trainer.pipe.vae_scale_factor
    img_shapes = [
        [
            (
                1,
                height // trainer.pipe.vae_scale_factor // 2,
                width // trainer.pipe.vae_scale_factor // 2,
            )
        ]
    ] * batch_size
    model_kwargs["img_shapes"] = img_shapes

    # 2. Encode Context (Text Instruction + Condition Image)
    # We need to construct prompt_embeds that include the visual tokens

    raw_prompts = batch.get(
        "raw_text", [""] * batch_size
    )  # From dataset if modified, else fallback
    if "raw_text" not in batch:
        # Fallback: decode input_ids if raw_text not available
        # This is slow, better to pass raw text from dataset
        input_ids = batch["input_ids"][0]
        raw_prompts = trainer.pipe.tokenizer.batch_decode(
            input_ids, skip_special_tokens=True
        )

    condition_pixel_values = batch.get("condition_pixel_values", None)

    # Use pipeline's encode_prompt to handle Qwen-VL complexity
    # Note: encode_prompt expects 'image' argument for condition image
    # We need to ensure condition_pixel_values is in correct format (likely Tensor BxCxHxW)

    # QwenImageEditPipeline.encode_prompt expects image tensor or list
    # If batch["condition_pixel_values"] is (B, C, H, W), we can pass it directly if pipeline supports batch
    # However, Qwen pipeline often processes one by one or expects specific list format for PIL

    # Hack: Qwen pipeline uses 'image' arg for condition image encoding
    if condition_pixel_values is not None:
        condition_pixel_values = condition_pixel_values.to(
            dtype=trainer.dtype, device=trainer.device
        )

        # We call encode_prompt.
        # IMPORTANT: This internally calls _get_qwen_prompt_embeds which runs the Qwen-VL text encoder
        prompt_embeds, prompt_embeds_mask = trainer.pipe.encode_prompt(
            prompt=raw_prompts,
            image=condition_pixel_values,  # This is the visual context!
            device=trainer.device,
            num_images_per_prompt=1,
        )
    else:
        # Text-only fallback (should not happen for editing task)
        prompt_embeds, prompt_embeds_mask = trainer.pipe.encode_prompt(
            prompt=raw_prompts, device=trainer.device, num_images_per_prompt=1
        )

    model_kwargs["encoder_hidden_states"] = prompt_embeds.to(dtype=trainer.dtype)
    model_kwargs["encoder_hidden_states_mask"] = prompt_embeds_mask

    # 3. Add Noise & Scheduler
    noise = torch.randn_like(img_latents)
    indices = torch.randint(
        0,
        trainer.scheduler.config.num_train_timesteps,
        (batch_size,),
        device=trainer.device,
    )
    schedule_timesteps_gpu = trainer.scheduler.timesteps.to(trainer.device)
    timesteps = schedule_timesteps_gpu[indices]

    # Prepare noisy latents
    # For Edit task, we might concatenate condition latents to noisy latents?
    # Qwen-Image-Edit pipeline: "latent_model_input = torch.cat([latents, image_latents], dim=1)"
    # WE MUST DO THIS for training too! The model expects 2x channels input.

    noisy_latents = trainer.scheduler.scale_noise(
        sample=img_latents,
        timestep=timesteps,
        noise=noise,
    )

    if condition_pixel_values is not None:
        # Encode condition image to latents (Visual Appearance Control)
        # This is separate from Qwen-VL visual tokens. This goes into UNet/DiT input.
        cond_latents = retrieve_latents(
            trainer.vae.encode(condition_pixel_values),
            generator=trainer.generator,
            sample_mode="argmax",
        )
        cond_latents = (cond_latents - latents_mean) / latents_std

        # Pack latents (Qwen specific packing)
        # Pipeline does: _pack_latents
        # We need to check if trainer.pipe has _pack_latents exposed or copy logic
        if hasattr(trainer.pipe, "_pack_latents"):
            # We need to pack both noisy_latents and cond_latents
            # But wait, input_preparer usually returns 'noisy_latents'
            # which trainer.py feeds to model(hidden_states=noisy_latents)
            # So we should concat HERE.

            # First, pack them individually?
            # Qwen pipeline:
            # latents = self._pack_latents(latents, ...)
            # image_latents = self._pack_latents(image_latents, ...)
            # latent_model_input = torch.cat([latents, image_latents], dim=1)

            # Let's pack first.
            # Check dimensions
            height_l, width_l = img_latents.shape[2], img_latents.shape[3]
            num_channels = img_latents.shape[1]

            noisy_latents_packed = trainer.pipe._pack_latents(
                noisy_latents, batch_size, num_channels, height_l, width_l
            )
            cond_latents_packed = trainer.pipe._pack_latents(
                cond_latents, batch_size, num_channels, height_l, width_l
            )

            # Concatenate for DiT input
            noisy_latents = torch.cat(
                [noisy_latents_packed, cond_latents_packed], dim=1
            )

            # Also update noise target?
            # Usually noise prediction is only on the first part (target image)
            # The model output will be same shape as input?
            # No, Qwen DiT output is only noise for the first part.
            # We don't need to change 'noise' variable shape,
            # trainer._calculate_main_loss uses noise_pred and noise.
            # We just need to make sure model() accepts the concatenated input.

        else:
            # Fallback if packing logic missing (should be there for Qwen pipeline)
            noisy_latents = torch.cat([noisy_latents, cond_latents], dim=1)

    return img_latents, noise, noisy_latents, timesteps, model_kwargs


INPUT_PREPARER_REGISTRY = {
    "flux-dev": prepare_flux_inputs,
    "pixart-sigma": prepare_pixart_sigma_inputs,
    "sd3": prepare_sd3_inputs,
    "qwen-image": prepare_qwen_image_inputs,
    "qwen-image-edit": prepare_qwen_image_edit_inputs,
}


def get_input_preparer(model_type):
    func = INPUT_PREPARER_REGISTRY.get(model_type)
    if func is None:
        raise NotImplementedError(
            f"_prepare_model_inputs not implemented for model_type: {model_type}"
        )
    return func
