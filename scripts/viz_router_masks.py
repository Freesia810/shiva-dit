import os
import torch
import copy
import json
import matplotlib.pyplot as plt
from shiva_dit.training.config import TrainerConfig
from shiva_dit.training.trainer import DiffusionModelTrainer
from torchvision import transforms

from shiva_dit.hook import HookManager
from shiva_dit.shiva_hook import ShivaDitBlockHook
from shiva_dit.utils import global_context

from scripts.config.shiva_router import hook_config_shared, lora_config

HOOK_CONFIG = copy.deepcopy(hook_config_shared)

HOOK_CONFIG["fixed_image_ratio"] = 1.0
HOOK_CONFIG["capture_mask_for_viz"] = True
HOOK_CONFIG["skip_layer_indices"] = [0]

IMAGE_PATH = "/path/to/your/dataset/MJHQ-30K/image_00001.jpg"
PROMPT = "A high quality photo of a cat"
OUTPUT_DIR = "./viz_results"
CHECKPOINT_PATH = "./checkpoints-shared/final.pt"

TIMESTEPS_TO_VIZ = [900, 500, 100]  # t=0.9, 0.5, 0.1
LAYERS_TO_VIZ = [2, 12, 22]
TEMP_META_FILE = "./viz_temp_meta.jsonl"


def create_temp_metadata():
    abs_image_path = os.path.abspath(IMAGE_PATH)
    image_dir = os.path.dirname(abs_image_path)
    data = {"file_name": os.path.basename(abs_image_path), "text": PROMPT}
    with open(TEMP_META_FILE, "w") as f:
        f.write(json.dumps(data) + "\n")
    return image_dir


def load_weights(trainer, ckpt_path):
    print(f"Loading weights from {ckpt_path}...")
    state_dict = torch.load(ckpt_path, map_location=trainer.device)
    if "model" in state_dict:
        state_dict = state_dict["model"]
    new_sd = {k.replace("module.", ""): v for k, v in state_dict.items()}
    trainer.model.load_state_dict(new_sd, strict=False)
    print("Weights loaded.")


def plot_heatmap_vector(score_map, save_path, title=None):
    fig, ax = plt.subplots(figsize=(4, 4))
    ax.imshow(score_map, cmap="viridis", vmin=0, vmax=1)
    if title:
        ax.set_title(title, fontsize=12)
    ax.axis("off")
    plt.subplots_adjust(0, 0, 1, 1, 0, 0)
    plt.savefig(save_path, format="pdf", bbox_inches="tight", pad_inches=0)
    plt.close(fig)


def main():
    image_dir = create_temp_metadata()

    trainer_config = {
        "model_type": "sd3",
        "model_path": None,
        "output_dir": OUTPUT_DIR,
        "training_mode": "lora",
        "mixed_precision": "bf16",
        "gradient_accumulation_steps": 1,
        "train_batch_size": 1,
        "seed": 42,
        "log_with_tensorboard": False,
        "metadata_file": TEMP_META_FILE,
        "image_dir": image_dir,
        "dataloader_num_workers": 0,
        "lora_config": lora_config,
        "trainable_hook_config": (ShivaDitBlockHook, HOOK_CONFIG),
    }

    print(">>> Initializing Trainer...")
    config = TrainerConfig(**trainer_config)
    trainer = DiffusionModelTrainer(config)

    if os.path.exists(CHECKPOINT_PATH):
        load_weights(trainer, CHECKPOINT_PATH)

    dataloader = trainer.train_dataloader
    batch = next(iter(dataloader))

    print(">>> Preparing Model Inputs...")
    with torch.no_grad():
        img_latents, noise, _, _, model_kwargs = trainer._prepare_model_inputs(batch)

    if config.model_type == "sd3":
        H, W = img_latents.shape[-2], img_latents.shape[-1]
        grid_h, grid_w = H // 2, W // 2
    else:
        grid_h = int(img_latents.shape[1] ** 0.5)
        grid_w = grid_h

    trainer.model.eval()
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    for t_val in TIMESTEPS_TO_VIZ:
        print(f"\nVisualizing t={t_val}...")

        B = img_latents.shape[0]
        dtype = trainer.dtype
        device = trainer.device
        target_t = torch.full((B,), t_val, device=device, dtype=dtype)

        if config.model_type in ["flux-dev", "sd3", "qwen-image-edit"]:
            noisy_latents_input = trainer.scheduler.scale_noise(
                sample=img_latents,
                timestep=target_t,
                noise=noise,
            )
        else:
            noisy_latents_input = trainer.scheduler.add_noise(
                original_samples=img_latents,
                noise=noise,
                timesteps=target_t,
            )

        if config.model_type in ["flux-dev", "qwen-image-edit"]:
            global_context.update(current_timestep=(target_t * 1000).to(device))
            model_t_input = target_t / 1000.0
        else:
            global_context.update(current_timestep=target_t)
            model_t_input = target_t

        with torch.no_grad():
            trainer.model(
                hidden_states=noisy_latents_input,
                timestep=model_t_input,
                **model_kwargs,
            )

        hooks = HookManager.get_all_hooks()
        for layer_idx in LAYERS_TO_VIZ:
            target_hook = None
            for h in hooks.values():
                if h.layer_idx == layer_idx:
                    target_hook = h
                    break

            if (
                target_hook
                and hasattr(target_hook, "last_token_scores")
                and target_hook.last_token_scores is not None
            ):
                scores = target_hook.last_token_scores
                s_map = scores[0].float().cpu().numpy().reshape(grid_h, grid_w)

                save_name = os.path.join(
                    OUTPUT_DIR, f"mask_L{layer_idx:02d}_t{t_val:03d}.pdf"
                )
                plot_heatmap_vector(s_map, save_name, title=f"L{layer_idx} @ t={t_val}")
                print(f"  Saved {save_name}")

                target_hook.last_token_scores = None

    if os.path.exists(TEMP_META_FILE):
        os.remove(TEMP_META_FILE)
    print("\nAll Done.")


if __name__ == "__main__":
    main()
