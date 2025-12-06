import os
import torch
import copy
import json
import matplotlib.pyplot as plt
from shiva_dit.training.config import TrainerConfig, LoraConfig
from shiva_dit.training.trainer import DiffusionModelTrainer

from shiva_dit.hook import HookManager
from shiva_dit.shiva_hook import ShivaDitBlockHook
from shiva_dit.utils import global_context

from scripts.config.shiva_router import hook_config, lora_config

mode = 'shared'

HOOK_CONFIG = copy.deepcopy(hook_config[mode])

HOOK_CONFIG["fixed_image_ratio"] = 0.6
HOOK_CONFIG["capture_mask_for_viz"] = True
HOOK_CONFIG["skip_layer_indices"] = [0]

IMAGE_PATH = "/home/admin/workspace/aop_lab/app_source/dataset/MJHQ-30K/indoor/45de4fb1e89b0d07d76b71bdcb6e6f02fabfdbb0.jpg"
PROMPT = "A jawdropping image of a bold red Ferrari SF90 Stradale, displayed inside a luxurious, highrise apartment in Tokyo, Japan, during a neonlit night. The cars striking design is accentuated by the citys dazzling lights, visible through the floortoceiling windows. This photograph embodies the fusion of automotive art and modern living, rendered in stunning 8K UHD resolution"
OUTPUT_DIR = "./viz_results"
CHECKPOINT_PATH = f"./checkpoints-{mode}/best_val_main_loss.pt"

TIMESTEPS_TO_VIZ = [800, 700, 500, 300]  # t=0.9, 0.5, 0.1
LAYERS_TO_VIZ = [1, 8, 15, 23]
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
    try:
        import torchvision
        grid_img_tensor = torchvision.utils.make_grid(
            score_map, nrow=score_map.shape[0], normalize=False, pad_value=0.5
        )
    except ImportError:
        raise Exception("torchvision not found.")

    grid_np = grid_img_tensor.permute(1, 2, 0).detach().cpu().numpy()

    aspect_ratio = grid_np.shape[1] / grid_np.shape[0] # W / H
    fig_height = 4
    fig_width = fig_height * aspect_ratio
    fig, ax = plt.subplots(figsize=(fig_width, fig_height))

    im = ax.imshow(grid_np[..., 0], cmap='viridis', vmin=0, vmax=1)

    if title:
        ax.set_title(title, fontsize=14)
    ax.axis('off')

    plt.subplots_adjust(top=1, bottom=0, right=1, left=0, hspace=0, wspace=0)
    plt.margins(0, 0)

    os.makedirs(os.path.dirname(os.path.abspath(save_path)), exist_ok=True)
    plt.savefig(save_path, format='pdf', bbox_inches='tight', pad_inches=0)
    plt.close(fig)
    
    print(f"Saved vector heatmap to {save_path}")


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
        "lora_config": LoraConfig(**lora_config),
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

        indices = torch.full((B,), t_val, device=device, dtype=dtype)
        indices = torch.randint(
            0,
            trainer.scheduler.config.num_train_timesteps,
            (B,),
            device=trainer.device,
        )
        schedule_timesteps_gpu = trainer.scheduler.timesteps.to(trainer.device)
        target_t = schedule_timesteps_gpu[indices]

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
                token_scores = target_hook.last_token_scores

                try:
                    import matplotlib.cm
                except ImportError:
                    print("matplotlib not found.")
                    return None

                if token_scores.dim() == 3:
                    token_scores = token_scores.squeeze(-1)
                
                B, total_tokens = token_scores.shape
                
                try:
                    score_map = token_scores.reshape(B, grid_h, grid_w)
                except Exception as e:
                    print(f"Reshape failed: {e}")
                    return None

                score_np = score_map.float().cpu().numpy() 
                
                cm = matplotlib.cm.get_cmap('jet') 
                colored_np = cm(score_np) # (B, H, W, 4)
                rgb_np = colored_np[..., :3] # (B, H, W, 3)

                # (B, H, W, 3) -> (B, 3, H, W)
                final_tensor = torch.from_numpy(rgb_np).permute(0, 3, 1, 2).float()

                save_name = os.path.join(
                    OUTPUT_DIR, f"mask_L{layer_idx:02d}_t{t_val:03d}.pdf"
                )
                plot_heatmap_vector(final_tensor, save_name, title=f"L{layer_idx} @ t={t_val}")
                print(f"  Saved {save_name}")

                target_hook.last_token_scores = None

    if os.path.exists(TEMP_META_FILE):
        os.remove(TEMP_META_FILE)
    print("\nAll Done.")


if __name__ == "__main__":
    main()
