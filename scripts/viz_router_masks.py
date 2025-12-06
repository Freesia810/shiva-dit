import os
import torch
import copy
import json
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
import numpy as np
from shiva_dit.training.config import TrainerConfig, LoraConfig
from shiva_dit.training.trainer import DiffusionModelTrainer

from shiva_dit.hook import HookManager
from shiva_dit.shiva_hook import ShivaDitBlockHook
from shiva_dit.utils import global_context

from scripts.config.shiva_router import hook_config, lora_config

FONT_PATH = "./Times New Roman.ttf"
custom_font_name = 'serif'

if os.path.exists(FONT_PATH):
    print(f"Found custom font: {FONT_PATH}")
    fm.fontManager.addfont(FONT_PATH)
    prop = fm.FontProperties(fname=FONT_PATH)
    custom_font_name = prop.get_name()
    print(f"Using font: {custom_font_name}")
else:
    print(f"Custom font not found at {FONT_PATH}. Using system serif.")
    plt.rcParams['font.serif'] = ['Times New Roman', 'Times', 'DejaVu Serif', 'serif']

plt.rcParams['font.family'] = custom_font_name
plt.rcParams['mathtext.fontset'] = 'custom'
plt.rcParams['mathtext.rm'] = custom_font_name
plt.rcParams['mathtext.it'] = f"{custom_font_name}:italic"
plt.rcParams['mathtext.bf'] = f"{custom_font_name}:bold"
plt.rcParams['axes.unicode_minus'] = False

mode = 'shared'
HOOK_CONFIG = copy.deepcopy(hook_config[mode])
HOOK_CONFIG["fixed_image_ratio"] = 0.6
HOOK_CONFIG["capture_mask_for_viz"] = True
HOOK_CONFIG["skip_layer_indices"] = [0]

IMAGE_PATH = "/home/admin/workspace/aop_lab/app_source/dataset/MJHQ-30K/indoor/45de4fb1e89b0d07d76b71bdcb6e6f02fabfdbb0.jpg"
PROMPT = "A jawdropping image of a bold red Ferrari SF90 Stradale, displayed inside a luxurious, highrise apartment in Tokyo, Japan, during a neonlit night. The cars striking design is accentuated by the citys dazzling lights, visible through the floortoceiling windows. This photograph embodies the fusion of automotive art and modern living, rendered in stunning 8K UHD resolution"
OUTPUT_DIR = "./viz_results"
CHECKPOINT_PATH = f"./checkpoints-{mode}/best_val_main_loss.pt"

TIMESTEPS_TO_VIZ = [900, 700, 500, 100]  
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

def plot_combined_grid(data_storage, row_labels, col_labels, save_path):
    rows = len(row_labels)
    cols = len(col_labels)
    
    fig, axes = plt.subplots(rows, cols, figsize=(10, 10), constrained_layout=True)
    
    cmap = 'viridis'
    vmin, vmax = 0.0, 1.0

    images = []

    for i, t_val in enumerate(row_labels):
        for j, layer_idx in enumerate(col_labels):
            ax = axes[i, j]
            
            key = (t_val, layer_idx)
            if key in data_storage:
                heatmap = data_storage[key]
                im = ax.imshow(heatmap, cmap=cmap, vmin=vmin, vmax=vmax, interpolation='nearest')
                if i == 0 and j == 0: images.append(im)
            else:
                ax.text(0.5, 0.5, "N/A", ha='center', va='center')
                ax.axis('off')
                continue

            ax.set_xticks([])
            ax.set_yticks([])
            # 如果想去掉子图的黑色边框，解开下面这行：
            # ax.axis('off') 
            
            if i == 0:
                ax.set_title(f"Layer {layer_idx}", fontsize=18, fontweight='bold', pad=8)
            
            if j == 0:
                t_display = int(t_val) if t_val > 1.0 else f"{t_val:.2f}"
                ax.set_ylabel(f"t ≈ {t_display}", fontsize=18, fontweight='bold', rotation=90, labelpad=8)

    cbar = fig.colorbar(images[0], ax=axes, orientation='horizontal', fraction=0.05, aspect=35, pad=0.02)
    cbar.set_label('Router Importance Score', fontsize=16, fontweight='bold')
    cbar.ax.tick_params(labelsize=12)
    cbar.outline.set_visible(False)

    os.makedirs(os.path.dirname(os.path.abspath(save_path)), exist_ok=True)
    plt.savefig(save_path, format='pdf', bbox_inches='tight', dpi=300)
    plt.close(fig)
    print(f"✅ Visualization saved to {save_path} (Layout: Constrained)")


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
    
    all_scheduler_timesteps = trainer.scheduler.timesteps.to(trainer.device)
    print(f"Scheduler has {len(all_scheduler_timesteps)} steps.")

    collected_maps = {}
    actual_timesteps_used = []

    print(f"\n>>> Starting Inference Loop...")

    for desired_t in TIMESTEPS_TO_VIZ:
        B = img_latents.shape[0]
        
        diffs = torch.abs(all_scheduler_timesteps - desired_t)
        nearest_idx = torch.argmin(diffs)
        actual_t_tensor = all_scheduler_timesteps[nearest_idx] 
        actual_timesteps_used.append(actual_t_tensor.item())
        
        print(f"  Target: {desired_t} -> Nearest: {actual_t_tensor.item():.4f}")

        target_t = actual_t_tensor.unsqueeze(0).repeat(B)

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
            global_context.update(current_timestep=target_t)
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

            if target_hook and hasattr(target_hook, "last_token_scores") and target_hook.last_token_scores is not None:
                token_scores = target_hook.last_token_scores
                if token_scores.dim() == 3:
                    token_scores = token_scores.squeeze(-1)
                
                try:
                    score_map = token_scores.reshape(B, grid_h, grid_w)
                    score_np = score_map[0].float().cpu().numpy() 
                    collected_maps[(actual_t_tensor.item(), layer_idx)] = score_np
                except Exception as e:
                    print(f"Reshape failed layer {layer_idx}: {e}")

                target_hook.last_token_scores = None

    print("\n>>> Generating Final PDF...")
    save_pdf_path = os.path.join(OUTPUT_DIR, "router_viz_grid.pdf")
    plot_combined_grid(collected_maps, actual_timesteps_used, LAYERS_TO_VIZ, save_pdf_path)

    if os.path.exists(TEMP_META_FILE):
        os.remove(TEMP_META_FILE)
    print("\nAll Done.")

if __name__ == "__main__":
    main()