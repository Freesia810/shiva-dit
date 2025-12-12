import os
import torch
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
import copy

from shiva_dit.training import DiffusionModelTrainer, TrainerConfig
from shiva_dit.hook import HookManager

try:
    from scripts.config.shiva_ratio_policy import trainer_config, hook_config, lora_config
except ImportError:
    print("Warning: Could not import config from scripts.config.shiva_ratio_policy.")
    exit(1)

CHECKPOINT_PATH = "./checkpoints_adaptive/checkpoint_step_000872" 
OUTPUT_DIR = "./analysis_results"
NUM_VAL_IMAGES = 64        
NUM_TIME_STEPS = 20
BATCH_SIZE = 1             

def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    print(">>> Configuring Analyzer...")
    
    analyze_config = copy.deepcopy(trainer_config)
    analyze_config["metadata_file"] = analyze_config.get("val_metadata_file", analyze_config["metadata_file"])
    analyze_config["image_dir"] = analyze_config.get("val_image_dir", analyze_config["image_dir"])
    analyze_config["train_batch_size"] = BATCH_SIZE
    analyze_config["run_validation"] = False 
    analyze_config["log_with_tensorboard"] = False
    
    analyze_hook_config = copy.deepcopy(hook_config)
    analyze_hook_config["capture_mask_for_viz"] = True
    analyze_hook_config["shiva_mode"] = "image"
    
    config = TrainerConfig(**analyze_config)
    trainer = DiffusionModelTrainer(config=config, load_evaluators=False)
    
    trainer.load_from_checkpoint(CHECKPOINT_PATH)

    trainer.model.eval()
    
    print(f">>> Preparing Dataloader from {analyze_config['metadata_file']}...")
    dataloader = trainer.train_dataloader 
    
    hooks = HookManager.get_all_hooks()
    for h in hooks.values():
        h.image_ratio_policy = None
        h.fixed_image_ratio = 1.0
    layer_indices = sorted([h.layer_idx for h in hooks.values()])
    num_layers = max(layer_indices) + 1 if layer_indices else 28
    print(f">>> Detected Layers: {layer_indices}")
    
    total_scheduler_steps = len(trainer.scheduler.timesteps)
    indices_grid = torch.linspace(0, total_scheduler_steps - 1, NUM_TIME_STEPS).long().to(trainer.device)
    
    all_real_timesteps = trainer.schedule_timesteps_gpu 
    
    heatmap_sum = np.zeros((NUM_TIME_STEPS, num_layers))
    heatmap_count = np.zeros((NUM_TIME_STEPS, num_layers))
    
    print(">>> Starting Analysis Loop...")
    
    count_images = 0
    total_batches = NUM_VAL_IMAGES // BATCH_SIZE
    
    with torch.no_grad():
        for step, batch in enumerate(tqdm(dataloader, total=total_batches)):
            if count_images >= NUM_VAL_IMAGES:
                break
            
            B = batch["pixel_values"].shape[0]
            count_images += B

            for t_seq_idx, t_index_val in enumerate(indices_grid):
                t_index_tensor = t_index_val.unsqueeze(0).repeat(B)
                
                img_latents, noise, noisy_latents, timesteps, model_kwargs = (
                    trainer.prepare_model_inputs(batch, timestep_index=t_index_tensor)
                )

                _ = trainer.model(
                    hidden_states=noisy_latents,
                    timestep=timesteps,
                    **model_kwargs
                )
                
                current_hooks = HookManager.get_all_hooks()
                for h in current_hooks.values():
                    if hasattr(h, "last_token_scores") and h.last_token_scores is not None:
                        raw_scores = h.last_token_scores
                        probs = torch.sigmoid(raw_scores)
                        avg_score = probs.mean().item()
                        
                        if h.layer_idx < num_layers:
                            heatmap_sum[t_seq_idx, h.layer_idx] += avg_score * B
                            heatmap_count[t_seq_idx, h.layer_idx] += B
                        
                        h.last_token_scores = None

    print(">>> Plotting Results...")
    
    heatmap_count[heatmap_count == 0] = 1e-6
    final_heatmap = heatmap_sum / heatmap_count # [T, L]

    plt.figure(figsize=(12, 8))
    ax = sns.heatmap(
        final_heatmap, 
        cmap='viridis', 
        vmin=0.0, vmax=1.0, 
        annot=False,
        cbar_kws={'label': 'Average Token Importance Score'}
    )
    
    sampled_real_timesteps = all_real_timesteps[indices_grid.cpu()].cpu().numpy()
    
    ax.set_yticks(np.arange(NUM_TIME_STEPS) + 0.5)
    ax.set_yticklabels([str(int(t)) for t in sampled_real_timesteps], rotation=0)
    
    ax.set_ylabel("Diffusion Timestep")
    ax.set_xlabel("Transformer Layer Index")
    
    title_str = f"Router Score Landscape (n={NUM_VAL_IMAGES} images)\n"
    title_str += "Y-axis: Real Timesteps sampled from Scheduler"
    ax.set_title(title_str)
    
    save_path_png = os.path.join(OUTPUT_DIR, "router_score_landscape.png")
    plt.savefig(save_path_png, dpi=300, bbox_inches='tight')
    
    print(f"✅ Analysis Complete! Saved to {save_path_png}")

if __name__ == "__main__":
    main()