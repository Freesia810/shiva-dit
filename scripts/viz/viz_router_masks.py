"""Render the paper's router heatmaps and score distributions (Figures 6 and 7)."""

import argparse
import json
import os
from pathlib import Path
from tempfile import TemporaryDirectory

import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
import matplotlib.gridspec as gridspec
import numpy as np


def plot_histogram_grid(data_storage, row_labels, col_labels, save_path, font=None):
    plt.rcParams['mathtext.fontset'] = 'stix'
    plt.rcParams['font.family'] = 'serif'

    if font is not None:
        prop_regular = fm.FontProperties(fname=font)
        prop_bold = fm.FontProperties(fname=font, weight='bold')
    else:
        prop_regular = fm.FontProperties(family='serif')
        prop_bold = fm.FontProperties(family='serif', weight='bold')

    rows = len(row_labels)
    n_layers = len(col_labels)

    fig, axes = plt.subplots(nrows=rows, ncols=n_layers,
                             figsize=(3 * n_layers, 2.0 * rows),
                             constrained_layout=True, squeeze=False)

    hist_color = '#4682B4'
    mean_color = '#D62728'


    for r, t_val in enumerate(row_labels):
        for c, layer_idx in enumerate(col_labels):
            ax = axes[r, c]

            mask_key = (t_val, layer_idx)
            if mask_key in data_storage:
                score_data = data_storage[mask_key].flatten()
                mean_val = np.mean(score_data)

                ax.hist(score_data, bins=40, range=(0, 1), density=True,
                        color=hist_color, alpha=0.8, edgecolor='none')

                ax.axvline(mean_val, color=mean_color, linestyle='--', linewidth=1.5)

                ax.text(0.95, 0.90, rf"$\mu$={mean_val:.2f}",
                        transform=ax.transAxes,
                        fontsize=12,
                        color=mean_color,
                        fontproperties=prop_bold,
                        ha='right', va='top',
                        bbox=dict(boxstyle="round,pad=0.2", fc="white", ec="none", alpha=0.7)
                        )
            else:
                ax.text(0.5, 0.5, "N/A", ha='center', va='center', fontproperties=prop_regular)

            ax.set_xlim(0, 1)
            ax.spines['top'].set_visible(False)
            ax.spines['right'].set_visible(False)
            ax.grid(True, axis='y', linestyle=':', alpha=0.4)

            if r == 0:
                ax.set_title(f"Layer {layer_idx}", fontsize=14, pad=10, fontproperties=prop_bold)

            if c == 0:
                t_display = int(t_val) if t_val > 1.0 else f"{t_val:.2f}"
                ax.set_ylabel(f"t ≈ {t_display}\nDensity", fontsize=12, fontproperties=prop_bold)
                for label in ax.get_yticklabels():
                    label.set_fontproperties(prop_regular)
            else:
                ax.set_yticklabels([])

            if r == rows - 1:
                ax.set_xlabel("Score", fontsize=12, fontproperties=prop_regular)
                for label in ax.get_xticklabels():
                    label.set_fontproperties(prop_regular)
            else:
                ax.set_xticklabels([])

    os.makedirs(os.path.dirname(os.path.abspath(save_path)), exist_ok=True)
    plt.savefig(save_path, format='pdf', bbox_inches='tight', dpi=300)
    plt.close(fig)
    print(f"Saved histogram to {save_path}")

def plot_paper_grid(data_storage, row_labels, col_labels, save_path):
    rows = len(row_labels)
    n_layers = len(col_labels)

    fig = plt.figure(figsize=(11, 2.3 * rows + 1), constrained_layout=True)
    width_ratios = [1, 0.15] + [1] * n_layers
    gs_main = gridspec.GridSpec(rows + 1, 2 + n_layers, figure=fig,
                                width_ratios=width_ratios,
                                height_ratios=[1.0] * rows + [0.08],
                                hspace=0.05, wspace=0.02)

    cmap = 'inferno'
    vmin, vmax = 0.0, 1.0
    mappable = plt.cm.ScalarMappable(cmap=cmap, norm=plt.Normalize(vmin, vmax))

    for row_idx, t_val in enumerate(row_labels):
        ax_img = fig.add_subplot(gs_main[row_idx, 0])
        img_key = (t_val, 'input')
        if img_key in data_storage:
            img_data = data_storage[img_key]
            ax_img.imshow(img_data, aspect='equal')
        else:
            ax_img.text(0.5, 0.5, "N/A", ha='center', va='center')

        ax_img.set_xticks([])
        ax_img.set_yticks([])
        t_display = int(t_val) if t_val > 1.0 else f"{t_val:.2f}"
        ax_img.text(-0.1, 0.5, f"t ≈ {t_display}", fontsize=16, fontweight='bold',
                    rotation=90, ha='right', va='center', transform=ax_img.transAxes)

        if row_idx == 0:
            ax_img.set_title("Input", fontsize=16, fontweight='bold', pad=10)

        for i, layer_idx in enumerate(col_labels):
            grid_col = 2 + i
            ax_mask = fig.add_subplot(gs_main[row_idx, grid_col])

            mask_key = (t_val, layer_idx)
            if mask_key in data_storage:
                heatmap = data_storage[mask_key]
                ax_mask.imshow(heatmap, cmap=cmap, vmin=vmin, vmax=vmax,
                                    interpolation='nearest', aspect='equal')
            else:
                ax_mask.axis('off')
                continue

            ax_mask.set_xticks([])
            ax_mask.set_yticks([])
            if row_idx == 0:
                ax_mask.set_title(f"Layer {layer_idx}", fontsize=16, fontweight='bold', pad=10)

    cbar_ax = fig.add_subplot(gs_main[rows, :])
    cbar = fig.colorbar(mappable, cax=cbar_ax, orientation='horizontal', fraction=0.5)
    cbar.set_label('Router Importance Score', fontsize=14, fontweight='bold', labelpad=5)
    cbar.ax.tick_params(labelsize=12)
    cbar.outline.set_visible(False)
    cbar_ax.set_aspect(0.03)

    os.makedirs(os.path.dirname(os.path.abspath(save_path)), exist_ok=True)
    plt.savefig(save_path, format='pdf', bbox_inches='tight', dpi=300)
    plt.close(fig)
    print(f"Saved score maps to {save_path}")

def collect_maps(trainer, config, timesteps, layers):
    import torch

    from shiva_dit.hook import HookManager

    dataloader = trainer.train_dataloader
    batch = next(iter(dataloader))

    with torch.no_grad():
        img_latents, _, _, _, _ = trainer.prepare_model_inputs(batch, timestep_index=0)

    if config.model_type == "sd3" or config.model_type == "pixart-sigma":
        H, W = img_latents.shape[-2], img_latents.shape[-1]
        grid_h, grid_w = H // 2, W // 2
    elif config.model_type == "flux-dev":
        H = W = int(img_latents.shape[-2] ** 0.5)
        grid_h, grid_w = H, W

    trainer.model.eval()

    all_scheduler_timesteps = trainer.scheduler.timesteps.to(trainer.device)
    print(f"Scheduler has {len(all_scheduler_timesteps)} steps.")

    collected_maps = {}
    actual_timesteps_used = []

    print(f"\n>>> Starting Inference Loop...")

    hooks = {hook.layer_idx: hook for hook in HookManager.get_all_hooks().values()}
    missing_layers = set(layers) - hooks.keys()
    if missing_layers:
        raise ValueError(f"No Shiva hooks for layer indices {sorted(missing_layers)}")

    for desired_t in timesteps:
        B = img_latents.shape[0]

        diffs = torch.abs(all_scheduler_timesteps - desired_t)
        nearest_idx = torch.argmin(diffs)

        actual_t_val = all_scheduler_timesteps[nearest_idx].item()
        actual_timesteps_used.append(actual_t_val)

        print(f"  Target: {desired_t} -> Nearest: {actual_t_val:.4f} (Index: {nearest_idx})")

        t_idx_tensor = nearest_idx.reshape(1).expand(B)

        with torch.no_grad():
            _, _, noisy_latents, model_timesteps, model_kwargs = trainer.prepare_model_inputs(
                batch, timestep_index=t_idx_tensor
            )

        with torch.no_grad():
            if config.model_type == "sd3":
                latents_for_viz = (noisy_latents / trainer.vae.config.scaling_factor) + trainer.vae.config.shift_factor
            elif config.model_type == "flux-dev":
                latents_for_viz = trainer.pipe._unpack_latents(noisy_latents, batch['pixel_values'].shape[2], batch['pixel_values'].shape[3], trainer.pipe.vae_scale_factor)
                latents_for_viz = (latents_for_viz / trainer.vae.config.scaling_factor) + trainer.vae.config.shift_factor
            else:
                latents_for_viz = noisy_latents / trainer.vae.config.scaling_factor

            decoded_img = trainer.vae.decode(latents_for_viz, return_dict=False)[0]
            decoded_img = (decoded_img / 2 + 0.5).clamp(0, 1)
            img_np = decoded_img[0].permute(1, 2, 0).cpu().float().numpy()

            collected_maps[(actual_t_val, 'input')] = img_np

        with torch.no_grad():
            trainer.model(
                hidden_states=noisy_latents,
                timestep=model_timesteps,
                **model_kwargs,
            )

        for layer_idx in layers:
            hook = hooks[layer_idx]
            token_scores = getattr(hook, "last_token_scores", None)
            if token_scores is None:
                raise RuntimeError(f"Layer {layer_idx} did not record router scores")
            score_map = token_scores.reshape(B, grid_h, grid_w)
            collected_maps[(actual_t_val, layer_idx)] = score_map[0].float().cpu().numpy()
            hook.last_token_scores = None

    return collected_maps, actual_timesteps_used


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--base-model-path", required=True)
    parser.add_argument("--image", type=Path, required=True)
    parser.add_argument("--prompt", required=True)
    parser.add_argument("--timesteps", type=int, nargs="+", default=[900, 700, 500, 100])
    parser.add_argument("--layers", type=int, nargs="+", default=[1, 8, 15, 23])
    parser.add_argument("--output-dir", type=Path, default=Path("runs/figures/router"))
    parser.add_argument("--font", type=Path)
    args = parser.parse_args()

    import torch

    from shiva_dit.shiva_hook import ShivaDitBlockHook
    from shiva_dit.training import DiffusionModelTrainer, LoraConfig, TrainerConfig

    checkpoint = args.checkpoint.expanduser().resolve(strict=True)
    image = args.image.expanduser().resolve(strict=True)
    if checkpoint.is_dir():
        meta = json.loads((checkpoint / "shiva_config.json").read_text())
    else:
        meta = torch.load(checkpoint, map_location="cpu", weights_only=True)
    hook_info = meta["hook_info"]
    if hook_info["class_name"] != "ShivaDitBlockHook":
        raise ValueError("Expected a Shiva checkpoint")
    if meta["model_type"] not in {"sd3", "flux-dev", "pixart-sigma"}:
        raise ValueError("Router maps support SD3, FLUX.1-dev, and PixArt-Sigma")
    trainer_settings = {
        "model_type": meta["model_type"],
        "model_path": args.base_model_path,
        "training_mode": meta["training_mode"],
        "mixed_precision": "bf16",
        "gradient_accumulation_steps": 1,
        "train_batch_size": 1,
        "seed": 42,
        "log_with_tensorboard": False,
        "image_dir": str(image.parent),
        "trainable_hook_config": (ShivaDitBlockHook, dict(hook_info["config"])),
    }
    if meta.get("lora_info"):
        trainer_settings["lora_config"] = LoraConfig(**meta["lora_info"])
    del meta

    with TemporaryDirectory(prefix="shiva-router-figure-") as workspace:
        metadata = Path(workspace) / "input.jsonl"
        metadata.write_text(json.dumps({"file_name": image.name, "text": args.prompt}) + "\n",
                            encoding="utf-8")
        config = TrainerConfig(**trainer_settings, metadata_file=str(metadata), output_dir=workspace)
        trainer = DiffusionModelTrainer(config, load_evaluators=False)
        trainer.load_from_checkpoint(str(checkpoint))
        maps, timesteps = collect_maps(trainer, config, args.timesteps, args.layers)

    if args.font is not None:
        fm.fontManager.addfont(args.font)
    family = fm.FontProperties(fname=args.font).get_name() if args.font else "STIXGeneral"
    with plt.rc_context({"font.family": family, "mathtext.fontset": "stix"}):
        plot_paper_grid(maps, timesteps, args.layers, args.output_dir / "router_viz_grid.pdf")
        plot_histogram_grid(maps, timesteps, args.layers, args.output_dir / "router_hist_grid.pdf", args.font)


if __name__ == "__main__":
    main()
