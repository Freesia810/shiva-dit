"""Render Figure 4(b) from a learned Shiva retention schedule."""

import argparse
from pathlib import Path

def plot_refined_landscape(df, output, font=None):
    import matplotlib.pyplot as plt
    import matplotlib.gridspec as gridspec
    from matplotlib import font_manager
    import numpy as np

    if df.empty or not np.isfinite(df.to_numpy(dtype=float)).all():
        raise ValueError("Expected a non-empty, finite timestep-by-layer matrix")
    if font is not None:
        font_manager.fontManager.addfont(font)
    font_family = font_manager.FontProperties(fname=font).get_name() if font else "STIXGeneral"
    plt.rcParams.update({
        "font.family": font_family, "mathtext.fontset": "stix", "font.size": 12,
        "axes.linewidth": 0.6, "xtick.major.width": 0.6, "ytick.major.width": 0.6,
        "xtick.direction": "out", "ytick.direction": "out", "axes.unicode_minus": False,
    })
    output.parent.mkdir(parents=True, exist_ok=True)
    layer_means = df.mean(axis=0)
    time_means = df.mean(axis=1)

    vmin, vmax = 0.2, 1.0
    tm_min, tm_max = time_means.min(), time_means.max()
    tm_range = max(tm_max - tm_min, 0.01)
    x_line_min = max(0, tm_min - tm_range * 0.1)
    x_line_max = min(1.05, tm_max + tm_range * 0.1)

    fig = plt.figure(figsize=(9, 3.2))

    gs = gridspec.GridSpec(2, 2, figure=fig,
                           height_ratios=[0.9, 3.0],
                           width_ratios=[6.0, 1.2],
                           wspace=0.08, hspace=0.08)

    cmap_name = 'plasma'
    cmap = plt.get_cmap(cmap_name)
    norm = plt.Normalize(vmin=vmin, vmax=vmax)

    # ============================================================
    # (A) Top: Spatial Importance
    # ============================================================
    ax_top = fig.add_subplot(gs[0, 0])

    colors = cmap(norm(layer_means.values))
    ax_top.bar(np.arange(len(layer_means)) + 0.5, layer_means.values,
               color=colors, width=0.9, edgecolor='none')

    ax_top.set_xlim(0, df.shape[1])
    y_top_max = layer_means.max() * 1.1
    ax_top.set_ylim(0, y_top_max)

    ax_top.set_xticks([])
    ax_top.set_yticks([0.5, 1.0])
    ax_top.tick_params(axis='y', labelsize=9, length=2, pad=2)
    ax_top.set_ylabel("Spatial Ratio", fontsize=9, labelpad=5)

    ax_top.spines[["top", "right", "bottom"]].set_visible(False)
    ax_top.spines['left'].set_linewidth(0.6)
    ax_top.grid(axis='y', linestyle='--', color='gray', alpha=0.3, linewidth=0.5)

    # ============================================================
    # (B) Center: Heatmap
    # ============================================================
    ax_heat = fig.add_subplot(gs[1, 0])

    ax_heat.pcolormesh(df.to_numpy(), cmap=cmap, norm=norm, rasterized=True)
    ax_heat.set_ylim(len(df), 0)

    for _, spine in ax_heat.spines.items():
        spine.set_visible(True)
        spine.set_linewidth(0.6)
        spine.set_edgecolor('black')

    ax_heat.set_xlabel('Layer Depth', fontsize=11, labelpad=4)
    x_steps = 2
    ax_heat.set_xticks(np.arange(0, df.shape[1], x_steps) + 0.5)
    ax_heat.set_xticklabels(np.arange(1, df.shape[1] + 1, x_steps), fontsize=9, rotation=0)
    ax_heat.tick_params(axis='x', length=3, width=0.6)

    ax_heat.set_ylabel('Timestep $t/T$', fontsize=11, labelpad=4)
    total_steps = df.shape[0]
    ax_heat.set_yticks([0.5, total_steps/2, total_steps-0.5])
    ax_heat.set_yticklabels(['1.0', '0.5', '0.0'], fontsize=9, rotation=0)
    ax_heat.tick_params(axis='y', length=3, width=0.6)

    # ============================================================
    # (C) Right: Temporal Schedule
    # ============================================================
    ax_right = fig.add_subplot(gs[1, 1], sharey=ax_heat)
    y_pos = np.arange(len(time_means)) + 0.5

    ax_right.plot(time_means.values, y_pos, color='#404040', linewidth=1.5)
    ax_right.fill_betweenx(y_pos, x_line_min, time_means.values, color='gray', alpha=0.15)

    ax_right.set_xlim(x_line_min, x_line_max)

    ax_right.locator_params(axis='x', nbins=3)
    ax_right.tick_params(axis='x', labelsize=8, length=2, width=0.6)
    ax_right.set_xlabel("Temporal Ratio", fontsize=9, labelpad=5)

    plt.setp(ax_right.get_yticklabels(), visible=False)
    ax_right.tick_params(axis='y', length=0)

    ax_right.spines[["top", "right", "left"]].set_visible(False)
    ax_right.spines['bottom'].set_linewidth(0.6)
    ax_right.grid(axis='x', linestyle=':', color='gray', alpha=0.5, linewidth=0.5)

    # ============================================================
    # (D) Colorbar (Embedded)
    # ============================================================
    ax_cbar_box = fig.add_subplot(gs[0, 1])
    ax_cbar_box.axis('off')
    ax_inset = ax_cbar_box.inset_axes([0.1, 0.2, 0.8, 0.15])

    cbar = plt.colorbar(plt.cm.ScalarMappable(norm=norm, cmap=cmap), cax=ax_inset, orientation='horizontal')
    cbar.set_ticks([vmin, vmax])
    cbar.set_ticklabels([f'{vmin}', f'{vmax}'])
    cbar.ax.tick_params(labelsize=9, length=2, width=0.6, pad=1)
    cbar.outline.set_linewidth(0.6)

    ax_cbar_box.text(0.5, 0.55, 'Keep Ratio', transform=ax_cbar_box.transAxes,
                     ha='center', va='bottom', fontsize=9)

    # ============================================================
    # Save
    # ============================================================
    fig.savefig(output.with_suffix(".pdf"), dpi=600, bbox_inches='tight', pad_inches=0.02)
    fig.savefig(output.with_suffix(".png"), dpi=600, bbox_inches='tight', pad_inches=0.02)

    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--csv", type=Path, help="Headerless matrix ordered from low to high timestep")
    source.add_argument("--lut", type=Path, help="LUT exported by benchmark.lut")
    parser.add_argument("--output", type=Path, default=Path("runs/figures/fig4b"))
    parser.add_argument("--font", type=Path)
    args = parser.parse_args()
    if args.lut:
        import torch

        table = torch.load(args.lut, map_location="cpu", weights_only=True)
        matrix = table["ratio"].float().T.numpy()

        import pandas as pd

        df = pd.DataFrame(matrix)
    else:
        import pandas as pd

        df = pd.read_csv(args.csv, header=None).iloc[::-1].reset_index(drop=True)
    plot_refined_landscape(df, args.output, args.font)


if __name__ == "__main__":
    main()
