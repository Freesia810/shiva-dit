import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import seaborn as sns
import os
from matplotlib import font_manager

# ============================================================
# 1. Font Configuration
# ============================================================
font_path = 'Times New Roman.ttf'
if os.path.exists(font_path):
    font_manager.fontManager.addfont(font_path)
    prop = font_manager.FontProperties(fname=font_path)
    plt.rcParams['font.family'] = prop.get_name()
    plt.rcParams['font.serif'] = [prop.get_name()]
else:
    plt.rcParams['font.family'] = 'serif'
    plt.rcParams['font.serif'] = ['Times New Roman']

plt.rcParams['axes.unicode_minus'] = False
plt.rcParams['mathtext.fontset'] = 'stix'


# ============================================================
# 2. Main Function
# ============================================================
def analyze_and_plot_advanced(csv_path, output_name="ratio_analysis_advanced"):

    if not os.path.exists(csv_path):
        print(f"Error: {csv_path} not found.")
        return
    
    df = pd.read_csv(csv_path, header=None)

    # Layer Means
    layer_means = df.mean(axis=0)
    idx_core = layer_means.idxmax()
    idx_detail = 0  

    # Temporal deviation
    df_deviation = df - df.mean(axis=0)

    # Lines for C panel
    line_core = df_deviation.iloc[:, idx_core]
    line_detail = df_deviation.iloc[:, idx_detail]
    line_mean = df.mean(axis=1) - df.mean(axis=1).mean()

    # ============================================================
    # 3. Figure Canvas
    # ============================================================
    fig = plt.figure(figsize=(12, 5.5), constrained_layout=True)

    # 调整 a 和 b 的相对高度
    gs = gridspec.GridSpec(
        2, 2, figure=fig,
        width_ratios=[3.4, 1.3],
        height_ratios=[2.0, 2.0]   # 🔥 a 更大, b 更小
    )

    # ------------------------------------------------------------
    # (a) Structural Baseline
    # ------------------------------------------------------------
    ax_bar = fig.add_subplot(gs[0, 0])

    norm = plt.Normalize(0, 1)
    colors = plt.cm.magma(norm(layer_means.values))

    ax_bar.bar(layer_means.index + 0.5, layer_means.values,
               color=colors, edgecolor='none', width=0.8)

    ax_bar.set_xlim(0, df.shape[1])
    ax_bar.set_ylim(0, 1.05)
    ax_bar.set_ylabel('Avg Ratio', fontsize=10)
    ax_bar.set_title('(a) Structural Baseline (Layer Importance)', fontsize=11, fontweight='bold', loc='left')
    ax_bar.grid(axis='y', linestyle=':', alpha=0.4)
    plt.setp(ax_bar.get_xticklabels(), visible=False)

    # ------------------------------------------------------------
    # (b) Heatmap – High Contrast “Seismic”
    # ------------------------------------------------------------
    ax_heat = fig.add_subplot(gs[1, 0], sharex=ax_bar)

    limit = max(abs(df_deviation.min().min()), abs(df_deviation.max().max()))

    sns.heatmap(
        df_deviation, cmap='seismic', center=0,
        vmin=-limit, vmax=limit,
        ax=ax_heat,
        cbar_kws={'label': 'Deviation from Mean ($\\Delta$ Ratio)', 'orientation': 'horizontal', 'pad': 0.18}
    )

    ax_heat.set_xlabel('Transformer Layer Depth', fontsize=11)
    ax_heat.set_ylabel('Timestep $t$', fontsize=11)

    ax_heat.set_xticks(np.arange(0, df.shape[1], 2) + 0.5)
    ax_heat.set_xticklabels(np.arange(1, df.shape[1] + 1, 2), fontsize=9)

    yticks = [0.5, 24.5, 49.5]
    ax_heat.set_yticks(yticks)
    ax_heat.set_yticklabels(['Image ($t=0$)', 'Transition', 'Noise ($t=1$)'], fontsize=9)

    ax_heat.set_title('(b) Temporal Deviation Map (Blue → Drop, Red → Rise)', fontsize=11, fontweight='bold', loc='left')

    # ------------------------------------------------------------
    # (c) Dynamics Line Plot
    # ------------------------------------------------------------
    ax_line = fig.add_subplot(gs[:, 1], sharey=ax_heat)

    y_axis = np.arange(df.shape[0]) + 0.5

    ax_line.plot(line_mean, y_axis, 'k--', linewidth=1.5, alpha=0.6, label='Global')
    ax_line.plot(line_core, y_axis, color='#d73027', linewidth=2.5, label=f'Layer {idx_core}')
    ax_line.plot(line_detail, y_axis, color='#4575b4', linewidth=2.5, label=f'Layer {idx_detail}')

    ax_line.axvline(0, color='gray', linestyle='-', linewidth=0.6, alpha=0.3)

    ax_line.set_title('(c) Dynamics Profile', fontsize=11, fontweight='bold', loc='left')
    ax_line.set_xlabel('$\\Delta$ Ratio (Deviation)', fontsize=10)

    x_limit = max(abs(line_core).max(), abs(line_detail).max()) * 1.25
    ax_line.set_xlim(-x_limit, x_limit)

    # 右上角 legend
    ax_line.legend(loc='upper right', fontsize=8, frameon=True)

    # 注释更明显
    ax_line.text(x_limit * 0.55, df.shape[0] * 0.70,
                 'Structure Layer\n(Rising Trend)',
                 color='#d73027', fontsize=9, fontweight='bold')

    ax_line.text(-x_limit * 0.55, df.shape[0] * 0.30,
                 'Detail Layer\n(Falling Trend)',
                 color='#4575b4', fontsize=9, ha='right', fontweight='bold')

    # 保存
    plt.savefig(f"{output_name}.pdf", dpi=300)
    plt.savefig(f"{output_name}.png", dpi=300)
    print(f"Saved: {output_name}.pdf / .png")

    plt.show()


if __name__ == "__main__":
    analyze_and_plot_advanced("adaptiva_ratio.csv")
