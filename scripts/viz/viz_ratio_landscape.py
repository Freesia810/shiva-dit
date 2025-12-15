import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import seaborn as sns
import os
from matplotlib import font_manager

# ============================================================
# 1. 精细化外观设置 (Publication Quality Style)
# ============================================================
def set_style():
    # 字体设置 (尝试 Times New Roman)
    font_path = 'Times New Roman.ttf'
    if os.path.exists(font_path):
        font_manager.fontManager.addfont(font_path)
        prop = font_manager.FontProperties(fname=font_path)
        plt.rcParams['font.family'] = prop.get_name()
    else:
        plt.rcParams['font.family'] = 'serif'
        plt.rcParams['font.serif'] = ['Times New Roman']
    
    # 关键：全局线条变细 (Refined Look)
    plt.rcParams['axes.linewidth'] = 0.6       # 边框线宽变细
    plt.rcParams['xtick.major.width'] = 0.6    # 刻度线宽变细
    plt.rcParams['ytick.major.width'] = 0.6
    plt.rcParams['xtick.direction'] = 'out'    # 刻度朝外
    plt.rcParams['ytick.direction'] = 'out'
    plt.rcParams['font.size'] = 12             # 基础字号
    plt.rcParams['axes.unicode_minus'] = False
    plt.rcParams['mathtext.fontset'] = 'stix'  # 数学公式字体优化

# ============================================================
# 2. 绘图主函数
# ============================================================
def plot_refined_landscape(csv_path, output_name="ratio_analysis"):
    set_style()

    if not os.path.exists(csv_path):
        print(f"Error: {csv_path} not found.")
        return

    # 1. 数据读取与预处理
    df_raw = pd.read_csv(csv_path, header=None)
    
    # 翻转数据：Top=Noise (t=1), Bottom=Image (t=0)
    df = df_raw.iloc[::-1].reset_index(drop=True)

    # 统计量
    layer_means = df.mean(axis=0)       # 顶部：空间维度平均
    time_means = df.mean(axis=1)        # 右侧：时间维度平均

    # ------------------------------------------------------------
    # 2. 自动计算 Range
    # ------------------------------------------------------------
    vmin, vmax = 0.2, 1.0
    tm_min, tm_max = time_means.min(), time_means.max()
    tm_range = tm_max - tm_min
    x_line_min = max(0, tm_min - tm_range * 0.1) 
    x_line_max = min(1.05, tm_max + tm_range * 0.1)

    # ------------------------------------------------------------
    # 3. 画布布局
    # ------------------------------------------------------------
    # 保持 (9, 3.2) 的矮宽比例
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
    
    sns.despine(ax=ax_top, top=True, right=True, bottom=True)
    ax_top.spines['left'].set_linewidth(0.6)
    ax_top.grid(axis='y', linestyle='--', color='gray', alpha=0.3, linewidth=0.5)

    # ============================================================
    # (B) Center: Heatmap
    # ============================================================
    ax_heat = fig.add_subplot(gs[1, 0])
    
    # 关键修改：
    # 1. 保持 rasterized=True 以消除白线
    # 2. 保存时提高 DPI 解决模糊
    sns.heatmap(df, ax=ax_heat, 
                cmap=cmap, norm=norm,
                cbar=False,
                rasterized=True) # 保持栅格化，防止矢量白线缝隙
    
    # 边框：加上细黑框 (0.6 width)
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
    
    sns.despine(ax=ax_right, top=True, right=True, left=True)
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
    # 关键修改：dpi=600 (High Resolution Rasterization)
    plt.savefig(f"{output_name}.pdf", dpi=600, bbox_inches='tight', pad_inches=0.02)
    plt.savefig(f"{output_name}.png", dpi=600, bbox_inches='tight', pad_inches=0.02)
    print(f"Saved: {output_name}.pdf")
    plt.show()

if __name__ == "__main__":
    plot_refined_landscape("adaptiva_ratio.csv")