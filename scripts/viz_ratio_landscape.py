import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import seaborn as sns


def analyze_and_plot_zoomed(csv_path):
    # 1. 读取数据 (假设没有表头，如果有请改为 header=0)
    try:
        df = pd.read_csv(csv_path, header=None)
    except Exception as e:
        print(f"读取失败: {e}")
        return

    # 计算均值
    layer_means = df.mean(axis=0)
    timestep_means = df.mean(axis=1)

    # --- 关键修改点 1: 获取数据的动态范围 ---
    # 我们不再假设范围是 0~1，而是看数据到底在哪
    # 为了防止贴边，稍微加一点 padding (例如上下各留 10% 的余量)
    
    # 全局数据的极值 (用于热力图)
    g_min, g_max = df.values.min(), df.values.max()
    
    # Layer均值的极值 (用于上方折线图)
    l_min, l_max = layer_means.min(), layer_means.max()
    l_span = l_max - l_min
    l_ylim = (l_min - l_span * 0.2, l_max + l_span * 0.2) # 上下扩充20%
    
    # Timestep均值的极值 (用于右侧折线图)
    t_min, t_max = timestep_means.min(), timestep_means.max()
    t_span = t_max - t_min
    t_xlim = (t_min - t_span * 0.2, t_max + t_span * 0.2)

    print(f"检测到数据范围: {g_min:.4f} ~ {g_max:.4f}")
    print(f"正在进行动态缩放可视化...")

    # 2. 绘图
    fig = plt.figure(figsize=(12, 10))
    gs = gridspec.GridSpec(4, 4, wspace=0.08, hspace=0.08)

    # --- A. 中间热力图 (Zoomed Color) ---
    ax_heatmap = fig.add_subplot(gs[1:4, 0:3])
    
    # 关键修改：vmin 和 vmax 不再是 0 和 1
    # 而是紧贴数据的真实范围，这样哪怕只有 0.01 的差别，颜色也会从“最黑”变成“最亮”
    sns.heatmap(df, cmap='magma', 
                vmin=g_min, vmax=g_max,  # <--- 核心改动
                ax=ax_heatmap, cbar=False,
                xticklabels=5, yticklabels=10)
    
    ax_heatmap.set_xlabel('Transformer Layers', fontsize=12)
    ax_heatmap.set_ylabel('Timesteps', fontsize=12)

    # --- B. 顶部 Layer 均值 (Zoomed Y-Axis) ---
    ax_layer = fig.add_subplot(gs[0, 0:3], sharex=ax_heatmap)
    ax_layer.plot(layer_means.index, layer_means.values, color='#1f77b4', linewidth=2, marker='o', markersize=3)
    
    # 关键修改：取消 set_ylim(0, 1)，使用动态范围
    ax_layer.set_ylim(l_ylim) 
    ax_layer.set_ylabel('Avg Ratio', fontsize=10)
    ax_layer.set_title(f'Layer-wise Avg (Range: {l_min:.3f}-{l_max:.3f})', fontsize=12) # 标题显示范围
    plt.setp(ax_layer.get_xticklabels(), visible=False)
    ax_layer.grid(True, linestyle=':', alpha=0.6)

    # --- C. 右侧 Timestep 均值 (Zoomed X-Axis) ---
    ax_time = fig.add_subplot(gs[1:4, 3], sharey=ax_heatmap)
    ax_time.plot(timestep_means.values, timestep_means.index, color='#d62728', linewidth=2, marker='.', markersize=2)
    
    # 关键修改：取消 set_xlim(0, 1)，使用动态范围
    ax_time.set_xlim(t_xlim)
    ax_time.set_xlabel('Avg Ratio', fontsize=10)
    # 在标题显示变化幅度，证明“虽然小，但是有趋势”
    ax_time.set_title(f'Time Trend\n(Δ={t_max-t_min:.3f})', fontsize=10) 
    
    plt.setp(ax_time.get_yticklabels(), visible=False)
    ax_time.grid(True, linestyle=':', alpha=0.6)

    # --- Colorbar ---
    # 这里的 Colorbar 会显示真实的数值范围（例如 0.61 - 0.63）
    cbar_ax = fig.add_axes([0.92, 0.15, 0.02, 0.25])
    norm = plt.Normalize(g_min, g_max)
    sm = plt.cm.ScalarMappable(cmap='magma', norm=norm)
    sm.set_array([])
    fig.colorbar(sm, cax=cbar_ax, label='Keep Ratio (Zoomed)')

    plt.suptitle('Adaptive Ratio Analysis (Auto-Scaled View)', fontsize=16, y=0.95)
    
    save_path = 'ratio_analysis_zoomed.pdf'
    plt.savefig(save_path, bbox_inches='tight', dpi=300)
    print(f"图表已保存至: {save_path}")
    plt.show()


analyze_and_plot_zoomed('adaptiva_ratio.csv')