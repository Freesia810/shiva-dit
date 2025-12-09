import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import os
from matplotlib import font_manager

file_adaptive = 'adaptive_val_loss.csv'
file_fixed = 'fix_val_loss.csv'
font_path = 'Times New Roman.ttf' 

col_step = 'step'
col_loss = 'value'

plt.rcParams['mathtext.fontset'] = 'stix'
plt.rcParams['font.family'] = 'STIXGeneral'

if not os.path.exists(font_path):
    print("Warning: Font file not found, using default font")
    plt.rcParams['font.family'] = 'serif'
else:
    try:
        font_manager.fontManager.addfont(font_path)
        prop = font_manager.FontProperties(fname=font_path)
        print(f"✅ Loaded font: {font_path}")
        print(f"   Font name: {prop.get_name()}")
        
        plt.rcParams['font.family'] = prop.get_name()
        plt.rcParams['axes.unicode_minus'] = False
        
    except Exception as e:
        print(f"⚠️ Error loading font: {e}")
        plt.rcParams['font.family'] = 'serif'

plt.rcParams['font.size'] = 14
plt.rcParams['axes.linewidth'] = 1.5
plt.rcParams['xtick.major.width'] = 1.5
plt.rcParams['ytick.major.width'] = 1.5
plt.rcParams['xtick.direction'] = 'in'
plt.rcParams['ytick.direction'] = 'in'
plt.rcParams['grid.alpha'] = 0.3

def read_data(filename):
    try:
        df = pd.read_csv(filename)
        return df
    except Exception as e:
        print(f"Load data from {filename}: {e}")
        return pd.DataFrame()

df_adaptive = read_data(file_adaptive)
df_fixed = read_data(file_fixed)

START_IDX = 8
if len(df_adaptive) > START_IDX:
    df_adaptive = df_adaptive.iloc[START_IDX:].reset_index(drop=True)
if len(df_fixed) > START_IDX:
    df_fixed = df_fixed.iloc[START_IDX:].reset_index(drop=True)

fig, ax = plt.subplots(figsize=(6, 4.5))

color_ours = '#1f77b4'  # Deep Blue
color_base = '#d62728'  # Brick Red

if not df_adaptive.empty and not df_fixed.empty:
    steps_adap = df_adaptive[col_step]
    loss_adap = df_adaptive[col_loss]
    
    steps_fixed = df_fixed[col_step]
    loss_fixed = df_fixed[col_loss]
    
    ax.plot(steps_fixed, loss_fixed, label='Fixed Ratio ($r=0.6$)', 
            color=color_base, linestyle='--', linewidth=2, alpha=0.8)
    ax.plot(steps_adap, loss_adap, label=r"Ours (Adaptive, $\bar{r}=0.6$)", 
            color=color_ours, linestyle='-', linewidth=2.5)
else:
    print("❌ Empty dataframe")

ax.set_xlabel('Training Steps', fontsize=16)
ax.set_ylabel('Validation MSE Loss', fontsize=16)

ax.legend(fontsize=14, frameon=True, fancybox=False, edgecolor='black')

ax.grid(True, linestyle='--', linewidth=0.5)
plt.tight_layout()

output_pdf = 'ablation_ratio_convergence.pdf'
output_png = 'ablation_ratio_convergence.png'

print(f"Saving figure to {output_pdf} ...")
plt.savefig(output_pdf, dpi=300, bbox_inches='tight')
plt.savefig(output_png, dpi=300, bbox_inches='tight')
print("✅ Convergence figure saved.")
