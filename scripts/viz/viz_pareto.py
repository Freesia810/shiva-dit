import matplotlib.pyplot as plt
import pandas as pd
import matplotlib.font_manager as fm
import numpy as np

FIG_WIDTH = 3.25
FIG_HEIGHT = 2.1   
FONT_PATH = './Times New Roman.ttf'
CSV_PATH = './efficiency_data.csv'

# --- 1. Load Font ---
try:
    fm.fontManager.addfont(FONT_PATH)
    plt.rcParams['font.family'] = 'Times New Roman'
except:
    plt.rcParams['font.family'] = 'serif'

# --- 2. Settings (Safe Version) ---
plt.rcParams.update({
    'font.size': 8,
    'axes.labelsize': 8,      
    'xtick.labelsize': 7,     
    'ytick.labelsize': 7,     
    'axes.linewidth': 1.0,
    'pdf.fonttype': 42,
    'ps.fonttype': 42
})

# --- 3. Data ---
df = pd.read_csv(CSV_PATH)

# --- 4. Plot Setup (Manual Adjust) ---
plt.figure(figsize=(FIG_WIDTH, FIG_HEIGHT))

plt.subplots_adjust(left=0.13, bottom=0.18, right=0.98, top=0.97)
ax = plt.gca()

# Colors
COLOR_OURS = '#d62728'     
COLOR_TEACHER = '#2ca02c'  
COLOR_BASE = '#7f7f7f'     

# Separate Groups
ours = df[df['Type'] == 'Ours']
teacher = df[df['Type'] == 'Teacher']
baselines = df[df['Type'] == 'Baseline']

# --- 5. Scatter Plots ---
plt.scatter(baselines['Latency_ms'], baselines['FID'], 
            c=COLOR_BASE, s=60, alpha=0.8, marker='o', 
            edgecolors='white', linewidth=0.5, label='Baselines', zorder=2)

plt.scatter(teacher['Latency_ms'], teacher['FID'], 
            c=COLOR_TEACHER, s=50, marker='D', 
            edgecolors='white', linewidth=0.5, label='Teacher', zorder=3)

plt.scatter(ours['Latency_ms'], ours['FID'], 
            c=COLOR_OURS, s=120, marker='*', 
            edgecolors='white', linewidth=0.5, label='Shiva (Ours)', zorder=10)

pareto_points = pd.concat([ours, teacher]).sort_values(by='Latency_ms', ascending=False)
plt.plot(pareto_points['Latency_ms'], pareto_points['FID'], 
         color=COLOR_OURS, linestyle='--', linewidth=1.0, alpha=0.5, zorder=1)

plt.axhline(y=teacher['FID'].values[0], color=COLOR_TEACHER, linestyle=':', linewidth=1.0, alpha=0.4, zorder=0)

# --- 7. Annotations (Same as before) ---
offset_map = {
    'Vanilla': (4, 0, 'left', 'center'), 'Finetuned': (5, -2.5, 'left', 'top'),     
    'ToMeSD': (-4, 0, 'right', 'center'), 'ToFu': (4, 0, 'left', 'center'),
    'SDTM': (0, -6, 'center', 'top'), 'ToMA': (4, 0, 'left', 'center'),
    'IBTM': (4, 0, 'left', 'center'), 'DiffCR': (0, 5, 'center', 'bottom'), 
    'DyDiT': (-4, 4, 'right', 'bottom'), 
    'Shiva-80%': (-3, 10, 'right', 'top'),   
    'Shiva-60%': (0, 5, 'center', 'bottom'), 
    'Shiva-70%': (0, -5, 'center', 'top'), 
    'Shiva-90%': (0, 5, 'left', 'bottom'), 
}

for i, row in df.iterrows():
    name = row['Method']
    settings = offset_map.get(name, (3, 3, 'left', 'bottom'))
    weight = 'bold' if 'Shiva' in name else 'normal'
    color = '#b01e1f' if 'Shiva' in name else 'black' 
    plt.annotate(name, (row['Latency_ms'], row['FID']), xytext=(settings[0], settings[1]), 
                 textcoords='offset points', fontsize=6.5, ha=settings[2], va=settings[3], 
                 fontweight=weight, color=color, zorder=15)

plt.xlabel('Inference Latency (ms)', fontweight='bold', labelpad=2)
plt.ylabel('FID', fontweight='bold', labelpad=1)

ax.tick_params(axis='both', which='major', pad=2)

plt.grid(True, linestyle='--', alpha=0.4, zorder=0)
plt.xlim(4500, 8300)
plt.ylim(10, 52)

plt.savefig('pareto_frontier.pdf', dpi=300, bbox_inches='tight', pad_inches=0.02)
print("Plot saved to pareto_frontier.pdf")