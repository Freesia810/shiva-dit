import matplotlib.pyplot as plt
import pandas as pd
import matplotlib.font_manager as fm
import numpy as np

# --- 1. Load Font ---
font_path = './Times New Roman.ttf'
try:
    fm.fontManager.addfont(font_path)
    plt.rcParams['font.family'] = 'Times New Roman'
except Exception as e:
    print(f"Warning: Could not load font from {font_path}. Using default.")

# --- 2. ICML Single-Column Settings ---
plt.rcParams.update({
    'font.size': 8,
    'axes.labelsize': 8,      
    'xtick.labelsize': 7,     
    'ytick.labelsize': 7,     
    'axes.linewidth': 1.0,
    'figure.autolayout': True,
    'pdf.fonttype': 42,
    'ps.fonttype': 42
})

# --- 3. Read & Preprocess Data ---
csv_path = './efficiency_data.csv'
try:
    df = pd.read_csv(csv_path)
except FileNotFoundError:
    print(f"Error: {csv_path} not found. Please ensure the CSV file is in the directory.")
    exit()

# Logic to strip prefix and rename
df['Model'] = df['Model'].str.replace('SD3-', '')
df['Model'] = df['Model'].replace({'80%': 'Shiva-80%', '60%': 'Shiva-60%'})

# --- 4. Plot Configuration ---
fig_width = 3.25
fig_height = 2.1  
plt.figure(figsize=(fig_width, fig_height))

colors = []
sizes = []
markers = []

for model in df['Model']:
    if 'Baseline' in model:
        colors.append('#7f7f7f') 
        sizes.append(80) 
        markers.append('X')
    elif 'Shiva' in model or '%' in model: 
        colors.append('#d62728') 
        sizes.append(120) 
        markers.append('*')
    else:
        colors.append('#1f77b4') 
        sizes.append(50) 
        markers.append('o')

plt.scatter(df['FLOPs(T)'], df['DiT Latency(ms)'], c=colors, s=sizes, alpha=0.9, edgecolors='white', linewidth=0.5, zorder=3)

plt.ylim(4700, 8000)

# --- 5. Annotations ---
offset_map = {
    'Baseline': (-4, -4, 'right', 'top'),
    'ToMeSD':   (0, 5, 'center', 'bottom'), 
    'ToFu':     (3, -5, 'left', 'top'),     
    'IBTM':     (-3, 3, 'right', 'bottom'),
    'ToMA':     (-5, 0, 'right', 'center'),
    'SDTM':     (0, -8, 'center', 'top'),
    'DiffCR':   (5, 0, 'left', 'center'),
    'DyDiT':    (5, 0, 'left', 'center'),
    'Shiva-80%': (7, 0, 'left', 'center'),
    'Shiva-60%': (7, 0, 'left', 'center'),
}

for i, row in df.iterrows():
    model = row['Model']
    x = row['FLOPs(T)']
    y = row['DiT Latency(ms)']
    
    settings = offset_map.get(model, (4, 4, 'left', 'bottom'))
    xytext = (settings[0], settings[1])
    ha = settings[2]
    va = settings[3]
    
    weight = 'bold' if 'Shiva' in model or '%' in model else 'normal'
    
    plt.annotate(model, (x, y), xytext=xytext, textcoords='offset points', 
                 fontsize=7, ha=ha, va=va, fontweight=weight)

plt.xlabel('FLOPs (T)')
plt.ylabel('DiT Latency (ms)')
plt.grid(True, linestyle='--', alpha=0.4, zorder=0)

plt.tight_layout(pad=0.2)
plt.savefig('efficiency_scatter.pdf', dpi=300, bbox_inches='tight')
print("Plot saved to efficiency_scatter.pdf")