import os
import glob
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics.pairwise import cosine_similarity

def load_evolution_data_to_tensor(data_dir):
    print(f"=== Scanning directory: {data_dir} ===")
    
    step_dirs = sorted(glob.glob(os.path.join(data_dir, "step_*")))
    if not step_dirs:
        print("Error: No step data found. Please check the path.")
        return None, None, None

    step_map = {} 
    for p in step_dirs:
        try:
            sid = int(os.path.basename(p).split('_')[1])
            step_map[sid] = p
        except:
            continue
            
    sorted_step_ids = sorted(step_map.keys())
    
    first_step_path = step_map[sorted_step_ids[0]]
    layer_files = glob.glob(os.path.join(first_step_path, "layer_*.npy"))
    if not layer_files:
        print("Error: Step folder is empty.")
        return None, None, None
        
    layer_ids = []
    for lp in layer_files:
        try:
            lid = int(os.path.basename(lp).split('_')[1].split('.')[0])
            layer_ids.append(lid)
        except:
            continue
    sorted_layer_ids = sorted(layer_ids)
    
    sample_vec = np.load(layer_files[0])
    num_tokens = sample_vec.shape[0]
    
    num_steps = len(sorted_step_ids)
    num_layers = len(sorted_layer_ids)
    
    print(f"Dimensions detected: [Steps: {num_steps}, Layers: {num_layers}, Tokens: {num_tokens}]")
    
    full_tensor = np.zeros((num_steps, num_layers, num_tokens), dtype=np.float32)
    
    print("Loading data into memory...")
    for i, sid in enumerate(sorted_step_ids):
        step_path = step_map[sid]
        for j, lid in enumerate(sorted_layer_ids):
            npy_path = os.path.join(step_path, f"layer_{lid:03d}.npy")
            if os.path.exists(npy_path):
                vec = np.load(npy_path)
                full_tensor[i, j, :] = vec
            else:
                print(f"Warning: Missing file {npy_path}")

    return full_tensor, sorted_step_ids, sorted_layer_ids

def analyze_npy_evolution(data_dir="evolution_data"):
    tensor, step_ids, layer_ids = load_evolution_data_to_tensor(data_dir)
    if tensor is None:
        return

    num_steps, num_layers, num_tokens = tensor.shape

    output_dir_temp = "plots_temporal_npy"
    os.makedirs(output_dir_temp, exist_ok=True)
    print(f"\n--- Task 1: Analyzing Temporal Evolution (Cosine Similarity) ---")

    for idx, real_layer_id in enumerate(layer_ids):
        layer_matrix = tensor[:, idx, :]
        sim_matrix = cosine_similarity(layer_matrix)

        plt.figure(figsize=(10, 8))
        
        tick_interval = 5
        xticks = np.arange(0, num_steps, tick_interval) + 0.5
        yticks = np.arange(0, num_steps, tick_interval) + 0.5
        xticklabels = [step_ids[i] for i in range(0, num_steps, tick_interval)]

        ax = sns.heatmap(sim_matrix, cmap="Blues", vmin=0, vmax=1)
        
        ax.set_xticks(xticks)
        ax.set_xticklabels(xticklabels, rotation=0)
        ax.set_yticks(yticks)
        ax.set_yticklabels(xticklabels, rotation=0)

        plt.title(f"Layer {real_layer_id} Temporal Similarity (Cosine)")
        plt.xlabel("Time Step")
        plt.ylabel("Time Step")
        
        save_path = os.path.join(output_dir_temp, f"layer_{real_layer_id}_temporal.png")
        plt.savefig(save_path, dpi=100)
        plt.close()
    
    print(f"Task 1 complete. Images saved in: {output_dir_temp}/")

    output_dir_spatial = "plots_spatial_npy"
    os.makedirs(output_dir_spatial, exist_ok=True)
    print(f"\n--- Task 2: Analyzing Inter-Layer Relationships (Cosine Similarity) ---")

    for idx, real_step_id in enumerate(step_ids):
        step_matrix = tensor[idx, :, :]
        sim_matrix = cosine_similarity(step_matrix)

        plt.figure(figsize=(10, 8))
        
        tick_interval = 5 if num_layers > 20 else 2
        if num_layers <= 10: tick_interval = 1
            
        xticks = np.arange(0, num_layers, tick_interval) + 0.5
        yticks = np.arange(0, num_layers, tick_interval) + 0.5
        xticklabels = [layer_ids[i] for i in range(0, num_layers, tick_interval)]

        ax = sns.heatmap(sim_matrix, cmap="Greens", vmin=0, vmax=1)
        
        ax.set_xticks(xticks)
        ax.set_xticklabels(xticklabels, rotation=0)
        ax.set_yticks(yticks)
        ax.set_yticklabels(xticklabels, rotation=0)

        plt.title(f"Step {real_step_id}: Inter-Layer Similarity (Cosine)")
        plt.xlabel("Layer Index")
        plt.ylabel("Layer Index")
        
        save_path = os.path.join(output_dir_spatial, f"step_{real_step_id}_spatial.png")
        plt.savefig(save_path, dpi=100)
        plt.close()

    print(f"Task 2 complete. Images saved in: {output_dir_spatial}/\n")
    print("Analysis finished.")

if __name__ == "__main__":
    analyze_npy_evolution("evolution_data")