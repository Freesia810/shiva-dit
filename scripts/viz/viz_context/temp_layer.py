import os
import numpy as np
import matplotlib.pyplot as plt
import glob
from sklearn.metrics.pairwise import cosine_similarity

def load_evolution_data(data_dir="evolution_data"):
    print("Loading data...")
    step_dirs = sorted(glob.glob(os.path.join(data_dir, "step_*")))
    
    if not step_dirs:
        print("Error: No data found. Please check the data path.")
        return None, None, None

    sample_step = step_dirs[0]
    layer_files = sorted(glob.glob(os.path.join(sample_step, "layer_*.npy")))
    sample_data = np.load(layer_files[0])
    
    num_steps = len(step_dirs)
    num_layers = len(layer_files)
    num_tokens = sample_data.shape[0]
    
    print(f"Detected: Steps={num_steps}, Layers={num_layers}, Tokens={num_tokens}")
    
    matrix = np.zeros((num_steps, num_layers, num_tokens))
    
    steps_indices = []
    layers_indices = []

    for t, s_dir in enumerate(step_dirs):
        s_idx = int(os.path.basename(s_dir).split('_')[1])
        steps_indices.append(s_idx)
        
        l_files = sorted(glob.glob(os.path.join(s_dir, "layer_*.npy")))
        for l, l_file in enumerate(l_files):
            if t == 0:
                l_idx = int(os.path.basename(l_file).split('_')[1].split('.')[0])
                layers_indices.append(l_idx)
            
            matrix[t, l, :] = np.load(l_file)
            
    return matrix, steps_indices, layers_indices

def verify_hypotheses(data_dir="evolution_data"):
    matrix, step_labels, layer_labels = load_evolution_data(data_dir)
    if matrix is None: return

    T, L, D = matrix.shape

    temporal_sims = [] 
    
    for l in range(L):
        layer_sims = []
        for t in range(T - 1):
            vec_t = matrix[t, l, :].reshape(1, -1)
            vec_next = matrix[t+1, l, :].reshape(1, -1)
            
            sim = cosine_similarity(vec_t, vec_next)[0][0]
            layer_sims.append(sim)
        
        temporal_sims.append(np.mean(layer_sims))

    plt.figure(figsize=(10, 6))
    plt.plot(layer_labels, temporal_sims, marker='o', linestyle='-', linewidth=2, color='royalblue')
    plt.title("Hypothesis 1: Temporal Stability vs. Layer Depth")
    plt.xlabel("Layer Index (Depth)")
    plt.ylabel("Avg Cosine Similarity (Step t vs t+1)")
    plt.grid(True, linestyle='--', alpha=0.6)
    
    plt.savefig("hypothesis_1_temporal_stability.png")
    print("Chart generated: hypothesis_1_temporal_stability.png")

    spatial_sims = [] 
    
    for t in range(T):
        step_layer_sims = []
        for l in range(L - 1):
            vec_l = matrix[t, l, :].reshape(1, -1)
            vec_next = matrix[t, l+1, :].reshape(1, -1)
            
            sim = cosine_similarity(vec_l, vec_next)[0][0]
            step_layer_sims.append(sim)
            
        spatial_sims.append(np.mean(step_layer_sims))

    plt.figure(figsize=(10, 6))
    plt.plot(step_labels, spatial_sims, marker='s', linestyle='-', linewidth=2, color='crimson')
    plt.title("Hypothesis 2: Layer Consistency vs. Time Step")
    plt.xlabel("Inference Step (0 = Start)")
    plt.ylabel("Avg Cosine Similarity (Layer l vs l+1)")
    plt.grid(True, linestyle='--', alpha=0.6)

    plt.savefig("hypothesis_2_layer_consistency.png")
    print("Chart generated: hypothesis_2_layer_consistency.png")

if __name__ == "__main__":
    verify_hypotheses()