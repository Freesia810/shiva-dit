import os
import glob
import numpy as np
import matplotlib.pyplot as plt
import scipy.stats

def softmax(x):
    e_x = np.exp(x - np.max(x))
    return e_x / e_x.sum(axis=-1, keepdims=True)

def calculate_metrics(data_dir="evolution_data"):
    step_dirs = sorted(glob.glob(os.path.join(data_dir, "step_*")))
    if not step_dirs:
        print("Error: No data found. Please run inference to generate evolution_data first.")
        return None, None

    metrics = {
        "entropy": [],      
        "variance": [],     
        "gini": [],         
        "top_10_mass": []   
    }
    
    steps_indices = []

    print(f"Analyzing data for {len(step_dirs)} time steps...")

    for s_dir in step_dirs:
        try:
            step_idx = int(os.path.basename(s_dir).split('_')[1])
        except:
            continue
        
        steps_indices.append(step_idx)
        
        layer_files = glob.glob(os.path.join(s_dir, "layer_*.npy"))
        
        step_entropies = []
        step_variances = []
        step_ginis = []
        step_top10 = []

        for l_file in layer_files:
            raw_scores = np.load(l_file)
            
            probs = softmax(raw_scores)
            entropy = scipy.stats.entropy(probs) 
            step_entropies.append(entropy)

            norm_scores = (raw_scores - raw_scores.min()) / (raw_scores.max() - raw_scores.min() + 1e-6)
            step_variances.append(np.var(norm_scores))

            sorted_probs = np.sort(probs)[::-1]
            k = max(1, int(len(probs) * 0.1)) 
            mass = np.sum(sorted_probs[:k])
            step_top10.append(mass)

        metrics["entropy"].append(np.mean(step_entropies))
        metrics["variance"].append(np.mean(step_variances))
        metrics["top_10_mass"].append(np.mean(step_top10))

    return steps_indices, metrics

def plot_rationale(steps, metrics):
    fig, axes = plt.subplots(1, 2, figsize=(18, 5))
    
    x_norm = np.array(steps) / max(steps)
    
    ax = axes[0]
    ax.plot(steps, metrics["entropy"], 'b-', linewidth=2, label='Attention Entropy')
    ax.set_title("Rationale 1: Attention Entropy")
    ax.set_xlabel("Inference Step (0=Start)")
    ax.set_ylabel("Shannon Entropy (Lower is Better for Pruning)")
    ax.grid(True, alpha=0.3)
    
    ax2 = ax.twinx()
    ax2.plot(steps, 1 - x_norm, 'r--', label='Linear Decay Schedule', alpha=0.5)
    ax2.set_ylabel("Keep Ratio (Linear Schedule)")
    ax2.legend(loc='upper right')
    
    ax = axes[1]
    ax.plot(steps, metrics["top_10_mass"], 'm-', linewidth=2, label='Mass of Top 10% Tokens')
    ax.set_title("Rationale 2: Information Concentration")
    ax.set_xlabel("Inference Step")
    ax.set_ylabel("Cumulative Probability (Mass)")
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig("pruning_rationale_analysis.png")
    print("Chart saved to: pruning_rationale_analysis.png")
    plt.show()

if __name__ == "__main__":
    steps, data = calculate_metrics()
    if steps:
        plot_rationale(steps, data)