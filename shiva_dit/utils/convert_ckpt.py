import torch
import os
import json
import glob
from safetensors.torch import load_file as load_safetensors

def filter_state_dict(state_dict, keep_keywords=None):
    if keep_keywords is None:
        keep_keywords = ["lora", "modules_to_save", "hook"]
    
    filtered_dict = {}
    pruned_count = 0
    
    for key, value in state_dict.items():
        if any(keyword in key for keyword in keep_keywords):
            print(f"  - Keeping key: {key}")
            filtered_dict[key] = value
        else:
            print(f"  - Pruning key: {key}")
            pruned_count += 1
            
    print(f"  - Original keys: {len(state_dict)}")
    print(f"  - Filtered keys: {len(filtered_dict)} (Pruned {pruned_count} base model keys)")
    
    return filtered_dict

def convert_folder_to_single_file(input_folder, output_path=None):
    print(f"Processing folder: {input_folder}")
    
    config_path = os.path.join(input_folder, "shiva_config.json")
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"Missing shiva_config.json in {input_folder}")
    
    with open(config_path, "r") as f:
        meta_config = json.load(f)
    print("  - Metadata loaded.")

    potential_weights = glob.glob(os.path.join(input_folder, "*model*.safetensors")) + \
                        glob.glob(os.path.join(input_folder, "*model*.bin"))
    
    if not potential_weights:
        raise FileNotFoundError(f"No weight files found in {input_folder}")
    
    weights_path = potential_weights[0]
    print(f"  - Loading heavy weights from: {os.path.basename(weights_path)}")
    
    if weights_path.endswith(".safetensors"):
        full_state_dict = load_safetensors(weights_path, device="cpu")
    else:
        full_state_dict = torch.load(weights_path, map_location="cpu")

    keywords = ["lora", "modules_to_save", "hook"] 
    filtered_weights = filter_state_dict(full_state_dict, keep_keywords=keywords)

    if len(filtered_weights) == 0:
        print("WARNING: Filtered state dict is empty! Check your keywords.")
        return

    save_dict = {
        "model_type": meta_config.get("model_type"),
        "training_mode": meta_config.get("training_mode"),
        "hook_info": meta_config.get("hook_info"),
        "lora_info": meta_config.get("lora_info"),
        "model": filtered_weights 
    }

    if output_path is None:
        output_path = os.path.join(input_folder, "shiva_model.pt")
    
    print(f"  - Saving to {output_path}...")
    torch.save(save_dict, output_path)
    
    original_size = os.path.getsize(weights_path) / (1024 * 1024)
    new_size = os.path.getsize(output_path) / (1024 * 1024)
    print(f"Done! Reduced size from {original_size:.2f} MB to {new_size:.2f} MB")

if __name__ == "__main__":
    folder_path = "./pixart_ckpt" 
    
    try:
        convert_folder_to_single_file(folder_path)
    except Exception as e:
        print(f"Error: {e}")