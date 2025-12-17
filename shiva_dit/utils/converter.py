import os
import torch
import json
import glob
from safetensors.torch import load_file as load_safetensors

class ModelConverter:
    @staticmethod
    def load_metadata(ckpt_dir):
        config_path = os.path.join(ckpt_dir, "shiva_config.json")
        if os.path.exists(config_path):
            with open(config_path, "r") as f:
                return json.load(f)
        return {}

    @staticmethod
    def convert_format_1_to_3(src_dir, output_file):
        """
        Format 1 (Accelerator Dir) -> Format 3 (Single PT File)
        Combines model.safetensors/bin and shiva_config.json into one .pt
        """
        print(f"Converting Format 1 -> 3: {src_dir} -> {output_file}")
        
        metadata = ModelConverter.load_metadata(src_dir)
        
        # Load Model Weights
        state_dict = {}
        
        safetensor_path = os.path.join(src_dir, "model.safetensors")
        bin_path = os.path.join(src_dir, "pytorch_model.bin")
        
        potential_safetensors = glob.glob(os.path.join(src_dir, "*.safetensors"))
        potential_bins = glob.glob(os.path.join(src_dir, "*.bin"))

        if os.path.exists(safetensor_path):
            state_dict = load_safetensors(safetensor_path, device="cpu")
        elif os.path.exists(bin_path):
            state_dict = torch.load(bin_path, map_location="cpu")
        elif len(potential_safetensors) > 0:
            target = next((f for f in potential_safetensors if "model" in os.path.basename(f)), potential_safetensors[0])
            print(f"  -> Found fallback weights: {os.path.basename(target)}")
            state_dict = load_safetensors(target, device="cpu")
        elif len(potential_bins) > 0:
            target = next((f for f in potential_bins if "model" in os.path.basename(f)), potential_bins[0])
            print(f"  -> Found fallback weights: {os.path.basename(target)}")
            state_dict = torch.load(target, map_location="cpu")
        else:
            raise FileNotFoundError(f"Could not find valid model weights (safetensors or bin) in {src_dir}")

        save_dict = {
            "model": state_dict,
            **metadata
        }
        
        os.makedirs(os.path.dirname(os.path.abspath(output_file)), exist_ok=True)
        
        torch.save(save_dict, output_file)
        print("Done.")

    @staticmethod
    def convert_format_2_to_3(src_dir, output_file):
        """
        Format 2 (Modular Dir) -> Format 3 (Single PT File)
        Combines modular_weights/*.pt and shiva_config.json into one .pt (Incremental)
        """
        print(f"Converting Format 2 -> 3: {src_dir} -> {output_file}")
        
        metadata = ModelConverter.load_metadata(src_dir)
        
        modular_dir = os.path.join(src_dir, "modular_weights")
        if not os.path.exists(modular_dir):
            modular_dir = src_dir # Try root if flattened
            
        pt_files = glob.glob(os.path.join(modular_dir, "*.pt"))
        full_sd = {}
        for pt_f in pt_files:
            full_sd.update(torch.load(pt_f, map_location="cpu"))
            
        save_dict = {
            "model": full_sd,
            **metadata
        }
        torch.save(save_dict, output_file)
        print("Done.")

    @staticmethod
    def convert_format_3_to_2(src_file, output_dir):
        """
        Format 3 (Single PT) -> Format 2 (Modular Dir)
        Splits single dict into modular weights based on keys.
        """
        print(f"Converting Format 3 -> 2: {src_file} -> {output_dir}")
        
        data = torch.load(src_file, map_location="cpu")
        
        # Extract metadata
        metadata_keys = ["model_type", "training_mode", "hook_info", "lora_info", "finetune_target_modules", "global_step"]
        metadata = {k: data[k] for k in metadata_keys if k in data}
        
        os.makedirs(output_dir, exist_ok=True)
        with open(os.path.join(output_dir, "shiva_config.json"), "w") as f:
            json.dump(metadata, f, indent=4)
            
        state_dict = data.get("model", {})
        modular_dir = os.path.join(output_dir, "modular_weights")
        os.makedirs(modular_dir, exist_ok=True)
        
        # Logic to split (borrowed from trainer)
        components_map = {
            "router": ["router.", "router_layer_embedder"],
            "gater_policy": ["gater_policy.", "gate_policy."],
            "image_ratio_policy": ["image_ratio_policy."],
            "context_ratio_policy": ["context_ratio_policy."],
            "context_pruner": ["context_pruner."],
            "fuser": ["fuser.", "fuse_expert."],
        }
        
        saved_keys = set()
        
        for name, prefixes in components_map.items():
            comp_dict = {k: v for k, v in state_dict.items() if any(p in k for p in prefixes)}
            if comp_dict:
                torch.save(comp_dict, os.path.join(modular_dir, f"{name}.pt"))
                saved_keys.update(comp_dict.keys())
        
        # Save remaining as lora/finetune/base
        remaining_dict = {k: v for k, v in state_dict.items() if k not in saved_keys}
        if remaining_dict:
            # Simple heuristic: if lora keys exist, name it lora, else finetune
            is_lora = any("lora_" in k for k in remaining_dict.keys())
            fname = "lora.pt" if is_lora else "finetune.pt"
            torch.save(remaining_dict, os.path.join(modular_dir, fname))
            
        print("Done.")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", type=str, required=True, choices=["1to3", "2to3", "3to2"])
    parser.add_argument("--src", type=str, required=True)
    parser.add_argument("--dst", type=str, required=True)
    args = parser.parse_args()
    
    if args.mode == "1to3":
        ModelConverter.convert_format_1_to_3(args.src, args.dst)
    elif args.mode == "2to3":
        ModelConverter.convert_format_2_to_3(args.src, args.dst)
    elif args.mode == "3to2":
        ModelConverter.convert_format_3_to_2(args.src, args.dst)