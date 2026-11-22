import torch
import argparse
from collections import defaultdict

from shiva_dit.utils.inference_utils import load_pipeline_for_inference
from shiva_dit.utils.utils import process_quick_start
from shiva_dit.hook import HookManager
from accelerate.utils import set_seed

def generate_lut(args):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = torch.bfloat16 if args.dtype == "bf16" else torch.float16
    
    set_seed(42)

    print(f"Loading model from {args.ckpt_path}...")
    pipe = load_pipeline_for_inference(
        ckpt_path=args.ckpt_path,
        base_model_path=args.base_model_path,
        device=device,
        dtype=dtype,
        gater_lut_path=None,      
        optimize_hooks=args.optimize_hooks,
        merge_lora=args.merge_lora
    )
    
    hooks = list(HookManager.get_all_hooks().values())
    if not hooks:
        raise RuntimeError("No hooks found.")

    try:
        if hasattr(pipe, "transformer"):
            # Flux / SD3
            total_layers = pipe.transformer.config.num_layers
            if hasattr(pipe.transformer.config, "num_single_layers"):
                total_layers += pipe.transformer.config.num_single_layers
        elif hasattr(pipe, "dit"):
            # PixArt / DiT
            total_layers = pipe.dit.config.depth
        else:
            raise AttributeError("Unknown architecture")
    except Exception:
        print("Warning: Could not determine total layers from config. Inferring from hooks.")
        max_hook_idx = max(h.layer_idx for h in hooks)
        total_layers = max_hook_idx + 1

    print(f"Total Model Layers: {total_layers}")
    print(f"Active Hooked Layers: {len(hooks)} (Skipped: {total_layers - len(hooks)})")
    
    gater_policy = hooks[0].gater_policy
    ratio_policy = hooks[0].image_ratio_policy
    
    gater_data = defaultdict(list)
    ratio_data = defaultdict(list)

    def capture_gater_hook(module, args, output):
        try:
            layer_idx_tensor = args[1]
            current_layer = layer_idx_tensor.flatten()[0].item()
            
            # output: gate_logits [B, 2]
            logits = output
            decision = torch.argmax(logits[0], dim=-1).cpu().item() # 0 or 1
            gater_data[current_layer].append(decision)
        except Exception as e:
            print(f"Error in gater hook: {e}")

    def capture_ratio_hook(module, args, output):
        try:
            layer_idx_tensor = args[1]
            current_layer = layer_idx_tensor.flatten()[0].item()
            
            # output: logits [B, 1]
            logits = output
            ratio_val = torch.sigmoid(logits).mean().cpu().item() # float 0-1
            ratio_data[current_layer].append(ratio_val)
        except Exception as e:
            print(f"Error in ratio hook: {e}")

    handle_gater = gater_policy.register_forward_hook(capture_gater_hook)
    handle_ratio = ratio_policy.register_forward_hook(capture_ratio_hook)
    
    print(f"Running inference (Steps={args.steps})...")
    prompt = "A high quality photo" 
    
    with torch.no_grad():
        _ = process_quick_start(
            pipe, 
            prompt=prompt, 
            num_inference_steps=args.steps, 
            height=args.height, 
            width=args.width
        )
    
    handle_gater.remove()
    handle_ratio.remove()
    
    final_gater_list = []
    final_ratio_list = []

    print("\nConstructing Dense LUT...")
    
    for layer_idx in range(total_layers):
        if layer_idx in gater_data:
            vals = gater_data[layer_idx]
            if len(vals) != args.steps:
                print(f"Warning: Layer {layer_idx} has {len(vals)} gater steps, expected {args.steps}. Truncating/Padding.")
                if len(vals) > args.steps:
                    vals = vals[:args.steps]
                else:
                    vals = vals + [0] * (args.steps - len(vals))
            final_gater_list.append(vals)
        else:
            final_gater_list.append([0] * args.steps)

        if layer_idx in ratio_data:
            vals = ratio_data[layer_idx]
            if len(vals) != args.steps:
                if len(vals) > args.steps:
                    vals = vals[:args.steps]
                else:
                    vals = vals + [1.0] * (args.steps - len(vals))
            final_ratio_list.append(vals)
        else:
            final_ratio_list.append([1.0] * args.steps)

    gater_tensor = torch.tensor(final_gater_list, dtype=torch.long) # [L, T]
    ratio_tensor = torch.tensor(final_ratio_list, dtype=torch.float32) # [L, T]
    
    fuse_ratio = (gater_tensor == 1).float().mean().item()
    avg_keep_ratio = ratio_tensor.mean().item()

    print("-" * 30)
    print(f"LUT Statistics:")
    print(f"  Gater Shape: {gater_tensor.shape} (Expected: [{total_layers}, {args.steps}])")
    print(f"  Fuse Ratio:  {fuse_ratio:.2%}")
    print(f"  Ratio Shape: {ratio_tensor.shape}")
    print(f"  Avg Ratio:   {avg_keep_ratio:.2%}")
    print("-" * 30)
    
    for i in range(total_layers):
        print(f"Layer {i+1}: {gater_tensor[i].tolist()}")
        print(f"Layer {i+1}: {ratio_tensor[i].tolist()}")

    save_data = {
        "gater": gater_tensor,
        "ratio": ratio_tensor
    }

    save_path = args.output_path
    if not save_path:
        save_path = f"gater_lut_{args.steps}steps.pt"
    
    torch.save(save_data, save_path)
    print(f"Saved Full LUT to: {save_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt_path", type=str, required=True)
    parser.add_argument("--base_model_path", type=str, default=None)
    parser.add_argument("--output_path", type=str, default=None)
    parser.add_argument("--steps", type=int, default=20)
    parser.add_argument("--height", type=int, default=1024)
    parser.add_argument("--width", type=int, default=1024)
    parser.add_argument("--dtype", type=str, default="bf16")
    parser.add_argument("--optimize_hooks", action="store_true", default=True)
    parser.add_argument("--merge_lora", action="store_true", default=True)
    
    args = parser.parse_args()
    generate_lut(args)