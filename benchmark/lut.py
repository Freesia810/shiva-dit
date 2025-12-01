import torch
import argparse
import os
import sys
import numpy as np

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
        optimize_hooks=False,
        merge_lora=args.merge_lora
    )
    
    hooks = list(HookManager.get_all_hooks().values())
    if not hooks:
        raise RuntimeError("No hooks found.")
    
    hooks.sort(key=lambda h: h.layer_idx)
    num_layers = len(hooks)
    print(f"Found {num_layers} hooked layers.")
    
    gater_policy = hooks[0].gater_policy
    ratio_policy = hooks[0].image_ratio_policy
    
    captured_gater_decisions = []
    captured_ratios = []

    # === Hook for Gater ===
    def capture_gater_hook(module, args, output):
        # output: gate_logits [B, 2]
        logits = output
        decision = torch.argmax(logits, dim=-1).cpu().item() # 0 or 1
        captured_gater_decisions.append(decision)

    # === Hook for Ratio ===
    def capture_ratio_hook(module, args, output):
        # output: logits [B, 1]
        logits = output
        ratio_val = torch.sigmoid(logits).mean().cpu().item() # float 0-1
        captured_ratios.append(ratio_val)

    handle_gater = gater_policy.register_forward_hook(capture_gater_hook)
    handle_ratio = ratio_policy.register_forward_hook(capture_ratio_hook)
    
    print(f"Running inference (Steps={args.steps})...")
    prompt = "A high quality photo of a cat" 
    
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
    
    total_gater = len(captured_gater_decisions)
    expected = num_layers * args.steps
    
    if total_gater != expected:
        if total_gater % num_layers == 0:
            actual_steps = total_gater // num_layers
            print(f"Adjusting inferred steps to {actual_steps}.")
            args.steps = actual_steps
        else:
            print(f"Warning: Expected {expected} gater decisions, got {total_gater}.")

    gater_tensor = torch.tensor(captured_gater_decisions, dtype=torch.long)
    gater_tensor = gater_tensor.view(args.steps, num_layers).t().contiguous() # [L, T]

    total_ratio = len(captured_ratios)
    ratio_tensor = torch.tensor(captured_ratios, dtype=torch.float32)
    if total_ratio == total_gater:
        ratio_tensor = ratio_tensor.view(args.steps, num_layers).t().contiguous() # [L, T]
    else:
        print(f"Warning: Ratio captured {total_ratio} vs Gater {total_gater}. Saving raw ratio list.")
    
    fuse_ratio = (gater_tensor == 1).float().mean().item()
    avg_keep_ratio = ratio_tensor.mean().item()

    print("-" * 30)
    print(f"LUT Statistics:")
    print(f"  Gater Shape: {gater_tensor.shape} | Fuse Ratio: {fuse_ratio:.2%}")
    print(f"  Ratio Shape: {ratio_tensor.shape} | Avg Ratio:  {avg_keep_ratio:.2%}")
    print("-" * 30)

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
    parser.add_argument("--merge_lora", action="store_true", default=True)
    
    args = parser.parse_args()
    generate_lut(args)