import torch
import time
import argparse
import os
import sys
import json
import statistics
import gc
from typing import Optional

from shiva_dit.utils.inference_utils import load_pipeline_for_inference
from shiva_dit.utils.utils import process_quick_start
from accelerate.utils import set_seed

try:
    from calflops import calculate_flops
    CALFLOPS_AVAILABLE = True
except ImportError:
    CALFLOPS_AVAILABLE = False
    print("Warning: 'calflops' not found. FLOPs measurement will be skipped.")
    print("Install via: pip install calflops")

def load_prompts(jsonl_path, limit=None):
    prompts = []
    if not os.path.exists(jsonl_path):
        print(f"Warning: JSONL file not found at {jsonl_path}. Using default prompt.")
        return None
    
    print(f"Loading prompts from {jsonl_path}...")
    try:
        with open(jsonl_path, "r", encoding="utf-8") as f:
            for i, line in enumerate(f):
                if limit and i >= limit:
                    break
                item = json.loads(line)
                if "text" in item:
                    prompts.append(item["text"])
    except Exception as e:
        print(f"Error reading JSONL: {e}")
        return None
    
    print(f"Loaded {len(prompts)} prompts.")
    return prompts

def count_parameters(model):
    total_params = sum(p.numel() for p in model.parameters())
    return total_params

def format_metric(value, base_unit=""):
    if value is None:
        return "N/A"
    if isinstance(value, str):
        return value
        
    if value >= 1e12:
        return f"{value / 1e12:.2f} T{base_unit}"
    elif value >= 1e9:
        return f"{value / 1e9:.2f} G{base_unit}"
    elif value >= 1e6:
        return f"{value / 1e6:.2f} M{base_unit}"
    elif value >= 1e3:
        return f"{value / 1e3:.2f} K{base_unit}"
    else:
        return f"{value:.2f} {base_unit}"

class FlopsMeasurer:
    def __init__(self, model):
        self.model = model
        self.flops = None
        self.macs = None
        self.params = None
        self.hook_handle = None

    def enable(self):
        self.hook_handle = self.model.register_forward_pre_hook(self._hook_fn, with_kwargs=True)

    def _hook_fn(self, module, args, kwargs):
        if self.hook_handle:
            self.hook_handle.remove()
            self.hook_handle = None

        if self.flops is not None:
            return

        print("\n[FLOPs] Capturing input for FLOPs calculation (DiT Single Step)...")
        try:
            clean_kwargs = {}
            for k, v in kwargs.items():
                if isinstance(v, torch.Tensor):
                    clean_kwargs[k] = v
                elif isinstance(v, (int, float)) and not isinstance(v, bool):
                    try:
                        clean_kwargs[k] = torch.tensor(v)
                    except:
                        pass
            
            flops, macs, params = calculate_flops(
                model=module,
                args=args,
                kwargs=clean_kwargs,
                print_results=False, 
                output_as_string=False,
                output_precision=4
            )
            self.flops = flops
            self.macs = macs
            self.params = params
            
            print(f"[FLOPs] Success: {format_metric(flops, 'FLOPs')}")
            
        except Exception as e:
            print(f"[FLOPs] Failed to calculate FLOPs: {e}")
            self.flops = "Error"
            self.macs = "Error"

def benchmark(args):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = torch.bfloat16 if args.dtype == "bf16" else torch.float16
    
    set_seed(42)

    pipe = load_pipeline_for_inference(
        ckpt_path=args.ckpt_path,
        base_model_path=args.base_model_path,
        model_type=args.model_type,
        device=device,
        dtype=dtype,
        gater_lut_path=args.gater_lut,
        optimize_hooks=args.optimize_hooks,
        merge_lora=args.merge_lora
    )
    
    model = None
    if hasattr(pipe, "transformer"):
        model = pipe.transformer
    elif hasattr(pipe, "unet"):
        model = pipe.unet
    
    total_p = 0
    if model:
        total_p = count_parameters(model)
    
    prompts = load_prompts(args.jsonl_path, args.limit)
    if prompts is None:
        prompts = ["A high quality photo of a cat", "A dog playing in the park"]
        if args.limit: prompts = prompts[:args.limit]

    flops_measurer = None
    if args.measure_flops and CALFLOPS_AVAILABLE and model:
        flops_measurer = FlopsMeasurer(model)
        flops_measurer.enable() 

    print("-" * 50)
    print("Warming up (and measuring FLOPs if enabled)...")
    warmup_steps = 4 
    _ = process_quick_start(pipe, prompt=prompts[0], num_inference_steps=warmup_steps)
    torch.cuda.synchronize()
    
    print("-" * 50)
    print(f"Running End-to-End Inference Benchmark on {len(prompts)} items...")
    
    latencies = []
    peak_memories = []
    
    save_dir = "benchmark_samples"
    if args.save_images:
        os.makedirs(save_dir, exist_ok=True)

    torch.cuda.reset_peak_memory_stats()
    torch.cuda.empty_cache()
    
    for i, prompt in enumerate(prompts):
        gc.collect()
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
        
        start_event = torch.cuda.Event(enable_timing=True)
        end_event = torch.cuda.Event(enable_timing=True)
        
        start_event.record()
        
        kwargs = {"prompt": prompt}
        if args.steps > 0:
            kwargs["num_inference_steps"] = args.steps
        if args.height > 0:
             kwargs["height"] = args.height
        if args.width > 0:
             kwargs["width"] = args.width
            
        image = process_quick_start(pipe, **kwargs)
        
        end_event.record()
        torch.cuda.synchronize()
        
        latency_ms = start_event.elapsed_time(end_event)
        latencies.append(latency_ms)
        
        current_peak = torch.cuda.max_memory_allocated() / 1024 / 1024 # MB
        peak_memories.append(current_peak)
        
        print(f"  [{i+1}/{len(prompts)}] E2E Time: {latency_ms:.2f}ms | Peak Mem: {current_peak:.0f}MB")

        if args.save_images:
            safe_name = "".join(c if c.isalnum() else "_" for c in prompt)[:30]
            image.save(os.path.join(save_dir, f"{i:03d}_{safe_name}.png"))

    avg_latency = statistics.mean(latencies)
    avg_memory = statistics.mean(peak_memories)

    display_name = os.path.basename(args.ckpt_path) if args.ckpt_path else "Baseline (No Checkpoint)"

    print("\n" + "=" * 20 + " Benchmark Report " + "=" * 20)
    print(f"Model Source: {display_name}")
    if args.model_type:
        print(f"Model Type:   {args.model_type}")
    print(f"Optimization: Merge LoRA={args.merge_lora}, Skip Layer0={args.optimize_hooks}")
    print(f"Resolution:   {args.height}x{args.width}")
    print(f"Steps:        {args.steps if args.steps > 0 else 'Default'}")
    print("-" * 50)
    
    if model:
        print(f"Model Parameters (DiT):")
        print(f"  Total:     {total_p / 1e9:.3f} B")
    
    if flops_measurer and flops_measurer.flops not in [None, "Error"]:
        print(f"Computational Cost (DiT Single Forward):")
        print(f"  FLOPs:     {format_metric(flops_measurer.flops, 'FLOPs')}")
        print(f"  MACs:      {format_metric(flops_measurer.macs, 'MACs')}")
    elif args.measure_flops:
        print("Computational Cost: Measurement failed or skipped.")

    print(f"System Performance (Avg over {len(prompts)} runs):")
    print(f"  Peak VRAM:          {avg_memory:.2f} MB")
    print(f"  End-to-End Latency: {avg_latency:.2f} ms ({avg_latency/1000:.2f} s)")
    print("=" * 58)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt_path", type=str, default=None, help="Path to checkpoint.pt. If None, runs baseline.")
    parser.add_argument("--model_type", type=str, default=None, help="Required if ckpt_path is None (e.g. 'flux-dev').")
    parser.add_argument("--base_model_path", type=str, default=None)
    parser.add_argument("--jsonl_path", type=str, default="test/training/dataset/meta.jsonl", help="Dataset jsonl file")
    parser.add_argument("--gater_lut", type=str, default=None, help="Path to gater LUT .pt file")
    
    parser.add_argument("--limit", type=int, default=None, help="Limit number of samples to test")
    parser.add_argument("--steps", type=int, default=-1, help="Override inference steps (default uses pipeline default)")
    parser.add_argument("--height", type=int, default=1024)
    parser.add_argument("--width", type=int, default=1024)
    parser.add_argument("--dtype", type=str, default="bf16", choices=["bf16", "fp16", "fp32"])
    
    parser.add_argument("--optimize_hooks", action="store_true", default=True, help="Enable Shiva optimizations (e.g. skip layer 0)")
    parser.add_argument("--no_optimize_hooks", action="store_false", dest="optimize_hooks")
    
    parser.add_argument("--merge_lora", action="store_true", default=True, help="Merge LoRA weights for speed")
    parser.add_argument("--no_merge_lora", action="store_false", dest="merge_lora")
    
    parser.add_argument("--measure_flops", action="store_true", default=True, help="Enable FLOPs measurement (requires calflops)")
    parser.add_argument("--no_measure_flops", action="store_false", dest="measure_flops")
    
    parser.add_argument("--save_images", action="store_true", help="Save generated images to disk")
    
    args = parser.parse_args()
    
    if args.ckpt_path is None and args.model_type is None:
        parser.error("--model_type is required when --ckpt_path is not provided.")
        
    benchmark(args)