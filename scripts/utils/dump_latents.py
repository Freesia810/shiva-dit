import torch
import os
import sys
import json
import argparse
from tqdm import tqdm
from accelerate.utils import set_seed

sys.path.append(os.getcwd()) 

try:
    from shiva_dit.utils.inference_utils import load_pipeline_for_inference
except ImportError:
    from shiva_dit.utils.utils import get_pipeline
    load_pipeline_for_inference = None

def load_data(jsonl_path, limit=None):
    data_items = []
    print(f"Loading prompts from {jsonl_path}...")
    with open(jsonl_path, "r", encoding="utf-8") as f:
        for i, line in enumerate(f):
            if limit and i >= limit:
                break
            item = json.loads(line)
            if "text" in item:
                data_items.append({"text": item["text"], "global_index": i})
    print(f"Loaded {len(data_items)} items.")
    return data_items

def main(args):
    os.makedirs(args.output_dir, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dtype = torch.bfloat16 if args.dtype == "bf16" else torch.float16

    print(f"Loading pipeline to ensure latent consistency...")
    
    if load_pipeline_for_inference:
        pipe = load_pipeline_for_inference(
            ckpt_path=args.ckpt_path,
            base_model_path=args.base_model_path,
            model_type=args.model_type,
            device=device,
            dtype=dtype,
        )
    else:
        print("Using shiva_dit.utils.utils.get_pipeline...")
        pipe, _ = get_pipeline(
            model_type=args.model_type,
            model_path=args.base_model_path or args.ckpt_path,
            dtype=dtype,
            device=device
        )

    if hasattr(pipe, "vae"):
        pipe.vae.requires_grad_(False)
    if hasattr(pipe, "transformer"):
        pipe.transformer.requires_grad_(False)

    try:
        num_channels_latents = pipe.transformer.config.in_channels
    except AttributeError:
        if hasattr(pipe, "unet"):
            num_channels_latents = pipe.unet.config.in_channels
        else:
            print("Warning: Could not infer latent channels from config, defaulting to 16 (SD3/Flux standard).")
            num_channels_latents = 16
    
    if not hasattr(pipe, "prepare_latents"):
        raise AttributeError("The loaded pipeline does not have a 'prepare_latents' method!")

    print("-" * 50)
    print(f"Dumping Latents using Pipeline: {pipe.__class__.__name__}")
    print(f"Inferred Latent Channels: {num_channels_latents}")
    print(f"Output Dir: {args.output_dir}")
    print("-" * 50)

    data_items = load_data(args.jsonl_path, args.limit)
    pbar = tqdm(total=len(data_items) * args.samples_per_prompt, desc="Dumping")

    for item in data_items:
        i = item["global_index"]
        
        for j in range(args.samples_per_prompt):
            current_seed = args.seed + (i * 100) + j
            
            file_name = f"latent_p{i:06d}_s{j:02d}.pt"
            save_path = os.path.join(args.output_dir, file_name)
            
            if not os.path.exists(save_path):
                set_seed(current_seed)
                generator = torch.Generator(device=device).manual_seed(current_seed)
                
                with torch.no_grad():
                    latent = pipe.prepare_latents(
                        batch_size=1,
                        num_channels_latents=num_channels_latents,
                        height=args.height,
                        width=args.width,
                        dtype=dtype,
                        device=device,
                        generator=generator,
                        latents=None
                    )
                
                torch.save(latent.clone(), save_path)
            
            pbar.update(1)

    pbar.close()
    print("Done.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt_path", type=str, default=None)
    parser.add_argument("--base_model_path", type=str, default=None)
    parser.add_argument("--model_type", type=str, default="sd3")
    
    parser.add_argument("--jsonl_path", type=str, required=True)
    parser.add_argument("--output_dir", type=str, default="latents_dump")
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--samples_per_prompt", type=int, default=5)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--height", type=int, default=1024)
    parser.add_argument("--width", type=int, default=1024)
    parser.add_argument("--dtype", type=str, default="bf16")
    
    args = parser.parse_args()
    
    if args.ckpt_path is None and args.model_type is None:
        parser.error("Please provide --ckpt_path or --model_type to load the pipeline.")

    main(args)