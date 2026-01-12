import torch
import os
import sys
import json
import argparse
from PIL import Image
from tqdm import tqdm
from types import ModuleType

from shiva_dit.utils.inference_utils import load_pipeline_for_inference
from shiva_dit.utils.utils import process_quick_start
from accelerate.utils import set_seed


def load_data(jsonl_path, image_dir, limit=None):
    """Load data from JSONL file."""
    data_items = []
    if not os.path.exists(jsonl_path):
        print(f"Error: JSONL file not found at {jsonl_path}")
        return []

    print(f"Loading data from {jsonl_path}...")
    try:
        with open(jsonl_path, "r", encoding="utf-8") as f:
            for i, line in enumerate(f):
                if limit and i >= limit:
                    break
                item = json.loads(line)

                img_rel_path = (
                    item.get("file_name") or item.get("image") or item.get("jpg")
                )
                
                full_img_path = None
                if img_rel_path and image_dir:
                    full_img_path = os.path.join(image_dir, img_rel_path)
                
                if "text" in item:
                    data_items.append(
                        {"text": item["text"], "image_path": full_img_path, "global_index": i}
                    )
    except Exception as e:
        print(f"Error reading JSONL: {e}")
        return []

    print(f"Loaded {len(data_items)} items.")
    return data_items


def generate_images(args):
    """Main image generation function."""
    # Multi-GPU Setup
    rank = int(os.environ.get("RANK", 0))
    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    world_size = int(os.environ.get("WORLD_SIZE", 1))

    if torch.cuda.is_available():
        device = f"cuda:{local_rank}"
        torch.cuda.set_device(device)
    else:
        device = "cpu"

    dtype = torch.bfloat16 if args.dtype == "bf16" else torch.float16

    print(f"[Process {rank}/{world_size}] Initialized on {device}")

    # Setup output directory
    if args.force_fixed_ratio > 0:
        if "fixed" not in args.output_dir:
            args.output_dir = f"{args.output_dir}_fixed_{args.force_fixed_ratio}"
    
    save_dir = args.output_dir
    if rank == 0:
        os.makedirs(save_dir, exist_ok=True)
        print(f"Output Directory: {save_dir}")

    # Load data
    all_data_items = load_data(args.jsonl_path, args.image_dir, args.limit)
    if not all_data_items:
        return

    # Data sharding for multi-GPU
    data_items = all_data_items[rank::world_size]
    print(f"[Process {rank}] Processing {len(data_items)} items (subset of {len(all_data_items)})")

    # Load metadata config
    if args.meta_config and os.path.exists(args.meta_config):
        print(f"Found metadata config at {args.meta_config}")
        with open(args.meta_config, "r") as f:
            meta_config = json.load(f)
    else:
        meta_config = None

    # Load pipeline
    print(f"Loading Pipeline...")
    pipe = load_pipeline_for_inference(
        ckpt_path=args.ckpt_path,
        base_model_path=args.base_model_path,
        model_type=args.model_type,
        device=device,
        dtype=dtype,
        gater_lut_path=args.gater_lut,
        optimize_hooks=args.optimize_hooks,
        merge_lora=args.merge_lora,
        meta_config=meta_config,
    )

    # Configure fixed ratio if specified
    if args.force_fixed_ratio > 0:
        print(f"[INFO] Disabling Policy Network...")
        print(f"[INFO] Forcing Fixed Ratio: {args.force_fixed_ratio}")
        from shiva_dit.hook import HookManager
        
        hooks_modified = 0
        for hook in HookManager.get_all_hooks().values():
            if hook.__class__.__name__ == "ShivaDitBlockHook":
                hook.image_ratio_policy = None
                hook.fixed_image_ratio = args.force_fixed_ratio
                hooks_modified += 1
        
        if hooks_modified == 0:
            raise Exception("[Error] Could not find 'shiva_hook' on pipeline. Please check hook registration.")
        else:
            print(f"[SUCCESS] Modified {hooks_modified} hook(s) for Fixed Baseline.")
    else:
        print(f"\n[INFO] Inference Mode: Adaptive (Policy Enabled)")

    total_samples = len(data_items) * args.samples_per_prompt
    
    if rank == 0:
        print("-" * 50)
        print(f"Starting Image Generation: {args.model_type}")
        print(f"Mode: {'FIXED ' + str(args.force_fixed_ratio) if args.force_fixed_ratio > 0 else 'ADAPTIVE'}")
        print(f"Global Prompts: {len(all_data_items)}")
        print(f"Local Prompts (Rank {rank}): {len(data_items)}")
        print(f"Samples per Prompt: {args.samples_per_prompt}")
        print(f"Total Target Samples (Local): {total_samples}")
        print(f"Global Seed: {args.seed}")
        print(f"Latent Source: {args.latent_source_dir if args.latent_source_dir else 'Random (On-the-fly)'}")
        print("-" * 50)

    # Generation loop
    pbar = tqdm(total=total_samples, desc=f"Gen Proc {rank}", position=rank)

    generation_metadata = []

    for item in data_items:
        prompt = item["text"]
        global_idx = item["global_index"]

        for j in range(args.samples_per_prompt):
            current_seed = args.seed + (global_idx * 100) + j
            file_name = f"p{global_idx:04d}_s{j:02d}.png"
            file_path = os.path.join(save_dir, file_name)

            gen_image = None

            # Skip existing files
            if os.path.exists(file_path):
                try:
                    gen_image = Image.open(file_path).convert("RGB")
                    gen_image.load()  # Verify image is valid
                    
                    # Add to metadata even if skipped
                    generation_metadata.append({
                        "file_name": file_name,
                        "prompt": prompt,
                        "global_index": global_idx,
                        "sample_index": j,
                        "seed": current_seed,
                        "status": "skipped_existing"
                    })
                    
                except Exception as e:
                    print(f"[Rank {rank}] Error loading {file_name}, will regenerate: {e}")
                    gen_image = None

            if gen_image is None:
                # Load latents if available
                latents = None
                if args.latent_source_dir:
                    latent_name = f"latent_p{global_idx:06d}_s{j:02d}.pt"
                    latent_path = os.path.join(args.latent_source_dir, latent_name)
                    
                    if os.path.exists(latent_path):
                        try:
                            latents = torch.load(latent_path, map_location=device)
                            latents = latents.to(dtype=dtype)
                        except Exception as e:
                            print(f"[Rank {rank}] Error loading latent {latent_path}: {e}")
                            latents = None
                    else:
                        if rank == 0:
                            print(f"[Warning] Latent file not found: {latent_path}. Falling back to random.")

                # Setup generator
                generator = torch.Generator(device=device).manual_seed(current_seed)
                set_seed(current_seed)

                # Prepare generation kwargs
                kwargs = {"prompt": prompt, "generator": generator}
                if args.steps > 0:
                    kwargs["num_inference_steps"] = args.steps
                if args.height > 0:
                    kwargs["height"] = args.height
                if args.width > 0:
                    kwargs["width"] = args.width
                if args.input_image: 
                    kwargs["image"] = Image.open(args.input_image).convert("RGB")
                if args.max_sequence_length: 
                    kwargs["max_sequence_length"] = args.max_sequence_length
                if latents is not None:
                    kwargs["latents"] = latents

                # Generate image
                gen_image = process_quick_start(pipe, **kwargs)
                gen_image.save(file_path)

                # Add to metadata
                generation_metadata.append({
                    "file_name": file_name,
                    "prompt": prompt,
                    "global_index": global_idx,
                    "sample_index": j,
                    "seed": current_seed,
                    "status": "generated",
                    "latent_source": "loaded" if latents is not None else "random"
                })

            pbar.update(1)

    pbar.close()

    # Save generation metadata
    if rank == 0:
        metadata_path = os.path.join(save_dir, "generation_metadata.json")
        with open(metadata_path, "w") as f:
            json.dump(generation_metadata, f, indent=2)
        print(f"Generation metadata saved to: {metadata_path}")

    print(f"[Process {rank}] Image generation completed. Generated {len(generation_metadata)} images.")


def main():
    parser = argparse.ArgumentParser(description="Independent Image Generation Script")
    
    # Model arguments
    parser.add_argument("--ckpt_path", type=str, default=None)
    parser.add_argument("--model_type", type=str, default=None)
    parser.add_argument("--base_model_path", type=str, default=None)
    parser.add_argument("--gater_lut", type=str, default=None)
    parser.add_argument("--meta_config", type=str, default=None)

    # Data arguments
    parser.add_argument("--jsonl_path", type=str, required=True, help="Path to val.jsonl")
    parser.add_argument("--image_dir", type=str, default=None, 
                       help="Root dir for GT images (only used for loading paths if needed)")
    parser.add_argument("--output_dir", type=str, default="generated_images", 
                       help="Where to save generated images")
    parser.add_argument("--limit", type=int, default=None)

    # Sampling configuration
    parser.add_argument("--samples_per_prompt", type=int, default=5,
                       help="Number of images per prompt")
    parser.add_argument("--seed", type=int, default=42, help="Global base seed")
    parser.add_argument("--steps", type=int, default=-1)
    parser.add_argument("--height", type=int, default=1024)
    parser.add_argument("--width", type=int, default=1024)
    parser.add_argument("--input_image", type=str, default=None)
    parser.add_argument("--max_sequence_length", type=int, default=None)
    parser.add_argument("--dtype", type=str, default="bf16")

    # Shiva-specific arguments
    parser.add_argument("--optimize_hooks", action="store_true", default=True)
    parser.add_argument("--no_optimize_hooks", action="store_false", dest="optimize_hooks")
    parser.add_argument("--merge_lora", action="store_true", default=True)
    parser.add_argument("--no_merge_lora", action="store_false", dest="merge_lora")
    parser.add_argument("--force_fixed_ratio", type=float, default=-1.0, 
                       help="If > 0, disables Policy and forces a fixed ratio. (e.g. 0.64)")
    parser.add_argument("--latent_source_dir", type=str, default=None, 
                       help="Path to directory containing pre-dumped latents")

    args = parser.parse_args()
    generate_images(args)


if __name__ == "__main__":
    main()
