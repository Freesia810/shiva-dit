import os
import json
import torch
import numpy as np
import pandas as pd
import lpips
import argparse
import time
import torch.multiprocessing as mp
from PIL import Image
from tqdm import tqdm
from itertools import product

from shiva_dit.utils import get_pipeline, process_quick_start
from shiva_dit.hook.hook_manager import HookManager
from shiva_dit.shiva_hook import ShivaDitBlockHook

# SEARCH_SPACE = {
#     "context_pool_size": [64, 16, 4, None],
#     "fixed_context_ratio": [(1.0, 0.1), (1.0, 0.05)],
#     "context_ratio_mode": ["cosine", "linear"],
#     "temporal_interval": [1, 10, 25, 50],
#     "spatial_group_size": [1],
# }

SEARCH_SPACE = {
    "context_pool_size": [48, 32], # 64, 16
    "fixed_context_ratio": [(1.0, 0.1)],
    "context_ratio_mode": ["cosine"],
    "temporal_interval": [2, 4, 6, 8], # 1, 10
    "spatial_group_size": [1, 2, 3],
}

BASE_SHIVA_CONFIG = {
    "shiva_mode": "context",
    "context_pruner_mode": "free",
    "context_ratio_mode": "linear",
    "init_Ts": 10.0,
    "init_Tk": 10.0,
    "context_shared_config": {
        "spatial_step_thresh": 1.0,
        "spatial_group_size": 1,
        "temporal_layer_thresh": 60,
        "temporal_interval": 1,
    },
}


def parse_args():
    parser = argparse.ArgumentParser(
        description="Shiva-DiT Hyperparameter Search Script"
    )

    parser.add_argument(
        "--dataset",
        type=str,
        required=True,
        help="Path to input jsonl file (e.g., test/dataset/context_qwen.jsonl)",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="search_results",
        help="Directory to save results and images",
    )
    parser.add_argument(
        "--image_root",
        type=str,
        default="",
        help="Root directory for input images (required for qwen-image-edit task)",
    )

    parser.add_argument(
        "--model",
        type=str,
        default="qwen-image",
        choices=["qwen-image", "qwen-image-edit", "flux", "sd3"],
        help="Model type",
    )
    parser.add_argument(
        "--max_seq_len",
        type=int,
        default=4096,
        help="Max sequence length for model input",
    )
    parser.add_argument(
        "--height", type=int, default=1024, help="Image height (default: 1024)"
    )
    parser.add_argument(
        "--width", type=int, default=1024, help="Image width (default: 1024)"
    )
    parser.add_argument(
        "--num_images_per_prompt",
        type=int,
        default=1,
        help="Number of images per prompt (default: 1)",
    )
    parser.add_argument("--device", type=str, default="cuda", help="Device to use")

    parser.add_argument(
        "--num_prompts",
        type=int,
        default=10,
        help="Number of prompts to use for evaluation (default: 10)",
    )
    parser.add_argument(
        "--seed", type=int, default=42, help="Random seed for generation"
    )

    return parser.parse_args()


def load_prompts(dataset_path, limit=None):
    if not os.path.exists(dataset_path):
        raise FileNotFoundError(f"Dataset file not found: {dataset_path}")

    print(f"[Data] Loading prompts from {dataset_path}")
    prompts = []
    with open(dataset_path, "r") as f:
        for line in f:
            if line.strip():
                try:
                    item = json.loads(line)
                    if "text" in item:
                        item["prompt"] = item["text"]

                    if "id" not in item:
                        if "file_name" in item and item["file_name"]:
                            item["id"] = os.path.splitext(item["file_name"])[0]
                        else:
                            item["id"] = str(hash(item.get("prompt", "")))[:8]
                    prompts.append(item)
                except json.JSONDecodeError:
                    continue

    if limit:
        print(f"[Data] Limiting to first {limit} prompts.")
        return prompts[:limit]
    return prompts


def get_gt_path(gt_dir, prompt_id):
    return os.path.join(gt_dir, f"{prompt_id}.png")


def generate_ground_truth(pipe, prompts, args, gt_dir):
    print(f"\n=== Checking Ground Truth in {gt_dir} ===")

    HookManager.disable()
    generator = torch.Generator(device=args.device)

    generated_count = 0
    for idx, item in enumerate(tqdm(prompts, desc="Checking GT")):
        p_id = str(item["id"])
        save_path = get_gt_path(gt_dir, p_id)

        generator.manual_seed(args.seed + idx)

        if os.path.exists(save_path):
            continue

        kwargs = {
            "prompt": item["prompt"],
            "max_sequence_length": args.max_seq_len,
            "generator": generator,
            "height": args.height,
            "width": args.width,
            "num_images_per_prompt": args.num_images_per_prompt,
        }

        if args.model == "qwen-image-edit":
            if not args.image_root:
                raise ValueError("For qwen-image-edit, you must provide --image_root")

            file_name = item.get("file_name")
            if not file_name:
                raise ValueError(f"Item {p_id} missing 'file_name' for edit task")

            img_input = os.path.join(args.image_root, file_name)
            kwargs["image"] = Image.open(img_input).convert("RGB")

        with torch.no_grad():
            res = process_quick_start(pipe, **kwargs)

        res.save(save_path)
        generated_count += 1

    if generated_count > 0:
        print(f"[GT] Generated {generated_count} missing ground truth images.")
    else:
        print("[GT] All ground truth images exist.")


def build_config_combinations(search_space):
    keys = search_space.keys()
    values = search_space.values()
    combinations = []
    for v in product(*values):
        combinations.append(dict(zip(keys, v)))
    return combinations

def worker(rank, world_size, args, configs, prompts, gt_dir):
    device = torch.device(f"cuda:{rank}")
    print(f"[Worker {rank}] Started. Processing on {device}")
    
    print(f"[Worker {rank}] Loading Model...")
    pipe, model_attrs = get_pipeline(args.model)
    pipe.to(device)
    
    loss_fn_alex = lpips.LPIPS(net="alex").to(device)

    my_configs = configs[rank::world_size]
    print(f"[Worker {rank}] Assigned {len(my_configs)} configs out of {len(configs)} total.")

    results = []

    generator = torch.Generator(device=device)
    
    for idx, cfg in enumerate(my_configs):
        original_idx = configs.index(cfg) 
        config_name = f"cfg_{original_idx:03d}"
        
        print(f"\n[Worker {rank}] >>> Running Config {config_name}: {cfg}")

        current_shiva_config = BASE_SHIVA_CONFIG.copy()
        current_shiva_config.update(cfg)
        
        shared_cfg = current_shiva_config["context_shared_config"].copy()
        if "spatial_group_size" in cfg:
            shared_cfg["spatial_group_size"] = cfg["spatial_group_size"]
            del current_shiva_config["spatial_group_size"]
        if "temporal_interval" in cfg:
            shared_cfg["temporal_interval"] = cfg["temporal_interval"]
            del current_shiva_config["temporal_interval"]
        current_shiva_config["context_shared_config"] = shared_cfg

        HookManager.configure_hooks(
            model_type=args.model,
            default=(ShivaDitBlockHook, current_shiva_config),
        )

        lpips_scores = []
        latencies = []

        with HookManager.enabled():
            HookManager.inject_hooks(model_type=args.model, pipe=pipe, model_attrs=model_attrs)

            for p_idx, item in enumerate(tqdm(prompts, desc=f"[W{rank}] Eval {config_name}", position=rank)):
                p_id = str(item["id"])
                gt_path = get_gt_path(gt_dir, p_id)
                generator.manual_seed(args.seed + p_idx)
                kwargs = {
                    "prompt": item["prompt"],
                    "max_sequence_length": args.max_seq_len,
                    "generator": generator,
                    "height": args.height,
                    "width": args.width,
                    "num_images_per_prompt": args.num_images_per_prompt,
                }
                
                if args.model == "qwen-image-edit":
                    file_name = item.get("file_name")
                    img_input = os.path.join(args.image_root, file_name)
                    kwargs["image"] = Image.open(img_input).convert("RGB")

                torch.cuda.synchronize(device)
                start_t = time.time()
                res = process_quick_start(pipe, **kwargs)
                torch.cuda.synchronize(device)
                latencies.append(time.time() - start_t)

                try:
                    pred_img = lpips.im2tensor(np.array(res)).to(device) # device
                    gt_img = lpips.im2tensor(lpips.load_image(gt_path)).to(device) # device
                    dist = loss_fn_alex(pred_img, gt_img).item()
                    lpips_scores.append(dist)
                except Exception as e:
                    print(f"[Worker {rank} Error] {e}")
                    lpips_scores.append(1.0)

        avg_lpips = np.mean(lpips_scores) if lpips_scores else 1.0
        avg_latency = np.mean(latencies) if latencies else 0.0
        record = {k: v for k, v in cfg.items()}
        record["config_id"] = config_name
        record["avg_lpips"] = avg_lpips
        record["avg_latency"] = avg_latency
        results.append(record)

    if results:
        df = pd.DataFrame(results)
        csv_path = os.path.join(args.output_dir, f"search_results_{args.model}_rank{rank}.csv")
        df.to_csv(csv_path, index=False)
        print(f"[Worker {rank}] Saved partial results to {csv_path}")

def run_search():
    args = parse_args()
    
    gt_dir = os.path.join(args.output_dir, "ground_truth", args.model)
    pred_dir_base = os.path.join(args.output_dir, "predictions", args.model)
    os.makedirs(gt_dir, exist_ok=True)
    os.makedirs(pred_dir_base, exist_ok=True)

    prompts = load_prompts(args.dataset, limit=args.num_prompts)
    
    print(f"[Init] Checking Ground Truth on Master Process...")
    temp_pipe, _ = get_pipeline(args.model)
    temp_pipe.to("cuda:0") 
    generate_ground_truth(temp_pipe, prompts, args, gt_dir)
    del temp_pipe
    torch.cuda.empty_cache()
    print("[Init] GT Check Done. Releasing memory for workers.")

    configs = build_config_combinations(SEARCH_SPACE)
    print(f"\n[Search] Start Grid Search over {len(configs)} configurations.")

    world_size = torch.cuda.device_count()
    if world_size < 1:
        raise RuntimeError("No CUDA devices available!")
    
    print(f"[System] Spawning {world_size} workers...")
    mp.spawn(
        worker,
        args=(world_size, args, configs, prompts, gt_dir),
        nprocs=world_size,
        join=True
    )

    print("\n[Merge] All workers finished. Merging results...")
    all_results = []
    for rank in range(world_size):
        part_csv = os.path.join(args.output_dir, f"search_results_{args.model}_rank{rank}.csv")
        if os.path.exists(part_csv):
            df_part = pd.read_csv(part_csv)
            all_results.append(df_part)
    
    if all_results:
        final_df = pd.concat(all_results, ignore_index=True)
        final_df = final_df.sort_values("config_id")
        
        final_csv_path = os.path.join(args.output_dir, f"search_results_{args.model}.csv")
        final_df.to_csv(final_csv_path, index=False)
        print(f"Saved merged results to {final_csv_path}")
        
        best_acc = final_df.sort_values("avg_lpips").iloc[0]
        print(f"Best Accuracy Config:\n{best_acc}")
        
        # for rank in range(world_size):
        #     os.remove(os.path.join(args.output_dir, f"search_results_{args.model}_rank{rank}.csv"))
    else:
        print("[Warning] No results collected.")

if __name__ == "__main__":
    run_search()
