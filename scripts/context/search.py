import os
import json
import torch
import numpy as np
import pandas as pd
import lpips
import argparse
import time
import copy
import torch.distributed as dist
from PIL import Image
from tqdm import tqdm
import itertools

from shiva_dit.utils import get_pipeline, process_quick_start
from shiva_dit.hook.hook_manager import HookManager
from shiva_dit.shiva_hook import ShivaDitBlockHook

SEARCH_SPACE = {
    "context_pool_size": [64, 32, 16],
    "fixed_context_ratio": [(1.0, 0.1)],
    "context_ratio_mode": ["cosine"],
    "temporal_layer_thresh": [20, 40, 60],
    "temporal_interval": [1, 2, 4, 8],
    "spatial_step_thresh": [0.2, 0.5],
    "spatial_group_size": [1, 2],
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
    parser = argparse.ArgumentParser(description="Shiva-DiT Hyperparameter Search Script (Torchrun)")
    parser.add_argument("--dataset", type=str, required=True, help="Path to input jsonl file")
    parser.add_argument("--output_dir", type=str, default="search_results", help="Directory to save results")
    parser.add_argument("--image_root", type=str, default="", help="Root directory for input images")
    parser.add_argument("--model", type=str, default="qwen-image", choices=["qwen-image", "qwen-image-edit", "flux", "sd3"])
    parser.add_argument("--max_seq_len", type=int, default=4096)
    parser.add_argument("--height", type=int, default=1024)
    parser.add_argument("--width", type=int, default=1024)
    parser.add_argument("--num_images_per_prompt", type=int, default=1)
    parser.add_argument("--num_prompts", type=int, default=10)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()

def load_prompts(dataset_path, limit=None):
    if not os.path.exists(dataset_path):
        raise FileNotFoundError(f"Dataset file not found: {dataset_path}")
    
    prompts = []
    with open(dataset_path, "r") as f:
        for line in f:
            if line.strip():
                try:
                    item = json.loads(line)
                    if "text" in item: item["prompt"] = item["text"]
                    if "id" not in item:
                        item["id"] = os.path.splitext(item["file_name"])[0] if item.get("file_name") else str(hash(item.get("prompt", "")))[:8]
                    prompts.append(item)
                except json.JSONDecodeError:
                    continue
    return prompts[:limit] if limit else prompts

def get_gt_path(gt_dir, prompt_id):
    return os.path.join(gt_dir, f"{prompt_id}.png")

def generate_ground_truth(pipe, prompts, args, gt_dir):
    HookManager.disable()
    generator = torch.Generator(device=pipe.device)
    
    for idx, item in enumerate(tqdm(prompts, desc="[Rank 0] Checking GT")):
        p_id = str(item["id"])
        save_path = get_gt_path(gt_dir, p_id)
        
        if os.path.exists(save_path):
            continue

        generator.manual_seed(args.seed + idx)
        kwargs = {
            "prompt": item["prompt"],
            "max_sequence_length": args.max_seq_len,
            "generator": generator,
            "height": args.height,
            "width": args.width,
            "num_images_per_prompt": args.num_images_per_prompt,
        }

        if args.model == "qwen-image-edit":
            img_input = os.path.join(args.image_root, item.get("file_name"))
            kwargs["image"] = Image.open(img_input).convert("RGB")

        with torch.no_grad():
            res = process_quick_start(pipe, **kwargs)
        res.save(save_path)

def build_config_combinations(search_space):
    keys = list(search_space.keys())
    values = list(search_space.values())

    default_spatial_thresh = search_space["spatial_step_thresh"][0]
    default_temporal_thresh = search_space["temporal_layer_thresh"][0]

    full_combinations = list(itertools.product(*values))
    print(f"[Config] All: {len(full_combinations)}")

    final_combinations = []
    
    for v in full_combinations:
        cfg = dict(zip(keys, v))
        
        if cfg["spatial_group_size"] == 1:
            if cfg["spatial_step_thresh"] != default_spatial_thresh:
                continue

        if cfg["temporal_interval"] == 1:
            if cfg["temporal_layer_thresh"] != default_temporal_thresh:
                continue
        
        final_combinations.append(cfg)

    print(f"[Config] Actual: {len(final_combinations)}")
    return final_combinations

def main():
    rank = int(os.environ["RANK"])
    local_rank = int(os.environ["LOCAL_RANK"])
    world_size = int(os.environ["WORLD_SIZE"])
    
    torch.cuda.set_device(local_rank)
    device = torch.device(f"cuda:{local_rank}")

    dist.init_process_group(
        backend="nccl", 
        device_id=torch.device(f"cuda:{local_rank}")
    )
    
    args = parse_args()

    gt_dir = os.path.join(args.output_dir, "ground_truth", args.model)
    if rank == 0:
        os.makedirs(gt_dir, exist_ok=True)
        os.makedirs(os.path.join(args.output_dir, "predictions", args.model), exist_ok=True)
    
    dist.barrier()

    prompts = load_prompts(args.dataset, limit=args.num_prompts)
    configs = build_config_combinations(SEARCH_SPACE)

    if rank == 0:
        print(f"[Init] Rank 0 generating Ground Truth...")
        gt_pipe, _ = get_pipeline(args.model, device=device)
        generate_ground_truth(gt_pipe, prompts, args, gt_dir)
        
        del gt_pipe
        torch.cuda.empty_cache()
        print("[Init] GT Check Done. Releasing memory.")
    
    dist.barrier()

    print(f"[Rank {rank}] Loading Model for evaluation...")
    pipe, model_attrs = get_pipeline(args.model, device=device)
    loss_fn_alex = lpips.LPIPS(net="alex").to(device)

    my_configs = configs[rank::world_size]
    print(f"[Rank {rank}] Assigned {len(my_configs)} configs.")

    results = []
    generator = torch.Generator(device=device)

    for cfg in my_configs:
        original_idx = configs.index(cfg)
        config_name = f"cfg_{original_idx:03d}"
        
        current_shiva_config = copy.deepcopy(BASE_SHIVA_CONFIG)
        current_shiva_config.update(cfg)
        
        shared_cfg = current_shiva_config["context_shared_config"].copy()
        if "spatial_step_thresh" in cfg:
            shared_cfg["spatial_step_thresh"] = cfg["spatial_step_thresh"]
            del current_shiva_config["spatial_step_thresh"]
        if "spatial_group_size" in cfg:
            shared_cfg["spatial_group_size"] = cfg["spatial_group_size"]
            del current_shiva_config["spatial_group_size"]
        if "temporal_layer_thresh" in cfg:
            shared_cfg["temporal_layer_thresh"] = cfg["temporal_layer_thresh"]
            del current_shiva_config["temporal_layer_thresh"]
        if "temporal_interval" in cfg:
            shared_cfg["temporal_interval"] = cfg["temporal_interval"]
            del current_shiva_config["temporal_interval"]
        current_shiva_config["context_shared_config"] = shared_cfg

        print(f"[Rank {rank}] Config_name: {config_name}, Config: {current_shiva_config}" )

        HookManager.configure_hooks(
            model_type=args.model,
            default=(ShivaDitBlockHook, current_shiva_config),
        )

        lpips_scores = []
        latencies = []

        with HookManager.enabled():
            HookManager.inject_hooks(model_type=args.model, pipe=pipe, model_attrs=model_attrs)

            for p_idx, item in enumerate(tqdm(prompts, desc=f"[R{rank}] {config_name}", position=local_rank)):
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
                    img_input = os.path.join(args.image_root, item.get("file_name"))
                    kwargs["image"] = Image.open(img_input).convert("RGB")

                torch.cuda.synchronize(device)
                start_t = time.time()
                res = process_quick_start(pipe, **kwargs)
                torch.cuda.synchronize(device)
                latencies.append(time.time() - start_t)

                try:
                    pred_np = np.array(res).astype(np.float32) / 255.0 * 2.0 - 1.0
                    pred_img = torch.from_numpy(pred_np).permute(2, 0, 1).unsqueeze(0).to(device) # (1, C, H, W)
                    gt_img = lpips.im2tensor(lpips.load_image(gt_path)).to(device)
                    dist_val = loss_fn_alex(pred_img, gt_img).item()
                    lpips_scores.append(dist_val)
                except Exception as e:
                    print(f"[Rank {rank} Error] {e}")
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
        print(f"[Rank {rank}] Saved results to {csv_path}")
    
    dist.barrier()

    if rank == 0:
        print("\n[Merge] Merging results from all ranks...")
        all_dfs = []
        for r in range(world_size):
            p_csv = os.path.join(args.output_dir, f"search_results_{args.model}_rank{r}.csv")
            if os.path.exists(p_csv):
                all_dfs.append(pd.read_csv(p_csv))
                # os.remove(p_csv)
        
        if all_dfs:
            final_df = pd.concat(all_dfs, ignore_index=True)
            final_df = final_df.sort_values("config_id")
            final_csv = os.path.join(args.output_dir, f"search_results_{args.model}.csv")
            final_df.to_csv(final_csv, index=False)
            
            best_cfg = final_df.sort_values("avg_lpips").iloc[0]
            print(f"Saved merged results to {final_csv}")
            print(f"Best Config:\n{best_cfg}")
        else:
            print("[Warning] No results found.")

    dist.destroy_process_group()

if __name__ == "__main__":
    main()