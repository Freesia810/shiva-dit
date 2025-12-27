import os
import sys
import json
import argparse
import glob
import subprocess
import re
import threading
import queue
import time
from tabulate import tabulate
import torch

print_lock = threading.Lock()

def safe_print(*args, **kwargs):
    with print_lock:
        print(*args, **kwargs)

def parse_metrics(metric_file):
    if not os.path.exists(metric_file):
        return None
    try:
        with open(metric_file, 'r') as f:
            return json.load(f)
    except Exception:
        return None

def find_checkpoints(root_dir):
    if not os.path.exists(root_dir):
        safe_print(f"Error: Directory {root_dir} does not exist.")
        sys.exit(1)
        
    candidates = glob.glob(os.path.join(root_dir, "checkpoint_step*"))
    valid_ckpts = []
    
    for path in candidates:
        if os.path.isdir(path):
            match = re.search(r'step[_\-](\d+)', os.path.basename(path))
            step = int(match.group(1)) if match else -1
            valid_ckpts.append({"path": path, "step": step})

    valid_ckpts.sort(key=lambda x: x["step"])
    return valid_ckpts

def worker(gpu_id, task_queue, results, results_lock, args, python_exe, script_path, total_tasks):
    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = str(gpu_id)

    while True:
        try:
            index, ckpt = task_queue.get_nowait()
        except queue.Empty:
            break

        step_name = f"step_{ckpt['step']}"
        ckpt_path = ckpt['path']
        eval_output_dir = os.path.join(args.output_root, f"eval_{step_name}")
        metrics_file = os.path.join(eval_output_dir, "metrics.json")

        skip = False
        if os.path.exists(metrics_file) and not args.force_rerun:
            metrics = parse_metrics(metrics_file)
            if metrics:
                with results_lock:
                    results.append({"step": ckpt['step'], "path": ckpt_path, "metrics": metrics})
                safe_print(f"[GPU {gpu_id}] [{index+1}/{total_tasks}] Skipping {step_name} (Metrics found).")
                skip = True
        
        if not skip:
            safe_print(f"[GPU {gpu_id}] [{index+1}/{total_tasks}] START Evaluating {step_name} ...")
            
            cmd = [
                python_exe, script_path,
                "--ckpt_path", ckpt_path,
                "--jsonl_path", args.jsonl_path,
                "--output_dir", eval_output_dir,
                "--samples_per_prompt", str(args.samples_per_prompt),
                "--limit", str(args.limit),
                "--dtype", args.dtype
            ]
            
            if args.meta_config:
                 cmd.extend(["--meta_config", args.meta_config])
            if args.image_dir:
                 cmd.extend(["--image_dir", args.image_dir])
            if args.model_type:
                cmd.extend(["--model_type", args.model_type])
            if args.base_model_path:
                cmd.extend(["--base_model_path", args.base_model_path])
            if args.fid_stats:
                cmd.extend(["--fid_stats", args.fid_stats])
            if args.gater_lut:
                cmd.extend(["--gater_lut", args.gater_lut])
            
            try:
                subprocess.run(cmd, check=True, env=env, capture_output=False)
                
                metrics = parse_metrics(metrics_file)
                if metrics:
                    with results_lock:
                        results.append({"step": ckpt['step'], "path": ckpt_path, "metrics": metrics})
                    
                    if args.cleanup_images:
                        img_files = glob.glob(os.path.join(eval_output_dir, "*.png"))
                        for f in img_files:
                            os.remove(f)
                    
                    safe_print(f"[GPU {gpu_id}] [{index+1}/{total_tasks}] DONE {step_name} -> FID: {metrics.get('FID', -1):.2f}")
                else:
                    safe_print(f"[GPU {gpu_id}] [{index+1}/{total_tasks}] FAILED {step_name} (No metrics file)")

            except subprocess.CalledProcessError as e:
                safe_print(f"[GPU {gpu_id}] Error evaluating {step_name}: {e}")
        
        task_queue.task_done()

def main(args):
    print(f"[1/4] Scanning checkpoints in: {args.checkpoint_root}")
    checkpoints = find_checkpoints(args.checkpoint_root)
    
    if not checkpoints:
        print("No checkpoints found!")
        return

    print(f"Found {len(checkpoints)} checkpoints.")

    try:
        num_gpus = torch.cuda.device_count()
    except:
        num_gpus = 1 # Fallback
    
    if num_gpus == 0:
        print("No GPUs detected! Running on CPU (slow).")
        num_gpus = 1
    
    print(f"[2/4] Starting Multi-GPU Evaluation Pool")
    print(f"      Workers: {num_gpus} (GPUs)")
    print(f"      Task Limit: {args.limit} prompts")

    task_queue = queue.Queue()
    for i, ckpt in enumerate(checkpoints):
        task_queue.put((i, ckpt))
    
    results = []
    results_lock = threading.Lock()
    
    python_exe = sys.executable
    script_path = args.generation_script_path

    if not os.path.exists(script_path):
        print(f"Error: Generation script not found at {script_path}")
        return

    threads = []
    for gpu_id in range(num_gpus):
        t = threading.Thread(
            target=worker, 
            args=(gpu_id, task_queue, results, results_lock, args, python_exe, script_path, len(checkpoints))
        )
        t.start()
        threads.append(t)
    
    task_queue.join()
    
    for t in threads:
        t.join()

    if not results:
        print("No valid results found.")
        return

    print(f"\n[3/4] Aggregating Results...")
    
    results.sort(key=lambda x: x['step'])

    has_fid = any(r['metrics'].get('FID', -1) > 0 for r in results)
    has_ir = any(r['metrics'].get('IR_SCORE', 0) > 0 for r in results)

    if args.sort_by == "auto":
        if has_fid:
            sort_key = "FID"
            reverse = False
        elif has_ir:
            sort_key = "IR"
            reverse = True
        else:
            sort_key = "CLIP_SCORE"
            reverse = True
    else:
        sort_key = args.sort_by
        reverse = False if "FID" in sort_key else True 

    valid_results = [r for r in results if r['metrics'].get(sort_key, -1) != -1]
    valid_results.sort(key=lambda x: x['metrics'].get(sort_key, 0), reverse=reverse)
    
    print(f"\n[4/4] Selection Report (Sorted by {sort_key} {'DESC' if reverse else 'ASC'})")
    
    table_data = []
    headers = ["Step", "FID", "CLIP Score", "CLIP IQA", "ImageReward", "Path"]
    
    for res in valid_results:
        m = res['metrics']
        fid = m.get('FID', -1)
        clip = m.get('CLIP_SCORE', 0)
        iqa = m.get('CLIP_IQA', 0)
        ir = m.get('IR', m.get('IMAGE_REWARD', 0))
        
        table_data.append([
            res['step'],
            f"{fid:.2f}" if fid > 0 else "N/A",
            f"{clip:.4f}",
            f"{iqa:.4f}",
            f"{ir:.4f}",
            os.path.basename(res['path'])
        ])

    try:
        print(tabulate(table_data, headers=headers, tablefmt="github"))
    except ImportError:
        print(f"{'Step':<10} | {'FID':<10} | {'CLIP':<10} | {'Path'}")
        for row in table_data:
            print(f"{row[0]:<10} | {row[1]:<10} | {row[2]:<10} | {row[5]}")

    if valid_results:
        best = valid_results[0]
        print(f"\n🏆 Best Checkpoint: {best['path']}")
        print(f"   Reason: {sort_key} = {best['metrics'].get(sort_key)}")
        
        print(f"\nCommand to run full generation with best model:")
        print(f"python {script_path} --ckpt_path {best['path']} --jsonl_path {args.jsonl_path} ...")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Multi-GPU Auto Select Best Checkpoint")
    
    parser.add_argument("--checkpoint_root", type=str, required=True)
    parser.add_argument("--generation_script_path", type=str, required=True)
    parser.add_argument("--jsonl_path", type=str, required=True)
    parser.add_argument("--image_dir", type=str, default=None)
    parser.add_argument("--meta_config", type=str, default=None)
    
    parser.add_argument("--output_root", type=str, default="model_selection_results")
    parser.add_argument("--limit", type=int, default=32)
    parser.add_argument("--samples_per_prompt", type=int, default=1)
    
    parser.add_argument("--model_type", type=str, default=None)
    parser.add_argument("--base_model_path", type=str, default=None)
    parser.add_argument("--fid_stats", type=str, default=None)
    parser.add_argument("--gater_lut", type=str, default=None)
    parser.add_argument("--dtype", type=str, default="bf16")

    parser.add_argument("--force_rerun", action="store_true")
    parser.add_argument("--cleanup_images", action="store_true", default=True)
    parser.add_argument("--sort_by", type=str, default="auto", choices=["auto", "FID", "CLIP_SCORE", "IR"])

    args = parser.parse_args()

    if not torch.cuda.is_available():
        print("Warning: CUDA not available. Will run in single-thread CPU mode.")
    
    main(args)