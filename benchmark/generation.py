import torch
import os
import sys
import json
import argparse
from PIL import Image
from tqdm import tqdm
from types import ModuleType

try:
    from pytorch_fid.fid_score import calculate_fid_given_paths
    from torchmetrics.multimodal.clip_score import CLIPScore
    from torchmetrics.multimodal import CLIPImageQualityAssessment
    import torchvision.transforms.functional as TF
except ImportError:
    print(
        "Error: Required libraries missing. Please run:\n"
        "pip install pytorch-fid torchmetrics[image]"
    )


LOCAL_BERT_PATH = "../bert-base-uncased"
from transformers import BertTokenizer, BertModel

_original_tokenizer_from_pretrained = BertTokenizer.from_pretrained
_original_model_from_pretrained = BertModel.from_pretrained

def _patched_tokenizer_loader(cls, pretrained_model_name_or_path, *args, **kwargs):
    if pretrained_model_name_or_path == "bert-base-uncased" and os.path.exists(
        LOCAL_BERT_PATH
    ):
        pretrained_model_name_or_path = LOCAL_BERT_PATH
    return _original_tokenizer_from_pretrained(
        pretrained_model_name_or_path, *args, **kwargs
    )

def _patched_model_loader(cls, pretrained_model_name_or_path, *args, **kwargs):
    if pretrained_model_name_or_path == "bert-base-uncased" and os.path.exists(
        LOCAL_BERT_PATH
    ):
        pretrained_model_name_or_path = LOCAL_BERT_PATH
    return _original_model_from_pretrained(
        pretrained_model_name_or_path, *args, **kwargs
    )

BertTokenizer.from_pretrained = classmethod(_patched_tokenizer_loader)
BertModel.from_pretrained = classmethod(_patched_model_loader)

sys.modules["ImageReward.ReFL"] = ModuleType("ImageReward.ReFL")

import transformers.modeling_utils
import transformers.pytorch_utils

if not hasattr(transformers.modeling_utils, "apply_chunking_to_forward"):
    if hasattr(transformers.pytorch_utils, "apply_chunking_to_forward"):
        transformers.modeling_utils.apply_chunking_to_forward = (
            transformers.pytorch_utils.apply_chunking_to_forward
        )
if not hasattr(transformers.modeling_utils, "find_pruneable_heads_and_indices"):
    if hasattr(transformers.pytorch_utils, "find_pruneable_heads_and_indices"):
        transformers.modeling_utils.find_pruneable_heads_and_indices = (
            transformers.pytorch_utils.find_pruneable_heads_and_indices
        )
if not hasattr(transformers.modeling_utils, "prune_linear_layer"):
    if hasattr(transformers.pytorch_utils, "prune_linear_layer"):
        transformers.modeling_utils.prune_linear_layer = (
            transformers.pytorch_utils.prune_linear_layer
        )

try:
    import ImageReward as RM
except ImportError:
    RM = None
    print("Warning: ImageReward not installed. IR score will be skipped.")

from shiva_dit.utils.inference_utils import load_pipeline_for_inference
from shiva_dit.utils.utils import process_quick_start
from accelerate.utils import set_seed


class GenerationEvaluator:
    def __init__(self, device, fid_stats_path=None):
        self.device = device
        self.fid_stats_path = fid_stats_path
        print(f"Initializing metrics on {device}...")
        if fid_stats_path:
            print(f"FID Reference Stats: {fid_stats_path}")
        else:
            print("Warning: No .npz provided. FID will be skipped.")

        try:
            self.clip_score = CLIPScore(
                model_name_or_path="../openai-clip-vit-base-patch16"
            ).to(device)

            self.clip_iqa = CLIPImageQualityAssessment(
                model_name_or_path="../openai-clip-vit-base-patch16"
            ).to(device)
        except Exception as e:
            print(f"Warning: Failed to load CLIP metrics: {e}")
            self.clip_score = None
            self.clip_iqa = None

        self.ir_model = None
        if RM:
            print("Loading ImageReward model...")
            try:
                self.ir_model = RM.load("../ImageReward/ImageReward.pt", device=device, med_config="../ImageReward/med_config.json")
            except Exception as e:
                print(f"Error loading ImageReward: {e}")

        self.results = {"ir_scores": [], "clip_scores": [], "clip_iqa_scores": []}

    def preprocess(self, image):
        return TF.to_tensor(image).to(self.device).unsqueeze(0)

    def preprocess_uint8(self, image):
        t = TF.to_tensor(image).to(self.device) * 255
        return t.to(torch.uint8).unsqueeze(0)

    @torch.no_grad()
    def update_fake(self, generated_pil, prompt=None):
        if self.clip_score is None:
            return

        gen_tensor = self.preprocess(generated_pil)
        gen_uint8 = self.preprocess_uint8(generated_pil)

        if prompt:
            score = self.clip_score(gen_uint8, [prompt])
            self.results["clip_scores"].append(score.item())

        iqa_out = self.clip_iqa(gen_tensor)
        if isinstance(iqa_out, dict):
            mean_iqa = torch.stack(list(iqa_out.values())).mean().item()
            self.results["clip_iqa_scores"].append(mean_iqa)
        else:
            self.results["clip_iqa_scores"].append(iqa_out.item())

        if self.ir_model and prompt:
            ir_score = self.ir_model.score(prompt, generated_pil)
            self.results["ir_scores"].append(ir_score)

    def compute(self, generated_images_dir):
        print("\nComputing aggregated metrics...")
        metrics = {}

        if self.fid_stats_path and os.path.exists(self.fid_stats_path) and os.path.exists(generated_images_dir):
            print(f"Running FID calculation between:\n - Gen: {generated_images_dir}\n - Ref: {self.fid_stats_path}")
            try:
                fid_value = calculate_fid_given_paths(
                    paths=[generated_images_dir, self.fid_stats_path],
                    batch_size=50,
                    device=self.device,
                    dims=2048
                )
                metrics["FID"] = fid_value
            except Exception as e:
                print(f"Error computing FID: {e}")
                metrics["FID"] = -1.0
        else:
            print("Skipping FID (Missing .npz file or output directory).")
            metrics["FID"] = -1.0

        for key, val_list in self.results.items():
            if val_list:
                metric_name = key.replace("_scores", "").upper()
                if metric_name == "CLIP":
                    metric_name = "CLIP_SCORE"
                metrics[metric_name] = sum(val_list) / len(val_list)

        return metrics


def load_data(jsonl_path, image_dir, limit=None):
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
                    # [MODIFIED] Save global_index for latent alignment
                    data_items.append(
                        {"text": item["text"], "image_path": full_img_path, "global_index": i}
                    )
    except Exception as e:
        print(f"Error reading JSONL: {e}")
        return []

    print(f"Loaded {len(data_items)} items.")
    return data_items


def main(args):
    # --- [MODIFIED] Multi-GPU Setup ---
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

    if args.force_fixed_ratio > 0:
        if "fixed" not in args.output_dir:
            args.output_dir = f"{args.output_dir}_fixed_{args.force_fixed_ratio}"
    
    save_dir = args.output_dir
    # Only Rank 0 creates directory to avoid race conditions
    if rank == 0:
        os.makedirs(save_dir, exist_ok=True)
        print(f"Output Directory: {save_dir}")
    
    # Simple barrier-like mechanism: wait for dir creation (optional, usually OS handles exist_ok=True fine)
    # But for strict safety with metrics file later, we utilize rank checks.

    # Only initialize evaluator on Rank 0 if we are strictly calculating metrics
    # Or initialize on all if we want real-time feedback per card (but final compute is tricky)
    # Here we initialize on all for simplicity, but only Rank 0 prints final aggregation.
    evaluator = None
    if not args.skip_metrics:
        evaluator = GenerationEvaluator(device, fid_stats_path=args.fid_stats)

    # Load ALL data first
    all_data_items = load_data(args.jsonl_path, args.image_dir, args.limit)
    if not all_data_items:
        return

    # --- [MODIFIED] Data Sharding ---
    data_items = all_data_items[rank::world_size]
    print(f"[Process {rank}] Processing {len(data_items)} items (subset of {len(all_data_items)})")

    if args.meta_config and os.path.exists(args.meta_config):
        print(f"Found metadata config at {args.meta_config}")
        with open(args.meta_config, "r") as f:
            meta_config = json.load(f)
    else:
        meta_config = None

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
        print(f"Starting Benchmark: {args.model_type}")
        print(f"Mode: {'FIXED ' + str(args.force_fixed_ratio) if args.force_fixed_ratio > 0 else 'ADAPTIVE'}")
        print(f"Global Prompts: {len(all_data_items)}")
        print(f"Local Prompts (Rank {rank}): {len(data_items)}")
        print(f"Samples per Prompt: {args.samples_per_prompt}")
        print(f"Total Target Samples (Local): {total_samples}")
        print(f"Global Seed: {args.seed}")
        print(f"Latent Source: {args.latent_source_dir if args.latent_source_dir else 'Random (On-the-fly)'}")
        print("-" * 50)

    # Position argument helps prevents progress bars from overlapping in console
    pbar = tqdm(total=total_samples, desc=f"Proc {rank}", position=rank)

    for item in data_items:
        prompt = item["text"]
        # [MODIFIED] Use global index for filename and latent retrieval
        global_idx = item["global_index"]

        for j in range(args.samples_per_prompt):
            # i = Global Prompt Index, j = Sample Index
            current_seed = args.seed + (global_idx * 100) + j

            file_name = f"p{global_idx:04d}_s{j:02d}.png"
            file_path = os.path.join(save_dir, file_name)

            gen_image = None

            # Skip existing files
            if os.path.exists(file_path):
                try:
                    gen_image = Image.open(file_path).convert("RGB")
                    # Force verify image is valid
                    gen_image.load() 
                except Exception as e:
                    print(f"[Rank {rank}] Error loading {file_name}, will regenerate: {e}")
                    gen_image = None

            if gen_image is None:
                # [MODIFIED] Latent Loading Logic
                latents = None
                if args.latent_source_dir:
                    # Expected format: latent_p{global_idx:06d}_s{j:02d}.pt
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

                # Setup Generator
                generator = torch.Generator(device=device).manual_seed(current_seed)
                set_seed(current_seed)

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
                
                # [MODIFIED] Inject latents if available
                if latents is not None:
                    kwargs["latents"] = latents

                gen_image = process_quick_start(pipe, **kwargs)

                gen_image.save(file_path)

            if evaluator:
                evaluator.update_fake(gen_image, prompt)
            
            pbar.update(1)

    pbar.close()
    
    if evaluator and not args.skip_metrics:
        if world_size > 1:
            print(f"[Process {rank}] Generation finished. Skipping aggregated metrics calculation in multi-GPU mode.")
            print(f"Please run a standalone evaluation script on '{save_dir}' once all processes are done.")
        else:
            print(f"\nComputing final metrics on {total_samples} samples...")
            metrics = evaluator.compute(generated_images_dir=save_dir)

            print("\n" + "=" * 20 + " Benchmark Report " + "=" * 20)
            print(f"Model: {args.ckpt_path or args.model_type}")
            print(f"Config: {'FIXED ' + str(args.force_fixed_ratio) if args.force_fixed_ratio > 0 else 'ADAPTIVE'}")
            print(f"Total Samples: {total_samples}")
            print("-" * 30)
            for k, v in metrics.items():
                print(f"{k:<15}: {v:.4f}")
            print("=" * 56)

            # Save metrics to json
            with open(os.path.join(save_dir, "metrics.json"), "w") as f:
                json.dump(metrics, f, indent=4)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    # Model
    parser.add_argument("--ckpt_path", type=str, default=None)
    parser.add_argument("--model_type", type=str, default=None)
    parser.add_argument("--base_model_path", type=str, default=None)
    parser.add_argument("--gater_lut", type=str, default=None)
    parser.add_argument("--meta_config", type=str, default=None)

    # Data
    parser.add_argument(
        "--jsonl_path", type=str, required=True, help="Path to val.jsonl"
    )
    parser.add_argument(
        "--image_dir",
        type=str,
        default=None,
        help="Root dir for GT images (only used for loading paths if needed, not for FID)",
    )
    parser.add_argument(
        "--fid_stats", 
        type=str, 
        default=None, 
        help="Path to precomputed .npz stats file for FID"
    )

    parser.add_argument(
        "--output_dir", type=str, default="eval_results", help="Where to save images"
    )
    parser.add_argument("--limit", type=int, default=None)

    # Sampling Config
    parser.add_argument(
        "--samples_per_prompt",
        type=int,
        default=5,
        help="Number of images per prompt (Total = N_prompts * N_samples)",
    )
    parser.add_argument("--seed", type=int, default=42, help="Global base seed")
    parser.add_argument("--steps", type=int, default=-1)
    parser.add_argument("--height", type=int, default=1024)
    parser.add_argument("--width", type=int, default=1024)
    parser.add_argument("--input_image", type=str, default=None)
    parser.add_argument("--max_sequence_length", type=int, default=None)
    parser.add_argument("--dtype", type=str, default="bf16")

    # Shiva
    parser.add_argument("--optimize_hooks", action="store_true", default=True)
    parser.add_argument(
        "--no_optimize_hooks", action="store_false", dest="optimize_hooks"
    )
    parser.add_argument("--merge_lora", action="store_true", default=True)
    parser.add_argument("--no_merge_lora", action="store_false", dest="merge_lora"
    )
    
    parser.add_argument(
        "--force_fixed_ratio", 
        type=float, 
        default=-1.0, 
        help="If > 0, disables Policy and forces a fixed ratio. (e.g. 0.64)"
    )

    parser.add_argument(
        "--latent_source_dir", 
        type=str, 
        default=None, 
        help="Path to directory containing pre-dumped latents (e.g. from machine A)."
    )
    parser.add_argument(
        "--skip_metrics", 
        action="store_true", 
        help="If set, skips FID/CLIP/IR calculation. Recommended for multi-GPU generation."
    )

    args = parser.parse_args()

    main(args)