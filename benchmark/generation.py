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
    sys.exit(1)


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

        self.clip_score = CLIPScore(
            model_name_or_path="../openai-clip-vit-base-patch16"
        ).to(device)

        self.clip_iqa = CLIPImageQualityAssessment(
            model_name_or_path="../openai-clip-vit-base-patch16"
        ).to(device)

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
                    data_items.append(
                        {"text": item["text"], "image_path": full_img_path}
                    )
    except Exception as e:
        print(f"Error reading JSONL: {e}")
        return []

    print(f"Loaded {len(data_items)} items.")
    return data_items


def main(args):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = torch.bfloat16 if args.dtype == "bf16" else torch.float16

    if args.force_fixed_ratio > 0:
        if "fixed" not in args.output_dir:
            args.output_dir = f"{args.output_dir}_fixed_{args.force_fixed_ratio}"
    
    save_dir = args.output_dir
    os.makedirs(save_dir, exist_ok=True)
    print(f"Output Directory: {save_dir}")

    evaluator = GenerationEvaluator(device, fid_stats_path=args.fid_stats)

    data_items = load_data(args.jsonl_path, args.image_dir, args.limit)
    if not data_items:
        return

    if args.meta_config and os.path.exists(args.meta_config):
        print(f"Found metadata config at {args.meta_config}")
        with open(args.meta_config, "r") as f:
            meta_config = json.load(f)

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
    print("-" * 50)
    print(f"Starting Benchmark: {args.model_type}")
    print(f"Mode: {'FIXED ' + str(args.force_fixed_ratio) if args.force_fixed_ratio > 0 else 'ADAPTIVE'}")
    print(f"Prompts: {len(data_items)}")
    print(f"Samples per Prompt: {args.samples_per_prompt}")
    print(f"Total Target Samples: {total_samples}")
    print(f"Global Seed: {args.seed}")
    print(f"FID Reference (.npz): {args.fid_stats}")
    
    print("-" * 50)

    pbar = tqdm(total=total_samples, desc="Processing")

    for i, item in enumerate(data_items):
        prompt = item["text"]

        for j in range(args.samples_per_prompt):
            # i = Prompt Index, j = Sample Index
            current_seed = args.seed + (i * 100) + j

            file_name = f"p{i:04d}_s{j:02d}.png"
            file_path = os.path.join(save_dir, file_name)

            gen_image = None

            if os.path.exists(file_path):
                try:
                    gen_image = Image.open(file_path).convert("RGB")
                except Exception as e:
                    print(f"Error loading {file_name}, will regenerate: {e}")
                    gen_image = None

            if gen_image is None:
                generator = torch.Generator(device=device).manual_seed(current_seed)
                set_seed(current_seed)

                kwargs = {"prompt": prompt, "generator": generator}
                if args.steps > 0:
                    kwargs["num_inference_steps"] = args.steps
                if args.height > 0:
                    kwargs["height"] = args.height
                if args.width > 0:
                    kwargs["width"] = args.width

                gen_image = process_quick_start(pipe, **kwargs)

                gen_image.save(file_path)

            evaluator.update_fake(gen_image, prompt)
            pbar.update(1)

    pbar.close()

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
    parser.add_argument("--dtype", type=str, default="bf16")

    # Shiva
    parser.add_argument("--optimize_hooks", action="store_true", default=True)
    parser.add_argument(
        "--no_optimize_hooks", action="store_false", dest="optimize_hooks"
    )
    parser.add_argument("--merge_lora", action="store_true", default=True)
    parser.add_argument("--no_merge_lora", action="store_false", dest="merge_lora")
    
    parser.add_argument(
        "--force_fixed_ratio", 
        type=float, 
        default=-1.0, 
        help="If > 0, disables Policy and forces a fixed ratio. (e.g. 0.64)"
    )

    args = parser.parse_args()

    if args.ckpt_path is None and args.model_type is None:
        parser.error("Please provide --ckpt_path or --model_type")

    main(args)