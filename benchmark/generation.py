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


LOCAL_BERT_PATH = os.environ.get("SHIVA_METRIC_BERT_PATH", "../bert-base-uncased")
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


class DirectClipScore:
    """CLIP Score computed straight from transformers, as a version-proof fallback.

    torchmetrics' CLIPScore reaches into `model.get_image_features`, whose return type
    changed in transformers 5.x (tensor -> BaseModelOutputWithPooling), so the metric
    raises on newer installs. Going through vision_model/text_model plus the projection
    heads is the canonical CLIP path and has been stable across both. The value matches
    torchmetrics' definition: 100 * max(cos(img, txt), 0).
    """

    def __init__(self, model_name_or_path, device):
        from transformers import CLIPModel, CLIPProcessor
        self.device = device
        self.model = CLIPModel.from_pretrained(model_name_or_path).to(device).eval()
        self.processor = CLIPProcessor.from_pretrained(model_name_or_path)

    def to(self, device):
        return self

    @torch.no_grad()
    def __call__(self, images_uint8, prompts):
        # images_uint8: (B, 3, H, W) uint8 tensor, matching the torchmetrics call site
        pil = [TF.to_pil_image(img.cpu()) for img in images_uint8]
        inputs = self.processor(text=list(prompts), images=pil, return_tensors="pt",
                                padding=True, truncation=True)
        inputs = {k: v.to(self.device) for k, v in inputs.items()}

        vision_out = self.model.vision_model(pixel_values=inputs["pixel_values"])
        img_emb = self.model.visual_projection(vision_out.pooler_output)
        text_out = self.model.text_model(
            input_ids=inputs["input_ids"],
            attention_mask=inputs.get("attention_mask"),
        )
        txt_emb = self.model.text_projection(text_out.pooler_output)

        img_emb = img_emb / img_emb.norm(p=2, dim=-1, keepdim=True)
        txt_emb = txt_emb / txt_emb.norm(p=2, dim=-1, keepdim=True)
        cos = (img_emb * txt_emb).sum(dim=-1)
        return 100.0 * cos.clamp(min=0).mean()


class GenerationEvaluator:
    def __init__(self, device, fid_stats_path=None):
        self.device = device
        self.fid_stats_path = fid_stats_path
        print(f"Initializing metrics on {device}...")
        if fid_stats_path:
            print(f"FID Reference Stats: {fid_stats_path}")
        else:
            print("Warning: No .npz provided. FID will be skipped.")

        # The generation boxes keep CLIP checkpoints beside the repo; a scoring-only
        # machine (e.g. the Mac that computes metrics) has no such directory, so fall
        # back to the hub id rather than losing CLIP Score and CLIP-IQA entirely.
        clip_path = os.environ.get("SHIVA_METRIC_CLIP_PATH", "openai/clip-vit-base-patch16")
        # Loaded independently: CLIP-IQA breaks on some torchmetrics/transformers
        # pairings, and sharing one try block would take CLIP Score down with it --
        # which is the metric the multi-seed comparison actually turns on.
        # Constructing CLIPScore can succeed while the first update() still raises on a
        # transformers-version mismatch, so probe it once and swap in the direct
        # implementation if it cannot actually score.
        self.clip_score = None
        try:
            probe = CLIPScore(model_name_or_path=clip_path).to(device)
            probe(torch.zeros(1, 3, 64, 64, dtype=torch.uint8, device=device), ["probe"])
            self.clip_score = probe
        except Exception as e:
            print(f"Warning: torchmetrics CLIPScore unusable ({e}); "
                  f"falling back to a direct transformers implementation.")
            try:
                self.clip_score = DirectClipScore(clip_path, device)
            except Exception as e2:
                print(f"Warning: Failed to load CLIP Score: {e2}")
        try:
            self.clip_iqa = CLIPImageQualityAssessment(
                model_name_or_path=clip_path
            ).to(device)
        except Exception as e:
            print(f"Warning: Failed to load CLIP-IQA: {e}")
            self.clip_iqa = None

        self.ir_model = None
        if RM:
            print("Loading ImageReward model...")
            try:
                self.ir_model = RM.load(
                    os.environ.get("SHIVA_METRIC_IR_PATH", "ImageReward-v1.0"),
                    device=device,
                    med_config=os.environ.get("SHIVA_METRIC_IR_CONFIG"),
                )
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
        if self.clip_score is None and self.clip_iqa is None and self.ir_model is None:
            return

        gen_tensor = self.preprocess(generated_pil)
        gen_uint8 = self.preprocess_uint8(generated_pil)

        if prompt and self.clip_score is not None:
            score = self.clip_score(gen_uint8, [prompt])
            self.results["clip_scores"].append(score.item())

        if self.clip_iqa is not None:
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
                if not line.strip():
                    continue
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
    elif torch.backends.mps.is_available():
        device = "mps"
    else:
        device = "cpu"

    dtype = {"bf16": torch.bfloat16, "fp16": torch.float16, "fp32": torch.float32}[args.dtype]

    print(f"[Process {rank}/{world_size}] Initialized on {device}")

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
    if not args.skip_metrics and world_size == 1:
        evaluator = GenerationEvaluator(device, fid_stats_path=args.fid_stats)

    # Load ALL data first
    all_data_items = load_data(args.jsonl_path, args.image_dir, args.limit)
    if not all_data_items:
        raise ValueError(f"No prompts loaded from {args.jsonl_path}")

    # --- [MODIFIED] Data Sharding ---
    data_items = all_data_items[rank::world_size]
    print(f"[Process {rank}] Processing {len(data_items)} items (subset of {len(all_data_items)})")

    if args.meta_config:
        print(f"Found metadata config at {args.meta_config}")
        with open(args.meta_config, "r") as f:
            meta_config = json.load(f)
    else:
        meta_config = None

    if args.model_type or args.ckpt_path or meta_config:
        print("Loading pipeline...")
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
            strict_checkpoint=args.strict_checkpoint,
            expected_lut_steps=args.steps if args.steps > 0 else None,
        )
        print(f"Pipeline device: {next(pipe.transformer.parameters()).device}")
    else:
        raise ValueError("Provide --model_type, --ckpt_path, or --meta_config")

    total_samples = len(data_items) * args.samples_per_prompt
    
    if rank == 0:
        print("-" * 50)
        print(f"Starting Benchmark: {args.model_type}")
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
                    with Image.open(file_path) as saved_image:
                        gen_image = saved_image.convert("RGB")
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
                            latents = torch.load(latent_path, map_location=device, weights_only=True)
                            latents = latents.to(dtype=dtype)
                            if args.strict_latents and not torch.isfinite(latents).all():
                                raise ValueError("Latent contains non-finite values")
                        except Exception as e:
                            if args.strict_latents:
                                raise RuntimeError(f"Cannot load required latent {latent_path}") from e
                            print(f"[Rank {rank}] Error loading latent {latent_path}: {e}")
                            latents = None
                    else:
                        if args.strict_latents:
                            raise FileNotFoundError(latent_path)
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
                
                # [MODIFIED] Inject latents if available
                if latents is not None:
                    kwargs["latents"] = latents

                gen_image = process_quick_start(pipe, **kwargs)

                gen_image.save(file_path)

            if evaluator:
                evaluator.update_fake(gen_image, prompt)
            
            pbar.update(1)

    pbar.close()
    
    if world_size > 1 and rank == 0:
        print(f"Run benchmark.evaluate on '{save_dir}' after all processes finish.")
    if evaluator:
        print(f"\nComputing final metrics on {total_samples} samples...")
        metrics = evaluator.compute(generated_images_dir=save_dir)

        print("\n" + "=" * 20 + " Benchmark Report " + "=" * 20)
        print(f"Model: {args.ckpt_path or args.model_type}")
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
    parser.add_argument("--dtype", choices=("bf16", "fp16", "fp32"), default="bf16")

    # Shiva
    parser.add_argument("--optimize_hooks", action="store_true", default=True)
    parser.add_argument(
        "--no_optimize_hooks", action="store_false", dest="optimize_hooks"
    )
    parser.add_argument("--merge_lora", action="store_true", default=True)
    parser.add_argument("--no_merge_lora", action="store_false", dest="merge_lora"
    )
    
    parser.add_argument(
        "--latent_source_dir", 
        type=str, 
        default=None, 
        help="Path to directory containing pre-dumped latents (e.g. from machine A)."
    )
    parser.add_argument("--strict_latents", action="store_true",
                        help="fail on missing/unreadable fixed latents instead of drawing random noise")
    parser.add_argument("--strict_checkpoint", action="store_true",
                        help="fail if checkpoint contains tensors that the loaded model does not use")
    parser.add_argument(
        "--skip_metrics", 
        action="store_true", 
        help="If set, skips FID/CLIP/IR calculation. Recommended for multi-GPU generation."
    )

    args = parser.parse_args()

    if args.samples_per_prompt < 1:
        parser.error("--samples_per_prompt must be positive")
    if args.gater_lut and args.steps < 1:
        parser.error("--gater_lut requires --steps matching the exported schedule")
    if args.strict_latents and not args.latent_source_dir:
        parser.error("--strict_latents requires --latent_source_dir")

    main(args)
