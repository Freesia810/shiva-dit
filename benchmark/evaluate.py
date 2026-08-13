"""Standalone metric computation over an already-generated image directory.

generation.py only computes metrics inline on a single GPU; under torchrun it prints
"please run a standalone evaluation script" and there was none. This is it. It is
also what multiseed.py calls once per (config, seed) run.

Images are matched back to their prompts through generation.py's naming scheme,
p{global_idx:04d}_s{sample:02d}.png, where global_idx is the line number in the
prompt jsonl.

Run:
    python -m benchmark.evaluate \
        --image_dir results/shiva60/seed42 \
        --jsonl_path ../dataset/MJHQ-30K/test.jsonl \
        --fid_stats ../dataset/MJHQ-30K/mjhq_30k_stats.npz
"""

import argparse
import glob
import json
import os
import re

import torch
from PIL import Image
from tqdm import tqdm

from benchmark.generation import GenerationEvaluator

NAME_RE = re.compile(r"p(\d+)_s(\d+)\.(?:png|jpg|jpeg)$", re.IGNORECASE)
# Archived runs are a mix of .png (vanilla / 80 / 60) and .jpg (finetuned / 90 / 70),
# so every path that filters by filename has to accept both.
IMG_EXTS = ("png", "jpg", "jpeg")


def prompt_table(jsonl_path):
    """global_index -> prompt, mirroring load_data's indexing (line number)."""
    table = {}
    with open(jsonl_path, "r", encoding="utf-8") as f:
        for i, line in enumerate(f):
            line = line.strip()
            if not line:
                continue
            try:
                item = json.loads(line)
            except json.JSONDecodeError:
                continue
            if "text" in item:
                table[i] = item["text"]
    return table


def pick_device(requested):
    if requested:
        return requested
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--image_dir", required=True)
    p.add_argument("--jsonl_path", required=True)
    p.add_argument("--fid_stats", default=None)
    p.add_argument("--device", default=None)
    p.add_argument("--limit", type=int, default=None)
    p.add_argument("--out", default=None, help="defaults to <image_dir>/metrics.json")
    p.add_argument("--overwrite", action="store_true")
    p.add_argument(
        "--seed_slice",
        type=int,
        default=None,
        help="score only images p*_s{NN}.png for this latent slice; "
        "FID is computed on that slice alone. Used by multiseed "
        "when every config injects the same fixed latents s00..s0k.",
    )
    args = p.parse_args()

    default_name = (
        "metrics.json"
        if args.seed_slice is None
        else f"metrics_s{args.seed_slice:02d}.json"
    )
    out_path = args.out or os.path.join(args.image_dir, default_name)
    if os.path.exists(out_path) and not args.overwrite:
        print(f"{out_path} exists; use --overwrite to recompute.")
        print(json.dumps(json.load(open(out_path)), indent=2))
        return

    device = pick_device(args.device)
    prompts = prompt_table(args.jsonl_path)
    files = sorted(
        f
        for ext in IMG_EXTS
        for f in glob.glob(os.path.join(args.image_dir, f"p*_s*.{ext}"))
    )
    if args.seed_slice is not None:
        # keep only this latent slice, e.g. p0000_s01.png for slice 1
        stem = f"_s{args.seed_slice:02d}."
        files = [f for f in files if stem in os.path.basename(f)]
    if args.limit:
        files = files[: args.limit]
    if not files:
        raise SystemExit(
            f"no generated images matching p*_s*.png under {args.image_dir}"
            + (
                f" for slice s{args.seed_slice:02d}"
                if args.seed_slice is not None
                else ""
            )
        )

    # FID scores a directory, not a file list, so a slice needs its own directory
    # of symlinks -- calculate_fid_given_paths then sees only this slice's images.
    fid_dir = args.image_dir
    if args.seed_slice is not None:
        fid_dir = os.path.join(args.image_dir, f"_fidslice_s{args.seed_slice:02d}")
        os.makedirs(fid_dir, exist_ok=True)
        for ext in IMG_EXTS:
            for stale in glob.glob(os.path.join(fid_dir, f"*.{ext}")):
                os.unlink(stale)
        for f in files:
            os.symlink(os.path.abspath(f), os.path.join(fid_dir, os.path.basename(f)))

    print(
        f"device={device}  images={len(files)}  prompts={len(prompts)}"
        + (f"  slice=s{args.seed_slice:02d}" if args.seed_slice is not None else "")
    )

    evaluator = GenerationEvaluator(device=device, fid_stats_path=args.fid_stats)

    missing = 0
    for path in tqdm(files, desc="scoring"):
        m = NAME_RE.search(os.path.basename(path))
        if not m:
            continue
        gidx = int(m.group(1))
        prompt = prompts.get(gidx)
        if prompt is None:
            missing += 1
        image = Image.open(path).convert("RGB")
        evaluator.update_fake(image, prompt)

    if missing:
        print(
            f"warning: {missing} images had no matching prompt "
            f"(CLIP/IR skipped for those)"
        )

    metrics = evaluator.compute(generated_images_dir=fid_dir)
    metrics["_n_images"] = len(files)
    if args.seed_slice is not None:
        metrics["_seed_slice"] = args.seed_slice

    with open(out_path, "w") as f:
        json.dump(metrics, f, indent=2)

    print("\n" + "=" * 46)
    for k, v in metrics.items():
        print(f"{k:<15}: {v}")
    print("=" * 46)
    print(f"written to {out_path}")


if __name__ == "__main__":
    main()
