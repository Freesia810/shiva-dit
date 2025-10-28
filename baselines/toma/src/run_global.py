import argparse
import numpy as np
import itertools
import os
from diffusers import StableDiffusionXLPipeline, StableDiffusionPipeline
import torch
import pandas as pd
from tqdm import tqdm
import json
from PIL import PngImagePlugin  # Import PNG plugin to handle metadata
from datetime import datetime

import gc

from tomesd_global.patch import apply_patch
from tomesd_attention_tile.unet_scheduler import UNetScheduler
from imageNet1k_class import IMAGENET2012_CLASSES

def clean_memory():
    """Utility function to clean up GPU memory after every generation."""
    torch.cuda.empty_cache()
    gc.collect()


def generate_image(
    pipeline,
    output_folder,
    index="warmup",
    ratio=0,
    prompt="default",
    random_seed=42,
    dst_selection=None,
    max_downsample=4,
    height=1024,
    width=1024,
    k=64,
    merge_method=None,
):
    toma_scheduler = UNetScheduler(
        timesteps=50,
        dst_recompute_timesteps=[_ for _ in range(0, 50, 1)],
        attn_recompute_timesteps=[_ for _ in range(0, 50, 1)],
    )
    merge_attn, merge_crossattn, merge_mlp = 1, 1, 1

    print("\n")
    print(
        f"Generating image with ratio: {ratio}, merge method: {merge_method}, dst_selection: {dst_selection}, merge_once: {False}"
    )

    if ratio != 0:
        apply_patch(
            pipeline,
            max_downsample=max_downsample,
            ratio=ratio,
            dst_selection=dst_selection,
            merge_attn=merge_attn,
            merge_crossattn=merge_crossattn,
            merge_mlp=merge_mlp,
            k=k,
            merge_method=merge_method,
            unet_scheduler=toma_scheduler,
        )

    generator = torch.Generator(device=device).manual_seed(random_seed)

    start_event = torch.cuda.Event(enable_timing=True)
    end_event = torch.cuda.Event(enable_timing=True)
    start_event.record()

    stable_diffusion_output, denosing_time = pipeline(
        prompt=prompt,
        height=height,
        width=width,
        generator=generator,
        guidance_scale=7.5,
        negative_prompt="worst quality, logo, banner, jpeg artifact, mutation, sketch, amputation, disconnected limbs, cartoon",
    )
    image = stable_diffusion_output.images[0]

    end_event.record()
    torch.cuda.synchronize()
    elapsed_time = start_event.elapsed_time(end_event) * 1e-3

    if index == "warmup":
        return None

    os.makedirs(output_folder, exist_ok=True)
    print(f"Elapsed time: {elapsed_time:.2f}s")

    file_name = f"{datetime.now().strftime('%Y-%m-%d-%H-%M-%S')}*{prompt}*{random_seed}*{ratio}*{dst_selection}*{merge_method}*{False}.png"
    image_path = os.path.join(output_folder, file_name)

    metadata_dict = {
        "Prompt": prompt,
        "Seed": random_seed,
        "Ratio": ratio,
        "Dst_Selection": dst_selection,
        "Merge_Method": merge_method,
        "Elapsed_Time": elapsed_time,
    }

    metadata_json = json.dumps(metadata_dict)

    metadata = PngImagePlugin.PngInfo()
    metadata.add_text("Metadata", metadata_json)

    image.save(image_path, "PNG", pnginfo=metadata)

    clean_memory()
    return elapsed_time, file_name


def evaluate_dst_selection(pipeline, output_folder, dst_selection_list, prompt_list, seed_list, ratio_list, num_tiles, warm_up=True):

    if warm_up:
        for _ in range(3):
            generate_image(pipeline, output_folder)

    configurations = itertools.product(ratio_list, prompt_list, dst_selection_list, seed_list)
    results = []

    for index, (ratio, prompt, dst_selection, seed) in tqdm(
        enumerate(configurations), desc="Processing configurations"
    ):
        merge_method = "attention"

        elapsed_time, file_name = generate_image(
            pipeline=pipeline,
            output_folder=output_folder,
            index=index,
            ratio=ratio,
            prompt=prompt,
            random_seed=seed,
            dst_selection=dst_selection,
            max_downsample=4,
            height=1024,
            width=1024,
            k=num_tiles,
            merge_method=merge_method,
        )

        results.append(elapsed_time)

    avg_time = np.mean(results)
    std_time = np.std(results)
    min_time = np.min(results)
    max_time = np.max(results)

    print(f"ratio={ratio}")
    print(f"Average time: {avg_time:.2f}s")
    print(f"Std time: {std_time:.2f}s")
    print(f"Min time: {min_time:.2f}s")
    print(f"Max time: {max_time:.2f}s")
    print(results)

    with open(results_file_path, "a") as f:
        f.write(f"ratio={ratio}\n")
        f.write(f"Average time: {avg_time:.2f}s\n")
        f.write(f"Std time: {std_time:.2f}s\n")
        f.write(f"Min time: {min_time:.2f}s\n")
        f.write(f"Max time: {max_time:.2f}s\n")
        f.write(f"{results}\n\n")

    clean_memory()


if __name__ == "__main__":
    # Argument parser setup
    parser = argparse.ArgumentParser(description="Generate images with different configurations.")
    # parser.add_argument("--num_tiles", type=int, default=256, help="Number of tiles for the operation.")

    
    args = parser.parse_args()

    ratio_list = [0.5]
    # num_tiles = args.num_tiles
    num_tiles = 256

    for ratio in ratio_list:
        prompt_list = list(IMAGENET2012_CLASSES.values())[:10]
        seed_list = [864]
        dst_method = "tile_wise_facility"
        output_folder = f"{ratio}"
        results_file_path = f"time.md"
        device = "cuda:0"

        pipeline = StableDiffusionXLPipeline.from_pretrained(
            "stabilityai/stable-diffusion-xl-base-1.0",
            torch_dtype=torch.float16,
            variant="fp16",
            use_safetensors=True,
            cache_dir="/home/wl2707/.cache",
            local_files_only=True,
        ).to(device)

        evaluate_dst_selection(
            pipeline=pipeline,
            output_folder=output_folder,
            dst_selection_list=[dst_method],
            prompt_list=prompt_list,
            seed_list=seed_list,
            ratio_list=[ratio],
            num_tiles=num_tiles,
            warm_up=True,
        )
