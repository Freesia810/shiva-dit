import os
import glob
import numpy as np
import torch
from pytorch_fid.fid_score import calculate_activation_statistics
from pytorch_fid.inception import InceptionV3

IMAGE_ROOT = "../dataset/MJHQ-30K" 
OUTPUT_PATH = "../dataset/MJHQ-30K/mjhq_30k_stats.npz"
DEVICE = "cuda:0"
BATCH_SIZE = 50
DIMS = 2048

def get_all_image_paths(root_dir):
    print(f"Scanning images in {root_dir}...")
    extensions = ['*.jpg', '*.jpeg', '*.png', '*.webp']
    image_paths = []
    
    for ext in extensions:
        # root_dir/**/*.png
        search_pattern = os.path.join(root_dir, '**', ext)
        found = glob.glob(search_pattern, recursive=True)
        image_paths.extend(found)
    
    image_paths = sorted(list(set(image_paths)))
    print(f"Found {len(image_paths)} images.")
    return image_paths

def main():
    paths = get_all_image_paths(IMAGE_ROOT)
    
    if len(paths) == 0:
        print("Error: No images found! Check your path.")
        return

    block_idx = InceptionV3.BLOCK_INDEX_BY_DIM[DIMS]
    model = InceptionV3([block_idx]).to(DEVICE)
    model.eval()

    print("Calculating statistics (this may take a while)...")
    mu, sigma = calculate_activation_statistics(
        files=paths,
        model=model,
        batch_size=BATCH_SIZE,
        dims=DIMS,
        device=DEVICE,
        num_workers=8
    )

    print(f"Saving statistics to {OUTPUT_PATH}...")
    np.savez_compressed(OUTPUT_PATH, mu=mu, sigma=sigma)
    print("Done! You can now use this .npz file for fast evaluation.")

if __name__ == "__main__":
    main()