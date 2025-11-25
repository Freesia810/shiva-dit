import json
import os
import random
import argparse
import shutil
from pathlib import Path
from tqdm import tqdm
from datasets import load_dataset
from PIL import Image


def parse_args():
    parser = argparse.ArgumentParser(
        description="Process locally available MagicBrush dataset into Shiva training format."
    )

    parser.add_argument(
        "--dataset_path",
        type=str,
        default="osu-nlp/MagicBrush",
        help="Path to local dataset or HF hub name",
    )

    parser.add_argument(
        "--root_dir",
        type=str,
        help="Path to save processed images and jsonl",
    )

    parser.add_argument(
        "--use_original_split",
        action="store_true",
        help="Use original dataset splits (train/dev/test).",
    )
    parser.add_argument(
        "--train_ratio",
        type=float,
        default=0.9,
        help="Train split ratio (if re-splitting)",
    )
    parser.add_argument(
        "--val_ratio",
        type=float,
        default=0.05,
        help="Validation split ratio (if re-splitting)",
    )
    parser.add_argument(
        "--train_samples",
        type=int,
        default=None,
        help="Explicitly set number of training samples (overrides ratio)",
    )

    parser.add_argument(
        "--seed", type=int, default=42, help="Random seed for shuffling"
    )

    args = parser.parse_args()
    return args


def write_jsonl(filename, data):
    with open(filename, "w", encoding="utf-8") as f:
        for item in data:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")


def process_dataset_items(dataset, img_save_dir):
    all_samples = []

    for split_name in dataset.keys():
        print(f"Processing split: {split_name}...")
        split_data = dataset[split_name]

        for idx, item in enumerate(tqdm(split_data)):
            try:
                # MagicBrush: {'source_img': PIL, 'target_img': PIL, 'instruction': str, ...}
                source_img = item["source_img"]
                target_img = item["target_img"]
                instruction = item["instruction"]

                img_id = item.get("img_id", f"{split_name}_{idx}")
                turn_index = item.get("turn_index", 0)

                src_filename = f"{img_id}_{turn_index}_src.png"
                tgt_filename = f"{img_id}_{turn_index}_tgt.png"

                src_path_abs = os.path.join(img_save_dir, src_filename)
                tgt_path_abs = os.path.join(img_save_dir, tgt_filename)

                src_path_rel = os.path.join("images", src_filename)
                tgt_path_rel = os.path.join("images", tgt_filename)

                if not os.path.exists(src_path_abs):
                    if source_img.mode != "RGB":
                        source_img = source_img.convert("RGB")
                    source_img.save(src_path_abs)

                if not os.path.exists(tgt_path_abs):
                    if target_img.mode != "RGB":
                        target_img = target_img.convert("RGB")
                    target_img.save(tgt_path_abs)

                entry = {
                    "file_name": tgt_path_rel,  # (Target)
                    "condition_image": src_path_rel,  # (Source)
                    "text": instruction,
                    "source": "magicbrush",
                    "original_split": split_name,
                }

                all_samples.append(entry)

            except Exception as e:
                print(f"Error processing item {idx} in {split_name}: {e}")
                continue

    return all_samples


def main():
    args = parse_args()
    random.seed(args.seed)

    root_dir = Path(args.root_dir)
    img_dir = root_dir / "images"

    if not root_dir.exists():
        root_dir.mkdir(parents=True)
    if not img_dir.exists():
        img_dir.mkdir(parents=True)

    print(f"Loading dataset from: {args.dataset_path}")
    try:
        dataset = load_dataset(args.dataset_path)
    except Exception as e:
        print(f"Failed to load dataset: {e}")
        print(
            "Please ensure 'dataset_path' points to a valid local directory or HF dataset name."
        )
        return

    print("Processing images and metadata...")
    all_samples = process_dataset_items(dataset, str(img_dir))
    print(f"Total samples collected: {len(all_samples)}")

    if args.use_original_split:
        print("Using original dataset splits...")
        train_data = [s for s in all_samples if s["original_split"] == "train"]
        val_data = [s for s in all_samples if s["original_split"] in ["dev", "val"]]
        test_data = [s for s in all_samples if s["original_split"] == "test"]
    else:
        print("Shuffling and re-splitting...")
        random.shuffle(all_samples)
        total = len(all_samples)

        if args.train_samples is not None:
            n_train = min(args.train_samples, total)
        else:
            n_train = int(total * args.train_ratio)

        n_val = int(total * args.val_ratio)
        n_test = total - n_train - n_val
        if n_test < 0:
            n_test = 0
            n_val = total - n_train

        train_data = all_samples[:n_train]
        val_data = all_samples[n_train : n_train + n_val]
        test_data = all_samples[n_train + n_val :]

    write_jsonl(root_dir / "train.jsonl", train_data)
    write_jsonl(root_dir / "val.jsonl", val_data)
    write_jsonl(root_dir / "test.jsonl", test_data)

    print("Done! Dataset prepared.")
    print(f"  Root: {root_dir}")
    print(f"  train: {len(train_data)}")
    print(f"  val:   {len(val_data)}")
    print(f"  test:  {len(test_data)}")


if __name__ == "__main__":
    main()
