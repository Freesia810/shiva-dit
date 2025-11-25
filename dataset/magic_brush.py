import json
import os
import random
import argparse
from pathlib import Path
from datasets import load_dataset


def parse_args():
    parser = argparse.ArgumentParser(
        description="Prepare MagicBrush metadata ONLY (Direct HF loading)."
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="./single_sample_data/magicbrush",
        help="Directory to save jsonl files",
    )
    parser.add_argument(
        "--dataset_path",
        type=str,
        default="osu-nlp/MagicBrush",
        help="HuggingFace dataset path",
    )

    parser.add_argument(
        "--train_ratio", type=float, default=0.9, help="Train split ratio"
    )
    parser.add_argument(
        "--val_ratio", type=float, default=0.05, help="Validation split ratio"
    )
    parser.add_argument(
        "--train_samples",
        type=int,
        default=None,
        help="Explicitly set number of training samples (overrides ratio)",
    )
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument(
        "--use_original_split",
        action="store_true",
        help="If set, ignore ratios and use original dataset splits.",
    )

    args = parser.parse_args()
    return args


def write_jsonl(filename, data):
    with open(filename, "w", encoding="utf-8") as f:
        for item in data:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")


def main():
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)

    print(f"Loading dataset metadata from: {args.dataset_path}")
    try:
        dataset = load_dataset(args.dataset_path)
    except Exception as e:
        print(f"Error loading dataset: {e}")
        return

    all_samples = []

    for split in dataset.keys():
        print(f"Collecting from {split}...")
        for idx, item in enumerate(dataset[split]):
            entry = {
                "file_name": f"hf://{args.dataset_path}/{split}/{idx}/target_img",
                "condition_image": f"hf://{args.dataset_path}/{split}/{idx}/source_img",
                "text": item["instruction"],
                "source": "magicbrush_hf",
                "hf_info": {"path": args.dataset_path, "split": split, "index": idx},
            }
            all_samples.append(entry)

    if args.use_original_split:
        print("Using original dataset splits...")
        train_data = [s for s in all_samples if s["hf_info"]["split"] == "train"]
        val_data = [
            s for s in all_samples if s["hf_info"]["split"] in ["dev", "validation"]
        ]
        test_data = [s for s in all_samples if s["hf_info"]["split"] == "test"]
    else:
        print(f"Shuffling and splitting {len(all_samples)} samples...")
        random.seed(args.seed)
        random.shuffle(all_samples)

        total = len(all_samples)
        if args.train_samples is not None:
            n_train = min(args.train_samples, total)
            print(f"Using fixed train samples: {n_train}")
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

    write_jsonl(os.path.join(args.output_dir, "train.jsonl"), train_data)
    write_jsonl(os.path.join(args.output_dir, "val.jsonl"), val_data)
    write_jsonl(os.path.join(args.output_dir, "test.jsonl"), test_data)

    print("Fast preparation done!")
    print(f"  Output: {args.output_dir}")
    print(f"  Train: {len(train_data)}")
    print(f"  Val:   {len(val_data)}")
    print(f"  Test:  {len(test_data)}")


if __name__ == "__main__":
    main()
