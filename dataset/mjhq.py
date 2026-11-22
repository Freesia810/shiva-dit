import json
import random
import argparse
from pathlib import Path


def parse_args():
    parser = argparse.ArgumentParser(description="Convert meta_data.json to train/val/test JSONL datasets.")
    parser.add_argument("--root_dir", type=str, help="Path to data root")
    parser.add_argument("--input", type=str, default="meta_data.json", help="Path to meta_data.json")
    parser.add_argument("--train_ratio", type=float, default=0.8, help="Train split ratio")
    parser.add_argument("--val_ratio", type=float, default=0.1, help="Validation split ratio")
    parser.add_argument("--train_samples", type=int, default=None, help="Explicitly set number of training samples (overrides ratio)")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for shuffling")
    args = parser.parse_args()
    return args


def write_jsonl(filename, data):
    with open(filename, "w", encoding="utf-8") as f:
        for item in data:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")


def main():
    args = parse_args()
    root_dir = Path(args.root_dir)
    input_path = root_dir / args.input

    with open(input_path, "r", encoding="utf-8") as f:
        meta = json.load(f)

    samples = []
    for file_id, info in meta.items():
        category = info["category"]
        prompt = info["prompt"]
        samples.append({
            "file_name": f"{category}/{file_id}.jpg",
            "text": prompt
        })

    total = len(samples)
    print(f"Total samples: {total}")

    random.seed(args.seed)
    random.shuffle(samples)

    if args.train_samples is not None:
        n_train = min(args.train_samples, total)
        print(f"Using specified train sample count: {n_train}")
    else:
        n_train = int(total * args.train_ratio)
        print(f"Using ratio-based train count: {n_train} (ratio={args.train_ratio})")

    n_val = int(total * args.val_ratio)
    n_test = total - n_train - n_val

    if n_test < 0:
        n_test = 0
        n_val = total - n_train

    train_data = samples[:n_train]
    val_data = samples[n_train:n_train + n_val]
    test_data = samples[n_train + n_val:]

    write_jsonl(root_dir / "train.jsonl", train_data)
    write_jsonl(root_dir / "val.jsonl", val_data)
    write_jsonl(root_dir / "test.jsonl", test_data)

    print("✅ Done! Files generated:")
    print(f"  train.jsonl  ({len(train_data)} samples)")
    print(f"  val.jsonl    ({len(val_data)} samples)")
    print(f"  test.jsonl   ({len(test_data)} samples)")


if __name__ == "__main__":
    main()
