"""Shared command-line arguments for the Shiva training recipes."""

import argparse

from shiva_dit.training import DiffusionModelTrainer, TrainerConfig


def train(recipe):
    parser = argparse.ArgumentParser(description=f"Train Shiva-DiT on {recipe['model_type']}")
    paths = {
        "model_path": "--base-model-path",
        "metadata_file": "--train-jsonl",
        "image_dir": "--image-dir",
        "val_metadata_file": "--val-jsonl",
        "val_image_dir": "--val-image-dir",
        "inference_metadata_file": "--inference-jsonl",
        "output_dir": "--output-dir",
    }
    for field, flag in paths.items():
        parser.add_argument(flag, dest=field, default=recipe[field])
    args = parser.parse_args()
    config = TrainerConfig(**(recipe | vars(args)))
    DiffusionModelTrainer(config=config).train()
