from project_name.training import DiffusionModelTrainer, LoraConfig, TrainerConfig
from baselines.diffcr import DiffCRDitBlockHook

HookType = DiffCRDitBlockHook
hook_config = {
    "ratio_min": 0.1,
    "ratio_max": 1.0,
    "ratio_step": -0.1,
}
lora_config = {
    "enable": True,
    "rank": 32,
    "lora_alpha": 32,
    "target_modules": ("to_q", "to_k", "to_v", "to_out.0"),
}

trainer_config = {
    "metadata_file": "../dataset/MJHQ-5K/train_meta.jsonl",
    "image_dir": "../dataset/MJHQ-5K/",
    "model_type": "pixart-sigma",  # "pixart-sigma", "flux-dev"
    "model_path": None,
    "output_dir": "./checkpoints/",
    "training_mode": "lora",  # "hook_only" "finetune" "auto-finetune" "lora" "auto-lora"
    "hook_warmup_steps": 0,
    "hook_only_steps": 0,
    "lora_config": LoraConfig(**lora_config),
    "trainable_hook_config": (HookType, hook_config),
    "caption_dropout_rate": 0.25,
    "num_epochs": 1000,
    "train_batch_size": 1,
    "gradient_accumulation_steps": 16,
    "learning_rate": 1e-4,
    "optimizer": "AdamW",
    "weight_decay": 1e-2,
    "lr_scheduler": "cosine", # "linear", "cosine", "constant"
    "num_warmup_steps": 0,
    "main_loss_weight": 1.0,
    "hook_loss_weight": 1.0,
    "seed": 42,
    "logging_steps": 5,
    "save_steps": 612,
    "run_validation": True,
    "validation_steps": 153,
    "val_batch_size": None,
    "val_metadata_file": "../dataset/MJHQ-5K/val_meta.jsonl",
    "val_image_dir": "../dataset/MJHQ-5K/",
    "run_inference_validation": True,
    "inference_steps": 612,
    "inference_metadata_file": "../dataset/MJHQ-5K/test_meta.jsonl",
    "inference_seed": 42,
    "inference_num_inference_steps": 20,
    "inference_output_dir": "inference_samples",
    "mixed_precision": "bf16",  # "no", "fp16", "bf16"
    "extra_config": {
        "ratio_loss_weight": 0,
        "target_ratio": 0.8,
    },
}

config = TrainerConfig(**trainer_config)
trainer = DiffusionModelTrainer(config=config)
trainer.train()
