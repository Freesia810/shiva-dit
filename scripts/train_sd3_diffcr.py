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
    "metadata_file": "../dataset/MJHQ-30K/train.jsonl",
    "image_dir": "../dataset/MJHQ-30K/",
    "model_type": "sd3",  # "pixart-sigma", "flux-dev"
    "model_path": None,
    "output_dir": "./checkpoints/",
    "training_mode": "lora",  # "hook_only" "finetune" "auto-finetune" "lora" "auto-lora"
    "hook_warmup_steps": 1750, # 1 epoch
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
    "save_steps": 3500, # 2 epoch
    "run_validation": True,
    "validation_steps": 875, # 0.5 epoch
    "val_batch_size": None,
    "val_metadata_file": "../dataset/MJHQ-30K/val.jsonl",
    "val_image_dir": "../dataset/MJHQ-30K/",
    "run_inference_validation": True,
    "inference_steps": 1750, # 1 epoch
    "inference_metadata_file": "../dataset/MJHQ-30K/test.jsonl",
    "inference_seed": 42,
    "inference_num_inference_steps": 30,
    "inference_output_dir": "inference_samples",
    "mixed_precision": "bf16",  # "no", "fp16", "bf16"
    "extra_config": {
        "ratio_loss_weight": 0.3,
        "target_ratio": 0.5,
    },
    "log_with_tensorboard": True,
    "log_with_wandb": True,
    "wandb_project": "generative-model-boost",
    "wandb_run_name": "sd3-diffcr",
    "log_with_mlflow": True,
    "mlflow_experiment_name": "generative-model-boost",
    "mlflow_run_name": "sd3-diffcr",
    "mlflow_tracking_uri": None,
}

config = TrainerConfig(**trainer_config)
trainer = DiffusionModelTrainer(config=config)
trainer.train()
