from shiva_dit.training import DiffusionModelTrainer, LoraConfig, TrainerConfig
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
    "target_modules": [
        "to_q", "to_k", "to_v", "to_out.0",
        "add_q_proj", "add_k_proj", "add_v_proj",
        "linear", "proj_mlp", 
        "net.2", 
    ],
    "modules_to_save": ["norm1", "norm1_context", "proj_out"],
    "use_dora": True,
}

trainer_config = {
    "metadata_file": "../dataset/MJHQ-30K/train.jsonl",
    "image_dir": "../dataset/MJHQ-30K/",
    "model_type": "sd3",
    "model_path": None,
    "output_dir": "./checkpoints/",
    "training_mode": "lora",
    "lora_config": LoraConfig(**lora_config),
    "trainable_hook_config": (HookType, hook_config),
    "caption_dropout_rate": 0.1,
    "num_epochs": 30,
    "train_batch_size": 1,
    "gradient_accumulation_steps": 2,
    "learning_rate": 1e-4,
    "lora_learning_rate": 1e-4,
    "optimizer": "AdamW",
    "weight_decay": 1e-2,
    "seed": 42,
    "logging_steps": 40,
    "save_steps": 1750, # 2 epoch
    "save_best": True,
    "run_validation": True,
    "validation_steps": 875, # 0.5 epoch
    "val_batch_size": None,
    "val_metadata_file": "../dataset/MJHQ-30K/val.jsonl",
    "val_image_dir": "../dataset/MJHQ-30K/",
    "run_inference_validation": True,
    "inference_steps": 1750, # 1 epoch
    "inference_metadata_file": "../dataset/MJHQ-30K/test.jsonl",
    "inference_seed": 42,
    "inference_num_inference_steps": 28,
    "inference_output_dir": "inference_samples",
    "mixed_precision": "bf16",  # "no", "fp16", "bf16"
    "extra_config": {
        "ratio_loss_weight": 0.3,
        "target_ratio": 0.65,
    },
    "log_with_tensorboard": True,
}

config = TrainerConfig(**trainer_config)
trainer = DiffusionModelTrainer(config=config)
trainer.train()
