from project_name.training import DiffusionModelTrainer, LoraConfig, TrainerConfig
from baselines.diffcr import DiffCRDitBlockHook

if __name__ == "__main__":
    HookType = DiffCRDitBlockHook
    hook_config = {
        "ratio_min": 0.1,
        "ratio_max": 1.0,
        "ratio_step": -0.1,
    }
    lora_config = {
        "enable": True,
        "rank": 32,
        "target_modules": ("to_q", "to_k", "to_v", "to_out.0"),
    }
    
    trainer_config = {
        "metadata_file": "./test/training/dataset/meta.json",
        "image_dir": "./test/training/dataset",
        "model_type": "flux-dev",
        "model_path": None,
        "output_dir": "./checkpoints/",
        "training_mode": "lora", # "hook_only" "finetune" "lora"
        "lora_config": LoraConfig(**lora_config),
        "trainable_hook_config": (HookType, hook_config),
        "num_epochs": 10000,
        "train_batch_size": 1,
        "gradient_accumulation_steps": 8,
        "learning_rate": 1e-4,
        "optimizer": "AdamW",
        "weight_decay": 3e-2,
        "lr_scheduler": "cosine",
        "num_warmup_steps": 0,
        "main_loss_weight": 1.0,
        "hook_loss_weight": 0.3,
        "seed": 42,
        "logging_steps": 1,
        "save_steps": 5000,
        "run_validation": True,
        "validation_steps": 20,
        "val_batch_size": None,
        "val_metadata_file": "./test/training/dataset/meta.json",
        "val_image_dir": "./test/training/dataset",
        "run_inference_validation": True,
        "inference_steps": 100,
        "inference_metadata_file": "./test/training/dataset/meta.json",
        "inference_seed": 42,
        "inference_num_inference_steps": 50,
        "inference_output_dir": "inference_samples",
        "mixed_precision": "bf16",
        "extra_config": {
            "target_ratio": 0.5,
        }
    }

    config = TrainerConfig(**trainer_config)
    trainer = DiffusionModelTrainer(config=config)
    trainer.train()