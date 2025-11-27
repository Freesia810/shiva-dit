import os
from shiva_dit.training import DiffusionModelTrainer, LoraConfig, TrainerConfig
from shiva_dit.shiva_hook import ShivaDitBlockHook

HookType = ShivaDitBlockHook
hook_config = {
    # --- Mode Selection ---
    "shiva_mode": "context",
    
    # --- Context Configs ---
    "context_ratio_mode": "fixed",
    "fixed_context_ratio": 0.8,  
    "context_proj_dim": 128,
    
    # --- Common ---
    "init_Ts": 2.0,
    "init_Tk": 2.0,

    "use_gater_lut": False,
}

lora_config = { "enable": False }

trainer_config = {
    "metadata_file": "../dataset/MagicBrush/train.jsonl",
    "image_dir": None,
    "model_type": "qwen-image-edit",
    "model_path": None,
    "output_dir": "./checkpoints/",
    "training_mode": "hook_only",
    "hook_only_steps": 0,
    "lora_config": LoraConfig(**lora_config),
    "trainable_hook_config": (HookType, hook_config),
    "caption_dropout_rate": 0.0,
    "num_epochs": 50,
    "train_batch_size": 1,
    "gradient_accumulation_steps": 16,
    "optimizer": "AdamW",
    "weight_decay": 1e-2,
    "distillation_final_weight": 1.0,
    "distillation_middle_weight": 0.0,
    "seed": 42,
    "logging_steps": 20,
    "save_steps": 1750,
    "save_best": True,
    "run_validation": True,
    "validation_steps": 875,
    "val_batch_size": None,
    "val_metadata_file": "../dataset/MagicBrush/val.jsonl",
    "val_image_dir": "../dataset/MJHQ-30K/",
    "run_inference_validation": True,
    "inference_steps": 1750,
    "inference_metadata_file": "../dataset/MagicBrush/test.jsonl",
    "inference_seed": 42,
    "inference_num_inference_steps": 28,
    "inference_output_dir": "inference_samples",
    "mixed_precision": "bf16",
    "extra_config": {
        "target_ratio": 1.0,
        # LR Order: [Base, LoRA, Gate, Fuser, ImageRatio, ContextRatio, ContextPruner]
        "stage_schedule": [
            {
                "steps": -1,
                "weights": {
                    "ratio": [0.0, 0.0, 0.0],
                    "gate": 0.0,
                    "distill": [0.0, 0.0],
                    "aux": 0.1,
                },
                "lr": [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1e-4],
                "temp": [2.0, 2.0, 2.0, 2.0],
            },
        ],
    },
    "log_with_tensorboard": True,
}

config = TrainerConfig(**trainer_config)
trainer = DiffusionModelTrainer(config=config)
trainer.train()
