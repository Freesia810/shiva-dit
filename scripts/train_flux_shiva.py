from shiva_dit.training import DiffusionModelTrainer, LoraConfig, TrainerConfig
from shiva_dit.shiva_hook import ShivaDitBlockHook

HookType = ShivaDitBlockHook
hook_config = {
    "num_router_groups": 19,
    "bottleneck_dim": 512,
    "ratio_bottleneck_dim": 32,
    "layer_embed_dim": 256,
    "fuse_expert_ffn_mult": 2,
    "use_gater_lut": False,
    "context_pruning_threshold": None,
    "context_pruning_ratio": None,
    "init_Ts": 10.0,
    "init_Tk": 10.0,
}

lora_config = {
    "enable": True,
    "rank": 32,
    "lora_alpha": 32,
    "target_modules": [
        "to_q",
        "to_k",
        "to_v",
        "to_out.0",
        "linear",
        "proj_mlp",
        "proj_out",
    ],
}

trainer_config = {
    "metadata_file": "../dataset/MJHQ-30K/train.jsonl",
    "image_dir": "../dataset/MJHQ-30K/",
    "model_type": "flux-dev",
    "model_path": None,
    "output_dir": "./checkpoints/",
    "training_mode": "lora",
    "hook_only_steps": 0,
    "lora_config": LoraConfig(**lora_config),
    "trainable_hook_config": (HookType, hook_config),
    
    "caption_dropout_rate": 0.1,
    "num_epochs": 50,
    "train_batch_size": 1,
    "gradient_accumulation_steps": 16, # 1750 steps * 16 = 28000 images (approx 1 epoch)
    
    "learning_rate": 1e-5,
    "lora_learning_rate": 1e-4,
    "policy_learning_rate": 1e-5,
    "optimizer": "AdamW",
    "weight_decay": 1e-2,
    "lr_scheduler": "cosine",
    "distillation_final_weight": 1.0, 
    "distillation_middle_weight": 0.0,
    "scheduler_num_warmup_steps": 0,
    "seed": 42,
    "logging_steps": 20,
    
    # 1 epoch = 1750 steps
    "save_steps": 1750,
    "save_best": True,
    "run_validation": True,
    "validation_steps": 875,
    "val_batch_size": None,
    "val_metadata_file": "../dataset/MJHQ-30K/val.jsonl",
    "val_image_dir": "../dataset/MJHQ-30K/",
    
    "run_inference_validation": True,
    "inference_steps": 1750,
    "inference_metadata_file": "../dataset/MJHQ-30K/test.jsonl",
    "inference_seed": 42,
    "inference_num_inference_steps": 28,
    "inference_output_dir": "inference_samples",
    "mixed_precision": "bf16",

    "extra_config": {
        "target_ratio": 0.8,
        "stage_schedule": [
            # Stage 1: Warmup (1 epoch)
            {
                "steps": 1750, 
                "weights": {"ratio": [0.0, 0.0, 0.0], "gate": 0.0, "distill": [0.0, 0.0]},
                "lr": [1e-5, 1e-4, 0.0, 0.0, 0.0],  # [Base, LoRA, Ratio, Gate, Fuser]
                "temp": [5.0, 5.0, 2.0, 2.0],
            },
            # Stage 2: Warmup Cont. (3 epochs)
            {
                "steps": 5250, 
                "weights": {"ratio": [0.0, 0.0, 0.0], "gate": 0.0, "distill": [0.0, 0.0]},
                "lr": [1e-5, 1e-5, 0.0, 0.0, 0.0], # [Base, LoRA, Ratio, Gate, Fuser]
                "temp": [5.0, 5.0, 2.0, 2.0],
            },
            # Stage 3: Train Ratio (4 epochs)
            {
                "steps": 7000, 
                "weights": {"ratio": [1.0, 50.0, 1.0], "gate": 0.0, "distill": [1.0, 0.0]},
                "lr": [0.0, 1e-4, 5e-4, 0.0, 0.0], # [Base, LoRA, Ratio, Gate, Fuser]
                "temp": [5.0, 10.0, 2.0, 10.0],
            },
            # Stage 4: Finetune Model (8 epochs)
            {
                "steps": 14000, 
                "weights": {"ratio": [1.0, 50.0, 1.0], "gate": 0.0, "distill": [1.0, 0.0]},
                "lr": [0.0, 1e-5, 1e-5, 0.0, 0.0], # [Base, LoRA, Ratio, Gate, Fuser]
                "temp": [10.0, 10.0, 10.0, 50.0],
            },
            # Stage 5: Finetune with High Temp (6 epochs)
            {
                "steps": 10500, 
                "weights": {"ratio": [1.0, 50.0, 1.0], "gate": 0.0, "distill": [1.0, 0.0]},
                "lr": [0.0, 1e-5, 1e-5, 0.0, 0.0], # [Base, LoRA, Ratio, Gate, Fuser]
                "temp": [10.0, 10.0, 50.0, 50.0],
            },
            # Stage 6: Train Gate (4 epochs)
            {
                "steps": 7000, 
                "weights": {"ratio": [1.0, 50.0, 1.0], "gate": 0.1, "distill": [1.0, 0.0]},
                "lr": [0.0, 0.0, 0.0, 1e-4, 2e-4], # [Base, LoRA, Ratio, Gate, Fuser]
                "temp": [10.0, 10.0, 50.0, 50.0],
            },
            # Final Stage
            {
                "steps": -1,
                "weights": {"ratio": [1.0, 50.0, 1.0], "gate": 0.1, "distill": [1.0, 0.0]},
                "lr": [0.0, 1e-5, 1e-5, 1e-5, 1e-5], # [Base, LoRA, Ratio, Gate, Fuser]
                "temp": [10.0, 10.0, 50.0, 50.0],
            },
        ],
    },
    "log_with_tensorboard": True,
    "log_with_wandb": False,
    "wandb_project": "shiva-dit",
    "wandb_run_name": "flux-dev-shiva",
    "log_with_mlflow": False,
    "mlflow_experiment_name": "shiva-dit",
    "mlflow_run_name": "flux-dev-shiva",
    "mlflow_tracking_uri": None,
}

print(hook_config)
print(lora_config)
print(trainer_config)

config = TrainerConfig(**trainer_config)
trainer = DiffusionModelTrainer(config=config)
trainer.train()