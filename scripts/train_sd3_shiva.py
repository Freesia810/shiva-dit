from shiva_dit.training import DiffusionModelTrainer, LoraConfig, TrainerConfig
from shiva_dit.shiva_hook import ShivaDitBlockHook

HookType = ShivaDitBlockHook
hook_config = {
    "num_router_groups": 24,
    "bottleneck_dim": 256,
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
        "net.2",
    ],
}

trainer_config = {
    "metadata_file": "../dataset/MJHQ-30K/train.jsonl",
    "image_dir": "../dataset/MJHQ-30K/",
    "model_type": "sd3",  # "pixart-sigma", "flux-dev"
    "model_path": None,
    "output_dir": "./checkpoints/",
    "training_mode": "lora",  # "hook_only" "finetune" "lora"
    "hook_only_steps": 0,
    "lora_config": LoraConfig(**lora_config),
    "trainable_hook_config": (HookType, hook_config),
    "caption_dropout_rate": 0.25,
    "num_epochs": 50,
    "train_batch_size": 1,
    "gradient_accumulation_steps": 16,
    "learning_rate": 1e-5,
    "lora_learning_rate": 1e-4,
    "policy_learning_rate": 1e-5,
    "optimizer": "AdamW",
    "weight_decay": 1e-2,
    "lr_scheduler": "cosine",  # "linear", "cosine", "constant"
    "scheduler_num_warmup_steps": 0,
    "seed": 42,
    "logging_steps": 40,
    "save_steps": 1750,  # 2 epoch
    "save_best": True,
    "run_validation": True,
    "validation_steps": 875,  # 0.5 epoch
    "val_batch_size": None,
    "val_metadata_file": "../dataset/MJHQ-30K/val.jsonl",
    "val_image_dir": "../dataset/MJHQ-30K/",
    "run_inference_validation": True,
    "inference_steps": 1750,  # 1 epoch
    "inference_metadata_file": "../dataset/MJHQ-30K/test.jsonl",
    "inference_seed": 42,
    "inference_num_inference_steps": 30,
    "inference_output_dir": "inference_samples",
    "mixed_precision": "bf16",  # "no", "fp16", "bf16"
    "extra_config": {
        "target_ratio": 0.5,
        "stage_schedule": [
            {
                "steps": 1750, # 1 epoch
                "weights": {"ratio": [0.0, 0.0, 0.0], "gate": 0.0},
                "lr": [1e-5, 1e-4, 0.0, 0.0, 0.0],  # [Base, LoRA, Ratio, Gate, Fuser]
                "temp": [5.0, 5.0, 2.0, 2.0], # [Ts-start, Ts-end, Tk-start, Tk-end]
            },
            # ------------------------------------------------------------------
            {
                "steps": 5250, # 3 epoch
                "weights": {"ratio": [0.0, 0.0, 0.0], "gate": 0.0},
                "lr": [1e-5, 1e-5, 0.0, 0.0, 0.0],
                "temp": [5.0, 5.0, 2.0, 2.0],
            },

            # ------------------------------------------------------------------
            {
                "steps": 7000, # 4 epoch
                "weights": {"ratio": [1.0, 50.0, 1.0], "gate": 0.0}, # [Budget(Hinge), Floor(Wall), MSE(Smooth)]
                "lr": [0.0, 1e-4, 5e-4, 0.0, 0.0],  # [Base, LoRA, Ratio, Gate, Fuser]
                "temp": [5.0, 10.0, 2.0, 10.0],
            },
            # ------------------------------------------------------------------
            {
                "steps": 14000, # 8 epoch
                "weights": {"ratio": [1.0, 50.0, 1.0], "gate": 0.0},
                "lr": [0.0, 1e-5, 1e-5, 0.0, 0.0],  # [Base, LoRA, Ratio, Gate, Fuser]
                "temp": [10.0, 10.0, 10.0, 50.0],
            },

            {
                "steps": 10500, # 6 epoch
                "weights": {"ratio": [1.0, 50.0, 1.0], "gate": 0.0},
                "lr": [0.0, 1e-5, 1e-5, 0.0, 0.0],  # [Base, LoRA, Ratio, Gate, Fuser]
                "temp": [10.0, 10.0, 50.0, 50.0],
            },
            # # ------------------------------------------------------------------
            {
                "steps": 7000, # 4 epoch
                "weights": {"ratio": [1.0, 50.0, 1.0], "gate": 0.1},
                "lr": [0.0, 0.0, 0.0, 1e-4, 2e-4],  # [Base, LoRA, Ratio, Gate, Fuser]
                "temp": [10.0, 10.0, 50.0, 50.0],
            },

            {
                "steps": -1,
                "weights": {"ratio": [1.0, 50.0, 1.0], "gate": 0.1},
                "lr": [0.0, 1e-5, 1e-5, 1e-5, 1e-5],  # [Base, LoRA, Ratio, Gate, Fuser]
                "temp": [10.0, 10.0, 50.0, 50.0],
            },
        ],
    },
    "log_with_tensorboard": True,
    "log_with_wandb": False,
    "wandb_project": "shiva-dit",
    "wandb_run_name": "sd3-shiva",
    "log_with_mlflow": False,
    "mlflow_experiment_name": "shiva-dit",
    "mlflow_run_name": "sd3-shiva",
    "mlflow_tracking_uri": None,
}

print(hook_config)
print(lora_config)
print(trainer_config)

config = TrainerConfig(**trainer_config)
trainer = DiffusionModelTrainer(config=config)
trainer.train()
