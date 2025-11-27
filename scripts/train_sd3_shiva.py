from shiva_dit.training import DiffusionModelTrainer, LoraConfig, TrainerConfig
from shiva_dit.shiva_hook import ShivaDitBlockHook

HookType = ShivaDitBlockHook
hook_config = {
    # --- Mode Selection ---
    "shiva_mode": "image",

    # --- Image Configs ---
    "image_ratio_mode": "adaptive",
    "num_router_groups": 24,
    "bottleneck_dim": 256,
    "ratio_bottleneck_dim": 32,
    "layer_embed_dim": 256,
    "fuse_expert_ffn_mult": 2,
    "use_gater_lut": False,
    
    # --- Common ---
    "init_Ts": 10.0,
    "init_Tk": 10.0,
}

lora_config = {
    "enable": True,
    "rank": 32,
    "lora_alpha": 32,
    "target_modules": [
        "to_q", "to_k", "to_v", "to_out.0",
        "linear", "proj_mlp", "proj_out", 
        "net.2", 
    ],
}

trainer_config = {
    "metadata_file": "../dataset/MJHQ-30K/train.jsonl",
    "image_dir": "../dataset/MJHQ-30K/",
    "model_type": "sd3",
    "model_path": None,
    "output_dir": "./checkpoints",
    "training_mode": "lora",
    "hook_only_steps": 0,
    "lora_config": LoraConfig(**lora_config),
    "trainable_hook_config": (HookType, hook_config),
    "caption_dropout_rate": 0.1,
    "num_epochs": 50,
    "train_batch_size": 1,
    "gradient_accumulation_steps": 16,
    "optimizer": "AdamW",
    "weight_decay": 1e-2,
    "distillation_final_weight": 1.0,
    "distillation_middle_weight": 0.0,
    "seed": 42,
    "logging_steps": 40,
    "save_steps": 1750,
    "save_best": True,
    "run_validation": True,
    "validation_steps": 875,
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
        # LR Order Mapping in Trainer: 
        # [0:Base/Router, 1:LoRA, 2:Gate, 3:Fuser, 4:ImageRatio(Policy), 5:CtxRatio, 6:CtxPruner]
        "stage_schedule": [
            # ------------------------------------------------------------------
            # Stage 0: Router Warmup (Random Ratio Phase)
            # ------------------------------------------------------------------
            {
                "steps": 3500, # 约 2 epochs
                "weights": {
                    "ratio": [1.0, 50.0, 1.0], 
                    "gate": 0.0,
                    "distill": [1.0, 0.0]
                },
                "lr": [5e-4, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                "temp": [2.0, 2.0, 2.0, 2.0],
            },

            # ------------------------------------------------------------------
            # Stage 1: Policy Wakeup & Adaptive Fusion
            # ------------------------------------------------------------------
            {
                "steps": 3500,
                "weights": {
                    "ratio": [1.0, 50.0, 1.0], 
                    "gate": 1.0,
                    "distill": [1.0, 0.0]
                },
                "lr": [1e-4, 0.0, 2e-4, 2e-4, 5e-4, 0.0, 0.0],
                "temp": [5.0, 5.0, 5.0, 5.0],
            },

            # ------------------------------------------------------------------
            # Stage 2: Final Finetune (Hardening)
            # ------------------------------------------------------------------
            {
                "steps": -1,
                "weights": {
                    "ratio": [1.0, 50.0, 1.0], 
                    "gate": 1.0,
                    "distill": [1.0, 0.0]
                },
                "lr": [1e-5, 1e-4, 1e-5, 1e-5, 1e-5, 0.0, 0.0],
                "temp": [5.0, 10.0, 5.0, 50.0],
            },
        ],
    },
    "log_with_tensorboard": True,
    "log_with_wandb": False,
    "wandb_project": "shiva-dit",
    "wandb_run_name": "sd3-shiva-optimized",
}

print(hook_config)
print(lora_config)
print(trainer_config)

config = TrainerConfig(**trainer_config)
trainer = DiffusionModelTrainer(config=config)
trainer.train()