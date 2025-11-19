from shiva_dit.training import DiffusionModelTrainer, LoraConfig, TrainerConfig
from shiva_dit.shiva_hook import ShivaDitBlockHook

HookType = ShivaDitBlockHook
hook_config = {
    "num_router_groups": 24,
    "bottleneck_dim": 256,
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
        "target_ratio": 0.8,
        "stage_schedule": [
            # ------------------------------------------------------------------
            # Phase 1: Router Warmup (0 - 1000步)
            # ------------------------------------------------------------------
            # 目的: 仅仅让 Router 学会“谁重要”。Policy 稍微动一点点 (5e-5)。
            # Ts=[20, 20]: 排序非常清晰 (High Confidence)。
            # Tk=[2, 2]:   选择非常模糊 (Soft Selection)，梯度回传给所有人。
            # RatioW=0.0:  不约束比例，先让 Router 自由生长。
            {
                "steps": 1750,
                "weights": {"ratio": 0.0, "gate": 0.0}, 
                "lr": [1e-5, 1e-4, 0.0],  # Policy LR = 0
                "temp": [20.0, 20.0, 2.0, 2.0]  # [Ts_start, Ts_end, Tk_start, Tk_end]
            },

            # ------------------------------------------------------------------
            # Phase 2: Anchor & Harden (1000 - 4500步)
            # ------------------------------------------------------------------
            # 目的: 强行把 Ratio 拉回 0.8 (Weight=1.0)，同时让选择变硬。
            # Ts=[20, 30]: 保持高精度排序。
            # Tk=[2, 5]:   选择开始变硬，梯度逐渐聚焦到 Top-K 边界上。
            {
                "steps": 3500, 
                "weights": {"ratio": 1.0, "gate": 0.0}, 
                "lr": [1e-5, 1e-4, 5e-5], 
                "temp": [20.0, 30.0, 2.0, 5.0] 
            },

            # ------------------------------------------------------------------
            # Phase 3: Differentiation (4500 - 25000步)
            # ------------------------------------------------------------------
            # 目的: 放开约束 (0.05)，让 Main Loss 驱动分化。
            # Ts=[30, 50]: 接近完美排序。
            # Tk=[5, 20]:  接近硬截断 (Hard Top-K)。此时梯度只在边界 Token 上有值。
            {
                "steps": 21000, 
                "weights": {"ratio": 0.05, "gate": 0.0}, 
                "lr": [1e-5, 1e-4, 2e-5], 
                "temp": [30.0, 50.0, 5.0, 20.0] 
            },

            # ------------------------------------------------------------------
            # Phase 4: Gater & Stability (25000+ 步)
            # ------------------------------------------------------------------
            # 目的: 引入 Fusion，稳定训练。
            {
                "steps": -1, 
                "weights": {"ratio": 0.05, "gate": 0.5}, 
                "lr": [1e-5, 1e-4, 1e-5], 
                "temp": [50.0, 50.0, 20.0, 20.0] 
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
