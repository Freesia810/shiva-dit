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
    "logging_steps": 5,
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
            # Phase 1: Router Warmup (0 - 1000步)
            # Policy 冻结 (LR=0)。让 Router 先学会“谁是垃圾 Token”。
            # Ratio 保持在初始值 (由 logit_bias=1.386 决定，约 0.8)。
            {
                "steps": 7000, 
                "weights": {"ratio": 0.0, "gate": 0.0}, 
                "lr": [1e-5, 1e-4, 0.0], 
                "temp": [1.0, 1.0]
            },

            # Phase 2: Anchor & Constraint (1000 - 4000步)
            # 开启 Policy 训练。
            # 使用 1.0 的强权重，对抗 Main Loss 的上升推力，确保平均值死死锚定在 0.8。
            # 这是为了让 Policy 学会：“由于 Budget 有限，我必须做选择”。
            {
                "steps": 3500, 
                "weights": {"ratio": 1.0, "gate": 0.0}, 
                "lr": [1e-5, 1e-4, 5e-5], 
                "temp": [1.0, 2.0]
            },

            # Phase 3: Differentiation / Exploration (4000 - 20000步) - 关键阶段
            # 只有在这里，我们才期望看到分化。
            # 权重降至 0.05。此时 Main Loss 的局部敏感度差异将主导梯度。
            # 如果这一阶段 Ratio 还是不分化，说明 Main Loss 对 Token 数量极其敏感。
            {
                "steps": 17500, 
                "weights": {"ratio": 0.05, "gate": 0.0}, 
                "lr": [1e-5, 1e-4, 5e-5], 
                "temp": [2.0, 4.0]
            },

            # Phase 4: Stability & Gater (20000+ 步)
            # 引入 Gate Loss，启用 Fusion 训练的正则化。
            {
                "steps": -1, 
                "weights": {"ratio": 0.05, "gate": 0.5}, 
                "lr": [1e-5, 1e-4, 1e-5], 
                "temp": [4.0, 5.0]
            }
        ]
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
