from shiva_dit.training import DiffusionModelTrainer, LoraConfig, TrainerConfig
from shiva_dit.shiva import ShivaDitBlockHook

HookType = ShivaDitBlockHook
hook_config = {
    "initial_kept_ratio": 1.0,
    
    "num_router_groups": 4,      # SD3 (L=24) / 4 = 6 layers per group

    "bottleneck_dim": 64,
    "layer_embed_dim": 64,

    "fuse_expert_ffn_mult": 2,
    "use_gater_lut": False,
    
    "context_pruning_threshold": None,
    "context_pruning_ratio": None,      
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
    "hook_warmup_steps": 3500, # 2 epoch
    "hook_only_steps": 0,
    "lora_config": LoraConfig(**lora_config),
    "trainable_hook_config": (HookType, hook_config),
    "caption_dropout_rate": 0.25,
    "num_epochs": 500,
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
    "save_steps": 1750, # 1 epoch
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
        "anneal_steps": 35000, # 20 epoch
        "ratio_loss_weight": 0.01,
        "target_ratio": 0.8,

        "schedule_steps": [8750, 14000], # 5 epoch, 8 epoch
        
        # (L_gate = Entropy)
        "gate_schedule": [1e-9, 1e-9, 0.01],
        
        # (L_con = MSE(prune, fuse))
        "consistency_schedule": [1.0, 0.1, 0.2],
    },
    "log_with_tensorboard": True,
    "log_with_wandb": True,
    "wandb_project": "shiva-dit",
    "wandb_run_name": "sd3-shiva",
    "log_with_mlflow": False,
    "mlflow_experiment_name": "shiva-dit",
    "mlflow_run_name": "sd3-shiva",
    "mlflow_tracking_uri": None,
}

config = TrainerConfig(**trainer_config)
trainer = DiffusionModelTrainer(config=config)
trainer.train()
