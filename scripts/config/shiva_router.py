from shiva_dit.training import DiffusionModelTrainer, LoraConfig, TrainerConfig
from shiva_dit.shiva_hook import ShivaDitBlockHook

HookType = ShivaDitBlockHook
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

hook_config_max = {
    "shiva_mode": "image",
    "image_ratio_mode": "adaptive",
    "fixed_image_ratio": 0.6,
    "num_router_groups": 24,
    "bottleneck_dim": 512,
    "layer_embed_dim": 64,
    "fuse_expert_ffn_mult": 2,
    "use_gater_lut": False,
    
    # --- Common ---
    "init_Ts": 10.0,
    "init_Tk": 10.0,
}

hook_config_slim = {
    "shiva_mode": "image",
    "image_ratio_mode": "adaptive",
    "fixed_image_ratio": 0.6,
    "num_router_groups": 24,
    "bottleneck_dim": 64,
    "layer_embed_dim": 64,
    "fuse_expert_ffn_mult": 2,
    "use_gater_lut": False,
    
    # --- Common ---
    "init_Ts": 10.0,
    "init_Tk": 10.0,
}

hook_config_tiny = {
    "shiva_mode": "image",
    "image_ratio_mode": "adaptive",
    "fixed_image_ratio": 0.6,
    "num_router_groups": 1,
    "bottleneck_dim": 64,
    "layer_embed_dim": 64,
    "fuse_expert_ffn_mult": 2,
    "use_gater_lut": False,
    
    # --- Common ---
    "init_Ts": 10.0,
    "init_Tk": 10.0,
}

hook_config_shared = {
    "shiva_mode": "image",
    "image_ratio_mode": "adaptive",
    "fixed_image_ratio": 0.6,
    "num_router_groups": 1,
    "bottleneck_dim": 512,
    "layer_embed_dim": 128,
    "fuse_expert_ffn_mult": 2,
    "use_gater_lut": False,
    
    # --- Common ---
    "init_Ts": 10.0,
    "init_Tk": 10.0,
}

hook_config_grouped_4 = {
    "shiva_mode": "image",
    "image_ratio_mode": "adaptive",
    "fixed_image_ratio": 0.6,
    "num_router_groups": 4,
    "bottleneck_dim": 128,
    "layer_embed_dim": 64,
    "fuse_expert_ffn_mult": 2,
    "use_gater_lut": False,
    
    # --- Common ---
    "init_Ts": 10.0,
    "init_Tk": 10.0,
}

hook_config_grouped_8 = {
    "shiva_mode": "image",
    "image_ratio_mode": "adaptive",
    "fixed_image_ratio": 0.6,
    
    "num_router_groups": 8,
    "bottleneck_dim": 64,
    "layer_embed_dim": 64,
    "fuse_expert_ffn_mult": 2,
    "use_gater_lut": False,
    
    "init_Ts": 10.0,
    "init_Tk": 10.0,
}

hook_config_grouped_12 = {
    "shiva_mode": "image",
    "image_ratio_mode": "adaptive",
    "fixed_image_ratio": 0.6,
    
    "num_router_groups": 12,
    "bottleneck_dim": 64,
    "layer_embed_dim": 64,
    "fuse_expert_ffn_mult": 2,
    "use_gater_lut": False,
    
    "init_Ts": 10.0,
    "init_Tk": 10.0,
}

hook_config = {
    "max": hook_config_max,
    "slim": hook_config_slim,
    "tiny": hook_config_tiny,
    "shared": hook_config_shared,
    "grouped_4": hook_config_grouped_4,
    "grouped_8": hook_config_grouped_8,
    "grouped_12": hook_config_grouped_12,
}

prefix = 'max'

trainer_config = {
    "metadata_file": "../dataset/MJHQ-30K/train.jsonl",
    "image_dir": "../dataset/MJHQ-30K/",
    "model_type": "sd3",
    "model_path": None,
    "output_dir": f"./checkpoints_{prefix}",
    "training_mode": "lora",
    "lora_config": LoraConfig(**lora_config),
    "trainable_hook_config": (HookType, hook_config[prefix]),
    "caption_dropout_rate": 0.1,
    "num_epochs": 2,
    "train_batch_size": 1,
    "gradient_accumulation_steps": 2,
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
    "run_inference_validation": False,
    "mixed_precision": "bf16",
    "extra_config": {
        "target_ratio": 0.6,
        "stage_schedule": [
            {
                "steps": 3500,
                "weights": {
                    "ratio": [0.0, 0.0, 0.0],
                    "gate": 0.0,
                    "distill": [2.0, 0.0],
                },
                "lr": {
                    "router": 3e-4,
                },
                "warmup": 200,
                "temp": [5.0, 15.0, 10.0, 20.0],
            },
        ],
    },
    "log_with_tensorboard": True,
}

if __name__ == "__main__":
    print(hook_config[prefix])
    print(lora_config)
    print(trainer_config)

    config = TrainerConfig(**trainer_config)
    trainer = DiffusionModelTrainer(config=config)
    trainer.train()