from shiva_dit.training import DiffusionModelTrainer, LoraConfig, TrainerConfig
from shiva_dit.shiva_hook import ShivaDitBlockHook

HookType = ShivaDitBlockHook

lora_config = {
    "enable": True,
    "rank": 128,
    "lora_alpha": 128,
    "target_modules": [
        "to_q", "to_k", "to_v", "to_out.0",
        "linear", "proj_mlp", "proj_out", 
        "net.2", 
    ],
}

mode = "fixed"
# mode = "adaptive"

hook_config = {
    # --- Mode Selection ---
    "shiva_mode": "image",

    # --- Image Configs ---
    "image_ratio_mode": mode,
    "fixed_image_ratio": 0.6,
    "num_router_groups": 24,
    "bottleneck_dim": 512,
    "ratio_bottleneck_dim": 256,
    "layer_embed_dim": 64,
    "fuse_expert_ffn_mult": 2,
    "use_gater_lut": False,
    
    # --- Common ---
    "init_Ts": 10.0,
    "init_Tk": 10.0,
}

stage_config_dict = {
    "adaptive": [
        {
            "steps": 7000, 
            "weights": {
                "ratio": [0.0, 0.0, 0.0], 
                "gate": 0.0,
                "distill": [0.0, 0.0]
            },
            # LR: [Base(Router), LoRA, Gate, Fuser, Policy, CtxRatio, CtxPruner]
            "lr": [5e-4, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
            "temp": [5.0, 15.0, 10.0, 20.0],
        },

        {
            "steps": 14000,
            "weights": {
                "ratio": [0.0, 10.0, 2.5],
                "gate": 0.0,
                "distill": [0.0, 0.0],
            },
            "lr": [1e-6, 1e-5, 0.0, 0.0, 2e-5, 0.0, 0.0],
            "temp": [15.0, 15.0, 20.0, 50.0],
        },

        {
            "steps": -1,
            "weights": {
                "ratio": [0.0, 10.0, 0.5],
                "gate": 0.0,
                "distill": [0.0, 0.0],
            },
            "lr": [1e-6, 1e-5, 0.0, 0.0, 1e-6, 0.0, 0.0],
            "temp": [15.0, 15.0, 50.0, 100.0],
        },
    ],

    "fixed": [
        {
            "steps": 7000, 
            "weights": {
                "ratio": [0.0, 0.0, 0.0], 
                "gate": 0.0,
                "distill": [0.0, 0.0]
            },
            # LR: [Base(Router), LoRA, Gate, Fuser, Policy, CtxRatio, CtxPruner]
            "lr": [5e-4, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
            "temp": [5.0, 10.0, 10.0, 20.0],
        },

        {
            "steps": 14000, 
            "weights": {
                "ratio": [0.0, 0.0, 0.0],
                "gate": 0.0,
                "distill": [0.0, 0.0],
            },
            "lr": [0.0, 1e-4, 0.0, 0.0, 0.0, 0.0, 0.0],
            "temp": [10.0, 10.0, 20.0, 50.0],
        },

        {
            "steps": -1, 
            "weights": {
                "ratio": [0.0, 0.0, 0.0],
                "gate": 0.0,
                "distill": [0.0, 0.0],
            },
            "lr": [0.0, 1e-4, 0.0, 0.0, 0.0, 0.0, 0.0],
            "temp": [10.0, 10.0, 50.0, 100.0],
        },
    ],
}

trainer_config = {
    "metadata_file": "../dataset/MJHQ-30K/train.jsonl",
    "image_dir": "../dataset/MJHQ-30K/",
    "model_type": "sd3",
    "model_path": None,
    "output_dir": f"./checkpoints_{mode}",
    "training_mode": "lora",
    "hook_only_steps": 0,
    "lora_config": LoraConfig(**lora_config),
    "trainable_hook_config": (HookType, hook_config),
    "caption_dropout_rate": 0.1,
    "num_epochs": 30,
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
        "target_ratio": 0.6,
        # LR Order Mapping in Trainer: 
        # [0:Base/Router, 1:LoRA, 2:Gate, 3:Fuser, 4:ImageRatio(Policy), 5:CtxRatio, 6:CtxPruner]
        "stage_schedule": stage_config_dict[mode],
    },
    "log_with_tensorboard": True,
}

print(hook_config)
print(lora_config)
print(trainer_config)

config = TrainerConfig(**trainer_config)
trainer = DiffusionModelTrainer(config=config)
trainer.train()
