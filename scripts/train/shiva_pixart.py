from shiva_dit.training import LoraConfig
from scripts.train.run_shiva import train
from shiva_dit.shiva_hook import ShivaDitBlockHook

HookType = ShivaDitBlockHook
hook_config = {
    "shiva_mode": "image",
    "image_ratio_mode": "adaptive",
    "fixed_image_ratio": 0.9,
    "num_router_groups": 14,
    "bottleneck_dim": 64,
    "ratio_bottleneck_dim": 256,
    "layer_embed_dim": 64,
    "fuse_expert_ffn_mult": 2,
    "use_gater_lut": False,
    "init_Ts": 10.0,
    "init_Tk": 10.0,
}

lora_config = {
    "enable": True,
    "rank": 64,
    "lora_alpha": 128,
    "target_modules": [
        "to_q", "to_k", "to_v", "to_out.0",
        "add_q_proj", "add_k_proj", "add_v_proj", "to_add_out",
        "net.0.proj", "net.2",
        "proj_mlp",
        "proj_out"
    ],
    "modules_to_save": [
        "norm1", 
        "norm1_context",
        "norm_out",
    ],
    "use_dora": True,
}

trainer_config = {
    "metadata_file": "../dataset/MJHQ-30K/train.jsonl",
    "image_dir": "../dataset/MJHQ-30K/",
    "model_type": "pixart-sigma",
    "model_path": None,
    "output_dir": "./checkpoints",
    "training_mode": "lora",
    "lora_config": LoraConfig(**lora_config),
    "trainable_hook_config": (HookType, hook_config),
    "caption_dropout_rate": 0.1,
    "num_epochs": 30,
    "train_batch_size": 1,
    "gradient_accumulation_steps": 2,
    "optimizer": "AdamW",
    "learning_rate": 1e-4,
    "lora_learning_rate": 1e-4,
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
        "target_ratio": 0.9,
        "stage_schedule": [
            # router stage
            {
                "steps": 3500,
                "weights": {
                    "ratio": [0.0, 0.0, 0.0],
                    "gate": 0.0,
                    "distill": [0.1, 1.0],
                },
                "lr": {
                    "router": 3e-4,
                },
                "warmup": 200,
                "temp": [5.0, 15.0, 10.0, 20.0],
            },
            # time stage
            {
                "steps": 5250,
                "weights": {
                    "ratio": [0.0, 0.0, 1.0],
                    "gate": 0.0,
                    "distill": [0.1, 0.0],
                },
                "lr": {
                    "lora": 5e-5,
                    "router": 1e-6,
                    "image_ratio_policy": 1e-4,
                },
                "warmup": 500,
                "temp": [15.0, 15.0, 20.0, 25.0],
            },
            # layer stage
            {
                "steps": 5250,
                "weights": {
                    "ratio": [0.0, 0.0, 1.0],
                    "gate": 0.0,
                    "distill": [0.1, 0.0],
                },
                "lr": {
                    "lora": 5e-5,
                    "router": 1e-6,
                    "image_ratio_policy": 1e-4,
                },
                "temp": [15.0, 15.0, 25.0, 30.0],
            },
            # final stage
            {
                "steps": -1,
                "weights": {
                    "ratio": [0.0, 0.0, 0.0],
                    "gate": 0.0,
                    "distill": [2.0, 0.0],
                },
                "lr": {
                    "lora": 1e-4,
                    "router": 1e-5,
                },
                "temp": [15.0, 15.0, 30.0, 33.0],
            },
        ],
    },
    "log_with_tensorboard": True,
}


if __name__ == "__main__":
    train(trainer_config)
