from project_name.training import DiffusionModelTrainer, LoraConfig, TrainerConfig
from baselines.dy_dit import DynamicDiTBlockHook

HookType = DynamicDiTBlockHook
hook_config = {
    "config": {
        "dynamic_width": {
            "enable": False,
            "select_threshold": 0.5,
        },
        "dynamic_token": {
            "enable": True,
            "select_threshold": 0.5,
            "tau": 5.0,
        },
    },
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
    "metadata_file": "../dataset/MJHQ-30K/mjhq_router_test_10.jsonl",
    "image_dir": "../dataset/MJHQ-30K/",
    "model_type": "sd3",  # "pixart-sigma", "flux-dev"
    "model_path": None,
    "output_dir": "./checkpoints/",
    "training_mode": "hook_only",  # "hook_only" "finetune" "lora"
    "hook_warmup_steps": 0,
    "hook_only_steps": 0,
    "lora_config": LoraConfig(**lora_config),
    "trainable_hook_config": (HookType, hook_config),
    "caption_dropout_rate": 0.25,
    "num_epochs": 1000,
    "train_batch_size": 1,
    "gradient_accumulation_steps": 4,
    "learning_rate": 1e-6,
    "optimizer": "AdamW",
    "weight_decay": 1e-2,
    "lr_scheduler": "cosine",  # "linear", "cosine", "constant"
    "num_warmup_steps": 0,
    "main_loss_weight": 1.0,
    "hook_loss_weight": 1.0,
    "seed": 42,
    "logging_steps": 5,
    "save_steps": 500,
    "distillation_final_weight": 0.1,
    "distillation_middle_weight": 1e-4,
    "distillation_layer_skip": 4,
    "run_validation": True,
    "validation_steps": 50,
    "val_batch_size": None,
    "val_metadata_file": "../dataset/MJHQ-30K/mjhq_router_test_10.jsonl",
    "val_image_dir": "../dataset/MJHQ-30K/",
    "run_inference_validation": True,
    "inference_steps": 100,
    "inference_metadata_file": "../dataset/MJHQ-30K/mjhq_router_test_10.jsonl",
    "inference_seed": 42,
    "inference_num_inference_steps": 20,
    "inference_output_dir": "inference_samples",
    "mixed_precision": "bf16",  # "no", "fp16", "bf16"
    "extra_config": {
        "target_flops": 0.7,
        "flops_weight": 5.0,
    },
}

config = TrainerConfig(**trainer_config)
trainer = DiffusionModelTrainer(config=config)
trainer.train()
