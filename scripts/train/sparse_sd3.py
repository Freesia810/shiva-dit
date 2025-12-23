from shiva_dit.training import DiffusionModelTrainer, LoraConfig, TrainerConfig
from baselines.sparse_dit import SparseDiTBlockHook

HookType = SparseDiTBlockHook

hook_config = {
    "poolingformer_layers": [0, 1], 
    "sdtm_ranges": [(2, 22)], 
    "r_min": 0.61,
    "r_max": 0.86,
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
        "add_q_proj",
        "add_k_proj",
        "add_v_proj",
        "linear",
        "proj_mlp",
        "net.2", 
    ],
    "modules_to_save": ["norm1", "norm1_context", "proj_out"],
    "use_dora": True,
}

trainer_config = {
    "metadata_file": "../dataset/MJHQ-30K/train.jsonl",
    "image_dir": "../dataset/MJHQ-30K/",
    "model_type": "sd3",
    "model_path": None,
    "output_dir": "./checkpoints/",
    "training_mode": "lora",
    "lora_config": LoraConfig(**lora_config),
    "trainable_hook_config": (HookType, hook_config),
    "caption_dropout_rate": 0.1,
    "num_epochs": 30, 
    "train_batch_size": 1,
    "gradient_accumulation_steps": 2,
    "learning_rate": 1e-4, 
    "lora_learning_rate": 1e-4,
    "optimizer": "AdamW",
    "weight_decay": 1e-2,
    "distillation_final_weight": 0.0,
    "distillation_middle_weight": 0.0,
    "seed": 42,
    "logging_steps": 40,
    "save_steps": 1750,
    "save_best": True,
    "run_validation": True,
    "validation_steps": 875,
    "val_batch_size": 1,
    "val_metadata_file": "../dataset/MJHQ-30K/val.jsonl",
    "val_image_dir": "../dataset/MJHQ-30K/",
    "run_inference_validation": True,
    "inference_steps": 1750,
    "inference_metadata_file": "../dataset/MJHQ-30K/test.jsonl",
    "inference_seed": 42,
    "inference_num_inference_steps": 28,
    "inference_output_dir": "inference_samples",
    "mixed_precision": "bf16",
    "log_with_tensorboard": True,
}

config = TrainerConfig(**trainer_config)
trainer = DiffusionModelTrainer(config=config, static_graph=True, find_unused_parameters=True)
trainer.train()