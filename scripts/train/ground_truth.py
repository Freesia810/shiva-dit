from shiva_dit.training import DiffusionModelTrainer, LoraConfig, TrainerConfig

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
    "model_type": "sd3",
    "model_path": None,
    "output_dir": f"./checkpoints",
    "training_mode": "lora",
    "lora_config": LoraConfig(**lora_config),
    "caption_dropout_rate": 0.1,
    "num_epochs": 30,
    "train_batch_size": 1,
    "gradient_accumulation_steps": 2,
    "optimizer": "AdamW",
    "learning_rate": 1e-4,
    "lora_learning_rate": 1e-4,
    "weight_decay": 1e-2,
    "distillation_final_weight": 0.0,
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
    "log_with_tensorboard": True,
}

if __name__ == "__main__":
    print(lora_config)
    print(trainer_config)

    config = TrainerConfig(**trainer_config)
    trainer = DiffusionModelTrainer(config=config)
    trainer.train()
