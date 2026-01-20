from shiva_dit.training import ContextPrunerDistiller, LoraConfig, TrainerConfig
from shiva_dit.shiva_hook import ShivaDitBlockHook

HookType = ShivaDitBlockHook
hook_config = {
    "shiva_mode": "context",
    "context_pruner_mode": "learned",
    "context_pool_size": 64,
    "context_proj_dim": 128,
}

lora_config = {
    "enable": False,
    "rank": 64,
    "lora_alpha": 128,
    "target_modules": [
        "to_q", "to_k", "to_v", "to_out.0",
        "add_q_proj", "add_k_proj", "add_v_proj", "to_add_out",
        "net.0.proj", "net.2",
        "single_transformer_blocks.proj_mlp", "single_transformer_blocks.proj_out",
        "proj_out",
    ],
    "use_dora": False
}

dataset = 'MagicBrush'

trainer_config = {
    "metadata_file": f"../dataset/{dataset}/train.jsonl",
    "image_dir": f"../dataset/{dataset}/",
    "model_type": "qwen-image-edit",
    "model_path": None,
    "dataloader_num_workers": 0,
    "output_dir": "./checkpoints",
    "training_mode": "hook_only",
    "lora_config": LoraConfig(**lora_config),
    "trainable_hook_config": (HookType, hook_config),
    "caption_dropout_rate": 0.0,
    "tokenizer_max_length": 1024,
    "enable_gradient_checkpointing": False,
    "num_epochs": 50,
    "train_batch_size": 1,
    "gradient_accumulation_steps": 2,
    "optimizer": "AdamW",
    "context_learning_rate": 1e-4,
    "weight_decay": 1e-2,
    "seed": 42,
    "logging_steps": 1,
    "save_steps": 200,
    "save_best": True,
    "run_validation": True,
    "validation_steps": 200,
    "val_metadata_file": f"../dataset/{dataset}/val.jsonl",
    "val_image_dir": f"../dataset/{dataset}/",
    "run_inference_validation": False,
    "mixed_precision": "bf16",
    "log_with_tensorboard": True,
}

print(hook_config)
print(lora_config)
print(trainer_config)

config = TrainerConfig(**trainer_config)
trainer = ContextPrunerDistiller(config=config, load_evaluators=False)
trainer.train()
