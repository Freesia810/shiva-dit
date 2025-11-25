import os
from shiva_dit.training import DiffusionModelTrainer, LoraConfig, TrainerConfig
from shiva_dit.shiva_hook import ShivaDitBlockHook

DATA_DIR = "./single_sample_data"
META_FILE = os.path.join(DATA_DIR, "meta.jsonl")

HookType = ShivaDitBlockHook
hook_config = {
    "context_pruning_ratio": 0.5,  
    "proj_dim": 128,
    "init_Ts": 2.0,
    "init_Tk": 2.0,
    
    "num_router_groups": 1,
    "use_gater_lut": False,
}

lora_config = { "enable": False }

trainer_config = {
    "metadata_file": META_FILE,
    "image_dir": DATA_DIR,
    "model_type": "flux-dev",
    "output_dir": "./checkpoints/",
    
    "training_mode": "hook_only",
    "hook_only_steps": 1000000, 
    
    "lora_config": LoraConfig(**lora_config),
    "trainable_hook_config": (HookType, hook_config),
    
    "learning_rate": 0.0,
    "policy_learning_rate": 0.0,
    "context_learning_rate": 1e-4,
    
    "optimizer": "AdamW",
    "lr_scheduler": "constant",
    "seed": 42,
    
    "num_epochs": 200,
    "train_batch_size": 1,
    "gradient_accumulation_steps": 1,
    "mixed_precision": "bf16",
    
    "logging_steps": 5,
    "save_steps": 50,
    "run_validation": False,
    "run_inference_validation": True,
    "inference_steps": 50,
    "inference_metadata_file": META_FILE,
    
    "log_with_tensorboard": True,
    "wandb_run_name": "context-pruning-clean",

    "extra_config": {
        "target_ratio": 0.8,
    }
}

print("[Script] Starting Context Pruning Training...")
config = TrainerConfig(**trainer_config)
trainer = DiffusionModelTrainer(config=config)
trainer.train()