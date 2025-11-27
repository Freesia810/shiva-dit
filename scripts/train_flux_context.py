import os
from shiva_dit.training import DiffusionModelTrainer, LoraConfig, TrainerConfig
from shiva_dit.shiva_hook import ShivaDitBlockHook

DATA_DIR = "test/training/dataset"
META_FILE = os.path.join(DATA_DIR, "meta.jsonl")

HookType = ShivaDitBlockHook
hook_config = {
    # --- Mode Selection ---
    "shiva_mode": "context",
    
    # --- Context Configs ---
    "context_ratio_mode": "fixed",
    "fixed_context_ratio": 0.1,  
    "context_proj_dim": 128,
    
    # --- Common ---
    "init_Ts": 2.0,
    "init_Tk": 2.0,

    "use_gater_lut": False,
}

lora_config = { "enable": False }

trainer_config = {
    "metadata_file": META_FILE,
    "image_dir": DATA_DIR,
    "model_type": "flux-dev",
    "output_dir": "./checkpoints/",
    "training_mode": "hook_only",
    "hook_only_steps": 10000,
    "caption_dropout_rate": 0.0,
    "lora_config": LoraConfig(**lora_config),
    "trainable_hook_config": (HookType, hook_config),
    "optimizer": "AdamW",
    "seed": 42,
    "num_epochs": 500,
    "train_batch_size": 1,
    "gradient_accumulation_steps": 1,
    "mixed_precision": "bf16",
    "logging_steps": 5,
    "save_steps": 1000,
    "run_validation": False,
    "run_inference_validation": True,
    "inference_steps": 50,
    "inference_metadata_file": META_FILE,
    "inference_num_inference_steps": 20,
    "distillation_final_weight": 2.0,
    "log_with_tensorboard": True,
    "wandb_run_name": "context-pruning-fixed",
    "extra_config": {
        "target_ratio": 1.0,
        # LR Order: [Base, LoRA, Gate, Fuser, ImageRatio, ContextRatio, ContextPruner]
        "stage_schedule": [
            {
                "steps": -1,
                "weights": {
                    "ratio": [0.0, 0.0, 0.0],
                    "gate": 0.0,
                    "distill": [2.0, 0.0],
                    "aux": 0.1,
                },
                # Only optimize ContextPruner (last item)
                "lr": [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1e-4],
                "temp": [2.0, 2.0, 2.0, 2.0],
            },
        ],
    },
}

print("[Script] Starting Context Pruning Training (Fixed Ratio)...")
config = TrainerConfig(**trainer_config)
trainer = DiffusionModelTrainer(config=config)
trainer.train()
