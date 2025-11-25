import os
import torch
import json
from shiva_dit.training import DiffusionModelTrainer, LoraConfig, TrainerConfig
from shiva_dit.shiva_hook import ShivaDitBlockHook
from shiva_dit.modules.pruner import ShivaContextPruner, ShivaPrunerExpert
import torch.nn.utils.parametrize as parametrize
import torch.nn as nn

def prepare_single_sample_data(image_source, output_dir):
    os.makedirs(output_dir, exist_ok=True)
    
    import shutil
    target_img_path = os.path.join(output_dir, "image_001.png")
    shutil.copy(image_source, target_img_path)
    
    long_prompt = "A high quality photo of a cat. The cat is cute. There is no dog. The background is simple. Lighting is soft. 4k resolution."
    meta_path = os.path.join(output_dir, "meta.jsonl")
    with open(meta_path, "w") as f:
        json.dump({"file_name": "image_001.png", "text": long_prompt}, f)
        f.write('\n')
    
    return meta_path, output_dir

DATA_DIR = "./single_sample_data"
META_FILE, IMG_DIR = prepare_single_sample_data("pic/cat.png", DATA_DIR)

HookType = ShivaDitBlockHook
hook_config = {
    "num_router_groups": 4,
    "bottleneck_dim": 64,
    "layer_embed_dim": 64,
    "fuse_expert_ffn_mult": 2,
    "use_gater_lut": False,
    
    "context_pruning_ratio": 0.5,
    "proj_dim": 128,
    "init_Ts": 2.0,
    "init_Tk": 2.0,
}

lora_config = {
    "enable": False,
}

trainer_config = {
    "metadata_file": META_FILE,
    "image_dir": IMG_DIR,
    "model_type": "flux-dev",
    "output_dir": "./checkpoints/",
    "training_mode": "hook_only",
    "hook_only_steps": 100000,
    "lora_config": LoraConfig(**lora_config),
    "trainable_hook_config": (HookType, hook_config),
    "caption_dropout_rate": 0.0,
    
    "num_epochs": 200, 
    "train_batch_size": 1,
    "gradient_accumulation_steps": 1, 
    
    "learning_rate": 0.0,       # DiT 不训
    "lora_learning_rate": 0.0,  # LoRA 不训
    "policy_learning_rate": 0.0,
    "context_learning_rate": 1e-4,
    
    "optimizer": "AdamW",
    "lr_scheduler": "constant",
    "seed": 42,
    
    "logging_steps": 5,
    "save_steps": 50,
    "run_validation": False, # 单样本不用 val
    "run_inference_validation": True, # 跑推理看图
    "inference_steps": 50,
    "inference_metadata_file": META_FILE,
    "inference_output_dir": "inference_samples",
    
    "mixed_precision": "bf16",
    "extra_config": {
        "ratio_loss_weight": 0.0, 
        "gate_loss_weight": 0.0,
        "target_ratio": 1.0, 
    },
    
    "log_with_tensorboard": True,
    "wandb_run_name": "context-pruning-single-test"
}

config = TrainerConfig(**trainer_config)
trainer = DiffusionModelTrainer(config=config)

print("\n[Script] Manually freezing parameters for Context Pruning Experiment...")
trainer.model.requires_grad_(False)

trainable_params = []
for hook in trainer.pipe.transformer.modules(): # 遍历 hook 模块
    if isinstance(hook, ShivaContextPruner):
        # 解锁投影层
        hook.q_proj.requires_grad_(True)
        hook.k_proj.requires_grad_(True)
        trainable_params.extend(list(hook.q_proj.parameters()))
        trainable_params.extend(list(hook.k_proj.parameters()))
        # STE 参数通常不需要导数 (buffer)，但如果 Ts/Tk 是 Parameter 则需要
        if isinstance(hook.ste.Ts, torch.nn.Parameter):
            hook.ste.Ts.requires_grad_(True)
            trainable_params.append(hook.ste.Ts)

trainer.optimizer = torch.optim.AdamW(trainable_params, lr=1e-3, weight_decay=1e-4)

print(f"[Script] Optimized parameters: {len(trainable_params)} tensors.")
print(f"[Script] Example: {trainable_params[0].shape}")

# 开始训练
trainer.train()