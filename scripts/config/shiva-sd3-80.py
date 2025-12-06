from shiva_dit.training import DiffusionModelTrainer, LoraConfig, TrainerConfig
from shiva_dit.shiva_hook import ShivaDitBlockHook

HookType = ShivaDitBlockHook
# hook_config = {
#     # --- Mode Selection ---
#     "shiva_mode": "image",

#     # --- Image Configs ---
#     "image_ratio_mode": "adaptive",
#     "fixed_image_ratio": 0.8,
#     "num_router_groups": 24,
#     "bottleneck_dim": 512,
#     "ratio_bottleneck_dim": 32,
#     "layer_embed_dim": 256,
#     "fuse_expert_ffn_mult": 2,
#     "use_gater_lut": False,
    
#     # --- Common ---
#     "init_Ts": 10.0,
#     "init_Tk": 10.0,
# }

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

# trainer_config = {
#     "metadata_file": "../dataset/MJHQ-30K/train.jsonl",
#     "image_dir": "../dataset/MJHQ-30K/",
#     "model_type": "sd3",
#     "model_path": None,
#     "output_dir": "./checkpoints",
#     "training_mode": "lora",
#     "hook_only_steps": 0,
#     "lora_config": LoraConfig(**lora_config),
#     "trainable_hook_config": (HookType, hook_config),
#     "caption_dropout_rate": 0.1,
#     "num_epochs": 30,
#     "train_batch_size": 1,
#     "gradient_accumulation_steps": 16,
#     "optimizer": "AdamW",
#     "weight_decay": 1e-2,
#     "distillation_final_weight": 1.0,
#     "distillation_middle_weight": 0.0,
#     "seed": 42,
#     "logging_steps": 40,
#     "save_steps": 1750,
#     "save_best": True,
#     "run_validation": True,
#     "validation_steps": 875,
#     "val_metadata_file": "../dataset/MJHQ-30K/val.jsonl",
#     "val_image_dir": "../dataset/MJHQ-30K/",
#     "run_inference_validation": True,
#     "inference_steps": 1750,
#     "inference_metadata_file": "../dataset/MJHQ-30K/test.jsonl",
#     "inference_seed": 42,
#     "inference_num_inference_steps": 28,
#     "inference_output_dir": "inference_samples",
#     "mixed_precision": "bf16",
#     "extra_config": {
#         "target_ratio": 0.8,
#         # LR Order Mapping in Trainer: 
#         # [0:Base/Router, 1:LoRA, 2:Gate, 3:Fuser, 4:ImageRatio(Policy), 5:CtxRatio, 6:CtxPruner]
#         "stage_schedule": [
#             # ------------------------------------------------------------------
#             # Stage 0: Router Warmup (Random Ratio Phase)
#             # ------------------------------------------------------------------
#             {
#                 "steps": 7000, # 约 4 epochs
#                 "weights": {
#                     "ratio": [1.0, 50.0, 1.0], 
#                     "gate": 0.0,
#                     "distill": [5.0, 0.0]
#                 },
#                 # LR: [Base(Router), LoRA, Gate, Fuser, Policy, CtxRatio, CtxPruner]
#                 "lr": [1e-5, 1e-4, 0.0, 0.0, 0.0, 0.0, 0.0],
#                 "temp": [2.0, 2.0, 2.0, 2.0],
#             },

#             {
#                 "steps": 3500,
#                 "weights": {
#                     "ratio": [1.0, 50.0, 0.1], 
#                     "gate": 0.0,
#                     "distill": [5.0, 0.0]
#                 },
#                 # LR: [Base(Router), LoRA, Gate, Fuser, Policy, CtxRatio, CtxPruner]
#                 "lr": [1e-5, 1e-5, 0.0, 0.0, 1e-3, 0.0, 0.0],
#                 "temp": [2.0, 5.0, 2.0, 5.0],
#             },

#             {
#                 "steps": 7000,
#                 "weights": {
#                     "ratio": [1.0, 50.0, 1.0], 
#                     "gate": 10.0,
#                     "distill": [1.0, 0.0]
#                 },
#                 "lr": [1e-5, 1e-4, 1e-5, 1e-5, 1e-5, 0.0, 0.0],
#                 "temp": [5.0, 10.0, 5.0, 10.0],
#             },

#             {
#                 "steps": 35000,
#                 "weights": {
#                     "ratio": [1.0, 50.0, 1.0], 
#                     "gate": 0.01,
#                     "distill": [1.0, 0.0]
#                 },
#                 "lr": [1e-5, 1e-4, 1e-5, 1e-5, 1e-5, 0.0, 0.0],
#                 "temp": [10.0, 10.0, 10.0, 50.0],
#             },
#         ],
#     },
#     "log_with_tensorboard": True,
# }


hook_config = {
    # --- Mode Selection ---
    "shiva_mode": "image",

    # --- Image Configs ---
    "image_ratio_mode": "adaptive",
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
trainer_config = {
    "metadata_file": "../dataset/MJHQ-30K/train.jsonl",
    "image_dir": "../dataset/MJHQ-30K/",
    "model_type": "sd3",
    "model_path": None,
    "output_dir": "./checkpoints",
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
        "target_ratio": 0.8,
        # LR Order Mapping in Trainer: 
        # [0:Base/Router, 1:LoRA, 2:Gate, 3:Fuser, 4:ImageRatio(Policy), 5:CtxRatio, 6:CtxPruner]
        "stage_schedule": [
            # ------------------------------------------------------------------
            # Stage 0: Router Warmup 
            # ------------------------------------------------------------------
            {
                "steps": 7000, 
                "weights": {
                    "ratio": [0.0, 0.0, 0.0], 
                    "gate": 0.0,
                    "distill": [1.0, 0.0]
                },
                # LR: [Base(Router), LoRA, Gate, Fuser, Policy, CtxRatio, CtxPruner]
                "lr": [5e-4, 1e-5, 0.0, 0.0, 0.0, 0.0, 0.0],
                "temp": [5.0, 10.0, 10.0, 20.0],
            },

            # ------------------------------------------------------------------
            # Stage 1: Ratio Policy Training
            # ------------------------------------------------------------------
            {
                "steps": 17500, 
                "weights": {
                    "ratio": [10.0, 100.0, 0.0],
                    "gate": 0.0,
                    "distill": [20.0, 50.0],
                },
                "lr": [1e-5, 1e-5, 0.0, 0.0, 1e-5, 0.0, 0.0],
                "temp": [10.0, 10.0, 20.0, 100.0],
            },

            # # ------------------------------------------------------------------
            # # Stage 2: Gater Cold Start (GATER_STAGE=2)
            # # ------------------------------------------------------------------
            # {
            #     "steps": 7000,
            #     "weights": {
            #         "ratio": [5.0, 50.0, 0.0], 
            #         "gate": 0.0, 
            #         "distill": [1.0, 0.0]
            #     },
            #     "lr": [1e-5, 1e-4, 1e-4, 1e-4, 1e-5, 0.0, 0.0],
            #     "temp": [5.0, 10.0, 5.0, 10.0],
            # },

            # # ------------------------------------------------------------------
            # # Stage 3: Gater Sparsity (Fine-tuning)
            # # ------------------------------------------------------------------
            # {
            #     "steps": 35000,
            #     "weights": {
            #         "ratio": [5.0, 50.0, 0.0], 
            #         "gate": 0.05,
            #         "distill": [1.0, 0.0]
            #     },
            #     "lr": [1e-5, 1e-4, 1e-5, 1e-5, 1e-5, 0.0, 0.0],
            #     "temp": [10.0, 10.0, 10.0, 50.0],
            # },
        ],
    },
    "log_with_tensorboard": True,
}

print(hook_config)
print(lora_config)
print(trainer_config)

config = TrainerConfig(**trainer_config)
trainer = DiffusionModelTrainer(config=config)
trainer.train()
