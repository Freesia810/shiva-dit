"""Stage-1 (Router Warmup) ablation arms for SD3.5.

Every arm reuses the *identical* recipe as scripts/train/shiva_router.py with the
Group-12 (pairwise) router -- same LoRA config, same stage schedule, same steps,
same lr, same distillation weights -- and varies exactly one knob. `full` is the
shared baseline for both ablation groups, so all rows land in one comparable table.

  Group A -- context modulation h_ctx (Eq. 8)
    full             time + prompt + layer            3.99M
    no_time          prompt + layer                   2.81M
    no_prompt        time + layer                     2.41M
    no_layer         time + prompt                    3.94M
    layer_only       layer                            1.23M
    no_ctx           (none)                           1.18M
    no_ctx_matched   (none), d'=216                   3.99M  <- param-matched control

  Group B -- selection-probability surrogate pi_i (Eq. 6)
    gumbel_tau{05,1,2}   independent Gumbel-Sigmoid, DyDiT-style

Usage:
    python -m scripts.train.ablation_sd3 --arm no_ctx_matched
    python -m scripts.train.ablation_sd3 --list
"""

import argparse

from shiva_dit.training import DiffusionModelTrainer, LoraConfig, TrainerConfig
from shiva_dit.shiva_hook import ShivaDitBlockHook

HookType = ShivaDitBlockHook

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

# Group-12 / pairwise sharing -- the best row of Table 2.
BASE_HOOK_CONFIG = {
    "shiva_mode": "image",
    "image_ratio_mode": "adaptive",
    "fixed_image_ratio": 0.6,
    "num_router_groups": 12,
    "bottleneck_dim": 64,
    "layer_embed_dim": 64,
    "fuse_expert_ffn_mult": 2,
    "use_gater_lut": False,
    "init_Ts": 10.0,
    "init_Tk": 10.0,
}

# arm name -> overrides on BASE_HOOK_CONFIG
ARMS = {
    # --- Group A: context modulation ---
    "full": {},
    "no_time": {"router_context_signals": ("prompt", "layer")},
    "no_prompt": {"router_context_signals": ("time", "layer")},
    "no_layer": {"router_context_signals": ("time", "prompt")},
    "layer_only": {"router_context_signals": ("layer",)},
    "no_ctx": {"router_context_signals": ()},
    "no_ctx_matched": {"router_context_signals": (), "bottleneck_dim": 216},
    # --- Group B: pi_i surrogate ---
    # tau is swept so the baseline is not crippled by a single unlucky temperature.
    "gumbel_tau05": {"pi_mode": "gumbel", "gumbel_tau": 0.5},
    "gumbel_tau1": {"pi_mode": "gumbel", "gumbel_tau": 1.0},
    "gumbel_tau2": {"pi_mode": "gumbel", "gumbel_tau": 2.0},
}


def _auto_grad_ckpt() -> bool:
    """Enable gradient checkpointing on cards that need the activation headroom.

    Distillation keeps a full teacher DiT resident, so on <~34GB cards (RTX 4090
    24GB, RTX 5090 32GB) we trade ~1.3x step time for the memory to fit. On A100
    40GB+ we leave it off for speed. Adapts to whatever GPU the job lands on.
    """
    import torch

    if not torch.cuda.is_available():
        return False
    total = torch.cuda.get_device_properties(0).total_memory
    return total < 34 * 1024**3


def build_trainer_config(arm: str) -> TrainerConfig:
    if arm not in ARMS:
        raise SystemExit(f"unknown arm '{arm}'. choose from: {', '.join(ARMS)}")

    hook_config = {**BASE_HOOK_CONFIG, **ARMS[arm]}

    trainer_config = {
        "metadata_file": "../dataset/MJHQ-30K/train.jsonl",
        "image_dir": "../dataset/MJHQ-30K/",
        "model_type": "sd3",
        "model_path": None,
        # Data disk: the 30GB system disk cannot hold these checkpoints.
        "output_dir": f"/root/autodl-tmp/runs/checkpoints_ablation_{arm}",
        "tensorboard_log_dir": f"logs_ablation_{arm}",
        "training_mode": "lora",
        "lora_config": LoraConfig(**lora_config),
        "trainable_hook_config": (HookType, hook_config),
        "caption_dropout_rate": 0.1,
        "num_epochs": 2,
        "train_batch_size": 1,
        "gradient_accumulation_steps": 2,
        "enable_gradient_checkpointing": _auto_grad_ckpt(),
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
        "run_inference_validation": False,
        "mixed_precision": "bf16",
        "extra_config": {
            "target_ratio": 0.6,
            "stage_schedule": [
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
            ],
        },
        "log_with_tensorboard": True,
    }
    return TrainerConfig(**trainer_config), hook_config


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--arm", default="full")
    parser.add_argument("--list", action="store_true", help="list arms and exit")
    # The trainer pins the LAST stage's end to num_training_steps
    # (= rows // grad_accum * num_epochs), ignoring the 3500 it declares, so on the
    # full 28k-row split one arm runs ~7h instead of 3500 steps. Sizing the train
    # subset is what actually bounds the run: rows = steps * accum / num_epochs.
    parser.add_argument(
        "--train_rows",
        type=int,
        default=None,
        help="cap the train split; 3500 rows == the intended 3500 steps",
    )
    parser.add_argument(
        "--val_rows",
        type=int,
        default=None,
        help="cap the val split; the full 1980 rows cost ~11 min a call",
    )
    args = parser.parse_args()

    if args.list:
        for name, override in ARMS.items():
            print(f"  {name:<16} {override if override else '(baseline)'}")
        raise SystemExit(0)

    config, hook_config = build_trainer_config(args.arm)

    def _subset(src, rows, tag):
        import os

        dst = os.path.join(os.path.dirname(src) or ".", f"{tag}_sub{rows}.jsonl")
        if not os.path.exists(dst):
            with open(src) as f:
                picked = [next(f) for _ in range(rows)]
            with open(dst, "w") as f:
                f.writelines(picked)
        return dst

    if args.train_rows:
        config.metadata_file = _subset(config.metadata_file, args.train_rows, "train")
        print(f"train subset: {config.metadata_file} ({args.train_rows} rows)")
    if args.val_rows:
        config.val_metadata_file = _subset(
            config.val_metadata_file, args.val_rows, "val"
        )
        print(f"val subset: {config.val_metadata_file} ({args.val_rows} rows)")
    print(f"=== ablation arm: {args.arm} ===")
    print(hook_config)
    print(lora_config)
    print(config)

    trainer = DiffusionModelTrainer(config=config)
    trainer.train()
