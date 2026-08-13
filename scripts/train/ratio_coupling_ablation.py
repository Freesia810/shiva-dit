"""Additive vs FiLM coupling in the Ratio Policy (Eq. 10) -- short, matched runs.

Reviewer V82a: *"The policy in Section 3.4 assumes the time trend is the same across
layers. Please show data that an additive policy is not much worse than a
multiplicative one."* Section 3.4 argues the point qualitatively (citing FiLM) but
never measures it, so this trains the two couplings against each other:

    additive (ours)  logit(r) = Phi_time(e_t) + Phi_layer(e_l) + b_anchor
    film             logit(r) = (1 + gamma(t)) * Phi_layer(e_l) + beta(t) + b_anchor

Everything else is held fixed. Both arms start from the *same* archived router
checkpoint, so the router is identical and only the ratio policy differs, and both
skip straight to the two stages that actually train the policy (time, then layer).
Capacity is matched: FiLM only widens the last time_mlp layer from 1 to 2 outputs.

The comparison is the validation loss the two reach under an identical budget --
if additive lands on the same loss, the simplification costs nothing.

    python -m scripts.train.ratio_coupling_ablation --mode additive --steps 800
    python -m scripts.train.ratio_coupling_ablation --mode film     --steps 800
"""

import argparse

import torch

from shiva_dit.training import DiffusionModelTrainer, LoraConfig, TrainerConfig
from shiva_dit.shiva_hook import ShivaDitBlockHook


def _auto_grad_ckpt() -> bool:
    if not torch.cuda.is_available():
        return False
    return torch.cuda.get_device_properties(0).total_memory < 34 * 1024**3


def build_config(args):
    meta = torch.load(args.init_ckpt, map_location="cpu", weights_only=False)
    hook_config = dict((meta.get("hook_info") or {}).get("config") or {})
    hook_config["policy_mode"] = args.mode
    lora_info = dict(meta.get("lora_info") or {})
    lora_info.pop("enable", None)

    stage_lr = {"lora": 5e-5, "router": 1e-6, "image_ratio_policy": 1e-4}
    ratio_w = {"ratio": [0.0, 0.0, 1.0], "gate": 0.0, "distill": [0.1, 0.0]}

    trainer_config = {
        "metadata_file": args.train_metadata,
        "image_dir": args.image_dir,
        "model_type": meta.get("model_type", "sd3"),
        "model_path": None,
        # Data disk, not the 30GB system disk: checkpoints here run to gigabytes.
        "output_dir": f"{args.out_root}/ckpt_coupling_{args.mode}",
        "tensorboard_log_dir": f"logs_coupling_{args.mode}",
        "training_mode": meta.get("training_mode", "lora"),
        "lora_config": LoraConfig(enable=True, **lora_info)
        if lora_info
        else LoraConfig(),
        "trainable_hook_config": (ShivaDitBlockHook, hook_config),
        "compile_model": False,
        "caption_dropout_rate": 0.1,
        # The trainer forces the LAST stage to end at num_training_steps
        # (= samples // grad_accum * num_epochs), ignoring its declared `steps`. So the
        # run length is set by how many rows the train metadata has: with accum 2 and
        # one epoch, `2 * (1 + 2*steps)` rows makes the layer stage close exactly on
        # schedule instead of running forever. --make_subset does that arithmetic.
        "num_epochs": 1,
        "train_batch_size": 1,
        "gradient_accumulation_steps": 2,
        "enable_gradient_checkpointing": _auto_grad_ckpt(),
        "optimizer": "AdamW",
        "weight_decay": 1e-2,
        "distillation_final_weight": 1.0,
        "distillation_middle_weight": 0.0,
        "seed": args.seed,
        "logging_steps": 25,
        "save_steps": 10**9,  # the policy curve is the artefact, not a checkpoint
        "save_best": False,
        "run_validation": True,
        "validation_steps": args.val_every,
        "val_metadata_file": args.val_metadata,
        "val_image_dir": args.image_dir,
        "run_inference_validation": False,
        "mixed_precision": "bf16",
        "log_with_tensorboard": True,
        "extra_config": {
            "target_ratio": args.target_ratio,
            # Stage 0 is the router warmup, already done in the loaded checkpoint; it is
            # kept with a single step purely so the time/layer stages land on indices 1
            # and 2, which is what ShivaRatioPolicy keys its stage behaviour off.
            "stage_schedule": [
                {
                    "steps": 1,
                    "weights": {
                        "ratio": [0.0, 0.0, 0.0],
                        "gate": 0.0,
                        "distill": [0.1, 1.0],
                    },
                    "lr": {"router": 0.0},
                    "warmup": 0,
                    "temp": [5.0, 15.0, 10.0, 20.0],
                },
                {  # time stage
                    "steps": args.steps,
                    "weights": ratio_w,
                    "lr": stage_lr,
                    "warmup": max(1, args.steps // 10),
                    "temp": [15.0, 15.0, 20.0, 25.0],
                },
                {  # layer stage
                    "steps": args.steps,
                    "weights": ratio_w,
                    "lr": stage_lr,
                    "temp": [15.0, 15.0, 25.0, 30.0],
                },
            ],
        },
    }
    return TrainerConfig(**trainer_config), hook_config


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=["additive", "film"], required=True)
    parser.add_argument(
        "--init_ckpt",
        default="/root/autodl-tmp/assets/models/router_grouped_12.pt",
        help="router checkpoint both arms start from",
    )
    parser.add_argument(
        "--steps",
        type=int,
        default=800,
        help="steps per ratio stage (time and layer each get this many)",
    )
    parser.add_argument("--val_every", type=int, default=200)
    parser.add_argument("--target_ratio", type=float, default=0.6)
    parser.add_argument("--train_metadata", default="../dataset/MJHQ-30K/train.jsonl")
    parser.add_argument("--val_metadata", default="../dataset/MJHQ-30K/val.jsonl")
    parser.add_argument("--image_dir", default="../dataset/MJHQ-30K/")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--out_root",
        default="/root/autodl-tmp/runs",
        help="data disk; the system disk is only 30GB",
    )
    parser.add_argument(
        "--make_subset",
        action="store_true",
        help="write a train-metadata subset sized so the schedule ends "
        "on time, and use it for this run",
    )
    parser.add_argument(
        "--val_rows",
        type=int,
        default=300,
        help="validation rows to score; the full 1980-row split costs "
        "~11 min per call, which would dwarf a short run. Both "
        "arms use the same subset, so the curves stay comparable.",
    )
    args = parser.parse_args()

    def _subset(src, rows, tag):
        import os

        dst = os.path.join(os.path.dirname(src) or ".", f"{tag}_sub{rows}.jsonl")
        with open(src) as f:
            picked = [next(f) for _ in range(rows)]
        with open(dst, "w") as f:
            f.writelines(picked)
        return dst

    if args.make_subset:
        need_rows = 2 * (1 + 2 * args.steps)  # accum 2, one epoch, 1+2*steps steps
        args.train_metadata = _subset(args.train_metadata, need_rows, "train")
        print(
            f"train subset: {args.train_metadata} "
            f"({need_rows} rows -> {1 + 2 * args.steps} steps)"
        )
    if args.val_rows:
        args.val_metadata = _subset(args.val_metadata, args.val_rows, "val")
        print(f"val subset: {args.val_metadata} ({args.val_rows} rows)")

    config, hook_config = build_config(args)
    print(f"=== ratio-policy coupling: {args.mode} ===")
    print(f"init from : {args.init_ckpt}")
    print(f"schedule  : time {args.steps} steps -> layer {args.steps} steps")

    trainer = DiffusionModelTrainer(config=config)

    # Load the router (and LoRA) but NOT the ratio policy. Two reasons: FiLM widens
    # time_mlp's last layer to 2 outputs, so the archived (1, 32) weights would not even
    # load; and more importantly, letting the additive arm inherit an already-trained
    # policy while FiLM starts cold would not be a comparison of couplings. Both arms
    # therefore begin from the identical router with a freshly initialised policy.
    state = torch.load(args.init_ckpt, map_location="cpu", weights_only=False)
    state = state.get("model", state)
    dropped = [k for k in state if "ratio_policy" in k]
    for k in dropped:
        state.pop(k)
    missing, unexpected = trainer.accelerator.unwrap_model(
        trainer.model
    ).load_state_dict(state, strict=False)
    print(
        f"loaded router weights; dropped {len(dropped)} ratio-policy tensors "
        f"(fresh policy in both arms); {len(unexpected)} unexpected keys"
    )

    trainer.train()

    # Does the model actually *use* the layer-specific temporal scaling FiLM offers?
    # gamma and beta are the two output rows of time_mlp's last layer, so comparing
    # their weight norms says whether the interaction term survived training: gamma
    # collapsing toward zero means the run rediscovered the additive assumption
    # (identical temporal trend at every depth) when free not to. This is evidence
    # about the assumption itself, and it comes from the FiLM arm, not from ours.
    policy = None
    for module in trainer.accelerator.unwrap_model(trainer.model).modules():
        if hasattr(module, "policy_mode") and hasattr(module, "time_mlp"):
            policy = module
            break
    if policy is not None:
        head = policy.time_mlp[-1]
        with torch.no_grad():
            if policy.policy_mode == "film":
                g_w, b_w = head.weight[0], head.weight[1]
                g_b, b_b = head.bias[0].item(), head.bias[1].item()
                print(f"[coupling] gamma head |w|={g_w.norm().item():.5f} b={g_b:+.5f}")
                print(f"[coupling] beta  head |w|={b_w.norm().item():.5f} b={b_b:+.5f}")
                ratio = g_w.norm().item() / max(b_w.norm().item(), 1e-8)
                print(
                    f"[coupling] |gamma|/|beta| = {ratio:.4f}  "
                    f"(-> 0 means the layer-specific scaling was not used)"
                )
            else:
                print(f"[coupling] time head |w|={head.weight.norm().item():.5f}")
        torch.save(
            {
                "policy_mode": policy.policy_mode,
                "state_dict": {k: v.cpu() for k, v in policy.state_dict().items()},
            },
            f"/root/policy_{args.mode}.pt",
        )
        print(f"[coupling] saved /root/policy_{args.mode}.pt")
