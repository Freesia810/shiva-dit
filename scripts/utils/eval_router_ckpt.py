"""Score an archived router checkpoint on the validation set -- no training.

Table 2's two columns come from validation, not a standalone evaluator:

    MSE = validation/distill_loss        trainer.py, F.mse_loss(student, teacher)
    Sim = validation/feature_cosine_sim  trainer.py, cosine over the feature dim

so this script rebuilds the trainer exactly as the checkpoint was written (hook and
LoRA config are read back out of the checkpoint itself), loads the weights, and calls
`trainer.validate()` without ever calling `trainer.train()`. Three uses:

  1. Reproduce a published row. `router_grouped_12.pt` should land on the paper's
     Group-12 numbers (MSE 0.894 / Sim 0.881). Because the text encoders now run in
     the training dtype rather than fp32 (a VRAM fix), reproducing that row is also
     what tells us the dtype change did not move the metric.

  2. h_ctx leave-one-out with no retraining. The three context terms of Eq. 8 enter
     the router as separate bias-free projections summed before a LayerNorm, so
     dropping one at eval time is well defined on a trained checkpoint. This measures
     how much the learned policy *relies* on each signal; it is not the same claim as
     retraining without that signal, but it carries no capacity confound at all --
     every arm is the identical network with identical parameters.

  3. Provide the residual-STE reference point for a Gumbel arm, so only the Gumbel
     run has to be trained.

    python -m scripts.utils.eval_router_ckpt --ckpt /path/router_grouped_12.pt
    python -m scripts.utils.eval_router_ckpt --ckpt ... --arms full,no_time,no_ctx
"""

import argparse
import json
import random

import numpy as np
import torch

from shiva_dit.training import DiffusionModelTrainer, LoraConfig, TrainerConfig
from shiva_dit.shiva_hook import ShivaDitBlockHook

# name -> which terms of h_ctx (Eq. 8) survive. Mirrors scripts/train/ablation_sd3.py
# so a no-training row and a trained row can sit in the same table.
ARMS = {
    "full": ("time", "prompt", "layer"),
    "no_time": ("prompt", "layer"),
    "no_prompt": ("time", "layer"),
    "no_layer": ("time", "prompt"),
    "layer_only": ("layer",),
    "no_ctx": (),
}


CONST_ARMS = {
    "const_time": ("time",),
    "const_prompt": ("prompt",),
    "const_layer": ("layer",),
    "const_all": ("time", "prompt", "layer"),
}

_SIGNAL_KW = {"time": "temb", "prompt": "pooled_encoder_states", "layer": "layer_emb"}


def freeze_context_signals(model, signals):
    """Hold a context input at a fixed value instead of deleting its term.

    Deleting a term is not a clean information ablation: the three terms of h_ctx are
    summed and then LayerNormed, so removing one lets the norm rescale the survivors --
    the context keeps its magnitude and only changes direction. Feeding a *constant*
    instead keeps the term, its projection and the norm's input structure intact, so
    the only thing taken away is the signal's information content.

    The constant is the first value the arm sees, which is an in-distribution sample;
    since every arm replays identical seeded batches, all arms freeze to the same value.
    """
    n = 0
    for module in model.modules():
        if not (hasattr(module, "context_signals") and hasattr(module, "use_context")):
            continue
        if not hasattr(module, "_orig_forward"):
            module._orig_forward = module.forward
        module._frozen = {}

        def make_forward(mod):
            def forward(hidden_states, pooled_encoder_states, temb, layer_emb):
                kw = {
                    "pooled_encoder_states": pooled_encoder_states,
                    "temb": temb,
                    "layer_emb": layer_emb,
                }
                for sig in signals:
                    key = _SIGNAL_KW[sig]
                    v = kw[key]
                    if v is None:
                        continue
                    if sig not in mod._frozen:
                        mod._frozen[sig] = v[:1].detach().clone()
                    kw[key] = mod._frozen[sig].expand_as(v)
                return mod._orig_forward(hidden_states=hidden_states, **kw)

            return forward

        module.forward = make_forward(module)
        n += 1
    return n


def restore_forwards(model):
    for module in model.modules():
        if hasattr(module, "_orig_forward"):
            module.forward = module._orig_forward
            module._frozen = {}


def set_context_signals(model, signals):
    """Point every router at `signals`, returning how many were touched.

    `use_context` has to move with `context_signals`: the forward pass feeds the
    summed context through `context_norm`, so leaving the flag on with an empty
    signal set would hand a None to the LayerNorm.
    """
    n = 0
    for module in model.modules():
        if hasattr(module, "context_signals") and hasattr(module, "use_context"):
            module.context_signals = tuple(signals)
            module.use_context = len(signals) > 0
            n += 1
    return n


def build_config(ckpt_path, args):
    """Rebuild the trainer config the checkpoint was written under."""
    meta = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    hook_info = meta.get("hook_info") or {}
    hook_config = dict(hook_info.get("config") or {})
    lora_info = dict(meta.get("lora_info") or {})
    lora_info.pop("enable", None)

    trainer_config = {
        # Never trained here, but the trainer builds a train dataloader on init;
        # pointing it at the validation split keeps the train split optional.
        "metadata_file": args.val_metadata,
        "image_dir": args.val_image_dir,
        "model_type": meta.get("model_type", "sd3"),
        "model_path": args.model_path,
        "output_dir": "./_eval_router_tmp",
        "training_mode": meta.get("training_mode", "lora"),
        "lora_config": LoraConfig(enable=True, **lora_info)
        if lora_info
        else LoraConfig(),
        "trainable_hook_config": (ShivaDitBlockHook, hook_config),
        "compile_model": False,
        "train_batch_size": 1,
        "val_batch_size": args.batch_size,
        # MSE/Sim *are* the distillation terms, so the teacher has to be resident.
        "distillation_final_weight": 1.0,
        "distillation_middle_weight": 0.0,
        "run_validation": True,
        "val_metadata_file": args.val_metadata,
        "val_image_dir": args.val_image_dir,
        "run_inference_validation": False,
        "log_with_tensorboard": False,
        "mixed_precision": args.mixed_precision,
        "seed": args.seed,
        "extra_config": {"target_ratio": args.target_ratio},
    }
    return TrainerConfig(**trainer_config), hook_config, meta.get("global_step")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt", required=True, help="archived router checkpoint (.pt)")
    p.add_argument(
        "--arms",
        default="full",
        help=f"comma-separated, from: {', '.join(ARMS)}; or 'all'",
    )
    p.add_argument("--val_metadata", default="../dataset/MJHQ-30K/val.jsonl")
    p.add_argument("--val_image_dir", default="../dataset/MJHQ-30K/")
    p.add_argument("--model_path", default=None)
    p.add_argument("--batch_size", type=int, default=1)
    p.add_argument("--mixed_precision", default="bf16", choices=["no", "fp16", "bf16"])
    p.add_argument(
        "--te_dtype",
        default=None,
        choices=["fp32", "bf16", "fp16"],
        help="override text-encoder dtype; fp32 reproduces the pre-fix "
        "setting but needs ~19GB extra VRAM for T5-XXL",
    )
    p.add_argument("--target_ratio", type=float, default=0.6)
    p.add_argument(
        "--val_limit",
        type=int,
        default=None,
        help="use only the first N validation batches -- for a smoke run; "
        "reproducing a published row needs the full split",
    )
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--out", default=None, help="write results as json")
    args = p.parse_args()

    all_arms = {**ARMS, **CONST_ARMS}
    if args.arms == "all":
        arms = list(ARMS)
    elif args.arms == "const":
        arms = ["full"] + list(CONST_ARMS)
    else:
        arms = [a.strip() for a in args.arms.split(",")]
    unknown = [a for a in arms if a not in all_arms]
    if unknown:
        raise SystemExit(
            f"unknown arm(s) {unknown}; choose from: {', '.join(all_arms)}"
        )

    config, hook_config, step = build_config(args.ckpt, args)
    print(f"checkpoint : {args.ckpt}")
    print(f"trained to : step {step}")
    print(f"hook config: {hook_config}")

    trainer = DiffusionModelTrainer(config=config)
    trainer.load_from_checkpoint(args.ckpt, model_only=True)

    # `global_step` is created by train(), which we never call; the forward path reads
    # it for its debug-interval bookkeeping, so validation dies on every batch without it.
    trainer.global_step = 0

    if args.te_dtype:
        dt = {"fp32": torch.float32, "bf16": torch.bfloat16, "fp16": torch.float16}
        for te in trainer.text_encoders:
            te.to(dtype=dt[args.te_dtype])
        print(f"text encoders cast to {args.te_dtype}")

    if args.val_limit:
        # validate() only iterates the loader and asks for its len(), so a
        # materialised list of the first N batches stands in for it.
        from itertools import islice

        trainer.val_dataloader = list(islice(trainer.val_dataloader, args.val_limit))
        print(
            f"smoke run: validation truncated to {len(trainer.val_dataloader)} batches"
        )

    # validate() ends with a policy visualisation meant for training runs; it needs a
    # live writer and contributes nothing here.
    trainer._log_policy_visualization = lambda *a, **k: None

    # validate() reports through the logger rather than returning MSE/Sim, so capture
    # what it logs instead of duplicating the loss code.
    captured = {}
    original_log = trainer.logger.log_scalars

    def capture(scalars, step=None):
        captured.update(scalars)
        return original_log(scalars, step=step)

    trainer.logger.log_scalars = capture

    results = {}
    model = trainer.accelerator.unwrap_model(trainer.model)
    for arm in arms:
        # Validation draws timesteps stochastically, and the arms run sequentially in
        # one process, so without reseeding each arm would be scored on a different
        # set of timesteps. Reseeding makes the comparison paired: every arm sees the
        # identical batches and identical noise, and the only difference is h_ctx.
        random.seed(args.seed)
        np.random.seed(args.seed)
        torch.manual_seed(args.seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(args.seed)

        # every arm starts from the unmodified router
        restore_forwards(model)
        set_context_signals(model, ARMS["full"])

        if arm in CONST_ARMS:
            frozen = CONST_ARMS[arm]
            signals = ARMS["full"]
            touched = freeze_context_signals(model, frozen)
            desc = "const:" + "+".join(frozen)
        else:
            signals = ARMS[arm]
            touched = set_context_signals(model, signals)
            desc = "+".join(signals) or "(none)"
        print(f"\n=== arm '{arm}'  {desc}  routers patched={touched} ===", flush=True)
        captured.clear()
        trainer.validate(0)
        # Table 2's "MSE" column is validation/main_loss, not the distillation term:
        # on the archived Group-12 checkpoint main_loss reproduces the published 0.894
        # (0.8963) while distill_loss lands at 0.6239. Sim is the feature cosine
        # (0.8793 vs a published 0.881). Both are kept so the mapping stays checkable.
        results[arm] = {
            "signals": desc,
            "MSE": captured.get("validation/main_loss"),
            "Sim": captured.get("validation/feature_cosine_sim"),
            "distill_loss": captured.get("validation/distill_loss"),
        }

    print(f"\n{'=' * 70}")
    print(f"{'arm':<12}{'signals':<22}{'MSE':>10}{'Sim':>10}{'distill':>12}")
    print(f"{'=' * 70}")
    for arm, r in results.items():
        sig = r["signals"]

        def fmt(v):
            return f"{v:.4f}" if v is not None else "n/a"

        print(
            f"{arm:<12}{sig:<22}{fmt(r['MSE']):>10}{fmt(r['Sim']):>10}"
            f"{fmt(r['distill_loss']):>12}"
        )
    print(
        "MSE = validation/main_loss (the column Table 2 reports); "
        "Group-12 published 0.894 / 0.881"
    )

    if args.out:
        payload = {
            "checkpoint": args.ckpt,
            "global_step": step,
            "hook_config": hook_config,
            "arms": results,
        }
        with open(args.out, "w") as f:
            json.dump(payload, f, indent=2)
        print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()
