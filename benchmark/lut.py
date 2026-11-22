"""Export the per-layer, per-step ratio policy of an adaptive Shiva checkpoint."""

import argparse
from collections import defaultdict
from pathlib import Path

import torch
from accelerate.utils import set_seed

from shiva_dit.hook import HookManager
from shiva_dit.shiva_hook import ShivaDitBlockHook
from shiva_dit.utils import process_quick_start
from shiva_dit.utils.inference_utils import load_pipeline_for_inference


def capture_lut(pipe, hooks, steps, height, width):
    if not hooks or any(not isinstance(hook, ShivaDitBlockHook) for hook in hooks):
        raise ValueError("LUT export requires a Shiva checkpoint")
    gater_policy = hooks[0].gater_policy
    ratio_policy = hooks[0].image_ratio_policy
    if gater_policy is None or ratio_policy is None or any(hook.use_gater_lut for hook in hooks):
        raise ValueError("LUT export requires an adaptive image policy without an active LUT")

    config = pipe.transformer.config
    total_layers = config.num_layers + getattr(config, "num_single_layers", 0)
    gater_values = defaultdict(list)
    ratio_values = defaultdict(list)

    def capture_gater(module, inputs, logits):
        layer = int(inputs[1].flatten()[0])
        gater_values[layer].append(int(logits[0].argmax()))

    def capture_ratio(module, inputs, logits):
        layer = int(inputs[1].flatten()[0])
        ratio_values[layer].append(float(torch.sigmoid(logits).mean()))

    with gater_policy.register_forward_hook(capture_gater), ratio_policy.register_forward_hook(capture_ratio):
        with torch.no_grad():
            process_quick_start(pipe, prompt="A high quality photo", num_inference_steps=steps,
                                height=height, width=width)

    gater = torch.zeros((total_layers, steps), dtype=torch.long)
    ratio = torch.ones((total_layers, steps), dtype=torch.float32)
    for hook in hooks:
        layer = hook.layer_idx
        if len(gater_values[layer]) != steps or len(ratio_values[layer]) != steps:
            raise RuntimeError(f"Layer {layer} did not record exactly {steps} policy decisions")
        gater[layer] = torch.tensor(gater_values[layer])
        ratio[layer] = torch.tensor(ratio_values[layer])
    if not torch.isfinite(ratio).all():
        raise RuntimeError("Ratio policy produced non-finite values")
    return {"gater": gater, "ratio": ratio}


def generate_lut(args):
    if not torch.cuda.is_available():
        raise RuntimeError("LUT export requires CUDA")
    dtype = {"bf16": torch.bfloat16, "fp16": torch.float16, "fp32": torch.float32}[args.dtype]
    set_seed(42)
    pipe = load_pipeline_for_inference(
        ckpt_path=args.ckpt_path,
        base_model_path=args.base_model_path,
        device="cuda",
        dtype=dtype,
        optimize_hooks=args.optimize_hooks,
        merge_lora=args.merge_lora,
    )
    lut = capture_lut(pipe, list(HookManager.get_all_hooks().values()),
                      args.steps, args.height, args.width)
    path = args.output_path or Path(f"gater_lut_{args.steps}steps.pt")
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(lut, path)
    print(f"Saved {tuple(lut['ratio'].shape)} LUT to {path}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ckpt_path", required=True)
    parser.add_argument("--base_model_path")
    parser.add_argument("--output_path", type=Path)
    parser.add_argument("--steps", type=int, default=20)
    parser.add_argument("--height", type=int, default=1024)
    parser.add_argument("--width", type=int, default=1024)
    parser.add_argument("--dtype", choices=("bf16", "fp16", "fp32"), default="bf16")
    parser.add_argument("--optimize_hooks", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--merge_lora", action=argparse.BooleanOptionalAction, default=True)
    args = parser.parse_args()
    if min(args.steps, args.height, args.width) < 1:
        parser.error("Steps, height, and width must be positive")
    generate_lut(args)


if __name__ == "__main__":
    main()
