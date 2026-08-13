"""Capture real inputs to a real SD3 block during a real denoising run.

The estimator study needs two things the random-init version cannot provide:
a *trained* block, and *real* latents that actually carry spatial redundancy.
This dumps both into a single .pt per (layer, step) so the analysis script can
reconstruct the exact situation offline.

Each dump contains the block's own trained weights alongside the tensors that
entered it, so the analysis never needs the full pipeline in memory.

Run (on the GPU box, weights required):
    python -m test.toy.capture_block_inputs \
        --model /root/autodl-tmp/sd3.5-medium \
        --layers 1 11 23 --steps 2 14 26 \
        --out /root/autodl-tmp/captures
"""

import argparse
import os

import torch

from diffusers.pipelines import StableDiffusion3Pipeline

DEFAULT_PROMPT = (
    "a photograph of an old fisherman mending a net on a harbour wall at dawn, "
    "weathered face, detailed hands, soft morning light"
)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model", required=True, help="local SD3/SD3.5 diffusers dir")
    p.add_argument("--prompt", default=DEFAULT_PROMPT)
    p.add_argument(
        "--prompts-jsonl",
        default=None,
        help="draw prompts from a MJHQ-style jsonl instead of --prompt; "
        "activation calibration needs several distinct prompts, since a "
        "single one gives a range that does not generalise",
    )
    p.add_argument("--num-prompts", type=int, default=8)
    p.add_argument(
        "--weights-once",
        action="store_true",
        help="write the block weights to their own file instead of into "
        "every dump -- they are ~380MB and identical across captures",
    )
    p.add_argument("--layers", type=int, nargs="+", default=[1, 11, 23])
    p.add_argument(
        "--steps",
        type=int,
        nargs="+",
        default=[2, 14, 26],
        help="denoising step indices at which to capture",
    )
    p.add_argument("--num-inference-steps", type=int, default=28)
    p.add_argument("--height", type=int, default=1024)
    p.add_argument("--width", type=int, default=1024)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--out", required=True)
    p.add_argument(
        "--no-t5",
        action="store_true",
        help="skip the T5 encoder (saves ~9GB, CLIP conditioning only)",
    )
    p.add_argument(
        "--device",
        default="cuda"
        if torch.cuda.is_available()
        else ("mps" if torch.backends.mps.is_available() else "cpu"),
    )
    p.add_argument("--dtype", default="fp16", choices=["fp16", "bf16", "fp32"])
    args = p.parse_args()

    dtype = {"fp16": torch.float16, "bf16": torch.bfloat16, "fp32": torch.float32}[
        args.dtype
    ]
    os.makedirs(args.out, exist_ok=True)

    load_kwargs = dict(torch_dtype=dtype)
    if args.no_t5:
        load_kwargs.update(text_encoder_3=None, tokenizer_3=None)

    print(f"loading {args.model} ...")
    pipe = StableDiffusion3Pipeline.from_pretrained(args.model, **load_kwargs)
    pipe = pipe.to(args.device)

    blocks = pipe.transformer.transformer_blocks
    print(
        f"{len(blocks)} transformer blocks, capturing layers {args.layers} "
        f"at steps {args.steps}"
    )

    state = {"step": 0, "prompt_idx": 0, "prompt_text": args.prompt}
    wanted_steps = set(args.steps)
    wanted_layers = set(args.layers)
    saved = []

    def make_hook(layer_idx):
        def hook(module, hargs, hkwargs):
            step = state["step"]
            if step not in wanted_steps:
                return
            hs = hkwargs.get("hidden_states", hargs[0] if hargs else None)
            ehs = hkwargs.get(
                "encoder_hidden_states", hargs[1] if len(hargs) > 1 else None
            )
            temb = hkwargs.get("temb", hargs[2] if len(hargs) > 2 else None)
            if hs is None or temb is None:
                return

            tag = f"L{layer_idx:02d}_S{step:02d}_P{state['prompt_idx']:02d}"
            path = os.path.join(args.out, f"{tag}.pt")
            torch.save(
                {
                    "layer": layer_idx,
                    "step": step,
                    "hidden_states": hs.detach().float().cpu(),
                    "encoder_hidden_states": None
                    if ehs is None
                    else ehs.detach().float().cpu(),
                    "temb": temb.detach().float().cpu(),
                    "block_state_dict": None
                    if args.weights_once
                    else {
                        k: v.detach().float().cpu()
                        for k, v in module.state_dict().items()
                    },
                    "block_config": {
                        # the block's own width, not config.joint_attention_dim
                        # (that one is the text embedding dim, 4096)
                        "hidden_dim": hs.shape[-1],
                        "num_attention_heads": pipe.transformer.config.num_attention_heads,
                        "attention_head_dim": pipe.transformer.config.attention_head_dim,
                        "context_pre_only": getattr(module, "context_pre_only", False),
                    },
                    "prompt": state["prompt_text"],
                },
                path,
            )
            saved.append(path)
            print(
                f"  captured L{layer_idx} S{step}  "
                f"hidden={tuple(hs.shape)} ctx="
                f"{None if ehs is None else tuple(ehs.shape)}"
            )

        return hook

    handles = []
    for i in sorted(wanted_layers):
        if i < len(blocks):
            handles.append(
                blocks[i].register_forward_pre_hook(make_hook(i), with_kwargs=True)
            )

    def step_counter(pipe_, step_index, timestep, cb_kwargs):
        state["step"] = step_index + 1
        return cb_kwargs

    # Calibration ranges have to come from more than one prompt, so the default
    # single-prompt run is only a fallback.
    prompts = [args.prompt]
    if args.prompts_jsonl:
        import json

        prompts = []
        with open(args.prompts_jsonl) as f:
            for line in f:
                if len(prompts) >= args.num_prompts:
                    break
                rec = json.loads(line)
                txt = rec.get("text") or rec.get("prompt")
                if txt:
                    prompts.append(txt)
        print(f"{len(prompts)} prompts from {args.prompts_jsonl}")

    if args.weights_once:
        for i in sorted(wanted_layers):
            if i >= len(blocks):
                continue
            wp = os.path.join(args.out, f"weights_L{i:02d}.pt")
            torch.save(
                {
                    k: v.detach().float().cpu()
                    for k, v in blocks[i].state_dict().items()
                },
                wp,
            )
            print(f"  block weights -> {wp} ({os.path.getsize(wp) / 1e6:.0f} MB)")

    for pi, prompt_text in enumerate(prompts):
        state["prompt_idx"] = pi
        state["prompt_text"] = prompt_text
        state["step"] = 0
        print(f"\n[{pi + 1}/{len(prompts)}] {prompt_text[:70]}")
        generator = torch.Generator(device="cpu").manual_seed(args.seed + pi)
        with torch.no_grad():
            pipe(
                prompt=prompt_text,
                negative_prompt="",
                num_inference_steps=args.num_inference_steps,
                guidance_scale=7.0,
                height=args.height,
                width=args.width,
                generator=generator,
                callback_on_step_end=step_counter,
            )

    for h in handles:
        h.remove()

    print(f"\nsaved {len(saved)} captures to {args.out}")
    for s in saved:
        print(f"  {s}  ({os.path.getsize(s) / 1e6:.0f} MB)")


if __name__ == "__main__":
    main()
