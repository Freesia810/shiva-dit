"""How much does W8A8 disturb a real SD3 block, layer by layer?

Before spending GPU hours generating images, the cheap question is whether an INT8
block still reproduces its FP output. This runs a captured, *trained* SD3 block on the
real tensors that entered it and reports the relative output error under simulated
W8A8 -- forward only, no generation, runs on a laptop.

Quantisation is done with optimum-quanto rather than a hand-rolled fake quantiser:
selective by module, per-channel weight scales and calibrated activation scales come
for free, and there is no bespoke arithmetic to get wrong.

The layer *choice* is driven by our own LUT. The learned keep-ratio per layer is a
redundancy map -- a layer the policy prunes to 55% of its tokens is, by the model's own
estimate, carrying far less irreplaceable signal than one held at 91%. Quantising the
redundant end first is the standard practice; here it is data-driven rather than a guess.

    python -m benchmark.quant_sensitivity --capture results/captures/L01_S02.pt
"""

import argparse

import torch

from diffusers.models.attention import JointTransformerBlock


def build_block(blob, device, dtype):
    cfg = blob["block_config"]
    block = JointTransformerBlock(
        dim=cfg["hidden_dim"],
        num_attention_heads=cfg["num_attention_heads"],
        attention_head_dim=cfg["attention_head_dim"],
        context_pre_only=cfg.get("context_pre_only", False),
    )
    block.load_state_dict(blob["block_state_dict"], strict=True)
    return block.to(device=device, dtype=dtype).eval()


def rel_err(a, b):
    return ((a - b).norm() / b.norm().clamp(min=1e-12)).item()


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--capture", default="results/captures/L01_S02.pt")
    p.add_argument(
        "--device",
        default="cpu",
        help="cpu keeps the numerics deterministic; mps is faster but "
        "diverges slightly, which would be read as quantisation error",
    )
    p.add_argument("--calib_batches", type=int, default=8)
    args = p.parse_args()

    from optimum.quanto import Calibration, freeze, qint8, quantize

    device = torch.device(args.device)
    dtype = torch.float32
    blob = torch.load(args.capture, map_location="cpu", weights_only=False)

    x = blob["hidden_states"][:1].to(device, dtype)
    ctx = blob["encoder_hidden_states"][:1].to(device, dtype)
    temb = blob["temb"][:1].to(device, dtype)
    print(
        f"capture: layer {blob.get('layer')}, step {blob.get('step')}, "
        f"tokens {x.shape[1]}, dim {x.shape[2]}"
    )

    with torch.no_grad():
        ref_ctx, ref = build_block(blob, device, dtype)(x, ctx, temb)

    # Which submodules to quantise. FFN tolerates INT8 better than attention, so the
    # sweep walks outward from the safest set rather than quantising everything at once.
    groups = {
        "ffn only": ["ff.net.0.proj", "ff.net.2"],
        "ffn + attn out": ["ff.net.0.proj", "ff.net.2", "attn.to_out.0"],
        "ffn + all attn": [
            "ff.net.0.proj",
            "ff.net.2",
            "attn.to_out.0",
            "attn.to_q",
            "attn.to_k",
            "attn.to_v",
        ],
        "everything": None,  # every nn.Linear in the block
    }

    print(f"\n{'scope':<18}{'W8 only':>12}{'W8A8':>12}")
    print("-" * 42)
    for name, include in groups.items():
        for mode in ("w8", "w8a8"):
            block = build_block(blob, device, dtype)
            targets = include
            if targets is not None:
                # quanto matches on module names; keep only those that exist
                have = {
                    n
                    for n, m in block.named_modules()
                    if isinstance(m, torch.nn.Linear)
                }
                targets = [t for t in targets if t in have]
                if not targets:
                    print(
                        f"  (no matching modules for {name}; module names are "
                        f"{sorted(have)[:6]}...)"
                    )
                    break

            kw = {"weights": qint8}
            if mode == "w8a8":
                kw["activations"] = qint8
            quantize(block, include=targets, **kw) if targets else quantize(block, **kw)

            if mode == "w8a8":
                with Calibration(), torch.no_grad():
                    for _ in range(args.calib_batches):
                        block(x, ctx, temb)
            freeze(block)

            with torch.no_grad():
                q_ctx, q_out = block(x, ctx, temb)
            err = rel_err(q_out, ref)
            if mode == "w8":
                row = f"{name:<18}{err:>11.4%}"
            else:
                print(row + f"{err:>11.4%}")


if __name__ == "__main__":
    main()
