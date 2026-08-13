"""INT8 latency on a real SD3 block, at the shapes Shiva actually runs.

End-to-end TensorRT would mean exporting a pipeline whose pruning hook has Python
control flow and data-dependent gathers -- days of work for a number that is really a
property of the block. The block is where the FLOPs are, so measuring INT8 there and
multiplying by the call count (24 layers x denoising steps) gives the same answer
without the export.

Shapes matter: Shiva feeds each block k tokens where k comes from the LUT, so latency
is measured at the retained-token counts the deployed model uses, not at full length.

    python -m benchmark.quant_latency --weights /root/autodl-tmp/captures2/weights_L23.pt
"""

import argparse

import torch

from diffusers.models.attention import JointTransformerBlock

DIM, HEADS, HEAD_DIM, CTX = 1536, 24, 64, 333


def build(weights, device, dtype):
    # The last block runs context_pre_only=True and so has no context FFN; inferring it
    # from the keys keeps any layer's weights loadable.
    ctx_pre_only = weights is not None and "ff_context.net.2.weight" not in weights
    blk = JointTransformerBlock(
        dim=DIM,
        num_attention_heads=HEADS,
        attention_head_dim=HEAD_DIM,
        context_pre_only=ctx_pre_only,
    )
    if weights is not None:
        blk.load_state_dict(weights, strict=True)
    return blk.to(device=device, dtype=dtype).eval()


@torch.no_grad()
def median_ms(fn, repeats, warmup=10):
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    ev = [
        (torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True))
        for _ in range(repeats)
    ]
    for s, e in ev:
        s.record()
        fn()
        e.record()
    torch.cuda.synchronize()
    xs = sorted(s.elapsed_time(e) for s, e in ev)
    return xs[len(xs) // 2]


def main():
    p = argparse.ArgumentParser()
    p.add_argument(
        "--weights",
        default=None,
        help="a weights_L*.pt from the capture run; random init if omitted "
        "(latency does not depend on the values, only the shapes)",
    )
    p.add_argument(
        "--tokens",
        type=int,
        nargs="+",
        default=[4096, 3277, 2458, 2253],
        help="full, 80%%, 60%% and 55%% retention -- the operating points "
        "of Shiva-100/80/60 and layer 23's own average",
    )
    p.add_argument("--repeats", type=int, default=100)
    p.add_argument("--dtype", default="fp16", choices=["fp16", "bf16"])
    args = p.parse_args()

    from optimum.quanto import Calibration, freeze, qint8, quantize

    device = torch.device("cuda")
    dtype = {"fp16": torch.float16, "bf16": torch.bfloat16}[args.dtype]
    weights = (
        torch.load(args.weights, map_location="cpu", weights_only=False)
        if args.weights
        else None
    )
    if weights is not None:
        weights = {k: v.to(dtype) for k, v in weights.items()}

    ffn = ["ff.net.0.proj", "ff.net.2"]
    configs = {
        f"{args.dtype} (baseline)": None,
        "W8 only  (ffn)": ("w8", ffn),
        "W8A8     (ffn)": ("w8a8", ffn),
        "W8 only  (all)": ("w8", None),
        "W8A8     (all)": ("w8a8", None),
    }

    print(f"SD3 block, dim={DIM}, ctx={CTX}, {args.dtype}, median of {args.repeats}")
    print(f"\n{'config':<20}" + "".join(f"{n:>12}" for n in args.tokens))
    print("-" * (20 + 12 * len(args.tokens)))

    base = {}
    for cname, spec in configs.items():
        row = []
        for n in args.tokens:
            x = torch.randn(1, n, DIM, device=device, dtype=dtype)
            ctx = torch.randn(1, CTX, DIM, device=device, dtype=dtype)
            temb = torch.randn(1, DIM, device=device, dtype=dtype)

            blk = build(weights, device, dtype)
            if spec is not None:
                mode, inc = spec
                kw = {"weights": qint8}
                if mode == "w8a8":
                    kw["activations"] = qint8
                if inc is None:
                    quantize(blk, **kw)
                else:
                    quantize(blk, include=inc, **kw)
                if mode == "w8a8":
                    with Calibration(), torch.no_grad():
                        for _ in range(4):
                            blk(x, ctx, temb)
                freeze(blk)

            ms = median_ms(
                lambda block=blk, image=x, context=ctx, time_embedding=temb: block(
                    image, context, time_embedding
                ),
                args.repeats,
            )
            if spec is None:
                base[n] = ms
                row.append(f"{ms:>11.3f}")
            else:
                row.append(f"{base[n] / ms:>10.3f}x")
            del blk
            torch.cuda.empty_cache()
        print(f"{cname:<20}" + "".join(row))

    print("\nbaseline row is milliseconds; the rest are speedups over it.")


if __name__ == "__main__":
    main()
