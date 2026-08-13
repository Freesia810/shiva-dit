"""INT8 latency on a real SD3 block, quantised through SegQuant's own API.

torch's stock int8 path (`torch._int_mm`) measured *slower* than fp16 on this 4090, and
slower even on 4096^3 and 8192^3 square GEMMs, so the limit is the kernel rather than
SD3's shapes. SegQuant ships a tuned CUTLASS int8 GEMM, so the question is whether a
real kernel turns that around.

The block is quantised via `segquant.torch.quantization.quantize` with
`real_quant=True` and `DType.INT8` on both activations and weights -- the same entry
point the repo's own `test_linear.py` uses -- and timed against the fp16 block at the
token counts Shiva actually retains.

    CUTLASS_PATH=... python -m benchmark.segquant_block --segquant /path/to/segquant \
        --weights /root/autodl-tmp/captures2/weights_L17.pt
"""

import argparse
import os
import sys

import torch

from diffusers.models.attention import JointTransformerBlock

DIM, HEADS, HEAD_DIM, CTX = 1536, 24, 64, 333


def build(weights, device, dtype):
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
def median_ms(fn, repeats=50, warmup=10):
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
    p.add_argument("--segquant", default="/root/autodl-tmp/segquant")
    p.add_argument("--weights", default=None)
    p.add_argument("--tokens", type=int, nargs="+", default=[4096, 3277, 2458])
    p.add_argument("--repeats", type=int, default=30)
    p.add_argument("--calib_batches", type=int, default=4)
    args = p.parse_args()

    sys.path.insert(0, args.segquant)
    os.chdir(args.segquant)  # the extension loader resolves sources from cwd
    from segquant.torch.quantization import quantize
    from segquant.config import DType, Optimum, Calibrate

    device = torch.device("cuda")
    dtype = torch.float16
    weights = (
        torch.load(args.weights, map_location="cpu", weights_only=False)
        if args.weights
        else None
    )
    if weights is not None:
        weights = {k: v.to(dtype) for k, v in weights.items()}

    cfg_real = {
        "default": {
            "enable": True,
            # seglinear triggers torch.fx symbolic tracing (quantization.py:301), and
            # a JointTransformerBlock has data-dependent control flow FX cannot trace.
            # The SegLinear pattern is orthogonal to the int8 kernel we are timing.
            "seglinear": False,
            "search_patterns": [],
            "real_quant": True,
            "opt": {"type": Optimum.DEFAULT, "alpha": 0.5},
            "calib": {"type": Calibrate.AMAX},
            "input_quant": {"type": DType.INT8, "axis": None},
            "weight_quant": {"type": DType.INT8, "axis": None},
        },
    }

    print(f"{torch.cuda.get_device_name(0)}, torch {torch.__version__}")
    print(f"\n{'tokens':>8}{'fp16 ms':>11}{'segquant int8':>15}{'speedup':>10}")
    print("-" * 44)

    for n in args.tokens:
        x = torch.randn(1, n, DIM, device=device, dtype=dtype)
        ctx = torch.randn(1, CTX, DIM, device=device, dtype=dtype)
        temb = torch.randn(1, DIM, device=device, dtype=dtype)

        blk = build(weights, device, dtype)
        t16 = median_ms(
            lambda block=blk, image=x, context=ctx, time_embedding=temb: block(
                image, context, time_embedding
            ),
            args.repeats,
        )

        # SegQuant calibrates from a loader of forward inputs
        loader = [((x, ctx, temb), {})] * args.calib_batches
        try:
            qblk = quantize(blk, loader, cfg_real, verbose=False)
            t8 = median_ms(
                lambda block=qblk, image=x, context=ctx, time_embedding=temb: block(
                    image, context, time_embedding
                ),
                args.repeats,
            )
            print(f"{n:>8}{t16:>11.3f}{t8:>15.3f}{t16 / t8:>9.2f}x", flush=True)
            del qblk
        except Exception as e:
            print(f"{n:>8}{t16:>11.3f}{'failed':>15}   {str(e)[:60]}", flush=True)

        del blk, x, ctx, temb
        torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
