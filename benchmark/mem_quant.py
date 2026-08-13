"""Peak VRAM across Vanilla, Shiva, and Shiva+INT8.

Token pruning and weight quantisation save memory in different places -- pruning shrinks
the activations that flow through each block, quantisation shrinks the weights that sit
resident -- so the two are worth reporting side by side rather than as one number.

Weights and peak activation are separated: `torch.cuda.memory_allocated()` right after
load is the resident model, and `max_memory_allocated()` over a forward adds what the
activations cost on top.

    python -m benchmark.mem_quant --segquant /root/autodl-tmp/segquant
"""

import argparse
import gc
import os
import sys

import torch


def snap():
    torch.cuda.synchronize()
    return torch.cuda.memory_allocated() / 2**30


@torch.no_grad()
def median_ms(fn, repeats=20, warmup=5):
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
    p.add_argument("--model", default="/root/stable-diffusion-3-medium-diffusers")
    p.add_argument("--ckpt", default="/root/autodl-tmp/assets/models/shiva-sd3-80.pt")
    p.add_argument("--lut", default="/root/autodl-tmp/assets/lut/lut-80.pt")
    p.add_argument("--res", type=int, default=1024)
    p.add_argument("--k_align", type=int, default=16)
    args = p.parse_args()

    repo = os.getcwd()
    device = torch.device("cuda")
    rows = []

    if args.k_align > 1:
        # CUTLASS int8 needs a 16-aligned M; the LUT's k is not, and an unaligned one
        # faults inside the kernel instead of raising.
        import shiva_dit.shiva_hook as _sh

        _sh.K_ALIGN = args.k_align
        print(f"k rounded up to a multiple of {args.k_align}")

    def measure(label, build_fn):
        gc.collect()
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
        tf, a, kw = build_fn()
        weights_gb = snap()
        torch.cuda.reset_peak_memory_stats()
        with torch.no_grad():
            for _ in range(3):
                tf(*a, **kw, return_dict=False)
        peak_gb = torch.cuda.max_memory_allocated() / 2**30
        ms = median_ms(
            lambda transformer=tf, args_=a, kwargs_=kw: transformer(
                *args_, **kwargs_, return_dict=False
            )
        )
        rows.append((label, weights_gb, peak_gb, ms))
        base = rows[0][3] if rows else ms
        print(
            f"{label:<20}{weights_gb:>10.2f}{peak_gb:>12.2f}"
            f"{peak_gb - weights_gb:>12.2f}{ms:>11.2f}{base / ms:>9.2f}x",
            flush=True,
        )
        del tf, a, kw
        gc.collect()
        torch.cuda.empty_cache()

    def load_pipe(ckpt=None, lut=None):
        from shiva_dit.utils.inference_utils import load_pipeline_for_inference

        pipe = load_pipeline_for_inference(
            ckpt_path=ckpt,
            base_model_path=args.model,
            model_type="sd3",
            device=device,
            dtype=torch.float16,
            gater_lut_path=lut,
            optimize_hooks=True,
            merge_lora=True,
        )
        tf = pipe.transformer
        # mask capture is a training/vis aid; it copies indices to CPU every call and
        # is what crashes once SegQuant replaces the Linears, so it is off for a
        # latency/memory measurement that never looks at the masks.
        for blk in getattr(tf, "transformer_blocks", []):
            hk = getattr(blk, "_hook", None)
            if hk is not None and hasattr(hk, "capture_mask_for_viz"):
                hk.capture_mask_for_viz = False
        grabbed = {}

        def grab(_m, aa, kk):
            if not grabbed:
                grabbed["a"], grabbed["kw"] = aa, kk
            return None

        h = tf.register_forward_pre_hook(grab, with_kwargs=True)
        with torch.no_grad():
            pipe(
                prompt="a cat",
                num_inference_steps=2,
                height=args.res,
                width=args.res,
                generator=torch.Generator(device="cuda").manual_seed(0),
                output_type="latent",
            )
        h.remove()
        a = tuple(t.clone() if torch.is_tensor(t) else t for t in grabbed["a"])
        kw = {
            k: (v.clone() if torch.is_tensor(v) else v)
            for k, v in grabbed["kw"].items()
        }
        kw.pop("return_dict", None)
        # the text encoders and VAE are not part of the backbone being compared
        for attr in ("vae", "text_encoder", "text_encoder_2", "text_encoder_3"):
            if getattr(pipe, attr, None) is not None:
                setattr(pipe, attr, None)
        gc.collect()
        torch.cuda.empty_cache()
        return tf, a, kw

    print(
        f"{'config':<20}{'weights GB':>10}{'peak GB':>12}{'activations':>12}"
        f"{'ms':>11}{'speedup':>10}"
    )
    print("-" * 75)

    measure("Vanilla", lambda: load_pipe())
    measure("Shiva-80", lambda: load_pipe(args.ckpt, args.lut))

    def shiva_quant():
        tf, a, kw = load_pipe(args.ckpt, args.lut)
        sys.path.insert(0, args.segquant)
        os.chdir(args.segquant)
        from segquant.torch.quantization import quantize
        from segquant.config import DType, Optimum, Calibrate
        import functools

        cfg = {
            # AdaLayerNorm's projection takes temb: shape (B, 1536), so M is 1 or 2.
            # CUTLASS int8 GEMMs fault on such a degenerate M, and these layers carry
            # almost no FLOPs, so excluding them costs nothing and is what a real
            # deployment would do anyway.
            # Only the FFN projections are quantised. The attention Linears take the
            # 333-token text stream (M=333, not 16-aligned) and AdaLayerNorm's take temb
            # (M=1..2); CUTLASS int8 GEMMs fault on both. The FFN sees the image tokens,
            # whose count the LUT fixes and K_ALIGN rounds to a multiple of 16, and it
            # is where the FLOPs are -- so this is both the workable and the useful set.
            "default": {"enable": False},
            "*ff.net.0.proj": {"enable": True},
            "*ff.net.2": {"enable": True},
            "_cfg": {
                "enable": True,
                "seglinear": False,
                "search_patterns": [],
                "real_quant": True,
                "opt": {"type": Optimum.DEFAULT, "alpha": 0.5},
                "calib": {"type": Calibrate.AMAX},
                "input_quant": {"type": DType.INT8, "axis": None},
                "weight_quant": {"type": DType.INT8, "axis": None},
            },
        }
        # the per-pattern entries need the same knobs as the default one
        base = {k: v for k, v in cfg["_cfg"].items()}
        cfg.pop("_cfg")
        for key in ("*ff.net.0.proj", "*ff.net.2"):
            cfg[key] = dict(base, enable=True)
        cfg["default"] = dict(base, enable=False)
        tf.forward = functools.partial(tf.forward, **kw)
        qtf = quantize(tf, [(tuple(a), {})], cfg, verbose=False)
        qtf.forward = functools.partial(type(qtf).forward, qtf, **kw)
        os.chdir(repo)
        return qtf, a, {}

    measure("Shiva-80 + INT8", shiva_quant)

    print("\nweights = allocated right after load; activations = peak minus weights.")


if __name__ == "__main__":
    main()
