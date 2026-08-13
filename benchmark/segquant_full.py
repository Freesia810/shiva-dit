"""INT8 latency on the whole SD3 transformer, not one block.

A single block leaves the per-Linear quantise/dequantise kernels exposed; across the
full 24-block backbone there is far more GEMM work to amortise them against, so the
end-to-end picture can differ from the block-level one. This quantises every Linear in
the transformer through SegQuant's own `quantize()` (real_quant, INT8 weights and
activations) and times the forward against fp16.

Inputs are the real tensors the pipeline feeds the transformer, captured from a short
run, so shapes and dtypes match deployment. Calibration is a single batch -- this
measures speed only, quality is covered by the generation arms.

    CUTLASS_PATH=... python -m benchmark.segquant_full \
        --segquant /root/autodl-tmp/segquant --model /root/stable-diffusion-3-medium-diffusers
"""

import argparse
import os
import sys

import torch


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
    p.add_argument("--res", type=int, default=1024)
    p.add_argument("--repeats", type=int, default=20)
    args = p.parse_args()

    repo = os.getcwd()
    device = torch.device("cuda")

    from diffusers.pipelines import StableDiffusion3Pipeline

    print(f"loading {args.model} ...")
    pipe = StableDiffusion3Pipeline.from_pretrained(
        args.model, torch_dtype=torch.float16
    ).to(device)
    tf = pipe.transformer

    # capture the real transformer inputs
    grabbed = {}

    def grab(_m, a, kw):
        if not grabbed:
            grabbed["a"], grabbed["kw"] = a, kw
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

    def clone(o):
        if torch.is_tensor(o):
            return o.clone()
        if isinstance(o, (list, tuple)):
            return type(o)(clone(x) for x in o)
        if isinstance(o, dict):
            return {k: clone(v) for k, v in o.items()}
        return o

    a, kw = clone(grabbed["a"]), clone(grabbed["kw"])
    kw.pop("return_dict", None)
    n_lin = sum(1 for _, m in tf.named_modules() if isinstance(m, torch.nn.Linear))
    print(f"transformer has {n_lin} Linear modules")

    def call():
        return tf(*a, **kw, return_dict=False)

    t16 = median_ms(call, args.repeats)
    print(f"\nfp16 forward: {t16:.3f} ms")

    # free the rest of the pipeline before quantising
    del pipe.vae, pipe.text_encoder, pipe.text_encoder_2, pipe.text_encoder_3
    torch.cuda.empty_cache()

    sys.path.insert(0, args.segquant)
    os.chdir(args.segquant)
    from segquant.torch.quantization import quantize
    from segquant.config import DType, Optimum, Calibrate

    cfg = {
        "default": {
            "enable": True,
            # FX cannot trace the transformer's control flow; SegLinear is orthogonal
            # to the int8 GEMM under test.
            "seglinear": False,
            "search_patterns": [],
            "real_quant": True,
            "opt": {"type": Optimum.DEFAULT, "alpha": 0.5},
            "calib": {"type": Calibrate.AMAX},
            "input_quant": {"type": DType.INT8, "axis": None},
            "weight_quant": {"type": DType.INT8, "axis": None},
        },
    }
    # quantization.py feeds batch[0] as *args when it is a tuple, so the captured
    # kwargs are bound here rather than passed through the loader
    import functools

    orig_forward = tf.forward
    tf.forward = functools.partial(orig_forward, **kw)
    loader = [(tuple(a), {})]  # one batch is enough for a speed-only run

    print("quantising the full transformer to INT8 ...")
    qtf = quantize(tf, loader, cfg, verbose=False)
    qtf.forward = functools.partial(type(qtf).forward, qtf, **kw)

    def qcall():
        return qtf(*a, return_dict=False)

    t8 = median_ms(qcall, args.repeats)
    os.chdir(repo)

    print(f"\n{'':<14}{'ms':>10}{'speedup':>10}")
    print("-" * 34)
    print(f"{'fp16':<14}{t16:>10.3f}{'1.00x':>10}")
    print(f"{'SegQuant INT8':<14}{t8:>10.3f}{t16 / t8:>9.2f}x")


if __name__ == "__main__":
    main()
