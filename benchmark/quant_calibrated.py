"""W8A8 sensitivity with calibration and evaluation on *different* prompts.

An earlier sweep calibrated activation ranges on the very tensor it then measured the
error on, and every capture came from one prompt. Both make the result an optimistic
lower bound: the scale is fitted to the test sample, and a single prompt's activation
range does not represent the distribution a deployed quantiser meets. This version
splits the prompts -- ranges are fitted on one set, error is reported on another --
which is the only way the number means anything about deployment.

Two modes:
  scope  : quantise a group of Linears (ffn / +attn out / +all attn / everything)
  module : quantise exactly one Linear at a time, to find which one carries the error

Captures come from `test.toy.capture_block_inputs --weights-once`, i.e.
`L{layer}_S{step}_P{prompt}.pt` for the activations and `weights_L{layer}.pt`
for the shared block weights.

    python -m benchmark.quant_calibrated --captures /root/autodl-tmp/captures2 \
        --layers 1 5 11 17 23 --step 14 --mode scope
"""

import argparse
import os

import torch

from diffusers.models.attention import JointTransformerBlock

SCOPES = {
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
    "everything": None,
}


def load_capture(path, device, dtype):
    b = torch.load(path, map_location="cpu", weights_only=False)
    return (
        b["hidden_states"][:1].to(device, dtype),
        b["encoder_hidden_states"][:1].to(device, dtype),
        b["temb"][:1].to(device, dtype),
        b["block_config"],
    )


def build_block(cfg, weights, device, dtype):
    blk = JointTransformerBlock(
        dim=cfg["hidden_dim"],
        num_attention_heads=cfg["num_attention_heads"],
        attention_head_dim=cfg["attention_head_dim"],
        context_pre_only=cfg.get("context_pre_only", False),
    )
    blk.load_state_dict(weights, strict=True)
    return blk.to(device=device, dtype=dtype).eval()


def rel_err(a, b):
    return ((a - b).norm() / b.norm().clamp(min=1e-12)).item()


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--captures", required=True)
    p.add_argument("--layers", type=int, nargs="+", default=[1, 5, 11, 17, 23])
    p.add_argument("--step", type=int, default=14)
    p.add_argument("--calib_prompts", type=int, nargs="+", default=[0, 1, 2, 3, 4, 5])
    p.add_argument("--eval_prompts", type=int, nargs="+", default=[6, 7])
    p.add_argument("--mode", choices=["scope", "module"], default="scope")
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = p.parse_args()

    from optimum.quanto import Calibration, freeze, qint8, quantize

    device, dtype = torch.device(args.device), torch.float32
    print(
        f"calib prompts {args.calib_prompts} -> eval prompts {args.eval_prompts} "
        f"(held out), step {args.step}, device {args.device}"
    )

    for layer in args.layers:
        wpath = os.path.join(args.captures, f"weights_L{layer:02d}.pt")
        if not os.path.exists(wpath):
            print(f"skip layer {layer}: no {wpath}")
            continue
        weights = torch.load(wpath, map_location="cpu", weights_only=False)

        def cap(pi):
            return os.path.join(
                args.captures, f"L{layer:02d}_S{args.step:02d}_P{pi:02d}.pt"
            )

        calib = [
            load_capture(cap(i), device, dtype)
            for i in args.calib_prompts
            if os.path.exists(cap(i))
        ]
        evals = [
            load_capture(cap(i), device, dtype)
            for i in args.eval_prompts
            if os.path.exists(cap(i))
        ]
        if not calib or not evals:
            print(f"skip layer {layer}: missing captures")
            continue
        cfg = calib[0][3]

        ref_blk = build_block(cfg, weights, device, dtype)
        with torch.no_grad():
            refs = [ref_blk(x, c, t)[1] for x, c, t, _ in evals]

        have = {n for n, m in ref_blk.named_modules() if isinstance(m, torch.nn.Linear)}
        targets = (
            {k: v for k, v in SCOPES.items()}
            if args.mode == "scope"
            else {n: [n] for n in sorted(have)}
        )

        print(f"\n=== layer {layer} ===")
        print(
            f"{'scope' if args.mode == 'scope' else 'module':<24}"
            f"{'W8A8':>10}{'W8 only':>10}"
        )
        print("-" * 44)
        for name, include in targets.items():
            inc = None if include is None else [t for t in include if t in have]
            if include is not None and not inc:
                continue
            out = {}
            for mode in ("w8a8", "w8"):
                blk = build_block(cfg, weights, device, dtype)
                kw = {"weights": qint8}
                if mode == "w8a8":
                    kw["activations"] = qint8
                if inc is None:
                    quantize(blk, **kw)
                else:
                    quantize(blk, include=inc, **kw)
                if mode == "w8a8":
                    with Calibration(), torch.no_grad():
                        for x, c, t, _ in calib:  # ranges from the calib prompts only
                            blk(x, c, t)
                freeze(blk)
                with torch.no_grad():
                    errs = [
                        rel_err(blk(x, c, t)[1], r)
                        for (x, c, t, _), r in zip(evals, refs)
                    ]
                out[mode] = sum(errs) / len(errs)
            print(f"{name:<24}{out['w8a8']:>9.3%}{out['w8']:>9.3%}")


if __name__ == "__main__":
    main()
