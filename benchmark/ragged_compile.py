"""Does a data-dependent token count actually prevent static graph compilation?

The paper claims threshold-masking methods "result in ragged tensors with variable
sequence lengths, which preclude the use of static graph compilation and introduce
significant kernel launch overheads". That is an assertion about shapes, so it is
testable directly -- and the honest way to test it is to change nothing except the
shape sequence.

Two stages:

  --record   run the real pipeline with real weights and log, for every block call,
             how many image tokens actually entered the block. Shiva reads k from the
             deployed LUT; DyDiT derives it from `.nonzero()` on a thresholded mask
             (baselines/dy_dit.py:187), so its length is data-dependent.

  --replay   take ONE clean JointTransformerBlock, torch.compile it once, and push
             each recorded sequence through it. Same block, same compiler, same
             number of calls -- the only difference is the sequence of shapes.
             Reports recompilations, whether the dynamo cache limit was hit (which
             silently drops the block back to eager), and wall time.

Run:
    python -m benchmark.ragged_compile --record shiva60 \
        --ckpt_path .../shiva-sd3-60.pt --gater_lut .../lut-60.pt --steps 50
    python -m benchmark.ragged_compile --record dydit --ckpt_path .../dydit.pt --steps 50
    python -m benchmark.ragged_compile --replay shiva60 dydit
"""

import argparse
import json
import os
import time
from collections import Counter

import torch

DIM, HEADS, HEAD_DIM = 1536, 24, 64
CTX_LEN = 333
OUT_DIR = "results/ragged"

# Deliberately spread across content types: a threshold on a data-dependent score
# has no reason to land on the same token count for a portrait and for a diagram.
PROMPTS = [
    "a photograph of an astronaut riding a horse on mars, detailed, dramatic lighting",
    "a close-up portrait of an elderly fisherman, weathered face, soft window light",
    "a minimalist line drawing of a bicycle on plain white background",
    "a dense aerial view of a night city, thousands of lights, rain, reflections",
    "a still life of three lemons on a wooden table, muted colours",
    "an intricate stained glass window depicting a phoenix, backlit",
    "a chalkboard covered in handwritten mathematical equations",
    "a single red balloon against an empty grey sky",
]


# --------------------------------------------------------------------------- #
# stage 1: record the shapes a real run actually produces
# --------------------------------------------------------------------------- #
def record(args):
    from shiva_dit.utils.inference_utils import load_pipeline_for_inference

    dtype = torch.float16
    pipe = load_pipeline_for_inference(
        ckpt_path=args.ckpt_path,
        base_model_path=args.base_model_path,
        model_type=args.model_type,
        device="cuda",
        dtype=dtype,
        gater_lut_path=args.gater_lut,
        optimize_hooks=True,
        merge_lora=True,
    )

    # Probe attention AND the FFN separately. DyDiT gathers only for the feed-forward
    # (baselines/dy_dit.py:210-221) and leaves attention on the full sequence, so a
    # single probe on `attn` would report "no ragged shapes" and miss the point --
    # and would also hide that DyDiT never prunes the quadratic term at all.
    seen = {"attn": [], "ffn": []}

    def probe(layer_idx, where):
        def pre_hook(module, pargs, kwargs):
            t = kwargs.get("hidden_states")
            if t is None:
                for c in pargs:
                    if torch.is_tensor(c) and c.dim() >= 2:
                        t = c
                        break
            if torch.is_tensor(t) and t.dim() >= 2:
                # (B, N, D) for attention, sometimes (N, D) after DyDiT's gather
                n = int(t.shape[1]) if t.dim() == 3 else int(t.shape[0])
                seen[where].append((layer_idx, n))

        return pre_hook

    blocks = pipe.transformer.transformer_blocks
    for i, b in enumerate(blocks):
        if hasattr(b, "attn"):
            b.attn.register_forward_pre_hook(probe(i, "attn"), with_kwargs=True)
        if hasattr(b, "ff"):
            b.ff.register_forward_pre_hook(probe(i, "ffn"), with_kwargs=True)

    # The decisive question is not how many shapes ONE image produces, but whether
    # that set is closed. Shiva's budget is a function of (layer, step) read from the
    # LUT, so it is identical for every prompt; DyDiT thresholds a score computed
    # from the input, so each new prompt can introduce shapes never seen before.
    # Running several prompts and watching the union grow separates the two.
    per_prompt = {"attn": [], "ffn": []}
    growth = {"attn": [], "ffn": []}
    union = {"attn": set(), "ffn": set()}

    for pi, prompt in enumerate(PROMPTS[: args.prompts]):
        seen["attn"].clear()
        seen["ffn"].clear()
        g = torch.Generator(device="cuda").manual_seed(42 + pi)
        pipe(
            prompt=prompt,
            num_inference_steps=args.steps,
            height=args.height,
            width=args.width,
            generator=g,
            num_images_per_prompt=args.batch,
        )
        for where in ("attn", "ffn"):
            calls = list(seen[where])
            per_prompt[where].append(calls)
            counts = [n for _, n in calls]
            if not counts:
                continue
            before = len(union[where])
            union[where] |= set(counts)
            growth[where].append(
                (
                    pi + 1,
                    len(set(counts)),
                    len(union[where]),
                    len(union[where]) - before,
                )
            )

    os.makedirs(OUT_DIR, exist_ok=True)
    path = os.path.join(OUT_DIR, f"{args.record}_shapes.json")
    with open(path, "w") as f:
        json.dump(
            {
                "label": args.record,
                "steps": args.steps,
                "prompts": args.prompts,
                "per_prompt": per_prompt,
                "growth": growth,
            },
            f,
        )

    print(f"\n{'=' * 74}")
    print(
        f"recorded {args.record}: {args.prompts} prompt(s), "
        f"{args.steps} steps, 1024x1024"
    )
    for where in ("attn", "ffn"):
        flat = [n for calls in per_prompt[where] for _, n in calls]
        if not flat:
            print(f"\n[{where}] no calls captured")
            continue
        hist = Counter(flat)
        print(
            f"\n[{where}]  {len(flat)} calls   "
            f"min={min(flat)}  max={max(flat)}  mean={sum(flat) / len(flat):.1f}"
        )
        print(f"  top shapes: {hist.most_common(5)}")
        print(f"  {'prompt':>7}{'distinct(this)':>16}{'union':>8}{'NEW':>7}")
        for i, d, u, new in growth[where]:
            print(f"  {i:>7}{d:>16}{u:>8}{new:>7}")
    print(f"\n-> {path}")
    print("=" * 74)


# --------------------------------------------------------------------------- #
# stage 2: replay the shape sequences through one compiled block
# --------------------------------------------------------------------------- #
def replay(args):
    """Same module, same compiler, same call count -- only the shape sequence differs.

    The carrier is the feed-forward, because that is the one module both methods
    drive with their own token count: Shiva feeds it k from the LUT, DyDiT feeds it
    whatever `.nonzero()` produced. Anything that differs between the two runs is
    therefore attributable to the shapes alone.
    """
    from diffusers.models.attention import FeedForward

    device, dtype = "cuda", torch.float16
    results = {}

    for label in args.replay:
        path = os.path.join(OUT_DIR, f"{label}_shapes.json")
        if not os.path.exists(path):
            print(f"missing {path} -- run --record {label} first")
            continue
        blob = json.load(open(path))
        per_prompt = blob["per_prompt"]
        seqs = per_prompt[args.where] if isinstance(per_prompt, dict) else per_prompt
        counts = [n for calls in seqs for _, n in calls][: args.limit]
        if not counts:
            print(f"{label}: no calls recorded for '{args.where}'")
            continue

        torch._dynamo.reset()
        torch._dynamo.config.cache_size_limit = args.cache_limit
        ff = FeedForward(dim=DIM, dim_out=DIM, activation_fn="gelu-approximate")
        ff = ff.to(device, dtype).eval()
        kw = {}
        if args.mode:
            kw["mode"] = args.mode
        compiled = torch.compile(ff, dynamic=args.dynamic, **kw)

        def run(n):
            x = torch.randn(1, n, DIM, device=device, dtype=dtype)
            with torch.no_grad():
                compiled(x)

        # warm the first shape so one-off compiler startup is not charged to anyone
        run(counts[0])
        torch.cuda.synchronize()

        from torch._dynamo.utils import counters

        counters.clear()
        t0 = time.perf_counter()
        for n in counts:
            run(n)
        torch.cuda.synchronize()
        wall = time.perf_counter() - t0

        hist = Counter(counts)
        graphs = counters.get("stats", {}).get("unique_graphs", 0)
        results[label] = dict(
            calls=len(counts),
            distinct=len(hist),
            wall=wall,
            ms_per_call=wall / len(counts) * 1e3,
            graphs=graphs,
            over_limit=len(hist) > args.cache_limit,
        )

    print(f"\n{'=' * 84}")
    print(f"replay of recorded '{args.where}' shapes through ONE compiled FeedForward")
    print(
        f"(dynamic={args.dynamic}, mode={args.mode}, "
        f"cache_size_limit={args.cache_limit})"
    )
    print(f"{'=' * 84}")
    print(
        f"{'sequence':<12}{'calls':>8}{'distinct':>10}{'graphs':>9}"
        f"{'ms/call':>10}{'total s':>10}{'over limit':>12}"
    )
    for label, r in results.items():
        print(
            f"{label:<12}{r['calls']:>8}{r['distinct']:>10}{r['graphs']:>9}"
            f"{r['ms_per_call']:>10.3f}{r['wall']:>10.2f}"
            f"{str(r['over_limit']):>12}"
        )
    print("=" * 84)
    if len(results) == 2:
        (la, a), (lb, b) = list(results.items())
        print(f"{lb} / {la} ms per call: {b['ms_per_call'] / a['ms_per_call']:.3f}x")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--record", default=None, help="label to record under")
    p.add_argument("--replay", nargs="+", default=None, help="labels to replay")
    p.add_argument("--ckpt_path", default=None)
    p.add_argument("--model_type", default="sd3")
    p.add_argument("--base_model_path", default=None)
    p.add_argument("--gater_lut", default=None)
    p.add_argument("--steps", type=int, default=50)
    p.add_argument(
        "--prompts",
        type=int,
        default=4,
        help="how many different prompts to record (max 8)",
    )
    p.add_argument(
        "--batch",
        type=int,
        default=1,
        help="images per prompt. DyDiT unions the per-sample masks "
        "(baselines/dy_dit.py:176), so its kept-token count should "
        "grow with the batch; a LUT budget cannot.",
    )
    p.add_argument("--height", type=int, default=1024)
    p.add_argument("--width", type=int, default=1024)
    p.add_argument(
        "--limit", type=int, default=1200, help="replay at most this many calls"
    )
    p.add_argument(
        "--mode",
        default=None,
        help="torch.compile mode, e.g. reduce-overhead (uses CUDA graphs, "
        "which is the 'static graph compilation' the paper appeals to)",
    )
    p.add_argument(
        "--where",
        default="ffn",
        choices=["attn", "ffn"],
        help="which recorded probe to replay",
    )
    p.add_argument(
        "--cache_limit",
        type=int,
        default=8,
        help="torch._dynamo.config.cache_size_limit (default is 8)",
    )
    p.add_argument(
        "--dynamic",
        default=None,
        type=lambda s: None if s == "none" else s == "true",
        help="torch.compile(dynamic=...): none|true|false",
    )
    args = p.parse_args()

    if args.record:
        record(args)
    if args.replay:
        replay(args)


if __name__ == "__main__":
    main()
