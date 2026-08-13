"""Phase breakdown measured on the REAL inference path.

`benchmark/phase_breakdown.py` times a synthetic single-block rig and runs the
ShivaRatioPolicy network, but deployment reads a LUT -- so its cost split describes
a configuration that never executes. This runs the actual pipeline, with the actual
checkpoint and LUT, and accumulates every phase over all 24 layers x every denoising
step through the instrumentation in shiva_dit/utils/phase_profiler.py.

Phases (as they occur in ShivaDitBlockHook):

    ratio (LUT lookup)   read k for this (layer, step)          overhead
    router               score every token                      overhead
    topk + gather        argsort scores, gather (B,N,D)->(B,k,D) overhead
    rope gather          gather the selected RoPE entries        overhead
    block (attn+FFN) @ k the transformer block on k tokens       compute
    scatter              (B,k,D) -> (B,N,D)                      overhead

Run:
    python -m benchmark.infer_phase_profile \
        --ckpt_path .../shiva-sd3-80.pt --gater_lut .../lut-80_90.pt \
        --steps 50 --label shiva90
"""

import argparse
import time

import torch

from shiva_dit.utils import phase_profiler as pp
from shiva_dit.utils.inference_utils import load_pipeline_for_inference

PROMPT = (
    "a photograph of an astronaut riding a horse on mars, "
    "detailed, high resolution, dramatic lighting"
)

# Sub-phases are printed indented (two leading spaces) and are already contained in
# their parent phase, so they are excluded from every total.
OVERHEAD_PHASES = {
    "ratio (LUT lookup)",
    "router",
    "topk + gather",
    "rope gather",
    "scatter",
    "selector: .nonzero() sync",
}
BLOCK_TOTAL = "BLOCK total (incl. all Shiva phases)"
ROWS = [
    "ratio (LUT lookup)",
    "router",
    "topk + gather",
    "  topk: argsort",
    "  topk: k .item() sync",
    "  topk: gather",
    "rope gather",
    "scatter",
    "selector: .nonzero() sync",
    "block (attn+FFN) @ k",
    "attention submodule (all layers)",
    "feed-forward submodule (all layers)",
    BLOCK_TOTAL,
]


ATTN_TOTAL = "attention submodule (all layers)"
FFN_TOTAL = "feed-forward submodule (all layers)"


def install_block_timers(pipe):
    """Time each transformer block end to end, plus the attention and feed-forward
    submodules separately. Works with or without the Shiva hook, so on vanilla these
    are the dense costs and on Shiva they are the costs at k -- which is exactly the
    "attention and FFN savings" split the reviewer asked for. attn/ffn are wrapped by
    their own pre/post hooks; they sum to less than the block because norm, gate and
    the pruning machinery live between them."""
    blocks = getattr(pipe.transformer, "transformer_blocks", None)
    if blocks is None:
        return 0
    for b in blocks:
        b.register_forward_pre_hook(lambda m, inp: pp.start(BLOCK_TOTAL, id(m)))
        b.register_forward_hook(lambda m, inp, out: pp.stop(BLOCK_TOTAL, id(m)))
        if hasattr(b, "attn"):
            b.attn.register_forward_pre_hook(lambda m, inp: pp.start(ATTN_TOTAL, id(m)))
            b.attn.register_forward_hook(lambda m, inp, out: pp.stop(ATTN_TOTAL, id(m)))
        if hasattr(b, "ff"):
            b.ff.register_forward_pre_hook(lambda m, inp: pp.start(FFN_TOTAL, id(m)))
            b.ff.register_forward_hook(lambda m, inp, out: pp.stop(FFN_TOTAL, id(m)))
    return len(blocks)


def main():
    p = argparse.ArgumentParser()
    p.add_argument(
        "--ckpt_path",
        default=None,
        help="omit to profile the unpruned baseline (no phases recorded)",
    )
    p.add_argument("--model_type", default="sd3")
    p.add_argument("--base_model_path", default=None)
    p.add_argument("--gater_lut", default=None)
    p.add_argument("--steps", type=int, default=50)
    p.add_argument(
        "--warmup",
        type=int,
        default=1,
        help="full pipeline runs discarded before measuring",
    )
    p.add_argument("--runs", type=int, default=3, help="measured pipeline runs")
    p.add_argument("--height", type=int, default=1024)
    p.add_argument("--width", type=int, default=1024)
    p.add_argument("--dtype", default="fp16", choices=["fp16", "bf16"])
    p.add_argument("--label", default="run")
    p.add_argument(
        "--batch",
        type=int,
        default=1,
        help="images generated per pipeline call (num_images_per_prompt); "
        "for batch-size sensitivity / throughput",
    )
    p.add_argument(
        "--k_jitter",
        type=int,
        default=0,
        help="perturb the LUT budget by +/- this many tokens on every "
        "call. Same model, same pruned modules, same mean retention "
        "-- only the regularity of the shapes changes. This is the "
        "controlled variable for the ragged-vs-regular question.",
    )
    p.add_argument(
        "--cfg",
        type=float,
        default=None,
        help="guidance scale; omit to use the pipeline default",
    )
    p.add_argument(
        "--skip_layers",
        type=int,
        nargs="+",
        default=None,
        help="block indices to leave dense; defaults to [0]",
    )
    p.add_argument(
        "--compile",
        default=None,
        help="torch.compile mode for the whole transformer, e.g. "
        "reduce-overhead (CUDA graphs). Phase timers are disabled "
        "when set: CUDA events cannot be recorded inside a captured "
        "graph, so only wall time is meaningful here.",
    )
    args = p.parse_args()

    if args.k_jitter:
        import shiva_dit.shiva_hook as _sh

        _sh.K_JITTER = args.k_jitter

    dtype = torch.float16 if args.dtype == "fp16" else torch.bfloat16
    device = "cuda"

    pipe = load_pipeline_for_inference(
        ckpt_path=args.ckpt_path,
        base_model_path=args.base_model_path,
        model_type=args.model_type,
        skip_layer_indices=args.skip_layers,
        device=device,
        dtype=dtype,
        gater_lut_path=args.gater_lut,
        optimize_hooks=True,
        merge_lora=True,
    )

    if args.compile:
        # Shiva needs one graph per distinct k (61 at 28 steps, 87 at 50), so the
        # dynamo cache must be raised well above its default of 8.
        torch._dynamo.config.cache_size_limit = 512
        torch._dynamo.config.accumulated_cache_size_limit = 4096
        # A diffusion pipeline calls the transformer once per step. Under CUDA graphs
        # the previous call's output buffer is reused, so each invocation has to be
        # announced or the pipeline reads a tensor that has already been overwritten.
        _inner = torch.compile(pipe.transformer.forward, mode=args.compile)

        def _clone(o):
            # CUDA graphs reuse their output buffers, so anything the caller (or a
            # hook holding state on self) reads after the next replay must be a copy.
            if torch.is_tensor(o):
                return o.clone()
            if isinstance(o, (tuple, list)):
                return type(o)(_clone(x) for x in o)
            if hasattr(o, "sample") and torch.is_tensor(getattr(o, "sample")):
                o.sample = o.sample.clone()
            return o

        def _wrapped(*a, **kw):
            torch.compiler.cudagraph_mark_step_begin()
            return _clone(_inner(*a, **kw))

        pipe.transformer.forward = _wrapped
        n_blocks = 0
    else:
        n_blocks = install_block_timers(pipe)

    def one_run():
        g = torch.Generator(device=device).manual_seed(42)
        extra = {} if args.cfg is None else {"guidance_scale": args.cfg}
        return pipe(
            prompt=PROMPT,
            num_inference_steps=args.steps,
            **extra,
            height=args.height,
            width=args.width,
            generator=g,
            num_images_per_prompt=args.batch,
        )

    for _ in range(args.warmup):
        one_run()

    torch.cuda.synchronize()
    if not args.compile:
        pp.enable()
    pp.reset()
    t0 = time.perf_counter()
    for _ in range(args.runs):
        one_run()
    torch.cuda.synchronize()
    wall = (time.perf_counter() - t0) / args.runs * 1e3
    totals, counts = pp.report()
    pp.disable()

    if args.compile:
        # Why did compilation help or hurt? Graph breaks fragment the compiled region
        # and add guard overhead with no fusion in return; recompiles mean the guards
        # keep failing. Both are visible in dynamo's counters.
        from torch._dynamo.utils import counters

        gb = counters.get("graph_break", {})
        print(
            f"\ndynamo: unique_graphs="
            f"{counters.get('stats', {}).get('unique_graphs', 0)}  "
            f"total graph breaks={sum(gb.values())}"
        )
        for reason, n in sorted(gb.items(), key=lambda x: -x[1])[:6]:
            print(f"   {n:6d}  {reason[:96]}")

    print(f"\n{'=' * 72}")
    print(
        f"REAL inference path -- {args.label}"
        + (f"   [k_jitter=+/-{args.k_jitter}]" if args.k_jitter else "   [regular k]")
    )
    print(f"ckpt={args.ckpt_path}  lut={args.gater_lut}")
    print(
        f"{args.steps} steps, {args.height}x{args.width}, {args.dtype}, "
        f"{args.runs} runs averaged"
    )
    print(f"{'=' * 72}")
    per_img = wall / args.batch
    print(
        f"pipeline wall time      {wall:9.1f} ms/call   "
        f"({n_blocks} blocks timed, batch={args.batch})"
    )
    print(
        f"per-image latency       {per_img:9.1f} ms/image   "
        f"throughput {1000 / per_img:6.3f} img/s"
    )

    if not totals:
        print("no phases recorded")
        print("=" * 72)
        return

    per_run = {k: v / args.runs for k, v in totals.items()}
    overhead = sum(v for k, v in per_run.items() if k in OVERHEAD_PHASES)
    block_k = per_run.get("block (attn+FFN) @ k", 0.0)
    block_total = per_run.get(BLOCK_TOTAL, 0.0)

    print(
        f"{'phase':<28}{'ms/image':>11}{'calls':>8}{'ms/call':>10}"
        f"{'% ovh':>8}{'% wall':>9}"
    )
    print("-" * 76)
    for name in ROWS:
        if name not in per_run:
            continue
        ms = per_run[name]
        n = max(counts[name] // args.runs, 1)
        share = (
            f"{100 * ms / overhead:6.1f}%" if name in OVERHEAD_PHASES else "       -"
        )
        print(
            f"{name:<28}{ms:11.2f}{n:8d}{ms / n:10.4f}{share:>8}{100 * ms / wall:8.1f}%"
        )
    print("-" * 76)
    if overhead:
        print(
            f"{'pruning overhead':<28}{overhead:11.2f}{'':8}{'':10}"
            f"{'100.0%':>8}{100 * overhead / wall:8.1f}%"
        )
    if block_k:
        print(
            f"{'block compute @ k':<28}{block_k:11.2f}{'':8}{'':10}{'':>8}"
            f"{100 * block_k / wall:8.1f}%"
        )
    if block_total:
        print(
            f"{'DiT blocks, all-in':<28}{block_total:11.2f}{'':8}{'':10}{'':>8}"
            f"{100 * block_total / wall:8.1f}%"
        )
        rest = wall - block_total
        print(
            f"{'outside DiT (VAE/TE/sched)':<28}{rest:11.2f}{'':8}{'':10}{'':>8}"
            f"{100 * rest / wall:8.1f}%"
        )
    print("=" * 76)


if __name__ == "__main__":
    main()
