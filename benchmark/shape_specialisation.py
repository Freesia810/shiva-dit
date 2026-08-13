"""What a bounded shape set actually buys from the compiler.

Two properties are claimed for a budget that is fixed per (layer, step):

  A. specialisation -- with the size known at compile time Inductor can fold it into
     the kernel (tile sizes, unrolling); a dynamic-shape kernel carries the size as a
     runtime argument. Measured at the SAME shape, so the only difference is whether
     the compiler was allowed to specialise.

  B. bounded capture -- CUDA Graphs record one graph per distinct size. A closed set
     is captured once and replayed forever; a set that keeps growing keeps paying
     capture cost and memory at serving time. Measured by feeding N distinct sizes
     and counting what gets recorded.

Run:
    python -m benchmark.shape_specialisation --tokens 2670 --repeats 200
"""

import argparse
import time

import torch

DIM, HEADS, HEAD_DIM = 1536, 24, 64
CTX_LEN = 333


def bench(mod, n, ctx, temb, repeats, device):
    def call():
        x = torch.randn(1, n, DIM, device=device, dtype=torch.float16)
        with torch.no_grad():
            mod(x, ctx, temb)

    for _ in range(10):
        call()
    torch.cuda.synchronize()
    ev = [
        (torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True))
        for _ in range(repeats)
    ]
    for s, e in ev:
        s.record()
        call()
        e.record()
    torch.cuda.synchronize()
    xs = sorted(s.elapsed_time(e) for s, e in ev)
    return xs[len(xs) // 2]


def main():
    from diffusers.models.attention import JointTransformerBlock

    p = argparse.ArgumentParser()
    p.add_argument(
        "--tokens", type=int, default=2670, help="the shape everything is measured at"
    )
    p.add_argument("--repeats", type=int, default=200)
    p.add_argument(
        "--shapes",
        type=int,
        nargs="+",
        default=[1, 8, 61, 200],
        help="how many distinct sizes to feed in the capture test",
    )
    args = p.parse_args()

    device = "cuda"
    ctx = torch.randn(1, CTX_LEN, DIM, device=device, dtype=torch.float16)
    temb = torch.randn(1, DIM, device=device, dtype=torch.float16)

    def fresh():
        torch._dynamo.reset()
        torch._dynamo.config.cache_size_limit = 1024
        torch._dynamo.config.accumulated_cache_size_limit = 8192
        return (
            JointTransformerBlock(
                dim=DIM, num_attention_heads=HEADS, attention_head_dim=HEAD_DIM
            )
            .to(device, torch.float16)
            .eval()
        )

    print(f"\n{'=' * 70}")
    print(f"A. specialisation, all measured at the same {args.tokens} tokens")
    print(f"{'=' * 70}")
    rows = {}
    for label, kw in [
        ("eager", None),
        ("compile dynamic=False", dict(dynamic=False)),
        ("compile dynamic=True", dict(dynamic=True)),
    ]:
        blk = fresh()
        mod = blk if kw is None else torch.compile(blk, **kw)
        ms = bench(mod, args.tokens, ctx, temb, args.repeats, device)
        rows[label] = ms
        print(f"  {label:<24}{ms:8.3f} ms/call")
    if "compile dynamic=False" in rows and "compile dynamic=True" in rows:
        st, dy = rows["compile dynamic=False"], rows["compile dynamic=True"]
        print(
            f"\n  static-shape kernel is {dy / st:.3f}x the speed of the dynamic one "
            f"({100 * (dy - st) / dy:+.1f}% time saved by specialising)"
        )

    print(f"\n{'=' * 70}")
    print("B. CUDA-graph capture as the number of distinct sizes grows")
    print(f"{'=' * 70}")
    print(f"  {'distinct sizes':>15}{'graphs':>9}{'capture s':>12}{'GiB held':>10}")
    for n_shapes in args.shapes:
        blk = fresh()
        mod = torch.compile(blk, mode="reduce-overhead")
        sizes = [args.tokens + 8 * i for i in range(n_shapes)]
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
        t0 = time.perf_counter()
        for n in sizes:
            x = torch.randn(1, n, DIM, device=device, dtype=torch.float16)
            with torch.no_grad():
                torch.compiler.cudagraph_mark_step_begin()
                mod(x, ctx, temb)
        torch.cuda.synchronize()
        cap = time.perf_counter() - t0
        from torch._dynamo.utils import counters

        graphs = counters.get("stats", {}).get("unique_graphs", 0)
        held = torch.cuda.max_memory_reserved() / 2**30
        print(f"  {n_shapes:>15}{graphs:>9}{cap:>12.1f}{held:>10.2f}")
    print("=" * 70)


if __name__ == "__main__":
    main()
