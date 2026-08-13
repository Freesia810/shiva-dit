"""Block-level CUDA Graph speedup: what a static token count actually buys.

The paper claims (p18) that the argsort+gather forward "produces static tensor shapes
amenable to CUDA Graph compilation", and (p1) that ragged shapes "preclude the use of
static graph compilation". Both are directly testable on one DiT block, without the
end-to-end hook (whose per-call state on `self` is what breaks the compiled CUDA-graph
backend). A block captured by hand has fully controlled input/output buffers, so it
runs where the end-to-end path currently does not.

  A. static k  -> capture ONE CUDA graph, replay it. Speedup vs eager at the same k.
  B. ragged    -> a threshold budget yields a new length each call; a manual graph is
                  captured per distinct shape, so N shapes cost N captures (time +
                  memory). We report how that grows -- the "not enumerable" cost.

Run:  python -m benchmark.block_cudagraph --repeats 200
"""

import argparse
import time

import torch

from diffusers.models.attention import JointTransformerBlock

DIM, HEADS, HEAD_DIM = 1536, 24, 64
CTX = 333


def make_block(device, dtype):
    b = JointTransformerBlock(
        dim=DIM, num_attention_heads=HEADS, attention_head_dim=HEAD_DIM
    )
    return b.to(device, dtype).eval()


def inputs(n, device, dtype):
    return (
        torch.randn(1, n, DIM, device=device, dtype=dtype),
        torch.randn(1, CTX, DIM, device=device, dtype=dtype),
        torch.randn(1, DIM, device=device, dtype=dtype),
    )


@torch.no_grad()
def median_ms(fn, repeats, warmup=20):
    """Per-call GPU time, bracketed by events.

    This is the optimistic number for eager execution: the CPU enqueues far ahead of
    the device, so the idle gaps *between* calls fall outside every measured window
    and only the kernels themselves are counted. Graph replay removes exactly those
    gaps, so this metric is structurally blind to most of what capture buys.
    """
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


@torch.no_grad()
def wall_ms(fn, repeats, warmup=20):
    """Amortised wall-clock per call: one sync at each end, nothing in between.

    Launch cost and inter-call gaps are inside the measured window here, so this is
    the number that corresponds to end-to-end latency.
    """
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(repeats):
        fn()
    torch.cuda.synchronize()
    return (time.perf_counter() - t0) * 1000.0 / repeats


@torch.no_grad()
def capture(block, n, device, dtype):
    """Capture one CUDA graph for a fixed length n; returns (replay_fn, static_in)."""
    x, ctx, temb = inputs(n, device, dtype)
    # warmup on a side stream (required before capture)
    s = torch.cuda.Stream()
    s.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(s):
        for _ in range(3):
            block(hidden_states=x, encoder_hidden_states=ctx, temb=temb)
    torch.cuda.current_stream().wait_stream(s)

    g = torch.cuda.CUDAGraph()
    with torch.cuda.graph(g):
        block(hidden_states=x, encoder_hidden_states=ctx, temb=temb)

    def replay():
        g.replay()

    return replay, (x, ctx, temb), g


def main():
    p = argparse.ArgumentParser()
    p.add_argument(
        "--tokens",
        type=int,
        nargs="+",
        default=[512, 1024, 2457, 4096],
        help="static token counts to capture (2457 = 60% of 4096)",
    )
    p.add_argument("--repeats", type=int, default=200)
    p.add_argument(
        "--ragged-shapes",
        type=int,
        default=60,
        help="distinct lengths a threshold budget would produce",
    )
    args = p.parse_args()

    device, dtype = "cuda", torch.float16
    block = make_block(device, dtype)

    print(
        f"\n{'=' * 78}\nA. static k: one CUDA graph, replay vs eager (same k)\n{'=' * 78}"
    )
    print(
        f"{'':>8}{'--- per-call GPU time ---':^33}{'--- amortised wall clock ---':^35}"
    )
    print(
        f"{'tokens':>8}{'eager':>11}{'graph':>11}{'speedup':>11}"
        f"{'eager':>11}{'graph':>11}{'speedup':>11}"
    )
    for n in args.tokens:
        x, ctx, temb = inputs(n, device, dtype)

        def call():
            return block(hidden_states=x, encoder_hidden_states=ctx, temb=temb)

        e_ev = median_ms(call, args.repeats)
        e_wall = wall_ms(call, args.repeats)
        replay, _, g = capture(block, n, device, dtype)
        g_ev = median_ms(replay, args.repeats)
        g_wall = wall_ms(replay, args.repeats)
        print(
            f"{n:>8}{e_ev:>11.3f}{g_ev:>11.3f}{e_ev / g_ev:>10.3f}x"
            f"{e_wall:>11.3f}{g_wall:>11.3f}{e_wall / g_wall:>10.3f}x",
            flush=True,
        )
        del g
        torch.cuda.empty_cache()

    print(
        f"\n{'=' * 66}\nB. ragged: capture cost grows with #distinct shapes\n{'=' * 66}"
    )
    # a static budget sees ONE shape (one capture); a threshold budget sees many.
    base = 2457
    shapes = [base + 8 * i for i in range(args.ragged_shapes)]
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    t0 = time.perf_counter()
    graphs = []
    mem_pts = []
    for i, n in enumerate(shapes):
        replay, _, g = capture(block, n, device, dtype)
        graphs.append(g)  # keep alive, as a real deploy must
        if (i + 1) in (1, 10, 30, 60):
            torch.cuda.synchronize()
            mem_pts.append(
                (
                    i + 1,
                    time.perf_counter() - t0,
                    torch.cuda.max_memory_allocated() / 2**30,
                )
            )
    print(f"{'#shapes':>8}{'capture s':>12}{'GiB held':>10}")
    for k, sec, gib in mem_pts:
        print(f"{k:>8}{sec:>12.2f}{gib:>10.2f}")
    print(
        "\nstatic budget = 1 shape = 1 capture (constant);"
        " threshold budget pays this per new shape."
    )


if __name__ == "__main__":
    main()
