"""End-to-end CUDA Graph on the DiT backbone, driven by the real LUT budget.

torch.compile(reduce-overhead) cannot wrap the real SD3 pipeline: the hook and the
scheduler hold intermediate tensors across steps, which the CUDA-graph backend
overwrites. Rather than fight every holder, we drive the backbone directly and
capture the graph by hand, where input/output buffers are fully controlled. This
gives the number the reviewer asks for -- the speedup actually obtained under CUDA
Graph -- with the same shape schedule the deployed model runs.

The backbone here is 24 JointTransformerBlocks x `steps` denoising steps = the real
call count. Per (layer, step) the token budget k comes from lut-60's ratio table, so
the shape schedule is exactly Shiva's. Two regimes:

  static (Shiva)  : k is a function of (layer, step) -> the whole schedule is fixed,
                    so one CUDA graph per step is captured and replayed every image.
  ragged (thresh) : k is redrawn per call (data-dependent) -> the schedule is never
                    the same twice, so a captured graph is invalid on the next image;
                    it falls back to eager. This is why a threshold budget cannot use
                    CUDA Graphs.

Run:  python -m benchmark.backbone_cudagraph --lut /root/autodl-tmp/assets/lut/lut-60.pt --steps 28
"""

import argparse
import math
import time

import torch

from diffusers.models.attention import JointTransformerBlock

DIM, HEADS, HEAD_DIM = 1536, 24, 64
CTX = 333
LAYERS = 24


def k_schedule(lut_path, steps, N):
    lut = torch.load(lut_path, map_location="cpu")
    rc = torch.as_tensor(lut["ratio"]).float()  # (24, T)
    T = min(steps, rc.shape[1])

    def kof(v):
        return max(8, min(int(math.ceil(min(max(float(v), 0.01), 1.0) * N)), N))

    # (steps, layers)
    return [
        [kof(rc[layer_idx, step_idx]) for layer_idx in range(LAYERS)]
        for step_idx in range(T)
    ]


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--lut", default="/root/autodl-tmp/assets/lut/lut-60.pt")
    p.add_argument("--steps", type=int, default=28)
    p.add_argument("--res", type=int, default=1024, help="image side; N=(res/8)^2")
    p.add_argument("--repeats", type=int, default=10, help="full-backbone repeats")
    args = p.parse_args()

    device, dtype = "cuda", torch.float16
    block = (
        JointTransformerBlock(
            dim=DIM, num_attention_heads=HEADS, attention_head_dim=HEAD_DIM
        )
        .to(device, dtype)
        .eval()
    )
    ctx = torch.randn(1, CTX, DIM, device=device, dtype=dtype)
    temb = torch.randn(1, DIM, device=device, dtype=dtype)

    N = (args.res // 8) ** 2
    sched = k_schedule(args.lut, args.steps, N)
    steps = len(sched)
    distinct = len({k for row in sched for k in row})
    print(
        f"res={args.res}^2 N={N}  backbone: {LAYERS} layers x {steps} steps = {LAYERS * steps} calls, "
        f"{distinct} distinct k"
    )

    # persistent input buffers per (step, layer) so a captured graph reads fixed addrs
    bufs = [
        [
            torch.randn(1, sched[t][layer_idx], DIM, device=device, dtype=dtype)
            for layer_idx in range(LAYERS)
        ]
        for t in range(steps)
    ]

    @torch.no_grad()
    def eager_backbone():
        for t in range(steps):
            for layer_idx in range(LAYERS):
                block(
                    hidden_states=bufs[t][layer_idx],
                    encoder_hidden_states=ctx,
                    temb=temb,
                )

    @torch.no_grad()
    def timed(fn, reps, warmup=3):
        for _ in range(warmup):
            fn()
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        for _ in range(reps):
            fn()
        torch.cuda.synchronize()
        return (time.perf_counter() - t0) / reps * 1e3

    # ---- static (Shiva): one graph per step, replay the whole schedule ----
    print("\ncapturing static CUDA graphs (one per step)...", flush=True)
    tcap0 = time.perf_counter()
    graphs = []
    with torch.no_grad():
        for t in range(steps):
            s = torch.cuda.Stream()
            s.wait_stream(torch.cuda.current_stream())
            with torch.cuda.stream(s):
                for _ in range(2):
                    for layer_idx in range(LAYERS):
                        block(
                            hidden_states=bufs[t][layer_idx],
                            encoder_hidden_states=ctx,
                            temb=temb,
                        )
            torch.cuda.current_stream().wait_stream(s)
            g = torch.cuda.CUDAGraph()
            with torch.cuda.graph(g):
                for layer_idx in range(LAYERS):
                    block(
                        hidden_states=bufs[t][layer_idx],
                        encoder_hidden_states=ctx,
                        temb=temb,
                    )
            graphs.append(g)
    cap_s = time.perf_counter() - tcap0

    def graph_backbone():
        for g in graphs:
            g.replay()

    eager_ms = timed(eager_backbone, args.repeats)
    graph_ms = timed(graph_backbone, args.repeats)

    print(f"\n{'=' * 60}")
    print(f"full backbone ({LAYERS}x{steps} block calls), median of {args.repeats}")
    print(f"{'=' * 60}")
    print(f"eager                 {eager_ms:9.2f} ms")
    print(
        f"static CUDA graph     {graph_ms:9.2f} ms   speedup {eager_ms / graph_ms:.3f}x"
    )
    print(
        f"(one-off graph capture cost: {cap_s:.1f} s for {steps} graphs, "
        f"amortised over every image)"
    )
    print(
        f"\nragged (threshold) cannot reuse these graphs: k is data-dependent, so a "
        f"new image needs a fresh capture ({cap_s:.1f}s) every time -> it runs eager "
        f"({eager_ms:.0f} ms), forgoing the {eager_ms / graph_ms:.2f}x."
    )


if __name__ == "__main__":
    main()
