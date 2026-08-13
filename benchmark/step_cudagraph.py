"""CUDA Graph on a whole denoising step, pruning path included.

`block_cudagraph.py` captures a bare block and finds almost nothing to save, because a
tight loop that does nothing but call one block keeps the device saturated: the host
runs far ahead and there are no gaps to remove. Real inference does not look like that.
Sampling `nvidia-smi` during generation puts mean GPU utilisation at 59%, so the device
spends a large share of the run waiting on the host.

What sits between the blocks is the pruning path: score every token, sort, gather the
survivors, run the block, scatter them back. These are many small kernels whose cost is
dominated by dispatch rather than arithmetic, and they are exactly what a captured graph
removes. Because our budget is a function of (layer, step) alone, every shape along that
path is known before the graph is recorded, so the entire step can be captured as one
graph. A threshold budget cannot: `nonzero()` returns a data-dependent length, which
CUDA Graph rejects outright rather than merely slowing down.

    python -m benchmark.step_cudagraph --lut /root/autodl-tmp/assets/lut/lut-60.pt --res 1024
"""

import argparse
import math
import time

import torch

from diffusers.models.attention import JointTransformerBlock

DIM, HEADS, HEAD_DIM, CTX, LAYERS = 1536, 24, 64, 333, 24


class Router(torch.nn.Module):
    """The scoring head, at the paper's shape: a bottleneck MLP down to one logit."""

    def __init__(self, dim=DIM, bottleneck=64):
        super().__init__()
        self.proj = torch.nn.Linear(dim, bottleneck)
        self.norm = torch.nn.LayerNorm(bottleneck)
        self.out = torch.nn.Linear(bottleneck, 1)

    def forward(self, x):
        return self.out(self.norm(torch.nn.functional.silu(self.proj(x)))).squeeze(-1)


def k_schedule(lut_path, steps, N):
    lut = torch.load(lut_path, map_location="cpu")
    rc = torch.as_tensor(lut["ratio"]).float()
    T = min(steps, rc.shape[1])

    def kof(v):
        return max(8, min(int(math.ceil(min(max(float(v), 0.01), 1.0) * N)), N))

    return [
        [kof(rc[layer_idx, step_idx]) for layer_idx in range(LAYERS)]
        for step_idx in range(T)
    ]


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--lut", default="/root/autodl-tmp/assets/lut/lut-60.pt")
    p.add_argument(
        "--res",
        type=int,
        default=1024,
        help="SD3 image side; token count is (res/16)^2",
    )
    p.add_argument("--steps", type=int, default=28)
    p.add_argument("--repeats", type=int, default=10)
    args = p.parse_args()

    device, dtype = "cuda", torch.float16
    N = (args.res // 16) ** 2  # SD3 packs 2x2 latent patches into one token
    sched = k_schedule(args.lut, args.steps, N)
    steps = len(sched)

    block = (
        JointTransformerBlock(
            dim=DIM, num_attention_heads=HEADS, attention_head_dim=HEAD_DIM
        )
        .to(device, dtype)
        .eval()
    )
    router = Router().to(device, dtype).eval()
    ctx = torch.randn(1, CTX, DIM, device=device, dtype=dtype)
    temb = torch.randn(1, DIM, device=device, dtype=dtype)
    # one persistent activation buffer per step, written in place by the scatter
    xs = [torch.randn(1, N, DIM, device=device, dtype=dtype) for _ in range(steps)]

    print(
        f"SD3 {args.res}^2 -> N={N} tokens, {LAYERS} layers x {steps} steps "
        f"= {LAYERS * steps} pruned block calls"
    )
    print(f"k range {min(min(r) for r in sched)}..{max(max(r) for r in sched)}")

    def one_step(t):
        """Score, select, gather, run the block, scatter back. All shapes static."""
        x = xs[t]
        for layer_idx in range(LAYERS):
            k = sched[t][layer_idx]
            scores = router(x)
            idx = torch.topk(scores, k, dim=1).indices
            gid = idx.unsqueeze(-1).expand(-1, -1, DIM)
            sel = torch.gather(x, 1, gid)
            out = block(hidden_states=sel, encoder_hidden_states=ctx, temb=temb)[1]
            x = x.scatter(1, gid, out)
        return x

    @torch.no_grad()
    def eager():
        for t in range(steps):
            one_step(t)

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

    print("\ncapturing one graph per step ...", flush=True)
    t0 = time.perf_counter()
    graphs = []
    with torch.no_grad():
        for t in range(steps):
            s = torch.cuda.Stream()
            s.wait_stream(torch.cuda.current_stream())
            with torch.cuda.stream(s):
                for _ in range(2):
                    one_step(t)
            torch.cuda.current_stream().wait_stream(s)
            g = torch.cuda.CUDAGraph()
            with torch.cuda.graph(g):
                one_step(t)
            graphs.append(g)
    cap_s = time.perf_counter() - t0

    def graphed():
        for g in graphs:
            g.replay()

    e = timed(eager, args.repeats)
    gm = timed(graphed, args.repeats)

    print(f"\n{'=' * 58}")
    print(f"{'eager':<26}{e:9.2f} ms")
    print(f"{'captured (one per step)':<26}{gm:9.2f} ms   speedup {e / gm:.3f}x")
    print(f"{'=' * 58}")
    print(
        f"capture cost {cap_s:.1f}s for {steps} graphs, paid once and replayed "
        f"for every image"
    )


if __name__ == "__main__":
    main()
