"""What INT8 can actually buy on this GPU, measured at the shapes SD3 runs.

The library route was a dead end: optimum-quanto's CUDA path dequantises rather than
running int8 kernels, so it measures slower than fp16, and torchao's current release
needs a newer torch than this environment has. Neither tells us anything about INT8
itself.

`torch._int_mm` is the int8 tensor-core matmul underneath every such library, so timing
it against the fp16 matmul at SD3's own shapes gives the ceiling any INT8 deployment is
working toward -- tool-independent, and honest about being a kernel-level number rather
than an end-to-end one.

Shapes are the Linear layers inside a JointTransformerBlock at the token counts Shiva
retains, so the weighting reflects where the block's time actually goes.

    python -m benchmark.int8_matmul
"""

import argparse

import torch

DIM = 1536


def median_ms(fn, repeats=200, warmup=20):
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
    p.add_argument("--tokens", type=int, nargs="+", default=[4096, 3277, 2458, 2253])
    p.add_argument("--repeats", type=int, default=200)
    args = p.parse_args()

    dev = torch.device("cuda")
    print(f"{torch.cuda.get_device_name(0)}, torch {torch.__version__}")

    # (name, K, N) for the Linears that dominate a block
    layers = [
        ("qkv proj   (1536->4608)", DIM, DIM * 3),
        ("attn out   (1536->1536)", DIM, DIM),
        ("ffn up     (1536->6144)", DIM, DIM * 4),
        ("ffn down   (6144->1536)", DIM * 4, DIM),
    ]

    print(f"\n{'layer':<26}{'tokens':>8}{'fp16 ms':>10}{'int8 ms':>10}{'speedup':>10}")
    print("-" * 64)
    for name, K, N in layers:
        for n in args.tokens:
            a16 = torch.randn(n, K, device=dev, dtype=torch.float16)
            b16 = torch.randn(K, N, device=dev, dtype=torch.float16)
            a8 = torch.randint(-127, 127, (n, K), device=dev, dtype=torch.int8)
            b8 = torch.randint(-127, 127, (K, N), device=dev, dtype=torch.int8)

            t16 = median_ms(lambda: a16 @ b16, args.repeats)
            t8 = median_ms(lambda: torch._int_mm(a8, b8), args.repeats)
            print(f"{name:<26}{n:>8}{t16:>10.4f}{t8:>10.4f}{t16 / t8:>9.2f}x")
        print()


if __name__ == "__main__":
    main()
