"""Pure CUDA Graph (no Inductor) on the full vanilla DiT forward.

`torch.compile(mode='reduce-overhead')` mixes Inductor fusion AND cuda graphs, so its
speedup cannot be attributed. This captures the real SD3 transformer.forward by hand
-- one manual CUDA graph, no Inductor -- and replays it against eager at the same
inputs. It isolates what CUDA Graph alone buys on this workload.

Run:  python -m benchmark.vanilla_pure_graph --steps 4 --repeats 30
"""

import argparse
import time

import torch

from shiva_dit.utils.inference_utils import load_pipeline_for_inference


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--repeats", type=int, default=30)
    p.add_argument("--res", type=int, default=1024)
    args = p.parse_args()

    device, dtype = "cuda", torch.float16
    pipe = load_pipeline_for_inference(
        model_type="sd3",
        device=device,
        dtype=dtype,
        base_model_path=None,
        optimize_hooks=True,
        merge_lora=True,
    )
    tf = pipe.transformer

    grabbed = {}

    def grab(m, a, kw):
        if not grabbed:
            grabbed["a"] = a
            grabbed["kw"] = kw
        return None

    h = tf.register_forward_pre_hook(grab, with_kwargs=True)
    with torch.no_grad():
        g0 = torch.Generator(device=device).manual_seed(0)
        pipe(
            prompt="a cat",
            num_inference_steps=2,
            height=args.res,
            width=args.res,
            generator=g0,
            output_type="latent",
        )
    h.remove()

    # freeze the grabbed inputs into persistent buffers the graph can read/write
    def clone(o):
        if torch.is_tensor(o):
            return o.clone()
        if isinstance(o, (list, tuple)):
            return type(o)(clone(x) for x in o)
        if isinstance(o, dict):
            return {k: clone(v) for k, v in o.items()}
        return o

    a = clone(grabbed["a"])
    kw = clone(grabbed["kw"])

    @torch.no_grad()
    def call():
        return tf(*a, **kw)

    @torch.no_grad()
    def timed(fn, reps, warmup=10):
        for _ in range(warmup):
            fn()
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        for _ in range(reps):
            fn()
        torch.cuda.synchronize()
        return (time.perf_counter() - t0) / reps * 1e3

    eager_ms = timed(call, args.repeats)

    # manual capture of the whole transformer.forward -- pure CUDA graph, no Inductor
    s = torch.cuda.Stream()
    s.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(s), torch.no_grad():
        for _ in range(3):
            call()
    torch.cuda.current_stream().wait_stream(s)

    graph = torch.cuda.CUDAGraph()
    with torch.no_grad(), torch.cuda.graph(graph):
        call()

    def replay():
        graph.replay()

    graph_ms = timed(replay, args.repeats)

    print(f"\n{'=' * 60}")
    print(
        f"vanilla full DiT forward ({args.res}^2, one call), median of {args.repeats}"
    )
    print(f"{'=' * 60}")
    print(f"eager                  {eager_ms:9.3f} ms")
    print(
        f"pure CUDA graph        {graph_ms:9.3f} ms   speedup {eager_ms / graph_ms:.3f}x"
    )
    print("(no Inductor here -- this is CUDA Graph alone)")


if __name__ == "__main__":
    main()
