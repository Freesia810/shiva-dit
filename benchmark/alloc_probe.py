"""Do data-dependent shapes cause allocator churn that slows inference down?

The reviewer asks for "memory-allocation effects" as part of a latency breakdown --
i.e. does a changing tensor shape make the caching allocator keep calling cudaMalloc
(a synchronising op) instead of reusing cached blocks, and does that show up as
latency.

The allocator caches a block per size. A bounded shape set (Shiva: 61/87 sizes from
the LUT) is seen in full within the first few prompts, after which every allocation
is a cache hit. A threshold budget produces new sizes on every new prompt, so it
keeps missing the cache and calling cudaMalloc.

Measured from a COLD allocator (the earlier version reset stats after warmup, so the
pool was already primed and both looked like zero -- that was the bug). For each of N
different prompts we record the NEW allocator segments cudaMalloc'd during that
prompt and the prompt's wall time.

    python -m benchmark.alloc_probe --configs shiva60 dydit --prompts 8
"""

import argparse
import time

import torch

from shiva_dit.utils.inference_utils import load_pipeline_for_inference

ASSETS = "/root/autodl-tmp/assets"
CFG = {
    "vanilla": {},
    "shiva60": dict(
        ckpt_path=f"{ASSETS}/models/shiva-sd3-60.pt",
        gater_lut_path=f"{ASSETS}/lut/lut-60.pt",
    ),
    "dydit": dict(ckpt_path=f"{ASSETS}/models/dydit.pt"),
}
PROMPTS = [
    "a photograph of an astronaut riding a horse on mars, dramatic lighting",
    "a close-up portrait of an elderly fisherman, weathered face",
    "a minimalist line drawing of a bicycle on plain white background",
    "a dense aerial view of a night city, thousands of lights, rain",
    "a still life of three lemons on a wooden table, muted colours",
    "an intricate stained glass window depicting a phoenix, backlit",
    "a chalkboard covered in handwritten mathematical equations",
    "a single red balloon against an empty grey sky",
    "a bowl of ramen with steam, chopsticks, dark background",
    "a snowy mountain village at dawn, warm cabin lights",
    "a robot barista pouring latte art in a neon cafe",
    "a field of sunflowers under a stormy purple sky",
]


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--configs", nargs="+", default=["shiva60", "dydit"])
    p.add_argument("--prompts", type=int, default=8)
    p.add_argument("--steps", type=int, default=28)
    p.add_argument("--res", type=int, default=1024)
    args = p.parse_args()

    for cfg in args.configs:
        pipe = load_pipeline_for_inference(
            model_type="sd3",
            device="cuda",
            dtype=torch.float16,
            base_model_path=None,
            optimize_hooks=True,
            merge_lora=True,
            **CFG[cfg],
        )

        # COLD start: clear cached blocks so the first prompts actually miss.
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()

        def seg():
            s = torch.cuda.memory_stats()
            return (s.get("segment.all.allocated", 0), s.get("num_alloc_retries", 0))

        print(
            f"\n{'=' * 70}\n{cfg}   {args.prompts} distinct prompts, "
            f"{args.res}^2, {args.steps} steps, COLD allocator\n{'=' * 70}"
        )
        print(f"{'prompt':>7}{'new cudaMalloc':>16}{'retries':>9}{'wall ms':>10}")
        prev_seg, prev_ret = seg()
        new_after_warm = 0
        for i in range(args.prompts):
            torch.cuda.synchronize()
            t0 = time.perf_counter()
            g = torch.Generator(device="cuda").manual_seed(100 + i)
            pipe(
                prompt=PROMPTS[i % len(PROMPTS)],
                num_inference_steps=args.steps,
                height=args.res,
                width=args.res,
                generator=g,
                output_type="latent",
            )
            torch.cuda.synchronize()
            ms = (time.perf_counter() - t0) * 1e3
            s, r = seg()
            new = s - prev_seg
            print(f"{i + 1:>7}{new:>16}{r - prev_ret:>9}{ms:>10.1f}", flush=True)
            if i >= 2:  # after the shape set is largely seen
                new_after_warm += new
            prev_seg, prev_ret = s, r

        print(
            f"  -> new cudaMalloc after prompt 3: {new_after_warm} "
            f"(0 = allocator fully reuses; >0 = still missing on new shapes)"
        )
        del pipe
        torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
