"""Peak MEMORY split into weights vs activations, across resolution.

The total peak is dominated by the ~16 GB of frozen weights (SD3 + T5), which are
identical for every config, so a total-peak comparison hides whatever the pruning
saves. This records the weight baseline right after load, then measures the peak of
the DiT stage (output_type='latent', so the VAE is excluded) and reports the
activation peak = total - weights. That is the number pruning can actually move.

OOM at a given (resolution) is reported rather than crashing the sweep.

    python -m benchmark.mem_probe --configs vanilla shiva60 dydit \
        --resolutions 1024 1536 1792 2048
"""

import argparse

import torch

from shiva_dit.utils.inference_utils import load_pipeline_for_inference

ASSETS = "/root/autodl-tmp/assets"
CFG = {
    "vanilla": {},
    "shiva60": dict(
        ckpt_path=f"{ASSETS}/models/shiva-sd3-60.pt",
        gater_lut_path=f"{ASSETS}/lut/lut-60.pt",
    ),
    "shiva80": dict(
        ckpt_path=f"{ASSETS}/models/shiva-sd3-80.pt",
        gater_lut_path=f"{ASSETS}/lut/lut-80.pt",
    ),
    "dydit": dict(ckpt_path=f"{ASSETS}/models/dydit.pt"),
}
PROMPT = "a photograph of an astronaut riding a horse on mars"


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--configs", nargs="+", default=["vanilla", "shiva60", "dydit"])
    p.add_argument("--resolutions", type=int, nargs="+", default=[1024, 1536, 1792])
    p.add_argument("--steps", type=int, default=8, help="peak is step-independent")
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

        torch.cuda.synchronize()
        weights = torch.cuda.memory_allocated() / 2**30

        print(f"\n{'=' * 60}\n{cfg}   weights baseline {weights:.2f} GiB\n{'=' * 60}")
        print(
            f"{'res':>6}{'tokens':>8}{'total GiB':>11}{'activation':>12}{'status':>8}"
        )
        for res in args.resolutions:
            torch.cuda.empty_cache()
            torch.cuda.reset_peak_memory_stats()
            try:
                g = torch.Generator(device="cuda").manual_seed(42)
                pipe(
                    prompt=PROMPT,
                    num_inference_steps=args.steps,
                    height=res,
                    width=res,
                    generator=g,
                    output_type="latent",
                )
                peak = torch.cuda.max_memory_allocated() / 2**30
                print(
                    f"{res:>6}{(res // 8) ** 2:>8}{peak:>11.2f}"
                    f"{peak - weights:>12.2f}{'ok':>8}",
                    flush=True,
                )
            except torch.OutOfMemoryError:
                print(
                    f"{res:>6}{(res // 8) ** 2:>8}{'-':>11}{'-':>12}{'OOM':>8}",
                    flush=True,
                )
                torch.cuda.empty_cache()

        del pipe
        torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
