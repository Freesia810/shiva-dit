"""Multi-seed evaluation: run every config under several seeds, report mean +/- std.

Table 1 currently reports single-seed numbers, and the headline gaps (Shiva-90% FID
12.15 vs Finetuned 13.22) are small enough that a reader cannot tell them from
run-to-run noise. This driver produces the error bars.

The "seed" here is NOT torch.manual_seed. The same seed on a different GPU draws a
different initial noise, so a cross-machine comparison would be confounded. Instead
every config is fed the *same* pre-dumped latents (latent_p{idx:06d}_s{j:02d}.pt);
the seed dimension is the sample index j = s00..s0k. Generation therefore happens
ONCE per config with samples_per_prompt = n_seeds, and each latent slice is scored
independently (all p*_s00 form one FID set, all p*_s01 the next, ...) to give the
mean +/- std. This is the whole point of shipping fixed latents.

Scope is deliberately narrow -- the five configs that carry the claim, on a prompt
subset -- not all nine baselines on full MJHQ-30K:

    vanilla, finetuned, shiva90, shiva80, shiva60

Runs are resumable: a config whose images already exist skips generation, and a
slice whose metrics_s{NN}.json exists skips scoring.

    # see the exact commands without running anything
    python -m benchmark.multiseed --configs configs.json \
        --jsonl_path .../eval.jsonl --latent_source_dir .../sd3_latents_dump --dry-run

    # single GPU, 5 latent slices
    python -m benchmark.multiseed --configs configs.json \
        --jsonl_path .../eval.jsonl --latent_source_dir .../sd3_latents_dump --n_seeds 5

    # 8 GPUs for the (single) generation run of each config
    python -m benchmark.multiseed --configs configs.json ... --nproc 8

The prompt jsonl must be the SAME one the latents were dumped from (line number =
p-index): for this dump that is eval.jsonl (2000 lines <-> p000000..p001999).

The configs file is a JSON object mapping a config name to the extra arguments
handed to benchmark.generation, e.g.

    {
      "vanilla":   {"model_type": "sd3"},
      "finetuned": {"model_type": "sd3", "ckpt_path": "ckpt/finetuned.pt"},
      "shiva60":   {"model_type": "sd3", "ckpt_path": "ckpt/shiva-sd3-60.pt",
                    "gater_lut": "lut/lut-60.pt"}
    }
"""

import argparse
import json
import math
import os
import subprocess
import sys

METRIC_ORDER = ["FID", "IR", "CLIP_SCORE", "CLIP_IQA"]
LOWER_IS_BETTER = {"FID"}


def build_generation_cmd(name, cfg, out_dir, args):
    """One generation run per config: samples_per_prompt = n_seeds, fixed latents."""
    if args.nproc > 1:
        cmd = [
            "torchrun",
            f"--nproc_per_node={args.nproc}",
            "-m",
            "benchmark.generation",
        ]
    else:
        cmd = [sys.executable, "-m", "benchmark.generation"]

    cmd += [
        "--jsonl_path",
        args.jsonl_path,
        "--output_dir",
        out_dir,
        "--samples_per_prompt",
        str(args.n_seeds),
        "--skip_metrics",  # scoring happens in a separate, uniform pass
    ]
    if args.latent_source_dir:
        cmd += ["--latent_source_dir", args.latent_source_dir]
    if args.limit:
        cmd += ["--limit", str(args.limit)]

    for key, val in cfg.items():
        flag = f"--{key}"
        if isinstance(val, bool):
            if val:
                cmd.append(flag)
        else:
            cmd += [flag, str(val)]
    return cmd


def build_eval_cmd(out_dir, seed_slice, args):
    cmd = [
        sys.executable,
        "-m",
        "benchmark.evaluate",
        "--image_dir",
        out_dir,
        "--jsonl_path",
        args.jsonl_path,
        "--seed_slice",
        str(seed_slice),
    ]
    if args.fid_stats:
        cmd += ["--fid_stats", args.fid_stats]
    return cmd


def run(cmd, dry_run):
    printable = " ".join(cmd)
    if dry_run:
        print(f"    $ {printable}")
        return True
    print(f"    $ {printable}", flush=True)
    result = subprocess.run(cmd)
    if result.returncode != 0:
        print(f"    !! exited {result.returncode}")
        return False
    return True


def aggregate(per_seed):
    """per_seed: list of metric dicts -> {metric: (mean, std, n)}"""
    out = {}
    keys = set()
    for m in per_seed:
        keys |= {k for k in m if not k.startswith("_")}
    for key in keys:
        vals = [m[key] for m in per_seed if key in m and m[key] is not None]
        vals = [v for v in vals if isinstance(v, (int, float)) and v != -1.0]
        if not vals:
            continue
        mean = sum(vals) / len(vals)
        if len(vals) > 1:
            var = sum((v - mean) ** 2 for v in vals) / (len(vals) - 1)
            std = math.sqrt(var)
        else:
            std = float("nan")
        out[key] = (mean, std, len(vals))
    return out


def report(results, n_seeds):
    ordered = [m for m in METRIC_ORDER if any(m in r for r in results.values())]
    extra = sorted({k for r in results.values() for k in r} - set(ordered))
    cols = ordered + extra

    print(f"\n{'=' * (22 + 20 * len(cols))}")
    print(f"Multi-seed results over {n_seeds} latent slices (s00..s{n_seeds - 1:02d})")
    print(f"{'=' * (22 + 20 * len(cols))}")
    header = f"{'config':<20}" + "".join(f"{c:>20}" for c in cols)
    print(header)
    for name, agg in results.items():
        row = f"{name:<20}"
        for c in cols:
            if c in agg:
                mean, std, n = agg[c]
                cell = f"{mean:.2f}+/-{std:.2f}" if std == std else f"{mean:.2f}"
                row += f"{cell:>20}"
            else:
                row += f"{'-':>20}"
        print(row)
    print(f"{'=' * (22 + 20 * len(cols))}")

    # The specific claim under dispute: does Shiva still beat Finetuned once error
    # bars exist? Every comparison is printed, including the ones that lose -- a
    # claim that quietly stops holding is the whole reason for running this.
    if "finetuned" in results:
        base = results["finetuned"]
        print("\nClaim check vs `finetuned` (separated = gap exceeds both stds):")
        for name, agg in results.items():
            if name == "finetuned":
                continue
            for metric in cols:
                if metric not in agg or metric not in base:
                    continue
                m1, s1, _ = agg[metric]
                m0, s0, _ = base[metric]
                if s1 != s1 or s0 != s0:
                    continue
                better = (m1 < m0) if metric in LOWER_IS_BETTER else (m1 > m0)
                if not better:
                    verdict = "LOST -- no longer better"
                elif abs(m1 - m0) > (s1 + s0):
                    verdict = "SEPARATED"
                else:
                    verdict = "overlaps -- not distinguishable"
                print(
                    f"  {name:<12} {metric:<12} {m1:>8.3f} vs {m0:>8.3f}  -> {verdict}"
                )
        print()


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--configs", required=True, help="JSON: name -> generation args")
    p.add_argument(
        "--n_seeds",
        type=int,
        default=5,
        help="number of fixed-latent slices to score (<= slices in the dump)",
    )
    p.add_argument(
        "--jsonl_path",
        default="../dataset/MJHQ-30K/eval.jsonl",
        help="MUST match the jsonl the latents were dumped from (line = p-index)",
    )
    p.add_argument(
        "--latent_source_dir",
        default=None,
        help="dir of latent_p{idx:06d}_s{j:02d}.pt; omit to fall back to random noise",
    )
    p.add_argument("--fid_stats", default="../dataset/MJHQ-30K/mjhq_30k_stats.npz")
    p.add_argument("--output_root", default="results/multiseed")
    p.add_argument(
        "--limit",
        type=int,
        default=2000,
        help="prompt subset size; the claim does not need full MJHQ-30K",
    )
    p.add_argument("--nproc", type=int, default=1)
    p.add_argument("--dry-run", action="store_true")
    p.add_argument(
        "--skip-generation",
        action="store_true",
        help="only (re)score directories that already hold images",
    )
    args = p.parse_args()

    with open(args.configs) as f:
        configs = json.load(f)

    if not args.latent_source_dir:
        print(
            "WARNING: --latent_source_dir not set. Each slice falls back to seeded "
            "random noise, which is NOT reproducible across machines. The multi-seed "
            "comparison is only valid with the shared fixed latents.\n"
        )

    print(
        f"configs={list(configs)}  n_seeds={args.n_seeds}  "
        f"limit={args.limit}  nproc={args.nproc}"
    )
    print(
        f"generation runs: {len(configs)}   scoring passes: "
        f"{len(configs) * args.n_seeds}\n"
    )

    results = {}
    for name, cfg in configs.items():
        out_dir = os.path.join(args.output_root, name)
        print(f"[{name}] -> {out_dir}")
        if not args.dry_run:
            os.makedirs(out_dir, exist_ok=True)

        # 1) generate ALL slices at once (samples_per_prompt = n_seeds)
        if not args.skip_generation:
            if not run(build_generation_cmd(name, cfg, out_dir, args), args.dry_run):
                continue

        # 2) score each latent slice independently
        per_slice = []
        for j in range(args.n_seeds):
            metrics_path = os.path.join(out_dir, f"metrics_s{j:02d}.json")
            if os.path.exists(metrics_path) and not args.dry_run:
                print(f"    slice s{j:02d}: metrics present, reusing")
                per_slice.append(json.load(open(metrics_path)))
                continue
            if not run(build_eval_cmd(out_dir, j, args), args.dry_run):
                continue
            if not args.dry_run and os.path.exists(metrics_path):
                per_slice.append(json.load(open(metrics_path)))

        if per_slice:
            results[name] = aggregate(per_slice)

    if args.dry_run:
        print("\ndry run: nothing executed.")
        return
    if results:
        report(results, args.n_seeds)
        summary = os.path.join(args.output_root, "summary.json")
        os.makedirs(args.output_root, exist_ok=True)
        with open(summary, "w") as f:
            json.dump(
                {
                    n: {
                        k: {"mean": v[0], "std": v[1], "n": v[2]}
                        for k, v in agg.items()
                    }
                    for n, agg in results.items()
                },
                f,
                indent=2,
            )
        print(f"summary written to {summary}")


if __name__ == "__main__":
    main()
