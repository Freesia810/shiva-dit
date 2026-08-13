"""Drive generation on a rented GPU box, pull results back, keep its disk bounded.

Runs on the local machine. For each config it generates ALL latent slices remotely
in one run (samples_per_prompt = n_seeds, fed the shared fixed latents), rsyncs the
images down, verifies the copy, deletes the remote copy, then scores every slice
locally. Peak remote usage is one config's batch (~1.5 GB per 1000 PNGs at 1024^2 x
n_seeds) instead of the whole sweep at once -- which matters on a 50 GB data disk.

The "seed" is the latent slice s00..s0k, NOT torch.manual_seed: the same seed draws
different noise on a different GPU, so every config is instead fed identical dumped
latents and compared slice by slice. See benchmark/multiseed.py for the rationale.

Metrics stay local on purpose: the GPU box needs no ImageReward / pytorch-fid /
torchmetrics stack, which is what lets it run a lean environment.

Remote images are only removed after the local copy is verified by file count. Pass
--keep-remote to disable deletion entirely.

    # push code, then run the whole sweep (5 latent slices)
    python -m benchmark.remote_pipeline --configs benchmark/multiseed_configs.example.json \
        --sync-code --n_seeds 5 --nproc 8 \
        --jsonl_path /root/autodl-tmp/assets/MJHQ-jsonl/eval.jsonl \
        --latent_source_dir /root/autodl-tmp/assets/sd3_latents_dump \
        --local_jsonl benchmark/prompts/eval.jsonl

    # generate + fetch only, score later
    python -m benchmark.remote_pipeline --configs ... --no-score
"""

import argparse
import json
import os
import subprocess
import sys

from benchmark.multiseed import aggregate, report

REPO_EXCLUDES = [
    ".git",
    "__pycache__",
    "*.pyc",
    ".DS_Store",
    "results",
    "figures",
    "checkpoints*",
    "*.png",
    "*.pt",
    "*.safetensors",
    "*.npz",
    "logs*",
]


def sh(cmd, check=True, capture=False):
    printable = " ".join(cmd) if isinstance(cmd, list) else cmd
    print(f"    $ {printable}", flush=True)
    r = subprocess.run(
        cmd, shell=isinstance(cmd, str), capture_output=capture, text=True
    )
    if check and r.returncode != 0:
        raise RuntimeError(f"command failed ({r.returncode}): {printable}")
    return r


class Remote:
    def __init__(self, host, port, user, repo, python):
        self.host, self.port, self.user = host, port, user
        self.repo, self.python = repo, python

    @property
    def target(self):
        return f"{self.user}@{self.host}"

    def ssh(self, remote_cmd, check=True, capture=False):
        return sh(
            ["ssh", "-p", str(self.port), self.target, remote_cmd],
            check=check,
            capture=capture,
        )

    def run_in_repo(self, remote_cmd, check=True):
        # login shell so conda/PATH and the HF_* exports in .bashrc apply
        wrapped = f"bash -lc 'cd {self.repo} && {remote_cmd}'"
        return self.ssh(wrapped, check=check)

    def sync_code(self, local_repo):
        excludes = []
        for pat in REPO_EXCLUDES:
            excludes += ["--exclude", pat]
        sh(
            [
                "rsync",
                "-az",
                "--delete-after",
                *excludes,
                "-e",
                f"ssh -p {self.port}",
                f"{local_repo.rstrip('/')}/",
                f"{self.target}:{self.repo}/",
            ]
        )

    def count_images(self, remote_dir):
        r = self.ssh(
            f"bash -lc 'ls {remote_dir}/p*_s*.png 2>/dev/null | wc -l'",
            check=False,
            capture=True,
        )
        try:
            return int((r.stdout or "0").strip().splitlines()[-1])
        except (ValueError, IndexError):
            return 0

    def fetch(self, remote_dir, local_dir):
        os.makedirs(local_dir, exist_ok=True)
        sh(
            [
                "rsync",
                "-az",
                "--partial",
                "-e",
                f"ssh -p {self.port}",
                f"{self.target}:{remote_dir}/",
                f"{local_dir}/",
            ]
        )

    def purge(self, remote_dir):
        self.ssh(f"rm -rf {remote_dir}")


def local_image_count(d):
    if not os.path.isdir(d):
        return 0
    return len([f for f in os.listdir(d) if f.startswith("p") and f.endswith(".png")])


def generation_cmd(cfg, remote_out, args):
    """One remote run per config: all n_seeds latent slices, fixed latents injected."""
    if args.nproc > 1:
        cmd = f"torchrun --nproc_per_node={args.nproc} -m benchmark.generation"
    else:
        cmd = "python -m benchmark.generation"
    cmd += (
        f" --jsonl_path {args.jsonl_path}"
        f" --output_dir {remote_out}"
        f" --samples_per_prompt {args.n_seeds} --skip_metrics"
    )
    if args.latent_source_dir:
        cmd += f" --latent_source_dir {args.latent_source_dir}"
    if args.limit:
        cmd += f" --limit {args.limit}"
    for key, val in cfg.items():
        if isinstance(val, bool):
            if val:
                cmd += f" --{key}"
        else:
            cmd += f" --{key} {val}"
    return cmd


def slice_metrics_path(local_dir, j):
    return os.path.join(local_dir, f"metrics_s{j:02d}.json")


def score_local_slices(local_dir, args):
    """Score each latent slice independently -> list of metric dicts."""
    per_slice = []
    for j in range(args.n_seeds):
        mpath = slice_metrics_path(local_dir, j)
        if not os.path.exists(mpath):
            cmd = [
                sys.executable,
                "-m",
                "benchmark.evaluate",
                "--image_dir",
                local_dir,
                "--jsonl_path",
                args.local_jsonl,
                "--seed_slice",
                str(j),
            ]
            if args.fid_stats:
                cmd += ["--fid_stats", args.fid_stats]
            sh(cmd, check=False)
        if os.path.exists(mpath):
            per_slice.append(json.load(open(mpath)))
    return per_slice


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--configs", required=True)
    p.add_argument(
        "--n_seeds",
        type=int,
        default=5,
        help="number of fixed-latent slices to generate and score",
    )
    p.add_argument("--host", default="connect.westd.seetacloud.com")
    p.add_argument("--port", type=int, default=53692)
    p.add_argument("--user", default="root")
    p.add_argument("--repo", default="/root/autodl-tmp/shiva-dit")
    p.add_argument("--python", default="python")
    p.add_argument("--remote-root", default="/root/autodl-tmp/results")
    p.add_argument("--local-root", default="results/multiseed")
    p.add_argument(
        "--jsonl_path",
        default="/root/autodl-tmp/assets/MJHQ-jsonl/eval.jsonl",
        help="prompt file path ON THE REMOTE (must match the latent dump)",
    )
    p.add_argument(
        "--latent_source_dir",
        default="/root/autodl-tmp/assets/sd3_latents_dump",
        help="dir of latent_p{idx:06d}_s{j:02d}.pt ON THE REMOTE",
    )
    p.add_argument(
        "--local_jsonl",
        default="benchmark/prompts/eval.jsonl",
        help="the same prompt file, locally, for scoring",
    )
    p.add_argument("--fid_stats", default="benchmark/prompts/mjhq_30k_stats.npz")
    p.add_argument("--limit", type=int, default=1000)
    p.add_argument("--nproc", type=int, default=1)
    p.add_argument("--sync-code", action="store_true")
    p.add_argument(
        "--keep-remote",
        action="store_true",
        help="do not delete remote batches after fetching",
    )
    p.add_argument(
        "--no-score", action="store_true", help="generate and fetch only; score later"
    )
    p.add_argument("--dry-run", action="store_true")
    args = p.parse_args()

    with open(args.configs) as f:
        configs = json.load(f)

    remote = Remote(args.host, args.port, args.user, args.repo, args.python)

    if args.sync_code and not args.dry_run:
        print("syncing code to remote...")
        remote.ssh(f"mkdir -p {args.repo} {args.remote_root}")
        remote.sync_code(os.path.dirname(os.path.abspath(__file__)) + "/..")

    print(
        f"configs={list(configs)} n_seeds={args.n_seeds} "
        f"-> {len(configs)} generation runs, {len(configs) * args.n_seeds} scoring passes\n"
    )

    results = {}
    for name, cfg in configs.items():
        remote_dir = f"{args.remote_root}/{name}"
        local_dir = os.path.join(args.local_root, name)
        expected = (args.limit or 0) * args.n_seeds
        print(f"[{name}]")

        # fully scored already?
        if all(
            os.path.exists(slice_metrics_path(local_dir, j))
            for j in range(args.n_seeds)
        ):
            print("    all slices scored locally, skipping")
            results[name] = aggregate(
                [
                    json.load(open(slice_metrics_path(local_dir, j)))
                    for j in range(args.n_seeds)
                ]
            )
            continue

        if args.dry_run:
            print(f"    $ (remote) {generation_cmd(cfg, remote_dir, args)}")
            print(f"    $ rsync {remote_dir} -> {local_dir}")
            print(f"    $ (remote) rm -rf {remote_dir}")
            for j in range(args.n_seeds):
                print(f"    $ evaluate --image_dir {local_dir} --seed_slice {j}")
            continue

        have_local = local_image_count(local_dir)
        if have_local < expected:
            remote.run_in_repo(f"mkdir -p {remote_dir}")
            remote.run_in_repo(generation_cmd(cfg, remote_dir, args))

            n_remote = remote.count_images(remote_dir)
            print(f"    remote produced {n_remote} images")
            remote.fetch(remote_dir, local_dir)

            n_local = local_image_count(local_dir)
            print(f"    fetched {n_local} images locally")

            # never destroy the only copy: delete remotely only on a match
            if args.keep_remote:
                print("    --keep-remote set, leaving remote batch in place")
            elif n_remote > 0 and n_local >= n_remote:
                remote.purge(remote_dir)
                print("    remote batch purged")
            else:
                print(
                    f"    !! transfer incomplete ({n_local}/{n_remote}), "
                    f"NOT deleting remote copy"
                )
        else:
            print(f"    {have_local} images already local, skipping generation")

        if not args.no_score:
            per_slice = score_local_slices(local_dir, args)
            if per_slice:
                results[name] = aggregate(per_slice)

    if results:
        report(results, args.n_seeds)


if __name__ == "__main__":
    main()
