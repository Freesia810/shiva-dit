"""Sweep latency / throughput over (config, resolution, batch, compiled) and emit a
flat CSV. Wraps benchmark.infer_phase_profile so every cell is measured by the exact
same instrumented path; this file only orchestrates and parses.

Every raw stdout block is also appended to results/latency_4090/<tag>_raw.log so no
number is lost if a ratio has to be recomputed later.

    python -m benchmark.latency_matrix --tag batch  \
        --configs vanilla shiva60 dydit --batches 1 2 4 8
    python -m benchmark.latency_matrix --tag res \
        --configs vanilla shiva60 --resolutions 512 1024 2048
"""

import argparse
import csv
import os
import re
import subprocess
import sys

ASSETS = "/root/autodl-tmp/assets"
CONFIGS = {
    "vanilla": [],
    "shiva90": [
        "--ckpt_path",
        f"{ASSETS}/models/shiva-sd3-80.pt",
        "--gater_lut",
        f"{ASSETS}/lut/lut-80_90.pt",
    ],
    "shiva80": [
        "--ckpt_path",
        f"{ASSETS}/models/shiva-sd3-80.pt",
        "--gater_lut",
        f"{ASSETS}/lut/lut-80.pt",
    ],
    "shiva70": [
        "--ckpt_path",
        f"{ASSETS}/models/shiva-sd3-60.pt",
        "--gater_lut",
        f"{ASSETS}/lut/lut-60_70.pt",
    ],
    "shiva60": [
        "--ckpt_path",
        f"{ASSETS}/models/shiva-sd3-60.pt",
        "--gater_lut",
        f"{ASSETS}/lut/lut-60.pt",
    ],
    "dydit": ["--ckpt_path", f"{ASSETS}/models/dydit.pt"],
}
OUT = "results/latency_4090"

WALL = re.compile(r"pipeline wall time\s+([\d.]+) ms/call")
PERIMG = re.compile(r"per-image latency\s+([\d.]+) ms/image\s+throughput\s+([\d.]+)")
DYN = re.compile(r"unique_graphs=(\d+)\s+total graph breaks=(\d+)")
OOM = re.compile(r"OutOfMemory|CUDA out of memory")


def run_cell(cfg, res, batch, compiled, steps, runs, warmup):
    cmd = [
        sys.executable,
        "-m",
        "benchmark.infer_phase_profile",
        "--steps",
        str(steps),
        "--runs",
        str(runs),
        "--warmup",
        str(warmup),
        "--label",
        cfg,
        "--height",
        str(res),
        "--width",
        str(res),
        "--batch",
        str(batch),
        *CONFIGS[cfg],
    ]
    if compiled:
        cmd += ["--compile", "default"]
    r = subprocess.run(cmd, capture_output=True, text=True)
    out = r.stdout + "\n" + r.stderr

    os.makedirs(OUT, exist_ok=True)
    with open(os.path.join(OUT, "matrix_raw.log"), "a") as f:
        f.write(
            f"\n@@@@@ {cfg} res={res} batch={batch} compiled={compiled} @@@@@\n{out}\n"
        )

    if OOM.search(out):
        return dict(wall="OOM", per_img="", thr="", graphs="", breaks="")
    w = WALL.search(out)
    pi = PERIMG.search(out)
    dy = DYN.search(out)
    return dict(
        wall=float(w.group(1)) if w else "",
        per_img=float(pi.group(1)) if pi else "",
        thr=float(pi.group(2)) if pi else "",
        graphs=int(dy.group(1)) if dy else "",
        breaks=int(dy.group(2)) if dy else "",
    )


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--tag", required=True, help="names the CSV/log for this sweep")
    p.add_argument("--configs", nargs="+", default=["vanilla", "shiva60", "dydit"])
    p.add_argument("--resolutions", type=int, nargs="+", default=[1024])
    p.add_argument("--batches", type=int, nargs="+", default=[1])
    p.add_argument(
        "--compiled",
        nargs="+",
        default=["0", "1"],
        help="0 = eager, 1 = torch.compile; give both to get both",
    )
    p.add_argument("--steps", type=int, default=50)
    p.add_argument("--runs", type=int, default=3)
    p.add_argument("--warmup", type=int, default=2)
    args = p.parse_args()

    csv_path = os.path.join(OUT, f"{args.tag}.csv")
    os.makedirs(OUT, exist_ok=True)
    rows = []
    done = set()
    if os.path.exists(csv_path):
        for r in csv.DictReader(open(csv_path)):
            rows.append(r)
            done.add((r["config"], r["res"], r["batch"], r["compiled"]))

    for cfg in args.configs:
        for res in args.resolutions:
            for batch in args.batches:
                for comp in args.compiled:
                    key = (cfg, str(res), str(batch), comp)
                    if key in done:
                        print(f"skip {key} (already in csv)")
                        continue
                    print(
                        f"run  {cfg} res={res} batch={batch} compiled={comp}",
                        flush=True,
                    )
                    m = run_cell(
                        cfg, res, batch, comp == "1", args.steps, args.runs, args.warmup
                    )
                    row = dict(
                        config=cfg,
                        res=res,
                        batch=batch,
                        compiled=comp,
                        steps=args.steps,
                        **m,
                    )
                    rows.append(row)
                    with open(csv_path, "w", newline="") as f:
                        w = csv.DictWriter(
                            f,
                            fieldnames=[
                                "config",
                                "res",
                                "batch",
                                "compiled",
                                "steps",
                                "wall",
                                "per_img",
                                "thr",
                                "graphs",
                                "breaks",
                            ],
                        )
                        w.writeheader()
                        w.writerows(rows)
                    print(
                        f"   -> wall={m['wall']} per_img={m['per_img']} "
                        f"thr={m['thr']} graphs={m['graphs']}",
                        flush=True,
                    )

    print(f"\nCSV: {csv_path}")


if __name__ == "__main__":
    main()
