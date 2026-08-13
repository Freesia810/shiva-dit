"""Turn the ablation training runs into one Table-2-style comparison.

Table 2's two columns come from validation, not from a standalone evaluator:

    MSE  = validation/distill_loss        trainer.py:1248, F.mse_loss(student, teacher)
    Sim  = validation/feature_cosine_sim  trainer.py:1245, cosine over the feature dim

Checkpoint selection, however, uses a *different* quantity -- `val_main_loss`
(trainer.py:1927, lower is better). So "the number at the saved checkpoint" and
"the best MSE this run ever reached" need not agree. All three readings are printed
rather than one, so a reporting convention can be chosen deliberately instead of by
accident; if they disagree materially, that itself is worth knowing before the table
goes into the paper.

    python -m scripts.utils.ablation_table --logs "logs_ablation_*"
"""

import argparse
import glob
import os

from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

TAG_MSE = "validation/distill_loss"
TAG_SIM = "validation/feature_cosine_sim"
TAG_SEL = "validation/main_loss"

# printed alongside each arm so a capacity confound cannot hide (SD3.5-M, G=12)
ARM_PARAMS = {
    "full": "3.99M",
    "no_time": "2.81M",
    "no_prompt": "2.41M",
    "no_layer": "3.94M",
    "layer_only": "1.23M",
    "no_ctx": "1.18M",
    "no_ctx_matched": "3.99M",
    "gumbel_tau05": "3.99M",
    "gumbel_tau1": "3.99M",
    "gumbel_tau2": "3.99M",
}
GROUP_A = [
    "full",
    "no_time",
    "no_prompt",
    "no_layer",
    "layer_only",
    "no_ctx",
    "no_ctx_matched",
]
GROUP_B = ["gumbel_tau05", "gumbel_tau1", "gumbel_tau2"]


def read_run(log_dir):
    files = (
        sorted(os.path.join(log_dir, f) for f in os.listdir(log_dir) if "tfevents" in f)
        if os.path.isdir(log_dir)
        else []
    )
    if not files:
        return None

    series = {TAG_MSE: {}, TAG_SIM: {}, TAG_SEL: {}}
    for path in files:
        try:
            ea = EventAccumulator(path, size_guidance={"scalars": 0})
            ea.Reload()
        except Exception as e:
            print(f"  ! {os.path.basename(path)}: {e}")
            continue
        available = ea.Tags()["scalars"]
        for tag in series:
            if tag in available:
                for s in ea.Scalars(tag):
                    series[tag][s.step] = s.value

    if not series[TAG_MSE]:
        return None

    mse, sim, sel = series[TAG_MSE], series[TAG_SIM], series[TAG_SEL]

    def at(step):
        return mse.get(step), sim.get(step)

    out = {"steps": max(mse), "n_val": len(mse)}

    # reading 1: at the step the checkpoint selector would have picked
    if sel:
        best_sel_step = min(sel, key=sel.get)
        out["sel_step"] = best_sel_step
        out["sel_mse"], out["sel_sim"] = at(best_sel_step)

    # reading 2: this run's best MSE, wherever it occurred
    best_mse_step = min(mse, key=mse.get)
    out["best_step"] = best_mse_step
    out["best_mse"], out["best_sim"] = at(best_mse_step)

    # reading 3: end of training
    last = max(mse)
    out["final_mse"], out["final_sim"] = at(last)
    return out


def fmt(v, nd=3):
    return "-" if v is None else f"{v:.{nd}f}"


def main():
    p = argparse.ArgumentParser()
    p.add_argument(
        "--logs",
        default="logs_ablation_*",
        help="glob of tensorboard dirs, one per arm",
    )
    p.add_argument("--prefix", default="logs_ablation_")
    args = p.parse_args()

    runs = {}
    for d in sorted(glob.glob(args.logs)):
        arm = os.path.basename(d.rstrip("/"))
        if arm.startswith(args.prefix):
            arm = arm[len(args.prefix) :]
        data = read_run(d)
        if data is None:
            print(f"  ! {d}: no `{TAG_MSE}` scalars found, skipped")
            continue
        runs[arm] = data

    if not runs:
        raise SystemExit(f"no usable runs matched {args.logs}")

    header = (
        f"{'arm':<16}{'params':>8}{'steps':>7}"
        f"{'MSE@sel':>10}{'Sim@sel':>9}"
        f"{'MSE best':>10}{'Sim@best':>10}"
        f"{'MSE final':>11}{'Sim final':>11}"
    )

    def block(title, arms, note=None):
        present = [a for a in arms if a in runs]
        if not present:
            return
        print(f"\n{title}")
        if note:
            print(f"  {note}")
        print(header)
        print("-" * len(header))
        for a in present:
            r = runs[a]
            print(
                f"{a:<16}{ARM_PARAMS.get(a, '?'):>8}{r['steps']:>7}"
                f"{fmt(r.get('sel_mse')):>10}{fmt(r.get('sel_sim')):>9}"
                f"{fmt(r.get('best_mse')):>10}{fmt(r.get('best_sim')):>10}"
                f"{fmt(r.get('final_mse')):>11}{fmt(r.get('final_sim')):>11}"
            )

    print("=" * len(header))
    print("Stage-1 ablation, Table-2 protocol (r=0.6, backbone frozen)")
    print(
        "MSE = validation/distill_loss (lower better) | Sim = feature cosine (higher better)"
    )
    print("@sel = at the step chosen by val_main_loss, i.e. the saved checkpoint")
    print("=" * len(header))

    block(
        "Group A -- context modulation h_ctx (Eq. 8)",
        GROUP_A,
        "no_ctx drops 71% of router params; no_ctx_matched (d'=216) is the "
        "capacity-matched control",
    )
    block(
        "Group B -- selection-probability surrogate pi_i (Eq. 6)",
        GROUP_B,
        "forward is bit-identical to `full`; only the backward surrogate differs",
    )

    leftover = [a for a in runs if a not in GROUP_A + GROUP_B]
    block("other runs", leftover)

    # flag disagreement between the reporting conventions
    print()
    for arm, r in runs.items():
        if r.get("sel_mse") is None or r.get("best_mse") is None:
            continue
        gap = r["sel_mse"] - r["best_mse"]
        if r["best_mse"] and abs(gap) > 0.05 * abs(r["best_mse"]):
            print(
                f"  note: {arm} -- MSE at the selected checkpoint ({r['sel_mse']:.3f}) "
                f"differs from its best ({r['best_mse']:.3f}); "
                f"pick one convention for the whole table"
            )


if __name__ == "__main__":
    main()
