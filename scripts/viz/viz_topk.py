"""Render the archived plotting routines for Figure 5(a) and (c)."""

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
import numpy as np
import pandas as pd


def font_properties(font, **kwargs):
    if font is not None:
        return fm.FontProperties(fname=font, **kwargs)
    return fm.FontProperties(family="serif", **kwargs)


def render_dynamics(csv_path, output_dir, font):
    steps, k_history, acc_history = pd.read_csv(
        csv_path, usecols=["step", "k", "accuracy"]
    )[["step", "k", "accuracy"]].to_numpy(dtype=float).T
    if steps.size == 0 or not np.isfinite([steps, k_history, acc_history]).all():
        raise ValueError("Expected finite step, k and accuracy columns")
    plt.rcParams["mathtext.fontset"] = "stix"
    plt.rcParams["font.family"] = "serif"

    def smooth_curve(points, factor=0.9):
        smoothed_points = []
        for point in points:
            if smoothed_points:
                previous = smoothed_points[-1]
                smoothed_points.append(previous * factor + point * (1 - factor))
            else:
                smoothed_points.append(point)
        return smoothed_points

    k_plot = smooth_curve(k_history, factor=0.92)
    acc_plot = smooth_curve(acc_history, factor=0.85)

    prop = font_properties(font, size=12)
    bold_prop = font_properties(font, size=12, weight="bold")

    fig, ax1 = plt.subplots(figsize=(7, 4.5))

    color_k = "#1F77B4"
    ax1.set_xlabel("Training Steps", fontproperties=prop)
    ax1.set_ylabel("Budget Size ($k$)", color=color_k, fontproperties=prop)
    ax1.tick_params(axis="y", labelcolor=color_k)
    ax1.set_ylim(0, 60)

    (line1,) = ax1.plot(
        steps, k_plot,
        color=color_k,
        linewidth=2.5,
        linestyle="-",
        label=r"Learned Budget $k$",
    )

    line_gt = ax1.axhline(
        y=20,
        color="#D62728",
        linewidth=1.5,
        linestyle=":",
        alpha=1.0,
        label=r"True Sparsity $k^*$",
    )

    ax2 = ax1.twinx()
    color_acc = "#FF7F0E"
    ax2.set_ylabel("Score Accuracy", color=color_acc, fontproperties=prop)
    ax2.tick_params(axis="y", labelcolor=color_acc)
    ax2.set_ylim(0, 1.1)

    (line2,) = ax2.plot(
        steps, acc_plot,
        color=color_acc,
        linewidth=2.0,
        linestyle="-",
        label="Score Accuracy",
    )

    ax1.axvline(
        x=100,
        color="gray",
        linestyle=":",
        linewidth=1.5,
        alpha=0.6,
    )

    ax1.text(
        30,
        2,
        "Warmup",
        ha="center",
        va="bottom",
        color="gray",
        fontproperties=bold_prop,
    )

    ax1.text(
        450,
        2,
        "Adaptation",
        ha="center",
        va="bottom",
        color="black",
        fontproperties=bold_prop,
    )

    lines = [line1, line2, line_gt]
    labels = [line.get_label() for line in lines]

    legend = ax1.legend(
        lines,
        labels,
        loc="center right",
        bbox_to_anchor=(0.96, 0.65),
        frameon=True,
        framealpha=0.95,
        edgecolor="#CCCCCC",
    )
    for text in legend.get_texts():
        text.set_font_properties(prop)

    for ax in [ax1, ax2]:
        for label in ax.get_xticklabels():
            label.set_fontproperties(prop)
            label.set_fontsize(10)
        for label in ax.get_yticklabels():
            label.set_fontproperties(prop)
            label.set_fontsize(10)

    ax1.spines["top"].set_visible(False)
    ax2.spines["top"].set_visible(False)

    plt.tight_layout()
    fig.savefig(output_dir / "exp_unified.pdf", format="pdf", bbox_inches="tight")
    fig.savefig(output_dir / "exp_unified.png", dpi=300, bbox_inches="tight")
    plt.close(fig)


def render_consistency(csv_path, output_dir, font):
    sim_data = pd.read_csv(csv_path, usecols=["cosine_similarity"]).cosine_similarity.to_numpy(dtype=float)
    if sim_data.size == 0 or not np.isfinite(sim_data).all():
        raise ValueError("Expected finite cosine similarities")
    prop = font_properties(font, size=12)
    mean_sim, std_sim = np.mean(sim_data), np.std(sim_data)
    fig, ax = plt.subplots(figsize=(6, 4))

    ax.hist(
        sim_data,
        bins=50,
        color="#4E79A7",
        alpha=0.85,
        edgecolor="white",
        linewidth=0.5,
        density=True,
    )

    ax.axvline(
        mean_sim,
        color="#E15759",
        linestyle="--",
        linewidth=2,
        label="Mean Similarity",
    )

    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_linewidth(1.0)
    ax.spines["bottom"].set_linewidth(1.0)

    ax.grid(axis="y", linestyle=":", alpha=0.4, zorder=0)
    ax.set_xlabel("Cosine Similarity", fontproperties=prop, fontsize=12)
    ax.set_ylabel("Probability Density", fontproperties=prop, fontsize=12)

    for label in ax.get_xticklabels() + ax.get_yticklabels():
        label.set_fontproperties(prop)
        label.set_fontsize(10)

    stats_text = f"Trials: {len(sim_data)}\nMean: {mean_sim:.3f}\nStd:  {std_sim:.3f}"

    props = dict(
        boxstyle="round",
        facecolor="white",
        alpha=0.9,
        edgecolor="gray",
        linewidth=0.5,
    )
    ax.text(
        0.05,
        0.95,
        stats_text,
        transform=ax.transAxes,
        verticalalignment="top",
        fontproperties=prop,
        fontsize=11,
        bbox=props,
    )

    plt.tight_layout()

    fig.savefig(output_dir / "exp3_consistency.pdf", format="pdf", bbox_inches="tight")
    fig.savefig(output_dir / "exp3_consistency.png", dpi=300, bbox_inches="tight")

    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dynamics", type=Path, help="CSV with step, k, and accuracy columns")
    parser.add_argument("--consistency", type=Path, help="CSV with a cosine_similarity column")
    parser.add_argument("--output-dir", type=Path, default=Path("runs/figures"))
    parser.add_argument("--font", type=Path)
    args = parser.parse_args()
    if not (args.dynamics or args.consistency):
        parser.error("Provide --dynamics, --consistency, or both")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    if args.dynamics:
        render_dynamics(args.dynamics, args.output_dir, args.font)
    if args.consistency:
        render_consistency(args.consistency, args.output_dir, args.font)


if __name__ == "__main__":
    main()
