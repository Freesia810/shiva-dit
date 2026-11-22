"""Render Figure 4(a) from exported validation MSE curves."""

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib import font_manager
import numpy as np
import pandas as pd


def load_curve(path):
    curve = pd.read_csv(path, usecols=["step", "value"])
    if curve.empty or not np.isfinite(curve.to_numpy(dtype=float)).all():
        raise ValueError(f"Expected finite step and value columns in {path}")
    return curve.iloc[8:] if len(curve) > 8 else curve


def render_figure(adaptive_csv, fixed_csv, output, font=None):
    adaptive, fixed = load_curve(adaptive_csv), load_curve(fixed_csv)
    if font is not None:
        font_manager.fontManager.addfont(font)
    font_family = font_manager.FontProperties(fname=font).get_name() if font else "STIXGeneral"
    style = {
        "font.family": font_family, "mathtext.fontset": "stix", "font.size": 14,
        "axes.linewidth": 1.5, "xtick.major.width": 1.5, "ytick.major.width": 1.5,
        "xtick.direction": "in", "ytick.direction": "in", "grid.alpha": 0.3,
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    with plt.rc_context(style):
        fig, ax = plt.subplots(figsize=(6, 4.5))
        ax.plot(fixed.step, fixed.value, label=r"Fixed Ratio ($r=0.6$)",
                color="#d62728", linestyle="--", linewidth=2, alpha=0.8)
        ax.plot(adaptive.step, adaptive.value, label=r"Ours (Adaptive, $\bar{r}=0.6$)",
                color="#1f77b4", linewidth=2.5)
        ax.set_xlabel("Training Steps", fontsize=16)
        ax.set_ylabel("Validation MSE Loss", fontsize=16)
        ax.legend(fontsize=14, frameon=True, fancybox=False, edgecolor="black")
        ax.grid(True, linestyle="--", linewidth=0.5)
        fig.tight_layout()
        fig.savefig(output.with_suffix(".pdf"), dpi=300, bbox_inches="tight")
        fig.savefig(output.with_suffix(".png"), dpi=300, bbox_inches="tight")
        plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--adaptive", required=True, type=Path)
    parser.add_argument("--fixed", required=True, type=Path)
    parser.add_argument("--output", type=Path, default=Path("runs/figures/fig4a"))
    parser.add_argument("--font", type=Path)
    args = parser.parse_args()
    render_figure(args.adaptive, args.fixed, args.output, args.font)


if __name__ == "__main__":
    main()
