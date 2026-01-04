import os
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
import numpy as np

from shiva_dit.utils.ste import ShivaTopkFunc

class ToyModel(nn.Module):
    def __init__(
        self, N, D, init_k, learn_k=False, init_logits_noise=0.1, ts=5.0, tk=10.0
    ):
        super().__init__()
        self.N = N
        self.D = D
        self.logits = nn.Parameter(torch.randn(1, N) * init_logits_noise)
        self.k_float = nn.Parameter(
            torch.tensor([float(init_k)]), requires_grad=learn_k
        )

        self.register_buffer("Ts", torch.tensor(float(ts)))
        self.register_buffer("Tk", torch.tensor(float(tk)))

    def forward(self, x):
        x_sel, x_rej, idx_sel, idx_rej = ShivaTopkFunc.apply(
            x, self.logits.expand(x.shape[0], -1), self.k_float, self.Ts, self.Tk
        )
        return x_sel, x_rej, idx_sel, idx_rej, self.k_float


def experiment_unified():
    print("\n>>> Running Unified Experiment: Sorting & Budgeting (Dotted Style)...")

    # === 1. 训练逻辑 (保持不变) ===
    N, D = 100, 16
    True_Signal_Count = 20
    Init_K = 50
    Total_Steps = 800
    Warmup_Steps = 100

    model = ToyModel(N, D, init_k=Init_K, learn_k=True, ts=5.0, tk=20.0)
    optimizer_router = optim.Adam([model.logits], lr=0.1)
    optimizer_k = optim.SGD([model.k_float], lr=2.0)

    x = torch.zeros(1, N, D)
    x[:, :True_Signal_Count, :] = 10.0
    torch.manual_seed(42)
    x = x + torch.randn(1, N, D) * 0.1

    k_history = []
    acc_history = []

    print(f"{'Step':<5} | {'K':<6} | {'Acc':<7} | {'Phase'}")
    print("-" * 40)

    for step in range(Total_Steps):
        optimizer_router.zero_grad()
        optimizer_k.zero_grad()

        if step < Warmup_Steps:
            phase = "Warmup"
            update_k = False
        else:
            phase = "Train"
            update_k = True

        x_sel, x_rej, idx_sel, _, current_k = model(x)

        loss_sel = torch.mean((x_sel - 10.0) ** 2)
        loss_rej = torch.mean((x_rej - 0.0) ** 2)
        loss_rec = loss_sel + loss_rej
        sparsity_penalty = 0.1 * current_k
        loss = loss_rec + sparsity_penalty

        loss.backward()
        optimizer_router.step()
        if update_k:
            optimizer_k.step()

        with torch.no_grad():
            model.k_float.clamp_(1.0, float(N))
            is_signal = idx_sel < True_Signal_Count
            recall = is_signal.float().sum() / True_Signal_Count

        k_val = current_k.item()
        k_history.append(k_val)
        acc_history.append(recall.item())

        if step % 100 == 0 or step == Total_Steps - 1:
            print(f"{step:<5} | {k_val:<6.2f} | {recall.item():<7.0%} | {phase}")

    # ==========================================
    #  Plotting: Dotted Style
    # ==========================================
    try:
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

        font_path = "Times New Roman.ttf"
        if os.path.exists(font_path):
            prop = fm.FontProperties(fname=font_path, size=12)
            bold_prop = fm.FontProperties(fname=font_path, size=12, weight="bold")
            title_prop = fm.FontProperties(fname=font_path, size=15)
        else:
            prop = fm.FontProperties(family="serif", size=12)
            bold_prop = fm.FontProperties(family="serif", size=12, weight="bold")
            title_prop = fm.FontProperties(family="serif", size=15)

        fig, ax1 = plt.subplots(figsize=(7, 4.5))

        color_k = "#1F77B4"
        ax1.set_xlabel("Training Steps", fontproperties=prop)
        ax1.set_ylabel("Budget Size ($k$)", color=color_k, fontproperties=prop)
        ax1.tick_params(axis="y", labelcolor=color_k)
        ax1.set_ylim(0, 60)

        (line1,) = ax1.plot(
            k_plot,
            color=color_k,
            linewidth=2.5,
            linestyle="-",
            label=r"Learned Budget $k$",
        )

        line_gt = ax1.axhline(
            y=True_Signal_Count,
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
            acc_plot,
            color=color_acc,
            linewidth=2.0,
            linestyle="-", 
            label="Score Accuracy",
        )

        ax1.axvline(
            x=Warmup_Steps,
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

        ax1.set_title(
            "Joint Learning of Sorting and Budgeting", fontproperties=title_prop, pad=12
        )

        lines = [line1, line2, line_gt]
        labels = [l.get_label() for l in lines]

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
        plt.savefig("exp_unified.pdf", format="pdf", bbox_inches="tight")
        plt.savefig("exp_unified.png", dpi=300, bbox_inches="tight")
        print("[Unified] Saved exp_unified.pdf & .png (Dotted Style)")

    except Exception as e:
        print(f"Plotting failed: {e}")
        import traceback

        traceback.print_exc()

def experiment_3_gradient_consistency():
    print("\n>>> Running Experiment 3: Gradient Sign Consistency (Monte Carlo)...")

    num_trials = 1000
    similarities = []

    ts_tensor = torch.tensor(5.0)
    tk_tensor = torch.tensor(10.0)

    print(f"Sampling {num_trials} trials...")

    for i in range(num_trials):
        N, D = 50, 16
        K = 25

        x = torch.randn(1, N, D, requires_grad=True)
        logits = torch.randn(1, N, requires_grad=True)
        k_tensor = torch.tensor([float(K)])

        x_sel, x_rej, _, _ = ShivaTopkFunc.apply(
            x, logits, k_tensor, ts_tensor, tk_tensor
        )

        loss = x_sel.sum() - x_rej.sum()

        if x.grad is not None:
            x.grad.zero_()
        if logits.grad is not None:
            logits.grad.zero_()

        loss.backward()

        grad_logits_shiva = logits.grad.clone().flatten()
        x_sum_flat = x.sum(dim=-1).detach().flatten()

        cos_sim = torch.nn.functional.cosine_similarity(
            x_sum_flat, grad_logits_shiva, dim=0
        ).item()
        similarities.append(cos_sim)

    sim_data = np.array(similarities)
    mean_sim = np.mean(sim_data)
    std_sim = np.std(sim_data)

    print(f"Done. Mean: {mean_sim:.4f} ± {std_sim:.4f}")

    try:
        font_path = "Times New Roman.ttf"
        if os.path.exists(font_path):
            prop = fm.FontProperties(fname=font_path, size=12)
            has_font = True
            print(f"[Plot] Loaded local font: {font_path}")
        else:
            prop = fm.FontProperties(family="serif", size=12)
            has_font = False
            print(f"[Plot] Local font not found, using system serif.")

        fig, ax = plt.subplots(figsize=(6, 4))

        n, bins, patches = ax.hist(
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

        ax.set_title(
            "Distribution of Gradient Consistency",
            fontproperties=prop,
            fontsize=14,
            pad=12,
        )
        ax.set_xlabel("Cosine Similarity", fontproperties=prop, fontsize=12)
        ax.set_ylabel("Probability Density", fontproperties=prop, fontsize=12)

        for label in ax.get_xticklabels() + ax.get_yticklabels():
            label.set_fontproperties(prop)
            label.set_fontsize(10)

        stats_text = (
            f"Trials: {num_trials}\n" f"Mean: {mean_sim:.3f}\n" f"Std:  {std_sim:.3f}"
        )

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

        plt.savefig("exp3_consistency.pdf", format="pdf", bbox_inches="tight")
        plt.savefig("exp3_consistency.png", dpi=300, bbox_inches="tight")

        print("[Exp 3] Saved exp3_consistency.pdf (Vector) & .png (Preview)")

    except Exception as e:
        print(f"[Exp 3] Plotting failed: {e}")
        import traceback

        traceback.print_exc()


if __name__ == "__main__":
    torch.manual_seed(42)
    experiment_unified()
    experiment_3_gradient_consistency()
