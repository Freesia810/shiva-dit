"""Render Figure 2 from the camera-ready FID and latency points."""

import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.font_manager as fm
import matplotlib.pyplot as plt
from matplotlib.markers import MarkerStyle
from matplotlib.patches import ConnectionPatch
import pandas as pd
from PIL import Image

DATA_PATH = Path(__file__).with_name("fig2_pareto.csv")
DPI = 400


FAMILY_COLORS = {'DiffCR': '#9467bd', 'IBTM': '#1f77b4', 'DyDiT': '#8c564b'}
DOT_AREA = 36
STAR_DIAMETER_SCALE = 1.35


def same_diameter_area(marker):
    style = MarkerStyle(marker)
    bounds = style.get_path().transformed(style.get_transform()).get_extents()
    return DOT_AREA / max(bounds.width, bounds.height)**2


def render_figure(output_dir, font=None):
    output_dir.mkdir(parents=True, exist_ok=True)
    font_family = "STIXGeneral"
    if font is not None:
        fm.fontManager.addfont(font)
        font_family = fm.FontProperties(fname=font).get_name()
    df = pd.read_csv(DATA_PATH)
    original = df[df.Label == 1]
    if len(df) != 21 or len(original) != 15:
        raise ValueError("Expected the 21 camera-ready operating points")
    width, height = 1255, 810
    plt.rcParams.update({'font.family': font_family, 'font.size': 8,
        'axes.labelsize': 8, 'xtick.labelsize': 7, 'ytick.labelsize': 7,
        'axes.linewidth': 1.0, 'pdf.fonttype': 42, 'ps.fonttype': 42})
    offset_map = {
        'Vanilla': (4, 0, 'left', 'center'), 'Finetuned': (5, -2.5, 'left', 'top'),
        'ToMeSD': (-4, 0, 'right', 'center'), 'ToFu': (4, 0, 'left', 'center'),
        'SDTM': (0, -6, 'center', 'top'), 'ToMA': (4, 0, 'left', 'center'),
        'IBTM': (4, 0, 'left', 'center'), 'DiffCR': (0, 5, 'center', 'bottom'),
        'DyDiT': (-4, 4, 'right', 'bottom'),
        'Shiva-80%': (0, -10, 'right', 'bottom'),
        'Shiva-60%': (0, 5, 'center', 'bottom'),
        'Shiva-70%': (0, 5, 'center', 'bottom'),
        'Shiva-90%': (0, 5, 'left', 'bottom'),
    }
    fig, (upper, lower) = plt.subplots(2, 1, sharex=True,
        figsize=(width / DPI, height / DPI),
        gridspec_kw={'height_ratios': [1, 6], 'hspace': .085})
    fig.subplots_adjust(left=.13, bottom=.18, right=.98, top=.97)
    for ax in (upper, lower):
        ax.patch.set_alpha(0)
        baselines = df[(df.Type == 'Baseline') &
            ~df.Method.str.startswith(tuple(FAMILY_COLORS))]
        teacher = df[df.Type == 'Teacher']
        ours = df[df.Type == 'Ours']
        ax.scatter(baselines.Latency_ms, baselines.FID, color='#7f7f7f',
            s=DOT_AREA, alpha=1, facecolors='#999999', marker='o', edgecolors='white', linewidth=.5, zorder=2)
        ax.scatter(teacher.Latency_ms, teacher.FID, color='#2ca02c',
            s=same_diameter_area('D'), marker='D', edgecolors='white', linewidth=.5, zorder=3)
        ax.scatter(ours.Latency_ms, ours.FID, color='#d62728',
            s=same_diameter_area('*') * STAR_DIAMETER_SCALE**2,
            marker='*', edgecolors='white', linewidth=.5, zorder=10)
        pareto = pd.concat([ours, teacher]).sort_values('Latency_ms', ascending=False)
        ax.plot(pareto.Latency_ms, pareto.FID, color='#d62728',
            linestyle='--', linewidth=1.0, alpha=.5, zorder=1)
        ax.axhline(teacher.FID.iloc[0], color='#2ca02c', linestyle=':', linewidth=1.0, alpha=.4, zorder=0)
        for family in ('DyDiT', 'IBTM', 'DiffCR'):
            points = df[df.Method.str.startswith(family)].sort_values('Latency_ms')
            assert len(points) == 3
            ax.scatter(points.Latency_ms, points.FID, color=FAMILY_COLORS[family],
                s=DOT_AREA, alpha=1, marker='o', edgecolors='white', linewidth=.5, zorder=2)
            # The high-to-middle DiffCR pair is drawn once as a straight screen segment.
            line_points = points[points.FID < 124] if family == 'DiffCR' else points
            ax.plot(line_points.Latency_ms, line_points.FID, color=FAMILY_COLORS[family], linestyle='--',
                linewidth=1.0, alpha=.8, zorder=1.5)
        ax.tick_params(axis='both', which='major', pad=2)
        ax.grid(True, linestyle='--', alpha=.4, zorder=0)
        ax.set_xlim(4500, 8300)
    lower.set_ylim(5, 72)
    lower.set_yticks(range(20, 71, 20))
    upper.set_ylim(124, 141)
    upper.set_yticks([140])
    upper.spines['bottom'].set_visible(False)
    upper.tick_params(axis='x', which='both', bottom=False, labelbottom=False)
    lower.spines['top'].set_visible(False)
    lower.set_xticks(range(4500, 8001, 500))
    lower.set_xlabel('Inference Latency (ms)', fontweight='normal', labelpad=2)
    fig.text(.017, .50, 'FID', rotation=90, va='center', fontweight='normal')
    # A single straight segment between the two dot centers, without an elbow.
    diffcr = df[df.Method.str.startswith('DiffCR')].sort_values('FID', ascending=False)
    a, b = diffcr.iloc[0], diffcr.iloc[1]
    bridge = ConnectionPatch(xyA=(a.Latency_ms, a.FID), coordsA=upper.transData,
        xyB=(b.Latency_ms, b.FID), coordsB=lower.transData, arrowstyle='-',
        connectionstyle='arc3,rad=0', color=FAMILY_COLORS['DiffCR'],
        alpha=.8, linewidth=1, linestyle='--', clip_on=False,
        shrinkA=0, shrinkB=0, zorder=-1)
    fig.add_artist(bridge)
    for ax, ypos in ((upper, 0), (lower, 1)):
        for xpos in (0, 1):
            ax.plot([xpos-.009, xpos+.009], [ypos-.018, ypos+.018],
                transform=ax.transAxes, color='black', linewidth=1, clip_on=False, zorder=20)
    offset_map.update({
        'Shiva-90%': (-8, 5, 'left', 'bottom'),
        'SDTM': (-2, 3, 'right', 'bottom'),
        'SparseDiT': (-4, 2, 'right', 'center'),
        'NeuralSort': (4, 2, 'left', 'bottom'),
        'ToMeSD': (4, 0, 'left', 'center'),
    })
    family_labels = {
        'DiffCR': (5880, 51, 'center'),
        'IBTM': (6940, 40, 'center'),
        'DyDiT': (6500, 9, 'center'),
    }
    annotations = []
    for row in original.itertuples():
        if row.Method in FAMILY_COLORS:
            x, y, ha = family_labels[row.Method]
            annotation = lower.text(x, y, row.Method, fontsize=6.5, ha=ha,
                va='center', fontweight='normal', color=FAMILY_COLORS[row.Method], zorder=15)
            annotations.append((row.Method, annotation))
            continue
        dx, dy, ha, va = offset_map.get(row.Method, (3, 3, 'left', 'bottom'))
        annotation = lower.annotate(row.Method, (row.Latency_ms, row.FID), xytext=(dx, dy),
            textcoords='offset points', fontsize=6.5, ha=ha, va=va, fontweight='normal',
            color='#b01e1f' if 'Shiva' in row.Method else 'black', zorder=15)
        annotations.append((row.Method, annotation))
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    overlaps = [(na, nb) for i, (na, aa) in enumerate(annotations)
        for nb, ab in annotations[i+1:] if aa.get_window_extent(renderer).overlaps(ab.get_window_extent(renderer))]
    print('Label overlaps:', overlaps)
    fig.savefig(output_dir / 'fig2-display.png', dpi=DPI, bbox_inches=None, pad_inches=0)
    fig.savefig(output_dir / 'fig2-display.pdf', bbox_inches=None, pad_inches=0)
    with Image.open(output_dir / 'fig2-display.png') as rendered:
        assert rendered.size == (width, height)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=Path("runs/figures"))
    parser.add_argument("--font", type=Path, help="Optional Times New Roman font file")
    args = parser.parse_args()
    render_figure(args.output_dir, args.font)


if __name__ == "__main__":
    main()
