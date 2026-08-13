"""Side-by-side comparison sheets across configs at a fixed seed.

fgGA asked for qualitative examples on prompts containing readable text, comparing
Vanilla / Shiva-80% / Shiva-60%, to show that failures are "limited and
understandable". That is a figure, not a metric -- this builds it.

Every config must have been generated from the same prompt jsonl with the same
--seed, so column j of a row is the same sample under a different budget. The
filenames generation.py writes (p{idx:04d}_s{n:02d}.png) make the pairing exact.

Run:
    python -m benchmark.contact_sheet \
        --dirs results/dense/vanilla results/dense/shiva80 results/dense/shiva60 \
        --labels Vanilla Shiva-80% Shiva-60% \
        --jsonl_path benchmark/prompts/dense_text.jsonl \
        --out figures/dense_text \
        --category text
"""

import argparse
import json
import os
import textwrap

from PIL import Image, ImageDraw, ImageFont

CELL = 320  # rendered size of each image cell
PAD = 8
HEADER = 28
CAPTION = 54


def load_prompts(jsonl_path):
    rows = {}
    with open(jsonl_path, "r", encoding="utf-8") as f:
        for i, line in enumerate(f):
            line = line.strip()
            if not line:
                continue
            item = json.loads(line)
            if "text" in item:
                rows[i] = item
    return rows


def load_font(size):
    for path in (
        "/System/Library/Fonts/Supplemental/Arial.ttf",
        "/System/Library/Fonts/Helvetica.ttc",
        "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
    ):
        if os.path.exists(path):
            try:
                return ImageFont.truetype(path, size)
            except OSError:
                continue
    return ImageFont.load_default()


def find_image(directory, gidx, sample=0):
    path = os.path.join(directory, f"p{gidx:04d}_s{sample:02d}.png")
    return path if os.path.exists(path) else None


def build_sheet(dirs, labels, prompts, indices, sample, out_path):
    cols, rows = len(dirs), len(indices)
    width = cols * CELL + (cols + 1) * PAD
    height = HEADER + rows * (CELL + CAPTION) + (rows + 1) * PAD

    sheet = Image.new("RGB", (width, height), "white")
    draw = ImageDraw.Draw(sheet)
    font_h = load_font(17)
    font_c = load_font(13)

    for j, label in enumerate(labels):
        x = PAD + j * (CELL + PAD)
        draw.text(
            (x + CELL // 2, HEADER // 2), label, fill="black", font=font_h, anchor="mm"
        )

    missing = 0
    for i, gidx in enumerate(indices):
        y = HEADER + PAD + i * (CELL + CAPTION + PAD)
        for j, directory in enumerate(dirs):
            x = PAD + j * (CELL + PAD)
            path = find_image(directory, gidx, sample)
            if path is None:
                missing += 1
                draw.rectangle([x, y, x + CELL, y + CELL], outline="red")
                draw.text(
                    (x + CELL // 2, y + CELL // 2),
                    "missing",
                    fill="red",
                    font=font_c,
                    anchor="mm",
                )
                continue
            img = Image.open(path).convert("RGB").resize((CELL, CELL), Image.LANCZOS)
            sheet.paste(img, (x, y))

        caption = textwrap.fill(prompts[gidx]["text"], width=int(cols * 46))
        draw.text((PAD, y + CELL + 5), caption[:400], fill="black", font=font_c)

    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    sheet.save(out_path)
    return missing


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--dirs", nargs="+", required=True)
    p.add_argument("--labels", nargs="+", default=None)
    p.add_argument("--jsonl_path", required=True)
    p.add_argument("--out", default="figures/contact_sheet")
    p.add_argument(
        "--category",
        default=None,
        help="only rows whose prompt has this category field",
    )
    p.add_argument("--sample", type=int, default=0)
    p.add_argument(
        "--per-sheet",
        type=int,
        default=4,
        help="rows per output image; keeps each sheet legible",
    )
    args = p.parse_args()

    labels = args.labels or [os.path.basename(d.rstrip("/")) for d in args.dirs]
    if len(labels) != len(args.dirs):
        raise SystemExit("--labels must match --dirs in length")

    prompts = load_prompts(args.jsonl_path)
    indices = sorted(
        i
        for i, item in prompts.items()
        if args.category is None or item.get("category") == args.category
    )
    if not indices:
        raise SystemExit(f"no prompts with category={args.category}")

    total_missing = 0
    sheets = 0
    for start in range(0, len(indices), args.per_sheet):
        chunk = indices[start : start + args.per_sheet]
        out_path = f"{args.out}_{sheets:02d}.png"
        total_missing += build_sheet(
            args.dirs, labels, prompts, chunk, args.sample, out_path
        )
        print(f"wrote {out_path}  ({len(chunk)} prompts x {len(args.dirs)} configs)")
        sheets += 1

    if total_missing:
        print(
            f"warning: {total_missing} cells had no image -- "
            f"were all configs generated from the same jsonl and seed?"
        )


if __name__ == "__main__":
    main()
