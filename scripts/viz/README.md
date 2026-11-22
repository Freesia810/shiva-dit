# Paper figures

Run these commands from the repository root. Install the optional plotting dependencies with `python -m pip install -r scripts/viz/requirements.txt`. Most plots read existing measurements. The router panels use a forward pass on a supplied checkpoint.

## Figure 2 · FID and inference latency

```bash
python -m scripts.viz.viz_pareto
```

The [21 plotted points](fig2_pareto.csv) retain the paper's FID and latency values. The renderer writes a PDF and a 1255 × 810 PNG. Pass `--font /path/to/Times-New-Roman.ttf` to use the original font.

## Figure 4 · Adaptive policy

```bash
python -m scripts.viz.viz_val_loss --adaptive adaptive_val_loss.csv --fixed fix_val_loss.csv
python -m scripts.viz.viz_ratio_landscape --lut runs/shiva-50steps.pt
```

The loss panel reads CSV exports with `step` and `value` columns and uses the same warmup exclusion as the paper plotting script. The retention panel accepts a LUT from `benchmark.lut` or a headerless CSV with timesteps as rows in low-to-high order and layers as columns.

## Figure 5(a, c) · Synthetic validation plots

```bash
python -m scripts.viz.viz_topk --dynamics budget_and_accuracy.csv --consistency cosine_similarity.csv
```

Provide saved `step,k,accuracy` and `cosine_similarity` data. Either input can be rendered on its own.

## Figures 6 and 7 · Router scores

```bash
python -m scripts.viz.viz_router_masks \
    --checkpoint /path/to/shiva-sd3.pt --base-model-path /path/to/sd3 \
    --image /path/to/input.jpg --prompt "Image caption" \
    --output-dir runs/figures/router
```

The script writes heatmap and histogram PDFs. The defaults select layers 1, 8, 15, and 23 and training timesteps near 900, 700, 500, and 100. The paper's comparison panels use the corresponding checkpoints and the same image and caption for each row of results.

The source release includes the Figure 2 points and the plotting code. Validation histories, other original measurements, images, and model checkpoints are external inputs.
