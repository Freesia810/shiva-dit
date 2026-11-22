# Generation and evaluation

Run the following commands from the repository root. Set `BASE_MODEL` and `SHIVA_CHECKPOINT` as in the [quick start](../README.md).

## Export an inference LUT

Export the learned schedule from an adaptive Shiva checkpoint. Use the same backbone, scheduler, and denoising grid to generate images with the LUT. Pass its step count explicitly with `--steps`.

```bash
python -m benchmark.lut \
    --base_model_path "$BASE_MODEL" \
    --ckpt_path "$SHIVA_CHECKPOINT" \
    --steps 50 --height 1024 --width 1024 \
    --output_path runs/shiva-50steps.pt
```

The file stores `ratio` and `gater` tensors of shape `[layers, steps]`. The deployed image pruning path uses the ratio entries for hard top-k. Dense layers skipped by the hook have ratio 1.

## Generate matched samples

Use a JSONL file with a `text` field on each line. With fixed latents, the index of a prompt in the file determines both its latent and image filename. Use a fresh output directory for each method or configuration. Generation resumes by using readable images already present at their expected paths.

```bash
torchrun --nproc_per_node=2 -m benchmark.generation \
    --model_type sd3 --base_model_path "$BASE_MODEL" \
    --ckpt_path "$SHIVA_CHECKPOINT" --gater_lut runs/shiva-50steps.pt \
    --jsonl_path /path/to/eval.jsonl --latent_source_dir /path/to/latents \
    --strict_latents --output_dir runs/shiva-mjhq \
    --steps 50 --samples_per_prompt 5 --skip_metrics
```

Input latents are named `latent_p000000_s00.pt` and images are named `p0000_s00.png`. For a new evaluation, `python -m scripts.utils.dump_latents --help` lists the options for generating a reusable set of initial latents. The paper protocol pairs five fixed latents per prompt across methods.

## Score generated images

```bash
python -m benchmark.evaluate \
    --image_dir runs/shiva-mjhq --jsonl_path /path/to/eval.jsonl \
    --fid_stats /path/to/mjhq_30k_stats.npz
```

This entry point computes FID, ImageReward, CLIP Score, and CLIP-IQA. It uses the same file selection for all four metrics. With 2,000 prompts and five latents each, score all 10,000 images in a single pass for pooled FID. Provide locally cached metric assets through `SHIVA_METRIC_IR_PATH`, `SHIVA_METRIC_IR_CONFIG`, `SHIVA_METRIC_BERT_PATH`, and `SHIVA_METRIC_CLIP_PATH` when needed.

## Measure inference cost

```bash
python -m benchmark.infer \
    --model_type sd3 --base_model_path "$BASE_MODEL" \
    --ckpt_path "$SHIVA_CHECKPOINT" --gater_lut runs/shiva-50steps.pt \
    --jsonl_path examples/prompts.jsonl --steps 50 --height 1024 --width 1024
```

This CUDA benchmark reports end-to-end latency, DiT latency, peak allocated memory, and average FLOPs per denoising step. Match the GPU, precision, batch, prompt set, resolution, and sampling settings across comparisons. See [baseline configurations](../baselines/README.md) and Appendix B.4 for the paper's comparison settings. [Paper](https://openreview.net/forum?id=uAAaT1KajI)
