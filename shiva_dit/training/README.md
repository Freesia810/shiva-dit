# Training Shiva-DiT

The training recipes use Accelerate and PEFT. They warm up the token router, learn the layer and timestep ratio policy, and jointly tune the sparse model. Model-specific hyperparameters and stage schedules are defined in [the recipes](../../scripts/train). See Appendix B.2 for the paper's training protocol. [Paper](https://openreview.net/forum?id=uAAaT1KajI)

## Prepare the dataset

Each line of a training or validation JSONL file gives a prompt and the path to its image. The loader resolves relative paths against the image root.

```json
{"file_name":"animals/example.jpg","text":"A red fox in a snowy forest"}
```

The helper below creates a new split from the MJHQ metadata. Use the saved paper split for exact paper reproduction.

```bash
python -m dataset.mjhq --root_dir /path/to/MJHQ-30K \
    --train_samples 28000 --val_ratio 0.0666666667
```

## Run a recipe

Run from the repository root. Choose a process count and mixed precision mode supported by your GPUs when configuring Accelerate. Write training outputs to a dedicated directory.

```bash
accelerate launch --module scripts.train.shiva_sd3_60 \
    --base-model-path /path/to/stable-diffusion-3-medium-diffusers \
    --train-jsonl /path/to/MJHQ-30K/train.jsonl \
    --image-dir /path/to/MJHQ-30K \
    --val-jsonl /path/to/MJHQ-30K/val.jsonl \
    --val-image-dir /path/to/MJHQ-30K \
    --inference-jsonl /path/to/MJHQ-30K/val.jsonl \
    --output-dir runs/shiva-sd3-60
```

Use `scripts.train.shiva_sd3_80`, `scripts.train.shiva_flux`, or `scripts.train.shiva_pixart` with the corresponding backbone to switch recipes. Pass a saved inference checkpoint to [the generation or LUT entry point](../../benchmark/README.md).

## Implementation

- [Trainer configuration](config.py) declares optimizer, data, checkpoint, and distillation options.
- [Input preparers](input_preparers.py) handle backbone-specific latent and text inputs.
- [Trainer](trainer.py) manages the student, teacher, stage schedule, and checkpoint lifecycle.
