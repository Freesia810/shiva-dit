# Benchmark & Evaluation

This directory contains three core scripts for comprehensive evaluation of Shiva-DiT and related baselines, covering image generation quality, system inference performance (Latency/FLOPs), and acceleration LUT generation.

## Image Generation (`generation.py`)

This script is used for batch image generation and supports calculating quality metrics such as FID, CLIP Score, and ImageReward. It supports multi-GPU parallel generation for efficiency.

### Features
* **Batch Generation**: Supports reading prompt lists in JSONL format.
* **Multi-GPU Parallelism**: Automatically distributes tasks via `torchrun` to accelerate large-scale sampling.
* **Metrics Calculation**: Integrates `pytorch-fid`, `torchmetrics` (CLIP), and `ImageReward`.
* **Shiva Support**: Supports loading trained Shiva checkpoints or using baseline methods.

### Usage Examples

**Single GPU Generation:**
```bash
python -m benchmark.generation \
    --model_type sd3 \
    --ckpt_path /path/to/shiva_ckpt.pt \
    --jsonl_path /path/to/prompts.jsonl \
    --output_dir results/shiva_sd3 \
    --samples_per_prompt 4 \
    --seed 42
```

**Multi-GPU Parallel Generation (Recommended):**
```bash
torchrun --nproc_per_node=8 -m benchmark.generation \
    --model_type sd3 \
    --ckpt_path /path/to/shiva_ckpt.pt \
    --jsonl_path /path/to/prompts.jsonl \
    --output_dir results/shiva_sd3 \
    --samples_per_prompt 4 \
    --skip_metrics  # Recommended to skip online metrics calculation in multi-GPU mode
```

**Arguments:**
* `--ckpt_path`: Path to the trained Shiva Checkpoint (optional for baselines).
* `--model_type`: Model architecture (e.g., `sd3`, `flux-dev`, `pixart-sigma`).
* `--jsonl_path`: Data file containing prompts (must include a `"text"` field).
* `--fid_stats`: (Optional) Path to reference `.npz` file for FID calculation.

---

## System Performance Benchmark (`infer.py`)

This script accurately measures model inference performance, including End-to-End Latency, DiT Module Latency, Peak Memory usage, and Computational Cost (FLOPs/MACs).

### Features
* **Precise Timing**: Uses `torch.cuda.Event` for millisecond-level accuracy.
* **FLOPs Statistics**: Integrates `fvcore` to automatically count actual inference FLOPs.
* **Breakdown**: Distinguishes between end-to-end time and pure DiT computation time.

### Usage Examples

```bash
python -m benchmark.infer \
    --model_type flux-dev \
    --ckpt_path /path/to/shiva_ckpt.pt \
    --height 1024 \
    --width 1024 \
    --steps 20 \
    --measure_flops
```

**Output Example:**
```text
System Performance (Avg over 50 runs):
  Peak VRAM:          14500.25 MB
  End-to-End Latency: 2150.40 ms
  DiT Latency:        1850.10 ms
Computational Cost:
  Avg FLOPs: 15.4 TFLOPs
```

---

## Look-Up Table (LUT) Generation (`lut.py`)

After training Shiva-DiT, the decisions of the Policy Network can be "frozen" for inference. This script records all decisions (Merge Ratio) by running a pre-inference pass and saves them as a `.pt` file.

In deployment, loading the LUT allows skipping the Policy network computation, further improving inference speed.

### Usage Examples

```bash
python -m benchmark.lut \
    --ckpt_path /path/to/shiva_ckpt.pt \
    --output_path shiva_lut_20steps.pt \
    --steps 20 \
    --height 1024 \
    --width 1024
```

**Using the Generated LUT:**
Load via the `--gater_lut` argument in `generation.py` or `infer.py`:

```bash
python -m benchmark.infer \
    --ckpt_path /path/to/shiva_ckpt.pt \
    --gater_lut shiva_lut_20steps.pt \
    ...
```