# Shiva-DiT

Residual-Based Differentiable Top-k Selection for Efficient Diffusion Transformers

Published in Transactions on Machine Learning Research, November 2026. [Paper](https://openreview.net/forum?id=uAAaT1KajI)

Shiva-DiT accelerates diffusion transformers by pruning image tokens within each denoising block. A router ranks the tokens for the current input, and a learned ratio policy sets how many the block will process. Attention and the feed-forward network run on the selected tokens. The other tokens pass through the residual path, and the block restores the full spatial layout for the next layer.

Hard top-k selection makes the number of processed tokens explicit, but the discrete ranking blocks gradients to both the scores and the budget. Shiva-DiT estimates these gradients during the backward pass from the path each token actually took. The same training objective can therefore learn both which tokens to keep and how many to keep, using a single sparse student forward pass.

## Environment

Use Python 3.10 and CUDA-enabled PyTorch. The dependencies specify PyTorch 2.6, Transformers 4.57.1, and PEFT 0.17.1. We recommend installing them in a virtual environment.

## Install

Run the following command from the repository root.

```bash
python -m pip install -r requirements.txt
```

## Usage

### Image generation

Set the paths to your SD3-Medium base model in Diffusers format and a trained Shiva checkpoint. Run the commands below from the repository root so that Python loads the bundled Diffusers implementation.

```bash
export BASE_MODEL=/path/to/stable-diffusion-3-medium-diffusers
export SHIVA_CHECKPOINT=/path/to/shiva-sd3.pt

python -m benchmark.generation \
    --model_type sd3 \
    --base_model_path "$BASE_MODEL" \
    --ckpt_path "$SHIVA_CHECKPOINT" \
    --jsonl_path examples/prompts.jsonl \
    --output_dir runs/quickstart \
    --samples_per_prompt 1 --steps 50 --skip_metrics
```

The example generates one image per prompt and writes them to `runs/quickstart/`. Replace `examples/prompts.jsonl` with a JSONL file containing a `text` field on each line to generate from your own prompts.

### Exporting a ratio schedule

The learned retention ratios depend on the layer and denoising timestep. Export them as a lookup table (LUT) to reuse the same schedule across prompts. During generation, the router still scores tokens for each input.

```bash
python -m benchmark.lut \
    --base_model_path "$BASE_MODEL" \
    --ckpt_path "$SHIVA_CHECKPOINT" \
    --steps 50 --height 1024 --width 1024 \
    --output_path runs/shiva-50steps.pt
```

Pass `--gater_lut runs/shiva-50steps.pt` to the generation command. Use the same checkpoint, backbone, scheduler, and denoising step count when exporting and applying the LUT.

### Training

The recipes warm up the router at a range of retention ratios, learn the ratio policy under an average retention target, and then jointly tune the router, policy, and backbone adapters. Distillation from a frozen teacher guides the sparse student. The training JSONL files pair `text` with `file_name`. Image paths are resolved relative to the image directory.

```bash
accelerate launch --module scripts.train.shiva_sd3_60 \
    --base-model-path "$BASE_MODEL" \
    --train-jsonl /path/to/MJHQ-30K/train.jsonl \
    --image-dir /path/to/MJHQ-30K \
    --val-jsonl /path/to/MJHQ-30K/val.jsonl \
    --val-image-dir /path/to/MJHQ-30K \
    --inference-jsonl /path/to/MJHQ-30K/val.jsonl \
    --output-dir runs/shiva-sd3-60
```

The repository also includes `scripts.train.shiva_sd3_80`, `scripts.train.shiva_flux`, and `scripts.train.shiva_pixart` for the other training recipes. Use the resulting checkpoint with the generation and LUT commands above. [Training guide](shiva_dit/training/README.md)

### Evaluation

For a matched image-quality comparison, use the same prompt file and initial latents for every method. The generator restores each latent by prompt and sample index. `--strict_latents` makes the run fail if a required latent is missing or unreadable.

```bash
torchrun --nproc_per_node=2 -m benchmark.generation \
    --model_type sd3 --base_model_path "$BASE_MODEL" \
    --ckpt_path "$SHIVA_CHECKPOINT" --gater_lut runs/shiva-50steps.pt \
    --jsonl_path /path/to/eval.jsonl --latent_source_dir /path/to/latents \
    --output_dir runs/shiva-eval --steps 50 --samples_per_prompt 5 \
    --strict_latents --skip_metrics

python -m benchmark.evaluate \
    --image_dir runs/shiva-eval --jsonl_path /path/to/eval.jsonl \
    --fid_stats /path/to/mjhq_30k_stats.npz
```

The evaluator supports FID, ImageReward, CLIP Score, and CLIP-IQA. It uses the FID reference statistics supplied by `--fid_stats`. To measure latency, use the inference benchmark with the same denoising settings.

```bash
python -m benchmark.infer \
    --model_type sd3 --base_model_path "$BASE_MODEL" \
    --ckpt_path "$SHIVA_CHECKPOINT" --gater_lut runs/shiva-50steps.pt \
    --jsonl_path examples/prompts.jsonl --steps 50 --height 1024 --width 1024
```

The benchmark reports end-to-end latency, DiT latency, peak allocated GPU memory, and average FLOPs per denoising step. [Benchmark guide](benchmark/README.md)

## Features

### Residual-based differentiable top-k

The forward pass gathers the top-$k$ tokens by score and scatters their updated features back to the original positions. In the backward pass, pairwise comparisons give each token a soft rank. Comparing that rank with $k$ defines a differentiable inclusion probability. The estimator uses gradients from the executed path to train the router scores and the budget through this probability. [Pruner](shiva_dit/modules/pruner.py) · [Estimator](shiva_dit/utils/ste.py)

### Context-aware router and adaptive ratio policy

The router combines per-token features with pooled text, timestep, and layer embeddings. The ratio policy adds the outputs of separate layer and timestep branches and passes the sum with a bias through a sigmoid. At layer $\ell$ and timestep $t$, the resulting ratio $r_{\ell,t}$ gives a budget of $k = \lceil r_{\ell,t}N \rceil$ for $N$ image tokens. During training, an exponential moving average tracks retention and guides it toward the target while allowing different layers and denoising steps to use different budgets. [Router](shiva_dit/modules/router.py) · [Ratio policy](shiva_dit/modules/gater.py)

## Contributing

### Other Methods for Diffusion Models

For comparison, we implement the training-free methods ToMeSD<sup>[1]</sup>, ToFu<sup>[2]</sup>, SDTM<sup>[3]</sup>, ToMA<sup>[4]</sup>, and IBTM<sup>[5]</sup>, as well as the trained methods DiffCR<sup>[6]</sup>, DyDiT<sup>[7]</sup>, SparseDiT<sup>[8]</sup>, and NeuralSort<sup>[9]</sup>. [Baseline guide](baselines/README.md)

These methods use the same generation and benchmark entry points as Shiva-DiT. For example, the following command benchmarks ToMeSD using the bundled configuration.

```bash
python -m benchmark.infer \
    --base_model_path "$BASE_MODEL" --meta_config benchmark/meta_config/tome_sd.json \
    --jsonl_path examples/prompts.jsonl --steps 50 --height 1024 --width 1024
```

For a trained baseline, pass its compatible checkpoint with `--ckpt_path`. Use the same hardware, precision, prompts, and sampling settings for timing comparisons. The settings used in the paper are in Appendix B.4. [Paper](https://openreview.net/forum?id=uAAaT1KajI)

## Citation

```bibtex
@article{zhang2026shivadit,
  title   = {Shiva-DiT: Residual-Based Differentiable Top-k Selection for Efficient Diffusion Transformers},
  author  = {Zhang, Jiaji and Zhao, Hailiang and Wu, Jiaju and Sun, Ruichao and Zhao, Xinkui and Deng, Shuiguang},
  journal = {Transactions on Machine Learning Research},
  year    = {2026},
  month   = nov,
  url     = {https://openreview.net/forum?id=uAAaT1KajI}
}
```

## References

[1] Daniel Bolya and Judy Hoffman. *Token Merging for Fast Stable Diffusion.* CVPR Workshops, 2023. [Paper](https://openaccess.thecvf.com/content/CVPR2023W/ECV/html/Bolya_Token_Merging_for_Fast_Stable_Diffusion_CVPRW_2023_paper.html)

[2] Minchul Kim, Shangqian Gao, Yen-Chang Hsu, Yilin Shen, and Hongxia Jin. *Token Fusion: Bridging the Gap Between Token Pruning and Token Merging.* WACV, 2024. [Paper](https://openaccess.thecvf.com/content/WACV2024/html/Kim_Token_Fusion_Bridging_the_Gap_Between_Token_Pruning_and_Token_WACV_2024_paper.html)

[3] Haipeng Fang, Sheng Tang, Juan Cao, Enshuo Zhang, Fan Tang, and Tong-Yee Lee. *Attend to Not Attended: Structure-then-Detail Token Merging for Post-training DiT Acceleration.* CVPR, 2025. [Paper](https://arxiv.org/abs/2505.11707)

[4] Wenbo Lu, Shaoyi Zheng, Yuxuan Xia, and Shengjie Wang. *ToMA: Token Merge with Attention for Diffusion Models.* ICML, 2025. [Paper](https://proceedings.mlr.press/v267/lu25v.html)

[5] Haoyu Wu, Jingyi Xu, Hieu Le, and Dimitris Samaras. *Importance-Based Token Merging for Efficient Image and Video Generation.* ICCV, 2025. [Paper](https://openaccess.thecvf.com/content/ICCV2025/html/Wu_Importance-Based_Token_Merging_for_Efficient_Image_and_Video_Generation_ICCV_2025_paper.html)

[6] Haoran You et al. *Layer- and Timestep-Adaptive Differentiable Token Compression Ratios for Efficient Diffusion Transformers.* CVPR, 2025. [Paper](https://arxiv.org/abs/2412.16822)

[7] Wangbo Zhao et al. *Dynamic Diffusion Transformer.* ICLR, 2025. [Paper](https://proceedings.iclr.cc/paper_files/paper/2025/hash/a44a70acd5d0abc1a252ada9719dd06d-Abstract-Conference.html)

[8] Shuning Chang, Pichao Wang, Jiasheng Tang, Fan Wang, and Yi Yang. *SparseDiT: Token Sparsification for Efficient Diffusion Transformer.* NeurIPS, 2025. [Paper](https://proceedings.neurips.cc/paper_files/paper/2025/hash/83fe5a77502e3d4cfab5960aed0ee6c3-Abstract-Conference.html)

[9] Aditya Grover, Eric Wang, Aaron Zweig, and Stefano Ermon. *Stochastic Optimization of Sorting Networks via Continuous Relaxations.* ICLR, 2019. [Paper](https://arxiv.org/abs/1903.08850)
