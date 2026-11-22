# Shiva-DiT

**Residual-Based Differentiable Top-k Selection for Efficient Diffusion Transformers**

Transactions on Machine Learning Research, November 2026 · [Paper and reviews](https://openreview.net/forum?id=uAAaT1KajI)

Shiva-DiT learns which image tokens to process and how many to retain in each diffusion transformer block. A context-aware router scores the tokens, and a layer- and timestep-dependent ratio policy determines the budget. The block processes the selected tokens and returns the others along the residual path. During training, a residual-based straight-through estimator propagates gradients to both the scores and the budget from a single sparse backbone pass.

## Environment and installation

Use Python 3.10 and a CUDA-enabled PyTorch installation for training, image generation, and latency measurement. The pinned environment uses PyTorch 2.6, Transformers 4.57.1, and PEFT 0.17.1.

```bash
conda create -n shiva python=3.10
conda activate shiva
python -m pip install -r requirements.txt
```

Run the examples from this repository's root. The checkout includes the modified Diffusers modules used by Shiva's hooks. Keep this checkout on the Python import path when running the scripts. [Third-party code](THIRD_PARTY.md)

To check the SD3 hook, checkpoint loading, and LUT locally without downloading model weights, run `python -m unittest test.test_inference_pipeline`.

## Quick start

The example uses an SD3-Medium base model in Diffusers format and a compatible Shiva checkpoint. Use your own checkpoint or train one with the [training guide](shiva_dit/training/README.md).

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

The output includes `runs/quickstart/p0000_s00.png`. Pass `--gater_lut path/to/lut.pt` to use a saved schedule for the same model and denoising grid. The [benchmark guide](benchmark/README.md) covers LUT export, matched-latent generation, quality metrics, and latency.

## Pipeline

| Component | Implementation |
| --- | --- |
| Context-aware router and ratio policy | [Router](shiva_dit/modules/router.py) · [Policy](shiva_dit/modules/gater.py) |
| Residual-based differentiable top-k | [Pruner](shiva_dit/modules/pruner.py) · [Autograd functions](shiva_dit/utils/ste.py) |
| Hook injection, sparse block execution, and restoration | [Installer](shiva_dit/shiva_installer.py) · [Block hook](shiva_dit/shiva_hook.py) |
| Staged training and checkpoint saving | [Training guide](shiva_dit/training/README.md) |
| Generation, evaluation, and latency | [Benchmark guide](benchmark/README.md) |
| Paper figures | [Plotting guide](scripts/viz/README.md) |

## Models and data

| Backbone | Model assets | Training recipe |
| --- | --- | --- |
| SD3-Medium | [Stability AI](https://huggingface.co/stabilityai/stable-diffusion-3-medium-diffusers) | [Shiva-60](scripts/train/shiva_sd3_60.py) · [Shiva-80](scripts/train/shiva_sd3_80.py) |
| FLUX.1-dev | [Black Forest Labs](https://huggingface.co/black-forest-labs/FLUX.1-dev) | [FLUX recipe](scripts/train/shiva_flux.py) |
| PixArt-Σ | [PixArt-α](https://huggingface.co/PixArt-alpha/PixArt-Sigma-XL-2-1024-MS) | [PixArt recipe](scripts/train/shiva_pixart.py) |

This is a source-code release. Obtain base models and training data from their respective providers. The same training and evaluation workflow supports new runs. Exact reproduction of the paper tables requires the original checkpoints, LUTs, prompt split, fixed latents, and FID reference statistics. [MJHQ-30K](https://huggingface.co/datasets/playgroundai/MJHQ-30K)

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

## License

Shiva-DiT source code is released under the [Apache License 2.0](LICENSE). Bundled components retain their existing notices. [Third-party code](THIRD_PARTY.md)
