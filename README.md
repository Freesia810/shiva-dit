# Shiva-DiT

Residual-Based Differentiable Top-k Selection for Efficient Diffusion Transformers

Published in Transactions on Machine Learning Research, November 2026. [Paper](https://openreview.net/forum?id=uAAaT1KajI)

## Environment

Use Python 3.10 and CUDA-enabled PyTorch. The dependencies specify PyTorch 2.6, Transformers 4.57.1, and PEFT 0.17.1.

## Install

Run the following command from the repository root.

```bash
python -m pip install -r requirements.txt
```

## Usage

To generate images with Shiva-DiT, set the paths to your SD3-Medium base model and Shiva checkpoint.

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

To use a cached ratio schedule, pass `--gater_lut` with a LUT exported for the same backbone and denoising steps. See the benchmark guide for LUT export, image quality evaluation, and latency measurement. [Benchmark guide](benchmark/README.md)

Training recipes are available for SD3-Medium, FLUX.1-dev, and PixArt-Σ. The training guide covers dataset preparation and launching a recipe with Accelerate. [Training guide](shiva_dit/training/README.md)

## Features

- **Context-aware routing and adaptive ratios.** The router scores image tokens using the prompt, timestep, and layer. The ratio policy allocates a retention budget to each layer and denoising step. [Router](shiva_dit/modules/router.py) · [Ratio policy](shiva_dit/modules/gater.py)
- **Residual-based differentiable top-k.** The block processes the selected tokens and carries the remaining tokens along the residual path. The estimator propagates gradients to the scores and the budget through a single sparse backbone pass. [Pruner](shiva_dit/modules/pruner.py) · [Estimator](shiva_dit/utils/ste.py)

## Contributing

### Other Methods for Diffusion Models

For comparison, we implement the training-free methods ToMeSD<sup>[1]</sup>, ToFu<sup>[2]</sup>, SDTM<sup>[3]</sup>, ToMA<sup>[4]</sup>, and IBTM<sup>[5]</sup>, as well as the trained methods DiffCR<sup>[6]</sup>, DyDiT<sup>[7]</sup>, SparseDiT<sup>[8]</sup>, and NeuralSort<sup>[9]</sup>. [Baseline guide](baselines/README.md)

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
