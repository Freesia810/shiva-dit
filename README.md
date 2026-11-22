# Shiva-DiT

[![TMLR 2026](https://img.shields.io/badge/TMLR-2026-blue.svg)](https://openreview.net/forum?id=uAAaT1KajI)
[![arXiv 2602.05605](https://img.shields.io/badge/arXiv-2602.05605-b31b1b.svg)](https://arxiv.org/abs/2602.05605)
[![GitHub repository](https://img.shields.io/badge/GitHub-Repository-black)](https://github.com/Freesia810/shiva-dit)
[![Apache 2.0 license](https://img.shields.io/badge/License-Apache--2.0-green.svg)](LICENSE)

Residual-Based Differentiable Top-k Selection for Efficient Diffusion Transformers

Shiva-DiT reduces the number of image tokens processed in each block of a diffusion transformer. The **context-aware router** scores tokens using their features, the prompt, the denoising timestep, and the layer. The **adaptive ratio policy** sets the retention budget for that layer and timestep. The block runs attention and the feed-forward network on the top-$k$ image tokens, carries the remaining tokens through the residual path, and restores the original token layout before the next block.

During training, a **residual-based straight-through estimator** makes this hard selection learnable. It assigns each token a soft rank and uses its position relative to $k$ to approximate its inclusion probability. Gradients from the path each token took propagate through that probability to both the router scores and the budget. This lets both decisions learn from one sparse student forward pass. At inference, the ratio policy can be exported to a lookup table (LUT) because its outputs depend on the layer and timestep. The router continues to score each input.

## Environment

The code was tested with **Python 3.10.12**. Use a virtual environment and a CUDA-enabled PyTorch installation for training and GPU benchmarking.

## Installation

Install the dependencies and run the examples from the repository root.

```bash
pip install -r requirements.txt
```

## Usage

### HookManager

Shiva-DiT and the baselines attach hooks to transformer blocks in the bundled Diffusers pipelines. `HookManager` configures which hook to install, then injects it into the transformer modules. **Callback** hooks intercept points within a block's forward pass. **Delegate** hooks control the block's forward pass. [Hook documentation](shiva_dit/hook/README.md)

### Training-free Methods

Training-free methods can be applied directly to the base model. This example injects the ToMeSD hook into SD3-Medium.

```python
import torch

from baselines.tome_sd import ToMeSDDitBlockHook
from shiva_dit.hook import HookManager
from shiva_dit.utils import get_pipeline, process_quick_start

model_type = "sd3"
pipe, model_attrs = get_pipeline(
    model_type,
    model_path="/path/to/stable-diffusion-3-medium-diffusers",
    device=torch.device("cuda"),
)

tomesd_config = {
    "stride": (2, 2),
    "merged_ratio": 0.15,
    "merge_config": {
        "self-attn": True,
        "cross-attn": False,
        "ffn": True,
        "parallel": False,
    },
    "generator": torch.Generator("cuda").manual_seed(0),
}

HookManager.configure_hooks(
    model_type=model_type,
    default=(ToMeSDDitBlockHook, tomesd_config),
)

with HookManager.enabled():
    HookManager.inject_hooks(model_type=model_type, pipe=pipe, model_attrs=model_attrs)
    image = process_quick_start(pipe)
    image.save("sd3_tomesd.png")
```

### Training-based Methods (Shiva-DiT)

Load a trained Shiva checkpoint alongside its base model. The loader reads the checkpoint configuration, attaches the corresponding hooks, and loads the trained weights.

```python
import torch

from shiva_dit.utils import process_quick_start
from shiva_dit.utils.inference_utils import load_pipeline_for_inference

pipe = load_pipeline_for_inference(
    ckpt_path="/path/to/shiva-sd3.pt",
    base_model_path="/path/to/stable-diffusion-3-medium-diffusers",
    model_type="sd3",
    device="cuda",
    dtype=torch.bfloat16,
    merge_lora=True,
)
image = process_quick_start(pipe, prompt="A ceramic teapot beside a window")
image.save("sd3_shiva.png")
```

To train a Shiva checkpoint, use the recipes for SD3-Medium, FLUX.1-dev, or PixArt-Sigma. The training guide covers data preparation, router warmup, ratio-policy learning, and joint tuning. [Training documentation](shiva_dit/training/README.md)

### Evaluation

Use `benchmark.lut` to export a learned ratio schedule, `benchmark.generation` to generate images, and `benchmark.infer` to measure latency and FLOPs. The example below uses the same checkpoint and 50-step schedule for generation and timing.

```bash
export BASE_MODEL=/path/to/stable-diffusion-3-medium-diffusers
export SHIVA_CHECKPOINT=/path/to/shiva-sd3.pt

python -m benchmark.lut     --base_model_path "$BASE_MODEL" --ckpt_path "$SHIVA_CHECKPOINT"     --steps 50 --output_path runs/shiva-50steps.pt

python -m benchmark.generation     --model_type sd3 --base_model_path "$BASE_MODEL"     --ckpt_path "$SHIVA_CHECKPOINT" --gater_lut runs/shiva-50steps.pt     --jsonl_path examples/prompts.jsonl --output_dir runs/shiva-sd3     --steps 50 --samples_per_prompt 1 --skip_metrics

python -m benchmark.infer     --model_type sd3 --base_model_path "$BASE_MODEL"     --ckpt_path "$SHIVA_CHECKPOINT" --gater_lut runs/shiva-50steps.pt     --jsonl_path examples/prompts.jsonl --steps 50
```

For image-quality evaluation, generate from the full evaluation prompt file and reuse the same initial latents across methods. The benchmark guide covers matched generation and scoring with FID, ImageReward, CLIP Score, and CLIP-IQA. [Benchmark documentation](benchmark/README.md)

## Contribution
### Baselines

The repository includes the following token-reduction methods. Each uses the same hook framework and benchmark entry points.

<table>
  <thead>
    <tr>
      <th>Category</th>
      <th>Method</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td rowspan="5"><b>Training-free</b><br>(Instant Injection)</td>
      <td>ToMeSD<sup>[1]</sup></td>
    </tr>
    <tr>
      <td>ToFu<sup>[2]</sup></td>
    </tr>
    <tr>
      <td>SDTM<sup>[3]</sup></td>
    </tr>
    <tr>
      <td>ToMA<sup>[4]</sup></td>
    </tr>
    <tr>
      <td>IBTM<sup>[5]</sup></td>
    </tr>
    <tr>
      <td rowspan="5"><b>Training-based</b></td>
      <td>DiffCR<sup>[6]</sup></td>
    </tr>
    <tr>
      <td>DyDiT<sup>[7]</sup></td>
    </tr>
    <tr>
      <td>SparseDiT<sup>[8]</sup></td>
    </tr>
    <tr>
      <td>NeuralSort<sup>[9]</sup></td>
    </tr>
    <tr>
      <td><b>Shiva-DiT</b> (ours)</td>
    </tr>
  </tbody>
</table>

The baseline guide lists the hooks, available configurations, and checkpoint requirements. [Baseline guide](baselines/README.md)

### Library

The repository includes a modified subset of Hugging Face Diffusers with block and pipeline adaptations for the hooks. [Third-party notices](THIRD_PARTY.md)

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

[1] Bolya, D., and Hoffman, J. Token Merging for Fast Stable Diffusion. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition Workshops (CVPRW), pp. 4599–4603, 2023.

[2] Kim, M., Gao, S., Hsu, Y.-C., Shen, Y., and Jin, H. Token Fusion: Bridging the Gap Between Token Pruning and Token Merging. In Proceedings of the IEEE/CVF Winter Conference on Applications of Computer Vision (WACV), pp. 1383–1392, 2024.

[3] Fang, H., Tang, S., Cao, J., Zhang, E., Tang, F., and Lee, T.-Y. Attend to Not Attended: Structure-then-Detail Token Merging for Post-training DiT Acceleration. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), pp. 18083–18092, 2025.

[4] Lu, W., Zheng, S., Xia, Y., and Wang, S. ToMA: Token Merge with Attention for Diffusion Models. In Proceedings of the 42nd International Conference on Machine Learning (ICML), Proceedings of Machine Learning Research, vol. 267, pp. 40930–40951, 2025.

[5] Wu, H., Xu, J., Le, H., and Samaras, D. Importance-Based Token Merging for Efficient Image and Video Generation. In Proceedings of the IEEE/CVF International Conference on Computer Vision (ICCV), pp. 4983–4995, 2025.

[6] You, H., Barnes, C., Zhou, Y., Kang, Y., Du, Z., Zhou, W., Zhang, L., Nitzan, Y., Liu, X., Lin, Z., Shechtman, E., Amirghodsi, S., and Lin, Y. C. Layer- and Timestep-Adaptive Differentiable Token Compression Ratios for Efficient Diffusion Transformers. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), pp. 18072–18082, 2025.

[7] Zhao, W., Han, Y., Tang, J., Wang, K., Song, Y., Huang, G., Wang, F., and You, Y. Dynamic Diffusion Transformer. In Proceedings of the 13th International Conference on Learning Representations (ICLR), 2025.

[8] Chang, S., Wang, P., Tang, J., Wang, F., and Yang, Y. SparseDiT: Token Sparsification for Efficient Diffusion Transformer. In Advances in Neural Information Processing Systems (NeurIPS), vol. 38, 2025.

[9] Grover, A., Wang, E., Zweig, A., and Ermon, S. Stochastic Optimization of Sorting Networks via Continuous Relaxations. In Proceedings of the 7th International Conference on Learning Representations (ICLR), 2019.
