# Shiva-DiT

## Environment
Our project has been tested with **Python 3.10** (specifically version 3.10.12). We highly recommend using a virtual environment, such as Anaconda3, to manage and install the required dependencies to avoid conflicts.

## Installation
Before installation, ensure all required Python dependencies are available. You can install them using:

```bash
pip install -r requirements.txt
```

## Usage

### HookManager
Shiva-DiT operates by dynamically injecting acceleration modules (hooks) into various Transformer blocks within a standard `diffusers` pipeline. This process is orchestrated by the `HookManager`.

The `HookManager` offers two modes to control and hook the behavior of a DiT block: **Callback** and **Delegate**. For development details and advanced usage of the `HookManager`, please refer to the [Hook Documentation](shiva_dit/hook/README.md).

### Training-free Methods
These methods do not require any learnable modules and can be injected directly into DiT blocks for immediate acceleration.

**Example Usage (ToMe)**:
```python
import torch
from shiva_dit.utils import get_pipeline, process_quick_start
from shiva_dit.hook.hook_manager import HookManager
# Assuming ToMeSDDitBlockHook is implemented in your baselines or hooks
from baselines.tome_sd import ToMeSDDitBlockHook 

model_type = "sd3"
pipe, model_attrs = get_pipeline(model_type)

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
    res = process_quick_start(pipe)
    res.save(f"{model_type}_tomesd.png")
```

### Training-based Methods (Shiva-DiT)
We provide a unified function to load a pipeline equipped with learnable acceleration modules.

```python
from shiva_dit.utils.inference_utils import load_pipeline_for_inference

load_pipeline_for_inference(
    ckpt_path: str = None,          # Path to the learned checkpoint
    base_model_path: str = None,    # Base model path (HuggingFace or local)
    model_type: str = None,         # Model architecture (e.g., 'sd3', 'flux-dev')
    device="cuda", 
    dtype=torch.bfloat16,
    gater_lut_path: str = None,     # Path to Shiva's LUT table for acceleration
    merge_lora: bool = True,
    meta_config: dict = None        # Config for training-free methods if needed
)
```

We also provide comprehensive training code. Please refer to the [Training Documentation](shiva_dit/training/README.md).

### Evaluation
We provide scripts for image generation and system metric measurement (FLOPs/Latency). Please refer to the [Benchmark Documentation](benchmark/README.md) for details.

**Example Usage:**
```
# Pics Generation
# Image Generation
python -m benchmark.generation \
    --ckpt_path /path/to/ckpt_path \
    --model_type sd3 \
    --jsonl_path /path/to/prompts.jsonl \
    --image_dir /path/to/save_images

# System Metrics (Latency/FLOPs)
python -m benchmark.infer \
    --ckpt_path /path/to/ckpt_path \
    --model_type sd3 \
    --gater_lut /path/to/lut
```

## Contribution
### Baselines
We have reproduced several state-of-the-art acceleration baselines based on their open-source implementations or papers. These implementations can be found in the `baselines/` directory:

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
      <td rowspan="4"><b>Training-based</b></td>
      <td>DiffCR<sup>[6]</sup></td>
    </tr>
    <tr>
      <td>DyDiT<sup>[7]</sup></td>
    </tr>
    <tr>
      <td>SparseDiT<sup>[8]</sup></td>
    </tr>
    <tr>
      <td><b>Shiva-DiT</b> (ours)</td>
    </tr>
  </tbody>
</table>

You can go to [this](baselines/README.md) for details.

### Library
**diffusers**: Future versions will provide our own forked and maintained version of `diffusers`. Currently, the integration relies on hard-coded adaptations.

### References
[1] Bolya, D. and Hoffman, J. Token merging for fast stable diffusion. CVPR Workshop on Efficient Deep Learning for Computer Vision, 2023.

[2] Kim, M., Gao, S., Hsu, Y.-C., Shen, Y., and Jin, H. Token fusion: Bridging the gap between token pruning and token merging. In 2024 IEEE/CVF Winter Conference on Applications of Computer Vision (WACV), pp. 1372–1381, 2024. doi: 10.1109/WACV57701.2024.00141.

[3] Fang, H., Tang, S., Cao, J., Zhang, E., Tang, F., and Lee, T.-Y. Attend to not attended: Structure-then-detail token merging for post-training dit acceleration, 2025.

[4] Lu, W., Zheng, S., Xia, Y., and Wang, S. ToMA: Token merge with attention for diffusion models. In Forty-second International Conference on Machine Learning, 2025.

[5] Wu, H., Xu, J., Le, H., and Samaras, D. Importance-based token merging for efficient image and video generation, 2025.

[6] You, H., Barnes, C., Zhou, Y., Kang, Y., Du, Z., Zhou, W., Zhang, L., Nitzan, Y., Liu, X., Lin, Z., Shechtman, E., Amirghodsi, S., and Lin, Y. C. Layer- and timestep-adaptive differentiable token compression ratios for efficient diffusion transformers, 2025.

[7] Zhao, W., Han, Y., Tang, J., Wang, K., Song, Y., Huang, G., Wang, F., and You, Y. Dynamic diffusion transformer. arXiv preprint arXiv:2410.03456, 2024.

[8] Chang, S., Wang, P., Tang, J., Wang, F., and Yang, Y. Sparsedit: Token sparsification for efficient diffusion transformer, 2025.