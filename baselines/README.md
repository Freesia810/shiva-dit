# Acceleration Baselines

We have reproduced several state-of-the-art acceleration baselines for Diffusion Transformers (DiTs). These methods utilize the **Shiva-DiT Hook Mechanism** to inject acceleration logic into standard pipelines (e.g., SD3, Flux).

## List of Baselines

### Training-free Methods (Instant Injection)
These methods can be applied immediately without any training.

1.  **ToMeSD** (Token Merging)
2.  **ToFu** (Token Fusion)
3.  **SDTM** (Structure-then-Detail Token Merging)
4.  **ToMA** (Token Merge with Attention)
5.  **IBTM** (Importance-based Token Merging)

### Training-based Methods
These methods require training learnable modules (e.g., routers or scoring networks) before inference.

1.  **DiffCR** (Differentiable Compression Ratios)
2.  **DyDiT** (Dynamic DiT)
3.  **SparseDiT** (Token Sparsification)

---

## Usage

All baselines are implemented as `DitBlockHook` classes. You can use `HookManager` to inject them into a pipeline.

### 1. ToMeSD: Token Merging for Fast Stable Diffusion
**Paper**: Token merging for fast stable diffusion

```python
from baselines.tome_sd import ToMeSDDitBlockHook

tomesd_config = {
    "stride": (2, 2),        # Stride for bipartite matching
    "merged_ratio": 0.15,    # Ratio of tokens to merge
    "merge_config": {        # Modules to apply merging
        "self-attn": True,
        "cross-attn": False,
        "ffn": True,
        "parallel": False,
    },
    "generator": torch.Generator("cuda").manual_seed(0),
}

HookManager.configure_hooks(model_type="sd3", default=(ToMeSDDitBlockHook, tomesd_config))
```

### 2. ToFu: Token Fusion
**Paper**: Token fusion: Bridging the gap between token pruning and token merging

```python
from baselines.tofu import ToFuDitBlockHook

tofu_config = {
    "stride": (2, 2),
    "merged_ratio": 0.2,
    "prune_layer_nums": 10,  # Number of layers to use pruning instead of merging
    "generator": torch.Generator("cuda").manual_seed(0),
}

HookManager.configure_hooks(model_type="sd3", default=(ToFuDitBlockHook, tofu_config))
```

### 3. SDTM: Structure-then-Detail Token Merging
**Paper**: Attend to not attended: Structure-then-detail token merging for post-training dit acceleration

```python
from baselines.sdtm import SDTMDiTBlockHook

sdtm_config = {
    "stride": (2, 2),
    "switch_step_fraction": 0.4, # Fraction of steps to switch from SSM to IDM
    "init_ratio": 0.5,           # Initial keep ratio
    "deviation": 0.2,            # Deviation for cosine schedule
    "alpha_s": 0.05,             # Weight for frequency-based score
    "mcw": 0.2,                  # Momentum coefficient for IDM weights
}

HookManager.configure_hooks(model_type="sd3", default=(SDTMDiTBlockHook, sdtm_config))
```

### 4. ToMA: Token Merge with Attention
**Paper**: ToMA: Token merge with attention for diffusion models

```python
from baselines.toma import ToMADitBlockHook

toma_config = {
    "tile_nums": (8, 8),      # Number of tiles (H, W)
    "merged_ratio": 0.25,
    "skip_layer_nums": 10,    # Skip merging in the first N layers
}

HookManager.configure_hooks(model_type="sd3", default=(ToMADitBlockHook, toma_config))
```

### 5. IBTM: Importance-based Token Merging
**Paper**: Importance-based token merging for efficient image and video generation

We have extended the original CFG-based importance concept to support newer models like **Flux-dev** and **Qwen-Image**.

```python
from baselines.ibtm import IBTMDiTBlockHook

ibtm_config = {
    "merged_ratio": 0.2,
    "pool_param": 0.4,        # Parameter for candidate pool size
    "dst_ratio_k": 0.25,      # Ratio of destination tokens
    "prune_timestep_ratio": 0.12, # Ratio of steps to use pruning
}

HookManager.configure_hooks(model_type="sd3", default=(IBTMDiTBlockHook, ibtm_config))
```

### 6. DiffCR: Differentiable Compression Ratios
**Paper**: Layer- and timestep-adaptive differentiable token compression ratios for efficient diffusion transformers

```python
from baselines.diffcr import DiffCRDitBlockHook

diffcr_config = {
    "ratio_min": 0.1,
    "ratio_max": 1.0,
    "ratio_step": -0.1,
}
# Note: Ensure you load the trained weights for the hooks
HookManager.configure_hooks(model_type="sd3", default=(DiffCRDitBlockHook, diffcr_config))
```

### 7. DyDiT: Dynamic Diffusion Transformer
**Paper**: Dynamic diffusion transformer

```python
from baselines.dy_dit import DynamicDiTBlockHook

dydit_config = {
    "dynamic_token": {
        "enable": True,
        "select_threshold": 0.5,
        "tau": 5.0,
    },
}
# Note: Requires trained scorer weights
HookManager.configure_hooks(model_type="flux-dev", default=(DynamicDiTBlockHook, dydit_config))
```

### 8. SparseDiT: Token Sparsification
**Paper**: Sparsedit: Token sparsification for efficient diffusion transformer

Currently, this implementation is **exclusive to SD3**, as handling the mixed logic for dense and sparse token processing requires complex, model-specific adaptations.

```python
from baselines.sparse_dit import SparseDiTBlockHook

sparsedit_config = {
    "sdtm_ranges": [(2, 6)],      # Layers to apply sparse logic (Start, End)
    "poolingformer_layers": [-1], # Layers to use PoolingFormer logic
    "r_min": 0.0,
    "r_max": 0.0,
}
# Note: Requires trained recovery weights
HookManager.configure_hooks(model_type="sd3", default=(SparseDiTBlockHook, sparsedit_config))
```