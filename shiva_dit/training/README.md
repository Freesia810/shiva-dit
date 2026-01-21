# Training Module

This directory contains the core infrastructure for training **Shiva-DiT**. The training pipeline is built on top of `accelerate` and `diffusers`, supporting various training modes (LoRA, Hook-only, Full Finetune) and advanced features like Knowledge Distillation.

## 1. Data Preparation

 The training pipeline uses the `ImageTextDataset` class, which expects a **JSONL** (JSON Lines) metadata file.

### JSONL Format
Each line in the JSONL file must be a valid JSON object containing at least a `text` field and an image reference field (`image` or `file_name`).

```json
{"text": "A cyberpunk city with neon lights", "image": "images/city_01.png"}
{"text": "A cute cat sleeping on a sofa", "image": "/absolute/path/to/cat.jpg"}
{"text": "An astronaut riding a horse", "image": "hf://datasets/user/dataset/train/0/image_col"}
```

### Supported Image Paths
The `image` field supports:
* **Relative Paths**: Relative to the `image_dir` defined in `TrainerConfig`.
* **Absolute Paths**: Local system paths.
* **HuggingFace URIs**: Format `hf://dataset_name/split/index/column_name` (requires `datasets` library).

---

## 2. Configuration

The training configuration is managed by dataclasses in `shiva_dit/training/config.py`.

### TrainerConfig
Controls the main training loop and hyperparameters.

| Parameter | Description |
| :--- | :--- |
| `model_type` | The model architecture (e.g., `sd3`, `flux-dev`, `pixart-sigma`, `qwen-image`). |
| `training_mode` | **`hook_only`**: Train only the Shiva hooks/router.<br>**`lora`**: Train LoRA adapters (and hooks if enabled).<br>**`finetune`**: Full model finetuning. |
| `distillation_final_weight` | Weight for the final output distillation loss (Student vs Frozen Teacher). |
| `distillation_middle_weight` | Weight for layer-wise feature distillation loss. |
| `mixed_precision` | Precision mode: `no`, `fp16`, or `bf16`. |
| `gradient_accumulation_steps` | Number of steps to accumulate gradients before updating. |

### LoraConfig
Specific settings when `training_mode="lora"`.

```python
lora_config = {
    "enable": True,
    "rank": 32,
    "lora_alpha": 32,
    "target_modules": ["to_q", "to_k", "to_v", "to_out.0"], # Targeting Attention
}
```

---

## 3. Training Workflow

The training process is defined in `shiva_dit/training/trainer.py`. It automatically handles:
1.  **Model Initialization**: Loads the DiT model, VAE, and Text Encoders.
2.  **Hook Injection**: Injects Shiva acceleration hooks (Router/Gater) into the DiT.
3.  **Distillation Setup**: If configured, loads a frozen "Teacher" model for knowledge distillation.
4.  **Optimization**: Manages optimizers for different parameter groups (Router, LoRA, Base Model).

### Example Usage

You can start training using the provided scripts in `scripts/train/`. For example, `scripts/train/shiva_router.py`:

```python
from shiva_dit.training import DiffusionModelTrainer, LoraConfig, TrainerConfig
from shiva_dit.shiva_hook import ShivaDitBlockHook

# 1. Define Hook Configuration
hook_config = {
    "shiva_mode": "image",
    "image_ratio_mode": "adaptive",
    "fixed_image_ratio": 0.6,
    "init_Ts": 10.0, # Temperature for Gumbel-Softmax
}

# 2. Define Trainer Configuration
trainer_config = {
    "metadata_file": "data/train.jsonl",
    "image_dir": "data/images/",
    "model_type": "sd3",
    "output_dir": "./checkpoints_shiva",
    "training_mode": "lora", # Train LoRA + Hooks
    "lora_config": LoraConfig(enable=True, rank=32),
    "trainable_hook_config": (ShivaDitBlockHook, hook_config),
    "distillation_final_weight": 1.0,
    "optimizer": "AdamW",
    "mixed_precision": "bf16",
}

# 3. Start Training
if __name__ == "__main__":
    config = TrainerConfig(**trainer_config)
    trainer = DiffusionModelTrainer(config=config)
    trainer.train()
```

### Running with Accelerate
It is recommended to run the script using `accelerate` for distributed training or mixed precision support:

```bash
accelerate launch scripts/train/shiva_router.py
```

---

## 4. Key Features

### Dynamic Distillation
The trainer supports distilling knowledge from a frozen original model (Teacher) to the accelerated model (Student).
* **Final Layer Distillation**: MSE loss between the student's and teacher's noise predictions.
* **Feature Distillation**: (Optional) MSE loss between intermediate layer activations.

### Stage Scheduling
You can define a `stage_schedule` in `extra_config` to change training hyperparameters dynamically during training (e.g., warmup, changing loss weights).

```python
"extra_config": {
    "stage_schedule": [
        {
            "steps": 2000,
            "weights": {"ratio": 0.0, "distill": 1.0}, # Phase 1: Pure Distillation
            "lr": {"router": 1e-4}
        },
        {
            "steps": -1, # Until end
            "weights": {"ratio": 0.1, "distill": 1.0}, # Phase 2: Add Ratio Loss
        }
    ]
}
```

### Monitoring
Training metrics are logged to **TensorBoard** (default) or **WandB**.
* **Losses**: Main task loss, Distillation loss, Router auxiliary losses.
* **Visualizations**: Policy heatmaps (Router decisions over time/layers), Token importance maps (for Context mode).