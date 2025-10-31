from dataclasses import dataclass, field
from typing import Literal, Optional, Tuple, Any, Dict

TrainingMode = Literal["hook_only", "finetune", "lora"]

@dataclass
class LoraConfig:
    enable: bool = False
    rank: int = 8
    target_modules: Tuple[str] = ("to_q", "to_k", "to_v", "to_out.0") 
    lora_alpha: int = 8
    lora_dropout: float = 0.1

@dataclass
class TrainerConfig:
    metadata_file: str = "./test/training/dataset/meta.json"
    image_dir: str = "./test/training/dataset" 
    model_type: str = "flux-dev"
    output_dir: str = "./checkpoints/"

    training_mode: TrainingMode = "hook_only"
    lora_config: LoraConfig = field(default_factory=LoraConfig)
    trainable_hook_config: Tuple[type, Dict[str, Any]] = None 

    num_epochs: int = 10
    train_batch_size: int = 1  # (micro-batch)
    gradient_accumulation_steps: int = 4
    # batch_size = train_batch_size * gradient_accumulation_steps

    learning_rate: float = 1e-4
    optimizer: str = "AdamW" # "Adam", "AdamW", "SGD"
    weight_decay: float = 1e-2
    lr_scheduler: str = "cosine" # "linear", "cosine", "constant"
    num_warmup_steps: int = None

    main_loss_weight: float = 1.0
    hook_loss_weight: float = 0.1

    seed: int = 42
    logging_steps: int = 50
    save_steps: int = 500

    mixed_precision: Optional[Literal["no", "fp16", "bf16"]] = "no"
