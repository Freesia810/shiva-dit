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
    model_path: str = None
    output_dir: str = "./checkpoints/"

    log_with_tensorboard: bool = True
    tensorboard_log_dir: str = "logs"

    training_mode: TrainingMode = "hook_only" # "hook_only" "finetune" "auto-finetune" "lora" "auto-lora"
    lora_config: LoraConfig = field(default_factory=LoraConfig)
    trainable_hook_config: Tuple[type, Dict[str, Any]] = None 

    # 0 -> hook_warmup_steps -> hook_only_steps
    hook_warmup_steps: int = 0
    hook_only_steps: int = 0

    caption_dropout_rate: float = 0.25

    num_epochs: int = 10
    train_batch_size: int = 1  # (micro-batch)
    gradient_accumulation_steps: int = 4 # batch_size = train_batch_size * gradient_accumulation_steps

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
    save_best: bool = False
    best_metric_higher_is_better: bool = False

    distillation_final_weight: float = 0.0 
    distillation_middle_weight: float = 0.0
    distillation_layer_skip: int = 4

    run_validation: bool = True
    validation_steps: int = 500
    val_batch_size: int = None # if not specified, use train_batch_size
    val_metadata_file: str = None
    val_image_dir: str = None

    run_inference_validation: bool = True
    inference_steps: int = 500
    inference_metadata_file: str = None
    inference_seed: int = 42
    inference_num_inference_steps: int = 50
    inference_output_dir: str = "inference_samples"

    mixed_precision: Optional[Literal["no", "fp16", "bf16"]] = "no"

    extra_config: Dict[str, Any] = field(default_factory=dict)
