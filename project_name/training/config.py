# 文件: project_name/training/config.py

from dataclasses import dataclass, field
from typing import Literal, Optional, Tuple, Any, Dict

# 定义三种训练模式
TrainingMode = Literal["hook_only", "finetune", "lora"]

@dataclass
class LoraConfig:
    """LoRA 微调配置"""
    enable: bool = False
    rank: int = 8
    # DiT/Transformer 中常见的 attention 模块
    target_modules: Tuple[str] = ("to_q", "to_k", "to_v", "to_out.0") 
    lora_alpha: int = 8
    lora_dropout: float = 0.1

@dataclass
class TrainerConfig:
    """主训练器配置"""
    # --- 数据和模型路径 ---
    metadata_file: str = "./data/metadata.jsonl" # 数据集元数据
    image_dir: str = "./data/images/"           # 数据集图像
    model_id: str = "flux-dev"                  # 要加载的 DiT/U-Net 模型
    model_type: str = "flux-dev"                # 用于 HookManager 的模型类型
    output_dir: str = "./checkpoints/"          # 模型保存路径

    # --- 训练模式 ---
    training_mode: TrainingMode = "hook_only"
    lora_config: LoraConfig = field(default_factory=LoraConfig)
    
    # --- 可训练 Hook 配置 ---
    # 格式: (Hook类, Hook的初始化kwargs)
    # 这使得 trainer 可以灵活适配 diffcr, dydit 或未来的 hook
    trainable_hook_config: Tuple[type, Dict[str, Any]] = None 

    # --- 训练超参 (含梯度累计) ---
    num_epochs: int = 10
    train_batch_size: int = 1  # (关键) 单次 forward 的 batch size (micro-batch)
    gradient_accumulation_steps: int = 4 # (关键) 梯度累计步数
    # (有效的 Batch Size = train_batch_size * gradient_accumulation_steps)
    
    learning_rate: float = 1e-4
    optimizer: str = "AdamW" # "Adam", "AdamW", "SGD"
    weight_decay: float = 1e-2
    lr_scheduler: str = "cosine" # "linear", "cosine", "constant"
    
    # --- 损失权重 ---
    main_loss_weight: float = 1.0
    hook_loss_weight: float = 0.1 # 例如 DiffCR 的 ratio 损失

    # --- 其他 ---
    seed: int = 42
    logging_steps: int = 50 # (基于 optimizer steps)
    save_steps: int = 500   # (基于 optimizer steps)
    
    # (可选) 混合精度训练
    mixed_precision: Optional[Literal["no", "fp16", "bf16"]] = "no"